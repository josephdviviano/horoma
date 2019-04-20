import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import data_augmentation as da
import copy


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATClassLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATClassLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x, self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x, r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds


class VATRegLoss(nn.Module):

    def __init__(self, var=0.1, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param var: Assumed Variance of the predicted Gaussian (default: 0.1)
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATRegLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.var = var

    def kldiv_gaussian(self, mu1, mu2, var):
        result = (torch.pow((mu1 - mu2), 2)) / (2 * var)
        return result.mean()

    def forward(self, model, x):
        with torch.no_grad():
            pred = model(x)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x, self.xi * d)
                adv_distance = self.kldiv_gaussian(pred_hat, pred, self.var)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x, r_adv)
            lds = self.kldiv_gaussian(pred_hat, pred, self.var)

        return lds


class MultiTaskVATLoss(nn.Module):

    def __init__(self, vat_type, weight, var=0.1, xi=10.0, eps=1.0, ip=1):
        """VAT loss (not needed for horoma as we have just one task)
        :param vat_type: Type (classification, regression) for each task considered
        :param weight: Weight contribution for each task considered
        :param var: Assumed Variance of the predicted Gaussian (default: 0.1)
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(MultiTaskVATLoss, self).__init__()
        self.vat_type = vat_type
        self.weight = weight
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.var = var
        self.is_valid = False
        for a in self.vat_type:
            if a is not None:
                self.is_valid = True
                break

    def kldiv_gaussian(self, mu1, mu2, var):
        result = (torch.pow((mu1 - mu2), 2)) / (2 * var)
        return result.mean()

    def compute_avg_distance(self, pred, target):
        result = 0.0
        for i in range(len(pred)):
            if self.vat_type[i] is None:
                continue
            if self.vat_type[i] == 0:
                result += self.weight[i] * self.kldiv_gaussian(pred[i], target[i], self.var)
            else:
                logp_hat = F.log_softmax(pred[i], dim=1)
                prob_tar = F.softmax(target[i], dim=1)
                result += self.weight[i] * F.kl_div(logp_hat, prob_tar, reduction='batchmean')
        return result

    def forward(self, model, x):
        if not self.is_valid:
            return torch.FloatTensor([0.0]).sum().to(x.device)
        with torch.no_grad():
            pred = model(x)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x, self.xi * d)
                if not isinstance(pred_hat, (list, tuple)):
                    pred_hat = [pred_hat]
                adv_distance = self.compute_avg_distance(pred_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x, r_adv)
            if not isinstance(pred_hat, (list, tuple)):
                pred_hat = [pred_hat]
            lds = self.compute_avg_distance(pred_hat, pred)

        return lds


class StochasticPertubationLoss(nn.Module):

    def __init__(self, vat_type, weight, var=0.1, xi=10.0, eps=1.0, ip=1):
        """Stochastic perturbation method
        :param vat_type: Type (classification, regression) for each task considered
        :param weight: Weight contribution for each task considered
        :param var: Assumed Variance of the predicted Gaussian (default: 0.1)
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(StochasticPertubationLoss, self).__init__()
        self.vat_type = vat_type
        self.weight = weight
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.var = var
        self.is_valid = False
        for a in self.vat_type:
            if a is not None:
                self.is_valid = True
                break

    def kldiv_gaussian(self, mu1, mu2, var):
        result = (torch.pow((mu1 - mu2), 2)) / (2 * var)
        return result.mean()

    def compute_avg_distance(self, pred, target):
        result = 0.0
        for i in range(len(pred)):
            if self.vat_type[i] is None:
                continue
            if self.vat_type[i] == 0:
                result += self.weight[i] * self.kldiv_gaussian(pred[i], target[i], self.var)
            else:
                logp_hat = F.log_softmax(pred[i], dim=1)
                prob_tar = F.softmax(target[i], dim=1)
                result += self.weight[i] * F.kl_div(logp_hat, prob_tar, reduction='batchmean')
        return result

    def forward(self, model, x):
        device = x[0].device if isinstance(x, (list, tuple)) else x.device
        if not self.is_valid:
            return torch.FloatTensor([0.0]).sum().to(device)
        if isinstance(x, (list, tuple)):
            x, x_noise = x
        else:
            x_noise = da.noisy_sample(x)
        with torch.no_grad():
            pred = model(x)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]

        with _disable_tracking_bn_stats(model):

            pred_hat = model(x_noise)
            if not isinstance(pred_hat, (list, tuple)):
                pred_hat = [pred_hat]
            lds = self.compute_avg_distance(pred_hat, pred)

        return lds


class MeanTeacherRegLoss(nn.Module):

    def __init__(self, vat_type, weight, var=0.1, xi=10.0, eps=1.0, ip=1):
        """Mean teacher method
        :param vat_type: Type (classification, regression) for each task considered
        :param weight: Weight contribution for each task considered
        :param var: Assumed Variance of the predicted Gaussian (default: 0.1)
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(MeanTeacherRegLoss, self).__init__()
        self.vat_type = vat_type
        self.weight = weight
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.var = var
        self.is_valid = False
        self.ema_model = None
        for a in self.vat_type:
            if a is not None:
                self.is_valid = True
                break

    def kldiv_gaussian(self, mu1, mu2, var):
        result = (torch.pow((mu1 - mu2), 2)) / (2 * var)
        return result.mean()

    def compute_avg_distance(self, pred, target):
        result = 0.0
        for i in range(len(pred)):
            if self.vat_type[i] is None:
                continue
            if self.vat_type[i] == 0:
                result += self.weight[i] * self.kldiv_gaussian(pred[i], target[i], self.var)
            else:
                logp_hat = F.log_softmax(pred[i], dim=1)
                prob_tar = F.softmax(target[i], dim=1)
                result += self.weight[i] * F.kl_div(logp_hat, prob_tar, reduction='batchmean')
        return result

    def forward(self, model, x):
        device = x[0].device if isinstance(x, (list, tuple)) else x.device
        if not self.is_valid:
            return torch.FloatTensor([0.0]).sum().to(device)
        if self.ema_model is None:
            self.ema_model = copy.deepcopy(model)
            self.ema_model.to(device)
            self.ema_model.train()

        if isinstance(x, (list, tuple)):
            x, x_noise = x
        else:
            x_noise = x  # da.noisy_sample(x)

        with torch.no_grad():
            pred = self.ema_model(x_noise)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]

        with _disable_tracking_bn_stats(model):

            pred_hat = model(x)
            if not isinstance(pred_hat, (list, tuple)):
                pred_hat = [pred_hat]
            lds = self.compute_avg_distance(pred_hat, pred)

        return lds

    def update_ema_variables(self, model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            # ema_param.mul_(alpha).add_(1 - alpha, param)


