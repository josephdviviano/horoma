import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from vat import MultiTaskVATLoss, StochasticPertubationLoss, MeanTeacherRegLoss
import scoring_function as scoreF
import evaluator
import utils


class Trainer:
    def __init__(
        self,
        args,
        device,
        criterion_dict,
        target_labels,
        weight=None,
        target_vat_dict=None,
        target_entropy_dict=None,
        init_iter=0,
    ):
        self.args = args
        self.init_iter = init_iter
        self.device = device
        self.target_labels = target_labels
        self.criterion_dict = criterion_dict
        self.target_entropy_dict = target_entropy_dict
        self.target_vat_dict = target_vat_dict
        self.criterion = [self.criterion_dict[a] for a in self.target_labels]
        self.entropy = [
            None if self.target_entropy_dict is None else self.target_entropy_dict[a]
            for a in self.target_labels
        ]
        self.vat_type = [
            None if self.target_vat_dict is None else self.target_vat_dict[a]
            for a in self.target_labels
        ]
        self.is_entropy_based = False
        for a in self.entropy:
            if a is not None:
                self.is_entropy_based = True
                break
        self.weight = weight
        if self.weight is None:
            self.weight = [1.0] * len(self.criterion)
        assert len(self.weight) == len(self.criterion)
        if args.cr_type == 3:
            self.vat_loss = MeanTeacherRegLoss(
                self.vat_type,
                self.weight,
                var=args.reg_vat_var,
                xi=args.xi,
                eps=args.eps,
                ip=args.ip,
            )
        elif args.cr_type == 2:
            self.vat_loss = MultiTaskVATLoss(
                self.vat_type,
                self.weight,
                var=args.reg_vat_var,
                xi=args.xi,
                eps=args.eps,
                ip=args.ip,
            )
        elif args.cr_type == 1:
            self.vat_loss = StochasticPertubationLoss(
                self.vat_type,
                self.weight,
                var=args.reg_vat_var,
                xi=args.xi,
                eps=args.eps,
                ip=args.ip,
            )
        else:
            raise ValueError(
                "Unknown consistency regularization type - {}".format(args.cr_type)
            )

    def train(
        self,
        model,
        data_iterators,
        optimizer,
        tb_prefix="exp/",
        prefix="neural_network",
    ):
        sup_losses = [utils.AverageMeter() for _ in range(len(self.criterion) + 1)]
        vat_losses = utils.AverageMeter()
        perfs = [utils.AverageMeter() for _ in range(3)]
        tb_dir = self.args.tensorboard_dir
        if not tb_dir.endswith("/"):
            tb_dir += "/"
        tb_dir += tb_prefix
        writer = utils.set_tensorborad_writer(tb_dir)

        model.train()

        criterion = self.criterion
        score_param_index = evaluator.get_scoring_func_param_index(self.target_labels)
        weight = self.weight
        if weight is None:
            weight = [1.0] * len(criterion)
        assert len(weight) == len(criterion)

        tbIndex = 0

        best_val_metric = 0.0

        # for k in tqdm(range(self.init_iter, self.args.iters)):
        for k in range(self.init_iter, self.args.iters):

            # reset
            if k > 0 and k % self.args.log_interval == 0:
                tbIndex += 1
                val_mean_loss, val_metrics, _ = self.eval(
                    model, data_iterators, key="val"
                )

                if val_metrics[0] > best_val_metric:
                    best_val_metric = val_metrics[0]
                    filename = (
                        self.args.checkpoint_dir + prefix + "_{}.pt".format("BestModel")
                    )
                    utils.set_path(filename)
                    utils.save_checkpoint(model, k, filename, optimizer, self.vat_loss)

                # writer.add_scalar("Train/Loss", sup_losses[0].avg, tbIndex)
                # writer.add_scalar("Train/VAT_Loss", vat_losses.avg, tbIndex)

                # Using defn:(loss = supervised_loss + reg_loss + self.args.alpha * lds)
                writer.add_scalar("Train/total_loss", loss, tbIndex)
                writer.add_scalar("Train/supervised_loss", supervised_loss, tbIndex)
                writer.add_scalar("Train/reg_loss", reg_loss, tbIndex)
                writer.add_scalar("Train/lds", self.args.alpha * lds, tbIndex)

                writer.add_scalar("Valid/Loss", val_mean_loss, tbIndex)

                writer.add_scalar("Train_perf/f1_score", perfs[0].avg, tbIndex)
                writer.add_scalar("Valid/f1_score", val_metrics[0], tbIndex)
                writer.add_scalar("Train_perf/accuracy", perfs[1].avg, tbIndex)
                writer.add_scalar("Valid/accuracy", val_metrics[1], tbIndex)

                train_metrics_avg = [p.avg for p in perfs]
                train_metrics_val = [p.val for p in perfs]

                print(
                    "Iteration: {}\t Loss {:.4f} ({:.4f})\t".format(
                        k, sup_losses[0].val, sup_losses[0].avg
                    ),
                    "VATLoss {:.4f} ({:.4f})\tValid_Loss {:.4f}".format(
                        vat_losses.val, vat_losses.avg, val_mean_loss
                    ),
                    "Train_Metrics: {}\t Train_Metrics_AVG {}\t".format(
                        train_metrics_val, train_metrics_avg
                    ),
                    "Valid_Metrics: {}\t".format(val_metrics),
                    "Best Perf: {}\t".format(best_val_metric),
                )
                print("-" * 80)
                for a in sup_losses:
                    a.reset()
                for a in perfs:
                    a.reset()
                vat_losses.reset()

                # re-activate trai mode
                model.train()

            x_l, y_l = next(data_iterators["labeled"])
            if not isinstance(y_l, (list, tuple)):
                y_l = [y_l]
            x_ul = next(data_iterators["unlabeled"])

            x_l, y_l = x_l.to(self.device), [t.to(self.device) for t in y_l]
            if not isinstance(x_ul, (list, tuple)):
                x_ul = x_ul.to(self.device)
            else:
                x_ul = [t.to(self.device) for t in x_ul]

            optimizer.zero_grad()

            lds = self.vat_loss(model, x_ul)

            if isinstance(x_ul, (list, tuple)):
                x_ul = x_ul[0]
            # print('')
            # print('LDS: ', lds)
            outputs = model(x_l)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]

            reg_loss = 0.0
            # print('is_entropy_based: ', self.is_entropy_based)
            if self.is_entropy_based:
                outputs_ul = model(x_ul)
                if not isinstance(outputs_ul, (list, tuple)):
                    outputs_ul = [outputs_ul]
                supervised_reg_losses = [
                    w * (0.0 if c is None else c(o))
                    for c, o, w in zip(self.entropy, outputs, self.weight)
                ]
                # print('supervised_reg_losses: ', supervised_reg_losses)
                unsupervised_reg_losses = [
                    w * (0.0 if c is None else c(o))
                    for c, o, w in zip(self.entropy, outputs_ul, self.weight)
                ]
                # print('unsupervised_reg_losses: ', unsupervised_reg_losses)

                # reg_losses = [
                #     (a+b)/(x_ul.size(0) + x_l.size(0))
                #     for a,b in zip(supervised_reg_losses, unsupervised_reg_losses)
                # ]
                reg_losses = [
                    ((a / (x_l.size(0))) + self.args.alpha * (b / (x_ul.size(0))))
                    / (1.0 + self.args.alpha)
                    for a, b in zip(supervised_reg_losses, unsupervised_reg_losses)
                ]

                # print('reg_losses: ', reg_losses)
                reg_loss = sum(reg_losses)
                # print('reg_loss: ', reg_loss)

            supervised_losses = [
                w * (c(o, gt) if o.size(1) == 1 else c(o, gt.squeeze(1)))
                for c, o, gt, w in zip(criterion, outputs, y_l, weight)
            ]
            supervised_loss = sum(supervised_losses)

            # print('supervised_losses: ', supervised_losses)
            # print('supervised_loss: ', supervised_loss)

            treeId_pred, treeId_true = None, None
            if score_param_index[0] is not None:
                i = score_param_index[0]
                _, pred_classes = torch.max(outputs[i], dim=1)
                treeId_true = y_l[i].view(-1).tolist()
                treeId_pred = pred_classes.view(-1).tolist()
                treeId_pred = np.array(treeId_pred, dtype=np.int32)
                treeId_true = np.array(treeId_true, dtype=np.int32)

            loss = supervised_loss + reg_loss + self.args.alpha * lds
            # loss = supervised_loss # + reg_loss + self.args.alpha * lds
            loss.backward()
            optimizer.step()

            if hasattr(self.vat_loss, "update_ema_variables"):
                self.vat_loss.update_ema_variables(model, self.args.ema_decay, k)

            # print('loss_final: ', loss)

            metrics = scoreF.scorePerformance(treeId_pred, treeId_true)

            for i in range(len(supervised_losses)):
                sup_losses[i + 1].update(supervised_losses[i].item(), x_l.shape[0])
            sup_losses[0].update(supervised_loss.item(), x_l.shape[0])
            for i in range(len(metrics)):
                perfs[i].update(metrics[i], x_l.shape[0])
            vat_losses.update(lds.item(), x_ul.shape[0])

            if k > 0 and k % self.args.chkpt_freq == 0:
                filename = self.args.checkpoint_dir + prefix + "_{}.pt".format(k)
                utils.set_path(filename)
                utils.save_checkpoint(model, k, filename, optimizer, self.vat_loss)

        filename = self.args.checkpoint_dir + prefix + "_{}.pt".format(self.args.iters)
        utils.set_path(filename)
        utils.save_checkpoint(model, self.args.iters, filename, optimizer)

    def eval(self, model, data_iterators, key="val"):
        assert key in ("val", "test")
        assert not (data_iterators[key] is None)
        criterion = self.criterion
        weight = self.weight
        device = self.device

        return evaluator.evaluate(
            model,
            device,
            data_iterators[key],
            self.target_labels,
            criterion,
            weight,
            labeled=True,
        )

