"""
Cloned from https://github.com/eelxpeng/UnsupervisedDeepLearning-Pytorch
on Mar 28th 2019. Style edits and refactoring by Joseph D Viviano.
"""
from sklearn.mixture import GaussianMixture
from sklearn.utils.linear_assignment_ import linear_assignment
from torch.autograd import Variable
from torch.nn import Parameter
from torchvision import datasets, transforms
from torchvision.utils import save_image
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

LOG2PI = math.log(2*math.pi)
EPS = 1e-10
CUDA = torch.cuda.is_available()

def cluster_acc(Y_pred, Y):
  assert Y_pred.size == Y.size

  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)

  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)

  return(sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w)


def build_network(layers, activation="relu", dropout=0):

    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))

        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())

        if dropout > 0:
            net.append(nn.Dropout(dropout))

    return(nn.Sequential(*net))


def log_likelihood_samples_unit_gaussian(samples):
    return(-0.5*LOG2PI*samples.size()[1] - torch.sum(0.5*(samples)**2, 1))


def log_likelihood_samplesImean_sigma(samples, mu, logvar):
    return(-0.5*LOG2PI*samples.size()[1] - torch.sum(
        0.5*(samples-mu)**2/torch.exp(logvar) + 0.5*logvar, 1))


class VaDE(nn.Module):

    model_hyperparameters_space = [
        Integer(2, 75, name="z_dim"),
        Integer(7, 150, name="n_centroids")
    ]

    def __init__(self, input_dim=784, z_dim=10, n_centroids=10, binary=True,
                 encode_layer=[500,500,2000], decode_layer=[2000,500,500]):
        super(self.__class__, self).__init__()

        self.z_dim = z_dim
        self.n_centroids = n_centroids

        self.encoder = build_network([input_dim] + encode_layer)
        self.decoder = build_network([z_dim] + decode_layer)

        self._enc_mu = nn.Linear(encode_layer[-1], z_dim)
        self._enc_log_sigma = nn.Linear(encode_layer[-1], z_dim)
        self._dec = nn.Linear(decode_layer[-1], input_dim)
        self._dec_act = None

        if binary:
            self._dec_act = nn.Sigmoid()

        self.create_gmm_param()

    def create_gmm_param(self):
        """
        t_p = Probability of each gaussian (all equal for now)
        u_p = Means of GMM
        l_p = Covariances of GMM.
        """

        # TODO: Differen't weights for each Gaussian?
        self.t_p = nn.Parameter(torch.ones(self.n_centroids)/self.n_centroids)

        # Means of GMM
        self.u_p = nn.Parameter(torch.zeros(self.z_dim, self.n_centroids))

        # Covariances of GMM
        self.l_p = nn.Parameter(torch.ones(self.z_dim, self.n_centroids))

    def initialize_gmm(self, dataloader):
        """
        Initializes a mixture of gaussians model by doing a forward pass of all
        samples in "dataloader", saving the latent spaces, and then fitting
        a gaussian mixture model (with self.n_centroids) to these latent 
        variables. We save the means as u_p and covariances as l_p.
        """
        if CUDA:
            self.cuda()

        self.eval()
        data = []

        # Get a collection of latent variables to fit the GMM to.
        for batch_idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.view(inputs.size(0), -1).float()

            if CUDA:
                inputs = inputs.cuda()

            inputs = Variable(inputs)
            z, _, _, _ = self.forward(inputs)
            data.append(z.data.cpu().numpy())

        data = np.concatenate(data)

        # Fit the GMM, saving the means and covariances for each gaussian.
        gmm = GaussianMixture(
            n_components=self.n_centroids, covariance_type='diag')
        gmm.fit(data)
        self.u_p.data.copy_(torch.from_numpy(
            gmm..means_.T.astype(np.float32)))
        self.l_p.data.copy_(torch.from_numpy(
            gmm.covariances_.T.astype(np.float32)))

    def reparameterize(self, mu, lv):
        """
        Reparameterizion trick to get the mean and log variance of the encoder.
        """
        if self.training:
          std = lv.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return(eps.mul(std).add_(mu))
        else:
          return(mu)

    def forward(self, x):
        """
        Return the latent, mu, logvar, and reconstruction for a minibatch.
        """
        z, mu, logvar = self.encode(x)
        x_recon = self.decode(z)

        return(x_recon, x, z, mu, logvar)

    def encode(self, x):
        """Encode x into latent z."""
        hid = self.encoder(x)
        mu = self._enc_mu(hid)
        logvar = self._enc_log_sigma(hid)
        z = self.reparameterize(mu, logvar)

        return(z, mu, logvar)

    def decode(self, z):
        """Reconstruct x from latent z."""
        hid = self.decoder(z)
        x_recon = self._dec(hid)
        if self._dec_act is not None:
            x_recon = self._dec_act(x_recon)

        return(x_recon)

    def get_t_2d(self, z):
        z_x = z.size()[0]
        t_2d = self.t_p.unsqueeze(0).expand(z_x, self.n_centroids)

        return(t_2d)

    def get_u_3d(self, z):
        """Means of GMM."""
        z_x = z.size()[0]
        u_p_x = self.u_p.size()[0]
        u_p_y = self.u_p.size()[1]
        u_3d = self.u_p.unsqueeze(0).expand(z_x, u_p_x, u_p_y)

        return(u_3d)

    def get_l_3d(self, z):
        """Covs of GMM.""" 
        z_x = z.size()[0]
        l_p_x = self.l_p.size()[0]
        l_p_y = self.l_p.size()[1]
        l_3d = self.l_p.unsqueeze(0).expand(z_x, l_p_x, l_p_y)

        return(l_3d)

    def get_z_mu_t(self, z_mu):
        z_mu_x = z_mu.size()[0]        
        z_mu_y = z_mu.size()[1]
        z_mu_t = z_mu.unsqueeze(2).expand(z_mu_x, z_mu_y, self.n_centroids)

        return(z_mu_t)

    def get_z_lv_t(self, z_lv):
        z_lv_x = z_lv.size()[0]
        z_lv_y = z_lv.size()[1]
        z_lv_t = z_lv.unsqueeze(2).expand(z_lv_x, z_lv_y, self.n_centroids)

        return(z_lv_t)


    def get_gamma(self, z, z_mu, z_lv):
        """
        z is the latent, z_mean is the mean of the latent, and z_lvar is the
        log variance of the latent.

        gamma is q_c_x, of which p_c_z estimates (eqn 15).
        """
        z_x = z.size()[0]
        z_y = z.size()[1]

        # NxDxK
        Z = z.unsqueeze(2).expand(z_x, z_y, self.n_centroids) 
        z_mu_t = self.get_z_mu_t(z_mu)  # Mean of z.
        z_lv_t = self.get_z_lv_t(z_lv)  # Log variance of z.
        u_3d = self.get_u_3d(z)  # Means of GMM.
        l_3d = self.get_l_3d(z)  # Covs of GMM.

        # NxK
        t_2d = self.get_t_2d(z)  # p(c)
        p_c_z = torch.exp(torch.log(t_2d) - torch.sum(0.5*torch.log(2*math.pi*l_3d) + (Z-u_3d)**2/(2*l_3d), dim=1)) + EPS
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return(gamma)

    def loss(self, recon_x, x, z, z_mu, z_lv):

        # NxDxK
        z_mu_t = self.get_z_mu_t(z_mu)  # Mean of z.
        z_lv_t = self.get_z_lv_t(z_lv)  # Log variance of z.
        u_3d = self.get_u_3d(z)  # Means of GMM.
        l_3d = self.get_l_3d(z)  # Covs of GMM.

        # NxK
        gamma = self.get_gamma(z, z_mu, z_lv) 
        t_2d = self.get_t_2d(z)  # for p(c)

        # log p(x|z)
        bce = -torch.sum(
            x*torch.log(torch.clamp(recon_x, min=EPS)) +
            (1-x)*torch.log(torch.clamp(1-recon_x, min=EPS)), 1)

        # log p(z|c)
        logpzc = torch.sum(
            0.5 * gamma * torch.sum(
                math.log(2*math.pi) + torch.log(l_3d) + 
                torch.exp(z_lv_t)/l_3d + (z_mu_t-u_3d)**2/l_3d, dim=1), dim=1)

        # log q(z|x)
        qentropy = -0.5 * torch.sum(1 + z_lv + math.log(2*math.pi), 1)

        # log p(c)
        logpc = -torch.sum(torch.log(t_2d)*gamma, 1)

        # log q(c|x)
        logqcx = torch.sum(torch.log(gamma)*gamma, 1)

        # Normalise by same number of elements as in reconstruction.
        loss = torch.mean(bce + logpzc + qentropy + logpc + logqcx)

        return(loss)

    def log_marginal_likelihood_estimate(self, x, num_samples):
        weight = torch.zeros(x.size(0))

        for i in range(num_samples):
            z, recon_x, mu, logvar = self.forward(x)
            zloglikelihood = log_likelihood_samples_unit_gaussian(z)

            dataloglikelihood = torch.sum(
                x*torch.log(torch.clamp(recon_x, min=EPS)) +
                (1-x)*torch.log(torch.clamp(1-recon_x, min=EPS)), 1)

            log_qz = log_likelihood_samplesImean_sigma(z, mu, logvar)
            weight += torch.exp(dataloglikelihood + zloglikelihood - log_qz).data

        return(torch.log(torch.clamp(weight/num_samples, min=1e-40)))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
