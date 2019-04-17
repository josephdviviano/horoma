"""
Cloned from https://github.com/eelxpeng/UnsupervisedDeepLearning-Pytorch
on Mar 28th 2019. Style edits and refactoring by Joseph D Viviano.
"""
from configs.constants import IMAGE_SIZE, INPUT_CHANNELS, IMAGE_H, IMAGE_W
from sklearn.mixture import GaussianMixture
from sklearn.utils.linear_assignment_ import linear_assignment
from skopt.space import Real, Integer
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

PAD = 0
STRIDE = 1
LOG2PI = math.log(2*math.pi)
EPS = 1e-10
CUDA = torch.cuda.is_available()


class VaDE(nn.Module):

    model_hyperparameters_space = [
        Integer(2, 75, name="z_dim"),
        Integer(7, 150, name="n_centroids"),
        Real(0.1, 0.5, name="dropout")
    ]

    def __init__(self, z_dim=10, n_centroids=10, dropout=0.1,
                 cnn1_out_channels=10, cnn2_out_channels=20, cnn_kernel_size=5,
                 lin2_in_channels=50, maxpool_kernel=2):
        super(self.__class__, self).__init__()

        # Save settings for later.
        self.z_dim = z_dim
        self.n_centroids = n_centroids
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SELU()
        self.maxpool = nn.MaxPool2d(maxpool_kernel, return_indices=True)
        self.cnn2_out_channels = cnn2_out_channels

        # Architecture copied from best performing model, conv_ae.py
        self.last_w = self._calc_output_size(
            IMAGE_W, cnn_kernel_size, PAD, STRIDE, maxpool_kernel, n_levels=2)
        self.last_h = self._calc_output_size(
            IMAGE_H, cnn_kernel_size, PAD, STRIDE, maxpool_kernel, n_levels=2)

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, cnn1_out_channels, cnn_kernel_size),
            self.maxpool,
            self.activation,
            self.dropout,
            nn.Conv2d(cnn1_out_channels, self.cnn2_out_channels, cnn_kernel_size),
            self.maxpool,
            self.activation,
            self.dropout
        )

        self.encoder_mlp = nn.Sequential(
            nn.Linear(
                self.cnn2_out_channels * self.last_w * self.last_h, lin2_in_channels),
            self.activation,
            self.dropout
        )

        # Encoding to the prior.
        self._enc_mu = nn.Linear(lin2_in_channels, z_dim)
        self._enc_log_sigma = nn.Linear(lin2_in_channels, z_dim)

        self.decoder_mlp = nn.Sequential(
            nn.Linear(z_dim, lin2_in_channels),
            self.activation,
            self.dropout,
            nn.Linear(
                lin2_in_channels, self.cnn2_out_channels * self.last_w * self.last_h),
            self.activation,
            self.dropout
        )

        # Transposed convolutions for decoder.
        self.decoder_cnn = nn.Sequential(
            nn.MaxUnpool2d(maxpool_kernel),
            nn.ConvTranspose2d(
                self.cnn2_out_channels, cnn1_out_channels, cnn_kernel_size),
            self.activation,
            self.dropout,
            nn.MaxUnpool2d(maxpool_kernel),
            nn.ConvTranspose2d(
                cnn1_out_channels, INPUT_CHANNELS, cnn_kernel_size),
            nn.Sigmoid()
        )

        #self.decoder = nn.Sequential(
        #    nn.Linear(self.z_dim, self.cnn2_out_channels * last_w * last_h),
        #    self.activation,
        #    nn.Linear(self.cnn2_out_channels * last_w * last_h,
        #              IMAGE_SIZE * INPUT_CHANNELS)
        #)

        self.create_gmm_param()

    def _calc_output_size(self, width, kernel, pad, stride, pool,
                          level=1, n_levels=1):
        """
        Recursively calculates the output image width / height given a square
        input. Assumes the same padding, kernel size, and stride were applied
        at all layers.
        """
        assert level <= n_levels
        assert level > 0 and n_levels > 0

        out = int(((width - kernel + (2*pad) / stride) + 1) / pool)

        if level < n_levels:
            out = self._calc_output_size(out, kernel, pad, stride, pool,
                                         level=level+1, n_levels=n_levels)

        return(out)

    def _get_t_2d(self, z):
        """Gaussian weights for z. TODO: currently shrinks to zero."""
        z_x = z.size()[0]
        t_2d = self.t_p.unsqueeze(0).expand(z_x, self.n_centroids)

        return(t_2d)

    def _get_u_3d(self, z):
        """Means of GMM."""
        z_x = z.size()[0]
        u_p_x = self.u_p.size()[0]
        u_p_y = self.u_p.size()[1]
        u_3d = self.u_p.unsqueeze(0).expand(z_x, u_p_x, u_p_y)

        return(u_3d)

    def _get_l_3d(self, z):
        """Covs of GMM."""
        z_x = z.size()[0]
        l_p_x = self.l_p.size()[0]
        l_p_y = self.l_p.size()[1]
        l_3d = self.l_p.unsqueeze(0).expand(z_x, l_p_x, l_p_y)

        return(l_3d)

    def _get_z_mu_t(self, z_mu):
        """Get means of latent."""
        z_mu_x = z_mu.size()[0]
        z_mu_y = z_mu.size()[1]
        z_mu_t = z_mu.unsqueeze(2).expand(z_mu_x, z_mu_y, self.n_centroids)

        return(z_mu_t)

    def _get_z_lv_t(self, z_lv):
        """Get log variances of latent."""
        z_lv_x = z_lv.size()[0]
        z_lv_y = z_lv.size()[1]
        z_lv_t = z_lv.unsqueeze(2).expand(z_lv_x, z_lv_y, self.n_centroids)

        return(z_lv_t)


    def _get_gamma(self, z, z_mu, z_lv):
        """
        z is the latent, z_mean is the mean of the latent, and z_lvar is the
        log variance of the latent.

        gamma is q_c_x, of which p_c_z estimates (eqn 15).
        """
        z_x = z.size()[0]
        z_y = z.size()[1]

        # NxDxK
        Z = z.unsqueeze(2).expand(z_x, z_y, self.n_centroids)
        z_mu_t = self._get_z_mu_t(z_mu)  # Mean of z.
        z_lv_t = self._get_z_lv_t(z_lv)  # Log variance of z.
        u_3d = self._get_u_3d(z)  # Means of GMM.
        l_3d = self._get_l_3d(z)  # Covs of GMM.

        # NxK
        # q(c|x) = p(c|z) = p(c)p(z|c), p(z|c) = normal(z|mu_c, sigma_c)
        # also see eq. 2.192 from Bishop 2006.
        t_2d = self._get_t_2d(z)
        p_c = torch.log(t_2d)
        p_z_c = 0.5*torch.log(2*math.pi*l_3d) + (Z-u_3d)**2/(2*l_3d)
        p_c_z = torch.exp(p_c - torch.sum(p_z_c, dim=1)) + EPS

        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return(gamma)

    def _log_pz_samples(samples):
        """Log liklihood of samples from the prior."""
        return(-0.5*LOG2PI*samples.size()[1] - torch.sum(0.5*(samples)**2, 1))

    def _log_qz_samples(samples, mu, logvar):
        """Log likelihood of samples from the posterior."""
        return(-0.5*LOG2PI*samples.size()[1] - torch.sum(
            0.5*(samples-mu)**2/torch.exp(logvar) + 0.5*logvar, 1))

    def _calc_bce(self, recon_x, x):
        bce = -torch.sum(
            x*torch.log(torch.clamp(recon_x, min=EPS)) +
            (1-x)*torch.log(torch.clamp(1-recon_x, min=EPS)), dim=[1,2,3])

        return(bce)

    def create_gmm_param(self):
        """
        t_p = Probability of each gaussian (all equal for now)
        u_p = Means of GMM
        l_p = Covariances of GMM.
        """
        # TODO: Differen't weights for each Gaussian?
        self.t_p = nn.Parameter(torch.ones(
            self.n_centroids)/self.n_centroids, requires_grad=False)

        # Means of GMM
        self.u_p = nn.Parameter(torch.zeros(
            self.z_dim, self.n_centroids), requires_grad=False)

        # Covariances of GMM
        self.l_p = nn.Parameter(torch.ones(
            self.z_dim, self.n_centroids), requires_grad=False)

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
        for batch_idx, inputs in enumerate(dataloader):
            #inputs = inputs.view(inputs.size(0), -1).float()

            if CUDA:
                inputs = inputs.cuda()

            inputs = Variable(inputs)
            z, _, _, _ = self.encode(inputs)
            data.append(z.data.cpu().numpy())

        data = np.concatenate(data)

        # Fit the GMM, saving the means and covariances for each gaussian.
        gmm = GaussianMixture(
            n_components=self.n_centroids, covariance_type='diag')
        gmm.fit(data)
        self.u_p.data.copy_(torch.from_numpy(
            gmm.means_.T.astype(np.float32)))
        self.l_p.data.copy_(torch.from_numpy(
            gmm.covariances_.T.astype(np.float32)))

    def forward(self, x):
        """
        If mode is train:
        Return the latent, mu, logvar, and reconstruction for a minibatch.

        Elif mode is pretrain:
        Return a reconstruction from a normal-style autoencoder.
        """
        z, mu, logvar, maxpool_indices = self.encode(x)
        x_recon = self.decode(z, maxpool_indices)
        return(x_recon, x, z, mu, logvar)

    def encode(self, x):
        """Encode x into latent z."""
        # Loop through CNN encoder to grab the indices for MaxUnpoolng
        indices = []
        for layer in self.encoder_cnn:
            if isinstance(layer, nn.MaxPool2d):
                x, idx = layer(x)
                indices.append(idx)
            else:
                x = layer(x)

        # Flatten for linear layers.
        hid = x.view([x.size(0), -1])

        # Apply linear layers to get samples (mu, sigma).
        hid = self.encoder_mlp(hid)
        mu = self._enc_mu(hid)
        logvar = self._enc_log_sigma(hid)

        # Generate latent layer.
        z = self.reparameterize(mu, logvar)

        return(z, mu, logvar, indices)

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

    def decode(self, z, indices):
        """Reconstruct x from latent z."""
        hid = self.decoder_mlp(z)
        hid = hid.view([hid.size(0),
                        self.cnn2_out_channels,
                        self.last_w,
                        self.last_h]
        )

        # Loop through CNN decoder to apply the indices for MaxUnpoolng
        indices = list(reversed(indices))
        maxunpool_idx = 0

        for layer in self.decoder_cnn:
            if isinstance(layer, nn.MaxUnpool2d):
                hid = layer(hid, indices[maxunpool_idx])
                maxunpool_idx += 1
            else:
                hid = layer(hid)

        return(hid)

    def vae_loss(self, recon_x, x, mu, logvar):
        """Standard VAE loss (for a single Gaussian prior)."""
        BCE = torch.mean(self._calc_bce(recon_x, x))

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
        return(BCE + KLD)


    def loss(self, recon_x, x, z, z_mu, z_lv):
        """VaDE loss, with a mixture of gaussians prior."""
        # NxDxK
        z_mu_t = self._get_z_mu_t(z_mu)  # Mean of z.
        z_lv_t = self._get_z_lv_t(z_lv)  # Log variance of z.
        u_3d = self._get_u_3d(z)  # Means of GMM.
        l_3d = self._get_l_3d(z)  # Covs of GMM.

        # NxK
        gamma = self._get_gamma(z, z_mu, z_lv)
        t_2d = self._get_t_2d(z)  # for p(c)

        # log p(x|z)
        bce = self._calc_bce(recon_x, x)

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
        print("bce={}, gammma={}, logpzc={}, qentropy={}, logpc={}, logqcx={}".format(
            torch.mean(bce),
            torch.mean(gamma),
            torch.mean(logpzc),
            torch.mean(qentropy),
            torch.mean(logpc),
            torch.mean(logqcx)))
        loss = torch.mean(bce + logpzc + qentropy + logpc + logqcx)

        return(loss)

    def log_mle(self, x, num_samples):
        weight = torch.zeros(x.size(0))

        for i in range(num_samples):
            z, recon_x, mu, logvar = self.forward(x)
            log_pz = self._log_pz_samples(z)

            log_px = torch.sum(
                x*torch.log(torch.clamp(recon_x, min=EPS)) +
                (1-x)*torch.log(torch.clamp(1-recon_x, min=EPS)), 1)

            log_qz = self._log_qz_samples(z, mu, logvar)
            weight += torch.exp(log_px + log_pz - log_qz).data

        return(torch.log(torch.clamp(weight/num_samples, min=1e-40)))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
