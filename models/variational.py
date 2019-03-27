import torch
from torch import nn
from skopt.space import Real, Integer

from utils.vision import UpSample
from configs.constants import INPUT_CHANNELS


class VariationalAutoEncoder(nn.Module):
    model_hyperparameters_space = [
        Integer(2, 75, name="latent_dimension"),
        Integer(75, 250, name="linear_size"),
        Real(0.1, 0.5, name="dropout")
    ]

    def __init__(self, ksi=1., latent_dimension=2, linear_size=100, dropout=.1,
                 reconstruction_criterion="MSELoss"):
        """
        Implementation of the Variational Convolutional auto-encoder.

        :param ksi: A hyper-parameter for tuning the importance of the
            KL Divergence loss compared to the reconstruction loss.
            In the seminal paper, a value of 1 seems to be a good candidate.
        :param latent_dimension: The dimension of the latent space.
        :param linear_size: Size of the last encoding linear.
        :param dropout: The dropout factor. Must belong to [0, 1).
        :param reconstruction_criterion: The criterion to use for the
            reconstruction loss.
        """

        assert 0 <= dropout < 1

        super(VariationalAutoEncoder, self).__init__()

        self.ksi = ksi
        self.hidden_dimension = latent_dimension

        self.reconstruction_criterion = getattr(nn, reconstruction_criterion)()

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.max = nn.MaxPool2d(2)

        self.encoder_conv32 = nn.Conv2d(INPUT_CHANNELS, 64, kernel_size=3,
                                        padding=1)
        self.encoder_conv16 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder_conv8 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.encoder_conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.encoder_lin_100 = nn.Linear(256 * 4 * 4, linear_size)
        self.encoder_lin_mu = nn.Linear(linear_size, latent_dimension)
        self.encoder_lin_sigma = nn.Linear(linear_size, latent_dimension)

        self.upsample = UpSample(scale_factor=2)

        self.decoder_lin_l = nn.Linear(latent_dimension, linear_size)
        self.decoder_lin_100 = nn.Linear(linear_size, 256 * 4 * 4)
        self.decoder_conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.decoder_conv8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.decoder_conv16 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder_conv32 = nn.Conv2d(64, INPUT_CHANNELS, kernel_size=3,
                                        padding=1)

    def encode_variance(self, x):
        """
        Encode the input to obtain the mu and sigma corresponding to it.

        :param x: the input to encode
        :return: mu and log sigma
        """
        # Get the batch size
        n = x.size(0)

        x = self.encoder_conv32(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.max(x)

        x = self.encoder_conv16(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.max(x)

        x = self.encoder_conv8(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.max(x)

        x = self.encoder_conv4(x)
        x = self.dropout(x)
        x = self.activation(x)

        # Flatten the input
        x = x.view(n, -1)

        x = self.encoder_lin_100(x)
        x = self.dropout(x)
        x = self.activation(x)

        mu = self.encoder_lin_mu(x)
        log_sigma = self.encoder_lin_sigma(x)

        return mu, log_sigma

    def encode(self, x):
        """
        Encoder in the VAE architecture.
        Takes a 32x32 image as input and returns its latent representation.

        :param x: A batch of 32x32 pre-processed images, as a tensor.
        :return: The encoding of x in the latent space.
        """
        mu, log_sigma = self.encode_variance(x)
        return self.sample(mu, log_sigma)

    def decode(self, x):
        """
        Decoder in the VAE architecture

        :param x: A latent space representation.
        :return: The reconstruction of x, from the latent space
        to the original space.
        """
        # Get the batch size
        n = x.size(0)

        x = self.decoder_lin_l(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.decoder_lin_100(x)
        x = self.dropout(x)
        x = self.activation(x)

        # Reshape the tensor
        x = x.view(n, 256, 4, 4)

        x = self.upsample(x)
        x = self.decoder_conv4(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.upsample(x)
        x = self.decoder_conv8(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.upsample(x)
        x = self.decoder_conv16(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.decoder_conv32(x)

        # Push values to (0, 1) to get an image representation
        x = torch.sigmoid(x)

        return x

    def sample(self, mu, log_sigma):
        """
        Samples a vector within the latent space.
        :param mu: The mean of the sampled tensor.
        :param log_sigma: The log-variance of the sampled tensor
        (log for stability reasons).
        :return: The latent stochastic encoding.
        """
        sigma = torch.exp(.5 * log_sigma)
        epsilon = torch.randn_like(sigma)

        z = mu + sigma * epsilon

        return z

    def forward(self, x):
        """
        Performs the forward pass:
        * encoding from the original space into the latent representation ;
        * reconstruction with loss in the original space.

        :param x: A tensor representation of a 32x32 image
        from the original space.
        :return:
        x_tilde (torch.Tensor): The reconstruction of the original image.
        mu: The mean of the sampled tensor.
        log_sigma: The log-variance of the sampled tensor
        """
        mu, log_sigma = self.encode_variance(x)
        z = self.sample(mu, log_sigma)
        x_tilde = self.decode(z)

        return x_tilde, mu, log_sigma

    def loss(self, x, x_tilde, mu, log_sigma):
        """
        VAE loss with KL divergence.

        See appendix B of the original paper.
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114
        KL = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        :param x: samples from the original space.
        :param x_tilde: reconstruction by the network.
        :param mu: Mean of the latent space representation.
        :param log_sigma: Log-variance of the latent space representation.
        :return: A combination of the reconstruction and KL divergence loss.
        """
        # Reconstruction loss
        reconstruction = self.reconstruction_criterion(x_tilde, x)

        # KL Divergence
        divergence = -0.5 * (1 + log_sigma - mu.pow(2) - log_sigma.exp()).sum()

        total_loss = reconstruction + self.ksi * divergence

        return total_loss
