import torch
from torch import nn

from utils.vision import UpSample


class WassersteinAutoEncoder(nn.Module):

    def __init__(self, ksi=10., latent_dimension=2, activation=nn.ReLU(),
                 dropout=.1, reconstruction_criterion=nn.MSELoss()):
        """
        Implementation of the Wasserstein auto-encoder.
        :param ksi: A hyper-parameter for tuning the importance of the
            Wasserstein loss compared to the reconstruction loss.
            In the seminal paper, a value of 10 seems to be a good candidate.
        :param latent_dimension: The dimension of the latent space.
            activation: The activation function to use.
        :param activation: The activation function between the layers.
        :param dropout: The dropout factor. Must belong to [0, 1).
        :param reconstruction_criterion: The criterion to use for the
            reconstruction loss.
        """
        assert 0 <= dropout < 1

        super(WassersteinAutoEncoder, self).__init__()

        self.ksi = ksi
        self.hidden_dimension = latent_dimension

        self.reconstruction_criterion = reconstruction_criterion

        self.activation = activation

        self.dropout = nn.Dropout(dropout)

        self.max = nn.MaxPool2d(2)

        self.encoder_conv32 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.encoder_conv16 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder_conv8 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.encoder_conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.encoder_lin_100 = nn.Linear(256 * 4 * 4, 100)
        self.encoder_lin_l = nn.Linear(100, latent_dimension)

        self.upsample = UpSample(scale_factor=2)

        self.decoder_lin_l = nn.Linear(latent_dimension, 100)
        self.decoder_lin_100 = nn.Linear(100, 256 * 4 * 4)
        self.decoder_conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.decoder_conv8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.decoder_conv16 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder_conv32 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def encode(self, x):
        """
        Encoder in the WAE architecture. Takes a 32x32 image as input and
        returns its latent representation.

        :param x: A batch of 32x32 pre-processed images, as a tensor.
        :return: The encoding of x in the latent space.
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

        x = self.encoder_lin_l(x)

        return x

    def decode(self, x):
        """
        Decoder in the WAE architecture

        :param x:  A latent space representation.
        :return: The reconstruction of x, from the latent space to the original space.
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

    def forward(self, x):
        """
        Performs the forward pass:
        * encoding from the original space into the latent representation ;
        * reconstruction with loss in the original space.

        :param x: An tensor representation of a 32x32 image
        from the original space.
        :return:
        x_tilde: The reconstruction of the original image.
        z: The latent space representation of the original image
        (useful for computing the loss).
        """
        z = self.encode(x)
        x_tilde = self.decode(z)

        return x_tilde, z

    def loss(self, x, x_tilde, z):
        """
        WAE loss with MMD divergence.

        :param x: samples from the original space.
        :param x_tilde: reconstruction by the network.
        :param z: latent space representation.
        :return: The MMD-based loss.
        """
        n = x.size(0)

        device = x.device

        recon_loss = self.reconstruction_criterion(x_tilde, x)

        z_fake = torch.randn(n, self.hidden_dimension).to(device)

        kernel_zf_zf = self.kernel(z_fake, z_fake)
        kernel_z_z = self.kernel(z, z)
        kernel_z_zf = self.kernel(z, z_fake)

        mmd_loss = ((1 - torch.eye(n).to(device)) * (kernel_zf_zf + kernel_z_z)).sum() / (n * (n - 1)) - 2 * kernel_z_zf.mean()

        total_loss = recon_loss + self.ksi * mmd_loss

        return total_loss

    def kernel(self, x, y):
        """
        Returns a matrix K where :math:`K_{i, j} = k(x_i, y_j)`

        Here we use the inverse multiquadratics kernel.

        :param x: a PyTorch Tensor
        :param y: a PyTorch Tensor
        :return: The kernel computed for every pair of x and y.
        """
        assert x.size() == y.size()

        # We use the advised constant, with sigma=1
        # c is the expected square distance between 2 vectors sampled from Pz
        c = self.hidden_dimension * 2

        x_ = x.unsqueeze(0)
        y_ = y.unsqueeze(1)

        ker = c / (c + (x_ - y_).pow(2).sum(2))

        return ker
