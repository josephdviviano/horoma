import torch.nn as nn


class SimpleAutoEncoder(nn.Module):
    def __init__(self, n_dims, n_latent_space):
        """
        Simple AE for testing purposes

        :param n_dims: input size
        :param n_latent_space: The dimension of the latent space.
            activation: The activation function to use.
        """
        super(SimpleAutoEncoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(n_dims, 12),
            nn.ReLU(True),
            nn.Linear(12, n_latent_space))
        self.decode = nn.Sequential(
            nn.Linear(n_latent_space, 12),
            nn.ReLU(True),
            nn.Linear(12, n_dims),
            nn.Tanh())

        self.mseloss = nn.MSELoss()

    def forward(self, x):
        """
        Performs the forward pass:
        * encoding from the original space into the latent representation ;
        * reconstruction with loss in the original space.

        :param x: An tensor representation of a 32x32 image
        from the original space.
        :return: An tensor in the latent space.
        """
        x = self.encode(x)
        x = self.decode(x)
        return x

    def loss(self, target, output):
        """
        Loss logic.

        :param target: tensor representing the target
        :param output: tensor that should be similar to the target
        :return: loss
        """
        return self.mseloss(output, target)
