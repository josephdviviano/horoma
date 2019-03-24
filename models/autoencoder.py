import torch.nn as nn
import torch
import torch.nn.functional as F
from skopt.space import Integer


class AutoEncoder(nn.Module):
    model_hyperparameters_space = [
        Integer(2, 50, name="latent_dimension"),
    ]

    def __init__(self, reconstruction_criterion=nn.MSELoss(),
                 latent_dimension=3):
        """
        Vanilla AE.

        :param reconstruction_criterion: The criterion to use for the
            reconstruction loss.
        :param latent_dimension: The dimension of the latent space.
        """
        super(AutoEncoder, self).__init__()

        self.reconstruction_criterion = reconstruction_criterion

        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, latent_dimension)

        self.fc5 = nn.Linear(latent_dimension, 64)
        self.fc6 = nn.Linear(64, 128)
        self.fc7 = nn.Linear(128, 512)
        self.fc8 = nn.Linear(512, 3 * 32 * 32)

    def encode(self, x):
        """
        Encoder in the Vanilla AE.

        :param x: image to encode
        :return: image in the latent space
        """
        # Get the batch size
        n = x.size(0)

        # Flatten the image
        x = x.view(n, -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return x

    def decode(self, x):
        """
        Decoder in the Vanilla AE.

        :param x: image in the latent space
        :return: reconstruction of the image
        """
        # Get the batch size
        n = x.size(0)

        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        # We push the pixels towards 0 and 1
        x = torch.sigmoid(self.fc8(x))

        # Transform the image to its original shape
        x = x.view(n, 3, 32, 32)

        return x

    def forward(self, x):
        """
        Performs the forward pass:
        * encoding from the original space into the latent representation ;
        * reconstruction with loss in the original space.

        :param x: image to reconstruct
        :return: image reconstruted
        """
        x = self.encode(x)
        x = self.decode(x)

        return x

    def loss(self, x, x_tilde):
        """
        Loss logic for the Vanilla AE.

        :param x: target image
        :param x_tilde: reconstruction by the network.
        :return:  A combination of the reconstruction and KL divergence loss.
        """
        # Reconstruction loss

        loss = self.reconstruction_criterion(x_tilde, x)

        return loss
