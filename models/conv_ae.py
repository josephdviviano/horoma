import torch
from torch import nn
import torch.nn.functional as F
from skopt.space import Real, Integer

from configs.constants import IMAGE_SIZE, INPUT_CHANNELS, IMAGE_H, IMAGE_W


class ConvolutionalAutoEncoder(nn.Module):
    model_hyperparameters_space = [
        Integer(30, 75, name="code_size"),
        Integer(50, 120, name="lin2_in_channels"),
        Real(0.1, 0.5, name="dropout")
    ]

    def __init__(self, code_size, cnn1_out_channels, cnn1_kernel_size,
                 cnn2_out_channels, cnn2_kernel_size, lin2_in_channels,
                 maxpool_kernel, dropout, loss_fct):
        """
        Convolutional AE.

        :param code_size: The dimension of the latent space.
        :param cnn1_out_channels: number of output channels on
        the first convolution
        :param cnn1_kernel_size: size of the kernel on the first convolution
        :param cnn2_out_channels: number of output channels on
        the second convolution
        :param cnn2_kernel_size: size of the kernel on the second convolution
        :param lin2_in_channels: size of the second linear layer
        :param maxpool_kernel: size of the maxpool
        :param dropout: The dropout factor. Must belong to [0, 1).
        :param loss_fct: The criterion to use for the
            reconstruction loss.
        """
        super(ConvolutionalAutoEncoder, self).__init__()
        self.code_size = code_size
        self.maxpool_kernel = maxpool_kernel
        self.loss_fct = getattr(nn, loss_fct)()
        self.dropout = nn.Dropout(dropout)

        self.encode_cnn_1 = nn.Conv2d(INPUT_CHANNELS, cnn1_out_channels,
                                      cnn1_kernel_size)
        self.encode_cnn_2 = nn.Conv2d(cnn1_out_channels, cnn2_out_channels,
                                      cnn2_kernel_size)

        # Calculating the size of the images before the first linear layer
        last_w = int(((IMAGE_W - cnn1_kernel_size + 1) / maxpool_kernel - cnn2_kernel_size + 1) / maxpool_kernel)
        last_h = int(((IMAGE_H - cnn1_kernel_size + 1) / maxpool_kernel - cnn2_kernel_size + 1) / maxpool_kernel)

        self.encode_lin_1 = nn.Linear(cnn2_out_channels * last_w * last_h,
                                      lin2_in_channels)
        self.encode_lin_2 = nn.Linear(lin2_in_channels, self.code_size)

        self.decode_lin_1 = nn.Linear(self.code_size,
                                      cnn2_out_channels * last_w * last_h)
        self.decode_lin_2 = nn.Linear(cnn2_out_channels * last_w * last_h,
                                      IMAGE_SIZE * INPUT_CHANNELS)

    def forward(self, image):
        """
        Performs the forward pass:
        * encoding from the original space into the latent representation ;
        * reconstruction with loss in the original space.

        :param image: A tensor representation of a 32x32 image
        from the original space.
        :return: the image in the latent space
        """
        code = self.encode(image)
        reconstruction = self.decode(code)

        return reconstruction

    def loss(self, image, reconstruction):
        """
        The loss function.

        :param image: target image
        :param reconstruction: the reconstruction of the target
        :return: loss
        """
        loss = self.loss_fct(reconstruction, image)

        return loss

    def encode(self, image):
        """
        Encoder in the CAE.

        :param image: image to encode.
        :return: image in the latent space.
        """
        code = self.encode_cnn_1(image)
        code = F.selu(F.max_pool2d(code, self.maxpool_kernel))
        code = self.dropout(code)

        code = self.encode_cnn_2(code)
        code = F.selu(F.max_pool2d(code, self.maxpool_kernel))
        code = self.dropout(code)

        code = code.view([code.size(0), -1])
        code = F.selu(self.encode_lin_1(code))
        code = self.encode_lin_2(code)

        return code

    def decode(self, code):
        """
        Decider in the CAE.

        :param code: image in the latent space
        :return: image reconstructed.
        """
        reconstruction = F.selu(self.decode_lin_1(code))
        reconstruction = torch.sigmoid(self.decode_lin_2(reconstruction))
        reconstruction = reconstruction.view((code.size(0), INPUT_CHANNELS,
                                              IMAGE_H, IMAGE_W))

        return reconstruction
