from abc import abstractmethod

import logging
import torch.nn as nn
import numpy as np


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def encode(self, x):
        """
        Encode a data point.

        :param x: the data point to encode
        :return: the encoded data point
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, x):
        """
        Decode a data point that was encoded.

        :param x: the encoded data point to decode
        :return: the decoded data point
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    @abstractmethod
    def loss(self, *args):
        """
        Loss logic since it can changed from model to model

        :param args: any args the loss function needs
        :return: loss
        """
        return NotImplementedError

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel,
                     self).__str__() + '\nTrainable parameters: {}'.format(
            params)
        # print(super(BaseModel, self))
