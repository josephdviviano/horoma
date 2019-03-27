from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import numpy as np


class CustomLabelledMNIST(MNIST):
    """
    Custom MNIST dataset used for testing purposes.
    """

    def __init__(self, root, train=True, target_transform=None, download=False,
                 flattened=False, subset=None):
        """

        :param root: path to the data.
        :param train: the split to use, training or test.
        :param target_transform: transform to apply to the target.
        :param download: if the data should be downloaded beforehand.
        :param flattened: if the images should be flattened
        :param subset: subset of the data to use, default: all.
        """
        super(CustomLabelledMNIST, self).__init__(root, train,
                                                  transforms.Compose(
                                                      [transforms.Resize(32),
                                                       transforms.ToTensor()]),
                                                  target_transform, download)

        self.flatten = flattened

        if subset is not None:
            if self.train:
                self.train_data = self.train_data[:subset]
                self.train_labels = self.train_labels[:subset]
            else:
                self.test_data = self.test_data[:subset]
                self.test_labels = self.test_labels[:subset]

        self.region_ids = np.random.choice(np.arange(100), size=len(self))

    def __getitem__(self, item):
        img, label = super(CustomLabelledMNIST, self).__getitem__(item)
        img = torch.cat((img, img, img), 0)
        if self.flatten:
            img = img.view(-1)
        return img, label


class CustomMNIST(CustomLabelledMNIST):

    def __getitem__(self, item):
        img, _ = super(CustomMNIST, self).__getitem__(item)
        return img
