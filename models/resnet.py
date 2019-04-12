from torch import nn
from torch.nn import CrossEntropyLoss
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, BasicBlock

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'}


class ResNet18(ResNet):
    def __init__(self, num_classes=17):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        self.fc = nn.Linear(512, num_classes)
        self.crossentropy = CrossEntropyLoss()

    def loss(self, x, label):
        """
        Loss logic for the Vanilla AE.

        :param x: target image
        :param x_tilde: reconstruction by the network.
        :return:  A combination of the reconstruction and KL divergence loss.
        """
        # Reconstruction loss

        loss = self.crossentropy(x, label)

        return loss
