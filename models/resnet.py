import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, BasicBlock

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'}


class ResNet18(ResNet):
    def __init__(self, num_classes=17):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

        # Freeze dem weights.
        for param in self.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(512, num_classes)
        self.crossentropy = CrossEntropyLoss(weight=torch.tensor([3.797049153371664, 2.9623605497447008, 1.611298036468296, 3.43174195631091, 3.9992895148743295, 1.8954284483805277, 0.1553497386864832, 0.9593591633310533, 1.6456965303961806, 0.88036092363079, 3.672995369869318, 1.52783087478394, 2.877434609539806, 0.6857456581685869, 1.5306280831565733, 3.627337111654075, 0.28470263288250225]))

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
