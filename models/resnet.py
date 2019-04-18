import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.utils.model_zoo as model_zoo
#from torchvision.models.resnet import ResNet, BasicBlock

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
             'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
             'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def feature_extract(self, x):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            features = x.view(x.size(0), -1)
        return features


    def forward(self, features):

        outputs = self.fc(features)

        return outputs


class ResNet18(ResNet):
    def __init__(self, num_classes=17):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

        # Freeze dem weights.
        for param in self.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(512, num_classes)
        self.crossentropy = CrossEntropyLoss(weight=torch.tensor([3.797049153371664, 2.9623605497447008, 1.611298036468296, 3.43174195631091, 3.9992895148743295, 1.8954284483805277, 0.1553497386864832, 0.9593591633310533, 1.6456965303961806, 0.88036092363079, 3.672995369869318, 1.52783087478394, 2.877434609539806, 0.6857456581685869, 1.5306280831565733, 3.627337111654075, 0.28470263288250225]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        outputs = self.fc(features)

        return outputs

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


class ResNet18_Ens(ResNet):
    def __init__(self, num_classes=17, num_heads = 10):
        super(ResNet18_Ens, self).__init__(BasicBlock,  [2, 2, 2, 2])
        self.load_state_dict(model_zoo.load_url(model_urls['resne18']))

        # Freeze dem weights.
        for param in self.parameters():
            param.requires_grad = False

        self.num_heads = num_heads

        for i in range(self.num_heads):
            setattr(self, "head{}".format(i),  nn.Linear(512, num_classes))
        self.curr_head = 0
        self.change_head(self.curr_head)

        self.crossentropy = CrossEntropyLoss(weight=torch.tensor([3.797049153371664, 2.9623605497447008, 1.611298036468296, 3.43174195631091, 3.9992895148743295, 1.8954284483805277, 0.1553497386864832, 0.9593591633310533, 1.6456965303961806, 0.88036092363079, 3.672995369869318, 1.52783087478394, 2.877434609539806, 0.6857456581685869, 1.5306280831565733, 3.627337111654075, 0.28470263288250225]))
        self.num_classes = num_classes

    def change_head(self, head_number):
        if 0 <= head_number < self.num_heads:
            self.curr_head = head_number
            self.fc = getattr(self, "head{}".format(head_number))
        else:
            print("change_head failed, current head is still {}".format(self.curr_head))

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

    def ensemble_predict(self, features, mode = "hard"):
        soft_predictions = torch.zeros(self.num_heads, features.size(0), self.num_classes)
        hard_predictions = torch.zeros(self.num_heads, features.size(0), self.num_classes)
        for i in range(self.num_heads):
            self.change_head(i)
            soft_predictions[i,:,:] = self.fc(features)
            max_predictions, _ = torch.max(soft_predictions[i,:,:],dim=1)
            hard_predictions[i,:,:] = (soft_predictions[i,:,:] == max_predictions.view(-1,1))
        hard_vote = torch.argmax(torch.mean(hard_predictions,dim=0),dim=1)
        soft_vote = torch.argmax(torch.mean(soft_predictions,dim=0),dim=1)
        if mode == "hard":
            return hard_vote
        elif mode == "soft":
            return soft_vote

class ResNet34_Ens(ResNet):
    def __init__(self, num_classes=17, num_heads=10, dropout=0.5):
        super(ResNet34_Ens, self).__init__(BasicBlock,  [3, 4, 6, 3])
        self.load_state_dict(model_zoo.load_url(model_urls['resnet34']))

        # Freeze dem weights.
        for param in self.parameters():
            param.requires_grad = False

        self.num_heads = num_heads

        for i in range(self.num_heads):
            setattr(self, "head{}".format(i),
                    nn.Sequential(nn.Linear(512, 512),
                                  nn.Dropout(p=dropout),
                                  nn.Linear(512, num_classes)
                    )
            )

        self.curr_head = 0
        self.change_head(self.curr_head)

        self.crossentropy = CrossEntropyLoss(weight=torch.tensor([3.797049153371664, 2.9623605497447008, 1.611298036468296, 3.43174195631091, 3.9992895148743295, 1.8954284483805277, 0.1553497386864832, 0.9593591633310533, 1.6456965303961806, 0.88036092363079, 3.672995369869318, 1.52783087478394, 2.877434609539806, 0.6857456581685869, 1.5306280831565733, 3.627337111654075, 0.28470263288250225]))
        self.num_classes = num_classes

    def change_head(self, head_number):
        if 0 <= head_number < self.num_heads:
            self.curr_head = head_number
            self.fc = getattr(self, "head{}".format(head_number))
        else:
            print("change_head failed, current head is still {}".format(self.curr_head))

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

    def ensemble_predict(self, features, mode = "hard"):
        soft_predictions = torch.zeros(self.num_heads, features.size(0), self.num_classes)
        hard_predictions = torch.zeros(self.num_heads, features.size(0), self.num_classes)
        for i in range(self.num_heads):
            self.change_head(i)
            soft_predictions[i,:,:] = self.fc(features)
            max_predictions, _ = torch.max(soft_predictions[i,:,:],dim=1)
            hard_predictions[i,:,:] = (soft_predictions[i,:,:] == max_predictions.view(-1,1))
        hard_vote = torch.argmax(torch.mean(hard_predictions,dim=0),dim=1)
        soft_vote = torch.argmax(torch.mean(soft_predictions,dim=0),dim=1)
        if mode == "hard":
            return hard_vote
        elif mode == "soft":
            return soft_vote

