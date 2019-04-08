import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None,
               track_running_stats=None):
    super(BasicBlock, self).__init__()

    assert (track_running_stats is not None)

    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNetTrunk(nn.Module):
  def __init__(self):
    super(ResNetTrunk, self).__init__()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion,
                       track_running_stats=self.batchnorm_track),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample,
                        track_running_stats=self.batchnorm_track))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(
        block(self.inplanes, planes, track_running_stats=self.batchnorm_track))

    return nn.Sequential(*layers)


class ResNet(nn.Module):
  def __init__(self):
    super(ResNet, self).__init__()

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        assert (m.track_running_stats == self.batchnorm_track)
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

class ClusterNet5gTrunk(ResNetTrunk):
  def __init__(self, config):
    super(ClusterNet5gTrunk, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    block = BasicBlock
    layers = [3, 4, 6, 3]

    in_channels = config.in_channels
    self.inplanes = 64
    self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                           padding=1,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64, track_running_stats=self.batchnorm_track)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    if config.input_sz == 96:
      avg_pool_sz = 7
    elif config.input_sz == 64:
      avg_pool_sz = 5
    elif config.input_sz == 32:
      avg_pool_sz = 3
    print("avg_pool_sz %d" % avg_pool_sz)

    self.avgpool = nn.AvgPool2d(avg_pool_sz, stride=1)

  def forward(self, x, penultimate_features=False):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    if not penultimate_features:
      # default
      x = self.layer4(x)
      x = self.avgpool(x)

    x = x.view(x.size(0), -1)

    return x


class ClusterNet5gHead(nn.Module):
  def __init__(self, config):
    super(ClusterNet5gHead, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.num_sub_heads = config.num_sub_heads

    self.heads = nn.ModuleList([nn.Sequential(
      nn.Linear(512 * BasicBlock.expansion, config.output_k),
      nn.Softmax(dim=1)) for _ in range(self.num_sub_heads)])

  def forward(self, x, kmeans_use_features=False):
    results = []
    for i in range(self.num_sub_heads):
      if kmeans_use_features:
        results.append(x)  # duplicates
      else:
        results.append(self.heads[i](x))
    return results


class IIC_semi_supervised(ResNet):
  def __init__(self, config):
    # no saving of configs
    super(IIC_semi_supervised, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.trunk = ClusterNet5gTrunk(config)
    self.head = ClusterNet5gHead(config)

    self._initialize_weights()

  def forward(self, x, kmeans_use_features=False, trunk_features=False,
              penultimate_features=False):
    x = self.trunk(x, penultimate_features=penultimate_features)

    if trunk_features:  # for semisup
      return x

    x = self.head(x, kmeans_use_features=kmeans_use_features)  # returns list
    return x


class ClusterNet5gTwoHeadHead(nn.Module):
  def __init__(self, config, output_k, semisup=False):
    super(ClusterNet5gTwoHeadHead, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.semisup = semisup

    if not semisup:
      self.num_sub_heads = config.num_sub_heads

      self.heads = nn.ModuleList([nn.Sequential(
        nn.Linear(512 * BasicBlock.expansion, output_k),
        nn.Softmax(dim=1)) for _ in range(self.num_sub_heads)])
    else:
      self.head = nn.Linear(512 * BasicBlock.expansion, output_k)

  def forward(self, x, kmeans_use_features=False):
    if not self.semisup:
      results = []
      for i in range(self.num_sub_heads):
        if kmeans_use_features:
          results.append(x)  # duplicates
        else:
          results.append(self.heads[i](x))
      return results
    else:

      return self.head(x)


class IIC_unsupervised(ResNet):
  def __init__(self, config):
    # no saving of configs
    super(IIC_unsupervised, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.trunk = ClusterNet5gTrunk(config)

    self.head_A = ClusterNet5gTwoHeadHead(config, output_k=config.output_k_A)

    semisup = (hasattr(config, "semisup") and
               config.semisup)
    print("semisup: %s" % semisup)

    self.head_B = ClusterNet5gTwoHeadHead(config, output_k=config.output_k_B,
                                          semisup=semisup)

    self._initialize_weights()

  def forward(self, x, head="B", kmeans_use_features=False,
              trunk_features=False,
              penultimate_features=False):
    # default is "B" for use by eval code
    # training script switches between A and B

    x = self.trunk(x, penultimate_features=penultimate_features)

    if trunk_features:  # for semisup
      return x

    # returns list or single
    if head == "A":
      x = self.head_A(x, kmeans_use_features=kmeans_use_features)
    elif head == "B":
      x = self.head_B(x, kmeans_use_features=kmeans_use_features)
    else:
      assert (False)

    return x