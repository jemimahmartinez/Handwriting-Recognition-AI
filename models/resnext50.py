import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, cardinality, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# The network should inherit from the nn.Module
class ResNeXt(nn.Module):
    def __init__(self, block, layers, cardinality=32):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.inplanes = 64
        # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1)

        # Max pooling (kernel_size)
        self.maxP1 = nn.MaxPool2d(3)
        # Average pooling (kernel_size)
        self.AvgP1 = nn.AdaptiveAvgPool2d((1, 1))
        # Batch Normalisation
        self.bn1 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Make Layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Fully connected layers (1)
        # parameters: (input size, number of classes)
        self.fc1 = nn.Linear(2048, 396)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality))

        return nn.Sequential(*layers)

    def forward1(self, x):
        # First set of convolutions
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxP1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.AvgP1(x)
        x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        # Making Full Connections
        x = self.fc1(x)
        # output = F.softmax(x, dim=1) # Can test logSoftmax vs. Softmax
        output = F.log_softmax(x, dim=1)
        return output

    def resnext50(**kwargs):
        model = ResNeXt(Bottleneck, [3, 4, 6, 3], **kwargs)
        return model

    def forward2(self, x):
        x = self.resnext50(x)
        return x
