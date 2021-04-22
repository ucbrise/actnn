'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import math
import torch
import torch.nn as nn
from actnn import QModule


class PreActBlock(nn.Module):
    expansion = 1
    M = 2

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None):
        super(PreActBlock, self).__init__()
        self.bn1 = builder.batchnorm(inplanes)
        self.relu = builder.activation()
        self.conv1 = builder.conv3x3(inplanes, planes, stride)
        self.bn2 = builder.batchnorm(planes, last_bn=True)
        self.conv2 = builder.conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        self.debug = False

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        if self.debug:
            out.retain_grad()
            self.conv1_in = out

        out = self.conv1(out)

        if self.debug:
            out.retain_grad()
            self.conv1_out = out

        out = self.bn2(out)
        out = self.relu(out)

        if self.debug:
            out.retain_grad()
            self.conv2_in = out

        out = self.conv2(out)

        if self.debug:
            out.retain_grad()
            self.conv2_out = out

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4
    M = 3

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = builder.batchnorm(inplanes)
        self.relu = builder.activation()
        self.conv1 = builder.conv1x1(inplanes, planes)
        self.bn2 = builder.batchnorm(planes)
        self.conv2 = builder.conv3x3(planes, planes, stride=stride)
        self.bn3 = builder.batchnorm(planes, last_bn=True)
        self.conv3 = builder.conv1x1(planes, planes*4)
        self.downsample = downsample
        self.stride = stride
        self.debug = False

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        if self.debug:
            out.retain_grad()
            self.conv1_in = out

        out = self.conv1(out)

        if self.debug:
            out.retain_grad()
            self.conv1_out = out

        out = self.bn2(out)
        out = self.relu(out)

        if self.debug:
            out.retain_grad()
            self.conv2_in = out

        out = self.conv2(out)

        if self.debug:
            out.retain_grad()
            self.conv2_out = out

        out = self.bn3(out)
        out = self.relu(out)

        if self.debug:
            out.retain_grad()
            self.conv3_in = out

        out = self.conv3(out)

        if self.debug:
            out.retain_grad()
            self.conv3_out = out

        out += residual

        return out


class PreActResNet(nn.Module):
    def __init__(self, builder, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.inplanes = 16
        self.builder = builder
        self.conv1 = builder.conv3x3(3, 16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0])
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.bn = builder.batchnorm(64 * block.expansion)
        self.relu = builder.activation()
        #self.avgpool = nn.AvgPool2d(8, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = builder.linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.builder.conv1x1(self.inplanes, planes * block.expansion, stride=stride)
            )

        layers = []
        layers.append(block(self.builder, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.builder, self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def set_debug(self, debug):
        for l in [self.layer1, self.layer2, self.layer3]:
            for b in l:
                b.debug = debug


    def set_name(self):
        self.linear_layers = [self.conv1]
        self.conv1.layer_name = 'conv_0'
        for lid, layer in enumerate([self.layer1, self.layer2, self.layer3]):
            for bid, block in enumerate(layer):
                for cid, convlayer in enumerate([block.conv1, block.conv2]):
                    convlayer.layer_name = 'conv_{}_{}_{}'.format(lid+1, bid+1, cid+1)
                    self.linear_layers.append(convlayer)
                if block.downsample is not None:
                    block.downsample[0].layer_name = 'conv_{}_{}_skip'.format(lid+1, bid+1)
                    self.linear_layers.append(block.downsample[0])

        self.fc.layer_name = 'fc'
        self.linear_layers.append(self.fc)
