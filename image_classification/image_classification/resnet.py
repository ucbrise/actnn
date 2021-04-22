import torch
import torch.nn as nn
from .preact_resnet import PreActBlock, PreActBottleneck, PreActResNet

from actnn import QConv2d, QLinear, QBatchNorm2d, QReLU, QSyncBatchNorm, QMaxPool2d, config

__all__ = ['ResNet', 'build_resnet', 'resnet_versions', 'resnet_configs']

# ResNetBuilder {{{

class ResNetBuilder(object):
    def __init__(self, version, config):
        self.config = config

        self.L = sum(version['layers'])
        self.M = version['block'].M

    def conv(self, kernel_size, in_planes, out_planes, stride=1):
        if kernel_size == 3:
            conv = self.config['conv'](in_planes, out_planes, kernel_size=3, stride=stride,
                                       padding=1, bias=False)
        elif kernel_size == 1:
            conv = self.config['conv'](in_planes, out_planes, kernel_size=1, stride=stride,
                                       bias=False)
        elif kernel_size == 5:
            conv = self.config['conv'](in_planes, out_planes, kernel_size=5, stride=stride,
                                       padding=2, bias=False)
        elif kernel_size == 7:
            conv = self.config['conv'](in_planes, out_planes, kernel_size=7, stride=stride,
                                       padding=3, bias=False)
        else:
            return None

        if self.config['nonlinearity'] == 'relu':
            nn.init.kaiming_normal_(conv.weight,
                    mode=self.config['conv_init'],
                    nonlinearity=self.config['nonlinearity'])

        return conv

    def conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, stride=stride)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride=stride)
        return c

    def conv7x7(self, in_planes, out_planes, stride=1):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride=stride)
        return c

    def conv5x5(self, in_planes, out_planes, stride=1):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride=stride)
        return c

    def batchnorm(self, planes, last_bn=False):
        if config.debug_remove_bn:
            return nn.Identity()
        bn = self.config['bn'](planes)

        gamma_init_val = 0 if last_bn and self.config['last_bn_0_init'] else 1
        nn.init.constant_(bn.weight, gamma_init_val)
        nn.init.constant_(bn.bias, 0)

        return bn

    def max_pool2d(self, *args, **kwargs):
        return self.config['max_pool2d'](*args, **kwargs)

    def linear(self, in_planes, out_planes):
        return self.config['linear'](in_planes, out_planes)

    def activation(self):
        if config.debug_remove_relu:
            return nn.Identity()
        return self.config['activation']()

# ResNetBuilder }}}

# BasicBlock {{{
class BasicBlock(nn.Module):
    M = 2
    expansion = 1

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = builder.conv3x3(inplanes, planes, stride)
        self.bn1 = builder.batchnorm(planes)
        self.relu = builder.activation()
        self.conv2 = builder.conv3x3(planes, planes)
        self.bn2 = builder.batchnorm(planes, last_bn=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
# BasicBlock }}}

# Bottleneck {{{
class Bottleneck(nn.Module):
    M = 3
    expansion = 4

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = builder.conv1x1(inplanes, planes)
        self.bn1 = builder.batchnorm(planes)
        self.conv2 = builder.conv3x3(planes, planes, stride=stride)
        self.bn2 = builder.batchnorm(planes)
        self.conv3 = builder.conv1x1(planes, planes * self.expansion)
        self.bn3 = builder.batchnorm(planes * self.expansion, last_bn=True)
        self.relu = builder.activation()
        self.downsample = downsample
        self.stride = stride
        self.debug = False

    def forward(self, x):
        residual = x

        if self.debug:
            x.retain_grad()
            self.conv1_in = x

        out = self.conv1(x)

        if self.debug:
            x.retain_grad()
            self.conv1_out = out

        out = self.bn1(out)

        if self.debug:
            x.retain_grad()
            self.conv1_bn_out = out

        out = self.relu(out)

        if self.debug:
            x.retain_grad()
            self.conv1_relu_out = out

        if self.debug:
            out.retain_grad()
            self.conv2_in = out

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.debug:
            out.retain_grad()
            self.conv3_in = out

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out
# Bottleneck }}}

# ResNet {{{
class ResNet(nn.Module):
    def __init__(self, builder, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = builder.conv7x7(3, 64, stride=2)
        self.bn1 = builder.batchnorm(64)
        self.relu = builder.activation()
        self.maxpool = builder.max_pool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(builder, block, 64, layers[0])
        self.layer2 = self._make_layer(builder, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(builder, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(builder, block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = builder.linear(512 * block.expansion, num_classes)

    def _make_layer(self, builder, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            dconv = builder.conv1x1(self.inplanes, planes * block.expansion,
                                    stride=stride)
            dbn = builder.batchnorm(planes * block.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        layers = []
        layers.append(block(builder, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(builder, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def set_precision(self):    # Hack
        self.bn1.scheme.bits = self.conv1.scheme.bits
        for block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for layer in block:
                layer.bn1.scheme.bits = layer.conv1.scheme.bits
                layer.bn2.scheme.bits = layer.conv2.scheme.bits
                layer.bn3.scheme.bits = layer.conv3.scheme.bits
                layer.bn1.scheme.conv_input_norm = layer.conv1.conv_input_norm
                layer.bn2.scheme.conv_input_norm = layer.conv2.conv_input_norm
                layer.bn3.scheme.conv_input_norm = layer.conv3.conv_input_norm
                if layer.downsample is not None:
                    layer.downsample[1].scheme.bits = layer.downsample[0].scheme.bits
                    layer.downsample[1].scheme.conv_input_norm = layer.downsample[0].conv_input_norm

    def set_debug(self, debug):
        self.debug = True
        for l in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for b in l:
                b.debug = debug

    def set_name(self):
        self.linear_layers = [self.conv1]
        self.conv1.layer_name = 'conv_0'
        for lid, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for bid, block in enumerate(layer):
                for cid, convlayer in enumerate([block.conv1, block.conv2, block.conv3]):
                    convlayer.layer_name = 'conv_{}_{}_{}'.format(lid+1, bid+1, cid+1)
                    self.linear_layers.append(convlayer)
                if block.downsample is not None:
                    block.downsample[0].layer_name = 'conv_{}_{}_skip'.format(lid+1, bid+1)
                    self.linear_layers.append(block.downsample[0])

        self.fc.layer_name = 'fc'
        self.linear_layers.append(self.fc)


# ResNet }}}


# ResNet {{{
class ResNetCifar(nn.Module):
    def __init__(self, builder, block, layers, num_classes=10):
        self.inplanes = 16
        super(ResNetCifar, self).__init__()
        self.conv1 = builder.conv3x3(3, 16)
        self.bn1 = builder.batchnorm(16)
        self.relu = builder.activation()
        self.layer1 = self._make_layer(builder, block, 16, layers[0])
        self.layer2 = self._make_layer(builder, block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(builder, block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = builder.linear(64 * block.expansion, num_classes)

    def _make_layer(self, builder, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            dconv = builder.conv1x1(self.inplanes, planes * block.expansion,
                                    stride=stride)
            dbn = builder.batchnorm(planes * block.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        layers = []
        layers.append(block(builder, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(builder, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def set_debug(self, debug):
        self.debug = True
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


# ResNet }}}

resnet_configs = {
        'classic' : {
            'conv' : nn.Conv2d,
            'linear' : nn.Linear,
            'bn' : nn.BatchNorm2d,
            'max_pool2d' : nn.MaxPool2d,
            'conv_init' : 'fan_out',
            'nonlinearity' : 'relu',
            'last_bn_0_init' : False,
            'activation' : lambda: nn.ReLU(inplace=True),
            'quantize_forward': False
            },
        'fanin' : {
            'conv' : nn.Conv2d,
            'linear' : nn.Linear,
            'bn' : nn.BatchNorm2d,
            'max_pool2d' : nn.MaxPool2d,
            'conv_init' : 'fan_in',
            'nonlinearity' : 'relu',
            'last_bn_0_init' : False,
            'activation' : lambda: nn.ReLU(inplace=True),
            'quantize_forward': False
            },
        'quantize' : {
            'conv' : QConv2d,
            'linear' : QLinear,
            'bn' : QBatchNorm2d,
            'max_pool2d' : QMaxPool2d,
            'conv_init' : 'fan_in',
            'nonlinearity' : 'relu',
            'last_bn_0_init' : False,
            'activation' : QReLU,
            'quantize_forward': True
            },
        'qlinear' : {
            'conv' : QConv2d,
            'linear' : QLinear,
            'bn' : nn.BatchNorm2d,
            'max_pool2d' : QMaxPool2d,
            'conv_init' : 'fan_in',
            'nonlinearity' : 'relu',
            'last_bn_0_init' : False,
            'activation' : lambda: nn.ReLU(inplace=True),
            'quantize_forward': True
            },
        'qsyncbn': {
            'conv': QConv2d,
            'linear': QLinear,
            'bn': QSyncBatchNorm,
            'max_pool2d' : QMaxPool2d,
            'conv_init': 'fan_in',
            'nonlinearity': 'relu',
            'last_bn_0_init': False,
            'activation': lambda: nn.ReLU(inplace=True),
            'quantize_forward': True
        },
}

resnet_versions = {
        'resnet18' : {
            'net' : ResNet,
            'block' : BasicBlock,
            'layers' : [2, 2, 2, 2],
            },
         'resnet34' : {
            'net' : ResNet,
            'block' : BasicBlock,
            'layers' : [3, 4, 6, 3],
            },
         'resnet50' : {
            'net' : ResNet,
            'block' : Bottleneck,
            'layers' : [3, 4, 6, 3],
            },
        'resnet101' : {
            'net' : ResNet,
            'block' : Bottleneck,
            'layers' : [3, 4, 23, 3],
            },
        'resnet152' : {
            'net' : ResNet,
            'block' : Bottleneck,
            'layers' : [3, 8, 36, 3],
            },
        'resnet56' : {
            'net' : ResNetCifar,
            'block' : BasicBlock,
            'layers' : [9, 9, 9],
            },
        'preact_resnet20' : {
            'net' : PreActResNet,
            'block' : PreActBlock,
            'layers' : [3, 3, 3],
            },
        'preact_resnet56' : {
            'net' : PreActResNet,
            'block' : PreActBlock,
            'layers' : [9, 9, 9],
            },
        'preact_resnet110' : {
            'net' : PreActResNet,
            'block' : PreActBlock,
            'layers' : [18, 18, 18],
            },
        'preact_resnet164' : {
            'net' : PreActResNet,
            'block' : PreActBottleneck,
            'layers' : [18, 18, 18],
            },
        'preact_resnet1001' : {
            'net' : PreActResNet,
            'block' : PreActBottleneck,
            'layers' : [111, 111, 111],
            },
        }


def build_resnet(version, config, num_classes, model_state=None):
    version = resnet_versions[version]
    config = resnet_configs[config]

    builder = ResNetBuilder(version, config)
    print("Version: {}".format(version))
    print("Config: {}".format(config))
    model = version['net'](builder,
                           version['block'],
                           version['layers'],
                           num_classes)

    return model
