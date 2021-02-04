from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from SoftPool import SoftPool2d


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return torch.flatten(F.adaptive_avg_pool2d(inputs, (1, 1)), 1)


def bnrelu(channels):
    return nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU(inplace=True))


def bnmish(channels):
    return nn.Sequential(nn.BatchNorm2d(channels), Mish())


def maxpool():
    return nn.MaxPool2d(3, stride=2, padding=1)


def softpool():
    return SoftPool2d(2, stride=2)


class IdentityResidualBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1, dilation=1, groups=1, norm_act=bnrelu, dropout=None):
        super(IdentityResidualBlock, self).__init__()

        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError('channels must contain either two or three values')
        if len(channels) == 2 and groups != 1:
            raise ValueError('groups > 1 are only valid if len(channels) == 3')

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [
                ('conv1', nn.Conv2d(in_channels, channels[0], 3, stride=stride, padding=dilation, bias=False,
                                    dilation=dilation)),
                ('bn2', norm_act(channels[0])),
                ('conv2', nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False,
                                    dilation=dilation))
            ]
            if dropout is not None:
                layers = layers[0:2] + [('dropout', dropout())] + layers[2:]
        else:
            layers = [
                ('conv1', nn.Conv2d(in_channels, channels[0], 1, stride=stride, padding=0, bias=False)),
                ('bn2', norm_act(channels[0])),
                ('conv2', nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False, groups=groups,
                                    dilation=dilation)),
                ('bn3', norm_act(channels[1])),
                ('conv3', nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False))
            ]
            if dropout is not None:
                layers = layers[0:4] + [('dropout', dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        if hasattr(self, 'proj_conv'):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)

        out = self.convs(bn1)
        out.add_(shortcut)
        return out


class WiderResNetA2(nn.Module):
    def __init__(self, structure, in_channels=3, norm_act=bnrelu, pool_func=maxpool, num_classes=1000, dilation=False):
        super(WiderResNetA2, self).__init__()
        self.structure = structure
        self.dilation = dilation

        if len(structure) != 6:
            raise ValueError('Expected a structure with six values')

        # Initial layers
        self.mod1 = torch.nn.Sequential(
            OrderedDict([('conv1', nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False))]))

        # Groups of residual blocks
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048), (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            # Create blocks for module
            blocks = []
            for block_id in range(num):
                if not dilation:
                    dil = 1
                    stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
                else:
                    if mod_id == 3:
                        dil = 2
                    elif mod_id > 3:
                        dil = 4
                    else:
                        dil = 1
                    stride = 2 if block_id == 0 and mod_id == 2 else 1

                if mod_id == 4:
                    drop = partial(nn.Dropout2d, p=0.3)
                elif mod_id == 5:
                    drop = partial(nn.Dropout2d, p=0.5)
                else:
                    drop = None

                blocks.append(('block%d' % (block_id + 1),
                               IdentityResidualBlock(in_channels, channels[mod_id], stride=stride, dilation=dil,
                                                     norm_act=norm_act, dropout=drop)))
                # Update channels and p_keep
                in_channels = channels[mod_id][-1]

            # Create module
            if mod_id < 2:
                self.add_module('pool%d' % (mod_id + 2), pool_func())
            self.add_module('mod%d' % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

        # Pooling and predictor
        self.bn_out = norm_act(in_channels)
        self.classifier = nn.Sequential(OrderedDict([
            ('avg_pool', GlobalAvgPool2d()),
            ('fc', nn.Linear(in_channels, num_classes))
        ]))

    def forward(self, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(out)
        out = self.mod5(out)
        out = self.mod6(out)
        out = self.mod7(out)
        out = self.bn_out(out)
        return self.classifier(out)


def wider_resnet38_a2(in_channels=3, norm_act=bnrelu, pool_func=maxpool, num_classes=1000, dilation=False):
    return WiderResNetA2(structure=[3, 3, 6, 3, 1, 1], in_channels=in_channels, norm_act=norm_act,
                         pool_func=pool_func, num_classes=num_classes, dilation=dilation)
