import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock
from torchvision.models.segmentation.deeplabv3 import ASPP

from utils import num_classes
from wrn import wider_resnet38_a2, bnrelu, maxpool, bnmish, softpool


class RegularStream(nn.Module):
    def __init__(self, in_channels=3, norm_act=bnrelu, pool_func=maxpool):
        super().__init__()
        self.backbone = wider_resnet38_a2(in_channels, norm_act=norm_act, pool_func=pool_func, dilation=True)

    def forward(self, x):
        res1 = self.backbone.mod1(x)
        # 1/2
        res2 = self.backbone.mod2(self.backbone.pool2(res1))
        # 1/4
        res3 = self.backbone.mod3(self.backbone.pool3(res2))
        # 1/8
        res4 = self.backbone.mod4(res3)
        res5 = self.backbone.mod5(res4)
        res6 = self.backbone.mod6(res5)
        res7 = self.backbone.mod7(res6)

        return res1, res2, res3, res4, res7


class ShapeStream(nn.Module):
    def __init__(self, norm_act=bnrelu):
        super().__init__()
        self.res3_conv = nn.Conv2d(256, 1, 1)
        self.res4_conv = nn.Conv2d(512, 1, 1)
        self.res7_conv = nn.Conv2d(4096, 1, 1)

        act = norm_act(1)[-1]
        self.res1 = BasicBlock(64, 64, 1)
        self.res1.relu = act
        self.res2 = BasicBlock(32, 32, 1)
        self.res2.relu = act
        self.res3 = BasicBlock(16, 16, 1)
        self.res3.relu = act

        self.res1_pre = nn.Conv2d(64, 32, 1)
        self.res2_pre = nn.Conv2d(32, 16, 1)
        self.res3_pre = nn.Conv2d(16, 8, 1)
        self.gate1 = GatedConv(32, 32, norm_act)
        self.gate2 = GatedConv(16, 16, norm_act)
        self.gate3 = GatedConv(8, 8, norm_act)
        self.gate = nn.Conv2d(8, 1, 1, bias=False)
        self.fuse = nn.Conv2d(2, 1, 1, bias=False)

    def forward(self, res1, res3, res4, res7, grad):
        size = grad.size()[-2:]
        res3 = F.interpolate(self.res3_conv(res3), size, mode='bilinear', align_corners=True)
        res4 = F.interpolate(self.res4_conv(res4), size, mode='bilinear', align_corners=True)
        res7 = F.interpolate(self.res7_conv(res7), size, mode='bilinear', align_corners=True)

        gate1 = self.gate1(self.res1_pre(self.res1(res1)), res3)
        gate2 = self.gate2(self.res2_pre(self.res2(gate1)), res4)
        gate3 = self.gate3(self.res3_pre(self.res3(gate2)), res7)
        gate = torch.sigmoid(self.gate(gate3))
        # C = 1
        feat = torch.sigmoid(self.fuse(torch.cat((gate, grad), dim=1)))
        return gate, feat


class GatedConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, norm_act=bnrelu):
        super().__init__(in_channels, out_channels, 1, bias=False)
        bn1, act = norm_act(in_channels + 1)
        bn2, _ = norm_act(1)
        self.attention = nn.Sequential(bn1, nn.Conv2d(in_channels + 1, in_channels + 1, 1), act,
                                       nn.Conv2d(in_channels + 1, 1, 1), bn2, nn.Sigmoid())

    def forward(self, feat, gate):
        attention = self.attention(torch.cat((feat, gate), dim=1))
        out = F.conv2d(feat * (attention + 1), self.weight)
        return out


class FeatureFusion(ASPP):
    def __init__(self, in_channels, atrous_rates=(6, 12, 18), out_channels=256, norm_act=bnrelu):
        # atrous_rates (6, 12, 18) is for stride 16
        super().__init__(in_channels, atrous_rates, out_channels)
        self.shape_conv = nn.Sequential(nn.Conv2d(1, out_channels, 1, bias=False), norm_act(out_channels))
        self.project = nn.Conv2d((len(atrous_rates) + 3) * out_channels, out_channels, 1, bias=False)
        self.fine = nn.Conv2d(128, 48, kernel_size=1, bias=False)
        # replace the activation func
        act = norm_act(1)[-1]
        for conv in self.convs:
            conv[-1] = act

    def forward(self, res2, res7, feat):
        res = []
        for conv in self.convs:
            res.append(conv(res7))
        res = torch.cat(res, dim=1)
        feat = F.interpolate(feat, res.size()[-2:], mode='bilinear', align_corners=True)
        res = torch.cat((res, self.shape_conv(feat)), dim=1)
        coarse = F.interpolate(self.project(res), res2.size()[-2:], mode='bilinear', align_corners=True)
        fine = self.fine(res2)
        out = torch.cat((coarse, fine), dim=1)
        return out


class GatedSCNN(nn.Module):
    def __init__(self, in_channels=3, norm_act=bnrelu, pool_func=softpool, num_classes=num_classes):
        super().__init__()
        assert norm_act in [bnrelu, bnmish] and pool_func in [maxpool, softpool], 'only support these types'

        self.regular_stream = RegularStream(in_channels, norm_act, pool_func)
        self.shape_stream = ShapeStream(norm_act)
        self.feature_fusion = FeatureFusion(in_channels=4096, out_channels=256, norm_act=norm_act)
        self.seg = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            norm_act(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            norm_act(256),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

    def forward(self, x, grad):
        res1, res2, res3, res4, res7 = self.regular_stream(x)
        gate, feat = self.shape_stream(res1, res3, res4, res7, grad)
        out = self.feature_fusion(res2, res7, feat)
        seg = F.interpolate(self.seg(out), grad.size()[-2:], mode='bilinear', align_corners=False)
        # [B, N, H, W], [B, 1, H, W]
        return seg, gate
