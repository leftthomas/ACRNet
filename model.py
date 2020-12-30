import torch
import torch.nn.functional as F
from pretrainedmodels import bninception
from torch import nn
from torchvision.models import resnet50, googlenet


class ProxyLinear(nn.Module):
    def __init__(self, num_proxy, in_features):
        super(ProxyLinear, self).__init__()
        self.num_proxy = num_proxy
        self.in_features = in_features
        # init proxy vector as unit random vector
        self.weight = nn.Parameter(F.normalize(torch.randn(num_proxy, in_features), dim=-1))

    def forward(self, x):
        normalized_weight = F.normalize(self.weight, dim=-1)
        output = x.mm(normalized_weight.t())
        return output

    def extra_repr(self):
        return 'num_proxy={}, in_features={}'.format(self.num_proxy, self.in_features)


class Model(nn.Module):
    def __init__(self, backbone_type, feature_dim, num_classes):
        super().__init__()

        # Backbone Network
        backbones = {'resnet50': (resnet50, 2048), 'inception': (bninception, 1024), 'googlenet': (googlenet, 1024)}
        backbone, middle_dim = backbones[backbone_type]
        backbone = backbone(pretrained='imagenet' if backbone_type == 'inception' else True)
        if backbone_type == 'inception':
            backbone.global_pool = nn.AdaptiveMaxPool2d(1)
            backbone.last_linear = nn.Identity()
        else:
            backbone.avgpool = nn.AdaptiveMaxPool2d(1)
            backbone.fc = nn.Identity()
        self.backbone = backbone

        # Refactor Layer
        self.refactor = nn.Linear(middle_dim, feature_dim, bias=False)
        self.fc = ProxyLinear(num_classes, feature_dim)

    def forward(self, x):
        features = self.backbone(x)
        features = F.layer_norm(features, features.size()[1:])
        features = F.normalize(self.refactor(features), dim=-1)
        classes = self.fc(features)
        return features, classes
