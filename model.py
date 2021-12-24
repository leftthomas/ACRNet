import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes, select_ratio, feat_dim=2048):
        super(Model, self).__init__()

        self.select_ratio = select_ratio
        self.conv = nn.Sequential(nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.proxy = nn.Parameter(torch.empty(num_classes, feat_dim, 1))
        # introduce dropout for robust, eliminate the noise
        self.drop_out = nn.Dropout()
        nn.init.kaiming_uniform_(self.proxy, a=math.sqrt(5))

    def forward(self, x):
        out = self.drop_out(self.conv(x.permute(0, 2, 1)))
        # [N, T, C]
        seg_score = F.conv1d(out, self.proxy).permute(0, 2, 1)

        k = max(int(seg_score.shape[1] * self.select_ratio), 1)
        top_score, top_idx = seg_score.topk(k=k, dim=1, largest=True)
        bottom_score, bottom_idx = seg_score.topk(k=k, dim=1, largest=False)

        # [N, C]
        act_score = torch.sigmoid(torch.mean(top_score, dim=1))
        bkg_score = torch.sigmoid(torch.mean(bottom_score, dim=1))
        # [N, T, C]
        seg_score = torch.sigmoid(seg_score)
        return act_score, bkg_score, seg_score
