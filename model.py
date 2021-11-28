import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes, feat_dim=2048):
        super(Model, self).__init__()

        self.conv = nn.Sequential(nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.proxy = nn.Parameter(torch.empty(num_classes, feat_dim, 1))
        # introduce dropout for robust, eliminate the noise
        self.drop_out = nn.Dropout(p=0.7)
        nn.init.kaiming_uniform_(self.proxy, a=math.sqrt(5))

    def forward(self, x):
        out = self.conv(x.permute(0, 2, 1))
        # [N, T, D]
        feat = out.permute(0, 2, 1)
        # [N, T, C]
        seg_score = F.conv1d(self.drop_out(out), self.proxy).permute(0, 2, 1)
        topk_score = seg_score.topk(k=max(int(seg_score.shape[1] * 0.1), 1), dim=1)[0]
        # [N, C]
        video_score = torch.softmax(torch.mean(topk_score, dim=1), dim=-1)
        # [N, T, C]
        seg_score = torch.softmax(seg_score, dim=-1)
        return feat, video_score, seg_score
