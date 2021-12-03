import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes, select_ratio, temperature, feat_dim=2048):
        super(Model, self).__init__()

        self.select_ratio = select_ratio
        self.temperature = temperature
        self.conv = nn.Sequential(nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.proxy = nn.Parameter(torch.empty(num_classes, feat_dim, 1))
        # introduce dropout for robust, eliminate the noise
        self.drop_out = nn.Dropout(p=0.7)
        nn.init.kaiming_uniform_(self.proxy, a=math.sqrt(5))

    def forward(self, x):
        out = self.drop_out(self.conv(x.permute(0, 2, 1)))
        # [N, T, D]
        feat = out.permute(0, 2, 1)
        # [N, T, C]
        seg_score = F.conv1d(out, self.proxy).permute(0, 2, 1)

        k = max(int(seg_score.shape[1] * self.select_ratio), 1)
        top_score, top_idx = seg_score.topk(k=k, dim=1, largest=True)
        bottom_score, bottom_idx = seg_score.topk(k=k, dim=1, largest=False)

        # [N, C]
        act_score = torch.softmax(torch.mean(top_score, dim=1), dim=-1)
        bkg_score = torch.softmax(torch.mean(bottom_score, dim=1), dim=-1)
        # [N, K*C, D]
        act_feat = torch.take_along_dim(feat, torch.flatten(top_idx, start_dim=1).unsqueeze(dim=-1), dim=1)
        bkg_feat = torch.take_along_dim(feat, torch.flatten(bottom_idx, start_dim=1).unsqueeze(dim=-1), dim=1)
        # [N, C]
        act_norm = torch.mean(torch.norm(act_feat.view(act_feat.shape[0], k, -1, act_feat.shape[-1]), p=2, dim=-1),
                              dim=1)
        bkg_norm = torch.mean(torch.norm(bkg_feat.view(bkg_feat.shape[0], k, -1, bkg_feat.shape[-1]), p=2, dim=-1),
                              dim=1)
        # [N, T, C]
        seg_score = torch.softmax(seg_score, dim=-1)
        return act_norm, bkg_norm, feat, act_score, bkg_score, seg_score

# if __name__ == '__main__':
#     import glob
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     import pandas as pd
#     from tqdm import tqdm
#
#     sns.set()
#
#     thumos_features = sorted(glob.glob('/data/thumos14/features/test/*_rgb.npy'))
#     activitynet_features = sorted(glob.glob('/data/activitynet/features/training/*_rgb.npy'))
#     thumos_frames, activitynet_frames = [], []
#     for feature in tqdm(thumos_features):
#         thumos_frames.append(len(np.load(feature)))
#     for feature in tqdm(activitynet_features):
#         activitynet_frames.append(len(np.load(feature)))
#
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     axes[0].set_title('thumos14', fontsize=18)
#     axes[1].set_title('activitynet', fontsize=18)
#     sns.histplot(pd.DataFrame({'count': thumos_frames}), ax=axes[0])
#     sns.histplot(pd.DataFrame({'count': activitynet_frames}), ax=axes[1])
#     plt.savefig('result/dist.pdf', bbox_inches='tight', pad_inches=0.1)
