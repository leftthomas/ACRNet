import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_classes, feat_dim=2048):
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.conv = nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=2)
        self.fc = nn.Conv1d(feat_dim, 2 * num_classes, kernel_size=1, groups=2)
        # introduce dropout for robust, eliminate the noise
        self.drop_out = nn.Dropout()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # [N, D, T]
        x = x.permute(0, 2, 1)
        out = self.drop_out(self.relu(self.conv(x)))
        # [N, T, C]
        seg_score = self.fc(out).permute(0, 2, 1)
        seg_score = (seg_score[:, :, :self.num_classes] + seg_score[:, :, self.num_classes:]) / 2

        k = max(int(seg_score.shape[1] * 0.1), 1)
        top_score, top_idx = seg_score.topk(k=k, dim=1, largest=True)
        bottom_score, bottom_idx = seg_score.topk(k=k, dim=1, largest=False)

        # [N, C]
        act_score = torch.sigmoid(torch.mean(top_score, dim=1))
        bkg_score = torch.sigmoid(torch.mean(bottom_score, dim=1))
        # [N, T, C]
        seg_score = torch.sigmoid(seg_score)
        return act_score, bkg_score, seg_score
