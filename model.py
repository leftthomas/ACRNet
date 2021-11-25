import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_classes, feat_dim=2048):
        super(Model, self).__init__()

        self.conv = nn.Sequential(nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.fc = nn.Conv1d(feat_dim, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(x.permute(0, 2, 1))
        # [N, T, D]
        feat = out.permute(0, 2, 1)
        # [N, T, C]
        score = self.fc(out).permute(0, 2, 1)
        return feat, score
