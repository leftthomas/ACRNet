import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes, feat_dim=2048):
        super(Model, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=2),
                                  nn.ReLU(inplace=True), nn.Dropout(),
                                  nn.Conv1d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=2),
                                  nn.ReLU(inplace=True), nn.Dropout())
        self.proxy = nn.Parameter(torch.empty(1, num_classes, feat_dim))
        nn.init.xavier_uniform_(self.proxy)

    def forward(self, x):
        # [N, D, T]
        out = self.conv(x.permute(0, 2, 1))
        rgb_out, flow_out = torch.chunk(out, 2, dim=1)
        rgb_out, flow_out = F.normalize(rgb_out, dim=1), F.normalize(flow_out, dim=1)
        rgb_proxy, flow_proxy = torch.chunk(self.proxy, 2, dim=-1)
        rgb_proxy, flow_proxy = F.normalize(rgb_proxy, dim=-1), F.normalize(flow_proxy, dim=-1)

        # [N, T, C]
        rgb_score = torch.softmax(torch.matmul(rgb_proxy, rgb_out).permute(0, 2, 1) / 0.07, dim=-1)
        flow_score = torch.softmax(torch.matmul(flow_proxy, flow_out).permute(0, 2, 1) / 0.07, dim=-1)

        rgb_sim = torch.softmax(torch.matmul(rgb_out.permute(0, 2, 1), rgb_out), dim=-1)
        flow_sim = torch.softmax(torch.matmul(flow_out.permute(0, 2, 1), flow_out), dim=-1)
        rgb_score = (rgb_sim.unsqueeze(dim=-1) * rgb_score.unsqueeze(dim=1)).sum(dim=-2)
        flow_score = (flow_sim.unsqueeze(dim=-1) * flow_score.unsqueeze(dim=1)).sum(dim=-2)
        seg_score = (rgb_score + flow_score) / 2

        # [N, C]
        act_score = torch.mean(torch.topk(seg_score, k=min(10, seg_score.shape[1]), dim=1, largest=True)[0], dim=1)
        bkg_score = torch.mean(torch.topk(seg_score, k=min(10, seg_score.shape[1]), dim=1, largest=False)[0], dim=1)

        return act_score, bkg_score, seg_score
