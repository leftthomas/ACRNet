import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(1, in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(-1))
        nn.init.uniform_(self.weight, -stdv, stdv)

    def forward(self, x, adj):
        # [N, T, D_i]  x  [1, D_i, D_o]
        support = torch.matmul(x, self.weight)
        # [N, T, T]  x  [N, T, D_o]
        output = torch.matmul(adj, support)
        return output


class Model(nn.Module):
    def __init__(self, num_classes, feat_dim=2048, th=0.2):
        super(Model, self).__init__()
        self.conv = GraphConvolution(feat_dim, feat_dim)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()
        self.cls = nn.Conv1d(feat_dim, num_classes, kernel_size=1)
        self.th = th

    def forward(self, x):
        # construct sim matrix [N, T, T]
        sim_matrix = torch.matmul(F.normalize(x, dim=-1), F.normalize(x.permute(0, 2, 1), dim=1))
        sim_min, sim_max = torch.aminmax(sim_matrix, dim=-1, keepdim=True)
        sim_matrix = (sim_matrix - sim_min) / (sim_max - sim_min)

        sim = torch.diagonal(torch.diff(torch.ge(sim_matrix, self.th), dim=1), dim1=-2, dim2=-1)
        a = torch.arange(sim.shape[-1]).to(sim.device).expand(sim.shape[0], -1)[torch.ne(sim, 0)]
        out = F.normalize(self.relu(self.conv(x, sim_matrix)), dim=-1)

        seg_score = torch.softmax(self.cls(self.drop(out).permute(0, 2, 1)).permute(0, 2, 1), dim=-1)
        # [N, C]
        act_score = torch.softmax(torch.mean(torch.topk(seg_score, k=max(1, seg_score.shape[1] // 8),
                                                        dim=1)[0], dim=1), dim=-1)
        bkg_score = torch.softmax(torch.mean(torch.topk(seg_score, k=max(1, seg_score.shape[1] // 8),
                                                        dim=1, largest=False)[0], dim=1), dim=-1)
        return act_score, bkg_score, seg_score
