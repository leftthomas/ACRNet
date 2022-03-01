import math

import torch
import torch.nn as nn


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
    def __init__(self, num_classes, feat_dim=2048, num_blocks=3):
        super(Model, self).__init__()
        self.conv_list = nn.ModuleList([GraphConvolution(feat_dim, feat_dim) for _ in range(num_blocks)])
        self.cls = nn.Conv1d(feat_dim, num_classes, kernel_size=1)

    def forward(self, x):
        for conv in self.conv_list:
            # construct sim matrix [N, T, T]
            sim_matrix = xxx
            # update x
            x = torch.relu(conv(x, sim_matrix))

        # [N, T, C]
        seg_score = torch.softmax(self.cls(x), dim=-1)
        # [N, C]
        return act_score, seg_score
