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


def construct_sim(x):
    sim_matrix = torch.matmul(F.normalize(x, dim=-1), F.normalize(x.permute(0, 2, 1), dim=1))
    sim_min = torch.amin(sim_matrix, dim=-1, keepdim=True)
    sim_max = torch.amax(sim_matrix, dim=-1, keepdim=True)
    sim_inter = sim_max - sim_min
    # avoid div by zero
    sim_inter = torch.where(torch.eq(sim_inter, 0), torch.ones_like(sim_inter), sim_inter)
    sim_matrix = (sim_matrix - sim_min) / sim_inter
    return sim_matrix


class Model(nn.Module):
    def __init__(self, num_classes, feat_dim=2048, num_blocks=3):
        super(Model, self).__init__()
        self.conv_list = nn.Sequential(*[GraphConvolution(feat_dim, feat_dim) for _ in range(num_blocks)])
        self.cls = nn.Conv1d(feat_dim, num_classes, kernel_size=1)

    def forward(self, x):
        for conv in self.conv_list:
            # construct sim matrix [N, T, T]
            sim_matrix = torch.softmax(construct_sim(x), dim=-1)
            # update x
            x = torch.relu(conv(x, sim_matrix))

        sim_matrix = construct_sim(x)
        sim_threshold = torch.mean(sim_matrix, dim=-1, keepdim=True)
        sim_filter = torch.diagonal(torch.ge(sim_matrix, sim_threshold), dim1=-2, dim2=-1).unsqueeze(dim=-1)
        act_count = torch.sum(sim_filter, dim=1)
        bkg_count = torch.sum(~sim_filter, dim=1)
        # avoid div by zero
        act_count = torch.where(torch.eq(act_count, 0), torch.ones_like(act_count), act_count)
        bkg_count = torch.where(torch.eq(bkg_count, 0), torch.ones_like(bkg_count), bkg_count)
        # [N, T, C]
        seg_score = torch.softmax(self.cls(x.permute(0, 2, 1)).permute(0, 2, 1), dim=-1)
        # [N, C]
        act_score = torch.where(sim_filter.expand_as(seg_score), seg_score, torch.zeros_like(seg_score)).sum(
            dim=1) / act_count
        bkg_score = torch.where(~sim_filter.expand_as(seg_score), seg_score, torch.zeros_like(seg_score)).sum(
            dim=1) / bkg_count
        return act_score, seg_score
