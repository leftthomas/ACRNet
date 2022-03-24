import torch
import torch.nn as nn
import torch.nn.functional as F


# ref: Cross-modal Consensus Network for Weakly Supervised Temporal Action Localization (ACM MM 2021)
class CCM(nn.Module):
    def __init__(self, feat_dim):
        super(CCM, self).__init__()
        self.global_context = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(feat_dim, feat_dim, 3, padding=1),
                                            nn.ReLU())
        self.local_context = nn.Sequential(nn.Conv1d(feat_dim, feat_dim, 3, padding=1), nn.ReLU())

    def forward(self, global_feat, local_feat):
        global_context = self.global_context(global_feat)
        local_context = self.local_context(local_feat)
        enhanced_feat = torch.sigmoid(global_context * local_context) * global_feat
        return enhanced_feat


class Model(nn.Module):
    def __init__(self, num_classes, factor):
        super(Model, self).__init__()

        self.factor = factor
        self.rgb_encoder = CCM(1024)
        self.flow_encoder = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=3, padding=1), nn.ReLU())
        self.cas = nn.Conv1d(1024, num_classes, kernel_size=1)

    def forward(self, x):
        # [N, D, T]
        rgb = x[:, :, :1024].transpose(-1, -2).contiguous()
        flow = x[:, :, 1024:].transpose(-1, -2).contiguous()
        rgb_feat = self.rgb_encoder(rgb, flow)
        flow_feat = self.flow_encoder(flow)
        # [N, T, C], class activation sequence
        rgb_cas = torch.softmax(self.cas(rgb_feat).transpose(-1, -2).contiguous(), dim=-1)
        flow_cas = torch.softmax(self.cas(flow_feat).transpose(-1, -2).contiguous(), dim=-1)
        seg_score = (rgb_cas + flow_cas) / 2

        k = max(seg_score.shape[1] // self.factor, 1)
        act_score = seg_score.topk(k, dim=1)[0].mean(dim=1)

        ori_rgb = F.normalize(rgb, p=2, dim=1)
        normed_rgb = F.normalize(rgb_feat, p=2, dim=1)
        normed_flow = F.normalize(flow, p=2, dim=1)
        # [N, T, T]
        ori_rgb_graph = torch.matmul(ori_rgb.transpose(-1, -2).contiguous(), ori_rgb)
        rgb_graph = torch.matmul(normed_rgb.transpose(-1, -2).contiguous(), normed_rgb)
        flow_graph = torch.matmul(normed_flow.transpose(-1, -2).contiguous(), normed_flow)

        return act_score, rgb_cas, flow_cas, seg_score, ori_rgb_graph, rgb_graph, flow_graph


def cross_entropy(score, label, eps=1e-8):
    num = label.sum(dim=-1)
    # avoid divide by zero
    num = torch.where(torch.eq(num, 0.0), torch.ones_like(num), num)
    act_loss = (-(label * torch.log(score + eps)).sum(dim=-1) / num).mean(dim=0)
    return act_loss


def graph_consistency(rgb_graph, flow_graph):
    return F.mse_loss(rgb_graph, flow_graph)


def split_pos_neg(ref_cas):
    sort_value, sort_index = torch.sort(ref_cas, descending=True)
    pos_list, neg_list = [], []
    if len(ref_cas) == 1:
        return sort_index, sort_index
    if len(ref_cas) == 2:
        return sort_index[[0]], sort_index[[-1]]
    # else
    pos_list.append(sort_index[0])
    neg_list.append(sort_index[-1])
    for i, value in zip(sort_index[1:-1], sort_value[1:-1]):
        pos_center = torch.mean(ref_cas[torch.stack(pos_list)])
        neg_center = torch.mean(ref_cas[torch.stack(neg_list)])
        pos_distance = torch.abs(value - pos_center)
        neg_distance = torch.abs(value - neg_center)
        if pos_distance <= neg_distance:
            pos_list.append(i)
        else:
            neg_list.append(i)
    return torch.stack(pos_list), torch.stack(neg_list)


def mutual_entropy(base_cas, ref_cas, label):
    poss, negs = [], []
    for i in range(base_cas.shape[0]):
        pos_list, neg_list = [], []
        for j in range(base_cas.shape[-1]):
            pos_index, neg_index = split_pos_neg(ref_cas[i, :, j])
            pos_value = base_cas[i, pos_index, j].mean()
            pos_list.append(pos_value)
            neg_value = base_cas[i, neg_index, j].mean()
            neg_list.append(neg_value)
        pos, neg = torch.stack(pos_list), torch.stack(neg_list)
        poss.append(pos)
        negs.append(neg)
    poss, negs = torch.stack(poss), torch.stack(negs)
    pos_loss = cross_entropy(poss, label)
    neg_loss = cross_entropy(1.0 - negs, label)
    loss = pos_loss + neg_loss
    return loss


def fuse_act_score(base_cas, ref_cas):
    poss, ths = [], []
    for i in range(base_cas.shape[0]):
        pos_list, pos_th = [], []
        for j in range(base_cas.shape[-1]):
            pos_index, neg_index = split_pos_neg(ref_cas[i, :, j])
            pos_value = base_cas[i, pos_index, j].mean()
            min_value = torch.amin(base_cas[i, pos_index, j])
            pos_list.append(pos_value)
            pos_th.append(min_value)
        poss.append(torch.stack(pos_list))
        ths.append(torch.stack(pos_th))
    poss, ths = torch.stack(poss), torch.stack(ths)
    return poss, ths
