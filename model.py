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
    def __init__(self, num_classes):
        super(Model, self).__init__()

        self.rgb_encoder = CCM(1024)
        self.flow_encoder = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=3, padding=1), nn.ReLU())
        self.rgb_cas = nn.Conv1d(1024, num_classes, kernel_size=1)
        self.flow_cas = nn.Conv1d(1024, num_classes, kernel_size=1)

    def forward(self, x):
        # [N, D, T]
        rgb = x[:, :, :1024].transpose(-1, -2).contiguous()
        flow = x[:, :, 1024:].transpose(-1, -2).contiguous()
        rgb_feat = self.rgb_encoder(rgb, flow)
        flow_feat = self.flow_encoder(flow)
        # [N, T, C], class activation sequence
        rgb_cas = torch.softmax(self.rgb_cas(rgb_feat).transpose(-1, -2).contiguous(), dim=-1)
        flow_cas = torch.softmax(self.flow_cas(flow_feat).transpose(-1, -2).contiguous(), dim=-1)
        seg_score = (rgb_cas + flow_cas) / 2

        ori_rgb = F.normalize(rgb, p=2, dim=1)
        normed_rgb = F.normalize(rgb_feat, p=2, dim=1)
        normed_flow = F.normalize(flow, p=2, dim=1)
        # [N, T, T]
        ori_rgb_graph = torch.matmul(ori_rgb.transpose(-1, -2).contiguous(), ori_rgb)
        rgb_graph = torch.matmul(normed_rgb.transpose(-1, -2).contiguous(), normed_rgb)
        flow_graph = torch.matmul(normed_flow.transpose(-1, -2).contiguous(), normed_flow)

        return rgb_cas, flow_cas, seg_score, ori_rgb_graph, rgb_graph, flow_graph


def cross_entropy(score, label, eps=1e-8):
    num = label.sum(dim=-1)
    # avoid divide by zero
    num = torch.where(torch.eq(num, 0.0), torch.ones_like(num), num)
    act_loss = (-(label * torch.log(score + eps)).sum(dim=-1) / num).mean(dim=0)
    return act_loss


def graph_consistency(rgb_graph, flow_graph):
    return F.mse_loss(rgb_graph, flow_graph)


def split_pos_neg(cas_score):
    n, t, c = cas_score.shape
    # [N*C, T]
    cas_score = cas_score.transpose(-1, -2).contiguous().view(-1, t)
    sort_value, sort_index = torch.sort(cas_score, dim=-1, descending=True)
    mask = torch.zeros_like(cas_score)
    # the index of the largest value is inited as positive
    mask[torch.arange(mask.shape[0], device=mask.device), sort_index[:, 0]] = 1
    pos_sum, neg_sum = sort_value[:, 0], sort_value[:, -1]
    pos_num, neg_num = torch.ones_like(pos_sum), torch.ones_like(neg_sum)
    for i in range(1, t - 1):
        index, value = sort_index[:, i], sort_value[:, i]
        pos_center = pos_sum / pos_num
        neg_center = neg_sum / neg_num
        pos_distance = torch.abs(value - pos_center)
        neg_distance = torch.abs(value - neg_center)
        condition = torch.le(pos_distance, neg_distance)
        pos_list = torch.where(condition, value, torch.zeros_like(value))
        neg_list = torch.where(~condition, value, torch.zeros_like(value))
        # update centers
        pos_num = pos_num + condition.float()
        neg_num = neg_num + (~condition).float()
        pos_sum = pos_sum + pos_list
        neg_sum = neg_sum + neg_list
        # update mask
        mask[torch.arange(mask.shape[0], device=mask.device), index] = condition.float()
    # [N, T, C]
    mask = mask.view(n, c, t).transpose(-1, -2).contiguous()
    return mask


def mutual_entropy(cas_score, label):
    pos_mask = split_pos_neg(cas_score)
    neg_mask = 1.0 - pos_mask
    pos_mask = F.dropout(pos_mask, p=0.5, training=True)
    neg_mask = F.dropout(neg_mask, p=0.5, training=True)

    pos_num = pos_mask.sum(dim=1)
    pos_num = torch.where(torch.eq(pos_num, 0.0), torch.ones_like(pos_num), pos_num)
    neg_num = neg_mask.sum(dim=1)
    neg_num = torch.where(torch.eq(neg_num, 0.0), torch.ones_like(neg_num), neg_num)
    pos = (cas_score * pos_mask).sum(dim=1) / pos_num
    neg = (cas_score * neg_mask).sum(dim=1) / neg_num
    pos_loss = cross_entropy(pos, label)
    neg_loss = cross_entropy(1.0 - neg, label)
    loss = pos_loss + neg_loss
    return loss


def fuse_act_score(cas_score):
    mask = split_pos_neg(cas_score)
    pos_num = mask.sum(dim=1)
    pos_num = torch.where(torch.eq(pos_num, 0.0), torch.ones_like(pos_num), pos_num)
    pos = (cas_score * mask).sum(dim=1) / pos_num
    # obtain the threshold
    ths = torch.where(mask.bool(), cas_score, torch.ones_like(cas_score))
    ths = torch.amin(ths, dim=1)
    return pos, ths
