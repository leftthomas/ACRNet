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
        rgb_cas = self.cas(rgb_feat).transpose(-1, -2).contiguous()
        flow_cas = self.cas(flow_feat).transpose(-1, -2).contiguous()
        cas = (rgb_cas + flow_cas) / 2
        seg_score = torch.softmax(cas, dim=-1)

        k = max(cas.shape[1] // self.factor, 1)
        act_score = torch.softmax(cas.topk(k, dim=1)[0].mean(dim=1), dim=-1)

        ori_rgb = F.normalize(rgb, p=2, dim=1)
        normed_rgb = F.normalize(rgb_feat, p=2, dim=1)
        normed_flow = F.normalize(flow, p=2, dim=1)
        # [N, T, T]
        ori_rgb_graph = torch.matmul(ori_rgb.transpose(-1, -2).contiguous(), ori_rgb)
        rgb_graph = torch.matmul(normed_rgb.transpose(-1, -2).contiguous(), normed_rgb)
        flow_graph = torch.matmul(normed_flow.transpose(-1, -2).contiguous(), normed_flow)

        return act_score, rgb_cas, flow_cas, seg_score, ori_rgb_graph, rgb_graph, flow_graph


def cross_entropy(score, label, eps=1e-8, reduce=True):
    num = label.sum(dim=-1)
    # avoid divide by zero
    num = torch.where(torch.eq(num, 0.0), torch.ones_like(num), num)
    act_loss = (-(label * torch.log(score + eps)).sum(dim=-1) / num)
    if reduce:
        act_loss = act_loss.mean(dim=0)
    return act_loss


def graph_consistency(rgb_graph, flow_graph):
    return F.mse_loss(rgb_graph, flow_graph)


def count_pos(ref_cas):
    # [N, T, C]
    ref_cas = torch.softmax(ref_cas.detach(), dim=-1)
    # [N, C]
    k = ref_cas.sum(dim=1)
    k_max = ref_cas.amax(dim=1)
    # avoid divided by zero
    k_max = torch.where(torch.eq(k_max, 0.0), torch.ones_like(k_max), k_max)
    k = k / k_max
    return k


def mutual_entropy(base_cas, ref_cas, label, factor):
    k = count_pos(ref_cas)
    pos_loss, neg_loss = 0.0, 0.0
    for i in range(base_cas.shape[0]):
        pos_list, neg_list = [], []
        for j in range(base_cas.shape[-1]):
            # avoid zero
            pos_k = max(int(k[i, j]), 1)
            pos_value = base_cas[i, :, j].topk(k=pos_k)[0]
            pos_num = min(base_cas.shape[1] // factor, pos_k)
            # hard positive
            pos_value = pos_value[-pos_num::].mean()
            pos_list.append(pos_value)

            neg_value = base_cas[i, :, j].topk(k=base_cas.shape[1] - pos_k, largest=False)[0]
            neg_num = min(base_cas.shape[1] // factor, base_cas.shape[1] - pos_k)
            # hard negative
            neg_value = neg_value[-neg_num::].mean()
            neg_list.append(neg_value)
        pos, neg = torch.softmax(torch.stack(pos_list), dim=-1), torch.softmax(torch.stack(neg_list), dim=-1)
        pos_loss = pos_loss + cross_entropy(pos, label[i, :], reduce=False)
        neg_loss = neg_loss + cross_entropy(1.0 - neg, label[i, :], reduce=False)
    loss = (pos_loss + neg_loss) / base_cas.shape[0]
    return loss


def fuse_act_score(base_cas, ref_cas, factor):
    k = count_pos(ref_cas)
    poss = []
    for i in range(base_cas.shape[0]):
        pos_list = []
        for j in range(base_cas.shape[-1]):
            pos_k = max(int(k[i, j]), 1)
            pos_value = base_cas[i, :, j].topk(k=pos_k)[0]
            pos_num = min(base_cas.shape[1] // factor, pos_k)
            # easy positive
            pos_value = pos_value[:pos_num].mean()
            pos_list.append(pos_value)
        pos = torch.stack(pos_list)
        poss.append(pos)
    poss = torch.stack(poss)
    return poss
