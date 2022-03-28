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
    def __init__(self, num_classes, hidden_dim, factor):
        super(Model, self).__init__()

        self.factor = factor
        self.cas_rgb_encoder = CCM(1024)
        self.cas_flow_encoder = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=3, padding=1), nn.ReLU())
        self.cas_rgb = nn.Conv1d(1024, num_classes, kernel_size=1)
        self.cas_flow = nn.Conv1d(1024, num_classes, kernel_size=1)

        self.sas_rgb_encoder = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=3, padding=1, bias=False), nn.ReLU())
        self.sas_flow_encoder = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=3, padding=1, bias=False), nn.ReLU())
        self.sas_rgb = nn.Conv1d(1024, 1, kernel_size=1)
        self.sas_flow = nn.Conv1d(1024, 1, kernel_size=1)

    def forward(self, x):
        # [N, D, T]
        rgb, flow = x[:, :, :1024].transpose(-1, -2).contiguous(), x[:, :, 1024:].transpose(-1, -2).contiguous()

        cas_rgb_feat, cas_flow_feat = self.cas_rgb_encoder(rgb, flow), self.cas_flow_encoder(flow)
        # [N, T, C], class activation sequence
        cas_rgb = self.cas_rgb(cas_rgb_feat).transpose(-1, -2).contiguous()
        cas_flow = self.cas_flow(cas_flow_feat).transpose(-1, -2).contiguous()
        cas = (cas_rgb + cas_flow) / 2
        cas_score = torch.softmax(cas, dim=-1)

        sas_rgb_feat, sas_flow_feat = self.sas_rgb_encoder(rgb), self.sas_flow_encoder(flow)
        # [N, T, 1], segment activation sequence
        sas_rgb = self.sas_rgb(sas_rgb_feat).transpose(-1, -2).contiguous()
        sas_rgb_score = torch.sigmoid(sas_rgb)
        sas_flow = self.sas_flow(sas_flow_feat).transpose(-1, -2).contiguous()
        sas_flow_score = torch.sigmoid(sas_flow)

        seg_score = (cas_score + sas_rgb_score + sas_flow_score) / 3

        act_index = seg_score.topk(k=max(seg_score.shape[1] // self.factor, 1), dim=1)[1]
        bkg_index = seg_score.topk(k=max(seg_score.shape[1] // self.factor, 1), dim=1, largest=False)[1]
        # [N, C], action classification score is aggregated by cas
        act_score = torch.softmax(torch.gather(cas, dim=1, index=act_index).mean(dim=1), dim=-1)
        bkg_score = torch.softmax(torch.gather(cas, dim=1, index=bkg_index).mean(dim=1), dim=-1)

        normed_rgb = F.normalize(cas_rgb_feat, p=2, dim=1)
        normed_flow = F.normalize(flow, p=2, dim=1)
        # [N, T, T]
        rgb_graph = torch.matmul(normed_rgb.transpose(-1, -2).contiguous(), normed_rgb)
        flow_graph = torch.matmul(normed_flow.transpose(-1, -2).contiguous(), normed_flow)

        return act_score, bkg_score, sas_rgb_score.squeeze(dim=-1), sas_flow_score.squeeze(
            dim=-1), act_index, seg_score, rgb_graph, flow_graph


def sas_label(act_index, num_seg, label):
    masks = []
    for i in range(act_index.shape[0]):
        pos_index = act_index[i][:, label[i].bool()].flatten()
        mask = torch.zeros(num_seg, device=act_index.device)
        mask[pos_index] = 1.0
        masks.append(mask)
    return torch.stack(masks)


def divide_label(label):
    pos_num = label.sum(dim=-1)
    neg_num = (1.0 - label).sum(dim=-1)
    # avoid divide by zero
    pos_num = torch.where(torch.eq(pos_num, 0.0), torch.ones_like(pos_num), pos_num)
    neg_num = torch.where(torch.eq(neg_num, 0.0), torch.ones_like(neg_num), neg_num)
    return pos_num, neg_num


def graph_consistency(rgb_graph, flow_graph):
    return F.mse_loss(rgb_graph, flow_graph)


def cross_entropy(act_score, bkg_score, label, eps=1e-8):
    act_num, bkg_num = divide_label(label)
    act_loss = (-(label * torch.log(act_score + eps)).sum(dim=-1) / act_num).mean(dim=0)
    bkg_loss = (-torch.log(1.0 - bkg_score + eps)).mean()
    return act_loss + bkg_loss


# ref: Weakly Supervised Action Selection Learning in Video (CVPR 2021)
def generalized_cross_entropy(score, label, q=0.7, eps=1e-8):
    pos_num, neg_num = divide_label(label)
    pos_loss = ((((1.0 - (score + eps) ** q) / q) * label).sum(dim=-1) / pos_num).mean(dim=0)
    neg_loss = ((((1.0 - (1.0 - score + eps) ** q) / q) * (1.0 - label)).sum(dim=-1) / neg_num).mean(dim=0)
    return pos_loss + neg_loss
