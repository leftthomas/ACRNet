import torch
import torch.nn as nn


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


class AttentionUnit(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(AttentionUnit, self).__init__()
        self.atte = nn.Sequential(nn.Conv1d(feat_dim, hidden_dim, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(hidden_dim, 1, 1), nn.Sigmoid())

    def forward(self, feat):
        return self.atte(feat)


class Model(nn.Module):
    def __init__(self, num_classes, hidden_dim, factor, temperature):
        super(Model, self).__init__()

        self.factor = factor
        self.rgb_ccm = CCM(1024)
        self.flow_ccm = CCM(1024)
        self.rgb_atte = AttentionUnit(1024, hidden_dim)
        self.flow_atte = AttentionUnit(1024, hidden_dim)
        self.cls = nn.Sequential(nn.Conv1d(2048, hidden_dim, 3, padding=1), nn.ReLU(),
                                 nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(),
                                 nn.Conv1d(hidden_dim, num_classes, 1))

    def forward(self, x):
        # [N, D, T]
        rgb, flow = x[:, :, :1024].transpose(-2, -1).contiguous(), x[:, :, 1024:].transpose(-2, -1).contiguous()
        rgb_feat = self.rgb_ccm(rgb, flow)
        flow_feat = self.flow_ccm(flow, rgb)
        # [N, 1, T]
        rgb_atte = self.rgb_atte(rgb_feat)
        flow_atte = self.flow_atte(flow_feat)
        # [N, T, 1]
        sas_score = ((rgb_atte + flow_atte) / 2).transpose(-2, -1).contiguous()

        # [N, T, C]
        cas = self.cls(torch.cat((rgb_feat, flow_feat), dim=1)).transpose(-2, -1).contiguous()
        cas_score = torch.softmax(cas, dim=-1)

        seg_score = cas * sas_score

        # [N, C], action score is aggregated by cas score
        k = max(cas_score.shape[1] // self.factor, 1)
        f_act_score = torch.softmax(cas.topk(k=k, dim=1)[0].mean(dim=1), dim=-1)
        act_score = torch.softmax(seg_score.topk(k=k, dim=1)[0].mean(dim=1), dim=-1)

        # [N, T, T]
        graph = torch.matmul(rgb_feat.transpose(-2, -1).contiguous(), flow_feat)

        return act_score, f_act_score, cas_score, sas_score.squeeze(dim=-1), seg_score, graph


def multiple_loss(act_score, label, eps=1e-8):
    act_num = torch.sum(label, dim=-1, keepdim=True)
    act_num = torch.where(torch.eq(act_num, 0.0), torch.ones_like(act_num), act_num)
    act_label = label / act_num
    loss = (-(act_label * torch.log(act_score + eps)).sum(dim=-1)).mean()
    return loss


def norm_loss(sas_score):
    return sas_score.abs().mean()
