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


class AttentionUnit(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(AttentionUnit, self).__init__()
        self.atte = nn.Sequential(nn.Conv1d(feat_dim, hidden_dim, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(hidden_dim, 1, 1), nn.Sigmoid())

    def forward(self, feat):
        return self.atte(feat)


class Model(nn.Module):
    def __init__(self, num_classes, hidden_dim, factor):
        super(Model, self).__init__()
        self.factor = factor
        self.rgb_ccm = CCM(1024)
        self.flow_ccm = CCM(1024)
        self.rgb_atte = AttentionUnit(1024, hidden_dim)
        self.flow_atte = AttentionUnit(1024, hidden_dim)
        self.fusion = nn.Sequential(nn.Conv1d(2048, hidden_dim, 3, padding=1), nn.ReLU())
        self.cls = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(),
                                 nn.Conv1d(hidden_dim, num_classes, 1))
        self.apply(weights_init)

    def forward(self, x):
        # [N, D, T]
        rgb, flow = x[:, :, :1024].transpose(-2, -1).contiguous(), x[:, :, 1024:].transpose(-2, -1).contiguous()
        rgb_feat = self.rgb_ccm(rgb, flow)
        flow_feat = self.flow_ccm(flow, rgb)
        # [N, T, 1]
        rgb_atte = self.rgb_atte(rgb_feat).transpose(-2, -1).contiguous()
        flow_atte = self.flow_atte(flow_feat).transpose(-2, -1).contiguous()
        sas = ((rgb_atte + flow_atte) / 2)

        feat = self.fusion(torch.cat((rgb_feat, flow_feat), dim=1))
        # [N, T, C]
        cas = self.cls(feat).transpose(-2, -1).contiguous()
        cas_score = torch.softmax(cas, dim=-1)
        #  [N, C]
        act_score = torch.softmax(torch.topk(cas, k=max(cas.shape[1] // self.factor, 1), dim=1)[0].mean(dim=1), dim=-1)

        seg_score = sas * cas_score
        return rgb_atte, flow_atte, act_score, seg_score, cas_score, sas.squeeze(dim=-1)


def multiple_loss(act_score, label):
    act_num = torch.sum(label, dim=-1, keepdim=True)
    act_num = torch.where(torch.eq(act_num, 0.0), torch.ones_like(act_num), act_num)
    act_label = label / act_num
    loss = ((-act_label * torch.log(act_score)).sum(dim=-1)).mean()
    return loss


def mutual_loss(rgb_atte, flow_atte):
    return (F.mse_loss(rgb_atte, flow_atte.detach()) + F.mse_loss(flow_atte, rgb_atte.detach())) / 2


def norm_loss(rgb_atte, flow_atte):
    return (rgb_atte.abs().mean() + flow_atte.abs().mean()) / 2


def supp_loss(cas_score, sas_score, label):
    score = cas_score * sas_score.unsqueeze(dim=-1)
    act_score = torch.softmax(torch.topk(score, k=max(score.shape[1] // 8, 1), dim=1)[0].mean(dim=1), dim=-1)

    act_num = torch.sum(label, dim=-1, keepdim=True)
    act_num = torch.where(torch.eq(act_num, 0.0), torch.ones_like(act_num), act_num)
    act_label = label / act_num

    loss = ((-act_label * torch.log(act_score)).sum(dim=-1)).mean()
    return loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch.nn.init.kaiming_uniform_(m.weight)
        if type(m.bias) != type(None):
            m.bias.data.fill_(0)
