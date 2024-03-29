import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch.nn.init.kaiming_uniform_(m.weight)
        if not isinstance(m.bias, type(None)):
            m.bias.data.fill_(0)


# ref: Weakly-supervised Temporal Action Localization with Multi-head Cross-modal Attention (PRICAI 2022)
class MCA(nn.Module):
    def __init__(self, feat_dim, num_head=4):
        super(MCA, self).__init__()
        self.rgb_proj = nn.Parameter(torch.empty(num_head, feat_dim, feat_dim // num_head))
        self.flow_proj = nn.Parameter(torch.empty(num_head, feat_dim, feat_dim // num_head))
        self.atte = nn.Parameter(torch.empty(num_head, feat_dim // num_head, feat_dim // num_head))

        nn.init.uniform_(self.rgb_proj, -math.sqrt(feat_dim), math.sqrt(feat_dim))
        nn.init.uniform_(self.flow_proj, -math.sqrt(feat_dim), math.sqrt(feat_dim))
        nn.init.uniform_(self.atte, -math.sqrt(feat_dim // num_head), math.sqrt(feat_dim // num_head))
        self.num_head = num_head

    def forward(self, rgb, flow):
        rgb, flow = rgb.mT.contiguous(), flow.mT.contiguous()
        n, t, d = rgb.shape
        # [N, H, T, D/H]
        o_rgb = F.normalize(torch.matmul(rgb.unsqueeze(dim=1), self.rgb_proj), dim=-1)
        o_flow = F.normalize(torch.matmul(flow.unsqueeze(dim=1), self.flow_proj), dim=-1)
        # [N, H, T, T]
        atte = torch.matmul(torch.matmul(o_rgb, self.atte), o_flow.mT.contiguous())
        rgb_atte = torch.softmax(atte, dim=-1)
        flow_atte = torch.softmax(atte.mT.contiguous(), dim=-1)

        # [N, H, T, D/H]
        e_rgb = F.gelu(torch.matmul(rgb_atte, o_rgb))
        e_flow = F.gelu(torch.matmul(flow_atte, o_flow))
        # [N, T, D]
        f_rgb = torch.tanh(e_rgb.mT.reshape(n, t, -1).contiguous() + rgb)
        f_flow = torch.tanh(e_flow.mT.reshape(n, t, -1).contiguous() + flow)

        f_rgb, f_flow = f_rgb.mT.contiguous(), f_flow.mT.contiguous()
        return f_rgb, f_flow


# ref: Dual-Evidential Learning for Weakly-supervised Temporal Action Localization (ECCV 2022)
class BWA(nn.Module):
    def __init__(self, feat_dim):
        super(BWA, self).__init__()
        self.attn = nn.Sequential(nn.Conv1d(feat_dim, feat_dim, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.conv = nn.Sequential(nn.Conv1d(feat_dim, feat_dim, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, base_feat, ref_feat):
        channel_attn = self.conv(self.avg(base_feat))
        bit_attn = self.attn(ref_feat)
        filter_feat = torch.sigmoid(bit_attn * channel_attn) * base_feat
        return filter_feat


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()

        self.mca = MCA(1024)
        self.rgb_bwa = BWA(1024)
        self.flow_bwa = BWA(1024)
        self.cas_rgb_encoder = nn.Sequential(nn.Conv1d(1024, 1024, 3, padding=1), nn.ReLU(),
                                             nn.Conv1d(1024, num_classes, kernel_size=1))
        self.cas_flow_encoder = nn.Sequential(nn.Conv1d(1024, 1024, 3, padding=1), nn.ReLU(),
                                              nn.Conv1d(1024, num_classes, kernel_size=1))

        self.aas_rgb_encoder = nn.Sequential(nn.Conv1d(1024, 512, 1), nn.ReLU(), nn.Conv1d(512, 1, 1))
        self.aas_flow_encoder = nn.Sequential(nn.Conv1d(1024, 512, 1), nn.ReLU(), nn.Conv1d(512, 1, 1))

        # ref: A Hybrid Attention Mechanism for Weakly-Supervised Temporal Action Localization (AAAI 2021)
        if num_classes != 20:
            pool = nn.AvgPool1d(13, 1, padding=6, count_include_pad=True)
            self.cas_rgb_encoder.append(pool)
            self.cas_flow_encoder.append(pool)
            self.aas_rgb_encoder.append(pool)
            self.aas_flow_encoder.append(pool)

        self.apply(weights_init)

    def forward(self, x):
        # [N, D, T]
        x = x.mT.contiguous()
        rgb, flow = self.mca(x[:, :1024, :], x[:, 1024:, :])
        rgb, flow = self.rgb_bwa(rgb, flow), self.flow_bwa(flow, rgb)

        # [N, T, C], class activation sequence
        cas_rgb = self.cas_rgb_encoder(rgb).mT.contiguous()
        cas_flow = self.cas_flow_encoder(flow).mT.contiguous()
        cas = cas_rgb + cas_flow
        cas_score = torch.softmax(cas, dim=-1)
        # [N, T, 1], action activation sequence
        aas_rgb = torch.sigmoid(self.aas_rgb_encoder(rgb).mT.contiguous())
        aas_flow = torch.sigmoid(self.aas_flow_encoder(flow).mT.contiguous())
        aas_score = (aas_rgb + aas_flow) / 2
        # [N, T, C]
        seg_score = (cas_score + aas_score) / 2
        seg_mask = temporal_clustering(seg_score)
        seg_mask = mask_refining(seg_score, seg_mask, cas)

        # [N, C]
        act_score, bkg_score = calculate_score(seg_score, seg_mask, cas)
        return act_score, bkg_score, seg_score, seg_mask, aas_rgb, aas_flow


def temporal_clustering(seg_score):
    n, t, c = seg_score.shape
    # [N*C, T]
    seg_score = seg_score.mT.contiguous().view(-1, t)
    sort_value, sort_index = torch.sort(seg_score, dim=-1, descending=True, stable=True)
    mask = torch.zeros_like(seg_score)
    row_index = torch.arange(mask.shape[0], device=mask.device)
    # the index of the largest value is inited as positive
    mask[row_index, sort_index[:, 0]] = 1
    # [N*C]
    pos_sum, neg_sum = sort_value[:, 0], sort_value[:, -1]
    pos_num, neg_num = torch.ones_like(pos_sum), torch.ones_like(neg_sum)
    for i in range(1, t - 1):
        pos_center = pos_sum / pos_num
        neg_center = neg_sum / neg_num
        index, value = sort_index[:, i], sort_value[:, i]
        pos_distance = torch.abs(value - pos_center)
        neg_distance = torch.abs(value - neg_center)
        condition = torch.le(pos_distance, neg_distance)
        pos_list = torch.where(condition, value, torch.zeros_like(value))
        neg_list = torch.where(~condition, value, torch.zeros_like(value))
        # update centers
        pos_num = pos_num + condition.float() / (i + 1)
        pos_sum = pos_sum + pos_list / (i + 1)
        neg_num = neg_num + (~condition).float()
        neg_sum = neg_sum + neg_list
        # update mask
        mask[row_index, index] = condition.float()
    # [N, T, C]
    mask = mask.view(n, c, t).mT.contiguous()
    return mask


def mask_refining(seg_score, seg_mask, cas):
    n, t, c = seg_score.shape
    sort_value, sort_index = torch.sort(seg_score, dim=1, descending=True, stable=True)
    # [N, T]
    ranks = torch.arange(2, t + 2, device=seg_score.device).reciprocal().view(1, -1).expand(n, -1).contiguous()
    row_index = torch.arange(n, device=seg_score.device).view(-1, 1).expand(-1, t).contiguous()
    # [N, C]
    act_score = torch.zeros(n, c, device=seg_score.device)
    mean_score = torch.zeros(n, c, device=seg_score.device)

    for i in range(c):
        # [N, T]
        index, value = sort_index[:, :, i], sort_value[:, :, i]
        mask = seg_mask[:, :, i][row_index, index]
        cs = cas[:, :, i][row_index, index]
        rank = ranks * mask
        # [N]
        tmp_score = (cs * rank).sum(dim=-1) / torch.clamp_min(rank.sum(dim=-1), 1.0)
        act_score[:, i] = tmp_score
        for j in range(n):
            ref_score = tmp_score[j]
            ref_val = cs[j][mask[j].bool()]
            sort_val = value[j][mask[j].bool()]
            if ref_val.shape[0] > 0:
                cum_cnts = torch.arange(1, mask[j].sum() + 1, device=seg_score.device)
                cum_scores = torch.cumsum(ref_val, dim=-1) / cum_cnts
                tmp_mask = torch.ge(cum_scores, ref_score).long()
                mean_score[j, i] = sort_val[min(tmp_mask.sum() - 1, sort_val.shape[0] - 1)]
            else:
                mean_score[j, i] = 0.0
    max_mask = torch.ge(seg_score, mean_score.unsqueeze(dim=1)).float()
    refined_mask = seg_mask * max_mask
    return refined_mask


def calculate_score(seg_score, seg_mask, cas):
    n, t, c = seg_score.shape
    # [N*C, T]
    seg_score = seg_score.mT.contiguous().view(-1, t)
    sort_value, sort_index = torch.sort(seg_score, dim=-1, descending=True, stable=True)
    seg_mask = seg_mask.mT.contiguous().view(-1, t)
    row_index = torch.arange(seg_mask.shape[0], device=seg_mask.device).view(-1, 1).expand(-1, t).contiguous()
    sort_mask = seg_mask[row_index, sort_index]
    cas = cas.mT.contiguous().view(-1, t)
    sort_cas = cas[row_index, sort_index]
    # [1, T]
    rank = torch.arange(2, t + 2, device=seg_score.device).unsqueeze(dim=0).reciprocal()
    # [N*C]
    act_num = (rank * sort_mask).sum(dim=-1)
    act_score = (sort_cas * rank * sort_mask).sum(dim=-1) / torch.clamp_min(act_num, 1.0)
    bkg_num = (1.0 - sort_mask).sum(dim=-1)
    bkg_score = (sort_cas * (1.0 - sort_mask)).sum(dim=-1) / torch.clamp_min(bkg_num, 1.0)
    act_score, bkg_score = torch.softmax(act_score.view(n, c), dim=-1), torch.softmax(bkg_score.view(n, c), dim=-1)
    return act_score, bkg_score


def cross_entropy(act_score, bkg_score, label, eps=1e-8):
    act_num = torch.clamp_min(torch.sum(label, dim=-1), 1.0)
    act_loss = (-(label * torch.log(torch.clamp_min(act_score, eps))).sum(dim=-1) / act_num).mean()
    bkg_loss = (-torch.log(torch.clamp_min(1.0 - bkg_score, eps))).mean()
    return act_loss + bkg_loss


# ref: Weakly Supervised Action Selection Learning in Video (CVPR 2021)
def generalized_cross_entropy(aas_score, seg_mask, label, q=0.7, eps=1e-8):
    # [N, T]
    aas_score = aas_score.squeeze(dim=-1)
    n, t, c = seg_mask.shape
    # [N, T]
    mask = torch.zeros(n, t, device=seg_mask.device)
    for i in range(n):
        mask[i, :] = torch.sum(seg_mask[i, :, label[i, :].bool()], dim=-1)
    # [N, T]
    mask = torch.clamp_max(mask, 1.0)
    # [N]
    pos_num = torch.clamp_min(torch.sum(mask, dim=1), 1.0)
    neg_num = torch.clamp_min(torch.sum(1.0 - mask, dim=1), 1.0)

    pos_loss = ((((1.0 - torch.clamp_min(aas_score, eps) ** q) / q) * mask).sum(dim=-1) / pos_num).mean()
    neg_loss = ((((1.0 - torch.clamp_min(1.0 - aas_score, eps) ** q) / q) * (1.0 - mask)).sum(dim=-1) / neg_num).mean()
    return pos_loss + neg_loss
