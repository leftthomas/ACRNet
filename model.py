import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch.nn.init.kaiming_uniform_(m.weight)
        if not isinstance(m.bias, type(None)):
            m.bias.data.fill_(0)


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()

        self.cas_rgb_encoder = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=3, padding=1), nn.Dropout(p=0.5),
                                             nn.ReLU(), nn.Conv1d(1024, num_classes, kernel_size=1))
        self.cas_flow_encoder = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=3, padding=1), nn.Dropout(p=0.5),
                                              nn.ReLU(), nn.Conv1d(1024, num_classes, kernel_size=1))

        self.aas_rgb_encoder = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=3, padding=1), nn.Dropout(p=0.5),
                                             nn.ReLU(), nn.Conv1d(1024, 1, kernel_size=1))
        self.aas_flow_encoder = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=3, padding=1), nn.Dropout(p=0.5),
                                              nn.ReLU(), nn.Conv1d(1024, 1, kernel_size=1))
        self.apply(weights_init)

    def forward(self, x):
        # [N, D, T]
        rgb, flow = x[:, :, :1024].mT.contiguous(), x[:, :, 1024:].mT.contiguous()
        # [N, T, C], class activation sequence
        cas_rgb, cas_flow = self.cas_rgb_encoder(rgb).mT.contiguous(), self.cas_flow_encoder(flow).mT.contiguous()
        cas = torch.softmax(cas_rgb + cas_flow, dim=-1)
        # [N, T, 1], action activation sequence
        aas_rgb, aas_flow = self.aas_rgb_encoder(rgb).mT.contiguous(), self.aas_flow_encoder(flow).mT.contiguous()
        aas = torch.sigmoid(aas_rgb + aas_flow)
        # [N, T, C]
        seg_score = (cas + aas) / 2
        seg_mask = temporal_clustering(seg_score)
        seg_mask = mask_refining(seg_score, seg_mask)

        # [N, C]
        act_num = torch.clamp_min(torch.sum(seg_mask, dim=1), 1.0)
        bkg_num = torch.clamp_min(torch.sum(1.0 - seg_mask, dim=1), 1.0)

        act_score = torch.softmax(torch.sum((cas_rgb + cas_flow) * seg_mask, dim=1) / act_num, dim=-1)
        bkg_score = torch.softmax(torch.sum((cas_rgb + cas_flow) * (1.0 - seg_mask), dim=1) / bkg_num, dim=-1)
        return act_score, bkg_score, aas, seg_score, seg_mask


def temporal_clustering(seg_score, soft=True):
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
        if soft:
            pos_num = pos_num + condition.float() / ((i + 1) ** 2)
            pos_sum = pos_sum + pos_list / ((i + 1) ** 2)
        else:
            pos_num = pos_num + condition.float()
            pos_sum = pos_sum + pos_list
        neg_num = neg_num + (~condition).float()
        neg_sum = neg_sum + neg_list
        # update mask
        mask[row_index, index] = condition.float()
    # [N, T, C]
    mask = mask.view(n, c, t).mT.contiguous()
    return mask


def mask_refining(seg_score, seg_mask, soft=True):
    n, t, c = seg_score.shape
    sort_value, sort_index = torch.sort(seg_score, dim=1, descending=True, stable=True)
    # [N, C]
    total_num = torch.count_nonzero(seg_mask, dim=1)
    act_score = torch.zeros(n, c, device=seg_score.device)
    for i in range(c):
        # [N, T]
        index, value = sort_index[:, :, i], sort_value[:, :, i]
        score, mask = seg_score[:, :, i], seg_mask[:, :, i]

        # generate pseudo action score as threshold
        if soft:
            weights = torch.ones_like(mask)
            for j in range(n):
                rank = torch.arange(1, total_num[j, i] + 1, device=seg_mask.device).reciprocal()
                weights[j, index[j, mask[j, :].bool()]] = rank
            pos_num = torch.clamp_min(torch.sum(mask * weights, dim=-1), 1.0)
            act_score[:, i] = torch.sum(score * mask * weights, dim=-1) / pos_num
        else:
            pos_num = torch.clamp_min(total_num[:, i], 1.0)
            act_score[:, i] = torch.sum(score * mask, dim=-1) / pos_num

    # suppress each segment in many actions concurrently
    refined_mask = torch.ge(seg_score, act_score.unsqueeze(dim=1)).float() * seg_mask

    return refined_mask


def cross_entropy(act_score, bkg_score, label, eps=1e-8):
    act_num = torch.clamp_min(torch.sum(label, dim=-1), 1.0)
    act_loss = (-(label * torch.log(act_score + eps)).sum(dim=-1) / act_num).mean(dim=0)
    bkg_loss = (-(label * torch.log(1.0 - bkg_score + eps)).sum(dim=-1) / act_num).mean(dim=0)
    return act_loss + bkg_loss


# ref: Weakly Supervised Action Selection Learning in Video (CVPR 2021)
def generalized_cross_entropy(aas_score, label, seg_mask, q=0.7, eps=1e-8):
    # [N, T]
    aas_score = aas_score.squeeze(dim=-1)
    n, t, c = seg_mask.shape
    # [N, T]
    mask = torch.zeros(n, t, device=seg_mask.device)
    for i in range(n):
        mask[i, :] = torch.sum(seg_mask[i, :, label[i, :].bool()], dim=-1)
    # [N, T]
    mask = torch.clamp_max(mask, 1.0)
    # [N, 1]
    pos_num = torch.clamp_min(torch.sum(mask, dim=1), 1.0)
    neg_num = torch.clamp_min(torch.sum(1.0 - mask, dim=1), 1.0)

    pos_loss = ((((1.0 - (aas_score + eps) ** q) / q) * mask).sum(dim=-1) / pos_num).mean(dim=0)
    neg_loss = ((((1.0 - (1.0 - aas_score + eps) ** q) / q) * (1.0 - mask)).sum(dim=-1) / neg_num).mean(dim=0)
    return pos_loss + neg_loss
