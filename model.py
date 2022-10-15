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
        # [N, C]
        # act_score, refined_mask = mask_refining(seg_score, seg_mask)
        act_score = torch.mean(seg_score * seg_mask, dim=1)

        return act_score, seg_score


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
            pos_num = pos_num + condition.float() / (i + 1)
            pos_sum = pos_sum + pos_list / (i + 1)
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


def mask_refining(seg_score, mask, soft=True):
    n, t, c = seg_score.shape
    sort_value, sort_index = torch.sort(seg_score, dim=1, descending=True, stable=True)
    # [N, T]
    if soft:
        ranks = torch.arange(2, t + 2, device=seg_score.device).reciprocal().view(-1, t).expand(n, -1).contiguous()
    else:
        ranks = torch.ones(n, t, device=seg_score.device)
    row_index = torch.arange(n, device=seg_score.device).view(n, -1).expand(-1, t).contiguous()
    # [N, C]
    act_score = torch.zeros(n, c, device=seg_score.device)
    for i in range(c):
        # [N, T]
        index, value = sort_index[:, :, i], sort_value[:, :, i]
        tmp_mask = mask[:, :, i][row_index, index]
        tmp_rank = ranks * tmp_mask
        tmp_score = tmp_rank * value
        # [N]
        tol_rank = torch.sum(tmp_rank, dim=-1)
        tol_rank = torch.where(torch.eq(tol_rank, 0.0), torch.ones_like(tol_rank), tol_rank)
        act_score[:, i] = tmp_score.sum(dim=-1) / tol_rank
    refined_mask = torch.ge(seg_score, act_score.unsqueeze(dim=1)).float() * mask

    return act_score, refined_mask


def cross_entropy(act_score, label, eps=1e-8):
    act_num = torch.sum(label, dim=-1)
    act_num = torch.where(torch.eq(act_num, 0.0), torch.ones_like(act_num), act_num)
    act_loss = (-(label * torch.log(act_score + eps)).sum(dim=-1) / act_num).mean(dim=0)
    return act_loss


# ref: Weakly Supervised Action Selection Learning in Video (CVPR 2021)
def generalized_cross_entropy(score, label, q=0.7, eps=1e-8):
    pos_num, neg_num = divide_label(label)
    pos_loss = ((((1.0 - (score + eps) ** q) / q) * label).sum(dim=-1) / pos_num).mean(dim=0)
    neg_loss = ((((1.0 - (1.0 - score + eps) ** q) / q) * (1.0 - label)).sum(dim=-1) / neg_num).mean(dim=0)
    return pos_loss + neg_loss
