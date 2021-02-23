import torch
import torch.nn.functional as F
from kornia.filters import spatial_gradient
from torch import nn

# constant values
ignore_label = 255
in_channels = 4
num_classes = 10
palette = [177, 191, 122, 0, 128, 0, 128, 168, 93, 62, 51, 0, 128, 128, 0, 128, 128, 128,
           192, 128, 0, 0, 128, 128, 132, 200, 173, 128, 64, 0]


class BoundaryBCELoss(nn.Module):
    def __init__(self, ignore_index=ignore_label):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, edge, target, boundary):
        edge = edge.squeeze(dim=1)
        mask = target != self.ignore_index
        pos_mask = (boundary == 1.0) & mask
        neg_mask = (boundary == 0.0) & mask
        num = torch.clamp(mask.sum(), min=1)
        pos_weight = neg_mask.sum() / num
        neg_weight = pos_mask.sum() / num

        weight = torch.zeros_like(boundary)
        weight[pos_mask] = pos_weight
        weight[neg_mask] = neg_weight
        loss = F.binary_cross_entropy(edge, boundary, weight, reduction='sum') / num
        return loss


class DualTaskLoss(nn.Module):
    def __init__(self, threshold=0.8, ignore_index=ignore_label):
        super().__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index

    def forward(self, seg, edge, target):
        edge = edge.squeeze(dim=1)
        logit = F.cross_entropy(seg, target, ignore_index=self.ignore_index, reduction='none')
        mask = target != self.ignore_index
        num = torch.clamp(((edge > self.threshold) & mask).sum(), min=1)
        reg_l_loss = (logit[edge > self.threshold]).sum() / num

        logit = torch.where(mask.unsqueeze(dim=1), seg, torch.zeros_like(seg))
        target = torch.where(mask, target, torch.zeros_like(target))
        logit = F.gumbel_softmax(logit, tau=0.5, dim=1)
        target = F.one_hot(target, num_classes=seg.size(1)).permute(0, 3, 1, 2).float()
        target = F.gumbel_softmax(target, tau=0.5, dim=1)
        logit_grad = spatial_gradient(logit)
        logit_mag = torch.norm(logit_grad, dim=-3)
        target_grad = spatial_gradient(target)
        target_mag = torch.norm(target_grad, dim=-3)
        reg_r_loss = F.l1_loss(logit_mag, target_mag, reduction='none')
        mask = ((logit_mag >= 1e-8) | (target_mag >= 1e-8)) & mask.unsqueeze(dim=1)
        num = torch.clamp(mask.sum(), min=1)
        reg_r_loss = reg_r_loss[mask].sum() / num
        loss = reg_l_loss + reg_r_loss
        return loss
