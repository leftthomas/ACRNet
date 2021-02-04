import torch
import torch.nn.functional as F
from torch import nn


def get_palette():
    palette = [177, 191, 122, 0, 128, 0, 128, 168, 93, 62, 51, 0, 128, 128, 0, 128, 128, 128,
               192, 128, 0, 0, 128, 128, 132, 200, 173, 128, 64, 0]
    return palette


class BoundaryBCELoss(nn.Module):
    def __init__(self, ignore_index=255):
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
    def __init__(self, threshold=0.8, ignore_index=255):
        super().__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index

    def forward(self, seg, edge, target):
        edge = edge.squeeze(dim=1)
        logit = F.cross_entropy(seg, target, ignore_index=self.ignore_index, reduction='none')
        mask = target != self.ignore_index
        num = torch.clamp(((edge > self.threshold) & mask).sum(), min=1)
        loss = (logit[edge > self.threshold]).sum() / num
        return loss
