import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes, hidden_dim, factor):
        super(Model, self).__init__()

        self.factor = factor
        self.cas_encoder = nn.Sequential(nn.Conv1d(in_channels=2048, out_channels=hidden_dim, kernel_size=1),
                                         nn.ReLU(), nn.Conv1d(in_channels=hidden_dim, out_channels=num_classes,
                                                              kernel_size=1))
        self.sas_encoder = nn.Sequential(nn.Conv1d(in_channels=2048, out_channels=hidden_dim, kernel_size=1),
                                         nn.ReLU(),
                                         nn.Conv1d(in_channels=hidden_dim, out_channels=1, kernel_size=1))

    def forward(self, x):
        # [N, L, C], class activation sequence
        cas = self.cas_encoder(x.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()
        sa_score = torch.softmax(cas, dim=-1)
        # [N, L, 1], segment activation sequence
        sas = self.sas_encoder(x.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()
        fb_score = torch.sigmoid(sas)

        seg_score = (sa_score + fb_score) / 2

        fore_index = seg_score.topk(k=min(seg_score.shape[1] // self.factor, seg_score.shape[1]), dim=1)[1]
        # [N, C], action classification score is aggregated by cas
        act_score = torch.gather(cas, dim=1, index=fore_index).mean(dim=1)
        return act_score, fb_score.squeeze(dim=-1), fore_index, seg_score


def form_fore_back(fore_index, num_seg, label):
    fb_mask = []
    for i in range(fore_index.shape[0]):
        pos_index = fore_index[i][:, label[i].bool()].flatten()
        mask = torch.zeros(num_seg, device=fore_index.device)
        mask[pos_index] = 1.0
        fb_mask.append(mask)
    return torch.stack(fb_mask)


def cross_entropy(score, label):
    num = label.sum(dim=-1, keepdim=True)
    # avoid divide by zero
    num = torch.where(num == 0.0, torch.ones_like(num), num)
    label = label / num
    loss = -(label * F.log_softmax(score, dim=-1)).sum(dim=-1).mean(dim=0)
    return loss
