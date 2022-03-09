import torch
import torch.nn as nn
import torch.nn.functional as F


# Multi-Head Graph Attention
class MGA(nn.Module):
    def __init__(self, feat_dim, num_head):
        super(MGA, self).__init__()
        self.num_heads = num_head
        self.temperature = nn.Parameter(torch.ones(1, num_head, 1, 1))

        self.qkv = nn.Conv1d(feat_dim, feat_dim * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv1d(feat_dim * 3, feat_dim * 3, kernel_size=3, padding=1, groups=feat_dim * 3, bias=False)
        self.project_out = nn.Conv1d(feat_dim, feat_dim, kernel_size=1, bias=False)

    def forward(self, x):
        n, d, l = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)
        # [N, H, L, D/H]
        q = q.reshape(n, self.num_heads, -1, l).transpose(-2, -1).contiguous()
        k = k.reshape(n, self.num_heads, -1, l).transpose(-2, -1).contiguous()
        v = v.reshape(n, self.num_heads, -1, l).transpose(-2, -1).contiguous()
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        # [N, H, L, L]
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)

        out = self.project_out(torch.matmul(attn, v).transpose(-2, -1).contiguous().reshape(n, -1, l))
        return out


# Gated Feed-Forward Network
class GFN(nn.Module):
    def __init__(self, feat_dim):
        super(GFN, self).__init__()

        self.project_in = nn.Conv1d(feat_dim, feat_dim * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv1d(feat_dim * 2, feat_dim * 2, kernel_size=3, padding=1, groups=feat_dim * 2, bias=False)
        self.project_out = nn.Conv1d(feat_dim, feat_dim, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, feat_dim, num_head):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn = MGA(feat_dim, num_head)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.ffn = GFN(feat_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x.transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous())
        x = x + self.ffn(self.norm2(x.transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous())
        return x


class Model(nn.Module):
    def __init__(self, num_classes, num_head, hidden_dim, factor):
        super(Model, self).__init__()

        self.factor = factor
        self.cas_encoder = nn.Sequential(nn.Conv1d(2048, hidden_dim, kernel_size=3, padding=1, bias=False),
                                         TransformerBlock(hidden_dim, num_head),
                                         nn.Conv1d(hidden_dim, num_classes, kernel_size=3, padding=1, bias=False))
        self.sas_encoder = nn.Sequential(nn.Conv1d(2048, hidden_dim, kernel_size=3, padding=1, bias=False),
                                         TransformerBlock(hidden_dim, num_head),
                                         nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1, bias=False))

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
