import torch
import torch.nn as nn
import torch.nn.functional as F


# Graph Attention
class GA(nn.Module):
    def __init__(self, feat_dim, factor):
        super(GA, self).__init__()
        self.factor = factor
        self.qkv = nn.Conv1d(feat_dim, feat_dim * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv1d(feat_dim * 3, feat_dim * 3, kernel_size=3, padding=1, groups=feat_dim * 3, bias=False)
        self.project_out = nn.Conv1d(feat_dim, feat_dim, kernel_size=1, bias=False)

    def forward(self, x):
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)
        q, k, v = F.normalize(q, dim=1), F.normalize(k, dim=1), v.transpose(-2, -1).contiguous()
        # [N, L, L]
        # attn = torch.matmul(q.transpose(-2, -1).contiguous(), k)
        # min_attn = torch.amin(attn, dim=-1, keepdim=True)
        # max_attn = torch.amax(attn, dim=-1, keepdim=True)
        # attn = (attn - min_attn) / torch.where(torch.eq(max_attn, 0.0), torch.ones_like(max_attn), max_attn)
        #
        # # ref: Graph Convolutional Networks for Temporal Action Localization (ICCV 2019)
        # attn = torch.diagonal_scatter(attn, torch.zeros(attn.shape[:-1], device=attn.device), dim1=-2, dim2=-1)
        # top_attn = torch.topk(attn, k=max(attn.shape[-1] // self.factor, 1), dim=-1)[0]
        # min_attn = torch.amin(top_attn, dim=-1, keepdim=True)
        # attn = torch.where(torch.ge(attn, min_attn), attn, torch.zeros_like(attn))
        # num = torch.count_nonzero(attn, dim=-1).unsqueeze(dim=-1)
        # v = torch.matmul(attn, v) / torch.where(torch.eq(num, 0.0), torch.ones_like(num), num) + v

        out = self.project_out(v.transpose(-2, -1).contiguous())
        return out


# Gated Feed-forward
class GF(nn.Module):
    def __init__(self, feat_dim):
        super(GF, self).__init__()

        self.project_in = nn.Conv1d(feat_dim, feat_dim * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv1d(feat_dim * 2, feat_dim * 2, kernel_size=3, padding=1, groups=feat_dim * 2, bias=False)
        self.project_out = nn.Conv1d(feat_dim, feat_dim, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, feat_dim, factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn = GA(feat_dim, factor)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.ffn = GF(feat_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x.transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous())
        x = x + self.ffn(self.norm2(x.transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous())
        return x


class Model(nn.Module):
    def __init__(self, num_classes, hidden_dim, factor):
        super(Model, self).__init__()

        self.factor = factor
        self.cas_branch = nn.Sequential(nn.Conv1d(2048, hidden_dim, kernel_size=3, padding=1, bias=False), nn.ReLU(),
                                        nn.Conv1d(in_channels=hidden_dim, out_channels=num_classes, kernel_size=1))
        self.sas_branch = nn.Sequential(nn.Conv1d(2048, hidden_dim, kernel_size=3, padding=1, bias=False), nn.ReLU(),
                                        nn.Conv1d(in_channels=hidden_dim, out_channels=1, kernel_size=1))

    def forward(self, x):
        # [N, L, C], class activation sequence
        cas = self.cas_branch(x.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()
        cas_score = torch.softmax(cas, dim=-1)

        # [N, L, 1], segment activation sequence
        sas = self.sas_branch(x.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()
        sas_score = torch.sigmoid(sas)

        seg_score = (cas_score + sas_score) / 2

        act_index = seg_score.topk(k=max(seg_score.shape[1] // self.factor, 1), dim=1)[1]
        bkg_index = seg_score.topk(k=max(seg_score.shape[1] // self.factor, 1), dim=1, largest=False)[1]
        # [N, C], action classification score is aggregated by cas
        act_score = torch.softmax(torch.gather(cas, dim=1, index=act_index).mean(dim=1), dim=-1)
        bkg_score = torch.softmax(torch.gather(cas, dim=1, index=bkg_index).mean(dim=1), dim=-1)
        return act_score, bkg_score, sas_score.squeeze(dim=-1), act_index, seg_score


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


def cross_entropy(act_score, bkg_score, label, eps=1e-8):
    act_num, bkg_num = divide_label(label)
    act_loss = (-(label * torch.log(act_score + eps)).sum(dim=-1) / act_num).mean(dim=0)
    bkg_loss = (-((1.0 - label) * torch.log(1.0 - bkg_score + eps)).sum(dim=-1) / bkg_num).mean(dim=0)
    return act_loss + bkg_loss


# ref: Weakly Supervised Action Selection Learning in Video (CVPR 2021)
def generalized_cross_entropy(score, label, q=0.7, eps=1e-8):
    pos_num, neg_num = divide_label(label)
    pos_loss = ((((1.0 - (score + eps) ** q) / q) * label).sum(dim=-1) / pos_num).mean(dim=0)
    neg_loss = ((((1.0 - (1.0 - score + eps) ** q) / q) * (1.0 - label)).sum(dim=-1) / neg_num).mean(dim=0)
    return pos_loss + neg_loss
