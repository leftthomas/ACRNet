import torch
import torch.nn as nn
import torch.nn.functional as F


# Graph Attention
class GA(nn.Module):
    def __init__(self, feat_dim, factor):
        super(GA, self).__init__()
        self.factor = factor
        # self.kv = nn.Conv1d(feat_dim, feat_dim * 2, kernel_size=1, bias=False)
        # self.kv_conv = nn.Conv1d(feat_dim * 2, feat_dim * 2, kernel_size=3, padding=1, groups=feat_dim * 2, bias=False)

    def forward(self, x):
        # k, v = self.kv_conv(self.kv(x)).chunk(2, dim=1)
        x, v = F.normalize(x, dim=1), x.transpose(-2, -1).contiguous()
        # [N, L, L]
        # attn = torch.matmul(q.transpose(-2, -1).contiguous(), k)
        # min_attn = torch.amin(attn, dim=-1, keepdim=True)
        # max_attn = torch.amax(attn, dim=-1, keepdim=True)
        # attn = (attn - min_attn) / torch.where(torch.eq(max_attn, 0.0), torch.ones_like(max_attn), max_attn)

        # diff_graph = torch.arange(v.shape[1], device=sim_graph.device).unsqueeze(dim=0)
        # diff_graph = diff_graph.sub(torch.arange(v.shape[1], device=sim_graph.device).unsqueeze(dim=-1))
        # diff_graph = torch.abs(diff_graph)
        # diff_graph = diff_graph.div(diff_graph.sum(dim=-1, keepdim=True))
        # diff_graph = torch.diagonal_scatter(diff_graph, torch.ones(diff_graph.shape[-1], device=diff_graph.device))
        # diff_graph = torch.reciprocal(diff_graph)
        # diff_graph = torch.diagonal_scatter(diff_graph, torch.zeros(diff_graph.shape[-1], device=diff_graph.device))
        # diff_graph = diff_graph.div(diff_graph.sum(dim=-1, keepdim=True))
        # diff_graph = torch.diagonal_scatter(diff_graph, torch.ones(diff_graph.shape[-1], device=diff_graph.device))

        # ref: Graph Convolutional Networks for Temporal Action Localization (ICCV 2019)
        # graph = (sim_graph + diff_graph.unsqueeze(dim=0)) / 2
        # attn = torch.diagonal_scatter(attn, torch.zeros(attn.shape[:-1], device=attn.device), dim1=-2, dim2=-1)
        # top_attn = torch.topk(attn, k=max(attn.shape[-1] // self.factor, 1), dim=-1)[0]
        # min_attn = torch.amin(top_attn, dim=-1, keepdim=True)
        # attn = torch.where(torch.ge(attn, min_attn), attn, torch.zeros_like(attn))
        # num = torch.count_nonzero(attn, dim=-1).unsqueeze(dim=-1)
        # v = torch.matmul(attn, v) / torch.where(torch.eq(num, 0.0), torch.ones_like(num), num) + v

        # ref: ACGNet: Action Complement Graph Network for Weakly-supervised Temporal Action Localization (AAAI 2022)
        sim_graph = torch.matmul(x.transpose(-2, -1).contiguous(), x)
        graph = sim_graph
        # select top k
        top_attn = torch.topk(graph, k=max(graph.shape[1] // self.factor, 1), dim=-1)[0]
        min_attn = torch.amin(top_attn, dim=-1, keepdim=True)
        graph = torch.where(torch.ge(graph, min_attn), graph, torch.zeros_like(graph))
        # row-normalized
        sum_attn = torch.sum(graph, dim=-1, keepdim=True)
        sum_attn = torch.where(torch.eq(sum_attn, 0.0), torch.ones_like(sum_attn), sum_attn)
        graph = graph.div(sum_attn)
        # graph average
        v = torch.matmul(graph, v) + v

        return v.transpose(-2, -1).contiguous(), sim_graph


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

        # self.norm1 = nn.LayerNorm(feat_dim)
        self.attn = GA(feat_dim, factor)
        # self.norm2 = nn.LayerNorm(feat_dim)
        # self.ffn = GF(feat_dim)

    def forward(self, x):
        x, graph = self.attn(x)
        # x = x + self.ffn(self.norm2(x.transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous())
        return x, graph


class Model(nn.Module):
    def __init__(self, num_classes, hidden_dim, factor, temperature):
        super(Model, self).__init__()

        self.factor = factor
        self.temperature = temperature
        self.encoder = nn.Sequential(nn.Conv1d(2048, hidden_dim, kernel_size=3, padding=1, bias=False),
                                     TransformerBlock(hidden_dim, factor))
        self.proxies = nn.Parameter(torch.randn(1, hidden_dim, num_classes))
        # self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        feat, graph = self.encoder(x.transpose(-1, -2).contiguous())
        # [N, L, D]
        feat = feat.transpose(-1, -2).contiguous()
        # [N, L, C], class activation sequence
        cas = torch.matmul(F.normalize(feat, dim=-1), F.normalize(self.proxies, dim=1)).div(self.temperature)
        cas_score = torch.softmax(cas, dim=-1)

        # [N, L, 1], segment activation sequence
        sas = torch.norm(feat, p=2, dim=-1, keepdim=True)
        min_norm = torch.amin(sas, dim=1, keepdim=True)
        max_norm = torch.amax(sas, dim=1, keepdim=True)
        sas_score = (sas - min_norm) / torch.where(torch.eq(max_norm, 0.0), torch.ones_like(max_norm), max_norm)

        seg_score = (cas_score + sas_score) / 2

        act_index = seg_score.topk(k=max(seg_score.shape[1] // self.factor, 1), dim=1)[1]
        bkg_index = seg_score.topk(k=max(seg_score.shape[1] // self.factor, 1), dim=1, largest=False)[1]
        # [N, C], action classification score is aggregated by cas
        act_score = torch.softmax(torch.gather(cas, dim=1, index=act_index).mean(dim=1), dim=-1)
        bkg_score = torch.softmax(torch.gather(cas, dim=1, index=bkg_index).mean(dim=1), dim=-1)
        return act_score, bkg_score, cas_score, sas_score.squeeze(dim=-1), seg_score, act_index, graph


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

