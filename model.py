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
    def __init__(self, feat_dim, expansion_factor):
        super(GFN, self).__init__()

        hidden_dim = int(feat_dim * expansion_factor)
        self.project_in = nn.Conv1d(feat_dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1,
                              groups=hidden_dim * 2, bias=False)
        self.project_out = nn.Conv1d(hidden_dim, feat_dim, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, feat_dim, num_head, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn = MGA(feat_dim, num_head)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.ffn = GFN(feat_dim, expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous()
        x = x + self.ffn(self.norm2(x).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous()
        return x


class Model(nn.Module):
    def __init__(self, num_classes, num_blocks, num_heads, feat_dims, expansion_factor, k):
        super(Model, self).__init__()

        self.k = k
        self.feat_conv = nn.Conv1d(2048, feat_dims[0], kernel_size=3, padding=1, bias=False)
        self.encoders = nn.ModuleList(
            [nn.Sequential(*[TransformerBlock(feat_dim, num_head, expansion_factor) for _ in range(num_block)]) for
             num_block, num_head, feat_dim in zip(num_blocks, num_heads, feat_dims)])
        # the number of down sample == the number of encoder - 1
        self.downs = nn.ModuleList([nn.Conv1d(feat_dims[i], feat_dims[i + 1], kernel_size=3, padding=1, bias=False)
                                    for i in range(len(feat_dims) - 1)])
        self.cls = nn.Conv1d(feat_dims[-1], num_classes, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # [N, L, D]
        x = self.feat_conv(x.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            # do not down sample the last encoder
            if i < len(self.encoders) - 1:
                x = self.downs[i](x.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()
        x = self.cls(x.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()
        # [N, L, C]
        seg_score = torch.softmax(x, dim=-1)
        # [N, C]
        act_score = torch.softmax(x.topk(k=min(self.k, x.shape[1]), dim=1)[0].mean(dim=1), dim=-1)
        return act_score, seg_score
