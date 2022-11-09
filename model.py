import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init

import model

torch.set_default_tensor_type('torch.cuda.FloatTensor')


# Multi-head Cross Modal Attention
class MCA(nn.Module):
    def __init__(self, feat_dim, num_head):
        super(MCA, self).__init__()
        self.rgb_proj = nn.Parameter(torch.empty(num_head, feat_dim, feat_dim // num_head))
        self.flow_proj = nn.Parameter(torch.empty(num_head, feat_dim, feat_dim // num_head))
        self.atte = nn.Parameter(torch.empty(num_head, feat_dim // num_head, feat_dim // num_head))

        nn.init.uniform_(self.rgb_proj, -math.sqrt(feat_dim), math.sqrt(feat_dim))
        nn.init.uniform_(self.flow_proj, -math.sqrt(feat_dim), math.sqrt(feat_dim))
        nn.init.uniform_(self.atte, -math.sqrt(feat_dim // num_head), math.sqrt(feat_dim // num_head))
        self.num_head = num_head

    def forward(self, rgb, flow):
        n, l, d = rgb.shape
        # [N, H, L, D/H]
        o_rgb = F.normalize(torch.matmul(rgb.unsqueeze(dim=1), self.rgb_proj), dim=-1)
        o_flow = F.normalize(torch.matmul(flow.unsqueeze(dim=1), self.flow_proj), dim=-1)
        # [N, H, L, L]
        atte = torch.matmul(torch.matmul(o_rgb, self.atte), o_flow.transpose(-1, -2))
        rgb_atte = torch.softmax(atte, dim=-1)
        flow_atte = torch.softmax(atte.transpose(-1, -2), dim=-1)

        # [N, H, L, D/H]
        e_rgb = F.gelu(torch.matmul(rgb_atte, o_rgb))
        e_flow = F.gelu(torch.matmul(flow_atte, o_flow))
        # [N, L, D]
        f_rgb = torch.tanh(e_rgb.transpose(1, 2).reshape(n, l, -1) + rgb)
        f_flow = torch.tanh(e_flow.transpose(1, 2).reshape(n, l, -1) + flow)
        return f_rgb, f_flow


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias) != type(None):
            m.bias.data.fill_(0)


class BWA_fusion_dropout_feat_v2(torch.nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()
        embed_dim = 1024
        self.bit_wise_attn = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.channel_conv = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
                                       nn.Dropout(0.5),
                                       nn.Sigmoid())
        self.channel_avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, vfeat, ffeat):
        channelfeat = self.channel_avg(vfeat)
        channel_attn = self.channel_conv(channelfeat)
        bit_wise_attn = self.bit_wise_attn(ffeat)
        filter_feat = torch.sigmoid(bit_wise_attn * channel_attn) * vfeat
        x_atn = self.attention(filter_feat)
        return x_atn, filter_feat


# fusion split modal single+ bit_wise_atten dropout+ contrastive + mutual learning +fusion feat(cat)
# ------TOP!!!!!!!!!!
class CO2(torch.nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()
        embed_dim = 2048
        mid_dim = 1024
        dropout_ratio = args['opt'].dropout_ratio
        reduce_ratio = args['opt'].reduce_ratio

        self.cma = MCA(n_feature // 2, args['opt'].num_head)

        self.vAttn = getattr(model, args['opt'].AWM)(1024, args)
        self.fAttn = getattr(model, args['opt'].AWM)(1024, args)

        self.feat_encoder = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(dropout_ratio))
        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 1, padding=0), nn.LeakyReLU(0.2), nn.Dropout(dropout_ratio))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Dropout(0.7), nn.Conv1d(embed_dim, n_class + 1, 1))

        self.channel_avg = nn.AdaptiveAvgPool1d(1)
        self.batch_avg = nn.AdaptiveAvgPool1d(1)
        self.ce_criterion = nn.BCELoss()

        self.apply(weights_init)

    def forward(self, inputs, is_training=True, **args):
        rgb, flow = inputs[:, :, :1024], inputs[:, :, 1024:]
        rgb, flow = self.cma(rgb, flow)
        feat = torch.cat((rgb, flow), dim=-1).transpose(-2, -1)

        b, c, n = feat.size()
        # feat = self.feat_encoder(x)
        v_atn, vfeat = self.vAttn(feat[:, :1024, :], feat[:, 1024:, :])
        f_atn, ffeat = self.fAttn(feat[:, 1024:, :], feat[:, :1024, :])
        x_atn = (f_atn + v_atn) / 2
        nfeat = torch.cat((vfeat, ffeat), 1)
        nfeat = self.fusion(nfeat)
        x_cls = self.classifier(nfeat)

        return {'feat': nfeat.transpose(-1, -2), 'cas': x_cls.transpose(-1, -2), 'attn': x_atn.transpose(-1, -2),
                'v_atn': v_atn.transpose(-1, -2), 'f_atn': f_atn.transpose(-1, -2)}

    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def criterion(self, outputs, labels, **args):
        feat, element_logits, element_atn = outputs['feat'], outputs['cas'], outputs['attn']
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']
        mutual_loss = 0.5 * F.mse_loss(v_atn, f_atn.detach()) + 0.5 * F.mse_loss(f_atn, v_atn.detach())
        # learning weight dynamic, lambda1 (1-lambda1)
        b, n, c = element_logits.shape
        element_logits_supp = self._multiply(element_logits, element_atn, include_min=True)
        loss_mil_orig, _ = self.topkloss(element_logits,
                                         labels,
                                         is_back=True,
                                         rat=args['opt'].k,
                                         reduce=None)
        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                         labels,
                                         is_back=False,
                                         rat=args['opt'].k,
                                         reduce=None)

        loss_3_supp_Contrastive = self.Contrastive(feat, element_logits_supp, labels, is_back=False)

        loss_norm = element_atn.mean()
        # guide loss
        loss_guide = (1 - element_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        v_loss_norm = v_atn.mean()
        # guide loss
        v_loss_guide = (1 - v_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.mean()
        # guide loss
        f_loss_guide = (1 - f_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        # total loss
        total_loss = (loss_mil_orig.mean() + loss_mil_supp.mean() +
                      args['opt'].alpha3 * loss_3_supp_Contrastive +
                      args['opt'].alpha4 * mutual_loss +
                      args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3 +
                      args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3)

        return total_loss

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 lab_rand=None,
                 rat=8,
                 reduce=None):

        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)
        instance_logits = torch.mean(
            topk_val,
            dim=-2,
        )
        labels_with_back = labels_with_back / (
                torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back *
                     F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))
        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind

    def Contrastive(self, x, element_logits, labels, is_back=False):
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        for i in range(0, 3 * 2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)
            atn2 = F.softmax(element_logits[i + 1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
            n2 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)  # (n_feature, n_class)
            Hf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1) / n1)
            Lf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), (1 - atn2) / n2)

            d1 = 1 - torch.sum(Hf1 * Hf2, dim=0) / (
                    torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))  # 1-similarity
            d2 = 1 - torch.sum(Hf1 * Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2 * Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d2 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d3 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
            n_tmp = n_tmp + torch.sum(labels[i, :] * labels[i + 1, :])
        sim_loss = sim_loss / n_tmp
        return sim_loss

    def decompose(self, outputs, **args):
        feat, element_logits, atn_supp, atn_drop, element_atn = outputs

        return element_logits, element_atn


class ANT_CO2(torch.nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()
        embed_dim = 2048
        mid_dim = 1024
        dropout_ratio = args['opt'].dropout_ratio
        reduce_ratio = args['opt'].reduce_ratio

        self.vAttn = getattr(model, args['opt'].AWM)(1024, args)
        self.fAttn = getattr(model, args['opt'].AWM)(1024, args)

        self.feat_encoder = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(dropout_ratio))
        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 1, padding=0), nn.LeakyReLU(0.2), nn.Dropout(dropout_ratio))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Dropout(0.7), nn.Conv1d(embed_dim, n_class + 1, 1))

        self.channel_avg = nn.AdaptiveAvgPool1d(1)
        self.batch_avg = nn.AdaptiveAvgPool1d(1)
        self.ce_criterion = nn.BCELoss()
        _kernel = ((args['opt'].max_seqlen // args['opt'].t) // 2 * 2 + 1)
        self.pool = nn.AvgPool1d(_kernel, 1, padding=_kernel // 2, count_include_pad=True) \
            if _kernel is not None else nn.Identity()
        self.apply(weights_init)

        self.cma = MCA(n_feature // 2, args['opt'].num_head)

    def forward(self, inputs, is_training=True, **args):
        rgb, flow = inputs[:, :, :1024], inputs[:, :, 1024:]
        rgb, flow = self.cma(rgb, flow)
        feat = torch.cat((rgb, flow), dim=-1).transpose(-2, -1)

        b, c, n = feat.size()
        # feat = self.feat_encoder(x)
        v_atn, vfeat = self.vAttn(feat[:, :1024, :], feat[:, 1024:, :])
        f_atn, ffeat = self.fAttn(feat[:, 1024:, :], feat[:, :1024, :])
        x_atn = (f_atn + v_atn) / 2
        nfeat = torch.cat((vfeat, ffeat), 1)
        nfeat = self.fusion(nfeat)
        x_cls = self.classifier(nfeat)
        x_cls = self.pool(x_cls)
        x_atn = self.pool(x_atn)
        f_atn = self.pool(f_atn)
        v_atn = self.pool(v_atn)

        return {'feat': nfeat.transpose(-1, -2), 'cas': x_cls.transpose(-1, -2), 'attn': x_atn.transpose(-1, -2),
                'v_atn': v_atn.transpose(-1, -2), 'f_atn': f_atn.transpose(-1, -2)}

    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def criterion(self, outputs, labels, **args):
        feat, element_logits, element_atn = outputs['feat'], outputs['cas'], outputs['attn']
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']
        mutual_loss = 0.5 * F.mse_loss(v_atn, f_atn.detach()) + 0.5 * F.mse_loss(f_atn, v_atn.detach())
        # learning weight dynamic, lambda1 (1-lambda1)
        b, n, c = element_logits.shape
        element_logits_supp = self._multiply(element_logits, element_atn, include_min=True)
        loss_mil_orig, _ = self.topkloss(element_logits,
                                         labels,
                                         is_back=True,
                                         rat=args['opt'].k,
                                         reduce=None)
        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                         labels,
                                         is_back=False,
                                         rat=args['opt'].k,
                                         reduce=None)

        loss_3_supp_Contrastive = self.Contrastive(feat, element_logits_supp, labels, is_back=False)

        loss_norm = element_atn.mean()
        # guide loss
        loss_guide = (1 - element_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        v_loss_norm = v_atn.mean()
        # guide loss
        v_loss_guide = (1 - v_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.mean()
        # guide loss
        f_loss_guide = (1 - f_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        # total loss
        total_loss = (loss_mil_orig.mean() + loss_mil_supp.mean() + args[
            'opt'].alpha3 * loss_3_supp_Contrastive + mutual_loss +
                      args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3 +
                      args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3)

        return total_loss

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 lab_rand=None,
                 rat=8,
                 reduce=None):

        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)
        instance_logits = torch.mean(
            topk_val,
            dim=-2,
        )
        labels_with_back = labels_with_back / (
                torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back *
                     F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))
        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind

    def Contrastive(self, x, element_logits, labels, is_back=False):
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        for i in range(0, 3 * 2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)
            atn2 = F.softmax(element_logits[i + 1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
            n2 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)  # (n_feature, n_class)
            Hf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1) / n1)
            Lf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), (1 - atn2) / n2)

            d1 = 1 - torch.sum(Hf1 * Hf2, dim=0) / (
                    torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))  # 1-similarity
            d2 = 1 - torch.sum(Hf1 * Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2 * Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d2 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d3 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
            n_tmp = n_tmp + torch.sum(labels[i, :] * labels[i + 1, :])
        sim_loss = sim_loss / n_tmp
        return sim_loss

    def decompose(self, outputs, **args):
        feat, element_logits, atn_supp, atn_drop, element_atn = outputs

        return element_logits, element_atn
