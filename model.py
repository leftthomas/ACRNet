import torch
import torch.nn as nn


class CAS(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(CAS, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=len_feature, kernel_size=3, stride=1, padding=1), nn.ReLU())

        self.classifier = nn.Conv1d(in_channels=len_feature, out_channels=num_classes, kernel_size=1, bias=False)
        self.drop_out = nn.Dropout(p=0.7)

    def forward(self, x):
        out = self.conv(x.permute(0, 2, 1))
        feat = out.permute(0, 2, 1)
        out = self.classifier(self.drop_out(out))
        out = out.permute(0, 2, 1)
        return out, feat


class Model(nn.Module):
    def __init__(self, r_act, r_bkg, num_classes, len_feature=2048):
        super(Model, self).__init__()

        self.r_act, self.r_bkg, self.num_classes = r_act, r_bkg, num_classes

        self.cas_module = CAS(len_feature, num_classes)

        self.drop_out = nn.Dropout(p=0.7)

    def forward(self, x):
        num_segments = x.shape[1]
        k_act = num_segments // self.r_act
        k_bkg = num_segments // self.r_bkg

        cas, feat = self.cas_module(x)
        feat_magnitudes = torch.norm(feat, dim=-1)
        # introduce dropout for better considering all features, eliminate the noise
        select_idx = self.drop_out(torch.ones_like(feat_magnitudes))

        feat_magnitudes = feat_magnitudes * select_idx

        idx_act = feat_magnitudes.topk(k=k_act, dim=-1, largest=True)[-1]
        feat_act = torch.take_along_dim(feat, indices=idx_act.unsqueeze(dim=-1), dim=1)

        idx_bkg = feat_magnitudes.topk(k=k_bkg, dim=-1, largest=False)[-1]
        feat_bkg = torch.take_along_dim(feat, indices=idx_bkg.unsqueeze(dim=-1), dim=1)

        topk_scores = cas.topk(k=k_act, dim=1)[0]
        bottomk_scores = torch.take_along_dim(cas, indices=idx_bkg.unsqueeze(dim=-1), dim=1)
        score_act = torch.softmax(torch.mean(topk_scores, dim=1), dim=-1)
        score_bkg = torch.softmax(torch.mean(bottomk_scores, dim=1), dim=-1)

        score_cas = torch.softmax(cas, dim=-1)

        return score_act, score_bkg, score_cas, feat_act, feat_bkg, feat
