import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # down sampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # up sampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # a bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Backbone(nn.Module):
    def __init__(self, hidden_dim, pretrained=False):
        super(Backbone, self).__init__()

        self.pretrained = pretrained
        self.f = []
        for name, module in resnet50(pretrained).named_children():
            if not isinstance(module, nn.Linear):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 2048, bias=False), nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
                               nn.Linear(2048, 2048, bias=False), nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
                               nn.Linear(2048, 2048, bias=False), nn.BatchNorm1d(2048))
        # prediction head
        self.h = nn.Sequential(nn.Linear(2048, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim),
                               nn.ReLU(inplace=True), nn.Linear(hidden_dim, 2048, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        if not self.pretrained:
            feature = self.g(feature)
        proj = self.h(feature)
        return F.normalize(feature, dim=-1), F.normalize(proj, dim=-1)


class SimCLRLoss(nn.Module):
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, proj_1, proj_2):
        batch_size = proj_1.size(0)
        # [2*B, Dim]
        out = torch.cat([proj_1, proj_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(proj_1 * proj_2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss


class NPIDLoss(nn.Module):
    def __init__(self, n, proj_dim, temperature, negs=4096, momentum=0.5):
        super(NPIDLoss, self).__init__()
        self.n = n
        self.negs = negs
        self.proj_dim = proj_dim
        self.momentum = momentum
        self.temperature = temperature
        # init memory bank as unit random vector ---> [N, Dim]
        self.register_buffer('bank', F.normalize(torch.randn(n, proj_dim), dim=-1))
        # z as normalizer, init with None
        self.z = None

    def forward(self, proj, pos_index):
        batch_size = proj.size(0)
        # randomly generate Negs+1 sample indexes for each batch ---> [B, Negs+1]
        idx = torch.randint(high=self.n, size=(batch_size, self.negs + 1))
        # make the first sample as positive
        idx[:, 0] = pos_index
        # select memory vectors from memory bank ---> [B, 1+Negs, Dim]
        samples = torch.index_select(self.bank, dim=0, index=idx.view(-1)).view(batch_size, -1, self.proj_dim)
        # compute cos similarity between each feature vector and memory bank ---> [B, 1+Negs]
        sim_matrix = torch.bmm(samples.to(device=proj.device), proj.unsqueeze(dim=-1)).view(batch_size, -1)
        out = torch.exp(sim_matrix / self.temperature)
        # Monte Carlo approximation, use the approximation derived from initial batches as z
        if self.z is None:
            self.z = out.detach().mean().item() * self.n
        # compute P(i|v) ---> [B, 1+Negs]
        output = out / self.z

        # compute loss
        # compute log(h(i|v))=log(P(i|v)/(P(i|v)+Negs*P_n(i))) ---> [B]
        p_d = (output.select(dim=-1, index=0) / (output.select(dim=-1, index=0) + self.negs / self.n)).log()
        # compute log(1-h(i|v'))=log(1-P(i|v')/(P(i|v')+Negs*P_n(i))) ---> [B, Negs]
        p_n = ((self.negs / self.n) / (output.narrow(dim=-1, start=1, length=self.negs) + self.negs / self.n)).log()
        # compute J_NCE(θ)=-E(P_d)-Negs*E(P_n)
        loss = - (p_d.sum() + p_n.sum()) / batch_size

        pos_samples = samples.select(dim=1, index=0)
        return loss, pos_samples

    def enqueue(self, proj, pos_index, pos_samples):
        # update memory bank ---> [B, Dim]
        pos_samples = proj.detach().cpu() * self.momentum + pos_samples * (1.0 - self.momentum)
        pos_samples = F.normalize(pos_samples, dim=-1)
        self.bank.index_copy_(0, pos_index, pos_samples)


class SimSiamLoss(nn.Module):
    def __init__(self):
        super(SimSiamLoss, self).__init__()

    def forward(self, feature_1, feature_2, proj_1, proj_2):
        sim_1 = -(proj_1 * feature_2.detach()).sum(dim=-1).mean()
        sim_2 = -(proj_2 * feature_1.detach()).sum(dim=-1).mean()
        loss = 0.5 * sim_1 + 0.5 * sim_2
        return loss
