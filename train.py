import json
import os

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataset import VideoDataset
from model import Model
from test import test_loop


class UMLoss(nn.Module):
    def __init__(self, magnitude):
        super(UMLoss, self).__init__()
        self.magnitude = magnitude

    def forward(self, feat_act, feat_bkg):
        loss_act = torch.relu(self.magnitude - torch.norm(torch.mean(feat_act, dim=-1), dim=-1))
        loss_bkg = torch.norm(torch.mean(feat_bkg, dim=-1), dim=-1)
        loss_um = torch.mean((loss_act + loss_bkg) ** 2)
        return loss_um


def train_loop(network, data_loader, train_optimizer, n_iter):
    network.train()
    data, label = next(data_loader)
    data, label = data.cuda(), label.cuda()
    label_act = label / torch.sum(label, dim=-1, keepdim=True)
    # a trick to flexible use bce Loss to formula be loss
    label_bkg = torch.ones_like(label)
    label_bkg /= torch.sum(label_bkg, dim=-1, keepdim=True)

    train_optimizer.zero_grad()
    score_act, score_bkg, _, feat_act, feat_bkg, _ = network(data)
    cls_loss = bce_criterion(score_act, label_act)
    um_loss = um_criterion(feat_act, feat_bkg)
    be_loss = bce_criterion(score_bkg, label_bkg)
    loss = cls_loss + args.alpha * um_loss + args.beta * be_loss
    loss.backward()
    train_optimizer.step()

    train_bar.set_description('Train Step: [{}/{}] Total Loss: {:.3f} CLS Loss: {:.3f} UM Loss: {:.3f} BE Loss: {:.3f}'
                              .format(n_iter, args.num_iter, loss.item(), cls_loss.item(), um_loss.item(),
                                      be_loss.item()))


if __name__ == '__main__':
    args = utils.parse_args()
    train_loader = DataLoader(VideoDataset(args.data_path, args.data_name, 'train', args.num_seg),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              worker_init_fn=args.worker_init_fn)
    test_loader = DataLoader(VideoDataset(args.data_path, args.data_name, 'test', args.num_seg), batch_size=1,
                             shuffle=False, num_workers=args.num_workers, worker_init_fn=args.worker_init_fn)

    net = Model(args.r_act, args.r_bkg, len(train_loader.dataset.class_name_to_idx)).cuda()

    best_mAP, um_criterion, bce_criterion, metric_info = -1, UMLoss(args.magnitude), nn.BCELoss(), {}
    optimizer = Adam(net.parameters(), lr=args.lr, weight_decay=args.decay)
    train_bar = tqdm(range(1, args.num_iter + 1), total=args.num_iter, initial=1, dynamic_ncols=True)
    for step in train_bar:
        if (step - 1) % len(train_loader) == 0:
            loader_iter = iter(train_loader)

        train_loop(net, loader_iter, optimizer, step)

        if step % args.eval_iter == 0:
            test_info = test_loop(net, args, test_loader, step)
            metric_info['Step {}'.format(step)] = test_info
            with open(os.path.join(args.save_path, '{}_metric.json'.format(args.data_name)), 'w') as f:
                json.dump(metric_info, f, indent=4)

            if test_info['mAP@AVG'] > best_mAP:
                best_mAP = test_info['mAP@AVG']
                with open(os.path.join(args.save_path, '{}_record.json'.format(args.data_name)), 'w') as f:
                    json.dump(test_info, f, indent=4)
                torch.save(net.state_dict(), os.path.join(args.save_path, '{}_model.pth'.format(args.data_name)))
