import json
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataset import VideoDataset
from model import Model
from test import test_loop


def train_loop(network, data_loader, train_optimizer, n_iter):
    network.train()
    data, label = next(data_loader)
    data, label = data.cuda(), label.cuda()
    act_label = label / torch.sum(label, dim=-1, keepdim=True)
    bkg_label = torch.ones_like(label)
    bkg_label /= torch.sum(bkg_label, dim=-1, keepdim=True)

    train_optimizer.zero_grad()
    act_norm, bkg_norm, _, act_score, bkg_score, _ = network(data)
    act_loss = bce_criterion(act_score, act_label)
    bkg_loss = bce_criterion(bkg_score, bkg_label)
    # compute norm loss
    min_act_norm, max_act_norm = torch.aminmax(act_norm, dim=-1, keepdim=True)
    min_bkg_norm, max_bkg_norm = torch.aminmax(bkg_norm, dim=-1, keepdim=True)
    min_norm = torch.where(torch.lt(min_act_norm, min_bkg_norm), min_act_norm, min_bkg_norm)
    max_norm = torch.where(torch.gt(max_act_norm, max_bkg_norm), max_act_norm, max_bkg_norm)
    # avoid div by zero
    max_norm = torch.where(torch.eq(max_norm, 0.0), torch.ones_like(max_norm), max_norm)

    act_norm = (act_norm - min_norm.detach()) / max_norm.detach()
    bkg_norm = (bkg_norm - min_norm.detach()) / max_norm.detach()
    norm_loss = torch.mean((1.0 - act_norm + bkg_norm) ** 2)

    loss = act_loss + bkg_loss + args.alpha * norm_loss
    loss.backward()
    train_optimizer.step()

    train_bar.set_description('Train Step: [{}/{}] Loss: {:.3f}'.format(n_iter, args.num_iter, act_loss.item()))


if __name__ == '__main__':
    args = utils.parse_args()
    train_data = VideoDataset(args.data_path, args.data_name, 'train', args.num_seg)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_data = VideoDataset(args.data_path, args.data_name, 'test', args.num_seg)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    net = Model(len(train_data.class_to_idx), args.select_ratio, args.temperature).cuda()
    optimizer = Adam(net.parameters())

    best_mAP, metric_info, bce_criterion = 0, {}, nn.BCELoss()
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
