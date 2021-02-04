import argparse
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import creat_dataset, Tianchi
from model import GatedSCNN
from utils import compute_metrics, BoundaryBCELoss, DualTaskLoss

# for reproducibility
np.random.seed(1)
torch.manual_seed(1)


# train for one epoch
def for_loop(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_time, total_num, preds, targets = 0.0, 0.0, 0, [], []
    data_bar = tqdm(data_loader, dynamic_ncols=True)
    for data, grad, target, boundary, name in data_bar:
        data, grad, target, boundary = data.cuda(), grad.cuda(), target.cuda(), boundary.cuda()
        torch.cuda.synchronize()
        start_time = time.time()
        seg, edge = net(data, grad)
        prediction = torch.argmax(seg.detach(), dim=1)
        torch.cuda.synchronize()
        end_time = time.time()
        semantic_loss = semantic_criterion(seg, target)
        edge_loss = edge_criterion(edge, target, boundary)
        task_loss = task_criterion(seg, edge, target)
        loss = semantic_loss + 20 * edge_loss + task_loss

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data.size(0)
        total_time += end_time - start_time
        total_loss += loss.item() * data.size(0)
        preds.append(prediction.cpu())
        targets.append(target.cpu())
        data_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} FPS: {:.0f}'
                                 .format(epoch, epochs, total_loss / total_num, total_num / total_time))
    # compute metrics
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    pa, mpa, class_iou = compute_metrics(preds, targets, num_classes=10)
    print('Train Epoch: [{}/{}] PA: {:.2f}% mPA: {:.2f}% mIOU: {:.2f}% '
          .format(epoch, epochs, pa * 100, mpa * 100, class_iou * 100))
    return total_loss / total_num, pa * 100, mpa * 100, class_iou * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Gated-SCNN')
    parser.add_argument('--data_path', default='../tcdata/suichang_round1_train_210120', type=str,
                        help='Data path for training dataset')
    parser.add_argument('--batch_size', default=16, type=int, help='Number of data for each batch to train')
    parser.add_argument('--epochs', default=230, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--save_path', default='../user_data', type=str, help='Save path for results')
    # args parse
    args = parser.parse_args()
    data_path, batch_size, epochs, save_path = args.data_path, args.batch_size, args.epochs, args.save_path

    # dataset, model setup, optimizer config and loss definition
    creat_dataset(data_path, num_classes=10, split='train')
    train_data = Tianchi(root=data_path, crop_size=256, split='train')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=min(4, batch_size),
                              drop_last=True)
    model = GatedSCNN(in_channels=4, num_classes=10).cuda()
    optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda eiter: math.pow(1 - eiter / epochs, 1.0))
    semantic_criterion = nn.CrossEntropyLoss(ignore_index=255)
    edge_criterion = BoundaryBCELoss(ignore_index=255)
    task_criterion = DualTaskLoss(threshold=0.8, ignore_index=255)

    results = {'train_loss': [], 'train_PA': [], 'train_mPA': [], 'train_class_mIOU': []}
    best_mIOU = 0.0
    # train loop
    for epoch in range(1, epochs + 1):
        train_loss, train_PA, train_mPA, train_class_mIOU = for_loop(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_PA'].append(train_PA)
        results['train_mPA'].append(train_mPA)
        results['train_class_mIOU'].append(train_class_mIOU)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/statistics.csv'.format(save_path), index_label='epoch')
        if train_class_mIOU > best_mIOU:
            best_mIOU = train_class_mIOU
            torch.save(model.state_dict(), '{}/model.pth'.format(save_path))
