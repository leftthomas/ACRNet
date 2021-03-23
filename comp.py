import os

import pandas as pd
import torch
from pytorch_metric_learning.losses import ProxyAnchorLoss, SoftTripleLoss
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Backbone, SimCLRLoss, NPIDLoss, SimSiamLoss
from utils import DomainDataset, val_contrast, parse_common_args

parser = parse_common_args()
# args parse
args = parser.parse_args()
data_root, data_name, method_name, train_domains = args.data_root, args.data_name, args.method_name, args.train_domains
val_domains, hidden_dim, temperature, batch_size = args.val_domains, args.hidden_dim, args.temperature, args.batch_size
total_iter, ranks, save_root = args.total_iter, args.ranks, args.save_root
# asserts
assert method_name != 'zsco', 'not support for {}'.format(method_name)

# data prepare
train_data = DomainDataset(data_root, data_name, domains=train_domains, train=True)
val_data = DomainDataset(data_root, data_name, domains=val_domains, train=False)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)

# model setup
model = Backbone(hidden_dim, pretrained=method_name == 'pretrained').cuda()
# optimizer config
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
if method_name == 'npid':
    loss_criterion = NPIDLoss(len(train_data), proj_dim=2048, temperature=temperature)
elif method_name == 'simclr':
    loss_criterion = SimCLRLoss(temperature)
elif method_name == 'simsiam':
    loss_criterion = SimSiamLoss()
elif method_name == 'proxyanchor':
    loss_criterion = ProxyAnchorLoss(len(train_data.classes), 2048).cuda()
    loss_optimizer = Adam(loss_criterion.parameters(), lr=1e-3, weight_decay=1e-6)
elif method_name == 'softtriple':
    loss_criterion = SoftTripleLoss(len(train_data.classes), 2048).cuda()
    loss_optimizer = Adam(loss_criterion.parameters(), lr=1e-3, weight_decay=1e-6)

results = {'train_loss': [], 'val_precise': []}
save_name_pre = '{}_{}_{}'.format(data_name, method_name, train_domains)
if not os.path.exists(save_root):
    os.makedirs(save_root)
best_precise, total_loss, current_iter = 0.0, 0.0, 0
epochs = (total_iter // (len(train_data) // batch_size)) + 1

if method_name == 'pretrained':
    current_iter += 1
    results['train_loss'].append(total_loss / current_iter)
    val_precise, features = val_contrast(model, val_loader, results, ranks, current_iter, total_iter)
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, current_iter + 1))
    data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='iter')
    torch.save(model.state_dict(), '{}/{}_model.pth'.format(save_root, save_name_pre))
    torch.save(features, '{}/{}_vectors.pth'.format(save_root, save_name_pre))
else:
    # train loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_bar = tqdm(train_loader, dynamic_ncols=True)
        for img_1, img_2, _, _, img_label, pos_index in train_bar:
            img_1, img_2 = img_1.cuda(), img_2.cuda()
            feature_1, proj_1 = model(img_1)

            if method_name == 'npid':
                loss, pos_samples = loss_criterion(proj_1, pos_index)
            elif method_name == 'simclr':
                feature_2, proj_2 = model(img_2)
                loss = loss_criterion(proj_1, proj_2)
            elif method_name == 'simsiam':
                feature_2, proj_2 = model(img_2)
                loss = loss_criterion(feature_1, feature_2, proj_1, proj_2)
            elif method_name == 'proxyanchor':
                loss = loss_criterion(proj_1, img_label)
            else:
                loss = loss_criterion(proj_1, img_label)
            optimizer.zero_grad()
            if method_name in ['proxyanchor', 'softtriple']:
                loss_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if method_name in ['proxyanchor', 'softtriple']:
                loss_optimizer.step()

            if method_name == 'npid':
                loss_criterion.enqueue(proj_1, pos_index, pos_samples)

            current_iter += 1
            total_loss += loss.item()
            train_bar.set_description(
                'Train Iter: [{}/{}] Loss: {:.4f}'.format(current_iter, total_iter, total_loss / current_iter))
            if current_iter % 100 == 0:
                results['train_loss'].append(total_loss / current_iter)
                # every 100 iters to val the model
                val_precise, features = val_contrast(model, val_loader, results, ranks, current_iter, total_iter)
                # save statistics
                data_frame = pd.DataFrame(data=results, index=range(1, current_iter // 100 + 1))
                data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='iter')

                if val_precise > best_precise:
                    best_precise = val_precise
                    torch.save(model.state_dict(), '{}/{}_model.pth'.format(save_root, save_name_pre))
                    torch.save(features, '{}/{}_vectors.pth'.format(save_root, save_name_pre))
            # stop iter data when arriving the total bp numbers
            if current_iter == total_iter:
                break
