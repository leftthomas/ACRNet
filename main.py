import itertools
import os
import random

import pandas as pd
import torch
from PIL import Image
from thop import clever_format, profile
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from model import Backbone, Generator, Discriminator, SimCLRLoss
from utils import DomainDataset, weights_init_normal, ReplayBuffer, parse_common_args, get_transform, val_contrast, \
    obtain_style_code

parser = parse_common_args()
parser.add_argument('--style_num', default=8, type=int, help='Number of used styles')
parser.add_argument('--gan_iter', default=4000, type=int, help='Number of bp to train gan model')
parser.add_argument('--rounds', default=5, type=int, help='Number of round to train whole model')

# args parse
args = parser.parse_args()
data_root, data_name, method_name, train_domains = args.data_root, args.data_name, args.method_name, args.train_domains
val_domains, hidden_dim, temperature, batch_size = args.val_domains, args.hidden_dim, args.temperature, args.batch_size
style_num, gan_iter, contrast_iter = args.style_num, args.gan_iter, args.total_iter
ranks, save_root, rounds = args.ranks, args.save_root, args.rounds
# asserts
assert method_name == 'zsco', 'not support for {}'.format(method_name)

# data prepare
train_contrast_data = DomainDataset(data_root, data_name, train_domains, train=True)
train_contrast_loader = DataLoader(train_contrast_data, batch_size=batch_size, shuffle=True, num_workers=8,
                                   drop_last=True)
val_contrast_data = DomainDataset(data_root, data_name, val_domains, train=False)
val_contrast_loader = DataLoader(val_contrast_data, batch_size=batch_size, shuffle=False, num_workers=8)
val_gan_data = DomainDataset(data_root, data_name, val_domains, train=False, style_num=style_num)
val_gan_loader = DataLoader(val_gan_data, batch_size=1, shuffle=False, num_workers=8)

# model setup
backbone = Backbone(hidden_dim).cuda()
# optimizer config
optimizer_backbone = Adam(backbone.parameters(), lr=1e-3, weight_decay=1e-6)

# loss setup
criterion_adversarial = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_contrast = SimCLRLoss(temperature)

contrast_results = {'train_loss': [], 'val_precise': []}
save_name_pre = '{}_{}_{}_{}_{}'.format(data_name, method_name, style_num, rounds, gan_iter)
if not os.path.exists(save_root):
    os.makedirs(save_root)
best_precise, total_contrast_loss = 0.0, 0.0

# training loop
for r in range(1, rounds + 1):
    # each round should refresh style images
    train_gan_data = DomainDataset(data_root, data_name, train_domains, train=True, style_num=style_num)
    style_images, style_names, style_categories, style_labels = train_gan_data.refresh(style_num)
    style_codes = obtain_style_code(style_num, size=[224, 224])
    train_gan_loader = DataLoader(train_gan_data, batch_size=1, shuffle=True, num_workers=8)
    # use conditional F, G, DF and DG
    F = Generator(3 + style_num, 3).cuda()
    G = Generator(3 + style_num, 3).cuda()
    DF = Discriminator(3 + style_num).cuda()
    DG = Discriminator(3 + style_num).cuda()
    F.apply(weights_init_normal)
    G.apply(weights_init_normal)
    DF.apply(weights_init_normal)
    DG.apply(weights_init_normal)
    optimizer_FG = Adam(itertools.chain(F.parameters(), G.parameters()), lr=2e-4, betas=(0.5, 0.999))
    optimizer_DF = Adam(DF.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_DG = Adam(DG.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # compute macs and params
    if r == 1:
        macs_f, params_f = profile(F, inputs=(torch.randn(style_num, 3 + style_num, 224, 224).cuda(),), verbose=False)
        macs_df, params_df = profile(DF, inputs=(torch.randn(style_num, 3 + style_num, 224, 224).cuda(),),
                                     verbose=False)
        macs, params = clever_format([(macs_f + macs_df) * 2, (params_f + params_df) * 2], '%.2f')
        print('Params: {}; MACs: {}'.format(params, macs))

    fake_style_buffer = [ReplayBuffer() for _ in range(style_num)]
    fake_content_buffer = [ReplayBuffer() for _ in range(style_num)]
    gan_epochs = (gan_iter // len(train_gan_data)) + 1
    contrast_epochs = (contrast_iter // (len(train_contrast_data) // batch_size)) + 1
    gan_results = {'train_fg_loss': [], 'train_df_loss': [], 'train_dg_loss': []}
    total_fg_loss, total_df_loss, total_dg_loss, current_gan_iter, current_contrast_iter = 0.0, 0.0, 0.0, 0, 0

    lr_scheduler_FG = LambdaLR(optimizer_FG,
                               lr_lambda=lambda eiter: 1.0 - max(0, eiter - gan_iter // 2) / float(gan_iter // 2))
    lr_scheduler_DF = LambdaLR(optimizer_DF,
                               lr_lambda=lambda eiter: 1.0 - max(0, eiter - gan_iter // 2) / float(gan_iter // 2))
    lr_scheduler_DG = LambdaLR(optimizer_DG,
                               lr_lambda=lambda eiter: 1.0 - max(0, eiter - gan_iter // 2) / float(gan_iter // 2))

    # GAN training loop
    for epoch in range(1, gan_epochs + 1):
        F.train()
        G.train()
        DF.train()
        DG.train()
        train_bar = tqdm(train_gan_loader, dynamic_ncols=True)
        for content, _, _, _, _, _ in train_bar:
            content = content.squeeze(dim=0).cuda()
            styles = torch.cat((torch.stack([(get_transform(train=True)(style)).cuda() for style in style_images],
                                            dim=0), style_codes.cuda()), dim=1)
            # F and G
            optimizer_FG.zero_grad()
            fake_style = F(content)
            fake_content = G(styles)
            pred_fake_style = DG(torch.cat((fake_style, style_codes.cuda()), dim=1))
            pred_fake_content = DF(torch.cat((fake_content, style_codes.cuda()), dim=1))
            # adversarial loss
            target_fake_style = torch.ones(pred_fake_style.size(), device=pred_fake_style.device)
            target_fake_content = torch.ones(pred_fake_content.size(), device=pred_fake_content.device)
            adversarial_loss = criterion_adversarial(pred_fake_style, target_fake_style) + criterion_adversarial(
                pred_fake_content, target_fake_content)
            # cycle loss
            cycle_loss = criterion_cycle(G(torch.cat((fake_style, style_codes.cuda()), dim=1)), content[:, :3, :, :]) \
                         + criterion_cycle(F(torch.cat((fake_content, style_codes.cuda()), dim=1)), styles[:, :3, :, :])
            fg_loss = adversarial_loss + 10 * cycle_loss
            fg_loss.backward()
            optimizer_FG.step()
            lr_scheduler_FG.step()
            total_fg_loss += fg_loss.item()
            # DF
            optimizer_DF.zero_grad()
            pred_real_content = DF(content)
            target_real_content = torch.ones(pred_real_content.size(), device=pred_real_content.device)
            fake_content = torch.stack([content_buffer.push_and_pop(fake_c[:3, :, :]) for fake_c, content_buffer in
                                        zip(fake_content, fake_content_buffer)], dim=0)
            pred_fake_content = DF(torch.cat((fake_content, style_codes.cuda()), dim=1))
            target_fake_content = torch.zeros(pred_fake_content.size(), device=pred_fake_content.device)
            adversarial_loss = (criterion_adversarial(pred_real_content, target_real_content)
                                + criterion_adversarial(pred_fake_content, target_fake_content)) / 2
            adversarial_loss.backward()
            optimizer_DF.step()
            lr_scheduler_DF.step()
            total_df_loss += adversarial_loss.item()
            # DG
            optimizer_DG.zero_grad()
            pred_real_style = DG(styles)
            target_real_style = torch.ones(pred_real_style.size(), device=pred_real_style.device)
            fake_style = torch.stack([style_buffer.push_and_pop(fake_s[:3, :, :]) for fake_s, style_buffer in
                                      zip(fake_style, fake_style_buffer)], dim=0)
            pred_fake_style = DG(torch.cat((fake_style, style_codes.cuda()), dim=1))
            target_fake_style = torch.zeros(pred_fake_style.size(), device=pred_fake_style.device)
            adversarial_loss = (criterion_adversarial(pred_real_style, target_real_style)
                                + criterion_adversarial(pred_fake_style, target_fake_style)) / 2
            adversarial_loss.backward()
            optimizer_DG.step()
            lr_scheduler_DG.step()
            total_dg_loss += adversarial_loss.item()

            current_gan_iter += 1
            train_bar.set_description('[{}/{}] Train Iter: [{}/{}] FG Loss: {:.4f}, DF Loss: {:.4f}, DG Loss: {:.4f}'
                                      .format(r, rounds, current_gan_iter, gan_iter, total_fg_loss / current_gan_iter,
                                              total_df_loss / current_gan_iter, total_dg_loss / current_gan_iter))
            if current_gan_iter % 100 == 0:
                gan_results['train_fg_loss'].append(total_fg_loss / current_gan_iter)
                gan_results['train_df_loss'].append(total_df_loss / current_gan_iter)
                gan_results['train_dg_loss'].append(total_dg_loss / current_gan_iter)
                # save statistics
                data_frame = pd.DataFrame(data=gan_results, index=range(1, current_gan_iter // 100 + 1))
                save_path = '{}/{}/round-{}/results.csv'.format(save_root, save_name_pre, r)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                data_frame.to_csv(save_path, index_label='iter')
            # stop iter data when arriving the gan bp numbers
            if current_gan_iter == gan_iter:
                # save the generated images for val data by current round model and styles
                F.eval()
                with torch.no_grad():
                    for image, name, category, label in zip(style_images, style_names, style_categories, style_labels):
                        domain = val_gan_data.domains[category]
                        name = os.path.basename(name)
                        label = val_gan_data.classes.inverse[label]
                        path = '{}/{}/round-{}/{}_{}_{}'.format(save_root, save_name_pre, r, domain, label, name)
                        if not os.path.exists(os.path.dirname(path)):
                            os.makedirs(os.path.dirname(path))
                        image.save(path)
                    for img, _, img_name, category, label, _ in tqdm(val_gan_loader,
                                                                     desc='Generate images for specific styles',
                                                                     dynamic_ncols=True):
                        fake_style = (F(img.squeeze(dim=0).cuda()) + 1.0) / 2
                        for fake_s, style_category, style_label, style_name in zip(fake_style, style_categories,
                                                                                   style_labels, style_names):
                            style_domain = val_gan_data.domains[style_category]
                            style_name = os.path.basename(style_name)
                            style_label = val_gan_data.classes.inverse[style_label]
                            domain = val_gan_data.domains[category[0].item()]
                            name = os.path.basename(img_name[0])
                            label_name = val_gan_data.classes.inverse[label[0].item()]
                            img_path = '{}/{}/round-{}/{}_{}_{}/{}_{}_{}'.format(save_root, save_name_pre, r,
                                                                                 style_domain, style_label,
                                                                                 style_name.split('.')[0], domain,
                                                                                 label_name, name)
                            if not os.path.exists(os.path.dirname(img_path)):
                                os.makedirs(os.path.dirname(img_path))
                            save_image(fake_s, img_path)
                F.train()
                # save models
                torch.save(F.state_dict(), '{}/{}/round-{}/F.pth'.format(save_root, save_name_pre, r))
                torch.save(G.state_dict(), '{}/{}/round-{}/G.pth'.format(save_root, save_name_pre, r))
                torch.save(DF.state_dict(), '{}/{}/round-{}/DF.pth'.format(save_root, save_name_pre, r))
                torch.save(DG.state_dict(), '{}/{}/round-{}/DG.pth'.format(save_root, save_name_pre, r))
                break
    # contrast training loop
    F.eval()
    for epoch in range(1, contrast_epochs + 1):
        backbone.train()
        train_bar = tqdm(train_contrast_loader, dynamic_ncols=True)
        for img_1, _, img_name, _, _, _ in train_bar:
            img_1 = img_1.cuda()
            _, proj_1 = backbone(img_1)
            with torch.no_grad():
                # TODO
                fs = random.choices(F, k=batch_size)
                img_2 = []
                for f, img in zip(fs, img_name):
                    img_2.append(f((get_transform('train')(Image.open(img))).unsqueeze(dim=0).cuda()))
                img_2 = torch.cat(img_2, dim=0)
            _, proj_2 = backbone(img_2)
            loss = criterion_contrast(proj_1, proj_2)
            optimizer_backbone.zero_grad()
            loss.backward()
            optimizer_backbone.step()
            current_contrast_iter += 1
            total_contrast_loss += loss.item()
            train_bar.set_description('Train Iter: [{}/{}] Contrast Loss: {:.4f}'
                                      .format(current_contrast_iter + contrast_iter * (r - 1), contrast_iter * rounds,
                                              total_contrast_loss / (current_contrast_iter + contrast_iter * (r - 1))))
            if current_contrast_iter % 100 == 0:
                contrast_results['train_loss'].append(
                    total_contrast_loss / (current_contrast_iter + contrast_iter * (r - 1)))
                # every 100 iters to val the model
                val_precise, features = val_contrast(backbone, val_contrast_loader, contrast_results,
                                                     ranks, current_contrast_iter + contrast_iter * (r - 1),
                                                     contrast_iter * rounds)
                # save statistics
                data_frame = pd.DataFrame(data=contrast_results,
                                          index=range(1, (current_contrast_iter + contrast_iter * (r - 1)) // 100 + 1))
                data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='iter')

                if val_precise > best_precise:
                    best_precise = val_precise
                    torch.save(backbone.state_dict(), '{}/{}_model.pth'.format(save_root, save_name_pre))
                    torch.save(features, '{}/{}_vectors.pth'.format(save_root, save_name_pre))
            # stop iter data when arriving the contrast bp numbers
            if current_contrast_iter == contrast_iter:
                break
    F.train()
