import argparse
import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm


def parse_common_args():
    # for reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    cudnn.deterministic = True
    cudnn.benchmark = False
    # common args
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--data_root', default='data', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='pacs', type=str, choices=['pacs', 'office'], help='Dataset name')
    parser.add_argument('--method_name', default='zsco', type=str,
                        choices=['zsco', 'simclr', 'npid', 'proxyanchor', 'softtriple', 'pretrained'],
                        help='Compared method name')
    parser.add_argument('--train_domains', nargs='+', default=['cartoon', 'photo'], type=str,
                        help='Selected domains to train')
    parser.add_argument('--val_domains', nargs='+', default=['sketch', 'art'], type=str,
                        help='Selected domains to val')
    parser.add_argument('--proj_dim', default=128, type=int, help='Projected feature dim for computing loss')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=32, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--total_iter', default=10000, type=int, help='Number of bp to train')
    parser.add_argument('--ranks', nargs='+', default=[1, 5, 10], type=int, help='Selected recall to val')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')
    return parser


def get_transform(split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, (1.0, 1.14)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class DomainDataset(Dataset):
    def __init__(self, data_root, data_name, domains, split='train'):
        super(DomainDataset, self).__init__()

        self.data_name = data_name
        self.domains = domains
        self.images, self.categories, self.labels = [], [], []
        for i, domain in enumerate(self.domains):
            images = sorted(glob.glob(os.path.join(data_root, data_name, split, domain, '*.png')))
            # which image
            self.images += images
            # which domain
            self.categories += [i] * len(images)
            # which instance
            self.labels += range(0, len(images))
            self.num_class = len(images)
        self.transform = get_transform(split)

    def __getitem__(self, index):
        img_name = self.images[index]
        img = Image.open(img_name)
        img_1 = self.transform(img)
        img_2 = self.transform(img)
        category = self.categories[index]
        label = self.labels[index]
        return img_1, img_2, img_name, category, label, index

    def __len__(self):
        return len(self.images)

    def refresh(self, style_num):
        images, names, categories, labels = [], [], [], []
        # need reverse del index to avoid the del index not exist error
        indexes = sorted(random.sample(range(0, len(self.images)), k=style_num), reverse=True)
        for i in indexes:
            name = self.images.pop(i)
            names.append(name)
            images.append(Image.open(name))
            categories.append(self.categories.pop(i))
            labels.append(self.labels.pop(i))
        return images, names, categories, labels


def recall(vectors, ranks, domains, labels):
    labels = torch.as_tensor(labels, device=vectors.device)
    acc = {}
    domain_a_vectors = vectors[: len(vectors) // 2]
    domain_b_vectors = vectors[len(vectors) // 2:]
    domain_a_labels = labels[: len(labels) // 2]
    domain_b_labels = labels[len(labels) // 2:]
    # A -> B
    sim_ab = domain_a_vectors.mm(domain_b_vectors.t())
    idx_ab = sim_ab.topk(k=ranks[-1], dim=-1, largest=True)[1]
    # B -> A
    sim_ba = domain_b_vectors.mm(domain_a_vectors.t())
    idx_ba = sim_ba.topk(k=ranks[-1], dim=-1, largest=True)[1]
    # cross A and B
    sim = vectors.mm(vectors.t())
    sim.fill_diagonal_(-np.inf)
    idx = sim.topk(k=ranks[-1], dim=-1, largest=True)[1]

    for r in ranks:
        correct_ab = (torch.eq(domain_b_labels[idx_ab[:, 0:r]], domain_a_labels.unsqueeze(dim=-1))).any(dim=-1)
        acc['{}->{}@{}'.format(domains[0], domains[1], r)] = (torch.sum(correct_ab) / correct_ab.size(0)).item()
        correct_ba = (torch.eq(domain_a_labels[idx_ba[:, 0:r]], domain_b_labels.unsqueeze(dim=-1))).any(dim=-1)
        acc['{}->{}@{}'.format(domains[1], domains[0], r)] = (torch.sum(correct_ba) / correct_ba.size(0)).item()
        correct = (torch.eq(labels[idx[:, 0:r]], labels.unsqueeze(dim=-1))).any(dim=-1)
        acc['{}<->{}@{}'.format(domains[0], domains[1], r)] = (torch.sum(correct) / correct.size(0)).item()
    # the cross recall is chosen as the representative of precise
    acc['val_precise'] = acc['{}<->{}@{}'.format(domains[0], domains[1], ranks[0])]
    return acc


# val for all val data
def val_contrast(net, data_loader, results, ranks, current_iter, total_iter):
    net.eval()
    vectors = []
    with torch.no_grad():
        for data, _, _, _, _, _ in tqdm(data_loader, desc='Feature extracting', dynamic_ncols=True):
            vectors.append(net(data.cuda())[0])
        vectors = torch.cat(vectors, dim=0)
        acc = recall(vectors, ranks, data_loader.dataset.domains, data_loader.dataset.labels)
        precise = acc['val_precise'] * 100
        print('Val Iter: [{}/{}] Precise: {:.2f}%'.format(current_iter, total_iter, precise))
        for key, value in acc.items():
            if key in results:
                results[key].append(value * 100)
            else:
                results[key] = [value * 100]
    net.train()
    return precise, vectors


class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.detach():
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)
