import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms


class Identity(object):
    def __call__(self, im):
        return im


class RGBToBGR(object):
    def __call__(self, im):
        assert im.mode == 'RGB'
        r, g, b = [im.getchannel(i) for i in range(3)]
        im = Image.merge('RGB', [b, g, r])
        return im


class ScaleIntensities(object):
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __call__(self, tensor):
        tensor = (tensor - self.in_range[0]) / (self.in_range[1] - self.in_range[0]) * (
                self.out_range[1] - self.out_range[0]) + self.out_range[0]
        return tensor


class ImageReader(Dataset):

    def __init__(self, data_path, data_name, data_type, backbone_type):
        data_dict = torch.load('{}/{}/uncropped_data_dicts.pth'.format(data_path, data_name))[data_type]
        self.class_to_idx = dict(zip(sorted(data_dict), range(len(data_dict))))
        if backbone_type == 'inception':
            normalize = transforms.Normalize([104, 117, 128], [1, 1, 1])
        else:
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if data_type == 'train':
            self.transform = transforms.Compose([
                RGBToBGR() if backbone_type == 'inception' else Identity(),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ScaleIntensities([0, 1], [0, 255]) if backbone_type == 'inception' else Identity(),
                normalize])
        else:
            self.transform = transforms.Compose([
                RGBToBGR() if backbone_type == 'inception' else Identity(),
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(),
                ScaleIntensities([0, 1], [0, 255]) if backbone_type == 'inception' else Identity(),
                normalize])
        self.images, self.labels = [], []
        for label, image_list in data_dict.items():
            self.images += image_list
            self.labels += [self.class_to_idx[label]] * len(image_list)

    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.images)


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()


def recall(feature_vectors, feature_labels, rank):
    feature_labels = torch.tensor(feature_labels, device=feature_vectors.device)
    sim_matrix = feature_vectors.mm(feature_vectors.t())
    sim_matrix.fill_diagonal_(-np.inf)

    idx = sim_matrix.topk(k=rank[-1], dim=-1, largest=True)[1]
    acc_list = []
    for r in rank:
        correct = (torch.eq(feature_labels[idx[:, 0:r]], feature_labels.unsqueeze(dim=-1))).any(dim=-1)
        acc_list.append((torch.sum(correct) / correct.size(0)).item())
    return acc_list


class UnifiedProxyLoss(nn.Module):
    def __init__(self, scale=32, margin=0.1, ratio=0.8):
        super(UnifiedProxyLoss, self).__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, output, label):
        pos_label = F.one_hot(label, num_classes=output.size(-1))
        neg_label = 1 - pos_label
        pos_num = pos_label.sum(dim=0)
        pos_num = torch.where(torch.ne(pos_num, 0), pos_num, torch.ones_like(pos_num))
        neg_num = neg_label.sum(dim=0)
        neg_num = torch.where(torch.ne(neg_num, 0), neg_num, torch.ones_like(neg_num))
        pos_output = torch.exp(-self.scale * (output - self.margin))
        neg_output = torch.exp(self.scale * (output + self.margin))
        pos_output = torch.where(torch.eq(pos_label, 1), pos_output, torch.zeros_like(pos_output)).sum(dim=0)
        neg_output = torch.where(torch.eq(neg_label, 1), neg_output, torch.zeros_like(neg_output)).sum(dim=0)
        loss = torch.mean(torch.log(pos_output / pos_num + neg_output / neg_num + 1))
        return loss
