import glob
import os
import random
import sys
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool

import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from torch.utils.data import Dataset

from utils import ignore_label

grad_progress = 0
boundary_progress = 0


class Tianchi(Dataset):
    def __init__(self, root, crop_size=256, split='train', ignore_label=ignore_label):
        self.crop_size = crop_size
        self.split = split
        self.ignore_label = ignore_label
        search_images = os.path.join(root, '*.tif')
        self.images = glob.glob(search_images)
        self.images.sort()
        search_grads = os.path.join(root.replace('tcdata', 'user_data'), '*grad.png')
        self.grads = glob.glob(search_grads)
        self.grads.sort()

        if self.split == 'train':
            search_labels = os.path.join(root, '*.png')
            self.labels = glob.glob(search_labels)
            self.labels.sort()
            search_boundaries = os.path.join(root.replace('tcdata', 'user_data'), '*boundary.png')
            self.boundaries = glob.glob(search_boundaries)
            self.boundaries.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        grad_path = self.grads[index]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        grad = cv2.imread(grad_path, cv2.IMREAD_GRAYSCALE)
        name = image_path.split('/')[-1]
        if self.split == 'train':
            label_path = self.labels[index]
            boundary_path = self.boundaries[index]
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            boundary = cv2.imread(boundary_path, cv2.IMREAD_GRAYSCALE)

        # random resize, multiple scale training
        if self.split == 'train':
            f_scale = random.choice([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
            image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
            grad = cv2.resize(grad, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
            label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
            boundary = cv2.resize(boundary, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)
        grad = np.asarray(grad, np.float32)
        if self.split == 'train':
            label -= 1
            boundary = np.asarray(boundary, np.float32)
        # change to Nir/RGB
        image = image[:, :, ::-1]
        # normalization
        image /= 255.0
        grad /= 255.0

        # random crop
        if self.split == 'train':
            img_h, img_w = label.shape
            pad_h = max(self.crop_size - img_h, 0)
            pad_w = max(self.crop_size - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                             pad_w, cv2.BORDER_CONSTANT,
                                             value=(0.0, 0.0, 0.0))
                grad_pad = cv2.copyMakeBorder(grad, 0, pad_h, 0,
                                              pad_w, cv2.BORDER_CONSTANT,
                                              value=(0.0,))
                label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                               pad_w, cv2.BORDER_CONSTANT,
                                               value=(self.ignore_label,))
                boundary_pad = cv2.copyMakeBorder(boundary, 0, pad_h, 0,
                                                  pad_w, cv2.BORDER_CONSTANT,
                                                  value=(0.0,))
            else:
                img_pad, grad_pad, label_pad, boundary_pad = image, grad, label, boundary

            img_h, img_w = label_pad.shape
            h_off = random.randint(0, img_h - self.crop_size)
            w_off = random.randint(0, img_w - self.crop_size)
            image = img_pad[h_off: h_off + self.crop_size, w_off: w_off + self.crop_size]
            grad = grad_pad[h_off: h_off + self.crop_size, w_off: w_off + self.crop_size]
            label = label_pad[h_off: h_off + self.crop_size, w_off: w_off + self.crop_size]
            boundary = boundary_pad[h_off: h_off + self.crop_size, w_off: w_off + self.crop_size]

        # HWC -> CHW
        image = image.transpose((2, 0, 1))
        # random horizontal flip
        if self.split == 'train':
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            grad = grad[:, ::flip]
            label = label[:, ::flip]
            boundary = boundary[:, ::flip]

        if self.split == 'train':
            label = np.asarray(label, np.long)
            return image.copy(), np.expand_dims(grad, axis=0).copy(), label.copy(), boundary.copy(), name
        else:
            return image.copy(), np.expand_dims(grad, axis=0).copy(), name


def generate_grad(image_name, total_num):
    # create the output filename
    dst = image_name.replace('tcdata', 'user_data')
    dst = dst.replace('.tif', '_grad.png')
    # do the conversion
    grad_image = cv2.Canny(cv2.imread(image_name, cv2.IMREAD_COLOR), 10, 100)
    cv2.imwrite(dst, grad_image)
    global grad_progress
    grad_progress += 1
    print("\rProgress: {:>3} %".format(grad_progress * 100 / total_num), end=' ')
    sys.stdout.flush()


def generate_boundary(image_name, num_classes, total_num):
    # create the output filename
    dst = image_name.replace('tcdata', 'user_data')
    dst = dst.replace('.png', '_boundary.png')
    # do the conversion
    semantic_image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    onehot_image = np.array([semantic_image == i + 1 for i in range(num_classes)]).astype(np.uint8)
    boundary_image = np.zeros(onehot_image.shape[1:])
    # for boundary conditions
    onehot_image = np.pad(onehot_image, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    for i in range(num_classes):
        dist = distance_transform_edt(onehot_image[i, :]) + distance_transform_edt(1.0 - onehot_image[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > 2] = 0
        boundary_image += dist
    boundary_image = (boundary_image > 0).astype(np.uint8)
    cv2.imwrite(dst, boundary_image)
    global boundary_progress
    boundary_progress += 1
    print("\rProgress: {:>3} %".format(boundary_progress * 100 / total_num), end=' ')
    sys.stdout.flush()


def creat_dataset(root, num_classes, split='train'):
    save_root = root.replace('tcdata', 'user_data')
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    search_path = os.path.join(save_root, '*grad.png')
    if not glob.glob(search_path):
        search_path = os.path.join(root, '*.tif')
        files = glob.glob(search_path)
        files.sort()
        # use multiprocessing to generate grad images
        print('\nGenerating {} grad images'.format(len(files)))
        print("Progress: {:>3} %".format(grad_progress * 100 / len(files)), end=' ')
        pool = ThreadPool()
        pool.map(partial(generate_grad, total_num=len(files)), files)
        pool.close()
        pool.join()

    if split == 'train':
        search_path = os.path.join(save_root, '*boundary.png')
        if not glob.glob(search_path):
            search_path = os.path.join(root, '*.png')
            files = glob.glob(search_path)
            files.sort()
            # use multiprocessing to generate boundary images
            print('\nGenerating {} boundary images'.format(len(files)))
            print("Progress: {:>3} %".format(boundary_progress * 100 / len(files)), end=' ')
            pool = ThreadPool()
            pool.map(partial(generate_boundary, num_classes=num_classes, total_num=len(files)), files)
            pool.close()
            pool.join()
