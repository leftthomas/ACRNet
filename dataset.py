import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, data_path, data_name, data_type, num_seg):

        self.data_name, self.data_type, self.num_seg = data_name, data_type, num_seg

        # prepare annotations
        if data_name == 'thumos14':
            data_type = 'val' if data_type == 'train' else 'test'
            label_name = 'annotations.json'
        else:
            data_type = 'training' if data_type == 'train' else 'validation'
            label_name = 'annotations_{}.json'.format(data_name[-3:])
            data_name = data_name[:-3]
        with open(os.path.join(data_path, data_name, label_name), 'r') as f:
            annotations = json.load(f)

        # prepare data
        self.rgb, self.flow, self.annotations, classes, self.class_to_idx, self.idx_to_class = [], [], {}, set(), {}, {}
        for key, value in annotations.items():
            if value['subset'] == data_type:
                self.rgb.append('{}/{}/features/{}/{}_rgb.npy'.format(data_path, data_name, data_type, key))
                self.flow.append('{}/{}/features/{}/{}_flow.npy'.format(data_path, data_name, data_type, key))
                # the prefix is added to compatible with ActivityNetLocalization class
                self.annotations['d_{}'.format(key)] = {'annotations': value['annotations']}
                for annotation in value['annotations']:
                    classes.add(annotation['label'])
        for i, key in enumerate(sorted(classes)):
            self.class_to_idx[key] = i
            self.idx_to_class[i] = key

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, index):
        rgb, flow = np.load(self.rgb[index]), np.load(self.flow[index])
        video_name, num_seg = os.path.basename(self.rgb[index]).split('.')[0][: -4], rgb.shape[0]
        annotation = self.annotations['d_{}'.format(video_name)]
        sample_idx = self.random_sampling(num_seg) if self.data_type == 'train' else self.uniform_sampling(num_seg)
        rgb, flow = torch.from_numpy(rgb[sample_idx]), torch.from_numpy(flow[sample_idx])

        label = torch.zeros(len(self.class_to_idx))
        for item in annotation['annotations']:
            label[self.class_to_idx[item['label']]] = 1
        feat = torch.cat((rgb, flow), dim=-1)
        if self.data_type == 'train':
            return feat, label
        else:
            return feat, label, video_name, num_seg

    def random_sampling(self, num_seg):
        sample_idx = np.append(np.arange(self.num_seg) * num_seg / self.num_seg, num_seg)
        for i in range(self.num_seg):
            if int(sample_idx[i]) == int(sample_idx[i + 1]):
                sample_idx[i] = int(sample_idx[i])
            else:
                sample_idx[i] = np.random.randint(int(sample_idx[i]), int(sample_idx[i + 1]))
        return sample_idx[:-1].astype(np.int)

    def uniform_sampling(self, num_seg):
        # because the length may different as these two line codes, make sure batch size == 1 in test mode
        if num_seg <= self.num_seg:
            return np.arange(num_seg).astype(np.int)
        else:
            return np.floor(np.arange(self.num_seg) * num_seg / self.num_seg).astype(np.int)