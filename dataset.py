import argparse
import glob
import json
import os
import subprocess

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import which_ffmpeg


class VideoDataset(Dataset):
    def __init__(self, data_path, data_name, data_type, num_seg, length=None):

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
                # ref: Weakly-supervised Temporal Action Localization by Uncertainty Modeling (AAAI 2021)
                if data_name == 'thumos14' and key in ['video_test_0000270', 'video_test_0001292',
                                                       'video_test_0001496']:
                    continue
                self.rgb.append('{}/{}/features/{}/{}_rgb.npy'.format(data_path, data_name, data_type, key))
                self.flow.append('{}/{}/features/{}/{}_flow.npy'.format(data_path, data_name, data_type, key))
                # the prefix is added to compatible with ActivityNetLocalization class
                self.annotations['d_{}'.format(key)] = {'annotations': value['annotations']}
                for annotation in value['annotations']:
                    classes.add(annotation['label'])
        for i, key in enumerate(sorted(classes)):
            self.class_to_idx[key] = i
            self.idx_to_class[i] = key
        # for train according to the given length, for test according to the real length
        self.num = len(self.rgb)
        self.sample_num = length if self.data_type == 'train' else self.num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        rgb, flow = np.load(self.rgb[index % self.num]), np.load(self.flow[index % self.num])
        video_key, num_seg = os.path.basename(self.rgb[index % self.num]).split('.')[0][: -4], rgb.shape[0]
        annotation = self.annotations['d_{}'.format(video_key)]
        sample_idx = self.random_sampling(num_seg) if self.data_type == 'train' else self.uniform_sampling(num_seg)
        rgb, flow = torch.from_numpy(rgb[sample_idx]), torch.from_numpy(flow[sample_idx])

        label = torch.zeros(len(self.class_to_idx))
        for item in annotation['annotations']:
            label[self.class_to_idx[item['label']]] = 1
        feat = torch.cat((rgb, flow), dim=-1)
        return feat, label, video_key, num_seg

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


if __name__ == '__main__':
    description = 'Extract the RGB and Flow features from videos with assigned fps'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_root', type=str, default='/home/data')
    parser.add_argument('--save_path', type=str, default='result')
    parser.add_argument('--dataset', type=str, default='thumos14', choices=['thumos14', 'activitynet'])
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--data_split', type=str, required=True)
    args = parser.parse_args()

    data_root, save_path, dataset, data_split = args.data_root, args.save_path, args.dataset, args.data_split
    fps, ffmpeg_path = args.fps, which_ffmpeg()
    videos = sorted(glob.glob('{}/{}/videos/{}/*'.format(data_root, dataset, data_split)))
    total = len(videos)

    for j, video_path in enumerate(videos):
        dir_name, video_name = os.path.dirname(video_path).split('/')[-1], os.path.basename(video_path).split('.')[0]
        save_root = '{}/{}/{}/{}'.format(save_path, dataset, dir_name, video_name)
        # pass the already precessed videos
        try:
            os.makedirs(save_root)
        except OSError:
            continue
        print('[{}/{}] Saving {} to {}/{}.mp4 with {} fps'.format(j + 1, total, video_path, save_root, video_name, fps))
        ffmpeg_cmd = '{} -hide_banner -loglevel panic -i {} -r {} -y {}/{}.mp4' \
            .format(ffmpeg_path, video_path, fps, save_root, video_name)
        subprocess.call(ffmpeg_cmd.split())
        flow_cmd = './denseFlow_gpu -f={}/{}.mp4 -o={}'.format(save_root, video_name, save_root)
        subprocess.call(flow_cmd.split())
