import json

import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, data_path, data_name, mode, num_segments):

        self.data_name, self.mode, self.num_segments = data_name, mode, num_segments

        # prepare features
        if data_name == 'thumos14':
            mode = 'val' if mode == 'train' else 'test'
            self.rgb = glob.glob(os.path.join(data_path, data_name, 'features', mode, 'rgb', '*'))
            self.flow = glob.glob(os.path.join(data_path, data_name, 'features', mode, 'flow', '*'))
            with open(os.path.join(data_path, data_name, 'annotations.json'), 'r') as f:
                annotations = json.load(f)['database']
        else:
            mode = 'train' if mode == 'train' else 'val'
            data_name, suffix = data_name[:-3], data_name[-3:]
            self.rgb = glob.glob(os.path.join(data_path, data_name, 'features_{}'.format(suffix), mode, 'rgb', '*'))
            self.flow = glob.glob(os.path.join(data_path, data_name, 'features_{}'.format(suffix), mode, 'flow', '*'))
            with open(os.path.join(data_path, data_name, 'annotations_{}.json'.format(suffix)), 'r') as f:
                annotations = json.load(f)['database']

        # prepare labels
        assert len(self.rgb) == len(self.flow)
        self.annotations, classes, self.class_name_to_idx, self.idx_to_class_name = {}, set(), {}, {}
        for key in self.rgb:
            video_name = os.path.basename(key).split('.')[0]
            value = annotations[video_name]
            # the prefix is added to compatible with ActivityNetLocalization class
            self.annotations['d_{}'.format(video_name)] = {'annotations': value['annotations']}
            for annotation in value['annotations']:
                classes.add(annotation['label'])
        for i, key in enumerate(sorted(classes)):
            self.class_name_to_idx[key] = i
            self.idx_to_class_name[i] = key

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, index):
        rgb, flow = np.load(self.rgb[index]), np.load(self.flow[index])
        video_name, num_seg = os.path.basename(self.rgb[index]).split('.')[0], rgb.shape[0]
        annotation = self.annotations['d_{}'.format(video_name)]
        sample_idx = self.random_sampling(num_seg) if self.mode == 'train' else self.uniform_sampling(num_seg)
        rgb, flow = torch.from_numpy(rgb[sample_idx]), torch.from_numpy(flow[sample_idx])

        label = torch.zeros(len(self.class_name_to_idx))
        for item in annotation['annotations']:
            label[self.class_name_to_idx[item['label']]] = 1
        feat = torch.cat((rgb, flow), dim=-1)
        if self.mode == 'train':
            return feat, label
        else:
            return feat, label, video_name, num_seg, annotation

    def random_sampling(self, length):
        if self.num_segments == length:
            return np.arange(length).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)

    def uniform_sampling(self, length):
        # because the length may different as these two line codes, make sure batch size == 1 in test mode
        if length <= self.num_segments:
            return np.arange(length).astype(int)
        else:
            return np.floor(np.arange(self.num_segments) * length / self.num_segments).astype(int)


if __name__ == '__main__':
    import glob
    import os
    import cv2.cv2 as cv2
    import numpy as np

    videos = sorted(glob.glob('/data/thumos14/splits/*/*/*.mp4'))
    new_features = sorted(glob.glob('/data/thumos14/features/*/*_rgb.npy'))
    news = {}
    for feature in new_features:
        news[os.path.basename(feature).split('.')[0][:-4]] = feature
    for video_name in videos:
        video = cv2.VideoCapture(video_name)
        fps = video.get(cv2.CAP_PROP_FPS)
        assert fps == 25
        frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        new_feature = np.load(news[os.path.basename(video_name).split('.')[0]])
        assert len(new_feature) == int(frames - 1) // 16
