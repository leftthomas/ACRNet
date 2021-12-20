import argparse
import os
import random
import subprocess

import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.backends import cudnn


def parse_args():
    desc = 'Pytorch Implementation of \'Mining Relations for Weakly-Supervised Action Localization\''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_path', type=str, default='/home/data')
    parser.add_argument('--save_path', type=str, default='result')
    parser.add_argument('--data_name', type=str, default='thumos14',
                        choices=['thumos14', 'activitynet1.2', 'activitynet1.3'])
    parser.add_argument('--act_th', type=float, default=0.2, help='threshold for action score')
    parser.add_argument('--iou_th', type=float, default=0.6, help='threshold for NMS IoU')
    parser.add_argument('--score_th', type=str, default='np.arange(0.0, 0.25, 0.025)',
                        help='threshold for candidate frames with scores')
    parser.add_argument('--norm_th', type=str, default='np.arange(0.4, 0.625, 0.025)',
                        help='threshold for candidate frames with norms')
    parser.add_argument('--num_seg', type=int, default=750, help='used segments for each video')
    parser.add_argument('--ratio', type=float, default=0.1,
                        help='selected top/bottom k segments for action/background')
    parser.add_argument('--alpha', type=float, default=0.0005)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--rate', type=int, default=16, help='number of frames in each segment')
    parser.add_argument('--num_iter', type=int, default=10000)
    parser.add_argument('--eval_iter', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 for no manual seed)')
    parser.add_argument('--model_file', type=str, default=None, help='the path of pre-trained model file')

    return init_args(parser.parse_args())


class Config(object):
    def __init__(self, arg):
        self.data_path = arg.data_path
        self.save_path = arg.save_path
        self.data_name = arg.data_name
        self.act_th = arg.act_th
        self.iou_th = arg.iou_th
        self.score_th = eval(arg.score_th)
        self.norm_th = eval(arg.norm_th)
        self.map_th = arg.map_th
        self.num_seg = arg.num_seg
        self.ratio = arg.ratio
        self.alpha = arg.alpha
        self.fps = arg.fps
        self.rate = arg.rate
        self.num_iter = arg.num_iter
        self.eval_iter = arg.eval_iter
        self.batch_size = arg.batch_size
        self.model_file = arg.model_file


def init_args(arg):
    if not os.path.exists(arg.save_path):
        os.makedirs(arg.save_path)

    if arg.seed >= 0:
        random.seed(arg.seed)
        np.random.seed(arg.seed)
        torch.manual_seed(arg.seed)
        torch.cuda.manual_seed_all(arg.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    arg.map_th = np.linspace(0.1, 0.7, 7) if arg.data_name == 'thumos14' else np.linspace(0.5, 0.95, 10)
    return Config(arg)


# change the segment based scores to frame based scores
def revert_frame(scores, num_frame):
    x = np.arange(scores.shape[0])
    f = interp1d(x, scores, kind='linear', axis=0, fill_value='extrapolate')
    scale = np.arange(num_frame) * scores.shape[0] / num_frame
    return f(scale)


# split frames to action regions
def grouping(frames):
    return np.split(frames, np.where(np.diff(frames) != 1)[0] + 1)


# rescale value to [0, 1]
def minmax_norm(value, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        min_val, max_val = torch.aminmax(value, dim=0, keepdim=True)

    delta = max_val - min_val
    delta = torch.where(delta <= 0.0, torch.ones_like(delta), delta)
    ret = torch.clamp((value - min_val) / delta, min=0.0, max=1.0)
    return ret


def result2json(result, class_dict):
    result_file = []
    for key, value in result.items():
        for line in value:
            result_file.append({'label': class_dict[key], 'score': float(line[-1]),
                                'segment': [float(line[0]), float(line[1])]})
    return result_file


# according OIC loss
def form_region(proposal, frame_score, act_score, fps, inflation=0.25, gamma=0.25):
    inner_score = np.mean(frame_score[proposal])
    outer_s = max(0, int(proposal[0] - inflation * len(proposal)))
    outer_e = min(len(frame_score) - 1, int(proposal[-1] + inflation * len(proposal)))
    outer_list = list(range(outer_s, proposal[0])) + list(range(proposal[-1] + 1, outer_e + 1))
    outer_score = 0.0 if len(outer_list) == 0 else np.mean(frame_score[outer_list])

    score = inner_score - outer_score + gamma * act_score
    # change frame index to second
    start, end = (proposal[0] + 1) / fps, (proposal[-1] + 2) / fps
    return [start, end, score]


def which_ffmpeg():
    result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.stdout.decode('utf-8').replace('\n', '')


if __name__ == '__main__':
    import cv2.cv2 as cv2
    import glob
    from tqdm import tqdm

    videos = sorted(glob.glob('/home/data/activitynet/splits/*/*/*.mp4'))
    new_features = sorted(glob.glob('/home/data/activitynet/features/*/*_rgb.npy'))
    news = {}
    for feature in tqdm(new_features):
        news[os.path.basename(feature).split('.')[0][:-4]] = feature
    for video_name in tqdm(videos):
        video = cv2.VideoCapture(video_name)
        fps = video.get(cv2.CAP_PROP_FPS)
        assert fps == 25
        frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        new_feature = np.load(news[os.path.basename(video_name).split('.')[0]])
        assert len(new_feature) == int(frames - 1) // 16


# if __name__ == '__main__':
#     import glob
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     import pandas as pd
#     from tqdm import tqdm
#
#     sns.set()
#
#     thumos_features = sorted(glob.glob('/data/thumos14/features/test/*_rgb.npy'))
#     activitynet_features = sorted(glob.glob('/data/activitynet/features/training/*_rgb.npy'))
#     thumos_frames, activitynet_frames = [], []
#     for feature in tqdm(thumos_features):
#         thumos_frames.append(len(np.load(feature)))
#     for feature in tqdm(activitynet_features):
#         activitynet_frames.append(len(np.load(feature)))
#
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     axes[0].set_title('thumos14', fontsize=18)
#     axes[1].set_title('activitynet', fontsize=18)
#     sns.histplot(pd.DataFrame({'count': thumos_frames}), ax=axes[0])
#     sns.histplot(pd.DataFrame({'count': activitynet_frames}), ax=axes[1])
#     plt.savefig('result/dist.pdf', bbox_inches='tight', pad_inches=0.1)
