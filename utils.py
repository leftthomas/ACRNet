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
    parser.add_argument('--num_seg', type=int, default=750, help='used segments for each video')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--rate', type=int, default=16, help='number of frames in each segment')
    parser.add_argument('--num_iter', type=int, default=10000)
    parser.add_argument('--eval_iter', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 for no manual seed)')
    parser.add_argument('--model_file', type=str, default=None, help='the path of pre-trained model file')

    return init_args(parser.parse_args())


class Config(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.save_path = args.save_path
        self.data_name = args.data_name
        self.act_th = args.act_th
        self.iou_th = args.iou_th
        self.score_th = eval(args.score_th)
        self.map_th = args.map_th
        self.num_seg = args.num_seg
        self.fps = args.fps
        self.rate = args.rate
        self.num_iter = args.num_iter
        self.eval_iter = args.eval_iter
        self.batch_size = args.batch_size
        self.model_file = args.model_file


def init_args(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    args.map_th = np.linspace(0.1, 0.7, 7) if args.data_name == 'thumos14' else np.linspace(0.5, 0.95, 10)
    return Config(args)


# change the segment based scores to frame based scores
def revert_frame(scores, num_frame):
    x = np.arange(scores.shape[0])
    f = interp1d(x, scores, kind='linear', axis=0, fill_value='extrapolate')
    scale = np.arange(num_frame) * scores.shape[0] / num_frame
    return f(scale)


# split frames to action regions
def grouping(frames):
    return np.split(frames, np.where(np.diff(frames) != 1)[0] + 1)


def result2json(result, class_dict):
    result_file = []
    for key, value in result.items():
        for line in value:
            result_file.append({'label': class_dict[key], 'score': float(line[-1]),
                                'segment': [float(line[0]), float(line[1])]})
    return result_file


def which_ffmpeg():
    result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.stdout.decode('utf-8').replace('\n', '')
