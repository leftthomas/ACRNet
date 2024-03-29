import argparse
import os
import random
import subprocess

import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.backends import cudnn


def parse_args():
    desc = 'Pytorch Implementation of \'Weakly-supervised Temporal Action Localization with Adaptive ' \
           'Clustering and Refining Network\''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_path', type=str, default='/home/data')
    parser.add_argument('--save_path', type=str, default='result')
    parser.add_argument('--data_name', type=str, default='thumos14',
                        choices=['thumos14', 'activitynet1.2', 'activitynet1.3'])
    parser.add_argument('--cls_th', type=float, default=0.1, help='threshold for action classification')
    parser.add_argument('--iou_th', type=float, default=0.1, help='threshold for NMS IoU')
    parser.add_argument('--act_th', type=str, default='np.arange(0.1, 1.0, 0.05)',
                        help='threshold for candidate frames')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha value for soft nms')
    parser.add_argument('--num_seg', type=int, default=750, help='sampled segments for each video')
    parser.add_argument('--fps', type=int, default=25, help='fps for each video')
    parser.add_argument('--rate', type=int, default=16, help='number of frames in each segment')
    parser.add_argument('--num_iter', type=int, default=2000, help='iterations of training')
    parser.add_argument('--eval_iter', type=int, default=100, help='iterations of evaluating')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of loading videos for training')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay for optimizer')
    parser.add_argument('--lamda', type=float, default=0.1, help='loss weight for aas loss')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 for no manual seed)')
    parser.add_argument('--model_file', type=str, default=None, help='the path of pre-trained model file')

    return init_args(parser.parse_args())


class Config(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.save_path = args.save_path
        self.data_name = args.data_name
        self.cls_th = args.cls_th
        self.iou_th = args.iou_th
        self.act_th = eval(args.act_th)
        self.map_th = args.map_th
        self.alpha = args.alpha
        self.num_seg = args.num_seg
        self.fps = args.fps
        self.rate = args.rate
        self.num_iter = args.num_iter
        self.eval_iter = args.eval_iter
        self.batch_size = args.batch_size
        self.init_lr = args.init_lr
        self.weight_decay = args.weight_decay
        self.lamda = args.lamda
        self.workers = args.workers
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


# ref: D2-Net: Weakly-Supervised Action Localization via Discriminative Embeddings and Denoised Activations (ICCV 2021)
def filter_results(results, ambi_file):
    ambi_list = [line.strip('\n').split(' ') for line in list(open(ambi_file, 'r'))]
    for key, value in results['results'].items():
        for filter_item in ambi_list:
            if filter_item[0] == key:
                filtered = []
                for item in value:
                    if max(float(filter_item[2]), item['segment'][0]) < min(float(filter_item[3]), item['segment'][1]):
                        continue
                    else:
                        filtered.append(item)
                value = filtered
        results['results'][key] = value
    return results


# ref: Completeness Modeling and Context Separation for Weakly Supervised Temporal Action Localization (CVPR 2019)
def oic_score(frame_scores, act_score, proposal, _lambda=0.25, gamma=0.2):
    inner_score = np.mean(frame_scores[proposal])
    outer_s = max(0, int(proposal[0] - _lambda * len(proposal)))
    outer_e = min(int(frame_scores.shape[0] - 1), int(proposal[-1] + _lambda * len(proposal)))
    outer_temp_list = list(range(outer_s, int(proposal[0]))) + list(range(int(proposal[-1] + 1), outer_e + 1))

    if len(outer_temp_list) == 0:
        outer_score = 0.0
    else:
        outer_score = np.mean(frame_scores[outer_temp_list])
    score = inner_score - outer_score + gamma * act_score
    return score
