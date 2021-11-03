import argparse
import os
import random

import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.backends import cudnn


def parse_args():
    description = 'Pytorch Implementation of \'Weakly-supervised Temporal Action Localization by Uncertainty Modeling\''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_path', type=str, default='/data')
    parser.add_argument('--save_path', type=str, default='result')
    parser.add_argument('--alpha', type=float, default=0.0005)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--magnitude', type=int, default=100)
    parser.add_argument('--r_act', type=int, default=9)
    parser.add_argument('--r_bkg', type=int, default=4)
    parser.add_argument('--act_th', type=float, default=0.2, help='threshold for action score')
    parser.add_argument('--iou_th', type=float, default=0.6, help='threshold for NMS IoU')
    parser.add_argument('--seg_th', type=str, default='np.arange(0.0, 0.25, 0.025)',
                        help='threshold for candidate segments')
    parser.add_argument('--mag_th', type=str, default='np.arange(0.4, 0.625, 0.025)',
                        help='threshold for candidate actions')
    parser.add_argument('--data_name', type=str, default='thumos14',
                        choices=['thumos14', 'activitynet1.2', 'activitynet1.3'])
    parser.add_argument('--num_segments', type=int, default=750)
    parser.add_argument('--scale', type=int, default=24)
    parser.add_argument('--fps', type=int, default=16)
    parser.add_argument('--sampling_frames', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--decay', type=float, default=0.0005, help='weight decay value for Adam')
    parser.add_argument('--num_iter', type=int, default=10000)
    parser.add_argument('--eval_iter', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 for no manual seed)')
    parser.add_argument('--model_file', type=str, default=None, help='the path of pre-trained model file')

    return init_args(parser.parse_args())


def init_args(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        args.worker_init_fn = np.random.seed(args.seed)
    else:
        args.worker_init_fn = None

    args.seg_th = eval(args.seg_th)
    args.mag_th = eval(args.mag_th)
    args.map_th = np.linspace(0.1, 0.7, 7) if args.data_name == 'thumos14' else np.linspace(0.5, 0.95, 10)
    return args


def upgrade_resolution(arr, scale):
    x = np.arange(0, arr.shape[0])
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale


def get_proposal(seg_list, seg_score, act_score, c_pred, scale, v_len, fps, sampling_frames, num_segments,
                 _lambda=0.25, gamma=0.2):
    t_factor = (fps * v_len) / (scale * num_segments * sampling_frames)
    temp = []
    for i in range(len(seg_list)):
        c_temp = []
        temp_list = np.array(seg_list[i])[0]
        if temp_list.any():
            # obtain the multi action temporal regions
            temp_regions = grouping(temp_list)
            for j in range(len(temp_regions)):
                len_proposal = len(temp_regions[j])
                # omit single frame
                if len_proposal < 2:
                    continue
                inner_score = np.mean(seg_score[temp_regions[j], i])
                outer_s = max(0, int(temp_regions[j][0] - _lambda * len_proposal))
                outer_e = min(int(seg_score.shape[0] - 1), int(temp_regions[j][-1] + _lambda * len_proposal))
                outer_temp_list = list(range(outer_s, int(temp_regions[j][0]))) + list(
                    range(int(temp_regions[j][-1] + 1), outer_e + 1))

                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(seg_score[outer_temp_list, i])
                # obtain the proposal
                c_score = inner_score - outer_score + gamma * act_score[c_pred[i]]
                t_start = temp_regions[j][0] * t_factor
                t_end = (temp_regions[j][-1] + 1) * t_factor
                c_temp.append([c_pred[i], t_start, t_end, c_score])
            temp.append(c_temp)
    return temp


def result2json(result, class_dict):
    result_file = []
    for key, value in result.items():
        for line in value:
            result_file.append({'label': class_dict[key], 'score': line[-1], 'segment': [line[0], line[1]]})
    return result_file


def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        min_val, max_val = torch.aminmax(act_map, dim=0, keepdim=True)
        min_val, max_val = torch.relu(min_val), torch.relu(max_val)

    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta

    ret[ret > 1] = 1
    ret[ret < 0] = 0

    return ret


if __name__ == '__main__':
    import glob
    from tqdm import tqdm
    import cv2.cv2 as cv2

    videos = glob.glob('/data/thumos14/videos/*/*')
    total_fps, min_frames, max_frames = set(), np.inf, -np.inf
    bar = tqdm(videos, dynamic_ncols=True)
    for video_name in bar:
        video = cv2.VideoCapture(video_name)
        fps = video.get(cv2.CAP_PROP_FPS)
        frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        if frames < min_frames:
            min_frames = frames
        if frames > max_frames:
            max_frames = frames
        total_fps.add(fps)
        bar.set_description('min-f: {} max-f: {} number fps: {}'.format(min_frames, max_frames, len(total_fps)))
    print('min-f: {} max-f: {} fps: {}'.format(min_frames, max_frames, total_fps))
