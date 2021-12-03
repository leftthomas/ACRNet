import argparse
import glob
import os
import random
import subprocess

import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.backends import cudnn


def parse_args():
    desc = 'Pytorch Implementation of \'Mining Relations for Weakly-Supervised Action Localization\''
    parse = argparse.ArgumentParser(description=desc)
    parse.add_argument('--data_path', type=str, default='/data')
    parse.add_argument('--save_path', type=str, default='result')
    parse.add_argument('--data_name', type=str, default='thumos14',
                       choices=['thumos14', 'activitynet1.2', 'activitynet1.3'])
    parse.add_argument('--act_th', type=float, default=0.2, help='threshold for action score')
    parse.add_argument('--iou_th', type=float, default=0.6, help='threshold for NMS IoU')
    parse.add_argument('--seg_th', type=str, default='np.arange(0.0, 0.25, 0.025)',
                       help='threshold for candidate segments')
    parse.add_argument('--mag_th', type=str, default='np.arange(0.4, 0.625, 0.025)',
                       help='threshold for candidate actions')
    parse.add_argument('--num_seg', type=int, default=750, help='used segments for each video')
    parse.add_argument('--select_ratio', type=float, default=0.1,
                       help='selected top/bottom k segments for action/background')
    parse.add_argument('--temperature', type=float, default=0.05,
                       help='used temperature scale for softmax')
    parse.add_argument('--alpha', type=float, default=0.0005)
    parse.add_argument('--fps', type=int, default=25)
    parse.add_argument('--rate', type=int, default=16, help='number of frames in each segment')
    parse.add_argument('--num_iter', type=int, default=10000)
    parse.add_argument('--eval_iter', type=int, default=100)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--seed', type=int, default=-1, help='random seed (-1 for no manual seed)')
    parse.add_argument('--model_file', type=str, default=None, help='the path of pre-trained model file')

    return init_args(parse.parse_args())


class Config(object):
    def __init__(self, arg):
        self.data_path = arg.data_path
        self.save_path = arg.save_path
        self.data_name = arg.data_name
        self.act_th = arg.act_th
        self.iou_th = arg.iou_th
        self.seg_th = eval(arg.seg_th)
        self.mag_th = eval(arg.mag_th)
        self.map_th = arg.map_th
        self.num_seg = arg.num_seg
        self.select_ratio = arg.select_ratio
        self.temperature = arg.temperature
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


if __name__ == '__main__':
    description = 'Extract the RGB and Flow features from videos with assigned fps'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_path', type=str, default='/data')
    parser.add_argument('--save_path', type=str, default='result')
    parser.add_argument('--data_name', type=str, default='thumos14', choices=['thumos14', 'activitynet'])
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--data_split', type=str, required=True)
    args = parser.parse_args()

    data_path, save_path, data_name, data_split = args.data_path, args.save_path, args.data_name, args.data_split
    fps, ffmpeg_path = args.fps, which_ffmpeg()
    videos = sorted(glob.glob('{}/{}/videos/{}/*'.format(data_path, data_name, data_split)))
    total = len(videos)

    for i, video_path in enumerate(videos):
        dir_name, video_name = os.path.dirname(video_path).split('/')[-1], os.path.basename(video_path).split('.')[0]
        save_root = '{}/{}/{}/{}'.format(save_path, data_name, dir_name, video_name)
        # pass the already precessed videos
        try:
            os.makedirs(save_root)
        except OSError:
            continue
        print('[{}/{}] Saving {} to {}/{}.mp4 with {} fps'.format(i + 1, total, video_path, save_root, video_name, fps))
        ffmpeg_cmd = '{} -hide_banner -loglevel panic -i {} -r {} -y {}/{}.mp4' \
            .format(ffmpeg_path, video_path, fps, save_root, video_name)
        subprocess.call(ffmpeg_cmd.split())
        flow_cmd = './denseFlow_gpu -f={}/{}.mp4 -o={}'.format(save_root, video_name, save_root)
        subprocess.call(flow_cmd.split())
