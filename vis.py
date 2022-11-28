import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataset import VideoDataset
from utils import grouping, revert_frame, parse_args

args = parse_args()
test_data = VideoDataset(args.data_path, args.data_name, 'test', args.num_seg)
gt = test_data.annotations

delu_dict = np.load('result/delu.npy', allow_pickle=True).item()
new_dict = np.load('result/acrnet.npy', allow_pickle=True).item()

delu_scores, acrnet_scores = {}, {}
for key, value in delu_dict.items():
    delu_scores[key.decode()] = torch.softmax(value['cas'], dim=-1) * value['attn']
for key, value in new_dict.items():
    acrnet_scores[key] = value['cas']

for key in tqdm(gt.keys()):
    key = key[2:]
    delu_score = delu_scores[key].squeeze(dim=0).cpu()
    acrnet_score = acrnet_scores[key]
    gts = gt['d_{}'.format(key)]
    num_frames = acrnet_score.shape[0]
    frame_indexes = np.arange(0, num_frames)
    delu_score = np.clip(revert_frame(delu_score.numpy(), num_frames), a_min=0.0, a_max=1.0)

    fig, axs = plt.subplots(5, 1, figsize=(7, 3))
    for i in range(len(gts['annotations'])):
        start, end = gts['annotations'][i]['segment']
        label = gts['annotations'][i]['label']
        label = test_data.class_to_idx[label]
        start, end = int(start * args.fps), min(int(end * args.fps), num_frames - 1)
        count = np.zeros(num_frames)
        count[start:end] = 1
        if start < end:
            axs[0].fill_between(frame_indexes, count, color='green')

    axs[2].plot(frame_indexes, delu_score[:, label], color='blue')
    axs[4].plot(frame_indexes, acrnet_score[:, label], color='red')

    delu_proposals = grouping(np.where(delu_score[:, label] >= 0.2)[0])
    for proposal in delu_proposals:
        if len(proposal) >= 2:
            start, end = proposal[0], proposal[-1]
            end = min(end, num_frames - 1)
            count = np.zeros(num_frames)
            count[start:end] = 1
            if start < end:
                axs[1].fill_between(frame_indexes, count, color='blue')

    acrnet_proposals = grouping(np.where(acrnet_score[:, label] >= 0.3)[0])
    for proposal in acrnet_proposals:
        if len(proposal) >= 2:
            start, end = proposal[0], proposal[-1]
            end = min(end, num_frames - 1)
            count = np.zeros(num_frames)
            count[start:end] = 1
            if start < end:
                axs[3].fill_between(frame_indexes, count, color='red')

    plt.setp(axs, xticks=[], yticks=[], xlim=(0, num_frames), ylim=(0, 1))

    save_name = 'result/{}.pdf'.format(key)
    plt.savefig(save_name, bbox_inches='tight')
    plt.cla()
    plt.close('all')
