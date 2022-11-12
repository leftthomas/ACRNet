import argparse
import os

import numpy as np
import torch
from PIL import Image
from pytorch_i3d import InceptionI3d
from torchvision.transforms import CenterCrop


def load_frame(frame_file):
    data = Image.open(frame_file)
    assert (min(data.size) == 256)
    data = CenterCrop(size=224)(data)
    data = np.array(data, dtype=np.float32)
    data = (data * 2 / 255) - 1

    assert (data.max() <= 1.0)
    assert (data.min() >= -1.0)

    return data


def load_rgb_batch(frames_dir, rgb_files, frame_indices):
    batch_data = np.zeros(frame_indices.shape + (224, 224, 3), dtype=np.float32)
    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i, j, :, :, :] = load_frame(os.path.join(frames_dir, 'rgb', rgb_files[frame_indices[i][j]]))

    return batch_data


def load_flow_batch(frames_dir, flow_x_files, flow_y_files, frame_indices):
    batch_data = np.zeros(frame_indices.shape + (224, 224, 2), dtype=np.float32)
    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i, j, :, :, 0] = load_frame(os.path.join(frames_dir, 'flow_x',
                                                                flow_x_files[frame_indices[i][j]]))
            batch_data[i, j, :, :, 1] = load_frame(os.path.join(frames_dir, 'flow_y',
                                                                flow_y_files[frame_indices[i][j]]))

    return batch_data


def run(mode='rgb', load_model='', frequency=16, chunk_size=16, input_dir='', output_dir='', batch_size=40):
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)

    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()
    # set model to evaluate mode
    i3d.eval()

    video_names = [i for i in os.listdir(input_dir)]
    for pro_i, video_name in enumerate(video_names):
        save_file = '{}_{}.npy'.format(video_name, mode)
        if save_file in os.listdir(output_dir):
            continue

        frames_dir = os.path.join(input_dir, video_name)
        if mode == 'rgb':
            rgb_files = [i for i in os.listdir(os.path.join(frames_dir, 'rgb'))]
            rgb_files.sort(key=lambda x: int(x.split('.')[0]))
            frame_cnt = len(rgb_files)
        else:
            flow_x_files = [i for i in os.listdir(os.path.join(frames_dir, 'flow_x'))]
            flow_y_files = [i for i in os.listdir(os.path.join(frames_dir, 'flow_y'))]
            flow_x_files.sort(key=lambda x: int(x.split('.')[0]))
            flow_y_files.sort(key=lambda x: int(x.split('.')[0]))
            assert (len(flow_y_files) == len(flow_x_files))
            frame_cnt = len(flow_y_files)

        # cut frames
        assert (frame_cnt > chunk_size)
        clipped_length = frame_cnt - chunk_size
        # the start of last chunk
        clipped_length = (clipped_length // frequency) * frequency
        # frames to chunks
        frame_indices = []
        for i in range(clipped_length // frequency + 1):
            frame_indices.append([j for j in range(i * frequency, i * frequency + chunk_size)])
        frame_indices = np.array(frame_indices)

        chunk_num = frame_indices.shape[0]
        # chunks to batches
        batch_num = int(np.ceil(chunk_num / batch_size))
        frame_indices = np.array_split(frame_indices, batch_num, axis=0)

        full_features = []
        for batch_id in range(batch_num):
            if mode == 'rgb':
                batch_data = load_rgb_batch(frames_dir, rgb_files, frame_indices[batch_id])
            else:
                batch_data = load_flow_batch(frames_dir, flow_x_files, flow_y_files, frame_indices[batch_id])
            with torch.no_grad():
                # [b, c, t, h, w]
                batch_data = torch.from_numpy(batch_data.transpose([0, 4, 1, 2, 3])).cuda()
                batch_feature = i3d.extract_features(batch_data)
                batch_feature = torch.flatten(batch_feature, start_dim=1).cpu().numpy()
            full_features.append(batch_feature)

        full_features = np.concatenate(full_features, axis=0)
        np.save(os.path.join(output_dir, save_file), full_features)
        print('[{}/{}] {} done: {} / {}, {}'.format(pro_i + 1, len(video_names), video_name, frame_cnt,
                                                    clipped_length, full_features.shape))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['rgb', 'flow'])
    parser.add_argument('--load_model', type=str, required=True)
    parser.add_argument('--input_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='result')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--frequency', type=int, default=16)
    parser.add_argument('--chunk_size', type=int, default=16)
    args = parser.parse_args()

    run(mode=args.mode, load_model=args.load_model, frequency=args.frequency, chunk_size=args.chunk_size,
        input_dir=args.input_dir, output_dir=args.output_dir, batch_size=args.batch_size)
