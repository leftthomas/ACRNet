import argparse
import os
import shutil

import numpy as np
import torch
from PIL import Image, ImageDraw

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Model')
    parser.add_argument('--query_img_name', default='/home/data/car/uncropped/008055.jpg', type=str,
                        help='query image name')
    parser.add_argument('--data_base', default='car_resnet50_512_data_base.pth', type=str, help='queried database')
    parser.add_argument('--retrieval_num', default=8, type=int, help='retrieval number')

    opt = parser.parse_args()

    query_img_name, data_base_name, retrieval_num = opt.query_img_name, opt.data_base, opt.retrieval_num
    data_name = data_base_name.split('_')[0]

    data_base = torch.load('results/{}'.format(data_base_name))

    if query_img_name not in data_base['test_images']:
        raise FileNotFoundError('{} not found'.format(query_img_name))
    query_index = data_base['test_images'].index(query_img_name)
    query_image = Image.open(query_img_name).convert('RGB').resize((224, 224), resample=Image.BILINEAR)
    query_label = torch.tensor(data_base['test_labels'][query_index])
    query_feature = data_base['test_features'][query_index]

    gallery_images = data_base['test_images']
    gallery_labels = torch.tensor(data_base['test_labels'])
    gallery_features = data_base['test_features']

    sim_matrix = query_feature.unsqueeze(0).mm(gallery_features.t()).squeeze()
    sim_matrix[query_index] = -np.inf
    idx = sim_matrix.topk(k=retrieval_num, dim=-1)[1]

    result_path = 'results/{}'.format(query_img_name.split('/')[-1].split('.')[0])
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.mkdir(result_path)
    query_image.save('{}/query_img.jpg'.format(result_path))
    for num, index in enumerate(idx):
        retrieval_image = Image.open(gallery_images[index.item()]).convert('RGB') \
            .resize((224, 224), resample=Image.BILINEAR)
        draw = ImageDraw.Draw(retrieval_image)
        retrieval_label = gallery_labels[index.item()]
        retrieval_status = torch.equal(retrieval_label, query_label)
        retrieval_sim = sim_matrix[index.item()].item()
        if retrieval_status:
            draw.rectangle((0, 0, 223, 223), outline='green', width=8)
        else:
            draw.rectangle((0, 0, 223, 223), outline='red', width=8)
        retrieval_image.save('{}/retrieval_img_{}_{}.jpg'.format(result_path, num + 1, '%.4f' % retrieval_sim))
