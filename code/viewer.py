import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

from model import GatedSCNN
from utils import get_palette

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict segmentation result from a given image')
    parser.add_argument('--model_weight', type=str, default='../user_data/model.pth', help='Pretrained model weight')
    parser.add_argument('--input_pic', type=str, default='../tcdata/suichang_round1_test_partA_210120/000001.tif',
                        help='Path to the input picture')
    # args parse
    args = parser.parse_args()
    model_weight, input_pic = args.model_weight, args.input_pic

    image = Image.open(input_pic)
    image_height, image_width = image.height, image.width
    num_width = 2 if 'test' in input_pic else 3
    target = Image.new('RGB', (image_width * num_width, image_height))
    images = [image.convert('RGB')]

    image = ToTensor()(cv2.imread(input_pic, cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()).unsqueeze(dim=0).cuda()
    grad = cv2.Canny(cv2.imread(input_pic, cv2.IMREAD_COLOR), 10, 100)
    grad = torch.from_numpy(np.expand_dims(np.asarray(grad, np.float32) / 255.0, axis=0).copy()).unsqueeze(dim=0).cuda()

    # model load
    model = GatedSCNN(in_channels=4, num_classes=10)
    model.load_state_dict(torch.load(model_weight, map_location=torch.device('cpu')))
    model = model.cuda()
    model.eval()

    # predict and save image
    with torch.no_grad():
        output, _ = model(image, grad)
        pred = torch.argmax(output, dim=1)
        pred_image = ToPILImage()(pred.byte().cpu())
        pred_image.putpalette(get_palette())
        if 'test' not in input_pic:
            gt_image = torch.from_numpy(cv2.imread(input_pic.replace('.tif', '.png'), cv2.IMREAD_GRAYSCALE)) - 1
            gt_image = ToPILImage()(gt_image.unsqueeze(dim=0))
            gt_image.putpalette(get_palette())
            images.append(gt_image)
        images.append(pred_image)
        # concat images
        for i in range(len(images)):
            left, top, right, bottom = image_width * i, 0, image_width * (i + 1), image_height
            target.paste(images[i], (left, top, right, bottom))
        target.save(os.path.split(input_pic)[-1].replace('.tif', '.png'))
