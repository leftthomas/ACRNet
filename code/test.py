import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from dataset import creat_dataset, Tianchi
from model import GatedSCNN


# test for loop all data
def for_loop(net, data_loader, save_root):
    net.eval()

    total_time, total_num, data_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    with torch.no_grad():
        for data, grad, name in data_bar:
            data, grad = data.cuda(), grad.cuda()
            torch.cuda.synchronize()
            start_time = time.time()
            seg, edge = net(data, grad)
            prediction = torch.argmax(seg.detach(), dim=1)
            torch.cuda.synchronize()
            end_time = time.time()

            total_num += data.size(0)
            total_time += end_time - start_time

            # save pred images
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            for pred_tensor, pred_name in zip(prediction, name):
                # revert train id to regular id
                prediction += 1
                pred_img = ToPILImage()(pred_tensor.unsqueeze(dim=0).byte().cpu())
                pred_name = pred_name.replace('.tif', '.png')
                pred_img.save('{}/{}'.format(save_root, pred_name))
            data_bar.set_description('Test Period FPS: {:.0f}'.format(total_num / total_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Gated-SCNN')
    parser.add_argument('--data_path', default='../tcdata/suichang_round1_test_partA_210120', type=str,
                        help='Data path for testing dataset')
    parser.add_argument('--model_weight', type=str, default='../user_data/model.pth', help='Pretrained model weight')
    parser.add_argument('--batch_size', default=16, type=int, help='Number of data for each batch to train')
    parser.add_argument('--save_path', default='../prediction_result', type=str, help='Save path for results')

    # args parse
    args = parser.parse_args()
    data_path, model_weight, batch_size, save_path = args.data_path, args.model_weight, args.batch_size, args.save_path

    # dataset and model setup
    creat_dataset(data_path, num_classes=10, split='test')
    test_data = Tianchi(root=data_path, crop_size=256, split='test')
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=min(4, batch_size))
    model = GatedSCNN(in_channels=4, num_classes=10)
    model.load_state_dict(torch.load(model_weight, map_location=torch.device('cpu')))
    model = model.cuda()

    # test loop
    for_loop(model, test_loader, save_path)
