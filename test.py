import json
import os

import numpy as np
import torch
from mmaction.core.evaluation import ActivityNetLocalization
from mmaction.localization import temporal_nms
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataset import VideoDataset
from model import Model


def test_loop(network, config, data_loader, step):
    metric_info = {}
    with torch.no_grad():
        if config.model_file is not None:
            network.load_state_dict(torch.load(config.model_file, 'cpu'))
            network = network.cuda()
        network.eval()

        results, num_correct, num_total = {'results': {}}, 0, 0
        for feat, label, video_name, num_seg in tqdm(data_loader, initial=1, dynamic_ncols=True):
            feat, label, video_name, num_seg = feat.cuda(), label.squeeze(0).cuda(), video_name[0], num_seg.squeeze(0)
            num_seg, used_seg = num_seg.item(), feat.shape[1]
            feat, video_score, seg_score = network(feat)
            feat, video_score, seg_score = feat.squeeze(0), video_score.squeeze(0), seg_score.squeeze(0)

            pred = torch.ge(video_score, config.act_th)
            num_correct += 1 if torch.equal(label, pred.float()) else 0
            num_total += 1

            frame_score = utils.revert_frame(seg_score.cpu().numpy(), config.rate * num_seg)
            # make sure the score between [0, 1]
            frame_score[frame_score < 0] = 0.0
            frame_score[frame_score > 1] = 1.0

            proposal_dict = {}
            for i, status in enumerate(pred):
                if status:
                    for frame_th in config.fra_th:
                        proposals = utils.grouping(np.where(frame_score[:, i] >= frame_th)[0])
                        for proposal in proposals:
                            # make sure the proposal to be regions
                            if len(proposal) >= 2:
                                if i not in proposal_dict.keys():
                                    proposal_dict[i] = []
                                start, end, score = proposal[0], proposal[-1], np.mean(frame_score[proposal, i])
                                # change frame index to second
                                start, end = (start + 1) / config.fps, (end + 1) / config.fps
                                proposal_dict[i].append([start, end, score])
                    # temporal nms
                    proposal_dict[i] = temporal_nms(np.array(proposal_dict[i]), config.iou_th).tolist()
            results['results'][video_name] = utils.result2json(proposal_dict, data_loader.dataset.idx_to_class)

        test_acc = num_correct / num_total

        gt_path = os.path.join(config.save_path, '{}_gt.json'.format(config.data_name))
        with open(gt_path, 'w') as json_file:
            json.dump(data_loader.dataset.annotations, json_file, indent=4)
        pred_path = os.path.join(config.save_path, '{}_pred.json'.format(config.data_name))
        with open(pred_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)

        # evaluate the metrics
        evaluator_atl = ActivityNetLocalization(gt_path, pred_path, tiou_thresholds=config.map_th, verbose=False)
        m_ap, m_ap_avg = evaluator_atl.evaluate()

        desc = 'Test Step: [{}/{}] Test ACC: {:.1f} mAP@AVG: {:.1f}'.format(step, config.num_iter, test_acc * 100,
                                                                            m_ap_avg * 100)
        metric_info['Test ACC'] = round(test_acc * 100, 1)
        metric_info['mAP@AVG'] = round(m_ap_avg * 100, 1)
        for i in range(config.map_th.shape[0]):
            desc += ' mAP@{:.2f}: {:.1f}'.format(config.map_th[i], m_ap[i] * 100)
            metric_info['mAP@{:.2f}'.format(config.map_th[i])] = round(m_ap[i] * 100, 1)
        print(desc)
        return metric_info


if __name__ == '__main__':
    args = utils.parse_args()
    dataset = VideoDataset(args.data_path, args.data_name, 'test', args.num_seg)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    net = Model(len(dataset.class_to_idx))

    test_info = test_loop(net, args, test_loader, 0)
    with open(os.path.join(args.save_path, '{}_record.json'.format(args.data_name)), 'w') as f:
        json.dump(test_info, f, indent=4)

#
# if __name__ == '__main__':
#     import cv2.cv2 as cv2
#     import glob
#     videos = sorted(glob.glob('/data/activitynet/splits/*/*/*.mp4'))
#     new_features = sorted(glob.glob('/data/activitynet/features/*/*_rgb.npy'))
#     news = {}
#     for feature in new_features:
#         news[os.path.basename(feature).split('.')[0][:-4]] = feature
#     for video_name in videos:
#         video = cv2.VideoCapture(video_name)
#         fps = video.get(cv2.CAP_PROP_FPS)
#         assert fps == 25
#         frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
#         new_feature = np.load(news[os.path.basename(video_name).split('.')[0]])
#         assert len(new_feature) == int(frames - 1) // 16
