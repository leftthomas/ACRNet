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
        for feat, label, video_name, num_seg, _ in tqdm(data_loader, initial=1, dynamic_ncols=True):
            feat, label, video_name = feat.cuda(), label.squeeze(0).cuda(), video_name[0]
            # pass the videos which contains only 1 frame
            if feat.shape[1] <= 1:
                continue
            num_seg, num_segments = num_seg.item(), feat.shape[1]
            score_act, score_bkg, score_cas, feat_act, feat_bkg, feat = network(feat)
            score_act, score_bkg, score_cas = score_act.squeeze(0), score_bkg.squeeze(0), score_cas.squeeze(0)
            feat_act, feat_bkg, feat = feat_act.squeeze(0), feat_bkg.squeeze(0), feat.squeeze(0)

            correct_pred = torch.sum(torch.eq(label, torch.ge(score_act, config.act_th).float()), dim=-1,
                                     keepdim=True)
            num_correct += torch.sum(correct_pred == network.num_classes).item()
            num_total += correct_pred.shape[0]

            # obtain the action class
            pred = torch.nonzero(torch.ge(score_act, config.act_th)).squeeze(dim=-1)
            if len(pred) == 0:
                pred = torch.argmax(score_act, dim=-1, keepdim=True)

            # calculate posterior probabilities for segments
            feat_magnitudes = torch.norm(feat, dim=-1)
            feat_magnitudes_act = torch.mean(torch.norm(feat_act, dim=-1), dim=-1, keepdim=True)
            feat_magnitudes_bkg = torch.mean(torch.norm(feat_bkg, dim=-1), dim=-1, keepdim=True)
            feat_magnitudes = utils.minmax_norm(feat_magnitudes, max_val=feat_magnitudes_act,
                                                min_val=feat_magnitudes_bkg)
            feat_magnitudes = feat_magnitudes.unsqueeze(dim=-1).expand(-1, network.num_classes)

            # do another minmax norm to rescale the posterior probabilities between [0, 1]
            score_cas = utils.minmax_norm(score_cas * feat_magnitudes)
            score_pred = utils.revert_frame(score_cas[:, pred].cpu().numpy(), config.scale)
            feat_magnitudes_pred = utils.revert_frame(feat_magnitudes[:, pred].cpu().numpy(), config.scale)

            proposal_dict, status = {}, True
            # enrich the proposal pool by using multiple thresholds
            for threshold, temp_pred in zip([config.seg_th, config.mag_th], [score_pred, feat_magnitudes_pred]):
                for i in range(len(threshold)):
                    filtered_pred = temp_pred.copy()
                    filtered_pred[np.where(filtered_pred < threshold[i])] = 0

                    seg_list = []
                    # select the candidate segments
                    for c in range(len(pred)):
                        seg_list.append(np.where(filtered_pred[:, c] > 0))
                    # obtain the proposals
                    scores = filtered_pred if status else score_pred
                    proposals = utils.get_proposal(seg_list, scores, score_act.cpu().numpy(), pred.cpu().numpy(),
                                                   config.scale, num_seg, config.fps, config.sampling_frames,
                                                   num_segments)
                    for j in range(len(proposals)):
                        if len(proposals[j]) == 0:
                            continue
                        class_id = proposals[j][0][0]
                        if class_id not in proposal_dict.keys():
                            proposal_dict[class_id] = []
                        proposal_dict[class_id] += np.array(proposals[j])[:, 1:].tolist()
                status = False
            final_proposals = {}
            # temporal nms
            for class_id in proposal_dict.keys():
                proposals = temporal_nms(np.array(proposal_dict[class_id]), config.iou_th)
                final_proposals[class_id] = proposals.tolist()
            results['results'][video_name] = utils.result2json(final_proposals, data_loader.dataset.idx_to_class_name)

        test_acc = num_correct / num_total

        gt_path = os.path.join(config.save_path, '{}_gt.json'.format(config.data_name))
        with open(gt_path, 'w') as f:
            json.dump(data_loader.dataset.annotations, f, indent=4)
        pred_path = os.path.join(config.save_path, '{}_pred.json'.format(config.data_name))
        with open(pred_path, 'w') as f:
            json.dump(results, f, indent=4)

        map_thresh = config.map_th
        # evaluate the metrics
        evaluator_atl = ActivityNetLocalization(gt_path, pred_path, tiou_thresholds=map_thresh, verbose=False)
        mAP, mAP_avg = evaluator_atl.evaluate()

        desc = 'Test Step: [{}/{}] Test ACC: {:.1f} mAP@AVG: {:.1f}'.format(step, config.num_iter, test_acc * 100,
                                                                            mAP_avg * 100)
        metric_info['Test ACC'] = round(test_acc * 100, 1)
        metric_info['mAP@AVG'] = round(mAP_avg * 100, 1)
        for i in range(map_thresh.shape[0]):
            desc += ' mAP@{:.2f}: {:.1f}'.format(map_thresh[i], mAP[i] * 100)
            metric_info['mAP@{:.2f}'.format(map_thresh[i])] = round(mAP[i] * 100, 1)
        print(desc)
        return metric_info


if __name__ == '__main__':
    args = utils.parse_args()
    test_loader = DataLoader(VideoDataset(args.data_path, args.data_name, 'test', args.num_segments), batch_size=1,
                             shuffle=False, num_workers=args.num_workers, worker_init_fn=args.worker_init_fn)

    net = Model(args.r_act, args.r_bkg, len(test_loader.dataset.class_name_to_idx))

    test_info = test_loop(net, args, test_loader, 0)
    with open(os.path.join(args.save_path, '{}_record.json'.format(args.data_name)), 'w') as f:
        json.dump(test_info, f, indent=4)
