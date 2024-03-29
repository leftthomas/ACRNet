import json

import numpy as np
import pandas as pd
import torch
from mmaction.core.evaluation import ActivityNetLocalization
from mmaction.localization import soft_nms
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VideoDataset
from model import Model, cross_entropy, generalized_cross_entropy
from utils import parse_args, oic_score, revert_frame, grouping, result2json, filter_results


def test_loop(net, data_loader, num_iter):
    net.eval()
    results, num_correct, num_total, test_info = {'results': {}}, 0, 0, {}
    with torch.no_grad():
        for data, gt, video_name, num_seg in tqdm(data_loader, initial=1, dynamic_ncols=True):
            data, gt = data.cuda(non_blocking=True), gt.squeeze(0).cuda(non_blocking=True)
            video_name, num_seg = video_name[0], num_seg.squeeze(0)
            act_score, _, seg_score, _, _, _ = net(data)
            # [C],  [T, C]
            act_score, seg_score = act_score.squeeze(0), seg_score.squeeze(0)

            pred = torch.ge(act_score, args.cls_th)
            # make sure at least one prediction
            if torch.sum(pred) == 0:
                pred[torch.argmax(act_score, dim=-1)] = True
            num_correct += 1 if torch.equal(gt, pred.float()) else 0
            num_total += 1

            frame_score = revert_frame(seg_score.cpu().numpy(), args.rate * num_seg.item())
            # make sure the score between [0, 1]
            frame_score = np.clip(frame_score, a_min=0.0, a_max=1.0)

            proposal_dict = {}
            for i, status in enumerate(pred):
                if status:
                    # enrich the proposal pool by using multiple thresholds
                    for threshold in args.act_th:
                        proposals = grouping(np.where(frame_score[:, i] >= threshold)[0])
                        # make sure the proposal to be regions
                        for proposal in proposals:
                            if len(proposal) >= 2:
                                if i not in proposal_dict:
                                    proposal_dict[i] = []
                                score = oic_score(frame_score[:, i], act_score[i].cpu().numpy(), proposal)
                                # change frame index to second
                                start, end = (proposal[0] + 1) / args.fps, (proposal[-1] + 2) / args.fps
                                proposal_dict[i].append([start, end, score])
                    # temporal soft nms
                    # ref: BSN: Boundary Sensitive Network for Temporal Action Proposal Generation (ECCV 2018)
                    if i in proposal_dict:
                        proposal_dict[i] = soft_nms(np.array(proposal_dict[i]), args.alpha, args.iou_th, args.iou_th,
                                                    top_k=len(proposal_dict[i])).tolist()

            results['results'][video_name] = result2json(proposal_dict, data_loader.dataset.idx_to_class)

        test_acc = num_correct / num_total

        if args.data_name == 'thumos14':
            results = filter_results(results, 'result/ambiguous.txt')
        gt_path = '{}/{}_gt.json'.format(args.save_path, args.data_name)
        with open(gt_path, 'w') as json_file:
            json.dump(data_loader.dataset.annotations, json_file, indent=4)
        pred_path = '{}/{}_pred.json'.format(args.save_path, args.data_name)
        with open(pred_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)

        # evaluate the metrics
        evaluator_atl = ActivityNetLocalization(gt_path, pred_path, tiou_thresholds=args.map_th, verbose=False)
        m_ap, m_ap_avg = evaluator_atl.evaluate()

        desc = 'Test Step: [{}/{}] ACC: {:.1f} mAP@AVG: {:.1f}'.format(num_iter, args.num_iter, test_acc * 100,
                                                                       m_ap_avg * 100)
        test_info['Test ACC'] = round(test_acc * 100, 1)
        test_info['mAP@AVG'] = round(m_ap_avg * 100, 1)
        for i in range(args.map_th.shape[0]):
            desc += ' mAP@{:.2f}: {:.1f}'.format(args.map_th[i], m_ap[i] * 100)
            test_info['mAP@{:.2f}'.format(args.map_th[i])] = round(m_ap[i] * 100, 1)
        print(desc)
        return test_info


def save_loop(net, data_loader, num_iter):
    global best_mAP
    test_info = test_loop(net, data_loader, num_iter)
    for key, value in test_info.items():
        if key not in metric_info:
            metric_info[key] = []
        metric_info[key].append('{:.3f}'.format(value))

    # save statistics
    data_frame = pd.DataFrame(data=metric_info, index=range(1, (num_iter if args.model_file
                                                                else num_iter // args.eval_iter) + 1))
    data_frame.to_csv('{}/{}.csv'.format(args.save_path, args.data_name), index_label='Step', float_format='%.3f')
    if test_info['mAP@AVG'] > best_mAP:
        best_mAP = test_info['mAP@AVG']
        torch.save(net.state_dict(), '{}/{}.pth'.format(args.save_path, args.data_name))


if __name__ == '__main__':
    args = parse_args()
    test_data = VideoDataset(args.data_path, args.data_name, 'test', args.num_seg)
    test_loader = DataLoader(test_data, 1, False, num_workers=args.workers, pin_memory=True)

    model = Model(len(test_data.class_to_idx)).cuda()
    best_mAP, metric_info = 0, {}
    if args.model_file:
        model.load_state_dict(torch.load(args.model_file))
        save_loop(model, test_loader, 1)

    else:
        model.train()
        train_data = VideoDataset(args.data_path, args.data_name, 'train', args.num_seg,
                                  args.batch_size * args.num_iter)
        train_loader = iter(DataLoader(train_data, args.batch_size, True, num_workers=args.workers, pin_memory=True))
        optimizer = Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iter)

        total_loss, total_num, metric_info['Loss'] = 0.0, 0, []
        train_bar = tqdm(range(1, args.num_iter + 1), initial=1, dynamic_ncols=True)
        for step in train_bar:
            feat, label, _, _ = next(train_loader)
            feat, label = feat.cuda(non_blocking=True), label.cuda(non_blocking=True)
            act_score, bkg_score, seg_score, seg_mask, aas_rgb, aas_flow = model(feat)
            cas_loss = cross_entropy(act_score, bkg_score, label)
            aas_rgb_loss = generalized_cross_entropy(aas_rgb, seg_mask, label)
            aas_flow_loss = generalized_cross_entropy(aas_flow, seg_mask, label)
            loss = cas_loss + args.lamda * (aas_rgb_loss + aas_flow_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_num += feat.size(0)
            total_loss += loss.item() * feat.size(0)
            train_bar.set_description('Train Step: [{}/{}] Loss: {:.3f}'
                                      .format(step, args.num_iter, total_loss / total_num))
            lr_scheduler.step()
            if step % args.eval_iter == 0:
                metric_info['Loss'].append('{:.3f}'.format(total_loss / total_num))
                save_loop(model, test_loader, step)
                model.train()
