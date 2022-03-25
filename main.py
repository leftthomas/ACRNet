import json

import numpy as np
import pandas as pd
import torch
from mmaction.core.evaluation import ActivityNetLocalization
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VideoDataset
from model import Model, cross_entropy, graph_consistency, mutual_entropy, fuse_act_score
from utils import parse_args, oic_score, revert_frame, grouping, result2json, draw_pred


def test_loop(net, data_loader, num_iter):
    net.eval()
    results, num_correct, num_total, test_info = {'results': {}}, 0, 0, {}
    with torch.no_grad():
        for data, gt, video_name, num_seg in tqdm(data_loader, initial=1, dynamic_ncols=True):
            data, gt, video_name, num_seg = data.cuda(), gt.squeeze(0).cuda(), video_name[0], num_seg.squeeze(0)
            _, rgb_cas, flow_cas, seg_score, ori_rgb_graph, rgb_graph, flow_graph = net(data)
            act_rgb_score, act_rgb_th = fuse_act_score(rgb_cas, flow_cas.detach())
            act_flow_score, act_flow_th = fuse_act_score(flow_cas, rgb_cas.detach())
            act_score = (act_rgb_score + act_flow_score) / 2
            act_th = (act_rgb_th + act_flow_th) / 2
            # [C],  [T, C],  [T, C]
            act_score, rgb_score, flow_score = act_score.squeeze(0), rgb_cas.squeeze(0), flow_cas.squeeze(0)
            # [T, C],  [T, T]
            seg_score, ori_rgb_graph = seg_score.squeeze(0), ori_rgb_graph.squeeze(0).cpu().numpy()
            # [T, T],  [T, T]
            rgb_graph, flow_graph = rgb_graph.squeeze(0).cpu().numpy(), flow_graph.squeeze(0).cpu().numpy()
            # [C]
            act_th = act_th.squeeze(0).cpu().numpy()

            pred = torch.ge(act_score, args.cls_th)
            num_correct += 1 if torch.equal(gt, pred.float()) else 0
            num_total += 1

            frame_score = revert_frame(seg_score.cpu().numpy(), args.rate * num_seg.item())
            # make sure the score between [0, 1]
            frame_score = np.clip(frame_score, a_min=0.0, a_max=1.0)

            rgb_score = revert_frame(rgb_score.cpu().numpy(), args.rate * num_seg.item())
            rgb_score = np.clip(rgb_score, a_min=0.0, a_max=1.0)
            flow_score = revert_frame(flow_score.cpu().numpy(), args.rate * num_seg.item())
            flow_score = np.clip(flow_score, a_min=0.0, a_max=1.0)

            proposal_dict = {}
            for i, status in enumerate(pred):
                if status:
                    proposals = grouping(np.where(frame_score[:, i] >= act_th[i])[0])
                    # make sure the proposal to be regions
                    for proposal in proposals:
                        if len(proposal) >= 2:
                            if i not in proposal_dict:
                                proposal_dict[i] = []
                            score = oic_score(frame_score[:, i], act_score[i].cpu().numpy(), proposal)
                            # change frame index to second
                            start, end = (proposal[0] + 1) / args.fps, (proposal[-1] + 2) / args.fps
                            proposal_dict[i].append([start, end, score])
            if args.save_vis:
                # draw the pred to vis
                draw_pred(frame_score, rgb_score, flow_score, ori_rgb_graph, rgb_graph, flow_graph, act_th,
                          data_loader.dataset.annotations, data_loader.dataset.idx_to_class,
                          data_loader.dataset.class_to_idx, video_name, args.fps, args.save_path, args.data_name)
            results['results'][video_name] = result2json(proposal_dict, data_loader.dataset.idx_to_class)

        test_acc = num_correct / num_total

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
        torch.save(model.state_dict(), '{}/{}.pth'.format(args.save_path, args.data_name))


if __name__ == '__main__':
    args = parse_args()
    test_data = VideoDataset(args.data_path, args.data_name, 'test', args.num_seg)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers)

    model = Model(len(test_data.class_to_idx), args.factor).cuda()
    best_mAP, metric_info = 0, {}
    if args.model_file:
        model.load_state_dict(torch.load(args.model_file))
        save_loop(model, test_loader, 1)

    else:
        train_data = VideoDataset(args.data_path, args.data_name, 'train', args.num_seg,
                                  args.batch_size * args.num_iter)
        train_loader = iter(DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers))
        optimizer = Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)

        total_loss, total_num, metric_info['Loss'] = 0.0, 0, []
        train_bar = tqdm(range(1, args.num_iter + 1), initial=1, dynamic_ncols=True)
        for step in train_bar:
            model.train()
            feat, label, _, _ = next(train_loader)
            feat, label = feat.cuda(), label.cuda()
            act_scores, rgb_cass, flow_cass, _, _, rgb_graphs, flow_graphs = model(feat)
            cas_loss = cross_entropy(act_scores, label)
            graph_loss = graph_consistency(rgb_graphs, flow_graphs)
            plus_cas_loss = (mutual_entropy(rgb_cass, flow_cass.detach(), label) +
                             mutual_entropy(flow_cass, rgb_cass.detach(), label)) / 2
            ori_weight = (args.num_iter - step + 1) / args.num_iter
            plus_weight = 1.0 - ori_weight
            loss = ori_weight * cas_loss + graph_loss + plus_weight * plus_cas_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_num += feat.size(0)
            total_loss += loss.item() * feat.size(0)
            train_bar.set_description('Train Step: [{}/{}] Loss: {:.3f}'
                                      .format(step, args.num_iter, total_loss / total_num))
            if step % args.eval_iter == 0:
                metric_info['Loss'].append('{:.3f}'.format(total_loss / total_num))
                save_loop(model, test_loader, step)
