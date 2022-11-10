import numpy as np


def getAP(conf, labels):
    assert len(conf) == len(labels)
    sortind = np.argsort(-conf)
    tp = labels[sortind] == 1
    fp = labels[sortind] != 1
    npos = np.sum(labels)

    fp = np.cumsum(fp).astype('float32')
    tp = np.cumsum(tp).astype('float32')
    prec = tp / (fp + tp)
    tmp = (labels[sortind] == 1).astype('float32')

    return np.sum(tmp * prec) / npos


def getClassificationMAP(confidence, labels):
    ''' confidence and labels are of dimension n_samples x n_label '''

    AP = []
    for i in range(np.shape(labels)[1]):
        AP.append(getAP(confidence[:, i], labels[:, i]))
    return 100 * sum(AP) / len(AP)


def str2ind(categoryname, classlist):
    return [i for i in range(len(classlist)) if categoryname == classlist[i].decode("utf-8")][0]


def strlist2indlist(strlist, classlist):
    return [str2ind(s, classlist) for s in strlist]


def strlist2multihot(strlist, classlist):
    return np.sum(np.eye(len(classlist))[strlist2indlist(strlist, classlist)], axis=0)


def write_to_file(dname, dmap, cmap, itr):
    fid = open(dname + "-results.log", "a+")
    string_to_write = str(itr)
    # if dmap:
    for item in dmap:
        string_to_write += " " + "%.2f" % item
    string_to_write += " " + "%.2f" % cmap
    fid.write(string_to_write + "\n")
    fid.close()


def soft_nms(dets, iou_thr=0.7, method='gaussian', sigma=0.3):
    dets = np.array(dets)
    x1 = dets[:, 2]
    x2 = dets[:, 3]

    areas = x2 - x1 + 1

    # expand dets with areas, and the second dimension is
    # x1, x2, score, area
    dets = np.concatenate((dets, areas[:, None]), axis=1)

    retained_box = []
    while dets.size > 0:
        max_idx = np.argmax(dets[:, 1], axis=0)
        dets[[0, max_idx], :] = dets[[max_idx, 0], :]
        retained_box.append(dets[0, :-1].tolist())

        xx1 = np.maximum(dets[0, 2], dets[1:, 2])
        xx2 = np.minimum(dets[0, 3], dets[1:, 3])

        inter = np.maximum(xx2 - xx1 + 1, 0.0)
        iou = inter / (dets[0, -1] + dets[1:, -1] - inter)

        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_thr] -= iou[iou > iou_thr]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:  # traditional nms
            weight = np.ones_like(iou)
            weight[iou > iou_thr] = 0

        dets[1:, 1] *= weight
        dets = dets[1:, :]

    return retained_box


def get_proposal_oic(tList, wtcam, final_score, c_pred, lambda_=0.25, gamma=0.2, loss_type="oic"):
    temp = []
    for i in range(len(tList)):
        c_temp = []
        temp_list = np.array(tList[i])[0]
        if temp_list.any():
            grouped_temp_list = grouping(temp_list)
            for j in range(len(grouped_temp_list)):
                inner_score = np.mean(wtcam[grouped_temp_list[j], i, 0])

                len_proposal = len(grouped_temp_list[j])
                outer_s = max(
                    0, int(grouped_temp_list[j][0] - lambda_ * len_proposal))
                outer_e = min(
                    int(wtcam.shape[0] - 1),
                    int(grouped_temp_list[j][-1] + lambda_ * len_proposal),
                )

                outer_temp_list = list(
                    range(outer_s, int(grouped_temp_list[j][0]))) + list(
                    range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))

                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(wtcam[outer_temp_list, i, 0])

                if loss_type == "oic":
                    c_score = inner_score - outer_score + gamma * final_score[
                        c_pred[i]]
                else:
                    c_score = inner_score
                t_start = grouped_temp_list[j][0]
                t_end = (grouped_temp_list[j][-1] + 1)
                c_temp.append([c_pred[i], c_score, t_start, t_end])
            temp.append(c_temp)
    return temp


def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)



