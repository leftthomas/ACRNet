import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def str2ind(categoryname, classlist):
    return [i for i in range(len(classlist)) if categoryname == classlist[i]][0]


class ANETdetection(object):

    def __init__(
            self,
            annotation_path,
            tiou_thresholds=np.array([0.1, 0.3, 0.5]),
            args=None,
            subset="test",
            verbose=False
    ):
        self.subset = subset
        self.args = args
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.ap = None
        self.annotation_path = annotation_path
        self.prediction = None
        self._import_ground_truth(self.annotation_path)

    def _import_ground_truth(self, annotation_path):
        gtsegments = np.load(annotation_path + "segments.npy", allow_pickle=True)
        gtlabels = np.load(annotation_path + "labels.npy", allow_pickle=True)
        videoname = np.load(annotation_path + "videoname.npy", allow_pickle=True)
        videoname = np.array([i.decode("utf8") for i in videoname])
        subset = np.load(annotation_path + "subset.npy", allow_pickle=True)
        subset = np.array([s.decode("utf-8") for s in subset])
        classlist = np.load(annotation_path + "classlist.npy", allow_pickle=True)
        classlist = np.array([c.decode("utf-8") for c in classlist])
        duration = np.load(annotation_path + "duration.npy", allow_pickle=True)
        ambilist = annotation_path + "Ambiguous_test.txt"

        try:
            ambilist = list(open(ambilist, "r"))
            ambilist = [a.strip("\n").split(" ") for a in ambilist]
        except:
            ambilist = []

        self.ambilist = ambilist
        self.classlist = classlist

        subset_ind = (subset == self.subset)
        gtsegments = gtsegments[subset_ind]
        gtlabels = gtlabels[subset_ind]
        videoname = videoname[subset_ind]

        self.idx_to_take = [i for i, s in enumerate(gtsegments)
                            if len(s) > 0]

        gtsegments = gtsegments[self.idx_to_take]
        gtlabels = gtlabels[self.idx_to_take]
        videoname = videoname[self.idx_to_take]

        self.videoname = videoname
        # which categories have temporal labels ?
        templabelcategories = sorted(list(set([l for gtl in gtlabels for l in gtl])))

        # # the number index for those categories.
        templabelidx = []
        for t in templabelcategories:
            templabelidx.append(str2ind(t, classlist))

        self.templabelidx = templabelidx

        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []

        for i in range(len(gtsegments)):
            for j in range(len(gtsegments[i])):
                video_lst.append(str(videoname[i]))
                t_start_lst.append(round(gtsegments[i][j][0] * 25 / 16))
                t_end_lst.append(round(gtsegments[i][j][1] * 25 / 16))
                label_lst.append(str2ind(gtlabels[i][j], self.classlist))
        ground_truth = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
            }
        )
        self.ground_truth = ground_truth
        self.activity_index = {i: templabelidx[i] for i in range(len(templabelidx))}

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            print("Warning: No predictions of label '%s' were provdied." % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby("label")
        prediction_by_label = self.prediction.groupby("label")

        results = Parallel(n_jobs=3)(
            delayed(compute_average_precision_detection)(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(
                    drop=True
                ),
                prediction=self._get_predictions_with_label(
                    prediction_by_label, label_name, cidx
                ),
                tiou_thresholds=self.tiou_thresholds,
            )
            for label_name, cidx in self.activity_index.items()
        )

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:, cidx] = results[i]

        return ap

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        if self.verbose:
            print("[INIT] Loaded annotations from {} subset.".format(self.subset))
            nr_gt = len(self.ground_truth)
            print("\tNumber of ground truth instances: {}".format(nr_gt))
            nr_pred = len(self.prediction)
            print("\tNumber of predictions: {}".format(nr_pred))
            print("\tFixed threshold for tiou score: {}".format(self.tiou_thresholds))

        self.ap = self.wrapper_compute_average_precision()
        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        if self.verbose:
            for k in range(len(self.tiou_thresholds)):
                print("Detection map @ %f = %f" % (self.tiou_thresholds[k], self.mAP[k]))
            print("Average-mAP: {}\n".format(self.mAP))
        return self.mAP


def compute_average_precision_detection(
        ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)
):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction["score"].values.argsort()[::-1]  # idx from high to low
    prediction = prediction.loc[sort_idx].reset_index(drop=True)  # value from high to low

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby("video-id")

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred["video-id"])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(
            this_pred[["t-start", "t-end"]].values, this_gt[["t-start", "t-end"]].values
        )
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]["index"]] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]["index"]] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(
            precision_cumsum[tidx, :], recall_cumsum[tidx, :]
        )

    return ap
