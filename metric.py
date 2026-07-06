from typing import Tuple

import numpy as np
import torch
from skimage import measure


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _logits_to_prob(pred):
    if torch.is_tensor(pred):
        return torch.sigmoid(pred).detach().cpu().numpy()
    pred = np.asarray(pred, dtype=np.float32)
    # If already in [0,1], keep it. Otherwise treat it as logits.
    if pred.min() >= 0.0 and pred.max() <= 1.0:
        return pred
    return 1.0 / (1.0 + np.exp(-pred))


def _prepare_arrays(pred, labels, thresh: float = 0.5):
    prob = _logits_to_prob(pred)
    gt = _to_numpy(labels).astype(np.float32)

    if prob.ndim == 3:
        prob = prob[:, None]
    if gt.ndim == 3:
        gt = gt[:, None]

    pred_bin = (prob > float(thresh)).astype(np.uint8)
    gt_bin = (gt > 0.5).astype(np.uint8)
    return pred_bin, gt_bin, prob


class SigmoidMetric:
    """Dataset-level pixel accuracy and IoU using sigmoid threshold 0.5."""

    def __init__(self, score_thresh: float = 0.5):
        self.score_thresh = float(score_thresh)
        self.reset()

    def update(self, pred, labels):
        pred_bin, gt_bin, _ = _prepare_arrays(pred, labels, self.score_thresh)
        tp = np.logical_and(pred_bin == 1, gt_bin == 1).sum()
        pred_pos = (pred_bin == 1).sum()
        gt_pos = (gt_bin == 1).sum()
        union = pred_pos + gt_pos - tp
        correct = np.logical_and(pred_bin == gt_bin, gt_bin == 1).sum()

        self.total_correct += float(correct)
        self.total_label += float(gt_pos)
        self.total_inter += float(tp)
        self.total_union += float(union)

    def get(self):
        pix_acc = self.total_correct / (self.total_label + np.spacing(1))
        iou = self.total_inter / (self.total_union + np.spacing(1))
        return pix_acc, iou

    def reset(self):
        self.total_inter = 0.0
        self.total_union = 0.0
        self.total_correct = 0.0
        self.total_label = 0.0


class SamplewiseSigmoidMetric:
    """Sample-wise nIoU metric."""

    def __init__(self, nclass: int = 1, score_thresh: float = 0.5):
        self.nclass = nclass
        self.score_thresh = float(score_thresh)
        self.reset()

    def update(self, preds, labels):
        pred_bin, gt_bin, _ = _prepare_arrays(preds, labels, self.score_thresh)
        b = pred_bin.shape[0]
        for i in range(b):
            tp = np.logical_and(pred_bin[i] == 1, gt_bin[i] == 1).sum()
            pred_pos = (pred_bin[i] == 1).sum()
            gt_pos = (gt_bin[i] == 1).sum()
            union = pred_pos + gt_pos - tp
            self.total_inter = np.append(self.total_inter, float(tp))
            self.total_union = np.append(self.total_union, float(union))

    def get(self):
        if self.total_union.size == 0:
            return np.array([]), 0.0
        iou = self.total_inter / (self.total_union + np.spacing(1))
        return iou, float(iou.mean())

    def reset(self):
        self.total_inter = np.array([], dtype=np.float64)
        self.total_union = np.array([], dtype=np.float64)


class ROCMetric:
    """ROC/PR curve arrays over thresholds from 0 to 1."""

    def __init__(self, nclass: int = 1, bins: int = 10):
        self.nclass = nclass
        self.bins = int(bins)
        self.reset()

    def update(self, preds, labels):
        prob = _logits_to_prob(preds)
        gt = _to_numpy(labels).astype(np.float32)
        if prob.ndim == 3:
            prob = prob[:, None]
        if gt.ndim == 3:
            gt = gt[:, None]
        gt = (gt > 0.5).astype(np.uint8)

        for i_bin in range(self.bins + 1):
            thresh = i_bin / float(self.bins)
            pred = (prob > thresh).astype(np.uint8)
            tp = np.logical_and(pred == 1, gt == 1).sum()
            fp = np.logical_and(pred == 1, gt == 0).sum()
            tn = np.logical_and(pred == 0, gt == 0).sum()
            fn = np.logical_and(pred == 0, gt == 1).sum()
            self.tp_arr[i_bin] += tp
            self.fp_arr[i_bin] += fp
            self.pos_arr[i_bin] += tp + fn
            self.neg_arr[i_bin] += fp + tn
            self.class_pos[i_bin] += tp + fp

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 1e-6)
        fp_rates = self.fp_arr / (self.neg_arr + 1e-6)
        recall = self.tp_arr / (self.pos_arr + 1e-6)
        precision = self.tp_arr / (self.class_pos + 1e-6)
        return tp_rates, fp_rates, recall, precision

    def reset(self):
        self.tp_arr = np.zeros(self.bins + 1, dtype=np.float64)
        self.pos_arr = np.zeros(self.bins + 1, dtype=np.float64)
        self.fp_arr = np.zeros(self.bins + 1, dtype=np.float64)
        self.neg_arr = np.zeros(self.bins + 1, dtype=np.float64)
        self.class_pos = np.zeros(self.bins + 1, dtype=np.float64)


class PD_FA:
    """
    Object-level probability of detection and false-alarm rate.

    A predicted connected component matches a GT component when the centroid
    distance is smaller than match_distance pixels. False alarms are counted by
    the area of unmatched predicted components, and Fa is normalized by the
    total number of image pixels.
    """

    def __init__(self, nclass: int = 1, bins: int = 10, match_distance: float = 3.0):
        self.nclass = nclass
        self.bins = int(bins)
        self.match_distance = float(match_distance)
        self.reset()

    def update(self, preds, labels):
        prob = _logits_to_prob(preds)
        gt = _to_numpy(labels).astype(np.float32)
        if prob.ndim == 3:
            prob = prob[:, None]
        if gt.ndim == 3:
            gt = gt[:, None]
        gt = (gt > 0.5).astype(np.uint8)

        b, _, h, w = prob.shape
        for i_bin in range(self.bins + 1):
            thresh = i_bin / float(self.bins)
            pred_bin = (prob > thresh).astype(np.uint8)
            self.total_pixels[i_bin] += b * h * w

            for bi in range(b):
                pred_2d = pred_bin[bi, 0]
                gt_2d = gt[bi, 0]

                pred_label = measure.label(pred_2d, connectivity=2)
                gt_label = measure.label(gt_2d, connectivity=2)
                pred_props = list(measure.regionprops(pred_label))
                gt_props = list(measure.regionprops(gt_label))

                self.target[i_bin] += len(gt_props)
                matched_pred = set()
                matched_gt = 0

                for g in gt_props:
                    gyx = np.asarray(g.centroid, dtype=np.float32)
                    best_dist = None
                    best_idx = None
                    for pi, p in enumerate(pred_props):
                        if pi in matched_pred:
                            continue
                        pyx = np.asarray(p.centroid, dtype=np.float32)
                        dist = np.linalg.norm(pyx - gyx)
                        if best_dist is None or dist < best_dist:
                            best_dist = dist
                            best_idx = pi
                    if best_idx is not None and best_dist is not None and best_dist < self.match_distance:
                        matched_pred.add(best_idx)
                        matched_gt += 1

                false_area = 0.0
                for pi, p in enumerate(pred_props):
                    if pi not in matched_pred:
                        false_area += float(p.area)

                self.PD[i_bin] += matched_gt
                self.FA[i_bin] += false_area

    def get(self, img_num=None):
        final_fa = self.FA / (self.total_pixels + 1e-6)
        final_pd = self.PD / (self.target + 1e-6)
        return final_fa, final_pd

    def reset(self):
        self.FA = np.zeros(self.bins + 1, dtype=np.float64)
        self.PD = np.zeros(self.bins + 1, dtype=np.float64)
        self.target = np.zeros(self.bins + 1, dtype=np.float64)
        self.total_pixels = np.zeros(self.bins + 1, dtype=np.float64)


def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):
    prob = torch.sigmoid(output)
    predict = (prob > score_thresh).float()
    if len(target.shape) == 3:
        target = target.unsqueeze(1).float()
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    target = (target > 0.5).float()
    tp = (predict * target).sum()
    fp = (predict * (1 - target)).sum()
    tn = ((1 - predict) * (1 - target)).sum()
    fn = ((1 - predict) * target).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos = tp + fp
    return tp, pos, fp, neg, class_pos
