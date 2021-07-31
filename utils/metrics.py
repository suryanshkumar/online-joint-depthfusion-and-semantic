import numpy as np
import torch


class runningScore(object):
# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
    def __init__(self, n_classes, ignore_index=-100):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.ignore_index = ignore_index

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        if self.ignore_index == 0:
            hist = self.confusion_matrix[1:, 1:]
        else:
            hist = self.confusion_matrix

        # acc = tp / tot_obs
        acc = np.diag(hist).sum() / hist.sum()

        # acc = mean(tp / (tp + fn))
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)

        # iou = mean(tp / (fn + fp + 2tp - tp))
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iou)

        # f = [(tp + fn) / tot_obs] * [tp / (fn + fp + tp)]
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        
        if self.ignore_index == 0:
            iou = np.insert(iou, 0, np.array(0))
        cls_iou = dict(zip(range(self.n_classes), iou))

        return (
            {
                "Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iou,
            },
            cls_iou,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def semantic_evaluation(est, target, mask, n_class):
    # Calculate mean IoU and accuracy on valid ids

    eps = np.finfo(np.float32).eps

    est = est.flatten() * mask.flatten()
    target = target.flatten() * mask.flatten()

    est_ids = np.bincount(np.unique(est.flatten()), minlength=n_class)
    gt_ids = np.bincount(np.unique(target.flatten()), minlength=n_class)

    mask = (target >= 0) & (target < n_class)
    hist = np.bincount(n_class * target[mask].astype(np.uint16) + est[mask], minlength=n_class*n_class)
    hist = hist.reshape(n_class, n_class) # target x estimate

    del est, target, mask

    tp = np.diag(hist)
    fp = hist.sum(axis=0) - tp
    fn = hist.sum(axis=1) - tp

    # valid = ids that are really in the scene, excluded 0 (free space or undefined)
    valid_ids = np.sum(gt_ids) - 1

    acc = tp / (tp + fn + eps)
    mean_cls = np.sum(acc[1:]) / valid_ids

    iou = tp / (tp + fn + fp + eps)
    mean_iou = np.sum(iou[1:]) / valid_ids

    valid = np.where(est_ids | gt_ids)[0]
    iou = iou[valid]
    cls_iou = dict(zip(valid, iou))

    metrics = {
        "Mean Acc": mean_cls,
        "Mean IoU": mean_iou, 
    }

    return metrics, cls_iou


def evaluation(est, target, mask=None):

    est = np.nan_to_num(est.astype(np.float32))
    target = np.nan_to_num(target.astype(np.float32))

    est = np.clip(est, -0.04, 0.04)
    target = np.clip(target, -0.04, 0.04)

    mse = mse_fn(est, target, mask)
    mad = mad_fn(est, target, mask)
    iou = iou_fn(est, target, mask)
    acc = acc_fn(est, target, mask)

    return {'mse': mse,
            'mad': mad,
            'iou': iou,
            'acc': acc}

def rmse_fn(est, target, mask=None):
    eps = 1.e-10

    if mask is not None:
        metric = np.sqrt(np.nansum(mask * np.power(est - target, 2)) / (np.nansum(mask) + eps))
    else:
        metric = np.sqrt(np.nanmean(np.power(est - target, 2)))

    return metric


def mse_fn(est, target, mask=None):
    eps = 1.e-10

    if mask is not None:
        metric = np.nansum(mask * np.power(est - target, 2)) / (np.nansum(mask) + eps)
    else:
        metric = np.nanmean(np.power(est - target, 2))

    return metric


def mad_fn(est, target, mask=None):
    eps = 1.e-10

    if mask is not None:
        grid = mask * np.abs(est - target)
        grid = grid.astype(np.float32)
        metric = np.nansum(grid) / (np.nansum(mask) + eps)
    else:
        metric = np.nanmean(np.abs(est - target))

    return metric


def iou_fn(est, target, mask=None):
    eps = 1.e-10

    if mask is not None:
        tp = (est < 0) & (target < 0) & (mask > 0)
        fp = (est < 0) & (target >= 0) & (mask > 0)
        fn = (est >= 0) & (target < 0) & (mask > 0)
    else:
        tp = (est < 0) & (target < 0)
        fp = (est < 0) & (target >= 0)
        fn = (est >= 0) & (target < 0)

    intersection = np.nansum(tp)
    union = np.nansum(tp) + np.nansum(fp) + np.nansum(fn)

    del tp, fp, fn
    metric = intersection / (union + eps)
    return metric


def acc_fn(est, target, mask=None):
    eps = 1.e-10

    if mask is not None:
        tp = (est < 0) & (target < 0) & (mask > 0)
        tn = (est >= 0) & (target >= 0) & (mask > 0)
    else:
        tp = (est < 0) & (target < 0)
        tn = (est >= 0) & (target >= 0)

    metric = (np.nansum(tp) + np.nansum(tn)) / (np.nansum(mask) + eps)

    del tp, tn
    return metric