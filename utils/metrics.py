"""Evaluation metrics for medical image segmentation."""

import numpy as np

try:
    from medpy import metric
    MEDPY_AVAILABLE = True
except ImportError:
    MEDPY_AVAILABLE = False
    print("Warning: medpy not installed. HD95/HD/ASSD will not be available.")


def calculate_metrics_binary(pred, gt):
    """Compute basic binary segmentation metrics (dice, iou, hd95)."""
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    pred = np.squeeze(pred)
    gt = np.squeeze(gt)

    if pred.ndim < 2:
        pred = pred.reshape(1, -1)
    if gt.ndim < 2:
        gt = gt.reshape(1, -1)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union > 0 else 0.0

    if pred.sum() > 0 and gt.sum() > 0:
        if MEDPY_AVAILABLE:
            dice = metric.binary.dc(pred, gt)
            try:
                hd95 = metric.binary.hd95(pred, gt)
            except Exception:
                hd95 = 0.0
        else:
            dice = 2 * intersection / (pred.sum() + gt.sum())
            hd95 = 0.0
    elif pred.sum() == 0 and gt.sum() == 0:
        dice = 1.0
        hd95 = 0.0
    else:
        dice = 0.0
        hd95 = 0.0

    return dice, iou, hd95


def calculate_metrics_comprehensive(pred, gt):
    """
    Compute comprehensive segmentation metrics.

    Returns dict with:
        Dice, IoU, HD95, HD, ASSD, Precision, Recall,
        Specificity, Accuracy, F2, MCC, VS
    """
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    pred = np.squeeze(pred)
    gt = np.squeeze(gt)

    if pred.ndim < 2:
        pred = pred.reshape(1, -1)
    if gt.ndim < 2:
        gt = gt.reshape(1, -1)

    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    TP = np.logical_and(pred_flat == 1, gt_flat == 1).sum()
    TN = np.logical_and(pred_flat == 0, gt_flat == 0).sum()
    FP = np.logical_and(pred_flat == 1, gt_flat == 0).sum()
    FN = np.logical_and(pred_flat == 0, gt_flat == 1).sum()

    metrics = {}

    # Precision (PPV)
    metrics['Precision'] = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # Recall (Sensitivity, TPR)
    metrics['Recall'] = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # Specificity (TNR)
    metrics['Specificity'] = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    # Accuracy
    total = TP + TN + FP + FN
    metrics['Accuracy'] = (TP + TN) / total if total > 0 else 0.0

    # Dice (F1-Score)
    if pred.sum() > 0 and gt.sum() > 0:
        if MEDPY_AVAILABLE:
            metrics['Dice'] = metric.binary.dc(pred, gt)
        else:
            metrics['Dice'] = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
    elif pred.sum() == 0 and gt.sum() == 0:
        metrics['Dice'] = 1.0
    else:
        metrics['Dice'] = 0.0

    # F2-Score (weights recall higher)
    beta = 2
    if (metrics['Precision'] + metrics['Recall']) > 0:
        metrics['F2'] = (1 + beta**2) * metrics['Precision'] * metrics['Recall'] / (beta**2 * metrics['Precision'] + metrics['Recall'])
    else:
        metrics['F2'] = 0.0

    # IoU (Jaccard Index)
    intersection = TP
    union = TP + FP + FN
    metrics['IoU'] = intersection / union if union > 0 else 0.0

    # MCC (Matthews Correlation Coefficient)
    mcc_denom = np.sqrt(float(TP + FP) * float(TP + FN) * float(TN + FP) * float(TN + FN))
    metrics['MCC'] = (float(TP) * float(TN) - float(FP) * float(FN)) / mcc_denom if mcc_denom > 0 else 0.0

    # Volumetric Similarity
    pred_vol = float(pred.sum())
    gt_vol = float(gt.sum())
    metrics['VS'] = 1.0 - abs(pred_vol - gt_vol) / (pred_vol + gt_vol) if (pred_vol + gt_vol) > 0 else 1.0

    # Distance-based metrics
    if pred.sum() > 0 and gt.sum() > 0:
        if MEDPY_AVAILABLE:
            try:
                metrics['HD95'] = metric.binary.hd95(pred, gt)
            except Exception:
                metrics['HD95'] = 0.0

            try:
                metrics['HD'] = metric.binary.hd(pred, gt)
            except Exception:
                metrics['HD'] = 0.0

            try:
                metrics['ASSD'] = metric.binary.assd(pred, gt)
            except Exception:
                metrics['ASSD'] = 0.0
        else:
            metrics['HD95'] = 0.0
            metrics['HD'] = 0.0
            metrics['ASSD'] = 0.0
    elif pred.sum() == 0 and gt.sum() == 0:
        metrics['HD95'] = 0.0
        metrics['HD'] = 0.0
        metrics['ASSD'] = 0.0
    else:
        metrics['HD95'] = 100.0
        metrics['HD'] = 100.0
        metrics['ASSD'] = 100.0

    return metrics
