import numpy as np
import torch


class MetricEvaluator:
    """
    Standard Change Detection Evaluation Metrics:
    - Precision (P)
    - Recall (R)
    - F1-Score (F1) / Dice Coefficient
    - Intersection over Union (IoU) on Change Class
    - Overall Accuracy (OA)
    - Cohen's Kappa Coefficient (Kappa)
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, preds, targets):
        """
        preds: Tensor or ndarray of shape (B, 1, H, W) or (B, H, W) (probabilities or logits)
        targets: Tensor or ndarray of shape (B, 1, H, W) or (B, H, W) (binary 0 or 1)
        """
        if isinstance(preds, torch.Tensor):
            if preds.shape != targets.shape:
                preds = preds.view_as(targets)
            # If logits, apply sigmoid
            if preds.min() < 0 or preds.max() > 1:
                preds = torch.sigmoid(preds)
            preds_bin = (preds > self.threshold).byte().cpu().numpy().flatten()
        else:
            preds_bin = (preds > self.threshold).astype(np.uint8).flatten()

        if isinstance(targets, torch.Tensor):
            targets_bin = (targets > 0.5).byte().cpu().numpy().flatten()
        else:
            targets_bin = (targets > 0.5).astype(np.uint8).flatten()

        self.tp += np.sum((preds_bin == 1) & (targets_bin == 1), dtype=np.int64)
        self.fp += np.sum((preds_bin == 1) & (targets_bin == 0), dtype=np.int64)
        self.fn += np.sum((preds_bin == 0) & (targets_bin == 1), dtype=np.int64)
        self.tn += np.sum((preds_bin == 0) & (targets_bin == 0), dtype=np.int64)

    def get_metrics(self):
        eps = 1e-7
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        total = tp + fp + fn + tn

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        oa = (tp + tn) / (total + eps)

        # Kappa Calculation
        pe = (((tp + fp) * (tp + fn)) + ((tn + fn) * (tn + fp))) / ((total + eps) ** 2)
        kappa = (oa - pe) / (1.0 - pe + eps)

        return {
            'Precision': float(precision) * 100.0,
            'Recall': float(recall) * 100.0,
            'F1': float(f1) * 100.0,
            'IoU': float(iou) * 100.0,
            'OA': float(oa) * 100.0,
            'Kappa': float(kappa) * 100.0,
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'TN': int(tn)
        }

    def summary_string(self):
        m = self.get_metrics()
        return (
            f"F1-Score: {m['F1']:.2f}% | IoU: {m['IoU']:.2f}% | "
            f"Precision: {m['Precision']:.2f}% | Recall: {m['Recall']:.2f}% | "
            f"OA: {m['OA']:.2f}% | Kappa: {m['Kappa']:.2f}%"
        )
