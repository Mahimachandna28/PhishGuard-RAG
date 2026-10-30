import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from typing import Dict

def compute_fpr_at_recall(y_true: np.ndarray, y_probs: np.ndarray, target_recall: float = 0.95) -> float:
    """
    Computes False Positive Rate (FPR) at fixed Recall (e.g. 95% Recall).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    # Find threshold achieving target recall
    idx = np.argmin(np.abs(recalls - target_recall))
    best_threshold = thresholds[min(idx, len(thresholds)-1)]

    y_pred_at_thresh = (y_probs >= best_threshold).astype(int)
    fp = np.sum((y_true == 0) & (y_pred_at_thresh == 1))
    tn = np.sum((y_true == 0) & (y_pred_at_thresh == 0))
    fpr = fp / max(fp + tn, 1)
    return float(fpr)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None) -> Dict[str, float]:
    """
    Computes standard evaluation protocol metrics (Section 9):
    Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, FPR@95%Recall.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = 0.0
    pr_auc = 0.0
    fpr_95 = 0.0

    if y_probs is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_probs)
            precisions, recalls, _ = precision_recall_curve(y_true, y_probs)
            pr_auc = auc(recalls, precisions)
            fpr_95 = compute_fpr_at_recall(y_true, y_probs, target_recall=0.95)
        except Exception:
            pass

    return {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "ROC-AUC": float(roc_auc),
        "PR-AUC": float(pr_auc),
        "FPR@95Recall": float(fpr_95)
    }
