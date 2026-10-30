import numpy as np
from typing import Dict, List

def compute_attack_success_rate(clean_y_true: np.ndarray, clean_y_pred: np.ndarray, attacked_y_pred: np.ndarray) -> float:
    """
    Computes Attack Success Rate (ASR):
    Fraction of correctly detected phishing emails flipped to 'Legitimate' (0).
    """
    correct_phish_mask = (clean_y_true == 1) & (clean_y_pred == 1)
    if np.sum(correct_phish_mask) == 0:
        return 0.0
    
    flipped_mask = correct_phish_mask & (attacked_y_pred == 0)
    asr = np.sum(flipped_mask) / np.sum(correct_phish_mask)
    return float(asr)

def compute_recovery_ratio(acc_clean: float, acc_attacked: float, acc_after_patch: float) -> float:
    """
    Computes Recovery Ratio (RR) metric (Contribution C3):
    RR = (Acc_after_patch - Acc_attacked) / (Acc_clean - Acc_attacked)
    """
    denom = acc_clean - acc_attacked
    if abs(denom) < 1e-6:
        return 1.0 if acc_after_patch >= acc_clean else 0.0
    rr = (acc_after_patch - acc_attacked) / denom
    return float(np.clip(rr, 0.0, 1.0))
