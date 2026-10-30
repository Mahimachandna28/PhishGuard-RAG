import numpy as np
from scipy import stats
from typing import List, Dict

def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> dict:
    """
    Performs McNemar's Test for statistical significance comparison between Model A and Model B.
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Contingency matrix
    n_01 = np.sum(~correct_a & correct_b) # Model A wrong, Model B right
    n_10 = np.sum(correct_a & ~correct_b) # Model A right, Model B wrong

    statistic = ((abs(n_01 - n_10) - 1)**2) / max(n_01 + n_10, 1)
    p_value = stats.chi2.sf(statistic, df=1)

    return {
        "n_01": int(n_01),
        "n_10": int(n_10),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant_p05": bool(p_value < 0.05)
    }

def confidence_interval_95(scores: List[float]) -> dict:
    mean = float(np.mean(scores))
    if len(scores) < 2:
        return {"mean": mean, "std": 0.0, "ci_lower": mean, "ci_upper": mean}
    
    std = float(np.std(scores, ddof=1))
    sem = std / np.sqrt(len(scores))
    ci = stats.t.interval(0.95, df=len(scores)-1, loc=mean, scale=sem)
    return {
        "mean": mean,
        "std": std,
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1])
    }
