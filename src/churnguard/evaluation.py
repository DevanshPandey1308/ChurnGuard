"""Ranking metrics for imbalanced churn classification."""

from math import ceil

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def ranking_metrics(y_true, probabilities) -> dict[str, float | int]:
    """Return PR-AUC (average precision) and top-decile lift, plus class counts."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    prevalence = float(y.mean()) if len(y) else float("nan")
    k = max(1, ceil(len(y) * 0.1)) if len(y) else 0
    if k:
        # Share cutoff-score ties proportionally instead of depending on row order.
        cutoff = np.partition(p, len(p) - k)[len(p) - k]
        above = p > cutoff
        tied = p == cutoff
        slots_for_ties = k - int(above.sum())
        expected_positives = float(y[above].sum()) + slots_for_ties * float(y[tied].mean())
        top_rate = expected_positives / k
    else:
        top_rate = float("nan")
    return {
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "lift_at_decile_1": top_rate / prevalence if prevalence else float("nan"),
        "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()),
        "rows": int(len(y)),
    }
