"""Sigmoid calibration and compact reliability tables.

The base classifier is fitted on March/April snapshots; Platt's sigmoid is fitted
only on raw May validation scores and labels. This uses May labels to fit the
calibrator, so May calibrated metrics and reliability are in-sample diagnostics.
June labels are reserved for final evaluation. With one validation month and only
four snapshots, calibration quality is uncertain; the existing overlapping
90-day train-label windows also mean this is not a fully purged temporal study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def calibration_features(probabilities) -> pd.DataFrame:
    """Convert raw probabilities to the single Platt input: clipped log odds."""
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Raw probabilities must be finite values in [0, 1]")
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return pd.DataFrame({"raw_logit_probability": np.log(clipped / (1 - clipped))})


class SigmoidProbabilityCalibrator:
    """Fit a deterministic Platt sigmoid to held-out base-model probabilities."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self._model: LogisticRegression | None = None
        self.n_calibration_rows_: int | None = None

    def fit(self, raw_probabilities, calibration_labels) -> "SigmoidProbabilityCalibrator":
        p = np.asarray(raw_probabilities, dtype=float).reshape(-1)
        y = np.asarray(calibration_labels, dtype=int).reshape(-1)
        if len(p) != len(y):
            raise ValueError("Probabilities and calibration labels must have equal length")
        if len(np.unique(y)) != 2:
            raise ValueError("Sigmoid calibration requires both target classes")
        features = calibration_features(p)
        self._model = LogisticRegression(random_state=self.random_seed, solver="lbfgs")
        self._model.fit(features, y)
        self.n_calibration_rows_ = len(y)
        return self

    def predict(self, raw_probabilities) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Fit calibrator before predicting")
        result = self._model.predict_proba(calibration_features(raw_probabilities))[:, 1]
        return np.clip(result, 0.0, 1.0)


def probability_metrics(y_true, probabilities) -> dict[str, float | int]:
    """Return consistent ranking metrics plus Brier score and class counts."""
    from .evaluation import ranking_metrics

    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError("Labels and probabilities must have equal length")
    result = ranking_metrics(y, p)
    result["brier_score"] = float(brier_score_loss(y, p))
    return result


def reliability_table(y_true, probabilities, n_bins: int = 5) -> pd.DataFrame:
    """Build an equal-frequency reliability table, reducing bins when ties require."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError("Labels and probabilities must have equal length")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if len(y) == 0:
        return pd.DataFrame(columns=["probability_bin", "mean_predicted_probability", "observed_churn_rate", "n_observations"])
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    frame = pd.DataFrame({"probability": p, "observed": y})
    try:
        bins = pd.qcut(frame["probability"], q=min(n_bins, len(frame)), duplicates="drop")
        if bins.isna().all():
            frame["probability_bin"] = "all"
        else:
            frame["probability_bin"] = bins
    except ValueError:
        frame["probability_bin"] = pd.Series(["all"] * len(frame), index=frame.index, dtype="string")
    grouped = frame.groupby("probability_bin", observed=True, sort=True).agg(
        mean_predicted_probability=("probability", "mean"),
        observed_churn_rate=("observed", "mean"),
        n_observations=("observed", "size"),
    ).reset_index()
    grouped["probability_bin"] = grouped["probability_bin"].astype(str)
    return grouped[["probability_bin", "mean_predicted_probability", "observed_churn_rate", "n_observations"]]
