"""Leakage-safe supervised modeling for predicted 90-day customer value."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .modeling import IDENTIFIER_COLUMNS, RFM_FEATURES, TARGET_DERIVED_COLUMNS, feature_columns, prepare_lightgbm_features
from .value_config import ValueModelConfig


def signed_log1p(values):
    """Signed log transform: sign(y) * log1p(abs(y)); valid for negative values."""
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def signed_expm1(values):
    """Inverse of signed_log1p: sign(z) * expm1(abs(z))."""
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.expm1(np.abs(values))


def inverse_transformed_predictions(model, features, training_transformed_target):
    """Invert model scores after clipping extrapolation to train-target support.

    This is chiefly important for linear regression: unconstrained extrapolation
    on signed-log scale can overflow dramatically after expm1. Bounds come only
    from training labels; the boosted-tree scores are naturally bounded by leaf
    averages but use the same conversion for a consistent output contract.
    """
    transformed = np.asarray(model.predict(features), dtype=float)
    train_y = np.asarray(training_transformed_target, dtype=float)
    bounded = np.clip(transformed, train_y.min(), train_y.max())
    return signed_expm1(bounded)


def value_feature_columns(table: pd.DataFrame, target_column: str = "future_net_spend_90d") -> list[str]:
    """Select historical predictors only; discard every identifier and outcome."""
    columns = feature_columns(table, target_column)
    forbidden = IDENTIFIER_COLUMNS | TARGET_DERIVED_COLUMNS | {target_column}
    invalid = forbidden.intersection(columns)
    if invalid:
        raise ValueError(f"Forbidden value-model features: {sorted(invalid)}")
    return columns


def training_baseline_predictions(training_target, n_validation: int, n_test: int):
    """Return training-only mean and median predictions for both evaluation sets."""
    train = np.asarray(training_target, dtype=float).reshape(-1)
    if len(train) == 0 or not np.isfinite(train).all():
        raise ValueError("Training target must contain finite observations")
    return {
        "mean_baseline": (np.full(n_validation, train.mean()), np.full(n_test, train.mean())),
        "median_baseline": (np.full(n_validation, np.median(train)), np.full(n_test, np.median(train))),
    }


def make_log_linear_rfm(config: ValueModelConfig) -> Pipeline:
    """Scaled, median-imputed linear regression over the core numeric RFM block."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("regressor", LinearRegression(**dict(config.linear_params))),
    ])


def make_lightgbm_regressor(config: ValueModelConfig):
    """Construct a seeded LightGBM regressor for signed-log transformed targets."""
    from lightgbm import LGBMRegressor

    params = dict(config.lightgbm_params)
    params.update({"objective": "regression", "random_state": config.random_seed, "n_jobs": -1})
    return LGBMRegressor(**params)


def regression_metrics(actual, predicted) -> dict[str, float | int]:
    """MAE, RMSE and Spearman rank correlation for value predictions."""
    y = np.asarray(actual, dtype=float).reshape(-1)
    p = np.asarray(predicted, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError("Actual and predicted values must have equal length")
    correlation = (
        spearmanr(y, p).statistic
        if len(y) > 1 and np.unique(y).size > 1 and np.unique(p).size > 1
        else float("nan")
    )
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "spearman": float(correlation) if np.isfinite(correlation) else float("nan"),
        "rows": len(y),
    }


def top_decile_value_capture(actual, predicted, fraction: float = 0.10) -> dict[str, float | int]:
    """Measure actual target-value share among the highest predicted-value rows."""
    y = np.asarray(actual, dtype=float).reshape(-1)
    p = np.asarray(predicted, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError("Actual and predicted values must have equal length")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if len(y) == 0:
        return {"selected_count": 0, "actual_value_captured": 0.0, "total_actual_value": 0.0,
                "capture_pct": float("nan"), "random_selection_reference_pct": fraction * 100}
    count = max(1, math.ceil(len(y) * fraction))
    cutoff = np.partition(p, len(p) - count)[len(p) - count]
    above = p > cutoff
    tied = p == cutoff
    remaining_slots = count - int(above.sum())
    # Expected capture under uniform tie-breaking; constant-score baselines
    # therefore have the random-selection reference rather than row-order bias.
    captured = float(y[above].sum() + remaining_slots * y[tied].mean())
    total = float(y.sum())
    return {
        "selected_count": int(count),
        "actual_value_captured": captured,
        "total_actual_value": total,
        "capture_pct": captured / total * 100 if total != 0 else float("nan"),
        "random_selection_reference_pct": fraction * 100,
    }


def target_distribution(values) -> dict[str, float | int]:
    """Summarize target values for reporting, not for model fitting or selection."""
    y = np.asarray(values, dtype=float).reshape(-1)
    return {
        "count": int(len(y)),
        "mean": float(np.mean(y)),
        "median": float(np.median(y)),
        "std": float(np.std(y, ddof=1)) if len(y) > 1 else float("nan"),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "zero_pct": float(np.mean(y == 0) * 100),
        "negative_pct": float(np.mean(y < 0) * 100),
    }
