"""Chronological walk-forward robustness evaluation for churn models.

The latest observed snapshot is reserved as the final test period and is
excluded from every fold. Each earlier snapshot after the first is used once
as validation, with all strictly earlier snapshots as training history. This
evaluates raw classifier probabilities; the existing May-fitted calibrator is
kept separate and is not refit across folds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from .evaluation import ranking_metrics
from .model_config import ModelConfig
from .modeling import (
    RFM_FEATURES,
    RFMRuleBaseline,
    make_lightgbm,
    make_logistic_pipeline,
    make_xy,
    prepare_lightgbm_features,
)

RESULT_COLUMNS = [
    "fold_id", "train_snapshot_dates", "train_start_date", "train_end_date",
    "validation_date", "train_rows", "validation_rows", "validation_churn_rate",
    "model", "pr_auc", "lift", "brier_score",
]
MODEL_NAMES = ("majority", "logistic_rfm", "rule_based_rfm", "lightgbm")


@dataclass(frozen=True)
class WalkForwardFold:
    """A train-before-validation fold; the reserved final date is never included."""

    fold_id: str
    train_snapshot_dates: tuple[str, ...]
    validation_date: str
    train: pd.DataFrame
    validation: pd.DataFrame


def build_walk_forward_folds(
    table: pd.DataFrame,
    date_column: str = "snapshot_date",
    min_train_snapshots: int = 1,
) -> tuple[WalkForwardFold, ...]:
    """Discover expanding-window folds and reserve the latest snapshot for test.

    For dates d0 < d1 < ... < dn (where dn is the final test snapshot), folds
    are d0→d1, (d0,d1)→d2, ... →d(n-1). No randomization is performed.
    """
    if date_column not in table:
        raise ValueError(f"Missing snapshot date column: {date_column}")
    if min_train_snapshots < 1:
        raise ValueError("min_train_snapshots must be at least one")
    dates = pd.to_datetime(table[date_column], errors="raise").dt.normalize()
    if dates.isna().any():
        raise ValueError("Snapshot dates cannot be missing")
    ordered_dates = tuple(pd.Timestamp(date) for date in sorted(dates.unique()))
    if len(ordered_dates) < 3:
        raise ValueError("At least three snapshot dates are required for walk-forward folds plus a final test")

    folds = []
    # The last observed snapshot is implicitly reserved: validation indices end
    # one date before it, and no returned training or validation data can include it.
    for validation_idx in range(1, len(ordered_dates) - 1):
        train_dates = ordered_dates[:validation_idx]
        if len(train_dates) < min_train_snapshots:
            continue
        validation_date = ordered_dates[validation_idx]
        train_mask = dates.isin(train_dates)
        validation_mask = dates.eq(validation_date)
        train = table.loc[train_mask].copy().reset_index(drop=True)
        validation = table.loc[validation_mask].copy().reset_index(drop=True)
        if train.empty or validation.empty:
            continue
        train_strings = tuple(d.strftime("%Y-%m-%d") for d in train_dates)
        validation_string = validation_date.strftime("%Y-%m-%d")
        folds.append(WalkForwardFold(
            fold_id=f"wf_{len(folds) + 1:02d}_{validation_string}",
            train_snapshot_dates=train_strings,
            validation_date=validation_string,
            train=train,
            validation=validation,
        ))
    return tuple(folds)


def _predict_fold(fold: WalkForwardFold, config: ModelConfig, target_column: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fit the existing models on fold training rows and predict validation."""
    train_x, train_y = make_xy(fold.train, target_column)
    validation_x, validation_y = make_xy(fold.validation, target_column, train_x.columns)
    if len(np.unique(train_y)) != 2:
        raise ValueError(
            f"{fold.fold_id} training target must contain both churn classes; "
            f"found {sorted(np.unique(train_y).tolist())}"
        )
    if train_x.empty or validation_x.empty:
        raise ValueError(f"{fold.fold_id} has no usable feature columns or rows")

    train_prior = float(train_y.mean())
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "majority": (np.full(len(validation_y), train_prior), validation_y.to_numpy(dtype=int)),
    }

    rfm_train_x = train_x.loc[:, RFM_FEATURES]
    rfm_validation_x = validation_x.loc[:, RFM_FEATURES]
    logistic = make_logistic_pipeline()
    logistic.fit(rfm_train_x, train_y)
    predictions["logistic_rfm"] = (
        logistic.predict_proba(rfm_validation_x)[:, 1], validation_y.to_numpy(dtype=int),
    )

    # Rule thresholds are fit only on this fold's training feature distribution.
    rules = RFMRuleBaseline().fit(fold.train)
    predictions["rule_based_rfm"] = (
        rules.predict_proba(fold.validation)[:, 1], validation_y.to_numpy(dtype=int),
    )

    lightgbm_train_x, lightgbm_validation_x = prepare_lightgbm_features(train_x, validation_x)
    lightgbm = make_lightgbm(config)
    lightgbm.fit(lightgbm_train_x, train_y, categorical_feature="auto")
    # Raw probabilities are intentional. The existing Platt calibrator was fit
    # on May and remains separate from this earlier-fold robustness exercise.
    predictions["lightgbm"] = (
        lightgbm.predict_proba(lightgbm_validation_x)[:, 1], validation_y.to_numpy(dtype=int),
    )
    return predictions


def evaluate_walk_forward(
    table: pd.DataFrame,
    model_config: ModelConfig | None = None,
    target_column: str = "churn_90d",
    date_column: str = "snapshot_date",
    min_train_snapshots: int = 1,
) -> dict:
    """Evaluate four churn models on all feasible pre-final chronological folds.

    June (or whichever date is latest in the supplied table) is excluded from
    all fold construction and therefore cannot influence fitting, validation,
    model selection, or the returned walk-forward metric tables.
    """
    config = model_config or ModelConfig()
    if target_column not in table:
        raise ValueError(f"Missing churn target column: {target_column}")
    folds = build_walk_forward_folds(table, date_column, min_train_snapshots)
    if not folds:
        raise ValueError("No feasible walk-forward folds under min_train_snapshots")
    final_test_date = pd.to_datetime(table[date_column]).max().normalize().strftime("%Y-%m-%d")
    records = []
    for fold in folds:
        if final_test_date in fold.train_snapshot_dates or final_test_date == fold.validation_date:
            raise RuntimeError("Final test date leaked into a walk-forward fold")
        model_predictions = _predict_fold(fold, config, target_column)
        y = model_predictions["majority"][1]
        for model_name in MODEL_NAMES:
            probability, model_y = model_predictions[model_name]
            metrics = ranking_metrics(model_y, probability)
            records.append({
                "fold_id": fold.fold_id,
                "train_snapshot_dates": fold.train_snapshot_dates,
                "train_start_date": fold.train_snapshot_dates[0],
                "train_end_date": fold.train_snapshot_dates[-1],
                "validation_date": fold.validation_date,
                "train_rows": int(len(fold.train)),
                "validation_rows": int(len(fold.validation)),
                "validation_churn_rate": float(model_y.mean()),
                "model": model_name,
                "pr_auc": metrics["pr_auc"],
                "lift": metrics["lift_at_decile_1"],
                "brier_score": float(brier_score_loss(model_y, probability)),
            })
    result_table = pd.DataFrame.from_records(records, columns=RESULT_COLUMNS)
    aggregate = result_table.groupby("model", sort=False).agg(
        folds_evaluated=("fold_id", "nunique"),
        mean_pr_auc=("pr_auc", "mean"),
        std_pr_auc=("pr_auc", "std"),
        mean_lift=("lift", "mean"),
        std_lift=("lift", "std"),
        mean_brier_score=("brier_score", "mean"),
        std_brier_score=("brier_score", "std"),
    ).reset_index()
    return {
        "results": result_table,
        "aggregate": aggregate,
        "folds": folds,
        "available_snapshot_dates": tuple(
            d.strftime("%Y-%m-%d") for d in sorted(pd.to_datetime(table[date_column]).dt.normalize().unique())
        ),
        "final_test_date_reserved": final_test_date,
        "calibration": "raw probabilities only; existing May calibrator is not refit",
    }
