"""Prediction-only customer ranking and hypothetical retention scenarios.

Targeting scores never consume observed future labels. Held-out outcomes are
joined in a separate evaluation step after customer scores and ranks exist.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .targeting_config import TargetingConfig

ID_COLUMNS = ["CustomerID", "snapshot_date"]
PREDICTION_COLUMNS = ID_COLUMNS + [
    "calibrated_churn_probability",
    "predicted_90d_value",
]
FUTURE_COLUMNS = {"churn_90d", "repurchased_90d", "future_net_spend_90d", "future_gross_purchase_value_90d"}
RANK_SCORE_COLUMNS = {
    "calibrated_churn_probability": "churn_rank",
    "predicted_90d_value": "value_rank",
    "risk_weighted_value": "risk_value_rank",
}


def score_customers(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build a prediction-only table and deterministic within-snapshot ranks.

    Ties are ordered by ascending CustomerID; duplicate CustomerID/snapshot
    pairs are rejected because they do not represent unique observations.
    Extra columns are rejected to make accidental future-label input explicit.
    """
    missing = set(PREDICTION_COLUMNS) - set(predictions.columns)
    forbidden = FUTURE_COLUMNS.intersection(predictions.columns)
    extras = set(predictions.columns) - set(PREDICTION_COLUMNS)
    if missing:
        raise ValueError(f"Missing targeting prediction columns: {sorted(missing)}")
    if forbidden:
        raise ValueError(f"Future outcomes are not valid targeting inputs: {sorted(forbidden)}")
    if extras:
        raise ValueError(f"Unexpected targeting inputs: {sorted(extras)}")

    scored = predictions.loc[:, PREDICTION_COLUMNS].copy().reset_index(drop=True)
    scored["snapshot_date"] = pd.to_datetime(scored["snapshot_date"])
    if scored[ID_COLUMNS].isna().any().any() or scored.duplicated(ID_COLUMNS).any():
        raise ValueError("Targeting inputs require unique, non-null CustomerID/snapshot_date pairs")
    probability = scored["calibrated_churn_probability"].to_numpy(dtype=float)
    value = scored["predicted_90d_value"].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("calibrated_churn_probability must be finite and in [0, 1]")
    if not np.isfinite(value).all():
        raise ValueError("predicted_90d_value must be finite")
    scored["risk_weighted_value"] = probability * value

    for score_column, rank_column in RANK_SCORE_COLUMNS.items():
        ranked = scored.sort_values(
            ["snapshot_date", score_column, "CustomerID"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        ranks = ranked.groupby("snapshot_date", sort=False).cumcount() + 1
        scored[rank_column] = ranks.reindex(scored.index).astype("int64")
    return scored


def add_scenario_values(
    scored: pd.DataFrame,
    retention_offer_cost: float,
    assumed_acceptance_rate: float,
    retained_value_fraction: float,
) -> pd.DataFrame:
    """Append per-customer scenario estimates, not measured intervention effects.

    Expected recovered value = churn probability × acceptance assumption ×
    retained-value assumption × max(predicted 90-day value, 0). Predicted
    negative net spend is clamped to zero because it is not treated as
    recoverable revenue in this hypothetical scenario. Offer cost is charged
    for every targeted customer, whether or not the offer is accepted.
    """
    if retention_offer_cost < 0:
        raise ValueError("retention_offer_cost cannot be negative")
    if not 0 <= assumed_acceptance_rate <= 1 or not 0 <= retained_value_fraction <= 1:
        raise ValueError("assumed_acceptance_rate and retained_value_fraction must be in [0, 1]")
    result = scored.copy()
    result["expected_recovered_value_under_scenario"] = (
        result["calibrated_churn_probability"]
        * assumed_acceptance_rate
        * retained_value_fraction
        * result["predicted_90d_value"].clip(lower=0)
    )
    result["offer_cost_under_scenario"] = float(retention_offer_cost)
    result["expected_net_value_under_scenario"] = (
        result["expected_recovered_value_under_scenario"] - result["offer_cost_under_scenario"]
    )
    return result


def _selected(frame: pd.DataFrame, fraction: float, rank_column: str) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    selections = []
    for _, group in frame.groupby("snapshot_date", sort=True):
        count = max(1, math.ceil(len(group) * fraction))
        selections.append(group.nsmallest(count, rank_column, keep="first"))
    if not selections:
        return frame.iloc[0:0].copy()
    return pd.concat(selections, ignore_index=False)


def _scenario_totals(selected: pd.DataFrame, config: TargetingConfig,
                     acceptance_rate: float | None = None,
                     retained_value_fraction: float | None = None) -> dict[str, float | int]:
    acceptance = config.assumed_acceptance_rate if acceptance_rate is None else acceptance_rate
    retained = config.retained_value_fraction if retained_value_fraction is None else retained_value_fraction
    recovered = (
        selected["calibrated_churn_probability"] * acceptance * retained
        * selected["predicted_90d_value"].clip(lower=0)
    )
    offer_cost = len(selected) * config.retention_offer_cost
    return {
        "number_targeted": int(len(selected)),
        "mean_churn_probability": float(selected["calibrated_churn_probability"].mean()) if len(selected) else float("nan"),
        "mean_predicted_90d_value": float(selected["predicted_90d_value"].mean()) if len(selected) else float("nan"),
        "total_predicted_value": float(selected["predicted_90d_value"].sum()),
        "total_modeled_expected_recovered_value": float(recovered.sum()),
        "total_modeled_offer_cost": float(offer_cost),
        "modeled_expected_net_value_under_scenario": float(recovered.sum() - offer_cost),
        "assumed_acceptance_rate": float(acceptance),
        "assumed_retained_value_fraction": float(retained),
    }


def expected_value_threshold_table(scored: pd.DataFrame, config: TargetingConfig) -> pd.DataFrame:
    """Return validation-style targeting trade-offs using prediction scores only."""
    if scored.empty:
        raise ValueError("Cannot build a threshold table for an empty prediction set")
    rank_column = RANK_SCORE_COLUMNS[config.primary_ranking_method]
    rows = []
    for fraction in config.targeting_thresholds:
        chosen = _selected(scored, fraction, rank_column)
        totals = _scenario_totals(chosen, config)
        totals["target_fraction"] = float(fraction)
        totals["percentage_targeted"] = float(len(chosen) / len(scored) * 100)
        rows.append(totals)
    return pd.DataFrame(rows).loc[:, [
        "target_fraction", "percentage_targeted", "number_targeted", "mean_churn_probability",
        "mean_predicted_90d_value", "total_predicted_value",
        "total_modeled_expected_recovered_value", "total_modeled_offer_cost",
        "modeled_expected_net_value_under_scenario", "assumed_acceptance_rate",
        "assumed_retained_value_fraction",
    ]]


def scenario_sensitivity(scored: pd.DataFrame, config: TargetingConfig) -> pd.DataFrame:
    """Evaluate a compact acceptance × retained-value grid on primary top fraction."""
    rank_column = RANK_SCORE_COLUMNS[config.primary_ranking_method]
    chosen = _selected(scored, config.top_fraction, rank_column)
    rows = []
    for acceptance in config.sensitivity_acceptance_rates:
        for retained in config.sensitivity_retained_value_fractions:
            row = _scenario_totals(chosen, config, acceptance, retained)
            row["target_fraction"] = config.top_fraction
            rows.append(row)
    return pd.DataFrame(rows).loc[:, [
        "target_fraction", "assumed_acceptance_rate", "assumed_retained_value_fraction",
        "number_targeted", "total_modeled_expected_recovered_value", "total_modeled_offer_cost",
        "modeled_expected_net_value_under_scenario",
    ]]


def top_decile_actual_value_capture(
    scored: pd.DataFrame,
    actual_outcomes: pd.DataFrame,
    rank_column: str,
    fraction: float = 0.10,
) -> dict[str, float | int]:
    """Evaluate selected predictions against future net spend after scoring."""
    if rank_column not in scored or rank_column not in RANK_SCORE_COLUMNS.values():
        raise ValueError(f"Unknown targeting rank column: {rank_column}")
    actual_col = "future_net_spend_90d"
    if set(actual_outcomes.columns) != set(ID_COLUMNS + [actual_col]):
        raise ValueError("Evaluation outcomes must contain only CustomerID, snapshot_date, and future_net_spend_90d")
    actual = actual_outcomes.copy()
    actual["snapshot_date"] = pd.to_datetime(actual["snapshot_date"])
    if actual.duplicated(ID_COLUMNS).any():
        raise ValueError("Evaluation outcomes must have unique CustomerID/snapshot_date pairs")
    if set(map(tuple, scored[ID_COLUMNS].to_numpy())) != set(map(tuple, actual[ID_COLUMNS].to_numpy())):
        raise ValueError("Prediction and held-out outcome customer snapshots must match exactly")
    evaluated = scored.merge(actual, on=ID_COLUMNS, how="inner", validate="one_to_one")
    chosen = _selected(evaluated, fraction, rank_column)
    total = float(evaluated[actual_col].sum())
    captured = float(chosen[actual_col].sum())
    return {
        "number_targeted": int(len(chosen)),
        "percentage_targeted": float(len(chosen) / len(evaluated) * 100) if len(evaluated) else float("nan"),
        "actual_future_net_spend_captured": captured,
        "total_actual_future_net_spend": total,
        "percentage_of_total_actual_future_net_spend": captured / total * 100 if total != 0 else float("nan"),
        "average_actual_future_net_spend_per_targeted_customer": float(chosen[actual_col].mean()) if len(chosen) else float("nan"),
        "overall_average_actual_future_net_spend": float(evaluated[actual_col].mean()) if len(evaluated) else float("nan"),
        "overall_customer_count": int(len(evaluated)),
    }


def _ranking_relationships(scored: pd.DataFrame, fraction: float) -> dict[str, dict[str, float | int]]:
    methods = {
        "risk_weighted_value": "risk_value_rank",
        "calibrated_churn_probability": "churn_rank",
        "predicted_90d_value": "value_rank",
    }
    relationships = {}
    for method, rank in methods.items():
        if method == "risk_weighted_value":
            continue
        left = set(_selected(scored, fraction, "risk_value_rank")[ID_COLUMNS].itertuples(index=False, name=None))
        right = set(_selected(scored, fraction, rank)[ID_COLUMNS].itertuples(index=False, name=None))
        union = left | right
        correlations = []
        for _, group in scored.groupby("snapshot_date", sort=True):
            if len(group) > 1:
                correlations.append(float(spearmanr(group["risk_value_rank"], group[rank]).statistic))
        relationships[method] = {
            "top_fraction_overlap_count": int(len(left & right)),
            "top_fraction_jaccard_pct": float(len(left & right) / len(union) * 100) if union else float("nan"),
            "mean_spearman_rank_correlation": float(np.nanmean(correlations)) if correlations else float("nan"),
        }
    return relationships


def evaluate_targeting_partitions(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    validation_outcomes: pd.DataFrame,
    test_outcomes: pd.DataFrame,
    config: TargetingConfig | None = None,
) -> dict:
    """Score validation and test independently; only validation gets scenario curves.

    Test labels are joined only after prediction-only scoring. No test-derived
    quantity is used to select a threshold or scenario assumption.
    """
    config = config or TargetingConfig()
    validation_scored = score_customers(validation_predictions)
    test_scored = score_customers(test_predictions)
    val_dates = set(validation_scored["snapshot_date"])
    test_dates = set(test_scored["snapshot_date"])
    if not val_dates or not test_dates or val_dates & test_dates:
        raise ValueError("Validation and test predictions must have distinct non-empty snapshot dates")
    for partition, outcomes in (("validation", validation_outcomes), ("test", test_outcomes)):
        outcome_dates = set(pd.to_datetime(outcomes["snapshot_date"])) if "snapshot_date" in outcomes else set()
        expected_dates = val_dates if partition == "validation" else test_dates
        if outcome_dates != expected_dates:
            raise ValueError(f"{partition} outcomes do not match their prediction snapshot dates")

    def evaluate_comparisons(scored, outcomes):
        return {
            method: top_decile_actual_value_capture(scored, outcomes, rank, config.top_fraction)
            for method, rank in (
                ("calibrated_churn_probability", "churn_rank"),
                ("predicted_90d_value", "value_rank"),
                ("risk_weighted_value", "risk_value_rank"),
            )
        }

    val_assumed = add_scenario_values(
        validation_scored, config.retention_offer_cost,
        config.assumed_acceptance_rate, config.retained_value_fraction,
    )
    test_assumed = add_scenario_values(
        test_scored, config.retention_offer_cost,
        config.assumed_acceptance_rate, config.retained_value_fraction,
    )
    return {
        "configuration": {
            "primary_ranking_method": config.primary_ranking_method,
            "targeting_thresholds": list(config.targeting_thresholds),
            "top_fraction": config.top_fraction,
            "retention_offer_cost": config.retention_offer_cost,
            "assumed_acceptance_rate": config.assumed_acceptance_rate,
            "retained_value_fraction": config.retained_value_fraction,
            "offer_cost_assumption": "cost is incurred for every targeted customer, regardless of acceptance",
            "scenario_status": "hypothetical assumptions, not learned from intervention outcomes",
        },
        "validation": {
            "customer_count": int(len(validation_scored)),
            "predictions": val_assumed,
            "top_decile_actual_value_capture": evaluate_comparisons(validation_scored, validation_outcomes),
            "expected_value_threshold_table": expected_value_threshold_table(validation_scored, config),
            "scenario_sensitivity": scenario_sensitivity(validation_scored, config),
            "ranking_relationships": _ranking_relationships(validation_scored, config.top_fraction),
        },
        "test": {
            "customer_count": int(len(test_scored)),
            "predictions": test_assumed,
            "top_decile_actual_value_capture": evaluate_comparisons(test_scored, test_outcomes),
            "ranking_relationships": _ranking_relationships(test_scored, config.top_fraction),
        },
    }
