"""Temporal fit and evaluation for the BG/NBD + Gamma-Gamma benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .classical_clv import (
    ClassicalCLVModel,
    build_customer_summary,
    future_gross_purchase_value,
    gamma_gamma_eligibility,
)
from .clv_config import ClassicalCLVConfig
from .temporal import TemporalDataSplit
from .value_model import regression_metrics, top_decile_value_capture


def _distribution(values) -> dict[str, float | int]:
    series = pd.Series(values, dtype=float)
    if series.empty:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "zero_pct": float("nan")}
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "zero_pct": float(series.eq(0).mean() * 100),
    }


def _rank_correlation(actual, predicted) -> float:
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if len(a) < 2 or np.unique(a).size < 2 or np.unique(p).size < 2:
        return float("nan")
    return float(spearmanr(a, p).statistic)


def _score_partition(
    transactions: pd.DataFrame,
    partition: pd.DataFrame,
    prediction_date,
    model: ClassicalCLVModel,
    supervised_predictions: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict]:
    snapshot_summary = build_customer_summary(transactions, prediction_date)
    classical_predictions = model.predict(snapshot_summary)
    scored = partition[["CustomerID", "snapshot_date", "future_net_spend_90d"]].merge(
        classical_predictions[["CustomerID", "expected_purchases_90d", "expected_average_order_value", "predicted_90d_gross_purchase_value"]],
        on="CustomerID", how="inner", validate="one_to_one",
    )
    actual_gross = future_gross_purchase_value(
        transactions, partition["CustomerID"], prediction_date, model.config.horizon_days
    )
    scored = scored.merge(actual_gross.reset_index(), on="CustomerID", how="left", validate="one_to_one")
    if supervised_predictions is not None:
        supervised = supervised_predictions[["CustomerID", "lightgbm_value"]]
        scored = scored.merge(supervised, on="CustomerID", how="left", validate="one_to_one")
    actual = scored["future_gross_purchase_value_90d"].to_numpy()
    pred = scored["predicted_90d_gross_purchase_value"].to_numpy()
    result = {
        "customers_in_partition": int(len(partition)),
        "customers_scored": int(len(scored)),
        "coverage_pct": float(len(scored) / len(partition) * 100) if len(partition) else float("nan"),
        "gamma_gamma_excluded_at_scoring": int(len(partition) - len(scored)),
        "aligned_gross_purchase_metrics": regression_metrics(actual, pred),
        "top_decile_actual_gross_value_capture": top_decile_value_capture(actual, pred, 0.10),
    }
    if supervised_predictions is not None and "lightgbm_value" in scored:
        result["supervised_lightgbm_vs_gross_actual"] = {
            "spearman": _rank_correlation(actual, scored["lightgbm_value"]),
            "top_decile_actual_gross_value_capture": top_decile_value_capture(
                actual, scored["lightgbm_value"], 0.10
            ),
        }
        result["prediction_rank_agreement_with_supervised_lightgbm"] = _rank_correlation(
            pred, scored["lightgbm_value"]
        )
    # Capture the number removed by Gamma-Gamma's positive repeat-value constraint.
    result["prediction_exclusions"] = classical_predictions.attrs.get("prediction_coverage", {})
    return scored, result


def evaluate_classical_clv(
    transactions: pd.DataFrame,
    split: TemporalDataSplit,
    supervised_predictions: dict[str, pd.DataFrame] | None = None,
    config: ClassicalCLVConfig | None = None,
) -> dict:
    """Fit through the validation cutoff and evaluate aligned 90-day spend rankings.

    Model parameters use customer order histories strictly before the first
    validation snapshot (May 1). June transactions and labels are not used for
    fitting. At each prediction date, customer summaries are rebuilt from rows
    strictly before that date; May purchase history may therefore inform June
    scores while fitted population parameters remain frozen.
    """
    config = config or ClassicalCLVConfig()
    if not {"CustomerID", "snapshot_date", "future_net_spend_90d"}.issubset(split.train.columns):
        raise ValueError("Temporal partitions must include identifiers and future_net_spend_90d")
    fit_end = pd.to_datetime(split.validation["snapshot_date"]).min()
    fit_summary = build_customer_summary(transactions, fit_end)
    model = ClassicalCLVModel(config).fit(fit_summary, fit_end)
    gamma_eligible, gamma_counts = gamma_gamma_eligibility(fit_summary)

    outputs = {}
    partition_metrics = {}
    for partition_name, partition in (("validation", split.validation), ("test", split.test)):
        prediction_date = pd.to_datetime(partition["snapshot_date"]).min()
        supervised = supervised_predictions.get(partition_name) if supervised_predictions else None
        scored, metrics = _score_partition(transactions, partition, prediction_date, model, supervised)
        outputs[partition_name] = scored
        partition_metrics[partition_name] = metrics

    return {
        "fit_period": {
            "strictly_before": str(fit_end),
            "first_order_date": str(transactions.loc[
                (transactions["InvoiceDate"] < fit_end) & transactions["is_purchase"], "InvoiceDate"
            ].min()),
            "fit_customers": len(fit_summary),
        },
        "bgnbd_fit_summary": {
            "frequency": _distribution(fit_summary["frequency"]),
            "recency_days": _distribution(fit_summary["recency"]),
            "T_days": _distribution(fit_summary["T"]),
        },
        "gamma_gamma_fit_summary": {
            **gamma_counts,
            "monetary_value_repeat_invoice_mean": _distribution(fit_summary.loc[gamma_eligible, "monetary_value"]),
            "fitted_parameters": {name: float(value) for name, value in model.ggf.params_.items()},
        },
        "metrics": partition_metrics,
        "predictions": outputs,
        "model": model,
        "horizon_days": config.horizon_days,
    }
