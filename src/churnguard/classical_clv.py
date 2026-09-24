"""BG/NBD + Gamma-Gamma benchmark using invoice-level Online Retail orders.

Order definition: one unique InvoiceNo with at least one positive-quantity,
non-C-prefixed line is one purchase. C-prefixed invoices are cancellations and
never count as purchases. An eligible order's monetary value is the signed sum
of its non-cancellation line values; cancellation invoices are not matched back
to original orders. The benchmark fits only strictly pre-cutoff transactions.

For each customer, frequency is repeat purchase invoices (invoice count minus 1),
recency is elapsed days from first to last purchase invoice, and T is elapsed days
from first invoice to the snapshot cutoff. Gamma-Gamma monetary_value is the mean
net value of repeat purchase invoices, excluding the first invoice per the
lifetimes convention. Its fit population requires repeat orders and every repeat
invoice value strictly positive. This is required by Gamma-Gamma's positive
monetary-value assumption; excluded customers are reported rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .clv_config import ClassicalCLVConfig


def build_order_transactions(transactions: pd.DataFrame, snapshot_date) -> pd.DataFrame:
    """Aggregate cleaned line items to one order row per customer/invoice before t."""
    required = {"CustomerID", "InvoiceNo", "InvoiceDate", "line_value", "is_cancellation", "is_purchase"}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"Cleaned transactions missing columns: {sorted(missing)}")
    t = pd.Timestamp(snapshot_date)
    history = transactions.loc[(transactions["InvoiceDate"] < t) & ~transactions["is_cancellation"]].copy()
    if history.empty:
        return pd.DataFrame(columns=["CustomerID", "InvoiceNo", "order_date", "order_value", "is_purchase_order"])
    orders = history.groupby(["CustomerID", "InvoiceNo"], sort=False).agg(
        order_date=("InvoiceDate", "min"),
        order_value=("line_value", "sum"),
        is_purchase_order=("is_purchase", "any"),
    ).reset_index()
    orders = orders.loc[orders["is_purchase_order"]].copy()
    return orders.sort_values(["CustomerID", "order_date", "InvoiceNo"], kind="stable").reset_index(drop=True)


def build_customer_summary(transactions: pd.DataFrame, snapshot_date) -> pd.DataFrame:
    """Build BG/NBD and Gamma-Gamma inputs for histories strictly before snapshot."""
    cutoff = pd.Timestamp(snapshot_date)
    orders = build_order_transactions(transactions, cutoff)
    columns = ["CustomerID", "frequency", "recency", "T", "monetary_value", "repeat_orders_positive", "order_count"]
    rows = []
    for customer_id, customer_orders in orders.groupby("CustomerID", sort=False):
        values = customer_orders["order_value"].to_numpy(dtype=float)
        dates = customer_orders["order_date"]
        first_date, last_date = dates.iloc[0], dates.iloc[-1]
        frequency = len(customer_orders) - 1
        repeat_values = values[1:]
        positive_repeats = bool(np.isfinite(repeat_values).all() and (repeat_values > 0).all())
        monetary_value = float(repeat_values.mean()) if frequency else 0.0
        rows.append({
            "CustomerID": int(customer_id),
            "frequency": int(frequency),
            "recency": (last_date - first_date).total_seconds() / 86400,
            "T": (cutoff - first_date).total_seconds() / 86400,
            "monetary_value": monetary_value,
            "repeat_orders_positive": positive_repeats,
            "order_count": int(len(customer_orders)),
        })
    return pd.DataFrame(rows, columns=columns)


def gamma_gamma_eligibility(summary: pd.DataFrame) -> tuple[pd.Series, dict[str, int]]:
    """Return valid repeat-customer mask and explicit population exclusion counts."""
    repeat = summary["frequency"].gt(0)
    positive = np.isfinite(summary["monetary_value"]) & summary["monetary_value"].gt(0)
    repeat_order_values_valid = summary["repeat_orders_positive"].astype(bool)
    eligible = repeat & positive & repeat_order_values_valid
    counts = {
        "customers_total": int(len(summary)),
        "eligible_customers": int(eligible.sum()),
        "excluded_single_purchase": int((~repeat).sum()),
        "excluded_nonpositive_or_invalid_repeat_monetary": int((repeat & ~(positive & repeat_order_values_valid)).sum()),
    }
    counts["excluded_total"] = counts["customers_total"] - counts["eligible_customers"]
    return eligible, counts


@dataclass
class ClassicalCLVModel:
    """Fitted lifetime models and a 90-day expected gross purchase-value scorer."""

    config: ClassicalCLVConfig
    bgf: object | None = None
    ggf: object | None = None
    fit_end: pd.Timestamp | None = None
    fit_summary: pd.DataFrame | None = None
    gamma_gamma_fit_counts: dict[str, int] | None = None

    def fit(self, summary: pd.DataFrame, fit_end) -> "ClassicalCLVModel":
        from lifetimes import BetaGeoFitter, GammaGammaFitter

        required = {"frequency", "recency", "T", "monetary_value", "repeat_orders_positive"}
        missing = required - set(summary.columns)
        if missing:
            raise ValueError(f"Summary missing required columns: {sorted(missing)}")
        if summary.empty or (summary["T"] <= 0).any() or (summary["recency"] > summary["T"]).any():
            raise ValueError("BG/NBD summary must have positive T and recency <= T")
        eligible, counts = gamma_gamma_eligibility(summary)
        if eligible.sum() < 2:
            raise ValueError("Gamma-Gamma requires at least two eligible repeat customers")
        self.bgf = BetaGeoFitter(penalizer_coef=self.config.penalizer_coefficient)
        self.bgf.fit(
            summary["frequency"].to_numpy(dtype=float),
            summary["recency"].to_numpy(dtype=float),
            summary["T"].to_numpy(dtype=float),
        )
        self.ggf = GammaGammaFitter(penalizer_coef=self.config.penalizer_coefficient)
        # Enforce q >= 1 for nonnegative population monetary mean. This dataset's
        # estimate lands at q=1; prediction uses an algebraically simplified
        # conditional expectation to avoid lifetimes' 0 * infinity numerical form.
        self.ggf.fit(
            summary.loc[eligible, "frequency"].to_numpy(dtype=float),
            summary.loc[eligible, "monetary_value"].to_numpy(dtype=float),
            q_constraint=True,
        )
        self.fit_end = pd.Timestamp(fit_end)
        self.fit_summary = summary.copy()
        self.gamma_gamma_fit_counts = counts
        return self

    def predict(self, summary: pd.DataFrame) -> pd.DataFrame:
        """Score valid summary rows; retain CustomerID for safe downstream joins."""
        if self.bgf is None or self.ggf is None:
            raise RuntimeError("Fit classical CLV models before predicting")
        repeat_eligible, _ = gamma_gamma_eligibility(summary)
        # Gamma-Gamma is fit to repeaters and one-time customers are not scored:
        # q=1 makes the population mean unbounded for frequency=0.
        eligible = repeat_eligible
        counts = {
            "customers_total": int(len(summary)),
            "scored_customers": int(eligible.sum()),
            "excluded_invalid_repeat_monetary": int((summary["frequency"].gt(0) & ~repeat_eligible).sum()),
        }
        scored = summary.loc[eligible].copy()
        if scored.empty:
            scored["expected_purchases_90d"] = pd.Series(dtype=float)
            scored["expected_average_order_value"] = pd.Series(dtype=float)
            scored["predicted_90d_gross_purchase_value"] = pd.Series(dtype=float)
            return scored
        expected_transactions = self.bgf.conditional_expected_number_of_purchases_up_to_time(
            self.config.horizon_days,
            scored["frequency"].to_numpy(dtype=float),
            scored["recency"].to_numpy(dtype=float),
            scored["T"].to_numpy(dtype=float),
        )
        frequency = scored["frequency"].to_numpy(dtype=float)
        monetary_value = scored["monetary_value"].to_numpy(dtype=float)
        p, q, v = (float(self.ggf.params_[name]) for name in ("p", "q", "v"))
        # Equivalent to lifetimes' conditional_expected_average_profit formula:
        # p*(v + f*m) / (p*f + q - 1). For q=1, f>0 remains well-defined.
        expected_order_value = p * (v + frequency * monetary_value) / (p * frequency + q - 1)
        expected_transactions = np.asarray(expected_transactions, dtype=float)
        expected_order_value = np.asarray(expected_order_value, dtype=float)
        valid_prediction = np.isfinite(expected_transactions) & np.isfinite(expected_order_value) & (expected_order_value > 0)
        counts["excluded_invalid_model_prediction"] = int((~valid_prediction).sum())
        scored = scored.loc[valid_prediction].copy()
        counts["scored_customers"] = int(len(scored))
        expected_transactions = expected_transactions[valid_prediction]
        expected_order_value = expected_order_value[valid_prediction]
        scored["expected_purchases_90d"] = expected_transactions
        scored["expected_average_order_value"] = expected_order_value
        scored["predicted_90d_gross_purchase_value"] = (
            scored["expected_purchases_90d"] * scored["expected_average_order_value"]
        )
        scored.attrs["prediction_coverage"] = counts
        return scored


def fit_classical_clv(transactions: pd.DataFrame, fit_end, config: ClassicalCLVConfig | None = None) -> ClassicalCLVModel:
    """Fit using order histories strictly before fit_end (normally May 1, 2011)."""
    config = config or ClassicalCLVConfig()
    cutoff = pd.Timestamp(fit_end)
    summary = build_customer_summary(transactions, cutoff)
    return ClassicalCLVModel(config).fit(summary, cutoff)


def future_gross_purchase_value(transactions: pd.DataFrame, customer_ids, snapshot_date, horizon_days: int = 90) -> pd.Series:
    """Evaluation-only gross purchase amount from non-cancellation positive lines."""
    t = pd.Timestamp(snapshot_date)
    end = t + pd.Timedelta(days=horizon_days)
    future = transactions.loc[
        transactions["CustomerID"].isin(customer_ids)
        & (transactions["InvoiceDate"] > t)
        & (transactions["InvoiceDate"] <= end)
        & transactions["is_purchase"]
    ]
    totals = future.groupby("CustomerID")["line_value"].sum()
    customer_index = pd.Index(pd.unique(customer_ids), name="CustomerID")
    return totals.reindex(customer_index, fill_value=0.0).rename("future_gross_purchase_value_90d")
