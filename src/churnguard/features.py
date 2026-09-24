"""Compact historical customer features computed strictly before snapshot time."""

from typing import Sequence

import numpy as np
import pandas as pd


def build_features(
    transactions: pd.DataFrame,
    snapshots: pd.DataFrame,
    trend_window_months: int = 3,
    activity_window_months: Sequence[int] = (3, 6),
) -> pd.DataFrame:
    """Build customer features from rows strictly earlier than each snapshot.

    Purchase/order statistics use positive non-cancellation lines. Net spend keeps
    signed values from all invoice types; no original-order cancellation matching
    is inferred. Trend/activity windows are calendar-month offsets from snapshot.
    """
    result = []
    by_customer = {key: group for key, group in transactions.groupby("CustomerID", sort=False)}
    for row in snapshots.itertuples(index=False):
        customer_tx = by_customer.get(row.CustomerID)
        if customer_tx is None:
            continue
        hist = customer_tx.loc[customer_tx["InvoiceDate"] < row.snapshot_date]
        purchases = hist.loc[hist["is_purchase"]]
        if purchases.empty:
            continue
        order_values = purchases.groupby("InvoiceNo")["line_value"].sum()
        order_dates = purchases.groupby("InvoiceNo")["InvoiceDate"].min().sort_values()
        purchase_dates = order_dates.drop_duplicates().sort_values()
        gaps = purchase_dates.diff().dt.total_seconds().div(86400).dropna()
        first_purchase = purchases["InvoiceDate"].min()
        gross_spend = float(purchases["line_value"].sum())
        returns = hist.loc[hist["is_return"]]
        all_negative = hist.loc[hist["Quantity"].lt(0)]
        value = float(hist["line_value"].sum())
        snapshot = row.snapshot_date
        recent_start = snapshot - pd.DateOffset(months=trend_window_months)
        previous_start = snapshot - pd.DateOffset(months=2 * trend_window_months)
        recent = hist.loc[hist["InvoiceDate"] >= recent_start]
        previous = hist.loc[(hist["InvoiceDate"] >= previous_start) & (hist["InvoiceDate"] < recent_start)]
        recent_orders = recent.loc[recent["is_purchase"], "InvoiceNo"].nunique()
        previous_orders = previous.loc[previous["is_purchase"], "InvoiceNo"].nunique()
        recent_spend = float(recent["line_value"].sum())
        previous_spend = float(previous["line_value"].sum())
        spend_ratio_name = f"spend_trend_ratio_{trend_window_months}m"
        row_features = {
            "CustomerID": row.CustomerID,
            "snapshot_date": snapshot,
            "recency_days": (snapshot - purchases["InvoiceDate"].max()).total_seconds() / 86400,
            "purchase_invoice_count": int(purchases["InvoiceNo"].nunique()),
            "purchase_line_count": int(len(purchases)),
            "units_purchased": float(purchases["Quantity"].sum()),
            "historical_net_spend": value,
            "log1p_purchase_invoice_count": float(np.log1p(purchases["InvoiceNo"].nunique())),
            "signed_log1p_historical_net_spend": float(np.sign(value) * np.log1p(abs(value))),
            "tenure_days": (snapshot - first_purchase).total_seconds() / 86400,
            "average_order_value": float(order_values.mean()),
            "average_items_per_order": float(purchases.groupby("InvoiceNo")["Quantity"].sum().mean()),
            "distinct_products": int(purchases["StockCode"].nunique()) if "StockCode" in purchases else 0,
            "mean_inter_purchase_gap_days": float(gaps.mean()) if not gaps.empty else float("nan"),
            "std_inter_purchase_gap_days": float(gaps.std()) if len(gaps) > 1 else float("nan"),
            f"recent_{trend_window_months}m_net_spend": recent_spend,
            f"previous_{trend_window_months}m_net_spend": previous_spend,
            f"spend_trend_difference_{trend_window_months}m": recent_spend - previous_spend,
            spend_ratio_name: recent_spend / previous_spend if previous_spend != 0 else float("nan"),
            f"recent_{trend_window_months}m_order_count": int(recent_orders),
            f"previous_{trend_window_months}m_order_count": int(previous_orders),
            f"order_trend_difference_{trend_window_months}m": int(recent_orders - previous_orders),
            f"order_trend_ratio_{trend_window_months}m": recent_orders / previous_orders if previous_orders else float("nan"),
            "return_quantity": float(-returns["Quantity"].sum()),
            "return_value": float(-returns["line_value"].sum()),
            "negative_adjustment_rate": float(-all_negative["line_value"].sum() / gross_spend) if gross_spend else float("nan"),
            "snapshot_month": int(snapshot.month),
            "inactive_days": (snapshot - purchases["InvoiceDate"].max()).days,
            "country": hist["Country"].dropna().iloc[-1] if "Country" in hist and hist["Country"].notna().any() else None,
        }
        for months in activity_window_months:
            start = snapshot - pd.DateOffset(months=months)
            active = hist.loc[hist["InvoiceDate"] >= start]
            row_features[f"activity_{months}m_order_count"] = int(active.loc[active["is_purchase"], "InvoiceNo"].nunique())
            row_features[f"active_{months}m"] = int(row_features[f"activity_{months}m_order_count"] > 0)
        result.append(row_features)
    return pd.DataFrame(result)
