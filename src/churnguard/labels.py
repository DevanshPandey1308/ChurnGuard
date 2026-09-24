"""Future-window repurchase and value targets."""

import pandas as pd


def build_labels(transactions: pd.DataFrame, snapshots: pd.DataFrame, horizon_days: int = 90) -> pd.DataFrame:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    result = []
    by_customer = {key: group for key, group in transactions.groupby("CustomerID", sort=False)}
    for row in snapshots.itertuples(index=False):
        end = row.snapshot_date + pd.Timedelta(days=horizon_days)
        customer_tx = by_customer.get(row.CustomerID)
        if customer_tx is None:
            future = transactions.iloc[0:0]
        else:
            dates = customer_tx["InvoiceDate"]
            future = customer_tx.loc[(dates > row.snapshot_date) & (dates <= end)]
        purchases = future.loc[future["Quantity"].gt(0) & ~future["is_cancellation"]]
        # Signed cancellation and return lines stay in net realized value; no
        # cancellation-to-original-invoice matching is inferred.
        net_value = future["line_value"].sum()
        result.append({
            "CustomerID": row.CustomerID,
            "snapshot_date": row.snapshot_date,
            "repurchased_90d": int(not purchases.empty),
            "churn_90d": int(purchases.empty),
            "future_net_spend_90d": float(net_value),
        })
    return pd.DataFrame(result)
