"""Point-in-time snapshot cohort construction."""

import pandas as pd


def build_snapshots(transactions: pd.DataFrame, snapshot_dates, horizon_days: int = 90) -> pd.DataFrame:
    """Create one row per customer with a purchase strictly before each date.

    Dates without a fully observed future horizon are omitted to keep labels complete.
    """
    dates = pd.to_datetime(list(snapshot_dates))
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    max_date = transactions["InvoiceDate"].max()
    rows = []
    for date in dates:
        if date + pd.Timedelta(days=horizon_days) > max_date:
            continue
        prior = transactions.loc[
            (transactions["InvoiceDate"] < date)
            & transactions["Quantity"].gt(0)
            & ~transactions["is_cancellation"]
        ]
        for customer_id in prior["CustomerID"].unique():
            rows.append({"CustomerID": int(customer_id), "snapshot_date": date})
    return pd.DataFrame(rows, columns=["CustomerID", "snapshot_date"]).drop_duplicates().reset_index(drop=True)
