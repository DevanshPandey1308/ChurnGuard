"""Normalize Online Retail transaction rows without discarding adjustments."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {"InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"}


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Return typed, deduplicated rows; cancellation and return rows remain available.

    C-prefixed invoices are cancellation records; other negative quantities are
    returns/negative adjustments. All source columns are retained. We do not match
    cancellations to original invoices: their signed line values remain in net
    monetary totals, but cancellation lines never count as purchase orders.
    """
    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = raw.drop_duplicates().copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype("string").str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df = df.dropna(subset=["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "CustomerID"])
    df = df.loc[df["CustomerID"] > 0].copy()
    df["CustomerID"] = df["CustomerID"].astype("int64")
    df["is_cancellation"] = df["InvoiceNo"].str.upper().str.startswith("C", na=False)
    df["is_return"] = df["Quantity"].lt(0) & ~df["is_cancellation"]
    df["is_purchase"] = df["Quantity"].gt(0) & ~df["is_cancellation"]
    df["transaction_type"] = "other"
    df.loc[df["is_purchase"], "transaction_type"] = "purchase"
    df.loc[df["is_return"], "transaction_type"] = "return_adjustment"
    df.loc[df["is_cancellation"], "transaction_type"] = "cancellation"
    df["line_value"] = df["Quantity"] * df["UnitPrice"]
    return df.sort_values(["InvoiceDate", "InvoiceNo"], kind="stable").reset_index(drop=True)
