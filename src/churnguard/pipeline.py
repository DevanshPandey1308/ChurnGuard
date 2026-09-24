"""Composable public entry points for the ChurnGuard data pipeline."""

import pandas as pd

from .cleaning import clean_transactions
from .config import PipelineConfig
from .features import build_features
from .labels import build_labels
from .snapshots import build_snapshots


def load_transactions(path, encoding: str = "latin1") -> pd.DataFrame:
    return clean_transactions(pd.read_csv(path, encoding=encoding))


def run_pipeline(config: PipelineConfig) -> pd.DataFrame:
    """Load the configured source and return a labeled customer-snapshot table."""
    transactions = load_transactions(config.data_path, config.encoding)
    return build_training_table(transactions, config)


def build_training_table(transactions: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    snapshots = build_snapshots(transactions, config.snapshot_dates, config.horizon_days)
    features = build_features(
        transactions,
        snapshots,
        trend_window_months=config.trend_window_months,
        activity_window_months=config.activity_window_months,
    )
    labels = build_labels(transactions, snapshots, config.horizon_days)
    table = features.merge(labels, on=["CustomerID", "snapshot_date"], validate="one_to_one")
    if table.duplicated(["CustomerID", "snapshot_date"]).any():
        raise ValueError("Training table contains duplicate customer snapshots")
    return table
