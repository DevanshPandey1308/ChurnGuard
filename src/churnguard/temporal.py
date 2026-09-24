"""Date-based dataset splits that keep repeated customers in temporal order."""

from dataclasses import dataclass

import pandas as pd

from .model_config import TemporalSplitConfig


@dataclass(frozen=True)
class TemporalDataSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_by_snapshot_date(
    table: pd.DataFrame,
    config: TemporalSplitConfig,
    date_column: str = "snapshot_date",
) -> TemporalDataSplit:
    """Split rows by configured snapshot dates, never by customer/random row.

    Since CustomerID repeats across snapshots, random splitting could put later
    observations for a customer into training while earlier snapshots are held out.
    """
    if date_column not in table:
        raise ValueError(f"Missing split date column: {date_column}")
    dates = pd.to_datetime(table[date_column]).dt.strftime("%Y-%m-%d")
    configured = set(config.train_dates) | set(config.validation_dates) | set(config.test_dates)
    observed = set(dates)
    extra = observed - configured
    if extra:
        raise ValueError(f"Unassigned snapshot dates: {sorted(extra)}")

    def rows_for(target_dates):
        return table.loc[dates.isin(target_dates)].copy().reset_index(drop=True)

    return TemporalDataSplit(
        train=rows_for(config.train_dates),
        validation=rows_for(config.validation_dates),
        test=rows_for(config.test_dates),
    )
