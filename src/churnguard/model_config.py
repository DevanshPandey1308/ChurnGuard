"""Configuration for chronological model experiments."""

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class TemporalSplitConfig:
    train_dates: Sequence[str] = ("2011-03-01", "2011-04-01")
    validation_dates: Sequence[str] = ("2011-05-01",)
    test_dates: Sequence[str] = ("2011-06-01",)
    target_column: str = "churn_90d"

    def __post_init__(self) -> None:
        parts = [set(self.train_dates), set(self.validation_dates), set(self.test_dates)]
        if any(not part for part in parts):
            raise ValueError("Train, validation, and test date sets must be non-empty")
        if parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2]:
            raise ValueError("Snapshot dates must not overlap between temporal splits")
        dates = [sorted(pd.to_datetime(date) for date in part) for part in parts]
        if not (max(dates[0]) < min(dates[1]) and max(dates[1]) < min(dates[2])):
            raise ValueError("Split dates must be strictly chronological: train < validation < test")


@dataclass(frozen=True)
class ModelConfig:
    random_seed: int = 42
    calibration_method: str = "sigmoid"
    reliability_bins: int = 5
    lightgbm_params: Mapping[str, object] = field(default_factory=lambda: {
        "n_estimators": 300,
        "learning_rate": 0.04,
        "num_leaves": 15,
        "max_depth": -1,
        "min_child_samples": 30,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "verbosity": -1,
    })

    def __post_init__(self) -> None:
        if self.calibration_method != "sigmoid":
            raise ValueError("Only sigmoid (Platt) calibration is currently supported")
        if self.reliability_bins < 2:
            raise ValueError("reliability_bins must be at least 2")
