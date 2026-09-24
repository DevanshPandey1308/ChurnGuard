"""Configuration for supervised 90-day customer-value experiments."""

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ValueModelConfig:
    target_column: str = "future_net_spend_90d"
    random_seed: int = 42
    target_transform: str = "signed_log1p"
    top_fraction: float = 0.10
    linear_params: Mapping[str, object] = field(default_factory=lambda: {"fit_intercept": True})
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
        if self.target_transform != "signed_log1p":
            raise ValueError("Only signed_log1p target transformation is currently supported")
        if not 0 < self.top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1]")
