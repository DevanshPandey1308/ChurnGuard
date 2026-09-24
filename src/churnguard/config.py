"""Configuration for repeatable snapshot dataset construction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class PipelineConfig:
    data_path: Path
    snapshot_dates: Sequence[str]
    horizon_days: int = 90
    encoding: str = "latin1"
    trend_window_months: int = 3
    activity_window_months: Sequence[int] = (3, 6)

    def __post_init__(self) -> None:
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if self.trend_window_months <= 0 or any(window <= 0 for window in self.activity_window_months):
            raise ValueError("feature windows must be positive")
