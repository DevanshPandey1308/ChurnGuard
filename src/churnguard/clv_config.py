"""Small configuration surface for the classical probabilistic CLV benchmark."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassicalCLVConfig:
    horizon_days: int = 90
    penalizer_coefficient: float = 0.001

    def __post_init__(self) -> None:
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if self.penalizer_coefficient < 0:
            raise ValueError("penalizer_coefficient cannot be negative")
