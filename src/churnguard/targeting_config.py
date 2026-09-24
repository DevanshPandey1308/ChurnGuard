"""Configuration for risk/value customer targeting scenarios."""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TargetingConfig:
    """Ranking and explicitly hypothetical retention-offer assumptions.

    ``retention_offer_cost`` is assumed to be incurred for every targeted
    customer, regardless of acceptance. Acceptance and retained-value fractions
    are scenario assumptions, not quantities learned from transaction data.
    """

    primary_ranking_method: str = "risk_weighted_value"
    targeting_thresholds: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
    retention_offer_cost: float = 50.0
    assumed_acceptance_rate: float = 0.25
    retained_value_fraction: float = 0.80
    sensitivity_acceptance_rates: Sequence[float] = (0.10, 0.25, 0.50)
    sensitivity_retained_value_fractions: Sequence[float] = (0.50, 0.80, 1.00)
    top_fraction: float = 0.10

    def __post_init__(self) -> None:
        methods = {"risk_weighted_value", "calibrated_churn_probability", "predicted_90d_value"}
        if self.primary_ranking_method not in methods:
            raise ValueError(f"primary_ranking_method must be one of {sorted(methods)}")
        if not self.targeting_thresholds or any(not 0 < x <= 1 for x in self.targeting_thresholds):
            raise ValueError("targeting_thresholds must contain fractions in (0, 1]")
        if len(set(self.targeting_thresholds)) != len(self.targeting_thresholds):
            raise ValueError("targeting_thresholds must be unique")
        if self.retention_offer_cost < 0:
            raise ValueError("retention_offer_cost cannot be negative")
        for name, rate in (("assumed_acceptance_rate", self.assumed_acceptance_rate),
                           ("retained_value_fraction", self.retained_value_fraction),
                           ("top_fraction", self.top_fraction)):
            if not 0 <= rate <= 1 or (name == "top_fraction" and rate == 0):
                raise ValueError(f"{name} must be in {'(0, 1]' if name == 'top_fraction' else '[0, 1]'}")
        for values, name in ((self.sensitivity_acceptance_rates, "sensitivity_acceptance_rates"),
                             (self.sensitivity_retained_value_fractions, "sensitivity_retained_value_fractions")):
            if not values or any(not 0 <= x <= 1 for x in values):
                raise ValueError(f"{name} must contain values in [0, 1]")
