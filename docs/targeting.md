# ChurnGuard targeting layer

The targeting layer ranks each customer snapshot using calibrated churn
probability and the supervised LightGBM prediction of future 90-day **net
spend**. It does not use BG/NBD + Gamma-Gamma lifetime value for its primary
score.

For each observation:

```text
risk_weighted_value = calibrated_churn_probability * predicted_90d_value
```

This is a prediction-based prioritization score, also described as value at
risk. It is not guaranteed revenue saved. Equal scores are ordered
deterministically by ascending `CustomerID` within each snapshot.

## Hypothetical offer scenario

For a customer selected for targeting:

```text
expected_recovered_value_under_scenario =
    calibrated_churn_probability
    * assumed_acceptance_rate
    * retained_value_fraction
    * max(predicted_90d_value, 0)

expected_net_value_under_scenario =
    expected_recovered_value_under_scenario - retention_offer_cost
```

The nonnegative clamp means a negative predicted net-spend value is treated as
zero recoverable value in this hypothetical calculation. The configured offer
cost is charged for every targeted customer, even if the assumed offer is not
accepted. This is a transparent scenario convention, not an observed cost or
learned response. The expected net value is not profit: the model includes only
the configured offer cost and does not estimate margin, intervention response,
or other costs.

Default demonstration assumptions in `TargetingConfig` are an offer cost of
50 currency units, 25% acceptance, and 80% retained value. They are illustrative
inputs only and should be replaced with approved scenario assumptions for any
business analysis. Validation provides the threshold curve and the configured
acceptance/retained-value sensitivity grid. June is reserved for final held-out
top-decile comparisons and does not select an assumption or threshold.

## Evaluation boundaries and limitations

Held-out actual `future_net_spend_90d` is joined only after prediction-only
scores and ranks are created. Top-decile actual value capture is an evaluation
statistic, not revenue generated, saved, or causal uplift. Risk-weighted,
churn-only, and value-only rankings are reported side by side.

The current evidence covers four snapshots, one validation month, and one final
test month. The 90-day label windows overlap. Acceptance and retained-value
assumptions are not estimated from interventions, and the project has no
causal/uplift data. Results therefore describe this temporal holdout and these
hypothetical inputs only.
