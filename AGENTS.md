# ChurnGuard project rules

- Goal: predict customer churn (no repurchase in the next 90 days) and net customer spend over that same future window.
- Build repeated point-in-time customer snapshots. Features use transactions strictly before snapshot date `t`; labels use `(t, t + horizon]`. Customers may appear in multiple snapshots.
- Prevent and test future-data leakage. Use chronological/walk-forward evaluation, never a random split.
- Preserve cancellations and returns in cleaning; do not drop all negative-quantity rows. Document any assumptions about matching or accounting for adjustments.
- Keep paths, snapshot dates, and the 90-day horizon configurable. Avoid machine-specific paths.
- RFM is a reusable feature block/baseline, not the final ML solution. Keep the Python pipeline modular for later models, SHAP, serving, reporting, tracking, deployment, and drift monitoring.
- PSI drift monitoring compares only frozen model input features; use the model's actual training snapshots as reference and the latest scoring batch as current. A PSI above 0.20 is a review/retraining signal, not proof of model inaccuracy; never use the final test cohort as reference or retrain automatically.
- Do not fabricate business-impact metrics. Do not remove existing notebook, dashboard, or data without review.
