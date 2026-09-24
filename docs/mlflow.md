# ChurnGuard MLflow tracking

MLflow records completed ChurnGuard experiment parameters, metrics, and
aggregate artifacts. The tracking wrapper consumes existing evaluation
results; it does not fit models, choose a winner, or alter predictions.

## Experiments and runs

`track_experiments` creates separate experiments/runs for:

- **Churn:** five individual runs named `majority_baseline`, `logistic_rfm`,
  `rule_based_rfm`, `lightgbm_raw`, and `lightgbm_calibrated`. Each logs
  validation/final-test PR-AUC, Lift@Decile 1, and Brier score. The calibrated
  run is self-contained and records raw and calibrated metrics side by side.
  The raw LightGBM run retains walk-forward fold and aggregate metrics/artifacts.
- **Future value:** the supervised LightGBM regression model for
  `future_net_spend_90d`, with its own validation/test MAE, RMSE, and Spearman
  metrics and its actual fitted estimator parameters.
- **Targeting analytics:** optional June top-decile actual future net-spend
  capture and scenario assumptions under names that do not imply revenue
  saved, profit, ROI, or campaign uplift.

The raw LightGBM churn run logs `walk_forward_metrics.csv` and
`walk_forward_aggregate.csv`; these contain fold/model metrics, not customer
rows. Raw transaction records, customer-level feature tables, June predictions,
and SHAP local explanation CSVs are never logged. Aggregate SHAP/global
importance files remain local as well.

## Local tracking setup

By default, `TrackingConfig` uses a local SQLite backend at
`mlruns/mlflow.db` and stores experiment artifacts under `mlruns/artifacts/`.
The directory is runtime-relative to the current project directory, not a
machine-specific path, and is ignored by git. No external tracking server is
required.

Configuration can be supplied in Python or through environment variables:

- `CHURNGUARD_MLFLOW_TRACKING_URI` overrides the local default URI.
- `CHURNGUARD_MLFLOW_DIR` selects the local tracking/artifact directory.
- `CHURNGUARD_MLFLOW_CHURN_EXPERIMENT`
- `CHURNGUARD_MLFLOW_VALUE_EXPERIMENT`
- `CHURNGUARD_MLFLOW_TARGETING_EXPERIMENT`

Example:

```python
from churnguard.tracking import TrackingConfig, track_experiments

tracked = track_experiments(
    evaluation_results,
    walk_forward_results,
    TrackingConfig(),
    split_config,
    model_config,
)
print(tracked["tracking_uri"])
```

To inspect local runs with the MLflow UI, use the printed local SQLite URI as
the backend store URI, for example:

```powershell
.venv\Scripts\mlflow ui --backend-store-uri sqlite:///C:/path/to/project/mlruns/mlflow.db
```

## Reproducibility and interpretation

Logged values are the outputs from the supplied run; tracking does not
recompute them or fit any model. `evaluate_models` now retains Brier scores for
each already-scored churn baseline so each run can report the same probability
metrics. Parameters record the training/validation/test snapshot dates,
random seed, feature-set identifier, and the model hyperparameters. Walk-forward
metrics use raw churn probabilities and are recorded alongside the separate
fixed-split metrics. Targeting capture is explicitly a held-out actual future
net-spend statistic. Scenario metrics are labeled as modeled expected net value
under assumptions, not measured business impact.

Reproduction still depends on the installed package versions, source data,
feature code, and runtime environment; this project does not yet use a lockfile
or container image. The four snapshots provide only two walk-forward folds,
and adjacent 90-day label windows overlap. Local MLflow tracking is
experiment-history storage, not a production tracking server or deployment
system.
