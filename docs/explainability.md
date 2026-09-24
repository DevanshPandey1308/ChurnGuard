# ChurnGuard SHAP explainability

## Model and data explained

The explainability module explains the existing LightGBM binary churn model; it
does not replace or refit that model. The fixed temporal model is fitted using
March and April snapshots, May is used for validation/calibration, and June is
the final held-out test period. SHAP explains June feature rows after the
existing model has been fitted. June labels are not accepted by the SHAP API
and are not used to compute feature contributions.

The June feature matrix is built in the fitted model's exact feature order using
the existing `prepare_lightgbm_features` preprocessing. Numeric missing values
and category levels are determined from the training features; `Country` is
preserved as a categorical feature. Feature order mismatches and identifiers or
future-derived target columns are rejected rather than silently reordered.

## Global and local outputs

`explain_lightgbm_churn` returns global importance with mean absolute SHAP
magnitude, signed mean SHAP value, and rank. It also selects deterministic June
rows with the lowest, median, and highest raw churn probabilities (ties break
by CustomerID and snapshot date). Local output lists each selected row's raw
and calibrated probability and its strongest positive and negative feature
contributions, including the feature value and contribution rank.

The default local artifact directory is `artifacts/shap/`, ignored by git. It
contains global importance, local explanations, selected-row metadata, and
summary/bar plots. The local CSVs can contain CustomerID; they are not uploaded
to MLflow. Reproduce them by running the existing `evaluate_models` workflow to
fit the churn model, preparing train/June feature frames in their existing
model feature order, then calling `explain_lightgbm_churn` with June prediction
probabilities. Set `artifact_dir` to use another location.

```python
from churnguard.temporal import split_by_snapshot_date
from churnguard.explainability import explain_lightgbm_churn

split = split_by_snapshot_date(training_table, split_config)
feature_names = evaluation["feature_columns"]["lightgbm"]
train_features = split.train.loc[:, feature_names]
june_features = split.test.loc[:, feature_names]  # Features only; no labels.
june_predictions = evaluation["predictions"]["test"]
metadata = june_predictions[[
    "CustomerID", "snapshot_date", "raw_lightgbm_probability",
    "calibrated_lightgbm_probability",
]].rename(columns={
    "raw_lightgbm_probability": "raw_churn_probability",
    "calibrated_lightgbm_probability": "calibrated_churn_probability",
})
explanations = explain_lightgbm_churn(
    evaluation["models"]["lightgbm"], train_features, june_features, metadata,
    metadata["raw_churn_probability"], metadata["calibrated_churn_probability"],
)
```

## SHAP output space and additivity

The TreeExplainer is configured with `model_output="raw"`. For binary
LightGBM, the contributions are in raw model-margin/log-odds space, not
probability space. The module checks that the expected/base value plus the sum
of feature SHAP values reconstructs `model.predict(..., raw_score=True)` within
a numerical tolerance and returns the maximum absolute error. Calibrated
probabilities are reported alongside local explanations for context; they are
not the values reconstructed by these SHAP contributions.

## Interpretation limits

SHAP explains the model's learned associations. It does NOT establish that
changing a feature will cause churn to change. Correlated predictors can share
or redistribute attribution, and a contribution describes the fitted model's
output relative to its baseline, not an intervention effect or customer
diagnosis. The available data has four snapshots, one validation month, and one
final test month; temporal label windows overlap. June explanations are
post-hoc explanations for an already-fitted model and do not alter June's role
as the final evaluation period.
