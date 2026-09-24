# ChurnGuard inference API

The service loads frozen inference objects created by the existing temporal
evaluation workflow. Its stack is:

```text
validated historical feature record
  → frozen LightGBM churn model
  → fitted sigmoid calibrator
  → frozen supervised LightGBM 90-day value model
  → risk-weighted score and optional batch ranks
```

The API performs inference using frozen artifacts; it does not retrain models.
The June final-test labels are not part of the exported artifacts. No customer
rows, prediction tables, or raw transactions are saved in the artifact bundle.

## Model artifacts

The default artifact directory is `models/`, configurable with
`CHURNGUARD_MODEL_DIR` or `create_app(artifact_dir=...)`. It is gitignored.
Export models from the normal fitted evaluation output:

```python
from churnguard.artifacts import export_inference_artifacts

artifact_path = export_inference_artifacts(
    evaluation_results,
    split_config=split_config,
    model_config=model_config,
    artifact_dir="models",
)
```

The directory contains `churn/model.joblib` (fitted churn classifier and
sigmoid calibrator), `future_value/model.joblib` (fitted supervised value
model), and `metadata.json` (format version, feature names/order, train-derived
imputation/category rules, target transform bounds, feature-set identifier,
training and validation snapshot dates, and seed). The exporter accepts fitted objects from `evaluate_models`;
it does not fit or alter them. Do not load untrusted joblib files.

## Input schema

Both scoring routes use this request structure:

```json
{
  "CustomerID": 12345,
  "snapshot_date": "2011-06-01",
  "features": {
    "recency_days": 14,
    "purchase_invoice_count": 3,
    "historical_net_spend": 215.75,
    "country": "United Kingdom"
  }
}
```

`features` must contain exactly the model feature names recorded in metadata.
Feature order is restored from metadata before inference. Numeric features must
be finite numbers. Categorical features, including `country`, must match a
category observed in the corresponding training feature schema. Missing,
unexpected, null, malformed, and unsupported categorical inputs are rejected;
the service does not invent feature values. `CustomerID` and `snapshot_date`
are optional metadata and are never passed to either model. Future outcomes
such as `churn_90d`, `repurchased_90d`, and `future_net_spend_90d` are not model
features and are rejected as unexpected keys.

## Routes

### `POST /predict`

Scores one customer. Example response:

```json
{
  "CustomerID": 12345,
  "snapshot_date": "2011-06-01T00:00:00",
  "churn_probability": 0.31,
  "predicted_90d_value": 428.7,
  "risk_weighted_value": 132.897
}
```

The numeric values above illustrate the response shape.

`churn_probability` is the fitted sigmoid/Platt calibrator applied to the raw
LightGBM churn probability. `predicted_90d_value` is the supervised LightGBM
prediction, inverted from the project's signed-log target transform using
training-only bounds. `risk_weighted_value` is their product. The
risk_weighted_value is a prioritization/value-at-risk score, not guaranteed
revenue saved.

### `POST /batch_score`

Accepts `{"customers": [<record>, ...]}` and returns `count` plus
`scored_customers`. Every result contains the same scores and metadata as
`/predict`, plus `churn_rank`, `value_rank`, and `risk_value_rank`. Ranks are
descending within snapshot date; ties use ascending CustomerID, then original
input order. When snapshot dates are omitted, the records are ranked together
as one batch. Optional metadata is preserved when supplied. No actual future
labels or transactions are accessed.

### `GET /health`

Returns `200` with `status: "ok"` and three loaded flags when all required
artifacts are available. Missing/invalid artifacts return `503` and
`status: "not_ready"`; the service never fits replacement models.

## Errors and local startup

Malformed Pydantic requests and invalid/missing/extra model features return
`422`. Inference requested while artifacts are unavailable returns `503`.
An error from a loaded model returns `500`; no synthetic score is returned.

From the repository root, install the project/dependencies once and export
artifacts from the existing evaluation output. Then start the API:

```powershell
pip install -e ".[analysis]"
$env:CHURNGUARD_MODEL_DIR = "models"
.venv\Scripts\uvicorn.exe churnguard.api:app --reload
```

The service listens on Uvicorn's local development address. The API returns
JSON suitable for downstream consumption; no Power BI connector is implemented.

## Limitations

The serving contract follows the current four-snapshot engineered feature set.
Unknown categories are rejected rather than guessed. The current Platt
calibrator is fitted on the single May validation snapshot, and validation
calibration diagnostics are in-sample for that mapping. This API is a local
inference foundation, not an authentication, deployment, drift-monitoring, or
model-refresh system. Predictions describe expected future net spend and risk
ranking; they do not estimate intervention impact.
