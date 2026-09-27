# ChurnGuard

ChurnGuard is an end-to-end customer churn and future-value prediction project built on the UCI Online Retail transaction dataset. It turns historical transactions into leakage-safe customer snapshots, estimates the chance that a customer will not purchase in the next 90 days, predicts their net spend over that period, and ranks customer snapshots for review.

The project is an engineering and modeling demonstration. It does not claim that a retention action will save revenue or cause a customer to return.

<!-- Once deployed, replace this with your real links:
## Live demo
- API: https://your-service.onrender.com/health
- App: https://your-app.vercel.app
-->

## Problem

Traditional RFM analysis describes past customer behavior. ChurnGuard builds on that foundation to answer two forward-looking questions:

1. Is a customer likely to make no purchase in the next 90 days?
2. What is their predicted net spend during that future period?

The dataset contains transactions from one UK-based online retailer between December 2010 and December 2011. It has no campaign response, intervention, margin, or demographic data, so the project cannot measure causal uplift or realized retention impact.

## How it works

```text
Online Retail transactions
        ↓
Cleaning and signed return/cancellation adjustments
        ↓
Repeated point-in-time customer snapshots
        ↓
Historical features + future 90-day labels
        ↓
Chronological model evaluation and experiment tracking
        ↓
Frozen churn/value models → FastAPI scoring → React interface
                       ↘ PSI drift check
                       ↘ Power BI-ready exports
```

For a snapshot at date `t`, features use transactions strictly before `t`; targets use `(t, t + 90 days]`. Customers may appear in multiple snapshots. The current configured dates are March 1, April 1, May 1, and June 1, 2011, yielding 8,968 customer-snapshot rows (not 8,968 distinct customers). Training uses March and April, May is validation, and June is the final chronological test period. Walk-forward evaluation supplements this split. Random row splitting is avoided because the same customer can occur in several periods.

Cleaning retains negative-quantity adjustments rather than deleting them. Cancellation rows are identified by a `C`-prefixed invoice number; other negative quantities are treated as returns. Their signed line values contribute to net spend, but cancellations are not matched to original invoices. Purchase behavior counts positive, non-cancellation lines. This is an explicit accounting assumption, not a reconstructed refund ledger.

## Models and evaluation

The churn comparison includes a majority-class baseline, RFM-only logistic regression, a deterministic RFM rule baseline, and LightGBM using the engineered feature set. The LightGBM churn probability can be sigmoid-calibrated using the validation period. A separate LightGBM regressor predicts future 90-day net spend, and BG/NBD + Gamma-Gamma provides a classical CLV benchmark.

Evaluation emphasizes PR-AUC, Lift@Decile 1, and Brier score rather than accuracy alone. SHAP provides global and selected local explanations. The risk-weighted ranking is:

```text
risk_weighted_value = calibrated_churn_probability × predicted_90d_value
```

It is a prioritization or value-at-risk score, not guaranteed savings. Optional offer scenarios use stated assumptions for acceptance, retained value, and cost; they are hypothetical and do not estimate intervention effects.

### Walk-forward churn results

The table below summarizes the two documented expanding-window validation folds. Probabilities are uncalibrated for all listed models; the rule baseline uses its deterministic score mapping. These values are a small robustness check, not the June final-test results or a claim of future performance.

| Model | Mean PR-AUC | Mean Lift@Decile 1 | Mean Brier score |
|---|---:|---:|---:|
| Majority baseline | 0.459 | 1.000 | 0.249 |
| Logistic regression (RFM) | 0.654 | 1.597 | 0.209 |
| Rule-based RFM | 0.514 | 1.149 | 0.331 |
| LightGBM | 0.718 | 1.734 | 0.188 |

There are only two walk-forward folds, and adjacent 90-day target windows overlap. See [`docs/walk_forward.md`](docs/walk_forward.md) for fold details and interpretation limits.

## Technology

- **Data and modeling:** Python, pandas, NumPy, scikit-learn, LightGBM, SciPy, lifetimes
- **Interpretability and tracking:** SHAP, MLflow
- **Serving:** FastAPI, Pydantic, Uvicorn
- **Web app:** React, TypeScript, Vite, TanStack Query, Recharts
- **Operations:** Docker, Docker Compose, GitHub Actions, pytest
- **Reporting and monitoring:** Power BI exports and a manual PSI utility

## Repository layout

```text
src/churnguard/       Cleaning, snapshots, features, labels, models, evaluation,
                      tracking, API, reporting, and PSI implementation
tests/                Python tests for pipeline, leakage, modeling, serving, and tools
frontend/             React + TypeScript scoring and analytics application
docs/                 API, Docker, CI, MLflow, SHAP, PSI, targeting, and Power BI guides
.github/workflows/    GitHub Actions CI workflow
Dockerfile            API inference image
docker-compose.yml    Local API service with external frozen model artifacts
pyproject.toml        Package and optional dependency configuration
```

## Setup and use

### 1. Prepare the data

Download the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online%2Bretail) and save/export its workbook as a CSV file. The pipeline reads CSV input; its default encoding is `latin1`. Keep the dataset local and place it somewhere such as `data/online_retail.csv` (the data directory is not required to be committed).

### 2. Install dependencies

Python 3.11 is used by CI. From the repository root:

```bash
python -m pip install -e ".[analysis,test]"
```

The `analysis` extra includes the CLV benchmark, SHAP, MLflow, and plotting dependencies; `test` installs pytest. Serving dependencies are declared separately in the base project requirements used by Docker.

### 3. Build a snapshot table

```python
from pathlib import Path

from churnguard.config import PipelineConfig
from churnguard.pipeline import run_pipeline

config = PipelineConfig(
    data_path=Path("data/online_retail.csv"),
    snapshot_dates=("2011-03-01", "2011-04-01", "2011-05-01", "2011-06-01"),
    horizon_days=90,
    encoding="latin1",
)
customer_snapshots = run_pipeline(config)
```

Snapshot dates, horizon, feature windows, and data path are configurable in Python. The temporal split and model parameters are defined in `src/churnguard/model_config.py`.

### 4. Train models and export artifacts

The API only serves frozen models — it never trains them. Fit the baselines and LightGBM models on the snapshot table, then export the fitted models and schema metadata to `models/` so the API (or Docker container) can load them:

```python
from churnguard.artifacts import export_inference_artifacts
from churnguard.experiment import evaluate_models
from churnguard.model_config import ModelConfig, TemporalSplitConfig

split_config = TemporalSplitConfig()
model_config = ModelConfig()
evaluation_results = evaluate_models(customer_snapshots, split_config, model_config)

export_inference_artifacts(
    evaluation_results, split_config, model_config, artifact_dir="models"
)
```

This writes `models/churn/model.joblib`, `models/future_value/model.joblib`, and `models/metadata.json`. `models/` is gitignored on purpose — it's a build output, not source code — so this step must be run at least once before `GET /health` will report `status: ok`. See [`docs/docker.md`](docs/docker.md) for the full explanation and Docker-specific notes.

### 5. Run checks

```bash
python -m pytest -q
python -m compileall -q src tests
```

Frontend checks are run from `frontend/`:

```bash
npm ci
npm test
npm run build
```

The GitHub Actions workflow runs Python tests, compilation, whitespace checks, Docker Compose configuration validation, and a Docker image build on pull requests and pushes to its configured branch. It does not train models, download data, or deploy.

## Scoring API and frontend

The FastAPI service exposes `GET /health`, `POST /predict`, and `POST /batch_score`. Requests contain the exact feature schema recorded with the frozen model; customer ID and snapshot date are metadata, not model inputs. Responses include calibrated churn probability, predicted future 90-day net spend, and risk-weighted value; batch responses also include within-snapshot ranks. See [`docs/api.md`](docs/api.md) for schemas and artifact generation.

Frozen files live under the gitignored `models/` directory and are supplied at runtime. The container does not contain model files and does not train or create replacements. Start the local API after valid artifacts are available:

```bash
docker compose up --build -d
```

Example request once the API is running (feature names/values must match `models/metadata.json` for your trained artifacts):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "CustomerID": 9001,
        "snapshot_date": "2011-06-01",
        "features": { "...": "see models/metadata.json for the exact feature schema" }
      }'
```

```json
{
  "CustomerID": 9001,
  "snapshot_date": "2011-06-01T00:00:00",
  "churn_probability": 0.42,
  "predicted_90d_value": 118.30,
  "risk_weighted_value": 49.69
}
```

The React app in `frontend/` is a separate local development server that calls the API. See [`docs/frontend.md`](docs/frontend.md) for startup instructions. This repository provides local serving/container foundations; it does not currently deploy a public endpoint.

## Explainability, tracking, drift, and reporting

- **SHAP:** Global and selected local explanations for the churn model. Values are in raw-margin/log-odds space and describe model behavior, not causality.
- **MLflow:** Local experiment tracking for churn, future value, and optional targeting analysis. It is not a hosted tracking service.
- **PSI:** A manual comparison between the frozen model's training-feature population and a supplied scoring batch. PSI above 0.20 is a review/retraining signal, not proof of model inaccuracy; no automatic retraining occurs. See [`docs/psi_drift.md`](docs/psi_drift.md).
- **Power BI:** The project exports scored customer snapshots and global feature importance with report measures/instructions. The Power BI Desktop report itself still requires rebuilding and visual verification; see [`docs/powerbi.md`](docs/powerbi.md).

## Limitations

- One public retailer dataset, about one year of history, and four configured snapshots.
- Only two walk-forward folds; the adjacent 90-day target windows overlap.
- Cancellation lines are not matched to original purchases.
- No causal/uplift model, observed campaign response, or verified business-impact metric.
- API authentication, public deployment, hosted tracking, automated alerts, and automatic retraining are outside the current implementation.

All reported model results describe the stated historical splits only. They should not be generalized to other retailers or interpreted as realized business impact.

## License

Released under the [MIT License](LICENSE) — free to use, modify, and share with attribution. Add a `LICENSE` file with the MIT text at the repo root if it isn't there yet (swap this section if you'd rather use a different license).
