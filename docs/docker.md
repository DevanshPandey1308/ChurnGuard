# ChurnGuard Docker and Compose

Docker packages the FastAPI inference application in a reproducible Python
runtime. The container serves frozen models; it does not train or recalibrate
them. This is a local container workflow, not a completed deployment, CI/CD, or
monitoring setup.

## Container contents

The image contains Python, the ChurnGuard package and its declared runtime
dependencies, Uvicorn, and the small OS OpenMP runtime needed by LightGBM. It
runs as an unprivileged user with a read-only root filesystem.

The image intentionally excludes raw datasets, notebooks, tests, Power BI
files, local MLflow databases/artifacts, generated SHAP artifacts, secrets,
and model files. `.dockerignore` enforces those build-context exclusions. No
Kubernetes, workflow scheduler, database, queue, or monitoring stack is added.

## Frozen model artifacts

Compose mounts the repository's `models/` directory read-only at
`/app/models`; `CHURNGUARD_MODEL_DIR=/app/models` tells the existing loader
where to load `churn/model.joblib`, `future_value/model.joblib`, and
`metadata.json`. The host path is relative (`./models`), not tied to a
developer's machine. The models are not copied into or built into the image.

Create or refresh artifacts explicitly from the existing fitted evaluation
workflow, outside the container. For example, from the repository root after
installing the project dependencies:

```python
from pathlib import Path

from churnguard.artifacts import export_inference_artifacts
from churnguard.config import PipelineConfig
from churnguard.experiment import evaluate_models
from churnguard.model_config import ModelConfig, TemporalSplitConfig
from churnguard.pipeline import run_pipeline

split_config = TemporalSplitConfig()
dates = (*split_config.train_dates, *split_config.validation_dates, *split_config.test_dates)
table = run_pipeline(PipelineConfig(data_path=Path("data[1].csv"), snapshot_dates=dates))
model_config = ModelConfig()
evaluation_results = evaluate_models(table, split_config, model_config)
export_inference_artifacts(
    evaluation_results, split_config, model_config, artifact_dir="models"
)
```

For the full offline evaluation/analysis workflow, install the optional
analysis dependencies first:

```powershell
pip install -e ".[analysis]"
```

Artifact export saves only fitted model objects and schema/transform metadata;
it does not serialize customer records, predictions, or June labels. The
existing workflow fits churn on its training snapshots, the sigmoid calibrator
on validation predictions/labels, and the supervised value model on training
labels. June remains evaluation-only. The container and API do not run this
workflow. If `models/` or any required artifact is absent/invalid, the API
returns `503` and its healthcheck remains unhealthy; no substitute model is
created.

After refreshing artifacts, recreate the service so it loads the new frozen
objects at process startup. Do not overwrite artifacts while a running process
is serving requests.

## Build and start

Ensure the three required artifacts are present under `models/`, then run:

```powershell
docker compose up --build -d
docker compose ps
```

Compose builds `churnguard-api:local` from the serving dependencies in
`pyproject.toml` (it does not install the optional SHAP, MLflow, CLV benchmark,
or plotting dependencies), starts one API service, maps host port
8000 to container port 8000, and mounts model files read-only. Set
`CHURNGUARD_API_PORT` to choose another host port. The container's command is
Uvicorn bound to `0.0.0.0:8000` without development reload. The image and
Compose healthcheck request `GET /health` every 10 seconds and require all
three loaded flags, with a 20-second startup grace period. Compose reports
healthy only after artifacts load successfully.

Check health from PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

## Call the scoring routes

The feature names and order are read from `models/metadata.json`. The following
PowerShell example creates a schema-complete sample using the saved training
medians and first known category. Replace values with a real historical
feature record for meaningful scoring:

```powershell
$schema = Get-Content models/metadata.json -Raw | ConvertFrom-Json
$features = @{}
foreach ($name in $schema.feature_columns) {
    $median = $schema.churn_preprocessing.numeric_medians.PSObject.Properties[$name]
    if ($null -ne $median) {
        $features[$name] = $median.Value
    } else {
        $features[$name] = $schema.churn_preprocessing.categorical_values.PSObject.Properties[$name].Value[0]
    }
}
$customer = @{ CustomerID = 9001; snapshot_date = "2011-06-01"; features = $features }
$json = $customer | ConvertTo-Json -Depth 6
Invoke-RestMethod http://localhost:8000/predict -Method Post -ContentType "application/json" -Body $json
```

`POST /predict` returns the calibrated churn probability, supervised predicted
90-day future net spend, and `risk_weighted_value`, their product. The score is
a value-at-risk prioritization score, not guaranteed revenue saved.

For `POST /batch_score`, send multiple records in a `customers` array. Example
using the record built above:

```powershell
$second = @{ CustomerID = 9002; snapshot_date = "2011-06-01"; features = $features }
$batch = @{ customers = @($customer, $second) } | ConvertTo-Json -Depth 8
Invoke-RestMethod http://localhost:8000/batch_score -Method Post -ContentType "application/json" -Body $batch
```

The response contains scored customers, a count, and deterministic churn,
value, and risk-value ranks. Both routes use the frozen artifacts; neither
reads future transactions or labels.

## Stop and limitations

Stop the local service with:

```powershell
docker compose down
```

This Compose setup is for local development and reproducibility. Its bind mount
requires a host-managed artifact directory and does not define how artifacts
are distributed, versioned, or secured in a future deployment. A deployment
phase will need an explicit artifact distribution and refresh strategy. Docker
does not complete deployment, CI/CD, authentication, or monitoring.
