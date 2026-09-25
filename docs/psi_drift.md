# ChurnGuard PSI drift check

Population Stability Index (PSI) compares a feature's distribution in a
reference population with its distribution in a current population. It is a
manual, lightweight drift signal; it does not measure predictive performance
and does not prove that a model is inaccurate.

## Populations and feature contract

The reference is reconstructed from the raw transactions used for the frozen
production model. The check reads `training_snapshot_dates` and the ordered
feature/type contract from `models/metadata.json`, cleans transactions with the
existing pipeline, and rebuilds only those point-in-time training snapshots.
For the current artifacts, the training snapshots are March and April 2011. June
is the final test snapshot and is never used as the reference. Rebuild from the
same source data/version used to train the model for a faithful comparison.

The current population is the latest scoring batch supplied by the operator.
It must be a flat CSV containing the model's input feature columns (the same
inputs sent to `/batch_score`). `CustomerID`, `snapshot_date`, labels, prediction
outputs, expected value, and ranks are never included because the comparison
uses only the feature names listed in the frozen model contract. Optional
feature selection can narrow this set, but cannot add fields outside the model
contract. Missing contract features fail with an explicit error.

## Calculation

Numeric features use up to ten quantile bins whose boundaries are calculated
from the reference values only. Tied boundaries are collapsed; out-of-range
current values land in the corresponding outer bin. Missing values occupy a
separate bin. Categorical features use the sorted reference category vocabulary;
missing and categories unseen in reference each have their own buckets. A
configurable epsilon (default `1e-6`) is added to every bucket before
normalization so a zero observed proportion does not make PSI infinite. For
reference rows, the check also repeats numeric missing-value imputation using
the frozen model's training-only medians, matching the feature preparation used
when the model was fitted. The current CSV represents the latest scoring inputs;
the API itself rejects null features before scoring.

Each result includes the numeric PSI, `stable`/`drift` status, and a boolean
`drift_flag`. The default trigger threshold is `0.20`; a feature is flagged only
when PSI is strictly greater than the configured threshold. A crossing is a
prompt to investigate distribution changes and consider retraining. It is not
proof of model degradation. ChurnGuard does not automatically retrain, alert,
or operate a full monitoring platform.

## Manual execution

From the repository root, provide the same raw source data used to train the
frozen model plus a current batch feature CSV:

```powershell
python -m churnguard.monitoring.psi_drift_check `
  --reference-transactions path/to/online_retail.csv `
  --current-batch path/to/latest_scoring_features.csv `
  --model-artifacts models `
  --output artifacts/monitoring/psi_latest.csv
```

Set `PYTHONPATH=src` when running from a source checkout that has not installed
the package (PowerShell: `$env:PYTHONPATH = "src"`). Use `--help` to set the
threshold, epsilon, numeric bin count, encoding, observation horizon, or repeat
`--feature` to select a subset of model inputs. The raw transaction source must
cover the training snapshots and their full observation horizons. The command
prints the per-feature report and whether any feature crossed the threshold;
the optional CSV records the same structured results.
