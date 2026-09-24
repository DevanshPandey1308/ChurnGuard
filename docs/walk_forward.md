# ChurnGuard temporal walk-forward evaluation

## Purpose and separation from the final test

Customers appear in multiple point-in-time snapshots. Random row splitting can
put one customer's later observation in training while an earlier observation
is evaluated, so this module measures robustness by fitting each model only on
snapshots strictly earlier than its validation snapshot.

Walk-forward validation measures model robustness across earlier chronological
validation periods. The latest observed snapshot is reserved as the final test
period and is not included in any walk-forward training or validation fold. In
the current ChurnGuard dataset that final period is June 2011. The existing
fixed March/April → May validation → June final test workflow remains unchanged.
This module does not tune models or thresholds using June.

## Current folds

The four available snapshot dates are:

- 2011-03-01
- 2011-04-01
- 2011-05-01
- 2011-06-01 (reserved final test)

Every feasible expanding-window fold before the latest date is:

| Fold | Training snapshot dates | Train rows | Validation date | Validation rows |
|---|---|---:|---|---:|
| wf_01_2011-04-01 | 2011-03-01 | 1,682 | 2011-04-01 | 2,134 |
| wf_02_2011-05-01 | 2011-03-01, 2011-04-01 | 3,816 | 2011-05-01 | 2,434 |

Fold dates and rows are discovered from the supplied snapshot feature table.
With only four snapshots, this yields two walk-forward folds; it is not
extensive cross-validation. June is observed only as the latest date to reserve
it and does not enter model fitting or validation.

## Measured walk-forward results

Results below are from the repository's current dataset and the existing model
configuration. Probabilities are raw for all three probabilistic approaches;
the rule baseline's fixed probabilities follow its existing deterministic RFM
logic.

| Fold | Model | Validation churn rate | PR-AUC | Lift@Decile 1 | Brier score |
|---|---|---:|---:|---:|---:|
| wf_01_2011-04-01 | Majority | 0.458 | 0.458 | 1.000 | 0.249 |
| wf_01_2011-04-01 | Logistic RFM | 0.458 | 0.644 | 1.602 | 0.211 |
| wf_01_2011-04-01 | Rule-based RFM | 0.458 | 0.502 | 1.140 | 0.339 |
| wf_01_2011-04-01 | LightGBM | 0.458 | 0.708 | 1.735 | 0.193 |
| wf_02_2011-05-01 | Majority | 0.461 | 0.461 | 1.000 | 0.249 |
| wf_02_2011-05-01 | Logistic RFM | 0.461 | 0.663 | 1.591 | 0.207 |
| wf_02_2011-05-01 | Rule-based RFM | 0.461 | 0.527 | 1.157 | 0.322 |
| wf_02_2011-05-01 | LightGBM | 0.461 | 0.728 | 1.734 | 0.182 |


| Model | Folds | Mean PR-AUC | SD PR-AUC | Mean lift | SD lift | Mean Brier | SD Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority | 2 | 0.459 | 0.002 | 1.000 | 0.000 | 0.249 | 0.000 |
| Logistic RFM | 2 | 0.654 | 0.014 | 1.597 | 0.008 | 0.209 | 0.002 |
| Rule-based RFM | 2 | 0.514 | 0.018 | 1.149 | 0.013 | 0.331 | 0.012 |
| LightGBM | 2 | 0.718 | 0.014 | 1.734 | 0.001 | 0.188 | 0.008 |

These are measurements across two folds, not a model ranking or winner
selection. Exact values are returned without rounding by the evaluator.

## Models and metrics

The evaluator reuses the existing model constructors and feature preparation
for majority, logistic RFM, rule-based RFM, and LightGBM churn models. RFM rule
quantile cuts are fit on each fold's training rows only. The majority baseline
uses the training churn prevalence as a constant probability. Results include
fold dates, row counts, validation churn rate, PR-AUC (average precision),
Lift@Decile 1, and Brier score. The aggregate table reports means and sample
standard deviations across folds as measurements, not as a winner ranking.

Walk-forward LightGBM metrics use raw probabilities. The existing sigmoid
calibration is not refit or applied across folds: that calibrator was fitted on
the May validation snapshot in the existing workflow, with June preserved for
the final test. Calibration and June final-test results remain a separate
evaluation stage.

## Temporal dependence and interpretation

Each target covers the next 90 days while snapshots are one month apart. Thus,
label windows from neighboring snapshots overlap in calendar time. Training and
validation observations are chronologically separated by snapshot date but
their label windows can still share future transaction periods. These folds are
not fully purged or independent cross-validation.

Two validation months provide only a small view of time variation. The first
fold also has a shorter training history than the second. Changes in metrics
can reflect both that training-history difference and month-to-month cohort or
label variation. Treat the summary as an early robustness check, not a precise
estimate of future performance. The separately reported June final test
remains the one final chronological holdout and must not be used for model
selection.

## Usage

```python
from churnguard import PipelineConfig, build_training_table, load_transactions
from churnguard.model_config import ModelConfig
from churnguard.walk_forward import evaluate_walk_forward

transactions = load_transactions(config.data_path, config.encoding)
snapshots = build_training_table(transactions, config)
walk_forward = evaluate_walk_forward(snapshots, ModelConfig())
fold_metrics = walk_forward["results"]
aggregate_metrics = walk_forward["aggregate"]
```
