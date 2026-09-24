"""Local MLflow tracking for existing churn, value, walk-forward, and targeting outputs.

This module only logs completed results; it never fits models or changes
predictions. Customer-level rows and local SHAP explanation files are not logged.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .model_config import ModelConfig, TemporalSplitConfig


@dataclass(frozen=True)
class TrackingConfig:
    """Local-file tracking settings with environment overrides.

    `CHURNGUARD_MLFLOW_TRACKING_URI` takes precedence; otherwise the URI is a
    local SQLite store rooted at `CHURNGUARD_MLFLOW_DIR` or `./mlruns`. Local
    artifacts are written under that directory as well.
    """

    tracking_uri: str | None = None
    local_tracking_dir: str = "mlruns"
    churn_experiment_name: str = "ChurnGuard-Churn"
    value_experiment_name: str = "ChurnGuard-Future-Value"
    targeting_experiment_name: str = "ChurnGuard-Targeting-Analytics"

    def __post_init__(self) -> None:
        if not self.churn_experiment_name or not self.value_experiment_name or not self.targeting_experiment_name:
            raise ValueError("MLflow experiment names must be non-empty")
        if self.tracking_uri is None:
            database_path = Path(self.local_tracking_dir).resolve() / "mlflow.db"
            uri = f"sqlite:///{database_path.as_posix()}"
            object.__setattr__(self, "tracking_uri", uri)

    @classmethod
    def from_env(cls) -> "TrackingConfig":
        """Build config from CHURNGUARD_MLFLOW_* environment variables."""
        local_dir = os.environ.get("CHURNGUARD_MLFLOW_DIR", "mlruns")
        tracking_uri = os.environ.get("CHURNGUARD_MLFLOW_TRACKING_URI")
        return cls(
            tracking_uri=tracking_uri,
            local_tracking_dir=local_dir,
            churn_experiment_name=os.environ.get("CHURNGUARD_MLFLOW_CHURN_EXPERIMENT", "ChurnGuard-Churn"),
            value_experiment_name=os.environ.get("CHURNGUARD_MLFLOW_VALUE_EXPERIMENT", "ChurnGuard-Future-Value"),
            targeting_experiment_name=os.environ.get("CHURNGUARD_MLFLOW_TARGETING_EXPERIMENT", "ChurnGuard-Targeting-Analytics"),
        )


def _metric(mlflow, name: str, value) -> None:
    number = float(value)
    if np.isfinite(number):
        mlflow.log_metric(name, number)


def _param(mlflow, name: str, value) -> None:
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, sort_keys=True, default=str)
    mlflow.log_param(name, str(value))


def _get_or_create_experiment(mlflow, name: str, artifact_location: str | None = None) -> str:
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        if artifact_location is None:
            return mlflow.create_experiment(name)
        return mlflow.create_experiment(name, artifact_location=artifact_location)
    return experiment.experiment_id


def _log_walk_forward(mlflow, walk_forward_results: dict) -> None:
    """Log the existing LightGBM walk-forward summary and aggregate tables."""
    fold_frame = walk_forward_results["results"].copy()
    temporary_artifact_dir = tempfile.TemporaryDirectory(prefix="churnguard-mlflow-")
    try:
        artifact_dir = Path(temporary_artifact_dir.name)
        fold_csv = artifact_dir / "walk_forward_metrics.csv"
        aggregate_csv = artifact_dir / "walk_forward_aggregate.csv"
        fold_frame.to_csv(fold_csv, index=False)
        walk_forward_results["aggregate"].to_csv(aggregate_csv, index=False)
        for ordinal, (_, row) in enumerate(
            fold_frame.loc[fold_frame["model"].eq("lightgbm")].sort_values("validation_date").iterrows(), start=1
        ):
            _metric(mlflow, f"wf_{ordinal:02d}_pr_auc", row["pr_auc"])
            _metric(mlflow, f"wf_{ordinal:02d}_lift", row["lift"])
            _metric(mlflow, f"wf_{ordinal:02d}_brier", row["brier_score"])
        lightgbm_aggregate = walk_forward_results["aggregate"].loc[
            walk_forward_results["aggregate"]["model"].eq("lightgbm")
        ]
        if len(lightgbm_aggregate) != 1:
            raise ValueError("Walk-forward results must include one LightGBM aggregate row")
        aggregate_row = lightgbm_aggregate.iloc[0]
        for output_name, column in (
            ("walk_forward_mean_pr_auc", "mean_pr_auc"),
            ("walk_forward_std_pr_auc", "std_pr_auc"),
            ("walk_forward_mean_lift", "mean_lift"),
            ("walk_forward_std_lift", "std_lift"),
            ("walk_forward_mean_brier", "mean_brier_score"),
            ("walk_forward_std_brier", "std_brier_score"),
        ):
            _metric(mlflow, output_name, aggregate_row[column])
        mlflow.set_tag("walk_forward_final_test_excluded", "true")
        mlflow.log_artifact(str(fold_csv), artifact_path="walk_forward")
        mlflow.log_artifact(str(aggregate_csv), artifact_path="walk_forward")
    finally:
        temporary_artifact_dir.cleanup()


def track_experiments(
    evaluation_results: dict,
    walk_forward_results: dict,
    tracking_config: TrackingConfig | None = None,
    split_config: TemporalSplitConfig | None = None,
    model_config: ModelConfig | None = None,
) -> dict:
    """Log completed evaluation outputs in separate local MLflow experiments.

    Five individual churn runs are produced, plus separate future-value and
    targeting runs. No fitting, prediction, customer-level artifact, raw
    transaction, or local SHAP artifact is passed to MLflow.
    """
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("MLflow is required; install the project dependencies before tracking") from exc

    tracking_config = tracking_config or TrackingConfig.from_env()
    split_config = split_config or TemporalSplitConfig()
    model_config = model_config or ModelConfig()
    mlflow.set_tracking_uri(tracking_config.tracking_uri)

    run_records = {}

    local_artifact_root = Path(tracking_config.local_tracking_dir).resolve() / "artifacts"
    if tracking_config.tracking_uri and tracking_config.tracking_uri.startswith("sqlite:"):
        local_artifact_root.mkdir(parents=True, exist_ok=True)

    def artifact_location(name: str) -> str | None:
        if tracking_config.tracking_uri and tracking_config.tracking_uri.startswith("sqlite:"):
            safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
            return (local_artifact_root / safe_name).as_uri()
        return None

    churn_experiment_id = _get_or_create_experiment(
        mlflow, tracking_config.churn_experiment_name,
        artifact_location(tracking_config.churn_experiment_name),
    )
    churn_models = (
        ("majority_baseline", "majority_class", "Majority-class baseline", "no_features"),
        ("logistic_rfm", "logistic_rfm", "Logistic regression RFM", "rfm_core_v1"),
        ("rule_based_rfm", "rule_based_rfm", "Rule-based RFM baseline", "rfm_core_v1"),
        ("lightgbm_raw", "lightgbm", "LightGBM churn classifier (raw)", "churnguard_engineered_features_v1"),
    )
    run_records["churn"] = {}
    for run_name, result_key, model_name, feature_set in churn_models:
        with mlflow.start_run(experiment_id=churn_experiment_id, run_name=run_name) as run:
            params = {
                "model_name": model_name,
                "model_type": "binary_classification",
                "feature_set_identifier": feature_set,
                "training_snapshot_dates": tuple(split_config.train_dates),
                "validation_snapshot_dates": tuple(split_config.validation_dates),
                "final_test_snapshot_dates": tuple(split_config.test_dates),
                "target_column": split_config.target_column,
            }
            if result_key in ("lightgbm", "logistic_rfm"):
                params["random_seed"] = model_config.random_seed
            if result_key == "lightgbm":
                params["probability_output"] = "raw"
                params.update({f"lightgbm_{key}": value for key, value in model_config.lightgbm_params.items()})
            for key, value in params.items():
                _param(mlflow, key, value)
            if result_key == "lightgbm":
                _param(mlflow, "logistic_feature_set", "not_applicable")

            churn_metrics = evaluation_results["results"][result_key]
            for partition, prefix in (("validation", "validation"), ("test", "final_test")):
                metrics = churn_metrics[partition]
                _metric(mlflow, f"{prefix}_pr_auc", metrics["pr_auc"])
                _metric(mlflow, f"{prefix}_lift_at_decile_1", metrics["lift_at_decile_1"])
                # All model probabilities are scored once by evaluate_models;
                # Brier values are persisted there and are never recomputed here.
                if "brier_score" in metrics:
                    _metric(mlflow, f"{prefix}_brier", metrics["brier_score"])

            # Keep the established walk-forward evidence attached to the raw
            # LightGBM run; it was computed independently of this logger.
            if result_key == "lightgbm":
                _log_walk_forward(mlflow, walk_forward_results)
            run_records["churn"][run_name] = {
                "experiment_name": tracking_config.churn_experiment_name,
                "run_id": run.info.run_id,
            }

    # The calibrated run stands on its own and reports both raw and calibrated
    # probabilities for validation and final test, with no test-based choice.
    with mlflow.start_run(experiment_id=churn_experiment_id, run_name="lightgbm_calibrated") as run:
        for key, value in {
            "model_name": "LightGBM churn classifier (sigmoid calibrated)",
            "model_type": "binary_classification",
            "feature_set_identifier": "churnguard_engineered_features_v1",
            "training_snapshot_dates": tuple(split_config.train_dates),
            "validation_snapshot_dates": tuple(split_config.validation_dates),
            "final_test_snapshot_dates": tuple(split_config.test_dates),
            "random_seed": model_config.random_seed,
            "target_column": split_config.target_column,
            "calibration_method": "sigmoid fitted on validation predictions",
        }.items():
            _param(mlflow, key, value)
        _param(mlflow, "probability_output_for_walk_forward", "raw")
        _param(mlflow, "lightgbm_params", model_config.lightgbm_params)
        for partition, prefix in (("validation", "validation"), ("test", "final_test")):
            for kind in ("raw", "calibrated"):
                metrics = evaluation_results["calibration_results"][kind][partition]
                _metric(mlflow, f"{kind}_{prefix}_pr_auc", metrics["pr_auc"])
                _metric(mlflow, f"{kind}_{prefix}_lift_at_decile_1", metrics["lift_at_decile_1"])
                _metric(mlflow, f"{kind}_{prefix}_brier", metrics["brier_score"])
        run_records["churn"]["lightgbm_calibrated"] = {
            "experiment_name": tracking_config.churn_experiment_name,
            "run_id": run.info.run_id,
        }

    # Walk-forward artifact logging is isolated in a helper so only the raw
    # LightGBM run receives the existing fold and aggregate outputs.
    # (See helper definition below.)

    value_experiment_id = _get_or_create_experiment(
        mlflow, tracking_config.value_experiment_name,
        artifact_location(tracking_config.value_experiment_name),
    )
    with mlflow.start_run(experiment_id=value_experiment_id, run_name="lightgbm-future-net-spend") as run:
        value_config = evaluation_results.get("value_results", {})
        value_metrics = value_config.get("metrics", {}).get("lightgbm_value")
        if value_metrics is None:
            raise ValueError("Evaluation results lack LightGBM future-value metrics")
        _param(mlflow, "model_name", "LightGBM future value regressor")
        _param(mlflow, "model_type", "regression")
        _param(mlflow, "target_column", "future_net_spend_90d")
        _param(mlflow, "feature_set_identifier", "churnguard_engineered_features_v1")
        _param(mlflow, "training_snapshot_dates", tuple(split_config.train_dates))
        _param(mlflow, "validation_snapshot_dates", tuple(split_config.validation_dates))
        _param(mlflow, "final_test_snapshot_dates", tuple(split_config.test_dates))
        value_estimator = value_config.get("models", {}).get("lightgbm_value")
        actual_params = value_estimator.get_params() if value_estimator is not None else {}
        _param(mlflow, "random_seed", actual_params.get("random_state", model_config.random_seed))
        if value_estimator is not None:
            for key in ("n_estimators", "learning_rate", "num_leaves", "max_depth", "min_child_samples",
                        "subsample", "colsample_bytree", "reg_lambda", "objective", "random_state"):
                if key in actual_params:
                    _param(mlflow, f"lightgbm_{key}", actual_params[key])
        for partition, prefix in (("validation", "validation"), ("test", "final_test")):
            for metric_name in ("mae", "rmse", "spearman"):
                _metric(mlflow, f"{prefix}_{metric_name}", value_metrics[partition][metric_name])
        run_records["future_value"] = {
            "experiment_name": tracking_config.value_experiment_name,
            "run_id": run.info.run_id,
        }

    targeting = evaluation_results.get("targeting_results")
    if targeting is not None:
        target_experiment_id = _get_or_create_experiment(
            mlflow, tracking_config.targeting_experiment_name,
            artifact_location(tracking_config.targeting_experiment_name),
        )
        with mlflow.start_run(experiment_id=target_experiment_id, run_name="june-targeting-evaluation") as run:
            assumptions = targeting["configuration"]
            _param(mlflow, "target_fraction", assumptions["top_fraction"])
            _param(mlflow, "retention_offer_cost_assumption", assumptions["retention_offer_cost"])
            _param(mlflow, "assumed_acceptance_rate", assumptions["assumed_acceptance_rate"])
            _param(mlflow, "retained_value_fraction", assumptions["retained_value_fraction"])
            _param(mlflow, "interpretation", "held-out actual future net spend capture, not business impact")
            test_capture = targeting["test"]["top_decile_actual_value_capture"]
            for method, metric_name in (
                ("predicted_90d_value", "june_top10_predicted_value_capture"),
                ("risk_weighted_value", "june_top10_risk_weighted_value_capture"),
            ):
                _metric(mlflow, metric_name, test_capture[method]["percentage_of_total_actual_future_net_spend"])
            risk = targeting["test"]["predictions"]
            if "expected_net_value_under_scenario" in risk:
                import math

                target_count = max(1, math.ceil(len(risk) * float(assumptions["top_fraction"])))
                top_risk = risk.nsmallest(target_count, "risk_value_rank")
                _metric(mlflow, "june_top10_modeled_expected_net_value_under_scenario",
                        top_risk["expected_net_value_under_scenario"].sum())
            mlflow.set_tag("june_metrics_are_held_out_capture_not_impact", "true")
            run_records["targeting"] = {
                "experiment_name": tracking_config.targeting_experiment_name,
                "run_id": run.info.run_id,
            }

    return {"tracking_uri": tracking_config.tracking_uri, "runs": run_records}
