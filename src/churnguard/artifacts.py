"""Export and load frozen inference artifacts from the existing evaluation flow."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib

from .model_config import ModelConfig, TemporalSplitConfig

ARTIFACT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class InferenceArtifacts:
    churn_model: object
    calibrator: object
    future_value_model: object
    metadata: dict


def export_inference_artifacts(
    evaluation_results: dict,
    split_config: TemporalSplitConfig | None = None,
    model_config: ModelConfig | None = None,
    artifact_dir: str | Path = "models",
) -> Path:
    """Persist fitted models and train-only schema metadata, without row data.

    Pass the completed output of the normal evaluation workflow. June/test labels
    and prediction tables are not read or included in the artifact.
    """
    split_config = split_config or TemporalSplitConfig()
    model_config = model_config or ModelConfig()
    models = evaluation_results.get("models", {})
    value_results = evaluation_results.get("value_results", {})
    value_models = value_results.get("models", {})
    required_models = (models.get("lightgbm"), models.get("lightgbm_calibrator"), value_models.get("lightgbm_value"))
    if any(model is None for model in required_models):
        raise ValueError("Evaluation results must contain fitted churn, calibrator, and supervised value models")
    churn_schema = evaluation_results.get("lightgbm_preprocessing")
    value_schema = value_results.get("lightgbm_preprocessing")
    if not churn_schema or not value_schema:
        raise ValueError("Evaluation results lack train-only inference preprocessing metadata")
    if churn_schema["feature_columns"] != value_schema["feature_columns"]:
        raise ValueError("Churn and future-value models have incompatible feature schemas")
    target_bounds = value_results.get("training_transformed_target_bounds")
    if not target_bounds or len(target_bounds) != 2:
        raise ValueError("Evaluation results lack the training-only target transform bounds")

    metadata = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "feature_set_identifier": "churnguard_engineered_features_v1",
        "model_types": {
            "churn": "lightgbm_binary_classifier",
            "calibrator": "sigmoid_platt",
            "future_value": "lightgbm_signed_log1p_regressor",
        },
        "feature_columns": churn_schema["feature_columns"],
        "churn_preprocessing": churn_schema,
        "future_value_preprocessing": value_schema,
        "future_value_training_transformed_target_bounds": target_bounds,
        "target_columns": {"churn": split_config.target_column, "future_value": "future_net_spend_90d"},
        "training_snapshot_dates": list(split_config.train_dates),
        "validation_snapshot_dates": list(split_config.validation_dates),
        "random_seed": model_config.random_seed,
    }
    destination = Path(artifact_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    bundle = {
        "churn_model": required_models[0],
        "calibrator": required_models[1],
        "future_value_model": required_models[2],
    }
    (destination / "churn").mkdir(exist_ok=True)
    (destination / "future_value").mkdir(exist_ok=True)
    # Write directly into the artifact directories. On Windows, moving files
    # from private temporary directories can carry restrictive ACLs into the
    # read-only Docker bind mount, making otherwise valid artifacts unreadable.
    joblib.dump({"churn_model": bundle["churn_model"], "calibrator": bundle["calibrator"]},
                destination / "churn" / "model.joblib")
    joblib.dump({"future_value_model": bundle["future_value_model"]},
                destination / "future_value" / "model.joblib")
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return destination


def load_inference_artifacts(artifact_dir: str | Path) -> InferenceArtifacts:
    """Load frozen artifacts; this function never fits or trains a model."""
    directory = Path(artifact_dir)
    churn_path = directory / "churn" / "model.joblib"
    value_path = directory / "future_value" / "model.joblib"
    metadata_path = directory / "metadata.json"
    missing = [str(path.relative_to(directory)) for path in (churn_path, value_path, metadata_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required model artifact files are missing: {', '.join(missing)}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError("Unsupported model artifact format version")
    churn_bundle = joblib.load(churn_path)
    value_bundle = joblib.load(value_path)
    if not {"churn_model", "calibrator"}.issubset(churn_bundle) or "future_value_model" not in value_bundle:
        raise ValueError("Model artifact bundle is missing required inference models")
    return InferenceArtifacts(
        churn_model=churn_bundle["churn_model"], calibrator=churn_bundle["calibrator"],
        future_value_model=value_bundle["future_value_model"], metadata=metadata,
    )
