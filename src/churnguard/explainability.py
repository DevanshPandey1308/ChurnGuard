"""SHAP explanations for the existing LightGBM churn classifier.

SHAP values use LightGBM's raw margin (log-odds) output space, not probability
space. The base value plus feature contributions is checked against the model's
raw-score prediction. June outcomes are not accepted by this API.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .modeling import TARGET_DERIVED_COLUMNS, IDENTIFIER_COLUMNS, prepare_lightgbm_features

FEATURE_FORBIDDEN = TARGET_DERIVED_COLUMNS | IDENTIFIER_COLUMNS
LOCAL_METADATA_COLUMNS = [
    "CustomerID", "snapshot_date", "raw_churn_probability", "calibrated_churn_probability",
]


def prepare_explanation_matrix(model, training_features: pd.DataFrame, explanation_features: pd.DataFrame) -> pd.DataFrame:
    """Apply the existing LightGBM preprocessing and enforce fitted column order.

    This deliberately rejects mismatched columns instead of silently
    reordering them. Training numeric imputations and categorical vocabularies
    come from the same `prepare_lightgbm_features` helper used in evaluation.
    """
    train_columns = list(training_features.columns)
    explanation_columns = list(explanation_features.columns)
    if train_columns != explanation_columns:
        raise ValueError("Training and explanation features must have identical column ordering")
    forbidden = FEATURE_FORBIDDEN.intersection(train_columns)
    if forbidden:
        raise ValueError(f"Identifiers/outcomes cannot be included in SHAP features: {sorted(forbidden)}")
    model_columns = list(getattr(model, "feature_name_", []))
    if model_columns and model_columns != train_columns:
        raise ValueError("Feature columns do not match the fitted model's feature ordering")
    _, prepared = prepare_lightgbm_features(training_features, explanation_features)
    if list(prepared.columns) != train_columns:
        raise RuntimeError("Feature preprocessing changed the fitted model feature ordering")
    return prepared


def select_representative_rows(metadata: pd.DataFrame) -> pd.DataFrame:
    """Select deterministic low, median, and high raw-probability observations."""
    missing = set(LOCAL_METADATA_COLUMNS) - set(metadata.columns)
    extras = set(metadata.columns) - set(LOCAL_METADATA_COLUMNS)
    if missing:
        raise ValueError(f"Missing local explanation metadata: {sorted(missing)}")
    if extras:
        raise ValueError(f"Unexpected local explanation metadata: {sorted(extras)}")
    if metadata.empty:
        raise ValueError("At least one observation is required for local explanations")
    frame = metadata.loc[:, LOCAL_METADATA_COLUMNS].copy().reset_index(drop=True)
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
    probability = frame["raw_churn_probability"].to_numpy(dtype=float)
    calibrated = frame["calibrated_churn_probability"].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or not np.isfinite(calibrated).all():
        raise ValueError("Churn probabilities must be finite")
    if ((probability < 0) | (probability > 1)).any() or ((calibrated < 0) | (calibrated > 1)).any():
        raise ValueError("Churn probabilities must be in [0, 1]")
    ordered = frame.sort_values(
        ["raw_churn_probability", "CustomerID", "snapshot_date"],
        ascending=[True, True, True], kind="mergesort",
    )
    positions = ordered.index.to_list()
    choices = (
        ("lowest_raw_probability", positions[0]),
        ("median_raw_probability", positions[(len(positions) - 1) // 2]),
        ("highest_raw_probability", positions[-1]),
    )
    selected = []
    seen = set()
    for label, position in choices:
        if position in seen:
            continue
        seen.add(position)
        row = frame.iloc[position].to_dict()
        row["representative_type"] = label
        row["source_row_position"] = int(position)
        selected.append(row)
    return pd.DataFrame(selected).reset_index(drop=True)


def _positive_class_values(shap_values, n_rows: int, n_features: int) -> np.ndarray:
    """Normalize supported SHAP binary-class return shapes to (rows, features)."""
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        elif len(shap_values) == 1:
            shap_values = shap_values[0]
        else:
            raise ValueError(f"Unexpected SHAP class-list output with {len(shap_values)} entries")
    values = np.asarray(shap_values)
    if values.ndim == 3:
        # Recent SHAP releases append the class/output axis.
        if values.shape[:2] == (n_rows, n_features):
            values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
        elif values.shape[1:] == (n_rows, n_features):
            values = values[1] if values.shape[0] > 1 else values[0]
    if values.shape != (n_rows, n_features):
        raise ValueError(f"Unexpected SHAP values shape {values.shape}; expected {(n_rows, n_features)}")
    return values.astype(float, copy=False)


def _positive_class_base_value(base_value) -> float:
    values = np.asarray(base_value, dtype=float).reshape(-1)
    if not len(values):
        raise ValueError("SHAP returned an empty base value")
    return float(values[-1])


def _local_explanations(
    matrix: pd.DataFrame,
    shap_values: np.ndarray,
    metadata: pd.DataFrame,
    max_features_per_direction: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    representatives = select_representative_rows(metadata)
    records = []
    feature_names = list(matrix.columns)
    for _, representative in representatives.iterrows():
        row_position = int(representative["source_row_position"])
        contribution = shap_values[row_position]
        positive = sorted((idx for idx, value in enumerate(contribution) if value > 0),
                          key=lambda idx: (-contribution[idx], feature_names[idx]))[:max_features_per_direction]
        negative = sorted((idx for idx, value in enumerate(contribution) if value < 0),
                          key=lambda idx: (contribution[idx], feature_names[idx]))[:max_features_per_direction]
        selected_features = sorted(set(positive + negative), key=lambda idx: (-abs(contribution[idx]), feature_names[idx]))
        for rank, feature_idx in enumerate(selected_features, start=1):
            shap_value = float(contribution[feature_idx])
            feature_value = matrix.iloc[row_position, feature_idx]
            if isinstance(feature_value, np.generic):
                feature_value = feature_value.item()
            records.append({
                "CustomerID": representative["CustomerID"],
                "snapshot_date": representative["snapshot_date"],
                "representative_type": representative["representative_type"],
                "raw_churn_probability": float(representative["raw_churn_probability"]),
                "calibrated_churn_probability": float(representative["calibrated_churn_probability"]),
                "feature": feature_names[feature_idx],
                "shap_value": shap_value,
                "feature_value": feature_value,
                "contribution_direction": "positive" if shap_value > 0 else "negative",
                "rank_within_customer": rank,
            })
    return pd.DataFrame(records), representatives


def explain_lightgbm_churn(
    model,
    training_features: pd.DataFrame,
    june_features: pd.DataFrame,
    june_metadata: pd.DataFrame,
    raw_churn_probabilities,
    calibrated_churn_probabilities,
    artifact_dir: str | Path = "artifacts/shap",
    max_features_per_direction: int = 5,
    create_plots: bool = True,
    additivity_tolerance: float = 1e-4,
) -> dict:
    """Explain an already-fitted churn LightGBM model on June feature rows.

    No labels are accepted. SHAP contributions are computed in raw margin
    (log-odds) space. June features are used only for post-hoc explanation of
    the already-fitted model; this function never fits or recalibrates a model.
    """
    if max_features_per_direction < 1:
        raise ValueError("max_features_per_direction must be at least one")
    if set(june_metadata.columns) != set(LOCAL_METADATA_COLUMNS):
        raise ValueError(f"June metadata must contain exactly {LOCAL_METADATA_COLUMNS}")
    metadata = june_metadata.loc[:, LOCAL_METADATA_COLUMNS].copy().reset_index(drop=True)
    if len(metadata) != len(june_features):
        raise ValueError("June metadata and feature rows must have matching lengths")
    raw_probabilities = np.asarray(raw_churn_probabilities, dtype=float).reshape(-1)
    calibrated_probabilities = np.asarray(calibrated_churn_probabilities, dtype=float).reshape(-1)
    if len(raw_probabilities) != len(metadata) or len(calibrated_probabilities) != len(metadata):
        raise ValueError("June probability arrays must align with June metadata rows")
    # Replace metadata probabilities with the supplied, aligned existing model outputs.
    metadata["raw_churn_probability"] = raw_probabilities
    metadata["calibrated_churn_probability"] = calibrated_probabilities

    matrix = prepare_explanation_matrix(model, training_features, june_features)
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(artifact_path / ".mplconfig"))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import shap
    except ImportError as exc:
        raise ImportError("SHAP and Matplotlib are required; install the project dependencies before explaining models") from exc
    explainer = shap.TreeExplainer(model, model_output="raw")
    shap_result = explainer.shap_values(matrix, check_additivity=False)
    values = _positive_class_values(shap_result, len(matrix), matrix.shape[1])
    base_value = _positive_class_base_value(explainer.expected_value)
    raw_margin = np.asarray(model.predict(matrix, raw_score=True), dtype=float).reshape(-1)
    reconstructed = base_value + values.sum(axis=1)
    max_additivity_error = float(np.max(np.abs(reconstructed - raw_margin))) if len(raw_margin) else 0.0
    tolerance = max(additivity_tolerance, additivity_tolerance * float(np.max(np.abs(raw_margin)))) if len(raw_margin) else additivity_tolerance
    if max_additivity_error > tolerance:
        raise RuntimeError(
            f"SHAP raw-margin reconstruction error {max_additivity_error:.6g} exceeds tolerance {tolerance:.6g}"
        )

    importance = pd.DataFrame({
        "feature": list(matrix.columns),
        "mean_abs_shap": np.mean(np.abs(values), axis=0),
        "mean_shap": np.mean(values, axis=0),
    }).sort_values(["mean_abs_shap", "feature"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    importance["rank"] = np.arange(1, len(importance) + 1)
    local, representatives = _local_explanations(matrix, values, metadata, max_features_per_direction)

    importance.to_csv(artifact_path / "global_feature_importance.csv", index=False)
    local.to_csv(artifact_path / "local_explanations.csv", index=False)
    representatives.to_csv(artifact_path / "representative_june_rows.csv", index=False)
    plot_paths = []
    if create_plots:
        import matplotlib.pyplot as plt

        shap.summary_plot(values, features=matrix, feature_names=list(matrix.columns), show=False,
                          max_display=min(20, matrix.shape[1]))
        plt.tight_layout()
        summary_path = artifact_path / "shap_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()
        plot_paths.append(summary_path)

        top = importance.head(min(20, len(importance))).sort_values("mean_abs_shap", ascending=True)
        height = max(4.0, 0.28 * len(top))
        figure, axis = plt.subplots(figsize=(9, height))
        axis.barh(top["feature"], top["mean_abs_shap"])
        axis.set_xlabel("Mean absolute SHAP value (raw margin / log-odds)")
        axis.set_title("June global SHAP importance")
        figure.tight_layout()
        bar_path = artifact_path / "shap_bar.png"
        figure.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        plot_paths.append(bar_path)

    return {
        "global_importance": importance,
        "local_explanations": local,
        "representative_rows": representatives,
        "n_observations_explained": int(len(matrix)),
        "n_features_explained": int(matrix.shape[1]),
        "shap_values_shape": tuple(values.shape),
        "feature_names": list(matrix.columns),
        "output_space": "raw LightGBM margin (log-odds), not probability space",
        "base_value": base_value,
        "additivity_max_abs_error": max_additivity_error,
        "additivity_tolerance": tolerance,
        "additivity_verified": True,
        "artifact_dir": str(artifact_path),
        "plot_paths": [str(path) for path in plot_paths],
    }
