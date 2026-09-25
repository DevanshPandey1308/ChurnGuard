"""Population Stability Index for ChurnGuard model input features.

Run manually with ``python -m churnguard.monitoring.psi_drift_check --help``.
The CLI rebuilds the production model's training snapshot features from the
supplied raw transactions, then compares them with a flat latest-batch CSV.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

NON_FEATURE_COLUMNS = {
    "CustomerID", "snapshot_date", "churn_90d", "repurchased_90d", "future_net_spend_90d",
    "churn_probability", "predicted_90d_value", "predicted_90d_spend", "expected_value",
    "risk_weighted_value", "churn_rank", "value_rank", "risk_value_rank",
}


@dataclass(frozen=True)
class PSIConfig:
    """Deterministic PSI controls; the default trigger is specified by blueprint."""

    threshold: float = 0.20
    epsilon: float = 1e-6
    numeric_bins: int = 10

    def __post_init__(self) -> None:
        if not 0 <= self.threshold:
            raise ValueError("threshold must be non-negative")
        if not 0 < self.epsilon < 1:
            raise ValueError("epsilon must be between 0 and 1")
        if self.numeric_bins < 2:
            raise ValueError("numeric_bins must be at least 2")


def _numeric_values(series: pd.Series, feature: str, population: str) -> np.ndarray:
    converted = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & converted.isna()
    if invalid.any():
        raise ValueError(f"Feature {feature!r} contains non-numeric values in {population} population")
    values = converted.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError(f"Feature {feature!r} contains infinite values in {population} population")
    return values


def _categorical_values(series: pd.Series) -> np.ndarray:
    return np.asarray([None if pd.isna(value) else str(value) for value in series], dtype=object)


def _numeric_distribution(
    reference: np.ndarray, current: np.ndarray, bins: int, epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    observed = reference[~np.isnan(reference)]
    boundaries = np.asarray([], dtype=float)
    if observed.size:
        quantiles = np.linspace(0, 1, bins + 1)[1:-1]
        boundaries = np.unique(np.quantile(observed, quantiles))
        ref_bucket = np.searchsorted(boundaries, reference, side="left")
        cur_bucket = np.searchsorted(boundaries, current, side="left")
    else:
        # A single numeric bucket plus missing keeps all-missing features defined.
        ref_bucket = np.zeros(reference.size, dtype=int)
        cur_bucket = np.zeros(current.size, dtype=int)
    # Missing is a separate, consistent bucket in both populations.
    ref_bucket = np.where(np.isnan(reference), len(boundaries) + 1 if observed.size else 1, ref_bucket)
    cur_bucket = np.where(np.isnan(current), len(boundaries) + 1 if observed.size else 1, cur_bucket)
    bucket_count = (len(boundaries) + 2) if observed.size else 2
    return _smoothed_proportions(ref_bucket, bucket_count, epsilon), _smoothed_proportions(cur_bucket, bucket_count, epsilon)


def _categorical_distribution(
    reference: np.ndarray, current: np.ndarray, epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Vocabulary comes only from reference. Unseen current levels get an explicit bucket.
    categories = sorted({value for value in reference if value is not None})
    positions = {value: index for index, value in enumerate(categories)}
    missing_index = len(categories)
    unseen_index = missing_index + 1
    ref_bucket = np.asarray([missing_index if value is None else positions[value] for value in reference])
    cur_bucket = np.asarray([
        missing_index if value is None else positions.get(value, unseen_index) for value in current
    ])
    return (
        _smoothed_proportions(ref_bucket, len(categories) + 2, epsilon),
        _smoothed_proportions(cur_bucket, len(categories) + 2, epsilon),
    )


def _smoothed_proportions(buckets: np.ndarray, bucket_count: int, epsilon: float) -> np.ndarray:
    counts = np.bincount(np.asarray(buckets, dtype=int), minlength=bucket_count).astype(float)
    return (counts + epsilon) / (counts.sum() + epsilon * bucket_count)


def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    *,
    feature: str,
    feature_type: str,
    config: PSIConfig | None = None,
) -> float:
    """Calculate one feature's PSI using reference-defined bins/categories."""
    config = config or PSIConfig()
    if reference.empty or current.empty:
        raise ValueError("Reference and current populations must both contain rows")
    if feature_type == "number":
        ref_values = _numeric_values(reference, feature, "reference")
        cur_values = _numeric_values(current, feature, "current")
        ref_p, cur_p = _numeric_distribution(ref_values, cur_values, config.numeric_bins, config.epsilon)
    elif feature_type == "category":
        ref_p, cur_p = _categorical_distribution(
            _categorical_values(reference), _categorical_values(current), config.epsilon,
        )
    else:
        raise ValueError(f"Unsupported feature type {feature_type!r} for {feature!r}")
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def compare_feature_populations(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_columns: Sequence[str],
    feature_types: Mapping[str, str],
    config: PSIConfig | None = None,
) -> pd.DataFrame:
    """Return PSI/status for explicit model-contract features only.

    Identifiers, labels, score outputs, and arbitrary columns are ignored unless
    explicitly (and incorrectly) passed as contract features. Required contract
    features missing from either population fail loudly.
    """
    config = config or PSIConfig()
    columns = list(feature_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("feature_columns must be a non-empty sequence without duplicates")
    forbidden = set(columns) & NON_FEATURE_COLUMNS
    if forbidden:
        raise ValueError(f"Non-feature fields cannot be monitored: {sorted(forbidden)}")
    missing_types = set(columns) - set(feature_types)
    if missing_types:
        raise ValueError(f"Missing feature types: {sorted(missing_types)}")
    for population_name, frame in (("reference", reference), ("current", current)):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required {population_name} features: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{population_name.capitalize()} population must contain rows")

    results = []
    for feature in columns:
        value = calculate_psi(
            reference[feature], current[feature], feature=feature,
            feature_type=feature_types[feature], config=config,
        )
        drift = value > config.threshold
        results.append({
            "feature": feature,
            "psi": value,
            "status": "drift" if drift else "stable",
            "drift_flag": drift,
            "threshold": config.threshold,
        })
    return pd.DataFrame(results, columns=["feature", "psi", "status", "drift_flag", "threshold"])


def _load_contract(artifact_dir: Path) -> tuple[dict, list[str], dict[str, str]]:
    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model feature contract not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    schema = metadata.get("churn_preprocessing", {})
    features = metadata.get("feature_columns", [])
    types = schema.get("feature_types", {})
    if not features or schema.get("feature_columns") != features:
        raise ValueError("Model metadata has no consistent churn feature contract")
    if set(features) - set(types):
        raise ValueError("Model metadata is missing feature types")
    training_dates = metadata.get("training_snapshot_dates", [])
    if not training_dates:
        raise ValueError("Model metadata does not identify its training snapshot dates")
    return metadata, list(features), types


def _build_reference_features(
    transactions_path: Path,
    training_dates: Sequence[str],
    feature_columns: Sequence[str],
    horizon_days: int,
    encoding: str,
) -> pd.DataFrame:
    # Import lazily so library-only PSI use does not couple to raw-data handling.
    from ..features import build_features
    from ..pipeline import load_transactions
    from ..snapshots import build_snapshots

    transactions = load_transactions(transactions_path, encoding)
    snapshots = build_snapshots(transactions, training_dates, horizon_days)
    actual_dates = set(snapshots["snapshot_date"].dt.strftime("%Y-%m-%d"))
    missing_dates = set(training_dates) - actual_dates
    if missing_dates:
        raise ValueError(
            "Raw reference transactions do not fully cover model training snapshots: "
            f"{sorted(missing_dates)} (including their {horizon_days}-day observation horizon)"
        )
    features = build_features(transactions, snapshots)
    missing_features = set(feature_columns) - set(features.columns)
    if missing_features:
        raise ValueError(f"Pipeline is missing model reference features: {sorted(missing_features)}")
    return features


def _apply_reference_preprocessing(
    reference: pd.DataFrame, feature_types: Mapping[str, str], medians: Mapping[str, float],
) -> pd.DataFrame:
    """Mirror the frozen model's train-only numeric imputation for reference rows."""
    prepared = reference.copy()
    for feature, kind in feature_types.items():
        if kind == "number":
            if feature not in medians:
                raise ValueError(f"Model metadata has no training median for numeric feature {feature!r}")
            prepared[feature] = pd.to_numeric(prepared[feature], errors="raise").fillna(float(medians[feature]))
    return prepared


def run_check(args: argparse.Namespace) -> pd.DataFrame:
    metadata, contract_features, feature_types = _load_contract(args.model_artifacts)
    selected = args.feature or contract_features
    invalid = set(selected) - set(contract_features)
    if invalid:
        raise ValueError(f"Requested features are outside the model contract: {sorted(invalid)}")
    reference = _build_reference_features(
        args.reference_transactions, metadata["training_snapshot_dates"], contract_features,
        args.horizon_days, args.encoding,
    )
    reference = _apply_reference_preprocessing(
        reference, feature_types, metadata["churn_preprocessing"].get("numeric_medians", {}),
    )
    current = pd.read_csv(args.current_batch)
    result = compare_feature_populations(
        reference, current, selected, feature_types,
        PSIConfig(threshold=args.threshold, epsilon=args.epsilon, numeric_bins=args.numeric_bins),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    flagged = result.loc[result["drift_flag"], "feature"].tolist()
    print("RETRAINING TRIGGER: " + (f"drift flagged for {', '.join(flagged)}" if flagged else "no monitored feature crossed threshold"))
    print("This is a drift signal, not evidence of model inaccuracy; no retraining is performed.")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare latest ChurnGuard model inputs with training-reference PSI.")
    parser.add_argument("--reference-transactions", type=Path, required=True,
                        help="Raw transaction CSV used to reconstruct production-model training features.")
    parser.add_argument("--current-batch", type=Path, required=True,
                        help="Flat CSV of the latest scoring batch's model features.")
    parser.add_argument("--model-artifacts", type=Path, default=Path("models"),
                        help="Frozen model artifact directory containing metadata.json (default: models).")
    parser.add_argument("--feature", action="append", help="Model feature to monitor; repeatable. Default: all.")
    parser.add_argument("--threshold", type=float, default=0.20, help="Drift threshold (default: 0.20).")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Zero-bin smoothing (default: 1e-6).")
    parser.add_argument("--numeric-bins", type=int, default=10, help="Maximum reference quantile bins (default: 10).")
    parser.add_argument("--horizon-days", type=int, default=90, help="Training snapshot coverage horizon (default: 90).")
    parser.add_argument("--encoding", default="latin1", help="Raw transaction CSV encoding (default: latin1).")
    parser.add_argument("--output", type=Path, help="Optional CSV report path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.horizon_days <= 0:
        raise SystemExit("--horizon-days must be positive")
    try:
        run_check(args)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"PSI check failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
