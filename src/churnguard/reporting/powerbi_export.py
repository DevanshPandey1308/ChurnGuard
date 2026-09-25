"""Export real ChurnGuard scoring and SHAP outputs for Power BI.

The exporter calls the existing FastAPI batch-scoring route. It never fits a
model, derives scores, or invents a risk band.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from ..cleaning import clean_transactions
from ..features import build_features
from ..model_config import TemporalSplitConfig
from ..snapshots import build_snapshots
from ..api import _rank_batch, score_feature_matrix
from ..artifacts import load_inference_artifacts


SCORE_COLUMNS = [
    "CustomerID", "snapshot_date", "country", "churn_probability",
    "predicted_90d_value", "risk_weighted_value", "churn_rank", "value_rank", "risk_value_rank",
]
API_SCORE_COLUMNS = [
    "CustomerID", "snapshot_date", "churn_probability", "predicted_90d_value",
    "risk_weighted_value", "churn_rank", "value_rank", "risk_value_rank",
]
IMPORTANCE_COLUMNS = ["feature", "mean_abs_shap", "mean_shap", "rank"]


def read_feature_contract(model_artifacts: str | Path = "models") -> tuple[list[str], dict[str, str]]:
    """Read and validate the exact frozen model input schema."""
    path = Path(model_artifacts) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Model feature contract not found: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    features = metadata.get("feature_columns")
    schema = metadata.get("churn_preprocessing", {})
    types = schema.get("feature_types", {})
    if not isinstance(features, list) or not features or schema.get("feature_columns") != features:
        raise ValueError("Frozen model metadata has no consistent feature contract")
    if set(features) - set(types):
        raise ValueError(f"Model feature types missing: {sorted(set(features) - set(types))}")
    forbidden = {"CustomerID", "snapshot_date", "churn_90d", "repurchased_90d", "future_net_spend_90d"}
    if forbidden.intersection(features):
        raise ValueError("Frozen feature contract contains identifier or outcome fields")
    return features, types


def _read_model_metadata(model_artifacts: str | Path = "models") -> dict[str, Any]:
    path = Path(model_artifacts) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Model feature contract not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_raw_feature_population(
    raw_transactions_path: str | Path,
    model_artifacts: str | Path = "models",
    horizon_days: int = 90,
    encoding: str = "latin1",
) -> dict[str, Any]:
    """Build model-ready snapshots through the canonical ChurnGuard pipeline.

    Snapshot dates come from the frozen model's train/validation/test metadata.
    This reporting path intentionally does not construct future labels.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    metadata = _read_model_metadata(model_artifacts)
    features, feature_types = read_feature_contract(model_artifacts)
    split_dates = [
        *metadata.get("training_snapshot_dates", []),
        *metadata.get("validation_snapshot_dates", []),
        *metadata.get("test_snapshot_dates", metadata.get("final_test_snapshot_dates", [])),
    ]
    if not split_dates:
        raise ValueError("Model metadata does not identify training/validation snapshot dates")
    if not metadata.get("test_snapshot_dates") and not metadata.get("final_test_snapshot_dates"):
        # Artifact format v1 stores train/validation dates. Use the repository's
        # configured final holdout, rather than guessing another reporting date.
        split_dates.extend(TemporalSplitConfig().test_dates)
    if not split_dates or len(set(split_dates)) != len(split_dates):
        raise ValueError("Model metadata must contain unique training, validation, and test snapshot dates")

    raw = pd.read_csv(raw_transactions_path, encoding=encoding)
    raw_row_count = len(raw)
    transactions = clean_transactions(raw)
    if transactions.empty:
        raise ValueError("No valid transactions remain after ChurnGuard cleaning")
    snapshots = build_snapshots(transactions, split_dates, horizon_days=horizon_days)
    actual_dates = set(snapshots["snapshot_date"].dt.strftime("%Y-%m-%d"))
    missing_dates = set(split_dates) - actual_dates
    if missing_dates:
        raise ValueError(f"Raw transactions do not cover configured snapshots and full horizons: {sorted(missing_dates)}")
    if snapshots.duplicated(["CustomerID", "snapshot_date"]).any():
        raise ValueError("Canonical snapshot builder returned duplicate customer snapshots")

    # This is the existing point-in-time feature builder. It filters transaction
    # dates strictly before each snapshot and does not accept or create labels.
    feature_table = build_features(transactions, snapshots)
    if feature_table.duplicated(["CustomerID", "snapshot_date"]).any():
        raise ValueError("Canonical feature builder returned duplicate customer snapshots")
    if len(feature_table) != len(snapshots):
        raise ValueError(
            f"Feature builder returned {len(feature_table)} rows for {len(snapshots)} snapshots"
        )
    missing_features = set(features) - set(feature_table.columns)
    if missing_features:
        raise ValueError(f"Generated population is missing model features: {sorted(missing_features)}")
    # Keep just identifiers and frozen model inputs; no labels or incidental
    # feature-table columns are sent to inference.
    model_input = feature_table.loc[:, ["CustomerID", "snapshot_date", *features]].copy()
    preprocessing = metadata["churn_preprocessing"]
    training_medians = preprocessing.get("numeric_medians", {})
    for name in features:
        if feature_types[name] == "number":
            if name not in training_medians:
                raise ValueError(f"Frozen model metadata has no training median for {name!r}")
            model_input[name] = pd.to_numeric(model_input[name], errors="raise").fillna(float(training_medians[name]))
        elif feature_types[name] == "category":
            allowed = set(preprocessing.get("categorical_values", {}).get(name, []))
            if not allowed:
                raise ValueError(f"Frozen model metadata has no categories for {name!r}")
            if model_input[name].isna().any():
                raise ValueError(f"Generated categorical feature {name!r} contains missing values")
            model_input[name] = model_input[name].astype(str)
        else:
            raise ValueError(f"Unsupported frozen model feature type for {name!r}: {feature_types[name]!r}")
    snapshot_counts = (
        model_input.groupby("snapshot_date", sort=True).size().rename("customer_snapshot_rows").to_dict()
    )
    return {
        "features": model_input,
        "raw_transaction_rows": raw_row_count,
        "cleaned_transaction_rows": len(transactions),
        "snapshot_row_count": len(snapshots),
        "snapshot_counts": {pd.Timestamp(date).strftime("%Y-%m-%d"): int(count) for date, count in snapshot_counts.items()},
    }


def score_raw_population(model_input: pd.DataFrame, features: Sequence[str], model_artifacts: str | Path) -> pd.DataFrame:
    """Score the full raw-derived population through the API's shared model logic.

    This uses the same frozen estimators and preprocessing as FastAPI, but calls
    the internal vectorized scorer to avoid the public API's intentional request
    validation of unseen categories and per-record HTTP overhead. Unseen values
    become missing categories under the model's frozen category schema.
    """
    artifacts = load_inference_artifacts(model_artifacts)
    scores = score_feature_matrix(model_input.loc[:, list(features)], artifacts)
    rows = pd.concat([
        model_input[["CustomerID", "snapshot_date"]].reset_index(drop=True),
        scores.reset_index(drop=True),
    ], axis=1).to_dict(orient="records")
    for row in rows:
        row["snapshot_date"] = pd.Timestamp(row["snapshot_date"]).strftime("%Y-%m-%d")
    _rank_batch(rows)
    return pd.DataFrame(rows).loc[:, API_SCORE_COLUMNS]


def _normalise_input(scoring_input: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    required = {"CustomerID", "snapshot_date", *features}
    missing = required - set(scoring_input.columns)
    extra = set(scoring_input.columns) - required
    if missing or extra:
        raise ValueError(f"Scoring input columns mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    frame = scoring_input.loc[:, ["CustomerID", "snapshot_date", *features]].copy()
    if frame.empty:
        raise ValueError("Scoring input must contain at least one customer")
    if frame[["CustomerID", "snapshot_date"]].isna().any().any():
        raise ValueError("CustomerID and snapshot_date must not be null")
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="raise").dt.strftime("%Y-%m-%d")
    if frame.duplicated(["CustomerID", "snapshot_date"]).any():
        raise ValueError("Scoring input has duplicate customer snapshots")
    if "country" not in frame:
        raise ValueError("Power BI reporting requires country in the model feature contract")
    if frame.loc[:, features].isna().any().any():
        raise ValueError("Scoring inputs contain null model features; these are not valid API records")
    return frame


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if pd.isna(value):
        raise ValueError("Null model input values cannot be sent to /batch_score")
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def score_via_batch_api(
    scoring_input: pd.DataFrame,
    features: Sequence[str],
    api_url: str = "http://localhost:8000",
    timeout_seconds: float = 120,
) -> pd.DataFrame:
    """Call the existing batch endpoint and return its untouched scored rows."""
    frame = _normalise_input(scoring_input, features)
    records = []
    for row in frame.to_dict(orient="records"):
        records.append({
            "CustomerID": _json_value(row["CustomerID"]),
            "snapshot_date": row["snapshot_date"],
            "features": {name: _json_value(row[name]) for name in features},
        })
    body = json.dumps({"customers": records}, allow_nan=False).encode("utf-8")
    request = Request(
        f"{api_url.rstrip('/')}/batch_score", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ChurnGuard /batch_score returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach ChurnGuard batch API at {api_url}: {exc.reason}") from exc
    rows = result.get("scored_customers")
    if result.get("count") != len(frame) or not isinstance(rows, list) or len(rows) != len(frame):
        raise ValueError("/batch_score response count does not match submitted input rows")
    scored = pd.DataFrame(rows)
    missing = set(API_SCORE_COLUMNS) - set(scored.columns)
    if missing:
        raise ValueError(f"/batch_score response is missing fields: {sorted(missing)}")
    if scored.duplicated(["CustomerID", "snapshot_date"]).any():
        raise ValueError("/batch_score returned duplicate customer snapshots")
    return scored.loc[:, API_SCORE_COLUMNS]


def prepare_customer_scores(scoring_input: pd.DataFrame, scored: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Join source country and API scores, validating one-to-one batch alignment."""
    frame = _normalise_input(scoring_input, features)
    scores = scored.loc[:, API_SCORE_COLUMNS].copy()
    scores["snapshot_date"] = pd.to_datetime(scores["snapshot_date"], errors="raise").dt.strftime("%Y-%m-%d")
    if scores.duplicated(["CustomerID", "snapshot_date"]).any():
        raise ValueError("Scored output has duplicate customer snapshots")
    joined = frame[["CustomerID", "snapshot_date", "country"]].merge(
        scores, on=["CustomerID", "snapshot_date"], how="outer", validate="one_to_one", indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Scored output customer snapshots do not match scoring inputs exactly")
    joined = joined.drop(columns="_merge")
    probability = joined["churn_probability"].to_numpy(dtype=float)
    value = joined["predicted_90d_value"].to_numpy(dtype=float)
    risk = joined["risk_weighted_value"].to_numpy(dtype=float)
    if not (np.isfinite(probability).all() and np.isfinite(value).all() and np.isfinite(risk).all()):
        raise ValueError("Scored output contains non-finite values")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("churn_probability must be in [0, 1]")
    if not np.allclose(risk, probability * value, rtol=1e-10, atol=1e-10):
        raise ValueError("risk_weighted_value does not match the existing API output contract")
    return joined.loc[:, SCORE_COLUMNS]


def prepare_global_importance(
    source_path: str | Path, destination_path: str | Path, features: Sequence[str],
) -> pd.DataFrame:
    """Copy the existing SHAP global summary after validating its model features."""
    source = Path(source_path)
    importance = pd.read_csv(source)
    missing = set(IMPORTANCE_COLUMNS) - set(importance.columns)
    if missing:
        raise ValueError(f"Global SHAP artifact is missing columns: {sorted(missing)}")
    if set(importance["feature"]) - set(features):
        raise ValueError("Global SHAP artifact contains features outside the frozen model contract")
    result = importance.loc[:, IMPORTANCE_COLUMNS].sort_values("rank", kind="stable").reset_index(drop=True)
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return result


def export_powerbi_data(
    input_csv: str | Path | None = None,
    output_dir: str | Path = "artifacts/powerbi",
    model_artifacts: str | Path = "models",
    shap_importance: str | Path = "artifacts/shap/global_feature_importance.csv",
    api_url: str = "http://localhost:8000",
    raw_transactions: str | Path | None = None,
    horizon_days: int = 90,
    encoding: str = "latin1",
    timeout_seconds: float = 600,
) -> dict[str, Any]:
    features, _ = read_feature_contract(model_artifacts)
    if (input_csv is None) == (raw_transactions is None):
        raise ValueError("Provide exactly one of input_csv (flat model inputs) or raw_transactions")
    raw_details = None
    if raw_transactions is not None:
        raw_details = build_raw_feature_population(raw_transactions, model_artifacts, horizon_days, encoding)
        source = raw_details["features"]
    else:
        source = pd.read_csv(input_csv)
    if raw_details is None:
        scored = score_via_batch_api(source, features, api_url, timeout_seconds)
    else:
        scored = score_raw_population(source, features, model_artifacts)
    customer_table = prepare_customer_scores(source, scored, features)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    customer_path = directory / "churnguard_customer_scores.csv"
    customer_table.to_csv(customer_path, index=False)
    importance_path = directory / "churnguard_global_feature_importance.csv"
    importance = prepare_global_importance(shap_importance, importance_path, features)
    return {
        "customer_scores_path": customer_path,
        "customer_snapshot_count": len(customer_table),
        "importance_path": importance_path,
        "importance_feature_count": len(importance),
        "raw_details": raw_details,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export real ChurnGuard outputs for Power BI.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="Flat CSV of model inputs for scoring (existing mode).")
    inputs.add_argument("--raw-transactions", type=Path,
                        help="Raw transaction CSV; build snapshots/features using the existing ChurnGuard pipeline.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/powerbi"))
    parser.add_argument("--model-artifacts", type=Path, default=Path("models"))
    parser.add_argument("--shap-importance", type=Path, default=Path("artifacts/shap/global_feature_importance.csv"))
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--timeout-seconds", type=float, default=600,
                        help="HTTP timeout for scoring a population (default: 600).")
    parser.add_argument("--horizon-days", type=int, default=90,
                        help="Snapshot coverage horizon used by the canonical snapshot builder (default: 90).")
    parser.add_argument("--encoding", default="latin1", help="Raw transaction CSV encoding (default: latin1).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        outputs = export_powerbi_data(
            args.input, args.output_dir, args.model_artifacts, args.shap_importance, args.api_url,
            raw_transactions=args.raw_transactions, horizon_days=args.horizon_days, encoding=args.encoding,
            timeout_seconds=args.timeout_seconds,
        )
    except (FileNotFoundError, OSError, ValueError, RuntimeError, KeyError) as exc:
        raise SystemExit(f"Power BI export failed: {exc}") from exc
    print(f"Scored customer snapshots: {outputs['customer_snapshot_count']} -> {outputs['customer_scores_path']}")
    print(f"Global SHAP features: {outputs['importance_feature_count']} -> {outputs['importance_path']}")
    details = outputs["raw_details"]
    if details is not None:
        print(f"Raw transaction rows read: {details['raw_transaction_rows']}")
        print(f"Transactions after canonical cleaning: {details['cleaned_transaction_rows']}")
        print(f"Customer snapshots generated: {details['snapshot_row_count']}")
        print(f"Customer-snapshot rows by date: {json.dumps(details['snapshot_counts'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
