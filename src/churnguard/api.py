"""FastAPI inference service backed only by frozen ChurnGuard artifacts."""

from __future__ import annotations

from datetime import date, datetime
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .artifacts import InferenceArtifacts, load_inference_artifacts
from .value_model import signed_expm1


class FeatureRecord(BaseModel):
    """One inference observation; model features are distinct from metadata."""

    model_config = ConfigDict(extra="forbid")
    CustomerID: int | str | None = None
    snapshot_date: date | datetime | str | None = None
    features: dict[str, Any]

    @field_validator("snapshot_date")
    @classmethod
    def validate_snapshot_date(cls, value):
        if value is not None:
            try:
                parsed = pd.Timestamp(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("snapshot_date must be a valid date") from exc
            if pd.isna(parsed):
                raise ValueError("snapshot_date must be a valid date")
        return value


class BatchScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[FeatureRecord] = Field(min_length=1)


class ScoreResponse(BaseModel):
    CustomerID: int | str | None = None
    snapshot_date: str | None = None
    churn_probability: float
    predicted_90d_value: float
    risk_weighted_value: float
    churn_rank: int | None = None
    value_rank: int | None = None
    risk_value_rank: int | None = None


def _normalise_snapshot(value) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).isoformat()


def _feature_frame(record: FeatureRecord, metadata: dict) -> pd.DataFrame:
    features = record.features
    expected = list(metadata["feature_columns"])
    missing = sorted(set(expected) - set(features))
    unexpected = sorted(set(features) - set(expected))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing required features: {missing}")
        if unexpected:
            details.append(f"unexpected features: {unexpected}")
        raise ValueError("; ".join(details))

    # Dict input order is irrelevant: explicitly restore the frozen training order.
    ordered = {name: features[name] for name in expected}
    types = metadata["churn_preprocessing"]["feature_types"]
    category_schemas = (
        metadata["churn_preprocessing"]["categorical_values"],
        metadata["future_value_preprocessing"]["categorical_values"],
    )
    for name, value in ordered.items():
        if value is None:
            raise ValueError(f"Feature {name!r} cannot be null")
        if types[name] == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Feature {name!r} must be a finite number")
        else:
            if not isinstance(value, str):
                raise ValueError(f"Categorical feature {name!r} must be a string")
            if any(value not in categories[name] for categories in category_schemas):
                raise ValueError(f"Invalid value for categorical feature {name!r}")
    return pd.DataFrame([ordered], columns=expected)


def _preprocess(frame: pd.DataFrame, schema: dict) -> pd.DataFrame:
    result = frame.copy()
    for column, median in schema["numeric_medians"].items():
        result[column] = pd.to_numeric(result[column], errors="raise").fillna(float(median))
    for column, categories in schema["categorical_values"].items():
        result[column] = pd.Categorical(result[column], categories=categories)
    return result.loc[:, schema["feature_columns"]]


def _score(record: FeatureRecord, artifacts: InferenceArtifacts) -> dict:
    raw = _feature_frame(record, artifacts.metadata)
    churn_schema = artifacts.metadata["churn_preprocessing"]
    value_schema = artifacts.metadata["future_value_preprocessing"]
    churn_matrix = _preprocess(raw, churn_schema)
    value_matrix = _preprocess(raw, value_schema)
    raw_probability = float(artifacts.churn_model.predict_proba(churn_matrix)[:, 1][0])
    churn_probability = float(artifacts.calibrator.predict([raw_probability])[0])
    transformed_value = float(artifacts.future_value_model.predict(value_matrix)[0])
    lower, upper = artifacts.metadata["future_value_training_transformed_target_bounds"]
    transformed_value = float(np.clip(transformed_value, lower, upper))
    predicted_value = float(signed_expm1([transformed_value])[0])
    if not all(math.isfinite(value) for value in (churn_probability, predicted_value)):
        raise ValueError("Model produced a non-finite score")
    return {
        "CustomerID": record.CustomerID,
        "snapshot_date": _normalise_snapshot(record.snapshot_date),
        "churn_probability": churn_probability,
        "predicted_90d_value": predicted_value,
        "risk_weighted_value": churn_probability * predicted_value,
    }


def _rank_batch(rows: list[dict]) -> None:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["snapshot_date"] or "__batch__", []).append(index)
    for indexes in groups.values():
        for score_name, rank_name in (
            ("churn_probability", "churn_rank"),
            ("predicted_90d_value", "value_rank"),
            ("risk_weighted_value", "risk_value_rank"),
        ):
            def tie_break(index):
                customer_id = rows[index]["CustomerID"]
                try:
                    return (0, float(customer_id), index)
                except (TypeError, ValueError):
                    return (1, "" if customer_id is None else str(customer_id), index)

            ordered = sorted(
                indexes,
                key=lambda index: (
                    -rows[index][score_name],
                    *tie_break(index),
                ),
            )
            for rank, index in enumerate(ordered, start=1):
                rows[index][rank_name] = rank


def create_app(artifact_dir: str | Path | None = None) -> FastAPI:
    """Construct an app and load frozen models once; absent artifacts stay unready."""
    directory = Path(artifact_dir or os.environ.get("CHURNGUARD_MODEL_DIR", "models"))
    app = FastAPI(title="ChurnGuard Inference API", version="1.0.0")
    # Local Vite development runs on a separate origin from the API. Restrict
    # browser access to local development hosts; deployed frontends can use a
    # same-origin reverse proxy or provide exact trusted origins at deployment.
    cors_origins = [
        origin.strip()
        for origin in os.environ.get(
            "CHURNGUARD_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    artifacts = None
    load_error = None
    try:
        artifacts = load_inference_artifacts(directory)
    except Exception as exc:
        load_error = exc
    app.state.artifacts = artifacts
    app.state.load_error = load_error

    def require_artifacts() -> InferenceArtifacts:
        if app.state.artifacts is None:
            raise HTTPException(status_code=503, detail="Inference artifacts are unavailable; service is not ready")
        return app.state.artifacts

    @app.get("/health")
    def health():
        loaded = app.state.artifacts is not None
        response = {
            "status": "ok" if loaded else "not_ready",
            "churn_model_loaded": loaded,
            "calibrator_loaded": loaded,
            "future_value_model_loaded": loaded,
        }
        return response if loaded else JSONResponse(status_code=503, content=response)

    @app.post("/predict", response_model=ScoreResponse)
    def predict(record: FeatureRecord):
        fitted = require_artifacts()
        try:
            return _score(record, fitted)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Model inference failed") from exc

    @app.post("/batch_score")
    def batch_score(request: BatchScoreRequest):
        fitted = require_artifacts()
        try:
            scored = [_score(record, fitted) for record in request.customers]
            _rank_batch(scored)
            return {"count": len(scored), "scored_customers": scored}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Model inference failed") from exc

    return app


app = create_app()
