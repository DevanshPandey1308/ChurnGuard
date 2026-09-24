"""Feature contracts, churn baselines, and LightGBM estimator construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model_config import ModelConfig

RFM_FEATURES = ["recency_days", "purchase_invoice_count", "historical_net_spend"]
TARGET_DERIVED_COLUMNS = {
    "repurchased_90d", "churn_90d", "future_net_spend_90d",
}
IDENTIFIER_COLUMNS = {"CustomerID", "snapshot_date"}


def feature_columns(table: pd.DataFrame, target_column: str = "churn_90d") -> list[str]:
    """Return engineered feature columns after excluding identifiers and outcomes."""
    excluded = IDENTIFIER_COLUMNS | TARGET_DERIVED_COLUMNS | {target_column}
    columns = [column for column in table.columns if column not in excluded]
    leaked = set(columns) & TARGET_DERIVED_COLUMNS
    if leaked:
        raise ValueError(f"Future-derived columns cannot be features: {sorted(leaked)}")
    return columns


def make_xy(table: pd.DataFrame, target_column: str = "churn_90d", columns=None):
    """Construct X/y with an explicit guard against outcome leakage."""
    if target_column not in table:
        raise ValueError(f"Missing target column: {target_column}")
    selected = list(columns) if columns is not None else feature_columns(table, target_column)
    forbidden = TARGET_DERIVED_COLUMNS | IDENTIFIER_COLUMNS | {target_column}
    bad = forbidden.intersection(selected)
    if bad:
        raise ValueError(f"Forbidden columns in features: {sorted(bad)}")
    missing = set(selected) - set(table.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    return table.loc[:, selected].copy(), table[target_column].astype("int64").copy()


def make_logistic_pipeline():
    """Interpretable scaled logistic regression on core RFM only."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    prep = ColumnTransformer([("rfm", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), RFM_FEATURES)])
    return Pipeline([("preprocess", prep), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])


@dataclass
class RFMRuleBaseline:
    """Train-calibrated version of notebook RFM rules; at-risk/dormant predict churn."""

    recency_cuts: np.ndarray | None = None
    frequency_cuts: np.ndarray | None = None

    def fit(self, table: pd.DataFrame) -> "RFMRuleBaseline":
        self.recency_cuts = np.nanquantile(table["recency_days"].astype(float), [0.2, 0.4, 0.6, 0.8])
        self.frequency_cuts = np.nanquantile(table["purchase_invoice_count"].astype(float), [0.2, 0.4, 0.6, 0.8])
        return self

    def predict_proba(self, table: pd.DataFrame) -> np.ndarray:
        if self.recency_cuts is None or self.frequency_cuts is None:
            raise RuntimeError("Fit RFMRuleBaseline before prediction")
        # Higher recency means stale/inactive; higher frequency means loyal.
        r_score = 5 - np.searchsorted(self.recency_cuts, table["recency_days"].to_numpy(), side="left")
        f_score = 1 + np.searchsorted(self.frequency_cuts, table["purchase_invoice_count"].to_numpy(), side="right")
        # Original notebook: high-value/loyal are safe; at-risk/dormant are churn.
        churn = (r_score < 4) | (f_score < 4)
        probability = churn.astype(float) * 0.75 + 0.125
        return np.column_stack([1 - probability, probability])


def make_lightgbm(config: ModelConfig):
    """Create a seeded LightGBM binary classifier; Country uses native categories."""
    from lightgbm import LGBMClassifier

    params = dict(config.lightgbm_params)
    params.update({"objective": "binary", "random_state": config.random_seed, "n_jobs": -1})
    return LGBMClassifier(**params)


def prepare_lightgbm_features(train_x: pd.DataFrame, other_x: pd.DataFrame | None = None):
    """Impute numeric features and encode categoricals as pandas categories."""
    frames = [train_x.copy()] + ([other_x.copy()] if other_x is not None else [])
    for column in train_x.columns:
        if pd.api.types.is_numeric_dtype(train_x[column]):
            median = train_x[column].median()
            for frame in frames:
                frame[column] = frame[column].fillna(median)
        else:
            categories = pd.Index(train_x[column].dropna().unique())
            for frame in frames:
                frame[column] = pd.Categorical(frame[column], categories=categories)
    if other_x is None:
        return frames[0]
    return frames[0], frames[1]


def lightgbm_feature_metadata(train_x: pd.DataFrame) -> dict:
    """Describe train-only imputation and category rules for frozen inference."""
    numeric_medians = {}
    categorical_values = {}
    feature_types = {}
    for column in train_x.columns:
        if pd.api.types.is_numeric_dtype(train_x[column]):
            numeric_medians[column] = float(train_x[column].median())
            feature_types[column] = "number"
        else:
            categorical_values[column] = [str(value) for value in pd.Index(train_x[column].dropna().unique())]
            feature_types[column] = "category"
    return {
        "feature_columns": list(train_x.columns),
        "feature_types": feature_types,
        "numeric_medians": numeric_medians,
        "categorical_values": categorical_values,
    }
