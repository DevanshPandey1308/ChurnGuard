"""Temporal evaluation workflow for supervised 90-day customer value."""

from __future__ import annotations

import pandas as pd

from .modeling import RFM_FEATURES, lightgbm_feature_metadata, prepare_lightgbm_features
from .temporal import TemporalDataSplit
from .value_config import ValueModelConfig
from .value_model import (
    make_lightgbm_regressor,
    make_log_linear_rfm,
    inverse_transformed_predictions,
    regression_metrics,
    signed_log1p,
    target_distribution,
    top_decile_value_capture,
    training_baseline_predictions,
    value_feature_columns,
)


def evaluate_value_models(split: TemporalDataSplit, config: ValueModelConfig | None = None) -> dict:
    """Fit value models on train only and evaluate validation/final test.

    Validation labels are used only for metrics. Final-test labels are not read
    until every model has been fitted and validation predictions are produced.
    Hyperparameters remain fixed; June results do not select or tune a model.
    """
    config = config or ValueModelConfig()
    target = config.target_column
    train = split.train
    validation = split.validation
    test = split.test
    if target not in train or target not in validation or target not in test:
        raise ValueError(f"All temporal partitions must contain value target {target!r}")

    columns = value_feature_columns(train, target)
    forbidden = {"CustomerID", "snapshot_date", "future_net_spend_90d", "repurchased_90d", "churn_90d"}
    if forbidden.intersection(columns):
        raise ValueError(f"Identifiers/targets leaked into value features: {sorted(forbidden.intersection(columns))}")
    x_train = train.loc[:, columns].copy()
    x_validation = validation.loc[:, columns].copy()
    y_train = train[target].astype(float).to_numpy()
    y_validation = validation[target].astype(float).to_numpy()

    # Negative net values exist in the training labels, so signed log is valid
    # where plain log1p is not. Only training labels determine this transform.
    transformed_train_target = signed_log1p(y_train)
    baseline_predictions = training_baseline_predictions(y_train, len(validation), len(test))

    linear = make_log_linear_rfm(config)
    linear.fit(x_train.loc[:, RFM_FEATURES], transformed_train_target)
    linear_validation = inverse_transformed_predictions(
        linear, x_validation.loc[:, RFM_FEATURES], transformed_train_target
    )

    lgb_train_x, lgb_validation_x = prepare_lightgbm_features(x_train, x_validation)
    lightgbm = make_lightgbm_regressor(config)
    lightgbm.fit(lgb_train_x, transformed_train_target, categorical_feature="auto")
    lgb_validation = inverse_transformed_predictions(lightgbm, lgb_validation_x, transformed_train_target)

    # Freeze both fitted models before any final-test labels are accessed.
    x_test = test.loc[:, columns].copy()
    _, lgb_test_x = prepare_lightgbm_features(x_train, x_test)
    linear_test = inverse_transformed_predictions(linear, test.loc[:, RFM_FEATURES], transformed_train_target)
    lgb_test = inverse_transformed_predictions(lightgbm, lgb_test_x, transformed_train_target)
    y_test = test[target].astype(float).to_numpy()

    predictions = {
        "training_mean": (baseline_predictions["mean_baseline"][0], baseline_predictions["mean_baseline"][1]),
        "training_median": (baseline_predictions["median_baseline"][0], baseline_predictions["median_baseline"][1]),
        "log_linear_rfm": (linear_validation, linear_test),
        "lightgbm_value": (lgb_validation, lgb_test),
    }
    metrics = {
        name: {"validation": regression_metrics(y_validation, pred_val), "test": regression_metrics(y_test, pred_test)}
        for name, (pred_val, pred_test) in predictions.items()
    }
    captures = {
        name: top_decile_value_capture(y_test, test_pred, config.top_fraction)
        for name, (_, test_pred) in predictions.items()
    }
    validation_output = validation[["CustomerID", "snapshot_date"]].copy()
    validation_output[target] = y_validation
    test_output = test[["CustomerID", "snapshot_date"]].copy()
    test_output[target] = y_test
    for name, (pred_val, pred_test) in predictions.items():
        validation_output[name] = pred_val
        test_output[name] = pred_test
    # Report full-sample distribution descriptively after fitting/evaluation;
    # transformation and all fitted quantities above depend on train targets only.
    return {
        "target_distribution": target_distribution(pd.concat([train[target], validation[target], test[target]], ignore_index=True)),
        "train_target_distribution": target_distribution(y_train),
        "metrics": metrics,
        "test_top_decile_value_capture": captures,
        "feature_columns": columns,
        "lightgbm_preprocessing": lightgbm_feature_metadata(x_train),
        "training_transformed_target_bounds": [float(transformed_train_target.min()), float(transformed_train_target.max())],
        "linear_feature_columns": list(RFM_FEATURES),
        "target_transform": config.target_transform,
        "top_fraction": config.top_fraction,
        "predictions": {"validation": validation_output, "test": test_output},
        "models": {"log_linear_rfm": linear, "lightgbm_value": lightgbm},
    }
