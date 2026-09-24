"""Fit churn baselines and LightGBM with validation-only selection and final test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import ranking_metrics
from .calibration import SigmoidProbabilityCalibrator, probability_metrics, reliability_table
from .model_config import ModelConfig, TemporalSplitConfig
from .modeling import RFM_FEATURES, RFMRuleBaseline, lightgbm_feature_metadata, make_lightgbm, make_logistic_pipeline, make_xy, prepare_lightgbm_features
from .temporal import split_by_snapshot_date
from .value_config import ValueModelConfig
from .value_experiment import evaluate_value_models
from .clv_config import ClassicalCLVConfig
from .clv_experiment import evaluate_classical_clv
from .targeting import evaluate_targeting_partitions
from .targeting_config import TargetingConfig


def evaluate_models(
    table: pd.DataFrame,
    split_config: TemporalSplitConfig,
    model_config: ModelConfig,
    value_config: ValueModelConfig | None = None,
    transactions: pd.DataFrame | None = None,
    classical_clv_config: ClassicalCLVConfig | None = None,
    targeting_config: TargetingConfig | None = None,
):
    """Evaluate existing churn models and the supervised 90-day value models."""
    split = split_by_snapshot_date(table, split_config)
    target = split_config.target_column
    train_x, train_y = make_xy(split.train, target)
    val_x, val_y = make_xy(split.validation, target, train_x.columns)
    rfm_train_x, _ = make_xy(split.train, target, RFM_FEATURES)
    rfm_val_x, _ = make_xy(split.validation, target, RFM_FEATURES)

    train_prior = float(train_y.mean())
    majority_probability = np.full(len(val_y), train_prior)

    logistic = make_logistic_pipeline()
    logistic.fit(rfm_train_x, train_y)
    logistic_val_probability = logistic.predict_proba(rfm_val_x)[:, 1]

    rules = RFMRuleBaseline().fit(split.train)
    rule_val_probability = rules.predict_proba(split.validation)[:, 1]
    lgb_train_x, lgb_val_x = prepare_lightgbm_features(train_x, val_x)
    lightgbm = make_lightgbm(model_config)
    lightgbm.fit(lgb_train_x, train_y, categorical_feature="auto")
    lgb_val_probability = lightgbm.predict_proba(lgb_val_x)[:, 1]

    # The sigmoid is fit only on validation predictions and labels. Validation
    # calibration metrics are in-sample to this mapping; no test labels are read.
    calibrator = SigmoidProbabilityCalibrator(model_config.random_seed)
    calibrator.fit(lgb_val_probability, val_y)
    calibrated_val_probability = calibrator.predict(lgb_val_probability)

    # Only after the calibrator is frozen do we score and evaluate final-test rows.
    test_x, test_y = make_xy(split.test, target, train_x.columns)
    majority_test_probability = np.full(len(test_y), train_prior)
    rfm_test_x, rfm_test_y = make_xy(split.test, target, RFM_FEATURES)
    logistic_test_probability = logistic.predict_proba(rfm_test_x)[:, 1]
    rule_test_probability = rules.predict_proba(split.test)[:, 1]
    _, lgb_test_x = prepare_lightgbm_features(train_x, test_x)
    lgb_test_probability = lightgbm.predict_proba(lgb_test_x)[:, 1]
    calibrated_test_probability = calibrator.predict(lgb_test_probability)

    preds = {
        "majority_class": (majority_probability, majority_test_probability),
        "logistic_rfm": (logistic_val_probability, logistic_test_probability),
        "rule_based_rfm": (rule_val_probability, rule_test_probability),
        "lightgbm": (lgb_val_probability, lgb_test_probability),
    }
    results = {}
    for name, (val_probability, test_probability) in preds.items():
        results[name] = {
            "validation": ranking_metrics(val_y, val_probability),
            "test": ranking_metrics(test_y, test_probability),
        }
        # Retain probability quality for each already-scored baseline too, so
        # tracking can compare all individual runs without refitting anything.
        results[name]["validation"]["brier_score"] = probability_metrics(val_y, val_probability)["brier_score"]
        results[name]["test"]["brier_score"] = probability_metrics(test_y, test_probability)["brier_score"]
    calibration_metrics = {
        "raw": {
            "validation": probability_metrics(val_y, lgb_val_probability),
            "test": probability_metrics(test_y, lgb_test_probability),
        },
        "calibrated": {
            "validation": probability_metrics(val_y, calibrated_val_probability),
            "test": probability_metrics(test_y, calibrated_test_probability),
        },
    }
    validation_predictions = split.validation[["CustomerID", "snapshot_date"]].copy()
    validation_predictions[target] = val_y.to_numpy()
    validation_predictions["raw_lightgbm_probability"] = lgb_val_probability
    validation_predictions["calibrated_lightgbm_probability"] = calibrated_val_probability
    test_predictions = split.test[["CustomerID", "snapshot_date"]].copy()
    test_predictions[target] = test_y.to_numpy()
    test_predictions["raw_lightgbm_probability"] = lgb_test_probability
    test_predictions["calibrated_lightgbm_probability"] = calibrated_test_probability
    reliability = {
        "validation_raw": reliability_table(val_y, lgb_val_probability, model_config.reliability_bins),
        "validation_calibrated": reliability_table(val_y, calibrated_val_probability, model_config.reliability_bins),
        "test_raw": reliability_table(test_y, lgb_test_probability, model_config.reliability_bins),
        "test_calibrated": reliability_table(test_y, calibrated_test_probability, model_config.reliability_bins),
    }
    value_results = evaluate_value_models(split, value_config)
    # Assemble a prediction-only input first. Actual held-out future spend is
    # passed separately and only joined by evaluate_targeting_partitions after
    # rankings have been produced.
    targeting_prediction_inputs = {}
    for partition_name in ("validation", "test"):
        churn_predictions = pd.DataFrame({
            "CustomerID": split.validation["CustomerID"].to_numpy()
                if partition_name == "validation" else split.test["CustomerID"].to_numpy(),
            "snapshot_date": split.validation["snapshot_date"].to_numpy()
                if partition_name == "validation" else split.test["snapshot_date"].to_numpy(),
            "calibrated_churn_probability": calibrated_val_probability
                if partition_name == "validation" else calibrated_test_probability,
        })
        value_predictions = value_results["predictions"][partition_name][
            ["CustomerID", "snapshot_date", "lightgbm_value"]
        ].rename(columns={"lightgbm_value": "predicted_90d_value"})
        targeting_prediction_inputs[partition_name] = churn_predictions.merge(
            value_predictions,
            on=["CustomerID", "snapshot_date"],
            how="inner",
            validate="one_to_one",
        )
    targeting_results = evaluate_targeting_partitions(
        targeting_prediction_inputs["validation"],
        targeting_prediction_inputs["test"],
        split.validation[["CustomerID", "snapshot_date", "future_net_spend_90d"]].copy(),
        split.test[["CustomerID", "snapshot_date", "future_net_spend_90d"]].copy(),
        targeting_config,
    )
    classical_clv_results = None
    if transactions is not None:
        classical_clv_results = evaluate_classical_clv(
            transactions,
            split,
            supervised_predictions=value_results["predictions"],
            config=classical_clv_config,
        )
    return {
        "results": results,
        "calibration_results": calibration_metrics,
        "predictions": {"validation": validation_predictions, "test": test_predictions},
        "reliability": reliability,
        "value_results": value_results,
        "classical_clv_results": classical_clv_results,
        "targeting_results": targeting_results,
        "split_counts": {"train": len(split.train), "validation": len(split.validation), "test": len(split.test)},
        "feature_columns": {"logistic_rfm": RFM_FEATURES, "lightgbm": list(train_x.columns)},
        "lightgbm_preprocessing": lightgbm_feature_metadata(train_x),
        "models": {"logistic_rfm": logistic, "rule_based_rfm": rules, "lightgbm": lightgbm, "lightgbm_calibrator": calibrator},
    }
