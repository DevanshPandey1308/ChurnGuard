import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from churnguard.temporal import TemporalDataSplit
from churnguard.value_config import ValueModelConfig
from churnguard.value_experiment import evaluate_value_models
from churnguard.value_model import (
    signed_expm1,
    signed_log1p,
    inverse_transformed_predictions,
    top_decile_value_capture,
    training_baseline_predictions,
    value_feature_columns,
)


class ValueModelTests(unittest.TestCase):
    def setUp(self):
        self.train = self._rows(40, "2011-03-01", 0)
        second = self._rows(40, "2011-04-01", 40)
        self.train = pd.concat([self.train, second], ignore_index=True)
        self.validation = self._rows(10, "2011-05-01", 80)
        self.test = self._rows(10, "2011-06-01", 90)

    @staticmethod
    def _rows(count, date, offset):
        index = np.arange(count) + offset
        return pd.DataFrame({
            "CustomerID": index + 1000,
            "snapshot_date": pd.to_datetime([date] * count),
            "recency_days": (index % 15) + 1,
            "purchase_invoice_count": (index % 7) + 1,
            "purchase_line_count": (index % 10) + 1,
            "units_purchased": (index % 20) + 1,
            "historical_net_spend": index.astype(float) + 2,
            "log1p_purchase_invoice_count": np.log1p((index % 7) + 1),
            "signed_log1p_historical_net_spend": np.log1p(index + 2),
            "tenure_days": index + 30,
            "average_order_value": index.astype(float) + 5,
            "average_items_per_order": (index % 4) + 1,
            "distinct_products": (index % 8) + 1,
            "mean_inter_purchase_gap_days": (index % 11) + 1,
            "std_inter_purchase_gap_days": (index % 5).astype(float),
            "recent_3m_net_spend": index.astype(float) * 1.5,
            "previous_3m_net_spend": index.astype(float),
            "spend_trend_difference_3m": index.astype(float) * 0.5,
            "spend_trend_ratio_3m": 1.5,
            "recent_3m_order_count": (index % 4) + 1,
            "previous_3m_order_count": (index % 3) + 1,
            "order_trend_difference_3m": (index % 2),
            "order_trend_ratio_3m": 1.1,
            "return_quantity": 0.0,
            "return_value": 0.0,
            "negative_adjustment_rate": 0.0,
            "snapshot_month": pd.to_datetime(date).month,
            "inactive_days": (index % 15) + 1,
            "country": np.where(index % 2, "UK", "France"),
            "activity_3m_order_count": (index % 4) + 1,
            "active_3m": 1,
            "activity_6m_order_count": (index % 5) + 1,
            "active_6m": 1,
            "churn_90d": (index % 2),
            "repurchased_90d": 1 - (index % 2),
            "future_net_spend_90d": (index - 45).astype(float) * 10,
        })

    def test_value_features_exclude_target_labels_and_identifiers(self):
        columns = value_feature_columns(self.train)
        self.assertNotIn("future_net_spend_90d", columns)
        self.assertNotIn("churn_90d", columns)
        self.assertNotIn("repurchased_90d", columns)
        self.assertNotIn("CustomerID", columns)
        self.assertNotIn("snapshot_date", columns)

    def test_training_mean_and_median_only_depend_on_training_target(self):
        original = training_baseline_predictions(self.train.future_net_spend_90d, 5, 4)
        modified_evaluation_targets = self.validation.future_net_spend_90d + 1_000_000
        changed = training_baseline_predictions(self.train.future_net_spend_90d, len(modified_evaluation_targets), len(self.test))
        np.testing.assert_allclose(original["mean_baseline"][0], changed["mean_baseline"][0][:5])
        np.testing.assert_allclose(original["median_baseline"][1], changed["median_baseline"][1][:4])
        self.assertEqual(original["mean_baseline"][0][0], self.train.future_net_spend_90d.mean())
        self.assertEqual(original["median_baseline"][0][0], self.train.future_net_spend_90d.median())

    def test_signed_log_transform_round_trips_negative_and_nonnegative_targets(self):
        values = np.array([-1591.2, -1.0, 0.0, 1.0, 114.425, 80014.34])
        np.testing.assert_allclose(signed_expm1(signed_log1p(values)), values)
        nonnegative = np.array([0.0, 1.0, 10.0])
        np.testing.assert_allclose(signed_log1p(nonnegative), np.log1p(nonnegative))
        np.testing.assert_allclose(signed_expm1(np.log1p(nonnegative)), nonnegative)

    def test_inverse_transform_clips_extrapolation_using_training_only_bounds(self):
        class FakeModel:
            def predict(self, features):
                return np.array([-100.0, 0.0, 100.0])

        transformed_training_target = signed_log1p([-5.0, 10.0])
        predictions = inverse_transformed_predictions(FakeModel(), [None] * 3, transformed_training_target)
        np.testing.assert_allclose(predictions, [-5.0, 0.0, 10.0])

    def test_top_decile_capture_and_tie_reference(self):
        values = np.arange(1, 11, dtype=float)
        capture = top_decile_value_capture(values, values, 0.1)
        self.assertEqual(capture["selected_count"], 1)
        self.assertEqual(capture["actual_value_captured"], 10.0)
        self.assertAlmostEqual(capture["capture_pct"], 10 / 55 * 100)
        tied = top_decile_value_capture(values, np.ones(10), 0.1)
        self.assertAlmostEqual(tied["capture_pct"], 10.0)

    def test_value_experiment_returns_numeric_predictions_with_expected_rows(self):
        split = TemporalDataSplit(self.train, self.validation, self.test)
        config = ValueModelConfig(lightgbm_params={
            "n_estimators": 8, "learning_rate": 0.05, "num_leaves": 5,
            "min_child_samples": 2, "verbosity": -1,
        })
        result = evaluate_value_models(split, config)
        for partition, expected in (("validation", len(self.validation)), ("test", len(self.test))):
            predictions = result["predictions"][partition]
            self.assertEqual(len(predictions), expected)
            for name in ("training_mean", "training_median", "log_linear_rfm", "lightgbm_value"):
                self.assertTrue(pd.api.types.is_numeric_dtype(predictions[name]))
        self.assertEqual(result["metrics"]["lightgbm_value"]["test"]["rows"], len(self.test))
        self.assertNotIn("CustomerID", result["feature_columns"])
        self.assertNotIn("future_net_spend_90d", result["feature_columns"])

    def test_test_target_changes_do_not_change_fitted_value_predictions(self):
        split = TemporalDataSplit(self.train, self.validation, self.test)
        changed_test = self.test.copy()
        changed_test["future_net_spend_90d"] += 500_000
        config = ValueModelConfig(lightgbm_params={
            "n_estimators": 8, "learning_rate": 0.05, "num_leaves": 5,
            "min_child_samples": 2, "verbosity": -1,
        })
        first = evaluate_value_models(split, config)
        second = evaluate_value_models(TemporalDataSplit(self.train, self.validation, changed_test), config)
        for name in first["predictions"]["test"].columns:
            if name not in {"future_net_spend_90d", "CustomerID", "snapshot_date"}:
                np.testing.assert_allclose(first["predictions"]["test"][name], second["predictions"]["test"][name])


if __name__ == "__main__":
    unittest.main()
