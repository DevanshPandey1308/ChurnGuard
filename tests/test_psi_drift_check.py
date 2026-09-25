import unittest

import numpy as np
import pandas as pd

from churnguard.monitoring.psi_drift_check import (
    PSIConfig,
    calculate_psi,
    compare_feature_populations,
)


class PSIDriftTests(unittest.TestCase):
    def setUp(self):
        self.config = PSIConfig()

    def test_identical_numeric_distributions_have_zero_psi(self):
        values = pd.Series(np.arange(100, dtype=float))
        self.assertAlmostEqual(calculate_psi(values, values.copy(), feature="x", feature_type="number"), 0.0)

    def test_small_change_is_lower_than_strong_numeric_shift(self):
        reference = pd.Series(np.arange(100, dtype=float))
        small = pd.Series(np.arange(100, dtype=float) + 1)
        strong = pd.Series(np.arange(100, dtype=float) + 100)
        small_psi = calculate_psi(reference, small, feature="x", feature_type="number")
        strong_psi = calculate_psi(reference, strong, feature="x", feature_type="number")
        self.assertLess(small_psi, strong_psi)

    def test_shift_crosses_default_threshold_and_is_flagged(self):
        reference = pd.DataFrame({"recency_days": np.zeros(100)})
        current = pd.DataFrame({"recency_days": np.full(100, 100.0)})
        result = compare_feature_populations(reference, current, ["recency_days"], {"recency_days": "number"})
        self.assertGreater(result.loc[0, "psi"], 0.20)
        self.assertEqual(result.loc[0, "status"], "drift")
        self.assertTrue(result.loc[0, "drift_flag"])

    def test_missing_required_features_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Missing required current features"):
            compare_feature_populations(pd.DataFrame({"x": [1]}), pd.DataFrame(), ["x"], {"x": "number"})

    def test_only_explicit_model_features_are_returned(self):
        reference = pd.DataFrame({"CustomerID": [1, 2], "churn_90d": [0, 1], "x": [1, 2]})
        current = pd.DataFrame({"CustomerID": [3, 4], "expected_value": [20, 30], "x": [1, 2]})
        result = compare_feature_populations(reference, current, ["x"], {"x": "number"})
        self.assertEqual(result.feature.tolist(), ["x"])

    def test_identifier_label_and_prediction_fields_cannot_be_monitored(self):
        frame = pd.DataFrame({"CustomerID": [1], "x": [1]})
        with self.assertRaisesRegex(ValueError, "Non-feature fields"):
            compare_feature_populations(frame, frame, ["CustomerID"], {"CustomerID": "number"})

    def test_missing_values_and_zero_bins_are_finite(self):
        reference = pd.Series([1.0, 1.0, np.nan, 1.0])
        current = pd.Series([2.0, 2.0, np.nan, np.nan])
        value = calculate_psi(reference, current, feature="x", feature_type="number")
        self.assertTrue(np.isfinite(value))
        self.assertGreater(value, 0)

    def test_categorical_unseen_level_is_explicit_and_stable(self):
        reference = pd.Series(["GB", "GB", "FR", None])
        current = pd.Series(["GB", "US", "FR", None])
        first = calculate_psi(reference, current, feature="country", feature_type="category")
        second = calculate_psi(reference, current, feature="country", feature_type="category")
        self.assertGreater(first, 0)
        self.assertEqual(first, second)

    def test_result_order_and_values_are_deterministic(self):
        reference = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "a"]})
        current = reference.copy()
        a = compare_feature_populations(reference, current, ["y", "x"], {"x": "number", "y": "category"})
        b = compare_feature_populations(reference, current, ["y", "x"], {"x": "number", "y": "category"})
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a.feature.tolist(), ["y", "x"])

    def test_threshold_flag_uses_strict_greater_than(self):
        reference = pd.DataFrame({"x": [1, 2, 3]})
        current = reference.copy()
        result = compare_feature_populations(reference, current, ["x"], {"x": "number"}, PSIConfig(threshold=0.0))
        self.assertEqual(result.loc[0, "psi"], 0.0)
        self.assertFalse(result.loc[0, "drift_flag"])

    def test_non_numeric_data_does_not_turn_into_missing_silently(self):
        with self.assertRaisesRegex(ValueError, "non-numeric values"):
            calculate_psi(pd.Series([1, "bad"]), pd.Series([1, 2]), feature="x", feature_type="number")

    def test_population_must_not_be_empty(self):
        with self.assertRaisesRegex(ValueError, "must contain rows"):
            compare_feature_populations(pd.DataFrame({"x": []}), pd.DataFrame({"x": [1]}), ["x"], {"x": "number"})


if __name__ == "__main__":
    unittest.main()
