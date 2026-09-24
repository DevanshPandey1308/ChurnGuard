import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from churnguard.calibration import (
    SigmoidProbabilityCalibrator,
    calibration_features,
    probability_metrics,
    reliability_table,
)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.validation_probability = np.array([0.05, 0.12, 0.2, 0.3, 0.45, 0.55, 0.7, 0.82, 0.91, 0.97])
        self.validation_target = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.test_probability = np.array([0.08, 0.25, 0.5, 0.75, 0.94])
        self.test_target = np.array([0, 0, 1, 0, 1])

    def test_calibrated_probabilities_are_bounded_and_preserve_row_count(self):
        calibrator = SigmoidProbabilityCalibrator(17).fit(self.validation_probability, self.validation_target)
        output = calibrator.predict(self.test_probability)
        self.assertEqual(len(output), len(self.test_probability))
        self.assertTrue(np.all((output >= 0) & (output <= 1)))

    def test_brier_score_calculation(self):
        metrics = probability_metrics([0, 1], [0.2, 0.8])
        self.assertAlmostEqual(metrics["brier_score"], 0.04)

    def test_reliability_table_columns_and_count(self):
        table = reliability_table(self.validation_target, self.validation_probability, n_bins=5)
        self.assertEqual(set(table.columns), {
            "probability_bin", "mean_predicted_probability", "observed_churn_rate", "n_observations"
        })
        self.assertLessEqual(len(table), 5)
        self.assertEqual(int(table.n_observations.sum()), len(self.validation_target))

    def test_reliability_table_handles_constant_scores(self):
        table = reliability_table([0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], n_bins=5)
        self.assertEqual(len(table), 1)
        self.assertEqual(int(table.n_observations.iloc[0]), 4)

    def test_calibration_features_contain_only_raw_model_score(self):
        features = calibration_features(self.validation_probability)
        self.assertEqual(list(features.columns), ["raw_logit_probability"])
        self.assertFalse({"churn_90d", "repurchased_90d", "future_net_spend_90d"} & set(features.columns))

    def test_calibrator_fit_api_accepts_only_calibration_labels(self):
        parameters = list(inspect.signature(SigmoidProbabilityCalibrator.fit).parameters)
        self.assertEqual(parameters, ["self", "raw_probabilities", "calibration_labels"])
        calibrator = SigmoidProbabilityCalibrator(17).fit(self.validation_probability, self.validation_target)
        self.assertEqual(calibrator.n_calibration_rows_, len(self.validation_target))
        # Test labels have no path into fit; only test probabilities are passed to transform.
        self.assertNotIn("test_labels", parameters)
        self.assertEqual(len(calibrator.predict(self.test_probability)), len(self.test_target))

    def test_repeated_calibration_is_deterministic(self):
        first = SigmoidProbabilityCalibrator(17).fit(self.validation_probability, self.validation_target).predict(self.test_probability)
        second = SigmoidProbabilityCalibrator(17).fit(self.validation_probability, self.validation_target).predict(self.test_probability)
        np.testing.assert_allclose(first, second)


if __name__ == "__main__":
    unittest.main()
