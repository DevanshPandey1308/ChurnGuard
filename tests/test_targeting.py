import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from churnguard.targeting import (
    add_scenario_values,
    evaluate_targeting_partitions,
    expected_value_threshold_table,
    scenario_sensitivity,
    score_customers,
    top_decile_actual_value_capture,
)
from churnguard.targeting_config import TargetingConfig


def prediction_frame(ids, date, churn, value):
    return pd.DataFrame({
        "CustomerID": ids,
        "snapshot_date": pd.to_datetime([date] * len(ids)),
        "calibrated_churn_probability": churn,
        "predicted_90d_value": value,
    })


class TargetingTests(unittest.TestCase):
    def test_risk_weighted_value_is_component_product(self):
        scored = score_customers(prediction_frame([10], "2011-05-01", [0.4], [1000]))
        self.assertAlmostEqual(scored.risk_weighted_value.iloc[0], 400)
        self.assertAlmostEqual(scored.calibrated_churn_probability.iloc[0], 0.4)
        self.assertAlmostEqual(scored.predicted_90d_value.iloc[0], 1000)

    def test_scenario_example_expected_recovery_and_net_value(self):
        scored = score_customers(prediction_frame([10], "2011-05-01", [0.4], [1000]))
        scenario = add_scenario_values(scored, 50, 0.25, 0.80)
        self.assertAlmostEqual(scenario.expected_recovered_value_under_scenario.iloc[0], 80)
        self.assertAlmostEqual(scenario.expected_net_value_under_scenario.iloc[0], 30)

    def test_negative_predicted_value_is_not_treated_as_recoverable(self):
        scored = score_customers(prediction_frame([10], "2011-05-01", [0.8], [-100]))
        scenario = add_scenario_values(scored, 25, 0.5, 0.5)
        self.assertAlmostEqual(scenario.expected_recovered_value_under_scenario.iloc[0], 0)
        self.assertAlmostEqual(scenario.expected_net_value_under_scenario.iloc[0], -25)

    def test_deterministic_ties_use_customer_id_not_input_row_order(self):
        first = prediction_frame([30, 10, 20], "2011-05-01", [0.5] * 3, [100] * 3)
        second = first.iloc[::-1].reset_index(drop=True)
        scored_first = score_customers(first).sort_values("CustomerID")
        scored_second = score_customers(second).sort_values("CustomerID")
        for rank in ("churn_rank", "value_rank", "risk_value_rank"):
            np.testing.assert_array_equal(scored_first[rank], scored_second[rank])
        ranks = score_customers(first).set_index("CustomerID").risk_value_rank.to_dict()
        self.assertEqual(ranks, {10: 1, 20: 2, 30: 3})

    def test_top_decile_uses_ceil_count_and_reports_population_comparison(self):
        scored = score_customers(prediction_frame(
            list(range(1, 12)), "2011-06-01", [0.5] * 11, list(range(1, 12))
        ))
        outcomes = pd.DataFrame({
            "CustomerID": list(range(1, 12)),
            "snapshot_date": pd.to_datetime(["2011-06-01"] * 11),
            "future_net_spend_90d": list(range(1, 12)),
        })
        result = top_decile_actual_value_capture(scored, outcomes, "value_rank", 0.10)
        self.assertEqual(result["number_targeted"], 2)
        self.assertAlmostEqual(result["percentage_targeted"], 2 / 11 * 100)
        self.assertEqual(result["actual_future_net_spend_captured"], 21)
        self.assertAlmostEqual(result["average_actual_future_net_spend_per_targeted_customer"], 10.5)
        self.assertAlmostEqual(result["overall_average_actual_future_net_spend"], 6)

    def test_threshold_grid_and_target_counts(self):
        config = TargetingConfig(targeting_thresholds=(0.25, 0.5, 1.0))
        scored = score_customers(prediction_frame(
            list(range(1, 9)), "2011-05-01", np.linspace(0.1, 0.8, 8), np.arange(1, 9) * 10,
        ))
        table = expected_value_threshold_table(scored, config)
        self.assertEqual(table.target_fraction.tolist(), [0.25, 0.5, 1.0])
        self.assertEqual(table.number_targeted.tolist(), [2, 4, 8])
        self.assertEqual(table.percentage_targeted.tolist(), [25.0, 50.0, 100.0])

    def test_future_labels_are_rejected_as_scoring_inputs(self):
        input_frame = prediction_frame([10], "2011-05-01", [0.4], [1000])
        input_frame["future_net_spend_90d"] = [999999]
        with self.assertRaisesRegex(ValueError, "Future outcomes"):
            score_customers(input_frame)

    def test_scenario_sensitivity_reports_small_configured_grid(self):
        config = TargetingConfig(
            sensitivity_acceptance_rates=(0.2, 0.4),
            sensitivity_retained_value_fractions=(0.5, 1.0),
        )
        scored = score_customers(prediction_frame(
            list(range(1, 11)), "2011-05-01", [0.5] * 10, [100] * 10,
        ))
        table = scenario_sensitivity(scored, config)
        self.assertEqual(len(table), 4)
        self.assertEqual(set(table.assumed_acceptance_rate), {0.2, 0.4})
        self.assertEqual(set(table.assumed_retained_value_fraction), {0.5, 1.0})
        self.assertEqual(table.number_targeted.tolist(), [1] * 4)

    def test_validation_test_separation_and_customer_counts(self):
        validation = prediction_frame([1, 2], "2011-05-01", [0.2, 0.8], [100, 200])
        test = prediction_frame([1, 3, 4], "2011-06-01", [0.4, 0.7, 0.3], [300, 100, 50])
        validation_actual = pd.DataFrame({
            "CustomerID": [1, 2], "snapshot_date": pd.to_datetime(["2011-05-01"] * 2),
            "future_net_spend_90d": [10, 20],
        })
        test_actual = pd.DataFrame({
            "CustomerID": [1, 3, 4], "snapshot_date": pd.to_datetime(["2011-06-01"] * 3),
            "future_net_spend_90d": [100, 200, 300],
        })
        result = evaluate_targeting_partitions(validation, test, validation_actual, test_actual)
        self.assertEqual(result["validation"]["customer_count"], 2)
        self.assertEqual(result["test"]["customer_count"], 3)
        self.assertIn("expected_value_threshold_table", result["validation"])
        self.assertNotIn("expected_value_threshold_table", result["test"])
        self.assertEqual(len(result["test"]["predictions"]), 3)
        with self.assertRaisesRegex(ValueError, "distinct non-empty"):
            evaluate_targeting_partitions(validation, validation, validation_actual, validation_actual)

    def test_scenario_formula_has_no_actual_label_dependency(self):
        # The scoring API's exact input contract contains predictions and IDs only.
        self.assertEqual(
            set(score_customers(prediction_frame([7], "2011-05-01", [0.4], [1000])).columns),
            {"CustomerID", "snapshot_date", "calibrated_churn_probability", "predicted_90d_value",
             "risk_weighted_value", "churn_rank", "value_rank", "risk_value_rank"},
        )


if __name__ == "__main__":
    unittest.main()
