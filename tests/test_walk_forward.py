import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from churnguard.model_config import ModelConfig
from churnguard.walk_forward import (
    MODEL_NAMES,
    RESULT_COLUMNS,
    build_walk_forward_folds,
    evaluate_walk_forward,
)


DATES = ("2011-03-01", "2011-04-01", "2011-05-01", "2011-06-01")


def snapshot_table(rows_per_date=80):
    frames = []
    customer_index = np.arange(rows_per_date)
    for date_index, date in enumerate(DATES):
        churn = ((customer_index + date_index) % 2).astype(int)
        frames.append(pd.DataFrame({
            "CustomerID": customer_index + 1000,
            "snapshot_date": pd.to_datetime([date] * rows_per_date),
            "recency_days": 1 + customer_index % 35 + date_index,
            "purchase_invoice_count": 1 + customer_index % 8,
            "historical_net_spend": 10.0 + customer_index * (1 + date_index),
            "country": np.where(customer_index % 2, "UK", "France"),
            "churn_90d": churn,
            "repurchased_90d": 1 - churn,
            "future_net_spend_90d": customer_index * 3.0,
        }))
    return pd.concat(frames, ignore_index=True)


def small_model_config():
    return ModelConfig(lightgbm_params={
        "n_estimators": 8,
        "learning_rate": 0.05,
        "num_leaves": 5,
        "min_child_samples": 2,
        "verbosity": -1,
    })


class WalkForwardTests(unittest.TestCase):
    def setUp(self):
        self.table = snapshot_table()

    def test_exactly_two_folds_with_chronological_expanding_train(self):
        folds = build_walk_forward_folds(self.table)
        self.assertEqual(len(folds), 2)
        self.assertEqual(folds[0].train_snapshot_dates, ("2011-03-01",))
        self.assertEqual(folds[0].validation_date, "2011-04-01")
        self.assertEqual(folds[1].train_snapshot_dates, ("2011-03-01", "2011-04-01"))
        self.assertEqual(folds[1].validation_date, "2011-05-01")
        for fold in folds:
            self.assertLess(max(fold.train_snapshot_dates), fold.validation_date)

    def test_final_latest_snapshot_is_never_training_or_validation(self):
        folds = build_walk_forward_folds(self.table)
        for fold in folds:
            self.assertNotIn("2011-06-01", fold.train_snapshot_dates)
            self.assertNotEqual(fold.validation_date, "2011-06-01")
            self.assertNotIn(pd.Timestamp("2011-06-01"), set(fold.train.snapshot_date))
            self.assertNotIn(pd.Timestamp("2011-06-01"), set(fold.validation.snapshot_date))

    def test_fold_generation_is_deterministic_and_preserves_row_order(self):
        first = build_walk_forward_folds(self.table)
        second = build_walk_forward_folds(self.table)
        self.assertEqual([(f.fold_id, f.train_snapshot_dates, f.validation_date) for f in first],
                         [(f.fold_id, f.train_snapshot_dates, f.validation_date) for f in second])
        for left, right in zip(first, second):
            pd.testing.assert_frame_equal(left.train, right.train)
            pd.testing.assert_frame_equal(left.validation, right.validation)

    def test_current_four_dates_produce_expected_row_counts(self):
        folds = build_walk_forward_folds(self.table)
        self.assertEqual([(len(f.train), len(f.validation)) for f in folds], [(80, 80), (160, 80)])

    def test_fold_partition_never_places_current_validation_label_in_train(self):
        folds = build_walk_forward_folds(self.table)
        for fold in folds:
            self.assertTrue((fold.train.snapshot_date < pd.Timestamp(fold.validation_date)).all())
            self.assertTrue((fold.validation.snapshot_date == pd.Timestamp(fold.validation_date)).all())
            train_dates = set(fold.train.snapshot_date.dt.strftime("%Y-%m-%d"))
            self.assertEqual(train_dates, set(fold.train_snapshot_dates))
            self.assertNotIn(fold.validation_date, train_dates)
        changed_april = self.table.copy()
        april = changed_april.snapshot_date == pd.Timestamp("2011-04-01")
        changed_april.loc[april, "churn_90d"] = 1
        april_folds = build_walk_forward_folds(changed_april)
        pd.testing.assert_frame_equal(folds[0].train, april_folds[0].train)
        self.assertTrue((april_folds[0].validation.churn_90d == 1).all())

        changed_may = self.table.copy()
        may = changed_may.snapshot_date == pd.Timestamp("2011-05-01")
        changed_may.loc[may, "churn_90d"] = 1
        may_folds = build_walk_forward_folds(changed_may)
        pd.testing.assert_frame_equal(folds[1].train, may_folds[1].train)
        self.assertTrue((may_folds[1].validation.churn_90d == 1).all())

    def test_model_result_schema_and_all_four_models(self):
        output = evaluate_walk_forward(self.table, small_model_config())
        result = output["results"]
        self.assertEqual(list(result.columns), RESULT_COLUMNS)
        self.assertEqual(len(result), 2 * len(MODEL_NAMES))
        self.assertEqual(set(result.model), set(MODEL_NAMES))
        self.assertEqual(output["final_test_date_reserved"], "2011-06-01")
        self.assertEqual(output["available_snapshot_dates"], DATES)
        self.assertNotIn("2011-06-01", set(result.validation_date))

    def test_metric_ranges_and_aggregate_schema(self):
        output = evaluate_walk_forward(self.table, small_model_config())
        result = output["results"]
        self.assertTrue(result.pr_auc.between(0, 1).all())
        self.assertTrue(result.lift.ge(0).all())
        self.assertTrue(result.brier_score.between(0, 1).all())
        self.assertTrue(result.validation_churn_rate.between(0, 1).all())
        self.assertTrue((result.train_rows > 0).all())
        self.assertTrue((result.validation_rows > 0).all())
        self.assertEqual(output["aggregate"].folds_evaluated.tolist(), [2, 2, 2, 2])
        self.assertEqual(set(output["aggregate"].columns), {
            "model", "folds_evaluated", "mean_pr_auc", "std_pr_auc", "mean_lift", "std_lift",
            "mean_brier_score", "std_brier_score",
        })

    def test_mutating_final_test_features_or_labels_cannot_change_walk_forward(self):
        original = evaluate_walk_forward(self.table, small_model_config())
        changed_table = self.table.copy()
        june = changed_table.snapshot_date == pd.Timestamp("2011-06-01")
        changed_table.loc[june, "churn_90d"] = 1 - changed_table.loc[june, "churn_90d"]
        changed_table.loc[june, "recency_days"] += 1_000_000
        changed_table.loc[june, "historical_net_spend"] *= -1_000_000
        changed = evaluate_walk_forward(changed_table, small_model_config())
        pd.testing.assert_frame_equal(original["results"], changed["results"])
        pd.testing.assert_frame_equal(original["aggregate"], changed["aggregate"])

    def test_insufficient_snapshot_dates_fail_transparently(self):
        with self.assertRaisesRegex(ValueError, "At least three snapshot dates"):
            build_walk_forward_folds(self.table[self.table.snapshot_date < pd.Timestamp("2011-05-01")])


if __name__ == "__main__":
    unittest.main()
