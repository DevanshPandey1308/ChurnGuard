import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from churnguard.model_config import TemporalSplitConfig
from churnguard.modeling import RFM_FEATURES, feature_columns, make_xy
from churnguard.temporal import split_by_snapshot_date


class TemporalModelingTests(unittest.TestCase):
    def setUp(self):
        self.table = pd.DataFrame({
            "CustomerID": [10, 11, 10, 12],
            "snapshot_date": pd.to_datetime(["2011-03-01", "2011-04-01", "2011-05-01", "2011-06-01"]),
            "recency_days": [2, 4, 7, 9],
            "purchase_invoice_count": [3, 1, 3, 4],
            "historical_net_spend": [100, 20, 130, 180],
            "country": ["UK", "UK", "FR", "UK"],
            "churn_90d": [0, 1, 0, 1],
            "repurchased_90d": [1, 0, 1, 0],
            "future_net_spend_90d": [5, 0, 10, 0],
        })
        self.config = TemporalSplitConfig()

    def test_split_uses_configured_dates_without_customer_randomization(self):
        parts = split_by_snapshot_date(self.table, self.config)
        self.assertEqual(parts.train["snapshot_date"].dt.strftime("%Y-%m-%d").tolist(), ["2011-03-01", "2011-04-01"])
        self.assertEqual(parts.validation["snapshot_date"].dt.strftime("%Y-%m-%d").tolist(), ["2011-05-01"])
        self.assertEqual(parts.test["snapshot_date"].dt.strftime("%Y-%m-%d").tolist(), ["2011-06-01"])
        self.assertIn(10, set(parts.train.CustomerID) & set(parts.validation.CustomerID))

    def test_snapshot_dates_do_not_overlap(self):
        parts = split_by_snapshot_date(self.table, self.config)
        date_sets = [set(part.snapshot_date) for part in (parts.train, parts.validation, parts.test)]
        self.assertFalse(date_sets[0] & date_sets[1])
        self.assertFalse(date_sets[0] & date_sets[2])
        self.assertFalse(date_sets[1] & date_sets[2])

    def test_repeated_split_is_reproducible_and_keeps_rows(self):
        first = split_by_snapshot_date(self.table, self.config)
        second = split_by_snapshot_date(self.table, self.config)
        for a, b in zip((first.train, first.validation, first.test), (second.train, second.validation, second.test)):
            pd.testing.assert_frame_equal(a, b)

    def test_features_exclude_all_future_derived_columns_and_target(self):
        columns = feature_columns(self.table)
        self.assertTrue(set(RFM_FEATURES).issubset(columns))
        self.assertFalse({"churn_90d", "repurchased_90d", "future_net_spend_90d"} & set(columns))
        with self.assertRaises(ValueError):
            make_xy(self.table, columns=["recency_days", "future_net_spend_90d"])

    def test_target_construction_uses_churn_90d(self):
        _, target = make_xy(self.table, target_column="churn_90d")
        self.assertEqual(target.tolist(), [0, 1, 0, 1])

    def test_temporal_config_rejects_overlap_or_nonchronological_dates(self):
        with self.assertRaises(ValueError):
            TemporalSplitConfig(train_dates=("2011-03-01",), validation_dates=("2011-03-01",), test_dates=("2011-04-01",))
        with self.assertRaises(ValueError):
            TemporalSplitConfig(train_dates=("2011-04-01",), validation_dates=("2011-03-01",), test_dates=("2011-06-01",))


if __name__ == "__main__":
    unittest.main()
