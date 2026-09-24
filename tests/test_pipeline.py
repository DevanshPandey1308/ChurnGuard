import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from churnguard.cleaning import clean_transactions
from churnguard.features import build_features
from churnguard.labels import build_labels
from churnguard.config import PipelineConfig
from churnguard.pipeline import build_training_table
from churnguard.snapshots import build_snapshots


def transactions(rows):
    return clean_transactions(pd.DataFrame(rows, columns=[
        "InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"
    ]))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tx = transactions([
            ("A", 2, "2020-01-01", 10, 1),
            ("B", 1, "2020-01-10", 5, 1),
            ("INV3", 1, "2020-01-09", 7, 2),
            ("R", -1, "2020-01-15", 5, 1),
            ("D", 3, "2020-04-09", 4, 1),
            ("E", 1, "2020-04-10", 100, 1),
            ("CANCEL", -1, "2020-02-01", 10, 2),
        ])
        self.snapshot = pd.DataFrame({"CustomerID": [1, 2], "snapshot_date": pd.to_datetime(["2020-01-10", "2020-01-10"])})

    def test_labels_use_open_start_closed_end_window(self):
        labels = build_labels(self.tx, self.snapshot, 90).set_index("CustomerID")
        self.assertEqual(labels.loc[1, "repurchased_90d"], 1)
        self.assertEqual(labels.loc[1, "future_net_spend_90d"], 7.0)
        self.assertEqual(labels.loc[2, "churn_90d"], 1)

    def test_features_use_only_history_strictly_before_snapshot(self):
        features = build_features(self.tx, self.snapshot).set_index("CustomerID")
        self.assertEqual(features.loc[1, "purchase_invoice_count"], 1)
        self.assertEqual(features.loc[1, "recency_days"], 9)
        self.assertEqual(features.loc[1, "historical_net_spend"], 20.0)

    def test_future_mutations_cannot_change_features(self):
        before = build_features(self.tx, self.snapshot)
        added = transactions([("F", 1, "2020-02-01", 9999, 1)])
        after = build_features(pd.concat([self.tx, added], ignore_index=True), self.snapshot)
        pd.testing.assert_frame_equal(before, after)

    def test_transactions_on_snapshot_date_are_excluded(self):
        snap = pd.DataFrame({"CustomerID": [1], "snapshot_date": pd.to_datetime(["2020-01-10"])})
        feature = build_features(self.tx, snap).iloc[0]
        self.assertEqual(feature.purchase_invoice_count, 1)
        self.assertEqual(feature.historical_net_spend, 20.0)

    def test_trend_windows_are_relative_and_exclude_snapshot_and_future(self):
        snap = pd.DataFrame({"CustomerID": [1], "snapshot_date": pd.to_datetime(["2020-04-10"])})
        feature = build_features(self.tx, snap).iloc[0]
        # Jan 10 and later are in the trailing three months; Apr 10's 100 is excluded.
        self.assertEqual(feature["recent_3m_net_spend"], 12.0)
        self.assertEqual(feature["recent_3m_order_count"], 2)

    def test_future_label_rows_do_not_enter_historical_features(self):
        future = self.tx.loc[self.tx["InvoiceDate"] > pd.Timestamp("2020-01-10")]
        snap = pd.DataFrame({"CustomerID": [1], "snapshot_date": pd.to_datetime(["2020-01-10"])})
        feature = build_features(self.tx, snap).iloc[0]
        self.assertEqual(feature.historical_net_spend, 20.0)
        self.assertFalse(future.empty)

    def test_cleaning_classifies_cancellation_purchase_and_return(self):
        rows = self.tx.set_index("InvoiceNo")
        self.assertEqual(rows.loc["CANCEL", "transaction_type"], "cancellation")
        self.assertEqual(rows.loc["R", "transaction_type"], "return_adjustment")
        self.assertEqual(rows.loc["A", "transaction_type"], "purchase")

    def test_pipeline_merges_features_and_labels_one_to_one(self):
        config = PipelineConfig(Path("unused.csv"), ["2020-01-10", "2020-04-10"])
        table = build_training_table(self.tx, config)
        self.assertFalse(table.duplicated(["CustomerID", "snapshot_date"]).any())
        self.assertTrue({"historical_net_spend", "repurchased_90d", "churn_90d", "future_net_spend_90d"}.issubset(table.columns))
        self.assertEqual(table["snapshot_date"].nunique(), 1)
        self.assertTrue((table["snapshot_date"] == pd.Timestamp("2020-01-10")).all())

    def test_order_and_return_features_use_order_level_values(self):
        snap = pd.DataFrame({"CustomerID": [1], "snapshot_date": pd.to_datetime(["2020-01-20"])})
        feature = build_features(self.tx, snap).iloc[0]
        self.assertEqual(feature.purchase_invoice_count, 2)
        self.assertEqual(feature.average_order_value, 12.5)
        self.assertEqual(feature.return_quantity, 1.0)
        self.assertEqual(feature.return_value, 5.0)

    def test_snapshots_require_prior_purchase_and_full_horizon(self):
        eligible = build_snapshots(self.tx, ["2020-01-10", "2020-04-10"], 90)
        self.assertEqual(set(eligible["CustomerID"]), {1, 2})
        self.assertEqual(eligible["snapshot_date"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()
