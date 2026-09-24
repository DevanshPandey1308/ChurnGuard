import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from churnguard.classical_clv import (
    ClassicalCLVModel,
    build_customer_summary,
    build_order_transactions,
    gamma_gamma_eligibility,
)
from churnguard.clv_config import ClassicalCLVConfig
from churnguard.cleaning import clean_transactions


def cleaned(rows):
    return clean_transactions(pd.DataFrame(rows, columns=[
        "InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "StockCode"
    ]))


class ClassicalCLVTests(unittest.TestCase):
    def setUp(self):
        self.tx = cleaned([
            ("A", 1, "2010-01-01 09:00", 10, 1, "P1"),
            ("A", 2, "2010-01-01 09:00", 5, 1, "P2"),
            ("B", 2, "2010-01-02 10:00", 10, 1, "P1"),
            ("CANCEL1", -2, "2010-01-03 11:00", 10, 1, "P1"),
            ("SINGLE", 1, "2010-01-04 12:00", 30, 2, "P3"),
            ("ON_CUTOFF", 10, "2010-02-01 00:00", 1000, 1, "P4"),
            ("AFTER", 100, "2010-02-02 12:00", 10000, 1, "P5"),
        ])

    def test_orders_aggregate_lines_and_count_invoice_not_product_rows(self):
        orders = build_order_transactions(self.tx, "2010-02-01")
        cust1 = orders.loc[orders.CustomerID == 1]
        self.assertEqual(len(cust1), 2)
        self.assertEqual(cust1.order_value.tolist(), [20.0, 20.0])
        self.assertNotIn("CANCEL1", set(orders.InvoiceNo))
        self.assertNotIn("ON_CUTOFF", set(orders.InvoiceNo))

    def test_frequency_recency_and_T_definitions(self):
        summary = build_customer_summary(self.tx, "2010-02-01").set_index("CustomerID")
        customer = summary.loc[1]
        self.assertEqual(customer.frequency, 1)  # two invoices, less the first
        self.assertAlmostEqual(customer.recency, 1 + 1 / 24)
        self.assertAlmostEqual(customer["T"], 30 + 15 / 24)
        self.assertEqual(customer.monetary_value, 20.0)  # repeat-order value only

    def test_snapshot_cutoff_and_future_mutation_do_not_change_summary(self):
        before = build_customer_summary(self.tx, "2010-02-01").sort_values("CustomerID").reset_index(drop=True)
        extra = cleaned([("LATER", 50, "2010-03-01", 5000, 1, "PX")])
        after = build_customer_summary(pd.concat([self.tx, extra], ignore_index=True), "2010-02-01").sort_values("CustomerID").reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after)
        self.assertTrue((pd.to_datetime(self.tx.loc[self.tx.InvoiceNo.isin(["A", "B", "SINGLE"]), "InvoiceDate"]) < pd.Timestamp("2010-02-01")).all())

    def test_gamma_gamma_eligibility_excludes_single_and_nonpositive_repeat_values(self):
        summary = pd.DataFrame({
            "frequency": [0, 1, 1, 2],
            "monetary_value": [0.0, 25.0, -2.0, 0.0],
            "repeat_orders_positive": [True, True, False, False],
        })
        eligible, counts = gamma_gamma_eligibility(summary)
        self.assertEqual(eligible.tolist(), [False, True, False, False])
        self.assertEqual(counts["eligible_customers"], 1)
        self.assertEqual(counts["excluded_single_purchase"], 1)
        self.assertEqual(counts["excluded_nonpositive_or_invalid_repeat_monetary"], 2)

    def test_only_positive_repeat_monetary_values_are_passed_to_gamma_gamma(self):
        summary = pd.DataFrame({
            "CustomerID": [1, 2, 3, 4],
            "frequency": [1, 1, 1, 0],
            "recency": [5.0, 8.0, 6.0, 0.0],
            "T": [20.0, 22.0, 24.0, 10.0],
            "monetary_value": [30.0, -4.0, 22.0, 0.0],
            "repeat_orders_positive": [True, False, True, True],
        })
        captured = {}

        class FakeBG:
            def __init__(self, **kwargs): pass
            def fit(self, *args, **kwargs): return self

        class FakeGG:
            def __init__(self, **kwargs): pass
            def fit(self, frequency, monetary, **kwargs):
                captured["frequency"] = np.asarray(frequency)
                captured["monetary"] = np.asarray(monetary)
                return self

        with patch("lifetimes.BetaGeoFitter", FakeBG), patch("lifetimes.GammaGammaFitter", FakeGG):
            ClassicalCLVModel(ClassicalCLVConfig()).fit(summary, "2010-02-01")
        self.assertEqual(captured["frequency"].tolist(), [1, 1])
        self.assertTrue((captured["monetary"] > 0).all())

    def test_prediction_keeps_customer_ids_and_only_scores_valid_population(self):
        class FakeBG:
            def conditional_expected_number_of_purchases_up_to_time(self, horizon, frequency, recency, T):
                return np.full(len(frequency), 2.0)

        class FakeGG:
            params_ = {"p": 1.0, "q": 1.0, "v": 5.0}

        model = ClassicalCLVModel(ClassicalCLVConfig(), bgf=FakeBG(), ggf=FakeGG())
        summary = pd.DataFrame({
            "CustomerID": [21, 22, 23],
            "frequency": [0, 1, 1], "recency": [0.0, 5.0, 3.0], "T": [20.0, 30.0, 25.0],
            "monetary_value": [0.0, 10.0, -1.0],
            "repeat_orders_positive": [True, True, False],
        })
        predictions = model.predict(summary)
        self.assertEqual(predictions.CustomerID.tolist(), [22])
        self.assertEqual(len(predictions), 1)
        self.assertTrue((predictions.predicted_90d_gross_purchase_value == 30.0).all())


if __name__ == "__main__":
    unittest.main()
