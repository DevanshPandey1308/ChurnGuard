import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd

from churnguard.reporting.powerbi_export import (
    SCORE_COLUMNS,
    build_raw_feature_population,
    export_powerbi_data,
    prepare_customer_scores,
    prepare_global_importance,
)

FEATURES = [
    "recency_days", "purchase_invoice_count", "purchase_line_count", "units_purchased",
    "historical_net_spend", "log1p_purchase_invoice_count", "signed_log1p_historical_net_spend",
    "tenure_days", "average_order_value", "average_items_per_order", "distinct_products",
    "mean_inter_purchase_gap_days", "std_inter_purchase_gap_days", "recent_3m_net_spend",
    "previous_3m_net_spend", "spend_trend_difference_3m", "spend_trend_ratio_3m",
    "recent_3m_order_count", "previous_3m_order_count", "order_trend_difference_3m",
    "order_trend_ratio_3m", "return_quantity", "return_value", "negative_adjustment_rate",
    "snapshot_month", "inactive_days", "country", "activity_3m_order_count", "active_3m",
    "activity_6m_order_count", "active_6m",
]


class PowerBIExportTests(unittest.TestCase):
    def setUp(self):
        self.features = ["country", "recency_days"]
        self.inputs = pd.DataFrame({
            "CustomerID": [101, 102],
            "snapshot_date": ["2011-06-01", "2011-06-01"],
            "country": ["United Kingdom", "France"],
            "recency_days": [10.0, 20.0],
        })
        self.scored = pd.DataFrame({
            "CustomerID": [101, 102],
            "snapshot_date": ["2011-06-01T00:00:00", "2011-06-01T00:00:00"],
            "churn_probability": [0.2, 0.7],
            "predicted_90d_value": [100.0, 50.0],
            "risk_weighted_value": [20.0, 35.0],
            "churn_rank": [2, 1],
            "value_rank": [1, 2],
            "risk_value_rank": [2, 1],
        })

    def test_customer_table_joins_country_and_preserves_api_score_outputs(self):
        table = prepare_customer_scores(self.inputs, self.scored, self.features)
        self.assertEqual(table.columns.tolist(), [
            "CustomerID", "snapshot_date", "country", "churn_probability",
            "predicted_90d_value", "risk_weighted_value", "churn_rank", "value_rank", "risk_value_rank",
        ])
        self.assertEqual(table.country.tolist(), ["United Kingdom", "France"])
        self.assertEqual(table.risk_weighted_value.tolist(), [20.0, 35.0])

    def test_join_rejects_missing_or_extra_customer_scores(self):
        with self.assertRaisesRegex(ValueError, "do not match"):
            prepare_customer_scores(self.inputs, self.scored.iloc[:1], self.features)

    def test_join_checks_risk_weighted_value_contract(self):
        incorrect = self.scored.copy()
        incorrect.loc[0, "risk_weighted_value"] = 99
        with self.assertRaisesRegex(ValueError, "does not match"):
            prepare_customer_scores(self.inputs, incorrect, self.features)

    def test_global_importance_uses_only_existing_model_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.csv"
            destination = Path(temporary) / "nested" / "importance.csv"
            pd.DataFrame({
                "feature": ["recency_days", "country"],
                "mean_abs_shap": [0.4, 0.1],
                "mean_shap": [-0.2, 0.01],
                "rank": [1, 2],
            }).to_csv(source, index=False)
            result = prepare_global_importance(source, destination, self.features)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertEqual(result.feature.tolist(), ["recency_days", "country"])

    def _raw_fixture(self, directory):
        raw_path = Path(directory) / "transactions.csv"
        raw = pd.DataFrame({
            "InvoiceNo": ["A100", "A101", "A102"],
            "StockCode": ["P1", "P1", "P2"],
            "Description": ["product", "product", "product"],
            "Quantity": [2, 10, 4],
            "InvoiceDate": ["2011-02-28 12:00", "2011-07-01 12:00", "2011-09-29 12:00"],
            "UnitPrice": [10.0, 100.0, 10.0],
            "CustomerID": ["123", "123", "123"],
            "Country": ["United Kingdom"] * 3,
        })
        raw.to_csv(raw_path, index=False)
        return raw_path, raw

    def _model_artifacts(self, directory):
        model_dir = Path(directory) / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        numeric = [name for name in FEATURES if name != "country"]
        metadata = {
            "feature_columns": FEATURES,
            "training_snapshot_dates": ["2011-03-01", "2011-04-01"],
            "validation_snapshot_dates": ["2011-05-01"],
            "churn_preprocessing": {
                "feature_columns": FEATURES,
                "feature_types": {name: ("category" if name == "country" else "number") for name in FEATURES},
                "numeric_medians": {name: 0.0 for name in numeric},
                "categorical_values": {"country": ["United Kingdom"]},
            },
        }
        (model_dir / "metadata.json").write_text(__import__("json").dumps(metadata), encoding="utf-8")
        return model_dir

    def test_raw_mode_uses_canonical_cleaning_snapshots_features_and_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_path, raw = self._raw_fixture(temporary)
            model_dir = self._model_artifacts(temporary)
            result = build_raw_feature_population(raw_path, model_dir)
            table = result["features"]
            self.assertEqual(result["raw_transaction_rows"], len(raw))
            self.assertEqual(result["snapshot_row_count"], 4)
            self.assertEqual(result["snapshot_counts"], {
                "2011-03-01": 1, "2011-04-01": 1, "2011-05-01": 1,
                "2011-06-01": 1,
            })
            self.assertEqual(set(table.columns), {"CustomerID", "snapshot_date", *FEATURES})
            self.assertEqual(table.duplicated(["CustomerID", "snapshot_date"]).sum(), 0)
            self.assertTrue(pd.api.types.is_string_dtype(table["country"]))
            self.assertTrue(table["country"].eq("United Kingdom").all())
            self.assertFalse({"churn_90d", "repurchased_90d", "future_net_spend_90d"} & set(table.columns))
            self.assertFalse(table[FEATURES].isna().any().any())

    def test_raw_features_are_point_in_time_and_do_not_use_later_spend(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_path, raw = self._raw_fixture(temporary)
            model_dir = self._model_artifacts(temporary)
            first = build_raw_feature_population(raw_path, model_dir)["features"]
            changed = raw.copy()
            changed.loc[changed["InvoiceNo"] == "A101", "UnitPrice"] = 9999.0
            changed_path = Path(temporary) / "changed.csv"
            changed.to_csv(changed_path, index=False)
            second = build_raw_feature_population(changed_path, model_dir)["features"]
            march_first = first.loc[first.snapshot_date.eq(pd.Timestamp("2011-03-01")), "historical_net_spend"].item()
            march_second = second.loc[second.snapshot_date.eq(pd.Timestamp("2011-03-01")), "historical_net_spend"].item()
            self.assertEqual(march_first, march_second)

    def test_flat_and_raw_export_modes_produce_powerbi_schema_without_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_path, _ = self._raw_fixture(temporary)
            model_dir = self._model_artifacts(temporary)
            raw_population = build_raw_feature_population(raw_path, model_dir)["features"]
            importance = Path(temporary) / "importance.csv"
            pd.DataFrame({
                "feature": ["country"], "mean_abs_shap": [0.5], "mean_shap": [-0.1], "rank": [1],
            }).to_csv(importance, index=False)
            api_scores = pd.DataFrame({
                "CustomerID": raw_population.CustomerID,
                "snapshot_date": raw_population.snapshot_date.astype(str),
                "churn_probability": [0.2] * len(raw_population),
                "predicted_90d_value": [100.0] * len(raw_population),
                "risk_weighted_value": [20.0] * len(raw_population),
                "churn_rank": range(1, len(raw_population) + 1),
                "value_rank": range(1, len(raw_population) + 1),
                "risk_value_rank": range(1, len(raw_population) + 1),
            })
            with patch("churnguard.reporting.powerbi_export.score_raw_population", return_value=api_scores):
                raw_out = export_powerbi_data(
                    None, Path(temporary) / "raw-output", model_dir, importance,
                    raw_transactions=raw_path,
                )
            flat_path = Path(temporary) / "flat.csv"
            flat = raw_population.iloc[:1].copy()
            flat.to_csv(flat_path, index=False)
            flat_scores = api_scores.iloc[:1].copy()
            with patch("churnguard.reporting.powerbi_export.score_via_batch_api", return_value=flat_scores):
                flat_out = export_powerbi_data(flat_path, Path(temporary) / "flat-output", model_dir, importance)
            raw_table = pd.read_csv(raw_out["customer_scores_path"])
            flat_table = pd.read_csv(flat_out["customer_scores_path"])
            self.assertEqual(raw_table.columns.tolist(), SCORE_COLUMNS)
            self.assertEqual(len(raw_table), 4)
            self.assertEqual(flat_table.columns.tolist(), SCORE_COLUMNS)
            self.assertEqual(len(flat_table), 1)
            self.assertFalse({"churn_90d", "repurchased_90d", "future_net_spend_90d"} & set(raw_table.columns))


if __name__ == "__main__":
    unittest.main()
