import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from churnguard.explainability import (
    explain_lightgbm_churn,
    prepare_explanation_matrix,
    select_representative_rows,
)
from churnguard.model_config import ModelConfig
from churnguard.modeling import make_lightgbm, prepare_lightgbm_features


class ExplainabilityTests(unittest.TestCase):
    def setUp(self):
        count = 80
        index = np.arange(count)
        self.train = pd.DataFrame({
            "recency_days": index % 31 + 1,
            "purchase_invoice_count": index % 8 + 1,
            "historical_net_spend": index * 9.0 + 10,
            "country": np.where(index % 2, "UK", "France"),
        })
        self.target = ((index % 8) > 3).astype(int)
        train_matrix = prepare_lightgbm_features(self.train)
        self.model = make_lightgbm(ModelConfig(lightgbm_params={
            "n_estimators": 12,
            "learning_rate": 0.05,
            "num_leaves": 5,
            "min_child_samples": 2,
            "verbosity": -1,
        }))
        self.model.fit(train_matrix, self.target, categorical_feature="auto")
        self.june = pd.DataFrame({
            "recency_days": [1, 10, 20, 30, 12, 17],
            "purchase_invoice_count": [8, 3, 1, 2, 4, 6],
            "historical_net_spend": [900, 300, 20, 10, 500, 700],
            "country": ["UK", "France", "UK", "France", "Other", "UK"],
        })
        self.june_meta = pd.DataFrame({
            "CustomerID": [11, 12, 13, 14, 15, 16],
            "snapshot_date": pd.to_datetime(["2011-06-01"] * 6),
            "raw_churn_probability": [0.9, 0.1, 0.5, 0.2, 0.5, 0.8],
            "calibrated_churn_probability": [0.85, 0.2, 0.48, 0.3, 0.52, 0.75],
        })

    def test_feature_order_and_country_preprocessing_match_fitted_model(self):
        matrix = prepare_explanation_matrix(self.model, self.train, self.june)
        self.assertEqual(list(matrix.columns), list(self.model.feature_name_))
        self.assertTrue(isinstance(matrix["country"].dtype, pd.CategoricalDtype))
        with self.assertRaisesRegex(ValueError, "identical column ordering"):
            prepare_explanation_matrix(self.model, self.train, self.june[self.june.columns[::-1]])

    def test_target_and_customer_identifier_columns_are_rejected(self):
        unsafe = self.train.assign(churn_90d=self.target)
        unsafe_june = self.june.assign(churn_90d=np.zeros(len(self.june), dtype=int))
        with self.assertRaisesRegex(ValueError, "cannot be included in SHAP features"):
            prepare_explanation_matrix(self.model, unsafe, unsafe_june)

    def test_representative_rows_are_deterministic_and_probability_ordered(self):
        first = select_representative_rows(self.june_meta)
        second = select_representative_rows(self.june_meta.iloc[::-1].reset_index(drop=True))
        self.assertEqual(first.representative_type.tolist(), [
            "lowest_raw_probability", "median_raw_probability", "highest_raw_probability"
        ])
        self.assertEqual(first.CustomerID.tolist(), [12, 13, 11])
        self.assertEqual(first.CustomerID.tolist(), second.CustomerID.tolist())

    def test_shap_shape_names_and_raw_margin_additivity(self):
        raw = self.model.predict_proba(prepare_explanation_matrix(self.model, self.train, self.june))[:, 1]
        calibrated = np.clip(raw * 0.9 + 0.05, 0, 1)
        with tempfile.TemporaryDirectory() as artifact_dir:
            result = explain_lightgbm_churn(
                self.model, self.train, self.june, self.june_meta, raw, calibrated,
                artifact_dir=artifact_dir, create_plots=False,
            )
            self.assertEqual(result["shap_values_shape"], (len(self.june), len(self.train.columns)))
            self.assertEqual(result["feature_names"], list(self.model.feature_name_))
            self.assertEqual(result["n_observations_explained"], len(self.june))
            self.assertEqual(result["n_features_explained"], len(self.train.columns))
            self.assertIn("log-odds", result["output_space"])
            self.assertTrue(result["additivity_verified"])
            self.assertLessEqual(result["additivity_max_abs_error"], result["additivity_tolerance"])
            self.assertEqual(set(result["global_importance"].columns), {
                "feature", "mean_abs_shap", "mean_shap", "rank"
            })
            local = result["local_explanations"]
            self.assertTrue({"CustomerID", "snapshot_date", "raw_churn_probability",
                             "calibrated_churn_probability", "feature", "shap_value",
                             "feature_value", "contribution_direction", "rank_within_customer"}.issubset(local.columns))
            self.assertTrue((local.contribution_direction.isin(["positive", "negative"])).all())


if __name__ == "__main__":
    unittest.main()
