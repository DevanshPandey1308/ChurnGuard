import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from churnguard.model_config import ModelConfig, TemporalSplitConfig
from churnguard.tracking import TrackingConfig, track_experiments


def sample_outputs():
    predictions = pd.DataFrame({
        "CustomerID": [101, 102],
        "snapshot_date": pd.to_datetime(["2011-06-01", "2011-06-01"]),
        "raw_lightgbm_probability": [0.1, 0.9],
        "future_net_spend_90d": [20, 40],
    })
    evaluation = {
        "results": {
            name: {
                "validation": {"pr_auc": 0.61, "lift_at_decile_1": 1.5, "brier_score": 0.20},
                "test": {"pr_auc": 0.58, "lift_at_decile_1": 1.4, "brier_score": 0.21},
            }
            for name in ("majority_class", "logistic_rfm", "rule_based_rfm", "lightgbm")
        },
        "calibration_results": {
            "raw": {
                "validation": {"pr_auc": 0.61, "lift_at_decile_1": 1.5, "brier_score": 0.20},
                "test": {"pr_auc": 0.58, "lift_at_decile_1": 1.4, "brier_score": 0.21},
            },
            "calibrated": {
                "validation": {"pr_auc": 0.61, "lift_at_decile_1": 1.5, "brier_score": 0.19},
                "test": {"pr_auc": 0.58, "lift_at_decile_1": 1.4, "brier_score": 0.20},
            },
        },
        "value_results": {
            "metrics": {"lightgbm_value": {
                "validation": {"mae": 100.0, "rmse": 200.0, "spearman": 0.5},
                "test": {"mae": 110.0, "rmse": 210.0, "spearman": 0.4},
            }},
            "models": {"lightgbm_value": type("Estimator", (), {
                "get_params": lambda self: {"n_estimators": 3, "learning_rate": 0.04, "objective": "regression"}
            })()},
        },
        "targeting_results": {
            "configuration": {
                "top_fraction": 0.1,
                "retention_offer_cost": 50,
                "assumed_acceptance_rate": 0.25,
                "retained_value_fraction": 0.8,
            },
            "test": {
                "top_decile_actual_value_capture": {
                    "predicted_90d_value": {"percentage_of_total_actual_future_net_spend": 55.0},
                    "risk_weighted_value": {"percentage_of_total_actual_future_net_spend": 53.0},
                },
                "predictions": pd.DataFrame({
                    "CustomerID": [101, 102],
                    "risk_value_rank": [2, 1],
                    "expected_net_value_under_scenario": [-10.0, -5.0],
                }),
            },
        },
        "predictions": {"test": predictions},
    }
    walk_forward = {
        "results": pd.DataFrame([
            {"fold_id": "wf_01_2011-04-01", "validation_date": "2011-04-01", "model": "lightgbm",
             "pr_auc": 0.70, "lift": 1.7, "brier_score": 0.18, "validation_rows": 10},
            {"fold_id": "wf_02_2011-05-01", "validation_date": "2011-05-01", "model": "lightgbm",
             "pr_auc": 0.72, "lift": 1.8, "brier_score": 0.17, "validation_rows": 12},
        ]),
        "aggregate": pd.DataFrame([{
            "model": "lightgbm", "folds_evaluated": 2,
            "mean_pr_auc": 0.71, "std_pr_auc": 0.01,
            "mean_lift": 1.75, "std_lift": 0.05,
            "mean_brier_score": 0.175, "std_brier_score": 0.007,
        }]),
    }
    return evaluation, walk_forward


class TrackingTests(unittest.TestCase):
    def test_configuration_defaults_to_deterministic_local_sqlite_uri(self):
        with tempfile.TemporaryDirectory() as directory:
            config = TrackingConfig(local_tracking_dir=directory)
            expected_path = (Path(directory).resolve() / "mlflow.db").as_posix()
            self.assertEqual(config.tracking_uri, f"sqlite:///{expected_path}")
            self.assertEqual(config.churn_experiment_name, "ChurnGuard-Churn")
            self.assertEqual(config, TrackingConfig(local_tracking_dir=directory))

    def test_environment_configuration_is_supported(self):
        with patch.dict("os.environ", {
            "CHURNGUARD_MLFLOW_DIR": "temporary-mlruns",
            "CHURNGUARD_MLFLOW_CHURN_EXPERIMENT": "Test-Churn",
            "CHURNGUARD_MLFLOW_VALUE_EXPERIMENT": "Test-Value",
            "CHURNGUARD_MLFLOW_TARGETING_EXPERIMENT": "Test-Targeting",
        }, clear=True):
            config = TrackingConfig.from_env()
        self.assertTrue(config.tracking_uri.startswith("sqlite:///"))
        self.assertEqual(config.churn_experiment_name, "Test-Churn")
        self.assertEqual(config.value_experiment_name, "Test-Value")
        self.assertEqual(config.targeting_experiment_name, "Test-Targeting")
        with patch.dict("os.environ", {
            "CHURNGUARD_MLFLOW_TRACKING_URI": "sqlite:///explicit-local.db",
        }, clear=True):
            explicit = TrackingConfig.from_env()
        self.assertEqual(explicit.tracking_uri, "sqlite:///explicit-local.db")

    def test_local_runs_log_params_metrics_and_aggregate_artifacts_without_changing_predictions(self):
        import mlflow
        from mlflow.tracking import MlflowClient

        evaluation, walk_forward = sample_outputs()
        before_predictions = evaluation["predictions"]["test"].copy(deep=True)
        before_targeting = evaluation["targeting_results"]["test"]["predictions"].copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            config = TrackingConfig(local_tracking_dir=directory)
            tracking = track_experiments(
                evaluation, walk_forward, config,
                TemporalSplitConfig(),
                ModelConfig(lightgbm_params={"n_estimators": 3, "num_leaves": 5}),
            )
            self.assertTrue(tracking["tracking_uri"].startswith("sqlite:///"))
            self.assertEqual(set(tracking["runs"]), {"churn", "future_value", "targeting"})
            self.assertEqual(set(tracking["runs"]["churn"]), {
                "majority_baseline", "logistic_rfm", "rule_based_rfm", "lightgbm_raw", "lightgbm_calibrated"
            })
            client = MlflowClient(tracking_uri=config.tracking_uri)
            churn = client.get_run(tracking["runs"]["churn"]["lightgbm_raw"]["run_id"])
            self.assertEqual(churn.data.params["model_name"], "LightGBM churn classifier (raw)")
            self.assertEqual(churn.data.params["training_snapshot_dates"], '["2011-03-01", "2011-04-01"]')
            self.assertEqual(churn.data.params["lightgbm_n_estimators"], "3")
            for key in ("validation_pr_auc", "validation_lift_at_decile_1", "validation_brier",
                        "final_test_pr_auc", "final_test_lift_at_decile_1", "final_test_brier",
                        "wf_01_pr_auc", "wf_01_lift", "wf_01_brier",
                        "wf_02_pr_auc", "wf_02_lift", "wf_02_brier",
                        "walk_forward_mean_pr_auc", "walk_forward_std_pr_auc",
                        "walk_forward_mean_lift", "walk_forward_std_lift",
                        "walk_forward_mean_brier", "walk_forward_std_brier"):
                self.assertIn(key, churn.data.metrics)
            artifact_paths = {item.path for item in client.list_artifacts(churn.info.run_id, "walk_forward")}
            self.assertEqual(artifact_paths, {
                "walk_forward/walk_forward_metrics.csv", "walk_forward/walk_forward_aggregate.csv"
            })

            for run_name, model_name in (
                ("majority_baseline", "Majority-class baseline"),
                ("logistic_rfm", "Logistic regression RFM"),
                ("rule_based_rfm", "Rule-based RFM baseline"),
            ):
                model_run = client.get_run(tracking["runs"]["churn"][run_name]["run_id"])
                self.assertEqual(model_run.data.params["model_name"], model_name)
                for key in ("validation_pr_auc", "validation_lift_at_decile_1", "validation_brier",
                            "final_test_pr_auc", "final_test_lift_at_decile_1", "final_test_brier"):
                    self.assertIn(key, model_run.data.metrics)
            calibrated = client.get_run(tracking["runs"]["churn"]["lightgbm_calibrated"]["run_id"])
            for key in ("raw_validation_pr_auc", "calibrated_validation_pr_auc",
                        "raw_validation_lift_at_decile_1", "calibrated_validation_lift_at_decile_1",
                        "raw_validation_brier", "calibrated_validation_brier",
                        "raw_final_test_pr_auc", "calibrated_final_test_pr_auc",
                        "raw_final_test_lift_at_decile_1", "calibrated_final_test_lift_at_decile_1",
                        "raw_final_test_brier", "calibrated_final_test_brier"):
                self.assertIn(key, calibrated.data.metrics)
            churn_experiment = client.get_experiment_by_name("ChurnGuard-Churn")
            self.assertEqual(len(client.search_runs([churn_experiment.experiment_id])), 5)

            value = client.get_run(tracking["runs"]["future_value"]["run_id"])
            self.assertEqual(value.data.params["model_type"], "regression")
            self.assertIn("validation_mae", value.data.metrics)
            self.assertIn("final_test_spearman", value.data.metrics)
            targeting = client.get_run(tracking["runs"]["targeting"]["run_id"])
            self.assertIn("june_top10_predicted_value_capture", targeting.data.metrics)
            self.assertIn("june_top10_risk_weighted_value_capture", targeting.data.metrics)
            self.assertNotIn("revenue_saved", targeting.data.metrics)
            self.assertFalse(any("CustomerID" in key for key in targeting.data.params))
            self.assertEqual(client.list_artifacts(targeting.info.run_id), [])
            # MLflow caches SQLAlchemy engines process-wide. Dispose this test's
            # local database connection before TemporaryDirectory removes it.
            client._tracking_client.store._dispose_engine()

        pd.testing.assert_frame_equal(evaluation["predictions"]["test"], before_predictions)
        pd.testing.assert_frame_equal(evaluation["targeting_results"]["test"]["predictions"], before_targeting)


if __name__ == "__main__":
    unittest.main()
