import hashlib
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from churnguard.api import create_app, score_feature_matrix
from churnguard.artifacts import export_inference_artifacts
from churnguard.model_config import ModelConfig, TemporalSplitConfig


class _Response:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.content = body
        self.text = body.decode("utf-8")

    def json(self):
        return json.loads(self.content)


class _ASGIClient:
    """Tiny synchronous ASGI caller to avoid adding a test-only HTTP client."""

    def __init__(self, app):
        self.app = app

    def request(self, method, path, payload=None):
        body = b"" if payload is None else json.dumps(payload).encode("utf-8")
        sent = False
        response_status = 500
        response_body = b""

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message):
            nonlocal response_status, response_body
            if message["type"] == "http.response.start":
                response_status = message["status"]
            elif message["type"] == "http.response.body":
                response_body += message.get("body", b"")

        parts = urlsplit(path)
        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": method, "scheme": "http", "path": parts.path,
            "raw_path": parts.path.encode(), "query_string": parts.query.encode(),
            "root_path": "", "headers": [(b"host", b"testserver"), (b"content-type", b"application/json")],
            "server": ("testserver", 80), "client": ("testclient", 12345),
        }
        asyncio.run(self.app(scope, receive, send))
        return _Response(response_status, response_body)

    def get(self, path):
        return self.request("GET", path)

    def post(self, path, json=None):
        return self.request("POST", path, json)


FEATURES = ["recency_days", "purchase_invoice_count", "historical_net_spend", "Country"]
PREPROCESSING = {
    "feature_columns": FEATURES,
    "feature_types": {
        "recency_days": "number", "purchase_invoice_count": "number",
        "historical_net_spend": "number", "Country": "category",
    },
    "numeric_medians": {
        "recency_days": 20.0, "purchase_invoice_count": 2.0, "historical_net_spend": 50.0,
    },
    "categorical_values": {"Country": ["UK", "France"]},
}


class FakeChurnModel:
    def __init__(self):
        self.predict_calls = 0
        self.fit_calls = 0

    def fit(self, *_args, **_kwargs):
        self.fit_calls += 1
        raise AssertionError("Inference must not train models")

    def predict_proba(self, frame):
        self.predict_calls += 1
        p = np.clip(frame["recency_days"].to_numpy(float) / 100.0, 0.0, 1.0)
        return np.column_stack([1 - p, p])


class FakeCalibrator:
    def __init__(self):
        self.calls = 0

    def fit(self, *_args, **_kwargs):
        raise AssertionError("Inference must not fit the calibrator")

    def predict(self, raw):
        self.calls += 1
        return np.asarray(raw) * 0.5


class FakeValueModel:
    def __init__(self):
        self.predict_calls = 0
        self.fit_calls = 0

    def fit(self, *_args, **_kwargs):
        self.fit_calls += 1
        raise AssertionError("Inference must not train a value model")

    def predict(self, frame):
        self.predict_calls += 1
        return frame["recency_days"].to_numpy(float) / 10.0


def write_fake_artifacts(directory):
    churn = FakeChurnModel()
    calibrator = FakeCalibrator()
    value_model = FakeValueModel()
    value_schema = json.loads(json.dumps(PREPROCESSING))
    outputs = {
        "models": {"lightgbm": churn, "lightgbm_calibrator": calibrator},
        "lightgbm_preprocessing": PREPROCESSING,
        "value_results": {
            "models": {"lightgbm_value": value_model},
            "lightgbm_preprocessing": value_schema,
            "training_transformed_target_bounds": [0.0, 3.0],
        },
    }
    path = export_inference_artifacts(outputs, TemporalSplitConfig(), ModelConfig(), directory)
    return path


def record(customer_id=22, recency=30, country="UK", snapshot_date="2011-06-01"):
    result = {"features": {
        "recency_days": recency,
        "purchase_invoice_count": 3,
        "historical_net_spend": 125.0,
        "Country": country,
    }}
    if customer_id is not None:
        result["CustomerID"] = customer_id
    if snapshot_date is not None:
        result["snapshot_date"] = snapshot_date
    return result


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.artifact_dir = write_fake_artifacts(Path(self.temporary.name) / "models")
        self.app = create_app(self.artifact_dir)
        self.client = _ASGIClient(self.app)

    def tearDown(self):
        self.temporary.cleanup()

    def test_health_reports_loaded_and_missing_artifact_states(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "status": "ok", "churn_model_loaded": True,
            "calibrator_loaded": True, "future_value_model_loaded": True,
        })
        missing = _ASGIClient(create_app(Path(self.temporary.name) / "absent")).get("/health")
        self.assertEqual(missing.status_code, 503)
        self.assertEqual(missing.json()["status"], "not_ready")
        self.assertFalse(missing.json()["future_value_model_loaded"])

    def test_predict_returns_scores_preserves_metadata_and_uses_calibrated_probability(self):
        response = self.client.post("/predict", json=record())
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["CustomerID"], 22)
        self.assertEqual(body["snapshot_date"], "2011-06-01T00:00:00")
        self.assertAlmostEqual(body["churn_probability"], 0.15)
        self.assertAlmostEqual(body["predicted_90d_value"], np.expm1(3.0))
        self.assertAlmostEqual(body["risk_weighted_value"], body["churn_probability"] * body["predicted_90d_value"])
        fitted = self.app.state.artifacts
        self.assertEqual(fitted.calibrator.calls, 1)
        self.assertEqual(fitted.churn_model.fit_calls, 0)

    def test_batch_has_deterministic_ranks_and_metadata_is_optional(self):
        request = {"customers": [
            record(customer_id=20, recency=60),
            record(customer_id=10, recency=60),
            record(customer_id=30, recency=20),
            record(customer_id=None, recency=40, snapshot_date=None),
        ]}
        response = self.client.post("/batch_score", json=request)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["count"], 4)
        self.assertEqual([row["churn_rank"] for row in body["scored_customers"]], [2, 1, 3, 1])
        self.assertEqual([row["CustomerID"] for row in body["scored_customers"][:2]], [20, 10])
        self.assertIsNone(body["scored_customers"][3]["snapshot_date"])
        again = self.client.post("/batch_score", json=request).json()
        self.assertEqual(body, again)

    def test_missing_extra_and_invalid_features_are_rejected(self):
        missing = record()
        del missing["features"]["Country"]
        self.assertEqual(self.client.post("/predict", json=missing).status_code, 422)
        unexpected = record()
        unexpected["features"]["churn_90d"] = 1
        self.assertEqual(self.client.post("/predict", json=unexpected).status_code, 422)
        invalid_category = record(country="Unknown")
        self.assertEqual(self.client.post("/predict", json=invalid_category).status_code, 422)
        invalid_number = record()
        invalid_number["features"]["recency_days"] = "bad"
        self.assertEqual(self.client.post("/predict", json=invalid_number).status_code, 422)
        extra_field = record()
        extra_field["future_net_spend_90d"] = 10
        self.assertEqual(self.client.post("/predict", json=extra_field).status_code, 422)

    def test_shared_internal_scorer_maps_unseen_category_to_model_missing_category(self):
        # Offline model preprocessing represents unseen levels as missing; the
        # public request validator remains strict for ordinary API requests.
        features = pd.DataFrame([{
            "recency_days": 30,
            "purchase_invoice_count": 3,
            "historical_net_spend": 125.0,
            "Country": "Unknown",
        }], columns=PREPROCESSING["feature_columns"])
        scores = score_feature_matrix(features, self.app.state.artifacts)
        self.assertTrue(np.isfinite(scores.to_numpy(dtype=float)).all())
        self.assertEqual(self.client.post("/predict", json=record(country="Unknown")).status_code, 422)

    def test_api_uses_supervised_value_model_without_future_labels_or_artifact_mutation(self):
        bundle_path = self.artifact_dir / "future_value" / "model.joblib"
        before = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
        response = self.client.post("/predict", json=record())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.app.state.artifacts.future_value_model.predict_calls, 1)
        self.assertEqual(self.app.state.artifacts.future_value_model.fit_calls, 0)
        self.assertEqual(hashlib.sha256(bundle_path.read_bytes()).hexdigest(), before)
        self.assertNotIn("future_net_spend_90d", response.json())

    def test_unready_service_refuses_inference(self):
        missing_client = _ASGIClient(create_app(Path(self.temporary.name) / "missing"))
        self.assertEqual(missing_client.post("/predict", json=record()).status_code, 503)
        self.assertEqual(missing_client.post("/batch_score", json={"customers": [record()]}).status_code, 503)


if __name__ == "__main__":
    unittest.main()
