import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from payment_risk.monitoring import MonitoringMiddleware, Telemetry
from scripts.check_operations import operational_alerts


class MonitoringTest(unittest.IsolatedAsyncioTestCase):
    async def test_alerts_require_volume_and_ignore_expected_client_errors(self):
        stats = {"count": 99, "status_counts": {"500": 99}, "latency_p95_ms": 2000}
        self.assertEqual(operational_alerts({"/predict": stats}), [])
        stats.update(count=100, status_counts={"200": 80, "422": 18, "500": 2})
        self.assertEqual(len(operational_alerts({"/predict": stats})), 2)
        stats.update(status_counts={"200": 90, "422": 10}, latency_p95_ms=10)
        self.assertEqual(operational_alerts({"/predict": stats}), [])

    async def test_bounded_latency_and_histogram_boundaries(self):
        metrics = Telemetry()
        with patch.dict(os.environ, {"RENDER_GIT_COMMIT": "deployed-sha", "APP_REVISION": "build-sha"}):
            self.assertEqual(metrics.snapshot()["revision"], "deployed-sha")
        for _ in range(1005):
            metrics.record_request("/predict", 200, 12)
        metrics.record_request("/predict", 422, 50)
        inputs = pd.DataFrame({"Amount": [0, 1, 10000], "Time": [0, 86400, 172800]})
        predictions = pd.DataFrame({"fraud_probability": [0, 0.003, 1], "decision": ["approve", "cancel", "cancel"]})
        metrics.record_predictions(inputs, predictions)
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["requests"]["/predict"]["count"], 1006)
        self.assertEqual(snapshot["requests"]["/predict"]["latency_sample_count"], 1000)
        self.assertEqual(snapshot["requests"]["/predict"]["latency_p95_ms"], 12)
        self.assertEqual(snapshot["predictions"]["payments"], 3)
        self.assertEqual(snapshot["predictions"]["cancels"], 2)
        self.assertEqual(snapshot["predictions"]["fraud_score"]["counts"], [1, 0, 1, 0, 0, 1])
        self.assertEqual(snapshot["predictions"]["amount"]["counts"], [1, 1, 0, 0, 0, 1])
        snapshot["predictions"]["amount"]["counts"][0] = -1
        self.assertEqual(metrics.snapshot()["predictions"]["amount"]["counts"][0], 1)

    async def test_middleware_records_validation_and_failure_without_request_data(self):
        metrics = Telemetry()

        async def send(message):
            pass

        async def receive():
            self.fail("Telemetry must not read the request body")

        async def invalid(scope, receive, send):
            await send({"type": "http.response.start", "status": 422, "headers": []})

        scope = {"type": "http", "path": "/predict", "query_string": b"private=secret",
                 "app": SimpleNamespace(state=SimpleNamespace(telemetry=metrics))}
        with self.assertLogs("uvicorn.error", level="INFO") as captured:
            await MonitoringMiddleware(invalid)(scope, receive, send)
        self.assertNotIn("secret", captured.output[0])
        self.assertEqual(metrics.snapshot()["requests"]["/predict"]["status_counts"], {"422": 1})
        self.assertEqual(metrics.snapshot()["predictions"]["payments"], 0)

        async def failed(scope, receive, send):
            raise RuntimeError("internal detail")

        scope["path"] = "/unknown-user-value"
        with self.assertLogs("uvicorn.error", level="INFO"), self.assertRaises(RuntimeError):
            await MonitoringMiddleware(failed)(scope, receive, send)
        self.assertEqual(metrics.snapshot()["requests"]["/other"]["status_counts"], {"500": 1})
        self.assertNotIn("unknown-user-value", json.dumps(metrics.snapshot()))


if __name__ == "__main__":
    unittest.main()
