"""Bounded, per-worker operational telemetry; never retain request bodies."""
from bisect import bisect_right
from collections import Counter, deque
from datetime import datetime, timezone
import json
import logging
import math
import os
from threading import Lock
import time

RELEASE_ID = "lightgbm-low-margin-dev-v1-explorer-v2"
SCORE_CUTS = (0.001, 0.003, 0.01, 0.1, 0.5)
AMOUNT_CUTS = (1, 10, 100, 1000, 10000)
TIME_CUTS = (43200, 86400, 129600, 172800)
ROUTES = {"/predict", "/explain", "/demo", "/docs", "/openapi.json", "/"}


def source_revision():
    return os.environ.get("RENDER_GIT_COMMIT") or os.environ.get("APP_REVISION", "local")


def histogram(cuts, counts):
    return {"boundaries": list(cuts), "counts": list(counts),
            "intervals": "[lower, upper); first starts at 0, last has no upper bound"}


class Telemetry:
    def __init__(self):
        self.lock = Lock()
        self.started = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.requests = {}
        self.payments = 0
        self.cancels = 0
        self.scores = [0] * (len(SCORE_CUTS) + 1)
        self.amounts = [0] * (len(AMOUNT_CUTS) + 1)
        self.times = [0] * (len(TIME_CUTS) + 1)

    def record_request(self, route, status, latency_ms):
        with self.lock:
            stats = self.requests.setdefault(route, {"statuses": Counter(), "latencies": deque(maxlen=1000)})
            stats["statuses"][str(status)] += 1
            stats["latencies"].append(latency_ms)

    def record_predictions(self, inputs, predictions):
        with self.lock:
            self.payments += len(predictions)
            self.cancels += int((predictions.decision == "cancel").sum())
            for scores, cuts, values in [
                (self.scores, SCORE_CUTS, predictions.fraud_probability),
                (self.amounts, AMOUNT_CUTS, inputs.Amount),
                (self.times, TIME_CUTS, inputs.Time),
            ]:
                for value in values:
                    scores[bisect_right(cuts, float(value))] += 1

    def snapshot(self):
        with self.lock:
            requests = {}
            for route, stats in self.requests.items():
                latencies = sorted(stats["latencies"])
                requests[route] = {
                    "count": sum(stats["statuses"].values()), "status_counts": dict(stats["statuses"]),
                    "latency_sample_count": len(latencies),
                    "latency_mean_ms": round(sum(latencies) / len(latencies), 3),
                    "latency_p95_ms": round(latencies[math.ceil(0.95 * len(latencies)) - 1], 3),
                }
            return {"release_id": RELEASE_ID, "revision": source_revision(),
                    "started_at": self.started_at,
                    "uptime_seconds": round(time.monotonic() - self.started, 3),
                    "scope": "single worker; resets on restart; latency uses last 1000 requests per route",
                    "requests": requests,
                    "predictions": {"payments": self.payments, "cancels": self.cancels,
                                    "fraud_score": histogram(SCORE_CUTS, self.scores),
                                    "amount": histogram(AMOUNT_CUTS, self.amounts),
                                    "elapsed_time": histogram(TIME_CUTS, self.times)}}


class MonitoringMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] in {"/health", "/metrics"}:
            return await self.app(scope, receive, send)
        route = scope["path"] if scope["path"] in ROUTES else "/other"
        started, status = time.monotonic(), 500

        async def observed_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, observed_send)
        finally:
            latency = (time.monotonic() - started) * 1000
            scope["app"].state.telemetry.record_request(route, status, latency)
            logging.getLogger("uvicorn.error").info(json.dumps({
                "event": "http_request", "release_id": RELEASE_ID,
                "route": route, "status": status, "latency_ms": round(latency, 3),
            }))
