"""Inspect aggregate service telemetry; optionally send the saved smoke examples."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def request(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    try:
        with urlopen(Request(url, data=data, headers={"Content-Type": "application/json"}), timeout=60) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def operational_alerts(requests):
    alerts = []
    for route in ["/predict", "/explain"]:
        stats = requests.get(route, {})
        if stats.get("count", 0) < 100:
            continue
        errors = sum(count for code, count in stats["status_counts"].items() if int(code) >= 500)
        if errors / stats["count"] > 0.01:
            alerts.append(f"{route}: server errors exceed 1% since worker start")
        if stats["latency_p95_ms"] > 1000:
            alerts.append(f"{route}: server latency p95 exceeds 1000 ms over retained samples")
    return alerts


def check(url, exercise=False):
    status, health = request(url + "/health")
    assert status == 200 and health["status"] == "ok", health
    status, before = request(url + "/metrics")
    assert status == 200 and before["release_id"] == health["release_id"], before
    for kind in ["fraud_score", "amount", "elapsed_time"]:
        assert sum(before["predictions"][kind]["counts"]) == before["predictions"]["payments"]
    if exercise:
        payload = json.loads((ROOT / "examples/predict-request.json").read_text())
        status, response = request(url + "/predict", payload)
        assert status == 200 and len(response["predictions"]) == len(payload["transactions"]), response
        status, response = request(url + "/predict", {"transactions": []})
        assert status == 422, response
    status, after = request(url + "/metrics")
    assert status == 200 and after["started_at"] == before["started_at"], "Worker restarted during check; retry"
    if exercise:
        assert after["predictions"]["payments"] >= before["predictions"]["payments"] + len(payload["transactions"])
        assert after["requests"]["/predict"]["status_counts"].get("422", 0) > before["requests"].get("/predict", {}).get("status_counts", {}).get("422", 0)
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "url": url,
            "health": health, "telemetry": after, "alerts": operational_alerts(after["requests"]),
            "smoke_traffic_generated": exercise,
            "limitations": "Per-worker aggregate telemetry, not persistent uptime monitoring or labeled model-quality evidence. Cold-start/network time is outside server latency."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--exercise", action="store_true", help="Send three saved payments and one expected-invalid request")
    args = parser.parse_args()
    result = check(args.url.rstrip("/"), args.exercise)
    print(json.dumps(result, indent=2))
    if result["alerts"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
