"""Check the pinned inference files and optionally smoke-test the running API (stdlib only)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def check_files(root=ROOT, manifest="deployment/release.json"):
    release = json.loads((root / manifest).read_text())
    for name, expected in release["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Release checksum mismatch: {name}")
    return release


def request_json(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=3) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def check_http(url, release):
    deadline = time.monotonic() + 45
    while True:
        try:
            status, health = request_json(url + "/health")
            break
        except (URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)
    assert status == 200 and health["status"] == "ok", health
    for key in ["model_id", "model_status", "scenario", "probability_calibrated"]:
        assert health[key] == release[key], (key, health)
    assert math.isclose(health["threshold"], release["threshold"], rel_tol=0, abs_tol=1e-15)

    payload = json.loads((ROOT / "examples/predict-request.json").read_text())
    expected = json.loads((ROOT / "examples/predict-expected.json").read_text())
    status, actual = request_json(url + "/predict", payload)
    assert status == 200 and len(actual["predictions"]) == len(expected), actual
    for result, reference in zip(actual["predictions"], expected):
        assert result.keys() == reference.keys()
        for key, value in reference.items():
            if isinstance(value, float):
                assert math.isclose(result[key], value, rel_tol=0, abs_tol=1e-12), (key, result)
            else:
                assert result[key] == value, (key, result)

    row = payload["transactions"][0]
    invalid = [
        {key: value for key, value in row.items() if key != "V1"},
        {**row, "Amount": -1}, {**row, "Time": -1}, {**row, "V1": "0.1"},
        {**row, "V1": True}, {**row, "V1": float("inf")}, {**row, "V1": float("nan")},
        {**row, "Class": 1},
    ]
    for body in [{"transactions": [bad]} for bad in invalid] + [{"transactions": []}]:
        status, error = request_json(url + "/predict", body)
        assert status == 422 and error.get("detail"), error
    status, schema = request_json(url + "/openapi.json")
    assert status == 200 and set(schema["components"]["schemas"]["Transaction"]["required"]) == set(row)
    return {"example_rows": len(expected), "saved_prediction_parity": True,
            "invalid_requests_rejected": len(invalid) + 1, "openapi_schema_verified": True}


def check_explorer(url):
    with urlopen(url + "/", timeout=5) as response:
        assert response.status == 200 and b"Payment decision workbench" in response.read()
    status, demo = request_json(url + "/demo")
    assert status == 200 and demo["source_split"] == "selection" and demo["curated"] is True
    assert len(demo["rows"]) == 200 and sum(row["label"] for row in demo["rows"]) == 10
    status, scores = request_json(url + "/predict", {"transactions": [row["features"] for row in demo["rows"]]})
    assert status == 200 and len(scores["predictions"]) == 200
    for row, prediction in zip(demo["rows"], scores["predictions"]):
        action = "cancel" if prediction["fraud_probability"] > demo["threshold"] and row["features"]["Amount"] > 0 else "approve"
        assert prediction["decision"] == action
    row = demo["rows"][0]
    status, explanation = request_json(url + "/explain", row["features"])
    assert status == 200 and len(explanation["contributions"]) == 30
    raw = explanation["base_log_odds"] + sum(item["log_odds"] for item in explanation["contributions"])
    assert math.isclose(raw, explanation["raw_log_odds"], rel_tol=0, abs_tol=1e-8)
    assert math.isclose(explanation["fraud_probability"], scores["predictions"][0]["fraud_probability"], rel_tol=0, abs_tol=1e-12)
    status, error = request_json(url + "/explain", {**row["features"], "Class": row["label"]})
    assert status == 422 and error.get("detail")
    for asset in ["style.css", "app.mjs"]:
        with urlopen(url + "/assets/" + asset, timeout=5) as response:
            assert response.status == 200
    return {"explorer_rows_scored": 200, "explorer_page_and_assets_verified": True,
            "explanation_reconstructs_score": True, "label_rejected_from_explanation_inputs": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Also check a running API; wait up to 45 seconds for startup")
    parser.add_argument("--manifest", default="deployment/release.json")
    args = parser.parse_args()
    release = check_files(manifest=args.manifest)
    result = {"release_id": release["release_id"], "model_id": release["model_id"],
              "artifact_sha256": release["artifact_sha256"], "release_files_verified": True}
    if args.url:
        result.update(check_http(args.url.rstrip("/"), release))
        if release.get("explorer"):
            result.update(check_explorer(args.url.rstrip("/")))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
