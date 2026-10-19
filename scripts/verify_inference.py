"""Verify local HTTP, batch, registry, and packaged inference agree on selection-derived examples."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from payment_risk import PaymentRiskPipeline
from payment_risk.pipeline import DEFAULT_ARTIFACT, FEATURES, file_sha256

ROOT = Path(__file__).resolve().parents[1]


def request_json(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    pipeline = PaymentRiskPipeline.load(DEFAULT_ARTIFACT)
    examples = pd.read_csv(ROOT / "examples/payment-inputs.csv")
    expected = pipeline.predict(examples)
    payload = {"transactions": examples.to_dict(orient="records")}
    status, health = request_json(args.url + "/health")
    assert status == 200 and health["model_id"] == pipeline.metadata["model_id"]
    assert health["model_status"] == "development_candidate" and health["probability_calibrated"] is False
    status, response = request_json(args.url + "/predict", payload)
    assert status == 200 and np.isclose(response["threshold"], pipeline.threshold)
    actual = pd.DataFrame(response["predictions"])
    np.testing.assert_allclose(actual.fraud_probability, expected.fraud_probability, rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual.modeled_policy_profit, expected.modeled_policy_profit, rtol=0, atol=1e-12)
    assert actual.decision.equals(expected.decision)
    row = payload["transactions"][0]
    missing = {key: value for key, value in row.items() if key != "V1"}
    invalid = {"missing_feature": missing, "negative_amount": {**row, "Amount": -1},
               "negative_time": {**row, "Time": -1}, "text": {**row, "V1": "0.1"},
               "boolean": {**row, "V1": True}, "infinity": {**row, "V1": float("inf")},
               "nan": {**row, "V1": float("nan")}, "unknown_feature": {**row, "Class": 1}}
    checks = {}
    for name, bad_row in invalid.items():
        status, error = request_json(args.url + "/predict", {"transactions": [bad_row]})
        assert status == 422 and error.get("detail"), name
        checks[name] = status
    status, error = request_json(args.url + "/predict", {"transactions": []})
    assert status == 422 and error.get("detail")
    checks["empty_batch"] = status
    status, schema = request_json(args.url + "/openapi.json")
    assert status == 200 and set(schema["components"]["schemas"]["Transaction"]["required"]) == set(FEATURES)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "predictions.csv"
        run = subprocess.run([sys.executable, "-m", "payment_risk", "--input", str(ROOT / "examples/payment-inputs.csv"),
                              "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(run.stderr)
        batch = pd.read_csv(output)
        np.testing.assert_allclose(batch.fraud_probability, expected.fraud_probability, rtol=0, atol=1e-12)
        assert batch.decision.equals(expected.decision)
        assert batch[FEATURES].shape == examples[FEATURES].shape
    package = json.loads((ROOT / "reports/candidate-package.json").read_text())
    import mlflow
    from mlflow import MlflowClient
    mlflow.set_tracking_uri(package["registry"]["tracking_uri"])
    restored = mlflow.pyfunc.load_model(package["registry"]["model_uri"])
    registered = restored.predict(examples)
    np.testing.assert_allclose(registered.fraud_probability, expected.fraud_probability, rtol=0, atol=1e-12)
    assert registered.decision.equals(expected.decision)
    model = MlflowClient().get_registered_model(package["registry"]["name"])
    assert str(model.aliases.get("candidate")) == str(package["registry"]["version"]) and "champion" not in model.aliases
    historical = json.loads((ROOT / "config/candidate-freeze.json").read_text())
    assert all(file_sha256(ROOT / path) == expected_hash for path, expected_hash in historical["lineage"].items())
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "url": args.url, "rows": len(examples),
              "model_id": pipeline.metadata["model_id"], "artifact_sha256": file_sha256(DEFAULT_ARTIFACT),
              "http_batch_registry_parity": True, "invalid_request_statuses": checks,
              "openapi_schema_verified": True, "historical_freeze_preserved": True,
              "status": "development_candidate", "independent_low_margin_test": False}
    (ROOT / "reports/inference-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(examples)} source-derived examples across package, CSV, HTTP, and registry.")
    print(f"Verified {len(checks)} invalid HTTP cases return JSON 422; candidate alias only; historical freeze preserved.")


if __name__ == "__main__":
    main()
