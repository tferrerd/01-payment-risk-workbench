"""Compare a hosted MLflow import with its local export, including model reload."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import numpy as np
import pandas as pd

from payment_risk.pipeline import PaymentRiskPipeline

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "mlflow_exim.run_info.run_id"


def read_json(path):
    return json.loads(path.read_text())


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


def verify(export, restore_registry=False):
    client = MlflowClient()
    runs = {}
    experiments = {}
    for experiment in client.search_experiments():
        experiments[experiment.name] = experiment
        for run in client.search_runs([experiment.experiment_id], max_results=1000):
            source_id = run.data.tags.get(SOURCE_ID)
            if source_id:
                assert source_id not in runs, f"Duplicate import of {source_id}"
                runs[source_id] = run
    sources = sorted((export / "experiments").glob("*/runs/*/run.json"))
    assert sources, "No exported runs found"
    assert len(runs) == len(sources), (len(runs), len(sources))

    def verify_run(path):
        source = read_json(path)["mlflow"]
        info = source["info"]
        run = runs[info["run_id"]]
        expected_name = read_json(path.parents[2] / "experiment.json")["mlflow"]["experiment"]["name"]
        assert run.info.experiment_id == experiments[expected_name].experiment_id
        assert run.info.status == info["status"]
        assert run.data.params == source["params"]
        for key, value in source["tags"].items():
            if not key.startswith("mlflow."):
                assert run.data.tags.get(key) == value, f"Run tag differs: {key}"
        assert set(run.data.metrics) == set(source["metrics"])
        for name, values in source["metrics"].items():
            expected = sorted((v["step"], v["timestamp"], v["value"]) for v in values)
            actual = sorted((v.step, v.timestamp, v.value) for v in client.get_metric_history(run.info.run_id, name))
            assert actual == expected, f"Metric history differs: {info['run_id']}/{name}"
        parent = source["tags"].get("mlflow.parentRunId")
        if parent:
            assert run.data.tags["mlflow.parentRunId"] == runs[parent].info.run_id
        for name in ("start_time", "end_time"):
            assert run.data.tags[f"mlflow_exim.run_info.{name}"] == str(info[name])
        expected_artifacts = hashes(path.parent / "artifacts")
        if expected_artifacts:
            with TemporaryDirectory() as temp:
                downloaded = Path(client.download_artifacts(run.info.run_id, "", temp))
                assert hashes(downloaded) == expected_artifacts, f"Artifacts differ: {info['run_id']}"
        return len(expected_artifacts), int(bool(parent)), len(source["metrics"])

    with ThreadPoolExecutor(max_workers=4) as executor:
        checks = list(executor.map(verify_run, sources))

    versions_checked = 0
    for path in (export / "models").glob("*/model.json"):
        source = read_json(path)["mlflow"]["registered_model"]
        name = source["name"]
        try:
            model = client.get_registered_model(name)
        except MlflowException as error:
            if not restore_registry or error.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
            model = client.create_registered_model(name, description=source["description"], tags=source.get("tags", {}))
        assert model.description == source["description"]
        imported_versions = list(client.search_model_versions(f"name = '{name}'"))
        for version in source["versions"]:
            target_run = runs[version["run_id"]].info.run_id
            matches = [v for v in imported_versions if v.run_id == target_run]
            # The bulk importer copies MLflow 3 logged models but can omit their
            # registered versions. Reuse the imported model; never retrain it.
            if not matches and restore_registry:
                outputs = runs[version["run_id"]].outputs.model_outputs
                assert len(outputs) == 1, "Expected exactly one imported logged model"
                model_id = outputs[0].model_id
                target = client.create_model_version(
                    name=name, source=f"models:/{model_id}", model_id=model_id,
                    run_id=target_run, description=version["description"],
                    tags={**version["tags"], "mlflow_exim.source_version": version["version"]})
                matches = [target]
            assert len(matches) == 1, f"Expected one version for source run {version['run_id']}"
            target = matches[0]
            for key, value in version["tags"].items():
                assert target.tags.get(key) == value, f"Model tag differs: {key}"
            for alias in version["aliases"]:
                if restore_registry:
                    client.set_registered_model_alias(name, alias, target.version)
                assert client.get_model_version_by_alias(name, alias).version == target.version
            with TemporaryDirectory() as temp:
                model_dir = Path(mlflow.artifacts.download_artifacts(
                    artifact_uri=f"models:/{name}/{target.version}", dst_path=temp))
                original = hashes(path.parent / version["source"].split("/")[-1] / "artifacts")
                copied = hashes(model_dir)
                # MLflow rewrites run/model IDs in this metadata file during import.
                original.pop("MLmodel", None)
                copied.pop("MLmodel", None)
                assert copied == original, "Registered model artifact bytes differ"
                restored = mlflow.pyfunc.load_model(str(model_dir))
                example = pd.DataFrame(read_json(ROOT / "examples/predict-request.json")["transactions"])
                expected = PaymentRiskPipeline.load(ROOT / "models/candidates/lightgbm-low-margin-dev-v1.joblib").predict(example)
                actual = restored.predict(example)
                np.testing.assert_allclose(actual.fraud_probability, expected.fraud_probability, rtol=0, atol=1e-12)
                assert actual.decision.equals(expected.decision)
            versions_checked += 1
    assert versions_checked == 1, "Expected the payment-risk registered candidate"
    return {"tracking_uri": mlflow.get_tracking_uri(), "verified": True,
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiments": len({runs[read_json(p)["mlflow"]["info"]["run_id"]].info.experiment_id for p in sources}),
            "runs": len(sources), "nested_parent_links": sum(c[1] for c in checks),
            "metric_histories": sum(c[2] for c in checks), "run_artifact_files": sum(c[0] for c in checks),
            "registered_model_versions": versions_checked, "candidate_reload_parity": True,
            "original_run_ids": "preserved in mlflow_exim.run_info.run_id tags",
            "original_run_times": "preserved in source tags; destination run timestamps are migration times"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restore-registry", action="store_true", help="Create missing registered versions from imported logged models and restore aliases")
    args = parser.parse_args()
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        parser.error("Set MLFLOW_TRACKING_URI to the destination first")
    report = verify(args.export, args.restore_registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
