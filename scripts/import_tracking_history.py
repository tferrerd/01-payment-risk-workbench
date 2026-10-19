"""Resume this project's MLflow export using batched metadata writes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from mlflow import MlflowClient
from mlflow.entities import LoggedModelOutput, LoggedModelStatus, Metric, Param, RunTag
from mlflow_export_import.common.source_tags import (
    ExportTags, mk_source_tags, mk_source_tags_mlflow_tag,
)
from mlflow_export_import.logged_model.logged_model_utils import update_logged_model_mlmodel_data
import requests

SOURCE_ID = "mlflow_exim.run_info.run_id"


def read(path):
    return json.loads(path.read_text())["mlflow"]


def source_tags(source):
    tags = {**source["tags"], **mk_source_tags_mlflow_tag(source["tags"]),
            **mk_source_tags(source["info"], ExportTags.PREFIX_RUN_INFO), "migration.transfer": "batched-v1"}
    tags.pop("mlflow.parentRunId", None)
    return tags


def migrate(export, workers):
    # A timeout after create_run commits can create a second run on automatic
    # retry. Source-ID tags make an explicit resume safer than retrying POSTs.
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
    uri = os.environ["MLFLOW_TRACKING_URI"]
    parsed = urlparse(uri)
    assert parsed.scheme == "https" and parsed.hostname == "dagshub.com"
    assert parsed.path.endswith(".mlflow")
    repo = parsed.path.removesuffix(".mlflow").strip("/")
    response = requests.get(
        f"https://dagshub.com/api/v1/repos/{repo}",
        auth=(os.environ["MLFLOW_TRACKING_USERNAME"], os.environ["MLFLOW_TRACKING_PASSWORD"]),
        timeout=30,
    )
    response.raise_for_status()
    assert response.json()["private"] is True, "History uploads require a private repository"

    client = MlflowClient()
    # Create experiment names sequentially before parallelizing run transfers.
    experiments = {e.name: e for e in client.search_experiments()}
    source_paths = sorted((export / "experiments").glob("*/runs/*/run.json"))
    assert source_paths, "No exported runs found"
    source_names = {read(p.parents[2] / "experiment.json")["experiment"]["name"] for p in source_paths}
    for name in sorted(source_names - experiments.keys()):
        eid = client.create_experiment(name)
        experiments[name] = client.get_experiment(eid)

    existing = {}
    unindexed = []
    for run in client.search_runs([experiments[n].experiment_id for n in source_names], max_results=1000):
        source_id = run.data.tags.get(SOURCE_ID)
        if source_id:
            assert source_id not in existing, f"Duplicate source run: {source_id}"
            existing[source_id] = run
        else:
            unindexed.append(run)
    # A stopped upstream import may have written params/metrics before its tags.
    # Repair only an unambiguous, still-running match to an unused source run.
    for run in unindexed:
        matches = []
        for path in source_paths:
            source = read(path)
            name = read(path.parents[2] / "experiment.json")["experiment"]["name"]
            latest = {k: max(v, key=lambda m: (m["step"], m["timestamp"], m["value"]))["value"]
                      for k, v in source["metrics"].items()}
            if (source["info"]["run_id"] not in existing and run.info.status == "RUNNING"
                    and experiments[name].experiment_id == run.info.experiment_id
                    and run.data.params == source["params"] and run.data.metrics == latest):
                matches.append(source)
        assert len(matches) == 1, f"Unindexed run requires review: {run.info.run_id}"
        source = matches[0]
        client.log_batch(run.info.run_id, tags=[RunTag(k, v) for k, v in source_tags(source).items()])
        existing[source["info"]["run_id"]] = client.get_run(run.info.run_id)
        print("Recovered interrupted run:", run.info.run_id, flush=True)

    def transfer(path):
        source = read(path)
        info = source["info"]
        source_id = info["run_id"]
        name = read(path.parents[2] / "experiment.json")["experiment"]["name"]
        run = existing.get(source_id)
        if run and run.info.status == info["status"]:
            return source_id, run.info.run_id, "reused"
        if run is None:
            run = client.create_run(experiments[name].experiment_id, tags=source_tags(source), run_name=info["run_name"])
        for key, value in run.data.params.items():
            assert source["params"].get(key) == value, f"Parameter conflict: {source_id}/{key}"
        params = [Param(k, v) for k, v in source["params"].items() if k not in run.data.params]
        metrics = []
        for key, values in source["metrics"].items():
            if len(values) == 1 and run.data.metrics.get(key) == values[0]["value"]:
                continue
            previous = set()
            if key in run.data.metrics:
                previous = {(m.value, m.timestamp, m.step) for m in client.get_metric_history(run.info.run_id, key)}
            metrics.extend(Metric(key, v["value"], v["timestamp"], v["step"]) for v in values
                           if (v["value"], v["timestamp"], v["step"]) not in previous)
        # This export's runs fit within one MLflow metadata batch.
        assert len(params) <= 100 and len(metrics) <= 1000
        if params or metrics:
            client.log_batch(run.info.run_id, params=params, metrics=metrics)
        artifacts = path.parent / "artifacts"
        if artifacts.exists():
            client.log_artifacts(run.info.run_id, str(artifacts))
        client.set_terminated(run.info.run_id, info["status"])
        return source_id, run.info.run_id, "copied"

    run_map = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(transfer, p) for p in source_paths]
        for future in as_completed(futures):
            source_id, destination_id, action = future.result()
            run_map[source_id] = destination_id
            print(f"{len(run_map)}/{len(source_paths)} {action} {source_id}", flush=True)

    for path in source_paths:
        source = read(path)
        parent = source["tags"].get("mlflow.parentRunId")
        if parent:
            client.set_tag(run_map[source["info"]["run_id"]], "mlflow.parentRunId", run_map[parent])

    for path in (export / "experiments").glob("*/logged_models/*/logged_model.json"):
        source = read(path)
        run_id = run_map[source["source_run_id"]]
        run = client.get_run(run_id)
        models = [m for m in client.search_logged_models(experiment_ids=[run.info.experiment_id])
                  if m.source_run_id == run_id]
        assert len(models) <= 1, "More than one imported model belongs to the candidate run"
        model = models[0] if models else client.create_logged_model(
            experiment_id=run.info.experiment_id, name=source["name"],
            source_run_id=run_id, tags=source["tags"], params=source["params"])
        if model.status != LoggedModelStatus.READY:
            client.log_model_artifacts(model.model_id, str(path.parent / "artifacts"))
            update_logged_model_mlmodel_data(client, model, str(path.parent / "artifacts/MLmodel"))
            client.finalize_logged_model(model.model_id, LoggedModelStatus.READY)
        if model.model_id not in [m.model_id for m in run.outputs.model_outputs]:
            client.log_outputs(run_id, models=[LoggedModelOutput(model.model_id, step=0)])
        client.set_terminated(run_id, "FINISHED")
    print("History copied. Run verify_tracking_migration before declaring migration complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    migrate(args.export, args.workers)
