"""Package the existing low-margin LightGBM candidate and verify selection parity."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from payment_risk.pipeline import DEFAULT_ARTIFACT, FEATURES, PaymentRiskPipeline, file_sha256
from scripts.run_baselines import ROOT, setup_tracking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true", help="Register only the development candidate in MLflow")
    args = parser.parse_args()
    scenario = json.loads((ROOT / "config/cost-scenario-low-margin.json").read_text())
    experiment = json.loads((ROOT / "reports/experiment-results.json").read_text())
    calibration = json.loads((ROOT / "reports/calibration-results.json").read_text())
    low_margin = json.loads((ROOT / "reports/low-margin-scenario.json").read_text())
    historical = json.loads((ROOT / "config/candidate-freeze.json").read_text())
    for path, expected in historical["lineage"].items():
        if file_sha256(ROOT / path) != expected:
            raise ValueError(f"Historical frozen input changed: {path}")
    source_path = ROOT / "models/experiments/lightgbm.joblib"
    evidence = next(row for row in calibration["results"] if row["name"] == "lightgbm")
    if file_sha256(source_path) != evidence["raw_model_sha256"]:
        raise ValueError("Source LightGBM artifact differs from calibration lineage")
    for path, expected in low_margin["metadata"]["lineage"].items():
        if file_sha256(ROOT / path) != expected:
            raise ValueError(f"Low-margin analysis input changed: {path}")
    source = joblib.load(source_path)
    if source["input_columns"] != FEATURES or source["lineage"]["code_sha256"] != experiment["metadata"]["code_sha256"]:
        raise ValueError("Source model schema or training lineage changed")
    paths = ["config/cost-scenario-low-margin.json", "reports/experiment-results.json", "reports/calibration-results.json",
             "reports/low-margin-scenario.json", "models/experiments/lightgbm.joblib", "scripts/package_candidate.py", "requirements-inference.txt"]
    paths += [str(path.relative_to(ROOT)) for path in sorted((ROOT / "payment_risk").glob("*.py"))]
    lineage = {path: file_sha256(ROOT / path) for path in paths}
    artifact_path = DEFAULT_ARTIFACT
    metadata = {"model_id": "lightgbm-low-margin-dev-v1", "status": "development_candidate", "features": FEATURES,
                "economics": scenario, "threshold": scenario["legitimate_margin_fraction"] / (scenario["legitimate_margin_fraction"] + scenario["fraud_loss_fraction"]),
                "calibration_method": "identity", "probability_calibrated": False, "model_refitted": False,
                "independent_low_margin_test": False,
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "lineage": lineage,
                "runtime_versions": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "lightgbm", "joblib"]}}
    if artifact_path.exists():
        pipeline = PaymentRiskPipeline.load(artifact_path)
        if pipeline.metadata["lineage"] != lineage:
            raise ValueError("Candidate v1 already exists with different lineage; create a new version instead of overwriting it")
    else:
        pipeline = PaymentRiskPipeline(source["model"], metadata)
    prediction_path = ROOT / experiment["selection_predictions"]["path"]
    if file_sha256(prediction_path) != experiment["selection_predictions"]["sha256"]:
        raise ValueError("Saved selection predictions changed")
    saved = pd.read_csv(prediction_path)
    saved = saved.loc[saved.model.eq("lightgbm")].set_index("source_row_id").sort_index()
    audit = json.loads((ROOT / "reports/duplicate-split-audit.json").read_text())
    parts, offset = [], 0
    for item in audit["source_files"]:
        path = ROOT / item["path"]
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"Source file changed: {path}")
        for chunk in pd.read_csv(path, usecols=FEATURES, chunksize=50000):
            chunk.index = np.arange(offset, offset + len(chunk))
            offset += len(chunk)
            parts.append(chunk.loc[chunk.index.isin(saved.index)])
    inputs = pd.concat(parts).loc[saved.index, FEATURES].astype(float)
    actual = pipeline.predict(inputs)
    difference = float(np.max(np.abs(actual.fraud_probability.to_numpy() - saved.score.to_numpy())))
    if difference > 1e-12:
        raise ValueError("Packaged inference differs from saved LightGBM selection scores")
    policies = low_margin["comparisons"]["selection_development"]
    policy_path = ROOT / policies["predictions"]["path"]
    if file_sha256(policy_path) != policies["predictions"]["sha256"]:
        raise ValueError("Low-margin policy predictions changed")
    decisions = pd.read_csv(policy_path)
    decisions = decisions.loc[decisions.policy.eq("lightgbm_break_even")].set_index("source_row_id").loc[saved.index]
    if not np.array_equal(actual.decision.eq("cancel"), decisions.canceled):
        raise ValueError("Packaged policy differs from stage 08 LightGBM decisions")
    if not artifact_path.exists():
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"artifact_format_version": 1, "model": source["model"], "metadata": metadata}, artifact_path)
        manifest = {"model_id": metadata["model_id"], "artifact_sha256": file_sha256(artifact_path), "metadata": metadata}
        artifact_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    restored = PaymentRiskPipeline.load(artifact_path).predict(inputs.iloc[:20])
    if not np.allclose(restored.fraud_probability, actual.fraud_probability.iloc[:20], rtol=0, atol=1e-12) or not restored.decision.equals(actual.decision.iloc[:20]):
        raise ValueError("Reloaded artifact differs from the validated pipeline")
    validation = {"rows": len(inputs), "split": "selection", "maximum_probability_difference": difference,
                  "decisions_match_stage_08": True, "no_refit": True, "independent_low_margin_test": False,
                  "stage_08_selection_metrics": next(row for row in policies["results"] if row["policy"] == "lightgbm_break_even")}
    # Small examples come only from selection, including observed action outcomes.
    chosen_ids = []
    for mask in [saved.Class.eq(0), saved.Class.eq(1), inputs.Amount.eq(0)]:
        if mask.any():
            chosen_ids.append(mask.index[mask][0])
    chosen_ids = list(dict.fromkeys(chosen_ids))
    examples = inputs.loc[chosen_ids].copy()
    sample_path = ROOT / "examples/payment-inputs.csv"
    sample_path.parent.mkdir(exist_ok=True)
    examples.to_csv(sample_path, index=False)
    (ROOT / "examples/predict-request.json").write_text(json.dumps({"transactions": examples.to_dict(orient="records")}, indent=2) + "\n")
    report_path = ROOT / "reports/candidate-package.json"
    report = {"model_id": pipeline.metadata["model_id"], "status": "development_candidate",
              "artifact_path": str(artifact_path.relative_to(ROOT)), "artifact_sha256": file_sha256(artifact_path),
              "threshold": pipeline.threshold, "lineage": lineage, "validation": validation,
              "example_source_row_ids": [int(value) for value in chosen_ids]}
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get("artifact_sha256") == report["artifact_sha256"] and "registry" in previous:
            report["registry"] = previous["registry"]
    if args.register:
        from payment_risk.registry import register_candidate
        setup_tracking("09-inference-candidate")
        report["registry"] = register_candidate(artifact_path, examples, validation)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Verified {len(inputs):,} selection scores and decisions; maximum probability difference {difference:.3g}.")
    print(f"Candidate artifact: {artifact_path.relative_to(ROOT)}; break-even probability {pipeline.threshold:.10f}")
    if "registry" in report:
        print("MLflow candidate:", report["registry"]["model_uri"])


if __name__ == "__main__":
    main()
