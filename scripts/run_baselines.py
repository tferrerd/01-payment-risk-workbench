"""Fit reproducible initial baselines; score selection only, never final evaluation."""

import hashlib
import json
import os
import subprocess
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def setup_tracking(experiment_name):
    uri = os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///" + str(ROOT / "mlflow.db")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    return uri


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def action_costs(labels, amounts, canceled, costs):
    """Opportunity loss versus approving only legitimate transactions."""
    labels = np.asarray(labels)
    amounts = np.asarray(amounts, dtype=float)
    canceled = np.asarray(canceled, dtype=bool)
    if not (labels.shape == amounts.shape == canceled.shape):
        raise ValueError("Labels, amounts and actions must have identical shapes")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("Expected binary labels and finite nonnegative amounts")
    return np.where(canceled, (1 - labels) * costs["legitimate_margin_fraction"] * amounts,
                    labels * costs["fraud_loss_fraction"] * amounts)


def action_profits(labels, amounts, canceled, costs):
    loss = action_costs(labels, amounts, canceled, costs)
    oracle_profit = (1 - np.asarray(labels)) * costs["legitimate_margin_fraction"] * np.asarray(amounts)
    return oracle_profit - loss


def cancel_from_probability(probability, amounts, costs):
    probability = np.asarray(probability, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    if probability.shape != amounts.shape or not np.isfinite(amounts).all() or (amounts < 0).any() or not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("Expected one finite probability in [0, 1] per nonnegative amount")
    margin = costs["legitimate_margin_fraction"]
    threshold = margin / (margin + costs["fraud_loss_fraction"])
    return (probability > threshold) & (amounts > 0)


def make_logistic():
    components = [f"V{i}" for i in range(1, 29)]
    transformers = [
        ("components", StandardScaler(), components),
        ("amount", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                             ("scale", StandardScaler())]), ["Amount"]),
    ]
    transformers.append(("time", StandardScaler(), ["Time"]))
    return Pipeline([
        ("preprocessing", ColumnTransformer(transformers)),
        ("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)),
    ])


def selection_metrics(labels, amounts, score, canceled, costs, probabilistic):
    y = np.asarray(labels)
    score = np.asarray(score)
    positives, negatives = y == 1, y == 0
    fpr, recall, thresholds = roc_curve(y, score)
    eligible = np.flatnonzero(fpr <= 0.001)
    best = eligible[np.argmax(recall[eligible])]
    threshold = float(thresholds[best])
    simulated = action_costs(y, amounts, canceled, costs)
    profits = action_profits(y, amounts, canceled, costs)
    oracle = float(((1 - y) * np.asarray(amounts) * costs["legitimate_margin_fraction"]).sum())
    return {
        "rows": len(y), "frauds": int(positives.sum()),
        "average_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "brier_score": float(brier_score_loss(y, score)) if probabilistic else None,
        "cancellation_rate": float(np.mean(canceled)),
        "fraud_canceled_recall": float(np.mean(canceled[positives])),
        "legitimate_cancellation_rate": float(np.mean(canceled[negatives])),
        "legitimate_approval_rate": float(np.mean(~canceled[negatives])),
        "simulated_total_profit": float(profits.sum()),
        "simulated_profit_per_1000": float(profits.mean() * 1000),
        "maximum_possible_profit": oracle,
        "profit_pct_of_maximum": float(100 * profits.sum() / oracle) if oracle > 0 else None,
        "simulated_total_cost": float(simulated.sum()),
        "simulated_cost_per_1000": float(simulated.mean() * 1000),
        "selection_fpr_diagnostic_recall": float(recall[best]),
        "selection_fpr_diagnostic_realized_fpr": float(fpr[best]),
        "selection_fpr_diagnostic_threshold": threshold if np.isfinite(threshold) else None,
        "selection_fpr_diagnostic_predict_none": not np.isfinite(threshold),
    }


def main():
    costs = json.loads((ROOT / "config/cost-scenario.json").read_text())
    protocol = json.loads((ROOT / "config/evaluation.json").read_text())
    audit = json.loads((ROOT / "reports/duplicate-split-audit.json").read_text())
    if sha256(ROOT / "config/evaluation.json") != audit["config_sha256"]:
        raise ValueError("Split configuration changed; rerun audit_splits.py")
    assignment_path = ROOT / audit["assignments"]["path"]
    if sha256(assignment_path) != audit["assignments"]["sha256"]:
        raise ValueError("Assignment checksum mismatch; rerun audit_splits.py")
    assignments = pd.read_csv(assignment_path).set_index("source_row_id")
    parts, offset = [], 0
    for source in audit["source_files"]:
        path = ROOT / source["path"]
        if sha256(path) != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path.name}")
        for chunk in pd.read_csv(path, chunksize=50000):
            chunk.index = np.arange(offset, offset + len(chunk))
            offset += len(chunk)
            development = assignments.loc[chunk.index, "split"].isin(["train", "selection"]).to_numpy()
            parts.append(chunk.loc[development])
    data = pd.concat(parts)
    if not data.index.is_unique or not np.isfinite(data.to_numpy()).all():
        raise ValueError("Invalid input values or source row IDs")
    split_labels = assignments.loc[data.index, "split"]
    if not split_labels.isin(["train", "selection"]).all() or not audit["group_disjoint"]:
        raise ValueError("Expected group-disjoint random development splits")
    if assignments.groupby("pattern_group_id")["split"].nunique().max() != 1:
        raise ValueError("Pattern groups overlap across splits")
    train = data.loc[split_labels.eq("train")]
    selection = data.loc[split_labels.eq("selection")]
    if train.empty or selection.empty or train.Class.nunique() != 2 or selection.Class.nunique() != 2:
        raise ValueError("Both development splits require both classes")
    y = selection.Class.to_numpy()
    amount = selection.Amount.to_numpy()
    model_dir = ROOT / "models/baselines"
    model_dir.mkdir(parents=True, exist_ok=True)
    prior = float(train.Class.mean())
    amount_threshold = float(train.Amount.quantile(0.99))
    # Hand-calculated contract: approve legitimate +25, approve fraud -100; cancel 0.
    assert np.allclose(action_profits([0, 1, 0, 1], [100] * 4, [False, False, True, True], costs),
                       [25, -100, 0, 0])
    assert np.array_equal(cancel_from_probability([0.19, 0.2, 0.21, 1], [100, 100, 100, 0], costs),
                          [False, False, True, False])
    metadata = {
        "evaluated_split": "selection", "train_rows": len(train), "train_frauds": int(train.Class.sum()),
        "selection_rows": len(selection), "selection_frauds": int(selection.Class.sum()),
        "evaluation_split_scored": False, "calibration_split_used": False,
        "split_strategy": protocol["strategy"], "random_seed": protocol["random_seed"],
        "training_prior": prior, "amount_rule_threshold_from_train_p99": amount_threshold,
        "code_sha256": sha256(Path(__file__)), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "protocol_sha256": sha256(ROOT / "config/evaluation.json"),
        "cost_config_sha256": sha256(ROOT / "config/cost-scenario.json"),
        "assignments_sha256": audit["assignments"]["sha256"],
        "poetry_lock_sha256": sha256(ROOT / "poetry.lock"), "source_files": audit["source_files"],
        "versions": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "joblib", "mlflow"]},
        "notes": ["Descriptive selection results, not final evaluation or rollout impact.",
                  "Logistic scores have not been post-hoc calibrated; cost-policy results are provisional.",
                  "FPR diagnostic thresholds are chosen and assessed on selection itself; not an independent estimate.",
                  "No class weighting, resampling, tuning, challenge/review action, or calibration in this initial comparison.",
                  "The binary approve/cancel policy has no review-capacity constraint.",
                  "Exact repeats and amounts above p99 remain in the fitted/scored populations."]
    }
    setup_tracking("02-baselines")
    results, prediction_frames = [], []
    for name in ["approve_all", "amount_rule", "logistic_regression"]:
        with mlflow.start_run(run_name=name) as run:
            probabilistic = name != "amount_rule"
            if name == "approve_all":
                score = np.full(len(selection), prior)
                canceled = np.zeros(len(selection), dtype=bool)
            elif name == "amount_rule":
                score = amount.copy()
                canceled = amount >= amount_threshold
            else:
                input_columns = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
                model = make_logistic()
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(train[input_columns], train.Class)
                score = model.predict_proba(selection[input_columns])[:, 1]
                canceled = cancel_from_probability(score, amount, costs)
                bundle = {"pipeline": model, "input_columns": input_columns, "policy": "cancel_when_expected_profit_negative",
                          "cost_scenario": costs, "protocol": protocol, "lineage": metadata,
                          "posthoc_calibrated": False}
                path = model_dir / f"{name}.joblib"
                joblib.dump(bundle, path)
                restored = joblib.load(path)["pipeline"]
                assert np.allclose(restored.predict_proba(selection[input_columns].iloc[:20])[:, 1], score[:20])
                mlflow.log_artifact(str(path), artifact_path="fitted_pipeline")
            metrics = selection_metrics(y, amount, score, canceled, costs, probabilistic)
            results.append({"name": name, "mlflow_run_id": run.info.run_id, **metrics})
            frame = pd.DataFrame({"source_row_id": selection.index, "model": name,
                                  "score": score, "score_is_probability": probabilistic,
                                  "canceled": canceled, "Class": y, "Amount": amount})
            frame["simulated_cost"] = action_costs(y, amount, canceled, costs)
            frame["simulated_profit"] = action_profits(y, amount, canceled, costs)
            prediction_frames.append(frame)
            mlflow.log_params({"split": "selection", "protocol": protocol["version"], "cost_scenario": costs["version"],
                               "seed": 42, "train_rows": len(train), "selection_rows": len(selection),
                               "classifier": name, "include_time": name == "logistic_regression", "calibrated": False})
            mlflow.set_tags({"code_sha256": metadata["code_sha256"], "protocol_sha256": metadata["protocol_sha256"],
                             "cost_config_sha256": metadata["cost_config_sha256"], "assignments_sha256": metadata["assignments_sha256"]})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None and np.isfinite(v)})
            mlflow.log_dict(metadata, "lineage.json")
            mlflow.log_artifact(str(ROOT / "config/cost-scenario.json"), artifact_path="configuration")
            mlflow.log_artifact(str(ROOT / "config/evaluation.json"), artifact_path="configuration")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    prediction_path = ROOT / "data/processed/baseline-selection-predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    baseline = results[0]["simulated_cost_per_1000"]
    for result in results:
        result["simulated_cost_reduction_vs_approve_all_pct"] = 100 * (baseline - result["simulated_cost_per_1000"]) / baseline
    report = {"metadata": metadata, "results": results,
              "selection_predictions": {"path": str(prediction_path.relative_to(ROOT)), "sha256": sha256(prediction_path)}}
    (ROOT / "reports/baseline-results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    columns = ["name", "average_precision", "brier_score", "fraud_canceled_recall", "cancellation_rate", "simulated_profit_per_1000", "profit_pct_of_maximum"]
    print(pd.DataFrame(results)[columns].round(6).to_string(index=False))
    print("Selection only; calibration and evaluation splits were not scored.")
    print("Saved fitted pipelines, selection predictions, report, and three local MLflow runs.")


if __name__ == "__main__":
    main()
