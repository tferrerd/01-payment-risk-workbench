"""Tune LightGBM using training-only grouped CV; compare the fixed winner on selection."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold

from payment_risk import PaymentRiskPipeline
from payment_risk.pipeline import DEFAULT_ARTIFACT, FEATURES
# Historical analysis scripts use sibling imports; reuse them without changing the freeze.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baselines import ROOT, action_profits, cancel_from_probability, setup_tracking, sha256
from run_final_evaluation import paired_group_bootstrap, policy_metrics

BASELINE = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=40,
                reg_alpha=0.0, reg_lambda=0.0, subsample=1.0, colsample_bytree=1.0)


def load_split(name, assignments, audit):
    if name not in {"train", "selection"}:
        raise ValueError("This experiment may only use training and selection payments")
    parts, offset = [], 0
    for source in audit["source_files"]:
        path = ROOT / source["path"]
        if sha256(path) != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
        for chunk in pd.read_csv(path, chunksize=50000):
            chunk.index = np.arange(offset, offset + len(chunk))
            offset += len(chunk)
            parts.append(chunk.loc[assignments.loc[chunk.index, "split"].eq(name).to_numpy()])
    frame = pd.concat(parts)
    if not frame.index.is_unique or not np.isfinite(frame.to_numpy()).all() or frame.Class.nunique() != 2:
        raise ValueError("Invalid development population")
    return frame


def training_folds(frame, assignments):
    assigned = assignments.loc[frame.index]
    if not assigned.split.eq("train").all():
        raise ValueError("Hyperparameter search folds must contain training rows only")
    groups = assigned.pattern_group_id
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(frame, frame.Class, groups))
    for fit, validation in folds:
        if set(groups.iloc[fit]) & set(groups.iloc[validation]):
            raise ValueError("Pattern leakage across CV folds")
        if any(frame.Class.iloc[index].nunique() != 2 for index in [fit, validation]):
            raise ValueError("Each fold must contain both labels")
    return folds


def parameters(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 47),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "subsample": trial.suggest_float("subsample", 0.7, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1),
    }


def classifier(params):
    return LGBMClassifier(**params, subsample_freq=1, random_state=42, n_jobs=4,
                          deterministic=True, force_col_wise=True, verbosity=-1)


def score_metrics(frame, score, costs):
    canceled = cancel_from_probability(score, frame.Amount.to_numpy(), costs)
    return policy_metrics(frame.Class.to_numpy(), frame.Amount.to_numpy(), score, canceled, costs)


def run(args):
    import optuna

    if args.trials < 2 or not re.fullmatch(r"[A-Za-z0-9_-]+", args.study_name):
        raise ValueError("Use at least two trials and a study name containing letters, digits, _ or -")
    output = ROOT / "reports/optuna" / args.study_name
    if output.exists():
        raise ValueError("Report directory already exists; choose a new --study-name to preserve prior runs")
    audit = json.loads((ROOT / "reports/duplicate-split-audit.json").read_text())
    assignment_path = ROOT / audit["assignments"]["path"]
    if sha256(assignment_path) != audit["assignments"]["sha256"] or sha256(ROOT / "config/evaluation.json") != audit["config_sha256"]:
        raise ValueError("Split configuration/assignment lineage differs")
    assignments = pd.read_csv(assignment_path).set_index("source_row_id")
    if not audit["group_disjoint"] or assignments.groupby("pattern_group_id").split.nunique().max() != 1:
        raise ValueError("Pattern groups cross outer splits")
    costs = json.loads((ROOT / "config/cost-scenario-low-margin.json").read_text())
    threshold = costs["legitimate_margin_fraction"] / (costs["legitimate_margin_fraction"] + costs["fraud_loss_fraction"])
    train = load_split("train", assignments, audit)
    folds = training_folds(train, assignments)
    x, y = train[FEATURES], train.Class.to_numpy()
    output.mkdir(parents=True)
    study_dir = ROOT / "data/processed/optuna"
    study_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "study_name": args.study_name,
        "objective": "maximize pooled training-only grouped 3-fold out-of-fold simulated profit per 1000",
        "trials_requested": args.trials, "sampler": "TPESampler(seed=42, n_startup_trials=10)",
        "pruner": "NopPruner; all trials complete all folds", "parallel_trials": 1,
        "model_threads": 4, "seed": 42, "train_rows": len(train), "train_frauds": int(y.sum()),
        "threshold": threshold, "threshold_tuned": False, "calibration_split_used": False,
        "evaluation_split_scored": False, "posthoc_calibrated": False,
        "features": FEATURES, "class_weighting_or_resampling": False,
        "selection_role": "Post-search development comparison only; no winner or parameter chosen using selection outcomes",
        "code_sha256": sha256(Path(__file__)), "poetry_lock_sha256": sha256(ROOT / "poetry.lock"),
        "requirements_tuning_sha256": sha256(ROOT / "requirements-tuning.txt"),
        "assignments_sha256": sha256(assignment_path), "protocol_sha256": audit["config_sha256"],
        "cost_config_sha256": sha256(ROOT / "config/cost-scenario-low-margin.json"),
        "deployed_artifact_sha256": sha256(DEFAULT_ARTIFACT), "source_files": audit["source_files"],
        "git_head_before_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "versions": {name: version(name) for name in ["optuna", "lightgbm", "scikit-learn", "mlflow", "numpy", "pandas"]},
        "folds": [{"fold": i, "fit_rows": len(fit), "validation_rows": len(val),
                   "validation_frauds": int(y[val].sum())} for i, (fit, val) in enumerate(folds)],
    }
    uri = setup_tracking("10-optuna-low-margin")
    print(f"Training-only search: {len(train):,} payments, {int(y.sum())} frauds; {args.trials} trials × 3 folds", flush=True)
    started = time.monotonic()
    study = optuna.create_study(study_name=args.study_name, direction="maximize",
                               storage="sqlite:///" + str(study_dir / f"{args.study_name}.sqlite3"),
                               sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
                               pruner=optuna.pruners.NopPruner())
    study.enqueue_trial(BASELINE)
    with mlflow.start_run(run_name=args.study_name) as parent:
        parent_id = parent.info.run_id
        mlflow.log_params({"trials": args.trials, "folds": 3, "threshold": threshold,
                           "objective": "training_cv_profit_per_1000", "scenario": costs["version"]})
        mlflow.set_tags({"scope": "development_training_cv", "model_promoted": "false"})
        mlflow.log_dict(metadata, "lineage.json")

        def objective(trial):
            params = parameters(trial)
            predictions = np.empty(len(train))
            fold_results = []
            with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True) as child:
                mlflow.log_params(params)
                trial.set_user_attr("mlflow_run_id", child.info.run_id)
                for fold, (fit, validation) in enumerate(folds):
                    model = classifier(params)
                    model.fit(x.iloc[fit], y[fit])
                    predictions[validation] = model.predict_proba(x.iloc[validation])[:, 1]
                    metrics = score_metrics(train.iloc[validation], predictions[validation], costs)
                    fold_results.append(metrics)
                    mlflow.log_metric(f"fold_{fold}_profit_per_1000", metrics["simulated_profit_per_1000"])
                metrics = score_metrics(train, predictions, costs)
                trial.set_user_attr("fold_metrics", fold_results)
                for name in ["average_precision", "brier_score", "fraud_canceled_recall", "legitimate_cancellation_rate"]:
                    trial.set_user_attr(name, metrics[name])
                mlflow.log_metrics({"cv_" + key: value for key, value in metrics.items() if isinstance(value, (int, float))})
                return metrics["simulated_profit_per_1000"]

        study.optimize(objective, n_trials=args.trials, n_jobs=1, gc_after_trial=True)
        best = study.best_trial
        # Freeze the winning parameters before loading or scoring selection payments.
        (output / "best-parameters.json").write_text(json.dumps({"trial": best.number, "parameters": best.params,
                                                               "training_cv_profit_per_1000": best.value}, indent=2) + "\n")
        tuned = classifier(best.params).fit(x, y)
        artifact_path = ROOT / "models/tuning" / f"{args.study_name}.joblib"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": tuned, "input_columns": FEATURES, "cost_scenario": costs,
                     "posthoc_calibrated": False, "lineage": metadata, "optuna_trial": best.number}, artifact_path)
        selection = load_split("selection", assignments, audit)
        deployed = PaymentRiskPipeline.load()
        scores = {"optuna_lightgbm": tuned.predict_proba(selection[FEATURES])[:, 1],
                  "deployed_lightgbm": deployed.model.predict_proba(selection[FEATURES])[:, 1]}
        np.testing.assert_allclose(joblib.load(artifact_path)["model"].predict_proba(selection[FEATURES].iloc[:20])[:, 1],
                                   scores["optuna_lightgbm"][:20], rtol=0, atol=1e-12)
        results = {name: score_metrics(selection, score, costs) for name, score in scores.items()}
        profits = pd.DataFrame({name: action_profits(selection.Class.to_numpy(), selection.Amount.to_numpy(),
                                                  cancel_from_probability(score, selection.Amount.to_numpy(), costs), costs)
                                for name, score in scores.items()})
        uncertainty, _ = paired_group_bootstrap(profits, assignments.loc[selection.index, "pattern_group_id"],
                                               "optuna_lightgbm", repeats=2000, seed=42)
        predictions_path = ROOT / "data/processed/optuna" / f"{args.study_name}-selection.csv"
        pd.DataFrame({"source_row_id": selection.index, **scores}).to_csv(predictions_path, index=False)
        study.trials_dataframe().to_csv(output / "trials.csv", index=False)
        baseline_cv = study.trials[0].value
        report = {"metadata": metadata, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                  "elapsed_seconds": time.monotonic() - started, "completed_trials": len(study.trials),
                  "mlflow_parent_run_id": parent_id, "tracking_uri": uri,
                  "best_trial": best.number, "best_parameters": best.params, "best_cv_profit_per_1000": best.value,
                  "baseline_cv_profit_per_1000": baseline_cv, "selection_results": results,
                  "selection_uncertainty": uncertainty, "model_promoted": False,
                  "model_artifact": {"path": str(artifact_path.relative_to(ROOT)), "sha256": sha256(artifact_path)},
                  "selection_predictions": {"path": str(predictions_path.relative_to(ROOT)), "sha256": sha256(predictions_path)},
                  "limitations": ["Raw-score economic policies remain provisional; the threshold was held fixed, not tuned.",
                                  "CV scores were optimized across trials and are not unbiased performance estimates.",
                                  "Selection has prior development/EDA exposure; the paired interval does not account for search uncertainty.",
                                  "Calibration and evaluation were not used in this experiment; no independent final-test or production claim."]}
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        mlflow.log_metrics({"best_cv_profit_per_1000": best.value, "baseline_cv_profit_per_1000": baseline_cv})
        for name, metrics in results.items():
            mlflow.log_metrics({name + "_selection_" + key: value for key, value in metrics.items() if isinstance(value, (int, float))})
        mlflow.log_artifact(str(artifact_path), artifact_path="tuned_model")
        for path in output.iterdir():
            mlflow.log_artifact(str(path), artifact_path="search_results")
    print(pd.DataFrame(results).T[["average_precision", "brier_score", "simulated_profit_per_1000", "frauds_canceled", "legitimate_canceled"]].to_string(), flush=True)
    print(f"Best trial {best.number}; CV profit {baseline_cv:.3f} → {best.value:.3f}; reports: {output}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--study-name", default="lightgbm-low-margin-optuna-v1")
    run(parser.parse_args())
