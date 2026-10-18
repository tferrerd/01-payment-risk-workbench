"""Audit duplicates and persist the temporal-v1 split without fitting a model."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    config_path = ROOT / "config/evaluation.json"
    config = json.loads(config_path.read_text())
    paths = sorted((ROOT / "data/raw").glob("creditcard_*.csv"))
    if len(paths) != 5:
        raise ValueError(f"Expected five source CSV parts; found {len(paths)}")
    frames, source_rows, sources = [], [], []
    for path in paths:
        part = pd.read_csv(path)
        frames.append(part)
        source_rows.append(pd.DataFrame({
            "source_file": path.name,
            "source_data_row": np.arange(1, len(part) + 1),
        }))
        sources.append({"path": str(path.relative_to(ROOT)), "rows": len(part), "sha256": sha256(path)})
    df = pd.concat(frames, ignore_index=True)
    expected = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    if list(df.columns) != expected or len(df) != 284807:
        raise ValueError("Unexpected source schema or population")
    if not np.isfinite(df.to_numpy()).all() or not df.Class.isin([0, 1]).all():
        raise ValueError("Non-finite numeric values or invalid labels")
    if (df.Time < 0).any() or not df.Time.is_monotonic_increasing:
        raise ValueError("Expected nonnegative chronological Time")
    boundaries = config["boundary_seconds"]
    names = config["split_names"]
    if boundaries != sorted(set(boundaries)) or len(names) != len(boundaries) + 1:
        raise ValueError("Invalid split boundaries")
    split_index = np.searchsorted(boundaries, df.Time, side="right")
    # Runnable boundary check: exact boundaries must enter the later window.
    assert np.array_equal(np.searchsorted(boundaries, boundaries, side="right"), [1, 2, 3])
    assignments = pd.concat(source_rows, ignore_index=True)
    assignments.insert(0, "source_row_id", np.arange(len(df)))
    assignments["Time"] = df.Time
    assignments["split"] = np.array(names)[split_index]
    split_summary = []
    for name in names:
        rows = df.loc[assignments.split.eq(name)]
        frauds = int(rows.Class.sum())
        if rows.empty or frauds == 0 or frauds == len(rows):
            raise ValueError(f"Split {name} needs both classes")
        split_summary.append({"split": name, "rows": len(rows), "frauds": frauds,
                              "fraud_pct": 100 * frauds / len(rows),
                              "time_min": float(rows.Time.min()), "time_max": float(rows.Time.max())})

    # Group exact parsed numeric values, rather than approximate similarity.
    # Excluding Time identifies repeated patterns, not confirmed repeat transactions.
    predictors = expected[:-1]
    definitions = {"all_columns": expected, "all_predictors": predictors,
                   "predictors_without_time": predictors[1:]}
    duplicate_summary = {}
    for definition, columns in definitions.items():
        ids = pd.Series(pd.factorize(pd.MultiIndex.from_frame(df[columns]), sort=False)[0])
        group_frame = pd.DataFrame({"group": ids, "split_index": split_index, "label": df.Class})
        groups = group_frame.groupby("group", sort=False).agg(
            rows=("label", "size"), label_count=("label", "nunique"),
            split_count=("split_index", "nunique"), first_split=("split_index", "min"))
        seen_before = ids.map(groups.first_split).to_numpy() < split_index
        assignments[f"{definition}_seen_in_earlier_split"] = seen_before
        per_split = {name: int(seen_before[split_index == i].sum()) for i, name in enumerate(names)}
        duplicate_summary[definition] = {
            "columns": columns, "unique_groups": len(groups),
            "duplicate_rows_beyond_first": len(df) - len(groups),
            "repeated_groups": int(groups.rows.gt(1).sum()),
            "conflicting_label_groups": int(groups.label_count.gt(1).sum()),
            "groups_spanning_splits": int(groups.split_count.gt(1).sum()),
            "rows_seen_in_earlier_split": per_split,
        }
    assert assignments.groupby("Time")["split"].nunique().max() == 1
    assert assignments.source_row_id.is_unique and len(assignments) == len(df)
    evaluation = assignments.split.eq("evaluation")
    unseen = evaluation & ~assignments.predictors_without_time_seen_in_earlier_split
    sensitivity = {"rows": int(unseen.sum()), "frauds": int(df.loc[unseen, "Class"].sum()),
                   "definition": "Evaluation V1–V28/Amount vector absent from every earlier split; duplicates within evaluation retained."}

    processed = ROOT / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    assignments_path = processed / "split-assignments.csv"
    assignments.to_csv(assignments_path, index=False)
    report = {
        "protocol": config, "source_files": sources,
        "config_sha256": sha256(config_path), "audit_script_sha256": sha256(Path(__file__)),
        "split_summary": split_summary, "duplicates": duplicate_summary,
        "unseen_pattern_evaluation": sensitivity,
        "assignments": {"path": str(assignments_path.relative_to(ROOT)),
                        "rows": len(assignments), "sha256": sha256(assignments_path)},
        "limitations": ["No customer/account identifiers; repeated patterns do not establish shared identity.",
                        "Upstream PCA fitting population and feature availability cannot be independently verified.",
                        "Earlier EDA used labels from all windows; final evaluation is not analytically untouched.",
                        "Only two days of observations; small fraud counts limit precision and temporal generalization."]
    }
    report_path = ROOT / "reports/duplicate-split-audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(pd.DataFrame(split_summary).to_string(index=False))
    for definition, summary in duplicate_summary.items():
        print(f"{definition}: {summary['duplicate_rows_beyond_first']:,} excess repeats; "
              f"{summary['conflicting_label_groups']} conflicting-label groups; "
              f"{summary['groups_spanning_splits']:,} cross-split groups")
    print(f"Previously unseen-pattern evaluation: {sensitivity['rows']:,} rows, {sensitivity['frauds']} frauds")
    print(f"Saved {assignments_path.relative_to(ROOT)} and {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
