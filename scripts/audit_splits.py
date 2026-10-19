"""Audit duplicates and persist seeded, stratified, group-disjoint random splits."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_group_splits(labels, group_ids, config):
    """Stratify indivisible exact-pattern groups; Time is not an argument."""
    names = config["split_names"]
    fractions = config["target_fractions"]
    if len(names) != 4 or len(fractions) != 4 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError("Expected four positive split fractions summing to one")
    rows = pd.DataFrame({"group": np.asarray(group_ids), "label": np.asarray(labels)})
    grouped = rows.groupby("group", sort=False).agg(label=("label", "first"), distinct_labels=("label", "nunique"))
    if grouped.distinct_labels.gt(1).any():
        raise ValueError("Identical-pattern groups contain conflicting labels")
    seed = config["random_seed"]
    train, remaining = train_test_split(grouped, train_size=fractions[0], stratify=grouped.label, random_state=seed)
    selection, remaining = train_test_split(remaining, train_size=fractions[1] / sum(fractions[1:]), stratify=remaining.label, random_state=seed)
    calibration, evaluation = train_test_split(remaining, train_size=fractions[2] / sum(fractions[2:]), stratify=remaining.label, random_state=seed)
    allocations = pd.concat([pd.Series(name, index=part.index) for name, part in zip(names, [train, selection, calibration, evaluation])])
    result = rows["group"].map(allocations)
    if result.isna().any() or allocations.index.has_duplicates:
        raise ValueError("Incomplete or overlapping group allocation")
    return result.to_numpy()


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
    if (df.Time < 0).any():
        raise ValueError("Expected nonnegative elapsed Time")
    names = config["split_names"]
    pattern_ids = pd.factorize(pd.MultiIndex.from_frame(df[config["group_columns"]]), sort=False)[0]
    split_labels = assign_group_splits(df.Class, pattern_ids, config)
    assignments = pd.concat(source_rows, ignore_index=True)
    assignments.insert(0, "source_row_id", np.arange(len(df)))
    assignments["Time"] = df.Time
    assignments["pattern_group_id"] = pattern_ids
    assignments["split"] = split_labels
    split_summary = []
    for name in names:
        rows = df.loc[assignments.split.eq(name)]
        frauds = int(rows.Class.sum())
        if rows.empty or frauds == 0 or frauds == len(rows):
            raise ValueError(f"Split {name} needs both classes")
        split_summary.append({"split": name, "rows": len(rows), "frauds": frauds,
                              "fraud_pct": 100 * frauds / len(rows),
                              "achieved_fraction": len(rows) / len(df),
                              "time_min": float(rows.Time.min()), "time_max": float(rows.Time.max())})

    # Group exact parsed numeric values, rather than approximate similarity.
    # Excluding Time identifies repeated patterns, not confirmed repeat transactions.
    predictors = expected[:-1]
    definitions = {"all_columns": expected, "all_predictors": predictors,
                   "predictors_without_time": predictors[1:]}
    duplicate_summary = {}
    for definition, columns in definitions.items():
        ids = pd.Series(pattern_ids if definition == "predictors_without_time" else pd.factorize(pd.MultiIndex.from_frame(df[columns]), sort=False)[0])
        group_frame = pd.DataFrame({"group": ids, "split": split_labels, "label": df.Class})
        groups = group_frame.groupby("group", sort=False).agg(
            rows=("label", "size"), label_count=("label", "nunique"),
            split_count=("split", "nunique"))
        overlaps = ids.map(groups.split_count).gt(1)
        per_split = {name: int(overlaps[split_labels == name].sum()) for name in names}
        duplicate_summary[definition] = {
            "columns": columns, "unique_groups": len(groups),
            "duplicate_rows_beyond_first": len(df) - len(groups),
            "repeated_groups": int(groups.rows.gt(1).sum()),
            "conflicting_label_groups": int(groups.label_count.gt(1).sum()),
            "groups_spanning_splits": int(groups.split_count.gt(1).sum()),
            "rows_overlapping_other_splits": per_split,
        }
    assert assignments.groupby("pattern_group_id")["split"].nunique().max() == 1
    assert all(result["groups_spanning_splits"] == 0 for result in duplicate_summary.values())
    assert assignments.source_row_id.is_unique and len(assignments) == len(df)

    processed = ROOT / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    assignments_path = processed / "split-assignments.csv"
    assignments.to_csv(assignments_path, index=False)
    report = {
        "protocol": config, "source_files": sources,
        "config_sha256": sha256(config_path), "audit_script_sha256": sha256(Path(__file__)),
        "split_summary": split_summary, "duplicates": duplicate_summary,
        "group_disjoint": True,
        "assignments": {"path": str(assignments_path.relative_to(ROOT)),
                        "rows": len(assignments), "sha256": sha256(assignments_path)},
        "limitations": ["No customer/account identifiers; repeated patterns do not establish shared identity.",
                        "Upstream PCA fitting population and feature availability cannot be independently verified.",
                        "Earlier full-dataset EDA and chronological baseline experiments preceded this random protocol; final evaluation is not analytically untouched.",
                        "Same-population random evaluation does not establish future-period or unseen-customer generalization."]
    }
    report_path = ROOT / "reports/duplicate-split-audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(pd.DataFrame(split_summary).to_string(index=False))
    for definition, summary in duplicate_summary.items():
        print(f"{definition}: {summary['duplicate_rows_beyond_first']:,} excess repeats; "
              f"{summary['conflicting_label_groups']} conflicting-label groups; "
              f"{summary['groups_spanning_splits']:,} cross-split groups")
    print("All four splits are group-disjoint; Time remains a model input.")
    print(f"Saved {assignments_path.relative_to(ROOT)} and {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
