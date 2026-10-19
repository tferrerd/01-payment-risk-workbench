"""Prepare 200 curated selection examples; never use evaluation payments for the demo."""
import json
from pathlib import Path

import pandas as pd

from payment_risk.pipeline import FEATURES, file_sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    audit = json.loads((ROOT / "reports/duplicate-split-audit.json").read_text())
    assignment = ROOT / "data/processed/split-assignments.csv"
    if file_sha256(assignment) != audit["assignments"]["sha256"]:
        raise ValueError("Split assignments differ from the saved audit")
    sources, lineage = [], {"data/processed/split-assignments.csv": file_sha256(assignment)}
    for source in audit["source_files"]:
        path = ROOT / source["path"]
        if file_sha256(path) != source["sha256"]:
            raise ValueError(f"Source changed: {source['path']}")
        sources.append(pd.read_csv(path))
        lineage[source["path"]] = source["sha256"]
    data = pd.concat(sources, ignore_index=True)
    split = pd.read_csv(assignment).set_index("source_row_id")
    selection = data.loc[split.index[split.split.eq("selection")]]
    parts = []
    for label, size in [(0, 190), (1, 10)]:
        pool = selection.loc[selection.Class.eq(label)]
        # Retain recognizable earlier examples, including a costly legitimate cancellation.
        required = pool.loc[pool.index.intersection([5, 4920, 514, 274771])]
        parts.append(pd.concat([required, pool.drop(required.index).sample(size - len(required), random_state=42)]))
    sample = pd.concat(parts).sort_index()
    rows = [{"payment_id": f"PAY-{index:06d}", "source_row_id": int(index), "label": int(row.Class),
             "features": {feature: float(row[feature]) for feature in FEATURES}} for index, row in sample.iterrows()]
    result = {"source_split": "selection", "curated": True, "sampling_seed": 42,
              "description": "200 selection payments: 190 legitimate and 10 fraud. Enriched for demonstration; not population-representative.",
              "rows": rows}
    path = ROOT / "examples/explorer-sample.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    report = {"rows": len(rows), "frauds": int(sample.Class.sum()), "scope": result["description"],
              "sample_path": str(path.relative_to(ROOT)), "sample_sha256": file_sha256(path),
              "source_row_ids": sample.index.tolist(), "lineage": lineage,
              "preparation_code_sha256": file_sha256(Path(__file__))}
    (ROOT / "reports/explorer-sample.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(rows)} selection-only demo payments, including {int(sample.Class.sum())} labeled frauds.")


if __name__ == "__main__":
    main()
