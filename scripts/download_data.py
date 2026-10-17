"""Download and verify the unchanged OpenML 1597 v1 fraud dataset; uses Python stdlib and curl."""

import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://www.openml.org/data/v1/download/1673544/creditcard.arff"
METADATA_URL = "https://www.openml.org/api/v1/json/data/1597"
EXPECTED_MD5 = "178bcf9bb1f31a3dfe12d0e577884add"
EXPECTED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]


def checksums(path):
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        subprocess.run([
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--retry", "2", "--max-time", "300", url, "--output", str(temporary),
        ], check=True)
        if path.suffix == ".arff" and checksums(temporary)["md5"] != EXPECTED_MD5:
            raise ValueError("Downloaded file does not match the publisher's OpenML checksum")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def profile(path):
    columns, rows, frauds = [], 0, 0
    ranges = {name: [float("inf"), float("-inf")] for name in ("Time", "Amount")}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip().lower().startswith("@attribute"):
                columns.append(shlex.split(line)[1])
            if line.strip().lower() == "@data":
                break
        if columns != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected ARFF schema: {columns}")
        for row in csv.reader(stream):
            if not row or row[0].lstrip().startswith("%"):
                continue
            if len(row) != len(columns):
                raise ValueError(f"Invalid column count on data row {rows + 1}")
            values = [float(value.strip(" '\"")) for value in row]
            if not all(math.isfinite(value) for value in values) or values[-1] not in (0, 1):
                raise ValueError(f"Invalid numeric values or fraud label on row {rows + 1}")
            rows += 1
            frauds += int(values[-1])
            for name, value in (("Time", values[0]), ("Amount", values[-2])):
                ranges[name][0] = min(ranges[name][0], value)
                ranges[name][1] = max(ranges[name][1], value)
    if (rows, frauds) != (284807, 492):
        raise ValueError(f"Unexpected dataset population: {rows} rows and {frauds} fraud labels")
    return {
        "rows": rows, "columns": columns, "fraud_count": frauds,
        "fraud_rate": frauds / rows, "numeric_values_finite": True,
        "time_seconds_min_max": ranges["Time"], "amount_min_max": ranges["Amount"],
        "note": "Schema/count/range checks only; this is not EDA or model evaluation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check local raw data without network access")
    args = parser.parse_args()
    raw = ROOT / "data/raw/creditcard.arff"
    metadata_path = ROOT / "data/source-openml-1597.json"
    manifest_path = ROOT / "data/manifest.json"
    if not args.verify_only:
        if not metadata_path.exists():
            download(METADATA_URL, metadata_path)
        metadata = json.loads(metadata_path.read_text())["data_set_description"]
        if (metadata["id"], metadata["version"], metadata["md5_checksum"]) != ("1597", "1", EXPECTED_MD5):
            raise ValueError("Unexpected OpenML dataset version or source checksum")
        if not raw.exists():
            download(SOURCE_URL, raw)
    digests = checksums(raw)
    if digests["md5"] != EXPECTED_MD5:
        raise ValueError("Local raw file does not match the published checksum; it will not be overwritten")
    summary = profile(raw)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["sha256"] != digests["sha256"]:
            raise ValueError("Local file does not match the saved provenance manifest")
    elif not args.verify_only:
        manifest = {
            "dataset": "ULB credit card fraud / OpenML 1597", "openml_version": 1,
            "source_url": SOURCE_URL, "metadata_url": METADATA_URL,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_path": "data/raw/creditcard.arff", "bytes": raw.stat().st_size,
            **digests, "source_license_label": "Public (OpenML metadata)",
            "publication_note": "Original data retained locally; confirm publisher terms before redistribution.",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.verify_only:
        report = ROOT / "reports/dataset-profile.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"verified": True, "raw_path": str(raw), **digests, **summary}, indent=2))


if __name__ == "__main__":
    main()
