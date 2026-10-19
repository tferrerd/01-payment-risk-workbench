"""Predict payment decisions for a CSV from the project root."""
import argparse
import csv
from pathlib import Path

import pandas as pd

from .pipeline import DEFAULT_ARTIFACT, PaymentRiskPipeline


def predict_csv(input_path, output_path, model_path=DEFAULT_ARTIFACT):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve() or output_path.exists():
        raise ValueError("Output must be a new file, separate from the input")
    with input_path.open(newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(header) != len(set(header)):
        raise ValueError("CSV needs a header with unique column names")
    pipeline = PaymentRiskPipeline.load(model_path)
    data = pd.read_csv(input_path)
    predictions = pipeline.predict(data)
    collisions = set(predictions.columns) & set(data.columns)
    if collisions:
        raise ValueError("Output fields already occur in input: " + ", ".join(sorted(collisions)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([data, predictions], axis=1).to_csv(output_path, index=False, mode="x")
    return len(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    try:
        rows = predict_csv(args.input, args.output, args.model)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
    print(f"Saved {rows:,} development-candidate predictions to {args.output}")


if __name__ == "__main__":
    main()
