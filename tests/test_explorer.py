import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from fastapi import Request

from payment_risk import PaymentRiskPipeline
from payment_risk.api import Transaction
from payment_risk.explorer import explain
from payment_risk.pipeline import FEATURES, file_sha256

ROOT = Path(__file__).resolve().parents[1]


class ExplorerTest(unittest.TestCase):
    def test_real_contributions_reconstruct_predictions(self):
        pipeline = PaymentRiskPipeline.load()
        request = Request({"type": "http", "app": SimpleNamespace(state=SimpleNamespace(pipeline=pipeline))})
        payload = json.loads((ROOT / "examples/predict-request.json").read_text())
        expected = json.loads((ROOT / "examples/predict-expected.json").read_text())
        for row, prediction in zip(payload["transactions"], expected):
            result = explain(Transaction(**row), request)
            self.assertEqual([item["feature"] for item in result["contributions"]], FEATURES)
            self.assertAlmostEqual(result["base_log_odds"] + sum(item["log_odds"] for item in result["contributions"]), result["raw_log_odds"], places=8)
            self.assertAlmostEqual(float(np.exp(-np.logaddexp(0, -result["raw_log_odds"]))), prediction["fraud_probability"], places=10)
            self.assertAlmostEqual(result["fraud_probability"], prediction["fraud_probability"], places=12)

    def test_curated_sample_matches_its_provenance_and_schema(self):
        path = ROOT / "examples/explorer-sample.json"
        sample = json.loads(path.read_text())
        report = json.loads((ROOT / "reports/explorer-sample.json").read_text())
        self.assertEqual(sample["source_split"], "selection")
        self.assertTrue(sample["curated"])
        self.assertEqual(len(sample["rows"]), 200)
        self.assertEqual(sum(row["label"] for row in sample["rows"]), 10)
        self.assertEqual(len({row["source_row_id"] for row in sample["rows"]}), 200)
        self.assertEqual([row["source_row_id"] for row in sample["rows"]], report["source_row_ids"])
        self.assertEqual(file_sha256(path), report["sample_sha256"])
        for row in sample["rows"]:
            self.assertEqual(set(row["features"]), set(FEATURES))
            Transaction(**row["features"])


if __name__ == "__main__":
    unittest.main()
