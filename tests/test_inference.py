import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from pydantic import ValidationError

from payment_risk import PaymentRiskPipeline
from payment_risk.__main__ import predict_csv
from payment_risk.api import PredictionRequest
from payment_risk.pipeline import FEATURES, file_sha256
from scripts.check_release import check_files


class ScoreModel:
    classes_ = np.array([0, 1])
    feature_name_ = FEATURES

    def predict_proba(self, data):
        if list(data.columns) != FEATURES:
            raise AssertionError("Feature order changed")
        p = data.V1.to_numpy()
        return np.column_stack([1-p, p])


class InferenceContractTest(unittest.TestCase):
    def setUp(self):
        self.costs = json.loads((Path(__file__).resolve().parents[1] / "config/cost-scenario-low-margin.json").read_text())
        self.threshold = self.costs["legitimate_margin_fraction"] / (self.costs["legitimate_margin_fraction"] + self.costs["fraud_loss_fraction"])
        self.metadata = {"features": FEATURES, "status": "development_candidate", "model_id": "test",
                         "economics": self.costs, "threshold": self.threshold,
                         "calibration_method": "identity", "probability_calibrated": False}
        self.pipeline = PaymentRiskPipeline(ScoreModel(), self.metadata)

    def inputs(self, probabilities):
        frame = pd.DataFrame(0.0, index=range(len(probabilities)), columns=FEATURES)
        frame["V1"], frame["Amount"] = probabilities, 100.0
        return frame

    def artifact(self, directory):
        path = Path(directory) / "candidate.joblib"
        joblib.dump({"artifact_format_version": 1, "model": ScoreModel(), "metadata": self.metadata}, path)
        path.with_suffix(".json").write_text(json.dumps({"model_id": "test", "artifact_sha256": file_sha256(path)}))
        return path

    def test_ties_zero_amount_and_reordered_inputs(self):
        frame = self.inputs([0, self.threshold, self.threshold+1e-8, 1])
        frame.loc[3, "Amount"] = 0
        frame["Class"] = [1, 0, 0, 1]
        result = self.pipeline.predict(frame[frame.columns[::-1]])
        self.assertEqual(result.decision.tolist(), ["approve", "approve", "cancel", "approve"])
        self.assertAlmostEqual(result.modeled_approval_profit.iloc[0], 0.3)
        np.testing.assert_allclose(result.modeled_policy_profit.iloc[2:], [0, 0])

    def test_rejects_bad_input_and_invalid_model_scores(self):
        valid = self.inputs([0.001])
        for column, value in [("Amount", -1), ("Time", -1), ("V2", np.nan), ("V2", np.inf), ("V2", True), ("V2", "1")]:
            bad = valid.copy()
            bad[column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                self.pipeline.predict(bad)
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.pipeline.predict(valid.drop(columns="V3"))
        with self.assertRaisesRegex(ValueError, "unique"):
            self.pipeline.predict(pd.concat([valid, valid[["V1"]]], axis=1))
        with self.assertRaisesRegex(ValueError, "invalid fraud probabilities"):
            self.pipeline.predict(self.inputs([1.1]))

    def test_artifact_integrity_and_csv_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.artifact(directory)
            frame = self.inputs([0.001, 0.01])
            frame["Class"] = [0, 1]
            source, output = Path(directory) / "input.csv", Path(directory) / "output.csv"
            frame.to_csv(source, index=False)
            self.assertEqual(predict_csv(source, output, path), 2)
            result = pd.read_csv(output)
            self.assertEqual(result.Class.tolist(), [0, 1])
            self.assertEqual(result.decision.tolist(), ["approve", "cancel"])
            with self.assertRaisesRegex(ValueError, "new file"):
                predict_csv(source, output, path)
            source.write_text("V1,V1\n1,1\n")
            with self.assertRaisesRegex(ValueError, "unique"):
                predict_csv(source, Path(directory) / "new.csv", path)
            with path.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                PaymentRiskPipeline.load(path)

    def test_api_payload_validation(self):
        row = self.inputs([0.001]).iloc[0].to_dict()
        self.assertEqual(len(PredictionRequest.model_validate({"transactions": [row]}).transactions), 1)
        for bad in [{**row, "Amount": -1}, {**row, "V1": True}, {**row, "V1": "0.01"},
                    {**row, "V1": float("inf")}, {**row, "Class": 1}]:
            with self.assertRaises(ValidationError):
                PredictionRequest.model_validate({"transactions": [bad]})
        with self.assertRaises(ValidationError):
            PredictionRequest.model_validate({"transactions": []})

    def test_release_snapshot_predictions_and_checksum_gate(self):
        root = Path(__file__).resolve().parents[1]
        release = check_files(root)
        pipeline = PaymentRiskPipeline.load()
        self.assertEqual(release["artifact_sha256"], file_sha256(root / "models/candidates/lightgbm-low-margin-dev-v1.joblib"))
        payload = json.loads((root / "examples/predict-request.json").read_text())
        expected = pd.DataFrame(json.loads((root / "examples/predict-expected.json").read_text()))
        pd.testing.assert_frame_equal(pipeline.predict(pd.DataFrame(payload["transactions"])), expected,
                                      check_exact=False, rtol=0, atol=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            (temporary / "deployment").mkdir()
            (temporary / "input.bin").write_bytes(b"original")
            (temporary / "deployment/release.json").write_text(json.dumps({
                "files": {"input.bin": file_sha256(temporary / "input.bin")}}))
            check_files(temporary)
            (temporary / "input.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                check_files(temporary)


if __name__ == "__main__":
    unittest.main()
