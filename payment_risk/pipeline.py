"""Validated model inputs and a versioned approve/cancel policy."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1] / "models/candidates/lightgbm-low-margin-dev-v1.joblib"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PaymentRiskPipeline:
    def __init__(self, model, metadata):
        self.model = model
        self.metadata = metadata
        if metadata.get("features") != FEATURES or metadata.get("status") != "development_candidate":
            raise ValueError("Unsupported artifact feature schema or status")
        if metadata.get("calibration_method") != "identity" or metadata.get("probability_calibrated") is not False:
            raise ValueError("This artifact format expects the raw-score development candidate")
        if list(model.classes_) != [0, 1] or list(model.feature_name_) != FEATURES:
            raise ValueError("Model class labels or feature order differ from the artifact schema")
        costs = metadata["economics"]
        self.margin = float(costs["legitimate_margin_fraction"])
        self.fraud_loss = float(costs["fraud_loss_fraction"])
        if not np.isfinite([self.margin, self.fraud_loss]).all() or self.margin <= 0 or self.fraud_loss <= 0:
            raise ValueError("Expected positive finite margin and fraud-loss fractions")
        if costs.get("cancel_profit") != 0 or costs.get("cancel_direct_cost") != 0:
            raise ValueError("This policy requires zero cancellation payoff")
        self.threshold = self.margin / (self.margin + self.fraud_loss)
        if not np.isclose(metadata["threshold"], self.threshold, rtol=0, atol=1e-15):
            raise ValueError("Artifact threshold differs from its economics")

    @classmethod
    def load(cls, path=DEFAULT_ARTIFACT, manifest_path=None):
        path = Path(path)
        manifest = json.loads(Path(manifest_path or path.with_suffix(".json")).read_text())
        if file_sha256(path) != manifest["artifact_sha256"]:
            raise ValueError("Candidate artifact checksum mismatch")
        bundle = joblib.load(path)
        if bundle.get("artifact_format_version") != 1 or bundle["metadata"]["model_id"] != manifest["model_id"]:
            raise ValueError("Unsupported candidate artifact or mismatched manifest")
        return cls(bundle["model"], bundle["metadata"])

    @staticmethod
    def validate_inputs(frame):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("Expected at least one payment in a DataFrame")
        if not frame.columns.is_unique:
            raise ValueError("Input column names must be unique")
        missing = [column for column in FEATURES if column not in frame]
        if missing:
            raise ValueError("Missing required features: " + ", ".join(missing))
        values = frame.loc[:, FEATURES]
        if any(not is_numeric_dtype(values[column]) or is_bool_dtype(values[column]) for column in FEATURES):
            raise ValueError("Features must be numeric; booleans and text are not accepted")
        try:
            numeric = values.to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Features must be complete finite numbers") from error
        if not np.isfinite(numeric).all():
            raise ValueError("Features must be complete finite numbers")
        if (values[["Time", "Amount"]] < 0).any().any():
            raise ValueError("Time and Amount must be nonnegative")
        return values.astype(float)

    def predict(self, frame):
        values = self.validate_inputs(frame)
        probability = np.asarray(self.model.predict_proba(values)[:, 1])
        if probability.shape != (len(values),) or not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
            raise ValueError("Model returned invalid fraud probabilities")
        amount = values.Amount.to_numpy()
        canceled = (probability > self.threshold) & (amount > 0)
        approval_profit = amount * (self.margin - (self.margin + self.fraud_loss) * probability)
        if not np.isfinite(approval_profit).all():
            raise ValueError("Modeled profit is outside the supported numeric range")
        return pd.DataFrame({"fraud_probability": probability,
                             "decision": np.where(canceled, "cancel", "approve"),
                             "modeled_approval_profit": approval_profit,
                             "modeled_policy_profit": np.where(canceled, 0.0, approval_profit),
                             "model_id": self.metadata["model_id"],
                             "scenario": self.metadata["economics"]["version"],
                             "model_status": self.metadata["status"],
                             "probability_calibrated": False}, index=frame.index)
