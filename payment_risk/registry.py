"""Register a verified inference artifact as a development candidate in MLflow."""
from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from mlflow.models import infer_signature
import numpy as np

from .pipeline import PaymentRiskPipeline, file_sha256

REGISTERED_MODEL = "payment-risk-low-margin"


class PaymentRiskModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.pipeline = PaymentRiskPipeline.load(context.artifacts["candidate"], context.artifacts["manifest"])

    def predict(self, context, model_input, params=None):
        return self.pipeline.predict(model_input)


def register_candidate(artifact_path, example, validation):
    artifact_path = Path(artifact_path)
    pipeline = PaymentRiskPipeline.load(artifact_path)
    digest = file_sha256(artifact_path)
    client = MlflowClient()
    matches = [item for item in client.search_model_versions(f"name = '{REGISTERED_MODEL}'")
               if item.tags.get("artifact_sha256") == digest]
    if matches:
        registered_version = max(matches, key=lambda item: int(item.version)).version
        model_uri = f"models:/{REGISTERED_MODEL}/{registered_version}"
        run_id = client.get_model_version(REGISTERED_MODEL, registered_version).run_id
    else:
        expected = pipeline.predict(example)
        with mlflow.start_run(run_name=pipeline.metadata["model_id"]) as run:
            mlflow.log_params({"model_id": pipeline.metadata["model_id"], "model_status": "development_candidate",
                               "scenario": pipeline.metadata["economics"]["version"], "threshold": pipeline.threshold,
                               "probability_calibrated": False, "refitted": False})
            mlflow.log_metrics({"selection_rows_verified": validation["rows"],
                                "maximum_probability_difference": validation["maximum_probability_difference"]})
            mlflow.log_dict(validation, "packaging-validation.json")
            info = mlflow.pyfunc.log_model(
                name="payment_pipeline", python_model=PaymentRiskModel(),
                artifacts={"candidate": str(artifact_path), "manifest": str(artifact_path.with_suffix('.json'))},
                code_paths=[str(Path(__file__).resolve().parent)],
                signature=infer_signature(example, expected), input_example=example,
                pip_requirements=[f"{name}=={version(name)}" for name in ["mlflow", "numpy", "pandas", "scikit-learn", "lightgbm", "joblib"]],
                registered_model_name=REGISTERED_MODEL,
                metadata={"status": "development_candidate", "independent_low_margin_test": False},
            )
            registered_version = str(info.registered_model_version)
            model_uri = f"models:/{REGISTERED_MODEL}/{registered_version}"
            run_id = run.info.run_id
        client.update_registered_model(REGISTERED_MODEL, description="Low-margin payment-risk development candidates. No production champion is established; fresh validation under this scenario remains pending.")
        client.update_model_version(REGISTERED_MODEL, registered_version,
                                    description="Existing LightGBM packaged with 0.3% legitimate margin / 100% fraud loss. Validated for inference parity on selection; uncalibrated scores and no independent final test under these economics.")
        for key, value in {"artifact_sha256": digest, "status": "development_candidate", "scenario": pipeline.metadata["economics"]["version"],
                           "independent_low_margin_test": "false", "probability_calibrated": "false"}.items():
            client.set_model_version_tag(REGISTERED_MODEL, registered_version, key, value)
    restored = mlflow.pyfunc.load_model(model_uri)
    actual, expected = restored.predict(example), pipeline.predict(example)
    if not np.allclose(actual.fraud_probability, expected.fraud_probability, rtol=0, atol=1e-12) or not actual.decision.equals(expected.decision):
        raise ValueError("Registered model differs from the verified local artifact")
    client.set_registered_model_alias(REGISTERED_MODEL, "candidate", registered_version)
    return {"name": REGISTERED_MODEL, "version": registered_version, "alias": "candidate", "model_uri": model_uri,
            "run_id": run_id, "tracking_uri": mlflow.get_tracking_uri(), "reload_verified": True}
