import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scripts.run_baselines import ROOT, setup_tracking


class TrackingConfigurationTest(unittest.TestCase):
    def test_local_store_and_server_override(self):
        with patch.dict(os.environ, {}, clear=True), patch("scripts.run_baselines.mlflow") as mlflow:
            self.assertEqual(setup_tracking("baseline"), "sqlite:///" + str(ROOT / "mlflow.db"))
            mlflow.set_experiment.assert_called_once_with("baseline")
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://127.0.0.1:5000"}), patch("scripts.run_baselines.mlflow") as mlflow:
            self.assertEqual(setup_tracking("experiment"), "http://127.0.0.1:5000")
            mlflow.set_tracking_uri.assert_called_once_with("http://127.0.0.1:5000")


if __name__ == "__main__":
    unittest.main()
