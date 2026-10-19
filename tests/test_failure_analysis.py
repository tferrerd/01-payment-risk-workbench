import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_failures
from run_baselines import sha256


class FailureAnalysisScopeTest(unittest.TestCase):
    def test_rejects_prediction_rows_outside_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "predictions.csv"
            assignments = pd.DataFrame({"split": ["train", "selection", "evaluation"]}, index=[0, 1, 2])
            report = {"selection_predictions": {"path": "predictions.csv", "sha256": ""}}
            with patch.object(analyze_failures, "ROOT", root):
                pd.DataFrame({"source_row_id": [1], "model": ["model"]}).to_csv(path, index=False)
                report["selection_predictions"]["sha256"] = sha256(path)
                self.assertEqual(len(analyze_failures.checked_predictions(report, {"model"}, assignments)), 1)
                pd.DataFrame({"source_row_id": [2], "model": ["model"]}).to_csv(path, index=False)
                report["selection_predictions"]["sha256"] = sha256(path)
                with self.assertRaisesRegex(ValueError, "Selection population mismatch"):
                    analyze_failures.checked_predictions(report, {"model"}, assignments)


if __name__ == "__main__":
    unittest.main()
