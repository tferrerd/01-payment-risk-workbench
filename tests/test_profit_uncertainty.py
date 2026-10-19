import sys
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_final_evaluation import paired_group_bootstrap
import run_final_evaluation


class PairedProfitBootstrapTest(unittest.TestCase):
    def test_pairing_and_sampled_payment_denominator(self):
        # Unequal group sizes must still yield a constant per-payment difference.
        profits = pd.DataFrame({"candidate": [2.0] * 5, "same": [2.0] * 5, "baseline": [1.0] * 5})
        result, draws = paired_group_bootstrap(profits, [1, 1, 1, 2, 3], "candidate", repeats=100)
        np.testing.assert_allclose(draws.candidate - draws.baseline, 1000)
        np.testing.assert_allclose(draws.candidate - draws.same, 0)
        differences = {row["comparator"]: row for row in result["paired_differences"]}
        self.assertEqual(differences["same"]["conclusion"], "inconclusive")
        self.assertEqual(differences["baseline"]["lower_95"], 1000)
        # A group contributes the sum of its rows; it is not one equally weighted row.
        result, _ = paired_group_bootstrap(pd.DataFrame({"candidate": [10, 0, 0], "baseline": [0, 0, 0]}),
                                          [1, 1, 2], "candidate", repeats=200, seed=7)
        self.assertAlmostEqual(result["paired_differences"][0]["profit_difference_per_1000"], 10000 / 3)
        self.assertEqual(result["paired_differences"][0]["lower_95"], 0)
        self.assertEqual(result["paired_differences"][0]["upper_95"], 5000)

    def test_cached_evaluation_does_not_load_models_or_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config").mkdir()
            (root / "reports").mkdir()
            freeze = root / "config/candidate-freeze.json"
            freeze.write_text("{}")
            predictions = root / "predictions.csv"
            predictions.write_text("cached")
            report = {"metadata": {"freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest()},
                      "predictions": {"path": "predictions.csv", "sha256": hashlib.sha256(predictions.read_bytes()).hexdigest()}}
            (root / "reports/final-evaluation.json").write_text(json.dumps(report))
            with patch.object(run_final_evaluation, "ROOT", root), patch.object(run_final_evaluation.joblib, "load", side_effect=AssertionError("must not load model")), patch.object(run_final_evaluation.pd, "read_csv", side_effect=AssertionError("must not read rows")):
                self.assertEqual(run_final_evaluation.evaluate({}, None, None, None, None), report)
                freeze.write_text('{"changed": true}')
                with self.assertRaisesRegex(ValueError, "different freeze"):
                    run_final_evaluation.evaluate({}, None, None, None, None)


if __name__ == "__main__":
    unittest.main()
