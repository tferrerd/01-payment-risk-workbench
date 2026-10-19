import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_baselines import action_profits, cancel_from_probability
from run_low_margin_scenario import summarize


class LowMarginEconomicsTest(unittest.TestCase):
    def test_percent_conversion_payoffs_and_break_even_policy(self):
        costs = json.loads((ROOT / "config/cost-scenario-low-margin.json").read_text())
        np.testing.assert_allclose(action_profits([0, 1, 0, 1], [100] * 4, [False, False, True, True], costs),
                                   [0.30, -100, 0, 0])
        threshold = costs["legitimate_margin_fraction"] / (costs["legitimate_margin_fraction"] + costs["fraud_loss_fraction"])
        self.assertAlmostEqual(threshold, 0.002991026919242273)
        np.testing.assert_array_equal(cancel_from_probability([threshold-1e-8, threshold, threshold+1e-8, 1],
                                                              [100, 100, 100, 0], costs),
                                      [False, False, True, False])
        result = summarize([0, 1], [100, 100], [False, True], costs)
        self.assertAlmostEqual(result["simulated_total_profit"], 0.30)
        self.assertEqual(result["frauds_canceled"], 1)


if __name__ == "__main__":
    unittest.main()
