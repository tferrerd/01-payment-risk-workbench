"""Hand-calculated checks for approve/cancel profit and opportunity loss."""
import json
import unittest
from pathlib import Path
import numpy as np
from scripts.run_baselines import action_costs, action_profits, cancel_from_probability

class BaselinePolicyTest(unittest.TestCase):
    def test_profit_cost_and_probability_boundary(self):
        costs = json.loads((Path(__file__).resolve().parents[1] / "config/cost-scenario.json").read_text())
        labels, amounts, canceled = [0, 1, 0, 1], [100] * 4, [False, False, True, True]
        np.testing.assert_allclose(action_profits(labels, amounts, canceled, costs), [25, -100, 0, 0])
        np.testing.assert_allclose(action_costs(labels, amounts, canceled, costs), [0, 100, 25, 0])
        np.testing.assert_array_equal(cancel_from_probability([0, 0.19, 0.20, 0.21, 1, 1], [100, 100, 100, 100, 100, 0], costs), [False, False, False, True, True, False])
        with self.assertRaises(ValueError):
            cancel_from_probability([1.2], [100], costs)
        with self.assertRaises(ValueError):
            action_profits([0], [-1], [False], costs)

if __name__ == "__main__":
    unittest.main()
