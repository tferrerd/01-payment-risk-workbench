"""Random group allocation preserves exact patterns and label strata."""
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_splits import assign_group_splits

class SplitAssignmentTest(unittest.TestCase):
    def test_group_isolation_strata_and_reproducibility(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "config/evaluation.json").read_text())
        groups = np.repeat(np.arange(400), 2)
        labels = np.repeat(np.r_[np.zeros(200), np.ones(200)], 2)
        actual = assign_group_splits(labels, groups, config)
        np.testing.assert_array_equal(actual, assign_group_splits(labels, groups, config))
        rows = pd.DataFrame(dict(group=groups, label=labels, split=actual))
        self.assertEqual(rows.groupby("group")["split"].nunique().max(), 1)
        self.assertTrue(rows.groupby("split")["label"].nunique().eq(2).all())
        self.assertFalse(np.array_equal(actual, assign_group_splits(labels, groups, {**config, "random_seed": 43})))
        labels[1] = 1
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            assign_group_splits(labels, groups, config)

if __name__ == "__main__":
    unittest.main()
