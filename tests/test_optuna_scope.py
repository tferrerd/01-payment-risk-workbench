import unittest

import numpy as np
import pandas as pd

from scripts.run_optuna import load_split, training_folds


class OptunaScopeTest(unittest.TestCase):
    def test_grouped_cv_contains_only_training_and_keeps_duplicates_together(self):
        labels = np.repeat([0] * 15 + [1] * 15, 2)
        frame = pd.DataFrame({"Class": labels}, index=np.arange(60))
        assigned = pd.DataFrame({"split": "train", "pattern_group_id": np.repeat(np.arange(30), 2)}, index=frame.index)
        seen = []
        for fit, held in training_folds(frame, assigned):
            self.assertFalse(set(assigned.pattern_group_id.iloc[fit]) & set(assigned.pattern_group_id.iloc[held]))
            self.assertEqual(frame.Class.iloc[held].nunique(), 2)
            seen.extend(held)
        self.assertEqual(sorted(seen), list(range(60)))
        assigned.loc[0, "split"] = "selection"
        with self.assertRaisesRegex(ValueError, "training rows only"):
            training_folds(frame, assigned)
        for forbidden in ["calibration", "evaluation"]:
            with self.assertRaisesRegex(ValueError, "only use training and selection"):
                load_split(forbidden, None, None)


if __name__ == "__main__":
    unittest.main()
