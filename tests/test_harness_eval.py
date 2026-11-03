"""FCB-9: baseline parity with the starter, holdout refusal, paired comparison shape."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from possession.evaluation import harness
from possession.predictors.baseline import majority
from starter import baseline as starter


class BaselineParityTests(unittest.TestCase):
    def test_majority_matches_starter_smooth(self):
        rng = np.random.default_rng(0)
        letters = rng.choice(["H", "A", ""], size=2000, p=[0.3, 0.3, 0.4])
        codes = np.where(letters == "H", starter.HOME, np.where(letters == "A", starter.AWAY, starter.NOBODY))
        for window in (0, 25, 50, 125):
            self.assertEqual(list(majority(letters, window)), list(starter.smooth(codes, window)))


class HarnessTests(unittest.TestCase):
    def test_holdout_refused(self):
        split = {"dev": ["M1"], "holdout": ["M2"], "split_sha256": "x"}
        with self.assertRaises(ValueError):
            harness.load_match("M2", split)

    def test_paired_differences(self):
        def rep(vals):
            return {"results": {"degraded/primary": {"pooled": {"spell_ci90": [0, 1]},
                                                     "per_match": {m: {"spell_f1": v} for m, v in vals.items()}}}}
        out = harness.paired(rep({"a": 0.5, "b": 0.4}), rep({"a": 0.4, "b": 0.45}))
        self.assertEqual(out["n_positive"], 1)
        self.assertAlmostEqual(out["mean"], 0.025)


if __name__ == "__main__":
    unittest.main()
