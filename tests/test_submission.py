"""FCB-19: submission assembly on synthetic data only (no target, dev or holdout files read)."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from possession import submission
from test_harness_eval import Spy, toy_refs, toy_tracking


class RecipeTests(unittest.TestCase):
    def test_reads_frozen_recipe(self):
        spec, variant, seed = submission.recipe()
        self.assertEqual((spec, variant, seed), ({"name": "short_spell_x", "mode": "carry"}, "degraded", 0))

    def test_rejects_non_frozen_recipe(self):
        for t in ({"decided_before_holdout": False, "labels": "primary", "seed": 0, "input_variant": "degraded"},
                  {"decided_before_holdout": True, "labels": "proxy_2s", "seed": 0, "input_variant": "degraded"},
                  {"decided_before_holdout": True, "labels": "primary", "seed": 0, "input_variant": "clean"}):
            with self.assertRaises(ValueError):
                submission.recipe({"predictor": {}, "target_training": t})


class TrainTests(unittest.TestCase):
    def test_fits_on_dev_only_with_variant_inputs(self):
        dev, hold = ["A", "B"], ["Z"]
        data = {m: (toy_tracking(40), None) for m in dev}
        data = {m: (t, toy_refs(t, "H")) for m, (t, _) in data.items()}
        seen = []
        Spy.log = []
        with patch.object(submission.split_mod, "load", return_value={"dev": dev, "holdout": hold}), \
             patch.dict(submission.make.__globals__["REGISTRY"], {"spy": Spy}):
            model, used = submission.train(
                {"name": "spy"}, "degraded", 0,
                load_match=lambda m, s: (seen.append(m), data[m])[1],
                variant_input=lambda t, m, v, s: (seen.append((m, v, s)), t)[1],
                gt_labels=lambda r, t, ls: np.where(t.period > 0, "H", "X"))
        self.assertEqual(used, dev)
        self.assertNotIn("Z", seen)
        self.assertEqual([x for x in seen if isinstance(x, tuple)], [("A", "degraded", 0), ("B", "degraded", 0)])
        self.assertEqual(len([e for e in Spy.log if e[0] == "fit"]), 1)


class OutputTests(unittest.TestCase):
    def tracking(self):
        period = np.zeros(submission.N_FRAMES, int)
        period[11185:90652], period[110808:190198] = 1, 2
        return type("T", (), {"period": period})()

    def test_x_outside_halves_and_official_validator_passes(self):
        tr = self.tracking()
        model = type("M", (), {"predict": lambda self, t: np.where(t.period > 0, "H", "X")})()
        labels = submission.predict_target(model, tr)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.csv"
            h1 = submission.write_csv(path, labels)
            self.assertEqual(h1, submission.write_csv(path, labels))  # byte-identical rewrite
            rows = list(csv.reader(path.open()))
            self.assertEqual(rows[0], ["frame", "label"])
            self.assertEqual(len(rows), submission.N_FRAMES + 1)
            r = subprocess.run([sys.executable, str(ROOT / "evaluation/validate_submission.py"), str(path),
                                "--rows", str(submission.N_FRAMES)], capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_rejects_wrong_length_bad_labels_or_labels_outside_halves(self):
        tr = self.tracking()
        for out in (np.array(["H"] * 10), np.where(tr.period > 0, "Q", "X"), np.full(submission.N_FRAMES, "H")):
            model = type("M", (), {"predict": lambda self, t, o=out: o})()
            with self.assertRaises(ValueError):
                submission.predict_target(model, tr)


if __name__ == "__main__":
    unittest.main()
