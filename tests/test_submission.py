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


import json as _json
import shutil


def frozen_manifest():
    return _json.loads((ROOT / "configs/freeze.json").read_text())


class RecipeTests(unittest.TestCase):
    def test_reads_frozen_recipe(self):
        spec, variant, seed, dev = submission.recipe()
        self.assertEqual((spec, variant, seed), ({"name": "short_spell_x", "mode": "carry"}, "degraded", 0))
        self.assertEqual(dev, list(submission.split_mod.load()["dev"]))

    def test_rejects_non_frozen_recipe(self):
        base = frozen_manifest()
        bad = []
        for key, value in (("decided_before_holdout", False), ("labels", "proxy_2s"), ("seed", 1),
                           ("input_variant", "degraded_long_gaps (alt)"), ("input_variant", "clean")):
            m = _json.loads(_json.dumps(base)); m["target_training"][key] = value; bad.append(m)
        m = _json.loads(_json.dumps(base)); m["predictor"] = {"name": "nearest_player"}; bad.append(m)
        m = _json.loads(_json.dumps(base)); m["split_sha256"] = "0" * 64; bad.append(m)
        for m in bad:
            with self.assertRaises(ValueError):
                submission.recipe(m)


class FreezeBindingTests(unittest.TestCase):
    """verify_freeze against a temporary copy of the repo; the git side is the real freeze tag."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rels = list(submission.FROZEN_FILES) + list(submission.FROZEN_MODULES)
        for d in submission.FROZEN_DIRS:
            rels += [str(p.relative_to(ROOT)) for p in (ROOT / d).rglob("*.py")]
        for rel in rels:
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, self.root / rel)

    def tearDown(self):
        self.tmp.cleanup()

    def verify(self):
        return submission.verify_freeze(root=self.root)

    def test_current_tree_matches_freeze(self):
        ident = self.verify()
        self.assertEqual(ident["tag"], "freeze-e005")

    def test_refuses_manifest_selecting_another_model(self):
        p = self.root / "configs/freeze.json"
        m = _json.loads(p.read_text()); m["predictor"] = {"name": "nearest_player"}
        p.write_text(_json.dumps(m))
        with self.assertRaises(ValueError):
            self.verify()

    def test_refuses_model_code_lock_or_split_drift(self):
        for rel in ("src/possession/predictors/xpolicy.py", "requirements.lock", "configs/split.yaml",
                    "src/possession/degrade.py"):
            with self.subTest(rel=rel):
                p = self.root / rel
                orig = p.read_bytes()
                p.write_bytes(orig + b"\n# drift\n")
                with self.assertRaises(ValueError):
                    self.verify()
                p.write_bytes(orig)

    def test_refuses_extra_predictor_file(self):
        (self.root / "src/possession/predictors/sneaky.py").write_text("x = 1\n")
        with self.assertRaises(ValueError):
            self.verify()

    def test_build_drift_or_bad_recipe_stops_before_any_loader(self):
        calls = []
        loader = lambda *a, **k: calls.append(a) or 1 / 0
        for patcher in (patch.object(submission, "verify_freeze", side_effect=ValueError("drift")),
                        patch.object(submission, "recipe", side_effect=ValueError("recipe"))):
            with patcher, self.assertRaises(ValueError):
                submission.build(self.root, 0, loader, loader, loader, loader, loader)
        with self.assertRaises(ValueError):  # seed mismatch also stops first
            submission.build(self.root, 1, loader, loader, loader, loader, loader)
        self.assertEqual(calls, [])


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
                {"name": "spy"}, "degraded", 0, dev,
                load_match=lambda m, s: (seen.append(m), data[m])[1],
                variant_input=lambda t, m, v, s: (seen.append((m, v, s)), t)[1],
                gt_labels=lambda r, t, ls: np.where(t.period > 0, "H", "X"))
        self.assertEqual(used, dev)
        self.assertNotIn("Z", seen)
        self.assertEqual([x for x in seen if isinstance(x, tuple)], [("A", "degraded", 0), ("B", "degraded", 0)])
        self.assertEqual(len([e for e in Spy.log if e[0] == "fit"]), 1)

    def test_refuses_dev_list_other_than_frozen(self):
        with patch.object(submission.split_mod, "load", return_value={"dev": ["A", "B"], "holdout": ["Z"]}):
            with self.assertRaises(ValueError):
                submission.train({"name": "spy"}, "degraded", 0, ["A", "Z"], lambda *a: 1 / 0, None, None)


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
