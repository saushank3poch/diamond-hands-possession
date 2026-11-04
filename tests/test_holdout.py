"""FCB-18: holdout gating and run-once guard (synthetic only; no holdout data is read)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from possession import holdout_gate
from possession.evaluation import harness
from test_harness_eval import Spy, toy_refs, toy_tracking

SPEC = {"name": "short_spell_x", "mode": "carry"}


def fake_git(head="abc", tag="abc", status=""):
    def git(*args):
        if args[:1] == ("status",):
            return status
        return head if args[1] == "HEAD" else tag
    return git


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.manifest = t / "freeze.json"
        self.manifest.write_text(json.dumps({"tag": "freeze-x", "predictor": SPEC}))
        self.patches = [patch.object(holdout_gate, "ROOT", t),
                        patch.object(holdout_gate, "MARKER", t / "runs" / "holdout" / "OPENED"),
                        patch.object(holdout_gate, "RESULT", t / "findings" / "HOLDOUT_metrics.json")]
        for p in self.patches:
            p.start()
        self.cfg = {"freeze": "freeze.json"}

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_passes_at_clean_tag(self):
        spec, freeze = holdout_gate.check_gate(self.cfg, git=fake_git())
        self.assertEqual((spec, freeze["commit"], freeze["tag"]), (SPEC, "abc", "freeze-x"))

    def test_refuses_off_tag_dirty_or_mismatched(self):
        for git in (fake_git(head="abc", tag="def"), fake_git(status=" M x.py")):
            with self.assertRaises(SystemExit):
                holdout_gate.check_gate(self.cfg, git=git)
        with self.assertRaises(SystemExit):
            holdout_gate.check_gate({**self.cfg, "predictor": {"name": "deadball"}}, git=fake_git())
        with self.assertRaises(SystemExit):
            holdout_gate.check_gate({**self.cfg, "predictors": [SPEC]}, git=fake_git())
        with self.assertRaises(SystemExit):
            holdout_gate.check_gate({}, git=fake_git())

    def test_runs_only_once(self):
        holdout_gate.check_gate(self.cfg, git=fake_git())
        holdout_gate.record_opened(Path("run"))
        with self.assertRaises(SystemExit):
            holdout_gate.check_gate(self.cfg, git=fake_git())

    def test_committed_result_also_blocks(self):
        holdout_gate.RESULT.parent.mkdir(parents=True)
        holdout_gate.RESULT.write_text("{}")
        with self.assertRaises(SystemExit):
            holdout_gate.check_gate(self.cfg, git=fake_git())


class LoadGateTests(unittest.TestCase):
    def test_dev_path_refuses_holdout_and_holdout_path_refuses_dev(self):
        split = {"dev": ["A"], "holdout": ["Z"]}
        with self.assertRaises(ValueError):
            harness.load_match("Z", split)
        with self.assertRaises(ValueError):
            harness.load_match("A", split, holdout=True)


class HoldoutIsolationTests(unittest.TestCase):
    def test_fit_on_all_dev_predict_holdout_only(self):
        dev, hold = ["A", "B", "C"], ["Y", "Z"]
        data = {m: (toy_tracking(40), None) for m in dev + hold}
        data = {m: (t, toy_refs(t, "H")) for m, (t, _) in data.items()}
        split = {"dev": dev, "holdout": hold, "split_sha256": "s"}
        loads = []

        def load(m, s, holdout=False):
            loads.append((m, holdout))
            return data[m]
        Spy.log = []
        with patch.object(harness.split_mod, "load", return_value=split), \
             patch.object(harness, "load_match", side_effect=load), \
             patch.object(harness, "_files", return_value=(Path("a"), Path("b"))), \
             patch.object(harness, "cache_identity", return_value={}), \
             patch.dict(harness.make.__globals__["REGISTRY"], {"spy": Spy}):
            rep = harness.evaluate_holdout({"name": "spy"}, bootstrap_n=0, variants=("clean",),
                                           label_sets=("primary",))
        self.assertEqual(sorted(loads), [("A", False), ("B", False), ("C", False), ("Y", True), ("Z", True)])
        fits = [e for e in Spy.log if e[0] == "fit"]
        preds = [e for e in Spy.log if e[0] == "predict"]
        self.assertEqual(len(fits), 1)
        self.assertEqual(fits[0][1], sorted(id(data[m][0]) for m in dev))
        self.assertEqual(sorted(p[1] for p in preds), sorted(id(data[m][0]) for m in hold))
        self.assertEqual(set(rep["results"]["clean/primary"]["per_match"]), set(hold))

    def test_refuses_overlapping_or_empty_holdout(self):
        for split in ({"dev": ["A"], "holdout": [], "split_sha256": "s"},
                      {"dev": ["A"], "holdout": ["A"], "split_sha256": "s"}):
            with patch.object(harness.split_mod, "load", return_value=split):
                with self.assertRaises(ValueError):
                    harness.evaluate_holdout({"name": "spy"}, bootstrap_n=0)


if __name__ == "__main__":
    unittest.main()
