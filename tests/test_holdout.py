"""FCB-18: holdout gating and run-once guard (synthetic only; no holdout data is read)."""
import hashlib
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
LOCK = b"lock-bytes\n"
CFG = json.dumps({"seed": 0, "bootstrap_n": 1000}).encode()


def manifest_bytes(**over):
    m = {"tag": "freeze-x", "predictor": SPEC, "split_sha256": "S",
         "requirements_lock_sha256": hashlib.sha256(LOCK).hexdigest(),
         "holdout_run": {"seed": 0, "bootstrap_n": 1000}}
    m.update(over)
    return json.dumps(m).encode()


def fake_git(head="abc", tag="abc", status=b"", blobs=None):
    def git(*args):
        if args[0] == "status":
            return status
        if args[0] == "show":
            return blobs[args[1].split(":", 1)[1]]
        return (head if args[1] == "HEAD" else tag).encode()
    return git


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        (t / "configs").mkdir()
        self.man = manifest_bytes()
        (t / holdout_gate.MANIFEST).write_bytes(self.man)
        (t / holdout_gate.CONFIG).write_bytes(CFG)
        (t / "requirements.lock").write_bytes(LOCK)
        self.blobs = {holdout_gate.MANIFEST: self.man, holdout_gate.CONFIG: CFG}
        self.patches = [patch.object(holdout_gate, "ROOT", t),
                        patch.object(holdout_gate, "MARKER", t / "runs" / "holdout" / "RESERVATION.json"),
                        patch.object(holdout_gate, "RESULT", t / "findings" / "HOLDOUT_metrics.json")]
        for p in self.patches:
            p.start()
        self.root = t

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def gate(self, cfg=CFG, seed=0, split="S", **git):
        return holdout_gate.check_gate(cfg, seed, split, git=fake_git(blobs=self.blobs, **git))

    def assertRefused(self, **kw):
        with self.assertRaises(SystemExit):
            self.gate(**kw)

    def test_passes_when_everything_matches(self):
        spec, freeze = self.gate()
        self.assertEqual((spec, freeze["commit"], freeze["tag"]), (SPEC, "abc", "freeze-x"))

    def test_refuses_off_tag_or_dirty(self):
        self.assertRefused(tag="def")
        self.assertRefused(status=b" M x.py")

    def test_refuses_manifest_not_matching_tag_blob(self):
        # Reviewer probe: a manifest naming the expected tag but selecting another predictor.
        (self.root / holdout_gate.MANIFEST).write_bytes(manifest_bytes(predictor={"name": "nearest_player"}))
        self.assertRefused()

    def test_refuses_altered_or_extended_run_config(self):
        self.assertRefused(cfg=json.dumps({"seed": 1, "bootstrap_n": 1000}).encode())
        bad = json.dumps({"seed": 0, "bootstrap_n": 1000, "freeze": "/tmp/x.json"}).encode()
        self.blobs[holdout_gate.CONFIG] = bad
        (self.root / holdout_gate.CONFIG).write_bytes(bad)
        self.assertRefused(cfg=bad)

    def test_refuses_env_seed_lock_or_split_mismatch(self):
        self.assertRefused(seed=7)
        self.assertRefused(split="other")
        (self.root / "requirements.lock").write_bytes(b"changed\n")
        self.assertRefused()

    def test_refuses_wrong_lock_hash_in_manifest(self):
        m = manifest_bytes(requirements_lock_sha256="0" * 64)
        self.blobs[holdout_gate.MANIFEST] = m
        (self.root / holdout_gate.MANIFEST).write_bytes(m)
        self.assertRefused()

    def test_gate_failure_never_reserves_or_evaluates(self):
        calls = []
        with self.assertRaises(SystemExit):
            holdout_gate.run_once(CFG, 7, "S", Path("r"), lambda s, n: calls.append(1), lambda r: None,
                                  git=fake_git(blobs=self.blobs))
        self.assertEqual(calls, [])
        self.assertFalse(holdout_gate.MARKER.exists())

    def test_success_records_completed_and_blocks_rerun(self):
        rep = holdout_gate.run_once(CFG, 0, "S", Path("r"), lambda s, n: {"spec": s}, lambda r: None,
                                    git=fake_git(blobs=self.blobs))
        self.assertEqual(rep["spec"], SPEC)
        self.assertEqual(json.loads(holdout_gate.MARKER.read_text())["status"], "completed")
        self.assertRefused()

    def test_failure_after_read_records_failed_and_blocks_retry(self):
        def boom(spec, n):
            raise RuntimeError("crash after reading holdout")
        with self.assertRaises(RuntimeError):
            holdout_gate.run_once(CFG, 0, "S", Path("r"), boom, lambda r: None, git=fake_git(blobs=self.blobs))
        rec = json.loads(holdout_gate.MARKER.read_text())
        self.assertEqual((rec["status"], rec["error_type"]), ("failed", "RuntimeError"))
        self.assertNotIn("crash", holdout_gate.MARKER.read_text())
        self.assertRefused()

    def test_competing_reservations_only_one_wins(self):
        self.gate()
        self.gate()  # two preflights may both pass ...
        holdout_gate.reserve(Path("a"))
        with self.assertRaises(SystemExit):  # ... but only one reservation succeeds
            holdout_gate.reserve(Path("b"))
        self.assertEqual(json.loads(holdout_gate.MARKER.read_text())["run_dir"], "a")

    def test_committed_result_also_blocks(self):
        holdout_gate.RESULT.parent.mkdir(parents=True)
        holdout_gate.RESULT.write_text("{}")
        self.assertRefused()


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
