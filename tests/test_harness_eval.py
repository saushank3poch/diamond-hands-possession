"""FCB-9: baseline parity, period-safe scoring, cache identity, paired-run safety, fold isolation."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from evaluation import possession_metrics as pm
from possession.evaluation import harness
from possession.evaluation.references import References
from possession.predictors.baseline import majority
from possession.tracking import Tracking
from starter import baseline as starter


def score(gt, pred, period):
    g, p = harness.scored_sequence(np.array(list(gt)), np.array(list(pred)), np.array(period))
    return pm.evaluate(pm.label_codes(list(g)), pm.label_codes(list(p)))


class BaselineParityTests(unittest.TestCase):
    def test_majority_matches_starter_smooth(self):
        rng = np.random.default_rng(0)
        letters = rng.choice(["H", "A", ""], size=2000, p=[0.3, 0.3, 0.4])
        codes = np.where(letters == "H", starter.HOME, np.where(letters == "A", starter.AWAY, starter.NOBODY))
        for window in (0, 25, 50, 125):
            self.assertEqual(list(majority(letters, window)), list(starter.smooth(codes, window)))


class PeriodScoringTests(unittest.TestCase):
    def test_review_probe_boundary_preserved(self):
        r = score("HHHH", "HHXX", [1, 1, 2, 2])
        self.assertEqual(r["n_gt_spells"], 2)
        self.assertAlmostEqual(r["spell_f1_mean"], 2 / 3)

    def test_second_period_all_x_is_not_filled_from_first(self):
        r = score("HHAA", "HHXX", [1, 1, 2, 2])
        self.assertEqual(r["n_pred_spells"], 1)  # period 2 stays unpredicted, not filled with H

    def test_same_and_different_labels_at_boundary(self):
        self.assertEqual(score("HHHH", "HHHH", [1, 1, 2, 2])["n_gt_spells"], 2)
        self.assertEqual(score("HHAA", "HHAA", [1, 1, 2, 2])["n_gt_spells"], 2)
        self.assertEqual(score("HHAA", "HHAA", [1, 1, 2, 2])["spell_f1_mean"], 1.0)

    def test_real_frames_kept_and_inactive_dropped(self):
        g, p = harness.scored_sequence(np.array(list("XHHAX")), np.array(list("XHHAX")), np.array([0, 1, 1, 2, 0]))
        self.assertEqual("".join(g), "HHXAX")
        self.assertEqual("".join(p), "HHXAX")


def toy_tracking(n, periods=(1, 2)):
    period = np.repeat(np.array(periods), n // len(periods))
    rng = np.random.default_rng(n)
    players = rng.uniform(-30, 30, (len(period), 4, 2))
    ball = players[:, 0] + 0.5
    return Tracking(np.arange(len(period)), np.arange(len(period)), period, np.array(["a", "b", "c", "d"]),
                    np.array(["H", "H", "A", "A"]), players, ball, np.ones(len(period), int), np.array([105.0, 68.0]))


def toy_refs(tracking, owner):
    n = len(tracking.frame)
    return References(tracking.frame.copy(), np.ones(n, np.int8), np.array([owner] * n),
                      np.zeros(n, bool), (), ())


class CacheTests(unittest.TestCase):
    def test_reuse_and_invalidation(self):
        tr = toy_tracking(40)
        refs = toy_refs(tr, "H")
        split = {"dev": ["M"], "holdout": [], "split_sha256": "s"}
        with tempfile.TemporaryDirectory() as tmp:
            info, raw = Path(tmp, "info.xml"), Path(tmp, "raw.xml")
            info.write_text("i")
            raw.write_text("r")
            ident = {"schema": 2, "match": "M", "sources": [{"x": 1}], "code": {"c": "1"}}
            calls = []

            def fake_load(*a):
                calls.append(a)
                return tr
            with patch.object(harness, "CACHE", Path(tmp) / "cache"), \
                 patch.object(harness, "_files", return_value=(info, raw)), \
                 patch.object(harness, "cache_identity", side_effect=lambda m, f: copy.deepcopy(ident)), \
                 patch.object(harness, "_sha256", return_value=None), \
                 patch.object(harness, "load_dfl", side_effect=fake_load), \
                 patch.object(harness, "load_dfl_references", return_value=refs):
                ident["sources"][0]["manifest_sha256"] = None
                with self.assertRaises(ValueError):  # unverifiable source is refused on a miss
                    harness.load_match("M", split)
                ident["sources"] = [{"manifest_sha256": "ok"}, {"manifest_sha256": "ok"}]
                with patch.object(harness, "_sha256", return_value="ok"):
                    harness.load_match("M", split)
                    harness.load_match("M", split)
                    self.assertEqual(len(calls), 1)  # second load is a cache hit
                    ident["code"] = {"c": "2"}       # parser change -> new key -> reparse
                    harness.load_match("M", split)
                    self.assertEqual(len(calls), 2)

    def test_misaligned_references_rejected(self):
        tr = toy_tracking(40)
        bad = References(np.arange(39), np.ones(39, np.int8), np.array(["H"] * 39), np.zeros(39, bool), (), ())
        split = {"dev": ["M"], "holdout": [], "split_sha256": "s"}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(harness, "CACHE", Path(tmp)), \
                 patch.object(harness, "_files", return_value=(Path(tmp, "a"), Path(tmp, "b"))), \
                 patch.object(harness, "cache_identity", return_value={"sources": [{"manifest_sha256": "ok"}] * 2}), \
                 patch.object(harness, "_sha256", return_value="ok"), \
                 patch.object(harness, "load_dfl", return_value=tr), \
                 patch.object(harness, "load_dfl_references", return_value=bad):
                with self.assertRaises(ValueError):
                    harness.load_match("M", split)

    def test_holdout_refused(self):
        with self.assertRaises(ValueError):
            harness.load_match("M2", {"dev": ["M1"], "holdout": ["M2"], "split_sha256": "x"})


class PairedTests(unittest.TestCase):
    def rep(self, vals, **prov):
        p = {f: f for f in harness.PAIRED_MUST_MATCH}
        p.update(prov)
        return {"provenance": p, "results": {"degraded/primary": {
            "pooled": {"spell_ci90": [0, 1]}, "per_match": {m: {"spell_f1": v} for m, v in vals.items()}}}}

    def test_differences(self):
        out = harness.paired(self.rep({"a": 0.5, "b": 0.4}), self.rep({"a": 0.4, "b": 0.45}))
        self.assertEqual(out["n_positive"], 1)
        self.assertAlmostEqual(out["mean"], 0.025)

    def test_incompatible_runs_refused(self):
        with self.assertRaises(ValueError):
            harness.paired(self.rep({"a": 0.5, "b": 0.4}), self.rep({"a": 0.4}))
        with self.assertRaises(ValueError):
            harness.paired(self.rep({}), self.rep({}))
        for field in ("split_sha256", "seed", "variants", "scorer_sha256", "label_sets", "inputs"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                harness.paired(self.rep({"a": 0.5}, **{field: "other"}), self.rep({"a": 0.4}))
        with self.assertRaises(ValueError):
            harness.paired({"results": {}}, self.rep({"a": 0.4}))


class Spy:
    log = []

    def __init__(self, **_):
        pass

    def fit(self, examples):
        Spy.log.append(("fit", sorted(id(t) for t, _ in examples), [set(np.unique(c)) for _, c in examples]))
        return self

    def predict(self, tracking):
        Spy.log.append(("predict", id(tracking), tuple(sorted(vars(tracking)) if hasattr(tracking, "__dict__") else ())))
        return np.where(tracking.period > 0, "H", "X")


class FoldIsolationTests(unittest.TestCase):
    def test_fit_sees_only_other_matches_and_predict_gets_tracking_only(self):
        matches = ["A", "B", "C"]
        data = {m: (toy_tracking(40), None) for m in matches}
        owners = {"A": "H", "B": "A", "C": "H"}
        data = {m: (t, toy_refs(t, owners[m])) for m, (t, _) in data.items()}
        split = {"dev": matches, "holdout": ["Z"], "split_sha256": "s",
                 "dev_folds": [{"validate": m, "train": [x for x in matches if x != m]} for m in matches]}
        Spy.log = []
        with patch.object(harness.split_mod, "load", return_value=split), \
             patch.object(harness, "load_match", side_effect=lambda m, s: data[m]), \
             patch.object(harness, "_files", return_value=(Path("a"), Path("b"))), \
             patch.object(harness, "cache_identity", return_value={}), \
             patch.dict(harness.make.__globals__["REGISTRY"], {"spy": Spy}):
            harness.evaluate({"name": "spy"}, bootstrap_n=0, variants=("clean",), label_sets=("primary",))
        fits = [e for e in Spy.log if e[0] == "fit"]
        predicts = [e for e in Spy.log if e[0] == "predict"]
        self.assertEqual(len(fits), 3)
        self.assertEqual(len(predicts), 3)
        for fold, fit, pred in zip(split["dev_folds"], fits, predicts):
            self.assertEqual(fit[1], sorted(id(data[m][0]) for m in fold["train"]))
            self.assertNotIn(pred[1], fit[1])
            self.assertEqual(pred[1], id(data[fold["validate"]][0]))


if __name__ == "__main__":
    unittest.main()
