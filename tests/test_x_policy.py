"""FCB-14: what X can and cannot buy under the official scorer (synthetic sequences only).

The temporal metrics score ``fill_abstentions(pred)``, which forward-fills X (back-filling only
the head of a graded run). So for spellF1 an X scores the same as the label it carries. That
does NOT mean X cannot improve a given prediction: replacing a wrong label with X can raise
spellF1 (and macroF1, the first tie-breaker), and can also lower them. Cases below cover both.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from evaluation import possession_metrics as pm
from possession.predictors.xpolicy import suppress_short_spells


def ev(gt, pred):
    return pm.evaluate(pm.label_codes(list(gt)), pm.label_codes(list(pred)), bootstrap_n=0)


def _random_seq(rng, n, labels, mean_run):
    out, k = [], rng.choice(labels)
    while len(out) < n:
        out += [k] * int(rng.geometric(1 / mean_run))
        k = rng.choice(labels)
    return np.array(out[:n])


class XPolicyTests(unittest.TestCase):
    def test_x_equals_carried_label_for_spell_f1(self):
        rng = np.random.default_rng(14)
        for _ in range(20):
            n = 3000
            gt = pm.label_codes(list(_random_seq(rng, n, ["H", "A", "D"], 60)))
            pred = pm.label_codes(list(_random_seq(rng, n, ["H", "A", "D"], 40)))
            with_x = pred.copy()
            with_x[rng.random(n) < 0.3] = pm.X
            runs = pm.valid_runs(gt)
            carried = pm.fill_abstentions(with_x, runs)
            a = pm.evaluate(gt, with_x, bootstrap_n=0)
            b = pm.evaluate(gt, carried, bootstrap_n=0)
            self.assertEqual(a["spell_f1_mean"], b["spell_f1_mean"])

    def test_x_after_dead_is_scored_as_dead(self):
        # A live run that opens with X inherits D, i.e. X cannot express "owner unknown yet".
        gt = pm.label_codes(list("D" * 50 + "H" * 100))
        x_head = pm.label_codes(list("D" * 50 + "X" * 30 + "H" * 70))
        late = pm.label_codes(list("D" * 80 + "H" * 70))
        runs = pm.valid_runs(gt)
        self.assertTrue((pm.fill_abstentions(x_head, runs) == late).all())
        self.assertEqual(pm.evaluate(gt, x_head, bootstrap_n=0)["spell_f1_mean"],
                         pm.evaluate(gt, late, bootstrap_n=0)["spell_f1_mean"])

    def test_x_can_improve_original_prediction(self):
        # Reviewer probe (FCB-14 review): X over a spurious flip merges the spell.
        self.assertAlmostEqual(ev("HHHHHH", "HHAAHH")["spell_f1_mean"], 0.1)
        self.assertAlmostEqual(ev("HHHHHH", "HHXXHH")["spell_f1_mean"], 1.0)

    def test_x_can_improve_macro_f1_with_equal_spell_f1(self):
        a, b = ev("HHHAAAD", "HHHHHHD"), ev("HHHAAAD", "HHHXXXD")
        self.assertAlmostEqual(a["spell_f1_mean"], b["spell_f1_mean"])
        self.assertGreater(b["macro_f1"], a["macro_f1"])

    def test_x_can_harm_original_prediction(self):
        # X over a real short possession erases it.
        gt = "H" * 20 + "A" * 10 + "H" * 20
        self.assertGreater(ev(gt, gt)["spell_f1_mean"], ev(gt, "H" * 20 + "X" * 10 + "H" * 20)["spell_f1_mean"])
        self.assertGreater(ev(gt, gt)["macro_f1"], ev(gt, "H" * 20 + "X" * 10 + "H" * 20)["macro_f1"])


class SuppressShortSpellTests(unittest.TestCase):
    def run_(self, s, n, mode="x"):
        return "".join(suppress_short_spells(np.array(list(s)), n, mode))

    def test_short_interior_spell_becomes_x_or_carry(self):
        self.assertEqual(self.run_("HHHHAAHHHH", 3), "HHHHXXHHHH")
        self.assertEqual(self.run_("HHHHAAHHHH", 3, "carry"), "HHHHHHHHHH")

    def test_first_spell_and_long_spells_kept(self):
        self.assertEqual(self.run_("AAHHHH", 3), "AAHHHH")
        self.assertEqual(self.run_("HHHAAAHHH", 3), "HHHAAAHHH")

    def test_short_spell_agreeing_with_carried_owner_is_kept(self):
        # H, short A (suppressed), short H (agrees with carried H: keep it).
        self.assertEqual(self.run_("HHHHAHHHHH"[:6] + "AAAA", 2), "HHHHXHAAAA")

    def test_x_and_carry_differ_across_reference_x_gap(self):
        # Reviewer probe: scorer fill restarts at reference-X gaps, predictor carry does not.
        original = "HHHAAHHH"
        x, carry = self.run_(original, 3, "x"), self.run_(original, 3, "carry")
        self.assertEqual((x, carry), ("HHHXXHHH", "HHHHHHHH"))
        truth = "XXXHHXXX"
        self.assertEqual(ev(truth, x)["spell_f1_mean"], 0.0)
        self.assertEqual(ev(truth, carry)["spell_f1_mean"], 1.0)

    def test_x_and_carry_equal_spell_f1_without_reference_x(self):
        rng = np.random.default_rng(5)
        gt = "".join(rng.choice(list("HA"), 1) [0] * int(rng.integers(5, 60)) for _ in range(80))
        pred = "".join(rng.choice(list("HA"), 1)[0] * int(rng.integers(1, 40)) for _ in range(150))[: len(gt)]
        pred = pred.ljust(len(gt), "H")
        x, c = self.run_(pred, 25, "x"), self.run_(pred, 25, "carry")
        self.assertEqual(ev(gt, x)["spell_f1_mean"], ev(gt, c)["spell_f1_mean"])

    def test_rejects_bad_mode(self):
        with self.assertRaises(ValueError):
            suppress_short_spells(np.array(list("HA")), 2, "drop")


if __name__ == "__main__":
    unittest.main()
