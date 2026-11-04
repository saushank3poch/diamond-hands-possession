"""FCB-14: what X can and cannot buy under the official scorer (synthetic sequences only).

The temporal metrics score ``fill_abstentions(pred)``, which forward-fills X (back-filling only
the head of a graded run). So for spellF1 an X is identical to writing the carried label, and
no X policy can do better on spellF1 than some explicit H/A/D policy.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm


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


if __name__ == "__main__":
    unittest.main()
