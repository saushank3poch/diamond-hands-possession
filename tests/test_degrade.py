"""FCB-8: degradation acts on input copies only and hits the published DATA.md targets."""
import ast
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from possession.degrade import Config, degrade
from possession.tracking import Tracking


def synthetic(n=25 * 600, players=22, seed=1):
    """Random-walk players and a ball that drifts between them; two periods; no labels."""
    rng = np.random.default_rng(seed)
    xy = np.cumsum(rng.normal(0, 0.05, (n, players, 2)), axis=0)
    xy = np.clip(xy + rng.uniform([-45, -30], [45, 30], (1, players, 2)), [-52, -34], [52, 34])
    ball = xy[:, 0] + np.cumsum(rng.normal(0, 0.3, (n, 2)), axis=0) * 0.05
    ball[n // 3: n // 3 + 100] = [60.0, 0.0]  # off the pitch for 4 s
    period = np.r_[np.ones(n // 2, int), np.full(n - n // 2, 2)]
    return Tracking(np.arange(n), np.arange(n), period, np.array([f"p{i}" for i in range(players)]),
                    np.array(["H"] * 11 + ["A"] * 11), xy, ball, np.ones(n, int), np.array([105.0, 68.0]))


class DegradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = synthetic()
        cls.before = (cls.clean.players.copy(), cls.clean.ball.copy())
        cls.out, cls.stats = degrade(cls.clean, Config(seed=7))

    def test_input_unchanged(self):
        np.testing.assert_array_equal(self.clean.players, self.before[0])
        np.testing.assert_array_equal(self.clean.ball, self.before[1])

    def test_deterministic_with_seed(self):
        again, stats = degrade(self.clean, Config(seed=7))
        np.testing.assert_array_equal(np.nan_to_num(again.players, nan=-999), np.nan_to_num(self.out.players, nan=-999))
        np.testing.assert_array_equal(np.nan_to_num(again.ball, nan=-999), np.nan_to_num(self.out.ball, nan=-999))
        other, _ = degrade(self.clean, Config(seed=8))
        self.assertFalse(np.array_equal(np.isnan(other.ball), np.isnan(self.out.ball)))

    def test_ball_visibility_near_published_target(self):
        self.assertAlmostEqual(self.stats["ball_visible_share"], 0.593, delta=0.05)

    def test_mean_objects_near_published_target(self):
        self.assertAlmostEqual(self.stats["mean_objects"], 20.5, delta=1.0)

    def test_fixed_slot_pool_and_frame_axis(self):
        self.assertEqual(len(self.out.slot_ids), 46)
        self.assertEqual(list(self.out.teams).count("H"), 23)
        np.testing.assert_array_equal(self.out.frame, self.clean.frame)
        np.testing.assert_array_equal(self.out.period, self.clean.period)

    def test_off_pitch_ball_mostly_missing(self):
        n = len(self.clean.frame)
        self.assertGreater(np.isnan(self.out.ball[n // 3: n // 3 + 100, 0]).mean(), 0.75)

    def test_ball_gaps_are_bursty(self):
        m = np.isnan(self.out.ball[:, 0]).astype(int)
        starts = np.sum((m[1:] == 1) & (m[:-1] == 0))
        self.assertGreater(m.sum() / max(starts, 1), 10)  # mean burst well over 10 frames

    def test_module_cannot_see_labels(self):
        tree = ast.parse((ROOT / "src/possession/degrade.py").read_text())
        names = [a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                 for a in n.names] + [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any("evaluation" in x or "labels" in x or "references" in x for x in names))


if __name__ == "__main__":
    unittest.main()
