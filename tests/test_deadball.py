"""FCB-11: detector mechanics on synthetic tracking (no references)."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from possession.predictors.deadball import DeadBall, _min_runs, _rolling_mean, _window_speed
from possession.tracking import Tracking


def synthetic(n=3000, seed=0):
    """Players run in 'live' stretches and stand still in 'dead' ones; ball leaves the pitch when dead."""
    rng = np.random.default_rng(seed)
    dead = (np.arange(n) // 500) % 2 == 1
    step = np.where(dead[:, None, None], 0.0, rng.normal(0, 0.12, (n, 6, 2)))
    players = np.cumsum(step, axis=0) + rng.uniform(-20, 20, (1, 6, 2))
    ball = players[:, 0] + 0.5
    ball[dead] = [60.0, 0.0]
    ball[dead & (rng.random(n) < 0.8)] = np.nan
    period = np.r_[np.ones(n // 2, int), np.full(n - n // 2, 2)]
    tr = Tracking(np.arange(n), np.arange(n), period, np.array(list("abcdef")), np.array(list("HHHAAA")),
                  players, ball, np.ones(n, int), np.array([105.0, 68.0]))
    codes = np.where(dead, "D", "H")
    return tr, codes


class DeadBallTests(unittest.TestCase):
    def test_learns_separable_synthetic_and_respects_periods(self):
        tr, codes = synthetic()
        model = DeadBall(min_dead_s=1.0, min_live_s=1.0).fit([(tr, codes)])
        pred = model.predict(tr)
        self.assertEqual(len(pred), len(codes))
        self.assertGreater(np.mean((pred == "D") == (codes == "D")), 0.85)
        self.assertTrue(set(np.unique(pred)) <= {"H", "A", "D", "X"})

    def test_window_speed_ignores_iid_jitter(self):
        n = 400
        rng = np.random.default_rng(1)
        still = np.zeros((n, 1, 2)) + rng.normal(0, 0.15, (n, 1, 2))
        v = _window_speed(still, [(0, n)])
        self.assertLess(np.nanmean(v), 0.5)  # per-frame differencing would give ~5 m/s

    def test_rolling_mean_does_not_cross_blocks(self):
        x = np.r_[np.zeros(10), np.ones(10)]
        out = _rolling_mean(x, 5, [(0, 10), (10, 20)])
        self.assertTrue(np.all(out[:10] == 0) and np.all(out[10:] == 1))

    def test_min_runs(self):
        d = np.array([0, 0, 1, 0, 0, 1, 1, 1, 0, 0], bool)
        self.assertEqual(_min_runs(d, 2, 1).astype(int).tolist(), [0, 0, 0, 0, 0, 1, 1, 1, 0, 0])

    def test_unknown_group_rejected(self):
        with self.assertRaises(ValueError):
            DeadBall(groups=["smart_tagging"])


if __name__ == "__main__":
    unittest.main()
