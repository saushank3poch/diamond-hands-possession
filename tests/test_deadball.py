"""FCB-11: detector mechanics on synthetic tracking (no references)."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from possession.predictors.deadball import DeadBall, _min_runs, _rolling_mean, _window_speed, live_owner, signals
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


def with_period(tr, period, players=None, ball=None):
    return Tracking(tr.frame, tr.source_frame, period, tr.slot_ids, tr.teams,
                    tr.players if players is None else players, tr.ball if ball is None else ball,
                    np.ones(len(tr.frame), int), tr.pitch)


def same(a, b):
    return np.array_equal(np.nan_to_num(a, nan=-9e9), np.nan_to_num(b, nan=-9e9))


class PeriodInvarianceTests(unittest.TestCase):
    def check(self, period):
        tr, _ = synthetic(len(period))
        first = np.flatnonzero(period == 1)
        other = np.flatnonzero(period != 1)
        players = tr.players.copy()
        ball = tr.ball.copy()
        players[other] = players[other] * 1.7 + np.random.default_rng(9).normal(0, 0.5, players[other].shape)
        ball[other] = np.where(np.isfinite(ball[other]), ball[other] * 0.3, np.nan)
        a = signals(with_period(tr, period))
        b = signals(with_period(tr, period, players, ball))
        for group in a:
            for (name, x), (_, y) in zip(a[group], b[group]):
                self.assertTrue(same(x[first], y[first]), f"{group}/{name} leaked across the boundary")

    def test_adjacent_periods(self):
        self.check(np.r_[np.ones(1500, int), np.full(1500, 2)])

    def test_period_zero_gap(self):
        self.check(np.r_[np.ones(1400, int), np.zeros(200, int), np.full(1400, 2)])

    def test_short_periods(self):
        self.check(np.r_[np.ones(100, int), np.full(100, 2), np.full(100, 3)])


class LiveOwnerTests(unittest.TestCase):
    def test_live_owner_never_d(self):
        self.assertEqual("".join(live_owner(np.array(["", "", "H", "", "", "A", ""]), 0)), "HHHHHAA")
        self.assertEqual("".join(live_owner(np.array(["", "", ""]), 5)), "XXX")

    def test_missing_ball_alone_never_produces_d(self):
        tr, codes = synthetic()
        model = DeadBall().fit([(tr, codes)])
        no_ball = with_period(tr, tr.period, ball=np.full_like(tr.ball, np.nan))
        model.dead_probability = lambda t: np.zeros(len(t.frame))
        out = model.predict(no_ball)
        self.assertNotIn("D", set(out))
        self.assertTrue(set(out) <= {"X"})
        no_players = with_period(tr, tr.period, players=np.full_like(tr.players, np.nan))
        self.assertNotIn("D", set(model.predict(no_players)))


class BallFreeAblationTests(unittest.TestCase):
    def test_no_ball_detector_ignores_ball(self):
        tr, codes = synthetic()
        model = DeadBall(groups=["motion", "formation", "restart"]).fit([(tr, codes)])
        rng = np.random.default_rng(3)
        moved = with_period(tr, tr.period, ball=np.where(rng.random((len(tr.frame), 1)) < 0.5, np.nan,
                                                         rng.uniform(-50, 50, tr.ball.shape)))
        self.assertTrue(same(model.dead_probability(tr), model.dead_probability(moved)))


if __name__ == "__main__":
    unittest.main()
