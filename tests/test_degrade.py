"""FCB-8: degradation acts on input copies only and hits the published DATA.md targets."""
import ast
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from possession.degrade import Config, _ball_entry_logits, _markov_mask, block_starts, degrade
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


class PeriodBoundaryTests(unittest.TestCase):
    def two_periods(self, gap=0):
        """Stationary players; the ball jumps between periods; optional period-0 rows between."""
        n1, n2 = 500, 500
        n = n1 + gap + n2
        period = np.r_[np.ones(n1, int), np.zeros(gap, int), np.full(n2, 2)]
        players = np.zeros((n, 2, 2))
        players[:, 0] = [-10.0, 0.0]
        players[:, 1] = [10.0, 0.0]
        ball = np.zeros((n, 2))
        ball[n1:] = [20.0, 10.0]
        return Tracking(np.arange(n), np.arange(n), period, np.array(["a", "b"]), np.array(["H", "A"]),
                        players, ball, np.r_[np.ones(n1, int), np.zeros(gap, int), np.ones(n2, int)],
                        np.array([105.0, 68.0]))

    def test_no_speed_spike_at_boundary(self):
        for gap in (0, 30):
            tr = self.two_periods(gap)
            logits = _ball_entry_logits(tr, Config(crowd_weight=0.0), block_starts(tr.period))
            self.assertEqual(float(np.abs(logits).max()), 0.0)

    def test_tracklets_are_cut_at_period_boundaries(self):
        for gap in (0, 30):
            tr = self.two_periods(gap)
            cfg = Config(seed=3, mean_objects=2.0, tracklet_break_per_min=0.0, duplicate_rate=0.0,
                         jitter_player_m=0.0, jitter_ball_m=0.0)
            _, stats = degrade(tr, cfg)
            # Two uninterrupted players, with 2 or 3 blocks (period 1, optional period 0, period 2).
            self.assertEqual(stats["tracklets"], 2 * (3 if gap else 2))

    def test_no_slot_carries_a_track_across_the_boundary(self):
        tr = self.two_periods(0)
        out, _ = degrade(tr, Config(seed=5, mean_objects=2.0, tracklet_break_per_min=0.0, duplicate_rate=0.0))
        # Each tracklet is period-bounded, so slot assignment is redrawn independently per period.
        first = ~np.isnan(out.players[499]).any(axis=-1)
        second = ~np.isnan(out.players[500]).any(axis=-1)
        self.assertEqual(first.sum(), 2)
        self.assertEqual(second.sum(), 2)

    def test_chains_reset_at_boundary(self):
        # Enter missing at frame 0 and never recover (q_off=0); the reset at frame 3 must clear it.
        p_on = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        resets = np.array([True, False, False, True, False])
        out = _markov_mask(np.random.default_rng(0), p_on, 0.0, resets)
        self.assertEqual(out.tolist(), [True, True, True, False, False])

class ReportSeedTests(unittest.TestCase):
    def test_nonzero_cli_seed_is_recorded_per_match(self):
        import importlib.util, json, tempfile
        from unittest.mock import patch
        spec = importlib.util.spec_from_file_location("degrade_report", ROOT / "scripts/degrade_report.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tr = synthetic(n=600)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "DFL_02_01_matchinformation_C_M.xml").write_text("")
            Path(tmp, "DFL_04_03_positions_raw_observed_C_M.xml").write_text("")
            out = Path(tmp) / "r.json"
            with patch.object(mod.split, "load", return_value={"split_sha256": "x", "dev": ["M"], "holdout": []}), \
                 patch.object(mod, "load_dfl", return_value=tr), \
                 patch.object(sys, "argv", ["x", "--data", tmp, "--out", str(out), "--seed", "7"]):
                mod.main()
            report = json.loads(out.read_text())
        self.assertEqual(report["base_seed"], 7)
        self.assertEqual(report["matches"][0]["effective_config"]["seed"], mod.match_seed(7, "M"))
        self.assertNotEqual(mod.match_seed(7, "M"), mod.match_seed(0, "M"))


if __name__ == "__main__":
    unittest.main()
