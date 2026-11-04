"""E006 dead-edge offset: synthetic mechanics only (no references at predict time)."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from possession.predictors.deadball import DeadBall
from possession.predictors.edgeoffset import EdgeOffset, _median_offset, shift_dead
from possession.predictors.xpolicy import ShortSpellX
from test_deadball import synthetic


def mask(s):
    return np.array([c == "D" for c in s])


def show(m):
    return "".join("D" if v else "." for v in m)


class ShiftDeadTests(unittest.TestCase):
    def test_moves_start_and_end(self):
        # stoppage 2 late -> start 2 earlier; restart 1 early (shift -1) -> end 1 later
        self.assertEqual(show(shift_dead(mask("......DDD......"), 2, -1)), "....DDDDDD.....")

    def test_block_boundaries_keep_their_side(self):
        self.assertEqual(show(shift_dead(mask("DDD......DDD"), 2, -1)), "DDDD...DDDDD")

    def test_collapsing_run_is_dropped(self):
        self.assertEqual(show(shift_dead(mask("....DD....."), -2, 2)), "...........")

    def test_overlapping_runs_merge(self):
        self.assertEqual(show(shift_dead(mask("..DD..DD...."), 2, -2)), "DDDDDDDDDD..")

    def test_clipped_to_block(self):
        self.assertEqual(show(shift_dead(mask(".DD......"), 5, 0)), "DDD......")

    def test_zero_shift_is_identity(self):
        m = mask("..DDD...DD.D.")
        self.assertTrue((shift_dead(m, 0, 0) == m).all())


class MedianOffsetTests(unittest.TestCase):
    def test_no_matched_edges_gives_zero(self):
        self.assertEqual(_median_offset([], [10, 20]), 0)
        self.assertEqual(_median_offset([10], []), 0)
        self.assertEqual(_median_offset([10], [500]), 0)

    def test_median_signed(self):
        self.assertEqual(_median_offset([100, 300, 500], [97, 296, 510]), -3)


class EdgeOffsetModelTests(unittest.TestCase):
    def test_shift_never_crosses_period_boundary(self):
        tr, codes = synthetic()
        model = EdgeOffset(min_dead_s=1.0, min_live_s=1.0).fit([(tr, codes)])
        b = int(np.flatnonzero(np.diff(tr.period))[0]) + 1
        n = len(codes)
        # Period 1 ends dead; period 2 starts live. A large late-release shift must not spill.
        m1 = np.zeros(b, bool); m1[b - 100:] = True
        m2 = np.zeros(n - b, bool); m2[300:400] = True
        model._dead_blocks = lambda t: iter([(0, b, m1), (b, n, m2)])
        model.stop_shift, model.release_shift = 50, -500
        pred = model.predict(tr)
        self.assertTrue((pred[b - 150:b] == "D").all())       # start moved 50 earlier, end kept
        self.assertTrue((pred[b:b + 250] != "D").all())       # nothing spilled into period 2
        self.assertTrue((pred[b + 250:b + 900] == "D").all())  # period-2 run: start -50, end +500
        self.assertTrue((pred[b + 900:] != "D").all())

    def test_zero_offsets_reproduce_e005(self):
        tr, codes = synthetic()
        e005 = ShortSpellX(mode="carry", min_dead_s=1.0, min_live_s=1.0).fit([(tr, codes)])
        e006 = EdgeOffset(min_dead_s=1.0, min_live_s=1.0).fit([(tr, codes)])
        e006.stop_shift = e006.release_shift = 0
        self.assertTrue((e005.predict(tr) == e006.predict(tr)).all())

    def test_fit_learns_known_lag_and_stays_within_periods(self):
        tr, codes = synthetic()
        # Reference dead runs shifted 10 frames later than the detector can see: prediction is early.
        model = EdgeOffset(min_dead_s=1.0, min_live_s=1.0).fit([(tr, codes)])
        base_stop, base_rel = model.stop_shift, model.release_shift
        late = np.r_[np.array(["H"] * 10), codes[:-10]]
        model._fit_offsets([(tr, late)])
        self.assertEqual((model.stop_shift, model.release_shift), (base_stop - 10, base_rel - 10))
        pred = model.predict(tr)
        self.assertEqual(len(pred), len(codes))
        self.assertTrue(set(np.unique(pred)) <= {"H", "A", "D", "X"})


if __name__ == "__main__":
    unittest.main()
