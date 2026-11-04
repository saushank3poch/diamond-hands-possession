"""Restart-timing audit (evaluation only): one-to-one edge matching and signed latency."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from possession.evaluation.restarts import edge_timing, match_edges, summarize_timing


def arr(s):
    return np.array(list(s))


class EdgeTimingTests(unittest.TestCase):
    def test_late_release_and_early_stop_signed(self):
        gt = arr("H" * 100 + "D" * 100 + "A" * 100)
        pr = arr("H" * 90 + "D" * 120 + "A" * 90)   # stop 10 early, release 10 late
        t = edge_timing(gt, pr, np.ones(300, int))
        self.assertEqual(t["stop"]["latencies"], [-10])
        self.assertEqual(t["release"]["latencies"], [10])

    def test_missed_and_spurious_edges_are_unmatched(self):
        gt = arr("H" * 100 + "D" * 100 + "A" * 100)
        pr = arr("H" * 300)
        t = edge_timing(gt, pr, np.ones(300, int))
        self.assertEqual((t["release"]["unmatched_ref"], t["release"]["n_pred"]), (1, 0))
        t = edge_timing(arr("H" * 300), arr("H" * 100 + "D" * 100 + "H" * 100), np.ones(300, int))
        self.assertEqual((t["release"]["n_ref"], t["release"]["unmatched_pred"]), (0, 1))

    def test_one_to_one_and_window(self):
        m = match_edges([100, 110], [105], np.ones(300, int), window=75)
        self.assertEqual((len(m["latencies"]), m["unmatched_ref"]), (1, 1))
        m = match_edges([100], [200], np.ones(300, int), window=75)
        self.assertEqual((m["latencies"], m["unmatched_ref"], m["unmatched_pred"]), ([], 1, 1))

    def test_no_cross_period_edges_or_pairs(self):
        period = np.r_[np.ones(100, int), np.full(100, 2)]
        gt = arr("D" * 100 + "H" * 100)  # D->H only at the period boundary: not an edge
        t = edge_timing(gt, gt, period)
        self.assertEqual(t["release"]["n_ref"], 0)
        m = match_edges([98], [102], period, window=75)
        self.assertEqual(m["latencies"], [])

    def test_x_in_prediction_is_carried(self):
        gt = arr("D" * 50 + "H" * 50)
        pr = arr("D" * 50 + "X" * 10 + "H" * 40)   # X after D acts as D: release 10 late
        self.assertEqual(edge_timing(gt, pr, np.ones(100, int))["release"]["latencies"], [10])

    def test_summary_pools(self):
        gt = arr("H" * 100 + "D" * 100 + "A" * 100)
        pr = arr("H" * 90 + "D" * 120 + "A" * 90)
        s = summarize_timing([edge_timing(gt, pr, np.ones(300, int))] * 2)
        self.assertEqual((s["release"]["matched"], s["release"]["median"], s["release"]["share_late"]), (2, 10.0, 1.0))


if __name__ == "__main__":
    unittest.main()
