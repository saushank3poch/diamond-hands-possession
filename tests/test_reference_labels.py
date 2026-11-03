"""Synthetic checks for the evaluator-side DFL label transform."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from possession.evaluation.labels import confirm_control, proxy_labels, raw_hd, reference_labels
from possession.evaluation.references import References


def lab(s, fps_per_char=1):
    return np.array(list(s))


class ConfirmControlTests(unittest.TestCase):
    def run_(self, s, confirm, periods=None):
        period = np.ones(len(s), dtype=int) if periods is None else np.array(periods)
        return "".join(confirm_control(lab(s), period, confirm_s=confirm, fps=1))

    def test_short_opposing_touch_is_absorbed(self):
        self.assertEqual(self.run_("HHHHAHHHH", 2), "HHHHHHHHH")

    def test_held_switch_is_kept_at_its_original_frame(self):
        self.assertEqual(self.run_("HHHHAAAHH", 2), "HHHHAAAHH")

    def test_touch_then_dead_keeps_previous_owner(self):
        self.assertEqual(self.run_("HHHADDD", 2), "HHHHDDD")

    def test_restart_spell_always_kept(self):
        self.assertEqual(self.run_("DDAHHH", 2), "DDAHHH")

    def test_ping_pong_collapses_to_owner(self):
        self.assertEqual(self.run_("HHHAHAHAAAA", 3), "HHHHHHHAAAA")

    def test_no_carry_across_periods_or_unknown(self):
        # Period 2 opens with a short A spell (a restart, kept); the 1-frame H after it is absorbed.
        self.assertEqual(self.run_("HHHAAH", 3, [1, 1, 1, 2, 2, 2]), "HHHAAA")
        self.assertEqual(self.run_("HHHXAH", 3), "HHHXAA")  # X breaks the chain: A restarts, H absorbed

    def test_raw_mapping(self):
        alive = np.array([0, 1, 1, -1])
        owner = np.array(["H", "H", "A", "A"])
        self.assertEqual("".join(raw_hd(alive, owner)), "DHAX")


class MaskTests(unittest.TestCase):
    def refs(self, alive, owner, unknown):
        n = len(alive)
        return References(np.arange(n), np.array(alive, dtype=np.int8), np.array(list(owner)),
                          np.array(unknown, dtype=bool), (None,) * n, (None,) * n)

    def test_unknown_mask_is_x_in_reference(self):
        r = self.refs([1, 1, 1], "HHH", [False, True, False])
        self.assertEqual("".join(reference_labels(r)), "HXH")

    def test_unknown_mask_survives_proxy_and_breaks_state(self):
        r = self.refs([1, 1, 1, 1, 1, 1], "HHHXAH", [False, False, False, True, False, False])
        out = "".join(proxy_labels(r, np.ones(6, dtype=int), confirm_s=3))
        self.assertEqual(out[3], "X")
        self.assertEqual(out, "HHHXAA")
        r = self.refs([1, 1, 1], "HHH", [False, True, False])
        self.assertEqual("".join(proxy_labels(r, np.ones(3, dtype=int), confirm_s=1)), "HXH")


if __name__ == "__main__":
    unittest.main()
