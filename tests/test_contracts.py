"""Published contracts exercised with generated inputs only; no fixture truth loaded."""

import csv
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluation import possession_metrics as pm
from games import GAME, GAMES, half_video_blocks
from possession.periods import predict_periods
from starter import baseline, load_atd, load_homography, load_smart_tagging


class AlignmentTests(unittest.TestCase):
    def test_atd_mapping_is_one_based_not_ground_truth_counter(self):
        for frame_id, video in [(1, 0), (11186, 11185), (110809, 110808), (199319, 199318)]:
            with self.subTest(frame_id=frame_id):
                self.assertEqual(load_atd.video_frame_of(frame_id), video)
                self.assertEqual(load_atd.frame_id_of(video), frame_id)

    def test_nearest_player_reader_uses_that_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic-atd.txt"
            path.write_text("1:0,0,0;1,1,0:0,0,0\n3:0,0,0;1,1,0:1,1,0\n")
            result = baseline.nearest_player_labels(path, [baseline.HOME, baseline.AWAY],
                                                     3, (100, 60))
        self.assertEqual(result.tolist(), [baseline.HOME, baseline.NOBODY, baseline.AWAY])

    def test_tag_seconds_round_to_video_frames_with_half_open_end(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic-tags.xml"
            path.write_text("""<root><instance><start>0.02</start><end>0.06</end>
<code>SET PIECES</code><label><group>Team</group><text>N/A</text></label>
</instance><instance><start>1.04</start><end>2.00</end><code>BUILD UP</code>
</instance></root>""")
            with patch.object(load_smart_tagging, "pattern_path", return_value=path):
                events = load_smart_tagging.load(fps=25)
        self.assertEqual([(e["start_frame"], e["end_frame"]) for e in events], [(0, 2), (26, 50)])
        self.assertEqual(events[0]["tags"]["Team"], "N/A")

    def test_homography_zero_based_alignment_and_projection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic-homography.csv"
            pd.DataFrame([
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.1, 1],
                [1, 2, 0, 0, 0, 3, 0, 0, 0, 1, 0.2, 1],
            ], columns=["frame"] + load_homography.MATRIX_COLUMNS + ["reprojection_error", "valid"]).to_csv(path, index=False)
            with patch.object(load_homography, "require", return_value=[path]), \
                 patch.object(load_homography, "source_frames", return_value=2):
                self.assertTrue(load_homography.check_alignment()["agrees"])
                matrices, _, valid = load_homography.load()
                self.assertEqual(load_homography.project(matrices[1], 0.5, 0.5), (1.0, 1.5))
                self.assertTrue(valid.all())
                table = pd.read_csv(path)
                table["frame"] += 1
                table.to_csv(path, index=False)
                self.assertFalse(load_homography.check_alignment()["agrees"])

    def test_published_period_endpoints_and_outside_mask(self):
        periods = half_video_blocks(GAME)
        self.assertEqual(periods, [(1, 11185, 90651), (2, 110808, 190197)])
        self.assertEqual(GAMES[GAME]["n_video_frames"], 199319)
        self.assertEqual(GAMES[GAME]["fps"], 25)
        blocks = [(start, end + 1) for _, start, end in periods]
        labels = predict_periods([None] * 199319, blocks, lambda: lambda rows: ["D"] * len(rows))
        self.assertEqual(labels.count("D"), 158857)
        self.assertEqual(labels.count("X"), 40462)
        for frame in (0, 11184, 90652, 110807, 190198, 199318):
            self.assertEqual(labels[frame], "X")
        for frame in (11185, 90651, 110808, 190197):
            self.assertEqual(labels[frame], "D")


class PeriodIsolationTests(unittest.TestCase):
    # A test double that remembers symbols, not a possession model.
    @staticmethod
    def new_carry():
        previous = "X"
        def predict(rows):
            nonlocal previous
            output = []
            for value in rows:
                if value is not None:
                    previous = value
                output.append(previous)
            return output
        return predict

    def test_half_or_observable_discontinuity_gets_fresh_state(self):
        labels = predict_periods(["H", None, None, "A", None, None], [(0, 2), (3, 6)], self.new_carry)
        self.assertEqual(labels, list("HHXAAA"))
        # Unknown first observation in the next block must not inherit H.
        labels = predict_periods(["H", None, None, None, "A", None], [(0, 2), (3, 6)], self.new_carry)
        self.assertEqual(labels, list("HHXXAA"))

    def test_reference_changes_only_reach_evaluator_and_never_mutate_predictions(self):
        observations = ["H", None, None, "A", None, None]
        blocks = [(0, 6)]
        expected = None
        for truth in ("HHHAAA", "HXXAAA", "DDDDDD"):
            labels = predict_periods(observations, blocks, self.new_carry)
            prediction = pm.label_codes(labels)
            before = prediction.tobytes()
            pm.evaluate(pm.label_codes(list(truth)), prediction)
            self.assertEqual(before, prediction.tobytes())
            expected = before if expected is None else expected
            self.assertEqual(before, expected)
            self.assertEqual(labels, list("HHHAAA"))  # GT-X did not reset at frames 1/2.
        with self.assertRaises(TypeError):
            predict_periods(observations, blocks, self.new_carry, reference_mask=[True] * 6)

    def test_invalid_blocks_rejected_before_predictor_runs(self):
        for blocks in ([(0, 0)], [(-1, 2)], [(0, 7)], [(0, 4), (3, 6)], [(3, 6), (0, 2)], [(False, 2)]):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                predict_periods([None] * 6, blocks, lambda: self.fail("invalid blocks reached predictor"))

    def test_prediction_length_and_alphabet_enforced(self):
        for result in ([], ["H", "A"], [None], ["contested"], ["h"]):
            with self.subTest(result=result), self.assertRaises(ValueError):
                predict_periods([None], [(0, 1)], lambda: lambda rows: result)


class SubmissionTests(unittest.TestCase):
    def test_exact_synthetic_output_passes_official_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.csv"
            blocks = [(start, end + 1) for _, start, end in half_video_blocks(GAME)]
            labels = predict_periods([None] * 199319, blocks, lambda: lambda rows: ["D"] * len(rows))
            baseline.write_submission(str(path), labels)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open(newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["frame", "label"])
            self.assertEqual(len(rows), 199320)
            self.assertEqual([int(row[0]) for row in rows[1:]], list(range(199319)))
            self.assertTrue({row[1] for row in rows[1:]} <= {"H", "A", "D", "X"})
            result = subprocess.run([sys.executable, str(ROOT / "evaluation/validate_submission.py"), str(path)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("199319 rows, frame 0..199318", result.stdout)
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_validator_rejects_malformed_rows(self):
        cases = [
            ([0, 1], ["H", "A"], 3),
            ([1, 2, 3], ["H", "A", "D"], 3),
            ([0, 1, 1], ["H", "A", "D"], 3),
            ([0, 2, 1], ["H", "A", "D"], 3),
            ([0.0, 1.0, 2.0], ["H", "A", "D"], 3),
            ([0, 1, 2], ["H", None, "D"], 3),
            ([0, 1, 2], ["H", "", "D"], 3),
            ([0, 1, 2], ["H", "C", "D"], 3),
        ]
        for frames, labels, expected in cases:
            with self.subTest(frames=frames, labels=labels):
                self.assertIsNotNone(pm.validate_submission(pd.DataFrame({"frame": frames, "label": labels}), expected))

    def test_missing_header_is_rejected_by_official_reader(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            path.write_text("0,H\n1,A\n")
            with self.assertRaises(ValueError):
                pm.read_label_csv(path)


class ScorerTests(unittest.TestCase):
    @staticmethod
    def score(truth, prediction):
        return pm.evaluate(pm.label_codes(list(truth)), pm.label_codes(list(prediction)))

    def test_six_reference_cases(self):
        cases = [
            ("HHH", "XHX", "HHH", 1.0, 1 / 6),
            ("HHH", "XXX", "XXX", 0.0, 0.0),
            ("HHDAA", "HXXXA", "HHHHA", 0.6, 4 / 9),
            ("HHXAA", "HXXXA", "HHXAA", 1.0, 4 / 9),
            ("HHAADD", "HHHADD", "HHHADD", 0.7, 37 / 45),
            ("HHHH", "HHHH", "HHHH", 1.0, 1 / 3),
        ]
        for truth, pred, filled, spell_f1, macro_f1 in cases:
            with self.subTest(truth=truth, pred=pred):
                gt, raw = pm.label_codes(list(truth)), pm.label_codes(list(pred))
                actual = pm.fill_abstentions(raw, pm.valid_runs(gt))
                self.assertEqual("".join(pm.LABELS[i] for i in actual), filled)
                result = self.score(truth, pred)
                self.assertAlmostEqual(result["spell_f1_mean"], spell_f1)
                self.assertAlmostEqual(result["macro_f1"], macro_f1)
                self.assertAlmostEqual(result["spell_f1_mean"], self.score(truth, filled)["spell_f1_mean"])

    def test_x_is_not_invariant_relative_to_original_prediction(self):
        self.assertEqual(self.score("HHDAA", "HHDAA")["spell_f1_mean"], 1.0)
        self.assertAlmostEqual(self.score("HHDAA", "HXXXA")["spell_f1_mean"], 0.6)

    def test_x_macro_f1_can_improve_or_worsen_relative_to_filled_prediction(self):
        self.assertGreater(self.score("HHHAAAD", "HHHXXXD")["macro_f1"],
                           self.score("HHHAAAD", "HHHHHHD")["macro_f1"])
        self.assertLess(self.score("HHH", "XHX")["macro_f1"], self.score("HHH", "HHH")["macro_f1"])

    def test_dead_carry_and_leading_backfill(self):
        for pred, expected in [("DXXA", "DDDA"), ("XXDA", "DDDA"), ("XXXX", "XXXX")]:
            filled = pm.fill_abstentions(pm.label_codes(list(pred)), [(0, 4)])
            self.assertEqual("".join(pm.LABELS[i] for i in filled), expected)

    def test_dead_ball_splits_same_team_spells(self):
        codes = pm.label_codes(list("HHDHH"))
        self.assertEqual(pm.possession_spells(codes, [(0, 5)]), [(0, 2, pm.H), (3, 5, pm.H)])

    def test_matching_is_same_team_one_to_one_and_threshold_inclusive(self):
        overlaps = pm.spell_overlaps([(0, 4, pm.H), (4, 8, pm.H)], [(0, 8, pm.H)])
        self.assertEqual(overlaps, [(0.5, 0, 0), (0.5, 1, 0)])
        self.assertEqual(pm.match_spells(overlaps, 2, 1, 0.5), (1, 0, 1))
        self.assertEqual(pm.match_spells(overlaps, 2, 1, 0.6), (0, 1, 2))
        self.assertEqual(pm.spell_overlaps([(0, 4, pm.H)], [(0, 4, pm.A)]), [])
        self.assertEqual(pm.SPELL_IOUS, (0.3, 0.4, 0.5, 0.6, 0.7))
        result = self.score("HHAADD", "HHHADD")
        self.assertAlmostEqual(result["spell_f1_mean"], np.mean([v["f1"] for v in result["spell_f1"].values()]))

    def test_ungraded_predictions_do_not_affect_scores(self):
        left = self.score("XXHHAAXX", "DDHHAADD")
        right = self.score("XXHHAAXX", "AAHHAAHH")
        for key in ("spell_f1_mean", "macro_f1", "n_graded", "n_gt_spells", "n_pred_spells"):
            self.assertEqual(left[key], right[key])


if __name__ == "__main__":
    unittest.main()
