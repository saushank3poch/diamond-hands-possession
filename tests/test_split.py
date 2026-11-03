"""The committed split is reproducible, disjoint, and covers every downloaded match."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from possession import split


class SplitTests(unittest.TestCase):
    def test_committed_split_matches_seeded_regeneration(self):
        committed = split.load()
        self.assertEqual(committed, split.build())

    def test_disjoint_complete_and_stratified(self):
        body = split.load()
        pairs = split.matches_from_manifest()
        self.assertEqual(len(pairs), 7)
        self.assertFalse(set(body["holdout"]) & set(body["dev"]))
        self.assertEqual(sorted(body["holdout"] + body["dev"]), sorted(m for _, m in pairs))
        self.assertGreaterEqual(len(body["holdout"]), 2)
        comps = {body["competition_of"][m] for m in body["holdout"]}
        self.assertEqual(comps, {c for c, _ in pairs})

    def test_folds_are_leave_one_match_out_over_dev_only(self):
        body = split.load()
        self.assertEqual([f["validate"] for f in body["dev_folds"]], body["dev"])
        for fold in body["dev_folds"]:
            self.assertNotIn(fold["validate"], fold["train"])
            self.assertFalse(set(fold["train"]) & set(body["holdout"]))

    def test_tampering_is_detected(self):
        body = split.load()
        body["holdout"] = body["dev"][:2]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.yaml"
            path.write_text(json.dumps(body))
            with self.assertRaises(ValueError):
                split.load(path)

    def test_draw_rejects_stratum_without_dev_match(self):
        with self.assertRaises(ValueError):
            split.draw([("C1", "M1"), ("C2", "M2"), ("C2", "M3")])


if __name__ == "__main__":
    unittest.main()
