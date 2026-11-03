"""Synthetic format contracts only: never open target or external match records."""
import ast
from dataclasses import fields
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from possession.adapters.atd import load_atd
from possession.adapters.dfl import load_dfl
from possession.evaluation.references import load_dfl_references
from possession.tracking import Tracking


def metadata():
    def team(role, name, pid):
        return f'''<Team TeamId="{name}" TeamName="{name}" Role="{role}">
<TrainerStaff/><Players><Player PersonId="{pid}" ShirtNumber="1" Shortname="test"
FirstName="test" LastName="test" Starting="true"/></Players></Team>'''
    return '''<PutDataRequest><MatchInformation><Environment PitchX="100" PitchY="60"/>
<General Result="0:0"/><OtherGameInformation TotalTimeFirstHalf="120" TotalTimeSecondHalf="40"/>
<Teams>''' + team("home", "h", "p1") + team("guest", "a", "p2") + '''</Teams>
</MatchInformation></PutDataRequest>'''


def positions(status="1", owner="1"):
    return f'''<Positions>
<FrameSet GameSection="firstHalf" TeamId="h" PersonId="p1">
<Frame N="10000" X="-25" Y="15"/><Frame N="10002" X="NaN" Y="8"/></FrameSet>
<FrameSet GameSection="firstHalf" TeamId="a" PersonId="p2">
<Frame N="10000" X="25" Y="-15"/></FrameSet>
<FrameSet GameSection="firstHalf" TeamId="BALL">
<Frame N="10000" X="0" Y="0" BallStatus="{status}" BallPossession="{owner}"/>
<Frame N="10002" Y="9" BallStatus="0" BallPossession="99"/></FrameSet>
<FrameSet GameSection="secondHalf" TeamId="h" PersonId="p1">
<Frame N="100000" X="25" Y="-15"/></FrameSet></Positions>'''


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.meta, self.raw = self.base / "meta.xml", self.base / "raw.xml"
        self.meta.write_text(metadata())
        self.raw.write_text(positions())

    def dfl(self):
        return load_dfl(self.meta, self.raw, directions={1: 1, 2: -1})

    def atd_fixture(self):
        self.meta.write_text('<Metadata><Team id="h"/><Team id="a"/>'
                            '<Player id="p1" teamId="h"/><Player id="p2" teamId="a"/></Metadata>')
        self.raw.write_text('1:0.25,0.25,0;0.75,0.75,0:0.5,0.5,0\n'
                            '2:NaN,0.4,0;NaN,NaN,0:NaN,NaN,0\n'
                            '3:NaN,NaN,0;NaN,NaN,0:NaN,0.3,0\n'
                            '4:0.75,0.75,0;NaN,NaN,0:NaN,NaN,0\n')

    def atd(self):
        with patch.dict(os.environ, {"POSSESSION_TARGET_LOG_DIR": str(self.base),
                                     "POSSESSION_RUN_DIR": str(self.base)}):
            return load_atd(self.meta, self.raw, n_frames=4, pitch=(100, 60),
                            blocks=[(1, 0, 3), (2, 3, 4)], directions={1: 1, 2: -1})

    def assert_tracking_equal(self, left, right):
        for field in fields(Tracking):
            np.testing.assert_array_equal(getattr(left, field.name), getattr(right, field.name))

    def test_dfl_gaps_missingness_and_periods(self):
        tracking = self.dfl()
        np.testing.assert_array_equal(tracking.source_frame, [10000, 10001, 10002, 100000])
        np.testing.assert_array_equal(tracking.period, [1, 1, 1, 2])
        np.testing.assert_array_equal(tracking.home_direction, [1, 1, 1, -1])
        np.testing.assert_array_equal(tracking.ball_missing, [False, True, True, True])
        self.assertTrue(tracking.player_missing[2].all())
        self.assertFalse(tracking.player_missing[3, 0])  # ball-less frame retained
        np.testing.assert_array_equal(tracking.players[0], [[-25, 15], [25, -15]])

    def test_source_coordinates_and_missingness_agree(self):
        dfl = self.dfl()
        self.atd_fixture()
        atd = self.atd()
        for name in ("players", "ball", "period", "teams", "home_direction", "pitch"):
            np.testing.assert_array_equal(getattr(dfl, name), getattr(atd, name))
        np.testing.assert_array_equal(atd.source_frame, [1, 2, 3, 4])

    def test_both_adapters_round_trip(self):
        dfl = self.dfl()
        self.atd_fixture()
        for tracking in (dfl, self.atd()):
            path = self.base / "tracking.npz"
            tracking.save(path)
            self.assert_tracking_equal(tracking, Tracking.load(path))
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {f.name for f in fields(Tracking)})

    def test_reference_changes_leave_every_input_and_toy_prediction_identical(self):
        baseline = self.dfl()
        def toy(tracking):
            # Deliberately not a football model: exercise the downstream input boundary.
            return np.where(tracking.ball_missing, "X", "H").tobytes()
        for status, owner in (("0", "2"), ("1", "99"), ("unexpected", "")):
            self.raw.write_text(positions(status, owner))
            changed = self.dfl()
            reference = load_dfl_references(self.raw, changed)
            self.assert_tracking_equal(baseline, changed)
            self.assertEqual(toy(baseline), toy(changed))
            self.assertEqual(reference.raw_status[0], status)
            self.assertEqual(reference.raw_owner[0], owner)
            reference.alive[:] = -1
            reference.unknown[:] = True
            reference.frame[:] = 999
            self.assert_tracking_equal(baseline, changed)

    def test_unknown_reference_is_not_away_or_dead(self):
        self.raw.write_text(positions("unexpected", "99"))
        reference = load_dfl_references(self.raw, self.dfl())
        np.testing.assert_array_equal(reference.alive, [-1, -1, 0, -1])
        np.testing.assert_array_equal(reference.owner, ["X", "X", "X", "X"])
        np.testing.assert_array_equal(reference.unknown, [True, True, False, True])

    def test_dfl_duplicate_and_unknown_object_rejected(self):
        for raw in (positions().replace('<Frame N="10000" X="-25" Y="15"/>',
                                        '<Frame N="10000" X="-25" Y="15"/>' * 2),
                    positions().replace('PersonId="p1"', 'PersonId="absent"')):
            self.raw.write_text(raw)
            with self.assertRaises(ValueError):
                self.dfl()

    def test_atd_requires_wrapper_before_file_open(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "containment"):
                load_atd("absent", "absent", n_frames=4, pitch=(100, 60), blocks=[])

    def test_atd_rejects_shifted_duplicate_short_and_malformed_input(self):
        self.atd_fixture()
        raw = self.raw.read_text()
        for bad in (raw.replace("1:", "0:", 1), raw.replace("2:", "1:", 1),
                    raw.splitlines()[0] + "\n", raw.replace("0.25,0.25,0", "0.25", 1)):
            self.raw.write_text(bad)
            with self.assertRaises(ValueError):
                self.atd()

    def test_home_attacking_rotation_is_per_period(self):
        tracking = self.dfl()
        players, ball = tracking.home_attacking_coordinates()
        np.testing.assert_array_equal(players[0, 0], players[3, 0])
        np.testing.assert_array_equal(players[3, 0], [-25, 15])
        self.assertTrue(np.isnan(ball[3]).all())
        with self.assertRaisesRegex(ValueError, "unknown"):
            load_dfl(self.meta, self.raw).home_attacking_coordinates()

    def test_unknown_direction_is_explicit_and_not_guessed(self):
        tracking = load_dfl(self.meta, self.raw)
        self.assertTrue((tracking.home_direction == 0).all())
        with self.assertRaises(ValueError):
            load_dfl(self.meta, self.raw, directions={1: 2})

    def test_schema_rejects_extra_label_column_and_partial_nan(self):
        tracking = self.dfl()
        values = {f.name: getattr(tracking, f.name) for f in fields(Tracking)}
        with self.assertRaises(TypeError):
            Tracking(**values, owner=np.array(["H"] * 4))
        values["ball"] = tracking.ball.copy()
        values["ball"][0, 1] = np.nan
        with self.assertRaises(ValueError):
            Tracking(**values)
        values["owner"] = np.array(["H"] * 4)
        path = self.base / "bad.npz"
        np.savez(path, **values)
        with self.assertRaises(ValueError):
            Tracking.load(path)

    def test_predictor_import_boundary_and_label_access(self):
        # All feature/model code must stay in possession outside adapters/evaluation.
        forbidden = {"BallStatus", "BallPossession", "ball_state", "ball_owning_team",
                     "reference_mask", "raw_status", "raw_owner", "alive", "owner", "unknown"}
        def violations(source):
            problems = []
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""] + [a.name for a in node.names]
                else:
                    names = []
                if any(any(part in name.split('.') for part in ("evaluation", "adapters", "kloppy"))
                       for name in names):
                    problems.append("import")
                if isinstance(node, ast.Attribute) and node.attr in forbidden:
                    problems.append("attribute")
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in forbidden:
                    problems.append("column")
            return problems
        self.assertTrue(violations('from possession.evaluation import references'))
        self.assertTrue(violations('x = table["BallPossession"]'))
        self.assertTrue(violations('x = table.alive'))
        for path in (ROOT / "src/possession").rglob("*.py"):
            relative = path.relative_to(ROOT / "src/possession")
            if relative.parts[0] in ("adapters", "evaluation"):
                continue
            self.assertEqual(violations(path.read_text()), [], str(relative))

    def test_atd_synthetic_execution_through_real_wrapper(self):
        self.atd_fixture()
        code = ('from possession.adapters.atd import load_atd; '
                f't=load_atd({str(self.meta)!r},{str(self.raw)!r},n_frames=4,pitch=(100,60),'
                'blocks=[(1,0,3),(2,3,4)]); assert len(t.frame)==4; print("SYNTHETIC_PRIVATE")')
        result = subprocess.run(["sh", "scripts/run_target_local.sh", "configs/scaffold.yaml", "--",
                                 sys.executable, "-c", code], cwd=ROOT,
                                env={**os.environ, "PYTHON": sys.executable},
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "TARGET_RUN PASS exit_code=0\n")
        self.assertEqual(result.stderr, "")
