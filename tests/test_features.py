"""Analytic synthetic fixtures; no target assets or reference files are read."""
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from possession.tracking import Tracking
from possession.features import Config, compute
from possession.feature_cache import cache_features


def fixture(n=10):
    t = np.arange(n) / 25
    ball = np.column_stack([t, np.zeros(n)])
    players = np.stack([ball + [-2, 0], ball + [2, 0], ball + [5, -3], ball + [5, 3]], axis=1)
    return Tracking(np.arange(n), np.arange(n), np.ones(n, dtype=int),
                    np.array(['h1','h2','a1','a2']), np.array(['H','H','A','A']),
                    players, ball, np.ones(n,dtype=int), np.array([100.,60.]))


class FeatureTests(unittest.TestCase):
    def test_analytic_motion_and_shape(self):
        x = compute(fixture())
        np.testing.assert_allclose(x['ball_vx'][1:], 1)
        np.testing.assert_allclose(x['ball_vy'][1:], 0)
        np.testing.assert_allclose(x['ball_ax'][2:], 0, atol=1e-12)
        np.testing.assert_allclose(x['H_nearest_distance'], 2)
        np.testing.assert_allclose(x['H_comotion_speed_difference'][1:], 0, atol=1e-12)
        np.testing.assert_allclose(x['H_depth'], 4)
        np.testing.assert_allclose(x['H_width'], 0)
        np.testing.assert_allclose(x['A_width'], 6)
        np.testing.assert_allclose(x['H_compactness'], 2)
        np.testing.assert_allclose(x['H_mean_speed'][1:], 1)
        self.assertTrue(np.isnan(x['ball_vx'][0]))
        self.assertTrue(np.isnan(x['H_mean_acceleration'][1]))

    def test_short_gap_interpolates_but_preserves_provenance(self):
        tr = fixture()
        ball = tr.ball.copy();ball[3:5] = np.nan
        x = compute(replace(tr, ball=ball), Config(max_ball_gap_frames=2))
        np.testing.assert_allclose(x['ball_x'], tr.ball[:,0])
        self.assertEqual(x['ball_interpolated'].sum(),2)
        np.testing.assert_array_equal(x['ball_quality'][3:5],[.5,.5])
        self.assertTrue(x['ball_missing'][3:5].all())
        np.testing.assert_allclose(x['ball_observation_age_s'][3:5],[.04,.08])
        long = compute(replace(tr,ball=ball),Config(max_ball_gap_frames=1))
        self.assertTrue(np.isnan(long['ball_x'][3:5]).all())
        self.assertTrue(np.isnan(long['ball_vx'][5]))

    def test_edges_and_periods_are_not_bridged(self):
        tr = fixture()
        for periods in ([1]*5+[2]*5, [1]*4+[0]*2+[2]*4):
            ball=tr.ball.copy();ball[4:6]=np.nan;ball[0]=np.nan;ball[-1]=np.nan
            x=compute(replace(tr,ball=ball,period=np.array(periods)))
            self.assertFalse(x['ball_interpolated'].any())
            self.assertTrue(np.isnan(x['ball_x'][4:6]).all())
            self.assertTrue(np.isnan(x['H_mean_speed'][5]))
            self.assertTrue(np.isnan(x['ball_observation_age_s'][5]))

    def test_period_jump_has_no_velocity_or_acceleration_spike(self):
        tr=fixture();ball=tr.ball.copy();players=tr.players.copy()
        ball[5:]+=20;players[5:]+=20
        x=compute(replace(tr,ball=ball,players=players,period=np.array([1]*5+[2]*5)))
        self.assertTrue(np.isnan(x['ball_speed'][5]))
        self.assertTrue(np.isnan(x['H_mean_speed'][5]))
        self.assertTrue(np.isnan(x['H_mean_acceleration'][6]))

    def test_slot_gap_and_teleport_motion_are_rejected(self):
        tr=fixture();p=tr.players.copy();p[3:5,:2]=np.nan;p[7:,:2,0]+=10
        x=compute(replace(tr,players=p))
        self.assertTrue(np.isnan(x['H_mean_speed'][5]))
        self.assertTrue(np.isnan(x['H_mean_speed'][7]))
        self.assertEqual(x['H_motion_count'][7],0)
        self.assertTrue(np.isfinite(x['H_centroid_x'][7]))

    def test_missing_team_and_ball_never_become_zero_distances(self):
        tr=fixture();p=tr.players.copy();p[:,2:]=np.nan
        x=compute(replace(tr,players=p,ball=np.full(tr.ball.shape,np.nan)))
        for name in ('A_nearest_distance','A_centroid_x','A_width','ball_speed','ball_out_of_bounds','ball_exiting'):
            self.assertTrue(np.isnan(x[name]).all(),name)
        self.assertFalse(x['ball_position_valid'].any())
        self.assertTrue((x['A_observation_quality']==0).all())

    def test_bounds_exiting_and_unknown_direction(self):
        tr=fixture();ball=tr.ball.copy();ball[:,0]=49.9+np.arange(10)*.04
        x=compute(replace(tr,ball=ball,home_direction=np.zeros(10,dtype=int)))
        self.assertEqual(x['ball_out_of_bounds'][1],0)
        self.assertEqual(x['ball_exiting'][1],1)
        self.assertEqual(x['ball_out_of_bounds'][3],1)
        self.assertEqual(x['ball_exiting'][3],0)
        self.assertTrue(np.isnan(x['H_attacking_line']).all())

    def test_inputs_and_reference_independence(self):
        tr=fixture();before={f.name:getattr(tr,f.name).copy() for f in fields(tr)}
        baseline=compute(tr)
        for labels,mask in [('HHHH',False),('AAAA',True),('XXXX',True)]:
            # Separate evaluator objects never enter the function or cache identity.
            refs={'labels':labels,'mask':mask}
            current=compute(tr)
            for key in baseline: np.testing.assert_array_equal(baseline[key],current[key])
        for key in before:np.testing.assert_array_equal(before[key],getattr(tr,key))
        with self.assertRaises(TypeError):compute(tr,reference_mask=np.ones(10))

    def test_config_validation(self):
        for kwargs in ({'max_ball_gap_frames':-1},{'max_ball_gap_frames':1.5},
                       {'max_player_speed_mps':0},{'exit_horizon_s':float('nan')}):
            with self.assertRaises(ValueError):Config(**kwargs)


class CacheTests(unittest.TestCase):
    def test_roundtrip_and_cache_invalidation(self):
        tr=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cache.parquet'
            table,hit=cache_features(tr,path);self.assertFalse(hit)
            again,hit=cache_features(tr,path);self.assertTrue(hit)
            self.assertTrue(table.equals(again,check_metadata=True))
            _,hit=cache_features(tr,path,Config(max_ball_gap_frames=0));self.assertFalse(hit)
            ball=tr.ball.copy();ball[0,0]+=1
            _,hit=cache_features(replace(tr,ball=ball),path);self.assertFalse(hit)
            changed=pq.read_table(path).replace_schema_metadata({b'possession_features':b'old'})
            pq.write_table(changed,path)
            _,hit=cache_features(tr,path);self.assertFalse(hit)
            identity=json.loads(pq.read_table(path).schema.metadata[b'possession_features'])
            self.assertFalse(identity['smart_tagging'])

    def test_cache_rejects_undeclared_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cache.parquet'
            table,_=cache_features(fixture(),path)
            table=table.append_column('forbidden_reference', table['frame'])
            pq.write_table(table,path)
            with self.assertRaisesRegex(ValueError, 'columns'):
                cache_features(fixture(),path)

    def test_real_wrapper_synthetic_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'input.npz';fixture().save(source)
            code=('import os;from pathlib import Path;from possession.tracking import Tracking;'
                  'from possession.feature_cache import cache_features;'
                  f't=Tracking.load({str(source)!r});'
                  'cache_features(t,Path(os.environ["POSSESSION_RUN_DIR"])/"synthetic.parquet");'
                  'print("SYNTHETIC_CACHE_DETAIL")')
            result=subprocess.run(['sh','scripts/run_target_local.sh','configs/scaffold.yaml','--',
                                   sys.executable,'-c',code],cwd=ROOT,
                                  env={**os.environ,'PYTHON':sys.executable},capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(result.stdout,'TARGET_RUN PASS exit_code=0\n')
            self.assertEqual(result.stderr,'')
