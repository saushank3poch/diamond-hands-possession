"""FCB-12 synthetic control, gap, lag and isolation contracts."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from possession.tracking import Tracking
from possession.predictors.turnover import Turnover, confirmed_owner, control_evidence, displacement_velocity
from possession.predictors.deadball import DeadBall, live_owner


def tracking(n=150):
    p = np.zeros((n, 2, 2)); p[:, 1, 0] = 10
    return Tracking(np.arange(n), np.arange(n), np.ones(n, int), np.array(['h','a']),
                    np.array(['H','A']), p, p[:, 0].copy(), np.ones(n, int), np.array([105.,68.]))


class ConfirmationTests(unittest.TestCase):
    def test_touch_contested_and_loose_retain_owner(self):
        codes = np.array(list('HHHAXXXHAA'))
        out, events = confirmed_owner(codes, 3)
        self.assertEqual(''.join(out), 'H'*len(codes))
        self.assertEqual(events, [(0,2,'H')])

    def test_pending_opponent_is_reset_by_missing(self):
        out, events = confirmed_owner(np.array(list('HHHAAAXAA')), 4)
        self.assertTrue(np.all(out == 'X'))
        self.assertFalse(events)

    def test_dwell_size_and_backdate_lag(self):
        for n in (5,10,20):
            evidence=np.array(list('H'*25+'X'*4+'A'*25))
            back, events=confirmed_owner(evidence,n)
            delayed,_=confirmed_owner(evidence,n,False)
            self.assertEqual(events[-1],(29,29+n-1,'A'))
            self.assertEqual(np.flatnonzero(back=='A')[0],29)
            self.assertEqual(np.flatnonzero(delayed=='A')[0],29+n-1)

    def test_no_evidence_and_short_tail(self):
        self.assertTrue(np.all(confirmed_owner(np.array(list('XXXX')),3)[0]=='X'))
        self.assertEqual(''.join(confirmed_owner(np.array(list('HHHAA')),3)[0]),'HHHHH')

    def test_invalid_config(self):
        for n in (0,1,-1,2.5):
            with self.assertRaises(ValueError): confirmed_owner(['H'],n)
        for seconds in (0, float('nan'), float('inf')):
            with self.assertRaises(ValueError): Turnover(confirmation_s=seconds)
        with self.assertRaises(ValueError): confirmed_owner(['D'],2)


class EvidenceTests(unittest.TestCase):
    def test_control_requires_close_uncontested_observations(self):
        tr=tracking()
        self.assertTrue(np.all(control_evidence(tr,0,150)[12:-12]=='H'))
        for ball in (np.full_like(tr.ball,np.nan), np.full_like(tr.ball,30.)):
            self.assertTrue(np.all(control_evidence(replace(tr,ball=ball),0,150)=='X'))
        p=tr.players.copy();p[:,1,0]=.1
        self.assertTrue(np.all(control_evidence(replace(tr,players=p),0,150)=='X'))
        p[:,1]=np.nan
        self.assertTrue(np.all(control_evidence(replace(tr,players=p),0,150)=='X'))

    def test_gap_and_slot_jump_break_motion_support(self):
        xy=np.zeros((100,1,2));xy[50]=np.nan
        v=displacement_velocity(xy)
        self.assertTrue(np.isnan(v[38:63]).all())
        xy=np.zeros((100,1,2));xy[50:]=30
        self.assertTrue(np.isnan(displacement_velocity(xy)[40:60]).all())
        self.assertTrue(np.isnan(displacement_velocity(xy[:10])).all())

    def test_fast_pass_by_stationary_player_is_not_control(self):
        tr=tracking();ball=tr.ball.copy();ball[:,0]=(np.arange(150)-75)*.4
        self.assertTrue(np.all(control_evidence(replace(tr,ball=ball),0,150)=='X'))

    def test_live_run_and_period_isolation(self):
        tr=tracking(200)
        period=np.r_[np.ones(100,int),np.full(100,2)]
        tr=replace(tr,period=period)
        p=tr.players.copy();p[100:]*=20
        b=tr.ball.copy();b[100:]=30
        model=Turnover();model.threshold=.5
        model.dead_probability=lambda t:np.zeros(len(t.frame))
        a=model.predict(tr);c=model.predict(replace(tr,players=p,ball=b))
        np.testing.assert_array_equal(a[:100],c[:100]);self.assertTrue(np.all(c[100:]=='X'))
        model.dead_probability=lambda t:np.r_[np.zeros(70),np.ones(60),np.zeros(70)]
        out=model.predict(tr)
        self.assertTrue(np.all(out[70:130]=='D'))
        self.assertNotIn('D',set(out[:70]));self.assertNotIn('D',set(out[130:]))

    def test_parent_hook_preserves_e002_owner(self):
        tr=tracking();codes=np.array(['H']*50+['']*50+['A']*50)
        np.testing.assert_array_equal(DeadBall().owner_run(tr,codes,0,150),live_owner(codes,50))

    def test_unknown_live_run_is_x_not_dead(self):
        tr=tracking();tr=replace(tr,ball=np.full_like(tr.ball,np.nan))
        model=Turnover();model.threshold=.5;model.dead_probability=lambda t:np.zeros(len(t.frame))
        self.assertTrue(np.all(model.predict(tr)=='X'))

if __name__=='__main__': unittest.main()
