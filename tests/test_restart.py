"""Restart rule and reference-only metric tests on invented tracking."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from possession.tracking import Tracking
from possession.predictors.restart import RestartOwner, restart_team
from possession.predictors.deadball import DeadBall
from possession.evaluation.restarts import restart_counts, summarize


def fixture(n=100):
    players=np.zeros((n,2,2));players[:,1,0]=8
    return Tracking(np.arange(n),np.arange(n),np.ones(n,int),np.array(['h','a']),
                    np.array(['H','A']),players,np.zeros((n,2)),np.ones(n,int),np.array([105.,68.]))


class RestartTests(unittest.TestCase):
    def test_nearest_taker_with_repeated_evidence(self):
        self.assertEqual(restart_team(fixture(),40,100),'H')

    def test_missing_contested_or_far_falls_back(self):
        tr=fixture()
        self.assertEqual(restart_team(replace(tr,ball=np.full_like(tr.ball,np.nan)),40,100),'X')
        p=tr.players.copy();p[:,1]=p[:,0]
        self.assertEqual(restart_team(replace(tr,players=p),40,100),'X')
        self.assertEqual(restart_team(replace(tr,players=tr.players+20),40,100),'X')
        p[:,1]=np.nan
        self.assertEqual(restart_team(replace(tr,players=p),40,100),'X')

    def test_thrower_geometry_not_tag(self):
        tr=fixture();p=tr.players.copy();b=tr.ball.copy();b[:]=[0,34]
        p[:,0]=[1.5,34];p[:,1]=[0,32.6]
        tr=replace(tr,players=p,ball=b)
        # Distance alone is ambiguous; the on-line taker wins with the line cue.
        self.assertEqual(restart_team(tr,40,100,False),'X')
        self.assertEqual(restart_team(tr,40,100,True),'H')

    def test_period_isolation_and_short_run(self):
        tr=fixture();period=np.r_[np.ones(50,int),np.full(50,2)];tr=replace(tr,period=period)
        b=tr.ball.copy();b[:50]=100;p=tr.players.copy();p[:50]=100
        self.assertEqual(restart_team(tr,50,100),restart_team(replace(tr,ball=b,players=p),50,100))
        self.assertEqual(restart_team(tr,50,51),'X')

    def test_rule_only_changes_initial_spell_up_to_one_second(self):
        tr=fixture();model=RestartOwner(window=0);codes=np.array(['A']*40+['H']*60)
        out=model.owner_run(tr,codes,0,100)
        self.assertTrue(np.all(out[:25]=='H'));self.assertTrue(np.all(out[25:40]=='A'))
        self.assertTrue(np.all(out[40:]=='H'))
        codes[:10]='A';codes[10:]='H'
        self.assertTrue(np.all(model.owner_run(tr,codes,0,100)=='H'))

    def test_ambiguous_rule_preserves_baseline_and_dead_mask(self):
        tr=fixture();tr=replace(tr,ball=np.full_like(tr.ball,np.nan));codes=np.array(['A']*100)
        np.testing.assert_array_equal(RestartOwner().owner_run(tr,codes,0,100),DeadBall().owner_run(tr,codes,0,100))
        model=RestartOwner();model.threshold=.5
        model.dead_probability=lambda t:np.r_[np.ones(50),np.zeros(50)]
        out=model.predict(tr);self.assertTrue(np.all(out[:50]=='D'));self.assertTrue(np.all(out[50:]=='X'))

    def test_no_reference_input(self):
        with self.assertRaises(TypeError): RestartOwner().owner_run(fixture(),np.array(['H']*100),0,100,reference_mask=None)


class RestartMetricTests(unittest.TestCase):
    def test_exact_edges_counts_and_misses(self):
        gt=np.array(list('DDHHDAADHXDA'));pred=np.array(list('DDAHDDADXXDH'));period=np.ones(len(gt),int)
        c=restart_counts(gt,pred,period)
        self.assertEqual(c,{'n':4,'correct':0,'team_predictions':2,'pred_d':1,'pred_x':1})
        self.assertEqual(summarize(c)['team_coverage'],.5)

    def test_excludes_unknown_and_period_boundaries(self):
        c=restart_counts(np.array(list('DXHDA')),np.array(list('DXHDA')),np.array([1,1,1,1,2]))
        self.assertEqual(c['n'],0);self.assertIsNone(summarize(c)['accuracy'])

    def test_alignment_validation(self):
        with self.assertRaises(ValueError):restart_counts(['D','H'],['H'],[1,1])

if __name__=='__main__':unittest.main()
