from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from trajectory_prediction.geometry import geodetic_to_enu,trailing_velocity,endpoint_rotation
from trajectory_prediction.data import history_actors,make_history,make_targets


class CausalDataTests(unittest.TestCase):
    def setUp(self):
        frames=np.arange(301)
        t=frames/10
        p=np.stack((np.column_stack((100*t,2*t,3*t)),np.column_stack((50*t+200,10*t+500,t+30))),axis=1)
        self.track={"frames":frames,"position":p,"valid":np.ones((301,2),bool),"segment":np.zeros((301,2),np.int32)}
        self.vel={"window_seconds":1,"min_samples":3,"min_span_seconds":.4}

    def history(self,track):
        return make_history(track,150,np.array([0,1]),100,2,self.vel)

    def test_velocity_causal_and_exact(self):
        t=np.arange(21)/10
        p=np.column_stack((3*t,2*t,-t))
        v,valid=trailing_velocity(t,p)
        self.assertFalse(valid[:4].any())
        np.testing.assert_allclose(v[valid],np.tile([3,2,-1],(valid.sum(),1)),atol=1e-10)
        p[11:]+=1e6
        other,_=trailing_velocity(t,p)
        np.testing.assert_array_equal(v[:11],other[:11])

    def test_future_perturbation_cannot_change_features(self):
        a=self.history(self.track)
        changed=copy.deepcopy(self.track)
        changed["position"][151:]=np.nan
        changed["valid"][151:]=False
        changed["segment"][151:]=7
        b=self.history(changed)
        for key in ("history_features","history_edges","history_velocity_shared"):
            np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(history_actors(self.track,150,100,2),history_actors(changed,150,100,2))

    def test_no_hidden_lookback(self):
        a=self.history(self.track)
        changed=copy.deepcopy(self.track)
        changed["position"][:50]=1e8
        b=self.history(changed)
        np.testing.assert_array_equal(a["history_features"],b["history_features"])
        self.assertFalse(a["history_velocity_valid"][:,:2].any())

    def test_permutation_and_shared_coordinate_recovery(self):
        a=self.history(self.track)
        b=make_history(self.track,150,np.array([1,0]),100,2,self.vel)
        np.testing.assert_allclose(a["history_features"][::-1],b["history_features"])
        np.testing.assert_allclose(a["history_edges"][::-1,::-1],b["history_edges"])
        ctx=a["history_context"]
        local=a["history_features"][:,:,:3]*1000.
        restored=np.einsum("ntc,ndc->ntd",local,ctx["R_local_to_shared"])+ctx["p0_shared"][:,None,:]
        np.testing.assert_allclose(restored,a["history_position_shared"],atol=.001)

    def test_reset_and_missing_future_do_not_select_actors(self):
        track=copy.deepcopy(self.track)
        track["segment"][201:,1]=1
        target,valid=make_targets(track,150,np.array([0,1]),200,2)
        self.assertTrue(valid[1,:25].all())
        self.assertFalse(valid[1,25:].any())
        self.assertFalse(valid[:,75:].any())
        np.testing.assert_array_equal(target[~valid],0)
        np.testing.assert_array_equal(history_actors(track,150,100,2),[0,1])

    def test_history_gap_excludes_only_affected_actor(self):
        self.track["valid"][51,1]=False
        # A source-rate gap on an unsampled frame must not be hidden by downsampling.
        np.testing.assert_array_equal(history_actors(self.track,150,100,2),[0])

    def test_enu_origin_and_axes(self):
        p=geodetic_to_enu(np.array([[0.,0.,0.],[0.,0.,10.],[.001,0.,0.],[0.,.001,0.]]),np.zeros(3))
        np.testing.assert_allclose(p[0],0,atol=1e-9)
        np.testing.assert_allclose(p[1],[0,0,10],atol=1e-9)
        self.assertGreater(p[2,0],100)
        self.assertGreater(p[3,1],100)

    def test_history_truncation_preserves_features(self):
        truncated={key:value[:151] for key,value in self.track.items()}
        np.testing.assert_array_equal(self.history(self.track)["history_features"],self.history(truncated)["history_features"])


if __name__=="__main__": unittest.main()
