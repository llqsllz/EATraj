from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"))
from cache_trajectory_features import cache_split
from trajectory_prediction.data import SceneDataset
from trajectory_prediction.packed_data import PackedSceneDataset
from trajectory_prediction.metrics import score_scene


class ReferenceCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/"tracks").mkdir()
        config={"source_rate_hz":10,"sample_rate_hz":5,"history_seconds":10,
                "max_forecast_seconds":20,
                "velocity":{"window_seconds":1,"min_samples":3,"min_span_seconds":.4}}
        (self.root/"dataset_config.json").write_text(json.dumps(config),encoding="utf-8")
        frames=np.arange(501)
        t=frames/10
        pos=np.stack((np.column_stack((100*t,2*t,3*t)),
                      np.column_stack((50*t+200,10*t+500,t+30))),axis=1)
        np.savez_compressed(self.root/"tracks/r.npz",frames=frames,position=pos,
                            valid=np.ones((501,2),bool),segment=np.zeros((501,2),np.int32))
        self.row={"sample_id":"ref","replay_id":"r","family_id":"g","split":"validation",
                  "endpoint_index":150,"actor_indices":[0,1]}
        focus=dict(self.row,sample_id="focus",endpoint_index=160)
        (self.root/"scene_index.jsonl").write_text(json.dumps(focus)+"\n",encoding="utf-8")
        (self.root/"reference_scene_index.jsonl").write_text(json.dumps(self.row)+"\n",encoding="utf-8")

    def build_reference(self):
        cache_split(self.root,"validation",index_name="reference_scene_index.jsonl",cache_name="reference_validation")

    def test_reference_roundtrip_and_common_scorer(self):
        self.build_reference()
        raw=SceneDataset(self.root,"validation",index_name="reference_scene_index.jsonl")
        packed=PackedSceneDataset(self.root,"validation",20,index_name="reference_scene_index.jsonl",
                                  cache_split="reference_validation")
        self.assertEqual(packed.rows[0]["sample_id"],"ref")
        for key in ("history_features","history_edges","target","future_valid"):
            np.testing.assert_array_equal(raw[0][key],packed[0][key])
        sample=packed[0]
        score=score_scene(sample["target"][None],sample["target"],sample["future_valid"])
        self.assertTrue(score["eligible"])
        self.assertEqual(score["scene_minADE_m"],0)
        self.assertFalse((self.root/"feature_cache/validation").exists())

    def test_reference_cache_cannot_be_used_as_focused_cache(self):
        self.build_reference()
        with self.assertRaisesRegex(ValueError,"source index mismatch"):
            PackedSceneDataset(self.root,"validation",20,cache_split="reference_validation")
        with self.assertRaisesRegex(ValueError,"separate cache name"):
            cache_split(self.root,"validation",index_name="reference_scene_index.jsonl")

    def test_reference_index_hash_and_no_overwrite(self):
        self.build_reference()
        with self.assertRaisesRegex(ValueError,"Refusing to overwrite"):
            self.build_reference()
        path=self.root/"reference_scene_index.jsonl"
        path.write_text(path.read_text(encoding="utf-8")+"\n",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"scene_index_sha256 mismatch"):
            PackedSceneDataset(self.root,"validation",20,index_name=path.name,
                               cache_split="reference_validation")

    def test_legacy_default_cache_remains_supported(self):
        cache_split(self.root,"validation")
        path=self.root/"feature_cache/validation/manifest.json"
        manifest=json.loads(path.read_text(encoding="utf-8"))
        manifest.pop("index_name")
        manifest.pop("cache_split")
        path.write_text(json.dumps(manifest),encoding="utf-8")
        ds=PackedSceneDataset(self.root,"validation",20)
        self.assertEqual(ds.rows[0]["sample_id"],"focus")


if __name__=="__main__": unittest.main()
