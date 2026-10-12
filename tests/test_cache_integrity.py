"""Regression checks for stale caches and exact forecast query times."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trajectory_prediction.artifacts import sha256_file
from trajectory_prediction.data import SceneDataset
from trajectory_prediction.packed_data import PackedSceneDataset


class CacheIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "feature_cache" / "validation"
        self.cache.mkdir(parents=True)
        (self.root / "tracks").mkdir()
        self.config = {"source_rate_hz": 10, "sample_rate_hz": 5, "history_seconds": 1,
                       "max_forecast_seconds": 2,
                       "velocity": {"window_seconds": 1, "min_samples": 3, "min_span_seconds": .4}}
        self.write_json(self.root / "dataset_config.json", self.config)
        frames = np.arange(51)
        time = frames / 10.
        position = np.stack((np.column_stack((3*time, time, time)),
                             np.column_stack((2*time+5, time+10, time+1))), axis=1)
        np.savez(self.root / "tracks" / "r0.npz", frames=frames, position=position,
                 valid=np.ones((51, 2), bool), segment=np.zeros((51, 2), np.int32))
        self.source_rows = [{"sample_id": f"s{i}", "replay_id": "r0", "family_id": "f0",
                             "split": "validation", "scenario": "2v2", "endpoint_index": end,
                             "endpoint_frame": end, "actor_indices": [0, 1]}
                            for i, end in enumerate((10, 30))]
        self.write_rows(self.root / "scene_index.jsonl", self.source_rows)
        dataset = SceneDataset(self.root, "validation")
        samples = [dataset[i] for i in range(len(dataset))]
        self.rows = [dict(row, actor_offset=2*i, actor_count=2, edge_offset=4*i)
                     for i, row in enumerate(self.source_rows)]
        self.write_rows(self.cache / "rows.jsonl", self.rows)
        arrays = {name: np.concatenate([sample[name] for sample in samples])
                  for name in ("history_features", "target", "future_valid")}
        arrays["history_edges"] = np.concatenate([sample["history_edges"].reshape(4, 3, 13) for sample in samples])
        for name in ("p0_shared", "v0_local", "R_local_to_shared"):
            arrays[name] = np.concatenate([sample["history_context"][name] for sample in samples])
        for name, array in arrays.items():
            np.save(self.cache / (name + ".npy"), array)
        self.manifest = {"status": "complete", "split": "validation", "scenes": 2,
                         "actors": 4, "pairs": 8, "sample_rate_hz": 5,
                         "dataset_config_sha256": sha256_file(self.root / "dataset_config.json"),
                         "scene_index_sha256": sha256_file(self.root / "scene_index.jsonl"),
                         "arrays": {name: {"shape": list(array.shape), "sha256": sha256_file(self.cache / (name + ".npy"))}
                                    for name, array in arrays.items()}}
        self.write_json(self.cache / "manifest.json", self.manifest)

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def write_rows(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def close_maps(self, dataset):
        # Windows cannot remove temporary mmap files while handles remain open.
        for array in dataset.arrays.values():
            array._mmap.close()

    def test_valid_cache_equals_live_targets_at_exact_horizon(self):
        packed = PackedSceneDataset(self.root, "validation", 1.2)
        try:
            live = SceneDataset(self.root, "validation", 1.2)
            self.assertTrue(packed.integrity["array_hashes_verified"])
            for i in range(len(live)):
                np.testing.assert_array_equal(packed[i]["forecast_times"], live[i]["forecast_times"])
                np.testing.assert_array_equal(packed[i]["target"], live[i]["target"])
                np.testing.assert_array_equal(packed[i]["future_valid"], live[i]["future_valid"])
                np.testing.assert_array_equal(packed[i]["history_features"], live[i]["history_features"])
        finally:
            self.close_maps(packed)

    def test_modified_protocol_is_rejected(self):
        self.config["max_forecast_seconds"] = 1
        self.write_json(self.root / "dataset_config.json", self.config)
        with self.assertRaisesRegex(ValueError, "dataset_config_sha256"):
            PackedSceneDataset(self.root, "validation", 1)

    def test_modified_source_scene_index_is_rejected(self):
        self.source_rows[0]["family_id"] = "different_family"
        self.write_rows(self.root / "scene_index.jsonl", self.source_rows)
        with self.assertRaisesRegex(ValueError, "scene_index_sha256"):
            PackedSceneDataset(self.root, "validation", 1)

    def test_modified_cached_metadata_is_rejected(self):
        self.rows[0]["family_id"] = "different_family"
        self.write_rows(self.cache / "rows.jsonl", self.rows)
        with self.assertRaisesRegex(ValueError, "row metadata"):
            PackedSceneDataset(self.root, "validation", 1)

    def test_shifted_actor_offsets_are_rejected(self):
        self.rows[0]["actor_offset"] = 1
        self.write_rows(self.cache / "rows.jsonl", self.rows)
        with self.assertRaisesRegex(ValueError, "offsets"):
            PackedSceneDataset(self.root, "validation", 1)

    def test_corrupted_array_rejected_by_default_and_opt_out_is_explicit(self):
        path = self.cache / "history_features.npy"
        array = np.load(path)
        array[0, 0, 0] += 10
        np.save(path, array)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            PackedSceneDataset(self.root, "validation", 1)
        packed = PackedSceneDataset(self.root, "validation", 1, verify_array_hashes=False)
        try:
            self.assertFalse(packed.integrity["array_hashes_verified"])
            self.assertTrue(packed.integrity["protocol_and_index_hashes_verified"])
        finally:
            self.close_maps(packed)

    def test_dtype_validation_still_applies_with_hash_opt_out(self):
        path = self.cache / "history_features.npy"
        np.save(path, np.load(path).astype(np.float64))
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            PackedSceneDataset(self.root, "validation", 1, verify_array_hashes=False)

    def test_repeated_getitem_does_not_repeat_hash_scan(self):
        packed = PackedSceneDataset(self.root, "validation", 1)
        try:
            with patch("trajectory_prediction.packed_data.sha256_file", side_effect=AssertionError("rehash")):
                self.assertEqual(packed[0]["target"].shape, (2, 5, 3))
                self.assertEqual(packed[1]["target"].shape, (2, 5, 3))
        finally:
            self.close_maps(packed)

    def test_both_loaders_reject_non_grid_or_invalid_horizons(self):
        for horizon in (1.1, .1, 0, -1, float("nan"), float("inf"), 2.2):
            for loader in (SceneDataset, PackedSceneDataset):
                with self.subTest(horizon=horizon, loader=loader.__name__):
                    with self.assertRaises(ValueError):
                        loader(self.root, "validation", horizon)

    def test_protocol_grid_changes_require_supported_exact_alignment(self):
        for key, value in (("history_seconds", 1.1), ("sample_rate_hz", 3), ("source_rate_hz", 20)):
            config = dict(self.config, **{key: value})
            self.write_json(self.root / "dataset_config.json", config)
            self.manifest["dataset_config_sha256"] = sha256_file(self.root / "dataset_config.json")
            self.write_json(self.cache / "manifest.json", self.manifest)
            for loader in (SceneDataset, PackedSceneDataset):
                with self.subTest(key=key, loader=loader.__name__):
                    with self.assertRaises(ValueError):
                        loader(self.root, "validation", 1)


if __name__ == "__main__":
    unittest.main()
