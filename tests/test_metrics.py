"""Hand-computable regression checks for the common trajectory scorer."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trajectory_prediction.metrics import (  # noqa: E402
    METRIC_NAMES,
    aggregate_scene_records,
    score_scene,
)


class ScoreSceneTests(unittest.TestCase):
    def test_k1_three_dimensional_euclidean_distance(self):
        target = np.zeros((2, 2, 3))
        prediction = np.broadcast_to([3.0, 4.0, 12.0], (1, 2, 2, 3))
        result = score_scene(prediction, target, np.ones((2, 2), bool))
        self.assertTrue(result["eligible"])
        for metric in METRIC_NAMES:
            self.assertEqual(result[metric], 13.0)
        self.assertEqual(result["top_score_mode_index"], 0)
        json.dumps(result, allow_nan=False)

    def test_joint_minimum_is_not_per_actor_oracle(self):
        target = np.zeros((2, 2, 3))
        prediction = np.zeros((2, 2, 2, 3))
        prediction[0, 1, :, 0] = 10
        prediction[1, 0, :, 0] = 10
        result = score_scene(prediction, target, np.ones((2, 2), bool))
        self.assertEqual(result["scene_minADE_m"], 5)
        self.assertEqual(result["scene_minFDE_m"], 5)
        self.assertEqual(result["marginal_minADE_m"], 0)
        self.assertEqual(result["marginal_minFDE_m"], 0)
        self.assertEqual(result["scene_ADE_mode_index"], 0)
        self.assertIsNone(result["top_score_ADE_m"])
        self.assertIsNone(result["top_score_mode_index"])

    def test_original_actor_modes_retain_marginals_when_scene_candidates_are_trimmed(self):
        """Per-actor K candidates must be scored before K-scene assembly."""
        target = np.zeros((2, 2, 3))
        actor_modes = np.zeros((2, 2, 2, 3))
        actor_modes[0, 1, :, 0] = 10.0
        actor_modes[1, 0, :, 0] = 10.0
        original = score_scene(actor_modes, target, np.ones((2, 2), bool))
        assembled_subset = score_scene(actor_modes[[0]], target, np.ones((2, 2), bool))
        self.assertEqual(original["marginal_minADE_m"], 0)
        self.assertEqual(original["marginal_minFDE_m"], 0)
        self.assertEqual(assembled_subset["marginal_minADE_m"], 5)
        self.assertEqual(assembled_subset["marginal_minFDE_m"], 5)

    def test_ade_and_fde_match_independently(self):
        prediction = np.zeros((2, 2, 3, 3))
        prediction[0, :, :, 0] = [0, 0, 6]
        prediction[1, :, :, 0] = [5, 5, 0]
        result = score_scene(
            prediction, np.zeros((2, 3, 3)), np.ones((2, 3), bool), [0.1, 0.9]
        )
        self.assertEqual(result["scene_ADE_mode_index"], 0)
        self.assertEqual(result["scene_FDE_mode_index"], 1)
        self.assertEqual(result["scene_minADE_m"], 2)
        self.assertEqual(result["scene_minFDE_m"], 0)
        self.assertEqual(result["scene_ADE_selected_FDE_m"], 6)
        self.assertAlmostEqual(result["top_score_ADE_m"], 10 / 3)
        self.assertEqual(result["top_score_FDE_m"], 0)

    def test_missing_exact_endpoint_excludes_actor_despite_coverage(self):
        valid = np.ones((3, 10), bool)
        valid[0, -1] = False
        prediction = np.zeros((1, 3, 10, 3))
        prediction[:, 0] = 99
        result = score_scene(prediction, np.zeros((3, 10, 3)), valid)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["n_actors_missing_endpoint"], 1)
        self.assertEqual(result["n_actors_below_coverage"], 0)
        self.assertEqual(result["scored_actor_indices"], [1, 2])
        self.assertEqual(result["scene_minFDE_m"], 0)

    def test_ineligible_scene_is_null_not_zero(self):
        valid = np.ones((2, 10), bool)
        valid[:, -1] = False
        result = score_scene(np.zeros((1, 2, 10, 3)), np.zeros((2, 10, 3)), valid)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["n_scored_actors"], 0)
        for metric in METRIC_NAMES:
            self.assertIsNone(result[metric])
        json.dumps(result, allow_nan=False)

    def test_nan_truth_outside_mask_is_safe(self):
        valid = np.ones((2, 10), bool)
        valid[:, 3] = False
        target = np.zeros((2, 10, 3))
        target[~valid] = np.nan
        prediction = np.zeros((1, 2, 10, 3))
        prediction[:, :, :, 0] = 1
        result = score_scene(prediction, target, valid)
        self.assertEqual(result["scene_minADE_m"], 1)
        self.assertEqual(result["n_scored_target_points"], 18)
        json.dumps(result, allow_nan=False)

    def test_actor_mean_does_not_weight_by_valid_point_count(self):
        valid = np.ones((2, 10), bool)
        valid[1, 0] = False
        prediction = np.zeros((1, 2, 10, 3))
        prediction[0, 0, :, 0] = 2
        prediction[0, 1, :, 0] = 4
        result = score_scene(prediction, np.zeros((2, 10, 3)), valid)
        self.assertEqual(result["scene_minADE_m"], 3)

    def test_actor_permutation_preserves_metrics(self):
        prediction = np.zeros((2, 3, 2, 3))
        prediction[0, :, :, 0] = [[1, 3], [4, 2], [6, 8]]
        prediction[1, :, :, 0] = [[3, 5], [1, 2], [4, 6]]
        target = np.zeros((3, 2, 3))
        valid = np.ones((3, 2), bool)
        original = score_scene(prediction, target, valid, [1, 2])
        permuted = score_scene(prediction[:, [2, 0, 1]], target[[2, 0, 1]], valid[[2, 0, 1]], [1, 2])
        for metric in METRIC_NAMES:
            self.assertAlmostEqual(original[metric], permuted[metric])

    def test_unscored_padding_may_be_nonfinite(self):
        prediction = np.zeros((1, 3, 2, 3))
        prediction[:, 2] = np.nan
        target = np.zeros((3, 2, 3))
        target[2] = np.nan
        valid = np.ones((3, 2), bool)
        valid[2] = False
        result = score_scene(prediction, target, valid)
        self.assertEqual(result["n_scored_actors"], 2)
        self.assertEqual(result["scene_minADE_m"], 0)

    def test_finite_validation_and_shape_errors(self):
        target = np.zeros((2, 2, 3))
        prediction = np.zeros((1, 2, 2, 3))
        valid = np.ones((2, 2), bool)
        bad_target = target.copy()
        bad_target[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "target contains nonfinite"):
            score_scene(prediction, bad_target, valid)
        bad_prediction = prediction.copy()
        bad_prediction[0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "prediction must be finite"):
            score_scene(bad_prediction, target, valid)
        bad_cases = (
            (prediction[0], target, valid),
            (prediction, target[:1], valid),
            (prediction, target, valid[:1]),
            (prediction, target, np.full((2, 2), 0.5)),
            (np.empty((0, 2, 2, 3)), target, valid),
            (np.empty((1, 2, 0, 3)), np.empty((2, 0, 3)), np.empty((2, 0), bool)),
        )
        for case in bad_cases:
            with self.subTest(shapes=[x.shape for x in case]), self.assertRaises(ValueError):
                score_scene(*case)
        for kwargs in ({"scores": [np.nan]}, {"scores": [1, 2]}, {"min_valid_fraction": 0}, {"min_valid_fraction": 1.1}, {"min_scored_actors": 0}, {"min_scored_actors": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                score_scene(prediction, target, valid, **kwargs)

    def test_nonfinite_prediction_at_unobserved_query_still_fails_for_scored_actor(self):
        valid = np.ones((2, 10), bool)
        valid[:, 0] = False
        prediction = np.zeros((1, 2, 10, 3))
        prediction[:, :, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "prediction must be finite"):
            score_scene(prediction, np.zeros((2, 10, 3)), valid)


class AggregationTests(unittest.TestCase):
    @staticmethod
    def record(replay, family, value):
        return {
            "replay_id": replay,
            "family_id": family,
            "eligible": value is not None,
            **{metric: value for metric in METRIC_NAMES},
        }

    def test_equal_hierarchy_with_unequal_windows_and_replays(self):
        records = [
            self.record("a1", "A", 0),
            self.record("a1", "A", 10),
            self.record("a2", "A", 25),
            self.record("b1", "B", 45),
            self.record("c1", "C", None),
        ]
        result = aggregate_scene_records(records)
        # A: mean(mean(0,10),25)=15; B:45. Equal family mean=30.
        for metric in METRIC_NAMES:
            self.assertEqual(result[metric], 30)
            self.assertEqual(result["metric_counts"][metric], {"windows": 4, "replays": 3, "families": 2})
        self.assertEqual(result["replay_records"][0]["scene_minADE_m"], 5)
        self.assertEqual(result["family_records"][0]["scene_minADE_m"], 15)
        self.assertEqual(result["counts"]["n_windows"], 5)
        self.assertEqual(result["counts"]["n_eligible_windows"], 4)
        self.assertEqual(result["counts"]["n_families"], 3)
        self.assertEqual(result["counts"]["n_eligible_families"], 2)
        json.dumps(result, allow_nan=False)

    def test_null_optional_metric_reports_its_own_counts(self):
        a = self.record("a", "A", 10)
        b = self.record("b", "B", 20)
        a["top_score_ADE_m"] = None
        result = aggregate_scene_records([a, b])
        self.assertEqual(result["scene_minADE_m"], 15)
        self.assertEqual(result["top_score_ADE_m"], 20)
        self.assertEqual(result["metric_counts"]["top_score_ADE_m"]["families"], 1)

    def test_empty_and_all_ineligible_have_null_metrics(self):
        for records in ([], [self.record("a", "A", None)]):
            result = aggregate_scene_records(records)
            self.assertFalse(result["eligible"])
            for metric in METRIC_NAMES:
                self.assertIsNone(result[metric])
            json.dumps(result, allow_nan=False)

    def test_inconsistent_replay_family_and_invalid_values_fail(self):
        with self.assertRaisesRegex(ValueError, "multiple families"):
            aggregate_scene_records([self.record("a", "A", 1), self.record("a", "B", 2)])
        invalid_records = [
            {**self.record("a", "A", 1), "family_id": None},
            {**self.record("a", "A", 1), "eligible": "yes"},
            {**self.record("a", "A", 1), "scene_minADE_m": np.nan},
            {**self.record("a", "A", 1), "scene_minADE_m": -1},
            {**self.record("a", "A", None), "scene_minADE_m": 0},
        ]
        for record in invalid_records:
            with self.subTest(record=record), self.assertRaises(ValueError):
                aggregate_scene_records([record])


if __name__ == "__main__":
    unittest.main()
