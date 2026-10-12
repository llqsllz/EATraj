"""Common, NumPy-only trajectory scores in metres.

These functions consume targets only for evaluation, never for model inputs.
An actor qualifies when its truth coverage meets the threshold and the exact
last query has truth. A scene qualifies when enough actors qualify. Every
joint mode contains all actors; marginal minima are reported separately.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
import math
from numbers import Integral, Real
from typing import Any

import numpy as np


METRIC_NAMES = (
    "scene_minADE_m",
    "scene_minFDE_m",
    "scene_ADE_selected_FDE_m",
    "top_score_ADE_m",
    "top_score_FDE_m",
    "marginal_minADE_m",
    "marginal_minFDE_m",
)


def score_scene(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    scores: np.ndarray | None = None,
    min_valid_fraction: float = 0.9,
    min_scored_actors: int = 2,
) -> dict[str, Any]:
    """Score ``prediction[K,N,H,3]`` against ``target[N,H,3]``.

    Inputs must already use the same metre coordinate system and query times.
    ``valid[N,H]`` is boolean (or exact 0/1). Nonfinite truth under a true mask
    is an error, rather than a silent change to the shared evaluation cohort.
    Truth outside the mask is never used in arithmetic. All K trajectories,
    at all H queries, must be finite for every qualifying actor; padding and
    actors excluded by the truth-only rule may contain nonfinite predictions.

    ADE averages valid times within each actor, then actors equally. FDE uses
    H-1, never the last available observation. Scene ADE and FDE minima may
    select different modes. Ties select the first mode. Scores may be logits
    or any finite ranking scores, with larger preferred; they need not sum to
    one. Without scores, top-score metrics are null for K>1 and use mode 0 for
    K=1. Ineligible scenes have null metrics and remain in the counts.

    The returned dictionary contains only JSON-safe built-in scalar/list types.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid)
    if prediction.ndim != 4 or prediction.shape[-1] != 3:
        raise ValueError("prediction must have shape [K,N,H,3]")
    k_count, n_count, h_count, _ = prediction.shape
    if k_count < 1 or h_count < 1:
        raise ValueError("K and H must be positive")
    if target.shape != (n_count, h_count, 3):
        raise ValueError("target must have shape [N,H,3] matching prediction")
    if valid.shape != (n_count, h_count):
        raise ValueError("valid must have shape [N,H] matching prediction")
    if valid.dtype.kind not in "biuf" or not np.all((valid == 0) | (valid == 1)):
        raise ValueError("valid must contain booleans or exact numeric 0/1")
    valid = valid.astype(bool, copy=False)
    if (
        isinstance(min_valid_fraction, bool)
        or not isinstance(min_valid_fraction, Real)
        or not math.isfinite(min_valid_fraction)
        or not 0 < min_valid_fraction <= 1
    ):
        raise ValueError("min_valid_fraction must be finite in (0,1]")
    if (
        isinstance(min_scored_actors, bool)
        or not isinstance(min_scored_actors, Integral)
        or min_scored_actors < 1
    ):
        raise ValueError("min_scored_actors must be a positive integer")
    if not np.isfinite(target[valid]).all():
        raise ValueError("target contains nonfinite values where valid is true")

    top_mode: int | None = 0 if k_count == 1 else None
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (k_count,) or not np.isfinite(scores).all():
            raise ValueError("scores must be finite with shape [K]")
        top_mode = int(np.argmax(scores))

    valid_counts = valid.sum(axis=1)
    coverage_ok = valid_counts / h_count >= min_valid_fraction
    endpoint_ok = valid[:, -1]
    actor_indices = np.flatnonzero(coverage_ok & endpoint_ok)
    n_scored = len(actor_indices)
    if not np.isfinite(prediction[:, actor_indices]).all():
        raise ValueError("prediction must be finite for all queries of scored actors")

    eligible = n_scored >= min_scored_actors
    result: dict[str, Any] = {
        "eligible": bool(eligible),
        "ineligible_reason": None if eligible else "insufficient_scored_actors",
        "units": "metres",
        "n_modes": int(k_count),
        "n_actors": int(n_count),
        "n_queries": int(h_count),
        "n_scored_actors": int(n_scored),
        "n_actors_below_coverage": int((~coverage_ok).sum()),
        "n_actors_missing_endpoint": int((~endpoint_ok).sum()),
        "n_valid_target_points": int(valid.sum()),
        "n_scored_target_points": int(valid_counts[actor_indices].sum()),
        "scored_actor_indices": actor_indices.tolist(),
        "min_valid_fraction": float(min_valid_fraction),
        "min_scored_actors": int(min_scored_actors),
        "scene_ADE_mode_index": None,
        "scene_FDE_mode_index": None,
        "top_score_mode_index": top_mode if eligible else None,
        **dict.fromkeys(METRIC_NAMES),
    }
    if not eligible:
        return result

    actor_ade = np.empty((k_count, n_scored), dtype=np.float64)
    actor_fde = np.empty_like(actor_ade)
    # Index before subtraction: multiplying NaN by a zero mask is not safe.
    with np.errstate(over="raise", invalid="raise"):
        try:
            for column, actor in enumerate(actor_indices):
                truth = target[actor, valid[actor]]
                estimates = prediction[:, actor, valid[actor], :]
                error = np.linalg.norm(estimates - truth[None], axis=-1)
                actor_ade[:, column] = error.mean(axis=1)
                actor_fde[:, column] = np.linalg.norm(
                    prediction[:, actor, -1] - target[actor, -1], axis=-1
                )
            scene_ade = actor_ade.mean(axis=1)
            scene_fde = actor_fde.mean(axis=1)
        except FloatingPointError as exc:
            raise ValueError("trajectory magnitude overflowed metre error calculation") from exc

    ade_mode = int(np.argmin(scene_ade))
    fde_mode = int(np.argmin(scene_fde))
    result.update(
        scene_ADE_mode_index=ade_mode,
        scene_FDE_mode_index=fde_mode,
        scene_minADE_m=float(scene_ade[ade_mode]),
        scene_minFDE_m=float(scene_fde[fde_mode]),
        scene_ADE_selected_FDE_m=float(scene_fde[ade_mode]),
        top_score_ADE_m=None if top_mode is None else float(scene_ade[top_mode]),
        top_score_FDE_m=None if top_mode is None else float(scene_fde[top_mode]),
        marginal_minADE_m=float(actor_ade.min(axis=0).mean()),
        marginal_minFDE_m=float(actor_fde.min(axis=0).mean()),
    )
    return result


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def aggregate_scene_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate scalar scores equally from windows to replays to families.

    Each record requires ``replay_id``, ``family_id``, and boolean ``eligible``;
    add these IDs to the output of :func:`score_scene`. One replay must belong
    to exactly one family. Unknown family provenance must be resolved by the
    caller, not invented here. Missing/null metrics are omitted individually
    and their contributing counts are reported. Nonfinite metrics and nonnull
    metrics in an ineligible record are errors. Pass one model/protocol at a
    time; this function does not pool seeds or establish family independence.

    Top-level metric values are the family macro means. ``replay_records`` and
    ``family_records`` expose intermediate means for audits and later paired
    group statistics. Counts include excluded scenes/groups. Empty input or
    all-ineligible input returns null metrics, not zeros.
    """
    replay_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    replay_family: dict[str, str] = {}
    n_scenes = n_eligible = 0
    for record in records:
        replay = record.get("replay_id")
        family = record.get("family_id")
        if not isinstance(replay, str) or not replay.strip():
            raise ValueError("each record requires a nonempty string replay_id")
        if not isinstance(family, str) or not family.strip():
            raise ValueError("each record requires a nonempty string family_id")
        if not isinstance(record.get("eligible"), (bool, np.bool_)):
            raise ValueError("each record requires boolean eligible")
        if replay in replay_family and replay_family[replay] != family:
            raise ValueError(f"replay {replay!r} is assigned to multiple families")
        for metric in METRIC_NAMES:
            value = record.get(metric)
            if value is not None:
                if not record["eligible"]:
                    raise ValueError("ineligible records must have null metrics")
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, Real)
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"{metric} must be a finite nonnegative scalar or null")
        replay_family[replay] = family
        replay_groups[replay].append(record)
        n_scenes += 1
        n_eligible += int(record["eligible"])

    replay_records: list[dict[str, Any]] = []
    family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metric_counts = {
        metric: {"windows": 0, "replays": 0, "families": 0}
        for metric in METRIC_NAMES
    }
    for replay, group in sorted(replay_groups.items()):
        summary: dict[str, Any] = {
            "replay_id": replay,
            "family_id": replay_family[replay],
            "n_windows": len(group),
            "n_eligible_windows": sum(bool(r["eligible"]) for r in group),
            "metric_window_counts": {},
        }
        for metric in METRIC_NAMES:
            values = [float(r[metric]) for r in group if r.get(metric) is not None]
            summary[metric] = _mean(values)
            summary["metric_window_counts"][metric] = len(values)
            metric_counts[metric]["windows"] += len(values)
            metric_counts[metric]["replays"] += int(bool(values))
        replay_records.append(summary)
        family_groups[summary["family_id"]].append(summary)

    family_records: list[dict[str, Any]] = []
    for family, group in sorted(family_groups.items()):
        summary = {
            "family_id": family,
            "n_replays": len(group),
            "n_eligible_replays": sum(r["n_eligible_windows"] > 0 for r in group),
            "n_windows": sum(r["n_windows"] for r in group),
            "n_eligible_windows": sum(r["n_eligible_windows"] for r in group),
            "metric_replay_counts": {},
        }
        for metric in METRIC_NAMES:
            values = [r[metric] for r in group if r[metric] is not None]
            summary[metric] = _mean(values)
            summary["metric_replay_counts"][metric] = len(values)
            metric_counts[metric]["families"] += int(bool(values))
        family_records.append(summary)

    result = {
        "eligible": bool(n_eligible),
        "units": "metres",
        "aggregation": "equal_windows_within_replay_then_equal_replays_within_family_then_equal_families",
        "counts": {
            "n_windows": n_scenes,
            "n_eligible_windows": n_eligible,
            "n_ineligible_windows": n_scenes - n_eligible,
            "n_replays": len(replay_records),
            "n_eligible_replays": sum(r["n_eligible_windows"] > 0 for r in replay_records),
            "n_families": len(family_records),
            "n_eligible_families": sum(r["n_eligible_windows"] > 0 for r in family_records),
        },
        "metric_counts": metric_counts,
        "replay_records": replay_records,
        "family_records": family_records,
    }
    for metric in METRIC_NAMES:
        result[metric] = _mean([r[metric] for r in family_records if r[metric] is not None])
    return result
