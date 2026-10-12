"""Endpoint-head diagnostics with the common trajectory evaluation cohort.

These diagnostics never change a model input or select a checkpoint. Goals
are actor-local displacement vectors; trajectories and targets are shared
metre coordinates. All oracle mode selections use a *single scene mode*;
the separately named marginal oracle is not a jointly achievable result.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
import math
from numbers import Real
from typing import Any

import numpy as np

from .metrics import score_scene


HIT_RADII_M = (20, 50, 100)
ENDPOINT_METRIC_NAMES = (
    "goal_scene_minFDE_m",
    "goal_marginal_minFDE_m",
    "goal_ade_selected_error_m",
    "goal_top_error_m",
    "goal_traj_endpoint_gap_m",
    "goal_all_mode_traj_endpoint_gap_m",
    *(f"goal_{selection}_hit{radius}" for selection in ("top", "ade_selected")
      for radius in HIT_RADII_M),
)
QUANTILE_NAMES = ("goal_top_median_m", "goal_top_p90_m")


def score_endpoint_scene(
    goals_local: np.ndarray | None,
    trajectories: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    p0: np.ndarray,
    R: np.ndarray,
    scores: np.ndarray | None = None,
    min_valid_fraction: float = 0.9,
    min_scored_actors: int = 2,
) -> dict[str, Any]:
    """Diagnose shared Joint endpoint heads for one scene.

    Shapes: ``goals_local[K,N,3]``, ``trajectories[K,N,H,3]``,
    ``target[N,H,3]``, ``valid[N,H]``, ``p0[N,3]``, ``R[N,3,3]``.
    A local column vector maps to shared coordinates by ``R @ goal + p0``.
    Eligibility, ADE winner, ties, and missing-score behavior are delegated
    to :func:`score_scene`. Only the exact final query is evaluated.

    ``goal_traj_endpoint_gap_m`` uses the trajectory ADE-selected scene mode;
    ``goal_all_mode_traj_endpoint_gap_m`` averages every scene mode equally.
    Hit metrics are fractions of eligible actors within an inclusive 3-D
    Euclidean radius, not classification accuracy. Top-score median/p90 are
    actor-window diagnostics, not family-weighted main metrics. Their raw
    actor errors are retained for correct pooled diagnostic quantiles.

    Missing goals return null metrics, never zero. Nonfinite data for a
    scored actor is an error; masked/padded actors are sliced out *before*
    coordinate arithmetic. This function does not silently drop predictions.
    """
    base = score_scene(
        trajectories, target, valid, scores,
        min_valid_fraction=min_valid_fraction,
        min_scored_actors=min_scored_actors,
    )
    result = {
        key: base[key] for key in (
            "eligible", "ineligible_reason", "n_modes", "n_actors", "n_queries",
            "n_scored_actors", "scored_actor_indices", "min_valid_fraction",
            "min_scored_actors", "scene_ADE_mode_index", "top_score_mode_index",
        )
    }
    result.update(
        endpoint_available=goals_local is not None,
        endpoint_unavailable_reason="missing_goal_head" if goals_local is None else None,
        units={"distance": "metres", "hit": "fraction"},
        goal_scene_FDE_mode_index=None,
        goal_top_actor_errors_m=[],
        **dict.fromkeys((*ENDPOINT_METRIC_NAMES, *QUANTILE_NAMES)),
    )
    if goals_local is None:
        return result
    goals = np.asarray(goals_local, dtype=np.float64)
    p0 = np.asarray(p0, dtype=np.float64)
    rotation = np.asarray(R, dtype=np.float64)
    k_count, n_count = base["n_modes"], base["n_actors"]
    if goals.shape != (k_count, n_count, 3):
        raise ValueError("goals_local must have shape [K,N,3] matching trajectories")
    if p0.shape != (n_count, 3) or rotation.shape != (n_count, 3, 3):
        raise ValueError("p0 and R must have shapes [N,3] and [N,3,3]")
    if not base["eligible"]:
        return result
    actors = np.asarray(base["scored_actor_indices"], dtype=int)
    # Slicing first prevents NaN padding from contaminating even a valid mode.
    goals, origin, rotation = goals[:, actors], p0[actors], rotation[actors]
    if not all(np.isfinite(value).all() for value in (goals, origin, rotation)):
        raise ValueError("goals_local, p0 and R must be finite for scored actors")
    gram = np.einsum("nji,njk->nik", rotation, rotation)
    if not np.allclose(gram, np.eye(3), rtol=1e-4, atol=1e-4):
        raise ValueError("R must be an orthogonal local-to-shared rotation")
    if not np.allclose(np.linalg.det(rotation), 1, rtol=1e-4, atol=1e-4):
        raise ValueError("R must be a proper rotation")
    truth = np.asarray(target, dtype=np.float64)[actors, -1]
    trajectory_end = np.asarray(trajectories, dtype=np.float64)[:, actors, -1]
    with np.errstate(over="raise", invalid="raise"):
        try:
            shared = np.einsum("nij,knj->kni", rotation, goals) + origin[None]
            errors = np.linalg.norm(shared - truth[None], axis=-1)
            gaps = np.linalg.norm(shared - trajectory_end, axis=-1)
            scene_errors = errors.mean(axis=1)
        except FloatingPointError as exc:
            raise ValueError("endpoint magnitude overflowed metre error calculation") from exc
    ade_mode = base["scene_ADE_mode_index"]
    top_mode = base["top_score_mode_index"]
    goal_mode = int(np.argmin(scene_errors))
    # Ineligible scenes returned above; the explicit check guards API misuse.
    if ade_mode is None:
        raise ValueError("eligible scene must have a trajectory ADE-selected mode")
    result.update(
        goal_scene_FDE_mode_index=goal_mode,
        goal_scene_minFDE_m=float(scene_errors[goal_mode]),
        goal_marginal_minFDE_m=float(errors.min(axis=0).mean()),
        goal_ade_selected_error_m=float(scene_errors[ade_mode]),
        goal_traj_endpoint_gap_m=float(gaps[ade_mode].mean()),
        goal_all_mode_traj_endpoint_gap_m=float(gaps.mean()),
    )
    for radius in HIT_RADII_M:
        result[f"goal_ade_selected_hit{radius}"] = float((errors[ade_mode] <= radius).mean())
    if top_mode is not None:
        top_errors = errors[top_mode]
        result.update(
            goal_top_error_m=float(scene_errors[top_mode]),
            goal_top_actor_errors_m=top_errors.tolist(),
            goal_top_median_m=float(np.median(top_errors)),
            goal_top_p90_m=float(np.quantile(top_errors, 0.9)),
        )
        for radius in HIT_RADII_M:
            result[f"goal_top_hit{radius}"] = float((top_errors <= radius).mean())
    return result


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def aggregate_endpoint_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Use equal windows → replays → families, matching main trajectory scores.

    Requires ``replay_id``, ``family_id`` and boolean ``eligible`` on each
    record. Missing endpoint outputs remain in counts with null metrics.
    Quantiles are explicitly separated as pooled actor-window diagnostics;
    overlapping windows do not supply independent statistical replicates.
    """
    replay_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    replay_family: dict[str, str] = {}
    actor_errors: list[float] = []
    n_eligible = n_available = n_windows = 0
    for record in records:
        replay, family = record.get("replay_id"), record.get("family_id")
        if not isinstance(replay, str) or not replay.strip():
            raise ValueError("each record requires a nonempty string replay_id")
        if not isinstance(family, str) or not family.strip():
            raise ValueError("each record requires a nonempty string family_id")
        if not isinstance(record.get("eligible"), (bool, np.bool_)):
            raise ValueError("each record requires boolean eligible")
        if replay in replay_family and replay_family[replay] != family:
            raise ValueError(f"replay {replay!r} is assigned to multiple families")
        for metric in ENDPOINT_METRIC_NAMES:
            value = record.get(metric)
            if value is None:
                continue
            if not record["eligible"]:
                raise ValueError("ineligible records must have null metrics")
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{metric} must be a finite nonnegative scalar or null")
            if "_hit" in metric and value > 1:
                raise ValueError("hit metrics must be fractions in [0,1]")
        errors = record.get("goal_top_actor_errors_m", [])
        if errors is None:
            errors = []
        if errors and (not record["eligible"] or record.get("goal_top_error_m") is None):
            raise ValueError("actor errors require an eligible, top-scored endpoint record")
        for error in errors:
            if (isinstance(error, (bool, np.bool_)) or not isinstance(error, Real)
                    or not math.isfinite(error) or error < 0):
                raise ValueError("actor errors must be finite nonnegative scalars")
            actor_errors.append(float(error))
        replay_family[replay] = family
        replay_groups[replay].append(record)
        n_windows += 1
        n_eligible += int(record["eligible"])
        n_available += int(bool(record.get("endpoint_available", False)) and record["eligible"])

    counts = {metric: {"windows": 0, "replays": 0, "families": 0}
              for metric in ENDPOINT_METRIC_NAMES}
    replay_records = []
    family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for replay, group in sorted(replay_groups.items()):
        summary = dict(replay_id=replay, family_id=replay_family[replay],
                       n_windows=len(group),
                       n_eligible_windows=sum(bool(r["eligible"]) for r in group),
                       metric_window_counts={})
        for metric in ENDPOINT_METRIC_NAMES:
            values = [float(r[metric]) for r in group if r.get(metric) is not None]
            summary[metric] = _mean(values)
            summary["metric_window_counts"][metric] = len(values)
            counts[metric]["windows"] += len(values)
            counts[metric]["replays"] += int(bool(values))
        replay_records.append(summary)
        family_groups[summary["family_id"]].append(summary)
    family_records = []
    for family, group in sorted(family_groups.items()):
        summary = dict(family_id=family, n_replays=len(group),
                       n_eligible_replays=sum(r["n_eligible_windows"] > 0 for r in group),
                       n_windows=sum(r["n_windows"] for r in group),
                       n_eligible_windows=sum(r["n_eligible_windows"] for r in group),
                       metric_replay_counts={})
        for metric in ENDPOINT_METRIC_NAMES:
            values = [r[metric] for r in group if r[metric] is not None]
            summary[metric] = _mean(values)
            summary["metric_replay_counts"][metric] = len(values)
            counts[metric]["families"] += int(bool(values))
        family_records.append(summary)
    return {
        "eligible": bool(n_eligible),
        "units": {"distance": "metres", "hit": "fraction"},
        "aggregation": "equal_windows_within_replay_then_equal_replays_within_family_then_equal_families",
        "counts": {"n_windows": n_windows, "n_eligible_windows": n_eligible,
                   "n_ineligible_windows": n_windows - n_eligible,
                   "n_eligible_endpoint_windows": n_available,
                   "n_replays": len(replay_records), "n_families": len(family_records),
                   "n_eligible_replays": sum(r["n_eligible_windows"] > 0 for r in replay_records),
                   "n_eligible_families": sum(r["n_eligible_windows"] > 0 for r in family_records)},
        "metric_counts": counts,
        "replay_records": replay_records,
        "family_records": family_records,
        **{metric: _mean([r[metric] for r in family_records if r[metric] is not None])
           for metric in ENDPOINT_METRIC_NAMES},
        "actor_window_diagnostics": {
            "aggregation": "pooled_scored_actor_windows_not_family_weighted",
            "n_top_scored_actor_windows": len(actor_errors),
            "goal_top_median_m": float(np.median(actor_errors)) if actor_errors else None,
            "goal_top_p90_m": float(np.quantile(actor_errors, 0.9)) if actor_errors else None,
        },
    }
