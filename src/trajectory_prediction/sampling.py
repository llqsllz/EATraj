"""Deterministic, history-only motion stratification (not intent labels)."""
from __future__ import annotations

import hashlib
import numpy as np


def stable_int(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16], 16)


def segment_age_frames(track: dict) -> np.ndarray:
    """Past-only continuous-segment age, computed once instead of per window."""
    segment = track["segment"]
    index = np.arange(len(segment))[:, None]
    starts = np.ones_like(segment, dtype=bool)
    starts[1:] = segment[1:] != segment[:-1]
    last_start = np.maximum.accumulate(np.where(starts, index, 0), axis=0)
    return index - last_start


def motion_features(track: dict, endpoint: int, actors: np.ndarray,
                    history_frames: int, source_rate: float, rules: dict,
                    ages: np.ndarray | None = None) -> dict:
    """Use OLS velocities in history bins to suppress quantization spikes."""
    span = round(rules["velocity_bin_seconds"] * source_rate)
    if history_frames < 3 * span or history_frames % span:
        raise ValueError("History must contain at least three complete velocity bins")
    begin = endpoint - history_frames
    if begin < 0 or len(actors) == 0:
        raise ValueError("Missing history or actors")
    if not track["valid"][begin:endpoint + 1, actors].all():
        raise ValueError("Motion scoring requires valid history")
    if np.any(track["segment"][begin, actors] != track["segment"][endpoint, actors]):
        raise ValueError("Motion history crosses a segment reset")
    t = np.arange(span + 1, dtype=float) / source_rate
    t -= t.mean()
    denominator = float(t @ t)
    velocities = []
    for offset in range(begin, endpoint, span):
        p = track["position"][offset:offset + span + 1, actors]
        velocities.append(np.einsum("t,tnc->nc", t, p - p.mean(axis=0)) / denominator)
    v = np.stack(velocities)
    speed = np.linalg.norm(v, axis=-1)
    horizontal_speed = np.linalg.norm(v[..., :2], axis=-1)
    moving = horizontal_speed >= rules["minimum_horizontal_speed_mps_assumed"]
    cross = v[:-1, :, 0] * v[1:, :, 1] - v[:-1, :, 1] * v[1:, :, 0]
    dot = np.sum(v[:-1, :, :2] * v[1:, :, :2], axis=-1)
    turn = np.abs(np.degrees(np.arctan2(cross, dot))) / rules["velocity_bin_seconds"]
    turn = np.where(moving[:-1] & moving[1:], turn, 0.)
    relative_acc = np.abs(np.diff(speed, axis=0)) / (rules["velocity_bin_seconds"] * np.maximum(np.median(speed, axis=0), 30.))
    slope = np.where(moving, np.abs(np.degrees(np.arctan2(v[..., 2], horizontal_speed))), 0.)
    actor_turn = np.median(turn, axis=0)
    actor_acc = np.median(relative_acc, axis=0)
    vertical_acc = np.abs(np.diff(v[..., 2], axis=0)) / rules["velocity_bin_seconds"]
    actor_vertical_acc = np.median(vertical_acc, axis=0)
    actor_slope = np.median(slope, axis=0)
    # Sustained climb/descent angle is retained as an audit feature but does
    # not alone define a maneuver: long stable climb segments are low-value
    # for the requested recut. Transitions in vertical speed do count.
    actor_score = np.maximum.reduce([actor_turn / rules["turn_rate_deg_s"],
                                    actor_acc / rules["relative_speed_change_per_s"],
                                    actor_vertical_acc / rules["vertical_acceleration_mps2"]])
    score = float(np.max(actor_score))
    if ages is None:
        ages = segment_age_frames({"segment": track["segment"][:endpoint + 1]})
    early_fraction = float(np.mean(ages[endpoint, actors] / source_rate < rules["early_segment_seconds"]))
    early = early_fraction >= rules["early_actor_fraction"]
    activity = ("early_segment_proxy" if early else "maneuver" if score >= rules["maneuver_score"]
                else "moderate_motion" if score >= rules["moderate_score"] else "level_like")
    return {"causal_motion_score": score, "activity_class": activity,
            "early_segment": early, "early_actor_fraction": early_fraction,
            "max_median_turn_rate_deg_s": float(np.max(actor_turn)),
            "max_median_relative_speed_change_per_s": float(np.max(actor_acc)),
            "max_median_vertical_acceleration_mps2": float(np.max(actor_vertical_acc)),
            "max_median_flight_path_angle_deg": float(np.max(actor_slope)),
            "maneuver_actor_count": int(np.sum(actor_score >= rules["maneuver_score"]))}


def retain_candidate(features: dict, replay_id: str, endpoint_frame: int,
                     seed: int, rules: dict) -> bool:
    probability = rules["retention"][features["activity_class"]]
    value = stable_int("retain", seed, replay_id, endpoint_frame) / 2**64
    return value < probability
