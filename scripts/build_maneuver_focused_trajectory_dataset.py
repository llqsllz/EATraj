"""Recut existing 2v2/5v5 trajectories with history-only motion sampling."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trajectory_prediction.artifacts import finish_run, sha256_file, start_run, write_json
from trajectory_prediction.data import history_actors, make_targets, resolve_time_grid
from trajectory_prediction.sampling import motion_features, retain_candidate, segment_age_frames, stable_int
from build_trajectory_development import clean_track


def assign_splits(frame: pd.DataFrame, seed: int, policy: dict) -> pd.DataFrame:
    """Keep source/exact/prefix duplicate groups together in an exploratory split."""
    frame = frame.copy().sort_values("replay_id").reset_index(drop=True)
    parents = list(range(len(frame)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for key in ("source_sha256", "exact_trajectory_fingerprint", "prefix_fingerprint_candidate", "development_group_id"):
        seen: dict[str, int] = {}
        if key not in frame:
            continue
        for index, value in enumerate(frame[key].fillna("").astype(str)):
            if not value.strip():
                continue
            if value in seen:
                left, right = find(index), find(seen[value])
                parents[max(left, right)] = min(left, right)
            else:
                seen[value] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(frame)):
        groups.setdefault(find(index), []).append(index)
    strata: dict[str, list[tuple[str, list[int]]]] = {}
    for members in groups.values():
        group_id = "recut_" + hashlib.sha256("|".join(frame.loc[members, "replay_id"]).encode()).hexdigest()[:20]
        frame.loc[members, "split_group_id"] = group_id
        stratum = "+".join(sorted(frame.loc[members, "scenario"].unique()))
        strata.setdefault(stratum, []).append((group_id, members))
    frame["previous_trajectory_split"] = frame["split"]
    for stratum, entries in sorted(strata.items()):
        entries.sort(key=lambda entry: stable_int("split", seed, stratum, entry[0]))
        n = len(entries)
        test_count = max(1, round(n * policy["test_fraction"])) if n >= 3 else 0
        validation_count = max(1, round(n * policy["validation_fraction"])) if n >= 3 else 0
        for index, (_, members) in enumerate(entries):
            value = policy["test_label"] if index < test_count else "validation" if index < test_count + validation_count else "train"
            frame.loc[members, "split"] = value
    for key in ("split_group_id", "source_sha256", "exact_trajectory_fingerprint", "prefix_fingerprint_candidate"):
        if key in frame:
            overlaps = frame[frame[key] != ""].groupby(key)["split"].nunique()
            if bool((overlaps > 1).any()):
                raise ValueError(f"Cross-split duplicate key: {key}")
    return frame


def build(config: dict, output: Path) -> dict:
    grid = resolve_time_grid(config)
    manifest_path = ROOT / config["input_manifest"]
    frame = pd.read_csv(manifest_path).fillna("")
    frame = frame[frame.scenario.isin(["2v2", "5v5"])].copy()
    if frame.empty:
        raise ValueError("No 2v2/5v5 replays")
    frame = assign_splits(frame, config["seed"], config["split"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "tracks").mkdir()
    config["input_manifest_sha256"] = sha256_file(manifest_path)
    write_json(output / "dataset_config.json", config)
    frame.to_csv(output / "split_manifest.csv", index=False, encoding="utf-8-sig")
    snapshots: dict[str, dict] = {}
    replay_summaries: list[dict] = []
    candidate_counts, retained_counts, coverage_counts = Counter(), Counter(), Counter()
    rules = config["sampling"]
    with (output / "scene_index.jsonl").open("w", encoding="utf-8") as selected, \
            (output / "reference_scene_index.jsonl").open("w", encoding="utf-8") as reference, \
            (output / "candidate_audit.jsonl").open("w", encoding="utf-8") as audit:
        for number, row in enumerate(frame.itertuples(index=False), 1):
            source = ROOT / config["source_root"] / row.replay_id / "aircraft_states.parquet"
            source_hash = sha256_file(source)
            if source_hash != row.aircraft_parquet_sha256:
                raise ValueError(f"Source hash mismatch: {row.replay_id}")
            columns = ["frame_idx", "aircraft_id", "longitude_raw", "latitude_raw", "altitude_raw"]
            data = pd.read_parquet(source, columns=columns)
            origin = np.asarray([row.origin_lon, row.origin_lat, row.origin_alt], dtype=float)
            track, quality = clean_track(data, origin, config["quality"])
            if np.any(np.diff(track["frames"]) != 1):
                raise ValueError(f"Non-contiguous canonical frame grid: {row.replay_id}")
            track_path = output / "tracks" / f"{row.replay_id}.npz"
            np.savez_compressed(track_path, **track)
            snapshots[row.replay_id] = {"source_parquet_sha256": source_hash, "track_sha256": sha256_file(track_path)}
            ages = segment_age_frames(track)
            candidate_local, retained_local, actor_local = Counter(), Counter(), Counter()
            no_history = no_supervision = 0
            stride = round(config["strides_seconds"][row.split] * config["source_rate_hz"])
            for endpoint in range(grid["history_frames"], len(track["frames"]) - grid["future_frames"], stride):
                actors = history_actors(track, endpoint, grid["history_frames"], grid["frame_step"])
                if len(actors) < config["minimum_history_actors"]:
                    no_history += 1
                    continue
                features = motion_features(track, endpoint, actors, grid["history_frames"], config["source_rate_hz"], rules, ages)
                sample = {
                    "sample_id": hashlib.sha256(f"{config['version']}|{row.replay_id}|{int(track['frames'][endpoint])}".encode()).hexdigest()[:24],
                    "replay_id": row.replay_id, "family_id": row.split_group_id,
                    "family_status": "duplicate_control_proxy_only", "split": row.split,
                    "scenario": row.scenario, "endpoint_index": endpoint,
                    "endpoint_frame": int(track["frames"][endpoint]),
                    "history_start_frame": int(track["frames"][endpoint - grid["history_frames"]]),
                    "future_end_frame": int(track["frames"][endpoint + grid["future_frames"]]),
                    "actor_indices": actors.tolist(), "actor_ids": track["actor_ids"][actors].tolist(),
                    **features,
                }
                _, valid = make_targets(track, endpoint, actors, grid["future_frames"], grid["frame_step"])
                has_supervision = bool(valid.any())
                retained_by_history = retain_candidate(features, row.replay_id, sample["endpoint_frame"], config["seed"], rules)
                keep = retained_by_history and has_supervision
                eligible = {}
                for horizon in config["evaluation_horizons_seconds"]:
                    steps = round(horizon * config["sample_rate_hz"])
                    horizon_valid = valid[:, :steps]
                    eligible[str(horizon)] = int(np.sum((horizon_valid.mean(axis=1) >= 0.9) & horizon_valid[:, -1]))
                    if keep and eligible[str(horizon)] >= config["minimum_history_actors"]:
                        coverage_counts[f"{row.split}:{horizon}s"] += 1
                audit.write(json.dumps({**sample, "retained": keep, "retained_by_history": retained_by_history,
                                       "has_supervision": has_supervision, "eligible_actors_by_horizon": eligible}, allow_nan=False) + "\n")
                if not has_supervision:
                    no_supervision += 1
                    continue
                reference.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                activity = features["activity_class"]
                candidate_local[activity] += 1
                candidate_counts[f"{row.split}:{activity}"] += 1
                if keep:
                    selected.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                    retained_local[activity] += 1
                    retained_counts[f"{row.split}:{activity}"] += 1
                    actor_local[str(len(actors))] += 1
            replay_summaries.append({"replay_id": row.replay_id, "scenario": row.scenario, "split": row.split,
                                     "candidate_counts": dict(candidate_local), "retained_counts": dict(retained_local),
                                     "actor_count_distribution": dict(actor_local), "history_ineligible": no_history,
                                     "no_future_supervision": no_supervision, "quality": quality})
            if number % 25 == 0 or number == len(frame):
                print(f"replays={number}/{len(frame)} candidates={sum(candidate_counts.values())} retained={sum(retained_counts.values())}", flush=True)
    write_json(output / "source_checksums.json", snapshots)
    summary = {
        "status": "completed", "dataset_root": str(output),
        "replay_counts": frame.groupby("split").size().to_dict(),
        "scenario_counts": {str(split): group.groupby("scenario").size().to_dict() for split, group in frame.groupby("split")},
        "group_counts": frame.groupby("split")["split_group_id"].nunique().to_dict(),
        "candidate_window_count": sum(candidate_counts.values()), "retained_window_count": sum(retained_counts.values()),
        "candidate_counts": dict(candidate_counts), "retained_counts": dict(retained_counts),
        "scorable_windows_by_horizon": dict(coverage_counts), "replay_summaries": replay_summaries,
        "selection_uses_future_motion": False, "new_external_data": False, "duplicate_group_cross_split_count": 0,
        "source_hashes_verified": len(snapshots), "formal_test_enabled": False,
        "artifact_sha256": {name: sha256_file(output / name) for name in ("dataset_config.json", "split_manifest.csv", "scene_index.jsonl", "reference_scene_index.jsonl", "candidate_audit.jsonl", "source_checksums.json")},
    }
    write_json(output / "build_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/trajectory_maneuver_focus_v4.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    if output.exists():
        raise SystemExit(f"Refusing overwrite: {output}")
    resolve_time_grid(config)
    run_dir, record = start_run("maneuver-focused-recut", config, [Path(__file__), args.config.resolve(), Path(__file__).with_name("build_trajectory_development.py")])
    try:
        summary = build(config, output)
        write_json(run_dir / "build_summary.json", summary)
        finish_run(run_dir, record, "completed", dataset_root=str(output), summary_sha256=sha256_file(output / "build_summary.json"))
        print(json.dumps({key: summary[key] for key in ("replay_counts", "scenario_counts", "candidate_window_count", "retained_window_count", "retained_counts")}, ensure_ascii=False, indent=2))
    except Exception:
        finish_run(run_dir, record, "failed", error=traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
