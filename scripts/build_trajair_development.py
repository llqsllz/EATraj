"""Build a causal, raw-observation TrajAir development cache.

This adapter deliberately does not consume TrajAir's interpolated processed files.
It creates 1 Hz, 10 s history / 20 s future windows from the released terminal
observations, with past-only hold support and a one-to-one future observation
match.  The packed arrays follow the project's eight-array cache contract; the
per-point target alignment offsets are kept in a sidecar because the current
PackedSceneDataset contract is intentionally closed over its array names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DEFAULT = ROOT / "data/public/trajair_source/terminal_raw_observations.parquet"
OUT_DEFAULT = ROOT / "data/trajectory_public_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def runway_transform() -> np.ndarray:
    """Reproduce the pinned TrajAir KBTP range/bearing frame.

    The release's official utility obtains the two runway bearings through a
    WGS84 inverse.  This local spherical inverse is sub-metre over the small
    camera-to-runway footprint and keeps the builder independent of optional
    geographiclib installations.
    """
    ref = (40.777888, -79.949864)
    r1 = (40.774548, -79.959237)
    r2 = (40.778630, -79.942803)

    def inv(a, b):
        lat1, lon1 = np.deg2rad(a[0]), np.deg2rad(a[1])
        lat2, lon2 = np.deg2rad(b[0]), np.deg2rad(b[1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        q = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        distance_km = 6371.0088 * 2 * np.arctan2(np.sqrt(q), np.sqrt(np.maximum(1 - q, 0)))
        bearing = np.arctan2(np.sin(dlon) * np.cos(lat2),
                             np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
        return distance_km, np.rad2deg(bearing)

    rr1, b1 = inv(ref, r1)
    rr2, b2 = inv(ref, r2)
    x1, y1 = rr1 * np.sin(np.deg2rad(b1)), rr1 * np.cos(np.deg2rad(b1))
    x2, y2 = rr2 * np.sin(np.deg2rad(b2)), rr2 * np.cos(np.deg2rad(b2))
    angle = -np.arctan2(y1 - y2, x1 - x2)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    p = -rot @ np.array([x1, y1])
    return -np.array([[np.cos(angle), -np.sin(angle), p[0]],
                      [np.sin(angle), np.cos(angle), p[1]], [0., 0., 1.]])


RUNWAY_R = runway_transform()


def local_xy_km(rng_km: np.ndarray, bearing_deg: np.ndarray) -> np.ndarray:
    x = rng_km * np.sin(np.deg2rad(bearing_deg))
    y = rng_km * np.cos(np.deg2rad(bearing_deg))
    return (RUNWAY_R @ np.vstack((x, y, np.ones_like(x))))[:2].T


def ls_velocity(times: np.ndarray, pos: np.ndarray, query: float, window: float = 5.0) -> tuple[np.ndarray, bool]:
    """Past-only least-squares velocity using actual observations only."""
    take = (times <= query + 1e-9) & (times >= query - window - 1e-9)
    t, p = times[take], pos[take]
    if len(t) < 2 or float(t[-1] - t[0]) < 0.4:
        return np.zeros(3), False
    tc = t - t[-1]
    den = float(np.sum((tc - tc.mean()) ** 2))
    if den <= 1e-10:
        return np.zeros(3), False
    v = np.sum((tc - tc.mean())[:, None] * (p - p.mean(axis=0)), axis=0) / den
    return v, bool(np.isfinite(v).all())


def endpoint_rotation(v: np.ndarray, valid: bool) -> tuple[np.ndarray, bool]:
    if not valid or np.linalg.norm(v[:2]) < 1.0:
        return np.eye(3), False
    theta = math.atan2(float(v[1]), float(v[0]))
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]), True


def actor_history(times: np.ndarray, pos: np.ndarray, grid: np.ndarray, max_age: float, match_tol: float):
    """Resolve a past-only held history without interpolation."""
    ix = np.searchsorted(times, grid, side="right") - 1
    present = ix >= 0
    safe = np.maximum(ix, 0)
    age = grid - times[safe]
    valid = present & (age >= -1e-9) & (age <= max_age + 1e-9)
    filled = pos[safe].copy()
    filled[~valid] = 0.
    observed = valid & (age <= match_tol + 1e-9)
    return filled, valid, observed, np.where(valid, age, np.nan), ix


def future_match(times: np.ndarray, pos: np.ndarray, grid: np.ndarray, tol: float):
    """Greedy monotone nearest matches; one source observation cannot repeat."""
    used: set[int] = set()
    out = np.zeros((len(grid), 3), dtype=np.float64)
    valid = np.zeros(len(grid), dtype=bool)
    offsets = np.full(len(grid), np.nan, dtype=np.float64)
    for k, q in enumerate(grid):
        j = int(np.searchsorted(times, q))
        # Search the complete tolerance interval, not only the immediate
        # predecessor/successor; irregular source intervals can skip a nearby
        # index while still containing a valid candidate.
        lo = int(np.searchsorted(times, q - tol - 1e-9, side="left"))
        hi = int(np.searchsorted(times, q + tol + 1e-9, side="right"))
        candidates = [x for x in range(lo, hi) if x not in used]
        if not candidates:
            continue
        best = min(candidates, key=lambda x: abs(float(times[x] - q)))
        delta = float(times[best] - q)
        if abs(delta) <= tol + 1e-9:
            used.add(best)
            out[k] = pos[best]
            offsets[k] = delta
            valid[k] = True
    return out, valid, offsets


def edge_tensor(p_shared: np.ndarray, v_shared: np.ndarray, v_valid: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    n, t, _ = p_shared.shape
    anchors = [0, (t - 1) // 2, t - 1]
    edges = np.zeros((n, n, 3, 13), dtype=np.float32)
    for ai, anchor in enumerate(anchors):
        rel_p = p_shared[None, :, anchor] - p_shared[:, None, anchor]
        rel_v = v_shared[None, :, anchor] - v_shared[:, None, anchor]
        vv = v_valid[:, anchor, None] & v_valid[None, :, anchor]
        rp = np.einsum("ijc,icd->ijd", rel_p, rotations)
        rv = np.einsum("ijc,icd->ijd", rel_v, rotations)
        dist = np.linalg.norm(rel_p, axis=-1)
        speed2 = np.sum(rel_v ** 2, axis=-1)
        tcpa_ok = vv & (speed2 >= 0.1 ** 2)
        dot = np.sum(rel_p * rel_v, axis=-1)
        tcpa = np.zeros_like(dist)
        tcpa[tcpa_ok] = np.clip(-dot[tcpa_ok] / speed2[tcpa_ok], 0, 15)
        dcpa = np.linalg.norm(rel_p + tcpa[..., None] * rel_v, axis=-1)
        closing = np.where(vv, -dot / np.maximum(dist, 1e-8), 0)
        rv[~vv] = 0
        edges[:, :, ai] = np.concatenate(
            (rp / 1000., rv / 100., dist[..., None] / 1000., closing[..., None] / 100.,
             tcpa[..., None] / 15., dcpa[..., None] / 1000., tcpa_ok[..., None],
             np.zeros((n, n, 1)), vv[..., None]), axis=-1)
    return edges


def build_window(actor_rows: list[dict], t0: float, history_seconds: int, future_seconds: int,
                 hold_age: float, tolerance: float):
    hgrid = t0 - np.arange(history_seconds, -1, -1, dtype=np.float64)
    actors = []
    for item in actor_rows:
        p, good, observed, age, _ = actor_history(item["times"], item["pos"], hgrid, hold_age, tolerance)
        # Actor membership is history-only. Endpoint support is tightened to the
        # declared <=0.5 s alignment tolerance, while older history uses ZOH.
        if bool(good.all()) and bool(observed[-1]):
            actors.append((item, p, good, observed, age))
    if len(actors) < 2:
        return None

    n, t = len(actors), len(hgrid)
    features, p_hist, v_hist, v_masks, rotations = [], [], [], [], []
    p0, v0_local, r_all = [], [], []
    for item, p, good, observed, age in actors:
        vv, vm = [], []
        for q in hgrid:
            v, ok = ls_velocity(item["times"], item["pos"], float(q))
            vv.append(v); vm.append(ok)
        vv, vm = np.asarray(vv), np.asarray(vm, dtype=bool)
        r, heading_ok = endpoint_rotation(vv[-1], bool(vm[-1]))
        p0.append(p[-1]); v0_local.append(vv[-1] @ r); r_all.append(r)
        local = (p - p[-1]) @ r
        rel_t = hgrid - t0
        source_dt = np.full(t, 1.0 / 10.0)
        f = np.column_stack((local / 1000., (vv @ r) / 100., np.ones(t), vm.astype(float),
                             observed.astype(float), np.full(t, float(heading_ok)),
                             np.nan_to_num(age, nan=hold_age) / 10., source_dt, rel_t / 10.))
        features.append(f)
        p_hist.append(p); v_hist.append(vv); v_masks.append(vm)
    p_hist = np.asarray(p_hist); v_hist = np.asarray(v_hist); v_masks = np.asarray(v_masks)
    rotations = np.asarray(r_all); p0 = np.asarray(p0); v0_local = np.asarray(v0_local)
    targets, valid, offsets = [], [], []
    fgrid = t0 + np.arange(1, future_seconds + 1, dtype=np.float64)
    for item, *_ in actors:
        target, mask, delta = future_match(item["times"], item["pos"], fgrid, tolerance)
        targets.append(target); valid.append(mask); offsets.append(delta)
    target = np.asarray(targets, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    target[~valid] = 0.
    return {
        "history_features": np.asarray(features, dtype=np.float32),
        "history_edges": edge_tensor(p_hist, v_hist, v_masks, rotations),
        "target": target,
        "future_valid": valid,
        "target_time_offsets": np.asarray(offsets, dtype=np.float32),
        "p0_shared": p0.astype(np.float32),
        "v0_local": v0_local.astype(np.float32),
        "R_local_to_shared": rotations.astype(np.float32),
        "actor_ids": [int(item["actor_id"]) for item, *_ in actors],
    }


def save_cache(out: Path, split: str, rows: list[dict], windows: list[dict], config: dict, index_hash: str):
    cache = out / "feature_cache" / split
    cache.mkdir(parents=True, exist_ok=True)
    n_actors = sum(len(w["actor_ids"]) for w in windows)
    n_pairs = sum(len(w["actor_ids"]) ** 2 for w in windows)
    arrays = {
        "history_features": np.concatenate([w["history_features"] for w in windows], axis=0),
        "history_edges": np.concatenate([w["history_edges"].reshape(-1, 3, 13) for w in windows], axis=0),
        "target": np.concatenate([w["target"] for w in windows], axis=0),
        "future_valid": np.concatenate([w["future_valid"] for w in windows], axis=0),
        "p0_shared": np.concatenate([w["p0_shared"] for w in windows], axis=0),
        "v0_local": np.concatenate([w["v0_local"] for w in windows], axis=0),
        "R_local_to_shared": np.concatenate([w["R_local_to_shared"] for w in windows], axis=0),
    }
    target_offsets = np.concatenate([w["target_time_offsets"] for w in windows], axis=0).astype(np.float32)
    for name, array in arrays.items():
        np.save(cache / f"{name}.npy", array.astype(bool if name == "future_valid" else np.float32), allow_pickle=False)
    np.save(cache / "target_time_offsets.npy", target_offsets, allow_pickle=False)
    cache_rows, actor_offset, edge_offset = [], 0, 0
    for row, w in zip(rows, windows):
        item = dict(row, actor_offset=actor_offset, actor_count=len(w["actor_ids"]), edge_offset=edge_offset)
        cache_rows.append(item); actor_offset += len(w["actor_ids"]); edge_offset += len(w["actor_ids"]) ** 2
    (cache / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in cache_rows), encoding="utf-8")
    manifest = {
        "status": "complete", "version": "trajectory_public_v1",
        "split": split, "sample_rate_hz": config["sample_rate_hz"],
        "scenes": len(rows), "actors": n_actors, "pairs": n_pairs,
        "arrays": {}, "rows_sha256": sha256(cache / "rows.jsonl"),
        "target_time_offsets": {"path": "target_time_offsets.npy", "shape": list(target_offsets.shape), "sha256": sha256(cache / "target_time_offsets.npy")},
        "dataset_config_sha256": sha256(out / "dataset_config.json"), "scene_index_sha256": index_hash,
        "protocol": "raw source Date+Time; 1 Hz query; past-only ZOH <=2 s; future nearest unused observation <=0.5 s",
    }
    expected = {
        "history_features": arrays["history_features"].shape,
        "history_edges": arrays["history_edges"].shape,
        "target": arrays["target"].shape,
        "future_valid": arrays["future_valid"].shape,
        "p0_shared": arrays["p0_shared"].shape,
        "v0_local": arrays["v0_local"].shape,
        "R_local_to_shared": arrays["R_local_to_shared"].shape,
    }
    for name, shape in expected.items():
        manifest["arrays"][name] = {"shape": list(shape), "dtype": "bool" if name == "future_valid" else "float32", "sha256": sha256(cache / f"{name}.npy")}
    write_json(cache / "manifest.json", manifest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--stride-train", type=int, default=5)
    ap.add_argument("--stride-validation", type=int, default=10)
    ap.add_argument("--gap-seconds", type=float, default=60.)
    ap.add_argument("--limit-endpoints", type=int, default=0, help="dry-run cap per split; 0 means all")
    args = ap.parse_args()
    raw, out = args.raw.resolve(), args.out.resolve()
    if not raw.exists():
        raise FileNotFoundError(raw)
    df = pd.read_parquet(raw).copy()
    required = {"ID", "Range", "Bearing", "Altitude", "source_time_seconds", "source_session", "Date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"raw artifact missing columns: {sorted(missing)}")
    df = df.sort_values(["source_session", "source_time_seconds", "ID", "source_row_id"], kind="stable")
    # Keep the source artifact's first row per actor/timestamp exactly as audited.
    df = df.drop_duplicates(["source_session", "ID", "source_time_seconds"], keep="first")
    xy = local_xy_km(df["Range"].to_numpy(float), df["Bearing"].to_numpy(float)) * 1000.
    df["x_m"], df["y_m"] = xy[:, 0], xy[:, 1]
    df["z_m"] = df["altitude_reported_m"].to_numpy(float) if "altitude_reported_m" in df else df["Altitude"].to_numpy(float) * 0.3048
    sessions = sorted(df["source_session"].unique().tolist())
    if len(sessions) < 2:
        raise ValueError("expected multiple raw sessions")
    train_sessions = sessions[: max(1, len(sessions) - 3)]
    validation_sessions = sessions[max(1, len(sessions) - 3): max(1, len(sessions) - 1)]
    reserved_sessions = sessions[-1:]
    split_map = {s: "train" for s in train_sessions} | {s: "validation" for s in validation_sessions} | {s: "reserved" for s in reserved_sessions}

    scene_rows = {"train": [], "validation": []}; windows = {"train": [], "validation": []}; episode_count = 0
    alignment = []
    for session in sessions:
        if split_map[session] == "reserved":
            continue
        part_session = df[df["source_session"] == session]
        # The release folder names are collection-session labels, while the
        # row Date field can cross midnight.  Date is therefore an explicit
        # boundary as well as the source-time gap boundary.
        for source_day, part in part_session.groupby("Date", sort=True):
            all_times = np.sort(part["source_time_seconds"].unique())
            cuts = np.flatnonzero(np.diff(all_times) > args.gap_seconds) + 1
            bounds = np.r_[0, cuts, len(all_times)]
            for bi, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
                start, end = float(all_times[lo]), float(all_times[hi - 1])
                seg = part[(part["source_time_seconds"] >= start) & (part["source_time_seconds"] <= end)]
                actor_rows = []
                for aid, actor in seg.groupby("ID", sort=True):
                    actor = actor.sort_values("source_time_seconds", kind="stable")
                    actor_rows.append({"actor_id": int(aid), "times": actor["source_time_seconds"].to_numpy(float),
                                       "pos": actor[["x_m", "y_m", "z_m"]].to_numpy(float)})
                if len(actor_rows) < 2:
                    continue
                split = split_map[session]; stride = args.stride_train if split == "train" else args.stride_validation
                first = int(math.ceil(start + 10.0)); last = int(math.floor(end - 1e-6))
                for t0_i in range(first, last + 1, stride):
                    if args.limit_endpoints and len(windows[split]) >= args.limit_endpoints:
                        break
                    w = build_window(actor_rows, float(t0_i), 10, 20, 2.0, 0.5)
                    if w is None:
                        continue
                    episode_id = f"{session}_{str(source_day).replace('/', '')}_ep{bi:04d}"
                    sid = f"trajair_{session}_{str(source_day).replace('/', '')}_ep{bi:04d}_{t0_i}"
                    row = {"scene_id": sid, "split": split, "replay_id": episode_id,
                           "endpoint_index": int(t0_i), "actor_indices": list(range(len(w["actor_ids"]))),
                           "actor_ids": w["actor_ids"], "source_session": session, "episode_id": episode_id,
                           "source_day": str(source_day), "endpoint_source_time_seconds": float(t0_i),
                           "history_seconds": 10, "forecast_seconds": 20,
                           "family_id": f"raw_{session}_{str(source_day).replace('/', '')}_ep{bi:04d}",
                           "family_status": "source-session/day episode; not an independent-flight claim"}
                    scene_rows[split].append(row); windows[split].append(w)
                episode_count += 1
    all_rows = scene_rows["train"] + scene_rows["validation"]
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "version": "trajectory_public_v1",
        "source_rate_hz": 10.0, "sample_rate_hz": 1.0,
        "raw_observation_rate_hz": "irregular; source Date+Time, not certified receiver clock",
        "history_seconds": 10.0, "max_forecast_seconds": 20.0,
        "history_hold_max_age_s": 2.0, "target_match_tolerance_s": 0.5,
        "endpoint_match_tolerance_s": 0.5, "target_observation_reuse": False,
        "interpolation": False, "source_artifact": str(raw.relative_to(ROOT)), "source_sha256": sha256(raw),
        "split_policy": {"train_sessions": train_sessions, "validation_sessions": validation_sessions,
                          "reserved_sessions_no_windows": reserved_sessions,
                          "split_unit": "source Date, source session, and contiguous source-time episode; no cross-boundary history/future"},
        "feature_schema": ["local_position_m/1000 x3", "causal_velocity_mps/100 x3", "history_support",
                            "velocity_valid", "matched_observation_mask", "heading_valid", "observation_age_s/10",
                            "nominal_query_dt_s/10", "relative_history_time_s/10"],
        "notes": ["Packed cache uses the project's canonical source_rate_hz=10 declaration solely for resolve_time_grid compatibility; the public adapter's query grid is 1 Hz.",
                   "Targets are nearest unused raw observations within +/-0.5 s and carry signed offsets in a sidecar.",
                   "No official processed train/test scene files are consumed; this is a raw-day development adapter, not an official benchmark split."]}
    write_json(out / "dataset_config.json", config)
    with (out / "scene_index.jsonl").open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    index_hash = sha256(out / "scene_index.jsonl")
    for split in ("train", "validation"):
        if windows[split]:
            save_cache(out, split, scene_rows[split], windows[split], config, index_hash)
    # A compact audit report is part of the public adapter, not a scientific test result.
    valid_offsets = np.concatenate([w["target_time_offsets"][w["target_time_offsets"] == w["target_time_offsets"]].ravel() for ws in windows.values() for w in ws]) if any(windows.values()) else np.array([])
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha256(raw),
              "episodes_with_windows": episode_count,
              "scene_counts": {k: len(v) for k, v in scene_rows.items()},
              "actor_counts": {k: int(sum(len(w["actor_ids"]) for w in v)) for k, v in windows.items()},
              "reserved_sessions": reserved_sessions,
              "target_alignment_offset_s": {"count": int(len(valid_offsets)), "q50": float(np.quantile(valid_offsets, .5)) if len(valid_offsets) else None,
                                               "q95": float(np.quantile(valid_offsets, .95)) if len(valid_offsets) else None,
                                               "q99": float(np.quantile(valid_offsets, .99)) if len(valid_offsets) else None,
                                               "max_abs": float(np.max(np.abs(valid_offsets))) if len(valid_offsets) else None},
              "future_valid_rate": {k: float(np.concatenate([w["future_valid"].ravel() for w in v]).mean()) if v else None for k, v in windows.items()},
              "scene_index_sha256": index_hash}
    write_json(out / "build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
