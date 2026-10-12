"""Acquire an official bounded TrajAir subset and inspect its simultaneous tracks.

Downloads are source data artifacts, never executable dependencies. The official
7days1 release is a subset of 111_days, not a replacement for the full benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import requests
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/public/trajair_source"
ARTICLE_URL = "https://widgets.figshare.com/public/articles/14866251"
FILES_URL = ARTICLE_URL + "/files"
COMMIT = "5a5bc751dabba09a668ca9acb6b05ae4112c5b3d"


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_reference_details() -> None:
    reference_dir = DEST / "official_preprocess_reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    preview_url = "https://ndownloader.figstatic.com/files/28625922/preview/28625922/content.json"
    preview = fetch(preview_url).json()
    save_json(reference_dir / "official_README_preview.json", preview)
    (reference_dir / "official_README_preview.txt").write_text("\n".join(preview) + "\n", encoding="utf-8")
    commit = fetch("https://api.github.com/repos/cyoung/stratux/commits/master").json()["sha"]
    receiver_url = f"https://raw.githubusercontent.com/cyoung/stratux/{commit}/main/traffic.go"
    target = reference_dir / "stratux_traffic.go"
    target.write_bytes(fetch(receiver_url).content)
    save_json(reference_dir / "supplemental_reference_manifest.json", {
        "official_readme_preview_url": preview_url,
        "readme_original_file_download_verified": False,
        "readme_preview_note": "Official text preview recovered; not claimed byte-identical to README.txt download",
        "receiver_source_url": receiver_url, "receiver_commit": commit,
        "receiver_source_sha256": hash_file(target),
        "receiver_scope": "Current upstream reference for Age semantics, not the certified dataset-era receiver firmware/logger",
    })


def fetch(url: str) -> requests.Response:
    for attempt in range(3):
        response = requests.get(url, timeout=(15, 60))
        if response.status_code not in {403, 502, 503, 504} or attempt == 2:
            response.raise_for_status()
            return response
    raise AssertionError("Unreachable")


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def acquire() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(DEST).free < 2 * 1024**3:
        raise RuntimeError("At least 2 GiB free space is required for this bounded acquisition")
    article = fetch(ARTICLE_URL).json()
    files = fetch(FILES_URL).json()
    if article["version"] != 1 or not article["isPublic"] or article["downloadDisabled"]:
        raise RuntimeError("Expected public downloadable TrajAir v1")
    save_json(DEST / "official_article_metadata.json", article)
    save_json(DEST / "official_file_metadata.json", files)
    selection = [f for f in files if f["name"] == "7days1.zip"]
    if len(selection) != 1 or sum(f["size"] for f in selection) > 50_000_000:
        raise RuntimeError("Expected the bounded official 7days1 selection")
    records = []
    for item in selection:
        target = DEST / "downloads" / item["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        expected_md5 = item["md5"].lower().zfill(32)
        if not target.exists():
            partial = target.with_suffix(target.suffix + ".partial")
            total = 0
            with requests.get(item["downloadUrl"], timeout=(15, 60), stream=True) as response:
                response.raise_for_status()
                with partial.open("wb") as out:
                    for chunk in response.iter_content(1024 * 1024):
                        total += len(chunk)
                        if total > item["size"]:
                            raise RuntimeError("Download exceeds official byte count")
                        out.write(chunk)
            if total != item["size"] or hash_file(partial, "md5") != expected_md5:
                raise RuntimeError("Official archive byte count/MD5 mismatch")
            partial.rename(target)
        if target.stat().st_size != item["size"] or hash_file(target, "md5") != expected_md5:
            raise RuntimeError(f"Existing downloaded file failed verification: {target}")
        records.append({"name": item["name"], "file_id": item["id"], "url": item["downloadUrl"],
                        "bytes": item["size"], "md5": expected_md5, "sha256": hash_file(target),
                        "relative_path": target.relative_to(ROOT).as_posix()})
        print(f"Verified {item['name']}: {item['size']} bytes", flush=True)
    archive = DEST / "downloads/7days1.zip"
    extract_root = DEST / "release_v1_7days1"
    members = []
    with zipfile.ZipFile(archive) as package:
        if sum(i.file_size for i in package.infolist()) > 2 * 1024**3:
            raise RuntimeError("Uncompressed archive exceeds 2 GiB bound")
        for item in package.infolist():
            parts = PurePosixPath(item.filename).parts
            if PurePosixPath(item.filename).is_absolute() or ".." in parts or "\\" in item.filename:
                raise RuntimeError("Unsafe archive member")
            target = (extract_root / item.filename).resolve()
            if not target.is_relative_to(extract_root.resolve()):
                raise RuntimeError("Unsafe archive destination")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.open(item) as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
            elif target.stat().st_size != item.file_size:
                raise RuntimeError("Existing extraction has unexpected size")
            if not item.is_dir():
                members.append({"path": target.relative_to(ROOT).as_posix(), "bytes": item.file_size,
                                "zip_crc": f"{item.CRC:08x}"})
    reference_dir = DEST / "official_preprocess_reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    references = []
    for name in ["README.md", "LICENSE", "adsb_preprocess/process.py", "adsb_preprocess/utils.py"]:
        url = f"https://raw.githubusercontent.com/castacks/trajairnet/{COMMIT}/{name}"
        target = reference_dir / name.replace("/", "__")
        content = fetch(url).content
        if target.exists() and target.read_bytes() != content:
            raise RuntimeError("Pinned reference content changed")
        if not target.exists():
            target.write_bytes(content)
        references.append({"path": target.relative_to(ROOT).as_posix(), "url": url, "sha256": hash_file(target)})
    save_json(DEST / "source_acquisition.json", {
        "acquired_utc": datetime.now(timezone.utc).isoformat(),
        "official_project_url": "https://theairlab.org/trajair/",
        "metadata_url": ARTICLE_URL, "file_metadata_url": FILES_URL,
        "dataset_doi": article["doi"], "version": article["version"],
        "selection": "Complete official 7days1.zip; deterministic first named week for adapter development",
        "full_111_days_benchmark_acquired": False,
        "data_license": "CC BY 4.0, as stated by official TrajAir project page",
        "official_preprocessing_code_commit": COMMIT,
        "downloads": records, "extracted_files": members, "references": references,
        "access_note": "KiltHub and api.figshare.com returned 403; official embedded widget public metadata and 7days1 figstatic download succeeded. The separate README.txt download returned 403 and is not claimed acquired.",
    })
    print(json.dumps({"files": len(members), "uncompressed_bytes": sum(m["bytes"] for m in members),
                      "root": str(extract_root)}, ensure_ascii=False), flush=True)


def causal_availability(record_seconds: np.ndarray, position_age: np.ndarray,
                        grid: np.ndarray, max_age: float) -> np.ndarray:
    """Availability only, no interpolation and no use of future receiver records."""
    index = np.searchsorted(record_seconds, grid, side="right") - 1
    present = index >= 0
    clipped = np.maximum(index, 0)
    effective_age = position_age[clipped] + grid - record_seconds[clipped]
    good = present & (position_age[clipped] >= 0) & (effective_age <= max_age)
    if np.any(record_seconds[clipped[good]] > grid[good]):
        raise AssertionError("Future record entered availability")
    return good


def inspect_data() -> None:
    release = DEST / "release_v1_7days1/7days1"
    raw_files = sorted((release / "raw_data").glob("*/1.csv"))
    if len(raw_files) != 7:
        raise RuntimeError("Expected all seven official raw source files")
    # Availability must be prefix invariant when future records are appended.
    test_grid = np.arange(4, dtype=float)
    np.testing.assert_array_equal(
        causal_availability(np.array([0., 2.]), np.array([0., 0.]), test_grid, 1.5),
        causal_availability(np.array([0., 2., 10.]), np.zeros(3), test_grid, 1.5))
    raw_results = []
    all_observations = []
    for path in raw_files:
        frame = pd.read_csv(path, usecols=["ID", "Date", "Time", "Lat", "Lon", "Altitude", "Age", "Range", "Bearing"])
        timestamp = pd.to_datetime(frame["Date"] + " " + frame["Time"],
                                   format="%m/%d/%Y %H:%M:%S.%f", errors="coerce")
        frame["timestamp_logged"] = timestamp
        # Explicitly retain the source Date/Time scale; assigning UTC requires
        # the official collection-time convention, not the folder's local date.
        # pandas 3 may infer microsecond datetimes; normalize explicitly before
        # converting integer storage to seconds.
        frame["record_seconds"] = timestamp.to_numpy(dtype="datetime64[ns]").astype("int64") / 1e9
        numeric_columns = ["ID", "Lat", "Lon", "Altitude", "Age", "Range", "Bearing"]
        finite = np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all(axis=1)
        geographic = frame["Lat"].between(-90, 90) & frame["Lon"].between(-180, 180)
        geographic &= ~((frame["Lat"] == 0) & (frame["Lon"] == 0))
        valid = finite & geographic & timestamp.notna()
        footprint = valid & frame["Range"].between(0, 5, inclusive="left") & (frame["Altitude"] < 6000)
        selected = frame.loc[footprint].copy()
        selected = selected.sort_values(["ID", "record_seconds"], kind="stable")
        duplicate_records = int(selected.duplicated(["ID", "record_seconds"]).sum())
        duplicate_rows = selected[selected.duplicated(["ID", "record_seconds"], keep=False)]
        conflicting = duplicate_rows.groupby(["ID", "record_seconds"])[["Lat", "Lon", "Altitude"]].nunique()
        conflict_count = int((conflicting > 1).any(axis=1).sum())
        if conflict_count:
            raise RuntimeError("Conflicting positions at a shared aircraft/source timestamp need explicit handling")
        # Repeated timestamps have identical positions but increasing Age. Keep
        # the first file record, never its later stale copy. Date+Time therefore
        # cannot be certified as each CSV row's polling/arrival time.
        selected = selected.drop_duplicates(["ID", "record_seconds"], keep="first")
        selected["source_session"] = path.parent.name
        selected["source_row_id"] = selected.index
        selected["source_relative_path"] = path.relative_to(ROOT).as_posix()
        selected["altitude_reported_m"] = selected["Altitude"] * 0.3048
        all_observations.append(selected)
        start = float(np.floor(selected.record_seconds.min()))
        end = float(np.ceil(selected.record_seconds.max()))
        grid = np.arange(start, end + 1., 1.)
        actor_groups = list(selected.groupby("ID", sort=True))
        deltas = []
        repeated = 0
        pair_count = 0
        for _, actor in actor_groups:
            dt = np.diff(actor.record_seconds.to_numpy())
            deltas.extend(dt[dt > 0].tolist())
            pos = actor[["Lat", "Lon", "Altitude"]].to_numpy()
            repeated += int(np.all(pos[1:] == pos[:-1], axis=1).sum())
            pair_count += max(0, len(actor) - 1)
        age_counts = {}
        for age_limit in [2.0, 5.0, 10.0]:
            matrix = np.stack([causal_availability(actor.record_seconds.to_numpy(),
                              actor.Age.to_numpy(), grid, age_limit) for _, actor in actor_groups])
            counts = matrix.sum(axis=0)
            # At 1 Hz a 10 s inclusive history contains 11 grid points.
            complete_history = np.stack([np.convolve(row.astype(np.int16), np.ones(11, dtype=np.int16), mode="valid") == 11
                                         for row in matrix])
            history_counts = complete_history.sum(axis=0)
            age_counts[str(age_limit)] = {
                "any_actor_grid_seconds": int((counts >= 1).sum()),
                "multi_actor_grid_seconds": int((counts >= 2).sum()),
                "max_simultaneous_fresh_actors": int(counts.max()),
                "actor_grid_seconds_histogram": {str(int(k)): int(v) for k, v in zip(*np.unique(counts[counts > 0], return_counts=True))},
                "complete_10s_history_multi_actor_endpoints": int((history_counts >= 2).sum()),
                "max_actors_with_complete_10s_history": int(history_counts.max()),
            }
        raw_results.append({
            "source_session": path.parent.name, "source_path": path.relative_to(ROOT).as_posix(),
            "sha256": hash_file(path), "raw_rows": len(frame), "valid_position_rows": int(valid.sum()),
            "terminal_footprint_rows": int(footprint.sum()), "unique_actor_count_in_footprint": len(actor_groups),
            "duplicate_actor_timestamp_rows_in_footprint": duplicate_records,
            "duplicate_actor_timestamp_conflicting_positions": conflict_count,
            "negative_age_rows_in_footprint": int((selected.Age < 0).sum()),
            "logged_dates": sorted(selected.Date.unique().tolist()),
            "first_logged_timestamp": str(selected.timestamp_logged.min()),
            "last_logged_timestamp": str(selected.timestamp_logged.max()),
            "record_interval_s_quantiles": {str(q): float(np.quantile(deltas, q)) for q in [.01, .5, .9, .99]},
            "age_s_quantiles": {str(q): float(selected.Age.quantile(q)) for q in [.01, .5, .9, .99]},
            "adjacent_identical_position_count": repeated,
            "adjacent_same_actor_record_pairs": pair_count,
            "availability_by_max_position_age_s": age_counts,
        })
        print(f"Audited {path.parent.name}: {len(frame)} rows, {len(actor_groups)} terminal aircraft", flush=True)
    observations = pd.concat(all_observations, ignore_index=True).rename(columns={"record_seconds": "source_time_seconds"})
    observations.to_parquet(DEST / "terminal_raw_observations.parquet", index=False)
    processed_results = []
    for path in sorted((release / "processed_data").glob("*/*.txt")):
        data = np.loadtxt(path, ndmin=2)
        if data.shape[1] != 7 or not np.isfinite(data).all():
            raise RuntimeError(f"Unexpected processed record layout: {path}")
        pairs = pd.DataFrame(data[:, :2], columns=["frame", "actor"]).drop_duplicates()
        counts = pairs.groupby("frame").actor.nunique()
        processed_results.append({"scene_file": path.relative_to(ROOT).as_posix(), "official_split": path.parent.name,
                                  "rows": len(data), "unique_actors": int(pairs.actor.nunique()),
                                  "active_seconds": len(counts), "multi_actor_seconds": int((counts >= 2).sum()),
                                  "max_simultaneous_actors": int(counts.max()),
                                  "first_frame": float(data[:, 0].min()), "last_frame": float(data[:, 0].max())})
    processed = pd.DataFrame(processed_results)
    processed.to_csv(DEST / "processed_scene_feasibility.csv", index=False)
    summary = {
        "status": "official_subset_acquired_and_actual_simultaneous_feasibility_verified",
        "scope": "Complete official 7days1 subset, not full 111_days benchmark; no forecast model training",
        "raw_fields": ["ID", "Time", "Date", "Altitude", "Speed", "Heading", "Lat", "Lon", "Age", "Range", "Bearing", "Tail", "Metar"],
        "footprint_rule": "finite valid geodetic position, 0<=Range<5 km, Altitude<6000 ft; per-row filtering only",
        "alignment_rule": "1 Hz absolute source Date+Time grid, first file row per actor/timestamp, last retained source record at or before query, diagnostic freshness=first raw Age+query_time-source_time; reject negative raw Age; no interpolation",
        "history_rule": "10-second inclusive history, 11 grid points, all available for actor; actors selected from history alone",
        "status_warning": "Age limits and 10-second history are feasibility probes, not frozen scientific protocol or final training windows. Source Date+Time causality is checked; actual receiver polling/arrival timestamps are not certified.",
        "causal_availability_prefix_check": "passed",
        "raw_days": raw_results,
        "totals": {"raw_rows": sum(x["raw_rows"] for x in raw_results), "terminal_rows": len(observations),
                   "terminal_footprint_rows_before_timestamp_dedup": sum(x["terminal_footprint_rows"] for x in raw_results),
                   "duplicate_actor_timestamp_rows": sum(x["duplicate_actor_timestamp_rows_in_footprint"] for x in raw_results),
                   "conflicting_same_actor_timestamp_positions": sum(x["duplicate_actor_timestamp_conflicting_positions"] for x in raw_results),
                   "negative_age_rows_after_timestamp_dedup": int((observations.Age < 0).sum()),
                   "source_sessions": len(raw_results), "processed_scenes": len(processed),
                   "processed_train_scenes": int((processed.official_split == "train").sum()),
                   "processed_test_scenes": int((processed.official_split == "test").sum()),
                   "processed_scenes_with_simultaneous_multiple_actors": int((processed.max_simultaneous_actors >= 2).sum()),
                   "processed_max_simultaneous_actors": int(processed.max_simultaneous_actors.max()),
                   "processed_rows": int(processed.rows.sum())},
        "raw_multi_actor_feasibility_by_age_s": {str(age): {
            "multi_actor_grid_seconds": sum(x["availability_by_max_position_age_s"][str(age)]["multi_actor_grid_seconds"] for x in raw_results),
            "complete_10s_history_multi_actor_endpoints": sum(x["availability_by_max_position_age_s"][str(age)]["complete_10s_history_multi_actor_endpoints"] for x in raw_results),
            "max_simultaneous_fresh_actors": max(x["availability_by_max_position_age_s"][str(age)]["max_simultaneous_fresh_actors"] for x in raw_results),
        } for age in [2.0, 5.0, 10.0]},
        "raw_observation_artifact": "data/public/trajair_source/terminal_raw_observations.parquet",
        "processed_scene_artifact": "data/public/trajair_source/processed_scene_feasibility.csv",
        "canonical_observations_sha256": hash_file(DEST / "terminal_raw_observations.parquet"),
        "cautions": ["Processed files already interpolate whole trajectories and have no original-observation mask.",
                     "Official website describes cubic interpolation; pinned published script uses pandas default linear interpolation with limit=60.",
                     "Whole-day resample('S').mean() may include records later within the labelled second; linear filling may access beyond a forecast cutoff.",
                     "Receiver-record timestamps and inferred position-fix timestamps are different. Never backdate availability by Age.",
                     "Repeated identical ID/Date/Time/position rows have varying Age; Date+Time is not demonstrably each row's polling time. Age-adjusted overlap remains a conditional source-time diagnostic, not certified online availability.",
                     "Folder dates and row Date fields differ around midnight; preserve both.",
                     "Raw altitude is in feet; barometric/GNSS flag is absent in released CSV, so a true geometric vertical datum is not certified.",
                     "No future targets were interpolated or prediction errors computed in this audit.",
                     "Raw 1 Hz held availability is an observation-support diagnostic, not newly measured 1 Hz positions."]}
    save_json(DEST / "feasibility_summary.json", summary)
    print(json.dumps({"totals": summary["totals"], "raw_multi_actor_feasibility_by_age_s": summary["raw_multi_actor_feasibility_by_age_s"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--references", action="store_true")
    args = parser.parse_args()
    if args.acquire:
        acquire()
    if args.inspect:
        inspect_data()
    if args.references:
        acquire_reference_details()
    if not (args.acquire or args.inspect or args.references):
        parser.error("Choose --acquire, --inspect and/or --references")
