"""Small, explicit experiment records and source snapshots."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def start_run(name: str, config: dict, sources: list[Path] | None = None) -> tuple[Path, dict]:
    now = datetime.now(timezone.utc)
    run_id = f"E-{now:%Y%m%dT%H%M%S%fZ}-{name}"
    run_dir = ROOT / ".experiments" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot = run_dir / "source"
    files = sorted(set(list((ROOT / "src" / "trajectory_prediction").glob("*.py")) + (sources or [])))
    hashes = {}
    for source in files:
        relative = source.relative_to(ROOT)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = sha256_file(source)
    packages = {}
    for package_name in ("numpy", "pandas", "pyarrow", "torch", "scipy", "mamba-ssm", "causal-conv1d"):
        try:
            packages[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            packages[package_name] = None
    record = {
        "run_id": run_id, "name": name, "started_utc": now.isoformat(),
        "status": "running", "evidence_status": "exploratory", "config": config,
        "command_argv": sys.argv, "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "git_commit": None, "source_sha256": hashes,
        "packages": packages,
    }
    write_json(run_dir / "record.json", record)
    with (ROOT / ".experiments" / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": run_id, "status": "running", "path": str(run_dir.relative_to(ROOT))}) + "\n")
    return run_dir, record


def finish_run(run_dir: Path, record: dict, status: str, **results: object) -> None:
    record.update(status=status, finished_utc=datetime.now(timezone.utc).isoformat(), **results)
    write_json(run_dir / "record.json", record)
    with (ROOT / ".experiments" / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": record["run_id"], "status": status, "path": str(run_dir.relative_to(ROOT))}) + "\n")
