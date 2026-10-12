"""Evaluate a frozen coordinate-only LSTM/GRU/TCN checkpoint on a chosen split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from trajectory_prediction.artifacts import write_json
from trajectory_prediction.metrics import aggregate_scene_records, score_scene
from trajectory_prediction.packed_data import PackedSceneDataset, collate_scenes
from train_coordinate_rnn import CoordinateRNN, CoordinateTCN, move_batch


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--kind", choices=["lstm", "gru", "tcn"], required=True)
    parser.add_argument("--split", default="test_dev", choices=["train", "validation", "test_dev"])
    parser.add_argument("--cache-split", default=None)
    parser.add_argument("--index-name", default="scene_index.jsonl")
    parser.add_argument("--horizon", type=float, default=5.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--skip-array-hash", action="store_true")
    parser.add_argument("--output-dir", default=None,
                        help="optional output directory relative to the project root")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    device = torch.device(args.device)
    data = PackedSceneDataset(ROOT / args.data, args.split, args.horizon,
                              verify_array_hashes=not args.skip_array_hash,
                              index_name=args.index_name, cache_split=args.cache_split)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = (CoordinateTCN(args.hidden, args.layers, data.h) if args.kind == "tcn"
             else CoordinateRNN(args.kind, args.hidden, args.layers, data.h)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_scenes, num_workers=0)
    out_dir = (ROOT / args.output_dir if args.output_dir else
               checkpoint_path.parent / f"{args.split}_h{int(args.horizon)}_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    max_actors = max(row["actor_count"] for row in data.rows)
    records, predictions, sample_ids, actor_counts, metadata_out = [], [], [], [], []
    for raw in loader:
        batch = move_batch(raw, device)
        pred = model(batch["history_features"], batch["actor_mask"], batch["history_context"]).cpu().numpy()
        target, valid = raw["target"].numpy(), raw["future_valid"].numpy()
        for i, meta in enumerate(raw["metadata"]):
            n = meta["actor_count"]
            result = score_scene(pred[i, :, :n], target[i, :n], valid[i, :n])
            sample_id = meta.get("sample_id", meta.get("endpoint_index", "unknown"))
            result.update(replay_id=meta.get("replay_id", "unknown"),
                          family_id=meta.get("family_id", "unknown"),
                          sample_id=sample_id, scenario=meta.get("scenario", "unknown"))
            records.append(result)
            padded = np.zeros((1, max_actors, data.h, 3), dtype=np.float32)
            padded[:, :n] = pred[i, :, :n]
            predictions.append(padded)
            sample_ids.append(sample_id)
            actor_counts.append(n)
            metadata_out.append(meta)
    summary = aggregate_scene_records(records)
    with (out_dir / f"{args.split}_scene_scores.jsonl").open("w", encoding="utf-8") as handle:
        for result in records:
            handle.write(json.dumps(result, allow_nan=False) + "\n")
    np.savez_compressed(out_dir / f"{args.split}_predictions.npz",
                        trajectories=np.asarray(predictions), scores=np.zeros((len(predictions), 1), dtype=np.float32),
                        sample_ids=np.asarray(sample_ids), actor_counts=np.asarray(actor_counts))
    write_json(out_dir / "prediction_index.json", metadata_out)
    write_json(out_dir / f"{args.split}_summary.json", summary)
    write_json(out_dir / "evaluation_config.json", vars(args) | {
        "checkpoint": str(checkpoint_path), "checkpoint_epoch": checkpoint.get("epoch"),
        "formal_test_enabled": False,
        "test_semantics": "development test_dev split from reused simulation cohort",
    })
    metric_keys = ("scene_minADE_m", "scene_minFDE_m", "top_score_ADE_m", "top_score_FDE_m")
    print(json.dumps({"output": str(out_dir), "counts": summary["counts"],
                      "metrics": {key: summary.get(key) for key in metric_keys}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
