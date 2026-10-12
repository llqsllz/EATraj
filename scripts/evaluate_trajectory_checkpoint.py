"""Evaluate a frozen trajectory checkpoint on an explicitly named split."""
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
from trajectory_prediction.artifacts import write_json
from trajectory_prediction.metrics import aggregate_scene_records, score_scene
from trajectory_prediction.models import TrajectoryModel, assemble_independent_scenes, assemble_factorized_scenes
from trajectory_prediction.packed_data import PackedSceneDataset, collate_scenes


def move_batch(batch, device):
    return {
        key: ({k: v.to(device) for k, v in value.items()} if key == "history_context" else
              value.to(device) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test_dev", choices=["train", "validation", "test_dev"])
    parser.add_argument("--cache-split", default=None)
    parser.add_argument("--index-name", default="scene_index.jsonl")
    parser.add_argument("--horizon", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-array-hash", action="store_true")
    parser.add_argument("--output-dir", default=None,
                        help="optional output directory relative to the project root")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    data_root = ROOT / args.data
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_config = checkpoint["model_config"]
    if model_config.get("temporal") == "mamba2":
        try:
            from mamba_ssm import Mamba2  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "This evaluator requires the official mamba_ssm runtime for a mamba2 checkpoint"
            ) from exc
    model = TrajectoryModel(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    data = PackedSceneDataset(data_root, args.split, args.horizon,
                              verify_array_hashes=not args.skip_array_hash,
                              index_name=args.index_name, cache_split=args.cache_split)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_scenes, num_workers=0)
    out_dir = (ROOT / args.output_dir if args.output_dir else
               checkpoint_path.parent / f"{args.split}_h{int(args.horizon)}_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    max_actors = max(row["actor_count"] for row in data.rows)
    records = []
    predictions = []
    scores_out = []
    sample_ids = []
    actor_counts = []
    metadata_out = []
    for raw in loader:
        batch = move_batch(raw, device)
        output = model(batch["history_features"], batch["actor_mask"], batch["forecast_times"],
                       batch["history_context"], batch["history_edges"])
        pred = output["trajectories"].detach().cpu().numpy()
        actor_logits = output.get("actor_logits")
        scene_logits = output.get("scene_logits")
        if actor_logits is not None:
            actor_logits = actor_logits.detach().cpu().numpy()
        if scene_logits is not None:
            scene_logits = scene_logits.detach().cpu().numpy()
        target = raw["target"].numpy()
        valid = raw["future_valid"].numpy()
        for i, meta in enumerate(raw["metadata"]):
            n = meta["actor_count"]
            factorized = bool(output.get("joint_factorized", False))
            candidates = pred[i, :, :, :n] if factorized else pred[i, :, :n]
            marginal = None
            if factorized:
                final_pred, scene_scores = assemble_factorized_scenes(
                    candidates, scene_logits[i], output["local_logits"].detach().cpu().numpy()[i, :, :n],
                    model_config["modes"]
                )
            elif scene_logits is None:
                marginal = score_scene(candidates, target[i, :n], valid[i, :n])
                final_pred, scene_scores = assemble_independent_scenes(candidates, actor_logits[i, :n], model_config["modes"])
            else:
                final_pred, scene_scores = candidates, scene_logits[i]
            result = score_scene(final_pred, target[i, :n], valid[i, :n], scores=scene_scores)
            if marginal is not None:
                result.update(marginal_minADE_m=marginal["marginal_minADE_m"],
                              marginal_minFDE_m=marginal["marginal_minFDE_m"])
            result.update(replay_id=meta.get("replay_id", "unknown"),
                          family_id=meta.get("family_id", "unknown"),
                          sample_id=meta.get("sample_id", meta.get("endpoint_index", "unknown")),
                          scenario=meta.get("scenario", "unknown"))
            records.append(result)
            padded = np.zeros((len(final_pred), max_actors, final_pred.shape[-2], 3), dtype=np.float32)
            padded[:, :n] = final_pred
            predictions.append(padded)
            scores_out.append(scene_scores)
            sample_ids.append(result["sample_id"])
            actor_counts.append(n)
            metadata_out.append(meta)
    summary = aggregate_scene_records(records)
    with (out_dir / f"{args.split}_scene_scores.jsonl").open("w", encoding="utf-8") as handle:
        for result in records:
            handle.write(json.dumps(result, allow_nan=False) + "\n")
    np.savez_compressed(out_dir / f"{args.split}_predictions.npz",
                        trajectories=np.asarray(predictions), scores=np.asarray(scores_out),
                        sample_ids=np.asarray(sample_ids), actor_counts=np.asarray(actor_counts))
    write_json(out_dir / "prediction_index.json", metadata_out)
    write_json(out_dir / f"{args.split}_summary.json", summary)
    write_json(out_dir / "evaluation_config.json", vars(args) | {
        "checkpoint": str(checkpoint_path), "checkpoint_epoch": checkpoint.get("epoch"),
        "model_config": model_config, "formal_test_enabled": False,
        "test_semantics": "development test_dev split from reused simulation cohort",
    })
    print(json.dumps({"output": str(out_dir), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
