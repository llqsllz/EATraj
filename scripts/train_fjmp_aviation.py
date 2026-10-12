"""Train/evaluate the map-free FJMP aviation adaptation."""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trajectory_prediction.artifacts import start_run, finish_run, write_json, sha256_file
from trajectory_prediction.packed_data import PackedSceneDataset, collate_scenes
from trajectory_prediction.metrics import score_scene, aggregate_scene_records
from trajectory_prediction.models import trajectory_loss
from trajectory_prediction.fjmp_adapted import FJMPAviationModel, aviation_relation_targets


class LimitedDataset:
    """Read-only prefix view used for CPU smoke tests without changing caches."""
    def __init__(self, dataset, limit=None):
        self.dataset = dataset
        self.root = dataset.root
        self.indices = list(range(len(dataset))) if not limit else list(range(min(int(limit), len(dataset))))
        self.rows = [dataset.rows[i] for i in self.indices]
    def __len__(self): return len(self.indices)
    def __getitem__(self, i): return self.dataset[self.indices[i]]


def move_batch(batch, device):
    return {k: ({x: y.to(device) for x, y in v.items()} if k == "history_context" else
                v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def draw_epoch(dataset, seed, epoch, per_group):
    groups = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(dataset.rows): groups[row["family_id"]][row["replay_id"]].append(i)
    rng = np.random.default_rng(seed + epoch * 10007); indices = []
    for family in sorted(groups):
        for _ in range(per_group):
            replay = sorted(groups[family])[int(rng.integers(len(groups[family])))]
            indices.append(int(rng.choice(groups[family][replay])))
    rng.shuffle(indices); return indices


@torch.inference_mode()
def evaluate(model, loader, device, output_dir=None):
    model.eval(); records = []; predictions = []; scores_all = []; metadata_all = []
    max_actors = max(row["actor_count"] for row in loader.dataset.rows)
    for raw in loader:
        batch = move_batch(raw, device); out = model(batch["history_features"], batch["actor_mask"], batch["forecast_times"], batch["history_context"], batch["history_edges"])
        pred, scores = out["trajectories"].cpu().numpy(), out["scene_logits"].cpu().numpy()
        for i, meta in enumerate(raw["metadata"]):
            n = meta["actor_count"]
            record = score_scene(pred[i, :, :n], raw["target"][i, :n].numpy(), raw["future_valid"][i, :n].numpy(), scores=scores[i])
            record.update(replay_id=str(meta.get("replay_id", "unknown")), family_id=str(meta.get("family_id", "unknown")), sample_id=str(meta.get("sample_id", "unknown")), scenario=meta.get("scenario", "public_multiaircraft"))
            records.append(record)
            if output_dir is not None:
                padded = np.zeros((pred.shape[1], max_actors, pred.shape[-2], 3), np.float32); padded[:, :n] = pred[i, :, :n]
                predictions.append(padded); scores_all.append(scores[i]); metadata_all.append(meta)
    summary = aggregate_scene_records(records)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "validation_scene_scores.jsonl").open("w", encoding="utf-8") as f:
            for r in records: f.write(json.dumps(r, allow_nan=False) + "\n")
        np.savez_compressed(output_dir / "validation_predictions.npz", trajectories=np.asarray(predictions), scores=np.asarray(scores_all))
        write_json(output_dir / "prediction_index.json", metadata_all)
        write_json(output_dir / "validation_summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/trajectory_joint_v5/common50_hist10_20260923")
    p.add_argument("--validation-index-name", default="reference_scene_index.jsonl")
    p.add_argument("--validation-cache-split", default="validation_reference")
    p.add_argument("--name", default="fjmp-aviation-adapted")
    p.add_argument("--horizon", type=float, default=20); p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=4); p.add_argument("--samples-per-group", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=17); p.add_argument("--width", type=int, default=64); p.add_argument("--layers", type=int, default=2); p.add_argument("--heads", type=int, default=4); p.add_argument("--modes", type=int, default=6)
    p.add_argument("--relation-threshold-m", type=float, default=1000.0); p.add_argument("--relation-weight", type=float, default=0.1)
    p.add_argument("--device", default="cuda"); p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--limit-train-scenes", type=int, default=0)
    p.add_argument("--limit-validation-scenes", type=int, default=0)
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    root = ROOT / args.data
    train = PackedSceneDataset(root, "train", args.horizon)
    val = PackedSceneDataset(root, "validation", args.horizon, index_name=args.validation_index_name, cache_split=args.validation_cache_split)
    if args.limit_train_scenes: train = LimitedDataset(train, args.limit_train_scenes)
    if args.limit_validation_scenes: val = LimitedDataset(val, args.limit_validation_scenes)
    config = dict(vars(args), model={"width": args.width, "layers": args.layers, "heads": args.heads, "modes": args.modes}, data_root=str(root), dataset_config_sha256=sha256_file(root / "dataset_config.json"), training_scene_index_sha256=sha256_file(root / "scene_index.jsonl"), validation_scene_index_sha256=sha256_file(root / args.validation_index_name), train_cache_manifest_sha256=sha256_file(train.root / "manifest.json"), validation_cache_manifest_sha256=sha256_file(val.root / "manifest.json"), official_fjmp=False, adaptation="map_free_3d_dag_fjmp_style", relation_supervision="weak_future_proximity_not_causal_ground_truth")
    run_dir, rec = start_run(args.name, config, [Path(__file__)])
    device = torch.device(args.device); model = FJMPAviationModel(args.width, args.layers, args.heads, args.modes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader_kw = dict(batch_size=args.batch_size, collate_fn=collate_scenes, num_workers=args.num_workers, pin_memory=args.device == "cuda")
    val_loader = DataLoader(val, shuffle=False, **loader_kw); best = float("inf"); best_epoch = 0; start = time.perf_counter(); curve = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []; rel_losses = []; sampler = draw_epoch(train, args.seed, epoch, args.samples_per_group); loader = DataLoader(train, sampler=sampler, **loader_kw)
        for raw in loader:
            batch = move_batch(raw, device); opt.zero_grad(set_to_none=True)
            out = model(batch["history_features"], batch["actor_mask"], batch["forecast_times"], batch["history_context"], batch["history_edges"])
            loss, _ = trajectory_loss(out, batch["target"], batch["future_valid"], batch["actor_mask"], joint=True)
            labels = aviation_relation_targets(batch["target"], batch["future_valid"], batch["actor_mask"], args.relation_threshold_m)
            rel = (torch.nn.functional.cross_entropy(out["relation_logits"].reshape(-1, 3), labels.reshape(-1), ignore_index=-1)
                   if (labels >= 0).any() else out["relation_logits"].sum() * 0.0)
            total = loss + args.relation_weight * rel; total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.detach())); rel_losses.append(float(rel.detach()))
        summary = evaluate(model, val_loader, device)
        metric = summary["scene_minADE_m"]; line = {"epoch": epoch, "training_loss": float(np.mean(losses)), "relation_loss": float(np.mean(rel_losses)), "validation_scene_minADE_m": metric, "validation_scene_minFDE_m": summary["scene_minFDE_m"]}; curve.append(line)
        with (run_dir / "learning_curve.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(line) + "\n")
        print(json.dumps(dict(line, run_id=rec["run_id"])), flush=True)
        if metric is not None and metric < best:
            best = metric; best_epoch = epoch; torch.save({"model": model.state_dict(), "model_config": config["model"], "epoch": epoch}, run_dir / "best.pt"); write_json(run_dir / "best_validation_summary.json", summary)
        if epoch - best_epoch >= args.patience: break
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True); model.load_state_dict(ckpt["model"]); final = evaluate(model, val_loader, device, run_dir); write_json(run_dir / "progress.json", {"status": "completed", "best_epoch": best_epoch, "validation": final, "curve": curve})
    finish_run(run_dir, rec, "completed", epochs_run=len(curve), best_epoch=best_epoch, parameter_count=sum(x.numel() for x in model.parameters()), elapsed_seconds=time.perf_counter() - start, checkpoint_sha256=sha256_file(run_dir / "best.pt"), validation_summary=final)
    print(json.dumps({"completed": rec["run_id"], "run_dir": str(run_dir), "best_epoch": best_epoch, "validation": final}), flush=True)


if __name__ == "__main__": main()
