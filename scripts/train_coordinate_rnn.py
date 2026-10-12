"""Minimal coordinate-only LSTM/GRU/TCN trajectory baselines.

The model receives only local-frame history positions (x,y,z), without
velocity, edge features, actor labels, or future-derived quantities.  It
predicts a single future trajectory per actor and is evaluated with the same
scene metrics as the main trajectory trainer.  This is a diagnostic baseline,
not a replacement for the joint model.
"""
from __future__ import annotations

import argparse, json, random, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from trajectory_prediction.artifacts import start_run, finish_run, write_json, sha256_file
from trajectory_prediction.packed_data import PackedSceneDataset, collate_scenes
from trajectory_prediction.metrics import score_scene, aggregate_scene_records


def move_batch(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_batch(v, device) for k, v in batch.items()}
    return batch


def draw_epoch(dataset, seed, epoch, per_group):
    groups = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(dataset.rows):
        groups[row["family_id"]][row["replay_id"]].append(i)
    rng = np.random.default_rng(seed + epoch * 10007)
    out = []
    for family in sorted(groups):
        replays = sorted(groups[family])
        for _ in range(per_group):
            replay = replays[int(rng.integers(len(replays)))]
            out.append(int(rng.choice(groups[family][replay])))
    rng.shuffle(out)
    return out


class CoordinateRNN(nn.Module):
    def __init__(self, kind, hidden, layers, horizon):
        super().__init__()
        self.horizon = horizon
        cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = cls(3, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, horizon * 3))

    def forward(self, history_features, actor_mask, context):
        # Feature positions are local and scaled by 1000 in the packed contract.
        x = history_features[..., :3]
        b, n, t, _ = x.shape
        h, _ = self.rnn(x.reshape(b * n, t, 3))
        delta = self.head(h[:, -1]).reshape(b, n, self.horizon, 3)
        p0 = context["p0_shared"]
        rot = context["R_local_to_shared"]
        # The head predicts local displacement in the same normalized units as
        # the packed coordinates (metres / 1000). Convert only after rotation.
        delta_shared = torch.einsum("bnij,bnhj->bnhi", rot, delta)
        pred = p0[:, :, None, :] + delta_shared * 1000.0
        pred = pred.unsqueeze(1)
        # [B,K=1,N,H,3]
        return pred


class CoordinateTCN(nn.Module):
    def __init__(self, hidden, layers, horizon, dropout=0.1):
        super().__init__()
        blocks = []
        channels = 3
        for i in range(layers):
            dilation = 2 ** i
            blocks.append(nn.ModuleDict({
                "conv": nn.Conv1d(channels, hidden, kernel_size=3,
                                  padding=2 * dilation, dilation=dilation),
                "residual": nn.Conv1d(channels, hidden, kernel_size=1),
                "dropout": nn.Dropout(dropout),
            }))
            channels = hidden
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, horizon * 3))
        self.horizon = horizon

    def forward(self, history_features, actor_mask, context):
        x = history_features[..., :3]
        b, n, t, _ = x.shape
        x = x.reshape(b * n, t, 3).transpose(1, 2)
        for block in self.blocks:
            y = block["conv"](x)[..., :x.shape[-1]]
            x = torch.relu(block["residual"](x) + block["dropout"](y))
        delta = self.head(x[..., -1]).reshape(b, n, self.horizon, 3)
        p0 = context["p0_shared"]
        rot = context["R_local_to_shared"]
        delta_shared = torch.einsum("bnij,bnhj->bnhi", rot, delta)
        pred = p0[:, :, None, :] + delta_shared * 1000.0
        return pred.unsqueeze(1)


@torch.inference_mode()
def evaluate(model, loader, device, horizon):
    model.eval(); records = []
    for raw in loader:
        batch = move_batch(raw, device)
        pred = model(batch["history_features"], batch["actor_mask"], batch["history_context"]).cpu().numpy()
        target = raw["target"].numpy(); valid = raw["future_valid"].numpy()
        for i, meta in enumerate(raw["metadata"]):
            n = meta["actor_count"]
            r = score_scene(pred[i, :, :n], target[i, :n], valid[i, :n])
            r.update(replay_id=meta.get("replay_id"), family_id=meta.get("family_id"),
                     sample_id=meta.get("sample_id", meta.get("scene_id")), scenario=meta.get("scenario", "air"))
            records.append(r)
    return aggregate_scene_records(records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True); p.add_argument("--data", required=True)
    p.add_argument("--kind", choices=["lstm", "gru", "tcn"], required=True)
    p.add_argument("--horizon", type=float, default=10); p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=15); p.add_argument("--samples-per-group", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=17); p.add_argument("--device", default="cpu")
    p.add_argument("--validation-index-name", default="reference_scene_index.jsonl")
    p.add_argument("--validation-cache-split", default="validation_reference")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    root = ROOT / a.data
    train = PackedSceneDataset(root, "train", a.horizon)
    val = PackedSceneDataset(root, "validation", a.horizon, index_name=a.validation_index_name, cache_split=a.validation_cache_split)
    device = torch.device(a.device)
    model = (CoordinateTCN(a.hidden, a.layers, train.h) if a.kind == "tcn"
             else CoordinateRNN(a.kind, a.hidden, a.layers, train.h)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    loader_args = dict(batch_size=a.batch_size, collate_fn=collate_scenes, num_workers=0)
    val_loader = DataLoader(val, shuffle=False, **loader_args)
    config = dict(vars(a), train_windows=len(train), validation_windows=len(val),
                  coordinate_only=True, future_inputs_used=False,
                  dataset_config_sha256=sha256_file(root / "dataset_config.json"),
                  validation_index_sha256=sha256_file(root / a.validation_index_name),
                  source_independence_verified=False, final_test_enabled=False)
    run_dir, record = start_run(a.name, config, [Path(__file__)])
    if a.eval_every < 1:
        raise ValueError("eval-every must be positive")
    best = float("inf"); best_epoch = 0; curve = []; started = time.perf_counter(); last_summary = None
    try:
        for epoch in range(1, a.epochs + 1):
            model.train(); losses=[]
            ids = draw_epoch(train, a.seed, epoch, a.samples_per_group)
            loader = DataLoader(train, sampler=ids, **loader_args)
            factor = min(1.0, epoch / 5.0) * .5 * (1 + np.cos(np.pi * max(0, epoch-5) / max(1, a.epochs-5)))
            for group in opt.param_groups: group["lr"] = max(1e-6, a.lr * factor)
            for raw in loader:
                b = move_batch(raw, device); opt.zero_grad(set_to_none=True)
                pred = model(b["history_features"], b["actor_mask"], b["history_context"])
                # Coordinate-only single-mode loss in shared coordinates.
                err = pred[:, 0] - b["target"]
                mask = b["future_valid"] & b["actor_mask"][:, :, None]
                loss = torch.linalg.vector_norm(err, dim=-1).masked_select(mask).mean()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.detach()))
            should_eval = (epoch % a.eval_every == 0) or (epoch == a.epochs)
            summary = evaluate(model, val_loader, device, train.h) if should_eval else None
            metric = summary["scene_minADE_m"] if summary is not None else None
            line = dict(epoch=epoch, training_loss=float(np.mean(losses)), validation_scene_minADE_m=metric,
                        validation_scene_minFDE_m=(summary["scene_minFDE_m"] if summary is not None else None),
                        evaluated=should_eval, lr=opt.param_groups[0]["lr"])
            curve.append(line)
            with (run_dir / "learning_curve.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(line) + "\n")
            if summary is not None:
                last_summary = summary
            if summary is not None and metric < best:
                best, best_epoch = metric, epoch
                torch.save({"model": model.state_dict(), "epoch": epoch}, run_dir / "best.pt")
                write_json(run_dir / "best_validation_summary.json", summary)
            print(json.dumps(dict(line, best_epoch=best_epoch, run_id=record["run_id"])), flush=True)
            if should_eval and epoch - best_epoch >= a.patience * a.eval_every: break
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True)["model"])
        summary = evaluate(model, val_loader, device, train.h)
        write_json(run_dir / "validation_summary.json", summary)
        finish_run(run_dir, record, "completed", epochs_run=len(curve), best_epoch=best_epoch,
                   parameter_count=sum(x.numel() for x in model.parameters()), elapsed_seconds=time.perf_counter()-started,
                   checkpoint_sha256=sha256_file(run_dir / "best.pt"), validation_summary=summary)
        print(json.dumps({"completed": record["run_id"], "best_epoch": best_epoch, **{k: summary[k] for k in ("scene_minADE_m", "scene_minFDE_m")}}), flush=True)
    except Exception:
        finish_run(run_dir, record, "failed", error="training failed")
        raise


if __name__ == "__main__": main()
