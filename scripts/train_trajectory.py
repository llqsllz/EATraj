"""Tracked GPU development training. No test split is accepted by this entrypoint."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from trajectory_prediction.artifacts import start_run,finish_run,write_json,sha256_file
from trajectory_prediction.packed_data import PackedSceneDataset,collate_scenes
from trajectory_prediction.metrics import score_scene,aggregate_scene_records
from trajectory_prediction.endpoint_metrics import score_endpoint_scene,aggregate_endpoint_records
from trajectory_prediction.models import TrajectoryModel,trajectory_loss,assemble_independent_scenes,assemble_factorized_scenes

FINAL_DATASET = "data/trajectory_joint_v5/common50_hist30_20260923"
FINAL_SAMPLE_RATE_HZ = 10.0
FINAL_HISTORY_SECONDS = 30.0
FINAL_DEVELOPMENT_HORIZON_SECONDS = 5.0
TRAINING_STATE_FORMAT = "trajectory-training-state-v1"


def parse_stop_at(value):
    """Require an explicit timezone so a midnight deadline is unambiguous."""
    if value is None:
        return None
    try:
        deadline = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("--stop-at requires an ISO datetime with UTC Z or an offset") from exc
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise ValueError("--stop-at requires an explicit timezone, e.g. 2026-09-30T08:00:00+08:00")
    return deadline.astimezone(timezone.utc)


def deadline_reached(deadline, now=None):
    return deadline is not None and (now or datetime.now(timezone.utc)) >= deadline


def capture_rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"kind": numpy_state[0], "keys": numpy_state[1].tolist(),
                      "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                      "cached_gaussian": float(numpy_state[4])},
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["kind"], np.asarray(n["keys"], dtype=np.uint32),
                         n["position"], n["has_gauss"], n["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact resume requires the same available CUDA device count")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def initial_schedule(args):
    return {"kind": "original_warmup_cosine", "start_epoch": 0,
            "end_epoch": args.epochs, "base_lr": args.lr, "minimum_lr": 1e-6}


def scheduled_learning_rate(epoch, stage):
    """Original-budget LR is unchanged by interruption/resume."""
    local_epoch = epoch - stage["start_epoch"]
    duration = stage["end_epoch"] - stage["start_epoch"]
    if not 1 <= local_epoch <= duration:
        raise ValueError("Epoch is outside the declared learning-rate stage")
    if stage["kind"] == "original_warmup_cosine":
        factor = min(1., local_epoch / 5.) * .5 * (1 + np.cos(
            np.pi * max(0, local_epoch - 5) / max(1, duration - 5)))
    elif stage["kind"] == "extension_cosine":
        factor = .5 * (1 + np.cos(np.pi * (local_epoch - 1) / max(1, duration - 1)))
    else:
        raise ValueError("Unknown learning-rate stage")
    return max(stage["minimum_lr"], stage["base_lr"] * factor)


def read_training_state(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            "Not a resumable training state; best.pt/init-checkpoint is weights-only warm-start, not resume"
        ) from exc
    required = {"format", "model", "model_config", "optimizer", "epoch", "best", "best_epoch",
                "rng", "curve", "training_args", "schedule_stages", "trainability", "run_id",
                "last_summary", "best_checkpoint_sha256", "source_sha256", "config", "elapsed_seconds"}
    if not isinstance(state, dict) or state.get("format") != TRAINING_STATE_FORMAT or not required.issubset(state):
        raise ValueError("Not a resumable training state; best.pt/init-checkpoint is weights-only warm-start, not resume")
    if [row["epoch"] for row in state["curve"]] != list(range(1, state["epoch"] + 1)):
        raise ValueError("Training-state learning curve is not a complete epoch sequence")
    return state


def resolve_resume_args(args, argv):
    """Resume inherits the saved scientific protocol; only runtime controls may differ."""
    if not args.resume_state:
        if args.extend_epochs is not None or args.extension_lr is not None:
            raise ValueError("--extend-epochs/--extension-lr require --resume-state")
        return args, None, None
    allowed = {"--resume-state", "--stop-at", "--save-training-state", "--extend-epochs", "--extension-lr"}
    provided = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
    if provided - allowed:
        raise ValueError("Resume restores all saved training options; remove overrides: " + ", ".join(sorted(provided - allowed)))
    state_path = Path(args.resume_state)
    if not state_path.is_absolute():
        state_path = ROOT / state_path
    state = read_training_state(state_path)
    restored = argparse.Namespace(**state["training_args"])
    restored.resume_state = str(state_path)
    restored.stop_at = args.stop_at
    restored.save_training_state = True
    restored.extend_epochs = args.extend_epochs
    restored.extension_lr = args.extension_lr
    if args.extension_lr is not None and (args.extend_epochs is None or not np.isfinite(args.extension_lr) or args.extension_lr <= 0):
        raise ValueError("A finite positive --extension-lr requires --extend-epochs")
    if args.extend_epochs is not None:
        if args.extend_epochs <= max(state["epoch"], restored.epochs):
            raise ValueError("--extend-epochs is a new total budget and must exceed the saved planned budget")
        restored.epochs = args.extend_epochs
    return restored, state, state_path


def save_training_state(path, *, model, model_config, optimizer, epoch, best, best_epoch,
                        curve, args, stages, record, summary, elapsed_seconds):
    """Atomic epoch-boundary state; old best.pt deliberately lacks this format."""
    best_path = path.parent / "best.pt"
    # Learning-curve values can contain NumPy scalar learning rates. Convert
    # them before torch.save so the state remains loadable with the safe
    # ``weights_only=True`` reader used for checkpoints.
    safe_curve = json.loads(json.dumps(curve, allow_nan=False))
    safe_stages = json.loads(json.dumps(stages, allow_nan=False))
    state = {"format": TRAINING_STATE_FORMAT, "model": model.state_dict(),
             "model_config": model_config, "optimizer": optimizer.state_dict(),
             "epoch": epoch, "best": best, "best_epoch": best_epoch, "rng": capture_rng_state(),
             "curve": safe_curve, "training_args": vars(args).copy(), "schedule_stages": safe_stages,
             "trainability": {name: p.requires_grad for name, p in model.named_parameters()},
             "run_id": record["run_id"], "last_summary": summary,
             "best_checkpoint_sha256": sha256_file(best_path) if best_path.exists() else None,
             "source_sha256": record["source_sha256"], "config": json.loads(json.dumps(record["config"])),
             "elapsed_seconds": elapsed_seconds,
             "saved_utc": datetime.now(timezone.utc).isoformat(),
             "resume_semantics": "exact_epoch_boundary_under_saved_schedule"}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def validate_resume_artifacts(state, state_path, config):
    run_dir = state_path.parent
    record = json.loads((run_dir / "record.json").read_text(encoding="utf-8"))
    if record["run_id"] != state["run_id"]:
        raise ValueError("Resume state belongs to a different run directory")
    for relative, expected in state["source_sha256"].items():
        if sha256_file(ROOT / relative.replace("\\", "/")) != expected:
            raise ValueError("Source changed since resumable state: " + relative)
    for field in ("model", "dataset_config_sha256", "training_scene_index_sha256", "validation_scene_index_sha256",
                  "split_manifest_sha256", "train_cache_manifest_sha256", "validation_cache_manifest_sha256",
                  "torch_version", "cuda_version", "gpu", "precision"):
        if config[field] != state["config"][field]:
            raise ValueError("Exact resume contract mismatch: " + field)
    curve_path = run_dir / "learning_curve.jsonl"
    disk_curve = [json.loads(line) for line in curve_path.read_text(encoding="utf-8").splitlines()] if curve_path.exists() else []
    if disk_curve != state["curve"]:
        raise ValueError("On-disk curve differs from the saved epoch boundary; preserve/audit it before resuming")
    best_path = run_dir / "best.pt"
    if state["best_checkpoint_sha256"] is not None and (not best_path.exists() or sha256_file(best_path) != state["best_checkpoint_sha256"]):
        raise ValueError("Best checkpoint changed since resumable state")
    return run_dir, record


def validate_final_dataset_protocol(args):
    """Fail closed when the frozen final development protocol is requested."""
    if Path(args.data).as_posix().rstrip("/") != FINAL_DATASET:
        return
    config_path = ROOT / args.data / "dataset_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if float(config.get("sample_rate_hz")) != FINAL_SAMPLE_RATE_HZ:
        raise ValueError("frozen dataset sample_rate_hz does not match the final protocol")
    if float(config.get("history_seconds")) != FINAL_HISTORY_SECONDS:
        raise ValueError("frozen dataset history_seconds does not match the final protocol")
    if float(args.horizon) != FINAL_DEVELOPMENT_HORIZON_SECONDS:
        raise ValueError(
            f"{FINAL_DATASET} is frozen for a 5 s development horizon; got --horizon {args.horizon}"
        )
    if args.validation_cache_split not in (None, "validation_reference"):
        raise ValueError("the frozen final protocol requires the validation_reference cache")
    if args.validation_cache_split == "validation_reference":
        if args.validation_index_name != "reference_scene_index.jsonl":
            raise ValueError("validation_reference requires reference_scene_index.jsonl")
    elif args.validation_index_name != "scene_index.jsonl":
        raise ValueError("the frozen final protocol requires scene_index.jsonl without validation_reference")


def move_batch(batch,device):
    return {key:({k:v.to(device,non_blocking=True) for k,v in value.items()} if key=="history_context" else
                 value.to(device,non_blocking=True) if torch.is_tensor(value) else value)
            for key,value in batch.items()}


def draw_epoch(dataset,seed,epoch,per_group):
    groups=defaultdict(lambda:defaultdict(list))
    for i,row in enumerate(dataset.rows):groups[row["family_id"]][row["replay_id"]].append(i)
    rng=np.random.default_rng(seed+epoch*10007);indices=[]
    for group in sorted(groups):
        replays=sorted(groups[group])
        for _ in range(per_group):
            rid=replays[int(rng.integers(len(replays)))]
            indices.append(int(rng.choice(groups[group][rid])))
    rng.shuffle(indices)
    return indices


def compact_goal_diagnostics(summary):
    return {k:v for k,v in summary.get("goal_diagnostics",{}).items()
            if k.startswith('goal_') or k in ('counts','actor_window_diagnostics')}


@torch.inference_mode()
def evaluate(model,loader,device,scene_modes,save_path=None):
    model.eval();records=[];predictions=[];scene_scores=[];sample_ids=[];actor_counts=[]
    goal_records=[]
    all_metadata=[]
    max_actors=max(row["actor_count"] for row in loader.dataset.rows)
    for raw in loader:
        batch=move_batch(raw,device)
        output=model(batch["history_features"],batch["actor_mask"],batch["forecast_times"],batch["history_context"],batch["history_edges"])
        pred=output["trajectories"].detach().cpu().numpy()
        actor_logits=output.get("actor_logits");scene_logits=output.get("scene_logits")
        factorized=bool(output.get("joint_factorized", False))
        if actor_logits is not None:actor_logits=actor_logits.detach().cpu().numpy()
        if scene_logits is not None:scene_logits=scene_logits.detach().cpu().numpy()
        local_logits=output.get("local_logits")
        if local_logits is not None:local_logits=local_logits.detach().cpu().numpy()
        target=raw["target"].numpy();valid=raw["future_valid"].numpy()
        goals_cpu = output.get("goals_local")
        if goals_cpu is not None:
            goals_cpu = goals_cpu.detach().cpu().numpy()
        context_cpu = raw["history_context"]
        p0_cpu = context_cpu["p0_shared"].numpy()
        rot_cpu = context_cpu["R_local_to_shared"].numpy()
        for i,meta in enumerate(raw["metadata"]):
            n=meta["actor_count"]
            p = pred[i, :, :, :n] if factorized else pred[i, :, :n]
            marginal=None
            if factorized:
                # Enumerate the highest-scoring global/local tuples. A single
                # local argmax per global mode discards heterogeneous scene
                # futures and is not the factorized joint distribution.
                p,scores=assemble_factorized_scenes(
                    p, scene_logits[i], local_logits[i,:,:n,:], scene_modes
                )
            elif scene_logits is None:
                # Marginal candidates are the ORIGINAL K options per actor. The
                # truth-free scene budget may discard some of these options.
                marginal=score_scene(p,target[i,:n],valid[i,:n])
                p,scores=assemble_independent_scenes(p,actor_logits[i,:n],scene_modes)
            else:scores=scene_logits[i]
            r=score_scene(p,target[i,:n],valid[i,:n],scores=scores)
            if goals_cpu is not None and not factorized and scene_logits is not None:
                goal_record=score_endpoint_scene(goals_cpu[i,:,:n],p,target[i,:n],valid[i,:n],
                    p0_cpu[i,:n],rot_cpu[i,:n],scores)
                goal_record.update(replay_id=meta.get("replay_id",meta.get("episode_id","unknown")),
                    family_id=meta.get("family_id",meta.get("episode_id","unknown")))
                goal_records.append(goal_record)
            if marginal is not None:
                for key in ("marginal_minADE_m","marginal_minFDE_m"):r[key]=marginal[key]
            r["marginal_candidate_source"]="original_actor_modes" if marginal is not None else ("factorized_global_plus_local_modes" if factorized else "joint_scene_modes")
            r.update(replay_id=meta.get("replay_id", meta.get("episode_id", "unknown")),
                     family_id=meta.get("family_id", meta.get("episode_id", "unknown")),
                     sample_id=meta.get("sample_id", meta.get("scene_id", meta.get("endpoint_index", "unknown"))),
                     scenario=meta.get("scenario", "public_multiaircraft"))
            records.append(r)
            if save_path is not None:
                padded=np.zeros((len(p),max_actors,p.shape[-2],3),np.float32);padded[:,:n]=p
                predictions.append(padded);scene_scores.append(scores);sample_ids.append(meta.get("sample_id", meta.get("scene_id", meta.get("endpoint_index", "unknown"))));actor_counts.append(n)
                all_metadata.append(meta)
    summary=aggregate_scene_records(records)
    if goal_records:
        summary["goal_diagnostics"]=aggregate_endpoint_records(goal_records)
    if save_path is not None:
        with (save_path/"validation_scene_scores.jsonl").open("w",encoding="utf-8") as f:
            for r in records:f.write(json.dumps(r,allow_nan=False)+"\n")
        np.savez_compressed(save_path/"validation_predictions.npz",trajectories=np.asarray(predictions),scores=np.asarray(scene_scores),
                            sample_ids=np.asarray(sample_ids),actor_counts=np.asarray(actor_counts))
        write_json(save_path/"prediction_index.json",all_metadata)
        # Preserve subgroup diagnostics without changing the main selection criterion.
        write_json(save_path/"validation_by_scenario.json",{s:aggregate_scene_records([r for r in records if r["scenario"]==s]) for s in sorted({r["scenario"] for r in records})})
    return summary


def build_parser():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/trajectory_joint_v3/dev_20260921")
    p.add_argument("--validation-index-name",default="scene_index.jsonl")
    p.add_argument("--validation-cache-split",default=None)
    p.add_argument("--name",default="transformer-independent-k1")
    p.add_argument("--temporal",choices=["transformer","mamba2"],default="transformer")
    p.add_argument("--temporal-variant",choices=["vanilla","conformer","patchtst","mamba_patch"],default="vanilla")
    p.add_argument("--patch-len",type=int,default=5)
    p.add_argument("--patch-stride",type=int,default=5)
    p.add_argument("--fusion-mode", choices=["add", "gated", "concat", "multiscale"], default="gated",
                   help="Mamba/Patch fusion for temporal_variant=mamba_patch")
    p.add_argument("--architecture",choices=["trajectory","ascent"],default="trajectory")
    p.add_argument("--interaction",action="store_true");p.add_argument("--joint",action="store_true")
    p.add_argument("--joint-variant", choices=["shared", "factorized", "influence"], default="shared",
                   help="Joint semantics: shared scene modes or global scene modes with actor-local residual modes")
    p.add_argument("--local-modes", type=int, default=3,
                   help="Actor-local latent modes used by factorized Joint")
    p.add_argument("--no-edge-geometry",action="store_true",
                   help="Zero historical edge inputs, retaining relation parameters and biases")
    p.add_argument("--edge-geometry-transform", choices=["none", "symmetric"], default="none",
                   help="Optional post-projection edge-embedding symmetrization; attention itself can remain directed")
    p.add_argument("--edge-keep-anchors", default=None,
                   help="Parameter-matched partial-edge ablation: comma-separated anchor indices to retain "
                        "(0=-30s start, 1=-15s mid, 2=0s current); e.g. '2' or '1,2'. Retains all if omitted.")
    p.add_argument("--edge-keep-feats", default=None,
                   help="Parameter-matched partial-edge ablation: comma-separated per-anchor feature indices "
                        "0..12 to retain (0-2 rel.pos, 3-5 rel.vel, 6 range, 7 closing, 8 TCPA, 9 DCPA, "
                        "10 tcpa_ok, 11 pad, 12 vel-valid); e.g. '0,1,2,3,4,5'. Retains all if omitted.")
    p.add_argument("--modes",type=int,default=1);p.add_argument("--width",type=int,default=64)
    p.add_argument("--heads",type=int,default=4)
    p.add_argument("--horizon",type=float,default=20);p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--patience",type=int,default=6);p.add_argument("--samples-per-group",type=int,default=16)
    p.add_argument("--batch-size",type=int,default=32);p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--score-weight", type=float, default=0.1,
                   help="Weight of candidate-score classification loss")
    p.add_argument("--seed",type=int,default=17);p.add_argument("--constraints",default=None)
    p.add_argument("--constraint-mode",choices=["full","acceleration_only"],default="full")
    p.add_argument("--device",default="cuda");p.add_argument("--num-workers",type=int,default=0)
    p.add_argument("--temporal-pool", choices=["last","mean","max"], default="last")
    p.add_argument("--relation-fusion", choices=["serial","gated"], default="serial")
    p.add_argument("--goal-head", action="store_true")
    p.add_argument("--goal-conditioning", action="store_true")
    p.add_argument("--goal-weight", type=float, default=0.0)
    p.add_argument("--goal-soft-temperature", type=float, default=10.0,
                   help="Metres; soft responsibility temperature for candidate endpoint loss")
    p.add_argument("--goal-radius-m", type=float, default=0.0,
                   help="Endpoint tolerance radius in metres; zero keeps point supervision")
    p.add_argument("--goal-loss-geometry", choices=["coordinate", "radial"], default="coordinate")
    p.add_argument("--endpoint-consistency-weight", type=float, default=0.0,
                   help="Weight for consistency between decoded trajectory endpoint and predicted endpoint")
    p.add_argument("--endpoint-correction", action="store_true",
                   help="Use a zero-initialized predicted-endpoint trajectory correction gate")
    p.add_argument("--goal-scale-m", type=float, default=1000.0)
    p.add_argument("--motion-decoder", choices=["acceleration", "coordinate", "ascent", "flight", "hybrid"], default="acceleration")
    p.add_argument("--ascent-pose", action="store_true",
                   help="Add ASCENT-style terminal position and orientation context after relation encoding")
    p.add_argument("--state-queries", action="store_true",
                   help="Add DeMo-style learnable per-future dynamic-state queries to the mode queries")
    p.add_argument("--max-future-steps", type=int, default=200,
                   help="Maximum future steps supported by state queries")
    p.add_argument("--future-relation", action="store_true",
                   help="Use history-only predicted goal tokens in a second joint relation block")
    p.add_argument("--intent-conditioning", choices=["none", "direction", "goal", "goal_relation"], default="none",
                   help="Explicit intent/endpoint conditioning variant")
    p.add_argument("--joint-score-temperature", type=float, default=1.0)
    p.add_argument("--joint-diversity-weight", type=float, default=0.0)
    p.add_argument("--init-checkpoint", default=None,
                   help="Weights-only warm-start of compatible tensors; NOT optimizer/RNG/epoch resume")
    p.add_argument("--save-training-state", action="store_true",
                   help="Atomically save resumable optimizer/RNG/epoch state after every complete epoch")
    p.add_argument("--resume-state", default=None,
                   help="Resume training_state.pt in its original run directory; inherits the saved scientific protocol")
    p.add_argument("--stop-at", default=None,
                   help="ISO datetime with timezone; pause at a complete epoch boundary and save training_state.pt")
    p.add_argument("--extend-epochs", type=int, default=None,
                   help="With resume, declare a NEW extension stage up to this total epoch budget, not original-plan exact resume")
    p.add_argument("--extension-lr", type=float, default=None,
                   help="Optional starting LR for the new extension stage; default is the saved optimizer LR")
    p.add_argument("--freeze-init", action="store_true",
                   help="Freeze parameters loaded from init checkpoint; train only newly added heads")
    p.add_argument("--freeze-prefixes", default="",
                   help="Comma-separated parameter-name prefixes to freeze after warm start")
    p.add_argument("--trainable-prefixes", default="",
                   help="Strict allowlist; all other parameters frozen. Each prefix must match.")
    p.add_argument("--evaluate-init", action="store_true", help="Evaluate and preserve epoch-zero checkpoint")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--decay-policy", choices=["legacy","selective"], default="legacy")
    return p


def validate_options(args):
    # Intent variants own their required endpoint/relation switches so that
    # experiment commands cannot silently construct an untrainable head.
    if args.intent_conditioning in ("goal", "goal_relation"):
        args.goal_head = True
        args.goal_conditioning = True
    if args.intent_conditioning == "goal_relation":
        args.future_relation = True
    if args.layers < 1 or not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("positive layers and finite nonnegative weight_decay required")
    if args.relation_fusion == "gated" and not args.interaction:
        raise ValueError("gated relation fusion requires interaction")
    if args.goal_conditioning and not args.goal_head:
        raise ValueError("goal-conditioning requires goal-head")
    if not np.isfinite(args.goal_weight) or args.goal_weight < 0:
        raise ValueError("goal-weight must be finite and nonnegative")
    if not np.isfinite(args.goal_soft_temperature) or args.goal_soft_temperature <= 0:
        raise ValueError("goal-soft-temperature must be finite and positive")
    if not np.isfinite(args.goal_radius_m) or args.goal_radius_m < 0:
        raise ValueError("goal-radius-m must be finite and nonnegative")
    if not np.isfinite(args.endpoint_consistency_weight) or args.endpoint_consistency_weight < 0:
        raise ValueError("endpoint-consistency-weight must be finite and nonnegative")
    if not np.isfinite(args.score_weight) or args.score_weight < 0:
        raise ValueError("score-weight must be finite and nonnegative")
    if args.architecture == "ascent" and (args.temporal_pool != "last" or args.relation_fusion != "serial"):
        raise ValueError("temporal pooling / relation fusion switches require trajectory architecture")
    if args.no_edge_geometry and (not args.interaction or args.architecture!="trajectory"):
        raise ValueError("--no-edge-geometry requires --interaction and --architecture trajectory")
    def _parse_index_list(raw, name):
        if raw is None:
            return None
        try:
            vals = [int(x.strip()) for x in str(raw).split(",") if x.strip() != ""]
        except ValueError:
            raise ValueError(f"{name} must be a comma-separated list of integers")
        if not vals:
            raise ValueError(f"{name} must contain at least one index")
        return vals
    args.edge_keep_anchors = _parse_index_list(args.edge_keep_anchors, "--edge-keep-anchors")
    args.edge_keep_feats = _parse_index_list(args.edge_keep_feats, "--edge-keep-feats")
    if (args.edge_keep_anchors is not None or args.edge_keep_feats is not None) and (
        not args.interaction or args.architecture != "trajectory"
    ):
        raise ValueError("--edge-keep-anchors/--edge-keep-feats require --interaction and --architecture trajectory")
    if args.edge_geometry_transform != "none" and (not args.interaction or args.architecture != "trajectory" or args.no_edge_geometry):
        raise ValueError("--edge-geometry-transform symmetric requires interaction with nonzero edge geometry")
    if args.joint_variant == "factorized" and not args.joint:
        raise ValueError("--joint-variant factorized requires --joint")
    if args.local_modes < 1:
        raise ValueError("--local-modes must be positive")
    if args.constraint_mode!="full" and not args.constraints:
        raise ValueError("--constraint-mode acceleration_only requires --constraints")
    if args.architecture=="ascent" and (args.interaction or args.joint or args.constraints or args.temporal!="transformer"):
        raise ValueError("ASCENT-adapted is an independent Transformer baseline without our motion constraints")
    if args.architecture=="ascent" and (args.motion_decoder != "acceleration" or args.ascent_pose):
        raise ValueError("ASCENT architecture selects its own decoder and pose path")
    if args.temporal_variant in ("conformer", "patchtst") and args.temporal != "transformer":
        raise ValueError("conformer and patchtst require --temporal transformer")
    if args.patch_len < 1 or args.patch_stride < 1:
        raise ValueError("patch-len and patch-stride must be positive")
    if not np.isfinite(args.joint_score_temperature) or args.joint_score_temperature <= 0:
        raise ValueError("joint-score-temperature must be finite and positive")
    if not np.isfinite(args.joint_diversity_weight) or args.joint_diversity_weight < 0:
        raise ValueError("joint-diversity-weight must be finite and nonnegative")


def load_verified_constraints(bounds_path, data_root, train):
    """Bind a training-history fit to the exact current source and cache."""
    data_root=Path(data_root).resolve()
    bounds_path=Path(bounds_path).resolve()
    fit=json.loads(bounds_path.read_text(encoding="utf-8"))
    if (ROOT/fit.get("data","")).resolve()!=data_root:
        raise ValueError("Motion bounds data source mismatch; refit on this dataset")
    expected_scope={"split":"train","observation":"history_endpoints","uses_future_values":False}
    if fit.get("fit_split")!="train" or fit.get("fit_scope")!=expected_scope:
        raise ValueError("Motion bounds require verified training-history-only fit scope")
    if train.manifest.get("split")!="train":
        raise ValueError("Motion bounds require a training cache")
    for key in ("protocol_and_index_hashes_verified","row_metadata_verified",
                "array_schema_verified","array_hashes_verified"):
        if fit.get("array_integrity",{}).get(key) is not True or train.integrity.get(key) is not True:
            raise ValueError(f"Motion bounds lack verified array integrity: {key}")
    sources={"dataset_config_sha256":data_root/"dataset_config.json",
             "training_scene_index_sha256":data_root/"scene_index.jsonl",
             "split_manifest_sha256":data_root/"split_manifest.csv",
             "cache_manifest_sha256":train.root/"manifest.json"}
    hashes={key:sha256_file(path) for key,path in sources.items()}
    for key,actual in hashes.items():
        if fit.get(key)!=actual:
            raise ValueError(f"Motion bounds {key} mismatch; refit on this dataset")
    for cache_key,fit_key in (("dataset_config_sha256","dataset_config_sha256"),
                              ("scene_index_sha256","training_scene_index_sha256")):
        if train.manifest.get(cache_key)!=hashes[fit_key]:
            raise ValueError(f"Motion bounds training cache {cache_key} mismatch")
    return fit["bounds"],dict(verified=True,fit_scope=expected_scope,
                             motion_bounds_sha256=sha256_file(bounds_path),**hashes)


def optimizer_groups(model, policy, weight_decay):
    """Respect official SSM no-decay markers and exempt scalar/vector offsets.

    Legacy is retained for exact historical comparisons; selecting this policy
    changes optimization, not the model, and must be separately registered.
    """
    if policy == "legacy":
        return [{"params": list(model.parameters()), "weight_decay": weight_decay}]
    if policy != "selective":
        raise ValueError("Unknown decay policy")
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        exempt = parameter.ndim < 2 or name.endswith("bias") or getattr(parameter, "_no_weight_decay", False)
        (no_decay if exempt else decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def apply_trainable_allowlist(model, prefixes_text):
    prefixes = tuple(x.strip().rstrip('.') for x in prefixes_text.split(',') if x.strip())
    params = dict(model.named_parameters())
    matches = {p: [n for n in params if n == p or n.startswith(p + '.')] for p in prefixes}
    if any(not names for names in matches.values()):
        raise ValueError(f"Unmatched trainable prefixes: {[p for p, names in matches.items() if not names]}")
    allowed = {name for names in matches.values() for name in names}
    if not allowed:
        raise ValueError("Trainable allowlist is empty")
    for name, param in params.items():
        param.requires_grad_(name in allowed)
    return {"matches": matches, "trainable_tensors": len(allowed),
            "frozen_tensors": len(params) - len(allowed)}


def main():
    argv = sys.argv[1:]
    args, resume_state, resume_path = resolve_resume_args(build_parser().parse_args(argv), argv)
    stop_at = parse_stop_at(args.stop_at)
    save_state = args.save_training_state or stop_at is not None or resume_state is not None
    validate_final_dataset_protocol(args)
    validate_options(args)
    if args.device=="cuda" and not torch.cuda.is_available():raise RuntimeError("Requested CUDA is unavailable; use verified runtime")
    torch.set_num_threads(4);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed);torch.cuda.reset_peak_memory_stats()
    train=PackedSceneDataset(ROOT/args.data,"train",args.horizon)
    validation=PackedSceneDataset(ROOT/args.data,"validation",args.horizon,
                                 index_name=("reference_scene_index.jsonl" if args.validation_cache_split == "validation_reference"
                                              else args.validation_index_name),
                                 cache_split=args.validation_cache_split)
    constraints=None;constraints_provenance=None
    if args.constraints:
        constraints,constraints_provenance=load_verified_constraints(ROOT/args.constraints,ROOT/args.data,train)
    model_config={"width":args.width,"layers":args.layers,"heads":args.heads,"modes":args.modes,"interaction":args.interaction,
                  "joint":args.joint,"joint_variant":args.joint_variant,"local_modes":args.local_modes,
                  "temporal":args.temporal,"constraints":constraints,"edge_anchors":3}
    if args.joint_score_temperature != 1.0:model_config["joint_score_temperature"] = args.joint_score_temperature
    if args.joint_diversity_weight > 0:model_config["joint_diversity_weight"] = args.joint_diversity_weight
    if args.endpoint_correction:model_config["endpoint_correction"] = True
    if args.temporal_variant != "vanilla":
        model_config["temporal_variant"] = args.temporal_variant
    if args.temporal_variant == "patchtst":
        model_config["patch_len"] = args.patch_len
        model_config["patch_stride"] = args.patch_stride
    if args.temporal_variant == "mamba_patch":
        model_config["patch_len"] = args.patch_len
        model_config["patch_stride"] = args.patch_stride
        model_config["fusion_mode"] = args.fusion_mode
    # Omit default keys so existing model configs still reconstruct exactly.
    if args.no_edge_geometry:model_config["use_edge_geometry"]=False
    if args.edge_geometry_transform != "none":model_config["edge_geometry_transform"] = args.edge_geometry_transform
    if args.edge_keep_anchors is not None:model_config["edge_keep_anchors"]=args.edge_keep_anchors
    if args.edge_keep_feats is not None:model_config["edge_keep_feats"]=args.edge_keep_feats
    if args.constraint_mode!="full":model_config["constraint_mode"]=args.constraint_mode
    if args.temporal_pool!="last":model_config["temporal_pool"]=args.temporal_pool
    if args.relation_fusion!="serial":model_config["relation_fusion"]=args.relation_fusion
    if args.goal_head:model_config["goal_head"]=True
    if args.goal_conditioning:model_config["goal_conditioning"]=True
    if args.goal_head and args.goal_scale_m != 1000.0:model_config["goal_scale_m"]=args.goal_scale_m
    if args.motion_decoder != "acceleration":model_config["motion_decoder"]=args.motion_decoder
    if args.ascent_pose:model_config["ascent_pose"]=True
    if args.state_queries:model_config["state_queries"]=True
    if args.state_queries and args.max_future_steps != 200:model_config["max_future_steps"]=args.max_future_steps
    if args.future_relation:model_config["future_relation"]=True
    if args.intent_conditioning != "none":model_config["intent_conditioning"]=args.intent_conditioning
    if args.intent_conditioning == "goal":model_config["goal_conditioning"]=True
    if args.intent_conditioning == "goal_relation":
        model_config["goal_conditioning"]=True
        model_config["future_relation"]=True
    if args.architecture=="ascent":
        model_config={"width":args.width,"layers":args.layers,"heads":args.heads,"modes":args.modes,
                      "history_steps":int(train.arrays["history_features"].shape[1]),
                      "future_steps":train.h,"sample_rate_hz":train.manifest["sample_rate_hz"]}
    split_manifest_path=ROOT/args.data/"split_manifest.csv"
    if not split_manifest_path.exists():
        split_manifest_path=ROOT/args.data/"scene_index.jsonl"
    config=dict(vars(args),model=model_config,training_windows=len(train),validation_windows=len(validation),
        dataset_config_sha256=sha256_file(ROOT/args.data/"dataset_config.json"),
        training_scene_index_sha256=sha256_file(ROOT/args.data/"scene_index.jsonl"),
        validation_scene_index_sha256=sha256_file(ROOT/args.data/args.validation_index_name),
        split_manifest_path=str(split_manifest_path.relative_to(ROOT)),
        split_manifest_sha256=sha256_file(split_manifest_path),
        train_cache_manifest_sha256=sha256_file(train.root/"manifest.json"),validation_cache_manifest_sha256=sha256_file(validation.root/"manifest.json"),
        sampler="uniform_candidate_prefix_cluster_then_replay_then_endpoint",source_independence_verified=False,
        final_test_enabled=False,selection="validation_candidate_prefix_macro_scene_minADE",precision="float32",
        torch_version=torch.__version__,cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    config["use_edge_geometry"]=args.interaction and not args.no_edge_geometry
    if constraints_provenance is not None:
        config["constraints_provenance"]=constraints_provenance
        config["active_constraint_bounds"]=["Amax"] if args.constraint_mode=="acceleration_only" else ["Vmax","Wmax","Amax"]
        config["velocity_projection_enabled"]=args.constraint_mode=="full"
    files=[Path(__file__)]+([ROOT/args.constraints] if args.constraints else [])
    if resume_state is None:
        run_dir,record=start_run(args.name,config,files)
        stages = [initial_schedule(args)]
    else:
        run_dir, record = validate_resume_artifacts(resume_state, resume_path, config)
        stages = list(resume_state["schedule_stages"])
        if args.extend_epochs is not None:
            stages.append({"kind": "extension_cosine", "start_epoch": resume_state["epoch"],
                           "end_epoch": args.epochs, "base_lr": args.extension_lr or resume_state["optimizer"]["param_groups"][0]["lr"],
                           "minimum_lr": 1e-6,
                           "evidence": "additional_training_stage_not_exact_original_schedule_continuation"})
        record.setdefault("resume_history", []).append({
            "resumed_utc": datetime.now(timezone.utc).isoformat(), "from_epoch": resume_state["epoch"],
            "state_sha256": sha256_file(resume_path), "stop_at": args.stop_at,
            "semantics": "new_extension_stage" if args.extend_epochs is not None else "exact_epoch_boundary_resume",
            "target_epochs": args.epochs, "command_argv": sys.argv})
        record["status"] = "running"
        record.pop("finished_utc", None)
        record["planned_epochs_current"] = args.epochs
        write_json(run_dir / "record.json", record)
    print(json.dumps({"run_id":record["run_id"],"model":model_config,"gpu":config["gpu"]}),flush=True)
    start=time.perf_counter()
    previous_elapsed = float(resume_state["elapsed_seconds"]) if resume_state is not None else 0.0
    try:
        device=torch.device(args.device)
        if args.architecture=="ascent":
            from trajectory_prediction.ascent_adapter import ASCENTAdapter
            model=ASCENTAdapter(**model_config).to(device)
        else:model=TrajectoryModel(**model_config).to(device)
        if args.init_checkpoint and resume_state is None:
            init_path = Path(args.init_checkpoint)
            if not init_path.is_absolute():
                init_path = ROOT / init_path
            init = torch.load(init_path, map_location="cpu", weights_only=True)
            record["initialization_sha256"] = sha256_file(init_path)
            init_config = init.get("model_config", {})
            if init_config.get("temporal") != model_config.get("temporal") or init_config.get("interaction") != model_config.get("interaction"):
                raise ValueError("init checkpoint temporal/interaction configuration mismatch")
            current = model.state_dict()
            loaded = 0
            for key, value in init.get("model", {}).items():
                if key in current and current[key].shape == value.shape:
                    current[key] = value
                    loaded += 1
            model.load_state_dict(current, strict=True)
            print(json.dumps({"warm_start": str(init_path), "loaded_parameter_tensors": loaded}), flush=True)
            if args.freeze_init:
                loaded_keys = set(init.get("model", {}).keys())
                frozen = 0
                for key, parameter in model.named_parameters():
                    if key in loaded_keys:
                        parameter.requires_grad_(False)
                        frozen += 1
                print(json.dumps({"warm_start_frozen_parameter_tensors": frozen}), flush=True)
            if args.freeze_prefixes:
                prefixes = tuple(x.strip() for x in args.freeze_prefixes.split(",") if x.strip())
                frozen = 0
                for key, parameter in model.named_parameters():
                    if key.startswith(prefixes):
                        parameter.requires_grad_(False)
                        frozen += 1
                print(json.dumps({"prefix_frozen_parameter_tensors": frozen, "prefixes": prefixes}), flush=True)
        if args.trainable_prefixes:
            print(json.dumps({"trainability": apply_trainable_allowlist(model, args.trainable_prefixes)}), flush=True)
        if resume_state is not None:
            if set(resume_state["trainability"]) != set(dict(model.named_parameters())):
                raise ValueError("Resume parameter names differ from the saved trainability contract")
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(resume_state["trainability"][name])
        trainability = {name: {"trainable": p.requires_grad, "elements": p.numel()}
                        for name, p in model.named_parameters()}
        write_json(run_dir/"trainability.json", trainability)
        write_json(run_dir/"record.json", record)
        parameters=sum(p.numel() for p in model.parameters())
        optimizer=torch.optim.AdamW(optimizer_groups(model,args.decay_policy,args.weight_decay),lr=args.lr,weight_decay=args.weight_decay)
        loader_options=dict(batch_size=args.batch_size,collate_fn=collate_scenes,num_workers=args.num_workers,pin_memory=args.device=="cuda")
        val_loader=DataLoader(validation,shuffle=False,**loader_options)
        best=float("inf");best_epoch=0;curve=[];summary=None;epoch=0
        if resume_state is not None:
            model.load_state_dict(resume_state["model"], strict=True)
            optimizer.load_state_dict(resume_state["optimizer"])
            epoch = int(resume_state["epoch"])
            best, best_epoch = resume_state["best"], resume_state["best_epoch"]
            curve, summary = list(resume_state["curve"]), resume_state["last_summary"]
            restore_rng_state(resume_state["rng"])
        if args.evaluate_init and resume_state is None:
            initial_summary = evaluate(model, val_loader, device, args.modes)
            best = initial_summary["scene_minADE_m"]
            write_json(run_dir/"initial_validation_summary.json", initial_summary)
            write_json(run_dir/"best_validation_summary.json", initial_summary)
            torch.save({"model":model.state_dict(),"model_config":model_config,"epoch":0,
                        "validation_metric":best}, run_dir/"best.pt")
            print(json.dumps({"epoch":0,"scene_minADE_m":best,
                              "goal_diagnostics":compact_goal_diagnostics(initial_summary)}),flush=True)
        state_path = run_dir / "training_state.pt"

        def checkpoint_training_boundary():
            save_training_state(state_path, model=model, model_config=model_config, optimizer=optimizer,
                epoch=epoch, best=best, best_epoch=best_epoch, curve=curve, args=args, stages=stages,
                record=record, summary=summary, elapsed_seconds=previous_elapsed + time.perf_counter() - start)

        def pause_training():
            checkpoint_training_boundary()
            progress = {"status": "paused", "pause_reason": "stop_at_epoch_boundary", "stop_at": args.stop_at,
                        "epoch": epoch, "best_epoch": best_epoch,
                        "best_validation_scene_minADE_m": best if np.isfinite(best) else None,
                        "resumable_state": str(state_path), "training_state_sha256": sha256_file(state_path)}
            write_json(run_dir / "progress.json", progress)
            finish_run(run_dir, record, "paused", epochs_run=epoch, best_epoch=best_epoch,
                       elapsed_seconds=previous_elapsed + time.perf_counter() - start,
                       parameter_count=parameters, **{k:v for k,v in progress.items() if k not in ("status", "epoch", "best_epoch")})
            print(json.dumps({"paused": record["run_id"], **progress}), flush=True)

        if deadline_reached(stop_at) and epoch < args.epochs:
            pause_training()
            return
        for epoch in range(epoch+1,args.epochs+1):
            epoch_start=time.perf_counter();model.train()
            indices=draw_epoch(train,args.seed,epoch,args.samples_per_group)
            loader=DataLoader(train,sampler=indices,**loader_options)
            losses=[];modes=np.zeros(args.modes,dtype=np.int64);skipped=0
            component_losses=defaultdict(list)
            # Fixed 5-epoch warmup, then cosine schedule, identical for comparison models.
            for group in optimizer.param_groups:group["lr"]=float(scheduled_learning_rate(epoch, stages[-1]))
            for raw in loader:
                batch=move_batch(raw,device);optimizer.zero_grad(set_to_none=True)
                output=model(batch["history_features"],batch["actor_mask"],batch["forecast_times"],batch["history_context"],batch["history_edges"])
                goal_target_local = goal_valid = None
                intent_target_direction = intent_valid = None
                if args.goal_weight > 0 or args.intent_conditioning == "direction":
                    context = batch["history_context"]
                    delta = batch["target"] - context["p0_shared"][:, :, None, :]
                    # History/context positions use row-vector local = shared @ R.
                    # Convert the future endpoint with the same convention.
                    goal_target_local = torch.einsum("bnij,bnhi->bnhj", context["R_local_to_shared"], delta)[:, :, -1]
                    goal_valid = batch["future_valid"][:, :, -1] & batch["actor_mask"]
                    if args.intent_conditioning == "direction":
                        intent_target_direction = goal_target_local / goal_target_local.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        intent_valid = goal_valid & torch.isfinite(intent_target_direction).all(dim=-1)
                loss,diag=trajectory_loss(output,batch["target"],batch["future_valid"],batch["actor_mask"],joint=args.joint,
                                          score_weight=args.score_weight,
                                          goal_weight=args.goal_weight,goal_target_local=goal_target_local,goal_valid=goal_valid,
                                          goal_soft_temperature=args.goal_soft_temperature,
                                          goal_radius_m=args.goal_radius_m,
                                          goal_loss_geometry=args.goal_loss_geometry,
                                          endpoint_consistency_weight=args.endpoint_consistency_weight,
                                          intent_target_direction=intent_target_direction,intent_valid=intent_valid)
                if not diag["n_supervised_scenes"]:skipped+=1;continue
                if not torch.isfinite(loss):raise FloatingPointError("Nonfinite trajectory loss")
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise FloatingPointError("Nonfinite gradients")
                optimizer.step();losses.append(float(loss.detach()));modes+=np.asarray(diag["mode_counts"])
                for key in ("loss_trajectory", "loss_score", "loss_goal", "loss_endpoint_consistency"):
                    component_losses[key].append(diag[key])
            summary=evaluate(model,val_loader,device,args.modes)
            metric=summary["scene_minADE_m"]
            if metric is None:raise RuntimeError("No eligible validation scenes")
            line={"epoch":epoch,"training_loss":float(np.mean(losses)),"validation_scene_minADE_m":metric,
                  "validation_top_score_ADE_m":summary["top_score_ADE_m"],"validation_scene_minFDE_m":summary["scene_minFDE_m"],
                  "mode_assignment_counts":modes.tolist(),"skipped_unsupervised_batches":skipped,
                  "seconds":time.perf_counter()-epoch_start,"lr":optimizer.param_groups[0]["lr"]}
            line["loss_components"]={key:float(np.mean(values)) for key,values in component_losses.items()}
            line["weighted_goal_loss"]=args.goal_weight * line["loss_components"]["loss_goal"]
            line["weighted_endpoint_consistency_loss"] = args.endpoint_consistency_weight * line["loss_components"]["loss_endpoint_consistency"]
            if "goal_diagnostics" in summary:
                line["goal_diagnostics"]=compact_goal_diagnostics(summary)
            curve.append(line)
            with (run_dir/"learning_curve.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(line)+"\n")
            if metric<best:
                best=metric;best_epoch=epoch
                torch.save({"model":model.state_dict(),"model_config":model_config,"epoch":epoch,"validation_metric":metric},run_dir/"best.pt")
                write_json(run_dir/"best_validation_summary.json",summary)
            print(json.dumps(dict(line,best_epoch=best_epoch,run_id=record["run_id"])),flush=True)
            write_json(run_dir/"progress.json",{"status":"running","epoch":epoch,"best_epoch":best_epoch,"best_validation_scene_minADE_m":best})
            if save_state:
                checkpoint_training_boundary()
            if epoch-best_epoch>=args.patience:break
            if deadline_reached(stop_at) and epoch < args.epochs:
                pause_training()
                return
        write_json(run_dir/"last_validation_summary.json",summary)
        torch.save({"model":model.state_dict(),"model_config":model_config,"epoch":len(curve),
                    "validation_metric":summary["scene_minADE_m"]},run_dir/"last.pt")
        checkpoint=torch.load(run_dir/"best.pt",map_location=device,weights_only=True)
        model.load_state_dict(checkpoint["model"])
        summary=evaluate(model,val_loader,device,args.modes,save_path=run_dir)
        write_json(run_dir/"validation_summary.json",summary)
        write_json(run_dir/"progress.json",{"status":"completed","epoch":len(curve),"best_epoch":best_epoch,
                   "best_validation_scene_minADE_m":summary["scene_minADE_m"]})
        finish_run(run_dir,record,"completed",epochs_run=len(curve),best_epoch=best_epoch,parameter_count=parameters,
            elapsed_seconds=previous_elapsed+time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            checkpoint_sha256=sha256_file(run_dir/"best.pt"),validation_summary=summary)
        print(json.dumps({"completed":record["run_id"],"best_epoch":best_epoch,"scene_minADE_m":summary["scene_minADE_m"],
                          "top_score_ADE_m":summary["top_score_ADE_m"],"seconds":time.perf_counter()-start}),flush=True)
    except Exception:
        write_json(run_dir/"progress.json",{"status":"failed","error":traceback.format_exc()})
        finish_run(run_dir,record,"failed",elapsed_seconds=time.perf_counter()-start,error=traceback.format_exc());raise


if __name__=="__main__":main()
