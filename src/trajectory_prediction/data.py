"""Pure-trajectory windows. Targets never participate in feature generation."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .geometry import trailing_velocity, endpoint_rotation


def resolve_time_grid(config: dict, horizon_seconds: float | None = None) -> dict:
    """Validate the canonical 10 Hz adapter's exact output grid.

    Reject unsupported times instead of evaluating a nearby, differently rounded
    target in the raw and packed loaders. A different source rate needs its own
    adapter: make_history still derives canonical time from frame_idx / 10.
    """
    def positive(value, name):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be positive and finite")
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
        return value

    def aligned_steps(seconds, rate, name):
        seconds = positive(seconds, name)
        steps = seconds * rate
        rounded = round(steps)
        if rounded < 1 or not np.isclose(steps, rounded, atol=1e-7, rtol=0):
            raise ValueError(f"{name} must align exactly with the {rate:g} Hz grid")
        return rounded

    source_rate = positive(config["source_rate_hz"], "source_rate_hz")
    sample_rate = positive(config["sample_rate_hz"], "sample_rate_hz")
    if not np.isclose(source_rate, 10., atol=1e-9, rtol=0):
        raise ValueError("This canonical frame adapter requires source_rate_hz=10")
    ratio = source_rate / sample_rate
    frame_step = round(ratio)
    if frame_step < 1 or not np.isclose(ratio, frame_step, atol=1e-9, rtol=0):
        raise ValueError("source_rate_hz / sample_rate_hz must be an integer")
    history_steps = aligned_steps(config["history_seconds"], sample_rate, "history_seconds")
    max_future_steps = aligned_steps(config["max_forecast_seconds"], sample_rate, "max_forecast_seconds")
    chosen = config["max_forecast_seconds"] if horizon_seconds is None else horizon_seconds
    future_steps = aligned_steps(chosen, sample_rate, "horizon_seconds")
    if future_steps > max_future_steps:
        raise ValueError("Requested horizon exceeds stored protocol")
    return {"sample_rate_hz": sample_rate, "frame_step": frame_step,
            "history_frames": history_steps * frame_step, "history_steps": history_steps + 1,
            "future_frames": future_steps * frame_step, "future_steps": future_steps,
            "max_future_steps": max_future_steps, "horizon_seconds": future_steps / sample_rate}


def history_actors(track: dict, endpoint: int, history_frames: int, frame_step: int) -> np.ndarray:
    begin = endpoint-history_frames
    if begin < 0 or endpoint >= len(track["frames"]):
        return np.empty(0,dtype=int)
    # Eligibility uses full source-rate history; sampled points alone cannot hide gaps.
    good = track["valid"][begin:endpoint+1].all(axis=0)
    good &= track["segment"][begin] == track["segment"][endpoint]
    return np.flatnonzero(good)


def make_history(track: dict, endpoint: int, actors: np.ndarray, history_frames: int,
                 frame_step: int, velocity_config: dict) -> dict:
    start = endpoint-history_frames
    source_positions = track["position"][start:endpoint+1,actors,:].astype(np.float64)
    t_raw = track["frames"][start:endpoint+1].astype(float)/10.
    t_hist = (t_raw[::frame_step]-t_raw[-1]).astype(np.float32)
    n, t = len(actors), len(t_hist)
    p0 = source_positions[-1].copy()
    rotations, velocities, vel_valid, feature_list, headings = [], [], [], [], []
    for i in range(n):
        v, vm = trailing_velocity(t_raw, source_positions[:,i],
            window_s=velocity_config["window_seconds"],min_samples=velocity_config["min_samples"],
            min_span_s=velocity_config["min_span_seconds"])
        r, heading = endpoint_rotation(v,vm)
        p = (source_positions[::frame_step,i]-p0[i])@r
        vl = v[::frame_step]@r
        features = np.column_stack((p/1000.,vl/100.,np.ones(t),vm[::frame_step],np.zeros(t),
            np.full(t,float(heading)),np.zeros(t),np.full(t,.1/10.),t_hist/10.))
        feature_list.append(features)
        rotations.append(r)
        velocities.append(v[::frame_step])
        vel_valid.append(vm[::frame_step])
        headings.append(heading)
    r = np.asarray(rotations)
    v_shared = np.asarray(velocities)
    velocity_mask = np.asarray(vel_valid)
    anchors = [0,(t-1)//2,t-1]
    p_shared = source_positions[::frame_step].transpose(1,0,2)
    edges = np.zeros((n,n,len(anchors),13),dtype=np.float32)
    for ai,anchor in enumerate(anchors):
        # i is receiver, j sender, vectors rotated with receiver's fixed frame.
        rel_p = p_shared[None,:,anchor,:]-p_shared[:,None,anchor,:]
        rel_v = v_shared[None,:,anchor,:]-v_shared[:,None,anchor,:]
        vv = velocity_mask[:,anchor,None]&velocity_mask[None,:,anchor]
        rp = np.einsum("ijc,icd->ijd",rel_p,r)
        rv = np.einsum("ijc,icd->ijd",rel_v,r)
        dist = np.linalg.norm(rel_p,axis=-1)
        speed2 = np.sum(rel_v**2,axis=-1)
        tcpa_ok = vv & (speed2>=.1**2)
        dot = np.sum(rel_p*rel_v,axis=-1)
        tcpa = np.zeros_like(dist)
        tcpa[tcpa_ok] = np.clip(-dot[tcpa_ok]/speed2[tcpa_ok],0,15)
        dcpa = np.linalg.norm(rel_p+tcpa[...,None]*rel_v,axis=-1)
        closing = np.where(vv,-dot/np.maximum(dist,1e-8),0)
        rv[~vv]=0
        edges[:,:,ai] = np.concatenate((rp/1000.,rv/100.,dist[...,None]/1000.,closing[...,None]/100.,
            tcpa[...,None]/15.,dcpa[...,None]/1000.,tcpa_ok[...,None],np.zeros((n,n,1)),vv[...,None]),axis=-1)
    return {"history_features":np.asarray(feature_list,dtype=np.float32),
            "actor_mask":np.ones(n,dtype=bool),"history_mask":np.ones((n,t),dtype=bool),
            "history_edges":edges,"history_context":{
                "p0_shared":p0.astype(np.float32),
                "v0_local":np.einsum("nc,ncd->nd",v_shared[:,-1],r).astype(np.float32),
                "R_local_to_shared":r.astype(np.float32)},
            "history_position_shared":p_shared.astype(np.float64),
            "history_velocity_shared":v_shared.astype(np.float64),
            "history_velocity_valid":velocity_mask,"history_times":t_hist}


def make_targets(track: dict, endpoint: int, actors: np.ndarray, horizon_frames: int, frame_step: int) -> tuple[np.ndarray,np.ndarray]:
    offsets=np.arange(frame_step,horizon_frames+1,frame_step)
    indices=endpoint+offsets
    inside=indices<len(track["frames"])
    safe=np.minimum(indices,len(track["frames"])-1)
    pos=track["position"][safe][:,actors,:].transpose(1,0,2).copy()
    mask=track["valid"][safe][:,actors].T & inside[None,:]
    mask &= track["segment"][safe][:,actors].T == track["segment"][endpoint,actors,None]
    pos[~mask]=0
    return pos.astype(np.float32),mask


class SceneDataset:
    """NPZ + JSONL data works in both data and GPU runtimes without PyArrow."""
    def __init__(self, root: Path | str, split: str, horizon_seconds: float | None = None,
                 *, index_name: str = "scene_index.jsonl"):
        self.root=Path(root)
        self.index_name=index_name
        self.config=json.loads((self.root/"dataset_config.json").read_text(encoding="utf-8"))
        self.rows=[json.loads(l) for l in (self.root/index_name).read_text(encoding="utf-8").splitlines() if l]
        self.rows=[r for r in self.rows if r["split"]==split]
        grid=resolve_time_grid(self.config,horizon_seconds)
        self.horizon_seconds=grid["horizon_seconds"]
        self.frame_step=grid["frame_step"]
        self.history_frames=grid["history_frames"]
        self.future_frames=grid["future_frames"]

    @lru_cache(maxsize=256)
    def track(self,replay_id):
        with np.load(self.root/"tracks"/(replay_id+".npz"),allow_pickle=False) as f:
            return {k:f[k] for k in f.files}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        track=self.track(row["replay_id"])
        actors=np.asarray(row["actor_indices"],dtype=int)
        sample=make_history(track,row["endpoint_index"],actors,self.history_frames,self.frame_step,self.config["velocity"])
        target,valid=make_targets(track,row["endpoint_index"],actors,self.future_frames,self.frame_step)
        sample.update(target=target,future_valid=valid,metadata=row,
            forecast_times=np.arange(1,target.shape[1]+1,dtype=np.float32)/self.config["sample_rate_hz"])
        return sample
