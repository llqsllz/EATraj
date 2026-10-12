"""Explicit-time TrajAir benchmark bridge; history-only features, SI internals.

Official cache supplies windows/actors, including its retrospective completeness
filter. This module does not call that an online deployment selection policy.
Cache agent_id is deliberately unused (upstream filtered-ID metadata bug).
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class OfficialCacheDataset(Dataset):
    def __init__(self, cache_path, obs_step=5):
        self.cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.obs_step = obs_step
        self.ranges = self.cache["seq_start_end"]

    def __len__(self):
        return len(self.ranges)

    def __getitem__(self, index):
        start, end = self.ranges[index]
        return (self.cache["obs_traj"][start:end, :, ::self.obs_step].transpose(1, 2),
                self.cache["pred_traj"][start:end].transpose(1, 2), index)


def collate_official(items):
    counts = [len(item[0]) for item in items]
    b, n = len(items), max(counts)
    obs = torch.zeros(b, n, *items[0][0].shape[1:])
    target = torch.zeros(b, n, *items[0][1].shape[1:])
    mask = torch.zeros(b, n, dtype=torch.bool)
    for i, (x, y, _) in enumerate(items):
        obs[i, :len(x)] = x * 1000  # official positions in km
        target[i, :len(x)] = y * 1000
        mask[i, :len(x)] = True
    return obs, target, mask, torch.tensor([x[2] for x in items])


def history_inputs(positions, actor_mask, history_dt):
    """Use only observed positions [B,N,T,3] in metres, sampled every dt seconds.

    Backward first differences use no unobserved 1-Hz positions. The first
    velocity is missing/zero. Rotate into each actor's latest reliable yaw.
    Feature and edge columns follow data.make_history; acquisition interval is
    explicit, not the simulation adapter's hard-coded frame_idx/10.
    """
    if history_dt <= 0 or positions.shape[-2] < 2:
        raise ValueError("positive history_dt and at least two observations required")
    b, n, t, _ = positions.shape
    p = torch.where(actor_mask[..., None, None], positions, torch.zeros_like(positions))
    v = torch.zeros_like(p)
    v[:, :, 1:] = torch.diff(p, dim=2) / history_dt
    vm = actor_mask[..., None].expand(b, n, t).clone()
    vm[:, :, 0] = False
    good = vm & (torch.linalg.vector_norm(v[..., :2], dim=-1) >= 1.)
    indices = torch.arange(t, device=p.device).expand(b, n, t)
    last = torch.where(good, indices, 0).max(dim=-1).values
    heading_v = v.gather(2, last[..., None, None].expand(b, n, 1, 3)).squeeze(2)
    angle = torch.atan2(heading_v[..., 1], heading_v[..., 0])
    c, s = torch.cos(angle), torch.sin(angle)
    r = torch.zeros(b, n, 3, 3, device=p.device, dtype=p.dtype)
    r[..., 0, 0] = c; r[..., 0, 1] = -s
    r[..., 1, 0] = s; r[..., 1, 1] = c; r[..., 2, 2] = 1
    p0 = p[:, :, -1]
    local_p = torch.einsum("bntc,bncd->bntd", p - p0[:, :, None], r)
    local_v = torch.einsum("bntc,bncd->bntd", v, r)
    features = p.new_zeros(b, n, t, 13)
    features[..., :3] = local_p / 1000
    features[..., 3:6] = local_v / 100
    features[..., 6] = actor_mask[..., None]
    features[..., 7] = vm
    features[..., 8] = 0.  # no history coordinate was imputed
    features[..., 9] = good.any(dim=-1)[..., None]
    features[..., 10] = 0.  # every benchmark input is an observed sample
    features[..., 11] = history_dt / 10
    features[..., 12] = (torch.arange(t, device=p.device) - (t - 1)) * history_dt / 10
    features = torch.where(actor_mask[..., None, None], features, torch.zeros_like(features))
    anchors = [0, (t - 1) // 2, t - 1]
    pa, va = p[:, :, anchors], v[:, :, anchors]
    rp = pa[:, None] - pa[:, :, None]
    rv = va[:, None] - va[:, :, None]
    vv = vm[:, None, :, anchors] & vm[:, :, None, anchors]
    distance = torch.linalg.vector_norm(rp, dim=-1)
    speed2 = (rv ** 2).sum(dim=-1)
    dot = (rp * rv).sum(dim=-1)
    ok = vv & (speed2 >= .01)
    tcpa = torch.where(ok, (-dot / speed2.clamp_min(.01)).clamp(0, 15), 0.)
    dcpa = torch.linalg.vector_norm(rp + tcpa[..., None] * rv, dim=-1)
    closing = torch.where(vv, -dot / distance.clamp_min(1e-8), 0.)
    rp_local = torch.einsum("bijac,bicd->bijad", rp, r)
    rv_local = torch.einsum("bijac,bicd->bijad", rv, r)
    edges = torch.cat((rp_local / 1000, torch.where(vv[..., None], rv_local / 100, 0.),
                       distance[..., None] / 1000, closing[..., None] / 100,
                       tcpa[..., None] / 15, dcpa[..., None] / 1000,
                       ok[..., None], torch.zeros_like(ok[..., None]), vv[..., None]), dim=-1)
    pair_mask = actor_mask[:, :, None] & actor_mask[:, None, :]
    edges = torch.where(pair_mask[..., None, None], edges, torch.zeros_like(edges))
    return features, {"p0_shared": p0, "v0_local": local_v[:, :, -1],
                      "R_local_to_shared": r}, edges


def marginal_errors(predictions, target, actor_mask):
    """Metres; independent minADE/minFDE per actor (official scoring semantics)."""
    distance = torch.linalg.vector_norm(predictions - target[:, None], dim=-1)
    ade = distance.mean(dim=-1).min(dim=1).values
    fde = distance[..., -1].min(dim=1).values
    return ade[actor_mask], fde[actor_mask]
