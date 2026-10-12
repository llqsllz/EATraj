"""Map-free FJMP-style joint predictor for multi-aircraft scenes.

This is an explicit aviation adaptation of FJMP, not the official road-map
implementation.  It keeps the two identifying ideas of FJMP: a learned
three-way directed interaction header and DAG-ordered joint decoding.  The
road/lane branch is replaced by the existing aircraft pair geometry cache and
the output remains the project's 3-D metre-space protocol.
"""
from __future__ import annotations

from typing import Dict, Any
import torch
from torch import Tensor, nn

from .models import integrate_acceleration, _positive_times


class FJMPAviationModel(nn.Module):
    """FJMP-style learned DAG interaction model for cached aviation scenes."""

    def __init__(self, width=64, layers=2, heads=4, modes=6,
                 edge_anchors=3, constraints=None):
        super().__init__()
        if width % heads or min(width, layers, heads, modes, edge_anchors) < 1:
            raise ValueError("invalid FJMP aviation dimensions")
        self.width, self.modes, self.edge_anchors = width, modes, edge_anchors
        self.constraints = constraints
        self.feature_projection = nn.Linear(13, width)
        self.history_time = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width))
        self.temporal = nn.GRU(width, width, num_layers=layers, batch_first=True)
        self.temporal_norm = nn.LayerNorm(width)
        self.edge_projection = nn.Sequential(
            nn.Linear(13 * edge_anchors, width), nn.GELU(), nn.Linear(width, width)
        )
        # FJMP relation header: no relation, i influences j, j influences i.
        self.relation_header = nn.Sequential(
            nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 3)
        )
        self.priority = nn.Linear(width, 1)
        self.message = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
        self.scene_queries = nn.Parameter(torch.randn(modes, width) * 0.02)
        self.scene_score = nn.Linear(width, 1)
        self.decoder = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))
        self.future_time = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width))

    def _relation(self, actors: Tensor, edges: Tensor, mask: Tensor):
        b, n, d = actors.shape
        e = self.edge_projection(edges.flatten(-2))
        src = actors[:, :, None, :].expand(-1, -1, n, -1)
        dst = actors[:, None, :, :].expand(-1, n, -1, -1)
        pair = torch.cat((src, dst, e), dim=-1)
        logits = self.relation_header(pair)
        valid = mask[:, :, None] & mask[:, None, :]
        eye = torch.eye(n, device=actors.device, dtype=torch.bool)[None]
        valid = valid & ~eye
        logits = logits.masked_fill(~valid[..., None], -1e4)
        # Self pairs cannot be an interaction.
        logits = logits.masked_fill(eye[..., None], -1e4)
        logits[..., 0] = torch.where(eye, torch.zeros_like(logits[..., 0]), logits[..., 0])
        # A scalar order makes the selected directed graph acyclic.  The
        # relation header still learns the edge type; the order is a
        # differentiable auxiliary representation used by the decoder.
        order = self.priority(actors).squeeze(-1)
        earlier = order[:, :, None] < order[:, None, :]
        # Keep only edges that agree with the learned topological order.  A
        # class-1 score denotes src -> dst; class-2 is interpreted as the
        # reverse proposal and is transposed into the same canonical edge.
        p = torch.softmax(logits, dim=-1)
        adjacency = torch.where(
            earlier, p[..., 1], p[..., 2]
        ) * earlier.float() * valid.float()
        return logits, adjacency, order

    def _dag_message(self, actors: Tensor, adjacency: Tensor, order: Tensor, mask: Tensor):
        # Sort by learned priority and process parents before children.  The
        # operation is intentionally explicit because the scene sizes are
        # small (2v2/5v5) and this mirrors FJMP's DAG traversal.
        b, n, d = actors.shape
        result = actors.clone()
        for bi in range(b):
            idx = torch.argsort(order[bi])
            state = actors[bi].clone()
            for pos in idx.tolist():
                if not bool(mask[bi, pos]):
                    continue
                parents = adjacency[bi, :, pos] * mask[bi].float()
                denom = parents.sum().clamp_min(1e-6)
                aggregate = (parents[:, None] * state).sum(0) / denom
                has_parent = parents.sum() > 1e-6
                update = self.message(torch.cat((state[pos], aggregate), dim=-1))
                # Avoid in-place writes into a tensor participating in the
                # autograd graph; rebuilding the state preserves gradients.
                replacement = state[pos] + torch.where(has_parent, update, torch.zeros_like(update))
                state = torch.cat((state[:pos], replacement[None], state[pos + 1:]), dim=0)
            result[bi] = state
        return result

    def forward(self, history_features: Tensor, actor_mask: Tensor,
                forecast_times: Tensor, history_context: Dict[str, Tensor],
                history_edges: Tensor) -> Dict[str, Any]:
        if history_features.ndim != 4 or history_features.shape[-1] != 13:
            raise ValueError("history_features must be [B,N,T,13]")
        b, n, _, _ = history_features.shape
        if history_edges.shape != (b, n, n, self.edge_anchors, 13):
            raise ValueError("history_edges shape mismatch")
        real = history_features[actor_mask]
        x = self.feature_projection(real) + self.history_time(real[..., 12:13])
        _, h = self.temporal(x)
        encoded = self.temporal_norm(h[-1])
        actors = encoded.new_zeros((b, n, self.width)); actors[actor_mask] = encoded
        safe_edges = torch.where(
            (actor_mask[:, :, None] & actor_mask[:, None, :])[..., None, None],
            history_edges, torch.zeros_like(history_edges))
        rel_logits, adjacency, order = self._relation(actors, safe_edges, actor_mask)
        actors = self._dag_message(actors, adjacency, order, actor_mask)
        query = self.scene_queries[None].expand(b, -1, -1)
        scene = query + (actors * actor_mask[..., None]).sum(1, keepdim=True) / actor_mask.sum(1, keepdim=True).clamp_min(1)[:, :, None]
        scene = scene + actors.mean(1, keepdim=True)
        mode_actors = actors[:, None] + scene[:, :, None, :]
        scores = self.scene_score(scene).squeeze(-1)
        times = forecast_times.to(history_features.device, history_features.dtype)
        steps = _positive_times(times)
        te = self.future_time(torch.stack((times / 10.0, steps), -1))
        dec = mode_actors[..., None, :] + te[None, None, None]
        proposal = self.decoder(dec)
        v0 = torch.where(actor_mask[..., None], history_context["v0_local"], torch.zeros_like(history_context["v0_local"]))
        local, velocities, diagnostics = integrate_acceleration(
            proposal, v0, times, self.constraints, acceleration_scale=10.0)
        rot = torch.where(actor_mask[..., None, None], history_context["R_local_to_shared"], torch.zeros_like(history_context["R_local_to_shared"]))
        p0 = torch.where(actor_mask[..., None], history_context["p0_shared"], torch.zeros_like(history_context["p0_shared"]))
        shared = torch.einsum("bnij,bknhj->bknhi", rot.to(local.dtype), local) + p0[:, None, :, None]
        shared = torch.where(actor_mask[:, None, :, None, None], shared, torch.zeros_like(shared))
        return {"trajectories": shared, "scene_logits": scores, "relation_logits": rel_logits,
                "dag_adjacency": adjacency, "dag_order": order,
                "local_velocities": velocities, "constraint_diagnostics": diagnostics}


def aviation_relation_targets(target: Tensor, valid: Tensor, actor_mask: Tensor,
                              distance_threshold_m: float = 1000.0) -> Tensor:
    """Weakly supervised three-way graph labels derived from future proximity.

    This is not a causal ground-truth label.  A pair is labelled as a weak
    directed interaction only when its future closest approach is below the
    configured threshold and the two actors have distinguishable pre-event
    motion changes.  The actor with the smaller pre-event change is used as a
    deterministic reference source; otherwise the pair is left unlabelled.
    """
    b, n, h, _ = target.shape
    out = torch.full((b, n, n), -1, dtype=torch.long, device=target.device)
    pos = torch.where(valid[..., None], target, torch.zeros_like(target))
    for bi in range(b):
        for i in range(n):
            if not bool(actor_mask[bi, i]): continue
            for j in range(i + 1, n):
                if not bool(actor_mask[bi, j]): continue
                out[bi, i, i] = -1
                out[bi, j, j] = -1
                both = valid[bi, i] & valid[bi, j]
                if not bool(both.any()): continue
                dist = torch.linalg.vector_norm(pos[bi, i] - pos[bi, j], dim=-1)
                dist = torch.where(both, dist, torch.full_like(dist, float("inf")))
                k = int(torch.argmin(dist))
                if float(dist[k]) <= distance_threshold_m:
                    # Use displacement change over the pre-event interval as
                    # a reproducible, non-causal orientation heuristic.
                    cut = max(1, k)
                    di = torch.linalg.vector_norm(target[bi, i, cut] - target[bi, i, 0])
                    dj = torch.linalg.vector_norm(target[bi, j, cut] - target[bi, j, 0])
                    if torch.isfinite(di) and torch.isfinite(dj) and abs(float(di - dj)) > 1e-3:
                        out[bi, i, j] = 1 if di < dj else 2
                        out[bi, j, i] = 2 if out[bi, i, j] == 1 else 1
                else:
                    out[bi, i, j] = 0
                    out[bi, j, i] = 0
    return out
