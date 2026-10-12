"""ASCENT adapted to this project's common history-only trajectory interface.

Derived from the active forward path of a-pru/ascent commit
814e0a18a8a7500dfb0498ab2ee873d022874e8a. This is an adapted baseline,
not a reproduction of the authors' benchmark, checkpoints or training loss.
See docs/ASCENT_ADAPTER_IMPLEMENTATION_20260921.md for the complete mapping.

Neural coordinates are metres / 1000. The official dt=1 parameterization is
preserved: its signed scalar is a normalized distance proxy per output step,
not a physical airspeed. Actual fixed query seconds are checked explicitly.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn


OFFICIAL_COMMIT = "814e0a18a8a7500dfb0498ab2ee873d022874e8a"


class _Projector(nn.Module):
    def __init__(self, dimension: int, width: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dimension, width), nn.ReLU(), nn.Linear(width, width))

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class _DropPath(nn.Module):
    """Per-actor stochastic depth, matching timm's scale_by_keep=True rule."""

    def __init__(self, probability: float):
        super().__init__()
        self.drop_prob = probability

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1 - self.drop_prob
        noise = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * (noise / keep)


class _Mlp(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, 4 * width)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0)
        self.fc2 = nn.Linear(4 * width, width)
        self.drop2 = nn.Dropout(0)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class _Block(nn.Module):
    """Only the default pre-norm path actually used by official Ascent."""

    def __init__(self, width: int, heads: int, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        # Official qkv_bias=False maps to add_bias_kv=False, not bias=False.
        self.attn = nn.MultiheadAttention(width, heads, add_bias_kv=False, dropout=0, batch_first=True)
        self.drop_path1 = _DropPath(drop_path)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = _Mlp(width)
        self.drop_path2 = _DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        normalized = self.norm1(x)
        message, _ = self.attn(normalized, normalized, normalized, need_weights=True)
        x = x + self.drop_path1(message)
        return x + self.drop_path2(self.mlp(self.norm2(x)))


def _angles(points: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Official heading/elevation features from the final two input points."""
    delta = points[:, -1] - points[:, -2]
    yaw = torch.atan2(delta[:, 1], delta[:, 0])
    pitch = torch.atan2(delta[:, 2], torch.linalg.vector_norm(delta[:, :2], dim=-1) + 1e-8)
    embedding = torch.stack((torch.cos(yaw), torch.sin(yaw), torch.cos(pitch), torch.sin(pitch)), dim=-1)
    return embedding, yaw, pitch


def _ascent_rotation(yaw: Tensor, pitch: Tensor) -> Tensor:
    """Official R_yaw @ R_pitch (including its original pitch sign)."""
    cy, sy, cp, sp = torch.cos(yaw), torch.sin(yaw), torch.cos(pitch), torch.sin(pitch)
    zero, one = torch.zeros_like(yaw), torch.ones_like(yaw)
    yaw_rotation = torch.stack((torch.stack((cy, -sy, zero), -1),
                                torch.stack((sy, cy, zero), -1),
                                torch.stack((zero, zero, one), -1)), dim=-2)
    pitch_rotation = torch.stack((torch.stack((cp, zero, sp), -1),
                                  torch.stack((zero, one, zero), -1),
                                  torch.stack((-sp, zero, cp), -1)), dim=-2)
    return yaw_rotation @ pitch_rotation


def _angle_head(pair: Tensor) -> Tensor:
    sine, cosine = pair[..., 0], pair[..., 1]
    norm = torch.sqrt(sine.square() + cosine.square() + 1e-8)
    return torch.atan2(sine / norm, cosine / norm)


def _parameter_positions(flight_params: Tensor) -> Tensor:
    """Vectorized official dt=1 recurrence, excluding the initial zero point.

    This preserves dx=s*cos(heading), dy=s*sin(heading), dz=s*sin(pitch),
    including its parameter semantics. No extra cos(pitch), positivity map,
    acceleration bound, initial-speed residual, or collision rule is added.
    """
    scalar, heading, pitch = flight_params.unbind(dim=-1)
    increments = torch.stack((scalar * torch.cos(heading), scalar * torch.sin(heading),
                              scalar * torch.sin(pitch)), dim=-1)
    return torch.cumsum(increments, dim=-2)


class ASCENTAdapter(nn.Module):
    """Independent actor ASCENT core under a fixed common sampling protocol.

    Default heads=4 is the requested development-budget adaptation; the
    official active architecture is width=128,layers=2,heads=8. Using those
    values allows direct active-weight transfer and reference-forward checks.
    This implementation keeps global position embedding and normalization on,
    and omits only official modules that its forward never executes.
    """

    def __init__(
        self,
        width: int = 128,
        layers: int = 2,
        heads: int = 4,
        modes: int = 5,
        history_steps: int = 51,
        future_steps: int = 100,
        sample_rate_hz: float = 5,
    ):
        super().__init__()
        dimensions = (width, layers, heads, modes, history_steps, future_steps)
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in dimensions):
            raise ValueError("model dimensions must be positive integers")
        if history_steps < 2 or width % heads:
            raise ValueError("history_steps >= 2 and width divisible by heads required")
        if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive and finite")
        self.width, self.layers, self.heads, self.modes = width, layers, heads, modes
        self.history_steps, self.future_steps = history_steps, future_steps
        self.sample_rate_hz = float(sample_rate_hz)
        self.coordinate_scale_m = 1000.0
        self.joint = False
        self.interaction = False
        self.official_commit = OFFICIAL_COMMIT
        self.baseline_label = "ASCENT-adapted"
        rates = torch.linspace(0, 0.2, layers).tolist()
        self.agent_blks = nn.ModuleList([_Block(width, heads, rate) for rate in rates])
        self.agent_xy_proj = _Projector(2, width)
        self.agent_z_proj = _Projector(1, width)
        self.agent_ts_proj = _Projector(1, width)

        def head(output: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(),
                                 nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, output))

        self.fp1 = head(future_steps)
        self.fp2 = head(2 * future_steps)
        self.fp3 = head(2 * future_steps)
        self.pi = head(1)
        self.pos_embed = nn.Sequential(nn.Linear(7, width), nn.GELU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)
        # Five rows are retained for active-state-dict compatibility; only 0 is
        # used, as in the official path. These are not dataset actor identities.
        self.type_embed = nn.Parameter(torch.empty(5, width))
        self.mode1_embed = nn.Parameter(torch.empty(modes, width))
        nn.init.normal_(self.type_embed, std=0.02)
        nn.init.normal_(self.mode1_embed, std=0.02)

    def _recover_history(
        self, history_features: Tensor, actor_mask: Tensor, history_context: Dict[str, Tensor]
    ) -> Tensor:
        """Undo the common yaw-local/metre-scale adapter; return shared m/1000."""
        b, n, t, _ = history_features.shape
        p0 = history_context.get("p0_shared")
        rotation = history_context.get("R_local_to_shared")
        if p0 is None or rotation is None or p0.shape != (b, n, 3) or rotation.shape != (b, n, 3, 3):
            raise ValueError("history_context requires p0_shared[B,N,3] and R_local_to_shared[B,N,3,3]")
        # No endpoint velocity is needed by the official path; it uses positions.
        if not torch.isfinite(p0[actor_mask]).all() or not torch.isfinite(rotation[actor_mask]).all():
            raise ValueError("real actors require finite position/rotation context")
        with torch.autocast(device_type=history_features.device.type, enabled=False):
            local = history_features[actor_mask][..., :3].float()
            real_rotation = rotation[actor_mask].float()
            identity = torch.eye(3, device=local.device).expand_as(real_rotation)
            if not torch.allclose(real_rotation.transpose(-1, -2) @ real_rotation, identity, atol=1e-4, rtol=1e-4):
                raise ValueError("common local-to-shared rotations must be orthogonal")
            normalized_shared = torch.bmm(local, real_rotation.transpose(-1, -2))
            normalized_shared = normalized_shared + p0[actor_mask].float()[:, None] / self.coordinate_scale_m
        return normalized_shared

    def forward(
        self,
        history_features: Tensor,
        actor_mask: Tensor,
        forecast_times: Tensor,
        history_context: Dict[str, Tensor],
        history_edges: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        # history_edges is intentionally unused: the official active model has
        # no scene interaction path despite legacy scene_blks/adj declarations.
        if history_features.ndim != 4 or history_features.shape[-1] != 13:
            raise ValueError("history_features must have shape [B,N,T,13]")
        b, n, t, _ = history_features.shape
        if b < 1 or n < 1 or t != self.history_steps or actor_mask.shape != (b, n) or actor_mask.dtype != torch.bool:
            raise ValueError("history dimensions/actor_mask must match configured history_steps")
        if not actor_mask.any(dim=1).all():
            raise ValueError("each scene must have at least one historical actor")
        real_features = history_features[actor_mask]
        if not torch.isfinite(real_features).all():
            raise ValueError("real actors require finite history features")
        expected_future = torch.arange(1, self.future_steps + 1, device=history_features.device, dtype=torch.float32) / self.sample_rate_hz
        if forecast_times.shape != (self.future_steps,) or not torch.allclose(
            forecast_times.to(device=expected_future.device, dtype=torch.float32), expected_future, atol=2e-5, rtol=1e-5
        ):
            raise ValueError("forecast_times must equal (1..future_steps)/sample_rate_hz seconds")
        expected_history = (torch.arange(t, device=history_features.device, dtype=torch.float32) - (t - 1)) / self.sample_rate_hz
        if not torch.allclose(real_features[..., 12].float() * 10.0, expected_history.expand(real_features.shape[0], t), atol=2e-5, rtol=1e-5):
            raise ValueError("historical relative_time feature must match fixed sample_rate_hz and end at zero")
        actor_shared = self._recover_history(history_features, actor_mask, history_context)
        real_count = actor_shared.shape[0]
        with torch.autocast(device_type=actor_shared.device.type, enabled=False):
            centers = actor_shared[:, -1]
            angle_embedding, yaw, pitch = _angles(actor_shared)
            rotation = _ascent_rotation(yaw, pitch)
            # Match official double-precision transform followed by float32.
            actor_local = torch.bmm((actor_shared.double() - centers.double()[:, None]), rotation.double()).float()
            position_features = torch.cat((centers, angle_embedding), dim=-1)
        step_indices = torch.arange(t, device=actor_shared.device, dtype=torch.float32).view(1, t, 1).expand(real_count, t, 1)
        encoded = self.agent_xy_proj(actor_local[..., :2]) + self.agent_z_proj(actor_local[..., 2:3])
        encoded = encoded + self.agent_ts_proj(step_indices)
        for block in self.agent_blks:
            encoded = block(encoded)
        encoded = encoded.max(dim=1).values
        encoded = encoded + self.type_embed[0][None] + self.pos_embed(position_features)
        encoded = self.norm(encoded)
        mode_features = encoded[:, None] + self.mode1_embed[None]
        logits = self.pi(mode_features).squeeze(-1)
        scalar = self.fp1(mode_features)
        yaw_pair = self.fp2(mode_features).reshape(real_count, self.modes, self.future_steps, 2)
        pitch_pair = self.fp3(mode_features).reshape(real_count, self.modes, self.future_steps, 2)
        with torch.autocast(device_type=actor_shared.device.type, enabled=False):
            flight_params = torch.stack((scalar.float(), _angle_head(yaw_pair.float()), _angle_head(pitch_pair.float())), dim=-1)
            local_prediction = _parameter_positions(flight_params)
            shared_prediction = torch.einsum("rij,rkhj->rkhi", rotation.double(), local_prediction.double())
            shared_prediction = (shared_prediction + centers.double()[:, None, None]).float() * self.coordinate_scale_m
        trajectories = shared_prediction.new_zeros((b, n, self.modes, self.future_steps, 3))
        trajectories[actor_mask] = shared_prediction
        actor_logits = logits.new_zeros((b, n, self.modes))
        actor_logits[actor_mask] = logits
        return {"trajectories": trajectories.permute(0, 2, 1, 3, 4), "actor_logits": actor_logits}
