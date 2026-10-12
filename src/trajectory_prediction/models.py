"""Development trajectory models with history-only forward interfaces.

Positions/velocities in history_context are metres and m/s. Input feature
normalization belongs to the dataset adapter; future times are seconds. No
future label or mask enters TrajectoryModel.forward.
"""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _positive_times(times: Tensor) -> Tensor:
    if times.ndim != 1 or times.numel() == 0 or not torch.isfinite(times).all():
        raise ValueError("forecast_times must be a nonempty finite vector")
    steps = torch.diff(times, prepend=times.new_zeros(1))
    if not torch.all(steps > 0):
        raise ValueError("forecast_times must be positive and strictly increasing")
    return steps


def _check_bounds(constraints: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if constraints is None:
        return None
    if set(constraints) != {"Vmax", "Wmax", "Amax"}:
        raise ValueError("constraints must contain exactly Vmax, Wmax (m/s), Amax (m/s^2)")
    bounds = {key: float(value) for key, value in constraints.items()}
    if any(not math.isfinite(value) or value < 0 for value in bounds.values()):
        raise ValueError("constraint bounds must be finite and nonnegative")
    return bounds


def integrate_acceleration(
    proposal: Tensor,
    initial_velocity: Tensor,
    forecast_times: Tensor,
    constraints: Optional[Dict[str, float]] = None,
    acceleration_scale: float = 10.0,
    constraint_mode: str = "full",
) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
    """Integrate [B,K,N,H,3] proposals from zero local displacement.

    Unconstrained acceleration is acceleration_scale * proposal. Constrained
    acceleration uses the same physical proposal scale before applying the
    V3 tanh/norm bound; vertical slab and speed ball project velocity, and
    positions use the same trapezoid in both variants.
    Bounds V/W expand to initial velocity once, never during the rollout.
    acceleration_only uses the same acceleration bound and trapezoid, without
    either velocity projection; stored V/W bounds are inactive in that mode.
    Physics runs in at least float32 even under mixed-precision autocast.
    """
    if proposal.ndim != 5 or proposal.shape[-1] != 3:
        raise ValueError("proposal must have shape [B,K,N,H,3]")
    b, k, n, h, _ = proposal.shape
    if min(b, k, n, h) < 1 or initial_velocity.shape != (b, n, 3):
        raise ValueError("positive B/K/N/H and initial_velocity[B,N,3] required")
    if not math.isfinite(acceleration_scale) or acceleration_scale <= 0:
        raise ValueError("acceleration_scale must be positive and finite")
    bounds = _check_bounds(constraints)
    if constraint_mode not in ("full", "acceleration_only"):
        raise ValueError("constraint_mode must be 'full' or 'acceleration_only'")
    if bounds is None and constraint_mode != "full":
        raise ValueError("acceleration_only requires constraint bounds")
    physics_dtype = torch.float64 if proposal.dtype == torch.float64 else torch.float32
    # Elementwise/reduction arithmetic is explicitly in a physics-safe dtype.
    proposal = proposal.to(dtype=physics_dtype)
    initial_velocity = initial_velocity.to(device=proposal.device, dtype=physics_dtype)
    times = forecast_times.to(device=proposal.device, dtype=physics_dtype)
    if times.shape != (h,):
        raise ValueError("forecast_times must have H elements")
    steps = _positive_times(times)
    if not torch.isfinite(proposal).all() or not torch.isfinite(initial_velocity).all():
        raise ValueError("proposal and initial_velocity must be finite")
    velocity = initial_velocity[:, None].expand(b, k, n, 3)
    position = torch.zeros_like(velocity)
    speed_bound = vertical_bound = None
    if bounds is None:
        # Algebraically the same recurrence as the projected branch, without
        # per-query Python/CUDA launches when there is no projection.
        acceleration = acceleration_scale * proposal
        dt = steps[None, None, None, :, None]
        velocities = velocity[..., None, :] + torch.cumsum(acceleration * dt, dim=-2)
        previous = torch.cat((velocity[..., None, :], velocities[..., :-1, :]), dim=-2)
        positions = torch.cumsum(0.5 * (previous + velocities) * dt, dim=-2)
        return positions, velocities, {
            "constrained": False,
            "constraint_mode": "none",
            "active_bounds": [],
            "velocity_projection_enabled": False,
            "unconstrained_acceleration_scale_m_per_s2": float(acceleration_scale),
            "effective_speed_bound": None,
            "effective_vertical_bound": None,
            "speed_bound_expanded": None,
            "vertical_bound_expanded": None,
            "projection_active": None,
        }
    if constraint_mode == "full":
        speed0 = torch.linalg.vector_norm(initial_velocity, dim=-1)
        vertical0 = initial_velocity[..., 2].abs()
        speed_bound = torch.maximum(speed0, speed0.new_tensor(bounds["Vmax"]))
        vertical_bound = torch.maximum(vertical0, vertical0.new_tensor(bounds["Wmax"]))
    amax = bounds["Amax"]
    if amax == 0.0:
        # Keep a differentiable zero path: zero acceleration must also have
        # finite (zero) gradients when a zero physical bound is configured.
        acceleration = proposal * 0.0
    else:
        # Match the unconstrained small-signal map a ~= acceleration_scale*u,
        # while retaining a hard norm bound of Amax for larger proposals.
        scale_denominator = max(amax, torch.finfo(physics_dtype).eps)
        bounded = torch.tanh(acceleration_scale * proposal / scale_denominator)
        acceleration = amax * bounded / torch.linalg.vector_norm(
            bounded, dim=-1, keepdim=True
        ).clamp_min(1.0)

    if constraint_mode == "acceleration_only":
        dt = steps[None, None, None, :, None]
        velocities = velocity[..., None, :] + torch.cumsum(acceleration * dt, dim=-2)
        previous = torch.cat((velocity[..., None, :], velocities[..., :-1, :]), dim=-2)
        positions = torch.cumsum(0.5 * (previous + velocities) * dt, dim=-2)
        return positions, velocities, {
            "constrained": True,
            "constraint_mode": "acceleration_only",
            "active_bounds": ["Amax"],
            "velocity_projection_enabled": False,
            "unconstrained_acceleration_scale_m_per_s2": float(acceleration_scale),
            "effective_speed_bound": None,
            "effective_vertical_bound": None,
            "speed_bound_expanded": None,
            "vertical_bound_expanded": None,
            "projection_active": None,
        }

    positions, velocities, active = [], [], []
    for index, dt in enumerate(steps.unbind()):
        trial = velocity + dt * acceleration[..., index, :]
        vertical = torch.maximum(
            torch.minimum(trial[..., 2], vertical_bound[:, None]),
            -vertical_bound[:, None],
        )
        slab = torch.cat((trial[..., :2], vertical[..., None]), dim=-1)
        norm = torch.linalg.vector_norm(slab, dim=-1)
        ratio = torch.where(
            norm > speed_bound[:, None],
            speed_bound[:, None] / norm.clamp_min(torch.finfo(physics_dtype).tiny),
            torch.ones_like(norm),
        )
        next_velocity = slab * ratio[..., None]
        active.append((next_velocity.detach() - trial.detach()).abs().amax(dim=-1) > 1e-6)
        position = position + 0.5 * (velocity + next_velocity) * dt
        velocity = next_velocity
        positions.append(position)
        velocities.append(velocity)
    diagnostics: Dict[str, Any] = {
        "constrained": True,
        "constraint_mode": "full",
        "active_bounds": ["Vmax", "Wmax", "Amax"],
        "velocity_projection_enabled": True,
        "unconstrained_acceleration_scale_m_per_s2": float(acceleration_scale),
        "effective_speed_bound": speed_bound,
        "effective_vertical_bound": vertical_bound,
        "speed_bound_expanded": speed0 > bounds["Vmax"],
        "vertical_bound_expanded": vertical0 > bounds["Wmax"],
        "projection_active": torch.stack(active, dim=-1),
    }
    return torch.stack(positions, dim=-2), torch.stack(velocities, dim=-2), diagnostics


def decode_coordinate_residual(
    proposal: Tensor,
    initial_velocity: Tensor,
    forecast_times: Tensor,
    acceleration_scale: float = 10.0,
) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
    """Decode local CV-residual positions without integrating proposals.

    Uses the same three-channel head and dimensionless proposal scale as the
    acceleration decoder: p(t) = v0*t + 0.5*acceleration_scale*t**2*u(t).
    A temporally constant u therefore has exactly the same position response
    as unconstrained acceleration integration. Each u(t) otherwise affects
    only its own position, removing the integrator's cross-time coupling.
    This is a CV-residual coordinate ablation, not a zero-prior coordinate
    regressor. Returned velocities are backward secants for diagnostics, not
    integrated physical states. No future label or bound is consumed.
    """
    if proposal.ndim != 5 or proposal.shape[-1] != 3:
        raise ValueError("proposal must have shape [B,K,N,H,3]")
    b, k, n, h, _ = proposal.shape
    if min(b, k, n, h) < 1 or initial_velocity.shape != (b, n, 3):
        raise ValueError("positive B/K/N/H and initial_velocity[B,N,3] required")
    if not math.isfinite(acceleration_scale) or acceleration_scale <= 0:
        raise ValueError("acceleration_scale must be positive and finite")
    physics_dtype = torch.float64 if proposal.dtype == torch.float64 else torch.float32
    proposal = proposal.to(dtype=physics_dtype)
    initial_velocity = initial_velocity.to(device=proposal.device, dtype=physics_dtype)
    times = forecast_times.to(device=proposal.device, dtype=physics_dtype)
    if times.shape != (h,):
        raise ValueError("forecast_times must have H elements")
    steps = _positive_times(times)
    if not torch.isfinite(proposal).all() or not torch.isfinite(initial_velocity).all():
        raise ValueError("proposal and initial_velocity must be finite")
    t = times[None, None, None, :, None]
    positions = initial_velocity[:, None, :, None, :] * t + 0.5 * acceleration_scale * t.square() * proposal
    previous = torch.cat((torch.zeros_like(positions[..., :1, :]), positions[..., :-1, :]), dim=-2)
    velocities = (positions - previous) / steps[None, None, None, :, None]
    return positions, velocities, {
        "constrained": False,
        "constraint_mode": "none",
        "active_bounds": [],
        "velocity_projection_enabled": False,
        "decoder": "coordinate",
        "coordinate_parameterization": "cv_residual_quadratic_time_scale",
        "coordinate_scale_acceleration_m_per_s2": float(acceleration_scale),
        "velocity_definition": "backward_position_secant",
        "effective_speed_bound": None,
        "effective_vertical_bound": None,
        "speed_bound_expanded": None,
        "vertical_bound_expanded": None,
        "projection_active": None,
    }


class _RelationLayer(nn.Module):
    """Receiver-i/sender-j attention with edge logits and edge value messages."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.width, self.heads, self.head_dim = width, heads, width // heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.edge_bias = nn.Linear(width, heads)
        self.edge_value = nn.Linear(width, width)
        self.projection = nn.Linear(width, width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, actors: Tensor, edges: Tensor, actor_mask: Tensor) -> Tensor:
        b, n, d = actors.shape
        q, k, v = self.qkv(self.norm(actors)).chunk(3, dim=-1)
        q = q.reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        bias = self.edge_bias(edges).permute(0, 3, 1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim) + bias
        logits = logits.masked_fill(~actor_mask[:, None, None, :], -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        edge_values = self.edge_value(edges).reshape(b, n, n, self.heads, self.head_dim)
        edge_values = edge_values.permute(0, 3, 1, 2, 4)
        values = v[:, :, None, :, :] + edge_values
        message = (weights[..., None] * values).sum(dim=-2).transpose(1, 2).reshape(b, n, d)
        actors = actors + self.projection(message)
        actors = actors + self.ffn(self.ffn_norm(actors))
        return torch.where(actor_mask[..., None], actors, torch.zeros_like(actors))


class _ConformerTemporalBlock(nn.Module):
    """Transformer layer followed by a local depthwise temporal mixer."""

    def __init__(self, width: int, heads: int, kernel_size: int = 5):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=2 * width, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.attention = layer
        self.conv_norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(
            width, width, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=width,
        )
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attention(x)
        residual = x
        y = self.conv_norm(x).transpose(1, 2)
        y = self.pointwise(torch.nn.functional.gelu(self.depthwise(y)))
        y = self.dropout(y.transpose(1, 2))
        return residual + y


class TrajectoryModel(nn.Module):
    """Shared actor temporal encoder, optional relations and joint futures.

    ``interaction=False,joint=False`` has no cross-actor information path.
    ``joint=True`` intentionally creates cross-actor paths in scene queries
    and per-mode actor attention even when the relation encoder is disabled.
    The only optional temporal implementation is the real official Mamba2.
    """

    def __init__(
        self,
        width: int = 64,
        layers: int = 2,
        heads: int = 4,
        modes: int = 1,
        interaction: bool = False,
        joint: bool = False,
        joint_variant: str = "shared",
        local_modes: int = 3,
        temporal: str = "transformer",
        constraints: Optional[Dict[str, float]] = None,
        edge_anchors: int = 3,
        use_edge_geometry: bool = True,
        constraint_mode: str = "full",
        temporal_pool: str = "last",
        temporal_variant: str = "vanilla",
        patch_len: int = 5,
        patch_stride: int = 5,
        fusion_mode: str = "gated",
        relation_fusion: str = "serial",
        goal_head: bool = False,
        goal_conditioning: bool = False,
        goal_scale_m: float = 1000.0,
        motion_decoder: str = "acceleration",
        ascent_pose: bool = False,
        state_queries: bool = False,
        max_future_steps: int = 200,
        future_relation: bool = False,
        intent_conditioning: str = "none",
        joint_score_temperature: float = 1.0,
        joint_diversity_weight: float = 0.0,
        endpoint_correction: bool = False,
        edge_geometry_transform: str = "none",
        edge_keep_anchors: Optional[Tuple[int, ...]] = None,
        edge_keep_feats: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        if min(width, layers, heads, modes, edge_anchors, local_modes) < 1 or width % heads:
            raise ValueError("positive dimensions and width divisible by heads required")
        if joint_variant not in ("shared", "factorized", "influence"):
            raise ValueError("joint_variant must be shared, factorized or influence")
        if joint_variant == "factorized" and not joint:
            raise ValueError("factorized joint requires joint=True")
        if not isinstance(use_edge_geometry, bool):
            raise ValueError("use_edge_geometry must be boolean")
        if not use_edge_geometry and not interaction:
            raise ValueError("disabling edge geometry requires interaction=True")
        if edge_geometry_transform not in ("none", "symmetric"):
            raise ValueError("edge_geometry_transform must be none or symmetric")
        if edge_geometry_transform != "none" and (not interaction or not use_edge_geometry):
            raise ValueError("edge_geometry_transform requires interaction=True and use_edge_geometry=True")
        # Parameter-matched partial-edge ablations: zero selected anchor/feature
        # slices of the edge tensor while keeping edge_projection dimensions and
        # every relation parameter unchanged. None retains the full descriptor.
        edge_keep_anchors = None if edge_keep_anchors is None else tuple(int(x) for x in edge_keep_anchors)
        edge_keep_feats = None if edge_keep_feats is None else tuple(int(x) for x in edge_keep_feats)
        if (edge_keep_anchors is not None or edge_keep_feats is not None) and not interaction:
            raise ValueError("edge keep masks require interaction=True")
        if edge_keep_anchors is not None and (
            len(edge_keep_anchors) == 0 or any(a < 0 or a >= edge_anchors for a in edge_keep_anchors)
        ):
            raise ValueError(f"edge_keep_anchors indices must be within 0..{edge_anchors - 1}")
        if edge_keep_feats is not None and (
            len(edge_keep_feats) == 0 or any(f < 0 or f >= 13 for f in edge_keep_feats)
        ):
            raise ValueError("edge_keep_feats indices must be within 0..12")
        self.edge_keep_anchors = edge_keep_anchors
        self.edge_keep_feats = edge_keep_feats
        if constraint_mode not in ("full", "acceleration_only"):
            raise ValueError("constraint_mode must be 'full' or 'acceleration_only'")
        if constraints is None and constraint_mode != "full":
            raise ValueError("acceleration_only requires constraint bounds")
        self.width, self.modes = width, modes
        self.interaction, self.joint = interaction, joint
        self.joint_variant, self.local_modes = joint_variant, int(local_modes)
        self.temporal_kind, self.edge_anchors = temporal.lower(), edge_anchors
        temporal_variant = temporal_variant.lower()
        if temporal_variant not in ("vanilla", "conformer", "patchtst", "mamba_patch"):
            raise ValueError("temporal_variant must be vanilla, conformer, patchtst or mamba_patch")
        if fusion_mode not in ("add", "gated", "concat", "multiscale"):
            raise ValueError("fusion_mode must be add, gated, concat or multiscale")
        if temporal_variant in ("conformer", "patchtst") and self.temporal_kind != "transformer":
            raise ValueError("conformer and patchtst require temporal='transformer'")
        if patch_len < 1 or patch_stride < 1:
            raise ValueError("patch_len and patch_stride must be positive")
        self.temporal_variant, self.fusion_mode = temporal_variant, fusion_mode
        self.patch_len, self.patch_stride = int(patch_len), int(patch_stride)
        self.constraints = _check_bounds(constraints)
        self.use_edge_geometry, self.constraint_mode = use_edge_geometry, constraint_mode
        self.edge_geometry_transform = edge_geometry_transform
        self.acceleration_scale = 10.0
        if temporal_pool not in ("last", "mean", "max"):
            raise ValueError("temporal_pool must be last, mean or max")
        if relation_fusion not in ("serial", "gated") or (relation_fusion == "gated" and not interaction):
            raise ValueError("gated relation_fusion requires interaction")
        if goal_conditioning and not goal_head:
            raise ValueError("goal_conditioning requires goal_head")
        if not math.isfinite(goal_scale_m) or goal_scale_m <= 0:
            raise ValueError("goal_scale_m must be finite and positive")
        if motion_decoder not in ("acceleration", "coordinate", "ascent", "flight", "hybrid"):
            raise ValueError("motion_decoder must be acceleration, coordinate, ascent, flight or hybrid")
        if motion_decoder != "acceleration" and constraints is not None:
            raise ValueError("motion decoders other than acceleration do not support constraints")
        self.temporal_pool, self.relation_fusion = temporal_pool, relation_fusion
        self.goal_head_enabled, self.goal_conditioning, self.goal_scale_m = bool(goal_head), bool(goal_conditioning), float(goal_scale_m)
        self.motion_decoder = motion_decoder
        self.ascent_pose = bool(ascent_pose)
        if max_future_steps < 1:
            raise ValueError("max_future_steps must be positive")
        self.state_queries_enabled = bool(state_queries)
        self.max_future_steps = int(max_future_steps)
        self.future_relation_enabled = bool(future_relation)
        if not math.isfinite(joint_score_temperature) or joint_score_temperature <= 0:
            raise ValueError("joint_score_temperature must be finite and positive")
        if not math.isfinite(joint_diversity_weight) or joint_diversity_weight < 0:
            raise ValueError("joint_diversity_weight must be finite and nonnegative")
        self.joint_score_temperature = float(joint_score_temperature)
        self.joint_diversity_weight = float(joint_diversity_weight)
        self.endpoint_correction_enabled = bool(endpoint_correction)
        if intent_conditioning not in ("none", "direction", "goal", "goal_relation"):
            raise ValueError("intent_conditioning must be none, direction, goal or goal_relation")
        if intent_conditioning in ("goal", "goal_relation") and not goal_head:
            raise ValueError("goal intent conditioning requires goal_head")
        self.intent_conditioning = intent_conditioning
        if self.future_relation_enabled and not self.goal_head_enabled:
            raise ValueError("future_relation requires goal_head")
        self.feature_projection = nn.Linear(13, width)
        self.history_time_encoding = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width))
        if self.temporal_kind == "transformer" and temporal_variant == "vanilla":
            layer = nn.TransformerEncoderLayer(
                width, heads, dim_feedforward=2 * width, dropout=0.1,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.temporal_norm = nn.LayerNorm(width)
        elif self.temporal_kind == "transformer" and temporal_variant == "conformer":
            self.temporal_encoder = nn.ModuleList([
                _ConformerTemporalBlock(width, heads) for _ in range(layers)
            ])
            self.temporal_norm = nn.LayerNorm(width)
        elif self.temporal_kind == "transformer" and temporal_variant == "patchtst":
            self.patch_projection = nn.Linear(self.patch_len * width, width)
            self.temporal_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    width, heads, dim_feedforward=2 * width, dropout=0.1,
                    activation="gelu", batch_first=True, norm_first=True,
                ),
                layers, enable_nested_tensor=False,
            )
            self.temporal_norm = nn.LayerNorm(width)
        elif temporal_variant == "mamba_patch":
            try:
                from mamba_ssm import Mamba2
                import mamba_ssm.modules.mamba2 as _mamba2_module
                _mamba2_module.causal_conv1d_fn = None
            except (ImportError, OSError, RuntimeError) as exc:
                raise ImportError("mamba_patch requires mamba_ssm") from exc
            if (2 * width) % 32:
                raise ValueError("Mamba2 expand=2 requires 2*width divisible by headdim=32")
            self.patch_projection = nn.Linear(self.patch_len * width, width)
            self.patch_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(width, heads, dim_feedforward=2 * width,
                                           dropout=0.1, activation="gelu", batch_first=True,
                                           norm_first=True),
                layers, enable_nested_tensor=False,
            )
            self.patch_norm = nn.LayerNorm(width)
            self.mamba_encoder = nn.ModuleList([
                Mamba2(d_model=width, d_state=64, d_conv=4, expand=2, headdim=32,
                       use_mem_eff_path=False) for _ in range(layers)
            ])
            self.mamba_norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
            self.mamba_dropout = nn.Dropout(0.1)
            self.patch_fusion = nn.Linear(2 * width, width)
            self.patch_add_projection = nn.Linear(width, width)
            self.patch_concat_projection = nn.Linear(2 * width, width)
            self.patch_gate_projection = nn.Linear(2 * width, width)
            self.fusion_gate = nn.Linear(2 * width, width)
            nn.init.zeros_(self.fusion_gate.weight)
            nn.init.constant_(self.fusion_gate.bias, 0.0)
            self.temporal_norm = nn.LayerNorm(width)
        elif self.temporal_kind in ("mamba", "mamba2"):
            try:
                from mamba_ssm import Mamba2
                # The installed causal-conv1d fused kernel requires channel-last
                # strides that boolean actor packing cannot satisfy. Disable
                # only that optional convolution fast path; Mamba2's explicit
                # causal convolution and scan remain the executed recurrence.
                import mamba_ssm.modules.mamba2 as _mamba2_module
                _mamba2_module.causal_conv1d_fn = None
            except (ImportError, OSError, RuntimeError) as exc:
                raise ImportError(
                    "temporal='mamba2' requires the official mamba_ssm.Mamba2 and compatible CUDA kernels; no substitute is used"
                ) from exc
            if (2 * width) % 32:
                raise ValueError("Mamba2 expand=2 requires 2*width divisible by headdim=32")
            self.temporal_encoder = nn.ModuleList([
                # The fused Triton path requires channel-last strides that are
                # not guaranteed after actor masking on this CUDA/runtime
                # combination. Keep the genuine Mamba2 recurrence but use its
                # explicit conv+scan path for deterministic batched training.
                Mamba2(d_model=width, d_state=64, d_conv=4, expand=2, headdim=32,
                       use_mem_eff_path=False,)
                for _ in range(layers)
            ])
            self.mamba_norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
            self.mamba_dropout = nn.Dropout(0.1)
            self.temporal_norm = nn.LayerNorm(width)
        else:
            raise ValueError("temporal must be 'transformer' or 'mamba2'")
        if interaction:
            self.edge_projection = nn.Sequential(
                nn.Linear(13 * edge_anchors, width), nn.GELU(), nn.Linear(width, width)
            )
            self.relation_layers = nn.ModuleList([_RelationLayer(width, heads) for _ in range(2)])
            # Partial-edge ablation masks; default all-ones keeps the full
            # 3-anchor x 13-feature descriptor. Buffers move with .to(device).
            anchor_keep = torch.ones(edge_anchors)
            if edge_keep_anchors is not None:
                anchor_keep.zero_()
                anchor_keep[list(edge_keep_anchors)] = 1.0
            feat_keep = torch.ones(13)
            if edge_keep_feats is not None:
                feat_keep.zero_()
                feat_keep[list(edge_keep_feats)] = 1.0
            # Non-persistent: the masks are ablation configuration, not learned
            # state. They are reconstructed from model_config on every build, so
            # warm-start weight loading can never overwrite a partial mask.
            self.register_buffer("edge_anchor_keep", anchor_keep, persistent=False)
            self.register_buffer("edge_feat_keep", feat_keep, persistent=False)
        self.mode_queries = nn.Parameter(torch.empty(modes, width))
        nn.init.normal_(self.mode_queries, std=0.02)
        if joint:
            self.scene_attention = nn.MultiheadAttention(width, heads, batch_first=True)
            self.scene_norm = nn.LayerNorm(width)
            self.mode_actor_attention = nn.MultiheadAttention(width, heads, batch_first=True)
            self.mode_actor_norm = nn.LayerNorm(width)
            self.mode_actor_ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
            self.scene_score = nn.Linear(width, 1)
            if joint_variant in ("factorized", "influence"):
                # A global scene hypothesis captures the interaction pattern;
                # actor-local modes then express heterogeneous responses under
                # that same scene hypothesis.
                self.local_mode_queries = nn.Parameter(torch.empty(local_modes, width))
                nn.init.normal_(self.local_mode_queries, std=0.02)
                self.factorized_actor_residual = nn.Sequential(
                    nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width)
                )
                # Start from the validated shared-scene solution. The local
                # branch is opened by gradient descent instead of injecting a
                # random trajectory perturbation at epoch 1.
                nn.init.zeros_(self.factorized_actor_residual[-1].weight)
                nn.init.zeros_(self.factorized_actor_residual[-1].bias)
                self.local_actor_score = nn.Linear(2 * width, local_modes)
                # Directed influence weights: row i is the soft set of
                # upstream influencers conditioning actor i's response.
                self.influence_score = nn.Sequential(
                    nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1)
                )
                self.influence_message = nn.Sequential(
                    nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width)
                )
                self.scene_score_fused = nn.Sequential(
                    nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 1)
                )
        else:
            self.actor_score = nn.Linear(width, modes)
        self.future_time_encoding = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width))
        if self.state_queries_enabled:
            # DeMo-style state queries: each future index gets a learnable
            # dynamic-state token in addition to the mode/intention token.
            # The query table is sliced to the requested horizon at runtime;
            # it never receives future labels.
            self.state_queries = nn.Parameter(torch.empty(self.max_future_steps, width))
            nn.init.normal_(self.state_queries, std=0.02)
        if self.motion_decoder in ("acceleration", "coordinate", "hybrid"):
            # Keep identical parameter names, shapes and initialization for
            # the non-integrating coordinate ablation. Only decoding changes.
            self.acceleration_head = nn.Sequential(
                nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3)
            )
            nn.init.normal_(self.acceleration_head[-1].weight, std=0.01)
            nn.init.zeros_(self.acceleration_head[-1].bias)
        if self.motion_decoder in ("ascent", "flight", "hybrid"):
            # ASCENT-style structured motion heads. The flight variant uses
            # the same angle representation but anchors speed and direction
            # to the causal endpoint velocity below.
            self.motion_scalar_head = nn.Sequential(
                nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)
            )
            self.motion_yaw_head = nn.Sequential(
                nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2)
            )
            self.motion_pitch_head = nn.Sequential(
                nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2)
            )
            nn.init.normal_(self.motion_scalar_head[-1].weight, std=0.01)
            nn.init.zeros_(self.motion_scalar_head[-1].bias)
            if self.motion_decoder == "hybrid":
                self.motion_gate = nn.Linear(width, 1)
                nn.init.zeros_(self.motion_gate.weight)
                nn.init.constant_(self.motion_gate.bias, -4.0)
        if self.ascent_pose:
            # [terminal position in km, heading cos/sin, pitch cos/sin].
            self.ascent_pose_embedding = nn.Sequential(
                nn.Linear(7, width), nn.GELU(), nn.Linear(width, width)
            )
        if self.goal_head_enabled:
            self.goal_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))
            self.goal_refine = nn.Sequential(nn.Linear(width + 3, width), nn.GELU(), nn.Linear(width, 3))
            self.goal_embedding = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
            # A warm-started model must initially reproduce the no-goal path.
            # Zero-initialize the residual projection so adding the auxiliary
            # endpoint head cannot abruptly perturb an already trained decoder.
            nn.init.zeros_(self.goal_head[-1].weight)
            nn.init.zeros_(self.goal_head[-1].bias)
            nn.init.zeros_(self.goal_refine[-1].weight)
            nn.init.zeros_(self.goal_refine[-1].bias)
            if self.endpoint_correction_enabled:
                # New warm-start parameters must leave the existing decoder
                # nearly unchanged at epoch zero. The gate is opened by the
                # endpoint/trajectory losses during adaptation.
                self.endpoint_correction_gate = nn.Sequential(
                    nn.Linear(width + 3, width), nn.GELU(), nn.Linear(width, 1)
                )
                nn.init.zeros_(self.endpoint_correction_gate[-1].weight)
                nn.init.constant_(self.endpoint_correction_gate[-1].bias, -6.0)
        if self.intent_conditioning == "direction":
            self.intent_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))
            self.intent_embedding = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        if self.future_relation_enabled:
            self.future_goal_embedding = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
            self.future_relation_attention = nn.MultiheadAttention(width, heads, batch_first=True)
            self.future_relation_norm = nn.LayerNorm(width)
            self.future_relation_ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        # Instantiate optional parameters last: common-path initial weights
        # remain identical under the same seed. Blend the relation increment,
        # not a second copy of the temporal residual.
        if relation_fusion == "gated":
            self.relation_gate = nn.Linear(2 * width, width)
            nn.init.zeros_(self.relation_gate.weight)
            nn.init.zeros_(self.relation_gate.bias)

    def forward(
        self,
        history_features: Tensor,
        actor_mask: Tensor,
        forecast_times: Tensor,
        history_context: Dict[str, Tensor],
        history_edges: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        if history_features.ndim != 4 or history_features.shape[-1] != 13:
            raise ValueError("history_features must have shape [B,N,T,13]")
        b, n, t, _ = history_features.shape
        if min(b, n, t) < 1 or actor_mask.shape != (b, n) or actor_mask.dtype != torch.bool:
            raise ValueError("positive B/N/T and boolean actor_mask[B,N] required")
        if not actor_mask.any(dim=1).all():
            raise ValueError("each scene must contain at least one historical actor")
        if not torch.isfinite(history_features[actor_mask]).all():
            raise ValueError("real actors require finite historical features")
        time_dtype = torch.float64 if history_features.dtype == torch.float64 else torch.float32
        times = forecast_times.to(device=history_features.device, dtype=time_dtype)
        steps = _positive_times(times)
        p0 = history_context.get("p0_shared")
        v0 = history_context.get("v0_local")
        rotation = history_context.get("R_local_to_shared")
        if p0 is None or v0 is None or rotation is None:
            raise ValueError("history_context requires p0_shared, v0_local and R_local_to_shared")
        if p0.shape != (b, n, 3) or v0.shape != (b, n, 3) or rotation.shape != (b, n, 3, 3):
            raise ValueError("history_context tensor shapes do not match B/N")
        if not all(torch.isfinite(x[actor_mask]).all() for x in (p0, v0, rotation)):
            raise ValueError("real actors require finite history_context")
        # Yaw-only orthogonal transforms preserve the physical vertical axis.
        with torch.autocast(device_type=rotation.device.type, enabled=False):
            real_rotation = rotation[actor_mask].float()
            eye = torch.eye(3, device=rotation.device, dtype=real_rotation.dtype)
            if not torch.allclose(real_rotation.transpose(-1, -2) @ real_rotation, eye.expand_as(real_rotation), atol=1e-4, rtol=1e-4):
                raise ValueError("R_local_to_shared must be orthogonal")
            if not torch.allclose(real_rotation[..., 2, :], eye[2].expand_as(real_rotation[..., 2, :]), atol=1e-4, rtol=1e-4) or not torch.all(torch.linalg.det(real_rotation) > 0):
                raise ValueError("R_local_to_shared must be a proper yaw-only rotation preserving up")

        real_features = history_features[actor_mask]
        temporal = self.feature_projection(real_features) + self.history_time_encoding(real_features[..., 12:13])
        if self.temporal_variant == "mamba_patch":
            total_steps = temporal.shape[1]
            starts = torch.arange(0, total_steps - self.patch_len + 1, self.patch_stride,
                                  device=temporal.device)
            tail_start = total_steps - self.patch_len
            if starts[-1] != tail_start:
                starts = torch.cat((starts, starts.new_tensor([tail_start])))
            patches = torch.stack([temporal[:, start:start + self.patch_len] for start in starts], dim=1)
            patch_tokens = self.patch_norm(self.patch_encoder(self.patch_projection(patches.flatten(start_dim=-2))))
            patch_encoded = patch_tokens[:, -1]
            mamba_temporal = temporal
            for norm, block in zip(self.mamba_norms, self.mamba_encoder):
                mamba_temporal = mamba_temporal + self.mamba_dropout(block(norm(mamba_temporal)))
            mamba_encoded = self.temporal_norm(mamba_temporal[:, -1])
            if self.fusion_mode == "add":
                encoded = mamba_encoded + self.patch_add_projection(patch_encoded)
            elif self.fusion_mode == "concat":
                encoded = self.patch_concat_projection(torch.cat((mamba_encoded, patch_encoded), dim=-1))
            elif self.fusion_mode == "multiscale":
                patch_mean = patch_tokens.mean(dim=1)
                encoded = self.patch_fusion(torch.cat((mamba_encoded, patch_mean), dim=-1))
            else:
                fusion_input = torch.cat((mamba_encoded, patch_encoded), dim=-1)
                gate = torch.sigmoid(self.fusion_gate(fusion_input))
                patch_delta = self.patch_gate_projection(fusion_input)
                encoded = mamba_encoded + gate * patch_delta
            encoded = self.temporal_norm(encoded)
        elif self.temporal_kind == "transformer" and self.temporal_variant == "vanilla":
            temporal = self.temporal_encoder(temporal)
        elif self.temporal_kind == "transformer" and self.temporal_variant == "conformer":
            for block in self.temporal_encoder:
                temporal = block(temporal)
        elif self.temporal_kind == "transformer" and self.temporal_variant == "patchtst":
            total_steps = temporal.shape[1]
            if total_steps < self.patch_len:
                raise ValueError("history is shorter than patch_len")
            starts = torch.arange(
                0, total_steps - self.patch_len + 1, self.patch_stride,
                device=temporal.device,
            )
            tail_start = total_steps - self.patch_len
            if starts[-1] != tail_start:
                # Preserve the most recent history point when the sequence
                # length is not divisible by the patch stride.
                starts = torch.cat((starts, starts.new_tensor([tail_start])))
            patches = torch.stack(
                [temporal[:, start:start + self.patch_len] for start in starts], dim=1
            )
            patches = patches.flatten(start_dim=-2)
            temporal = self.patch_projection(patches)
            temporal = self.temporal_encoder(temporal)
        else:
            for norm, block in zip(self.mamba_norms, self.temporal_encoder):
                temporal = temporal + self.mamba_dropout(block(norm(temporal)))
        # The mamba_patch branch already produces its fused representation in
        # ``encoded``.  Do not overwrite it with the generic temporal pooling
        # path; doing so silently disabled every patch-fusion mode.
        if self.temporal_variant != "mamba_patch":
            pooled = (temporal[:, -1] if self.temporal_pool == "last" else
                      temporal.mean(dim=1) if self.temporal_pool == "mean" else
                      temporal.max(dim=1).values)
            encoded = self.temporal_norm(pooled)
        actors = encoded.new_zeros((b, n, self.width))
        actors[actor_mask] = encoded
        if self.interaction:
            expected = (b, n, n, self.edge_anchors, 13)
            if history_edges is None or history_edges.shape != expected:
                raise ValueError(f"interaction requires history_edges with shape {expected}")
            pair_mask = actor_mask[:, :, None] & actor_mask[:, None, :]
            if not torch.isfinite(history_edges[pair_mask]).all():
                raise ValueError("real actor pairs require finite history_edges")
            safe_edges = torch.where(pair_mask[..., None, None], history_edges, torch.zeros_like(history_edges))
            if not self.use_edge_geometry:
                # Remove only edge inputs; retain all projection biases and
                # relation-message parameters for a parameter-matched ablation.
                safe_edges = torch.zeros_like(safe_edges)
            elif self.edge_keep_anchors is not None or self.edge_keep_feats is not None:
                # Partial descriptor ablation: keep edge_projection and every
                # relation parameter; zero only deselected anchor/feature slices.
                # The [1,1,A,13] mask broadcasts over batch and actor pairs.
                keep = (self.edge_anchor_keep[None, None, :, None]
                        * self.edge_feat_keep[None, None, None, :])
                safe_edges = safe_edges * keep
            edges = self.edge_projection(safe_edges.flatten(start_dim=-2))
            if self.edge_geometry_transform == "symmetric":
                # Raw vectors live in different receiver-local frames. Do
                # not average them as though they shared a coordinate frame.
                # Symmetrize the learned edge representations instead, while
                # preserving actor slots, pair masks and all head parameters.
                # q/k attention can still be directional: this removes only
                # directed *edge representations*, not every directed path.
                edges = 0.5 * (edges + edges.transpose(1, 2))
            temporal_actors = actors
            for relation in self.relation_layers:
                actors = relation(actors, edges, actor_mask)
            if self.relation_fusion == "gated":
                gate = torch.sigmoid(self.relation_gate(torch.cat((temporal_actors, actors), dim=-1)))
                actors = temporal_actors + gate * (actors - temporal_actors)
                actors = torch.where(actor_mask[..., None], actors, torch.zeros_like(actors))

        if self.ascent_pose:
            # Add ASCENT's global endpoint/orientation context after relation
            # encoding so relative edge messages remain translation invariant.
            safe_p0 = torch.where(actor_mask[..., None], p0, torch.zeros_like(p0))
            safe_v0 = torch.where(actor_mask[..., None], v0, torch.zeros_like(v0))
            safe_rotation = torch.where(
                actor_mask[..., None, None], rotation, torch.zeros_like(rotation)
            )
            v_shared = torch.einsum("bnij,bnj->bni", safe_rotation, safe_v0)
            horizontal = torch.linalg.vector_norm(v_shared[..., :2], dim=-1).clamp_min(1e-6)
            speed = torch.linalg.vector_norm(v_shared, dim=-1).clamp_min(1e-6)
            pose = torch.cat((
                safe_p0 / 1000.0,
                (v_shared[..., 0:1] / horizontal[..., None]),
                (v_shared[..., 1:2] / horizontal[..., None]),
                (v_shared[..., 2:3] / speed[..., None]),
                (horizontal[..., None] / speed[..., None]),
            ), dim=-1)
            pose = torch.where(actor_mask[..., None], pose, torch.zeros_like(pose))
            actors = actors + self.ascent_pose_embedding(pose)

        if self.joint:
            query = self.mode_queries[None].expand(b, -1, -1)
            scene_context, _ = self.scene_attention(query, actors, actors, key_padding_mask=~actor_mask, need_weights=False)
            scene_context = self.scene_norm(query + scene_context)
            if self.joint_variant in ("factorized", "influence"):
                actor_tokens = actors[:, None, :, None, :].expand(-1, self.modes, -1, self.local_modes, -1)
                scene_tokens = scene_context[:, :, None, None, :].expand(-1, -1, n, self.local_modes, -1)
                # Directed soft influence graph. Edge i->j is scored from
                # source i and receiver j; the receiver gets a weighted
                # message before its local response modes are generated.
                source = actors[:, :, None, :].expand(-1, -1, n, -1)
                receiver = actors[:, None, :, :].expand(-1, n, -1, -1)
                pair_features = torch.cat((source, receiver), dim=-1)
                influence_logits = self.influence_score(pair_features).squeeze(-1)
                eye = torch.eye(n, device=actors.device, dtype=torch.bool)[None]
                influence_logits = influence_logits.masked_fill(eye, float("-inf"))
                influence_weights = torch.softmax(influence_logits, dim=1)
                influence_weights = torch.nan_to_num(influence_weights, nan=0.0)
                directed_messages = self.influence_message(pair_features)
                influence_context = (influence_weights[..., None] * directed_messages).sum(dim=1)
                conditioned_actors = actors + influence_context
                conditioned_actors = torch.where(actor_mask[..., None], conditioned_actors, torch.zeros_like(conditioned_actors))
                actor_tokens = conditioned_actors[:, None, :, None, :].expand(-1, self.modes, -1, self.local_modes, -1)
                local_queries = self.local_mode_queries[None, None, None, :, :]
                local_tokens = actor_tokens + local_queries
                local_residual = self.factorized_actor_residual(
                    torch.cat((local_tokens, scene_tokens), dim=-1)
                )
                local_logits = self.local_actor_score(
                    torch.cat((conditioned_actors[:, None, :, :].expand(-1, self.modes, -1, -1),
                               scene_context[:, :, None, :].expand(-1, -1, n, -1)), dim=-1)
                )
                if self.joint_variant == "influence":
                    # Influence-conditioned Joint retains one local response
                    # factor per receiver, but explicitly exposes the edge
                    # distribution for joint scoring and diagnostics.
                    influence_prior = influence_weights.mean(dim=1)
                    local_logits = local_logits + 0.25 * torch.log(
                        influence_prior.clamp_min(1e-6)
                    )[:, None, :, None]
                # Keep every local mode explicitly. Different actors may
                # choose different l under the same global scene mode g.
                # The local index is intentionally actor-specific: the
                # factorized decoder must not apply one shared l to every
                # actor before interaction attention.
                mode_actors = (actor_tokens + scene_tokens + local_residual).permute(0, 1, 3, 2, 4).contiguous()
            else:
                mode_actors = actors[:, None] + scene_context[:, :, None]
            if self.joint_variant == "shared":
                mode_count = self.modes
                flattened = mode_actors.reshape(b * mode_count, n, self.width)
                mode_mask = actor_mask[:, None].expand(b, mode_count, n).reshape(b * mode_count, n)
                message, _ = self.mode_actor_attention(flattened, flattened, flattened, key_padding_mask=~mode_mask, need_weights=False)
                flattened = self.mode_actor_norm(flattened + message)
                flattened = flattened + self.mode_actor_ffn(flattened)
                mode_actors = flattened.reshape(b, self.modes, n, self.width)
            if self.joint_variant in ("factorized", "influence"):
                mask = actor_mask[:, None, None, :, None].to(mode_actors.dtype)
                denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
                pooled_mean = (mode_actors * mask).sum(dim=(2, 3)) / denom
                masked = mode_actors.masked_fill(~actor_mask[:, None, None, :, None], float("-inf"))
                pooled_max = masked.amax(dim=(2, 3)).masked_fill(~torch.isfinite(masked.amax(dim=(2, 3))), 0.0)
                score_input = torch.cat((scene_context, pooled_mean, pooled_max), dim=-1)
                logits = {
                    "scene_logits": self.scene_score_fused(score_input).squeeze(-1),
                    "local_logits": local_logits,
                    "influence_weights": influence_weights,
                    "joint_factorized": True,
                    "joint_influence": self.joint_variant == "influence",
                    "joint_score_temperature": self.joint_score_temperature,
                    "joint_diversity_weight": self.joint_diversity_weight,
                }
            else:
                logits = {
                    "scene_logits": self.scene_score(scene_context).squeeze(-1),
                    "joint_score_temperature": self.joint_score_temperature,
                    "joint_diversity_weight": self.joint_diversity_weight,
                }
        else:
            mode_actors = actors[:, None] + self.mode_queries[None, :, None]
            actor_logits = self.actor_score(actors)
            logits = {"actor_logits": torch.where(actor_mask[..., None], actor_logits, torch.zeros_like(actor_logits))}
        factorized_shape = self.joint and self.joint_variant in ("factorized", "influence")
        if factorized_shape:
            mode_actors = mode_actors.reshape(b, self.modes * self.local_modes, n, self.width)
        decoder_mode_count = mode_actors.shape[1]
        goals_local = None
        if self.goal_head_enabled:
            # The goal is a residual around constant-velocity extrapolation. It
            # is predicted from history and relation features only; truth enters
            # only through trajectory_loss.
            coarse_residual = torch.tanh(self.goal_head(mode_actors)) * self.goal_scale_m
            initial_v = torch.where(actor_mask[..., None], v0, torch.zeros_like(v0))
            coarse_goal = initial_v[:, None] * times[-1] + coarse_residual
            refine_input = torch.cat((mode_actors, (coarse_goal / self.goal_scale_m).detach()), dim=-1)
            refinement = torch.tanh(self.goal_refine(refine_input)) * self.goal_scale_m
            goals_local = coarse_goal + refinement
            goals_local = torch.where(actor_mask[:, None, :, None], goals_local, torch.zeros_like(goals_local))
            if self.intent_conditioning == "goal":
                mode_actors = mode_actors + self.goal_embedding(goals_local / self.goal_scale_m)
            if self.goal_conditioning and self.intent_conditioning != "goal":
                condition = self.goal_embedding((goals_local / self.goal_scale_m).detach())
                mode_actors = mode_actors + condition
            if self.future_relation_enabled:
                # FINet-inspired future-aware interaction: predicted intent
                # endpoints become scene tokens before temporal decoding.
                # The endpoints are history-only predictions, never labels.
                goal_tokens = mode_actors + self.future_goal_embedding(goals_local / self.goal_scale_m)
                flattened = goal_tokens.reshape(b * decoder_mode_count, n, self.width)
                goal_mask = actor_mask[:, None].expand(b, decoder_mode_count, n).reshape(b * decoder_mode_count, n)
                message, _ = self.future_relation_attention(
                    flattened, flattened, flattened, key_padding_mask=~goal_mask, need_weights=False
                )
                flattened = self.future_relation_norm(flattened + message)
                flattened = flattened + self.future_relation_ffn(flattened)
                mode_actors = flattened.reshape(b, decoder_mode_count, n, self.width)
        if self.intent_conditioning == "direction":
            direction_logits = self.intent_head(mode_actors)
            direction = direction_logits / direction_logits.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            mode_actors = mode_actors + self.intent_embedding(direction)
            logits["intent_direction"] = direction
        time_features = torch.stack((times / 10.0, steps), dim=-1).to(history_features.dtype)
        time_embedding = self.future_time_encoding(time_features)
        if self.state_queries_enabled:
            if times.numel() > self.max_future_steps:
                raise ValueError("forecast horizon exceeds state-query table")
            state_queries = self.state_queries[:times.numel()].to(dtype=mode_actors.dtype)
            decoder_features = (
                mode_actors[..., None, :]
                + time_embedding[None, None, None]
                + state_queries[None, None, None]
            )
        else:
            decoder_features = mode_actors[..., None, :] + time_embedding[None, None, None]
        safe_v0 = torch.where(actor_mask[..., None], v0, torch.zeros_like(v0))
        if self.motion_decoder in ("acceleration", "coordinate"):
            proposal = self.acceleration_head(decoder_features)
            proposal = torch.where(actor_mask[:, None, :, None, None], proposal, torch.zeros_like(proposal))
            if self.motion_decoder == "coordinate":
                local_positions, velocities, diagnostics = decode_coordinate_residual(
                    proposal, safe_v0, times, self.acceleration_scale,
                )
            else:
                local_positions, velocities, diagnostics = integrate_acceleration(
                    proposal, safe_v0, times, self.constraints, self.acceleration_scale,
                    constraint_mode=self.constraint_mode,
                )
        else:
            scalar = self.motion_scalar_head(decoder_features).squeeze(-1)
            yaw_pair = self.motion_yaw_head(decoder_features)
            pitch_pair = self.motion_pitch_head(decoder_features)
            yaw_norm = torch.linalg.vector_norm(yaw_pair, dim=-1).clamp_min(1e-6)
            pitch_norm = torch.linalg.vector_norm(pitch_pair, dim=-1).clamp_min(1e-6)
            yaw = torch.atan2(yaw_pair[..., 0] / yaw_norm, yaw_pair[..., 1] / yaw_norm)
            pitch = torch.atan2(pitch_pair[..., 0] / pitch_norm, pitch_pair[..., 1] / pitch_norm)
            if self.motion_decoder == "flight":
                base_speed = torch.linalg.vector_norm(safe_v0, dim=-1).clamp_min(1.0)
                base_yaw = torch.atan2(safe_v0[..., 1], safe_v0[..., 0])
                base_pitch = torch.atan2(
                    safe_v0[..., 2], torch.linalg.vector_norm(safe_v0[..., :2], dim=-1).clamp_min(1e-6)
                )
                scalar = base_speed[:, None, :, None] * torch.exp(0.5 * torch.tanh(scalar))
                yaw = base_yaw[:, None, :, None] + 0.8 * torch.tanh(yaw)
                pitch = base_pitch[:, None, :, None] + 0.5 * torch.tanh(pitch)
            direction = torch.stack((
                torch.cos(yaw) * torch.cos(pitch),
                torch.sin(yaw) * torch.cos(pitch),
                torch.sin(pitch),
            ), dim=-1)
            dt = steps.to(dtype=decoder_features.dtype)[None, None, None, :, None]
            increments = scalar[..., None] * direction * dt
            if self.motion_decoder == "ascent":
                # Official ASCENT predicts a normalized displacement per
                # output index. Use metres here while retaining its dt=1-like
                # cumulative parameterization as a controlled comparison.
                increments = increments * 1000.0
            if self.motion_decoder == "hybrid":
                # Convert the structured velocity prior to acceleration and
                # fuse it with the learned acceleration head before using the
                # existing physical integrator. The gate starts near zero so
                # the hybrid path is initialized close to the baseline.
                flight_velocity = scalar[..., None] * direction
                initial_velocity = safe_v0[:, None, :, None, :].expand(-1, decoder_mode_count, -1, -1, -1)
                previous_velocity = torch.cat((initial_velocity, flight_velocity[..., :-1, :]), dim=-2)
                flight_acceleration = (flight_velocity - previous_velocity) / dt.clamp_min(torch.finfo(flight_velocity.dtype).eps)
                base_proposal = self.acceleration_head(decoder_features)
                flight_proposal = flight_acceleration / self.acceleration_scale
                gate = torch.sigmoid(self.motion_gate(mode_actors))[..., None, :]
                proposal = (1.0 - gate) * base_proposal + gate * flight_proposal
                proposal = torch.where(actor_mask[:, None, :, None, None], proposal, torch.zeros_like(proposal))
                local_positions, velocities, diagnostics = integrate_acceleration(
                    proposal, safe_v0, times, self.constraints, self.acceleration_scale,
                    constraint_mode=self.constraint_mode,
                )
                diagnostics["decoder"] = "hybrid"
            else:
                local_positions = torch.cumsum(increments, dim=-2)
                velocities = increments / dt.clamp_min(torch.finfo(increments.dtype).eps)
                diagnostics = {
                    "constrained": False,
                    "constraint_mode": "none",
                    "active_bounds": [],
                    "velocity_projection_enabled": False,
                    "decoder": self.motion_decoder,
                    "projection_active": None,
                }
        safe_rotation = torch.where(actor_mask[..., None, None], rotation, torch.zeros_like(rotation))
        safe_p0 = torch.where(actor_mask[..., None], p0, torch.zeros_like(p0))
        # Autocast must not reduce precision in metre-space coordinate recovery.
        with torch.autocast(device_type=local_positions.device.type, enabled=False):
            shared = torch.einsum("bnij,bknhj->bknhi", safe_rotation.to(local_positions.dtype), local_positions)
            shared = shared + safe_p0.to(local_positions.dtype)[:, None, :, None]
            goals_shared = None
            if goals_local is not None:
                goals_shared = torch.einsum(
                    "bnij,bknj->bkni", safe_rotation.to(goals_local.dtype), goals_local
                ) + safe_p0.to(goals_local.dtype)[:, None]
                if self.endpoint_correction_enabled:
                    terminal_delta = goals_shared - shared[..., -1, :]
                    gate_input = torch.cat(
                        (mode_actors, terminal_delta / self.goal_scale_m), dim=-1
                    )
                    gate = torch.sigmoid(self.endpoint_correction_gate(gate_input)).squeeze(-1)
                    progress = (steps / steps[-1].clamp_min(torch.finfo(steps.dtype).eps)).to(shared.dtype)
                    shared = shared + gate[..., None, None] * progress[None, None, None, :, None] * terminal_delta[..., None, :]
        shared = torch.where(actor_mask[:, None, :, None, None], shared, torch.zeros_like(shared))
        if factorized_shape:
            shared = shared.reshape(b, self.modes, self.local_modes, n, shared.shape[-2], 3)
            velocities = velocities.reshape(b, self.modes, self.local_modes, n, velocities.shape[-2], 3)
            if goals_local is not None:
                goals_local = goals_local.reshape(b, self.modes, self.local_modes, n, 3)
            if goals_shared is not None:
                goals_shared = goals_shared.reshape(b, self.modes, self.local_modes, n, 3)
        return {
            "trajectories": shared,
            **logits,
            **({"goals_local": goals_local} if goals_local is not None else {}),
            **({"goals_shared": goals_shared} if goals_shared is not None else {}),
            "local_velocities": velocities,
            "constraint_diagnostics": diagnostics,
        }


def endpoint_region_error(prediction: Tensor, target: Tensor, radius_m: float = 0.0,
                          geometry: str = "coordinate") -> Tensor:
    """Per-actor endpoint loss, with identical geometry at every radius.

    This is a training tolerance ball, not a calibrated uncertainty region.
    Radial mode uses the same metre norm and Huber scale including radius=0.
    Coordinate mode at radius=0 exactly retains the historical point loss.
    """
    if geometry not in ("coordinate", "radial"):
        raise ValueError("Unknown endpoint loss geometry")
    if not math.isfinite(radius_m) or radius_m < 0:
        raise ValueError("radius must be finite and nonnegative")
    delta = prediction - target
    distance = torch.linalg.vector_norm(delta, dim=-1)
    if geometry == "radial":
        excess = F.relu(distance - radius_m) / 1000.0
        return F.smooth_l1_loss(excess, torch.zeros_like(excess), beta=.01, reduction="none")
    if radius_m == 0:
        return F.smooth_l1_loss(prediction / 1000., target / 1000., beta=.01, reduction="none").mean(-1)
    excess_delta = delta * (F.relu(distance - radius_m) / distance.clamp_min(1e-8)).unsqueeze(-1)
    return F.smooth_l1_loss(excess_delta / 1000., torch.zeros_like(delta), beta=.01, reduction="none").mean(-1)


def trajectory_loss(
    output: Dict[str, Tensor],
    target: Tensor,
    valid: Tensor,
    actor_mask: Tensor,
    score_weight: float = 0.1,
    joint: bool = True,
    goal_weight: float = 0.0,
    goal_target_local: Optional[Tensor] = None,
    goal_valid: Optional[Tensor] = None,
    goal_soft_temperature: float = 10.0,
    goal_radius_m: float = 0.0,
    goal_loss_geometry: str = "coordinate",
    endpoint_consistency_weight: float = 0.0,
    intent_target_direction: Optional[Tensor] = None,
    intent_valid: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, Any]]:
    """Multimodal trajectory loss with shared or factorized Joint semantics.

    Factorized Joint first selects a global scene mode and then an independent
    local response mode for each actor under that scene. SmoothL1 uses
    coordinates /1000 m, beta=.01; score targets use detached ADE winners.
    """
    prediction = output["trajectories"]
    factorized = bool(output.get("joint_factorized", False))
    if factorized:
        if prediction.ndim != 6 or prediction.shape[-1] != 3:
            raise ValueError("factorized trajectories must have shape [B,G,L,N,H,3]")
        b, g, l, n, h, _ = prediction.shape
        k = g
    else:
        if prediction.ndim != 5 or prediction.shape[-1] != 3:
            raise ValueError("trajectories must have shape [B,K,N,H,3]")
        b, k, n, h, _ = prediction.shape
    if target.shape != (b, n, h, 3) or valid.shape != (b, n, h) or actor_mask.shape != (b, n):
        raise ValueError("target, valid and actor_mask shapes must match trajectories")
    if valid.dtype != torch.bool or actor_mask.dtype != torch.bool:
        raise ValueError("valid and actor_mask must be boolean")
    if not math.isfinite(score_weight) or score_weight < 0:
        raise ValueError("score_weight must be finite and nonnegative")
    if not math.isfinite(goal_weight) or goal_weight < 0:
        raise ValueError("goal_weight must be finite and nonnegative")
    if not math.isfinite(endpoint_consistency_weight) or endpoint_consistency_weight < 0:
        raise ValueError("endpoint_consistency_weight must be finite and nonnegative")
    if not math.isfinite(goal_soft_temperature) or goal_soft_temperature <= 0:
        raise ValueError("goal_soft_temperature must be finite and positive")
    if not math.isfinite(goal_radius_m) or goal_radius_m < 0:
        raise ValueError("goal_radius_m must be finite and nonnegative")
    if goal_loss_geometry not in ("coordinate", "radial"):
        raise ValueError("goal_loss_geometry must be coordinate or radial")
    valid = valid & actor_mask[..., None]
    if not torch.isfinite(target[valid]).all():
        raise ValueError("target must be finite under the effective true mask")
    valid_pred = valid[:, None, :, :, None]
    if factorized:
        valid_pred = valid[:, None, None, :, :, None].expand_as(prediction)
    else:
        valid_pred = valid_pred.expand_as(prediction)
    if not torch.isfinite(prediction[valid_pred]).all():
        raise ValueError("predictions must be finite at supervised coordinates")
    count = valid.sum(dim=-1)
    supervised_actor = count > 0
    supervised_scene = supervised_actor.any(dim=-1)
    if not supervised_scene.any():
        zero = torch.nan_to_num(prediction.reshape(-1)[:1], nan=0.0, posinf=0.0, neginf=0.0).sum() * 0
        return zero, {"loss_trajectory": 0.0, "loss_score": 0.0, "matched_ADE_m": None,
                      "n_supervised_scenes": 0, "n_supervised_actors": 0, "mode_counts": [0] * k}
    safe_target = torch.where(valid[..., None], target, torch.zeros_like(target))
    if factorized:
        safe_prediction = torch.where(valid[:, None, None, :, :, None], prediction, torch.zeros_like(prediction))
        actor_ade = torch.linalg.vector_norm(safe_prediction - safe_target[:, None, None], dim=-1).sum(dim=-1) / count[:, None, None].clamp_min(1)
    else:
        safe_prediction = torch.where(valid[:, None, :, :, None], prediction, torch.zeros_like(prediction))
        actor_ade = torch.linalg.vector_norm(safe_prediction - safe_target[:, None], dim=-1).sum(dim=-1) / count[:, None].clamp_min(1)
    scene_actor_count = supervised_actor.sum(dim=-1).clamp_min(1)
    if joint:
        if factorized:
            local_best = actor_ade.detach().argmin(dim=2)  # [B,G,N]
            local_selected_ade = actor_ade.gather(2, local_best[:, :, None, :]).squeeze(2)
            scene_ade = (local_selected_ade * supervised_actor[:, None]).sum(dim=-1) / scene_actor_count[:, None]
            best = scene_ade.detach().argmin(dim=1)
            chosen_local = local_best[torch.arange(b, device=prediction.device), best]
            per_actor = safe_prediction[
                torch.arange(b, device=prediction.device)[:, None],
                best[:, None], chosen_local,
                torch.arange(n, device=prediction.device)[None, :]
            ]
            selected = per_actor
            scores = output.get("scene_logits")
            local_scores = output.get("local_logits")
            if scores is None or scores.shape != (b, g) or not torch.isfinite(scores[supervised_scene]).all():
                raise ValueError("factorized joint requires finite scene_logits[B,G]")
            if local_scores is None or local_scores.shape != (b, g, n, l) or not torch.isfinite(local_scores[supervised_actor[:, None].expand(b, g, n)]).all():
                raise ValueError("factorized joint requires finite local_logits[B,G,N,L]")
            local_log_probs = F.log_softmax(local_scores, dim=-1)
            # The scene classifier is trained on the same factorized joint
            # score used at inference: global log-probability plus the sum of
            # actor-local log-probabilities for the target-matching tuple.
            best_local_log_prob = local_log_probs.gather(
                -1, local_best[..., None]
            ).squeeze(-1)
            joint_candidate_scores = scores / float(output.get("joint_score_temperature", 1.0)) + (
                best_local_log_prob * supervised_actor[:, None]
            ).sum(dim=-1)
            score_loss = F.cross_entropy(
                joint_candidate_scores[supervised_scene], best[supervised_scene]
            )
            chosen_local_logits = local_scores[
                torch.arange(b, device=prediction.device), best
            ]
            chosen_local_ce = F.cross_entropy(
                chosen_local_logits.reshape(b * n, l),
                chosen_local.reshape(-1), reduction="none"
            ).reshape(b, n)
            local_score_loss = (
                (chosen_local_ce * supervised_actor).sum(dim=-1)
                / scene_actor_count
            )[supervised_scene].mean()
            score_loss = score_loss + local_score_loss
            matched_ade = scene_ade[supervised_scene, best[supervised_scene]].mean()
            selected_modes = best[supervised_scene]
        else:
            scene_ade = (actor_ade * supervised_actor[:, None]).sum(dim=-1) / scene_actor_count[:, None]
            best = scene_ade.detach().argmin(dim=1)
            selected = safe_prediction[torch.arange(b, device=prediction.device), best]
            scores = output.get("scene_logits")
            if scores is None or scores.shape != (b, k) or not torch.isfinite(scores[supervised_scene]).all():
                raise ValueError("joint loss requires finite scene_logits[B,K] for supervised scenes")
            temperature = float(output.get("joint_score_temperature", 1.0))
            score_loss = F.cross_entropy((scores / temperature)[supervised_scene], best[supervised_scene])
            matched_ade = scene_ade[supervised_scene, best[supervised_scene]].mean()
            selected_modes = best[supervised_scene]
    else:
        best = actor_ade.detach().argmin(dim=1)
        candidates = safe_prediction.permute(0, 2, 1, 3, 4)
        selected = candidates.gather(2, best[:, :, None, None, None].expand(b, n, 1, h, 3)).squeeze(2)
        scores = output.get("actor_logits")
        if scores is None or scores.shape != (b, n, k) or not torch.isfinite(scores[supervised_actor]).all():
            raise ValueError("independent loss requires finite actor_logits[B,N,K] for supervised actors")
        safe_scores = torch.where(supervised_actor[..., None], scores, torch.zeros_like(scores))
        actor_ce = F.cross_entropy(safe_scores.reshape(b * n, k), best.reshape(-1), reduction="none").reshape(b, n)
        score_loss = ((actor_ce * supervised_actor).sum(dim=-1) / scene_actor_count)[supervised_scene].mean()
        selected_ade = actor_ade.gather(1, best[:, None]).squeeze(1)
        matched_ade = ((selected_ade * supervised_actor).sum(dim=-1) / scene_actor_count)[supervised_scene].mean()
        selected_modes = best[supervised_actor]
    # Intent direction is supervised on the same target-matching candidate
    # selected above, so it cannot leak a future mode choice into the loss.
    intent_direction = output.get("intent_direction")
    intent_loss = prediction.new_zeros(())
    if intent_direction is not None:
        if intent_target_direction is None or intent_valid is None:
            raise ValueError("direction intent output requires target and validity")
        expected_intent_shape = (b, g * l, n, 3) if factorized else (b, k, n, 3)
        if intent_direction.shape != expected_intent_shape:
            raise ValueError("direction intent output shape is invalid")
        if intent_target_direction.shape != (b, n, 3) or intent_valid.shape != (b, n) or intent_valid.dtype != torch.bool:
            raise ValueError("direction intent target shapes are invalid")
        safe_dir = torch.where(intent_valid[..., None], intent_target_direction, torch.zeros_like(intent_target_direction))
        if factorized:
            if not joint:
                raise ValueError("factorized direction intent requires joint loss")
            flat_mode = best[:, None] * l + chosen_local
            chosen_dir = intent_direction[
                torch.arange(b, device=prediction.device)[:, None], flat_mode,
                torch.arange(n, device=prediction.device)[None, :]
            ]
        elif joint:
            chosen_dir = intent_direction[torch.arange(b, device=prediction.device), best]
        else:
            chosen_dir = intent_direction.permute(0, 2, 1, 3).gather(
                2, best[:, :, None, None].expand(b, n, 1, 3)
            ).squeeze(2)
        err = (chosen_dir - safe_dir).square().mean(dim=-1)
        intent_loss = (err * intent_valid).sum() / intent_valid.sum().clamp_min(1)
    coordinate_loss = F.smooth_l1_loss(selected / 1000.0, safe_target / 1000.0, beta=0.01, reduction="none").mean(dim=-1)
    actor_loss = (coordinate_loss * valid).sum(dim=-1) / count.clamp_min(1)
    scene_loss = (actor_loss * supervised_actor).sum(dim=-1) / scene_actor_count
    trajectory = scene_loss[supervised_scene].mean()
    endpoint_consistency = prediction.new_zeros(())
    if endpoint_consistency_weight > 0:
        goals_shared = output.get("goals_shared")
        endpoint_valid = valid[..., -1]
        if goals_shared is None:
            raise ValueError("endpoint consistency requires output goals_shared")
        if factorized:
            expected = (b, g, l, n, 3)
            if goals_shared.shape != expected:
                raise ValueError("factorized goals_shared shape is invalid")
            # Use the same actor-local response tuple as scene ADE for each
            # global candidate; no future target enters this selection.
            candidate_end = prediction[..., -1, :].permute(0, 1, 3, 2, 4).gather(
                3, local_best[..., None, None].expand(b, g, n, 1, 3)
            ).squeeze(3)
            candidate_goal = goals_shared.permute(0, 1, 3, 2, 4).gather(
                3, local_best[..., None, None].expand(b, g, n, 1, 3)
            ).squeeze(3)
        else:
            expected = (b, k, n, 3)
            if goals_shared.shape != expected:
                raise ValueError("goals_shared shape is invalid")
            candidate_end = prediction[..., -1, :]
            candidate_goal = goals_shared
        error = F.smooth_l1_loss(
            (candidate_end - candidate_goal) / 1000.0,
            torch.zeros_like(candidate_end), beta=0.01, reduction="none"
        ).mean(dim=-1)
        candidate_actor = (error * endpoint_valid[:, None]).sum(dim=-1) / endpoint_valid.sum(dim=-1, keepdim=True).clamp_min(1)
        if joint:
            responsibility = torch.softmax(-scene_ade.detach() / float(goal_soft_temperature), dim=1)
            endpoint_consistency = (candidate_actor * responsibility).sum(dim=1)
            endpoint_consistency = endpoint_consistency[endpoint_valid.any(dim=-1)].mean()
        else:
            selected_end = candidate_actor.gather(1, best[:, None]).squeeze(1)
            endpoint_consistency = selected_end[endpoint_valid.any(dim=-1)].mean()
    goal_loss = prediction.new_zeros(())
    goals = output.get("goals_local")
    if goal_weight > 0:
        if goals is None or goal_target_local is None or goal_valid is None:
            raise ValueError("goal output, local target and validity are required")
        expected_goal_shape = (b, g, l, n, 3) if factorized else (b, k, n, 3)
        if goals.shape != expected_goal_shape or goal_target_local.shape != (b, n, 3) or goal_valid.shape != (b, n) or goal_valid.dtype != torch.bool:
            raise ValueError("goal tensor shapes or dtype are invalid")
        goal_actor_mask = actor_mask[:, None, None].expand(b, g, l, n) if factorized else actor_mask[:, None].expand(b, k, n)
        if not torch.isfinite(goal_target_local[goal_valid]).all() or not torch.isfinite(goals[goal_actor_mask]).all():
            raise ValueError("goal tensors must be finite on effective actors")
        # Sanitize masked targets before the elementwise loss. Multiplying a
        # NaN loss by False later does not remove the NaN in IEEE arithmetic.
        safe_goal_target = torch.where(
            goal_valid.unsqueeze(-1), goal_target_local, torch.zeros_like(goal_target_local)
        )
        if joint:
            if factorized:
                chosen_goals = goals[torch.arange(b, device=prediction.device)[:, None], best[:, None], chosen_local, torch.arange(n, device=prediction.device)[None, :]]
            else:
                chosen_goals = goals[torch.arange(b, device=prediction.device), best]
            # Candidate-level soft endpoint supervision.  Hard winner-only
            # supervision is noisy and leaves the other scene modes
            # unconstrained.  Use trajectory ADE as a detached responsibility
            # distribution so every candidate receives a stable endpoint
            # signal while the best candidate remains dominant.
            if factorized:
                # Same actor-local tuples used to compute each global scene ADE.
                candidate_goals = goals.permute(0, 1, 3, 2, 4).gather(
                    3, local_best[..., None, None].expand(b, g, n, 1, 3)
                ).squeeze(3)
                candidate_ade = scene_ade
            else:
                candidate_goals = goals
                candidate_ade = scene_ade
            responsibility = torch.softmax(-candidate_ade.detach() / float(goal_soft_temperature), dim=1)
            all_goal_err = endpoint_region_error(candidate_goals,
                safe_goal_target[:, None].expand_as(candidate_goals), goal_radius_m, goal_loss_geometry)
            actor_goal_err = (all_goal_err * goal_valid[:, None]).sum(dim=-1) / goal_valid.sum(dim=-1, keepdim=True).clamp_min(1)
            goal_loss = (actor_goal_err * responsibility).sum(dim=1)
            goal_loss = goal_loss[goal_valid.any(dim=-1)].mean() if goal_valid.any(dim=-1).any() else prediction.new_zeros(())
        else:
            chosen_goals = goals.permute(0, 2, 1, 3).gather(2, best[:, :, None, None].expand(b, n, 1, 3)).squeeze(2)
            goal_err = endpoint_region_error(chosen_goals, safe_goal_target, goal_radius_m, goal_loss_geometry)
            goal_loss = (goal_err * goal_valid).sum(dim=-1) / goal_valid.sum(dim=-1).clamp_min(1)
            goal_loss = goal_loss[goal_valid.any(dim=-1)].mean() if goal_valid.any(dim=-1).any() else prediction.new_zeros(())
    diversity_loss = prediction.new_zeros(())
    if joint and getattr(output, "get", None) is not None and output.get("joint_factorized", False):
        # Keep the factorized path's local candidates separated. This is only
        # active when explicitly enabled and is computed on predicted endpoint
        # distances, never on future labels.
        diversity_weight = float(output.get("joint_diversity_weight", 0.0))
        if diversity_weight > 0:
            flat_pred = prediction[..., -1, :]
            if flat_pred.shape[2] > 1:
                pair = torch.cdist(flat_pred.transpose(1, 2), flat_pred.transpose(1, 2))
                diversity_loss = diversity_weight * torch.exp(-pair / 100.0).mean()
    total = (trajectory + score_weight * score_loss + goal_weight * goal_loss
             + endpoint_consistency_weight * endpoint_consistency
             + 0.05 * intent_loss + diversity_loss)
    diagnostics = {
        "loss_trajectory": float(trajectory.detach()),
        "loss_score": float(score_loss.detach()),
        "loss_goal": float(goal_loss.detach()),
        "loss_endpoint_consistency": float(endpoint_consistency.detach()),
        "loss_intent": float(intent_loss.detach()),
        "loss_diversity": float(diversity_loss.detach()),
        "matched_ADE_m": float(matched_ade.detach()),
        "n_supervised_scenes": int(supervised_scene.sum()),
        "n_supervised_actors": int(supervised_actor.sum()),
        "mode_counts": torch.bincount(selected_modes, minlength=k).detach().cpu().tolist(),
    }
    return total, diagnostics


def assemble_independent_scenes(
    trajectory: np.ndarray,
    actor_logits: np.ndarray,
    scene_modes: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select up to scene_modes unique product-score actor-mode tuples.

    Input trajectories are [K,N,H,3], logits [N,K]. Returns [S,N,H,3]
    and [S] summed actor log-probabilities, S=min(scene_modes,K**N).
    The heap explores only top tuples, not all K**N combinations. Ties are
    broken by mode rank/index; no truth is accepted or consulted. Unique
    refers to actor-mode tuples, not necessarily distinct trajectory geometry.
    K=1 returns one scene instead of duplicating it to fill the budget.
    """
    trajectory = np.asarray(trajectory)
    actor_logits = np.asarray(actor_logits, dtype=np.float64)
    if trajectory.ndim != 4 or trajectory.shape[-1] != 3:
        raise ValueError("trajectory must have shape [K,N,H,3]")
    k, n, h, _ = trajectory.shape
    if min(k, n, h) < 1 or actor_logits.shape != (n, k):
        raise ValueError("positive K/N/H and actor_logits[N,K] required")
    if isinstance(scene_modes, bool) or not isinstance(scene_modes, (int, np.integer)) or scene_modes < 1:
        raise ValueError("scene_modes must be a positive integer")
    if not np.isfinite(trajectory).all() or not np.isfinite(actor_logits).all():
        raise ValueError("trajectory and actor_logits must be finite")
    log_probs = actor_logits - np.logaddexp.reduce(actor_logits, axis=1, keepdims=True)
    order = np.argsort(-log_probs, axis=1, kind="stable")
    sorted_log_probs = np.take_along_axis(log_probs, order, axis=1)
    start = (0,) * n

    def tuple_score(ranks: Tuple[int, ...]) -> float:
        return float(sum(sorted_log_probs[i, rank] for i, rank in enumerate(ranks)))

    heap = [(-tuple_score(start), start)]
    seen = {start}
    scenes, scores = [], []
    limit = min(int(scene_modes), k ** n)
    while heap and len(scenes) < limit:
        negative_score, ranks = heapq.heappop(heap)
        modes = np.array([order[i, rank] for i, rank in enumerate(ranks)])
        scenes.append(trajectory[modes, np.arange(n)])
        scores.append(-negative_score)
        for actor in range(n):
            if ranks[actor] + 1 >= k:
                continue
            neighbor = ranks[:actor] + (ranks[actor] + 1,) + ranks[actor + 1:]
            if neighbor not in seen:
                seen.add(neighbor)
                heapq.heappush(heap, (-tuple_score(neighbor), neighbor))
    return np.stack(scenes), np.asarray(scores, dtype=np.float64)


def assemble_factorized_scenes(
    trajectory: np.ndarray,
    scene_logits: np.ndarray,
    local_logits: np.ndarray,
    scene_modes: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble top joint tuples from global and actor-local factors.

    ``trajectory`` is ``[G,L,N,H,3]``; ``local_logits`` is ``[G,N,L]``.
    A candidate is a tuple ``(g,l_1,...,l_N)`` and its score is the global
    log-probability plus the sum of actor-local log-probabilities.  The heap
    enumerates the highest-scoring tuples without consulting future labels.
    This is the inference counterpart of the factorized training objective.
    """
    trajectory = np.asarray(trajectory)
    scene_logits = np.asarray(scene_logits, dtype=np.float64)
    local_logits = np.asarray(local_logits, dtype=np.float64)
    if trajectory.ndim != 5 or trajectory.shape[-1] != 3:
        raise ValueError("factorized trajectory must have shape [G,L,N,H,3]")
    g, l, n, h, _ = trajectory.shape
    if scene_logits.shape != (g,) or local_logits.shape != (g, n, l):
        raise ValueError("factorized logits must have shapes [G] and [G,N,L]")
    if min(g, l, n, h) < 1 or scene_modes < 1:
        raise ValueError("factorized dimensions and scene_modes must be positive")
    if not np.isfinite(trajectory).all() or not np.isfinite(scene_logits).all() or not np.isfinite(local_logits).all():
        raise ValueError("factorized candidates and logits must be finite")
    scene_logp = scene_logits - np.logaddexp.reduce(scene_logits)
    local_logp = local_logits - np.logaddexp.reduce(local_logits, axis=-1, keepdims=True)
    scene_order = np.argsort(-scene_logp, kind="stable")
    local_order = np.argsort(-local_logp, axis=-1, kind="stable")
    local_sorted = np.take_along_axis(local_logp, local_order, axis=-1)
    # Heap state: (negative score, global rank, local rank tuple).
    start = (0,) * n
    def score(grank: int, ranks: Tuple[int, ...]) -> float:
        gi = int(scene_order[grank])
        return float(scene_logp[gi] + sum(local_sorted[gi, i, rank] for i, rank in enumerate(ranks)))
    # Seed every global mode. Otherwise a low-probability global mode cannot
    # enter the top-S list even when its local product has a better tuple.
    heap = [(-score(grank, start), grank, start) for grank in range(g)]
    heapq.heapify(heap)
    seen = {(grank, start) for grank in range(g)}
    candidates, scores = [], []
    limit = min(int(scene_modes), g * (l ** n))
    while heap and len(candidates) < limit:
        neg, grank, ranks = heapq.heappop(heap)
        gi = int(scene_order[grank])
        modes = np.asarray([local_order[gi, i, rank] for i, rank in enumerate(ranks)], dtype=np.int64)
        candidates.append(trajectory[gi, modes, np.arange(n)])
        scores.append(float(-neg))
        for actor in range(n):
            if ranks[actor] + 1 < l:
                nxt = ranks[:actor] + (ranks[actor] + 1,) + ranks[actor + 1:]
                state = (grank, nxt)
                if state not in seen:
                    seen.add(state)
                    heapq.heappush(heap, (-score(grank, nxt), grank, nxt))
        if grank + 1 < g:
            state = (grank + 1, start)
            if state not in seen:
                seen.add(state)
                heapq.heappush(heap, (-score(grank + 1, start), grank + 1, start))
    return np.stack(candidates), np.asarray(scores, dtype=np.float64)
