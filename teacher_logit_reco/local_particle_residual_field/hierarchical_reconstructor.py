"""Step 6 local streams and non-hierarchical capacity controls.

The hierarchical/global components intentionally remain absent here.  This
module establishes the exact particle interface that Step 7 consumes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import hashlib
import math
from typing import Any, Mapping, Sequence

import torch

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from .bridge_contracts import with_content_hash
from .bridge_losses import (
    DirectedNeighborGraph,
    build_directed_neighbor_graph,
    compute_c0_objective,
    resolve_c0_loss_recipe,
)
from .bridge_reconstructor import (
    C0CorrectionOutput,
    NUMERICAL_SPACE_CONDITIONING,
    NUMERICAL_SPACE_CORRECTION,
    NUMERICAL_SPACE_LOSS,
    NUMERICAL_SPACE_PHYSICAL,
    PHYSICAL_CHANNELS,
    TorchBridgeScalers,
    _masked_jet_summary,
)


PREDICTION_ANCHORED_LOCAL_CONFIG_CONTRACT = (
    "prediction_anchored_local_correction_config_v1"
)
PREDICTION_ANCHORED_LOCAL_GRAPH_CONTRACT = (
    "prediction_anchored_local_graph_features_v1"
)
PREDICTION_ANCHORED_PARTICLE_INTERFACE_CONTRACT = (
    "prediction_anchored_particle_reasoning_interface_v1"
)
PREDICTION_ANCHORED_RESOURCE_MEASUREMENT_CONTRACT = (
    "prediction_anchored_correction_resource_measurement_v1"
)
PREDICTION_ANCHORED_STEP6_MEASUREMENT_CONTRACT = (
    "prediction_anchored_step6_registry_measurement_v1"
)
PREDICTION_ANCHORED_STEP6_RELOAD_CONTRACT = (
    "prediction_anchored_step6_tiny_train_reload_v1"
)

ARCH_A0M_CAPACITY_PARTICLE = "D10_A0M_capacity_particle"
ARCH_A1_MULTISCALE_LOCAL = "D10_A1_multiscale_local"
ARCH_A1H_HARD_RADIUS = "D10_A1H_hard_radius"
STEP6_ARCHITECTURE_IDS = (
    ARCH_A0M_CAPACITY_PARTICLE,
    ARCH_A1_MULTISCALE_LOCAL,
    ARCH_A1H_HARD_RADIUS,
)
KERNEL_GAUSSIAN = "gaussian"
KERNEL_HARD_RADIUS = "hard_radius"
KERNEL_CAPACITY_UNIFORM = "capacity_uniform"
LOCAL_KERNEL_MODES = (KERNEL_GAUSSIAN, KERNEL_HARD_RADIUS)
LOCAL_RADII = (0.02, 0.05, 0.10)
LOCAL_EDGE_FEATURE_NAMES = (
    "source_minus_target_delta_eta",
    "sin_source_minus_target_delta_phi",
    "cos_source_minus_target_delta_phi",
    "delta_r",
    "clipped_log_source_target_pt_ratio",
    "clipped_log_source_target_energy_ratio",
)


@dataclass(frozen=True)
class LocalCorrectionConfig:
    architecture_id: str = ARCH_A1_MULTISCALE_LOCAL
    particle_dim: int = RAW_TOKEN_DIM
    field_dim: int = 50
    h0_dim: int = 160
    width: int = 160
    graph_cap: int = 32
    graph_support: float = 0.30
    radii: tuple[float, float, float] = LOCAL_RADII
    kernel_mode: str = KERNEL_GAUSSIAN
    local_layers: int = 2
    capacity_particle_blocks: int = 0
    capacity_hidden_dim: int = 320
    dropout: float = 0.05
    layer_norm_epsilon: float = 1.0e-5
    aggregation_epsilon: float = 1.0e-6
    ratio_epsilon: float = 1.0e-8
    ratio_log_clip: float = 8.0
    trust_bound_enabled: bool = True
    zero_initialize_heads: bool = True

    def __post_init__(self) -> None:
        if self.architecture_id not in STEP6_ARCHITECTURE_IDS:
            raise ValueError(f"unknown Step 6 architecture {self.architecture_id!r}")
        if int(self.particle_dim) != RAW_TOKEN_DIM or int(self.field_dim) != 50:
            raise ValueError("Step 6 requires repository raw width 14 and field width 50")
        if int(self.h0_dim) != 160 or int(self.width) != 160:
            raise ValueError("Step 6 local/global particle interfaces are locked to width 160")
        if int(self.graph_cap) != 32 or float(self.graph_support) != 0.30:
            raise ValueError("Step 6 graph is locked to cap 32 and support 0.30")
        if tuple(float(value) for value in self.radii) != LOCAL_RADII:
            raise ValueError("Step 6 target-kernel radii are locked to 0.02/0.05/0.10")
        if self.architecture_id == ARCH_A1_MULTISCALE_LOCAL:
            expected = (KERNEL_GAUSSIAN, 2, 0)
        elif self.architecture_id == ARCH_A1H_HARD_RADIUS:
            expected = (KERNEL_HARD_RADIUS, 2, 0)
        else:
            expected = (KERNEL_CAPACITY_UNIFORM, 6, 2)
        actual = (
            str(self.kernel_mode),
            int(self.local_layers),
            int(self.capacity_particle_blocks),
        )
        if actual != expected:
            raise ValueError(
                f"{self.architecture_id} requires kernel/layers/blocks {expected}, got {actual}"
            )
        expected_capacity_width = 288 if self.architecture_id == ARCH_A0M_CAPACITY_PARTICLE else 320
        if int(self.capacity_hidden_dim) != expected_capacity_width:
            raise ValueError(
                f"{self.architecture_id} capacity hidden width is locked to {expected_capacity_width}"
            )
        if float(self.dropout) < 0 or float(self.dropout) >= 1:
            raise ValueError("Step 6 dropout must be in [0,1)")
        if float(self.layer_norm_epsilon) != 1.0e-5:
            raise ValueError("Step 6 LayerNorm epsilon is locked to 1e-5")
        if float(self.aggregation_epsilon) != 1.0e-6:
            raise ValueError("Step 6 aggregation epsilon is locked to 1e-6")
        if float(self.ratio_epsilon) != 1.0e-8 or float(self.ratio_log_clip) != 8.0:
            raise ValueError("Step 6 ratio epsilon/clip are locked to 1e-8 and 8")
        if not bool(self.trust_bound_enabled) or not bool(self.zero_initialize_heads):
            raise ValueError("Step 6 architectures require bounded, zero-initialized heads")

    @classmethod
    def for_architecture(
        cls,
        architecture_id: str,
        *,
        dropout: float = 0.05,
    ) -> "LocalCorrectionConfig":
        if architecture_id == ARCH_A1_MULTISCALE_LOCAL:
            return cls(architecture_id=architecture_id, kernel_mode=KERNEL_GAUSSIAN, dropout=dropout)
        if architecture_id == ARCH_A1H_HARD_RADIUS:
            return cls(architecture_id=architecture_id, kernel_mode=KERNEL_HARD_RADIUS, dropout=dropout)
        if architecture_id == ARCH_A0M_CAPACITY_PARTICLE:
            return cls(
                architecture_id=architecture_id,
                kernel_mode=KERNEL_CAPACITY_UNIFORM,
                local_layers=6,
                capacity_particle_blocks=2,
                capacity_hidden_dim=288,
                dropout=dropout,
            )
        raise ValueError(f"unknown Step 6 architecture {architecture_id!r}")

    def to_artifact(self) -> dict[str, Any]:
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_LOCAL_CONFIG_CONTRACT,
                **asdict(self),
                "radii": list(self.radii),
                "edge_feature_names": list(LOCAL_EDGE_FEATURE_NAMES),
                "message_mlp": [326, 160, 160],
                "update_mlp": [320, 160, 160],
                "radius_head": [480, 160, 128, 64, 15],
                "five_channel_pass_through": True,
                "global_or_region_module_present": False,
            }
        )


@dataclass(frozen=True)
class LocalGraphFeatures:
    graph: DirectedNeighborGraph
    edge_features: torch.Tensor
    kernel_weights: torch.Tensor
    kernel_mode: str
    radii: tuple[float, float, float]

    def audit_artifact(self) -> dict[str, Any]:
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_LOCAL_GRAPH_CONTRACT,
                "cap": int(self.graph.cap),
                "support": float(self.graph.support),
                "self_edges_present": False,
                "directed_edge_count": int(self.graph.edge_valid.sum().detach().cpu()),
                "edge_feature_names": list(LOCAL_EDGE_FEATURE_NAMES),
                "edge_feature_shape": list(self.edge_features.shape),
                "kernel_weight_shape": list(self.kernel_weights.shape),
                "kernel_mode": self.kernel_mode,
                "radii": list(self.radii),
                "tie_policy": "ascending_source_particle_index",
                "implicit_reverse_edges_inserted": False,
            }
        )


@dataclass(frozen=True)
class ParticleReasoningState:
    base_hidden: torch.Tensor
    radius_streams: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    readback: torch.Tensor
    particle_mask: torch.Tensor
    region_tokens: torch.Tensor | None
    region_mask: torch.Tensor | None
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class LocalCorrectionOutput(C0CorrectionOutput):
    reasoning_state: ParticleReasoningState


def _gather_neighbors(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = int(values.shape[0])
    batch_index = torch.arange(batch, device=values.device)[:, None, None]
    return values[batch_index, indices]


def build_local_graph_features(
    hlt_tokens: torch.Tensor,
    mask: torch.Tensor,
    *,
    kernel_mode: str,
    graph: DirectedNeighborGraph | None = None,
    ratio_epsilon: float = 1.0e-8,
    ratio_log_clip: float = 8.0,
) -> LocalGraphFeatures:
    """Build the one locked graph, six edge features, and three kernels."""

    tokens = hlt_tokens.float()
    valid = mask.to(device=tokens.device, dtype=torch.bool)
    if tokens.ndim != 3 or tokens.shape[-1] != RAW_TOKEN_DIM or valid.shape != tokens.shape[:2]:
        raise ValueError("Step 6 graph tokens/mask do not align")
    if kernel_mode not in (KERNEL_GAUSSIAN, KERNEL_HARD_RADIUS, KERNEL_CAPACITY_UNIFORM):
        raise ValueError("unknown Step 6 local kernel mode")
    graph = graph or build_directed_neighbor_graph(tokens, valid, cap=32, support=0.30)
    source = _gather_neighbors(tokens, graph.neighbor_indices)
    target = tokens[:, :, None, :].expand_as(source)
    delta_eta = source[..., 1] - target[..., 1]
    delta_phi = torch.remainder(source[..., 2] - target[..., 2] + math.pi, 2 * math.pi) - math.pi
    epsilon = float(ratio_epsilon)
    clip = float(ratio_log_clip)
    log_pt_ratio = torch.log(torch.clamp(source[..., 0], min=epsilon)) - torch.log(
        torch.clamp(target[..., 0], min=epsilon)
    )
    log_energy_ratio = torch.log(torch.clamp(source[..., 3], min=epsilon)) - torch.log(
        torch.clamp(target[..., 3], min=epsilon)
    )
    features = torch.stack(
        (
            delta_eta,
            torch.sin(delta_phi),
            torch.cos(delta_phi),
            graph.delta_r,
            torch.clamp(log_pt_ratio, min=-clip, max=clip),
            torch.clamp(log_energy_ratio, min=-clip, max=clip),
        ),
        dim=-1,
    )
    edge_valid = graph.edge_valid.to(device=tokens.device, dtype=torch.bool)
    features = features.masked_fill(~edge_valid.unsqueeze(-1), 0.0)
    if kernel_mode == KERNEL_GAUSSIAN:
        weights = torch.stack(
            [torch.exp(-0.5 * graph.delta_r.square() / radius**2) for radius in LOCAL_RADII],
            dim=-1,
        )
    elif kernel_mode == KERNEL_HARD_RADIUS:
        weights = torch.stack(
            [(graph.delta_r <= radius).to(dtype=tokens.dtype) for radius in LOCAL_RADII],
            dim=-1,
        )
    else:
        weights = torch.ones((*graph.delta_r.shape, 1), device=tokens.device, dtype=tokens.dtype)
    weights = weights * edge_valid.unsqueeze(-1).to(dtype=weights.dtype)
    if not torch.isfinite(features).all() or not torch.isfinite(weights).all():
        raise ValueError("Step 6 graph produced non-finite features/weights")
    return LocalGraphFeatures(
        graph=graph,
        edge_features=features,
        kernel_weights=weights,
        kernel_mode=kernel_mode,
        radii=LOCAL_RADII,
    )


class PredictionAnchoredBaseFusion(torch.nn.Module):
    """Locked raw/f0/h0/jet fusion shared by local and future HLG models."""

    def __init__(self, *, dropout: float, layer_norm_epsilon: float) -> None:
        super().__init__()
        epsilon = float(layer_norm_epsilon)
        self.raw_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(RAW_TOKEN_DIM, eps=epsilon),
            torch.nn.Linear(RAW_TOKEN_DIM, 64),
            torch.nn.GELU(),
        )
        self.f0_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(50, eps=epsilon),
            torch.nn.Linear(50, 64),
            torch.nn.GELU(),
        )
        self.h0_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(160, eps=epsilon),
            torch.nn.Linear(160, 96),
            torch.nn.GELU(),
        )
        self.jet_projection = torch.nn.Sequential(
            torch.nn.Linear(11, 32),
            torch.nn.GELU(),
        )
        self.fusion = torch.nn.Sequential(
            torch.nn.LayerNorm(256, eps=epsilon),
            torch.nn.Linear(256, 160),
            torch.nn.GELU(),
            torch.nn.LayerNorm(160, eps=epsilon),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        standardized_f0: torch.Tensor,
        h0: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.raw_projection(tokens)
        field = self.f0_projection(standardized_f0)
        state = self.h0_projection(h0)
        jet = self.jet_projection(_masked_jet_summary(tokens, mask))
        jet = jet[:, None, :].expand(-1, tokens.shape[1], -1)
        hidden = self.fusion(torch.cat((raw, field, state, jet), dim=-1))
        return hidden.masked_fill(~mask.unsqueeze(-1), 0.0)


class SharedLocalMessageLayer(torch.nn.Module):
    """One M_l/U_l pair; one instance is reused for all three radii."""

    def __init__(self, config: LocalCorrectionConfig) -> None:
        super().__init__()
        width = int(config.width)
        epsilon = float(config.layer_norm_epsilon)
        self.aggregation_epsilon = float(config.aggregation_epsilon)
        self.pre_norm = torch.nn.LayerNorm(width, eps=epsilon)
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * width + len(LOCAL_EDGE_FEATURE_NAMES), width),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(width, width),
        )
        self.update_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * width, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, width),
        )
        self.residual_dropout = torch.nn.Dropout(float(config.dropout))

    def aggregate_messages(
        self,
        hidden: torch.Tensor,
        edge_features: torch.Tensor,
        neighbor_indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.pre_norm(hidden)
        source = _gather_neighbors(normalized, neighbor_indices)
        target = normalized[:, :, None, :].expand_as(source)
        messages = self.message_mlp(torch.cat((target, source, edge_features), dim=-1))
        numerator = (weights.unsqueeze(-1) * messages).sum(dim=2)
        denominator = weights.sum(dim=2, keepdim=True) + self.aggregation_epsilon
        return numerator / denominator, normalized

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        edge_features: torch.Tensor,
        neighbor_indices: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        aggregate, normalized = self.aggregate_messages(
            hidden, edge_features, neighbor_indices, weights
        )
        update = self.update_mlp(torch.cat((normalized, aggregate), dim=-1))
        output = hidden + self.residual_dropout(update)
        return output.masked_fill(~mask.unsqueeze(-1), 0.0)


class TargetKernelLocalProcessor(torch.nn.Module):
    def __init__(self, config: LocalCorrectionConfig) -> None:
        super().__init__()
        if config.kernel_mode not in LOCAL_KERNEL_MODES:
            raise ValueError("target-kernel processor requires Gaussian or hard-radius mode")
        self.config = config
        self.layers = torch.nn.ModuleList(
            [SharedLocalMessageLayer(config) for _ in range(int(config.local_layers))]
        )

    def forward(
        self,
        base_hidden: torch.Tensor,
        graph_features: LocalGraphFeatures,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        streams = [base_hidden, base_hidden, base_hidden]
        for layer in self.layers:
            streams = [
                layer(
                    stream,
                    edge_features=graph_features.edge_features,
                    neighbor_indices=graph_features.graph.neighbor_indices,
                    weights=graph_features.kernel_weights[..., radius_index],
                    mask=mask,
                )
                for radius_index, stream in enumerate(streams)
            ]
        return (streams[0], streams[1], streams[2])


class _WideParticleCapacityBlock(torch.nn.Module):
    def __init__(self, config: LocalCorrectionConfig) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(
            int(config.width), eps=float(config.layer_norm_epsilon)
        )
        self.network = torch.nn.Sequential(
            torch.nn.Linear(int(config.width), int(config.capacity_hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(int(config.capacity_hidden_dim), int(config.width)),
            torch.nn.Dropout(float(config.dropout)),
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output = hidden + self.network(self.norm(hidden))
        return output.masked_fill(~mask.unsqueeze(-1), 0.0)


class ParticleCapacityProcessor(torch.nn.Module):
    """Six-layer single-stream particle control with no region/global tokens."""

    def __init__(self, config: LocalCorrectionConfig) -> None:
        super().__init__()
        if config.architecture_id != ARCH_A0M_CAPACITY_PARTICLE:
            raise ValueError("particle capacity processor requires the A0M config")
        self.config = config
        self.layers = torch.nn.ModuleList(
            [SharedLocalMessageLayer(config) for _ in range(int(config.local_layers))]
        )
        self.capacity_blocks = torch.nn.ModuleList(
            [_WideParticleCapacityBlock(config) for _ in range(int(config.capacity_particle_blocks))]
        )

    def forward(
        self,
        base_hidden: torch.Tensor,
        graph_features: LocalGraphFeatures,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = base_hidden
        weights = graph_features.kernel_weights[..., 0]
        for layer in self.layers:
            hidden = layer(
                hidden,
                edge_features=graph_features.edge_features,
                neighbor_indices=graph_features.graph.neighbor_indices,
                weights=weights,
                mask=mask,
            )
        for block in self.capacity_blocks:
            hidden = block(hidden, mask)
        return hidden


class RadiusCorrectionHead(torch.nn.Module):
    """Locked canonical 480->160->128->64->15 radius head."""

    def __init__(self, config: LocalCorrectionConfig) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(480, eps=float(config.layer_norm_epsilon)),
            torch.nn.Linear(480, 160),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(160, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(128, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(64, 15),
        )
        if bool(config.zero_initialize_heads):
            final = self.network[-1]
            assert isinstance(final, torch.nn.Linear)
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.zeros_(final.bias)

    def forward(
        self,
        base_hidden: torch.Tensor,
        radius_hidden: torch.Tensor,
        readback: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(torch.cat((base_hidden, radius_hidden, readback), dim=-1))


class PredictionAnchoredLocalCorrection(torch.nn.Module):
    """A1/A1H/A0M correction model with the future HLG particle interface."""

    def __init__(
        self,
        scaler_artifact: Mapping[str, Any],
        config: LocalCorrectionConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.scalers = TorchBridgeScalers(scaler_artifact)
        self.base_fusion = PredictionAnchoredBaseFusion(
            dropout=float(config.dropout),
            layer_norm_epsilon=float(config.layer_norm_epsilon),
        )
        if config.architecture_id == ARCH_A0M_CAPACITY_PARTICLE:
            self.local_processor: torch.nn.Module = ParticleCapacityProcessor(config)
        else:
            self.local_processor = TargetKernelLocalProcessor(config)
        self.radius_heads = torch.nn.ModuleList([RadiusCorrectionHead(config) for _ in LOCAL_RADII])
        if any("region" in name or "global" in name for name, _ in self.named_modules()):
            raise AssertionError("Step 6 model unexpectedly instantiated hierarchy/global modules")
        if any("gate" in name.lower() for name, _ in self.named_parameters()):
            raise AssertionError("Step 6 bounded primary architectures have no learned gate")

    @property
    def scaler_sha256(self) -> str:
        return self.scalers.artifact_sha256

    def config_artifact(self) -> dict[str, Any]:
        config = self.config.to_artifact()
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_PARTICLE_INTERFACE_CONTRACT,
                "architecture_id": self.config.architecture_id,
                "config_sha256": config["content_hash"],
                "config": config,
                "scaler_sha256": self.scaler_sha256,
                "input_spaces": {
                    "consumer": NUMERICAL_SPACE_PHYSICAL,
                    "conditioning": NUMERICAL_SPACE_CONDITIONING,
                    "correction_head": NUMERICAL_SPACE_CORRECTION,
                    "loss": NUMERICAL_SPACE_LOSS,
                },
                "particle_interface": {
                    "base_width": 160,
                    "radius_stream_count": 3,
                    "radius_stream_width": 160,
                    "readback_width": 160,
                    "region_tokens": None,
                    "region_mask": None,
                    "no_global_readback_policy": "identity_base_hidden",
                },
                "physical_correction_channels": list(range(45)),
                "pass_through_channels": list(range(45, 50)),
                "learned_gate_present": False,
            }
        )

    def _validate_inputs(
        self,
        hlt_tokens: torch.Tensor,
        mask: torch.Tensor,
        f0_live: torch.Tensor,
        h0: torch.Tensor,
        f0_space: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = hlt_tokens.float()
        valid = mask.to(device=tokens.device, dtype=torch.bool)
        anchor = f0_live.to(device=tokens.device, dtype=torch.float32).detach()
        frozen_h0 = h0.to(device=tokens.device, dtype=torch.float32).detach()
        if str(f0_space) != NUMERICAL_SPACE_PHYSICAL:
            raise ValueError("Step 6 conditioning requires physical-field f0")
        if tokens.ndim != 3 or tokens.shape[-1] != RAW_TOKEN_DIM:
            raise ValueError("Step 6 raw HLT tokens have the wrong shape")
        if valid.shape != tokens.shape[:2] or anchor.shape != (*valid.shape, 50):
            raise ValueError("Step 6 token/mask/f0 shapes do not align")
        if frozen_h0.shape != (*valid.shape, 160):
            raise ValueError("Step 6 h0 must have shape [B,P,160]")
        if not (
            torch.isfinite(tokens).all()
            and torch.isfinite(anchor).all()
            and torch.isfinite(frozen_h0).all()
        ):
            raise ValueError("Step 6 inputs contain non-finite values")
        padding = ~valid
        if bool(padding.any()) and (
            bool(torch.count_nonzero(tokens[padding]))
            or bool(torch.count_nonzero(anchor[padding]))
            or bool(torch.count_nonzero(frozen_h0[padding]))
        ):
            raise ValueError("Step 6 padded raw/f0/h0 inputs must be exactly zero")
        if bool(valid.any()) and (
            bool((tokens[..., 0][valid] <= 0).any())
            or bool((tokens[..., 3][valid] <= 0).any())
        ):
            raise ValueError("valid HLT particles require positive pT and energy")
        return tokens, valid, anchor, frozen_h0

    def forward(
        self,
        hlt_tokens: torch.Tensor,
        mask: torch.Tensor,
        f0_live: torch.Tensor,
        h0: torch.Tensor,
        *,
        f0_space: str = NUMERICAL_SPACE_PHYSICAL,
    ) -> LocalCorrectionOutput:
        tokens, valid, anchor, frozen_h0 = self._validate_inputs(
            hlt_tokens, mask, f0_live, h0, f0_space
        )
        standardized_f0 = self.scalers.conditioning_standardize(
            anchor, valid, input_space=NUMERICAL_SPACE_PHYSICAL
        )
        base = self.base_fusion(tokens, valid, standardized_f0, frozen_h0)
        graph_features = build_local_graph_features(
            tokens,
            valid,
            kernel_mode=str(self.config.kernel_mode),
            ratio_epsilon=float(self.config.ratio_epsilon),
            ratio_log_clip=float(self.config.ratio_log_clip),
        )
        if self.config.architecture_id == ARCH_A0M_CAPACITY_PARTICLE:
            capacity = self.local_processor(base, graph_features, valid)
            streams = (capacity, capacity, capacity)
        else:
            streams = self.local_processor(base, graph_features, valid)
        readback = base
        standardized_raw = torch.cat(
            [
                head(base, streams[index], readback)
                for index, head in enumerate(self.radius_heads)
            ],
            dim=-1,
        )
        standardized_raw = standardized_raw.masked_fill(~valid.unsqueeze(-1), 0.0)
        correction, saturation = self.scalers.physical_correction(
            standardized_raw,
            valid,
            trust_bound_enabled=True,
        )
        f_hat = torch.cat((anchor[..., :PHYSICAL_CHANNELS] + correction, anchor[..., 45:]), dim=-1)
        f_hat = f_hat.masked_fill(~valid.unsqueeze(-1), 0.0)
        reasoning = ParticleReasoningState(
            base_hidden=base,
            radius_streams=streams,
            readback=readback,
            particle_mask=valid,
            region_tokens=None,
            region_mask=None,
            diagnostics={
                "contract": PREDICTION_ANCHORED_PARTICLE_INTERFACE_CONTRACT,
                "architecture_id": self.config.architecture_id,
                "kernel_mode": self.config.kernel_mode,
                "radii": list(LOCAL_RADII),
                "radius_streams_preserved": True,
                "readback_source": "identity_base_hidden_no_global",
                "region_or_global_path_present": False,
            },
        )
        valid_physical = int(valid.sum().detach().cpu().item()) * PHYSICAL_CHANNELS
        diagnostics = {
            **reasoning.diagnostics,
            "scaler_sha256": self.scaler_sha256,
            "consumer_output_space": NUMERICAL_SPACE_PHYSICAL,
            "conditioning_space": NUMERICAL_SPACE_CONDITIONING,
            "correction_space": NUMERICAL_SPACE_CORRECTION,
            "loss_space": NUMERICAL_SPACE_LOSS,
            "trust_bound_enabled": True,
            "saturation_fraction": float(saturation.sum().detach().cpu().item())
            / max(valid_physical, 1),
            "saturation_definition": "abs_delta_gte_0.99_times_component_trust_scale",
            "reliability_channels_exact_pass_through": bool(
                torch.equal(f_hat[..., 45:], anchor[..., 45:])
            ),
            "learned_gate_present": False,
            "f0_and_h0_stop_gradient": True,
            "graph_cap": int(graph_features.graph.cap),
            "graph_support": float(graph_features.graph.support),
            "directed_edge_count": int(graph_features.graph.edge_valid.sum().detach().cpu()),
        }
        return LocalCorrectionOutput(
            f_hat=f_hat,
            physical_correction=correction,
            standardized_raw_correction=standardized_raw,
            hidden=base,
            mask=valid,
            saturation_mask=saturation,
            diagnostics=diagnostics,
            reasoning_state=reasoning,
        )


def build_step6_correction_model(
    architecture_id: str,
    *,
    scaler_artifact: Mapping[str, Any],
    dropout: float = 0.05,
) -> PredictionAnchoredLocalCorrection:
    return PredictionAnchoredLocalCorrection(
        scaler_artifact,
        LocalCorrectionConfig.for_architecture(architecture_id, dropout=dropout),
    )


@dataclass(frozen=True)
class CorrectionResourceProfile:
    architecture_id: str
    trainable_parameters: int
    total_parameters: int
    forward_flops: int
    batch_size: int
    particle_width: int
    valid_particles: int
    method: str

    def to_artifact(self) -> dict[str, Any]:
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_RESOURCE_MEASUREMENT_CONTRACT,
                **asdict(self),
                "flop_convention": (
                    "linear_multiply_add_is_two_plus_bias;layernorm_five;gelu_eight;"
                    "explicit_pair_kernel_attention_terms"
                ),
                "correction_path_only": True,
            }
        )


def _parameter_counts(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(int(parameter.numel()) for parameter in module.parameters())
    trainable = sum(
        int(parameter.numel()) for parameter in module.parameters() if parameter.requires_grad
    )
    return trainable, total


def _profile_inputs(
    *,
    particle_width: int,
    valid_particles: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(particle_width) <= 0 or not 0 < int(valid_particles) <= int(particle_width):
        raise ValueError("resource profile requires 0 < valid_particles <= particle_width")
    tokens = torch.zeros((1, int(particle_width), RAW_TOKEN_DIM), device=device)
    mask = torch.zeros((1, int(particle_width)), dtype=torch.bool, device=device)
    mask[:, : int(valid_particles)] = True
    positions = torch.arange(int(valid_particles), device=device, dtype=torch.float32)
    tokens[0, : int(valid_particles), 0] = 1.0 + positions / max(int(valid_particles), 1)
    tokens[0, : int(valid_particles), 1] = 0.002 * positions
    tokens[0, : int(valid_particles), 2] = 0.003 * positions
    tokens[0, : int(valid_particles), 3] = 2.0 + positions / max(int(valid_particles), 1)
    f0 = torch.zeros((1, int(particle_width), 50), device=device)
    h0 = torch.zeros((1, int(particle_width), 160), device=device)
    return tokens, mask, f0, h0


def measure_correction_resources(
    model: PredictionAnchoredLocalCorrection,
    *,
    particle_width: int,
    valid_particles: int | None = None,
) -> CorrectionResourceProfile:
    """Execute one batch-1 forward and count locked primitive operations."""

    valid_count = int(particle_width if valid_particles is None else valid_particles)
    device = next(model.parameters()).device
    inputs = _profile_inputs(
        particle_width=int(particle_width),
        valid_particles=valid_count,
        device=device,
    )
    flops = 0

    def linear_hook(module: torch.nn.Linear, args: tuple[Any, ...], output: Any) -> None:
        nonlocal flops
        tensor = output if isinstance(output, torch.Tensor) else output[0]
        flops += int(tensor.numel()) * (
            2 * int(module.in_features) + (1 if module.bias is not None else 0)
        )

    def layer_norm_hook(module: torch.nn.LayerNorm, args: tuple[Any, ...], output: Any) -> None:
        nonlocal flops
        flops += 5 * int(output.numel())

    def gelu_hook(module: torch.nn.GELU, args: tuple[Any, ...], output: Any) -> None:
        nonlocal flops
        flops += 8 * int(output.numel())

    hooks = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, torch.nn.LayerNorm):
            hooks.append(module.register_forward_hook(layer_norm_hook))
        elif isinstance(module, torch.nn.GELU):
            hooks.append(module.register_forward_hook(gelu_hook))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(*inputs)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    edge_count = int(output.diagnostics["directed_edge_count"])
    stream_passes = int(model.config.local_layers) * (
        3 if model.config.kernel_mode in LOCAL_KERNEL_MODES else 1
    )
    # Six edge features, kernel construction, weighted sums/division, residuals,
    # correction inverse-scaling/trust, and physical addition.
    flops += edge_count * (28 + (18 if model.config.kernel_mode == KERNEL_GAUSSIAN else 3))
    flops += stream_passes * edge_count * int(model.config.width) * 3
    flops += stream_passes * valid_count * int(model.config.width) * 2
    flops += valid_count * PHYSICAL_CHANNELS * 8
    trainable, total = _parameter_counts(model)
    return CorrectionResourceProfile(
        architecture_id=model.config.architecture_id,
        trainable_parameters=trainable,
        total_parameters=total,
        forward_flops=int(flops),
        batch_size=1,
        particle_width=int(particle_width),
        valid_particles=valid_count,
        method="executed_forward_hooks_plus_explicit_local_operations_v1",
    )


def canonical_a3_resource_reference(
    *,
    scaler_artifact: Mapping[str, Any],
    particle_width: int,
    valid_particles: int | None = None,
) -> CorrectionResourceProfile:
    """Analytical locked-profile reference; this does not instantiate an A3 model."""

    from teacher_logit_reco.multiscale_subjet_part.assignment import (
        SoftSubjetAssignment,
        SoftSubjetAssignmentConfig,
    )
    from teacher_logit_reco.multiscale_subjet_part.features import (
        multiscale_subjet_scale_specs_for_profile,
    )
    from teacher_logit_reco.multiscale_subjet_part.seeds import SubjetSeedBuilderConfig
    from teacher_logit_reco.multiscale_subjet_part.subjet_transformer import (
        MultiScaleSubjetPairFeatureConfig,
        MultiScaleSubjetTransformer,
        MultiScaleSubjetTransformerConfig,
    )

    valid_count = int(particle_width if valid_particles is None else valid_particles)
    local_model = build_step6_correction_model(
        ARCH_A1_MULTISCALE_LOCAL,
        scaler_artifact=scaler_artifact,
        dropout=0.05,
    )
    local = measure_correction_resources(
        local_model,
        particle_width=int(particle_width),
        valid_particles=valid_count,
    )
    specs = multiscale_subjet_scale_specs_for_profile("few_subjets")
    seed_config = SubjetSeedBuilderConfig(
        scale_specs=specs,
        method_by_scale={
            "small": "leading_pt",
            "medium": "local_density",
            "large": "farthest_point",
        },
        density_pt_weight=1.0,
        include_self_in_density=False,
        eps=1.0e-8,
    )
    assignment = SoftSubjetAssignment(
        SoftSubjetAssignmentConfig(
            scale_specs=specs,
            query_mode="seeded",
            embed_dim=64,
            hidden_dim=128,
            temperature=0.50,
            geometry_bias_strength=2.0,
            use_scale_embedding=True,
            radius_floor=0.03,
            dead_token_weight_threshold=1.0e-3,
            seed_config=seed_config,
        )
    )
    transformer = MultiScaleSubjetTransformer(
        MultiScaleSubjetTransformerConfig(
            token_dim=160,
            num_layers=2,
            num_heads=4,
            ffn_dim=320,
            dropout=0.05,
            attention_dropout=0.05,
            use_pairwise_bias=True,
            use_scale_pair_embedding=True,
            pair_bias_hidden_dim=64,
            num_scales=3,
            mask_value=-1.0e4,
            pair_feature_config=MultiScaleSubjetPairFeatureConfig(
                num_scales=3,
                eps=1.0e-6,
                delta_r_scale=5.0,
                radius_scale=0.5,
                max_log_value=14.0,
                max_log_ratio=8.0,
            ),
        )
    )
    assignment_params, _ = _parameter_counts(assignment)
    transformer_params, _ = _parameter_counts(transformer)
    # 389->320->160, scale embedding, output LayerNorm.
    region_projection_params = (
        2 * 389 + 389 * 320 + 320 + 320 * 160 + 160 + 3 * 160 + 2 * 160
    )
    # Query pre-norm, MHA(160,4), 160->160 output fusion, final LayerNorm.
    readback_params = 2 * 160 + (3 * 160 * 160 + 3 * 160) + (
        160 * 160 + 160
    ) + (160 * 160 + 160) + 2 * 160
    parameter_count = (
        int(local.trainable_parameters)
        + int(assignment_params)
        + int(transformer_params)
        + int(region_projection_params)
        + int(readback_params)
    )

    particles = valid_count
    regions = 10
    dim = 160
    # Locked learned assignment projections and particle/seed similarity.
    assignment_flops = (
        particles * (2 * assignment.particle_input_dim * 128 + 128 + 8 * 128)
        + particles * (2 * 128 * 64 + 64)
        + regions * (2 * assignment.seed_input_dim * 128 + 128 + 8 * 128)
        + regions * (2 * 128 * 64 + 64)
        + 2 * particles * regions * 64
    )
    region_flops = regions * (
        5 * 389
        + 2 * 389 * 320
        + 320
        + 8 * 320
        + 2 * 320 * 160
        + 160
        + 5 * 160
    )
    pair_bias_flops = regions * regions * (
        2 * 19 * 64 + 64 + 8 * 64 + 2 * 64 * 4 + 4
    )
    transformer_flops = 0
    for _ in range(2):
        transformer_flops += regions * (2 * dim * 3 * dim + 3 * dim)
        transformer_flops += regions * (2 * dim * dim + dim)
        transformer_flops += 4 * regions * regions * dim
        transformer_flops += regions * (2 * dim * 320 + 320 + 8 * 320)
        transformer_flops += regions * (2 * 320 * dim + dim)
        transformer_flops += regions * 10 * dim
    readback_flops = (
        particles * (2 * dim * dim + dim)
        + 2 * regions * (2 * dim * dim + dim)
        + particles * (2 * dim * dim + dim)
        + 4 * particles * regions * dim
        + particles * (2 * dim * dim + dim)
        + particles * 10 * dim
    )
    return CorrectionResourceProfile(
        architecture_id="D10_A3_hlg_primary_locked_profile_reference",
        trainable_parameters=int(parameter_count),
        total_parameters=int(parameter_count),
        forward_flops=int(
            local.forward_flops
            + assignment_flops
            + region_flops
            + pair_bias_flops
            + transformer_flops
            + readback_flops
        ),
        batch_size=1,
        particle_width=int(particle_width),
        valid_particles=valid_count,
        method="locked_section_12_7_analytical_components_v1_no_a3_checkpoint",
    )


def particle_capacity_match(
    control: CorrectionResourceProfile,
    reference: CorrectionResourceProfile,
) -> dict[str, Any]:
    parameter_ratio = control.trainable_parameters / reference.trainable_parameters
    flop_ratio = control.forward_flops / reference.forward_flops
    parameter_relative_error = abs(parameter_ratio - 1.0)
    flop_relative_error = abs(flop_ratio - 1.0)
    return with_content_hash(
        {
            "contract": "prediction_anchored_particle_capacity_match_v1",
            "control": control.to_artifact(),
            "reference": reference.to_artifact(),
            "parameter_ratio": parameter_ratio,
            "flop_ratio": flop_ratio,
            "parameter_relative_error": parameter_relative_error,
            "flop_relative_error": flop_relative_error,
            "parameter_tolerance": 0.05,
            "flop_tolerance": 0.10,
            "parameter_tolerance_passed": parameter_relative_error <= 0.05,
            "flop_tolerance_passed": flop_relative_error <= 0.10,
            "passed": parameter_relative_error <= 0.05 and flop_relative_error <= 0.10,
            "control_not_resized_after_selection": True,
        }
    )


STEP7_DEFERRED_ARCHITECTURE_IDS = (
    "D10_A2_regions_no_global",
    "D10_A3_hlg_primary",
    "D10_A4_hlg_refine",
    "D10_A5_hlg_absolute_conditioned",
    "D10_A5S_hlg_scratch_physical45",
    "D10_A6_hlg_no_pair_bias",
    "D10_A7_hlg_no_h0",
    "D10_A7F_hlg_no_f0",
    "D10_A7X_hlg_no_raw_skip",
    "D10_A8_hlg_fused_radius_heads",
    "D10_A9_hlg_group_gate",
    "D10_AS_hlg_regions_2_2_1",
    "D10_AL_hlg_regions_8_8_4",
    "D10_AFIX_hlg_fixed_assignment",
    "D10_ASAME_hlg_same_scale_only",
    "D10_AGLOBAL_hlg_one_global_token",
    "A0_CAP500_direct_hlt",
    "A0_CAP500_r0rep_direct",
    "D10_XA3_bridge_only",
    "D10_XA3_ce_only",
    "D10_XA3_kd_only",
    "D10_XA3_kd_bridge",
    "D10_XA3_kd_ce",
    "D10_XA3_full_no_warmup",
    "D10_XA3_full_no_smooth",
    "D10_B1_all50_fullhead",
    "D10_B2_all50_physical45_only",
    "D10_TALT_A0",
    "D10_TALT_A3",
    "D10_N0_shuffled_logit_kd",
    "D10_N1_shuffled_bridge_field",
    "D10_N2_shuffled_primary",
    "D10_N3_nonprivileged_teacher_kd",
)


def _torch_bytes(value: Any) -> bytes:
    buffer = BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def measure_step6_registry_states(
    registry: Mapping[str, Any],
    *,
    scaler_artifact: Mapping[str, Any],
    particle_width: int,
    source_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure C0 plus the three implemented non-hierarchical architectures."""

    from .bridge_campaign import (
        MEASUREMENT_UNMEASURED,
        record_registry_measurements,
        validate_campaign_registry,
    )
    from .bridge_reconstructor_train import measure_c0_registry_states

    validate_campaign_registry(registry)
    source_hash = str(source_manifest_sha256)
    if len(source_hash) != 64 or any(character not in "0123456789abcdef" for character in source_hash):
        raise ValueError("Step 6 measurement requires the source-manifest SHA-256")
    by_id = {row["canonical_run_id"]: row for row in registry["runs"]}
    changed = [
        run_id
        for run_id in STEP7_DEFERRED_ARCHITECTURE_IDS
        if by_id[run_id]["measurement_status"] != MEASUREMENT_UNMEASURED
    ]
    if changed:
        raise ValueError(
            f"Step 6 cannot measure or inherit Step 7 architecture states: {changed}"
        )
    c0_updated, c0_measurement = measure_c0_registry_states(
        registry,
        scaler_artifact=scaler_artifact,
        model_width=160,
    )
    state_bytes: dict[str, int] = {}
    model_contracts: dict[str, str] = {}
    resources: dict[str, Any] = {}
    for architecture_id in STEP6_ARCHITECTURE_IDS:
        model = build_step6_correction_model(
            architecture_id,
            scaler_artifact=scaler_artifact,
            dropout=0.05,
        )
        payload = {
            "checkpoint_contract": "prediction_anchored_step6_weights_v1",
            "architecture_id": architecture_id,
            "model_config": model.config_artifact(),
            "model_state_dict": model.state_dict(),
            "scaler_sha256": model.scaler_sha256,
            "weights_only": True,
            "optimizer_state_persisted": False,
            "generated_fields_persisted": False,
        }
        state_bytes[architecture_id] = len(_torch_bytes(payload))
        model_contracts[architecture_id] = model.config_artifact()["content_hash"]
        resources[architecture_id] = measure_correction_resources(
            model,
            particle_width=int(particle_width),
            valid_particles=int(particle_width),
        ).to_artifact()
    reference = canonical_a3_resource_reference(
        scaler_artifact=scaler_artifact,
        particle_width=int(particle_width),
        valid_particles=int(particle_width),
    )
    control_profile = CorrectionResourceProfile(
        **{
            key: resources[ARCH_A0M_CAPACITY_PARTICLE][key]
            for key in (
                "architecture_id",
                "trainable_parameters",
                "total_parameters",
                "forward_flops",
                "batch_size",
                "particle_width",
                "valid_particles",
                "method",
            )
        }
    )
    capacity = particle_capacity_match(control_profile, reference)
    if not bool(capacity["passed"]):
        raise AssertionError("A0M no longer matches the locked A3 resource profile")
    updated = record_registry_measurements(c0_updated, state_bytes)
    updated_by_id = {row["canonical_run_id"]: row for row in updated["runs"]}
    if any(
        updated_by_id[run_id]["measurement_status"] != MEASUREMENT_UNMEASURED
        for run_id in STEP7_DEFERRED_ARCHITECTURE_IDS
    ):
        raise AssertionError("Step 6 changed a deferred HLG/direct registry row")
    all_measured = {
        **dict(c0_measurement["measured_state_bytes"]),
        **state_bytes,
    }
    artifact = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_STEP6_MEASUREMENT_CONTRACT,
            "input_registry_sha256": registry["content_hash"],
            "c0_measurement_sha256": c0_measurement["content_hash"],
            "updated_registry_sha256": updated["content_hash"],
            "source_manifest_sha256": source_hash,
            "source_manifest_particle_width": int(particle_width),
            "measured_state_bytes": all_measured,
            "new_step6_state_bytes": state_bytes,
            "model_contract_sha256": model_contracts,
            "resource_profiles": resources,
            "canonical_a3_reference": reference.to_artifact(),
            "particle_capacity_match": capacity,
            "implemented_configuration_count": len(all_measured),
            "step7_deferred_unmeasured_run_ids": list(STEP7_DEFERRED_ARCHITECTURE_IDS),
            "serialization_method": "torch_save_weights_only_to_verified_ram",
        }
    )
    return updated, artifact


def tiny_train_reload_step6_model(
    architecture_id: str,
    *,
    scaler_artifact: Mapping[str, Any],
    batch: Mapping[str, Any],
    learning_rate: float = 1.0e-3,
) -> dict[str, Any]:
    """One CPU bridge-regression step followed by a strict state reload."""

    required = {"hlt_tokens", "mask", "f0", "h0", "bridge_fields"}
    missing = sorted(required - set(batch))
    if missing:
        raise ValueError(f"Step 6 tiny train/reload batch is missing {missing}")
    tensors = {
        "hlt_tokens": torch.as_tensor(batch["hlt_tokens"], dtype=torch.float32),
        "mask": torch.as_tensor(batch["mask"], dtype=torch.bool),
        "f0": torch.as_tensor(batch["f0"], dtype=torch.float32),
        "h0": torch.as_tensor(batch["h0"], dtype=torch.float32),
        "bridge_fields": torch.as_tensor(batch["bridge_fields"], dtype=torch.float32),
    }
    torch.manual_seed(96_000 + STEP6_ARCHITECTURE_IDS.index(architecture_id))
    model = build_step6_correction_model(
        architecture_id,
        scaler_artifact=scaler_artifact,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    model.train()
    output = model(tensors["hlt_tokens"], tensors["mask"], tensors["f0"], tensors["h0"])
    loss, objective = compute_c0_objective(
        output,
        tensors,
        resolve_c0_loss_recipe("D10_L8_full_c0"),
        model.scalers,
        phase="field_warmup",
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("Step 6 tiny train loss is non-finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or any(not torch.isfinite(value).all() for value in gradients):
        raise FloatingPointError("Step 6 tiny train gradients are invalid")
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = model(
            tensors["hlt_tokens"], tensors["mask"], tensors["f0"], tensors["h0"]
        ).f_hat
    encoded = _torch_bytes(
        {
            "architecture_id": architecture_id,
            "config": model.config.to_artifact(),
            "state_dict": model.state_dict(),
        }
    )
    payload = torch.load(BytesIO(encoded), map_location="cpu", weights_only=False)
    reloaded = build_step6_correction_model(
        architecture_id,
        scaler_artifact=scaler_artifact,
        dropout=0.0,
    )
    reloaded.load_state_dict(payload["state_dict"], strict=True)
    reloaded.eval()
    with torch.no_grad():
        actual = reloaded(
            tensors["hlt_tokens"], tensors["mask"], tensors["f0"], tensors["h0"]
        ).f_hat
    if not torch.equal(expected, actual):
        difference = float((expected - actual).abs().max().item())
        raise AssertionError(f"Step 6 strict reload changed f_hat; max_abs={difference}")
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_STEP6_RELOAD_CONTRACT,
            "architecture_id": architecture_id,
            "loss": float(loss.detach().cpu().item()),
            "loss_coefficients": objective["coefficients"],
            "gradient_tensor_count": len(gradients),
            "checkpoint_sha256": hashlib.sha256(encoded).hexdigest(),
            "serialized_state_bytes": len(encoded),
            "strict_reload": True,
            "reload_exact_f_hat": True,
            "optimizer_state_persisted": False,
            "generated_fields_persisted": False,
            "scientific_results_allowed": False,
        }
    )


__all__ = [
    "PREDICTION_ANCHORED_LOCAL_CONFIG_CONTRACT",
    "PREDICTION_ANCHORED_LOCAL_GRAPH_CONTRACT",
    "PREDICTION_ANCHORED_PARTICLE_INTERFACE_CONTRACT",
    "PREDICTION_ANCHORED_RESOURCE_MEASUREMENT_CONTRACT",
    "PREDICTION_ANCHORED_STEP6_MEASUREMENT_CONTRACT",
    "PREDICTION_ANCHORED_STEP6_RELOAD_CONTRACT",
    "ARCH_A0M_CAPACITY_PARTICLE",
    "ARCH_A1_MULTISCALE_LOCAL",
    "ARCH_A1H_HARD_RADIUS",
    "STEP6_ARCHITECTURE_IDS",
    "STEP7_DEFERRED_ARCHITECTURE_IDS",
    "KERNEL_GAUSSIAN",
    "KERNEL_HARD_RADIUS",
    "KERNEL_CAPACITY_UNIFORM",
    "LOCAL_KERNEL_MODES",
    "LOCAL_RADII",
    "LOCAL_EDGE_FEATURE_NAMES",
    "LocalCorrectionConfig",
    "LocalGraphFeatures",
    "ParticleReasoningState",
    "LocalCorrectionOutput",
    "PredictionAnchoredBaseFusion",
    "SharedLocalMessageLayer",
    "TargetKernelLocalProcessor",
    "ParticleCapacityProcessor",
    "RadiusCorrectionHead",
    "PredictionAnchoredLocalCorrection",
    "CorrectionResourceProfile",
    "build_local_graph_features",
    "build_step6_correction_model",
    "measure_correction_resources",
    "canonical_a3_resource_reference",
    "particle_capacity_match",
    "measure_step6_registry_states",
    "tiny_train_reload_step6_model",
]
