"""Scaler-aware C0 correction path for the prediction-anchored bridge pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Callable, Mapping

import numpy as np
import torch

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from .bridge import (
    BRIDGE_CHANNEL_PHYSICAL45,
    PREDICTION_ANCHORED_BRIDGE_SCALER_CONTRACT,
    BridgeScalers,
)
from .bridge_contracts import validate_content_hash, with_content_hash


PREDICTION_ANCHORED_C0_CONFIG_CONTRACT = "prediction_anchored_c0_correction_config_v1"
PREDICTION_ANCHORED_C0_MODEL_CONTRACT = "prediction_anchored_c0_correction_model_v1"
PREDICTION_ANCHORED_FROZEN_LIVE_CONSUMER_CONTRACT = (
    "prediction_anchored_frozen_live_consumer_v1"
)

PHYSICAL_CHANNELS = 45
RELIABILITY_CHANNELS = 5
NUMERICAL_SPACE_PHYSICAL = "physical_field"
NUMERICAL_SPACE_CONDITIONING = "conditioning_standardized"
NUMERICAL_SPACE_CORRECTION = "correction_standardized"
NUMERICAL_SPACE_LOSS = "loss_standardized"


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _masked_jet_summary(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The existing 11-value HLT-only jet summary used by R0."""

    valid = mask.to(dtype=tokens.dtype)
    pt = torch.where(mask, tokens[..., 0], torch.zeros_like(tokens[..., 0]))
    eta = torch.where(mask, tokens[..., 1], torch.zeros_like(tokens[..., 1]))
    phi = torch.where(mask, tokens[..., 2], torch.zeros_like(tokens[..., 2]))
    energy = torch.where(mask, tokens[..., 3], torch.zeros_like(tokens[..., 3]))
    count = valid.sum(dim=1)
    sum_pt = pt.sum(dim=1)
    sum_energy = energy.sum(dim=1)
    denominator = torch.clamp(sum_pt, min=1.0e-6)
    eta_centroid = (pt * eta).sum(dim=1) / denominator
    sin_phi = (pt * torch.sin(phi)).sum(dim=1) / denominator
    cos_phi = (pt * torch.cos(phi)).sum(dim=1) / denominator
    phi_centroid = torch.atan2(sin_phi, cos_phi)
    d_eta = eta - eta_centroid[:, None]
    d_phi = torch.remainder(phi - phi_centroid[:, None] + math.pi, 2.0 * math.pi) - math.pi
    radial = torch.sqrt(d_eta.square() + d_phi.square() + 1.0e-12)
    radial_mean = (pt * radial).sum(dim=1) / denominator
    if tokens.shape[-1] >= 10:
        pid = tokens[..., 5:10]
    else:  # pragma: no cover - production tokens always have the PID block
        pid = tokens.new_zeros((*tokens.shape[:2], 5))
    pid_fraction = (pid * pt[..., None]).sum(dim=1) / denominator[:, None]
    return torch.cat(
        (
            torch.log1p(sum_pt)[:, None],
            torch.log1p(sum_energy)[:, None],
            torch.log1p(count)[:, None],
            eta_centroid[:, None],
            phi_centroid[:, None],
            radial_mean[:, None],
            pid_fraction,
        ),
        dim=1,
    )


@dataclass(frozen=True)
class C0CorrectionConfig:
    particle_dim: int = RAW_TOKEN_DIM
    h0_dim: int = 160
    field_dim: int = 50
    d_model: int = 160
    raw_projection_dim: int = 64
    f0_projection_dim: int = 64
    h0_projection_dim: int = 96
    jet_projection_dim: int = 32
    particle_mlp_layers: int = 2
    head_hidden_dim: int = 64
    dropout: float = 0.05
    trust_bound_enabled: bool = True
    zero_initialize_heads: bool = True

    def __post_init__(self) -> None:
        if int(self.particle_dim) != RAW_TOKEN_DIM:
            raise ValueError(f"C0 requires the repository raw-HLT width {RAW_TOKEN_DIM}")
        if int(self.h0_dim) != 160 or int(self.field_dim) != 50:
            raise ValueError("C0 requires h0 width 160 and field width 50")
        if (
            int(self.raw_projection_dim)
            + int(self.f0_projection_dim)
            + int(self.h0_projection_dim)
            + int(self.jet_projection_dim)
            != 256
        ):
            raise ValueError("C0 base fusion must concatenate to exactly 256 values")
        for name in (
            "d_model",
            "raw_projection_dim",
            "f0_projection_dim",
            "h0_projection_dim",
            "jet_projection_dim",
            "particle_mlp_layers",
            "head_hidden_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= float(self.dropout) < 1:
            raise ValueError("dropout must be in [0,1)")

    def to_artifact(self) -> dict[str, Any]:
        return with_content_hash(
            {"contract": PREDICTION_ANCHORED_C0_CONFIG_CONTRACT, **asdict(self)}
        )


class TorchBridgeScalers(torch.nn.Module):
    """Immutable scaler buffers with explicit numerical-space operations."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        super().__init__()
        validate_content_hash(
            artifact, expected_contract=PREDICTION_ANCHORED_BRIDGE_SCALER_CONTRACT
        )
        scaler = BridgeScalers.from_artifact(artifact)
        if scaler.channel_policy != BRIDGE_CHANNEL_PHYSICAL45:
            raise ValueError("the primary C0 correction path requires physical45 scalers")
        self.artifact_sha256 = str(artifact["content_hash"])
        for name in (
            "mu_f0",
            "sigma_f0",
            "q99_delta",
            "sigma_delta",
            "trust_scale",
            "epsilon",
        ):
            self.register_buffer(
                name,
                torch.as_tensor(getattr(scaler, name), dtype=torch.float32),
                persistent=True,
            )
        self.register_buffer(
            "active", torch.as_tensor(scaler.active, dtype=torch.bool), persistent=True
        )
        self.register_buffer(
            "sparse_nonzero_fallback",
            torch.as_tensor(scaler.sparse_nonzero_fallback, dtype=torch.bool),
            persistent=True,
        )

    def conditioning_standardize(
        self,
        f0: torch.Tensor,
        mask: torch.Tensor,
        *,
        input_space: str,
    ) -> torch.Tensor:
        if str(input_space) != NUMERICAL_SPACE_PHYSICAL:
            raise ValueError("C0 conditioning requires physical-field f0")
        if f0.ndim != 3 or f0.shape[-1] != 50 or mask.shape != f0.shape[:2]:
            raise ValueError("C0 f0/mask shapes do not align")
        standardized = (f0 - self.mu_f0) / self.sigma_f0
        return standardized.masked_fill(~mask.unsqueeze(-1), 0.0)

    def physical_correction(
        self,
        standardized_raw: torch.Tensor,
        mask: torch.Tensor,
        *,
        trust_bound_enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if standardized_raw.shape[-1] != PHYSICAL_CHANNELS:
            raise ValueError("C0 correction head must emit exactly 45 standardized channels")
        sigma = self.sigma_delta[:PHYSICAL_CHANNELS]
        trust = self.trust_scale[:PHYSICAL_CHANNELS]
        active = self.active[:PHYSICAL_CHANNELS]
        physical_raw = standardized_raw * sigma
        if bool(trust_bound_enabled):
            correction = trust * torch.tanh(physical_raw / trust)
            saturation = correction.abs() >= 0.99 * trust
            saturation = saturation & mask.unsqueeze(-1) & active
        else:
            correction = physical_raw
            saturation = None
        correction = correction * active.to(dtype=correction.dtype)
        correction = correction.masked_fill(~mask.unsqueeze(-1), 0.0)
        return correction, saturation


class _ResidualParticleMlp(torch.nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(width, eps=1.0e-5)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(width, width),
            torch.nn.Dropout(dropout),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.net(self.norm(hidden))


class _RadiusCorrectionHead(torch.nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float, zero_init: bool) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(width, eps=1.0e-5),
            torch.nn.Linear(width, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 15),
        )
        if zero_init:
            final = self.net[-1]
            assert isinstance(final, torch.nn.Linear)
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.zeros_(final.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


@dataclass(frozen=True)
class C0CorrectionOutput:
    f_hat: torch.Tensor
    physical_correction: torch.Tensor
    standardized_raw_correction: torch.Tensor
    hidden: torch.Tensor
    mask: torch.Tensor
    saturation_mask: torch.Tensor | None
    diagnostics: Mapping[str, Any]


class PredictionAnchoredC0Correction(torch.nn.Module):
    """Simple f0+h0+raw-HLT correction baseline with no hierarchy or gate."""

    def __init__(
        self,
        scaler_artifact: Mapping[str, Any],
        config: C0CorrectionConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or C0CorrectionConfig()
        self.scalers = TorchBridgeScalers(scaler_artifact)
        width = int(self.config.d_model)
        self.raw_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(int(self.config.particle_dim)),
            torch.nn.Linear(int(self.config.particle_dim), int(self.config.raw_projection_dim)),
            torch.nn.GELU(),
        )
        self.f0_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(50),
            torch.nn.Linear(50, int(self.config.f0_projection_dim)),
            torch.nn.GELU(),
        )
        self.h0_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(160),
            torch.nn.Linear(160, int(self.config.h0_projection_dim)),
            torch.nn.GELU(),
        )
        self.jet_projection = torch.nn.Sequential(
            torch.nn.Linear(11, int(self.config.jet_projection_dim)),
            torch.nn.GELU(),
        )
        self.fusion = torch.nn.Sequential(
            torch.nn.LayerNorm(256, eps=1.0e-5),
            torch.nn.Linear(256, width),
            torch.nn.GELU(),
            torch.nn.LayerNorm(width, eps=1.0e-5),
        )
        self.particle_mlp = torch.nn.ModuleList(
            [
                _ResidualParticleMlp(width, float(self.config.dropout))
                for _ in range(int(self.config.particle_mlp_layers))
            ]
        )
        self.radius_heads = torch.nn.ModuleList(
            [
                _RadiusCorrectionHead(
                    width,
                    int(self.config.head_hidden_dim),
                    float(self.config.dropout),
                    bool(self.config.zero_initialize_heads),
                )
                for _ in range(3)
            ]
        )
        if any("gate" in name.lower() for name, _ in self.named_parameters()):
            raise AssertionError("primary C0 must not instantiate a learned gate")

    @property
    def scaler_sha256(self) -> str:
        return self.scalers.artifact_sha256

    def config_artifact(self) -> dict[str, Any]:
        payload = self.config.to_artifact()
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_C0_MODEL_CONTRACT,
                "config_sha256": payload["content_hash"],
                "config": payload,
                "scaler_sha256": self.scaler_sha256,
                "input_spaces": {
                    "consumer": NUMERICAL_SPACE_PHYSICAL,
                    "conditioning": NUMERICAL_SPACE_CONDITIONING,
                    "correction_head": NUMERICAL_SPACE_CORRECTION,
                    "loss": NUMERICAL_SPACE_LOSS,
                },
                "physical_correction_channels": list(range(45)),
                "pass_through_channels": list(range(45, 50)),
                "learned_gate_present": False,
            }
        )

    def forward(
        self,
        hlt_tokens: torch.Tensor,
        mask: torch.Tensor,
        f0_live: torch.Tensor,
        h0: torch.Tensor,
        *,
        f0_space: str = NUMERICAL_SPACE_PHYSICAL,
    ) -> C0CorrectionOutput:
        tokens = hlt_tokens.to(dtype=torch.float32)
        valid = mask.to(device=tokens.device, dtype=torch.bool)
        anchor = f0_live.to(device=tokens.device, dtype=torch.float32).detach()
        frozen_h0 = h0.to(device=tokens.device, dtype=torch.float32).detach()
        if tokens.ndim != 3 or tokens.shape[-1] != int(self.config.particle_dim):
            raise ValueError("C0 raw HLT tokens have the wrong shape")
        if valid.shape != tokens.shape[:2] or anchor.shape != (*valid.shape, 50):
            raise ValueError("C0 token/mask/f0 shapes do not align")
        if frozen_h0.shape != (*valid.shape, 160):
            raise ValueError("C0 h0 must have shape [B,P,160]")
        if not torch.isfinite(tokens).all() or not torch.isfinite(anchor).all() or not torch.isfinite(frozen_h0).all():
            raise ValueError("C0 inputs contain non-finite values")
        standardized_f0 = self.scalers.conditioning_standardize(
            anchor, valid, input_space=f0_space
        )
        raw = self.raw_projection(tokens)
        field = self.f0_projection(standardized_f0)
        state = self.h0_projection(frozen_h0)
        jet = self.jet_projection(_masked_jet_summary(tokens, valid))
        jet = jet[:, None, :].expand(-1, tokens.shape[1], -1)
        hidden = self.fusion(torch.cat((raw, field, state, jet), dim=-1))
        hidden = hidden.masked_fill(~valid.unsqueeze(-1), 0.0)
        for layer in self.particle_mlp:
            hidden = layer(hidden).masked_fill(~valid.unsqueeze(-1), 0.0)
        standardized_raw = torch.cat(
            [head(hidden) for head in self.radius_heads], dim=-1
        )
        standardized_raw = standardized_raw.masked_fill(~valid.unsqueeze(-1), 0.0)
        correction, saturation = self.scalers.physical_correction(
            standardized_raw,
            valid,
            trust_bound_enabled=bool(self.config.trust_bound_enabled),
        )
        f_hat = anchor.clone()
        f_hat_physical = anchor[..., :45] + correction
        f_hat = torch.cat((f_hat_physical, anchor[..., 45:]), dim=-1)
        f_hat = f_hat.masked_fill(~valid.unsqueeze(-1), 0.0)
        valid_physical = int(valid.sum().detach().cpu().item()) * PHYSICAL_CHANNELS
        saturation_fraction = (
            None
            if saturation is None
            else float(saturation.sum().detach().cpu().item()) / max(valid_physical, 1)
        )
        diagnostics = {
            "contract": PREDICTION_ANCHORED_C0_MODEL_CONTRACT,
            "scaler_sha256": self.scaler_sha256,
            "f0_input_space": str(f0_space),
            "consumer_output_space": NUMERICAL_SPACE_PHYSICAL,
            "trust_bound_enabled": bool(self.config.trust_bound_enabled),
            "saturation_fraction": saturation_fraction,
            "saturation_definition": (
                "not_applicable"
                if saturation is None
                else "abs_delta_gte_0.99_times_component_trust_scale"
            ),
            "reliability_channels_exact_pass_through": bool(
                torch.equal(f_hat[..., 45:], anchor[..., 45:])
            ),
            "learned_gate_present": False,
            "f0_and_h0_stop_gradient": True,
        }
        return C0CorrectionOutput(
            f_hat=f_hat,
            physical_correction=correction,
            standardized_raw_correction=standardized_raw,
            hidden=hidden,
            mask=valid,
            saturation_mask=saturation,
            diagnostics=diagnostics,
        )


class FrozenLiveBridgeConsumer(torch.nn.Module):
    """Frozen/eval T10 parameters with a differentiable physical-field input."""

    def __init__(
        self,
        consumer: torch.nn.Module,
        *,
        checkpoint_sha256: str,
        forward_adapter: Callable[[torch.nn.Module, Mapping[str, Any], torch.Tensor], Any],
    ) -> None:
        super().__init__()
        if not _valid_sha256(checkpoint_sha256):
            raise ValueError("frozen live consumer requires a checkpoint SHA-256")
        self.consumer = consumer
        self.checkpoint_sha256 = str(checkpoint_sha256)
        self.forward_adapter = forward_adapter
        for parameter in self.consumer.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.consumer.eval()

    def train(self, mode: bool = True) -> "FrozenLiveBridgeConsumer":
        super().train(mode)
        self.consumer.eval()
        return self

    def forward(
        self,
        batch: Mapping[str, Any],
        physical_fields: torch.Tensor,
        *,
        input_space: str = NUMERICAL_SPACE_PHYSICAL,
    ) -> torch.Tensor:
        if str(input_space) != NUMERICAL_SPACE_PHYSICAL:
            raise ValueError("T10 must receive physical residual fields")
        self.consumer.eval()
        output = self.forward_adapter(self.consumer, batch, physical_fields)
        logits = getattr(output, "logits", output)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("frozen live consumer adapter must return [B,C] logits")
        if any(parameter.requires_grad for parameter in self.consumer.parameters()):
            raise AssertionError("a frozen T10 parameter became trainable")
        return logits

    def audit_artifact(self) -> dict[str, Any]:
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_FROZEN_LIVE_CONSUMER_CONTRACT,
                "checkpoint_sha256": self.checkpoint_sha256,
                "parameters_frozen": True,
                "consumer_eval_mode": not self.consumer.training,
                "live_field_input_gradient_enabled": True,
                "input_space": NUMERICAL_SPACE_PHYSICAL,
                "no_grad_wrapper_used": False,
            }
        )


__all__ = [
    "PREDICTION_ANCHORED_C0_CONFIG_CONTRACT",
    "PREDICTION_ANCHORED_C0_MODEL_CONTRACT",
    "PREDICTION_ANCHORED_FROZEN_LIVE_CONSUMER_CONTRACT",
    "PHYSICAL_CHANNELS",
    "RELIABILITY_CHANNELS",
    "NUMERICAL_SPACE_PHYSICAL",
    "NUMERICAL_SPACE_CONDITIONING",
    "NUMERICAL_SPACE_CORRECTION",
    "NUMERICAL_SPACE_LOSS",
    "C0CorrectionConfig",
    "TorchBridgeScalers",
    "C0CorrectionOutput",
    "PredictionAnchoredC0Correction",
    "FrozenLiveBridgeConsumer",
]
