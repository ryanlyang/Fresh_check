"""Privileged particle-view consumers with exact A0 zero-scale endpoints."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn

from .contracts import canonical_sha256
from .target_generator import masked_particle_mean_center


PARTICLE_VIEW_CONSUMER_CONFIG_CONTRACT = "particle_view_consumer_config_v1"
PARTICLE_VIEW_CONSUMER_PATHS = (
    "raw_projected",
    "token_only",
    "pair_only",
    "token_and_pair",
)


@dataclass(frozen=True)
class ParticleViewConsumerConfig:
    view_dim: int
    hidden_dim: int = 128
    num_heads: int = 8
    injection_block: int = 0
    view_path: str = "token_and_pair"
    learned_trust: bool = True
    view_clip: float = 6.0
    clean_coordinate_dropout: float = 0.05
    clean_noise_sigma: float = 0.02
    contract: str = PARTICLE_VIEW_CONSUMER_CONFIG_CONTRACT

    def __post_init__(self) -> None:
        if self.view_dim not in {1, 2, 4, 8}:
            raise ValueError("view_dim must be one of 1, 2, 4, or 8")
        if self.hidden_dim <= 0 or self.num_heads <= 0:
            raise ValueError("consumer dimensions must be positive")
        if self.view_path not in PARTICLE_VIEW_CONSUMER_PATHS:
            raise ValueError(f"view_path must be one of {PARTICLE_VIEW_CONSUMER_PATHS}")
        if self.injection_block < -1:
            raise ValueError("injection_block must be -1 or nonnegative")
        if self.view_path == "raw_projected" and self.injection_block != -1:
            raise ValueError("raw-projected consumer uses injection_block=-1")
        if not isinstance(self.learned_trust, bool):
            raise ValueError("learned_trust must be boolean")
        for name, value, upper in (
            ("view_clip", self.view_clip, None),
            ("clean_coordinate_dropout", self.clean_coordinate_dropout, 1.0),
            ("clean_noise_sigma", self.clean_noise_sigma, None),
        ):
            if not math.isfinite(value) or value < 0 or (
                upper is not None and value >= upper
            ):
                raise ValueError(f"{name} is invalid")

    @property
    def token_enabled(self) -> bool:
        return self.view_path in {"token_only", "token_and_pair"}

    @property
    def raw_enabled(self) -> bool:
        return self.view_path == "raw_projected"

    @property
    def pair_enabled(self) -> bool:
        return self.view_path in {"pair_only", "token_and_pair"}

    def to_payload(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "view_dim": self.view_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "injection_block": self.injection_block,
            "view_path": self.view_path,
            "learned_trust": self.learned_trust,
            "view_clip": self.view_clip,
            "clean_coordinate_dropout": self.clean_coordinate_dropout,
            "clean_noise_sigma": self.clean_noise_sigma,
            "token_scale": "tanh_bounded_zero_initialized",
            "pair_scale": "tanh_bounded_zero_initialized",
            "gate_initial_value": 0.5 if self.learned_trust else 1.0,
            "trust_regularizer_weight": 0.01 if self.learned_trust else 0.0,
            "token_injection_location": (
                "projected_into_raw_hlt_features_pre_embedding"
                if self.raw_enabled
                else "post_embedding_pre_particle_blocks"
                if self.injection_block == -1
                else "post_complete_particle_block"
            ),
            "pair_injection_location": "post_part_pair_embed_pre_attention",
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_payload())


@dataclass
class ParticleViewConsumerOutput:
    logits: torch.Tensor
    gate: torch.Tensor
    token_correction: torch.Tensor
    pair_bias: torch.Tensor
    raw_token_scale: torch.Tensor
    raw_pair_scale: torch.Tensor
    effective_token_scale: torch.Tensor
    effective_pair_scale: torch.Tensor
    trust_loss: torch.Tensor


def prepare_clean_consumer_view(
    view: torch.Tensor,
    mask: torch.Tensor,
    *,
    training: bool,
    coordinate_dropout: float = 0.05,
    noise_sigma: float = 0.02,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply the locked cache->dropout->noise->recenter->zero ordering."""

    if view.ndim != 3 or mask.shape != view.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("view/mask shapes must be [B,P,D] and boolean [B,P]")
    result = view.clamp(-6.0, 6.0)
    if training and coordinate_dropout:
        keep = torch.rand(
            (view.shape[0], 1, view.shape[2]),
            device=view.device,
            generator=generator,
        ) >= coordinate_dropout
        result = result * keep.to(result.dtype)
    if training and noise_sigma:
        noise = torch.randn(
            result.shape,
            device=result.device,
            dtype=result.dtype,
            generator=generator,
        ) * noise_sigma
        result = torch.where(mask[:, :, None], result + noise, result)
    # The registered float32 cache is the exact evaluation coordinate.  Its
    # values must not be silently transformed a second time.  Re-centering is
    # part of the locked *training augmentation* order only.
    if training:
        result = masked_particle_mean_center(result, mask)
    return torch.where(mask[:, :, None], result, torch.zeros_like(result))


def _xavier_linear(linear: nn.Linear) -> None:
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _first_tensor(value):
    return value[0] if isinstance(value, (tuple, list)) else value


def _replace_first(original, replacement):
    if isinstance(original, tuple):
        return (replacement, *original[1:])
    if isinstance(original, list):
        return [replacement, *original[1:]]
    return replacement


class ParticleViewConsumer(nn.Module):
    """Wrap an A0 ParT without modifying its frozen zero-view endpoint."""

    def __init__(self, a0_model: nn.Module, config: ParticleViewConsumerConfig):
        super().__init__()
        if not hasattr(a0_model, "mod"):
            raise ValueError("A0 model must expose the repository ParT as .mod")
        if not hasattr(a0_model.mod, "blocks"):
            raise ValueError("A0 ParT does not expose particle blocks")
        if (
            config.injection_block >= 0
            and config.injection_block >= len(a0_model.mod.blocks)
        ):
            raise ValueError("consumer injection block is outside A0")
        if config.pair_enabled and not hasattr(a0_model.mod, "pair_embed"):
            raise ValueError("pair-bias consumer requires A0.mod.pair_embed")
        self.a0_model = a0_model
        self.config = config
        hidden, view_dim = config.hidden_dim, config.view_dim
        self.view_adapter = nn.Sequential(
            nn.Linear(view_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.raw_adapter = nn.Linear(view_dim, 17)
        self.gate = nn.Sequential(
            nn.Linear(hidden + view_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.pair_adapter = nn.Sequential(
            nn.Linear(4 * view_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.num_heads),
        )
        for module in (self.view_adapter, self.pair_adapter):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    _xavier_linear(layer)
        _xavier_linear(self.raw_adapter)
        for layer in self.gate:
            if isinstance(layer, nn.Linear):
                _xavier_linear(layer)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.raw_token_scale = nn.Parameter(torch.zeros(()))
        self.raw_pair_scale = nn.Parameter(torch.zeros(()))

    def _token_layout(self, values: torch.Tensor, batch: int, particles: int):
        if tuple(values.shape[:2]) == (batch, particles):
            return values, "batch_particle_hidden"
        if tuple(values.shape[:2]) == (particles, batch):
            return values.permute(1, 0, 2), "particle_batch_hidden"
        # Weaver's embedding stack is channel-first [B,C,P], whereas its
        # particle blocks are sequence-first [P,B,C].
        if values.ndim == 3 and (
            values.shape[0] == batch and values.shape[2] == particles
        ):
            return values.permute(0, 2, 1), "batch_hidden_particle"
        raise ValueError("A0 particle-block output layout changed")

    @staticmethod
    def _restore_token_layout(values: torch.Tensor, layout: str):
        if layout == "batch_particle_hidden":
            return values
        if layout == "particle_batch_hidden":
            return values.permute(1, 0, 2)
        if layout == "batch_hidden_particle":
            return values.permute(0, 2, 1)
        raise ValueError("unknown A0 token layout")

    def _pair_features(self, view: torch.Tensor) -> torch.Tensor:
        left = view[:, :, None, :]
        right = view[:, None, :, :]
        return torch.cat(
            (
                left.expand(-1, -1, view.shape[1], -1),
                right.expand(-1, view.shape[1], -1, -1),
                left - right,
                left * right,
            ),
            dim=-1,
        )

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        view: torch.Tensor,
        *,
        augment_clean_view: bool = False,
        augmentation_generator: torch.Generator | None = None,
    ) -> ParticleViewConsumerOutput:
        if mask.ndim != 3 or mask.shape[1] != 1 or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean [B,1,P]")
        valid = mask[:, 0]
        if view.shape[:2] != valid.shape or view.shape[2] != self.config.view_dim:
            raise ValueError("view shape differs from consumer configuration")
        view = prepare_clean_consumer_view(
            view,
            valid,
            training=bool(self.training and augment_clean_view),
            coordinate_dropout=self.config.clean_coordinate_dropout,
            noise_sigma=self.config.clean_noise_sigma,
            generator=augmentation_generator,
        )
        batch, particles = valid.shape
        embedded = self.a0_model.mod.embed(features)
        embedded, _ = self._token_layout(
            _first_tensor(embedded), batch, particles
        )
        if embedded.shape[-1] != self.config.hidden_dim:
            raise ValueError("A0 embedding width differs from consumer")
        if self.config.learned_trust:
            initial_gate = torch.sigmoid(
                self.gate(torch.cat((embedded, view), dim=-1))
            )
        else:
            initial_gate = torch.ones(
                (*valid.shape, 1), device=view.device, dtype=view.dtype
            )
        initial_gate = torch.where(
            valid[:, :, None], initial_gate, torch.zeros_like(initial_gate)
        )
        state: dict[str, torch.Tensor] = {"gate": initial_gate}
        model_features = features
        if self.config.raw_enabled:
            raw_correction = self.raw_adapter(view) * initial_gate
            raw_correction = torch.where(
                valid[:, :, None],
                raw_correction,
                torch.zeros_like(raw_correction),
            )
            model_features = features + torch.tanh(
                self.raw_token_scale
            ) * raw_correction.transpose(1, 2)
            state["token_correction"] = raw_correction

        def token_hook(_module, _inputs, output):
            values, layout = self._token_layout(
                _first_tensor(output), batch, particles
            )
            if values.shape[-1] != self.config.hidden_dim:
                raise ValueError("A0 hidden width differs from consumer")
            gate = state["gate"]
            correction = self.view_adapter(view) * gate
            correction = torch.where(
                valid[:, :, None], correction, torch.zeros_like(correction)
            )
            effective = torch.tanh(self.raw_token_scale)
            updated = values + effective * correction
            state["gate"] = gate
            state["token_correction"] = correction
            updated = self._restore_token_layout(updated, layout)
            return _replace_first(output, updated)

        def pair_hook(_module, _inputs, output):
            gate = state["gate"]
            pair = self.pair_adapter(self._pair_features(view)).permute(0, 3, 1, 2)
            # A zero gate belongs to padding.  Clamping before sqrt avoids the
            # infinite derivative at zero; padding is then removed exactly by
            # pair_valid below.
            trust = torch.sqrt(
                (gate[:, :, None, 0] * gate[:, None, :, 0]).clamp_min(
                    1.0e-12
                )
            )
            pair = pair * trust[:, None]
            pair_valid = valid[:, None, :, None] & valid[:, None, None, :]
            pair = torch.where(pair_valid, pair, torch.zeros_like(pair))
            effective = torch.tanh(self.raw_pair_scale)
            original = _first_tensor(output)
            if original.shape == pair.shape:
                updated = original + effective * pair
            elif original.shape == (
                batch * self.config.num_heads,
                particles,
                particles,
            ):
                updated = original + effective * pair.reshape_as(original)
            elif original.shape == (
                batch,
                particles,
                particles,
                self.config.num_heads,
            ):
                updated = original + effective * pair.permute(0, 2, 3, 1)
            else:
                raise ValueError("A0 pair-bias tensor layout changed")
            state["pair_bias"] = pair
            return _replace_first(output, updated)

        handles = []
        if self.config.pair_enabled:
            handles.append(self.a0_model.mod.pair_embed.register_forward_hook(pair_hook))
        if self.config.token_enabled:
            token_module = (
                self.a0_model.mod.embed
                if self.config.injection_block == -1
                else self.a0_model.mod.blocks[self.config.injection_block]
            )
            handles.append(
                token_module.register_forward_hook(token_hook)
            )
        try:
            logits = self.a0_model(
                points, model_features, lorentz_vectors, mask
            )
        finally:
            for handle in handles:
                handle.remove()
        gate = state.get(
            "gate",
            torch.ones((*valid.shape, 1), device=view.device, dtype=view.dtype),
        )
        token_correction = state.get(
            "token_correction",
            torch.zeros(
                (batch, particles, self.config.hidden_dim),
                device=view.device,
                dtype=view.dtype,
            ),
        )
        pair_bias = state.get(
            "pair_bias",
            torch.zeros(
                (batch, self.config.num_heads, particles, particles),
                device=view.device,
                dtype=view.dtype,
            ),
        )
        trust_loss = (
            gate[valid].mean()
            if self.config.learned_trust and valid.any()
            else logits.new_zeros(())
        )
        return ParticleViewConsumerOutput(
            logits=logits,
            gate=gate,
            token_correction=token_correction,
            pair_bias=pair_bias,
            raw_token_scale=self.raw_token_scale,
            raw_pair_scale=self.raw_pair_scale,
            effective_token_scale=torch.tanh(self.raw_token_scale),
            effective_pair_scale=torch.tanh(self.raw_pair_scale),
            trust_loss=trust_loss,
        )


def consumer_diagnostics(
    output: ParticleViewConsumerOutput, mask: torch.Tensor
) -> dict[str, float]:
    valid = mask[:, 0]
    gates = output.gate[..., 0][valid]
    if gates.numel():
        quantiles = torch.quantile(
            gates.detach().float(), torch.tensor([0.01, 0.5, 0.99], device=gates.device)
        )
    else:
        quantiles = torch.zeros(3, device=output.logits.device)
    token = output.effective_token_scale.detach() * output.token_correction.detach()
    token_norm = token.norm(dim=-1)[valid]
    pair_valid = valid[:, None, :, None] & valid[:, None, None, :]
    pair = output.effective_pair_scale.detach() * output.pair_bias.detach()
    pair_values = pair[pair_valid.expand_as(pair)]

    def _rms(values: torch.Tensor) -> float:
        if not values.numel():
            return 0.0
        return float(values.float().square().mean().sqrt().item())

    def _p99(values: torch.Tensor) -> float:
        if not values.numel():
            return 0.0
        return float(torch.quantile(values.detach().float(), 0.99).item())

    result = {
        "gate_mean": float(gates.mean().item()) if gates.numel() else 0.0,
        "gate_p01": float(quantiles[0].item()),
        "gate_p50": float(quantiles[1].item()),
        "gate_p99": float(quantiles[2].item()),
        "gate_fraction_below_001": float((gates < 0.01).float().mean().item())
        if gates.numel()
        else 0.0,
        "gate_fraction_above_099": float((gates > 0.99).float().mean().item())
        if gates.numel()
        else 0.0,
        "raw_token_scale": float(output.raw_token_scale.detach().item()),
        "effective_token_scale": float(output.effective_token_scale.detach().item()),
        "raw_pair_scale": float(output.raw_pair_scale.detach().item()),
        "effective_pair_scale": float(output.effective_pair_scale.detach().item()),
        "effective_token_correction_rms": _rms(token_norm),
        "effective_token_correction_norm_p99": _p99(token_norm),
        "effective_pair_bias_rms": _rms(pair_values),
        "effective_pair_bias_abs_p99": _p99(pair_values.abs()),
    }
    for head in range(pair.shape[1]):
        selected = pair[:, head][
            valid[:, :, None] & valid[:, None, :]
        ]
        result[f"effective_pair_bias_head{head}_rms"] = _rms(selected)
        result[f"effective_pair_bias_head{head}_abs_p99"] = _p99(
            selected.abs()
        )
    return result


def audit_zero_scaled_a0_endpoint(
    consumer: ParticleViewConsumer,
    *,
    points: torch.Tensor,
    features: torch.Tensor,
    lorentz_vectors: torch.Tensor,
    mask: torch.Tensor,
    view: torch.Tensor,
    tolerance: float = 1.0e-7,
) -> dict[str, float | bool]:
    """Verify the exact warm-start endpoint before any optimizer update."""

    if float(consumer.raw_token_scale.detach().abs().item()) != 0.0:
        raise ValueError("token residual scale is not exactly zero")
    if float(consumer.raw_pair_scale.detach().abs().item()) != 0.0:
        raise ValueError("pair residual scale is not exactly zero")
    was_training = consumer.training
    consumer.eval()
    with torch.no_grad():
        wrapped = consumer(
            points, features, lorentz_vectors, mask, view
        ).logits
        reference = consumer.a0_model(
            points, features, lorentz_vectors, mask
        )
    consumer.train(was_training)
    maximum = float((wrapped - reference).abs().max().item())
    if maximum > tolerance:
        raise ValueError(
            "zero-scaled particle-view consumer does not reproduce A0: "
            f"max_abs={maximum:.9g}"
        )
    return {
        "ok": True,
        "maximum_absolute_logit_difference": maximum,
        "tolerance": tolerance,
        "token_scale_exactly_zero": True,
        "pair_scale_exactly_zero": True,
    }


__all__ = [
    "PARTICLE_VIEW_CONSUMER_CONFIG_CONTRACT",
    "PARTICLE_VIEW_CONSUMER_PATHS",
    "ParticleViewConsumer",
    "ParticleViewConsumerConfig",
    "ParticleViewConsumerOutput",
    "audit_zero_scaled_a0_endpoint",
    "consumer_diagnostics",
    "prepare_clean_consumer_view",
]
