"""Denoiser-augmented Particle Transformer tagger wrapper.

Step 4 keeps the canonical HLT ParT inputs as the scoring backbone input.  The
target-conditioned denoiser sees the same HLT particles, predicts per-particle
correction hypotheses, and a small zero-initialized adapter converts those
fields into a residual in ParT embedding space.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import build_hlt_classifier, require_torch
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES
from teacher_logit_reco.local_compression_part.features import (
    LocalCompressionCanonicalInputs,
    build_local_compression_canonical_inputs,
)

from .data import DENOISING_TARGET_NAMES
from .model import (
    TARGET_DENOISING_MODEL_CONTRACT,
    TargetConditionedDenoiserConfig,
    TargetConditionedPairwiseDenoiser,
    TargetDenoisingOutput,
)
from .train import TARGET_DENOISING_TRAINING_CONTRACT

try:  # Keep imports cheap where torch is absent.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


TARGET_DENOISING_STEP4 = "target_conditioned_denoising_part_step4_augmented_part"
TARGET_DENOISING_TAGGER_CONTRACT = "target_conditioned_denoising_augmented_part_tagger_v1"

TARGET_DENOISING_VARIANT_HLT_PART_BASELINE = "hlt_part_baseline"
TARGET_DENOISING_VARIANT_FEATURE_MLP_ADAPTER_TAG_ONLY = "feature_mlp_adapter_tag_only"
TARGET_DENOISING_VARIANT_DENOISER_FEATURES_FROZEN = "denoiser_features_frozen"
TARGET_DENOISING_VARIANT_DENOISER_FEATURES_JOINT = "denoiser_features_joint"
TARGET_DENOISING_VARIANT_DENOISER_TAG_ONLY_SAME_ARCH = "denoiser_tag_only_same_arch"
TARGET_DENOISING_VARIANT_DENOISER_SHUFFLED_TARGETS = "denoiser_shuffled_targets"
TARGET_DENOISING_VARIANT_DENOISER_NO_PAIR_BIAS = "denoiser_no_pair_bias"
TARGET_DENOISING_VARIANT_DENOISER_LOCAL_KERNEL_ONLY = "denoiser_local_kernel_only"

TARGET_DENOISING_TAGGER_VARIANTS = (
    TARGET_DENOISING_VARIANT_HLT_PART_BASELINE,
    TARGET_DENOISING_VARIANT_FEATURE_MLP_ADAPTER_TAG_ONLY,
    TARGET_DENOISING_VARIANT_DENOISER_FEATURES_FROZEN,
    TARGET_DENOISING_VARIANT_DENOISER_FEATURES_JOINT,
    TARGET_DENOISING_VARIANT_DENOISER_TAG_ONLY_SAME_ARCH,
    TARGET_DENOISING_VARIANT_DENOISER_SHUFFLED_TARGETS,
    TARGET_DENOISING_VARIANT_DENOISER_NO_PAIR_BIAS,
    TARGET_DENOISING_VARIANT_DENOISER_LOCAL_KERNEL_ONLY,
)
TARGET_DENOISING_DENOISER_VARIANTS = {
    TARGET_DENOISING_VARIANT_DENOISER_FEATURES_FROZEN,
    TARGET_DENOISING_VARIANT_DENOISER_FEATURES_JOINT,
    TARGET_DENOISING_VARIANT_DENOISER_TAG_ONLY_SAME_ARCH,
    TARGET_DENOISING_VARIANT_DENOISER_SHUFFLED_TARGETS,
    TARGET_DENOISING_VARIANT_DENOISER_NO_PAIR_BIAS,
    TARGET_DENOISING_VARIANT_DENOISER_LOCAL_KERNEL_ONLY,
}


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))


def _coerce_tokens_and_mask(tokens_or_batch: Any, mask: Any | None = None) -> tuple[Any, Any]:
    torch = require_torch()
    if isinstance(tokens_or_batch, Mapping):
        batch = tokens_or_batch
        token_keys = ("hlt_tokens", "tokens", "raw_tokens", "x")
        mask_keys = ("hlt_constituent_mask", "hlt_mask", "mask", "raw_mask", "particle_mask")
        tokens = None
        resolved_mask = mask
        for key in token_keys:
            if key in batch:
                tokens = batch[key]
                break
        if tokens is None:
            raise ValueError(f"batch mapping must contain one of {token_keys}")
        if resolved_mask is None:
            for key in mask_keys:
                if key in batch:
                    resolved_mask = batch[key]
                    break
        if resolved_mask is None:
            raise ValueError(f"batch mapping must contain one of {mask_keys} when mask is not provided")
    else:
        if mask is None:
            raise ValueError("mask is required when passing raw token tensors")
        tokens = tokens_or_batch
        resolved_mask = mask
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.as_tensor(tokens, dtype=torch.float32)
    else:
        tokens = tokens.float()
    if not isinstance(resolved_mask, torch.Tensor):
        resolved_mask = torch.as_tensor(resolved_mask, dtype=torch.bool, device=tokens.device)
    else:
        resolved_mask = resolved_mask.to(device=tokens.device, dtype=torch.bool)
    if int(tokens.ndim) != 3:
        raise ValueError(f"tokens must have shape [batch, particles, raw_dim], got {tuple(tokens.shape)}")
    if int(resolved_mask.ndim) != 2 or tuple(resolved_mask.shape) != tuple(tokens.shape[:2]):
        raise ValueError(f"mask shape {tuple(resolved_mask.shape)} does not match token rows {tuple(tokens.shape[:2])}")
    return _nan_to_num_torch(tokens), resolved_mask


def _resolve_embed_module(part_model: Any) -> tuple[str, Any]:
    candidates = (
        ("mod.embed", getattr(getattr(part_model, "mod", None), "embed", None)),
        ("embed", getattr(part_model, "embed", None)),
    )
    for name, module in candidates:
        if module is not None and hasattr(module, "register_forward_hook"):
            return name, module
    for name, module in part_model.named_modules():
        if str(name).endswith("embed") and hasattr(module, "register_forward_hook"):
            return str(name), module
    raise ValueError("Could not locate a ParT particle embedding module for denoising delta_h injection.")


def _delta_for_embed_output(delta_h: Any, embed_output: Any) -> Any:
    torch = require_torch()
    if not isinstance(embed_output, torch.Tensor):
        raise TypeError(f"ParT embed output must be a tensor, got {type(embed_output)!r}")
    if int(embed_output.ndim) != 3:
        raise ValueError(f"ParT embed output must be rank 3, got {tuple(embed_output.shape)}")
    batch, particles, dim = delta_h.shape
    shape = tuple(embed_output.shape)
    if int(shape[-1]) == int(dim):
        if int(shape[0]) == int(batch):
            if int(shape[1]) > int(particles):
                raise ValueError(f"embed particle dimension {shape[1]} exceeds delta_h particles {particles}")
            return delta_h[:, : int(shape[1]), :]
        if int(shape[1]) == int(batch):
            if int(shape[0]) > int(particles):
                raise ValueError(f"embed particle dimension {shape[0]} exceeds delta_h particles {particles}")
            return delta_h[:, : int(shape[0]), :].permute(1, 0, 2).contiguous()
    if int(shape[0]) == int(batch) and int(shape[1]) == int(dim):
        if int(shape[2]) > int(particles):
            raise ValueError(f"embed particle dimension {shape[2]} exceeds delta_h particles {particles}")
        return delta_h[:, : int(shape[2]), :].transpose(1, 2).contiguous()
    raise ValueError(f"Cannot align delta_h shape {tuple(delta_h.shape)} with embed output shape {shape}")


def _batch_first_from_embed_output(embed_output: Any, *, batch: int, particles: int, dim: int) -> Any:
    if int(embed_output.ndim) != 3:
        raise ValueError(f"ParT embed output must be rank 3, got {tuple(embed_output.shape)}")
    shape = tuple(embed_output.shape)
    if int(shape[-1]) == int(dim):
        if int(shape[0]) == int(batch):
            return embed_output[:, :particles, :]
        if int(shape[1]) == int(batch):
            return embed_output[:particles, :, :].permute(1, 0, 2).contiguous()
    if int(shape[0]) == int(batch) and int(shape[1]) == int(dim):
        return embed_output[:, :, :particles].transpose(1, 2).contiguous()
    raise ValueError(f"Cannot convert embed output shape {shape} to batch-first [B, P, D]")


def _tensor_quantile(value: Any, q: float) -> Any:
    torch = require_torch()
    flat = value.reshape(-1).float()
    if int(flat.numel()) == 0:
        return torch.zeros((), dtype=flat.dtype, device=value.device)
    if hasattr(torch, "quantile"):
        return torch.quantile(flat, float(q))
    sorted_flat = torch.sort(flat).values
    index = min(max(int(round(float(q) * (int(sorted_flat.numel()) - 1))), 0), int(sorted_flat.numel()) - 1)
    return sorted_flat[index]


@dataclass(frozen=True)
class TargetDenoisingAugmentedParTConfig:
    """Configuration for the Step 4 denoiser-augmented ParT wrapper."""

    variant: str = TARGET_DENOISING_VARIANT_DENOISER_FEATURES_FROZEN
    num_classes: int = 10
    model_size: str = "base"
    part_embed_dim: int = 128
    max_constits: int = 128
    weight_threshold: float = 0.0
    adapter_hidden_dim: int = 128
    adapter_dropout: float = 0.0
    adapter_gate_bias_init: float = -2.0
    zero_init_adapter: bool = True
    include_canonical_features: bool = True
    include_denoiser_deltas: bool = True
    include_denoiser_log_variances: bool = True
    include_denoiser_reliability: bool = True
    freeze_denoiser: bool | None = None
    denoiser_config: TargetConditionedDenoiserConfig = field(default_factory=TargetConditionedDenoiserConfig)

    def __post_init__(self) -> None:
        variant = str(self.variant)
        if variant not in TARGET_DENOISING_TAGGER_VARIANTS:
            raise ValueError(f"unknown target-denoising tagger variant {variant!r}")
        object.__setattr__(self, "variant", variant)
        for name in ("num_classes", "part_embed_dim", "max_constits", "adapter_hidden_dim"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in ("adapter_dropout",):
            value = float(getattr(self, name))
            if value < 0.0 or value >= 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
            object.__setattr__(self, name, value)
        if float(self.weight_threshold) < 0.0:
            raise ValueError("weight_threshold must be non-negative")
        object.__setattr__(self, "weight_threshold", float(self.weight_threshold))
        object.__setattr__(self, "adapter_gate_bias_init", float(self.adapter_gate_bias_init))
        denoiser_config = self.denoiser_config
        if not isinstance(denoiser_config, TargetConditionedDenoiserConfig):
            denoiser_config = TargetConditionedDenoiserConfig(**dict(denoiser_config))
        if variant == TARGET_DENOISING_VARIANT_DENOISER_NO_PAIR_BIAS:
            denoiser_config = replace(denoiser_config, use_pair_bias=False, use_local_kernel=False)
        elif variant == TARGET_DENOISING_VARIANT_DENOISER_LOCAL_KERNEL_ONLY:
            denoiser_config = replace(denoiser_config, use_pair_bias=False, use_local_kernel=True)
        object.__setattr__(self, "denoiser_config", denoiser_config)

    @property
    def uses_denoiser(self) -> bool:
        return self.variant in TARGET_DENOISING_DENOISER_VARIANTS

    @property
    def uses_embedding_adapter(self) -> bool:
        return self.variant != TARGET_DENOISING_VARIANT_HLT_PART_BASELINE

    @property
    def effective_freeze_denoiser(self) -> bool:
        if self.freeze_denoiser is not None:
            return bool(self.freeze_denoiser)
        return self.variant == TARGET_DENOISING_VARIANT_DENOISER_FEATURES_FROZEN

    @property
    def denoiser_condition_dim(self) -> int:
        dim = 0
        if bool(self.include_canonical_features):
            dim += len(PF_FEATURE_NAMES)
        if self.uses_denoiser and bool(self.include_denoiser_deltas):
            dim += len(DENOISING_TARGET_NAMES)
        if self.uses_denoiser and bool(self.include_denoiser_log_variances):
            dim += len(DENOISING_TARGET_NAMES)
        if self.uses_denoiser and bool(self.include_denoiser_reliability):
            dim += 1
        if dim <= 0:
            raise ValueError("embedding adapter needs at least one conditioning field")
        return int(dim)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["denoiser_config"] = asdict(self.denoiser_config)
        data["uses_denoiser"] = bool(self.uses_denoiser)
        data["uses_embedding_adapter"] = bool(self.uses_embedding_adapter)
        data["effective_freeze_denoiser"] = bool(self.effective_freeze_denoiser)
        data["denoiser_condition_dim"] = int(self.denoiser_condition_dim)
        return data


class DenoisingEmbeddingResidualAdapter(_ModuleBase):
    """Zero-init MLP that maps denoiser fields to ParT embedding residuals."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        gate_bias_init: float = -2.0,
        zero_init: bool = True,
    ) -> None:
        torch = require_torch()
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(self.input_dim),
            torch.nn.Linear(self.input_dim, int(hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(int(hidden_dim), int(output_dim)),
        )
        self.gate = torch.nn.Sequential(
            torch.nn.LayerNorm(self.input_dim),
            torch.nn.Linear(self.input_dim, 1),
        )
        if bool(zero_init):
            torch.nn.init.zeros_(self.network[-1].weight)
            torch.nn.init.zeros_(self.network[-1].bias)
        torch.nn.init.zeros_(self.gate[-1].weight)
        torch.nn.init.constant_(self.gate[-1].bias, float(gate_bias_init))

    def forward(self, conditioning: Any, mask: Any) -> tuple[Any, Any, dict[str, Any]]:
        torch = require_torch()
        conditioning = _nan_to_num_torch(conditioning.float())
        mask = mask.to(device=conditioning.device, dtype=torch.bool)
        if int(conditioning.ndim) != 3 or int(conditioning.shape[-1]) != int(self.input_dim):
            raise ValueError(f"conditioning must have shape [B, P, {self.input_dim}], got {tuple(conditioning.shape)}")
        if tuple(mask.shape) != tuple(conditioning.shape[:2]):
            raise ValueError("mask shape must match conditioning leading dimensions")
        gate = torch.sigmoid(self.gate(conditioning))
        delta_h = gate * self.network(conditioning)
        delta_h = torch.where(mask[:, :, None], _nan_to_num_torch(delta_h), torch.zeros_like(delta_h))
        gate = torch.where(mask[:, :, None], _nan_to_num_torch(gate), torch.zeros_like(gate))
        if bool(mask.any()):
            valid_delta = delta_h[mask]
            valid_gate = gate.squeeze(-1)[mask]
            delta_norm = valid_delta.norm(dim=-1)
            diagnostics = {
                "adapter_output_norm_mean": float(delta_norm.mean().detach().cpu().item()),
                "adapter_output_norm_p90": float(_tensor_quantile(delta_norm, 0.90).detach().cpu().item()),
                "adapter_output_norm_max": float(delta_norm.max().detach().cpu().item()),
                "gate_mean": float(valid_gate.mean().detach().cpu().item()),
                "gate_p10": float(_tensor_quantile(valid_gate, 0.10).detach().cpu().item()),
                "gate_p90": float(_tensor_quantile(valid_gate, 0.90).detach().cpu().item()),
            }
        else:
            diagnostics = {
                "adapter_output_norm_mean": 0.0,
                "adapter_output_norm_p90": 0.0,
                "adapter_output_norm_max": 0.0,
                "gate_mean": 0.0,
                "gate_p10": 0.0,
                "gate_p90": 0.0,
            }
        return delta_h, gate, diagnostics


@dataclass(frozen=True)
class TargetDenoisingAugmentedParTOutput:
    """Full output of the Step 4 denoiser-augmented ParT wrapper."""

    logits: Any
    canonical_inputs: LocalCompressionCanonicalInputs
    config: TargetDenoisingAugmentedParTConfig
    denoiser_output: TargetDenoisingOutput | None
    delta_h: Any | None
    gate: Any | None
    injection_summary: Mapping[str, Any]
    adapter_diagnostics: Mapping[str, Any]
    denoiser_checkpoint_report: Mapping[str, Any] | None = None

    @property
    def output_contract(self) -> str:
        return TARGET_DENOISING_TAGGER_CONTRACT

    @property
    def variant(self) -> str:
        return str(self.config.variant)

    def diagnostics(self) -> dict[str, Any]:
        torch = require_torch()
        mask = self.canonical_inputs.particle_mask.bool()
        diagnostics = {
            "contract": TARGET_DENOISING_TAGGER_CONTRACT,
            "step": TARGET_DENOISING_STEP4,
            "variant": str(self.variant),
            "logits_shape": list(self.logits.shape),
            "canonical_features_shape": list(self.canonical_inputs.features.shape),
            "valid_particle_count": int(mask.detach().cpu().sum().item()),
            "uses_denoiser": bool(self.config.uses_denoiser),
            "uses_embedding_adapter": bool(self.config.uses_embedding_adapter),
            "original_hlt_part_inputs_unchanged": True,
            "denoiser_checkpoint": dict(self.denoiser_checkpoint_report or {}) or None,
            "injection_summary": dict(self.injection_summary),
            "adapter_diagnostics": dict(self.adapter_diagnostics),
        }
        if self.delta_h is not None:
            delta = self.delta_h.detach()
            diagnostics["delta_h_shape"] = list(delta.shape)
            diagnostics["delta_h_abs_max"] = float(delta.abs().max().cpu().item()) if int(delta.numel()) else 0.0
            if bool(mask.any()):
                aligned_mask = mask[:, : int(delta.shape[1])]
                norm = delta[:, : int(aligned_mask.shape[1])].norm(dim=-1)[aligned_mask]
                diagnostics["delta_h_norm_mean"] = float(norm.mean().cpu().item()) if int(norm.numel()) else 0.0
        else:
            diagnostics["delta_h_shape"] = None
            diagnostics["delta_h_abs_max"] = 0.0
        if self.denoiser_output is not None:
            diagnostics["denoiser_output_summary"] = self.denoiser_output.summary()
            diagnostics["denoiser_delta_abs_mean"] = float(
                self.denoiser_output.deltas.detach()[self.denoiser_output.token_mask.bool()].abs().mean().cpu().item()
            ) if bool(self.denoiser_output.token_mask.bool().any()) else 0.0
            diagnostics["denoiser_attention_bias_abs_max"] = float(
                self.denoiser_output.attention_bias.detach().abs().max().cpu().item()
            ) if int(self.denoiser_output.attention_bias.numel()) else 0.0
        else:
            diagnostics["denoiser_output_summary"] = None
        return diagnostics


class TargetDenoisingAugmentedParT(_ModuleBase):
    """ParT tagger that consumes target-conditioned denoising context."""

    def __init__(
        self,
        config: TargetDenoisingAugmentedParTConfig | Mapping[str, Any] | None = None,
        *,
        part_model: Any | None = None,
        denoiser: TargetConditionedPairwiseDenoiser | None = None,
    ) -> None:
        torch = require_torch()
        super().__init__()
        self.config = config if isinstance(config, TargetDenoisingAugmentedParTConfig) else TargetDenoisingAugmentedParTConfig(**dict(config or {}))
        self.part_model = part_model or build_hlt_classifier(
            num_classes=int(self.config.num_classes),
            model_size=str(self.config.model_size),
        )
        self.denoiser = denoiser or (
            TargetConditionedPairwiseDenoiser(self.config.denoiser_config) if self.config.uses_denoiser else None
        )
        self.embedding_adapter = (
            DenoisingEmbeddingResidualAdapter(
                input_dim=int(self.config.denoiser_condition_dim),
                output_dim=int(self.config.part_embed_dim),
                hidden_dim=int(self.config.adapter_hidden_dim),
                dropout=float(self.config.adapter_dropout),
                gate_bias_init=float(self.config.adapter_gate_bias_init),
                zero_init=bool(self.config.zero_init_adapter),
            )
            if self.config.uses_embedding_adapter
            else None
        )
        self.embed_module_name, self.embed_module = (None, None)
        if self.config.uses_embedding_adapter:
            self.embed_module_name, self.embed_module = _resolve_embed_module(self.part_model)
        self.denoiser_checkpoint_report: dict[str, Any] | None = None
        self._pending_delta_h: Any | None = None
        self._last_injection_summary: dict[str, Any] = {}
        if self.denoiser is not None and bool(self.config.effective_freeze_denoiser):
            self.set_denoiser_trainable(False)
        elif self.denoiser is not None:
            self.set_denoiser_trainable(True)
        self._last_part_embedding: Any | None = None

    @property
    def output_contract(self) -> str:
        return TARGET_DENOISING_TAGGER_CONTRACT

    @property
    def variant_name(self) -> str:
        return str(self.config.variant)

    def no_weight_decay(self) -> set[str]:
        names: set[str] = set()
        if hasattr(self.part_model, "no_weight_decay"):
            names.update({f"part_model.{name}" for name in self.part_model.no_weight_decay()})
        return names

    def set_denoiser_trainable(self, trainable: bool) -> None:
        if self.denoiser is None:
            return
        for parameter in self.denoiser.parameters():
            parameter.requires_grad = bool(trainable)

    def adapter_modules(self) -> dict[str, Any]:
        modules: dict[str, Any] = {}
        if self.embedding_adapter is not None:
            modules["embedding_adapter"] = self.embedding_adapter
        if self.denoiser is not None:
            modules["denoiser"] = self.denoiser
        return modules

    def variant_behavior(self) -> dict[str, Any]:
        return {
            "variant": str(self.variant_name),
            "uses_denoiser": bool(self.config.uses_denoiser),
            "uses_embedding_adapter": bool(self.config.uses_embedding_adapter),
            "uses_feature_mlp_tag_only": self.variant_name == TARGET_DENOISING_VARIANT_FEATURE_MLP_ADAPTER_TAG_ONLY,
            "uses_pretrained_denoiser_features": self.variant_name
            in {
                TARGET_DENOISING_VARIANT_DENOISER_FEATURES_FROZEN,
                TARGET_DENOISING_VARIANT_DENOISER_FEATURES_JOINT,
                TARGET_DENOISING_VARIANT_DENOISER_SHUFFLED_TARGETS,
                TARGET_DENOISING_VARIANT_DENOISER_NO_PAIR_BIAS,
                TARGET_DENOISING_VARIANT_DENOISER_LOCAL_KERNEL_ONLY,
            },
            "freezes_denoiser": bool(self.config.effective_freeze_denoiser),
            "joint_denoiser_finetune": bool(self.config.uses_denoiser and not self.config.effective_freeze_denoiser),
            "expects_shuffled_denoiser_pretraining": self.variant_name == TARGET_DENOISING_VARIANT_DENOISER_SHUFFLED_TARGETS,
            "denoiser_use_pair_bias": None if self.denoiser is None else bool(self.denoiser.config.use_pair_bias),
            "denoiser_use_local_kernel": None if self.denoiser is None else bool(self.denoiser.config.use_local_kernel),
            "original_hlt_part_inputs_unchanged": True,
            "integration": "denoiser_outputs_to_embedding_delta_h",
        }

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "contract": TARGET_DENOISING_TAGGER_CONTRACT,
            "step": TARGET_DENOISING_STEP4,
            "config": self.config.to_dict(),
            "part_model_config": dict(getattr(self.part_model, "config", {}) or {}),
            "denoiser_config": None if self.denoiser is None else asdict(self.denoiser.config),
            "embed_module_name": self.embed_module_name,
            "variant_behavior": self.variant_behavior(),
            "denoiser_checkpoint": dict(self.denoiser_checkpoint_report or {}),
        }

    def build_canonical_inputs(
        self,
        tokens: Any,
        mask: Any,
        *,
        weights: Any | None = None,
        max_constits: int | None = None,
        weight_threshold: float | None = None,
    ) -> LocalCompressionCanonicalInputs:
        if max_constits is None:
            max_constits = int(self.config.max_constits)
        if weight_threshold is None:
            weight_threshold = float(self.config.weight_threshold)
        return build_local_compression_canonical_inputs(
            tokens,
            mask,
            weights=weights,
            max_constits=int(max_constits),
            weight_threshold=float(weight_threshold),
        )

    def _conditioning_fields(
        self,
        canonical: LocalCompressionCanonicalInputs,
        denoiser_output: TargetDenoisingOutput | None,
    ) -> Any:
        torch = require_torch()
        fields: list[Any] = []
        if bool(self.config.include_canonical_features):
            fields.append(canonical.feature_rows())
        if self.config.uses_denoiser:
            if denoiser_output is None:
                raise ValueError("denoiser variant requires denoiser output")
            if bool(self.config.include_denoiser_deltas):
                fields.append(denoiser_output.deltas)
            if bool(self.config.include_denoiser_log_variances):
                fields.append(denoiser_output.log_variances)
            if bool(self.config.include_denoiser_reliability):
                fields.append(denoiser_output.reliability[:, :, None])
        if not fields:
            raise ValueError("no conditioning fields were enabled")
        conditioning = torch.cat([field.to(device=canonical.features.device, dtype=canonical.features.dtype) for field in fields], dim=-1)
        return torch.where(canonical.particle_mask[:, :, None], _nan_to_num_torch(conditioning), torch.zeros_like(conditioning))

    def _embed_injection_hook(self, module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        del module, inputs
        if self._pending_delta_h is None:
            return output
        delta = self._pending_delta_h
        output_tensor = output[0] if isinstance(output, tuple) else output
        if isinstance(output, tuple):
            first = output[0] + _delta_for_embed_output(delta, output[0])
            output = (first, *output[1:])
            output_tensor_for_diag = first
        else:
            residual = _delta_for_embed_output(delta, output)
            output = output + residual
            output_tensor_for_diag = output
        self._last_part_embedding = _batch_first_from_embed_output(
            output_tensor_for_diag,
            batch=int(delta.shape[0]),
            particles=int(delta.shape[1]),
            dim=int(self.config.part_embed_dim),
        )
        self._last_injection_summary = {
            "embed_module_name": str(self.embed_module_name),
            "delta_h_shape": list(delta.shape),
            "embed_output_shape": list(output_tensor.shape),
            "delta_h_abs_max": float(delta.detach().abs().max().cpu().item()) if int(delta.numel()) else 0.0,
            "injection_applied": True,
            "adapter_kind": "target_denoising_embedding_delta_h",
        }
        return output

    def _part_forward_with_delta(self, canonical: LocalCompressionCanonicalInputs, delta_h: Any | None) -> Any:
        self._last_part_embedding = None
        self._pending_delta_h = delta_h
        self._last_injection_summary = {
            "embed_module_name": None if self.embed_module_name is None else str(self.embed_module_name),
            "delta_h_shape": None if delta_h is None else list(delta_h.shape),
            "delta_h_abs_max": 0.0
            if delta_h is None or not int(delta_h.numel())
            else float(delta_h.detach().abs().max().cpu().item()),
            "injection_applied": False,
            "adapter_kind": "none" if delta_h is None else "target_denoising_embedding_delta_h",
        }
        handle = None
        if delta_h is not None:
            if self.embed_module is None:
                raise ValueError("embedding adapter is active but no embed module was resolved")
            handle = self.embed_module.register_forward_hook(self._embed_injection_hook)
        try:
            return self.part_model(canonical.points, canonical.features, canonical.lorentz_vectors, canonical.mask)
        finally:
            if handle is not None:
                handle.remove()
            self._pending_delta_h = None

    def forward_outputs(
        self,
        tokens_or_batch: Any,
        mask: Any | None = None,
        *,
        weights: Any | None = None,
        max_constits: int | None = None,
        weight_threshold: float | None = None,
        need_denoiser_weights: bool = False,
    ) -> TargetDenoisingAugmentedParTOutput:
        torch = require_torch()
        raw_tokens, raw_mask = _coerce_tokens_and_mask(tokens_or_batch, mask)
        canonical = self.build_canonical_inputs(
            raw_tokens,
            raw_mask,
            weights=weights,
            max_constits=max_constits,
            weight_threshold=weight_threshold,
        )
        denoiser_output = None
        if self.config.uses_denoiser:
            if self.denoiser is None:
                raise ValueError("denoiser variant requires a denoiser module")
            if bool(self.config.effective_freeze_denoiser):
                with torch.no_grad():
                    denoiser_output = self.denoiser(raw_tokens, raw_mask, need_weights=need_denoiser_weights)
            else:
                denoiser_output = self.denoiser(raw_tokens, raw_mask, need_weights=need_denoiser_weights)
        delta_h = None
        gate = None
        adapter_diagnostics: dict[str, Any] = {}
        if self.embedding_adapter is not None:
            conditioning = self._conditioning_fields(canonical, denoiser_output)
            delta_h, gate, adapter_diagnostics = self.embedding_adapter(conditioning, canonical.particle_mask)
            adapter_diagnostics.update(
                {
                    "conditioning_dim": int(conditioning.shape[-1]),
                    "conditioning_abs_mean": float(conditioning.detach().abs().mean().cpu().item())
                    if int(conditioning.numel())
                    else 0.0,
                }
            )
        logits = self._part_forward_with_delta(canonical, delta_h)
        return TargetDenoisingAugmentedParTOutput(
            logits=logits,
            canonical_inputs=canonical,
            config=self.config,
            denoiser_output=denoiser_output,
            delta_h=delta_h,
            gate=gate,
            injection_summary=dict(self._last_injection_summary),
            adapter_diagnostics=adapter_diagnostics,
            denoiser_checkpoint_report=self.denoiser_checkpoint_report,
        )

    def forward(
        self,
        tokens_or_batch: Any,
        mask: Any | None = None,
        *,
        return_outputs: bool = False,
        return_diagnostics: bool = False,
        **kwargs: Any,
    ) -> Any:
        output = self.forward_outputs(tokens_or_batch, mask, **kwargs)
        if return_outputs:
            return output
        if return_diagnostics:
            return output.logits, output.diagnostics()
        return output.logits


def load_target_denoising_pretrained_checkpoint(
    checkpoint_path: str | Path,
    model_or_denoiser: TargetDenoisingAugmentedParT | TargetConditionedPairwiseDenoiser,
    *,
    map_location: Any = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a Step 3 denoiser checkpoint into a Step 4 wrapper or denoiser."""

    torch = require_torch()
    payload = torch.load(str(checkpoint_path), map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("denoiser checkpoint payload must be a mapping")
    if payload.get("output_contract") != TARGET_DENOISING_TRAINING_CONTRACT:
        raise ValueError(
            f"expected denoiser checkpoint contract {TARGET_DENOISING_TRAINING_CONTRACT!r}, "
            f"got {payload.get('output_contract')!r}"
        )
    if payload.get("model_contract") != TARGET_DENOISING_MODEL_CONTRACT:
        raise ValueError(
            f"expected denoiser model contract {TARGET_DENOISING_MODEL_CONTRACT!r}, got {payload.get('model_contract')!r}"
        )
    target = model_or_denoiser.denoiser if isinstance(model_or_denoiser, TargetDenoisingAugmentedParT) else model_or_denoiser
    if target is None:
        raise ValueError("target wrapper has no denoiser module")
    incompatible = target.load_state_dict(payload["model_state_dict"], strict=bool(strict))
    report = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_contract": payload.get("output_contract"),
        "model_contract": payload.get("model_contract"),
        "strict": bool(strict),
        "missing_keys": list(getattr(incompatible, "missing_keys", []) or []),
        "unexpected_keys": list(getattr(incompatible, "unexpected_keys", []) or []),
        "metrics": dict(payload.get("metrics") or {}),
        "config": dict(payload.get("config") or {}),
        "model_config": dict(payload.get("model_config") or {}),
        "train_dataset": dict(payload.get("train_dataset") or {}),
        "model_val_dataset": dict(payload.get("model_val_dataset") or {}),
    }
    if isinstance(model_or_denoiser, TargetDenoisingAugmentedParT):
        model_or_denoiser.denoiser_checkpoint_report = dict(report)
    return report


__all__ = [
    "TARGET_DENOISING_DENOISER_VARIANTS",
    "TARGET_DENOISING_STEP4",
    "TARGET_DENOISING_TAGGER_CONTRACT",
    "TARGET_DENOISING_TAGGER_VARIANTS",
    "TARGET_DENOISING_VARIANT_DENOISER_FEATURES_FROZEN",
    "TARGET_DENOISING_VARIANT_DENOISER_FEATURES_JOINT",
    "TARGET_DENOISING_VARIANT_DENOISER_LOCAL_KERNEL_ONLY",
    "TARGET_DENOISING_VARIANT_DENOISER_NO_PAIR_BIAS",
    "TARGET_DENOISING_VARIANT_DENOISER_SHUFFLED_TARGETS",
    "TARGET_DENOISING_VARIANT_DENOISER_TAG_ONLY_SAME_ARCH",
    "TARGET_DENOISING_VARIANT_FEATURE_MLP_ADAPTER_TAG_ONLY",
    "TARGET_DENOISING_VARIANT_HLT_PART_BASELINE",
    "DenoisingEmbeddingResidualAdapter",
    "TargetDenoisingAugmentedParT",
    "TargetDenoisingAugmentedParTConfig",
    "TargetDenoisingAugmentedParTOutput",
    "load_target_denoising_pretrained_checkpoint",
]
