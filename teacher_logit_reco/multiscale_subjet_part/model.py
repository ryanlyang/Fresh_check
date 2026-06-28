"""Step 9 multi-scale subjet classifier variants.

This module is the first integrated classifier surface for the multi-scale
subjet plan.  The mainline model is a zero-initialized residual adapter into the
real HLT ParT input contract; branch-only and latent/random variants are kept
explicitly labelled as controls.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, build_hlt_classifier, require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from .assignment import (
    MULTISCALE_SUBJET_ASSIGNMENT_QUERY_LEARNED,
    MULTISCALE_SUBJET_ASSIGNMENT_QUERY_SEEDED,
    SoftSubjetAssignmentConfig,
)
from .cross_attention import (
    ParticleSubjetCrossAttentionConfig,
    ParticleSubjetCrossAttentionOutput,
    ParticleSubjetCrossAttentionReadback,
)
from .features import (
    CANONICAL_PART_FEATURE_NAMES,
    MULTISCALE_SUBJET_SCALE_PROFILES,
    CanonicalPartInputs,
    build_canonical_part_inputs,
    multiscale_subjet_scale_specs_for_profile,
)
from .part_wrapper import (
    MultiScaleSubjetHLTPartBaselineClassifier,
    MultiScaleSubjetReferencePartConfig,
    build_multiscale_subjet_hlt_part_baseline,
)
from .protocol import (
    MULTISCALE_SUBJET_BASELINE_VARIANT,
    MULTISCALE_SUBJET_DEFAULT_VARIANTS,
    MULTISCALE_SUBJET_PRIMARY_VARIANT,
)
from .subjet_transformer import (
    MultiScaleSubjetTransformer,
    MultiScaleSubjetTransformerConfig,
    MultiScaleSubjetTransformerOutput,
)
from .tokens import (
    MultiScaleSubjetTokenBuilder,
    MultiScaleSubjetTokenBuilderConfig,
    MultiScaleSubjetTokenBuilderOutput,
)


try:  # Keep package imports cheap on machines without PyTorch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


MULTISCALE_SUBJET_HLT_PART_BASELINE_STEP = "multiscale_subjet_part_step8_hlt_part_baseline_wrapper"
MULTISCALE_SUBJET_HLT_PART_BASELINE_CONTRACT = "multiscale_subjet_hlt_part_baseline_wrapper_v1"
MULTISCALE_SUBJET_CLASSIFIER_STEP = "multiscale_subjet_part_step9_classifier"
MULTISCALE_SUBJET_CLASSIFIER_CONTRACT = "multiscale_subjet_part_classifier_v1"

MULTISCALE_SUBJET_VARIANT_HLT_PART_BASELINE = MULTISCALE_SUBJET_BASELINE_VARIANT
MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER = MULTISCALE_SUBJET_PRIMARY_VARIANT
MULTISCALE_SUBJET_VARIANT_SUBJET_BRANCH_ONLY = "subjet_branch_only"
MULTISCALE_SUBJET_VARIANT_PURE_PERCEIVER_LATENT_CONTROL = "pure_perceiver_latent_control"
MULTISCALE_SUBJET_VARIANT_RANDOM_SUBJET_CONTROL = "part_plus_random_subjet_control"
MULTISCALE_SUBJET_VARIANT_LATE_FUSION = "part_plus_subjet_late_fusion"
MULTISCALE_SUBJET_VARIANT_CLS_FUSION = "part_plus_subjet_cls_fusion"
MULTISCALE_SUBJET_VARIANT_CROSS_ATTENTION_FUSION = "part_plus_subjet_cross_attention_fusion"
MULTISCALE_SUBJET_VARIANT_TWO_HLT_PART_ENSEMBLE = "two_hlt_part_ensemble_control"
MULTISCALE_SUBJET_CLASSIFIER_VARIANTS = (
    MULTISCALE_SUBJET_VARIANT_HLT_PART_BASELINE,
    MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER,
    MULTISCALE_SUBJET_VARIANT_SUBJET_BRANCH_ONLY,
    MULTISCALE_SUBJET_VARIANT_PURE_PERCEIVER_LATENT_CONTROL,
    MULTISCALE_SUBJET_VARIANT_RANDOM_SUBJET_CONTROL,
    MULTISCALE_SUBJET_VARIANT_LATE_FUSION,
    MULTISCALE_SUBJET_VARIANT_CLS_FUSION,
    MULTISCALE_SUBJET_VARIANT_CROSS_ATTENTION_FUSION,
    MULTISCALE_SUBJET_VARIANT_TWO_HLT_PART_ENSEMBLE,
)
MULTISCALE_SUBJET_PART_ANCHORED_VARIANTS = (
    MULTISCALE_SUBJET_VARIANT_HLT_PART_BASELINE,
    MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER,
    MULTISCALE_SUBJET_VARIANT_PURE_PERCEIVER_LATENT_CONTROL,
    MULTISCALE_SUBJET_VARIANT_RANDOM_SUBJET_CONTROL,
    MULTISCALE_SUBJET_VARIANT_LATE_FUSION,
    MULTISCALE_SUBJET_VARIANT_CLS_FUSION,
    MULTISCALE_SUBJET_VARIANT_CROSS_ATTENTION_FUSION,
    MULTISCALE_SUBJET_VARIANT_TWO_HLT_PART_ENSEMBLE,
)


def normalize_multiscale_subjet_variant(value: str) -> str:
    """Resolve Step 9 variant aliases without conflating controls."""

    clean = str(value).strip().lower().replace("-", "_")
    aliases = {
        "baseline": MULTISCALE_SUBJET_VARIANT_HLT_PART_BASELINE,
        "hlt_baseline": MULTISCALE_SUBJET_VARIANT_HLT_PART_BASELINE,
        "hlt_part": MULTISCALE_SUBJET_VARIANT_HLT_PART_BASELINE,
        "hlt_part_baseline": MULTISCALE_SUBJET_VARIANT_HLT_PART_BASELINE,
        MULTISCALE_SUBJET_VARIANT_HLT_PART_BASELINE: MULTISCALE_SUBJET_VARIANT_HLT_PART_BASELINE,
        "primary": MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER,
        "residual": MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER,
        "residual_adapter": MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER,
        "subjet_residual_part_adapter": MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER,
        MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER: MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER,
        "subjet_only": MULTISCALE_SUBJET_VARIANT_SUBJET_BRANCH_ONLY,
        "subjet_branch": MULTISCALE_SUBJET_VARIANT_SUBJET_BRANCH_ONLY,
        "subjet_branch_only": MULTISCALE_SUBJET_VARIANT_SUBJET_BRANCH_ONLY,
        MULTISCALE_SUBJET_VARIANT_SUBJET_BRANCH_ONLY: MULTISCALE_SUBJET_VARIANT_SUBJET_BRANCH_ONLY,
        "perceiver": MULTISCALE_SUBJET_VARIANT_PURE_PERCEIVER_LATENT_CONTROL,
        "pure_latent": MULTISCALE_SUBJET_VARIANT_PURE_PERCEIVER_LATENT_CONTROL,
        "pure_perceiver": MULTISCALE_SUBJET_VARIANT_PURE_PERCEIVER_LATENT_CONTROL,
        "pure_perceiver_latent_control": MULTISCALE_SUBJET_VARIANT_PURE_PERCEIVER_LATENT_CONTROL,
        MULTISCALE_SUBJET_VARIANT_PURE_PERCEIVER_LATENT_CONTROL: MULTISCALE_SUBJET_VARIANT_PURE_PERCEIVER_LATENT_CONTROL,
        "random_subjet": MULTISCALE_SUBJET_VARIANT_RANDOM_SUBJET_CONTROL,
        "random_subjet_control": MULTISCALE_SUBJET_VARIANT_RANDOM_SUBJET_CONTROL,
        "part_plus_random_subjet": MULTISCALE_SUBJET_VARIANT_RANDOM_SUBJET_CONTROL,
        "part_plus_random_subjet_control": MULTISCALE_SUBJET_VARIANT_RANDOM_SUBJET_CONTROL,
        MULTISCALE_SUBJET_VARIANT_RANDOM_SUBJET_CONTROL: MULTISCALE_SUBJET_VARIANT_RANDOM_SUBJET_CONTROL,
        "late_fusion": MULTISCALE_SUBJET_VARIANT_LATE_FUSION,
        "part_plus_subjet_late_fusion": MULTISCALE_SUBJET_VARIANT_LATE_FUSION,
        MULTISCALE_SUBJET_VARIANT_LATE_FUSION: MULTISCALE_SUBJET_VARIANT_LATE_FUSION,
        "cls_fusion": MULTISCALE_SUBJET_VARIANT_CLS_FUSION,
        "part_plus_subjet_cls_fusion": MULTISCALE_SUBJET_VARIANT_CLS_FUSION,
        MULTISCALE_SUBJET_VARIANT_CLS_FUSION: MULTISCALE_SUBJET_VARIANT_CLS_FUSION,
        "cross_attention_fusion": MULTISCALE_SUBJET_VARIANT_CROSS_ATTENTION_FUSION,
        "part_plus_subjet_cross_attention_fusion": MULTISCALE_SUBJET_VARIANT_CROSS_ATTENTION_FUSION,
        MULTISCALE_SUBJET_VARIANT_CROSS_ATTENTION_FUSION: MULTISCALE_SUBJET_VARIANT_CROSS_ATTENTION_FUSION,
        "two_part": MULTISCALE_SUBJET_VARIANT_TWO_HLT_PART_ENSEMBLE,
        "two_hlt_part": MULTISCALE_SUBJET_VARIANT_TWO_HLT_PART_ENSEMBLE,
        "two_hlt_part_ensemble": MULTISCALE_SUBJET_VARIANT_TWO_HLT_PART_ENSEMBLE,
        "two_hlt_part_ensemble_control": MULTISCALE_SUBJET_VARIANT_TWO_HLT_PART_ENSEMBLE,
        MULTISCALE_SUBJET_VARIANT_TWO_HLT_PART_ENSEMBLE: MULTISCALE_SUBJET_VARIANT_TWO_HLT_PART_ENSEMBLE,
    }
    if clean not in aliases:
        raise ValueError(f"multiscale subjet variant must be one of {MULTISCALE_SUBJET_CLASSIFIER_VARIANTS}, got {value!r}")
    return aliases[clean]


@dataclass(frozen=True)
class HLTPartBaselineRawTokenConfig:
    """Configuration for the exact HLT ParT baseline raw-token wrapper."""

    num_classes: int = 2
    model_size: str = "base"
    max_constits: int = 128
    weight_threshold: float = 0.0

    def __post_init__(self) -> None:
        for name in ("num_classes", "max_constits"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if str(self.model_size) not in {"base", "tiny", "large"}:
            raise ValueError("model_size must be 'base', 'tiny', or 'large'")
        weight_threshold = float(self.weight_threshold)
        if not math.isfinite(weight_threshold) or weight_threshold < 0.0:
            raise ValueError("weight_threshold must be non-negative and finite")
        object.__setattr__(self, "weight_threshold", weight_threshold)

    @property
    def variant(self) -> str:
        return MULTISCALE_SUBJET_VARIANT_HLT_PART_BASELINE

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["variant"] = self.variant
        payload["canonical_part_feature_names"] = list(CANONICAL_PART_FEATURE_NAMES)
        return payload


@dataclass(frozen=True)
class MultiScaleSubjetClassifierConfig:
    """Shared configuration for Step 9 classifier variants."""

    variant: str = MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER
    num_classes: int = 2
    model_size: str = "base"
    max_constits: int = 128
    weight_threshold: float = 0.0
    token_dim: int = 128
    token_hidden_dim: int = 256
    assignment_embed_dim: int = 64
    assignment_hidden_dim: int = 128
    assignment_temperature: float = 1.0
    assignment_geometry_bias_strength: float = 2.0
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ffn_dim: int = 256
    transformer_pair_bias_hidden_dim: int = 64
    readback_hidden_dim: int = 128
    readback_heads: int = 4
    readback_delta_hidden_dim: int = 256
    branch_hidden_dim: int = 256
    residual_gamma_init: float = 0.0
    dropout: float = 0.05
    attention_dropout: float = 0.05
    scale_profile: str = "default"
    use_assignment_scale_embedding: bool = True
    use_token_scale_embedding: bool = True
    use_subjet_pair_bias: bool = True
    use_scale_pair_embedding: bool = True
    random_control_seed: int = 2027

    def __post_init__(self) -> None:
        variant = normalize_multiscale_subjet_variant(self.variant)
        object.__setattr__(self, "variant", variant)
        for name in (
            "num_classes",
            "max_constits",
            "token_dim",
            "token_hidden_dim",
            "assignment_embed_dim",
            "assignment_hidden_dim",
            "transformer_heads",
            "transformer_ffn_dim",
            "transformer_pair_bias_hidden_dim",
            "readback_hidden_dim",
            "readback_heads",
            "readback_delta_hidden_dim",
            "branch_hidden_dim",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        transformer_layers = int(self.transformer_layers)
        if transformer_layers < 0:
            raise ValueError("transformer_layers must be non-negative")
        object.__setattr__(self, "transformer_layers", transformer_layers)
        if int(self.token_dim) % int(self.transformer_heads) != 0:
            raise ValueError("token_dim must be divisible by transformer_heads")
        if int(self.readback_hidden_dim) % int(self.readback_heads) != 0:
            raise ValueError("readback_hidden_dim must be divisible by readback_heads")
        if str(self.model_size) not in {"base", "tiny", "large"}:
            raise ValueError("model_size must be 'base', 'tiny', or 'large'")
        scale_profile = str(self.scale_profile).strip().lower().replace("-", "_")
        # Validate and canonicalize aliases through the shared scale-profile helper.
        specs = multiscale_subjet_scale_specs_for_profile(scale_profile)
        for candidate in MULTISCALE_SUBJET_SCALE_PROFILES:
            if specs == multiscale_subjet_scale_specs_for_profile(candidate):
                scale_profile = candidate
                break
        object.__setattr__(self, "scale_profile", scale_profile)
        for name in ("assignment_temperature", "assignment_geometry_bias_strength"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")
            if name == "assignment_temperature" and value <= 0.0:
                raise ValueError("assignment_temperature must be positive")
            object.__setattr__(self, name, value)
        for name in ("dropout", "attention_dropout"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0 or value >= 1.0:
                raise ValueError(f"{name} must be finite and satisfy 0 <= {name} < 1")
            object.__setattr__(self, name, value)
        weight_threshold = float(self.weight_threshold)
        if not math.isfinite(weight_threshold) or weight_threshold < 0.0:
            raise ValueError("weight_threshold must be non-negative and finite")
        residual_gamma_init = float(self.residual_gamma_init)
        if not math.isfinite(residual_gamma_init):
            raise ValueError("residual_gamma_init must be finite")
        object.__setattr__(self, "weight_threshold", weight_threshold)
        object.__setattr__(self, "residual_gamma_init", residual_gamma_init)
        object.__setattr__(self, "random_control_seed", int(self.random_control_seed))
        object.__setattr__(self, "use_assignment_scale_embedding", bool(self.use_assignment_scale_embedding))
        object.__setattr__(self, "use_token_scale_embedding", bool(self.use_token_scale_embedding))
        object.__setattr__(self, "use_subjet_pair_bias", bool(self.use_subjet_pair_bias))
        object.__setattr__(self, "use_scale_pair_embedding", bool(self.use_scale_pair_embedding))

    @property
    def query_mode(self) -> str:
        if self.variant == MULTISCALE_SUBJET_VARIANT_PURE_PERCEIVER_LATENT_CONTROL:
            return MULTISCALE_SUBJET_ASSIGNMENT_QUERY_LEARNED
        return MULTISCALE_SUBJET_ASSIGNMENT_QUERY_SEEDED

    @property
    def uses_part_anchor(self) -> bool:
        return self.variant in MULTISCALE_SUBJET_PART_ANCHORED_VARIANTS

    @property
    def is_baseline(self) -> bool:
        return self.variant == MULTISCALE_SUBJET_VARIANT_HLT_PART_BASELINE

    @property
    def is_branch_only(self) -> bool:
        return self.variant == MULTISCALE_SUBJET_VARIANT_SUBJET_BRANCH_ONLY

    @property
    def is_random_control(self) -> bool:
        return self.variant == MULTISCALE_SUBJET_VARIANT_RANDOM_SUBJET_CONTROL

    @property
    def is_primary(self) -> bool:
        return self.variant == MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER

    @property
    def is_late_fusion(self) -> bool:
        return self.variant == MULTISCALE_SUBJET_VARIANT_LATE_FUSION

    @property
    def is_cls_fusion(self) -> bool:
        return self.variant == MULTISCALE_SUBJET_VARIANT_CLS_FUSION

    @property
    def is_cross_attention_fusion(self) -> bool:
        return self.variant == MULTISCALE_SUBJET_VARIANT_CROSS_ATTENTION_FUSION

    @property
    def is_two_part_ensemble(self) -> bool:
        return self.variant == MULTISCALE_SUBJET_VARIANT_TWO_HLT_PART_ENSEMBLE

    @property
    def uses_subjet_stack(self) -> bool:
        return not (self.is_baseline or self.is_two_part_ensemble)

    @property
    def uses_readback(self) -> bool:
        return bool(
            self.uses_part_anchor
            and not self.is_baseline
            and not self.is_two_part_ensemble
            and not self.is_late_fusion
            and not self.is_cls_fusion
        )

    @property
    def uses_subjet_branch_logits(self) -> bool:
        return bool(self.is_branch_only or self.is_late_fusion or self.is_cls_fusion or self.is_cross_attention_fusion)

    @property
    def effective_use_subjet_pair_bias(self) -> bool:
        # The random-subjet control should test random latent content, not leak
        # the real learned assignment geometry through pairwise attention bias.
        return bool(self.use_subjet_pair_bias and not self.is_random_control)

    @property
    def baseline_recoverable_at_zero_gamma(self) -> bool:
        return bool(self.uses_part_anchor and not self.is_baseline and float(self.residual_gamma_init) == 0.0)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["query_mode"] = self.query_mode
        payload["uses_part_anchor"] = bool(self.uses_part_anchor)
        payload["default_required_variants"] = list(MULTISCALE_SUBJET_DEFAULT_VARIANTS)
        payload["canonical_part_feature_names"] = list(CANONICAL_PART_FEATURE_NAMES)
        payload["effective_use_subjet_pair_bias"] = bool(self.effective_use_subjet_pair_bias)
        payload["random_subjet_pair_bias_disabled"] = bool(self.is_random_control)
        payload["baseline_recoverable_at_zero_gamma"] = bool(self.baseline_recoverable_at_zero_gamma)
        payload["scale_specs"] = [spec.to_dict() for spec in multiscale_subjet_scale_specs_for_profile(self.scale_profile)]
        payload["uses_readback"] = bool(self.uses_readback)
        payload["uses_subjet_branch_logits"] = bool(self.uses_subjet_branch_logits)
        payload["uses_two_part_ensemble"] = bool(self.is_two_part_ensemble)
        return payload


@dataclass(frozen=True)
class HLTPartBaselineRawTokenOutput:
    """Output for the exact HLT ParT baseline raw-token wrapper."""

    logits: Any
    part_inputs: CanonicalPartInputs
    config: HLTPartBaselineRawTokenConfig
    uses_reference_part_backbone: bool

    def summary(self) -> dict[str, Any]:
        return {
            "step": MULTISCALE_SUBJET_HLT_PART_BASELINE_STEP,
            "contract": MULTISCALE_SUBJET_HLT_PART_BASELINE_CONTRACT,
            "variant": self.config.variant,
            "uses_reference_part_backbone": bool(self.uses_reference_part_backbone),
            "serious_comparison_ready": bool(self.uses_reference_part_backbone),
            "logits_shape": list(self.logits.shape),
            "part_inputs": self.part_inputs.summary(),
        }

    def diagnostics(self) -> dict[str, Any]:
        torch = require_torch()
        part_mask = self.part_inputs.mask.squeeze(1).to(dtype=self.logits.dtype)
        valid_counts = part_mask.sum(dim=1)
        return {
            "step": MULTISCALE_SUBJET_HLT_PART_BASELINE_STEP,
            "contract": MULTISCALE_SUBJET_HLT_PART_BASELINE_CONTRACT,
            "variant": self.config.variant,
            "batch_size": int(self.logits.shape[0]),
            "valid_particle_count_mean": valid_counts.mean(),
            "valid_particle_count_min": valid_counts.min(),
            "valid_particle_count_max": valid_counts.max(),
            "logit_abs_mean": self.logits.detach().abs().mean(),
            "uses_reference_part_backbone": torch.tensor(
                1.0 if bool(self.uses_reference_part_backbone) else 0.0,
                dtype=self.logits.dtype,
                device=self.logits.device,
            ),
            "serious_comparison_ready": torch.tensor(
                1.0 if bool(self.uses_reference_part_backbone) else 0.0,
                dtype=self.logits.dtype,
                device=self.logits.device,
            ),
        }


@dataclass(frozen=True)
class MultiScaleSubjetClassifierOutput:
    """Debug-rich output for one Step 9 classifier variant."""

    logits: Any
    config: MultiScaleSubjetClassifierConfig
    part_inputs: CanonicalPartInputs | None
    token_output: MultiScaleSubjetTokenBuilderOutput | None
    transformer_output: MultiScaleSubjetTransformerOutput | None
    readback_output: ParticleSubjetCrossAttentionOutput | None
    subjet_branch_attention: Any | None
    uses_reference_part_backbone: bool

    def summary(self) -> dict[str, Any]:
        return {
            "step": MULTISCALE_SUBJET_CLASSIFIER_STEP,
            "contract": MULTISCALE_SUBJET_CLASSIFIER_CONTRACT,
            "variant": self.config.variant,
            "logits_shape": list(self.logits.shape),
            "uses_reference_part_backbone": bool(self.uses_reference_part_backbone),
            "serious_comparison_ready": bool(self.uses_reference_part_backbone and self.config.uses_part_anchor),
            "baseline_recoverable_at_zero_gamma": bool(self.config.baseline_recoverable_at_zero_gamma),
            "subjet_branch_only": bool(self.config.is_branch_only),
            "pure_perceiver_latent_control": self.config.variant == MULTISCALE_SUBJET_VARIANT_PURE_PERCEIVER_LATENT_CONTROL,
            "random_subjet_token_control": bool(self.config.is_random_control),
            "random_subjet_pair_bias_disabled": bool(self.config.is_random_control),
            "query_mode": self.config.query_mode,
            "scale_profile": self.config.scale_profile,
            "uses_readback": bool(self.config.uses_readback),
            "uses_subjet_branch_logits": bool(self.config.uses_subjet_branch_logits),
            "uses_two_part_ensemble": bool(self.config.is_two_part_ensemble),
            "part_inputs": None if self.part_inputs is None else self.part_inputs.summary(),
            "token_builder": None if self.token_output is None else self.token_output.summary(),
            "subjet_transformer": None if self.transformer_output is None else self.transformer_output.summary(),
            "readback": None if self.readback_output is None else self.readback_output.summary(),
        }

    def diagnostics(self) -> dict[str, Any]:
        torch = require_torch()
        diagnostics: dict[str, Any] = {
            "step": MULTISCALE_SUBJET_CLASSIFIER_STEP,
            "contract": MULTISCALE_SUBJET_CLASSIFIER_CONTRACT,
            "variant": self.config.variant,
            "batch_size": int(self.logits.shape[0]),
            "logit_abs_mean": self.logits.detach().abs().mean(),
            "uses_reference_part_backbone": torch.tensor(
                1.0 if bool(self.uses_reference_part_backbone) else 0.0,
                dtype=self.logits.dtype,
                device=self.logits.device,
            ),
            "serious_comparison_ready": torch.tensor(
                1.0 if bool(self.uses_reference_part_backbone and self.config.uses_part_anchor) else 0.0,
                dtype=self.logits.dtype,
                device=self.logits.device,
            ),
        }
        if self.part_inputs is not None:
            part_mask = self.part_inputs.mask.squeeze(1).to(dtype=self.logits.dtype)
            valid_counts = part_mask.sum(dim=1)
            diagnostics["valid_particle_count_mean"] = valid_counts.mean()
            diagnostics["valid_particle_count_min"] = valid_counts.min()
            diagnostics["valid_particle_count_max"] = valid_counts.max()
        if self.token_output is not None:
            diagnostics["subjet_valid_fraction"] = torch.tensor(
                float(self.token_output.diagnostics.get("valid_subjet_fraction", 0.0)),
                dtype=self.logits.dtype,
                device=self.logits.device,
            )
            diagnostics["subjet_assignment_entropy_mean"] = torch.tensor(
                float(self.token_output.diagnostics.get("assignment_entropy_mean", 0.0) or 0.0),
                dtype=self.logits.dtype,
                device=self.logits.device,
            )
        if self.transformer_output is not None:
            diagnostics["subjet_pair_bias_abs_mean"] = torch.tensor(
                float(self.transformer_output.diagnostics.get("pair_bias_abs_mean") or 0.0),
                dtype=self.logits.dtype,
                device=self.logits.device,
            )
        if self.readback_output is not None:
            for key, value in self.readback_output.diagnostics.items():
                diagnostics[f"readback_{key}"] = (
                    torch.tensor(float(value), dtype=self.logits.dtype, device=self.logits.device)
                    if isinstance(value, (float, int, bool))
                    else value
                )
        return diagnostics


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, value.new_full((), float(nan)))


def _coerce_tokens_and_mask(tokens_or_batch: Any, mask: Any | None = None) -> tuple[Any, Any]:
    torch = require_torch()
    if isinstance(tokens_or_batch, Mapping):
        tokens = (
            tokens_or_batch.get("tokens")
            if "tokens" in tokens_or_batch
            else tokens_or_batch.get("hlt_tokens", tokens_or_batch.get("raw_tokens"))
        )
        if tokens is None:
            raise ValueError("input mapping must contain 'tokens', 'hlt_tokens', or 'raw_tokens'")
        if mask is None:
            mask = tokens_or_batch.get("mask", tokens_or_batch.get("hlt_mask"))
    else:
        tokens = tokens_or_batch
    if mask is None:
        raise ValueError("mask is required")
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.as_tensor(tokens, dtype=torch.float32)
    else:
        tokens = tokens.float()
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask, dtype=torch.bool, device=tokens.device)
    else:
        mask = mask.to(device=tokens.device, dtype=torch.bool)
    if int(tokens.ndim) != 3:
        raise ValueError(f"tokens must have shape [batch, particles, features], got {tuple(tokens.shape)}")
    if int(tokens.shape[-1]) != RAW_TOKEN_DIM:
        raise ValueError(f"tokens last dimension must be RAW_TOKEN_DIM={RAW_TOKEN_DIM}, got {int(tokens.shape[-1])}")
    if int(mask.ndim) != 2 or tuple(mask.shape) != tuple(tokens.shape[:2]):
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match tokens shape {tuple(tokens.shape[:2])}")
    return tokens, mask


class HLTPartBaselineRawTokenClassifier(_ModuleBase):
    """Exact HLT ParT baseline with a raw-token forward API."""

    def __init__(
        self,
        config: HLTPartBaselineRawTokenConfig | Mapping[str, Any] | None = None,
        *,
        part_model: Any | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = HLTPartBaselineRawTokenConfig()
        elif isinstance(config, Mapping):
            config = HLTPartBaselineRawTokenConfig(**dict(config))
        self.config = config
        self.part_model = part_model or build_hlt_classifier(
            num_classes=int(config.num_classes),
            model_size=str(config.model_size),
        )
        if not self.uses_reference_part_backbone:
            raise ValueError(
                "Multi-scale subjet HLT ParT baseline requires the real "
                "ParticleTransformerHLTClassifier backbone."
            )

    @property
    def uses_reference_part_backbone(self) -> bool:
        return isinstance(self.part_model, ParticleTransformerHLTClassifier)

    @property
    def output_contract(self) -> str:
        return MULTISCALE_SUBJET_HLT_PART_BASELINE_CONTRACT

    def no_weight_decay(self) -> set[str]:
        if hasattr(self.part_model, "no_weight_decay"):
            return {f"part_model.{name}" for name in self.part_model.no_weight_decay()}
        return set()

    def to_config_dict(self) -> dict[str, Any]:
        payload = self.config.to_dict()
        payload["part_model_config"] = dict(getattr(self.part_model, "config", {}) or {})
        payload["uses_reference_part_backbone"] = bool(self.uses_reference_part_backbone)
        return payload

    def forward_outputs(self, tokens_or_batch: Any, mask: Any | None = None) -> HLTPartBaselineRawTokenOutput:
        raw_tokens, raw_mask = _coerce_tokens_and_mask(tokens_or_batch, mask)
        raw_tokens = _nan_to_num_torch(raw_tokens)
        canonical = build_canonical_part_inputs(
            raw_tokens,
            raw_mask,
            max_constits=int(self.config.max_constits),
            weight_threshold=float(self.config.weight_threshold),
        )
        logits = self.part_model(
            canonical.points,
            canonical.features,
            canonical.lorentz_vectors,
            canonical.mask,
        )
        return HLTPartBaselineRawTokenOutput(
            logits=_nan_to_num_torch(logits),
            part_inputs=canonical,
            config=self.config,
            uses_reference_part_backbone=bool(self.uses_reference_part_backbone),
        )

    def forward(
        self,
        tokens_or_batch: Any,
        mask: Any | None = None,
        *,
        return_outputs: bool = False,
        return_diagnostics: bool = False,
    ):
        output = self.forward_outputs(tokens_or_batch, mask)
        if bool(return_outputs):
            return output
        if bool(return_diagnostics):
            return output.logits, output.diagnostics()
        return output.logits


class MultiScaleSubjetPartClassifier(_ModuleBase):
    """Step 9 multi-scale subjet classifier and controls."""

    def __init__(
        self,
        config: MultiScaleSubjetClassifierConfig | Mapping[str, Any] | None = None,
        *,
        part_model: Any | None = None,
        part_model_2: Any | None = None,
    ) -> None:
        torch = require_torch()
        super().__init__()
        if config is None:
            config = MultiScaleSubjetClassifierConfig()
        elif isinstance(config, Mapping):
            config = MultiScaleSubjetClassifierConfig(**dict(config))
        self.config = config
        self.part_model = None
        self.part_model_2 = None
        if bool(config.uses_part_anchor):
            self.part_model = part_model or build_hlt_classifier(
                num_classes=int(config.num_classes),
                model_size=str(config.model_size),
            )
            if not isinstance(self.part_model, ParticleTransformerHLTClassifier):
                raise ValueError(
                    "Multi-scale subjet part-anchored variants require the real "
                    "ParticleTransformerHLTClassifier backbone. Use the branch-only "
                    "control for non-ParT experiments."
                )
            if config.is_two_part_ensemble:
                self.part_model_2 = part_model_2 or (
                    copy.deepcopy(part_model)
                    if part_model is not None
                    else build_hlt_classifier(
                        num_classes=int(config.num_classes),
                        model_size=str(config.model_size),
                    )
                )
        if config.is_baseline:
            self.token_builder = None
            self.subjet_transformer = None
            self.readback = None
            self.random_control_tokens = None
            self.subjet_pool_query = None
            self.subjet_head = None
            self.fusion_gamma = None
            self.fusion_delta_head = None
            return
        if config.is_two_part_ensemble:
            self.token_builder = None
            self.subjet_transformer = None
            self.readback = None
            self.random_control_tokens = None
            self.subjet_pool_query = None
            self.subjet_head = None
            self.fusion_gamma = None
            self.fusion_delta_head = None
            return

        scale_specs = multiscale_subjet_scale_specs_for_profile(str(config.scale_profile))
        assignment_config = SoftSubjetAssignmentConfig(
            query_mode=config.query_mode,
            scale_specs=scale_specs,
            embed_dim=int(config.assignment_embed_dim),
            hidden_dim=int(config.assignment_hidden_dim),
            temperature=float(config.assignment_temperature),
            geometry_bias_strength=float(config.assignment_geometry_bias_strength),
            use_scale_embedding=bool(config.use_assignment_scale_embedding),
        )
        self.token_builder = MultiScaleSubjetTokenBuilder(
            MultiScaleSubjetTokenBuilderConfig(
                assignment_config=assignment_config,
                token_dim=int(config.token_dim),
                hidden_dim=int(config.token_hidden_dim),
                dropout=float(config.dropout),
                use_scale_embedding=bool(config.use_token_scale_embedding),
            )
        )
        self.subjet_transformer = MultiScaleSubjetTransformer(
            MultiScaleSubjetTransformerConfig(
                token_dim=int(config.token_dim),
                num_layers=int(config.transformer_layers),
                num_heads=int(config.transformer_heads),
                ffn_dim=int(config.transformer_ffn_dim),
                dropout=float(config.dropout),
                attention_dropout=float(config.attention_dropout),
                use_pairwise_bias=bool(config.effective_use_subjet_pair_bias),
                use_scale_pair_embedding=bool(config.use_scale_pair_embedding),
                pair_bias_hidden_dim=int(config.transformer_pair_bias_hidden_dim),
                num_scales=len(scale_specs),
            )
        )
        self.readback = None
        if bool(config.uses_readback):
            self.readback = ParticleSubjetCrossAttentionReadback(
                ParticleSubjetCrossAttentionConfig(
                    feature_dim=len(CANONICAL_PART_FEATURE_NAMES),
                    subjet_token_dim=int(config.token_dim),
                    hidden_dim=int(config.readback_hidden_dim),
                    num_heads=int(config.readback_heads),
                    delta_hidden_dim=int(config.readback_delta_hidden_dim),
                    dropout=float(config.dropout),
                    attention_dropout=float(config.attention_dropout),
                    residual_gamma_init=float(config.residual_gamma_init),
                    max_constits=int(config.max_constits),
                    weight_threshold=float(config.weight_threshold),
                )
            )
        self.random_control_tokens = None
        if config.is_random_control:
            self.random_control_tokens = torch.nn.Parameter(torch.empty(1, self.token_builder.config.total_num_subjets, int(config.token_dim)))
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(config.random_control_seed))
            with torch.no_grad():
                init = torch.randn(self.random_control_tokens.shape, generator=generator, dtype=self.random_control_tokens.dtype)
                self.random_control_tokens.copy_(0.02 * init.to(device=self.random_control_tokens.device))
        self.subjet_pool_query = None
        self.subjet_head = None
        self.fusion_gamma = None
        self.fusion_delta_head = None
        if config.uses_subjet_branch_logits:
            self.subjet_pool_query = torch.nn.Parameter(torch.empty(int(config.token_dim)))
            self.subjet_head = torch.nn.Sequential(
                torch.nn.LayerNorm(int(config.token_dim)),
                torch.nn.Linear(int(config.token_dim), int(config.branch_hidden_dim)),
                torch.nn.GELU(),
                torch.nn.Dropout(float(config.dropout)),
                torch.nn.Linear(int(config.branch_hidden_dim), int(config.num_classes)),
            )
            torch.nn.init.normal_(self.subjet_pool_query, mean=0.0, std=0.02)
        if config.is_late_fusion or config.is_cross_attention_fusion:
            self.fusion_gamma = torch.nn.Parameter(torch.tensor(float(config.residual_gamma_init), dtype=torch.float32))
        if config.is_cls_fusion:
            self.fusion_gamma = torch.nn.Parameter(torch.tensor(float(config.residual_gamma_init), dtype=torch.float32))
            self.fusion_delta_head = torch.nn.Sequential(
                torch.nn.LayerNorm(int(config.token_dim) + int(config.num_classes)),
                torch.nn.Linear(int(config.token_dim) + int(config.num_classes), int(config.branch_hidden_dim)),
                torch.nn.GELU(),
                torch.nn.Dropout(float(config.dropout)),
                torch.nn.Linear(int(config.branch_hidden_dim), int(config.num_classes)),
            )

    @property
    def uses_reference_part_backbone(self) -> bool:
        return isinstance(self.part_model, ParticleTransformerHLTClassifier)

    @property
    def output_contract(self) -> str:
        return MULTISCALE_SUBJET_HLT_PART_BASELINE_CONTRACT if self.config.is_baseline else MULTISCALE_SUBJET_CLASSIFIER_CONTRACT

    def no_weight_decay(self) -> set[str]:
        names: set[str] = set()
        if self.subjet_pool_query is not None:
            names.add("subjet_pool_query")
        if self.part_model is not None and hasattr(self.part_model, "no_weight_decay"):
            names.update({f"part_model.{name}" for name in self.part_model.no_weight_decay()})
        if self.part_model_2 is not None and hasattr(self.part_model_2, "no_weight_decay"):
            names.update({f"part_model_2.{name}" for name in self.part_model_2.no_weight_decay()})
        return names

    def to_config_dict(self) -> dict[str, Any]:
        payload = self.config.to_dict()
        payload["part_model_config"] = dict(getattr(self.part_model, "config", {}) or {}) if self.part_model is not None else None
        payload["part_model_2_config"] = dict(getattr(self.part_model_2, "config", {}) or {}) if self.part_model_2 is not None else None
        payload["uses_reference_part_backbone"] = bool(self.uses_reference_part_backbone)
        payload["serious_comparison_ready"] = bool(self.uses_reference_part_backbone and self.config.uses_part_anchor)
        payload["baseline_recoverable_at_zero_gamma"] = bool(self.config.baseline_recoverable_at_zero_gamma)
        payload["random_subjet_pair_bias_disabled"] = bool(self.config.is_random_control)
        return payload

    def _run_subjet_stack(self, raw_tokens: Any, raw_mask: Any) -> tuple[MultiScaleSubjetTokenBuilderOutput, MultiScaleSubjetTransformerOutput]:
        if self.token_builder is None or self.subjet_transformer is None:
            raise RuntimeError("subjet stack is not initialized")
        token_output = self.token_builder(raw_tokens, raw_mask, query_mode=self.config.query_mode)
        if self.config.is_random_control:
            if self.random_control_tokens is None:
                raise RuntimeError("random control tokens are not initialized")
            control_tokens = self.random_control_tokens.to(device=raw_tokens.device, dtype=token_output.subjet_tokens.dtype)
            control_tokens = control_tokens.expand(int(raw_tokens.shape[0]), -1, -1)
            control_tokens = control_tokens * token_output.subjet_mask[:, :, None].to(dtype=control_tokens.dtype)
            token_output = replace(token_output, subjet_tokens=control_tokens)
        transformer_output = self.subjet_transformer(token_output)
        return token_output, transformer_output

    def _subjet_branch_pool(self, transformer_output: MultiScaleSubjetTransformerOutput) -> tuple[Any, Any]:
        torch = require_torch()
        if self.subjet_pool_query is None:
            raise RuntimeError("subjet branch pooling is not initialized")
        tokens = transformer_output.subjet_tokens
        mask = transformer_output.subjet_mask.to(device=tokens.device, dtype=torch.bool)
        query = self.subjet_pool_query.to(device=tokens.device, dtype=tokens.dtype)
        scores = torch.einsum("bmd,d->bm", tokens, query) / math.sqrt(float(tokens.shape[-1]))
        scores = scores.masked_fill(~mask, -1.0e4)
        attention = torch.softmax(scores, dim=-1)
        attention = torch.where(mask, attention, torch.zeros_like(attention))
        attention = attention / torch.clamp(attention.sum(dim=-1, keepdim=True), min=1.0e-12)
        pooled = torch.einsum("bm,bmd->bd", attention, tokens)
        return pooled, attention

    def _subjet_branch_logits(self, transformer_output: MultiScaleSubjetTransformerOutput) -> tuple[Any, Any]:
        if self.subjet_head is None:
            raise RuntimeError("subjet branch head is not initialized")
        pooled, attention = self._subjet_branch_pool(transformer_output)
        return _nan_to_num_torch(self.subjet_head(pooled)), attention

    def forward_outputs(self, tokens_or_batch: Any, mask: Any | None = None) -> MultiScaleSubjetClassifierOutput | HLTPartBaselineRawTokenOutput:
        torch = require_torch()
        raw_tokens, raw_mask = _coerce_tokens_and_mask(tokens_or_batch, mask)
        raw_tokens = _nan_to_num_torch(raw_tokens)
        raw_mask = raw_mask.to(device=raw_tokens.device, dtype=torch.bool)
        if self.config.is_baseline:
            baseline = MultiScaleSubjetHLTPartBaselineClassifier(
                MultiScaleSubjetReferencePartConfig(
                    num_classes=int(self.config.num_classes),
                    model_size=str(self.config.model_size),
                    max_constits=int(self.config.max_constits),
                    weight_threshold=float(self.config.weight_threshold),
                    require_reference_part_backbone=True,
                ),
                part_model=self.part_model,
            )
            return baseline.forward_outputs(raw_tokens, raw_mask)
        if self.config.is_two_part_ensemble:
            if self.part_model is None or self.part_model_2 is None:
                raise RuntimeError("two-HLT-ParT ensemble is not initialized")
            canonical = build_canonical_part_inputs(
                raw_tokens,
                raw_mask,
                max_constits=int(self.config.max_constits),
                weight_threshold=float(self.config.weight_threshold),
            )
            kwargs = {
                "points": canonical.points,
                "features": canonical.features,
                "lorentz_vectors": canonical.lorentz_vectors,
                "mask": canonical.mask,
            }
            logits_1 = self.part_model(kwargs["points"], kwargs["features"], kwargs["lorentz_vectors"], kwargs["mask"])
            logits_2 = self.part_model_2(kwargs["points"], kwargs["features"], kwargs["lorentz_vectors"], kwargs["mask"])
            logits = 0.5 * (_nan_to_num_torch(logits_1) + _nan_to_num_torch(logits_2))
            return MultiScaleSubjetClassifierOutput(
                logits=_nan_to_num_torch(logits),
                config=self.config,
                part_inputs=canonical,
                token_output=None,
                transformer_output=None,
                readback_output=None,
                subjet_branch_attention=None,
                uses_reference_part_backbone=bool(self.uses_reference_part_backbone),
            )

        token_output, transformer_output = self._run_subjet_stack(raw_tokens, raw_mask)
        part_inputs = None
        readback_output = None
        branch_attention = None
        if self.config.is_branch_only:
            logits, branch_attention = self._subjet_branch_logits(transformer_output)
        elif self.config.is_late_fusion:
            if self.part_model is None or self.fusion_gamma is None:
                raise RuntimeError("late-fusion path is not initialized")
            canonical = build_canonical_part_inputs(
                raw_tokens,
                raw_mask,
                max_constits=int(self.config.max_constits),
                weight_threshold=float(self.config.weight_threshold),
            )
            part_inputs = canonical
            part_logits = self.part_model(canonical.points, canonical.features, canonical.lorentz_vectors, canonical.mask)
            branch_logits, branch_attention = self._subjet_branch_logits(transformer_output)
            gamma = self.fusion_gamma.to(device=part_logits.device, dtype=part_logits.dtype)
            logits = _nan_to_num_torch(part_logits) + gamma * _nan_to_num_torch(branch_logits)
        elif self.config.is_cls_fusion:
            if self.part_model is None or self.fusion_gamma is None or self.fusion_delta_head is None:
                raise RuntimeError("CLS-fusion path is not initialized")
            canonical = build_canonical_part_inputs(
                raw_tokens,
                raw_mask,
                max_constits=int(self.config.max_constits),
                weight_threshold=float(self.config.weight_threshold),
            )
            part_inputs = canonical
            part_logits = self.part_model(canonical.points, canonical.features, canonical.lorentz_vectors, canonical.mask)
            pooled, branch_attention = self._subjet_branch_pool(transformer_output)
            fusion_delta = self.fusion_delta_head(torch.cat([_nan_to_num_torch(part_logits), pooled], dim=-1))
            gamma = self.fusion_gamma.to(device=part_logits.device, dtype=part_logits.dtype)
            logits = _nan_to_num_torch(part_logits) + gamma * _nan_to_num_torch(fusion_delta)
        else:
            if self.readback is None or self.part_model is None:
                raise RuntimeError("residual ParT readback path is not initialized")
            readback_output = self.readback(raw_tokens, raw_mask, transformer_output)
            part_inputs = readback_output.canonical_inputs
            kwargs = readback_output.part_inputs()
            logits = self.part_model(
                kwargs["points"],
                kwargs["features"],
                kwargs["lorentz_vectors"],
                kwargs["mask"],
            )
            if self.config.is_cross_attention_fusion:
                if self.fusion_gamma is None:
                    raise RuntimeError("cross-attention branch fusion is not initialized")
                branch_logits, branch_attention = self._subjet_branch_logits(transformer_output)
                gamma = self.fusion_gamma.to(device=logits.device, dtype=logits.dtype)
                logits = _nan_to_num_torch(logits) + gamma * _nan_to_num_torch(branch_logits)
        return MultiScaleSubjetClassifierOutput(
            logits=_nan_to_num_torch(logits),
            config=self.config,
            part_inputs=part_inputs,
            token_output=token_output,
            transformer_output=transformer_output,
            readback_output=readback_output,
            subjet_branch_attention=branch_attention,
            uses_reference_part_backbone=bool(self.uses_reference_part_backbone),
        )

    def forward(
        self,
        tokens_or_batch: Any,
        mask: Any | None = None,
        *,
        return_outputs: bool = False,
        return_diagnostics: bool = False,
    ):
        output = self.forward_outputs(tokens_or_batch, mask)
        if bool(return_outputs):
            return output
        if bool(return_diagnostics):
            return output.logits, output.diagnostics()
        return output.logits


MultiscaleSubjetClassifierConfig = MultiScaleSubjetClassifierConfig
MultiscaleSubjetClassifierOutput = MultiScaleSubjetClassifierOutput
MultiscaleSubjetPartClassifier = MultiScaleSubjetPartClassifier


def build_hlt_part_baseline_raw_token_classifier(
    config: HLTPartBaselineRawTokenConfig | Mapping[str, Any] | None = None,
    *,
    part_model: Any | None = None,
    **kwargs: Any,
) -> HLTPartBaselineRawTokenClassifier:
    """Build the exact HLT ParT baseline for the multiscale subjet protocol."""

    if kwargs:
        payload = {} if config is None else (config.to_dict() if isinstance(config, HLTPartBaselineRawTokenConfig) else dict(config))
        payload.pop("variant", None)
        payload.pop("canonical_part_feature_names", None)
        payload.update(kwargs)
        config = payload
    return HLTPartBaselineRawTokenClassifier(config, part_model=part_model)


def build_multiscale_subjet_part_classifier(
    config: MultiScaleSubjetClassifierConfig | Mapping[str, Any] | None = None,
    *,
    part_model: Any | None = None,
    **kwargs: Any,
) -> MultiScaleSubjetPartClassifier:
    """Build one Step 9 classifier variant."""

    if kwargs:
        payload = {} if config is None else (config.to_dict() if isinstance(config, MultiScaleSubjetClassifierConfig) else dict(config))
        payload.pop("query_mode", None)
        payload.pop("uses_part_anchor", None)
        payload.pop("default_required_variants", None)
        payload.pop("canonical_part_feature_names", None)
        payload.update(kwargs)
        config = payload
    return MultiScaleSubjetPartClassifier(config, part_model=part_model)


def build_multiscale_subjet_comparison_classifier(
    variant: str,
    *,
    num_classes: int = 2,
    model_size: str = "base",
    max_constits: int = 128,
    token_dim: int = 128,
    dropout: float = 0.05,
    attention_dropout: float = 0.05,
    residual_gamma_init: float = 0.0,
    weight_threshold: float = 0.0,
    part_model: Any | None = None,
    **kwargs: Any,
):
    """Build a comparison model by protocol variant name."""

    variant = normalize_multiscale_subjet_variant(variant)
    if variant == MULTISCALE_SUBJET_VARIANT_HLT_PART_BASELINE:
        return build_multiscale_subjet_hlt_part_baseline(
            MultiScaleSubjetReferencePartConfig(
                num_classes=int(num_classes),
                model_size=str(model_size),
                max_constits=int(max_constits),
                weight_threshold=float(weight_threshold),
                require_reference_part_backbone=True,
            ),
            part_model=part_model,
        )
    payload = {
        "variant": variant,
        "num_classes": int(num_classes),
        "model_size": str(model_size),
        "max_constits": int(max_constits),
        "token_dim": int(token_dim),
        "dropout": float(dropout),
        "attention_dropout": float(attention_dropout),
        "residual_gamma_init": float(residual_gamma_init),
        "weight_threshold": float(weight_threshold),
    }
    payload.update(kwargs)
    return build_multiscale_subjet_part_classifier(payload, part_model=part_model)


__all__ = [
    "MULTISCALE_SUBJET_CLASSIFIER_CONTRACT",
    "MULTISCALE_SUBJET_CLASSIFIER_STEP",
    "MULTISCALE_SUBJET_CLASSIFIER_VARIANTS",
    "MULTISCALE_SUBJET_HLT_PART_BASELINE_CONTRACT",
    "MULTISCALE_SUBJET_HLT_PART_BASELINE_STEP",
    "MULTISCALE_SUBJET_PART_ANCHORED_VARIANTS",
    "MULTISCALE_SUBJET_VARIANT_HLT_PART_BASELINE",
    "MULTISCALE_SUBJET_VARIANT_CLS_FUSION",
    "MULTISCALE_SUBJET_VARIANT_CROSS_ATTENTION_FUSION",
    "MULTISCALE_SUBJET_VARIANT_LATE_FUSION",
    "MULTISCALE_SUBJET_VARIANT_PURE_PERCEIVER_LATENT_CONTROL",
    "MULTISCALE_SUBJET_VARIANT_RANDOM_SUBJET_CONTROL",
    "MULTISCALE_SUBJET_VARIANT_RESIDUAL_PART_ADAPTER",
    "MULTISCALE_SUBJET_VARIANT_SUBJET_BRANCH_ONLY",
    "MULTISCALE_SUBJET_VARIANT_TWO_HLT_PART_ENSEMBLE",
    "HLTPartBaselineRawTokenClassifier",
    "HLTPartBaselineRawTokenConfig",
    "HLTPartBaselineRawTokenOutput",
    "MultiScaleSubjetClassifierConfig",
    "MultiScaleSubjetClassifierOutput",
    "MultiScaleSubjetPartClassifier",
    "MultiscaleSubjetClassifierConfig",
    "MultiscaleSubjetClassifierOutput",
    "MultiscaleSubjetPartClassifier",
    "build_hlt_part_baseline_raw_token_classifier",
    "build_multiscale_subjet_comparison_classifier",
    "build_multiscale_subjet_part_classifier",
    "normalize_multiscale_subjet_variant",
]
