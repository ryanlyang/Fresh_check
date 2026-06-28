"""Baseline-faithful HLT ParT wrappers for multi-scale subjet experiments.

Step 8 is intentionally conservative: the baseline path must use the same
Weaver ``ParticleTransformer`` wrapper as existing HLT training.  Multi-scale
subjet modules can adapt canonical PF features before this backbone, but they
must not replace the HLT ParT baseline with a custom raw-token transformer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, build_hlt_classifier, require_torch

from .cross_attention import ParticleSubjetCrossAttentionOutput
from .features import CANONICAL_PART_FEATURE_NAMES, CanonicalPartInputs, build_canonical_part_inputs
from .protocol import MULTISCALE_SUBJET_BASELINE_VARIANT


try:  # Keep imports cheap when PyTorch is not available.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


MULTISCALE_SUBJET_REFERENCE_PART_BACKBONE_CONTRACT = "multiscale_subjet_reference_hlt_part_backbone_v1"
MULTISCALE_SUBJET_REFERENCE_PART_BACKBONE_STEP = "multiscale_subjet_part_step8_reference_hlt_part_backbone"
MULTISCALE_SUBJET_HLT_PART_BASELINE_CONTRACT = "multiscale_subjet_hlt_part_baseline_wrapper_v1"
MULTISCALE_SUBJET_HLT_PART_BASELINE_STEP = "multiscale_subjet_part_step8_hlt_part_baseline_wrapper"


@dataclass(frozen=True)
class MultiScaleSubjetReferencePartConfig:
    """Configuration for the real HLT ParT backbone used by this branch."""

    num_classes: int = 2
    model_size: str = "base"
    max_constits: int = 128
    weight_threshold: float = 0.0
    variant: str = MULTISCALE_SUBJET_BASELINE_VARIANT
    baseline_variant: str = MULTISCALE_SUBJET_BASELINE_VARIANT
    require_reference_part_backbone: bool = True

    def __post_init__(self) -> None:
        for name in ("num_classes", "max_constits"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        model_size = str(self.model_size)
        if model_size not in {"base", "tiny", "large"}:
            raise ValueError("model_size must be 'base', 'tiny', or 'large'")
        weight_threshold = float(self.weight_threshold)
        if weight_threshold < 0.0:
            raise ValueError("weight_threshold must be non-negative")
        baseline_variant = str(self.baseline_variant)
        if baseline_variant != MULTISCALE_SUBJET_BASELINE_VARIANT:
            raise ValueError("baseline_variant must be hlt_part_baseline")
        object.__setattr__(self, "model_size", model_size)
        object.__setattr__(self, "weight_threshold", weight_threshold)
        object.__setattr__(self, "variant", str(self.variant))
        object.__setattr__(self, "baseline_variant", baseline_variant)
        object.__setattr__(self, "require_reference_part_backbone", bool(self.require_reference_part_backbone))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["part_feature_names"] = list(CANONICAL_PART_FEATURE_NAMES)
        payload["uses_custom_raw_token_transformer"] = False
        return payload


@dataclass(frozen=True)
class MultiScaleSubjetPartBackboneOutput:
    """Output from the real HLT ParT backbone, optionally with adapted features."""

    logits: Any
    canonical_inputs: CanonicalPartInputs
    part_inputs: Mapping[str, Any]
    config: MultiScaleSubjetReferencePartConfig
    uses_reference_part_backbone: bool
    adapted_feature_rows: Any | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "step": MULTISCALE_SUBJET_REFERENCE_PART_BACKBONE_STEP,
            "contract": MULTISCALE_SUBJET_REFERENCE_PART_BACKBONE_CONTRACT,
            "variant": self.config.variant,
            "baseline_variant": self.config.baseline_variant,
            "uses_reference_part_backbone": bool(self.uses_reference_part_backbone),
            "uses_custom_raw_token_transformer": False,
            "serious_comparison_ready": bool(self.uses_reference_part_backbone),
            "logits_shape": list(self.logits.shape),
            "canonical_inputs": self.canonical_inputs.summary(),
            "part_inputs_shapes": {key: list(value.shape) for key, value in self.part_inputs.items()},
            "adapted_features": self.adapted_feature_rows is not None,
            "part_feature_names": list(CANONICAL_PART_FEATURE_NAMES),
        }

    def diagnostics(self) -> dict[str, Any]:
        torch = require_torch()
        part_mask = self.part_inputs["mask"].squeeze(1).to(dtype=self.logits.dtype)
        valid_counts = part_mask.sum(dim=1)
        diagnostics = {
            "step": MULTISCALE_SUBJET_REFERENCE_PART_BACKBONE_STEP,
            "contract": MULTISCALE_SUBJET_REFERENCE_PART_BACKBONE_CONTRACT,
            "variant": self.config.variant,
            "baseline_variant": self.config.baseline_variant,
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
            "uses_custom_raw_token_transformer": torch.zeros((), dtype=self.logits.dtype, device=self.logits.device),
            "serious_comparison_ready": torch.tensor(
                1.0 if bool(self.uses_reference_part_backbone) else 0.0,
                dtype=self.logits.dtype,
                device=self.logits.device,
            ),
        }
        if self.adapted_feature_rows is not None:
            canonical_rows = self.canonical_inputs.feature_rows().to(device=self.adapted_feature_rows.device)
            delta = self.adapted_feature_rows - canonical_rows
            diagnostics["adapted_feature_delta_norm_mean"] = (delta.norm(dim=-1) * part_mask).sum() / part_mask.sum().clamp_min(1.0)
        return diagnostics


@dataclass(frozen=True)
class MultiScaleSubjetHLTPartBaselineOutput:
    """Output wrapper for the exact raw-token HLT ParT baseline."""

    logits: Any
    backbone_output: MultiScaleSubjetPartBackboneOutput

    @property
    def part_inputs(self) -> Mapping[str, Any]:
        return self.backbone_output.part_inputs

    @property
    def config(self) -> MultiScaleSubjetReferencePartConfig:
        return self.backbone_output.config

    @property
    def uses_reference_part_backbone(self) -> bool:
        return bool(self.backbone_output.uses_reference_part_backbone)

    def summary(self) -> dict[str, Any]:
        payload = self.backbone_output.summary()
        payload["step"] = MULTISCALE_SUBJET_HLT_PART_BASELINE_STEP
        payload["contract"] = MULTISCALE_SUBJET_HLT_PART_BASELINE_CONTRACT
        payload["variant"] = MULTISCALE_SUBJET_BASELINE_VARIANT
        payload["baseline_variant"] = MULTISCALE_SUBJET_BASELINE_VARIANT
        return payload

    def diagnostics(self) -> dict[str, Any]:
        payload = self.backbone_output.diagnostics()
        payload["step"] = MULTISCALE_SUBJET_HLT_PART_BASELINE_STEP
        payload["contract"] = MULTISCALE_SUBJET_HLT_PART_BASELINE_CONTRACT
        payload["variant"] = MULTISCALE_SUBJET_BASELINE_VARIANT
        payload["baseline_variant"] = MULTISCALE_SUBJET_BASELINE_VARIANT
        return payload


def normalize_reference_part_config(
    config: MultiScaleSubjetReferencePartConfig | Mapping[str, Any] | None = None,
) -> MultiScaleSubjetReferencePartConfig:
    if config is None:
        return MultiScaleSubjetReferencePartConfig()
    if isinstance(config, MultiScaleSubjetReferencePartConfig):
        return config
    return MultiScaleSubjetReferencePartConfig(**dict(config))


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
    if int(mask.ndim) != 2 or tuple(mask.shape) != tuple(tokens.shape[:2]):
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match tokens shape {tuple(tokens.shape[:2])}")
    return tokens, mask


def _is_reference_part_backbone(part_model: Any) -> bool:
    return isinstance(part_model, ParticleTransformerHLTClassifier)


class MultiScaleSubjetReferencePartBackbone(_ModuleBase):
    """Thin wrapper around the real reference HLT Particle Transformer."""

    def __init__(
        self,
        config: MultiScaleSubjetReferencePartConfig | Mapping[str, Any] | None = None,
        *,
        part_model: Any | None = None,
    ) -> None:
        torch = require_torch()
        super().__init__()
        self.config = normalize_reference_part_config(config)
        self.part_model = part_model or build_hlt_classifier(
            num_classes=int(self.config.num_classes),
            model_size=str(self.config.model_size),
        )
        if bool(self.config.require_reference_part_backbone) and not self.uses_reference_part_backbone:
            raise ValueError(
                "Multi-scale subjet Step 8 requires the real ParticleTransformerHLTClassifier backbone. "
                "Do not substitute a custom raw-token transformer for hlt_part_baseline."
            )

    @property
    def uses_reference_part_backbone(self) -> bool:
        return _is_reference_part_backbone(self.part_model)

    @property
    def output_contract(self) -> str:
        return MULTISCALE_SUBJET_REFERENCE_PART_BACKBONE_CONTRACT

    def no_weight_decay(self) -> set[str]:
        names: set[str] = set()
        if hasattr(self.part_model, "no_weight_decay"):
            names.update({f"part_model.{name}" for name in self.part_model.no_weight_decay()})
        return names

    def to_config_dict(self) -> dict[str, Any]:
        payload = self.config.to_dict()
        payload["part_model_config"] = dict(getattr(self.part_model, "config", {}) or {})
        payload["uses_reference_part_backbone"] = bool(self.uses_reference_part_backbone)
        payload["contract"] = MULTISCALE_SUBJET_REFERENCE_PART_BACKBONE_CONTRACT
        return payload

    def _part_inputs_from_canonical(
        self,
        canonical_inputs: CanonicalPartInputs,
        *,
        adapted_feature_rows: Any | None = None,
    ) -> dict[str, Any]:
        torch = require_torch()
        part_inputs = canonical_inputs.as_part_kwargs()
        if adapted_feature_rows is not None:
            adapted = adapted_feature_rows.to(device=canonical_inputs.features.device, dtype=canonical_inputs.features.dtype)
            expected_shape = tuple(canonical_inputs.feature_rows().shape)
            if tuple(adapted.shape) != expected_shape:
                raise ValueError(f"adapted_feature_rows shape {tuple(adapted.shape)} does not match {expected_shape}")
            adapted = _nan_to_num_torch(adapted)
            particle_mask = canonical_inputs.mask.squeeze(1).to(device=adapted.device, dtype=torch.bool)
            adapted = torch.where(particle_mask[:, :, None], adapted, torch.zeros_like(adapted))
            part_inputs = dict(part_inputs)
            part_inputs["features"] = adapted.transpose(1, 2).contiguous()
        return dict(part_inputs)

    def forward_canonical(
        self,
        canonical_inputs: CanonicalPartInputs,
        *,
        adapted_feature_rows: Any | None = None,
    ) -> MultiScaleSubjetPartBackboneOutput:
        part_inputs = self._part_inputs_from_canonical(canonical_inputs, adapted_feature_rows=adapted_feature_rows)
        logits = self.part_model(
            part_inputs["points"],
            part_inputs["features"],
            part_inputs["lorentz_vectors"],
            part_inputs["mask"],
        )
        logits = _nan_to_num_torch(logits)
        return MultiScaleSubjetPartBackboneOutput(
            logits=logits,
            canonical_inputs=canonical_inputs,
            part_inputs=part_inputs,
            config=self.config,
            uses_reference_part_backbone=bool(self.uses_reference_part_backbone),
            adapted_feature_rows=adapted_feature_rows,
        )

    def forward_readback(self, readback_output: ParticleSubjetCrossAttentionOutput) -> MultiScaleSubjetPartBackboneOutput:
        return self.forward_canonical(
            readback_output.canonical_inputs,
            adapted_feature_rows=readback_output.adapted_features,
        )

    def forward(self, canonical_or_readback: Any, *, return_outputs: bool = False):
        if isinstance(canonical_or_readback, ParticleSubjetCrossAttentionOutput):
            output = self.forward_readback(canonical_or_readback)
        elif isinstance(canonical_or_readback, CanonicalPartInputs):
            output = self.forward_canonical(canonical_or_readback)
        else:
            raise TypeError("forward expects CanonicalPartInputs or ParticleSubjetCrossAttentionOutput")
        return output if bool(return_outputs) else output.logits


class MultiScaleSubjetHLTPartBaselineClassifier(_ModuleBase):
    """Exact HLT ParT baseline with a raw-token forward API."""

    def __init__(
        self,
        config: MultiScaleSubjetReferencePartConfig | Mapping[str, Any] | None = None,
        *,
        part_model: Any | None = None,
    ) -> None:
        super().__init__()
        resolved = normalize_reference_part_config(config)
        if resolved.variant != MULTISCALE_SUBJET_BASELINE_VARIANT:
            resolved = MultiScaleSubjetReferencePartConfig(
                num_classes=resolved.num_classes,
                model_size=resolved.model_size,
                max_constits=resolved.max_constits,
                weight_threshold=resolved.weight_threshold,
                variant=MULTISCALE_SUBJET_BASELINE_VARIANT,
                baseline_variant=resolved.baseline_variant,
                require_reference_part_backbone=resolved.require_reference_part_backbone,
            )
        self.config = resolved
        self.reference_part = MultiScaleSubjetReferencePartBackbone(self.config, part_model=part_model)

    @property
    def part_model(self) -> Any:
        return self.reference_part.part_model

    @property
    def uses_reference_part_backbone(self) -> bool:
        return bool(self.reference_part.uses_reference_part_backbone)

    @property
    def output_contract(self) -> str:
        return MULTISCALE_SUBJET_HLT_PART_BASELINE_CONTRACT

    def no_weight_decay(self) -> set[str]:
        return self.reference_part.no_weight_decay()

    def to_config_dict(self) -> dict[str, Any]:
        payload = self.reference_part.to_config_dict()
        payload["contract"] = MULTISCALE_SUBJET_HLT_PART_BASELINE_CONTRACT
        payload["step"] = MULTISCALE_SUBJET_HLT_PART_BASELINE_STEP
        payload["variant"] = MULTISCALE_SUBJET_BASELINE_VARIANT
        payload["baseline_variant"] = MULTISCALE_SUBJET_BASELINE_VARIANT
        return payload

    def forward_outputs(self, tokens_or_batch: Any, mask: Any | None = None) -> MultiScaleSubjetHLTPartBaselineOutput:
        raw_tokens, raw_mask = _coerce_tokens_and_mask(tokens_or_batch, mask)
        raw_tokens = _nan_to_num_torch(raw_tokens)
        raw_mask = raw_mask.to(device=raw_tokens.device, dtype=_torch.bool if _torch is not None else bool)
        canonical = build_canonical_part_inputs(
            raw_tokens,
            raw_mask,
            max_constits=int(self.config.max_constits),
            weight_threshold=float(self.config.weight_threshold),
        )
        backbone_output = self.reference_part.forward_canonical(canonical)
        return MultiScaleSubjetHLTPartBaselineOutput(logits=backbone_output.logits, backbone_output=backbone_output)

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


def build_multiscale_subjet_reference_part_backbone(
    config: MultiScaleSubjetReferencePartConfig | Mapping[str, Any] | None = None,
    *,
    part_model: Any | None = None,
    **kwargs: Any,
) -> MultiScaleSubjetReferencePartBackbone:
    if kwargs:
        payload = {} if config is None else (config.to_dict() if isinstance(config, MultiScaleSubjetReferencePartConfig) else dict(config))
        payload.update(kwargs)
        config = payload
    return MultiScaleSubjetReferencePartBackbone(config, part_model=part_model)


def build_multiscale_subjet_hlt_part_baseline(
    config: MultiScaleSubjetReferencePartConfig | Mapping[str, Any] | None = None,
    *,
    part_model: Any | None = None,
    **kwargs: Any,
) -> MultiScaleSubjetHLTPartBaselineClassifier:
    if kwargs:
        payload = {} if config is None else (config.to_dict() if isinstance(config, MultiScaleSubjetReferencePartConfig) else dict(config))
        payload.update(kwargs)
        config = payload
    return MultiScaleSubjetHLTPartBaselineClassifier(config, part_model=part_model)


MultiscaleSubjetReferencePartConfig = MultiScaleSubjetReferencePartConfig
MultiscaleSubjetPartBackboneOutput = MultiScaleSubjetPartBackboneOutput
MultiscaleSubjetHLTPartBaselineOutput = MultiScaleSubjetHLTPartBaselineOutput
MultiscaleSubjetReferencePartBackbone = MultiScaleSubjetReferencePartBackbone
MultiscaleSubjetHLTPartBaselineClassifier = MultiScaleSubjetHLTPartBaselineClassifier
