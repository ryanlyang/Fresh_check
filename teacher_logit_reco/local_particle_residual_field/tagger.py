"""Augmented Particle Transformer tagger for local residual-field features."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from jetclass_fresh.hlt_baseline import build_hlt_classifier, require_torch
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES

from .controls import (
    CONTROL_RESIDUAL_FIELD_SOURCES,
    LOCAL_RESIDUAL_FIELD_CONTROL_CONTRACT,
    RESIDUAL_FIELD_SOURCE_CROSS_JET_SHUFFLE,
    RESIDUAL_FIELD_SOURCE_LEARNED_NO_TARGET,
    RESIDUAL_FIELD_SOURCE_RADIUS_PERMUTED,
    RESIDUAL_FIELD_SOURCE_RANDOM,
    RESIDUAL_FIELD_SOURCE_WITHIN_JET_SHUFFLE,
    LocalResidualFieldControlConfig,
    LocalResidualFieldControlGenerator,
    apply_local_residual_field_control,
    normalize_residual_field_control_source,
)
from .model import (
    LOCAL_RESIDUAL_RECONSTRUCTOR_CONTRACT,
    LocalResidualFieldReconstructorConfig,
    LocalResidualFieldReconstructorOutput,
    build_local_residual_field_reconstructor,
)
from .train import _torch_load_checkpoint

try:  # Keep metadata/report imports cheap when PyTorch is unavailable.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


LOCAL_RESIDUAL_FIELD_TAGGER_CONTRACT = "local_particle_residual_field_augmented_part_v1"
LOCAL_RESIDUAL_FIELD_TAGGER_STEP = "local_particle_residual_field_step5_augmented_part_tagger"

RESIDUAL_FIELD_SOURCE_ZERO = "zero"
RESIDUAL_FIELD_SOURCE_HLT_ONLY = "hlt_only"
RESIDUAL_FIELD_SOURCE_ORACLE = "oracle"
RESIDUAL_FIELD_SOURCE_ORACLE_SCALED = "oracle_scaled"
RESIDUAL_FIELD_SOURCE_ORACLE_FIELD_SUBSET = "oracle_field_subset"
RESIDUAL_FIELD_SOURCE_ORACLE_NOISY = "oracle_noisy"
RESIDUAL_FIELD_SOURCE_ORACLE_DROPOUT = "oracle_dropout"
RESIDUAL_FIELD_SOURCE_FROZEN_RECONSTRUCTOR = "frozen_reconstructor"
RESIDUAL_FIELD_SOURCE_JOINT_RECONSTRUCTOR = "joint_reconstructor"
ORACLE_RESIDUAL_FIELD_SOURCES = (
    RESIDUAL_FIELD_SOURCE_ORACLE,
    RESIDUAL_FIELD_SOURCE_ORACLE_SCALED,
    RESIDUAL_FIELD_SOURCE_ORACLE_FIELD_SUBSET,
    RESIDUAL_FIELD_SOURCE_ORACLE_NOISY,
    RESIDUAL_FIELD_SOURCE_ORACLE_DROPOUT,
)
RESIDUAL_FIELD_SOURCES = (
    RESIDUAL_FIELD_SOURCE_HLT_ONLY,
    RESIDUAL_FIELD_SOURCE_ZERO,
    *ORACLE_RESIDUAL_FIELD_SOURCES,
    RESIDUAL_FIELD_SOURCE_FROZEN_RECONSTRUCTOR,
    RESIDUAL_FIELD_SOURCE_JOINT_RECONSTRUCTOR,
    *CONTROL_RESIDUAL_FIELD_SOURCES,
)


def normalize_residual_field_source(value: str) -> str:
    clean = str(value).strip().lower().replace("-", "_")
    aliases = {
        "": RESIDUAL_FIELD_SOURCE_HLT_ONLY,
        "none": RESIDUAL_FIELD_SOURCE_HLT_ONLY,
        "hlt": RESIDUAL_FIELD_SOURCE_HLT_ONLY,
        "hlt_only": RESIDUAL_FIELD_SOURCE_HLT_ONLY,
        "hlt_part": RESIDUAL_FIELD_SOURCE_HLT_ONLY,
        "part": RESIDUAL_FIELD_SOURCE_HLT_ONLY,
        "baseline": RESIDUAL_FIELD_SOURCE_HLT_ONLY,
        "blank": RESIDUAL_FIELD_SOURCE_ZERO,
        "zero": RESIDUAL_FIELD_SOURCE_ZERO,
        "zero_augmented": RESIDUAL_FIELD_SOURCE_ZERO,
        "oracle": RESIDUAL_FIELD_SOURCE_ORACLE,
        "target": RESIDUAL_FIELD_SOURCE_ORACLE,
        "targets": RESIDUAL_FIELD_SOURCE_ORACLE,
        "oracle_scaled": RESIDUAL_FIELD_SOURCE_ORACLE_SCALED,
        "scaled_oracle": RESIDUAL_FIELD_SOURCE_ORACLE_SCALED,
        "alpha_oracle": RESIDUAL_FIELD_SOURCE_ORACLE_SCALED,
        "oracle_field_subset": RESIDUAL_FIELD_SOURCE_ORACLE_FIELD_SUBSET,
        "oracle_subset": RESIDUAL_FIELD_SOURCE_ORACLE_FIELD_SUBSET,
        "subset_oracle": RESIDUAL_FIELD_SOURCE_ORACLE_FIELD_SUBSET,
        "oracle_noisy": RESIDUAL_FIELD_SOURCE_ORACLE_NOISY,
        "noisy_oracle": RESIDUAL_FIELD_SOURCE_ORACLE_NOISY,
        "oracle_noise": RESIDUAL_FIELD_SOURCE_ORACLE_NOISY,
        "oracle_dropout": RESIDUAL_FIELD_SOURCE_ORACLE_DROPOUT,
        "dropout_oracle": RESIDUAL_FIELD_SOURCE_ORACLE_DROPOUT,
        "frozen": RESIDUAL_FIELD_SOURCE_FROZEN_RECONSTRUCTOR,
        "frozen_reco": RESIDUAL_FIELD_SOURCE_FROZEN_RECONSTRUCTOR,
        "frozen_reconstructor": RESIDUAL_FIELD_SOURCE_FROZEN_RECONSTRUCTOR,
        "predicted": RESIDUAL_FIELD_SOURCE_FROZEN_RECONSTRUCTOR,
        "joint": RESIDUAL_FIELD_SOURCE_JOINT_RECONSTRUCTOR,
        "joint_reco": RESIDUAL_FIELD_SOURCE_JOINT_RECONSTRUCTOR,
        "joint_reconstructor": RESIDUAL_FIELD_SOURCE_JOINT_RECONSTRUCTOR,
    }
    if clean in aliases:
        return aliases[clean]
    try:
        return normalize_residual_field_control_source(clean)
    except ValueError as exc:
        raise ValueError(f"residual_field_source must be one of {RESIDUAL_FIELD_SOURCES}, got {value!r}")


@dataclass(frozen=True)
class LocalResidualFieldTaggerConfig:
    """Configuration for the augmented residual-field ParT wrapper."""

    num_classes: int = 10
    field_dim: int = 50
    base_feature_dim: int = len(PF_FEATURE_NAMES)
    hlt_anchored_field_normalization: bool = False
    field_source: str = RESIDUAL_FIELD_SOURCE_FROZEN_RECONSTRUCTOR
    model_size: str = "base"
    residual_field_scale: float = 1.0
    residual_field_clip_value: float = 8.0
    field_dropout: float = 0.0
    oracle_field_alpha: float = 1.0
    oracle_field_noise_std: float = 0.0
    oracle_field_dropout: float = 0.0
    oracle_field_group_dropout: float = 0.0
    field_names: Sequence[str] = field(default_factory=tuple)
    field_groups: Mapping[str, Sequence[int]] = field(default_factory=dict)
    source_field_indices: Sequence[int] = field(default_factory=tuple)
    reconstructor_config: LocalResidualFieldReconstructorConfig | Mapping[str, Any] | None = None
    control_config: LocalResidualFieldControlConfig | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        source = normalize_residual_field_source(self.field_source)
        object.__setattr__(self, "field_source", source)
        for name in ("num_classes", "base_feature_dim"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        field_dim = int(self.field_dim)
        if field_dim < 0 or (field_dim == 0 and source != RESIDUAL_FIELD_SOURCE_HLT_ONLY):
            raise ValueError("field_dim must be positive except for hlt_only baselines")
        object.__setattr__(self, "field_dim", field_dim)
        anchored_normalization = bool(self.hlt_anchored_field_normalization)
        if anchored_normalization and (
            source == RESIDUAL_FIELD_SOURCE_HLT_ONLY or field_dim <= 0
        ):
            raise ValueError(
                "hlt_anchored_field_normalization requires a positive residual-field input"
            )
        object.__setattr__(
            self, "hlt_anchored_field_normalization", anchored_normalization
        )
        scale = float(self.residual_field_scale)
        if scale < 0.0:
            raise ValueError("residual_field_scale must be non-negative")
        clip_value = float(self.residual_field_clip_value)
        if clip_value < 0.0:
            raise ValueError("residual_field_clip_value must be non-negative")
        dropout = float(self.field_dropout)
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("field_dropout must be in [0, 1)")
        object.__setattr__(self, "residual_field_scale", scale)
        object.__setattr__(self, "residual_field_clip_value", clip_value)
        object.__setattr__(self, "field_dropout", dropout)
        alpha = float(self.oracle_field_alpha)
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError("oracle_field_alpha must be finite and non-negative")
        noise_std = float(self.oracle_field_noise_std)
        if not math.isfinite(noise_std) or noise_std < 0.0:
            raise ValueError("oracle_field_noise_std must be finite and non-negative")
        oracle_dropout = float(self.oracle_field_dropout)
        if not math.isfinite(oracle_dropout) or oracle_dropout < 0.0 or oracle_dropout >= 1.0:
            raise ValueError("oracle_field_dropout must be finite and in [0, 1)")
        group_dropout = float(self.oracle_field_group_dropout)
        if not math.isfinite(group_dropout) or group_dropout < 0.0 or group_dropout >= 1.0:
            raise ValueError("oracle_field_group_dropout must be finite and in [0, 1)")
        if source not in {
            RESIDUAL_FIELD_SOURCE_ORACLE_NOISY,
            RESIDUAL_FIELD_SOURCE_ORACLE_DROPOUT,
        } and any(value > 0.0 for value in (noise_std, oracle_dropout, group_dropout)):
            raise ValueError(
                f"field_source={source!r} does not apply oracle corruption; use "
                "oracle_noisy or oracle_dropout"
            )
        if source == RESIDUAL_FIELD_SOURCE_ORACLE_DROPOUT and noise_std > 0.0:
            raise ValueError("oracle_dropout does not apply Gaussian noise; use oracle_noisy")
        if source == RESIDUAL_FIELD_SOURCE_ORACLE_DROPOUT and oracle_dropout <= 0.0 and group_dropout <= 0.0:
            raise ValueError("oracle_dropout requires oracle_field_dropout or oracle_field_group_dropout")
        if source == RESIDUAL_FIELD_SOURCE_ORACLE_NOISY and noise_std <= 0.0:
            raise ValueError("oracle_noisy requires a positive oracle_field_noise_std")
        object.__setattr__(self, "oracle_field_alpha", alpha)
        object.__setattr__(self, "oracle_field_noise_std", noise_std)
        object.__setattr__(self, "oracle_field_dropout", oracle_dropout)
        object.__setattr__(self, "oracle_field_group_dropout", group_dropout)
        names = tuple(str(name) for name in self.field_names)
        if names and len(names) != int(self.field_dim):
            raise ValueError("field_names length must match field_dim")
        object.__setattr__(self, "field_names", names)
        groups = {
            str(group): tuple(int(index) for index in indices)
            for group, indices in dict(self.field_groups or {}).items()
        }
        for group, indices in groups.items():
            for index in indices:
                if index < 0 or index >= int(self.field_dim):
                    raise ValueError(f"field group {group!r} index {index} outside field_dim={self.field_dim}")
        object.__setattr__(self, "field_groups", groups)
        source_indices = tuple(int(index) for index in self.source_field_indices)
        if source_indices and len(source_indices) != int(self.field_dim):
            raise ValueError("source_field_indices length must match field_dim")
        if len(set(source_indices)) != len(source_indices):
            raise ValueError("source_field_indices must be unique")
        for index in source_indices:
            if index < 0:
                raise ValueError("source_field_indices must be non-negative")
        object.__setattr__(self, "source_field_indices", source_indices)
        if self.control_config is not None and not isinstance(self.control_config, LocalResidualFieldControlConfig):
            control_payload = dict(self.control_config)
            control_payload.pop("contract", None)
            control_payload.setdefault("field_names", names)
            object.__setattr__(self, "control_config", LocalResidualFieldControlConfig(**control_payload))

    @property
    def augmented_feature_dim(self) -> int:
        if self.field_source == RESIDUAL_FIELD_SOURCE_HLT_ONLY:
            return int(self.base_feature_dim)
        return int(self.base_feature_dim) + int(self.field_dim)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract"] = LOCAL_RESIDUAL_FIELD_TAGGER_CONTRACT
        payload["augmented_feature_dim"] = int(self.augmented_feature_dim)
        payload["field_names"] = list(self.field_names)
        payload["field_groups"] = {
            str(group): [int(index) for index in indices]
            for group, indices in dict(self.field_groups).items()
        }
        payload["source_field_indices"] = [int(index) for index in self.source_field_indices]
        if isinstance(self.reconstructor_config, LocalResidualFieldReconstructorConfig):
            payload["reconstructor_config"] = self.reconstructor_config.to_dict()
        if isinstance(self.control_config, LocalResidualFieldControlConfig):
            payload["control_config"] = self.control_config.to_dict()
        return payload


@dataclass(frozen=True)
class LocalResidualFieldTaggerOutput:
    """Forward output from the augmented residual-field ParT."""

    logits: Any
    augmented_features: Any
    residual_features: Any
    residual_fields: Any
    field_mask: Any
    diagnostics: Mapping[str, Any]
    reconstructor_output: LocalResidualFieldReconstructorOutput | None = None
    control_diagnostics: Mapping[str, Any] | None = None


class HLTAnchoredSplitLayerNorm(_ModuleBase):
    """Normalize HLT and residual-field channels without perturbing the HLT path.

    Weaver's particle embedder applies LayerNorm over its complete input vector
    before the widened input projection.  A conventional ``LayerNorm(67)``
    therefore changes the original 17 HLT activations merely because 50 field
    channels were appended, even when the new projection columns are zero.
    This module retains one state-dict-compatible affine vector while
    normalizing the two channel blocks independently.
    """

    def __init__(
        self,
        base_feature_dim: int,
        field_dim: int,
        *,
        eps: float = 1.0e-5,
        elementwise_affine: bool = True,
        device: Any | None = None,
        dtype: Any | None = None,
    ) -> None:
        torch = require_torch()
        super().__init__()
        self.base_feature_dim = int(base_feature_dim)
        self.field_dim = int(field_dim)
        self.normalized_shape = (self.base_feature_dim + self.field_dim,)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.base_feature_dim <= 0 or self.field_dim <= 0:
            raise ValueError("split LayerNorm channel dimensions must be positive")
        if self.elementwise_affine:
            factory_kwargs = {"device": device, "dtype": dtype}
            self.weight = torch.nn.Parameter(
                torch.ones(self.normalized_shape, **factory_kwargs)
            )
            self.bias = torch.nn.Parameter(
                torch.zeros(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, value: Any) -> Any:
        torch = require_torch()
        if int(value.shape[-1]) != int(self.normalized_shape[0]):
            raise ValueError(
                f"anchored LayerNorm expected {self.normalized_shape[0]} channels, "
                f"got {value.shape[-1]}"
            )
        base = value[..., : self.base_feature_dim]
        fields = value[..., self.base_feature_dim :]
        base_weight = (
            None if self.weight is None else self.weight[: self.base_feature_dim]
        )
        base_bias = None if self.bias is None else self.bias[: self.base_feature_dim]
        field_weight = (
            None if self.weight is None else self.weight[self.base_feature_dim :]
        )
        field_bias = (
            None if self.bias is None else self.bias[self.base_feature_dim :]
        )
        base = torch.nn.functional.layer_norm(
            base,
            (self.base_feature_dim,),
            base_weight,
            base_bias,
            self.eps,
        )
        fields = torch.nn.functional.layer_norm(
            fields,
            (self.field_dim,),
            field_weight,
            field_bias,
            self.eps,
        )
        return torch.cat((base, fields), dim=-1)


def install_hlt_anchored_field_normalization(
    part_model: Any,
    *,
    base_feature_dim: int,
    field_dim: int,
) -> dict[str, Any]:
    """Replace only Weaver's widened input LayerNorm with a split normalizer."""

    torch = require_torch()
    core = getattr(part_model, "mod", part_model)
    embedder = getattr(core, "embed", None)
    embed_stack = getattr(embedder, "embed", None)
    if not isinstance(embed_stack, torch.nn.Sequential) or len(embed_stack) < 2:
        raise TypeError(
            "HLT-anchored field normalization requires Weaver's sequential "
            "particle input embedder"
        )
    original = embed_stack[0]
    if not isinstance(original, torch.nn.LayerNorm):
        raise TypeError(
            "HLT-anchored field normalization expected LayerNorm as the first "
            "particle embedding operation"
        )
    total = int(base_feature_dim) + int(field_dim)
    if tuple(int(value) for value in original.normalized_shape) != (total,):
        raise ValueError(
            "particle input LayerNorm shape does not match base plus field dimensions"
        )
    parameter = next(original.parameters(), None)
    replacement = HLTAnchoredSplitLayerNorm(
        int(base_feature_dim),
        int(field_dim),
        eps=float(original.eps),
        elementwise_affine=bool(original.elementwise_affine),
        device=None if parameter is None else parameter.device,
        dtype=None if parameter is None else parameter.dtype,
    )
    with torch.no_grad():
        if original.elementwise_affine:
            replacement.weight.copy_(original.weight)
            replacement.bias.copy_(original.bias)
    embed_stack[0] = replacement
    return {
        "installed": True,
        "base_feature_dim": int(base_feature_dim),
        "field_dim": int(field_dim),
        "normalized_shape": [total],
        "input_projection_index": 1,
        "hlt_and_field_blocks_normalized_independently": True,
    }


def _coerce_reconstructor_config(
    config: LocalResidualFieldTaggerConfig,
) -> LocalResidualFieldReconstructorConfig:
    if isinstance(config.reconstructor_config, LocalResidualFieldReconstructorConfig):
        return config.reconstructor_config
    payload = dict(config.reconstructor_config or {})
    payload.pop("contract", None)
    payload.setdefault("field_dim", int(config.field_dim))
    payload.setdefault("field_names", tuple(config.field_names))
    payload.setdefault("field_groups", dict(config.field_groups))
    return LocalResidualFieldReconstructorConfig(**payload)


def _masked_field_stats(fields: Any, mask: Any) -> dict[str, Any]:
    if mask.ndim == 3:
        mask = mask.squeeze(1)
    valid = mask.bool()
    if not bool(valid.any().detach().cpu().item()):
        return {
            "residual_abs_mean": 0.0,
            "residual_l2_mean": 0.0,
            "valid_particles": 0,
        }
    selected = fields[valid]
    return {
        "residual_abs_mean": float(selected.detach().abs().mean().cpu().item()),
        "residual_l2_mean": float((selected.detach().square().sum(dim=-1) + 1.0e-12).sqrt().mean().cpu().item()),
        "valid_particles": int(valid.detach().sum().cpu().item()),
    }


class LocalResidualFieldAugmentedParT(_ModuleBase):
    """ParT classifier that receives HLT particle features plus residual fields."""

    def __init__(
        self,
        config: LocalResidualFieldTaggerConfig | Mapping[str, Any] | None = None,
        *,
        part_model: Any | None = None,
        reconstructor: Any | None = None,
    ) -> None:
        torch = require_torch()
        super().__init__()
        self.config = config if isinstance(config, LocalResidualFieldTaggerConfig) else LocalResidualFieldTaggerConfig(**dict(config or {}))
        self.part_model = part_model or build_hlt_classifier(
            num_classes=int(self.config.num_classes),
            model_size=str(self.config.model_size),
            overrides={"input_dim": int(self.config.augmented_feature_dim)},
        )
        self.hlt_anchored_normalization_report = None
        if self.config.hlt_anchored_field_normalization:
            self.hlt_anchored_normalization_report = (
                install_hlt_anchored_field_normalization(
                    self.part_model,
                    base_feature_dim=int(self.config.base_feature_dim),
                    field_dim=int(self.config.field_dim),
                )
            )
        self.field_dropout = torch.nn.Dropout(float(self.config.field_dropout))
        if reconstructor is not None:
            self.reconstructor = reconstructor
        elif self.config.field_source in {
            RESIDUAL_FIELD_SOURCE_FROZEN_RECONSTRUCTOR,
            RESIDUAL_FIELD_SOURCE_JOINT_RECONSTRUCTOR,
        }:
            self.reconstructor = build_local_residual_field_reconstructor(_coerce_reconstructor_config(self.config))
        else:
            self.reconstructor = None
        if self.config.field_source == RESIDUAL_FIELD_SOURCE_LEARNED_NO_TARGET:
            control_config = self.config.control_config
            if not isinstance(control_config, LocalResidualFieldControlConfig):
                control_config = LocalResidualFieldControlConfig(field_names=tuple(self.config.field_names))
            self.control_generator = LocalResidualFieldControlGenerator(
                field_dim=int(self.config.field_dim),
                hidden_dim=int(control_config.learned_hidden_dim),
                dropout=float(control_config.learned_dropout),
            )
        else:
            self.control_generator = None
        if self.config.field_source == RESIDUAL_FIELD_SOURCE_FROZEN_RECONSTRUCTOR:
            self.set_reconstructor_trainable(False)

    def _select_source_fields(self, fields: Any) -> Any:
        indices = tuple(int(index) for index in self.config.source_field_indices)
        if not indices:
            return fields
        torch = require_torch()
        index_tensor = torch.as_tensor(indices, device=fields.device, dtype=torch.long)
        if int(index_tensor.max().detach().cpu().item()) >= int(fields.shape[-1]):
            raise ValueError(
                f"source_field_indices max {int(index_tensor.max().detach().cpu().item())} "
                f"is outside source residual field dim {int(fields.shape[-1])}"
            )
        return fields.index_select(dim=-1, index=index_tensor)

    def _select_reconstructor_output(
        self,
        output: LocalResidualFieldReconstructorOutput,
    ) -> LocalResidualFieldReconstructorOutput:
        selected = self._select_source_fields(output.predicted_fields)
        if selected is output.predicted_fields:
            return output
        log_sigma = None if output.log_sigma is None else self._select_source_fields(output.log_sigma)
        return LocalResidualFieldReconstructorOutput(
            predicted_fields=selected,
            field_mask=output.field_mask,
            hidden=output.hidden,
            diagnostics=dict(output.diagnostics),
            log_sigma=log_sigma,
        )

    def _field_valid_mask(self, *, fields: Any, mask: Any, raw_mask: Any | None) -> Any:
        torch = require_torch()
        field_mask = raw_mask if raw_mask is not None else mask
        if field_mask.ndim == 3:
            field_mask = field_mask.squeeze(1)
        field_mask = field_mask.to(device=fields.device, dtype=torch.bool)
        if field_mask.shape != fields.shape[:2]:
            raise ValueError(
                f"residual-field mask shape {tuple(field_mask.shape)} is incompatible with "
                f"fields shape {tuple(fields.shape)}"
            )
        return field_mask[:, :, None].to(dtype=fields.dtype)

    def _sanitize_residual_fields(self, fields: Any, valid_mask: Any) -> Any:
        torch = require_torch()
        clip_value = float(self.config.residual_field_clip_value)
        fields = fields * valid_mask
        if clip_value > 0.0:
            fields = torch.nan_to_num(fields, nan=0.0, posinf=clip_value, neginf=-clip_value)
            fields = fields.clamp(min=-clip_value, max=clip_value)
        else:
            fields = torch.nan_to_num(fields, nan=0.0, posinf=0.0, neginf=0.0)
        return fields * valid_mask

    def _transform_oracle_fields(self, fields: Any, *, mask: Any, raw_mask: Any | None) -> tuple[Any, Mapping[str, Any]]:
        torch = require_torch()
        fields = self._select_source_fields(fields)
        valid_mask = self._field_valid_mask(fields=fields, mask=mask, raw_mask=raw_mask)
        valid_values = valid_mask.expand_as(fields).to(dtype=torch.bool)
        source = self.config.field_source
        alpha = float(self.config.oracle_field_alpha)
        transformed = fields * alpha
        noise_std = float(self.config.oracle_field_noise_std)
        field_dropout = float(self.config.oracle_field_dropout)
        group_dropout = float(self.config.oracle_field_group_dropout)
        corruption_active = bool(self.training and source in {
            RESIDUAL_FIELD_SOURCE_ORACLE_NOISY,
            RESIDUAL_FIELD_SOURCE_ORACLE_DROPOUT,
        })
        input_nonfinite = (~torch.isfinite(fields)) & valid_values
        element_drop_mask = torch.zeros_like(valid_values)
        group_drop_events = 0
        group_draws = 0
        if bool(corruption_active) and noise_std > 0.0:
            transformed = transformed + torch.randn_like(transformed) * noise_std * valid_mask
        if bool(corruption_active) and field_dropout > 0.0:
            keep_bool = torch.rand_like(transformed) >= field_dropout
            element_drop_mask = (~keep_bool) & valid_values
            transformed = transformed * keep_bool.to(dtype=transformed.dtype)
        if bool(corruption_active) and group_dropout > 0.0:
            for group, indices in dict(self.config.field_groups).items():
                group_indices = tuple(int(index) for index in indices)
                if not group_indices:
                    continue
                index_tensor = torch.as_tensor(group_indices, device=transformed.device, dtype=torch.long)
                if int(index_tensor.max().detach().cpu().item()) >= int(transformed.shape[-1]):
                    raise ValueError(
                        f"field group {group!r} index {int(index_tensor.max().detach().cpu().item())} "
                        f"outside transformed field dim {int(transformed.shape[-1])}"
                    )
                keep_group_bool = torch.rand((int(transformed.shape[0]), 1, 1), device=transformed.device) >= group_dropout
                keep_group = keep_group_bool.to(dtype=transformed.dtype)
                group_draws += int(keep_group_bool.numel())
                group_drop_events += int((~keep_group_bool).detach().sum().cpu().item())
                group_values = transformed.index_select(dim=-1, index=index_tensor) * keep_group
                transformed = transformed.index_copy(dim=-1, index=index_tensor, source=group_values)
        clip_value = float(self.config.residual_field_clip_value)
        pre_sanitize_nonfinite = (~torch.isfinite(transformed)) & valid_values
        clipped_values = torch.zeros_like(valid_values)
        if clip_value > 0.0:
            clipped_values = torch.isfinite(transformed) & (transformed.abs() > clip_value) & valid_values
        transformed = self._sanitize_residual_fields(transformed, valid_mask)
        valid_value_count = int(valid_values.detach().sum().cpu().item())
        denominator = float(max(valid_value_count, 1))
        diagnostics = {
            "oracle_field_transform": True,
            "oracle_field_source": source,
            "oracle_field_alpha": alpha,
            "oracle_field_noise_std": noise_std,
            "oracle_field_dropout": field_dropout,
            "oracle_field_group_dropout": group_dropout,
            "oracle_field_corruption_active": bool(corruption_active),
            "oracle_field_selected_indices": [int(index) for index in self.config.source_field_indices],
            "oracle_field_selected_names": list(self.config.field_names),
            "oracle_field_valid_value_count": valid_value_count,
            "oracle_field_input_nonfinite_count": int(input_nonfinite.detach().sum().cpu().item()),
            "oracle_field_pre_sanitize_nonfinite_count": int(pre_sanitize_nonfinite.detach().sum().cpu().item()),
            "oracle_field_clipped_value_count": int(clipped_values.detach().sum().cpu().item()),
            "oracle_field_clip_fraction": float(clipped_values.detach().sum().cpu().item()) / denominator,
            "oracle_field_element_dropout_count": int(element_drop_mask.detach().sum().cpu().item()),
            "oracle_field_element_dropout_fraction": float(element_drop_mask.detach().sum().cpu().item()) / denominator,
            "oracle_field_group_dropout_events": int(group_drop_events),
            "oracle_field_group_dropout_draws": int(group_draws),
        }
        return transformed, diagnostics

    def set_reconstructor_trainable(self, trainable: bool) -> dict[str, int]:
        total = 0
        changed = 0
        if self.reconstructor is not None:
            for parameter in self.reconstructor.parameters():
                total += 1
                if bool(parameter.requires_grad) != bool(trainable):
                    changed += 1
                parameter.requires_grad_(bool(trainable))
        return {"reconstructor_parameter_tensors": total, "changed": changed, "trainable": int(bool(trainable))}

    def set_part_trainable(self, trainable: bool) -> dict[str, int]:
        total = 0
        changed = 0
        for parameter in self.part_model.parameters():
            total += 1
            if bool(parameter.requires_grad) != bool(trainable):
                changed += 1
            parameter.requires_grad_(bool(trainable))
        return {"part_parameter_tensors": total, "changed": changed, "trainable": int(bool(trainable))}

    def residual_fields_from_batch(
        self,
        *,
        features: Any,
        mask: Any,
        tokens: Any | None = None,
        raw_mask: Any | None = None,
        indices: Any | None = None,
        residual_fields: Any | None = None,
        residual_features: Any | None = None,
        oracle_fields: Any | None = None,
        target_fields: Any | None = None,
    ) -> tuple[Any, Any, Any | None, Mapping[str, Any] | None]:
        torch = require_torch()
        batch_size, _, particles = features.shape
        if residual_features is not None:
            field_features = residual_features.to(device=features.device, dtype=features.dtype)
            if field_features.ndim != 3:
                raise ValueError("residual_features must have shape [B, F, P]")
            fields = field_features.transpose(1, 2).contiguous()
            fields = self._select_source_fields(fields)
            field_features = fields.transpose(1, 2).contiguous()
            return fields, field_features, None, None
        if residual_fields is not None:
            fields = residual_fields.to(device=features.device, dtype=features.dtype)
            if fields.ndim != 3:
                raise ValueError("residual_fields must have shape [B, P, F]")
            fields = self._select_source_fields(fields)
            return fields, fields.transpose(1, 2).contiguous(), None, None
        source = self.config.field_source
        if source in ORACLE_RESIDUAL_FIELD_SOURCES:
            fields = oracle_fields if oracle_fields is not None else target_fields
            if fields is None:
                raise ValueError("oracle residual-field source requires oracle_fields or target_fields")
            fields = fields.to(device=features.device, dtype=features.dtype)
            fields, diagnostics = self._transform_oracle_fields(fields, mask=mask, raw_mask=raw_mask)
            return fields, fields.transpose(1, 2).contiguous(), None, diagnostics
        if source == RESIDUAL_FIELD_SOURCE_ZERO:
            fields = torch.zeros((batch_size, particles, int(self.config.field_dim)), device=features.device, dtype=features.dtype)
            return fields, fields.transpose(1, 2).contiguous(), None, None
        if source in CONTROL_RESIDUAL_FIELD_SOURCES:
            control_config = self.config.control_config
            if not isinstance(control_config, LocalResidualFieldControlConfig):
                control_config = LocalResidualFieldControlConfig(field_names=tuple(self.config.field_names))
            control_output = apply_local_residual_field_control(
                source=source,
                target_fields=None if target_fields is None else target_fields.to(device=features.device, dtype=features.dtype),
                mask=raw_mask if raw_mask is not None else mask.squeeze(1),
                field_names=tuple(self.config.field_names),
                config=control_config,
                tokens=None if tokens is None else tokens.to(device=features.device, dtype=features.dtype),
                generator=self.control_generator,
                indices=None if indices is None else indices.to(device=features.device),
            )
            fields = control_output.fields.to(device=features.device, dtype=features.dtype)
            return fields, fields.transpose(1, 2).contiguous(), None, dict(control_output.diagnostics)
        if self.reconstructor is None:
            raise ValueError(f"residual-field source {source!r} requires a reconstructor")
        if tokens is None:
            raise ValueError("predicted residual-field sources require raw HLT tokens")
        tokens = tokens.to(device=features.device, dtype=features.dtype)
        raw_mask = raw_mask.to(device=features.device, dtype=torch.bool) if raw_mask is not None else None
        if source == RESIDUAL_FIELD_SOURCE_FROZEN_RECONSTRUCTOR:
            with torch.no_grad():
                reco_output = self.reconstructor(tokens, raw_mask)
        else:
            reco_output = self.reconstructor(tokens, raw_mask)
        reco_output = self._select_reconstructor_output(reco_output)
        fields = reco_output.predicted_fields.to(device=features.device, dtype=features.dtype)
        return fields, fields.transpose(1, 2).contiguous(), reco_output, None

    def augment_features(
        self,
        features: Any,
        mask: Any,
        *,
        residual_fields: Any,
        residual_features: Any | None = None,
    ) -> tuple[Any, Any]:
        torch = require_torch()
        if features.ndim != 3:
            raise ValueError("features must have shape [B, C, P]")
        field_features = residual_features
        if field_features is None:
            if residual_fields.ndim != 3:
                raise ValueError("residual_fields must have shape [B, P, F]")
            field_features = residual_fields.transpose(1, 2).contiguous()
        field_features = field_features.to(device=features.device, dtype=features.dtype)
        if field_features.shape[0] != features.shape[0] or field_features.shape[2] != features.shape[2]:
            raise ValueError(
                f"residual field features shape {tuple(field_features.shape)} is incompatible with "
                f"features shape {tuple(features.shape)}"
            )
        if int(field_features.shape[1]) != int(self.config.field_dim):
            raise ValueError(f"residual field dim {field_features.shape[1]} != configured {self.config.field_dim}")
        mask_features = mask
        if mask_features.ndim == 2:
            mask_features = mask_features[:, None, :]
        mask_features = mask_features.to(device=features.device, dtype=torch.bool)
        field_features = field_features * mask_features.to(dtype=field_features.dtype)
        clip_value = float(self.config.residual_field_clip_value)
        if clip_value > 0.0:
            field_features = torch.nan_to_num(field_features, nan=0.0, posinf=clip_value, neginf=-clip_value)
            field_features = field_features.clamp(min=-clip_value, max=clip_value)
        else:
            field_features = torch.nan_to_num(field_features, nan=0.0, posinf=0.0, neginf=0.0)
        field_features = self.field_dropout(field_features) * float(self.config.residual_field_scale)
        augmented = torch.cat([features, field_features], dim=1)
        return augmented, field_features

    def forward(
        self,
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        *,
        tokens: Any | None = None,
        raw_mask: Any | None = None,
        indices: Any | None = None,
        residual_fields: Any | None = None,
        residual_features: Any | None = None,
        oracle_fields: Any | None = None,
        target_fields: Any | None = None,
        return_outputs: bool = False,
    ) -> Any:
        if self.config.field_source == RESIDUAL_FIELD_SOURCE_HLT_ONLY:
            logits = self.part_model(points, features, lorentz_vectors, mask)
            if not bool(return_outputs):
                return logits
            torch = require_torch()
            batch_size = int(features.shape[0])
            particles = int(features.shape[2])
            empty_fields = torch.zeros((batch_size, particles, 0), device=features.device, dtype=features.dtype)
            empty_features = torch.zeros((batch_size, 0, particles), device=features.device, dtype=features.dtype)
            field_mask = raw_mask if raw_mask is not None else mask.squeeze(1)
            diagnostics = {
                "contract": LOCAL_RESIDUAL_FIELD_TAGGER_CONTRACT,
                "field_source": self.config.field_source,
                "hlt_only": True,
                "augmented_feature_dim": int(features.shape[1]),
                "base_feature_dim": int(features.shape[1]),
                "field_dim": 0,
                "residual_abs_mean": 0.0,
                "residual_l2_mean": 0.0,
                "valid_particles": int(field_mask.bool().detach().sum().cpu().item()),
            }
            return LocalResidualFieldTaggerOutput(
                logits=logits,
                augmented_features=features,
                residual_features=empty_features,
                residual_fields=empty_fields,
                field_mask=field_mask,
                diagnostics=diagnostics,
                reconstructor_output=None,
                control_diagnostics=None,
            )
        fields, field_features, reco_output, control_diagnostics = self.residual_fields_from_batch(
            features=features,
            mask=mask,
            tokens=tokens,
            raw_mask=raw_mask,
            indices=indices,
            residual_fields=residual_fields,
            residual_features=residual_features,
            oracle_fields=oracle_fields,
            target_fields=target_fields,
        )
        augmented, field_features = self.augment_features(
            features,
            mask,
            residual_fields=fields,
            residual_features=field_features,
        )
        logits = self.part_model(points, augmented, lorentz_vectors, mask)
        if not bool(return_outputs):
            return logits
        field_mask = raw_mask if raw_mask is not None else mask.squeeze(1)
        diagnostics = {
            "contract": LOCAL_RESIDUAL_FIELD_TAGGER_CONTRACT,
            "field_source": self.config.field_source,
            "augmented_feature_dim": int(augmented.shape[1]),
            "base_feature_dim": int(features.shape[1]),
            "field_dim": int(field_features.shape[1]),
            "residual_field_clip_value": float(self.config.residual_field_clip_value),
            **_masked_field_stats(fields, field_mask),
        }
        if isinstance(control_diagnostics, Mapping) and self.config.field_source in CONTROL_RESIDUAL_FIELD_SOURCES:
            diagnostics["control_contract"] = LOCAL_RESIDUAL_FIELD_CONTROL_CONTRACT
            diagnostics["control_diagnostics"] = dict(control_diagnostics)
        elif isinstance(control_diagnostics, Mapping):
            diagnostics["source_diagnostics"] = dict(control_diagnostics)
            if control_diagnostics.get("oracle_field_transform"):
                diagnostics["oracle_field_transform"] = dict(control_diagnostics)
        if reco_output is not None:
            diagnostics["reconstructor_contract"] = LOCAL_RESIDUAL_RECONSTRUCTOR_CONTRACT
            diagnostics["reconstructor_variant"] = getattr(getattr(self.reconstructor, "config", None), "variant", None)
        return LocalResidualFieldTaggerOutput(
            logits=logits,
            augmented_features=augmented,
            residual_features=field_features,
            residual_fields=fields,
            field_mask=field_mask,
            diagnostics=diagnostics,
            reconstructor_output=reco_output,
            control_diagnostics=control_diagnostics,
        )


def _extract_checkpoint_state_dict(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                return value
        if payload and all(hasattr(value, "shape") for value in payload.values()):
            return payload
    raise ValueError("checkpoint does not contain a recognizable model state dict")


def _strip_prefixes(key: str) -> str:
    output = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model."):
            if output.startswith(prefix):
                output = output[len(prefix) :]
                changed = True
    return output


def _part_warm_start_candidates(source_key: str) -> tuple[str, ...]:
    clean = _strip_prefixes(source_key)
    candidates: list[str] = []
    if clean.startswith("part_model."):
        candidates.append(clean)
    else:
        candidates.append(f"part_model.{clean}")
    if clean.startswith("mod."):
        candidates.append(f"part_model.{clean}")
    if clean.startswith("model."):
        candidates.append(f"part_model.{clean[len('model.'):]}")
    output: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = _strip_prefixes(candidate)
        if candidate not in seen:
            seen.add(candidate)
            output.append(candidate)
    return tuple(output)


def _can_partial_copy(source: Any, target: Any) -> bool:
    if len(tuple(source.shape)) != len(tuple(target.shape)):
        return False
    return all(int(target_dim) >= int(source_dim) for source_dim, target_dim in zip(source.shape, target.shape)) and (
        tuple(source.shape) != tuple(target.shape)
    )


def _partial_copy_tensor(source: Any, target: Any) -> Any:
    updated = target.detach().clone()
    updated.zero_()
    slices = tuple(slice(0, int(dim)) for dim in source.shape)
    updated[slices] = source.detach().to(device=target.device, dtype=target.dtype)
    return updated


def warm_start_local_residual_field_tagger_part(
    model: LocalResidualFieldAugmentedParT,
    checkpoint_path: str | Path,
    *,
    map_location: Any = "cpu",
    require: bool = False,
) -> dict[str, Any]:
    """Load an HLT ParT checkpoint into the augmented ParT backbone.

    Matching tensors are copied exactly.  If the target tensor is larger only
    because the feature dimension expanded, the source tensor is copied into
    the leading slice and the new residual-field slice is zeroed so the initial
    model starts close to the baseline.
    """

    payload = _torch_load_checkpoint(checkpoint_path, map_location=map_location)
    source_state = _extract_checkpoint_state_dict(payload)
    target_state = model.state_dict()
    updated_state = dict(target_state)
    loaded: list[dict[str, str]] = []
    partial: list[dict[str, Any]] = []
    skipped_shape: list[dict[str, Any]] = []
    unmatched: list[str] = []
    non_tensor: list[str] = []
    for source_key, source_value in source_state.items():
        if not hasattr(source_value, "shape"):
            non_tensor.append(str(source_key))
            continue
        matched = False
        for candidate in _part_warm_start_candidates(str(source_key)):
            if not candidate.startswith("part_model.") or candidate not in target_state:
                continue
            target_value = target_state[candidate]
            if tuple(target_value.shape) == tuple(source_value.shape):
                updated_state[candidate] = source_value.detach().to(device=target_value.device, dtype=target_value.dtype)
                loaded.append({"source_key": str(source_key), "target_key": candidate})
                matched = True
                break
            if _can_partial_copy(source_value, target_value):
                updated_state[candidate] = _partial_copy_tensor(source_value, target_value)
                partial.append(
                    {
                        "source_key": str(source_key),
                        "target_key": candidate,
                        "source_shape": list(source_value.shape),
                        "target_shape": list(target_value.shape),
                    }
                )
                matched = True
                break
            skipped_shape.append(
                {
                    "source_key": str(source_key),
                    "target_key": candidate,
                    "source_shape": list(source_value.shape),
                    "target_shape": list(target_value.shape),
                }
            )
        if not matched:
            unmatched.append(str(source_key))
    model.load_state_dict(updated_state, strict=True)
    report = {
        "contract": LOCAL_RESIDUAL_FIELD_TAGGER_CONTRACT,
        "step": LOCAL_RESIDUAL_FIELD_TAGGER_STEP,
        "checkpoint": str(checkpoint_path),
        "loaded_key_count": len(loaded),
        "partial_loaded_key_count": len(partial),
        "loaded_keys": loaded[:50],
        "partial_loaded_keys": partial[:50],
        "shape_mismatch_count": len(skipped_shape),
        "shape_mismatches": skipped_shape[:50],
        "unmatched_key_count": len(unmatched),
        "unmatched_source_keys_sample": unmatched[:50],
        "non_tensor_key_count": len(non_tensor),
        "non_tensor_keys_sample": non_tensor[:50],
        "required": bool(require),
    }
    if bool(require) and (len(loaded) + len(partial)) == 0:
        raise ValueError(f"warm start loaded zero ParT keys from {checkpoint_path}")
    return report


def load_local_residual_reconstructor_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: Any = "cpu",
) -> tuple[Any, Mapping[str, Any]]:
    payload = _torch_load_checkpoint(checkpoint_path, map_location=map_location)
    model_config = payload.get("model_config") if isinstance(payload, Mapping) else None
    if not isinstance(model_config, Mapping):
        raise ValueError("reconstructor checkpoint is missing model_config")
    clean_config = dict(model_config)
    clean_config.pop("contract", None)
    model = build_local_residual_field_reconstructor(LocalResidualFieldReconstructorConfig(**clean_config))
    model.load_state_dict(_extract_checkpoint_state_dict(payload))
    return model, payload


__all__ = [
    "LOCAL_RESIDUAL_FIELD_TAGGER_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_TAGGER_STEP",
    "RESIDUAL_FIELD_SOURCE_HLT_ONLY",
    "RESIDUAL_FIELD_SOURCE_ZERO",
    "RESIDUAL_FIELD_SOURCE_ORACLE",
    "RESIDUAL_FIELD_SOURCE_ORACLE_SCALED",
    "RESIDUAL_FIELD_SOURCE_ORACLE_FIELD_SUBSET",
    "RESIDUAL_FIELD_SOURCE_ORACLE_NOISY",
    "RESIDUAL_FIELD_SOURCE_ORACLE_DROPOUT",
    "RESIDUAL_FIELD_SOURCE_FROZEN_RECONSTRUCTOR",
    "RESIDUAL_FIELD_SOURCE_JOINT_RECONSTRUCTOR",
    "ORACLE_RESIDUAL_FIELD_SOURCES",
    "RESIDUAL_FIELD_SOURCES",
    "LocalResidualFieldTaggerConfig",
    "LocalResidualFieldTaggerOutput",
    "HLTAnchoredSplitLayerNorm",
    "install_hlt_anchored_field_normalization",
    "LocalResidualFieldAugmentedParT",
    "normalize_residual_field_source",
    "warm_start_local_residual_field_tagger_part",
    "load_local_residual_reconstructor_from_checkpoint",
]
