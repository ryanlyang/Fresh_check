"""Embedding-injected exact-HLT-ParT wrapper for architecture views.

Step 2 keeps the real HLT ParT as the scoring backbone.  Architecture-view
branches produce only a gated residual in ParT's particle embedding space.  A
zero-initialized residual projection recovers the exact baseline logits.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, build_hlt_classifier, require_torch

from teacher_logit_reco.local_compression_part.adapter import (
    LOCAL_COMPRESSION_GEOMETRY_FEATURES,
    LOCAL_COMPRESSION_PID_FEATURES,
)
from teacher_logit_reco.local_compression_part.config import LocalCompressionFeatureConfig
from teacher_logit_reco.local_compression_part.features import (
    LocalCompressionCanonicalInputs,
    build_local_compression_canonical_inputs,
)

from .config import (
    ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER_WIDE,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_PLUS_FEATURE_MLP_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER,
    ARCHITECTURE_VIEW_PART_CONTRACT,
    ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
    ARCHITECTURE_VIEW_VARIANT_RANDOM_VIEW_CONTROL,
    ArchitectureViewConfig,
    architecture_view_effective_variant,
    architecture_view_variant_spec,
    enabled_views_for_variant,
    normalize_architecture_view_variant,
)
from .fusion import ArchitectureViewFusionOutput, ArchitectureViewParticleViews, align_particle_mask_to_length

try:  # Keep package imports cheap on machines without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


ARCHITECTURE_VIEW_MODEL_STEP = "architecture_view_part_step2_embedding_injected_part"
ARCHITECTURE_VIEW_MODEL_CONTRACT = f"{ARCHITECTURE_VIEW_PART_CONTRACT}_embedding_injected_exact_part_v1"


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, value.new_full((), float(nan)))


def _normalize_view_config(config: ArchitectureViewConfig | Mapping[str, Any] | None = None) -> ArchitectureViewConfig:
    if config is None:
        return ArchitectureViewConfig()
    if isinstance(config, ArchitectureViewConfig):
        return config
    return ArchitectureViewConfig.from_dict(dict(config))


def _coerce_tokens_and_mask(tokens_or_batch: Any, mask: Any | None = None) -> tuple[Any, Any]:
    torch = require_torch()
    if isinstance(tokens_or_batch, Mapping):
        batch = tokens_or_batch
        token_keys = ("tokens", "hlt_tokens", "raw_tokens", "x")
        mask_keys = ("mask", "hlt_mask", "raw_mask", "particle_mask")
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
    tokens = torch.as_tensor(tokens).float()
    resolved_mask = torch.as_tensor(resolved_mask, device=tokens.device).bool()
    if int(tokens.ndim) != 3:
        raise ValueError(f"tokens must have shape [batch, particles, raw_dim], got {tuple(tokens.shape)}")
    if int(resolved_mask.ndim) != 2:
        raise ValueError(f"mask must have shape [batch, particles], got {tuple(resolved_mask.shape)}")
    if tuple(tokens.shape[:2]) != tuple(resolved_mask.shape):
        raise ValueError(f"tokens/mask leading shapes differ: {tuple(tokens.shape[:2])} vs {tuple(resolved_mask.shape)}")
    return _nan_to_num_torch(tokens), resolved_mask


def _resolve_embed_module(part_model: Any) -> Any:
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
    raise ValueError(
        "Could not locate the HLT ParT particle embedding module. "
        "Expected part_model.mod.embed or an equivalent named module ending in 'embed'."
    )


def _delta_for_embed_output(delta_h: Any, embed_output: Any) -> Any:
    """Broadcast ``[B, P, D]`` residuals to common ParT embedding layouts."""

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
    raise ValueError(
        f"Cannot align delta_h shape {tuple(delta_h.shape)} with ParT embed output shape {shape}. "
        "Check ArchitectureViewConfig.part_embed_dim against the HLT ParT embed dimension."
    )


def _batch_first_from_embed_output(embed_output: Any, *, batch: int, particles: int, dim: int) -> Any:
    """Convert common ParT embed layouts to ``[B, P, D]`` for local adapters."""

    torch = require_torch()
    if not isinstance(embed_output, torch.Tensor):
        raise TypeError(f"ParT embed output must be a tensor, got {type(embed_output)!r}")
    if int(embed_output.ndim) != 3:
        raise ValueError(f"ParT embed output must be rank 3, got {tuple(embed_output.shape)}")
    shape = tuple(embed_output.shape)
    if int(shape[-1]) == int(dim):
        if int(shape[0]) == int(batch):
            if int(shape[1]) > int(particles):
                raise ValueError(f"embed particle dimension {shape[1]} exceeds mask particles {particles}")
            return embed_output
        if int(shape[1]) == int(batch):
            if int(shape[0]) > int(particles):
                raise ValueError(f"embed particle dimension {shape[0]} exceeds mask particles {particles}")
            return embed_output.permute(1, 0, 2).contiguous()
    if int(shape[0]) == int(batch) and int(shape[1]) == int(dim):
        if int(shape[2]) > int(particles):
            raise ValueError(f"embed particle dimension {shape[2]} exceeds mask particles {particles}")
        return embed_output.transpose(1, 2).contiguous()
    raise ValueError(
        f"Cannot convert ParT embed output shape {shape} to [B, P, D] with "
        f"batch={batch}, particles={particles}, dim={dim}."
    )


def _infer_part_embed_dim(part_model: Any, fallback: int) -> int:
    config = getattr(part_model, "config", {}) or {}
    if isinstance(config, Mapping):
        embed_dims = config.get("embed_dims")
        if isinstance(embed_dims, (list, tuple)) and embed_dims:
            try:
                return int(embed_dims[-1])
            except (TypeError, ValueError):
                pass
        for key in ("embed_dim", "part_embed_dim"):
            if key in config:
                try:
                    return int(config[key])
                except (TypeError, ValueError):
                    pass
    return int(fallback)


def _tensor_quantile(value: Any, q: float) -> Any:
    torch = require_torch()
    if int(value.numel()) == 0:
        return value.new_zeros(())
    if hasattr(torch, "quantile"):
        return torch.quantile(value.float(), float(q))
    flat = value.reshape(-1).float().sort().values
    index = min(int(flat.numel()) - 1, max(0, int(round(float(q) * (int(flat.numel()) - 1)))))
    return flat[index]


@dataclass(frozen=True)
class ArchitectureViewFeatureDeltaOutput:
    """Bounded residual edits to canonical ParT feature rows."""

    delta_F_rows: Any
    raw_delta_rows: Any
    bounded_delta_rows: Any
    adapted_feature_rows: Any
    mask: Any
    feature_rows: Any
    feature_delta_scales: Any
    feature_active_mask: Any
    feature_names: tuple[str, ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = require_torch()
        if int(self.delta_F_rows.ndim) != 3:
            raise ValueError("delta_F_rows must have shape [batch, particles, feature_dim]")
        batch_size, num_particles, feature_dim = tuple(self.delta_F_rows.shape)
        expected = (batch_size, num_particles, feature_dim)
        for name in ("raw_delta_rows", "bounded_delta_rows", "adapted_feature_rows", "feature_rows"):
            if tuple(getattr(self, name).shape) != expected:
                raise ValueError(f"{name} shape must match delta_F_rows")
        if tuple(self.mask.shape) != (batch_size, num_particles):
            raise ValueError("mask shape must match delta_F_rows leading dimensions")
        if tuple(self.feature_delta_scales.shape) != (feature_dim,):
            raise ValueError("feature_delta_scales must have shape [feature_dim]")
        if tuple(self.feature_active_mask.shape) != (feature_dim,):
            raise ValueError("feature_active_mask must have shape [feature_dim]")
        if len(tuple(self.feature_names)) != feature_dim:
            raise ValueError("feature_names length must match delta_F_rows feature dimension")
        for name in (
            "delta_F_rows",
            "raw_delta_rows",
            "bounded_delta_rows",
            "adapted_feature_rows",
            "feature_rows",
            "feature_delta_scales",
            "feature_active_mask",
        ):
            if not bool(torch.isfinite(getattr(self, name)).all()):
                raise ValueError(f"{name} contains non-finite values")
        invalid = torch.where(self.mask[:, :, None].bool(), torch.zeros_like(self.delta_F_rows), self.delta_F_rows.abs())
        if float(invalid.max().detach().cpu().item()) > 1.0e-6:
            raise ValueError("invalid particles must have zero delta_F_rows")
        scale = self.feature_delta_scales.abs() * self.feature_active_mask.abs()
        over_bound = torch.clamp(self.delta_F_rows.abs() - scale.view(1, 1, -1), min=0.0)
        if float(over_bound.max().detach().cpu().item()) > 1.0e-5:
            raise ValueError("delta_F_rows exceed configured feature bounds")
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def summary(self) -> dict[str, Any]:
        return {
            "delta_F_rows_shape": list(self.delta_F_rows.shape),
            "adapted_feature_rows_shape": list(self.adapted_feature_rows.shape),
            "feature_names": list(self.feature_names),
            "feature_delta_scales": [float(v) for v in self.feature_delta_scales.detach().cpu().tolist()],
            "active_feature_count": int(self.feature_active_mask.detach().cpu().sum().item()),
            "diagnostics": dict(self.diagnostics),
        }


class ArchitectureViewFeatureDeltaMLP(_ModuleBase):
    """Small LC-style MLP that repairs canonical ParT feature rows before embedding."""

    def __init__(
        self,
        feature_config: LocalCompressionFeatureConfig,
        config: ArchitectureViewConfig,
    ) -> None:
        super().__init__()
        torch = require_torch()
        self.feature_names = tuple(feature_config.canonical_feature_names)
        self.feature_dim = len(self.feature_names)
        hidden_dim = max(int(config.fusion_hidden_dim), self.feature_dim * 4)
        self.projector = torch.nn.Sequential(
            torch.nn.LayerNorm(self.feature_dim),
            torch.nn.Linear(self.feature_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(hidden_dim, self.feature_dim),
        )
        torch.nn.init.zeros_(self.projector[-1].weight)
        torch.nn.init.zeros_(self.projector[-1].bias)
        scales = torch.as_tensor(tuple(feature_config.feature_delta_scales), dtype=torch.float32)
        if not bool(config.use_feature_wise_input_delta_scales):
            scales = torch.ones_like(scales)
        scales = scales * float(config.input_delta_scale)
        active = torch.ones(self.feature_dim, dtype=torch.float32)
        if bool(config.freeze_input_delta_pid):
            for index, name in enumerate(self.feature_names):
                if name in LOCAL_COMPRESSION_PID_FEATURES:
                    active[index] = 0.0
        if bool(config.freeze_input_delta_geometry):
            for index, name in enumerate(self.feature_names):
                if name in LOCAL_COMPRESSION_GEOMETRY_FEATURES:
                    active[index] = 0.0
        self.register_buffer("feature_delta_scales", scales)
        self.register_buffer("feature_active_mask", active)
        self.input_delta_scale = float(config.input_delta_scale)
        self.use_feature_wise_input_delta_scales = bool(config.use_feature_wise_input_delta_scales)
        self.freeze_input_delta_pid = bool(config.freeze_input_delta_pid)
        self.freeze_input_delta_geometry = bool(config.freeze_input_delta_geometry)

    def _diagnostics(self, delta: Any, feature_rows: Any, mask: Any) -> dict[str, Any]:
        torch = require_torch()
        diagnostics: dict[str, Any] = {
            "adapter_kind": "lc_mlp_delta_features",
            "zero_init_delta_projection": True,
            "input_delta_scale": float(self.input_delta_scale),
            "uses_feature_wise_input_delta_scales": bool(self.use_feature_wise_input_delta_scales),
            "freeze_input_delta_pid": bool(self.freeze_input_delta_pid),
            "freeze_input_delta_geometry": bool(self.freeze_input_delta_geometry),
            "active_feature_count": int(self.feature_active_mask.detach().cpu().sum().item()),
        }
        if not bool(mask.any()):
            diagnostics.update(
                {
                    "delta_F_l2_mean": 0.0,
                    "delta_F_l2_p90": 0.0,
                    "delta_F_l2_max": 0.0,
                    "delta_F_abs_mean": 0.0,
                    "delta_F_abs_p90": 0.0,
                    "delta_F_abs_max": 0.0,
                    "delta_F_to_feature_norm_ratio": 0.0,
                }
            )
            return diagnostics
        active_delta = delta[mask]
        active_features = feature_rows[mask]
        l2 = active_delta.square().sum(dim=-1)
        abs_delta = active_delta.abs()
        feature_l2 = active_features.square().sum(dim=-1).mean().clamp_min(1.0e-12)
        delta_l2_mean = l2.mean()
        diagnostics.update(
            {
                "delta_F_l2_mean": float(delta_l2_mean.detach().cpu().item()),
                "delta_F_l2_p90": float(_tensor_quantile(l2, 0.90).detach().cpu().item()),
                "delta_F_l2_max": float(l2.max().detach().cpu().item()),
                "delta_F_abs_mean": float(abs_delta.mean().detach().cpu().item()),
                "delta_F_abs_p90": float(_tensor_quantile(abs_delta.reshape(-1), 0.90).detach().cpu().item()),
                "delta_F_abs_max": float(abs_delta.max().detach().cpu().item()),
                "delta_F_to_feature_norm_ratio": float(
                    (delta_l2_mean.sqrt() / feature_l2.sqrt()).detach().cpu().item()
                ),
                "final_delta_projection_weight_norm": float(
                    self.projector[-1].weight.detach().norm().cpu().item()
                ),
                "final_delta_projection_bias_norm": float(
                    self.projector[-1].bias.detach().norm().cpu().item()
                ),
            }
        )
        per_feature_abs_mean = abs_delta.mean(dim=0)
        per_feature_abs_p90 = torch.stack(
            [_tensor_quantile(abs_delta[:, index], 0.90) for index in range(self.feature_dim)]
        )
        per_feature_sq_mean = active_delta.square().mean(dim=0)
        per_feature_rms = per_feature_sq_mean.sqrt()
        for index, name in enumerate(self.feature_names):
            diagnostics[f"per_feature_delta_F_abs_mean.{name}"] = float(
                per_feature_abs_mean[index].detach().cpu().item()
            )
            diagnostics[f"per_feature_delta_F_abs_p90.{name}"] = float(
                per_feature_abs_p90[index].detach().cpu().item()
            )
            diagnostics[f"per_feature_delta_F_sq_mean.{name}"] = float(
                per_feature_sq_mean[index].detach().cpu().item()
            )
            diagnostics[f"per_feature_delta_F_rms.{name}"] = float(
                per_feature_rms[index].detach().cpu().item()
            )
            diagnostics[f"feature_delta_scale.{name}"] = float(
                self.feature_delta_scales[index].detach().cpu().item()
            )
            diagnostics[f"feature_delta_active.{name}"] = float(
                self.feature_active_mask[index].detach().cpu().item()
            )
        return diagnostics

    def forward(self, canonical: LocalCompressionCanonicalInputs) -> ArchitectureViewFeatureDeltaOutput:
        torch = require_torch()
        mask = canonical.particle_mask.bool()
        feature_rows = canonical.feature_rows()
        feature_rows = torch.where(mask[:, :, None], feature_rows, torch.zeros_like(feature_rows))
        raw_delta = _nan_to_num_torch(self.projector(feature_rows))
        bounded_delta = torch.tanh(raw_delta)
        scale = (self.feature_delta_scales * self.feature_active_mask).to(
            dtype=bounded_delta.dtype,
            device=bounded_delta.device,
        )
        delta = bounded_delta * scale.view(1, 1, -1)
        delta = torch.where(mask[:, :, None], delta, torch.zeros_like(delta))
        raw_delta = torch.where(mask[:, :, None], raw_delta, torch.zeros_like(raw_delta))
        bounded_delta = torch.where(mask[:, :, None], bounded_delta, torch.zeros_like(bounded_delta))
        adapted = torch.where(mask[:, :, None], feature_rows + delta, torch.zeros_like(feature_rows))
        return ArchitectureViewFeatureDeltaOutput(
            delta_F_rows=delta,
            raw_delta_rows=raw_delta,
            bounded_delta_rows=bounded_delta,
            adapted_feature_rows=adapted,
            mask=mask,
            feature_rows=feature_rows,
            feature_delta_scales=self.feature_delta_scales.to(dtype=delta.dtype, device=delta.device).detach(),
            feature_active_mask=self.feature_active_mask.to(dtype=delta.dtype, device=delta.device).detach(),
            feature_names=self.feature_names,
            diagnostics=self._diagnostics(delta, feature_rows, mask),
        )


@dataclass(frozen=True)
class ArchitectureViewResidualPartOutput:
    """Full Step 2 forward output."""

    logits: Any
    canonical_inputs: LocalCompressionCanonicalInputs
    view_output: ArchitectureViewFusionOutput
    config: ArchitectureViewConfig
    variant: str
    part_model_class: str
    embed_module_name: str
    injection_summary: Mapping[str, Any]
    feature_delta_output: ArchitectureViewFeatureDeltaOutput | None = None
    variant_behavior: Mapping[str, Any] | None = None
    parameter_accounting: Mapping[str, Any] | None = None
    baseline_checkpoint_report: Mapping[str, Any] | None = None
    init_logit_diff_vs_baseline: Mapping[str, Any] | None = None
    part_embedding: Any | None = None

    @property
    def output_contract(self) -> str:
        return ARCHITECTURE_VIEW_MODEL_CONTRACT

    @property
    def baseline_recoverable_at_zero_delta(self) -> bool:
        return bool(self.view_output.diagnostics.get("baseline_recovery_zero_projection", False)) or not bool(
            self.config.enabled_views
        )

    def diagnostics(self) -> dict[str, Any]:
        delta = self.view_output.delta_h.detach()
        mask = align_particle_mask_to_length(self.view_output.mask.detach().bool(), int(delta.shape[1]))
        valid = mask.to(dtype=delta.dtype)
        denom = valid.sum().clamp_min(1.0)
        delta_norm = delta.norm(dim=-1)
        gate_values = self.view_output.gate.squeeze(-1)
        gate_mask = align_particle_mask_to_length(self.view_output.mask.detach().bool(), int(gate_values.shape[1]))
        gate_valid = gate_mask.to(dtype=gate_values.dtype)
        gate_denom = gate_valid.sum().clamp_min(1.0)
        accounting = dict(self.parameter_accounting or {})
        checkpoint = dict(self.baseline_checkpoint_report or {})
        init_diff = dict(self.init_logit_diff_vs_baseline or {})
        diagnostics = {
            "contract": ARCHITECTURE_VIEW_MODEL_CONTRACT,
            "step": ARCHITECTURE_VIEW_MODEL_STEP,
            "variant": str(self.variant),
            "logits_shape": list(self.logits.shape),
            "enabled_views": list(self.config.enabled_views),
            "part_model_class": str(self.part_model_class),
            "embed_module_name": str(self.embed_module_name),
            "valid_particle_count": int(mask.sum().cpu().item()),
            "delta_h_abs_max": float(delta.abs().max().cpu().item()) if int(delta.numel()) else 0.0,
            "delta_h_norm_mean": float(((delta_norm * valid).sum() / denom).cpu().item()),
            "delta_h_norm_p90": float(_tensor_quantile(delta_norm[mask], 0.90).detach().cpu().item())
            if bool(mask.any())
            else 0.0,
            "delta_h_norm_max": float(delta_norm[mask].max().detach().cpu().item()) if bool(mask.any()) else 0.0,
            "gate_mean": float(((gate_values * gate_valid).sum() / gate_denom).detach().cpu().item()),
            "part_model_trainable": bool((accounting.get("trainable_part_params") or 0) > 0),
            "active_adapter_module_count": int(len(self.variant_behavior.get("active_adapter_module_names", [])))
            if isinstance(self.variant_behavior, Mapping)
            else 0,
            "trainable_part_params": float(accounting.get("trainable_part_params") or 0.0),
            "trainable_adapter_params": float(accounting.get("trainable_adapter_params") or 0.0),
            "dormant_adapter_params": float(accounting.get("dormant_adapter_params") or 0.0),
            "baseline_recoverable_at_zero_delta": bool(self.baseline_recoverable_at_zero_delta),
            "view_fusion": self.view_output.summary(),
            "embed_injection": dict(self.injection_summary),
            "variant_behavior": dict(self.variant_behavior or {}),
            "feature_delta": None
            if self.feature_delta_output is None
            else self.feature_delta_output.summary(),
            "baseline_checkpoint": checkpoint or None,
            "init_logit_diff_vs_baseline": init_diff or None,
            "part_embedding_shape": None if self.part_embedding is None else list(self.part_embedding.shape),
        }
        if self.part_embedding is not None:
            embedding = self.part_embedding.detach()
            embed_mask = align_particle_mask_to_length(mask, int(embedding.shape[1]))
            if bool(embed_mask.any()):
                embedding_norm = embedding.norm(dim=-1)[embed_mask]
                diagnostics.update(
                    {
                        "part_embedding_norm_mean": float(embedding_norm.mean().detach().cpu().item()),
                        "part_embedding_norm_p90": float(
                            _tensor_quantile(embedding_norm, 0.90).detach().cpu().item()
                        ),
                    }
                )
        if checkpoint:
            diagnostics.update(
                {
                    "baseline_checkpoint_path": checkpoint.get("baseline_checkpoint_path"),
                    "baseline_checkpoint_hash": checkpoint.get("baseline_checkpoint_hash"),
                    "baseline_checkpoint_selection_metric": checkpoint.get(
                        "baseline_checkpoint_selection_metric"
                    ),
                    "baseline_checkpoint_hlt_degradation_strength": checkpoint.get(
                        "baseline_checkpoint_hlt_degradation_strength"
                    ),
                    "baseline_checkpoint_split_manifest_hash": checkpoint.get(
                        "baseline_checkpoint_split_manifest_hash"
                    ),
                }
            )
        return diagnostics


class ArchitectureViewResidualParT(_ModuleBase):
    """Architecture-view residual injector for the exact HLT ParT backbone."""

    def __init__(
        self,
        config: ArchitectureViewConfig | Mapping[str, Any] | None = None,
        *,
        variant: str = "av_all_views",
        part_model: Any | None = None,
        feature_config: LocalCompressionFeatureConfig | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.variant = normalize_architecture_view_variant(variant)
        self.effective_variant = architecture_view_effective_variant(self.variant)
        self.variant_spec = architecture_view_variant_spec(self.variant)
        view_config = _normalize_view_config(config)
        view_config = ArchitectureViewConfig.from_dict(
            {**view_config.to_dict(), "enabled_views": enabled_views_for_variant(self.variant)}
        )
        self.feature_config = (
            feature_config
            if isinstance(feature_config, LocalCompressionFeatureConfig)
            else LocalCompressionFeatureConfig(**dict(feature_config or {}))
        )
        self.part_model_size = "large" if self.variant_spec.part_size == "larger" else "base"
        self.part_model = part_model or build_hlt_classifier(
            num_classes=int(view_config.num_classes),
            model_size=str(self.part_model_size),
        )
        if not isinstance(self.part_model, ParticleTransformerHLTClassifier):
            raise ValueError("ArchitectureViewResidualParT requires the real ParticleTransformerHLTClassifier backbone")
        inferred_part_embed_dim = _infer_part_embed_dim(self.part_model, int(view_config.part_embed_dim))
        if inferred_part_embed_dim != int(view_config.part_embed_dim):
            view_config = ArchitectureViewConfig.from_dict(
                {**view_config.to_dict(), "part_embed_dim": int(inferred_part_embed_dim)}
            )
        self.config = view_config
        self.view_module = ArchitectureViewParticleViews(self.config, enabled_views=self.config.enabled_views)
        torch = require_torch()
        canonical_feature_dim = len(self.feature_config.canonical_feature_names)
        context_hidden_dim = int(self.config.fusion_hidden_dim)
        if self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER_WIDE:
            context_hidden_dim *= 2
            self.context_control = torch.nn.Sequential(
                torch.nn.LayerNorm(canonical_feature_dim),
                torch.nn.Linear(canonical_feature_dim, context_hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(float(self.config.dropout)),
                torch.nn.Linear(context_hidden_dim, context_hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(float(self.config.dropout)),
                torch.nn.Linear(context_hidden_dim, int(self.config.part_embed_dim)),
            )
            self.context_control_gate = torch.nn.Sequential(
                torch.nn.LayerNorm(canonical_feature_dim),
                torch.nn.Linear(canonical_feature_dim, context_hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(context_hidden_dim, context_hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(context_hidden_dim, 1),
            )
        else:
            self.context_control = torch.nn.Sequential(
                torch.nn.LayerNorm(canonical_feature_dim),
                torch.nn.Linear(canonical_feature_dim, context_hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(float(self.config.dropout)),
                torch.nn.Linear(context_hidden_dim, int(self.config.part_embed_dim)),
            )
            self.context_control_gate = torch.nn.Sequential(
                torch.nn.LayerNorm(canonical_feature_dim),
                torch.nn.Linear(canonical_feature_dim, context_hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(context_hidden_dim, 1),
            )
        torch.nn.init.zeros_(self.context_control[-1].weight)
        torch.nn.init.zeros_(self.context_control[-1].bias)
        torch.nn.init.zeros_(self.context_control_gate[-1].weight)
        torch.nn.init.constant_(self.context_control_gate[-1].bias, float(self.config.gate_bias_init))
        part_embed_dim = int(self.config.part_embed_dim)
        adapter_heads = max(1, min(8, part_embed_dim // 32))
        while part_embed_dim % adapter_heads != 0 and adapter_heads > 1:
            adapter_heads -= 1
        self.part_only_adapter = torch.nn.Sequential(
            torch.nn.LayerNorm(part_embed_dim),
            torch.nn.Linear(part_embed_dim, int(self.config.fusion_hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(int(self.config.fusion_hidden_dim), part_embed_dim),
        )
        self.part_only_gate = torch.nn.Sequential(
            torch.nn.LayerNorm(part_embed_dim),
            torch.nn.Linear(part_embed_dim, int(self.config.fusion_hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Linear(int(self.config.fusion_hidden_dim), 1),
        )
        torch.nn.init.zeros_(self.part_only_adapter[-1].weight)
        torch.nn.init.zeros_(self.part_only_adapter[-1].bias)
        torch.nn.init.zeros_(self.part_only_gate[-1].weight)
        torch.nn.init.constant_(self.part_only_gate[-1].bias, float(self.config.gate_bias_init))
        extra_layer = torch.nn.TransformerEncoderLayer(
            d_model=part_embed_dim,
            nhead=int(adapter_heads),
            dim_feedforward=max(part_embed_dim * 2, int(self.config.fusion_hidden_dim) * 2),
            dropout=float(self.config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.extra_part_block = torch.nn.TransformerEncoder(extra_layer, num_layers=1)
        self.extra_part_block_out = torch.nn.Linear(part_embed_dim, part_embed_dim)
        torch.nn.init.zeros_(self.extra_part_block_out.weight)
        torch.nn.init.zeros_(self.extra_part_block_out.bias)
        if self.variant in (
            ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
            ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_PLUS_FEATURE_MLP_ADAPTER,
        ):
            self.feature_delta_adapter = ArchitectureViewFeatureDeltaMLP(self.feature_config, self.config)
        else:
            self.feature_delta_adapter = torch.nn.Identity()
        generator = torch.Generator()
        generator.manual_seed(int(self.config.random_control_seed))
        self.register_buffer(
            "random_view_feature_permutation",
            torch.randperm(int(self.config.raw_token_dim), generator=generator),
            persistent=False,
        )
        self.register_buffer(
            "shuffled_context_feature_permutation",
            torch.randperm(int(canonical_feature_dim), generator=generator),
            persistent=False,
        )
        self.embed_module_name, self.embed_module = _resolve_embed_module(self.part_model)
        self.baseline_checkpoint_report: dict[str, Any] | None = None
        self.init_logit_diff_vs_baseline: dict[str, Any] | None = None
        self._pending_delta_h: Any | None = None
        self._pending_particle_mask: Any | None = None
        self._last_delta_h: Any | None = None
        self._last_part_embedding: Any | None = None
        self._last_injection_summary: dict[str, Any] = {}
        self._freeze_inactive_adapter_modules()

    @property
    def output_contract(self) -> str:
        return ARCHITECTURE_VIEW_MODEL_CONTRACT

    @property
    def uses_reference_part_backbone(self) -> bool:
        return isinstance(self.part_model, ParticleTransformerHLTClassifier)

    def no_weight_decay(self) -> set[str]:
        names: set[str] = set()
        if hasattr(self.part_model, "no_weight_decay"):
            names.update({f"part_model.{name}" for name in self.part_model.no_weight_decay()})
        return names

    def adapter_module_map(self) -> dict[str, Any]:
        return {
            "view_module": self.view_module,
            "context_control": self.context_control,
            "context_control_gate": self.context_control_gate,
            "part_only_adapter": self.part_only_adapter,
            "part_only_gate": self.part_only_gate,
            "extra_part_block": self.extra_part_block,
            "extra_part_block_out": self.extra_part_block_out,
            "feature_delta_adapter": self.feature_delta_adapter,
        }

    def active_adapter_module_names(self) -> tuple[str, ...]:
        if self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_PLUS_FEATURE_MLP_ADAPTER:
            return ("feature_delta_adapter", "context_control", "context_control_gate")
        if self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES:
            return ("feature_delta_adapter",)
        if self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK:
            return ("extra_part_block", "extra_part_block_out")
        if self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER:
            return ("part_only_adapter", "part_only_gate")
        if self.effective_variant == ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL:
            return ("context_control", "context_control_gate")
        if self.effective_variant == ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK:
            return ()
        return ("view_module",) if self.config.enabled_views else ()

    def adapter_modules(self, *, active_only: bool = True) -> tuple[Any, ...]:
        modules = self.adapter_module_map()
        if bool(active_only):
            return tuple(modules[name] for name in self.active_adapter_module_names())
        return tuple(modules.values())

    def _freeze_inactive_adapter_modules(self) -> None:
        active_names = set(self.active_adapter_module_names())
        for name, module in self.adapter_module_map().items():
            trainable = name in active_names
            for parameter in module.parameters():
                parameter.requires_grad_(bool(trainable))

    def active_parameters(self):
        yield from self.part_model.parameters()
        for module in self.adapter_modules(active_only=True):
            yield from module.parameters()

    @staticmethod
    def _parameter_count(modules: tuple[Any, ...], *, trainable_only: bool = False) -> int:
        seen: set[int] = set()
        count = 0
        for module in modules:
            for parameter in module.parameters():
                parameter_id = id(parameter)
                if parameter_id in seen:
                    continue
                seen.add(parameter_id)
                if trainable_only and not bool(parameter.requires_grad):
                    continue
                count += int(parameter.numel())
        return int(count)

    def parameter_accounting(self) -> dict[str, Any]:
        active_modules = (self.part_model, *self.adapter_modules(active_only=True))
        all_modules = (self,)
        part_modules = (self.part_model,)
        active_adapter_modules = self.adapter_modules(active_only=True)
        all_adapter_modules = self.adapter_modules(active_only=False)
        part_head_modules = self._part_head_parameter_modules()
        active_adapter_params = self._parameter_count(active_adapter_modules)
        all_adapter_params = self._parameter_count(all_adapter_modules)
        return {
            "total_params": self._parameter_count(active_modules),
            "trainable_params": self._parameter_count(active_modules, trainable_only=True),
            "actual_allocated_params": self._parameter_count(all_modules),
            "actual_trainable_allocated_params": self._parameter_count(all_modules, trainable_only=True),
            "part_params": self._parameter_count(part_modules),
            "trainable_part_params": self._parameter_count(part_modules, trainable_only=True),
            "part_head_params": self._parameter_count(part_head_modules),
            "trainable_part_head_params": self._parameter_count(part_head_modules, trainable_only=True),
            "adapter_params": active_adapter_params,
            "trainable_adapter_params": self._parameter_count(active_adapter_modules, trainable_only=True),
            "all_adapter_params": all_adapter_params,
            "dormant_adapter_params": max(0, int(all_adapter_params) - int(active_adapter_params)),
            "active_adapter_module_names": list(self.active_adapter_module_names()),
            "dormant_adapter_module_names": [
                name for name in self.adapter_module_map() if name not in set(self.active_adapter_module_names())
            ],
        }

    def _part_head_parameter_modules(self) -> tuple[Any, ...]:
        torch = require_torch()
        matches: list[Any] = []
        for name, module in self.part_model.named_modules():
            lower = name.lower()
            if lower.endswith("fc") or ".fc" in lower or "classifier" in lower or "head" in lower:
                matches.append(module)
        if not matches:
            return ()
        return tuple(matches)

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "contract": ARCHITECTURE_VIEW_MODEL_CONTRACT,
            "step": ARCHITECTURE_VIEW_MODEL_STEP,
            "variant": str(self.variant),
            "effective_variant": str(self.effective_variant),
            "config": self.config.to_dict(),
            "feature_config": self.feature_config.to_dict(),
            "part_model_config": dict(getattr(self.part_model, "config", {}) or {}),
            "part_model_size": str(self.part_model_size),
            "enabled_views": list(self.config.enabled_views),
            "embed_module_name": str(self.embed_module_name),
            "baseline_checkpoint": dict(self.baseline_checkpoint_report or {}),
            "init_logit_diff_vs_baseline": dict(self.init_logit_diff_vs_baseline or {}),
            "uses_reference_part_backbone": bool(self.uses_reference_part_backbone),
            "variant_spec": self.variant_spec.to_dict(),
            "variant_behavior": self.variant_behavior(),
            "parameter_accounting": self.parameter_accounting(),
        }

    def variant_behavior(self) -> dict[str, Any]:
        uses_extra_part_block = self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK
        uses_part_only_adapter = self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER
        uses_combined_lc_plus_feature_mlp = (
            self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_PLUS_FEATURE_MLP_ADAPTER
        )
        uses_lc_mlp_delta_features = self.variant in (
            ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
            ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_PLUS_FEATURE_MLP_ADAPTER,
        )
        uses_larger_part = self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART
        uses_wide_feature_mlp = self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER_WIDE
        uses_frozen_feature_mlp = self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER
        uses_shuffled_feature_mlp = self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER
        uses_context_mlp = (
            self.effective_variant == ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL
            and not uses_part_only_adapter
        )
        forces_zero_delta = (
            self.effective_variant == ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK
            and not uses_extra_part_block
            and not uses_part_only_adapter
            and not uses_lc_mlp_delta_features
        )
        return {
            "variant": str(self.variant),
            "effective_variant": str(self.effective_variant),
            "variant_spec": self.variant_spec.to_dict(),
            "input_source": str(self.variant_spec.input_source),
            "enabled_views": list(self.config.enabled_views),
            "active_adapter_module_names": list(self.active_adapter_module_names()),
            "dormant_adapter_module_names": [
                name for name in self.adapter_module_map() if name not in set(self.active_adapter_module_names())
            ],
            "uses_architecture_views": bool(self.config.enabled_views)
            and self.effective_variant != ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
            "uses_randomized_view_semantics": self.effective_variant == ARCHITECTURE_VIEW_VARIANT_RANDOM_VIEW_CONTROL,
            "uses_context_mlp_control": uses_context_mlp,
            "uses_lc_mlp_delta_features": bool(uses_lc_mlp_delta_features),
            "uses_combined_lc_plus_feature_mlp_adapter": bool(uses_combined_lc_plus_feature_mlp),
            "combined_adapter": bool(uses_combined_lc_plus_feature_mlp),
            "input_delta_adapter_active": bool(uses_lc_mlp_delta_features),
            "embedding_delta_adapter_active": bool(
                uses_combined_lc_plus_feature_mlp or (not forces_zero_delta and not uses_lc_mlp_delta_features)
            ),
            "input_feature_delta_policy": "bounded_tanh_feature_scales"
            if uses_lc_mlp_delta_features
            else "none",
            "uses_wide_feature_mlp_adapter": bool(uses_wide_feature_mlp),
            "uses_frozen_part_feature_adapter": bool(uses_frozen_feature_mlp),
            "uses_shuffled_feature_adapter": bool(uses_shuffled_feature_mlp),
            "feature_shuffle_policy": "cross_particle_roll_plus_feature_permutation"
            if uses_shuffled_feature_mlp
            else "none",
            "uses_part_only_adapter": bool(uses_part_only_adapter),
            "uses_extra_part_block": bool(uses_extra_part_block),
            "uses_larger_part_backbone": bool(uses_larger_part),
            "part_model_size": str(self.part_model_size),
            "forces_zero_delta": bool(forces_zero_delta),
            "injects_embedding_delta": bool(
                uses_combined_lc_plus_feature_mlp or (not forces_zero_delta and not uses_lc_mlp_delta_features)
            ),
            "adapts_input_features": bool(uses_lc_mlp_delta_features),
        }

    def build_canonical_inputs(
        self,
        tokens: Any,
        mask: Any,
        *,
        weights: Any | None = None,
        max_constits: int | None = None,
        weight_threshold: float = 0.0,
    ) -> LocalCompressionCanonicalInputs:
        if max_constits is None:
            max_constits = int(tokens.shape[1])
        return build_local_compression_canonical_inputs(
            tokens,
            mask,
            weights=weights,
            max_constits=int(max_constits),
            weight_threshold=float(weight_threshold),
            config=self.feature_config,
        )

    def _embed_injection_hook(self, module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        del module, inputs
        torch = require_torch()
        if self._pending_delta_h is None and self._pending_particle_mask is None:
            return output
        output_tensor = output[0] if isinstance(output, tuple) else output
        delta = self._pending_delta_h
        adapter_kind = "precomputed_delta"
        gate_mean: float | None = None
        if delta is None:
            if self._pending_particle_mask is None:
                return output
            mask = self._pending_particle_mask.bool()
            batch_size, num_particles = mask.shape
            base = _batch_first_from_embed_output(
                output_tensor,
                batch=int(batch_size),
                particles=int(num_particles),
                dim=int(self.config.part_embed_dim),
            )
            local_mask = mask[:, : int(base.shape[1])]
            valid = local_mask[:, :, None].to(dtype=base.dtype)
            if self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER:
                gate = torch.sigmoid(self.part_only_gate(base))
                delta = gate * self.part_only_adapter(base)
                gate_denom = local_mask.sum().clamp_min(1)
                gate_mean = float(
                    ((gate.squeeze(-1) * local_mask.to(dtype=gate.dtype)).sum() / gate_denom).detach().cpu().item()
                )
                adapter_kind = "part_embedding_mlp"
            elif self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK:
                key_padding_mask = ~local_mask
                all_padded = key_padding_mask.all(dim=1)
                if bool(all_padded.any().item()):
                    key_padding_mask = key_padding_mask.clone()
                    key_padding_mask[all_padded, 0] = False
                refined = self.extra_part_block(base, src_key_padding_mask=key_padding_mask)
                if bool(all_padded.any().item()):
                    refined = refined.clone()
                    refined[all_padded] = 0.0
                delta = self.extra_part_block_out(refined)
                gate_mean = 1.0
                adapter_kind = "extra_part_block"
            else:
                delta = torch.zeros_like(base)
                adapter_kind = "no_injection"
            delta = delta * valid
        self._last_delta_h = delta
        embedding_norm_mean: float | None = None
        embedding_norm_p90: float | None = None
        delta_to_embedding_norm_ratio: float | None = None
        try:
            batch_size, num_particles = int(delta.shape[0]), int(delta.shape[1])
            base_for_diag = _batch_first_from_embed_output(
                output_tensor,
                batch=batch_size,
                particles=num_particles,
                dim=int(self.config.part_embed_dim),
            )
            if self._pending_particle_mask is not None:
                diag_mask = self._pending_particle_mask.bool()[:, : int(base_for_diag.shape[1])]
            else:
                diag_mask = torch.ones(
                    int(base_for_diag.shape[0]),
                    int(base_for_diag.shape[1]),
                    dtype=torch.bool,
                    device=base_for_diag.device,
                )
            if bool(diag_mask.any()):
                embedding_norm = base_for_diag.norm(dim=-1)[diag_mask]
                delta_norm = delta[:, : int(base_for_diag.shape[1])].norm(dim=-1)[diag_mask]
                embedding_norm_mean_t = embedding_norm.mean()
                embedding_norm_mean = float(embedding_norm_mean_t.detach().cpu().item())
                embedding_norm_p90 = float(_tensor_quantile(embedding_norm, 0.90).detach().cpu().item())
                delta_to_embedding_norm_ratio = float(
                    (delta_norm.mean() / embedding_norm_mean_t.clamp_min(1.0e-12)).detach().cpu().item()
                )
        except Exception:
            embedding_norm_mean = None
            embedding_norm_p90 = None
            delta_to_embedding_norm_ratio = None
        if isinstance(output, tuple):
            first = output[0] + _delta_for_embed_output(delta, output[0])
            output = (first, *output[1:])
            output_shape = tuple(first.shape)
            output_tensor_for_representation = first
        else:
            residual = _delta_for_embed_output(delta, output)
            output = output + residual
            output_shape = tuple(output.shape)
            output_tensor_for_representation = output
        self._last_part_embedding = _batch_first_from_embed_output(
            output_tensor_for_representation,
            batch=int(delta.shape[0]),
            particles=int(delta.shape[1]),
            dim=int(self.config.part_embed_dim),
        )
        self._last_injection_summary = {
            "embed_module_name": str(self.embed_module_name),
            "delta_h_shape": list(delta.shape),
            "embed_output_shape": list(output_shape),
            "delta_h_abs_max": float(delta.detach().abs().max().cpu().item()) if int(delta.numel()) else 0.0,
            "adapter_kind": str(adapter_kind),
            "gate_mean": gate_mean,
            "embedding_norm_mean": embedding_norm_mean,
            "embedding_norm_p90": embedding_norm_p90,
            "delta_to_embedding_norm_ratio": delta_to_embedding_norm_ratio,
            "injection_applied": bool(adapter_kind != "no_injection"),
        }
        return output

    def _part_forward_with_delta(self, canonical: LocalCompressionCanonicalInputs, delta_h: Any | None) -> tuple[Any, Any]:
        torch = require_torch()
        self._last_part_embedding = None
        self._pending_delta_h = delta_h
        self._pending_particle_mask = canonical.particle_mask
        self._last_delta_h = delta_h
        self._last_injection_summary = {
            "embed_module_name": str(self.embed_module_name),
            "delta_h_shape": None if delta_h is None else list(delta_h.shape),
            "delta_h_abs_max": 0.0
            if delta_h is None or not int(delta_h.numel())
            else float(delta_h.detach().abs().max().cpu().item()),
            "adapter_kind": "hook_computed_delta" if delta_h is None else "precomputed_delta",
            "injection_applied": False,
        }
        handle = self.embed_module.register_forward_hook(self._embed_injection_hook)
        try:
            logits = self.part_model(
                canonical.points,
                canonical.features,
                canonical.lorentz_vectors,
                canonical.mask,
            )
        finally:
            handle.remove()
            self._pending_delta_h = None
            self._pending_particle_mask = None
        actual_delta = self._last_delta_h
        if actual_delta is None:
            actual_delta = torch.zeros(
                int(canonical.particle_mask.shape[0]),
                int(canonical.particle_mask.shape[1]),
                int(self.config.part_embed_dim),
                dtype=canonical.features.dtype,
                device=canonical.features.device,
            )
        return _nan_to_num_torch(logits.float()), actual_delta

    def _context_control_view_output(self, canonical: LocalCompressionCanonicalInputs) -> ArchitectureViewFusionOutput:
        torch = require_torch()
        feature_rows = canonical.feature_rows()
        diagnostics: dict[str, Any] = {
            "context_mlp_control": True,
            "baseline_recovery_zero_projection": True,
            "feature_mlp_adapter_type": str(self.variant_spec.adapter_type),
        }
        if self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER:
            permutation = self.shuffled_context_feature_permutation.to(device=feature_rows.device)
            feature_rows = feature_rows.index_select(dim=-1, index=permutation)
            flat_rows = feature_rows.reshape(-1, int(feature_rows.shape[-1]))
            if int(flat_rows.shape[0]) > 1:
                flat_rows = flat_rows.roll(shifts=max(1, int(flat_rows.shape[0]) // 2), dims=0)
            feature_rows = flat_rows.reshape_as(feature_rows)
            diagnostics.update(
                {
                    "shuffled_feature_adapter": True,
                    "feature_shuffle_policy": "cross_particle_roll_plus_feature_permutation",
                }
            )
        elif self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER_WIDE:
            diagnostics["wide_feature_mlp_adapter"] = True
        elif self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER:
            diagnostics["frozen_part_feature_adapter"] = True
        gate = torch.sigmoid(self.context_control_gate(feature_rows))
        adapter_output = self.context_control(feature_rows)
        delta_h = gate * adapter_output
        delta_h = delta_h * canonical.particle_mask[:, :, None].to(dtype=delta_h.dtype)
        gate = gate * canonical.particle_mask[:, :, None].to(dtype=gate.dtype)
        valid_mask = canonical.particle_mask.bool()
        adapter_norm = adapter_output.norm(dim=-1)
        if bool(valid_mask.any()):
            valid_adapter_norm = adapter_norm[valid_mask]
            diagnostics.update(
                {
                    "adapter_output_norm_mean": float(valid_adapter_norm.mean().detach().cpu().item()),
                    "adapter_output_norm_p90": float(
                        _tensor_quantile(valid_adapter_norm, 0.90).detach().cpu().item()
                    ),
                    "adapter_output_norm_max": float(valid_adapter_norm.max().detach().cpu().item()),
                }
            )
        else:
            diagnostics.update(
                {
                    "adapter_output_norm_mean": 0.0,
                    "adapter_output_norm_p90": 0.0,
                    "adapter_output_norm_max": 0.0,
                }
            )
        combined = torch.zeros(
            int(feature_rows.shape[0]),
            int(feature_rows.shape[1]),
            0,
            dtype=feature_rows.dtype,
            device=feature_rows.device,
        )
        return ArchitectureViewFusionOutput(
            view_embeddings={},
            combined_view=combined,
            delta_h=delta_h,
            gate=gate,
            mask=canonical.particle_mask,
            diagnostics=diagnostics,
        )

    def _combined_adapter_diagnostics(
        self,
        feature_delta_output: ArchitectureViewFeatureDeltaOutput,
        view_output: ArchitectureViewFusionOutput,
    ) -> dict[str, Any]:
        torch = require_torch()
        delta_f = feature_delta_output.delta_F_rows.detach().float()
        delta_h = view_output.delta_h.detach().float()
        mask_f = feature_delta_output.mask.detach().bool()
        mask_h = align_particle_mask_to_length(view_output.mask.detach().bool(), int(delta_h.shape[1]))
        particles = min(int(delta_f.shape[1]), int(delta_h.shape[1]), int(mask_f.shape[1]), int(mask_h.shape[1]))
        if particles <= 0:
            return {
                "combined_adapter": True,
                "input_delta_adapter_active": True,
                "embedding_delta_adapter_active": True,
                "combined_delta_F_delta_h_corr": None,
                "fraction_jets_delta_F_only": 0.0,
                "fraction_jets_delta_h_only": 0.0,
                "fraction_jets_both_adapters_active": 0.0,
            }
        delta_f_norm = delta_f[:, :particles].norm(dim=-1)
        delta_h_norm = delta_h[:, :particles].norm(dim=-1)
        valid = mask_f[:, :particles] & mask_h[:, :particles]
        batch_valid = valid.any(dim=1)
        if bool(batch_valid.any()):
            jet_delta_f = (delta_f_norm * valid.to(dtype=delta_f_norm.dtype)).sum(dim=1) / valid.sum(
                dim=1
            ).clamp_min(1).to(dtype=delta_f_norm.dtype)
            jet_delta_h = (delta_h_norm * valid.to(dtype=delta_h_norm.dtype)).sum(dim=1) / valid.sum(
                dim=1
            ).clamp_min(1).to(dtype=delta_h_norm.dtype)
            jet_delta_f = jet_delta_f[batch_valid]
            jet_delta_h = jet_delta_h[batch_valid]
            f_active = jet_delta_f > 1.0e-8
            h_active = jet_delta_h > 1.0e-8
            if int(jet_delta_f.numel()) > 1:
                centered_f = jet_delta_f - jet_delta_f.mean()
                centered_h = jet_delta_h - jet_delta_h.mean()
                denom = centered_f.norm() * centered_h.norm()
                corr = None if float(denom.detach().cpu().item()) <= 1.0e-12 else float(
                    ((centered_f * centered_h).sum() / denom).detach().cpu().item()
                )
            else:
                corr = None
            denom_jets = float(jet_delta_f.numel())
            return {
                "combined_adapter": True,
                "input_delta_adapter_active": True,
                "embedding_delta_adapter_active": True,
                "combined_delta_F_mean_per_jet": float(jet_delta_f.mean().detach().cpu().item()),
                "combined_delta_h_mean_per_jet": float(jet_delta_h.mean().detach().cpu().item()),
                "combined_delta_F_delta_h_corr": corr,
                "fraction_jets_delta_F_only": float((f_active & ~h_active).float().sum().detach().cpu().item())
                / denom_jets,
                "fraction_jets_delta_h_only": float((~f_active & h_active).float().sum().detach().cpu().item())
                / denom_jets,
                "fraction_jets_both_adapters_active": float((f_active & h_active).float().sum().detach().cpu().item())
                / denom_jets,
            }
        return {
            "combined_adapter": True,
            "input_delta_adapter_active": True,
            "embedding_delta_adapter_active": True,
            "combined_delta_F_delta_h_corr": None,
            "fraction_jets_delta_F_only": 0.0,
            "fraction_jets_delta_h_only": 0.0,
            "fraction_jets_both_adapters_active": 0.0,
        }

    def _zero_view_output(
        self,
        canonical: LocalCompressionCanonicalInputs,
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> ArchitectureViewFusionOutput:
        torch = require_torch()
        batch_size, num_particles = canonical.particle_mask.shape
        combined = torch.zeros(
            int(batch_size),
            int(num_particles),
            0,
            dtype=canonical.features.dtype,
            device=canonical.features.device,
        )
        delta_h = torch.zeros(
            int(batch_size),
            int(num_particles),
            int(self.config.part_embed_dim),
            dtype=canonical.features.dtype,
            device=canonical.features.device,
        )
        gate = torch.zeros(
            int(batch_size),
            int(num_particles),
            1,
            dtype=canonical.features.dtype,
            device=canonical.features.device,
        )
        return ArchitectureViewFusionOutput(
            view_embeddings={},
            combined_view=combined,
            delta_h=delta_h,
            gate=gate,
            mask=canonical.particle_mask,
            diagnostics=dict(diagnostics or {}),
        )

    def _tokens_for_view_branches(self, canonical: LocalCompressionCanonicalInputs) -> Any:
        tokens = canonical.selected_tokens
        if self.effective_variant == ARCHITECTURE_VIEW_VARIANT_RANDOM_VIEW_CONTROL:
            permutation = self.random_view_feature_permutation.to(device=tokens.device)
            return tokens.index_select(dim=-1, index=permutation)
        return tokens

    def forward_outputs(
        self,
        tokens_or_batch: Any,
        mask: Any | None = None,
        *,
        weights: Any | None = None,
        max_constits: int | None = None,
        weight_threshold: float = 0.0,
    ) -> ArchitectureViewResidualPartOutput:
        raw_tokens, raw_mask = _coerce_tokens_and_mask(tokens_or_batch, mask)
        canonical = self.build_canonical_inputs(
            raw_tokens,
            raw_mask,
            weights=weights,
            max_constits=max_constits,
            weight_threshold=float(weight_threshold),
        )
        if self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER:
            view_output = self._zero_view_output(
                canonical,
                diagnostics={
                    "part_only_adapter": True,
                    "baseline_recovery_zero_projection": True,
                },
            )
            delta_for_part = None
        elif self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK:
            view_output = self._zero_view_output(
                canonical,
                diagnostics={
                    "extra_part_block": True,
                    "baseline_recovery_zero_projection": True,
                },
            )
            delta_for_part = None
        elif self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART:
            view_output = self._zero_view_output(
                canonical,
                diagnostics={
                    "larger_part_capacity_control": True,
                    "no_embedding_injection": True,
                },
            )
            delta_for_part = None
        elif self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_PLUS_FEATURE_MLP_ADAPTER:
            feature_delta_output = self.feature_delta_adapter(canonical)
            adapted_canonical = canonical.with_features(feature_delta_output.adapted_feature_rows)
            view_output = self._context_control_view_output(adapted_canonical)
            view_output = replace(
                view_output,
                diagnostics={
                    **view_output.diagnostics,
                    "lc_plus_feature_mlp_adapter": True,
                    "baseline_recovery_zero_projection": True,
                    "feature_input_delta_adapter": True,
                    "embedding_delta_adapter": True,
                    **self._combined_adapter_diagnostics(feature_delta_output, view_output),
                },
            )
            logits, actual_delta_h = self._part_forward_with_delta(adapted_canonical, view_output.delta_h)
            if actual_delta_h is not view_output.delta_h:
                view_output = replace(view_output, delta_h=actual_delta_h)
            return ArchitectureViewResidualPartOutput(
                logits=logits,
                canonical_inputs=adapted_canonical,
                view_output=view_output,
                config=self.config,
                variant=str(self.variant),
                part_model_class=type(self.part_model).__name__,
                embed_module_name=str(self.embed_module_name),
                injection_summary=dict(self._last_injection_summary),
                feature_delta_output=feature_delta_output,
                part_embedding=self._last_part_embedding,
                variant_behavior=self.variant_behavior(),
                parameter_accounting=self.parameter_accounting(),
                baseline_checkpoint_report=self.baseline_checkpoint_report,
                init_logit_diff_vs_baseline=self.init_logit_diff_vs_baseline,
            )
        elif self.variant == ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES:
            feature_delta_output = self.feature_delta_adapter(canonical)
            adapted_canonical = canonical.with_features(feature_delta_output.adapted_feature_rows)
            view_output = self._zero_view_output(
                adapted_canonical,
                diagnostics={
                    "lc_mlp_delta_features": True,
                    "baseline_recovery_zero_projection": True,
                    "feature_input_delta_adapter": True,
                    "no_embedding_injection": True,
                },
            )
            logits, actual_delta_h = self._part_forward_with_delta(adapted_canonical, None)
            return ArchitectureViewResidualPartOutput(
                logits=logits,
                canonical_inputs=adapted_canonical,
                view_output=view_output,
                config=self.config,
                variant=str(self.variant),
                part_model_class=type(self.part_model).__name__,
                embed_module_name=str(self.embed_module_name),
                injection_summary=dict(self._last_injection_summary),
                feature_delta_output=feature_delta_output,
                part_embedding=self._last_part_embedding,
                variant_behavior=self.variant_behavior(),
                parameter_accounting=self.parameter_accounting(),
                baseline_checkpoint_report=self.baseline_checkpoint_report,
                init_logit_diff_vs_baseline=self.init_logit_diff_vs_baseline,
            )
        elif self.effective_variant == ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK:
            view_output = self._zero_view_output(
                canonical,
                diagnostics={
                    "baseline_recheck": True,
                    "no_embedding_injection": True,
                },
            )
            delta_for_part = None
        elif self.effective_variant == ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL:
            view_output = self._context_control_view_output(canonical)
            delta_for_part = view_output.delta_h
        else:
            view_tokens = self._tokens_for_view_branches(canonical)
            view_output = self.view_module(view_tokens, canonical.particle_mask)
            delta_for_part = view_output.delta_h
        logits, actual_delta_h = self._part_forward_with_delta(canonical, delta_for_part)
        if actual_delta_h is not view_output.delta_h:
            view_output = replace(
                view_output,
                delta_h=actual_delta_h,
                diagnostics={
                    **view_output.diagnostics,
                    "hook_computed_delta": bool(delta_for_part is None and self.variant in (
                        ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK,
                        ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER,
                    )),
                },
            )
        return ArchitectureViewResidualPartOutput(
            logits=logits,
            canonical_inputs=canonical,
            view_output=view_output,
            config=self.config,
            variant=str(self.variant),
            part_model_class=type(self.part_model).__name__,
            embed_module_name=str(self.embed_module_name),
            injection_summary=dict(self._last_injection_summary),
            feature_delta_output=None,
            part_embedding=self._last_part_embedding,
            variant_behavior=self.variant_behavior(),
            parameter_accounting=self.parameter_accounting(),
            baseline_checkpoint_report=self.baseline_checkpoint_report,
            init_logit_diff_vs_baseline=self.init_logit_diff_vs_baseline,
        )

    def forward(
        self,
        tokens_or_batch: Any,
        mask: Any | None = None,
        *,
        return_outputs: bool = False,
        return_diagnostics: bool = False,
        weights: Any | None = None,
        max_constits: int | None = None,
        weight_threshold: float = 0.0,
    ) -> Any:
        output = self.forward_outputs(
            tokens_or_batch,
            mask,
            weights=weights,
            max_constits=max_constits,
            weight_threshold=weight_threshold,
        )
        if bool(return_outputs):
            return output
        if bool(return_diagnostics):
            return output.logits, output.diagnostics()
        return output.logits


def build_architecture_view_residual_part(
    config: ArchitectureViewConfig | Mapping[str, Any] | None = None,
    *,
    variant: str = "av_all_views",
    part_model: Any | None = None,
    feature_config: LocalCompressionFeatureConfig | Mapping[str, Any] | None = None,
) -> ArchitectureViewResidualParT:
    return ArchitectureViewResidualParT(
        config,
        variant=variant,
        part_model=part_model,
        feature_config=feature_config,
    )


__all__ = [
    "ARCHITECTURE_VIEW_MODEL_CONTRACT",
    "ARCHITECTURE_VIEW_MODEL_STEP",
    "ArchitectureViewResidualParT",
    "ArchitectureViewResidualPartOutput",
    "build_architecture_view_residual_part",
]
