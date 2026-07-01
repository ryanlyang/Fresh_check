"""Embedding-injected exact-HLT-ParT wrapper for architecture views.

Step 2 keeps the real HLT ParT as the scoring backbone.  Architecture-view
branches produce only a gated residual in ParT's particle embedding space.  A
zero-initialized residual projection recovers the exact baseline logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, build_hlt_classifier, require_torch

from teacher_logit_reco.local_compression_part.config import LocalCompressionFeatureConfig
from teacher_logit_reco.local_compression_part.features import (
    LocalCompressionCanonicalInputs,
    build_local_compression_canonical_inputs,
)

from .config import (
    ARCHITECTURE_VIEW_PART_CONTRACT,
    ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
    ARCHITECTURE_VIEW_VARIANT_RANDOM_VIEW_CONTROL,
    ArchitectureViewConfig,
    enabled_views_for_variant,
    normalize_architecture_view_variant,
)
from .fusion import ArchitectureViewFusionOutput, ArchitectureViewParticleViews

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
    baseline_checkpoint_report: Mapping[str, Any] | None = None
    init_logit_diff_vs_baseline: Mapping[str, Any] | None = None

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
        mask = self.view_output.mask.detach().bool()
        valid = mask.to(dtype=delta.dtype)
        denom = valid.sum().clamp_min(1.0)
        delta_norm = delta.norm(dim=-1)
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
            "gate_mean": float(((self.view_output.gate.squeeze(-1) * valid).sum() / denom).detach().cpu().item()),
            "baseline_recoverable_at_zero_delta": bool(self.baseline_recoverable_at_zero_delta),
            "view_fusion": self.view_output.summary(),
            "embed_injection": dict(self.injection_summary),
            "baseline_checkpoint": checkpoint or None,
            "init_logit_diff_vs_baseline": init_diff or None,
        }
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
        view_config = _normalize_view_config(config)
        view_config = ArchitectureViewConfig.from_dict(
            {**view_config.to_dict(), "enabled_views": enabled_views_for_variant(self.variant)}
        )
        self.config = view_config
        self.feature_config = (
            feature_config
            if isinstance(feature_config, LocalCompressionFeatureConfig)
            else LocalCompressionFeatureConfig(**dict(feature_config or {}))
        )
        self.part_model = part_model or build_hlt_classifier(num_classes=2)
        if not isinstance(self.part_model, ParticleTransformerHLTClassifier):
            raise ValueError("ArchitectureViewResidualParT requires the real ParticleTransformerHLTClassifier backbone")
        self.view_module = ArchitectureViewParticleViews(self.config, enabled_views=self.config.enabled_views)
        torch = require_torch()
        canonical_feature_dim = len(self.feature_config.canonical_feature_names)
        self.context_control = torch.nn.Sequential(
            torch.nn.LayerNorm(canonical_feature_dim),
            torch.nn.Linear(canonical_feature_dim, int(self.config.fusion_hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(int(self.config.fusion_hidden_dim), int(self.config.part_embed_dim)),
        )
        self.context_control_gate = torch.nn.Sequential(
            torch.nn.LayerNorm(canonical_feature_dim),
            torch.nn.Linear(canonical_feature_dim, int(self.config.fusion_hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Linear(int(self.config.fusion_hidden_dim), 1),
        )
        torch.nn.init.zeros_(self.context_control[-1].weight)
        torch.nn.init.zeros_(self.context_control[-1].bias)
        torch.nn.init.zeros_(self.context_control_gate[-1].weight)
        torch.nn.init.constant_(self.context_control_gate[-1].bias, float(self.config.gate_bias_init))
        generator = torch.Generator()
        generator.manual_seed(int(self.config.random_control_seed))
        self.register_buffer(
            "random_view_feature_permutation",
            torch.randperm(int(self.config.raw_token_dim), generator=generator),
            persistent=False,
        )
        self.embed_module_name, self.embed_module = _resolve_embed_module(self.part_model)
        self.baseline_checkpoint_report: dict[str, Any] | None = None
        self.init_logit_diff_vs_baseline: dict[str, Any] | None = None
        self._pending_delta_h: Any | None = None
        self._last_injection_summary: dict[str, Any] = {}

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

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "contract": ARCHITECTURE_VIEW_MODEL_CONTRACT,
            "step": ARCHITECTURE_VIEW_MODEL_STEP,
            "variant": str(self.variant),
            "config": self.config.to_dict(),
            "feature_config": self.feature_config.to_dict(),
            "part_model_config": dict(getattr(self.part_model, "config", {}) or {}),
            "enabled_views": list(self.config.enabled_views),
            "embed_module_name": str(self.embed_module_name),
            "baseline_checkpoint": dict(self.baseline_checkpoint_report or {}),
            "init_logit_diff_vs_baseline": dict(self.init_logit_diff_vs_baseline or {}),
            "uses_reference_part_backbone": bool(self.uses_reference_part_backbone),
            "variant_behavior": self.variant_behavior(),
        }

    def variant_behavior(self) -> dict[str, Any]:
        return {
            "variant": str(self.variant),
            "enabled_views": list(self.config.enabled_views),
            "uses_architecture_views": bool(self.config.enabled_views)
            and self.variant != ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
            "uses_randomized_view_semantics": self.variant == ARCHITECTURE_VIEW_VARIANT_RANDOM_VIEW_CONTROL,
            "uses_context_mlp_control": self.variant == ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
            "forces_zero_delta": self.variant == ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
            "injects_embedding_delta": self.variant != ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
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
        if self._pending_delta_h is None:
            return output
        delta = self._pending_delta_h
        if isinstance(output, tuple):
            first = output[0] + _delta_for_embed_output(delta, output[0])
            output = (first, *output[1:])
            output_shape = tuple(first.shape)
        else:
            residual = _delta_for_embed_output(delta, output)
            output = output + residual
            output_shape = tuple(output.shape)
        self._last_injection_summary = {
            "embed_module_name": str(self.embed_module_name),
            "delta_h_shape": list(delta.shape),
            "embed_output_shape": list(output_shape),
            "delta_h_abs_max": float(delta.detach().abs().max().cpu().item()) if int(delta.numel()) else 0.0,
            "injection_applied": True,
        }
        return output

    def _part_forward_with_delta(self, canonical: LocalCompressionCanonicalInputs, delta_h: Any) -> Any:
        self._pending_delta_h = delta_h
        self._last_injection_summary = {
            "embed_module_name": str(self.embed_module_name),
            "delta_h_shape": list(delta_h.shape),
            "delta_h_abs_max": float(delta_h.detach().abs().max().cpu().item()) if int(delta_h.numel()) else 0.0,
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
        return _nan_to_num_torch(logits.float())

    def _context_control_view_output(self, canonical: LocalCompressionCanonicalInputs) -> ArchitectureViewFusionOutput:
        torch = require_torch()
        feature_rows = canonical.feature_rows()
        gate = torch.sigmoid(self.context_control_gate(feature_rows))
        delta_h = gate * self.context_control(feature_rows)
        delta_h = delta_h * canonical.particle_mask[:, :, None].to(dtype=delta_h.dtype)
        gate = gate * canonical.particle_mask[:, :, None].to(dtype=gate.dtype)
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
            diagnostics={
                "context_mlp_control": True,
                "baseline_recovery_zero_projection": True,
            },
        )

    def _tokens_for_view_branches(self, canonical: LocalCompressionCanonicalInputs) -> Any:
        tokens = canonical.selected_tokens
        if self.variant == ARCHITECTURE_VIEW_VARIANT_RANDOM_VIEW_CONTROL:
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
        if self.variant == ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL:
            view_output = self._context_control_view_output(canonical)
        else:
            view_tokens = self._tokens_for_view_branches(canonical)
            view_output = self.view_module(view_tokens, canonical.particle_mask)
        logits = self._part_forward_with_delta(canonical, view_output.delta_h)
        return ArchitectureViewResidualPartOutput(
            logits=logits,
            canonical_inputs=canonical,
            view_output=view_output,
            config=self.config,
            variant=str(self.variant),
            part_model_class=type(self.part_model).__name__,
            embed_module_name=str(self.embed_module_name),
            injection_summary=dict(self._last_injection_summary),
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
