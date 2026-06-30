"""Exact-HLT-ParT wrapper for the local-compression feature adapter.

Step 10 keeps the canonical HLT ParT classifier as the scoring backbone.  The
local-compression branch predicts only a bounded residual delta in the ParT PF
feature space, so a zero-initialized adapter recovers the baseline exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, build_hlt_classifier, require_torch

from .adapter import LocalCompressionDeltaFAdapter, LocalCompressionDeltaFOutput
from .compressor import LocalCompressionCompressorOutput, LocalModalityCompressor
from .config import LOCAL_COMPRESSION_PART_CONTRACT, LocalCompressionPartConfig
from .context import ParticleContextBlock, ParticleContextOutput
from .features import (
    LocalCompressionCanonicalInputs,
    LocalCompressionModalities,
    build_local_compression_modalities,
    build_local_compression_canonical_inputs,
)
from .gates import ContextAwareModalityGate, LocalCompressionGateOutput
from .pooling import LocalCompressionPoolOutput, LocalCompressionProvisionalPooler
from .subtokens import LocalCompressionSubtokenEncoder, LocalCompressionSubtokenOutput

try:  # Keep package imports cheap on machines without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


LOCAL_COMPRESSION_MODEL_STEP = "local_compression_part_step10_model"
LOCAL_COMPRESSION_MODEL_CONTRACT = f"{LOCAL_COMPRESSION_PART_CONTRACT}_exact_part_adapter_model_v1"


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, value.new_full((), float(nan)))


def _normalize_model_config(config: LocalCompressionPartConfig | Mapping[str, Any] | None = None) -> LocalCompressionPartConfig:
    if config is None:
        return LocalCompressionPartConfig()
    if isinstance(config, LocalCompressionPartConfig):
        return config
    return LocalCompressionPartConfig(**dict(config))


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


@dataclass(frozen=True)
class LocalCompressionFeatureAdapterOutput:
    """Full local-compression wrapper output."""

    logits: Any
    canonical_inputs: LocalCompressionCanonicalInputs
    adapted_inputs: LocalCompressionCanonicalInputs
    modalities: LocalCompressionModalities
    subtoken_output: LocalCompressionSubtokenOutput
    compressor_output: LocalCompressionCompressorOutput
    pool_output: LocalCompressionPoolOutput
    context_output: ParticleContextOutput
    gate_output: LocalCompressionGateOutput
    delta_output: LocalCompressionDeltaFOutput
    config: LocalCompressionPartConfig
    uses_reference_part_backbone: bool
    part_model_class: str
    baseline_checkpoint_report: Mapping[str, Any] | None = None
    init_logit_diff_vs_baseline: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        torch = require_torch()
        if int(self.logits.ndim) != 2:
            raise ValueError(f"logits must have shape [batch, classes], got {tuple(self.logits.shape)}")
        if int(self.logits.shape[0]) != int(self.canonical_inputs.batch_size):
            raise ValueError("logits batch size must match canonical inputs")
        if tuple(self.adapted_inputs.feature_rows().shape) != tuple(self.delta_output.adapted_feature_rows.shape):
            raise ValueError("adapted input features must match delta output")
        if not bool(torch.isfinite(self.logits).all()):
            raise ValueError("logits contain non-finite values")

    @property
    def batch_size(self) -> int:
        return int(self.logits.shape[0])

    @property
    def baseline_recoverable_at_zero_delta(self) -> bool:
        return bool(self.config.baseline_recoverable_at_zero_delta and self.config.zero_init_delta_projection)

    @property
    def output_contract(self) -> str:
        return LOCAL_COMPRESSION_MODEL_CONTRACT

    def diagnostics(self) -> dict[str, Any]:
        torch = require_torch()
        delta = self.delta_output.delta_F_rows
        adapted_shift = self.adapted_inputs.feature_rows() - self.canonical_inputs.feature_rows()
        checkpoint = dict(self.baseline_checkpoint_report or {})
        init_diff = dict(self.init_logit_diff_vs_baseline or {})
        diagnostics = {
            "contract": LOCAL_COMPRESSION_MODEL_CONTRACT,
            "logits_shape": list(self.logits.shape),
            "valid_particle_count": int(self.canonical_inputs.particle_mask.detach().cpu().sum().item()),
            "delta_abs_mean": float(delta[self.delta_output.mask].detach().abs().mean().cpu().item())
            if bool(self.delta_output.mask.any().detach().cpu().item())
            else 0.0,
            "delta_abs_max": float(delta.detach().abs().max().cpu().item()),
            "adapted_feature_shift_abs_max": float(adapted_shift.detach().abs().max().cpu().item()),
            "baseline_recoverable_at_zero_delta": bool(self.baseline_recoverable_at_zero_delta),
            "uses_reference_part_backbone": bool(self.uses_reference_part_backbone),
            "part_model_class": str(self.part_model_class),
            "gate_summary": self.gate_output.summary(),
            "delta_summary": self.delta_output.summary(),
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


class LocalCompressionFeatureAdapterParT(_ModuleBase):
    """Local-compression residual feature adapter feeding exact HLT ParT."""

    def __init__(
        self,
        config: LocalCompressionPartConfig | Mapping[str, Any] | None = None,
        *,
        part_model: Any | None = None,
    ) -> None:
        super().__init__()
        self.config = _normalize_model_config(config)
        self.part_model = part_model or build_hlt_classifier(num_classes=int(self.config.num_classes))
        if not isinstance(self.part_model, ParticleTransformerHLTClassifier):
            raise ValueError("LocalCompressionFeatureAdapterParT requires the real ParticleTransformerHLTClassifier backbone")
        self.subtoken_encoder = LocalCompressionSubtokenEncoder(self.config)
        self.compressor = LocalModalityCompressor(self.config)
        self.pooler = LocalCompressionProvisionalPooler(self.config)
        self.context_block = ParticleContextBlock(self.config)
        self.gate = ContextAwareModalityGate(self.config)
        self.adapter = LocalCompressionDeltaFAdapter(self.config)
        self.baseline_checkpoint_report: dict[str, Any] | None = None
        self.init_logit_diff_vs_baseline: dict[str, Any] | None = None

    @property
    def uses_reference_part_backbone(self) -> bool:
        return isinstance(self.part_model, ParticleTransformerHLTClassifier)

    @property
    def baseline_recoverable_at_zero_delta(self) -> bool:
        return bool(self.config.baseline_recoverable_at_zero_delta and self.config.zero_init_delta_projection)

    @property
    def output_contract(self) -> str:
        return LOCAL_COMPRESSION_MODEL_CONTRACT

    def no_weight_decay(self) -> set[str]:
        names: set[str] = set()
        if hasattr(self.part_model, "no_weight_decay"):
            names.update({f"part_model.{name}" for name in self.part_model.no_weight_decay()})
        return names

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "contract": LOCAL_COMPRESSION_MODEL_CONTRACT,
            "config": self.config.to_dict(),
            "part_model_config": dict(getattr(self.part_model, "config", {}) or {}),
            "adapter_config": self.config.to_dict(),
            "baseline_checkpoint": dict(self.baseline_checkpoint_report or {}),
            "init_logit_diff_vs_baseline": dict(self.init_logit_diff_vs_baseline or {}),
            "uses_reference_part_backbone": bool(self.uses_reference_part_backbone),
            "baseline_recoverable_at_zero_delta": bool(self.baseline_recoverable_at_zero_delta),
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
            max_constits=max_constits,
            weight_threshold=float(weight_threshold),
            config=self.config.feature_config,
        )

    def forward_outputs(
        self,
        tokens_or_batch: Any,
        mask: Any | None = None,
        *,
        weights: Any | None = None,
        max_constits: int | None = None,
        weight_threshold: float = 0.0,
    ) -> LocalCompressionFeatureAdapterOutput:
        torch = require_torch()
        raw_tokens, raw_mask = _coerce_tokens_and_mask(tokens_or_batch, mask)
        canonical = self.build_canonical_inputs(
            raw_tokens,
            raw_mask,
            weights=weights,
            max_constits=max_constits,
            weight_threshold=weight_threshold,
        )
        modalities = build_local_compression_modalities(canonical, config=self.config.feature_config)
        subtoken_output = self.subtoken_encoder(canonical, modalities)
        compressor_output = self.compressor(subtoken_output)
        pool_output = self.pooler(compressor_output)
        context_output = self.context_block(pool_output)
        gate_output = self.gate(compressor_output, context_output, canonical)
        delta_output = self.adapter(canonical, pool_output, context_output, gate_output)
        adapted = canonical.with_features(delta_output.adapted_feature_rows)
        logits = self.part_model(
            adapted.points,
            adapted.features,
            adapted.lorentz_vectors,
            adapted.mask,
        )
        if isinstance(logits, torch.Tensor):
            logits = logits.to(device=adapted.features.device).float()
        else:
            logits = torch.as_tensor(logits, device=adapted.features.device).float()
        logits = _nan_to_num_torch(logits)
        return LocalCompressionFeatureAdapterOutput(
            logits=logits,
            canonical_inputs=canonical,
            adapted_inputs=adapted,
            modalities=modalities,
            subtoken_output=subtoken_output,
            compressor_output=compressor_output,
            pool_output=pool_output,
            context_output=context_output,
            gate_output=gate_output,
            delta_output=delta_output,
            config=self.config,
            uses_reference_part_backbone=bool(self.uses_reference_part_backbone),
            part_model_class=type(self.part_model).__name__,
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


def build_local_compression_feature_adapter_part(
    config: LocalCompressionPartConfig | Mapping[str, Any] | None = None,
    *,
    part_model: Any | None = None,
) -> LocalCompressionFeatureAdapterParT:
    return LocalCompressionFeatureAdapterParT(config, part_model=part_model)


__all__ = [
    "LOCAL_COMPRESSION_MODEL_CONTRACT",
    "LOCAL_COMPRESSION_MODEL_STEP",
    "LocalCompressionFeatureAdapterOutput",
    "LocalCompressionFeatureAdapterParT",
    "build_local_compression_feature_adapter_part",
]
