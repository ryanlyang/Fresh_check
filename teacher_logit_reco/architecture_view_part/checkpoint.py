"""Checkpoint warm-start helpers for Architecture-View Residual ParT."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from jetclass_fresh.hlt_baseline import require_torch

from teacher_logit_reco.local_compression_part.checkpoint import (
    LocalCompressionBaselineCheckpointReport,
    load_hlt_part_baseline_checkpoint,
    sha256_file,
)

from .config import (
    ARCHITECTURE_VIEW_BINARY_LABEL_FILTER,
    ARCHITECTURE_VIEW_HLT_DEGRADATION_STRENGTH,
    ARCHITECTURE_VIEW_LABEL_NAMES,
    ARCHITECTURE_VIEW_PART_CONTRACT,
    ARCHITECTURE_VIEW_PRIMARY_METRIC,
)


ARCHITECTURE_VIEW_CHECKPOINT_STEP = "architecture_view_part_step2_checkpoint_warm_start"
ARCHITECTURE_VIEW_CHECKPOINT_CONTRACT = f"{ARCHITECTURE_VIEW_PART_CONTRACT}_checkpoint_warm_start_v1"


def load_architecture_view_hlt_part_checkpoint(
    checkpoint_path: str | Path,
    model_or_part_model: Any,
    *,
    map_location: Any = "cpu",
    run_report_path: str | Path | None = None,
    expected_selection_metric: str | None = ARCHITECTURE_VIEW_PRIMARY_METRIC,
    expected_hlt_degradation_strength: float | None = ARCHITECTURE_VIEW_HLT_DEGRADATION_STRENGTH,
    expected_split_manifest_hash: str | None = None,
    expected_label_names: Sequence[str] | None = ARCHITECTURE_VIEW_LABEL_NAMES,
    expected_label_filter: Sequence[int] | None = ARCHITECTURE_VIEW_BINARY_LABEL_FILTER,
    expected_num_classes: int | None = 2,
    require_metadata: bool = True,
    require_all_part_keys: bool = True,
) -> LocalCompressionBaselineCheckpointReport:
    """Load the exact HLT ParT baseline with architecture-view protocol checks."""

    return load_hlt_part_baseline_checkpoint(
        checkpoint_path,
        model_or_part_model,
        map_location=map_location,
        run_report_path=run_report_path,
        expected_selection_metric=expected_selection_metric,
        expected_hlt_degradation_strength=expected_hlt_degradation_strength,
        expected_split_manifest_hash=expected_split_manifest_hash,
        expected_label_names=expected_label_names,
        expected_label_filter=expected_label_filter,
        expected_num_classes=expected_num_classes,
        require_metadata=bool(require_metadata),
        require_all_part_keys=bool(require_all_part_keys),
    )


def warm_start_architecture_view_part_model(
    model: Any,
    checkpoint_path: str | Path,
    **kwargs: Any,
) -> LocalCompressionBaselineCheckpointReport:
    report = load_architecture_view_hlt_part_checkpoint(checkpoint_path, model, **kwargs)
    if hasattr(model, "baseline_checkpoint_report"):
        setattr(model, "baseline_checkpoint_report", report.to_dict())
    return report


def compute_architecture_view_init_logit_diff_vs_baseline(
    model: Any,
    tokens_or_batch: Any,
    mask: Any | None = None,
    *,
    max_constits: int | None = None,
    weight_threshold: float = 0.0,
    attach: bool = True,
) -> dict[str, Any]:
    """Compare architecture-view logits with direct HLT ParT baseline logits."""

    torch = require_torch()
    if not hasattr(model, "part_model") or not hasattr(model, "build_canonical_inputs"):
        raise ValueError("init-logit comparison requires ArchitectureViewResidualParT")
    from .model import _coerce_tokens_and_mask

    was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        raw_tokens, raw_mask = _coerce_tokens_and_mask(tokens_or_batch, mask)
        if max_constits is None:
            max_constits = int(raw_tokens.shape[1])
        canonical = model.build_canonical_inputs(
            raw_tokens,
            raw_mask,
            max_constits=int(max_constits),
            weight_threshold=float(weight_threshold),
        )
        baseline_logits = model.part_model(
            canonical.points,
            canonical.features,
            canonical.lorentz_vectors,
            canonical.mask,
        ).float()
        output = model(
            raw_tokens,
            raw_mask,
            return_outputs=True,
            max_constits=int(max_constits),
            weight_threshold=float(weight_threshold),
        )
        diff = (output.logits - baseline_logits).detach().float()
        result = {
            "contract": ARCHITECTURE_VIEW_CHECKPOINT_CONTRACT,
            "step": ARCHITECTURE_VIEW_CHECKPOINT_STEP,
            "baseline_logits_shape": list(baseline_logits.shape),
            "architecture_view_logits_shape": list(output.logits.shape),
            "max_abs_logit_diff": float(diff.abs().max().cpu().item()) if int(diff.numel()) else 0.0,
            "mean_abs_logit_diff": float(diff.abs().mean().cpu().item()) if int(diff.numel()) else 0.0,
            "allclose_atol_1e_6": bool(torch.allclose(output.logits, baseline_logits, atol=1.0e-6, rtol=1.0e-6)),
            "embed_injection": dict(output.injection_summary),
        }
    if was_training:
        model.train()
    if bool(attach) and hasattr(model, "init_logit_diff_vs_baseline"):
        setattr(model, "init_logit_diff_vs_baseline", dict(result))
    return result


__all__ = [
    "ARCHITECTURE_VIEW_CHECKPOINT_CONTRACT",
    "ARCHITECTURE_VIEW_CHECKPOINT_STEP",
    "LocalCompressionBaselineCheckpointReport",
    "compute_architecture_view_init_logit_diff_vs_baseline",
    "load_architecture_view_hlt_part_checkpoint",
    "sha256_file",
    "warm_start_architecture_view_part_model",
]
