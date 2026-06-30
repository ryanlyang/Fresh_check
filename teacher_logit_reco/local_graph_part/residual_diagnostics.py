"""Diagnostics for local-graph residual expert corrections."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np

from .fusion import binary_margin_from_logits
from .residual_cache import operating_point_from_scores


LOCAL_GRAPH_RESIDUAL_DIAGNOSTICS_STEP = "local_graph_residual_expert_step5_diagnostics"
LOCAL_GRAPH_RESIDUAL_DIAGNOSTICS_CONTRACT = "local_graph_residual_diagnostics_v1"


def _jsonable_float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _safe_fraction(numerator: int, denominator: int) -> float | None:
    denominator = int(denominator)
    if denominator <= 0:
        return None
    return float(int(numerator) / float(denominator))


def _summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "abs_mean": None,
            "min": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "abs_mean": float(np.mean(np.abs(finite))),
        "min": float(np.min(finite)),
        "p10": float(np.quantile(finite, 0.10)),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "max": float(np.max(finite)),
    }


def _margin_from_logits_or_scores(
    *,
    logits: np.ndarray | None,
    scores: np.ndarray | None,
    name: str,
) -> np.ndarray:
    if scores is not None:
        output = np.asarray(scores, dtype=np.float64).reshape(-1)
    elif logits is not None:
        output = binary_margin_from_logits(np.asarray(logits, dtype=np.float64)).reshape(-1)
    else:
        raise ValueError(f"{name} logits or scores are required")
    if not np.isfinite(output).all():
        raise FloatingPointError(f"{name} scores contain non-finite values")
    return output


def _mask_indices(mask: np.ndarray, *, limit: int = 20) -> list[int]:
    return [int(index) for index in np.nonzero(mask)[0][: int(limit)].tolist()]


@dataclass(frozen=True)
class LocalGraphResidualDiagnosticsConfig:
    """Configuration for residual correction diagnostics."""

    target_signal_efficiency: float = 0.50
    near_tau_fraction: float = 0.25
    signal_boundary_quantile_low: float = 0.45
    signal_boundary_quantile_high: float = 0.55
    hard_background_quantile: float = 0.90
    include_index_samples: bool = True
    max_index_samples: int = 20

    def __post_init__(self) -> None:
        for field_name in ("target_signal_efficiency", "near_tau_fraction"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0.0 or value > 1.0:
                raise ValueError(f"{field_name} must be in (0, 1]")
            object.__setattr__(self, field_name, value)
        for field_name in ("signal_boundary_quantile_low", "signal_boundary_quantile_high", "hard_background_quantile"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
            object.__setattr__(self, field_name, value)
        if float(self.signal_boundary_quantile_low) > float(self.signal_boundary_quantile_high):
            raise ValueError("signal boundary quantile low cannot exceed high")
        max_index_samples = int(self.max_index_samples)
        if max_index_samples < 0:
            raise ValueError("max_index_samples must be nonnegative")
        object.__setattr__(self, "max_index_samples", max_index_samples)
        object.__setattr__(self, "include_index_samples", bool(self.include_index_samples))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def residual_correction_diagnostics(
    *,
    labels: np.ndarray,
    baseline_logits: np.ndarray | None = None,
    fused_logits: np.ndarray | None = None,
    residual_logits: np.ndarray | None = None,
    baseline_logit: np.ndarray | None = None,
    fused_logit: np.ndarray | None = None,
    residual_logit: np.ndarray | None = None,
    indices: np.ndarray | None = None,
    alpha_report: Mapping[str, Any] | None = None,
    config: LocalGraphResidualDiagnosticsConfig | None = None,
) -> dict[str, Any]:
    """Compare frozen baseline and fused residual scores at FPR@50 style boundaries."""

    config = config or LocalGraphResidualDiagnosticsConfig()
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.size == 0:
        raise ValueError("labels cannot be empty")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary encoded as QCD=0, Hgg=1")
    baseline = _margin_from_logits_or_scores(logits=baseline_logits, scores=baseline_logit, name="baseline")
    fused = _margin_from_logits_or_scores(logits=fused_logits, scores=fused_logit, name="fused")
    residual = _margin_from_logits_or_scores(logits=residual_logits, scores=residual_logit, name="residual")
    if labels.shape != baseline.shape or labels.shape != fused.shape or labels.shape != residual.shape:
        raise ValueError("labels, baseline, fused, and residual scores must have matching shape")
    if indices is None:
        indices_np = np.arange(labels.shape[0], dtype=np.int64)
    else:
        indices_np = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices_np.shape != labels.shape:
            raise ValueError("indices must match labels shape")

    signal = labels == 1
    background = labels == 0
    baseline_op = operating_point_from_scores(labels, baseline, float(config.target_signal_efficiency))
    fused_op = operating_point_from_scores(labels, fused, float(config.target_signal_efficiency))
    tau_base = float(baseline_op["threshold"])
    tau_fused = float(fused_op["threshold"])

    baseline_fp = background & (baseline >= tau_base)
    fused_fp = background & (fused >= tau_fused)
    baseline_fn = signal & (baseline < tau_base)
    fused_fn = signal & (fused < tau_fused)
    old_fp_removed = baseline_fp & ~fused_fp
    new_fp_introduced = fused_fp & ~baseline_fp
    fp_intersection = baseline_fp & fused_fp
    old_fn_removed = baseline_fn & ~fused_fn
    new_fn_introduced = fused_fn & ~baseline_fn
    fn_intersection = baseline_fn & fused_fn

    abs_delta_tau = np.abs(baseline - tau_base)
    finite_abs_delta = abs_delta_tau[np.isfinite(abs_delta_tau)]
    near_width = float(np.quantile(finite_abs_delta, float(config.near_tau_fraction))) if finite_abs_delta.size else 0.0
    near_tau = abs_delta_tau <= near_width
    baseline_fp_near = baseline_fp & near_tau
    baseline_fn_near = baseline_fn & near_tau
    baseline_fp_near_corrected = baseline_fp_near & (fused < tau_fused)
    baseline_fn_near_corrected = baseline_fn_near & (fused >= tau_fused)

    if int(np.sum(background)) > 0:
        hard_threshold = float(np.quantile(baseline[background], float(config.hard_background_quantile)))
        hard_background = background & (baseline >= hard_threshold)
    else:
        hard_threshold = float("nan")
        hard_background = np.zeros_like(background)
    if int(np.sum(signal)) > 0:
        q_low = float(np.quantile(baseline[signal], float(config.signal_boundary_quantile_low)))
        q_high = float(np.quantile(baseline[signal], float(config.signal_boundary_quantile_high)))
        hgg_boundary = signal & (baseline >= q_low) & (baseline <= q_high)
    else:
        q_low = float("nan")
        q_high = float("nan")
        hgg_boundary = np.zeros_like(signal)

    score_shift = fused - baseline
    fp_union_count = int(np.sum(baseline_fp | fused_fp))
    fn_union_count = int(np.sum(baseline_fn | fused_fn))
    false_positive_overlap = {
        "baseline_fp_count": int(np.sum(baseline_fp)),
        "fused_fp_count": int(np.sum(fused_fp)),
        "intersection_count": int(np.sum(fp_intersection)),
        "old_false_positives_removed": int(np.sum(old_fp_removed)),
        "new_false_positives_introduced": int(np.sum(new_fp_introduced)),
        "union_count": fp_union_count,
        "old_false_positive_removed_fraction": _safe_fraction(int(np.sum(old_fp_removed)), int(np.sum(baseline_fp))),
        "new_false_positive_fraction_of_fused": _safe_fraction(int(np.sum(new_fp_introduced)), int(np.sum(fused_fp))),
        "jaccard": _safe_fraction(int(np.sum(fp_intersection)), fp_union_count),
    }
    false_negative_overlap = {
        "baseline_fn_count": int(np.sum(baseline_fn)),
        "fused_fn_count": int(np.sum(fused_fn)),
        "intersection_count": int(np.sum(fn_intersection)),
        "old_false_negatives_removed": int(np.sum(old_fn_removed)),
        "new_false_negatives_introduced": int(np.sum(new_fn_introduced)),
        "union_count": fn_union_count,
        "old_false_negative_removed_fraction": _safe_fraction(int(np.sum(old_fn_removed)), int(np.sum(baseline_fn))),
        "new_false_negative_fraction_of_fused": _safe_fraction(int(np.sum(new_fn_introduced)), int(np.sum(fused_fn))),
        "jaccard": _safe_fraction(int(np.sum(fn_intersection)), fn_union_count),
    }
    boundary_corrections = {
        "near_tau_width": near_width,
        "baseline_fp_near_tau50_count": int(np.sum(baseline_fp_near)),
        "baseline_fp_near_tau50_corrected_count": int(np.sum(baseline_fp_near_corrected)),
        "baseline_FP_near_tau50_corrected_fraction": _safe_fraction(
            int(np.sum(baseline_fp_near_corrected)),
            int(np.sum(baseline_fp_near)),
        ),
        "baseline_fn_near_tau50_count": int(np.sum(baseline_fn_near)),
        "baseline_fn_near_tau50_corrected_count": int(np.sum(baseline_fn_near_corrected)),
        "baseline_FN_near_tau50_corrected_fraction": _safe_fraction(
            int(np.sum(baseline_fn_near_corrected)),
            int(np.sum(baseline_fn_near)),
        ),
    }

    region_shift_summary = {
        "score_shift_all": _summary(score_shift),
        "score_shift_qcd": _summary(score_shift[background]),
        "score_shift_hgg": _summary(score_shift[signal]),
        "hard_QCD_tail_score_shift_mean": _jsonable_float(np.mean(score_shift[hard_background]))
        if int(np.sum(hard_background)) > 0
        else None,
        "Hgg_boundary_score_shift_mean": _jsonable_float(np.mean(score_shift[hgg_boundary]))
        if int(np.sum(hgg_boundary)) > 0
        else None,
        "hard_background_threshold": _jsonable_float(hard_threshold),
        "hard_background_count": int(np.sum(hard_background)),
        "hgg_boundary_score_low": _jsonable_float(q_low),
        "hgg_boundary_score_high": _jsonable_float(q_high),
        "hgg_boundary_count": int(np.sum(hgg_boundary)),
    }

    alpha_summary = None
    if isinstance(alpha_report, Mapping):
        alpha_summary = {
            "selected_alpha": alpha_report.get("selected_alpha"),
            "selected_fpr": alpha_report.get("selected_fpr"),
            "baseline_fpr": alpha_report.get("baseline_fpr"),
            "delta_fpr_vs_baseline": alpha_report.get("delta_fpr_vs_baseline"),
            "collapsed_to_zero": alpha_report.get("collapsed_to_zero"),
        }

    report: dict[str, Any] = {
        "step": LOCAL_GRAPH_RESIDUAL_DIAGNOSTICS_STEP,
        "contract": LOCAL_GRAPH_RESIDUAL_DIAGNOSTICS_CONTRACT,
        "config": config.to_dict(),
        "n_jets": int(labels.shape[0]),
        "n_signal": int(np.sum(signal)),
        "n_background": int(np.sum(background)),
        "baseline_operating_point": baseline_op,
        "fused_operating_point": fused_op,
        "fused_delta_FPR50_vs_baseline": float(fused_op["false_positive_rate"]) - float(baseline_op["false_positive_rate"]),
        "fused_delta_background_rejection_at_50_vs_baseline": (
            _jsonable_float(float(fused_op["background_rejection"]) - float(baseline_op["background_rejection"]))
            if math.isfinite(float(fused_op["background_rejection"]))
            and math.isfinite(float(baseline_op["background_rejection"]))
            else None
        ),
        "false_positive_overlap": false_positive_overlap,
        "false_negative_overlap": false_negative_overlap,
        "boundary_corrections": boundary_corrections,
        "residual_summary": _summary(residual),
        "baseline_score_summary": _summary(baseline),
        "fused_score_summary": _summary(fused),
        "region_shift_summary": region_shift_summary,
        "alpha_summary": alpha_summary,
    }
    if bool(config.include_index_samples):
        limit = int(config.max_index_samples)
        report["index_samples"] = {
            "old_false_positives_removed": indices_np[old_fp_removed][:limit].astype(int).tolist(),
            "new_false_positives_introduced": indices_np[new_fp_introduced][:limit].astype(int).tolist(),
            "old_false_negatives_removed": indices_np[old_fn_removed][:limit].astype(int).tolist(),
            "new_false_negatives_introduced": indices_np[new_fn_introduced][:limit].astype(int).tolist(),
        }
    return report


__all__ = [
    "LOCAL_GRAPH_RESIDUAL_DIAGNOSTICS_CONTRACT",
    "LOCAL_GRAPH_RESIDUAL_DIAGNOSTICS_STEP",
    "LocalGraphResidualDiagnosticsConfig",
    "residual_correction_diagnostics",
]
