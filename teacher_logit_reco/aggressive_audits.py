"""Audits for the aggressive cross-architecture reconstructor branch."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from jetclass_fresh.fusion import STACK_SPLITS, load_prediction_block, prediction_paths
from jetclass_fresh.hlt_baseline import save_json

from .crossarch_experiment import (
    AGGRESSIVE_RECONSTRUCTOR_ARCHITECTURES,
    TEACHER_ARCHITECTURES,
    aggressive_reco_domain_tagger_model_name,
    aggressive_reco_model_name,
    build_fusion_groups,
    hlt_model_name,
)


AGGRESSIVE_AUDIT_STEP = "teacher_logit_reco_step13_aggressive_audit"
PARENT_WEIGHT_HISTOGRAM_EDGES = tuple(float(x) for x in np.linspace(0.0, 1.0, 11))


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


@dataclass
class RunningStats:
    """Streaming scalar statistics for audit-sized tensor summaries."""

    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def update(self, values: Any) -> None:
        arr = _to_numpy(values)
        if arr is None:
            return
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        self.count += int(arr.size)
        self.total += float(arr.sum())
        self.total_sq += float(np.square(arr).sum())
        current_min = float(arr.min())
        current_max = float(arr.max())
        self.minimum = current_min if self.minimum is None else min(self.minimum, current_min)
        self.maximum = current_max if self.maximum is None else max(self.maximum, current_max)

    def to_dict(self) -> dict[str, Any]:
        if self.count <= 0:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        mean = self.total / float(self.count)
        variance = max(0.0, self.total_sq / float(self.count) - mean * mean)
        return {
            "count": int(self.count),
            "mean": float(mean),
            "std": float(np.sqrt(variance)),
            "min": None if self.minimum is None else float(self.minimum),
            "max": None if self.maximum is None else float(self.maximum),
        }


@dataclass
class HistogramAccumulator:
    edges: Sequence[float]
    counts: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if len(self.edges) < 2:
            raise ValueError("Histogram edges must contain at least two values")
        self.counts = np.zeros(len(self.edges) - 1, dtype=np.int64)

    def update(self, values: Any) -> None:
        arr = _to_numpy(values)
        if arr is None:
            return
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        counts, _ = np.histogram(arr, bins=np.asarray(self.edges, dtype=np.float64))
        self.counts += counts.astype(np.int64)

    def to_dict(self) -> dict[str, Any]:
        return {"edges": [float(x) for x in self.edges], "counts": [int(x) for x in self.counts.tolist()]}


@dataclass
class AggressiveReconstructionDiagnosticsAccumulator:
    """Aggregate aggressive soft-view diagnostics without storing per-jet tensors."""

    parent_weight: RunningStats = field(default_factory=RunningStats)
    parent_weight_sum: RunningStats = field(default_factory=RunningStats)
    parent_valid_count: RunningStats = field(default_factory=RunningStats)
    parent_keep_fraction: RunningStats = field(default_factory=RunningStats)
    parent_prune_rate_lt_010: RunningStats = field(default_factory=RunningStats)
    parent_prune_rate_lt_025: RunningStats = field(default_factory=RunningStats)
    parent_prune_rate_lt_050: RunningStats = field(default_factory=RunningStats)
    parent_delta_abs: dict[str, RunningStats] = field(
        default_factory=lambda: {
            "logpt": RunningStats(),
            "eta": RunningStats(),
            "phi": RunningStats(),
            "loge": RunningStats(),
        }
    )
    parent_weight_histogram: HistogramAccumulator = field(
        default_factory=lambda: HistogramAccumulator(PARENT_WEIGHT_HISTOGRAM_EDGES)
    )
    extra_weight: RunningStats = field(default_factory=RunningStats)
    extra_weight_sum: RunningStats = field(default_factory=RunningStats)
    extra_pt_fraction: RunningStats = field(default_factory=RunningStats)
    extra_slot_usage: RunningStats = field(default_factory=RunningStats)
    extra_slot_usage_histogram_counts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    extra_slot_activation_counts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    extra_slot_observed_jets: int = 0
    global_correction: dict[str, RunningStats] = field(
        default_factory=lambda: {
            "logpt_scale": RunningStats(),
            "loge_scale": RunningStats(),
            "eta_shift": RunningStats(),
            "phi_shift": RunningStats(),
        }
    )
    global_correction_abs: dict[str, RunningStats] = field(
        default_factory=lambda: {
            "logpt_scale": RunningStats(),
            "loge_scale": RunningStats(),
            "eta_shift": RunningStats(),
            "phi_shift": RunningStats(),
        }
    )
    batch_count: int = 0
    jet_count: int = 0

    def update_from_soft_view(self, soft_view: Any) -> None:
        aux = getattr(soft_view, "aux", {}) or {}
        parent_weights = _to_numpy(aux.get("parent_weights"))
        parent_mask_value = aux.get("sanitized_hlt_mask")
        if parent_mask_value is None:
            parent_mask_value = aux.get("parent_mask")
        parent_mask = _to_numpy(parent_mask_value)
        if parent_weights is not None:
            parent_weights = np.asarray(parent_weights, dtype=np.float64)
            if parent_mask is None:
                parent_mask = np.isfinite(parent_weights)
            else:
                parent_mask = np.asarray(parent_mask).astype(bool)
            valid_parent_weights = parent_weights[parent_mask & np.isfinite(parent_weights)]
            self.parent_weight.update(valid_parent_weights)
            self.parent_weight_histogram.update(np.clip(valid_parent_weights, 0.0, 1.0))
            if parent_weights.ndim == 2:
                weighted = np.where(parent_mask, parent_weights, 0.0)
                valid_counts = np.maximum(parent_mask.sum(axis=1), 1)
                parent_weight_sum = weighted.sum(axis=1)
                self.parent_weight_sum.update(parent_weight_sum)
                self.parent_valid_count.update(parent_mask.sum(axis=1))
                self.parent_keep_fraction.update(parent_weight_sum / valid_counts)
                for threshold, stats in (
                    (0.10, self.parent_prune_rate_lt_010),
                    (0.25, self.parent_prune_rate_lt_025),
                    (0.50, self.parent_prune_rate_lt_050),
                ):
                    pruned = ((parent_weights < threshold) & parent_mask).sum(axis=1) / valid_counts
                    stats.update(pruned)

        parent_delta = _to_numpy(aux.get("parent_delta"))
        if parent_delta is not None and parent_delta.ndim == 3:
            if parent_mask is None:
                parent_mask = np.ones(parent_delta.shape[:2], dtype=bool)
            for index, key in enumerate(("logpt", "eta", "phi", "loge")):
                values = np.abs(parent_delta[:, :, index])
                self.parent_delta_abs[key].update(values[parent_mask & np.isfinite(values)])

        extra_weights = _to_numpy(aux.get("extra_weights"))
        extra_mask = _to_numpy(aux.get("extra_mask"))
        if extra_weights is not None:
            extra_weights = np.asarray(extra_weights, dtype=np.float64)
            if extra_mask is None:
                extra_mask = np.isfinite(extra_weights)
            else:
                extra_mask = np.asarray(extra_mask).astype(bool)
            valid_extra_weights = extra_weights[extra_mask & np.isfinite(extra_weights)]
            self.extra_weight.update(valid_extra_weights)

        for key, stats in (
            ("extra_weight_sum", self.extra_weight_sum),
            ("extra_pt_fraction", self.extra_pt_fraction),
            ("extra_slot_usage", self.extra_slot_usage),
        ):
            stats.update(aux.get(key))

        slot_usage_hist = aux.get("extra_slot_usage_histogram")
        if slot_usage_hist is not None:
            hist = np.asarray(_to_numpy(slot_usage_hist), dtype=np.int64).reshape(-1)
            if hist.size > self.extra_slot_usage_histogram_counts.size:
                grown = np.zeros(hist.size, dtype=np.int64)
                grown[: self.extra_slot_usage_histogram_counts.size] = self.extra_slot_usage_histogram_counts
                self.extra_slot_usage_histogram_counts = grown
            self.extra_slot_usage_histogram_counts[: hist.size] += hist

        active_mask = _to_numpy(aux.get("extra_slot_active_mask"))
        if active_mask is not None and active_mask.ndim == 2:
            active_mask = np.asarray(active_mask).astype(bool)
            slot_counts = active_mask.sum(axis=0).astype(np.int64)
            if slot_counts.size > self.extra_slot_activation_counts.size:
                grown = np.zeros(slot_counts.size, dtype=np.int64)
                grown[: self.extra_slot_activation_counts.size] = self.extra_slot_activation_counts
                self.extra_slot_activation_counts = grown
            self.extra_slot_activation_counts[: slot_counts.size] += slot_counts
            self.extra_slot_observed_jets += int(active_mask.shape[0])

        global_correction = aux.get("global_correction")
        if global_correction is None:
            global_correction = aux.get("global_calibration")
        if global_correction is None:
            global_correction = {}
        for key in ("logpt_scale", "loge_scale", "eta_shift", "phi_shift"):
            values = _to_numpy(global_correction.get(key) if isinstance(global_correction, Mapping) else None)
            self.global_correction[key].update(values)
            if values is not None:
                self.global_correction_abs[key].update(np.abs(values))

        labels = _to_numpy(getattr(soft_view, "labels", None))
        self.jet_count += int(labels.shape[0]) if labels is not None and labels.ndim >= 1 else 0
        self.batch_count += 1

    def to_dict(self) -> dict[str, Any]:
        slot_fractions: list[float] = []
        if self.extra_slot_observed_jets > 0 and self.extra_slot_activation_counts.size:
            slot_fractions = [
                float(count) / float(self.extra_slot_observed_jets)
                for count in self.extra_slot_activation_counts.tolist()
            ]
        return {
            "batch_count": int(self.batch_count),
            "jet_count": int(self.jet_count),
            "parent_weight": self.parent_weight.to_dict(),
            "parent_weight_sum": self.parent_weight_sum.to_dict(),
            "parent_valid_count": self.parent_valid_count.to_dict(),
            "parent_keep_fraction": self.parent_keep_fraction.to_dict(),
            "parent_prune_rate_lt_010": self.parent_prune_rate_lt_010.to_dict(),
            "parent_prune_rate_lt_025": self.parent_prune_rate_lt_025.to_dict(),
            "parent_prune_rate_lt_050": self.parent_prune_rate_lt_050.to_dict(),
            "parent_delta_abs": {key: value.to_dict() for key, value in self.parent_delta_abs.items()},
            "parent_weight_histogram": self.parent_weight_histogram.to_dict(),
            "extra_weight": self.extra_weight.to_dict(),
            "extra_weight_sum": self.extra_weight_sum.to_dict(),
            "extra_pt_fraction": self.extra_pt_fraction.to_dict(),
            "extra_slot_usage": self.extra_slot_usage.to_dict(),
            "extra_slot_usage_histogram": [int(x) for x in self.extra_slot_usage_histogram_counts.tolist()],
            "extra_slot_activation_fraction_by_slot": slot_fractions,
            "global_correction": {key: value.to_dict() for key, value in self.global_correction.items()},
            "global_correction_abs": {key: value.to_dict() for key, value in self.global_correction_abs.items()},
        }


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _maybe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def _prediction_metadata_path(prediction_dir: Path, model_name: str, split: str) -> Path:
    _, meta_path = prediction_paths(prediction_dir, model_name, split)
    return meta_path


def _checkpoint_report_path(root: Path, reco_architecture: str, teacher_architecture: str) -> Path:
    return root / reco_architecture / teacher_architecture / "run_report.json"


def _adapted_report_path(root: Path, reco_architecture: str, teacher_architecture: str) -> Path:
    return root / reco_architecture / teacher_architecture / "run_report.json"


def _expected_size(split: str, expected_split_sizes: Mapping[str, int]) -> int | None:
    value = expected_split_sizes.get(split)
    return None if value is None else int(value)


def summarize_prediction_metadata(
    *,
    prediction_dir: str | Path,
    model_names: Sequence[str],
    splits: Sequence[str],
    expected_split_sizes: Mapping[str, int],
    require_reconstruction_diagnostics: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prediction_root = Path(prediction_dir)
    report: dict[str, Any] = {}
    flags: list[dict[str, Any]] = []
    for model_name in model_names:
        by_split: dict[str, Any] = {}
        for split in splits:
            path = _prediction_metadata_path(prediction_root, model_name, split)
            metadata = _maybe_load_json(path)
            expected_n = _expected_size(split, expected_split_sizes)
            if metadata is None:
                flags.append(
                    {
                        "severity": "error",
                        "name": "missing_prediction_metadata",
                        "model_name": model_name,
                        "split": split,
                        "path": str(path),
                    }
                )
                by_split[split] = {"exists": False, "path": str(path)}
                continue
            n_jets = int(metadata.get("n_jets", -1))
            diagnostics = metadata.get("reconstruction_diagnostics")
            if expected_n is not None and n_jets != int(expected_n):
                flags.append(
                    {
                        "severity": "error",
                        "name": "split_size_mismatch",
                        "model_name": model_name,
                        "split": split,
                        "expected_n_jets": int(expected_n),
                        "actual_n_jets": n_jets,
                    }
                )
            if require_reconstruction_diagnostics and diagnostics is None:
                flags.append(
                    {
                        "severity": "warning",
                        "name": "missing_reconstruction_diagnostics",
                        "model_name": model_name,
                        "split": split,
                        "path": str(path),
                    }
                )
            by_split[split] = {
                "exists": True,
                "path": str(path),
                "n_jets": n_jets,
                "expected_n_jets": expected_n,
                "metrics": dict(metadata.get("metrics") or {}),
                "hlt_content_hash": metadata.get("hlt_content_hash"),
                "reconstructor_checkpoint": metadata.get("reconstructor_checkpoint"),
                "reconstructor_checkpoint_epoch": metadata.get("reconstructor_checkpoint_epoch"),
                "teacher_architecture": metadata.get("teacher_architecture") or metadata.get("tagger_architecture"),
                "allowed_inputs": metadata.get("allowed_inputs"),
                "reconstruction_diagnostics": diagnostics,
            }
        report[model_name] = by_split
    return report, flags


def summarize_prediction_arrays(
    *,
    prediction_dir: str | Path,
    model_names: Sequence[str],
    splits: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prediction_root = Path(prediction_dir)
    report: dict[str, Any] = {}
    flags: list[dict[str, Any]] = []
    for model_name in model_names:
        by_split: dict[str, Any] = {}
        for split in splits:
            try:
                block = load_prediction_block(prediction_root, model_name, split)
            except Exception as exc:
                flags.append(
                    {
                        "severity": "error",
                        "name": "prediction_block_load_failed",
                        "model_name": model_name,
                        "split": split,
                        "error": str(exc),
                    }
                )
                by_split[split] = {"loaded": False, "error": str(exc)}
                continue
            logits = np.asarray(block.logits)
            probs = np.asarray(block.probs)
            labels = np.asarray(block.labels)
            row_count = int(labels.shape[0])
            split_report = {
                "loaded": True,
                "n_jets": row_count,
                "logits_shape": [int(x) for x in logits.shape],
                "probs_shape": [int(x) for x in probs.shape],
                "labels_shape": [int(x) for x in labels.shape],
                "logits_all_finite": bool(np.isfinite(logits).all()),
                "probs_all_finite": bool(np.isfinite(probs).all()),
                "labels_all_finite": bool(np.isfinite(labels).all()),
            }
            if logits.shape[0] != row_count or probs.shape[0] != row_count:
                flags.append(
                    {
                        "severity": "error",
                        "name": "prediction_block_row_mismatch",
                        "model_name": model_name,
                        "split": split,
                        "n_labels": row_count,
                        "n_logits": int(logits.shape[0]),
                        "n_probs": int(probs.shape[0]),
                    }
                )
            for array_name, ok in (
                ("logits", split_report["logits_all_finite"]),
                ("probs", split_report["probs_all_finite"]),
                ("labels", split_report["labels_all_finite"]),
            ):
                if not ok:
                    flags.append(
                        {
                            "severity": "error",
                            "name": "prediction_block_nonfinite_values",
                            "model_name": model_name,
                            "split": split,
                            "array": array_name,
                        }
                    )
            by_split[split] = split_report
        report[model_name] = by_split
    return report, flags


def summarize_checkpoint_reports(
    *,
    model_root: str | Path,
    adapted_tagger_root: str | Path,
    reconstructors: Sequence[str],
    teachers: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    model_root = Path(model_root)
    adapted_tagger_root = Path(adapted_tagger_root)
    reco_reports: dict[str, Any] = {}
    adapted_reports: dict[str, Any] = {}
    flags: list[dict[str, Any]] = []
    for reco in reconstructors:
        for teacher in teachers:
            reco_name = aggressive_reco_model_name(reco, teacher)
            adapted_name = aggressive_reco_domain_tagger_model_name(reco, teacher)
            reco_path = _checkpoint_report_path(model_root, reco, teacher)
            adapted_path = _adapted_report_path(adapted_tagger_root, reco, teacher)
            reco_report = _maybe_load_json(reco_path)
            adapted_report = _maybe_load_json(adapted_path)
            if reco_report is None:
                flags.append({"severity": "error", "name": "missing_reconstructor_run_report", "model_name": reco_name, "path": str(reco_path)})
                reco_reports[reco_name] = {"exists": False, "path": str(reco_path)}
            else:
                config = dict(reco_report.get("config") or {})
                source = dict(reco_report.get("source") or {})
                no_final = bool(reco_report.get("no_final_test_evaluation", False))
                split_ok = config.get("train_split", "model_train") == "model_train" and config.get("val_split", "model_val") == "model_val"
                if not no_final or not split_ok:
                    flags.append(
                        {
                            "severity": "error",
                            "name": "possible_final_test_training_leakage",
                            "model_name": reco_name,
                            "no_final_test_evaluation": no_final,
                            "train_split": config.get("train_split"),
                            "val_split": config.get("val_split"),
                        }
                    )
                reco_reports[reco_name] = {
                    "exists": True,
                    "path": str(reco_path),
                    "best_epoch": reco_report.get("best_epoch"),
                    "best_model_val_total_loss": reco_report.get("best_model_val_total_loss"),
                    "best_model_val_reco_argmax_accuracy": reco_report.get("best_model_val_reco_argmax_accuracy"),
                    "epochs_completed": reco_report.get("epochs_completed"),
                    "checkpoint": reco_report.get("checkpoint"),
                    "teacher": dict(reco_report.get("teacher") or {}),
                    "train_n_jets": source.get("train_n_jets"),
                    "val_n_jets": source.get("val_n_jets"),
                    "train_pair": source.get("train_pair"),
                    "val_pair": source.get("val_pair"),
                    "no_final_test_evaluation": no_final,
                    "train_split": config.get("train_split"),
                    "val_split": config.get("val_split"),
                    "model_config": dict(reco_report.get("model_config") or {}),
                    "loss_config": dict(reco_report.get("loss_config") or {}),
                }
            if adapted_report is None:
                flags.append({"severity": "error", "name": "missing_adapted_tagger_run_report", "model_name": adapted_name, "path": str(adapted_path)})
                adapted_reports[adapted_name] = {"exists": False, "path": str(adapted_path)}
            else:
                source_metadata_path = adapted_report.get("source_metadata_path")
                source_metadata = None
                if source_metadata_path:
                    source_metadata = _maybe_load_json(Path(str(source_metadata_path)))
                no_final = bool(adapted_report.get("no_final_test_evaluation", False))
                if not no_final:
                    flags.append(
                        {
                            "severity": "error",
                            "name": "possible_adapted_tagger_final_test_training_leakage",
                            "model_name": adapted_name,
                            "no_final_test_evaluation": no_final,
                        }
                    )
                adapted_reports[adapted_name] = {
                    "exists": True,
                    "path": str(adapted_path),
                    "best_epoch": adapted_report.get("best_epoch"),
                    "best_model_val_accuracy": adapted_report.get("best_model_val_accuracy"),
                    "best_model_val_loss": adapted_report.get("best_model_val_loss"),
                    "epochs_completed": adapted_report.get("epochs_completed"),
                    "checkpoint": adapted_report.get("checkpoint"),
                    "source_teacher_reco_model_name": adapted_report.get("source_teacher_reco_model_name"),
                    "source_reconstructor_implementation": adapted_report.get("source_reconstructor_implementation"),
                    "source_metadata_path": source_metadata_path,
                    "train_n_jets": (source_metadata or {}).get("train_n_jets"),
                    "val_n_jets": (source_metadata or {}).get("val_n_jets"),
                    "train_split": (source_metadata or {}).get("train_split"),
                    "val_split": (source_metadata or {}).get("val_split"),
                    "no_final_test_evaluation": no_final,
                }
    return reco_reports, adapted_reports, flags


def summarize_fusion_groups(
    *,
    prediction_dir: str | Path,
    group_names: Sequence[str],
    splits: Sequence[str],
    fusion_report_path: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_groups = build_fusion_groups(include_optional=True)
    prediction_root = Path(prediction_dir)
    fusion_report = _maybe_load_json(Path(fusion_report_path)) if fusion_report_path else None
    fusion_groups_from_report = dict((fusion_report or {}).get("groups") or {})
    flags: list[dict[str, Any]] = []
    output: dict[str, Any] = {}
    for group_name in group_names:
        spec = all_groups.get(group_name)
        if spec is None:
            flags.append({"severity": "error", "name": "unknown_fusion_group", "group": group_name})
            output[group_name] = {"exists": False}
            continue
        missing: list[dict[str, str]] = []
        for model_name in spec.model_names:
            for split in splits:
                path = _prediction_metadata_path(prediction_root, model_name, split)
                if not path.exists():
                    missing.append({"model_name": model_name, "split": split, "path": str(path)})
        if missing:
            flags.append({"severity": "error", "name": "fusion_group_missing_prediction_sources", "group": group_name, "missing_count": len(missing)})
        group_report = fusion_groups_from_report.get(group_name) or {}
        output[group_name] = {
            "exists": True,
            "description": spec.description,
            "model_names": list(spec.model_names),
            "n_models": len(spec.model_names),
            "missing_prediction_sources": missing,
            "fusion_report": group_report,
        }
    return output, flags


@dataclass
class AggressiveAuditConfig:
    prediction_dir: str
    reco_model_dir: str
    adapted_tagger_dir: str
    output_dir: str
    fusion_report: str | None = None
    reconstructors: Sequence[str] = AGGRESSIVE_RECONSTRUCTOR_ARCHITECTURES
    teachers: Sequence[str] = TEACHER_ARCHITECTURES
    splits: Sequence[str] = tuple(STACK_SPLITS)
    fusion_groups: Sequence[str] = (
        "hlt4",
        "aggressive_all16",
        "aggressive_all16_plus_hlt4",
        "aggressive_cross12_plus_hlt4",
        "aggressive_part_teacher4_plus_hlt4",
        "aggressive_pn_teacher4_plus_hlt4",
        "aggressive_mixed4_plus_hlt4",
        "aggressive_adapted_all16_plus_hlt4",
    )
    expected_split_sizes: Mapping[str, int] = field(
        default_factory=lambda: {"stack_train": 500_000, "stack_val": 150_000, "final_test": 500_000}
    )
    check_prediction_arrays: bool = False


def write_aggressive_audit_summary(report: Mapping[str, Any], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    lines = [
        "# Aggressive Reconstructor Audit",
        "",
        f"ok: `{bool(report.get('ok'))}`",
        f"error_count: `{report.get('error_count')}`",
        f"warning_count: `{report.get('warning_count')}`",
        "",
        "## Fusion Groups",
    ]
    for name, group in sorted((report.get("fusion_groups") or {}).items()):
        lines.append(f"- `{name}`: {group.get('n_models')} models, missing={len(group.get('missing_prediction_sources') or [])}")
    lines.extend(["", "## Flags"])
    for flag in report.get("flags") or []:
        lines.append(f"- `{flag.get('severity')}` `{flag.get('name')}` {json.dumps(flag, sort_keys=True)}")
    path = output_dir / "aggressive_audit_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_aggressive_audit(config: AggressiveAuditConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstructors = tuple(config.reconstructors)
    teachers = tuple(config.teachers)
    splits = tuple(config.splits)
    frozen_names = tuple(aggressive_reco_model_name(reco, teacher) for reco in reconstructors for teacher in teachers)
    adapted_names = tuple(aggressive_reco_domain_tagger_model_name(reco, teacher) for reco in reconstructors for teacher in teachers)
    hlt_names = tuple(hlt_model_name(architecture) for architecture in ("part", "pn", "pfn", "pcnn"))

    reco_reports, adapted_reports, checkpoint_flags = summarize_checkpoint_reports(
        model_root=config.reco_model_dir,
        adapted_tagger_root=config.adapted_tagger_dir,
        reconstructors=reconstructors,
        teachers=teachers,
    )
    frozen_predictions, frozen_flags = summarize_prediction_metadata(
        prediction_dir=config.prediction_dir,
        model_names=frozen_names,
        splits=splits,
        expected_split_sizes=config.expected_split_sizes,
    )
    adapted_predictions, adapted_flags = summarize_prediction_metadata(
        prediction_dir=config.prediction_dir,
        model_names=adapted_names,
        splits=splits,
        expected_split_sizes=config.expected_split_sizes,
    )
    hlt_predictions, hlt_flags = summarize_prediction_metadata(
        prediction_dir=config.prediction_dir,
        model_names=hlt_names,
        splits=splits,
        expected_split_sizes=config.expected_split_sizes,
        require_reconstruction_diagnostics=False,
    )
    fusion_groups, fusion_flags = summarize_fusion_groups(
        prediction_dir=config.prediction_dir,
        group_names=tuple(config.fusion_groups),
        splits=splits,
        fusion_report_path=config.fusion_report,
    )
    prediction_array_audit: dict[str, Any] | None = None
    array_flags: list[dict[str, Any]] = []
    if bool(config.check_prediction_arrays):
        all_model_names = tuple(dict.fromkeys((*hlt_names, *frozen_names, *adapted_names)))
        prediction_array_audit, array_flags = summarize_prediction_arrays(
            prediction_dir=config.prediction_dir,
            model_names=all_model_names,
            splits=splits,
        )
    flags = checkpoint_flags + frozen_flags + adapted_flags + hlt_flags + fusion_flags + array_flags
    error_count = sum(1 for flag in flags if flag.get("severity") == "error")
    warning_count = sum(1 for flag in flags if flag.get("severity") == "warning")
    report = {
        "experiment_step": AGGRESSIVE_AUDIT_STEP,
        "ok": error_count == 0,
        "error_count": int(error_count),
        "warning_count": int(warning_count),
        "flags": flags,
        "config": {
            "prediction_dir": str(config.prediction_dir),
            "reco_model_dir": str(config.reco_model_dir),
            "adapted_tagger_dir": str(config.adapted_tagger_dir),
            "output_dir": str(config.output_dir),
            "fusion_report": config.fusion_report,
            "reconstructors": list(reconstructors),
            "teachers": list(teachers),
            "splits": list(splits),
            "fusion_groups": list(config.fusion_groups),
            "expected_split_sizes": dict(config.expected_split_sizes),
        },
        "split_sizes": dict(config.expected_split_sizes),
        "checkpoint_metadata": {
            "aggressive_reconstructors": reco_reports,
            "aggressive_adapted_taggers": adapted_reports,
        },
        "prediction_metadata": {
            "hlt4": hlt_predictions,
            "aggressive_frozen_teacher": frozen_predictions,
            "aggressive_adapted_taggers": adapted_predictions,
        },
        "prediction_array_audit": prediction_array_audit,
        "fusion_groups": fusion_groups,
        "leakage_rules": {
            "model_train": "train reconstructors and adapted reco-domain taggers only",
            "model_val": "select reconstructor and adapted tagger checkpoints only",
            "stack_train": "fit fusion models only",
            "stack_val": "select/validate fusion models only",
            "final_test": "locked evaluation only; never used for checkpoint or fuser selection",
        },
    }
    save_json(output_dir / "aggressive_audit_report.json", report)
    write_aggressive_audit_summary(report, output_dir)
    return report


__all__ = [
    "AGGRESSIVE_AUDIT_STEP",
    "AggressiveAuditConfig",
    "AggressiveReconstructionDiagnosticsAccumulator",
    "HistogramAccumulator",
    "RunningStats",
    "run_aggressive_audit",
    "summarize_checkpoint_reports",
    "summarize_fusion_groups",
    "summarize_prediction_arrays",
    "summarize_prediction_metadata",
    "write_aggressive_audit_summary",
]
