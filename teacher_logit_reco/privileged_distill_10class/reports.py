"""PD10 Step 8 final report writer."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.fusion import load_prediction_block
from jetclass_fresh.hlt_baseline import save_json
from jetclass_fresh.jetclass_data import LABEL_NAMES

from .config import (
    PD10_DEFAULT_ALPHA,
    PD10_DEFAULT_TEMPERATURE,
    PD10_EXPERIMENT_NAME,
    PD10_NUM_CLASSES,
    PD10_STUDENT_INIT_SCRATCH,
    PD10_STUDENT_INIT_WARM_START,
    PD10_TARGET_CONFIDENCE_WEIGHTED_PLUS_REP,
    PD10_TARGET_FULL_LOGITS,
    PD10_TARGET_FULL_LOGITS_PLUS_REP,
    PD10_TARGET_REP_ONLY,
    PD10_TARGET_TOP3_PLUS_REP,
    PD10_TEACHER_DUAL_VIEW,
    PD10_TEACHER_HLT,
    PD10_TEACHER_NONE,
    PD10_TEACHER_OFFLINE,
    PD10_TEACHER_PARTICLE_DUAL_VIEW,
    build_pd10_core_student_variants,
    build_pd10_priority_student_variants,
    default_pd10_experiment_layout,
    pd10_extended_student_variant_name,
    pd10_extended_teacher_model_name,
    pd10_target_mode_uses_logits,
    pd10_target_mode_uses_representations,
)
from .dual_view_teacher import load_pd10_dual_view_logit_block
from .logits import load_pd10_teacher_logit_block
from .metrics import pd10_prediction_metrics_from_logits
from .particle_dual_view_teacher import load_pd10_particle_dual_view_logit_block


PD10_STEP8_EXPERIMENT_STEP = "pd10_step8_final_report"
PD10_REPORT_CONTRACT = "pd10_privileged_distillation_final_report_v1"
PD10_REPORT_JSON = "pd10_report.json"
PD10_REPORT_MD = "pd10_report.md"
PD10_REPORT_RUN_JSON = "run_report.json"

PD10_REPORT_TABLES: dict[str, str] = {
    "teacher_metrics": "teacher_metrics.csv",
    "student_metrics": "student_metrics.csv",
    "student_core_matrix": "student_core_matrix.csv",
    "warm_start_comparisons": "warm_start_comparisons.csv",
    "scratch_comparisons": "scratch_comparisons.csv",
    "teacher_target_comparison": "teacher_target_comparison.csv",
    "topk_confidence_ablations": "topk_confidence_ablations.csv",
    "binary_projection_table": "binary_projection_table.csv",
    "gap_closure_table": "gap_closure_table.csv",
    "calibration_table": "calibration_table.csv",
    "class_pair_improvements": "class_pair_improvements.csv",
    "leakage_audit_summary": "leakage_audit_summary.csv",
    "v2_comparisons": "v2_comparisons.csv",
    "v2_diagnostics": "v2_diagnostics.csv",
}


@dataclass(frozen=True)
class PD10ReportConfig:
    """Locations and policy for the PD10 final report."""

    output_dir: str
    teachers_dir: str
    students_dir: str
    teacher_logit_dir: str | None = None
    audit_dir: str | None = None
    student_variants: tuple[str, ...] = field(default_factory=tuple)
    include_priority_students: bool = True
    require_core_students: bool = True
    require_teacher_reports: bool = True
    require_audit: bool = True
    include_prediction_metrics: bool = True
    include_v2_students: bool = True
    confirm_final_test: bool = False

    def __post_init__(self) -> None:
        if not bool(self.confirm_final_test):
            raise ValueError("PD10 final report requires confirm_final_test=True")
        variants = tuple(str(item) for item in self.student_variants)
        if not variants:
            specs = list(build_pd10_core_student_variants())
            if self.include_priority_students:
                specs.extend(build_pd10_priority_student_variants())
            variants = tuple(spec.name for spec in specs)
            if self.include_v2_students:
                variants = tuple(dict.fromkeys([*variants, *_discover_v2_student_variants(self.students_dir)]))
        if len(set(variants)) != len(variants):
            raise ValueError("student_variants contains duplicates")
        object.__setattr__(self, "student_variants", variants)


def pd10_report_dir(*, output_root: str | Path = "checkpoints") -> Path:
    return default_pd10_experiment_layout(output_root=output_root).final_report_dir


def default_pd10_report_config(*, output_root: str | Path = "checkpoints", confirm_final_test: bool = False) -> PD10ReportConfig:
    layout = default_pd10_experiment_layout(output_root=output_root)
    return PD10ReportConfig(
        output_dir=str(layout.final_report_dir),
        teachers_dir=str(layout.teachers_dir),
        students_dir=str(layout.students_dir),
        teacher_logit_dir=str(layout.teacher_logits_dir),
        audit_dir=str(layout.step2_audit_dir),
        confirm_final_test=confirm_final_test,
    )


def _read_json(path: str | Path | None) -> Any | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _default_v2_student_variants() -> tuple[str, ...]:
    """Student directory names for the planned additive PD10-V2 matrix."""

    return (
        pd10_extended_student_variant_name(
            PD10_STUDENT_INIT_WARM_START,
            PD10_TEACHER_PARTICLE_DUAL_VIEW,
            PD10_TARGET_FULL_LOGITS,
        ),
        pd10_extended_student_variant_name(
            PD10_STUDENT_INIT_WARM_START,
            PD10_TEACHER_PARTICLE_DUAL_VIEW,
            PD10_TARGET_REP_ONLY,
        ),
        pd10_extended_student_variant_name(
            PD10_STUDENT_INIT_WARM_START,
            PD10_TEACHER_PARTICLE_DUAL_VIEW,
            PD10_TARGET_FULL_LOGITS_PLUS_REP,
        ),
        pd10_extended_student_variant_name(
            PD10_STUDENT_INIT_WARM_START,
            PD10_TEACHER_DUAL_VIEW,
            PD10_TARGET_FULL_LOGITS_PLUS_REP,
        ),
        pd10_extended_student_variant_name(
            PD10_STUDENT_INIT_WARM_START,
            PD10_TEACHER_PARTICLE_DUAL_VIEW,
            PD10_TARGET_FULL_LOGITS_PLUS_REP,
            representation_beta=0.3,
        ),
        pd10_extended_student_variant_name(
            PD10_STUDENT_INIT_WARM_START,
            PD10_TEACHER_PARTICLE_DUAL_VIEW,
            PD10_TARGET_TOP3_PLUS_REP,
        ),
        pd10_extended_student_variant_name(
            PD10_STUDENT_INIT_WARM_START,
            PD10_TEACHER_PARTICLE_DUAL_VIEW,
            PD10_TARGET_CONFIDENCE_WEIGHTED_PLUS_REP,
        ),
        pd10_extended_student_variant_name(
            PD10_STUDENT_INIT_SCRATCH,
            PD10_TEACHER_PARTICLE_DUAL_VIEW,
            PD10_TARGET_FULL_LOGITS_PLUS_REP,
        ),
    )


def _target_mode_uses_logits(target_mode: Any) -> bool:
    try:
        return bool(pd10_target_mode_uses_logits(str(target_mode)))
    except Exception:
        return False


def _target_mode_uses_representations(target_mode: Any) -> bool:
    try:
        return bool(pd10_target_mode_uses_representations(str(target_mode)))
    except Exception:
        return False


def _is_v2_student_report(report: Mapping[str, Any] | None, *, variant: str | None = None) -> bool:
    if variant in set(_default_v2_student_variants()):
        return True
    if not isinstance(report, Mapping):
        return False
    teacher = report.get("teacher_target")
    target_mode = report.get("target_mode")
    return (
        teacher == PD10_TEACHER_PARTICLE_DUAL_VIEW
        or _target_mode_uses_representations(target_mode)
        or bool(report.get("uses_representations"))
    )


def _discover_v2_student_variants(students_dir: str | Path) -> tuple[str, ...]:
    root = Path(students_dir)
    if not root.exists():
        return ()
    variants: list[str] = []
    for report_path in sorted(root.glob("*/run_report.json")):
        try:
            report = _read_json(report_path)
        except Exception:
            continue
        if isinstance(report, Mapping):
            variant = str(report.get("variant_name") or report_path.parent.name)
            candidate_report: Mapping[str, Any] | None = report
        else:
            variant = report_path.parent.name
            candidate_report = None
        if _is_v2_student_report(candidate_report, variant=variant):
            variants.append(report_path.parent.name)
    return tuple(dict.fromkeys(variants))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_jsonable(value), sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(str(key))
                seen.add(str(key))
    if not keys:
        rows = [{"available": False, "reason": "no rows"}]
        keys = ["available", "reason"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _metric_value(metrics: Mapping[str, Any] | None, *keys: str) -> Any:
    if not isinstance(metrics, Mapping):
        return None
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    return None


def _normalize_metrics(metrics: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metrics, Mapping):
        return None
    accuracy = _metric_value(metrics, "accuracy", "acc")
    cross_entropy = _metric_value(metrics, "cross_entropy", "ce_loss", "loss", "best_model_val_loss")
    n_jets = _metric_value(metrics, "n_jets", "num_jets", "n")
    result = {
        "accuracy": _finite_float(accuracy),
        "cross_entropy": _finite_float(cross_entropy),
        "n_jets": None if n_jets is None else int(n_jets),
    }
    for key in (
        "loss",
        "ce_loss",
        "kd_loss",
        "effective_kd_alpha",
        "rep_loss",
        "effective_representation_beta",
        "teacher_student_representation_cosine",
        "teacher_student_logit_kl",
        "representation_beta",
        "representation_dim",
        "representation_mode",
        "expected_calibration_error",
        "ece",
        "mean_confidence",
        "macro_ovr_auc",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "fpr_at_signal_eff_0p30_macro",
        "fpr_at_signal_eff_0p50_macro",
        "background_rejection_at_signal_eff_0p30_macro",
        "background_rejection_at_signal_eff_0p50_macro",
        "validation_threshold_fpr_at_signal_eff_0p30_macro",
        "validation_threshold_fpr_at_signal_eff_0p50_macro",
        "validation_binary_fpr_at_signal_eff_0p30_macro",
        "validation_binary_fpr_at_signal_eff_0p50_macro",
        "validation_binary_background_rejection_at_signal_eff_0p30_macro",
        "validation_binary_background_rejection_at_signal_eff_0p50_macro",
        "confusion_matrix",
        "binary_metrics",
        "binary_projection_results",
        "binary_score_thresholds",
        "per_class_accuracy",
        "per_class_metrics",
        "score_thresholds_by_class",
        "validation_threshold_fpr",
        "validation_binary_threshold_fpr",
    ):
        if key in metrics:
            result[key] = metrics[key]
    return result


def _metrics_from_report(report: Mapping[str, Any] | None, split: str) -> dict[str, Any] | None:
    if not isinstance(report, Mapping):
        return None
    candidates: list[Any] = []
    if split == "model_val":
        candidates.extend(
            [
                report.get("selected_model_val_metrics"),
                report.get("model_val_metrics"),
                report.get("best_model_val_metrics"),
                report.get("model_val"),
                report.get("model_val_report"),
            ]
        )
        candidates.append(
            {
                "accuracy": report.get("best_model_val_accuracy"),
                "cross_entropy": report.get("best_model_val_cross_entropy"),
                "loss": report.get("best_model_val_loss"),
            }
        )
    elif split == "final_test":
        candidates.extend(
            [
                report.get("final_test_metrics"),
                report.get("final_test"),
                report.get("metrics") if report.get("split") == "final_test" else None,
            ]
        )
        final_report = report.get("final_test_report")
        if isinstance(final_report, Mapping):
            candidates.extend(
                [
                    final_report.get("metrics"),
                    final_report.get("final_test_metrics"),
                    final_report.get("source_final_test_report"),
                    final_report,
                ]
            )
        source = report.get("source_final_test_report")
        if isinstance(source, Mapping):
            candidates.extend([source.get("metrics"), source])
    for candidate in candidates:
        metrics = _normalize_metrics(candidate if isinstance(candidate, Mapping) else None)
        if metrics is not None and (metrics.get("accuracy") is not None or metrics.get("cross_entropy") is not None):
            return metrics
    return None


def _load_teacher_logit_block(config: PD10ReportConfig, teacher_target: str, split: str):
    if teacher_target == PD10_TEACHER_DUAL_VIEW:
        return load_pd10_dual_view_logit_block(config.teacher_logit_dir, split)
    if teacher_target == PD10_TEACHER_PARTICLE_DUAL_VIEW:
        return load_pd10_particle_dual_view_logit_block(config.teacher_logit_dir, split)
    return load_pd10_teacher_logit_block(config.teacher_logit_dir, teacher_target, split)


def _prediction_metrics(config: PD10ReportConfig, teacher_target: str, split: str) -> dict[str, Any] | None:
    if not config.include_prediction_metrics or not config.teacher_logit_dir:
        return None
    try:
        block = _load_teacher_logit_block(config, teacher_target, split)
        validation_thresholds = None
        validation_binary_thresholds = None
        if split == "final_test":
            try:
                val_block = _load_teacher_logit_block(config, teacher_target, "model_val")
                val_metrics = pd10_prediction_metrics_from_logits(val_block.logits, val_block.labels)
                validation_thresholds = val_metrics.get("score_thresholds_by_class")
                validation_binary_thresholds = val_metrics.get("binary_score_thresholds")
            except Exception:
                validation_thresholds = None
                validation_binary_thresholds = None
    except Exception:
        return None
    metrics = pd10_prediction_metrics_from_logits(
        block.logits,
        block.labels,
        validation_thresholds_by_class=validation_thresholds,
        validation_binary_thresholds=validation_binary_thresholds,
    )
    metrics["n_jets"] = int(block.labels.shape[0])
    metrics["prediction_content_hash"] = block.metadata.get("prediction_content_hash")
    metrics["jet_identity_hash"] = block.metadata.get("jet_identity_hash")
    return metrics


def _student_prediction_metrics(config: PD10ReportConfig, variant: str, split: str) -> dict[str, Any] | None:
    if not config.include_prediction_metrics:
        return None
    prediction_dir = Path(config.students_dir) / variant / "student_predictions"
    try:
        block = load_prediction_block(prediction_dir, variant, split)
        validation_thresholds = None
        validation_binary_thresholds = None
        if split == "final_test":
            try:
                val_block = load_prediction_block(prediction_dir, variant, "model_val")
                val_metrics = pd10_prediction_metrics_from_logits(val_block.logits, val_block.labels)
                validation_thresholds = val_metrics.get("score_thresholds_by_class")
                validation_binary_thresholds = val_metrics.get("binary_score_thresholds")
            except Exception:
                validation_thresholds = None
                validation_binary_thresholds = None
    except Exception:
        return None
    metrics = pd10_prediction_metrics_from_logits(
        block.logits,
        block.labels,
        validation_thresholds_by_class=validation_thresholds,
        validation_binary_thresholds=validation_binary_thresholds,
    )
    metrics["n_jets"] = int(block.labels.shape[0])
    metrics["prediction_content_hash"] = block.metadata.get("prediction_content_hash")
    metrics["jet_identity_hash"] = block.metadata.get("jet_identity_hash")
    metrics["prediction_metadata_path"] = block.metadata.get("metadata_path")
    return metrics


def _merge_metric_payloads(
    report_metrics: Mapping[str, Any] | None,
    cache_metrics: Mapping[str, Any] | None,
    *,
    cache_source: str,
) -> tuple[dict[str, Any] | None, str | None]:
    report_norm = _normalize_metrics(report_metrics)
    cache_norm = _normalize_metrics(cache_metrics)
    if report_norm is None and cache_norm is None:
        return None, None
    if report_norm is not None and cache_norm is not None:
        merged = {**report_norm, **cache_norm}
        return merged, f"{cache_source}+run_report"
    if cache_norm is not None:
        return cache_norm, cache_source
    return report_norm, "run_report"


def _teacher_report_path(config: PD10ReportConfig, teacher_target: str) -> Path:
    return Path(config.teachers_dir) / pd10_extended_teacher_model_name(teacher_target) / "run_report.json"


def _teacher_logit_cache_exists(config: PD10ReportConfig, teacher_target: str) -> bool:
    if not config.teacher_logit_dir:
        return False
    model_name = pd10_extended_teacher_model_name(teacher_target)
    root = Path(config.teacher_logit_dir) / model_name
    return any((root / f"{split}_predictions_metadata.json").exists() for split in ("model_val", "final_test"))


def _teacher_targets_for_report(config: PD10ReportConfig) -> tuple[str, ...]:
    targets = [PD10_TEACHER_HLT, PD10_TEACHER_OFFLINE, PD10_TEACHER_DUAL_VIEW]
    particle_path = _teacher_report_path(config, PD10_TEACHER_PARTICLE_DUAL_VIEW)
    if particle_path.exists() or _teacher_logit_cache_exists(config, PD10_TEACHER_PARTICLE_DUAL_VIEW):
        targets.append(PD10_TEACHER_PARTICLE_DUAL_VIEW)
    return tuple(targets)


def _teacher_metric_rows(config: PD10ReportConfig, problems: list[str], warnings: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in _teacher_targets_for_report(config):
        path = _teacher_report_path(config, target)
        report = _read_json(path)
        if not isinstance(report, Mapping):
            message = f"missing teacher run_report for {target}: {path}"
            required_teacher = target != PD10_TEACHER_PARTICLE_DUAL_VIEW
            (problems if required_teacher and config.require_teacher_reports else warnings).append(message)
            report = {}
        for split in ("model_val", "final_test"):
            report_metrics = _metrics_from_report(report, split)
            cache_metrics = _prediction_metrics(config, target, split)
            metrics, source = _merge_metric_payloads(
                report_metrics,
                cache_metrics,
                cache_source="teacher_logit_cache",
            )
            if metrics is None:
                if split == "final_test":
                    warnings.append(f"missing {split} metrics for teacher {target}")
                continue
            rows.append(
                {
                    "row_type": "teacher",
                    "teacher_target": target,
                    "model_name": pd10_extended_teacher_model_name(target),
                    "split": split,
                    "accuracy": metrics.get("accuracy"),
                    "cross_entropy": metrics.get("cross_entropy"),
                    "n_jets": metrics.get("n_jets"),
                    "macro_ovr_auc": metrics.get("macro_ovr_auc"),
                    "expected_calibration_error": metrics.get("expected_calibration_error") or metrics.get("ece"),
                    "mean_confidence": metrics.get("mean_confidence"),
                    "fpr_at_signal_eff_0p50_macro": metrics.get("fpr_at_signal_eff_0p50_macro"),
                    "background_rejection_at_signal_eff_0p50_macro": metrics.get(
                        "background_rejection_at_signal_eff_0p50_macro"
                    ),
                    "validation_threshold_fpr_at_signal_eff_0p50_macro": metrics.get(
                        "validation_threshold_fpr_at_signal_eff_0p50_macro"
                    ),
                    "validation_binary_fpr_at_signal_eff_0p50_macro": metrics.get(
                        "validation_binary_fpr_at_signal_eff_0p50_macro"
                    ),
                    "metrics_source": source,
                    "report_path": str(path),
                    "metrics": metrics,
                }
            )
    return rows


def _student_specs_by_name() -> dict[str, Any]:
    specs = list(build_pd10_core_student_variants()) + list(build_pd10_priority_student_variants())
    return {spec.name: spec for spec in specs}


def _student_group(variant: str, report: Mapping[str, Any], *, spec: Any | None, is_core: bool) -> str:
    if is_core:
        return "core"
    if spec is not None:
        return "priority"
    if _is_v2_student_report(report, variant=variant):
        return "v2"
    return "custom"


def _student_metric_rows(config: PD10ReportConfig, problems: list[str], warnings: list[str]) -> list[dict[str, Any]]:
    specs_by_name = _student_specs_by_name()
    core_names = {spec.name for spec in build_pd10_core_student_variants()}
    rows: list[dict[str, Any]] = []
    for variant in config.student_variants:
        spec = specs_by_name.get(variant)
        path = Path(config.students_dir) / variant / "run_report.json"
        report = _read_json(path)
        is_core = variant in core_names
        if not isinstance(report, Mapping):
            message = f"missing student run_report for {variant}: {path}"
            (problems if is_core and config.require_core_students else warnings).append(message)
            continue
        init_mode = report.get("student_init") or (None if spec is None else spec.init_mode)
        teacher_target = report.get("teacher_target") or (None if spec is None else spec.teacher_target)
        target_mode = report.get("target_mode") or (None if spec is None else spec.target_mode)
        temperature = report.get("temperature", None if spec is None else spec.temperature)
        kd_alpha = report.get("kd_alpha", None if spec is None else spec.kd_alpha)
        top_k = report.get("top_k", None if spec is None else spec.top_k)
        group = _student_group(variant, report, spec=spec, is_core=is_core)
        uses_logit_teacher = report.get("uses_logit_teacher")
        if uses_logit_teacher is None:
            uses_logit_teacher = teacher_target != PD10_TEACHER_NONE and _target_mode_uses_logits(target_mode)
        uses_representations = report.get("uses_representations")
        if uses_representations is None:
            uses_representations = teacher_target != PD10_TEACHER_NONE and _target_mode_uses_representations(target_mode)
        for split in ("model_val", "final_test"):
            report_metrics = _metrics_from_report(report, split)
            cache_metrics = _student_prediction_metrics(config, variant, split)
            metrics, source = _merge_metric_payloads(
                report_metrics,
                cache_metrics,
                cache_source="student_prediction_cache",
            )
            if metrics is None:
                if split == "final_test" and is_core and config.require_core_students:
                    problems.append(f"missing final_test metrics for core student {variant}")
                continue
            rows.append(
                {
                    "row_type": "student",
                    "variant": variant,
                    "group": group,
                    "student_init": init_mode,
                    "teacher_target": teacher_target,
                    "target_mode": target_mode,
                    "temperature": temperature,
                    "kd_alpha": kd_alpha,
                    "top_k": top_k,
                    "representation_beta": report.get("representation_beta"),
                    "representation_dim": report.get("representation_dim"),
                    "representation_mode": report.get("representation_mode"),
                    "uses_logit_teacher": bool(uses_logit_teacher),
                    "uses_representations": bool(uses_representations),
                    "split": split,
                    "accuracy": metrics.get("accuracy"),
                    "cross_entropy": metrics.get("cross_entropy"),
                    "kd_loss": metrics.get("kd_loss"),
                    "rep_loss": metrics.get("rep_loss"),
                    "effective_kd_alpha": metrics.get("effective_kd_alpha"),
                    "effective_representation_beta": metrics.get("effective_representation_beta"),
                    "teacher_student_representation_cosine": metrics.get("teacher_student_representation_cosine"),
                    "teacher_student_logit_kl": metrics.get("teacher_student_logit_kl"),
                    "n_jets": metrics.get("n_jets"),
                    "macro_ovr_auc": metrics.get("macro_ovr_auc"),
                    "expected_calibration_error": metrics.get("expected_calibration_error") or metrics.get("ece"),
                    "mean_confidence": metrics.get("mean_confidence"),
                    "fpr_at_signal_eff_0p50_macro": metrics.get("fpr_at_signal_eff_0p50_macro"),
                    "background_rejection_at_signal_eff_0p50_macro": metrics.get(
                        "background_rejection_at_signal_eff_0p50_macro"
                    ),
                    "validation_threshold_fpr_at_signal_eff_0p50_macro": metrics.get(
                        "validation_threshold_fpr_at_signal_eff_0p50_macro"
                    ),
                    "validation_binary_fpr_at_signal_eff_0p50_macro": metrics.get(
                        "validation_binary_fpr_at_signal_eff_0p50_macro"
                    ),
                    "best_epoch": report.get("best_epoch"),
                    "selection_metric": report.get("selection_metric"),
                    "checkpoint": report.get("checkpoint"),
                    "report_path": str(path),
                    "metrics_source": source,
                    "teacher_logits_train_time_only": report.get("teacher_logits_train_time_only"),
                    "teacher_representations_train_time_only": report.get("teacher_representations_train_time_only"),
                    "inference_requires_teacher_logits": report.get("inference_requires_teacher_logits"),
                    "inference_requires_teacher_representations": report.get(
                        "inference_requires_teacher_representations"
                    ),
                    "inference_requires_offline_inputs": report.get("inference_requires_offline_inputs"),
                    "inference_export_requires_teacher_features": report.get(
                        "inference_export_requires_teacher_features"
                    ),
                    "teacher_logit_cache": report.get("teacher_logit_cache"),
                    "teacher_representation_cache": report.get("teacher_representation_cache"),
                    "metrics": metrics,
                }
            )
    return rows


def _final_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("split") == "final_test" and _finite_float(row.get("accuracy")) is not None]


def _row_metric_value(row: Mapping[str, Any] | None, key: str) -> float | None:
    if row is None:
        return None
    value = _finite_float(row.get(key))
    if value is not None:
        return value
    metrics = row.get("metrics")
    if isinstance(metrics, Mapping):
        return _finite_float(metrics.get(key))
    return None


def _row_fpr_value(row: Mapping[str, Any] | None) -> float | None:
    value = _row_metric_value(row, "validation_threshold_fpr_at_signal_eff_0p50_macro")
    if value is not None:
        return value
    return _row_metric_value(row, "fpr_at_signal_eff_0p50_macro")


def _best_outcome_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = _final_rows(rows)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            _row_metric_value(row, "cross_entropy")
            if _row_metric_value(row, "cross_entropy") is not None
            else float("inf"),
            -(_row_metric_value(row, "accuracy") or float("-inf")),
            -(_row_metric_value(row, "macro_ovr_auc") or float("-inf")),
            _row_fpr_value(row) if _row_fpr_value(row) is not None else float("inf"),
        ),
    )


def _best_accuracy_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Backward-compatible name; ranking is now CE-first per the PD10-V2 plan."""

    return _best_outcome_row(rows)


def _best_final_accuracy_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = _final_rows(rows)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            _row_metric_value(row, "accuracy") or float("-inf"),
            -(_row_metric_value(row, "cross_entropy") or float("inf")),
        ),
    )


def _teacher_final_accuracy(teacher_rows: Sequence[Mapping[str, Any]], target: str) -> float | None:
    for row in teacher_rows:
        if row.get("teacher_target") == target and row.get("split") == "final_test":
            return _finite_float(row.get("accuracy"))
    return None


def _teacher_final_metric(teacher_rows: Sequence[Mapping[str, Any]], target: str, key: str) -> float | None:
    for row in teacher_rows:
        if row.get("teacher_target") != target or row.get("split") != "final_test":
            continue
        value = _finite_float(row.get(key))
        if value is not None:
            return value
        metrics = row.get("metrics")
        if isinstance(metrics, Mapping):
            return _finite_float(metrics.get(key))
    return None


def _student_final_by_condition(student_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, float, float, int], Mapping[str, Any]]:
    result: dict[tuple[str, str, str, float, float, int], Mapping[str, Any]] = {}
    for row in _final_rows(student_rows):
        key = (
            str(row.get("student_init")),
            str(row.get("teacher_target")),
            str(row.get("target_mode")),
            float(row.get("temperature") or PD10_DEFAULT_TEMPERATURE),
            float(row.get("kd_alpha") or 0.0),
            int(row.get("top_k") or 0),
        )
        result[key] = row
    return result


def _find_default_student(
    condition_rows: Mapping[tuple[str, str, str, float, float, int], Mapping[str, Any]],
    init_mode: str,
    teacher_target: str,
) -> Mapping[str, Any] | None:
    target_alpha = 0.0 if teacher_target == PD10_TEACHER_NONE else PD10_DEFAULT_ALPHA
    for key, row in condition_rows.items():
        init, teacher, mode, temperature, alpha, _top_k = key
        if (
            init == init_mode
            and teacher == teacher_target
            and mode == PD10_TARGET_FULL_LOGITS
            and np.isclose(temperature, PD10_DEFAULT_TEMPERATURE)
            and np.isclose(alpha, target_alpha)
        ):
            return row
    return None


def _comparison(candidate: Mapping[str, Any] | None, baseline: Mapping[str, Any] | None) -> dict[str, Any]:
    candidate_acc = _row_metric_value(candidate, "accuracy")
    baseline_acc = _row_metric_value(baseline, "accuracy")
    candidate_ce = _row_metric_value(candidate, "cross_entropy")
    baseline_ce = _row_metric_value(baseline, "cross_entropy")
    candidate_auc = _row_metric_value(candidate, "macro_ovr_auc")
    baseline_auc = _row_metric_value(baseline, "macro_ovr_auc")
    candidate_fpr = _row_fpr_value(candidate)
    baseline_fpr = _row_fpr_value(baseline)
    delta = None if candidate_acc is None or baseline_acc is None else candidate_acc - baseline_acc
    delta_ce = None if candidate_ce is None or baseline_ce is None else candidate_ce - baseline_ce
    delta_auc = None if candidate_auc is None or baseline_auc is None else candidate_auc - baseline_auc
    delta_fpr = None if candidate_fpr is None or baseline_fpr is None else candidate_fpr - baseline_fpr
    required = (delta, delta_ce, delta_auc, delta_fpr)
    beats = None
    if all(value is not None for value in required):
        beats = bool(delta > 0.0 and delta_ce < 0.0 and delta_auc > 0.0 and delta_fpr < 0.0)
    return {
        "available": candidate_acc is not None and baseline_acc is not None,
        "candidate_accuracy": candidate_acc,
        "baseline_accuracy": baseline_acc,
        "delta_accuracy": delta,
        "candidate_cross_entropy": candidate_ce,
        "baseline_cross_entropy": baseline_ce,
        "delta_cross_entropy": delta_ce,
        "candidate_macro_ovr_auc": candidate_auc,
        "baseline_macro_ovr_auc": baseline_auc,
        "delta_macro_ovr_auc": delta_auc,
        "candidate_fpr_at_signal_eff_0p50_macro": candidate_fpr,
        "baseline_fpr_at_signal_eff_0p50_macro": baseline_fpr,
        "delta_fpr_at_signal_eff_0p50_macro": delta_fpr,
        "beats_baseline": beats,
        "candidate_variant": None if candidate is None else candidate.get("variant"),
        "baseline_variant": None if baseline is None else baseline.get("variant"),
    }


def _init_comparison_rows(student_rows: Sequence[Mapping[str, Any]], init_mode: str) -> list[dict[str, Any]]:
    condition_rows = _student_final_by_condition(student_rows)
    baseline = _find_default_student(condition_rows, init_mode, PD10_TEACHER_NONE)
    rows: list[dict[str, Any]] = []
    for teacher in (PD10_TEACHER_HLT, PD10_TEACHER_OFFLINE, PD10_TEACHER_DUAL_VIEW):
        candidate = _find_default_student(condition_rows, init_mode, teacher)
        row = {
            "student_init": init_mode,
            "teacher_target": teacher,
            "baseline": "ce_only",
            **_comparison(candidate, baseline),
        }
        rows.append(row)
    return rows


def _teacher_target_comparison_rows(student_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    condition_rows = _student_final_by_condition(student_rows)
    rows: list[dict[str, Any]] = []
    for init_mode in (PD10_STUDENT_INIT_SCRATCH, PD10_STUDENT_INIT_WARM_START):
        hlt = _find_default_student(condition_rows, init_mode, PD10_TEACHER_HLT)
        offline = _find_default_student(condition_rows, init_mode, PD10_TEACHER_OFFLINE)
        dual = _find_default_student(condition_rows, init_mode, PD10_TEACHER_DUAL_VIEW)
        for target, candidate in ((PD10_TEACHER_OFFLINE, offline), (PD10_TEACHER_DUAL_VIEW, dual)):
            rows.append(
                {
                    "student_init": init_mode,
                    "candidate_teacher_target": target,
                    "baseline_teacher_target": PD10_TEACHER_HLT,
                    **_comparison(candidate, hlt),
                }
            )
        rows.append(
            {
                "student_init": init_mode,
                "candidate_teacher_target": PD10_TEACHER_DUAL_VIEW,
                "baseline_teacher_target": PD10_TEACHER_OFFLINE,
                **_comparison(dual, offline),
            }
        )
    return rows


def _topk_confidence_ablation_rows(student_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    condition_rows = _student_final_by_condition(student_rows)
    baseline = _find_default_student(condition_rows, PD10_STUDENT_INIT_WARM_START, PD10_TEACHER_DUAL_VIEW)
    rows: list[dict[str, Any]] = []
    for row in _final_rows(student_rows):
        if row.get("group") != "priority":
            continue
        rows.append(
            {
                "variant": row.get("variant"),
                "student_init": row.get("student_init"),
                "teacher_target": row.get("teacher_target"),
                "target_mode": row.get("target_mode"),
                "temperature": row.get("temperature"),
                "kd_alpha": row.get("kd_alpha"),
                "top_k": row.get("top_k"),
                "baseline": None if baseline is None else baseline.get("variant"),
                **_comparison(row, baseline),
            }
        )
    if not rows:
        rows.append({"available": False, "reason": "no priority ablation student reports found"})
    return rows


def _final_student_by_variant(student_rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("variant")): row for row in _final_rows(student_rows) if row.get("variant") is not None}


def _teacher_final_row(teacher_rows: Sequence[Mapping[str, Any]], teacher_target: str) -> Mapping[str, Any] | None:
    return next(
        (
            row
            for row in teacher_rows
            if row.get("teacher_target") == teacher_target and row.get("split") == "final_test"
        ),
        None,
    )


def _row_source_name(row: Mapping[str, Any] | None) -> Any:
    if row is None:
        return None
    return row.get("variant") or row.get("teacher_target")


def _row_metric(row: Mapping[str, Any] | None, key: str) -> float | None:
    return _row_metric_value(row, key)


def _row_fpr_0p50(row: Mapping[str, Any] | None) -> float | None:
    return _row_fpr_value(row)


def _v2_comparison(
    candidate: Mapping[str, Any] | None,
    baseline: Mapping[str, Any] | None,
    *,
    comparison_type: str,
    baseline_label: str,
) -> dict[str, Any]:
    row = {
        "comparison_type": comparison_type,
        "baseline_label": baseline_label,
        "candidate_source": _row_source_name(candidate),
        "baseline_source": _row_source_name(baseline),
        "candidate_group": None if candidate is None else candidate.get("group"),
        "student_init": None if candidate is None else candidate.get("student_init"),
        "candidate_teacher_target": None if candidate is None else candidate.get("teacher_target"),
        "baseline_teacher_target": None if baseline is None else baseline.get("teacher_target"),
        "candidate_target_mode": None if candidate is None else candidate.get("target_mode"),
        "baseline_target_mode": None if baseline is None else baseline.get("target_mode"),
        "candidate_rep_loss": _row_metric(candidate, "rep_loss"),
        "candidate_effective_representation_beta": _row_metric(candidate, "effective_representation_beta"),
        "candidate_representation_beta": None if candidate is None else candidate.get("representation_beta"),
        "candidate_representation_mode": None if candidate is None else candidate.get("representation_mode"),
        **_comparison(candidate, baseline),
    }
    candidate_ce = _row_metric(candidate, "cross_entropy")
    baseline_ce = _row_metric(baseline, "cross_entropy")
    candidate_fpr = _row_fpr_0p50(candidate)
    baseline_fpr = _row_fpr_0p50(baseline)
    row.update(
        {
            "candidate_cross_entropy": candidate_ce,
            "baseline_cross_entropy": baseline_ce,
            "delta_cross_entropy": None if candidate_ce is None or baseline_ce is None else candidate_ce - baseline_ce,
            "candidate_fpr_at_signal_eff_0p50_macro": candidate_fpr,
            "baseline_fpr_at_signal_eff_0p50_macro": baseline_fpr,
            "delta_fpr_at_signal_eff_0p50_macro": None
            if candidate_fpr is None or baseline_fpr is None
            else candidate_fpr - baseline_fpr,
        }
    )
    return row


def _v2_comparison_rows(
    teacher_rows: Sequence[Mapping[str, Any]],
    student_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    condition_rows = _student_final_by_condition(student_rows)
    by_variant = _final_student_by_variant(student_rows)
    v2_students = [
        row
        for row in _final_rows(student_rows)
        if row.get("group") == "v2" or _is_v2_student_report(row, variant=str(row.get("variant")))
    ]
    for candidate in v2_students:
        init_mode = str(candidate.get("student_init") or PD10_STUDENT_INIT_WARM_START)
        ce_anchor = _find_default_student(condition_rows, init_mode, PD10_TEACHER_NONE)
        dual_anchor = _find_default_student(condition_rows, init_mode, PD10_TEACHER_DUAL_VIEW)
        rows.append(
            _v2_comparison(
                candidate,
                ce_anchor,
                comparison_type="student_vs_v04_anchor",
                baseline_label=f"{init_mode}_ce_only",
            )
        )
        rows.append(
            _v2_comparison(
                candidate,
                dual_anchor,
                comparison_type="student_vs_v04_anchor",
                baseline_label=f"{init_mode}_dual_view_full_logits_t2_a0p5",
            )
        )

    particle_logit = by_variant.get(
        pd10_extended_student_variant_name(
            PD10_STUDENT_INIT_WARM_START,
            PD10_TEACHER_PARTICLE_DUAL_VIEW,
            PD10_TARGET_FULL_LOGITS,
        )
    )
    particle_logit_rep = by_variant.get(
        pd10_extended_student_variant_name(
            PD10_STUDENT_INIT_WARM_START,
            PD10_TEACHER_PARTICLE_DUAL_VIEW,
            PD10_TARGET_FULL_LOGITS_PLUS_REP,
        )
    )
    particle_rep = by_variant.get(
        pd10_extended_student_variant_name(
            PD10_STUDENT_INIT_WARM_START,
            PD10_TEACHER_PARTICLE_DUAL_VIEW,
            PD10_TARGET_REP_ONLY,
        )
    )
    logit_fusion_rep = by_variant.get(
        pd10_extended_student_variant_name(
            PD10_STUDENT_INIT_WARM_START,
            PD10_TEACHER_DUAL_VIEW,
            PD10_TARGET_FULL_LOGITS_PLUS_REP,
        )
    )
    if particle_logit_rep is not None or particle_logit is not None:
        rows.append(
            _v2_comparison(
                particle_logit_rep,
                particle_logit,
                comparison_type="v2_ablation",
                baseline_label="particle_dual_logit_only",
            )
        )
    if particle_rep is not None or particle_logit is not None:
        rows.append(
            _v2_comparison(
                particle_rep,
                particle_logit,
                comparison_type="v2_ablation",
                baseline_label="particle_dual_logit_only",
            )
        )
    if particle_logit_rep is not None or logit_fusion_rep is not None:
        rows.append(
            _v2_comparison(
                particle_logit_rep,
                logit_fusion_rep,
                comparison_type="v2_ablation",
                baseline_label="logit_fusion_dual_logit_rep_kd",
            )
        )

    particle_teacher = _teacher_final_row(teacher_rows, PD10_TEACHER_PARTICLE_DUAL_VIEW)
    if particle_teacher is not None:
        for baseline_target in (PD10_TEACHER_HLT, PD10_TEACHER_OFFLINE, PD10_TEACHER_DUAL_VIEW):
            rows.append(
                _v2_comparison(
                    particle_teacher,
                    _teacher_final_row(teacher_rows, baseline_target),
                    comparison_type="teacher_vs_anchor",
                    baseline_label=baseline_target,
                )
            )

    if not rows:
        rows.append({"available": False, "reason": "no V2 teacher/student reports found"})
    return rows


def _v2_diagnostic_rows(student_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in student_rows:
        if row.get("split") != "model_val":
            continue
        if row.get("group") != "v2" and not _is_v2_student_report(row, variant=str(row.get("variant"))):
            continue
        rows.append(
            {
                "variant": row.get("variant"),
                "student_init": row.get("student_init"),
                "teacher_target": row.get("teacher_target"),
                "target_mode": row.get("target_mode"),
                "uses_logit_teacher": row.get("uses_logit_teacher"),
                "uses_representations": row.get("uses_representations"),
                "model_val_cross_entropy": _row_metric(row, "cross_entropy"),
                "model_val_accuracy": _row_metric(row, "accuracy"),
                "model_val_kd_loss": _row_metric(row, "kd_loss"),
                "model_val_rep_loss": _row_metric(row, "rep_loss"),
                "model_val_teacher_student_logit_kl": _row_metric(row, "teacher_student_logit_kl"),
                "model_val_teacher_student_representation_cosine": _row_metric(
                    row,
                    "teacher_student_representation_cosine",
                ),
                "effective_kd_alpha": _row_metric(row, "effective_kd_alpha"),
                "effective_representation_beta": _row_metric(row, "effective_representation_beta"),
                "representation_beta": row.get("representation_beta"),
                "representation_mode": row.get("representation_mode"),
                "metrics_source": row.get("metrics_source"),
                "report_path": row.get("report_path"),
            }
        )
    if not rows:
        rows.append({"available": False, "reason": "no V2 model_val diagnostic rows found"})
    return rows


def _gap_closure_rows(
    student_rows: Sequence[Mapping[str, Any]],
    *,
    hlt_part_accuracy: float | None,
    offline_part_accuracy: float | None,
    hlt_part_fpr_0p50: float | None = None,
    offline_part_fpr_0p50: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gap = None
    if hlt_part_accuracy is not None and offline_part_accuracy is not None:
        gap = offline_part_accuracy - hlt_part_accuracy
    fpr_gap = None
    if hlt_part_fpr_0p50 is not None and offline_part_fpr_0p50 is not None:
        fpr_gap = hlt_part_fpr_0p50 - offline_part_fpr_0p50
    for row in _final_rows(student_rows):
        acc = _finite_float(row.get("accuracy"))
        closure = None
        remaining = None
        if acc is not None and hlt_part_accuracy is not None and gap is not None and abs(gap) > 1.0e-12:
            closure = (acc - hlt_part_accuracy) / gap
            remaining = offline_part_accuracy - acc if offline_part_accuracy is not None else None
        metrics = row.get("metrics")
        student_fpr = _finite_float(row.get("validation_threshold_fpr_at_signal_eff_0p50_macro"))
        if student_fpr is None:
            student_fpr = _finite_float(row.get("fpr_at_signal_eff_0p50_macro"))
        if student_fpr is None and isinstance(metrics, Mapping):
            student_fpr = _finite_float(metrics.get("validation_threshold_fpr_at_signal_eff_0p50_macro"))
        if student_fpr is None and isinstance(metrics, Mapping):
            student_fpr = _finite_float(metrics.get("fpr_at_signal_eff_0p50_macro"))
        fpr_closure = None
        remaining_fpr = None
        if (
            student_fpr is not None
            and hlt_part_fpr_0p50 is not None
            and offline_part_fpr_0p50 is not None
            and fpr_gap is not None
            and abs(fpr_gap) > 1.0e-12
        ):
            fpr_closure = (hlt_part_fpr_0p50 - student_fpr) / fpr_gap
            remaining_fpr = student_fpr - offline_part_fpr_0p50
        rows.append(
            {
                "variant": row.get("variant"),
                "student_init": row.get("student_init"),
                "teacher_target": row.get("teacher_target"),
                "target_mode": row.get("target_mode"),
                "student_accuracy": acc,
                "hlt_part_accuracy": hlt_part_accuracy,
                "offline_part_accuracy": offline_part_accuracy,
                "offline_gap": gap,
                "delta_vs_hlt_part": None if acc is None or hlt_part_accuracy is None else acc - hlt_part_accuracy,
                "gap_closure_fraction": closure,
                "remaining_gap_to_offline_part": remaining,
                "student_fpr_at_signal_eff_0p50_macro": student_fpr,
                "hlt_part_fpr_at_signal_eff_0p50_macro": hlt_part_fpr_0p50,
                "offline_part_fpr_at_signal_eff_0p50_macro": offline_part_fpr_0p50,
                "fpr_gap_hlt_to_offline": fpr_gap,
                "fpr_gap_closure_fraction": fpr_closure,
                "remaining_fpr_gap_to_offline_part": remaining_fpr,
            }
        )
    return rows


def _flatten_binary_rows(metric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in metric_rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        for key in ("binary_projection_results", "binary_metrics"):
            payload = metrics.get(key)
            if not isinstance(payload, Mapping):
                continue
            for name, value in payload.items():
                if isinstance(value, Mapping):
                    row_id = (
                        str(row.get("variant") or row.get("teacher_target")),
                        str(row.get("row_type")),
                        str(row.get("split")),
                        str(name),
                    )
                    if row_id in seen:
                        continue
                    seen.add(row_id)
                    rows.append(
                        {
                            "source": row.get("variant") or row.get("teacher_target"),
                            "row_type": row.get("row_type"),
                            "split": row.get("split"),
                            "binary_task": name,
                            "available": value.get("available", True),
                            "auc": value.get("auc"),
                            "fpr_at_signal_eff_0p30": value.get("fpr_at_signal_eff_0p30"),
                            "fpr_at_signal_eff_0p50": value.get("fpr_at_signal_eff_0p50"),
                            "background_rejection_at_signal_eff_0p50": value.get(
                                "background_rejection_at_signal_eff_0p50"
                            ),
                            "validation_thresholds_from_split": value.get("validation_thresholds_from_split"),
                            "validation_threshold_at_signal_eff_0p50": value.get(
                                "validation_threshold_at_signal_eff_0p50"
                            ),
                            "validation_signal_eff_at_threshold_0p50": value.get(
                                "validation_signal_eff_at_threshold_0p50"
                            ),
                            "validation_fpr_at_signal_eff_0p30": value.get(
                                "validation_fpr_at_signal_eff_0p30"
                            ),
                            "validation_fpr_at_signal_eff_0p50": value.get(
                                "validation_fpr_at_signal_eff_0p50"
                            ),
                            "validation_background_rejection_at_signal_eff_0p50": value.get(
                                "validation_background_rejection_at_signal_eff_0p50"
                            ),
                            "metrics": value,
                        }
                    )
    if not rows:
        rows.append(
            {
                "available": False,
                "reason": "No binary projection metrics were found in teacher/student run reports.",
            }
        )
    return rows


def _calibration_rows(metric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in metric_rows:
        ece = _finite_float(row.get("expected_calibration_error"))
        if ece is None:
            continue
        rows.append(
            {
                "source": row.get("variant") or row.get("teacher_target"),
                "row_type": row.get("row_type"),
                "split": row.get("split"),
                "accuracy": row.get("accuracy"),
                "cross_entropy": row.get("cross_entropy"),
                "expected_calibration_error": ece,
                "mean_confidence": row.get("mean_confidence"),
                "available": True,
            }
        )
    if not rows:
        rows.append(
            {
                "available": False,
                "reason": "No calibration metrics were found; cache prediction logits or add ECE to run reports.",
            }
        )
    return rows


def _confusion_matrix(metrics: Mapping[str, Any] | None) -> np.ndarray | None:
    if not isinstance(metrics, Mapping) or "confusion_matrix" not in metrics:
        return None
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
    if matrix.shape != (PD10_NUM_CLASSES, PD10_NUM_CLASSES):
        return None
    return matrix


def _class_pair_improvement_rows(
    teacher_rows: Sequence[Mapping[str, Any]],
    student_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hlt_teacher = next(
        (row for row in teacher_rows if row.get("teacher_target") == PD10_TEACHER_HLT and row.get("split") == "final_test"),
        None,
    )
    hlt_matrix = _confusion_matrix(hlt_teacher.get("metrics") if isinstance(hlt_teacher, Mapping) else None)
    if hlt_matrix is not None:
        for candidate in teacher_rows:
            if candidate.get("teacher_target") == PD10_TEACHER_HLT or candidate.get("split") != "final_test":
                continue
            cand_matrix = _confusion_matrix(candidate.get("metrics"))
            if cand_matrix is None:
                continue
            for truth in range(PD10_NUM_CLASSES):
                for pred in range(PD10_NUM_CLASSES):
                    if truth == pred:
                        continue
                    reduction = hlt_matrix[truth, pred] - cand_matrix[truth, pred]
                    if reduction > 0:
                        rows.append(
                            {
                                "source": candidate.get("teacher_target"),
                                "baseline": PD10_TEACHER_HLT,
                                "truth_class": int(truth),
                                "truth_label": LABEL_NAMES[truth],
                                "predicted_as": int(pred),
                                "predicted_label": LABEL_NAMES[pred],
                                "baseline_confusions": int(hlt_matrix[truth, pred]),
                                "candidate_confusions": int(cand_matrix[truth, pred]),
                                "confusion_reduction": int(reduction),
                            }
                        )
    condition_rows = _student_final_by_condition(student_rows)
    for init_mode in (PD10_STUDENT_INIT_SCRATCH, PD10_STUDENT_INIT_WARM_START):
        baseline = _find_default_student(condition_rows, init_mode, PD10_TEACHER_NONE)
        baseline_matrix = _confusion_matrix(baseline.get("metrics") if isinstance(baseline, Mapping) else None)
        if baseline_matrix is None:
            continue
        for candidate in _final_rows(student_rows):
            if candidate.get("student_init") != init_mode or candidate.get("teacher_target") == PD10_TEACHER_NONE:
                continue
            cand_matrix = _confusion_matrix(candidate.get("metrics"))
            if cand_matrix is None:
                continue
            for truth in range(PD10_NUM_CLASSES):
                for pred in range(PD10_NUM_CLASSES):
                    if truth == pred:
                        continue
                    reduction = baseline_matrix[truth, pred] - cand_matrix[truth, pred]
                    if reduction > 0:
                        rows.append(
                            {
                                "source": candidate.get("variant"),
                                "baseline": baseline.get("variant"),
                                "truth_class": int(truth),
                                "truth_label": LABEL_NAMES[truth],
                                "predicted_as": int(pred),
                                "predicted_label": LABEL_NAMES[pred],
                                "baseline_confusions": int(baseline_matrix[truth, pred]),
                                "candidate_confusions": int(cand_matrix[truth, pred]),
                                "confusion_reduction": int(reduction),
                            }
                        )
    if not rows:
        rows.append(
            {
                "available": False,
                "reason": "No confusion matrices were available to identify improved class-pair confusions.",
            }
        )
    return rows


def _audit_summary_rows(config: PD10ReportConfig, problems: list[str], warnings: list[str]) -> list[dict[str, Any]]:
    if not config.audit_dir:
        warnings.append("no audit_dir configured")
        return [{"available": False, "reason": "no audit_dir configured"}]
    audit_path = Path(config.audit_dir) / "pd10_step2_audit_report.json"
    audit = _read_json(audit_path)
    if not isinstance(audit, Mapping):
        message = f"missing Step 2 audit report: {audit_path}"
        (problems if config.require_audit else warnings).append(message)
        return [{"available": False, "reason": message, "path": str(audit_path)}]
    if not bool(audit.get("ok")):
        problems.append(f"Step 2 audit report is not ok: {audit_path}")
    rows = [
        {
            "available": True,
            "ok": audit.get("ok"),
            "experiment_name": audit.get("experiment_name"),
            "manifest_hash": audit.get("manifest_hash"),
            "hlt_degradation_strength": audit.get("hlt_degradation_strength"),
            "problems": audit.get("problems", []),
            "path": str(audit_path),
        }
    ]
    split_audit = audit.get("audits", {}).get("split_manifest", {}) if isinstance(audit.get("audits"), Mapping) else {}
    hlt_audit = audit.get("audits", {}).get("hlt_cache", {}) if isinstance(audit.get("audits"), Mapping) else {}
    rows.append(
        {
            "available": True,
            "section": "split_manifest",
            "ok": split_audit.get("ok"),
            "duplicate_within_split_count": split_audit.get("split_audit", {}).get("duplicate_within_split_count"),
            "cross_split_overlap_count": split_audit.get("split_audit", {}).get("cross_split_overlap_count"),
            "file_level_separation_claimed": split_audit.get("split_audit", {}).get("file_level_separation_claimed"),
        }
    )
    rows.append(
        {
            "available": True,
            "section": "hlt_cache",
            "ok": hlt_audit.get("ok"),
            "distinct_hlt_hashes_ok": hlt_audit.get("distinct_hlt_hashes_ok"),
        }
    )
    return rows


def _answer_summary(
    teacher_rows: Sequence[Mapping[str, Any]],
    student_rows: Sequence[Mapping[str, Any]],
    teacher_target_rows: Sequence[Mapping[str, Any]],
    warm_rows: Sequence[Mapping[str, Any]],
    scratch_rows: Sequence[Mapping[str, Any]],
    gap_rows: Sequence[Mapping[str, Any]],
    class_pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    hlt_acc = _teacher_final_accuracy(teacher_rows, PD10_TEACHER_HLT)
    offline_acc = _teacher_final_accuracy(teacher_rows, PD10_TEACHER_OFFLINE)
    particle_acc = _teacher_final_accuracy(teacher_rows, PD10_TEACHER_PARTICLE_DUAL_VIEW)
    best_student = _best_accuracy_row(student_rows)
    best_student_acc = None if best_student is None else _finite_float(best_student.get("accuracy"))
    best_accuracy_student = _best_final_accuracy_row(student_rows)
    best_accuracy_student_acc = _row_metric(best_accuracy_student, "accuracy")
    did_any_student_beat_hlt_part = None
    if hlt_acc is not None:
        student_final_accuracies = [_row_metric(row, "accuracy") for row in _final_rows(student_rows)]
        did_any_student_beat_hlt_part = any(
            accuracy is not None and accuracy > hlt_acc for accuracy in student_final_accuracies
        )
    v2_students = [
        row
        for row in _final_rows(student_rows)
        if row.get("group") == "v2" or _is_v2_student_report(row, variant=str(row.get("variant")))
    ]
    best_v2_student = _best_accuracy_row(v2_students)
    best_v2_acc = None if best_v2_student is None else _finite_float(best_v2_student.get("accuracy"))
    best_v2_ce = _row_metric(best_v2_student, "cross_entropy")
    best_v2_auc = _row_metric(best_v2_student, "macro_ovr_auc")
    best_v2_fpr = _row_fpr_0p50(best_v2_student)
    condition_rows = _student_final_by_condition(student_rows)
    warm_ce = _find_default_student(condition_rows, PD10_STUDENT_INIT_WARM_START, PD10_TEACHER_NONE)
    warm_dual = _find_default_student(condition_rows, PD10_STUDENT_INIT_WARM_START, PD10_TEACHER_DUAL_VIEW)
    warm_ce_acc = None if warm_ce is None else _finite_float(warm_ce.get("accuracy"))
    warm_dual_acc = None if warm_dual is None else _finite_float(warm_dual.get("accuracy"))
    best_v2_vs_warm_ce = _comparison(best_v2_student, warm_ce)
    best_v2_vs_warm_dual = _comparison(best_v2_student, warm_dual)
    best_gap = None
    finite_gap_rows = [row for row in gap_rows if _finite_float(row.get("gap_closure_fraction")) is not None]
    if finite_gap_rows:
        best_gap = max(finite_gap_rows, key=lambda row: _finite_float(row.get("gap_closure_fraction")) or -1.0e9)
    best_fpr_gap = None
    finite_fpr_gap_rows = [row for row in gap_rows if _finite_float(row.get("fpr_gap_closure_fraction")) is not None]
    if finite_fpr_gap_rows:
        best_fpr_gap = max(
            finite_fpr_gap_rows,
            key=lambda row: _finite_float(row.get("fpr_gap_closure_fraction")) or -1.0e9,
        )
    dual_vs_hlt = [
        row
        for row in teacher_target_rows
        if row.get("candidate_teacher_target") == PD10_TEACHER_DUAL_VIEW
        and row.get("baseline_teacher_target") == PD10_TEACHER_HLT
    ]
    dual_vs_offline = [
        row
        for row in teacher_target_rows
        if row.get("candidate_teacher_target") == PD10_TEACHER_DUAL_VIEW
        and row.get("baseline_teacher_target") == PD10_TEACHER_OFFLINE
    ]
    return {
        "hlt_part_final_test_accuracy": hlt_acc,
        "offline_part_final_test_accuracy": offline_acc,
        "particle_dual_view_teacher_final_test_accuracy": particle_acc,
        "best_student_variant": None if best_student is None else best_student.get("variant"),
        "best_student_final_test_accuracy": best_student_acc,
        "best_student_delta_vs_hlt_part": None if best_student_acc is None or hlt_acc is None else best_student_acc - hlt_acc,
        "best_accuracy_student_variant": None if best_accuracy_student is None else best_accuracy_student.get("variant"),
        "best_accuracy_student_final_test_accuracy": best_accuracy_student_acc,
        "did_any_student_beat_hlt_part": did_any_student_beat_hlt_part,
        "v2_students_available": bool(v2_students),
        "best_v2_student_variant": None if best_v2_student is None else best_v2_student.get("variant"),
        "best_v2_student_final_test_accuracy": best_v2_acc,
        "best_v2_student_final_test_cross_entropy": best_v2_ce,
        "best_v2_student_final_test_macro_ovr_auc": best_v2_auc,
        "best_v2_student_final_test_fpr_at_signal_eff_0p50_macro": best_v2_fpr,
        "best_v2_delta_vs_hlt_part": None if best_v2_acc is None or hlt_acc is None else best_v2_acc - hlt_acc,
        "best_v2_delta_vs_warm_start_ce_only": None
        if best_v2_acc is None or warm_ce_acc is None
        else best_v2_acc - warm_ce_acc,
        "best_v2_multimetric_vs_warm_start_ce_only": best_v2_vs_warm_ce,
        "did_best_v2_beat_warm_start_ce_only": best_v2_vs_warm_ce.get("beats_baseline"),
        "best_v2_delta_vs_v04_warm_dual_view_kd": None
        if best_v2_acc is None or warm_dual_acc is None
        else best_v2_acc - warm_dual_acc,
        "best_v2_multimetric_vs_v04_warm_dual_view_kd": best_v2_vs_warm_dual,
        "did_best_v2_beat_v04_warm_dual_view_kd": best_v2_vs_warm_dual.get("beats_baseline"),
        "did_dual_view_kd_beat_hlt_self_kd": {
            str(row.get("student_init")): row.get("beats_baseline") for row in dual_vs_hlt
        },
        "did_dual_view_kd_beat_offline_kd": {
            str(row.get("student_init")): row.get("beats_baseline") for row in dual_vs_offline
        },
        "did_warm_start_kd_beat_warm_start_ce_only": any(
            row.get("beats_baseline") is True for row in warm_rows
        ),
        "did_scratch_kd_beat_scratch_ce_only": any(row.get("beats_baseline") is True for row in scratch_rows),
        "best_gap_closure_variant": None if best_gap is None else best_gap.get("variant"),
        "best_gap_closure_fraction": None if best_gap is None else best_gap.get("gap_closure_fraction"),
        "best_fpr_gap_closure_variant": None if best_fpr_gap is None else best_fpr_gap.get("variant"),
        "best_fpr_gap_closure_fraction": None if best_fpr_gap is None else best_fpr_gap.get("fpr_gap_closure_fraction"),
        "class_pair_improvements_available": bool(class_pair_rows and class_pair_rows[0].get("available", True)),
    }


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    answers = report.get("answers", {})
    outputs = report.get("outputs", {})
    lines = [
        "# PD10 Privileged Distillation Report",
        "",
        f"- Overall ok: {report.get('ok')}",
        f"- HLT ParT final-test accuracy: {answers.get('hlt_part_final_test_accuracy')}",
        f"- Offline ParT final-test accuracy: {answers.get('offline_part_final_test_accuracy')}",
        f"- Particle dual-view teacher final-test accuracy: {answers.get('particle_dual_view_teacher_final_test_accuracy')}",
        f"- Best student: `{answers.get('best_student_variant')}`",
        f"- Best student final-test accuracy: {answers.get('best_student_final_test_accuracy')}",
        f"- Accuracy-best student: `{answers.get('best_accuracy_student_variant')}`",
        f"- Accuracy-best student final-test accuracy: {answers.get('best_accuracy_student_final_test_accuracy')}",
        f"- Best student delta vs HLT ParT: {answers.get('best_student_delta_vs_hlt_part')}",
        "",
        "## Answers",
        "",
        f"- Did any student beat HLT ParT? {answers.get('did_any_student_beat_hlt_part')}",
        f"- Did dual-view KD beat HLT self-KD? {answers.get('did_dual_view_kd_beat_hlt_self_kd')}",
        f"- Did dual-view KD beat offline-only KD? {answers.get('did_dual_view_kd_beat_offline_kd')}",
        f"- Did warm-start KD beat warm-start CE-only? {answers.get('did_warm_start_kd_beat_warm_start_ce_only')}",
        f"- Did scratch KD beat scratch CE-only? {answers.get('did_scratch_kd_beat_scratch_ce_only')}",
        f"- Best offline-gap closure: `{answers.get('best_gap_closure_variant')}` at {answers.get('best_gap_closure_fraction')}",
        (
            f"- Best FPR gap closure: `{answers.get('best_fpr_gap_closure_variant')}` "
            f"at {answers.get('best_fpr_gap_closure_fraction')}"
        ),
        f"- Class-pair improvements available? {answers.get('class_pair_improvements_available')}",
        "",
        "## V2",
        "",
        f"- V2 students available? {answers.get('v2_students_available')}",
        f"- Best V2 student: `{answers.get('best_v2_student_variant')}`",
        f"- Best V2 final-test accuracy: {answers.get('best_v2_student_final_test_accuracy')}",
        f"- Best V2 delta vs HLT ParT: {answers.get('best_v2_delta_vs_hlt_part')}",
        f"- Best V2 delta vs warm-start CE-only: {answers.get('best_v2_delta_vs_warm_start_ce_only')}",
        f"- Best V2 delta vs v0.4 warm dual-view KD: {answers.get('best_v2_delta_vs_v04_warm_dual_view_kd')}",
        f"- Did best V2 beat warm-start CE-only? {answers.get('did_best_v2_beat_warm_start_ce_only')}",
        f"- Did best V2 beat v0.4 warm dual-view KD? {answers.get('did_best_v2_beat_v04_warm_dual_view_kd')}",
        "",
        "## Tables",
        "",
    ]
    for name, value in outputs.items():
        if name.endswith("_csv"):
            lines.append(f"- `{name}`: `{value}`")
    problems = report.get("problems") or []
    warnings = report.get("warnings") or []
    if problems:
        lines.extend(["", "## Problems", ""])
        lines.extend(f"- {problem}" for problem in problems)
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_pd10_report(config: PD10ReportConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    warnings: list[str] = []

    teacher_rows = _teacher_metric_rows(config, problems, warnings)
    student_rows = _student_metric_rows(config, problems, warnings)
    core_names = {spec.name for spec in build_pd10_core_student_variants()}
    student_core_matrix = [
        row
        for row in student_rows
        if row.get("variant") in core_names and row.get("split") == "final_test"
    ]
    warm_rows = _init_comparison_rows(student_rows, PD10_STUDENT_INIT_WARM_START)
    scratch_rows = _init_comparison_rows(student_rows, PD10_STUDENT_INIT_SCRATCH)
    teacher_target_rows = _teacher_target_comparison_rows(student_rows)
    ablation_rows = _topk_confidence_ablation_rows(student_rows)
    v2_rows = _v2_comparison_rows(teacher_rows, student_rows)
    v2_diagnostics = _v2_diagnostic_rows(student_rows)
    hlt_part_accuracy = _teacher_final_accuracy(teacher_rows, PD10_TEACHER_HLT)
    offline_part_accuracy = _teacher_final_accuracy(teacher_rows, PD10_TEACHER_OFFLINE)
    hlt_part_fpr_0p50 = _teacher_final_metric(
        teacher_rows,
        PD10_TEACHER_HLT,
        "validation_threshold_fpr_at_signal_eff_0p50_macro",
    )
    if hlt_part_fpr_0p50 is None:
        hlt_part_fpr_0p50 = _teacher_final_metric(teacher_rows, PD10_TEACHER_HLT, "fpr_at_signal_eff_0p50_macro")
    offline_part_fpr_0p50 = _teacher_final_metric(
        teacher_rows,
        PD10_TEACHER_OFFLINE,
        "validation_threshold_fpr_at_signal_eff_0p50_macro",
    )
    if offline_part_fpr_0p50 is None:
        offline_part_fpr_0p50 = _teacher_final_metric(
            teacher_rows,
            PD10_TEACHER_OFFLINE,
            "fpr_at_signal_eff_0p50_macro",
        )
    gap_rows = _gap_closure_rows(
        student_rows,
        hlt_part_accuracy=hlt_part_accuracy,
        offline_part_accuracy=offline_part_accuracy,
        hlt_part_fpr_0p50=hlt_part_fpr_0p50,
        offline_part_fpr_0p50=offline_part_fpr_0p50,
    )
    all_metric_rows = list(teacher_rows) + list(student_rows)
    binary_rows = _flatten_binary_rows(all_metric_rows)
    calibration_rows = _calibration_rows(all_metric_rows)
    class_pair_rows = _class_pair_improvement_rows(teacher_rows, student_rows)
    audit_rows = _audit_summary_rows(config, problems, warnings)
    answers = _answer_summary(
        teacher_rows,
        student_rows,
        teacher_target_rows,
        warm_rows,
        scratch_rows,
        gap_rows,
        class_pair_rows,
    )

    outputs = {
        "report_json": str(output_dir / PD10_REPORT_JSON),
        "report_md": str(output_dir / PD10_REPORT_MD),
        "run_report": str(output_dir / PD10_REPORT_RUN_JSON),
    }
    for key, filename in PD10_REPORT_TABLES.items():
        outputs[f"{key}_csv"] = str(output_dir / filename)

    report = {
        "ok": not problems,
        "contract": PD10_REPORT_CONTRACT,
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_STEP8_EXPERIMENT_STEP,
        "config": asdict(config),
        "answers": answers,
        "teacher_metric_rows": teacher_rows,
        "student_metric_rows": student_rows,
        "student_core_matrix": student_core_matrix,
        "warm_start_comparisons": warm_rows,
        "scratch_comparisons": scratch_rows,
        "teacher_target_comparison_rows": teacher_target_rows,
        "topk_confidence_ablation_rows": ablation_rows,
        "v2_comparison_rows": v2_rows,
        "v2_diagnostic_rows": v2_diagnostics,
        "binary_projection_rows": binary_rows,
        "gap_closure_rows": gap_rows,
        "calibration_rows": calibration_rows,
        "class_pair_improvement_rows": class_pair_rows,
        "leakage_audit_summary": audit_rows,
        "problems": problems,
        "warnings": warnings,
        "outputs": outputs,
    }

    _write_csv(output_dir / PD10_REPORT_TABLES["teacher_metrics"], teacher_rows)
    _write_csv(output_dir / PD10_REPORT_TABLES["student_metrics"], student_rows)
    _write_csv(output_dir / PD10_REPORT_TABLES["student_core_matrix"], student_core_matrix)
    _write_csv(output_dir / PD10_REPORT_TABLES["warm_start_comparisons"], warm_rows)
    _write_csv(output_dir / PD10_REPORT_TABLES["scratch_comparisons"], scratch_rows)
    _write_csv(output_dir / PD10_REPORT_TABLES["teacher_target_comparison"], teacher_target_rows)
    _write_csv(output_dir / PD10_REPORT_TABLES["topk_confidence_ablations"], ablation_rows)
    _write_csv(output_dir / PD10_REPORT_TABLES["v2_comparisons"], v2_rows)
    _write_csv(output_dir / PD10_REPORT_TABLES["v2_diagnostics"], v2_diagnostics)
    _write_csv(output_dir / PD10_REPORT_TABLES["binary_projection_table"], binary_rows)
    _write_csv(output_dir / PD10_REPORT_TABLES["gap_closure_table"], gap_rows)
    _write_csv(output_dir / PD10_REPORT_TABLES["calibration_table"], calibration_rows)
    _write_csv(output_dir / PD10_REPORT_TABLES["class_pair_improvements"], class_pair_rows)
    _write_csv(output_dir / PD10_REPORT_TABLES["leakage_audit_summary"], audit_rows)
    save_json(output_dir / PD10_REPORT_JSON, _jsonable(report))
    save_json(output_dir / PD10_REPORT_RUN_JSON, _jsonable(report))
    _write_markdown(output_dir / PD10_REPORT_MD, report)
    return report


def write_pd10_report(config: PD10ReportConfig) -> dict[str, Any]:
    return build_pd10_report(config)


__all__ = [
    "PD10_REPORT_CONTRACT",
    "PD10_REPORT_JSON",
    "PD10_REPORT_MD",
    "PD10_REPORT_RUN_JSON",
    "PD10_REPORT_TABLES",
    "PD10_STEP8_EXPERIMENT_STEP",
    "PD10ReportConfig",
    "build_pd10_report",
    "default_pd10_report_config",
    "pd10_report_dir",
    "write_pd10_report",
]
