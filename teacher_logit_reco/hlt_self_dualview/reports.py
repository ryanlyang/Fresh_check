"""Step 7 report aggregation for deployable HLT self-dualview studies."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.fusion import load_prediction_block
from jetclass_fresh.hlt_baseline import save_json

from teacher_logit_reco.privileged_distill_10class.config import (
    PD10_DEFAULT_ALPHA,
    PD10_DEFAULT_TEMPERATURE,
    PD10_STUDENT_INIT_WARM_START,
    PD10_TARGET_FULL_LOGITS,
    PD10_TEACHER_DUAL_VIEW,
    PD10_TEACHER_HLT,
    PD10_TEACHER_NONE,
    default_pd10_experiment_layout,
    pd10_extended_teacher_model_name,
    pd10_student_variant_name,
)
from teacher_logit_reco.privileged_distill_10class.metrics import pd10_prediction_metrics_from_logits

from .config import (
    HLT_SDV_DEFAULT_STRENGTHS,
    HLT_SDV_EXPERIMENT_NAME,
    HLT_SDV_PRIMARY_STRENGTH,
    HLT_SDV_VARIANT_HLT2_ONLY,
    HLT_SDV_VARIANT_SAME_VIEW,
    HLT_SDV_VARIANT_TTA,
    build_hlt_sdv_required_variants,
    default_hlt_sdv_experiment_layout,
    hlt_sdv_dual_hlt2_variant_name,
    hlt_sdv_strength_from_variant,
    normalize_hlt_sdv_variant,
)


HLT_SDV_STEP7_EXPERIMENT_STEP = "hlt_sdv_step7_final_report"
HLT_SDV_REPORT_CONTRACT = "hlt_self_dualview_10class_final_report_v1"
HLT_SDV_REPORT_JSON = "hlt_self_dualview_report.json"
HLT_SDV_REPORT_MD = "hlt_self_dualview_report.md"
HLT_SDV_REPORT_SUMMARY_JSON = "summary.json"
HLT_SDV_REPORT_RUN_JSON = "run_report.json"
HLT_SDV_METRIC_TABLE_CSV = "metric_table.csv"
HLT_SDV_COMPARISON_TABLE_CSV = "comparison_table.csv"
HLT_SDV_BINARY_TABLE_CSV = "binary_projection_table.csv"


@dataclass(frozen=True)
class HLTSDVReportConfig:
    """Locations and policy for the HLT self-dualview final report."""

    pd10_root: str
    output_dir: str | None = None
    sdv_models_dir: str | None = None
    pd10_teacher_logits_dir: str | None = None
    pd10_students_dir: str | None = None
    pd10_teachers_dir: str | None = None
    pd10_final_report_json: str | None = None
    variants: tuple[str, ...] = field(default_factory=tuple)
    hlt2_strengths: tuple[float, ...] = HLT_SDV_DEFAULT_STRENGTHS
    primary_strength: float = HLT_SDV_PRIMARY_STRENGTH
    include_prediction_metrics: bool = True
    require_sdv_variants: bool = True
    require_anchors: bool = False
    confirm_final_test: bool = False

    def __post_init__(self) -> None:
        if not bool(self.confirm_final_test):
            raise ValueError("HLT-SDV final report requires confirm_final_test=True")
        variants = tuple(normalize_hlt_sdv_variant(item) for item in self.variants)
        if not variants:
            variants = build_hlt_sdv_required_variants(self.hlt2_strengths)
        if len(set(variants)) != len(variants):
            raise ValueError("HLT-SDV report variants contain duplicates")
        object.__setattr__(self, "variants", variants)


def default_hlt_sdv_report_config(
    pd10_root: str | Path,
    *,
    confirm_final_test: bool = False,
) -> HLTSDVReportConfig:
    return HLTSDVReportConfig(pd10_root=str(pd10_root), confirm_final_test=confirm_final_test)


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _read_json(path: str | Path | None) -> Any | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _config_paths(config: HLTSDVReportConfig) -> dict[str, Path]:
    pd10_root = Path(config.pd10_root)
    sdv_layout = default_hlt_sdv_experiment_layout(
        output_root=pd10_root.parent,
        pd10_experiment_name=pd10_root.name,
    )
    pd10_layout = default_pd10_experiment_layout(
        output_root=pd10_root.parent,
        experiment_name=pd10_root.name,
    )
    return {
        "pd10_root": pd10_root,
        "output_dir": Path(config.output_dir) if config.output_dir else sdv_layout.final_report_dir,
        "sdv_models_dir": Path(config.sdv_models_dir) if config.sdv_models_dir else sdv_layout.models_dir,
        "pd10_teacher_logits_dir": Path(config.pd10_teacher_logits_dir)
        if config.pd10_teacher_logits_dir
        else pd10_layout.teacher_logits_dir,
        "pd10_students_dir": Path(config.pd10_students_dir) if config.pd10_students_dir else pd10_layout.students_dir,
        "pd10_teachers_dir": Path(config.pd10_teachers_dir) if config.pd10_teachers_dir else pd10_layout.teachers_dir,
        "pd10_final_report_json": Path(config.pd10_final_report_json)
        if config.pd10_final_report_json
        else pd10_layout.final_report_dir / "pd10_report.json",
    }


def _metric_from_mapping(metrics: Mapping[str, Any] | None, *keys: str) -> Any:
    if not isinstance(metrics, Mapping):
        return None
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def _metrics_from_prediction_block(
    prediction_dir: Path,
    model_name: str,
    split: str,
    *,
    include_prediction_metrics: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    if not include_prediction_metrics:
        return None, None
    try:
        block = load_prediction_block(prediction_dir, model_name, split)
    except Exception:
        return None, None
    validation_thresholds = None
    validation_binary_thresholds = None
    if split == "final_test":
        try:
            val_block = load_prediction_block(prediction_dir, model_name, "model_val")
            val_metrics = pd10_prediction_metrics_from_logits(val_block.logits, val_block.labels)
            validation_thresholds = val_metrics.get("score_thresholds_by_class")
            validation_binary_thresholds = val_metrics.get("binary_score_thresholds")
        except Exception:
            validation_thresholds = None
            validation_binary_thresholds = None
    metrics = pd10_prediction_metrics_from_logits(
        block.logits,
        block.labels,
        validation_thresholds_by_class=validation_thresholds,
        validation_binary_thresholds=validation_binary_thresholds,
    )
    metrics.update(
        {
            "n_jets": int(block.labels.shape[0]),
            "prediction_content_hash": block.metadata.get("prediction_content_hash"),
            "jet_identity_hash": block.metadata.get("jet_identity_hash"),
            "prediction_metadata_path": block.metadata.get("metadata_path"),
        }
    )
    return metrics, "prediction_cache"


def _normalize_metrics(metrics: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metrics, Mapping):
        return None
    result = dict(metrics)
    if "cross_entropy" not in result:
        value = _metric_from_mapping(result, "ce_loss", "loss", "best_model_val_loss")
        if value is not None:
            result["cross_entropy"] = value
    if "expected_calibration_error" not in result and "ece" in result:
        result["expected_calibration_error"] = result["ece"]
    return result


def _merge_metrics(
    report_metrics: Mapping[str, Any] | None,
    cache_metrics: Mapping[str, Any] | None,
    *,
    cache_source: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    report_norm = _normalize_metrics(report_metrics)
    cache_norm = _normalize_metrics(cache_metrics)
    if report_norm is None and cache_norm is None:
        return None, None
    if report_norm is not None and cache_norm is not None:
        return {**report_norm, **cache_norm}, f"{cache_source}+run_report"
    if cache_norm is not None:
        return cache_norm, cache_source
    return report_norm, "run_report"


def _row_from_metrics(
    *,
    row_type: str,
    name: str,
    split: str,
    group: str,
    metrics: Mapping[str, Any],
    metrics_source: str | None,
    report_path: Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = dict(metrics)
    row = {
        "row_type": row_type,
        "name": name,
        "variant": name if row_type in {"sdv", "control", "anchor_student"} else None,
        "model_name": name,
        "group": group,
        "split": split,
        "accuracy": metrics.get("accuracy"),
        "cross_entropy": metrics.get("cross_entropy"),
        "n_jets": metrics.get("n_jets"),
        "macro_ovr_auc": metrics.get("macro_ovr_auc"),
        "fpr_at_signal_eff_0p30_macro": metrics.get("fpr_at_signal_eff_0p30_macro"),
        "fpr_at_signal_eff_0p50_macro": metrics.get("fpr_at_signal_eff_0p50_macro"),
        "background_rejection_at_signal_eff_0p30_macro": metrics.get(
            "background_rejection_at_signal_eff_0p30_macro"
        ),
        "background_rejection_at_signal_eff_0p50_macro": metrics.get(
            "background_rejection_at_signal_eff_0p50_macro"
        ),
        "validation_threshold_fpr_at_signal_eff_0p30_macro": metrics.get(
            "validation_threshold_fpr_at_signal_eff_0p30_macro"
        ),
        "validation_threshold_fpr_at_signal_eff_0p50_macro": metrics.get(
            "validation_threshold_fpr_at_signal_eff_0p50_macro"
        ),
        "validation_binary_fpr_at_signal_eff_0p30_macro": metrics.get(
            "validation_binary_fpr_at_signal_eff_0p30_macro"
        ),
        "validation_binary_fpr_at_signal_eff_0p50_macro": metrics.get(
            "validation_binary_fpr_at_signal_eff_0p50_macro"
        ),
        "expected_calibration_error": metrics.get("expected_calibration_error") or metrics.get("ece"),
        "mean_confidence": metrics.get("mean_confidence"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "macro_f1": metrics.get("macro_f1"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "per_class_metrics": metrics.get("per_class_metrics"),
        "binary_metrics": metrics.get("binary_metrics") or metrics.get("binary_projection_results"),
        "metrics_source": metrics_source,
        "report_path": None if report_path is None else str(report_path),
        "metrics": metrics,
    }
    if extra:
        row.update(dict(extra))
    return row


def _sdv_group_for_variant(variant: str) -> str:
    if variant == HLT_SDV_VARIANT_SAME_VIEW:
        return "same_view_dual_fusion"
    if variant == HLT_SDV_VARIANT_HLT2_ONLY:
        return "hlt2_only_control"
    if variant == HLT_SDV_VARIANT_TTA:
        return "tta_control"
    if variant.startswith("sdv_hlt_hlt2_"):
        return "sdv_hlt_hlt2"
    return "custom_hlt_sdv"


def _sdv_report_metrics(report: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    if split == "model_val":
        return (
            report.get("model_val_prediction_metrics")
            or report.get("model_val_metrics")
            or report.get("best_model_val_training_metrics")
            or {
                "accuracy": report.get("best_model_val_accuracy"),
                "cross_entropy": report.get("best_model_val_cross_entropy"),
            }
        )
    return report.get("final_test_metrics") or report.get("metrics")


def _collect_sdv_rows(config: HLTSDVReportConfig, paths: Mapping[str, Path], warnings: list[str], problems: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    models_dir = paths["sdv_models_dir"]
    for variant in config.variants:
        run_path = models_dir / variant / "run_report.json"
        report = _read_json(run_path)
        if not isinstance(report, Mapping):
            message = f"missing HLT-SDV run_report for {variant}: {run_path}"
            (problems if config.require_sdv_variants else warnings).append(message)
            continue
        prediction_dir = models_dir / variant / "predictions"
        for split in ("model_val", "final_test"):
            cache_metrics, cache_source = _metrics_from_prediction_block(
                prediction_dir,
                variant,
                split,
                include_prediction_metrics=bool(config.include_prediction_metrics),
            )
            metrics, source = _merge_metrics(_sdv_report_metrics(report, split), cache_metrics, cache_source=cache_source)
            if metrics is None:
                warnings.append(f"missing {split} metrics for HLT-SDV variant {variant}")
                continue
            rows.append(
                _row_from_metrics(
                    row_type="sdv" if _sdv_group_for_variant(variant) == "sdv_hlt_hlt2" else "control",
                    name=variant,
                    split=split,
                    group=_sdv_group_for_variant(variant),
                    metrics=metrics,
                    metrics_source=source,
                    report_path=run_path,
                    extra={
                        "hlt2_strength": hlt_sdv_strength_from_variant(variant),
                        "selection_metric": report.get("selection_metric", "model_val_cross_entropy"),
                        "best_epoch": report.get("best_epoch"),
                        "checkpoint": report.get("checkpoint"),
                        "requires_offline_inputs": report.get("requires_offline_inputs"),
                        "requires_teacher_features": report.get("requires_teacher_features"),
                        "requires_deterministic_hlt2_transform": report.get(
                            "requires_deterministic_hlt2_transform"
                        ),
                        "no_final_test_used_for_selection": report.get("no_final_test_used_for_selection", True),
                    },
                )
            )
    return rows


def _teacher_anchor_report_metrics(report: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    if split == "final_test":
        final_report = report.get("final_test_report")
        if isinstance(final_report, Mapping):
            return final_report.get("metrics") or final_report
    if split == "model_val":
        model_val = report.get("model_val_report")
        if isinstance(model_val, Mapping):
            return model_val.get("metrics") or model_val
    return None


def _collect_teacher_anchor(
    config: HLTSDVReportConfig,
    paths: Mapping[str, Path],
    warnings: list[str],
    problems: list[str],
) -> list[dict[str, Any]]:
    target = PD10_TEACHER_HLT
    model_name = pd10_extended_teacher_model_name(target)
    run_path = paths["pd10_teachers_dir"] / model_name / "run_report.json"
    report = _read_json(run_path)
    if not isinstance(report, Mapping):
        message = f"missing PD10 HLT teacher run_report: {run_path}"
        (problems if config.require_anchors else warnings).append(message)
        report = {}
    rows: list[dict[str, Any]] = []
    prediction_dir = paths["pd10_teacher_logits_dir"]
    for split in ("model_val", "final_test"):
        cache_metrics, cache_source = _metrics_from_prediction_block(
            prediction_dir,
            model_name,
            split,
            include_prediction_metrics=bool(config.include_prediction_metrics),
        )
        metrics, source = _merge_metrics(
            _teacher_anchor_report_metrics(report, split),
            cache_metrics,
            cache_source=cache_source,
        )
        if metrics is None:
            warnings.append(f"missing {split} metrics for PD10 HLT ParT anchor")
            continue
        rows.append(
            _row_from_metrics(
                row_type="anchor_teacher",
                name="hlt_part",
                split=split,
                group="pd10_anchor",
                metrics=metrics,
                metrics_source=source,
                report_path=run_path,
                extra={"teacher_target": target, "pd10_model_name": model_name},
            )
        )
    return rows


def _student_anchor_metrics(report: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    if split == "model_val":
        return (
            report.get("model_val_prediction_metrics")
            or report.get("best_model_val_metrics")
            or report.get("best_model_val_training_metrics")
            or {
                "accuracy": report.get("best_model_val_accuracy"),
                "cross_entropy": report.get("best_model_val_cross_entropy") or report.get("best_model_val_loss"),
            }
        )
    return report.get("final_test_metrics")


def _collect_student_anchor(
    config: HLTSDVReportConfig,
    paths: Mapping[str, Path],
    *,
    variant: str,
    alias: str,
    group: str,
    warnings: list[str],
    problems: list[str],
) -> list[dict[str, Any]]:
    run_path = paths["pd10_students_dir"] / variant / "run_report.json"
    report = _read_json(run_path)
    if not isinstance(report, Mapping):
        message = f"missing PD10 student anchor run_report for {alias}: {run_path}"
        (problems if config.require_anchors else warnings).append(message)
        report = {}
    rows: list[dict[str, Any]] = []
    prediction_dir = paths["pd10_students_dir"] / variant / "student_predictions"
    for split in ("model_val", "final_test"):
        cache_metrics, cache_source = _metrics_from_prediction_block(
            prediction_dir,
            variant,
            split,
            include_prediction_metrics=bool(config.include_prediction_metrics),
        )
        metrics, source = _merge_metrics(_student_anchor_metrics(report, split), cache_metrics, cache_source=cache_source)
        if metrics is None:
            warnings.append(f"missing {split} metrics for PD10 student anchor {alias}")
            continue
        rows.append(
            _row_from_metrics(
                row_type="anchor_student",
                name=alias,
                split=split,
                group=group,
                metrics=metrics,
                metrics_source=source,
                report_path=run_path,
                extra={"pd10_variant": variant},
            )
        )
    return rows


def _anchor_variants() -> dict[str, str]:
    warm_ce = pd10_student_variant_name(PD10_STUDENT_INIT_WARM_START, PD10_TEACHER_NONE)
    warm_dual = pd10_student_variant_name(
        PD10_STUDENT_INIT_WARM_START,
        PD10_TEACHER_DUAL_VIEW,
        PD10_TARGET_FULL_LOGITS,
        temperature=PD10_DEFAULT_TEMPERATURE,
        kd_alpha=PD10_DEFAULT_ALPHA,
    )
    return {
        "warm_start_ce_only": warm_ce,
        "v1_warm_dual_view_kd": warm_dual,
    }


def _collect_anchor_rows(config: HLTSDVReportConfig, paths: Mapping[str, Path], warnings: list[str], problems: list[str]) -> list[dict[str, Any]]:
    rows = _collect_teacher_anchor(config, paths, warnings, problems)
    anchors = _anchor_variants()
    rows.extend(
        _collect_student_anchor(
            config,
            paths,
            variant=anchors["warm_start_ce_only"],
            alias="warm_start_ce_only",
            group="pd10_anchor",
            warnings=warnings,
            problems=problems,
        )
    )
    rows.extend(
        _collect_student_anchor(
            config,
            paths,
            variant=anchors["v1_warm_dual_view_kd"],
            alias="v1_warm_dual_view_kd",
            group="pd10_anchor",
            warnings=warnings,
            problems=problems,
        )
    )
    return rows


def _row_metric(row: Mapping[str, Any] | None, key: str) -> float | None:
    if row is None:
        return None
    value = _finite_float(row.get(key))
    if value is not None:
        return value
    metrics = row.get("metrics")
    if isinstance(metrics, Mapping):
        return _finite_float(metrics.get(key))
    return None


def _row_fpr(row: Mapping[str, Any] | None) -> float | None:
    for key in (
        "validation_threshold_fpr_at_signal_eff_0p50_macro",
        "fpr_at_signal_eff_0p50_macro",
        "validation_binary_fpr_at_signal_eff_0p50_macro",
    ):
        value = _row_metric(row, key)
        if value is not None:
            return value
    return None


def _row_by_name(rows: Sequence[Mapping[str, Any]], name: str, split: str = "final_test") -> Mapping[str, Any] | None:
    for row in rows:
        if row.get("name") == name and row.get("split") == split:
            return row
    return None


def _sdv_candidates(rows: Sequence[Mapping[str, Any]], split: str) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("split") == split
        and row.get("group") == "sdv_hlt_hlt2"
        and _row_metric(row, "accuracy") is not None
    ]


def _best_sdv_by_model_val(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = _sdv_candidates(rows, "model_val")
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            _row_metric(row, "cross_entropy") if _row_metric(row, "cross_entropy") is not None else float("inf"),
            -(_row_metric(row, "accuracy") or float("-inf")),
            -(_row_metric(row, "macro_ovr_auc") or float("-inf")),
            _row_fpr(row) if _row_fpr(row) is not None else float("inf"),
        ),
    )


def _best_sdv_by_final_accuracy(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = _sdv_candidates(rows, "final_test")
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            _row_metric(row, "accuracy") or float("-inf"),
            -(_row_metric(row, "cross_entropy") or float("inf")),
        ),
    )


def _same_name_final_row(rows: Sequence[Mapping[str, Any]], model_val_row: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if model_val_row is None:
        return None
    return _row_by_name(rows, str(model_val_row.get("name")), split="final_test")


def _comparison(candidate: Mapping[str, Any] | None, baseline: Mapping[str, Any] | None, *, baseline_name: str) -> dict[str, Any]:
    candidate_acc = _row_metric(candidate, "accuracy")
    baseline_acc = _row_metric(baseline, "accuracy")
    candidate_ce = _row_metric(candidate, "cross_entropy")
    baseline_ce = _row_metric(baseline, "cross_entropy")
    candidate_auc = _row_metric(candidate, "macro_ovr_auc")
    baseline_auc = _row_metric(baseline, "macro_ovr_auc")
    candidate_fpr = _row_fpr(candidate)
    baseline_fpr = _row_fpr(baseline)
    delta_acc = None if candidate_acc is None or baseline_acc is None else candidate_acc - baseline_acc
    delta_ce = None if candidate_ce is None or baseline_ce is None else candidate_ce - baseline_ce
    delta_auc = None if candidate_auc is None or baseline_auc is None else candidate_auc - baseline_auc
    delta_fpr = None if candidate_fpr is None or baseline_fpr is None else candidate_fpr - baseline_fpr
    accuracy_beats = None if delta_acc is None else bool(delta_acc > 0.0)
    multimetric = None
    if None not in (delta_acc, delta_ce, delta_auc, delta_fpr):
        multimetric = bool(delta_acc > 0.0 and delta_ce < 0.0 and delta_auc > 0.0 and delta_fpr < 0.0)
    return {
        "baseline_name": baseline_name,
        "candidate_name": None if candidate is None else candidate.get("name"),
        "baseline_row_name": None if baseline is None else baseline.get("name"),
        "available": candidate_acc is not None and baseline_acc is not None,
        "candidate_accuracy": candidate_acc,
        "baseline_accuracy": baseline_acc,
        "delta_accuracy": delta_acc,
        "candidate_cross_entropy": candidate_ce,
        "baseline_cross_entropy": baseline_ce,
        "delta_cross_entropy": delta_ce,
        "candidate_macro_ovr_auc": candidate_auc,
        "baseline_macro_ovr_auc": baseline_auc,
        "delta_macro_ovr_auc": delta_auc,
        "candidate_fpr_at_signal_eff_0p50_macro": candidate_fpr,
        "baseline_fpr_at_signal_eff_0p50_macro": baseline_fpr,
        "delta_fpr_at_signal_eff_0p50_macro": delta_fpr,
        "beats_by_accuracy": accuracy_beats,
        "beats_multimetric": multimetric,
    }


def _baseline_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any] | None]:
    return {
        "hlt_part": _row_by_name(rows, "hlt_part"),
        "warm_start_ce_only": _row_by_name(rows, "warm_start_ce_only"),
        "v1_warm_dual_view_kd": _row_by_name(rows, "v1_warm_dual_view_kd"),
        "same_view_dual_fusion": _row_by_name(rows, HLT_SDV_VARIANT_SAME_VIEW),
        "tta_hlt_part_hlt_plus_hlt2": _row_by_name(rows, HLT_SDV_VARIANT_TTA),
    }


def _headline_comparison_rows(rows: Sequence[Mapping[str, Any]], selected_final: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    return [
        _comparison(selected_final, baseline, baseline_name=baseline_name)
        for baseline_name, baseline in _baseline_rows(rows).items()
    ]


def _all_variant_comparison_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    baselines = {
        name: row
        for name, row in _baseline_rows(rows).items()
        if row is not None
    }
    candidate_rows = [
        row
        for row in rows
        if row.get("split") == "final_test"
        and row.get("group") in {"sdv_hlt_hlt2", "same_view_dual_fusion", "hlt2_only_control", "tta_control"}
        and _row_metric(row, "accuracy") is not None
    ]
    comparison_rows: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        for baseline_name, baseline in baselines.items():
            if baseline is not None and candidate.get("name") == baseline.get("name"):
                continue
            comparison = _comparison(candidate, baseline, baseline_name=baseline_name)
            comparison["candidate_group"] = candidate.get("group")
            comparison["candidate_hlt2_strength"] = candidate.get("hlt2_strength")
            comparison_rows.append(comparison)
    return comparison_rows


def _flatten_binary_rows(metric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in metric_rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        binary = metrics.get("binary_metrics") or metrics.get("binary_projection_results")
        if not isinstance(binary, Mapping):
            continue
        for task_name, payload in binary.items():
            if not isinstance(payload, Mapping) or str(task_name).startswith("macro_"):
                continue
            out = {
                "name": row.get("name"),
                "row_type": row.get("row_type"),
                "group": row.get("group"),
                "split": row.get("split"),
                "binary_task": task_name,
                "available": payload.get("available"),
                "signal_label": payload.get("signal_label"),
                "background_label": payload.get("background_label"),
                "n_signal": payload.get("n_signal"),
                "n_background": payload.get("n_background"),
                "auc": payload.get("auc"),
                "fpr_at_signal_eff_0p30": payload.get("fpr_at_signal_eff_0p30"),
                "fpr_at_signal_eff_0p50": payload.get("fpr_at_signal_eff_0p50"),
                "validation_fpr_at_signal_eff_0p30": payload.get("validation_fpr_at_signal_eff_0p30"),
                "validation_fpr_at_signal_eff_0p50": payload.get("validation_fpr_at_signal_eff_0p50"),
            }
            rows.append(out)
    return rows


def _answer_payload(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    best_val = _best_sdv_by_model_val(rows)
    selected_final = _same_name_final_row(rows, best_val)
    best_final_acc = _best_sdv_by_final_accuracy(rows)
    comparisons = _headline_comparison_rows(rows, selected_final)
    comparison_by_name = {str(row["baseline_name"]): row for row in comparisons}
    selected_name = None if selected_final is None else selected_final.get("name")
    selected_acc = _row_metric(selected_final, "accuracy")
    selected_ce = _row_metric(selected_final, "cross_entropy")
    selected_auc = _row_metric(selected_final, "macro_ovr_auc")
    selected_fpr = _row_fpr(selected_final)
    best_val_strength = None if best_val is None else best_val.get("hlt2_strength")
    return {
        "best_sdv_selected_by_model_val_ce": selected_name,
        "best_sdv_model_val_variant": None if best_val is None else best_val.get("name"),
        "best_sdv_model_val_cross_entropy": _row_metric(best_val, "cross_entropy"),
        "best_sdv_model_val_accuracy": _row_metric(best_val, "accuracy"),
        "best_sdv_hlt2_strength_by_model_val_ce": best_val_strength,
        "best_sdv_final_test_accuracy": selected_acc,
        "best_sdv_final_test_cross_entropy": selected_ce,
        "best_sdv_final_test_macro_ovr_auc": selected_auc,
        "best_sdv_final_test_fpr_at_signal_eff_0p50_macro": selected_fpr,
        "accuracy_best_sdv_final_test_variant": None if best_final_acc is None else best_final_acc.get("name"),
        "accuracy_best_sdv_final_test_accuracy": _row_metric(best_final_acc, "accuracy"),
        "did_hlt_hlt2_beat_hlt_part": comparison_by_name["hlt_part"].get("beats_by_accuracy"),
        "did_hlt_hlt2_beat_warm_start_ce_only": comparison_by_name["warm_start_ce_only"].get("beats_by_accuracy"),
        "did_hlt_hlt2_beat_hlt_hlt_same_view": comparison_by_name["same_view_dual_fusion"].get("beats_by_accuracy"),
        "did_hlt_hlt2_beat_tta_averaging": comparison_by_name["tta_hlt_part_hlt_plus_hlt2"].get(
            "beats_by_accuracy"
        ),
        "delta_vs_hlt_part_accuracy": comparison_by_name["hlt_part"].get("delta_accuracy"),
        "delta_vs_warm_start_ce_only_accuracy": comparison_by_name["warm_start_ce_only"].get("delta_accuracy"),
        "delta_vs_v1_warm_dual_view_kd_accuracy": comparison_by_name["v1_warm_dual_view_kd"].get("delta_accuracy"),
        "delta_vs_same_view_dual_fusion_accuracy": comparison_by_name["same_view_dual_fusion"].get("delta_accuracy"),
        "delta_vs_tta_averaging_accuracy": comparison_by_name["tta_hlt_part_hlt_plus_hlt2"].get("delta_accuracy"),
        "comparisons": comparison_by_name,
        "did_final_test_winner_use_final_test_information_for_selection": False,
        "selection_rule": "HLT+HLT2 winner selected by model_val cross_entropy, accuracy/AUC/FPR tie-break only",
    }


def render_hlt_sdv_report_markdown(report: Mapping[str, Any]) -> str:
    answers = report.get("answers", {}) if isinstance(report.get("answers"), Mapping) else {}
    lines = [
        "# HLT Self-Dualview Report",
        "",
        f"- OK: {report.get('ok')}",
        f"- Selected HLT+HLT2 SDV: `{answers.get('best_sdv_selected_by_model_val_ce')}`",
        f"- Selected final-test accuracy: {answers.get('best_sdv_final_test_accuracy')}",
        f"- Delta vs HLT ParT: {answers.get('delta_vs_hlt_part_accuracy')}",
        f"- Delta vs warm-start CE-only: {answers.get('delta_vs_warm_start_ce_only_accuracy')}",
        f"- Delta vs V1 warm dual-view KD: {answers.get('delta_vs_v1_warm_dual_view_kd_accuracy')}",
        f"- Delta vs same-view dual fusion: {answers.get('delta_vs_same_view_dual_fusion_accuracy')}",
        f"- Delta vs TTA averaging: {answers.get('delta_vs_tta_averaging_accuracy')}",
        "",
        "## Questions",
        "",
        f"- Did HLT+HLT2 beat HLT ParT? {answers.get('did_hlt_hlt2_beat_hlt_part')}",
        f"- Did HLT+HLT2 beat warm-start CE-only? {answers.get('did_hlt_hlt2_beat_warm_start_ce_only')}",
        f"- Did HLT+HLT2 beat HLT+HLT same-view? {answers.get('did_hlt_hlt2_beat_hlt_hlt_same_view')}",
        f"- Did HLT+HLT2 beat cheap TTA averaging? {answers.get('did_hlt_hlt2_beat_tta_averaging')}",
        f"- Which HLT2 strength was best by model-val CE? {answers.get('best_sdv_hlt2_strength_by_model_val_ce')}",
        "- Did the final-test winner use final-test information for selection? No.",
    ]
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.get("warnings", []))
    if report.get("problems"):
        lines.extend(["", "## Problems", ""])
        lines.extend(f"- {problem}" for problem in report.get("problems", []))
    return "\n".join(lines) + "\n"


def write_hlt_sdv_report(config: HLTSDVReportConfig) -> dict[str, Any]:
    paths = _config_paths(config)
    warnings: list[str] = []
    problems: list[str] = []
    sdv_rows = _collect_sdv_rows(config, paths, warnings, problems)
    anchor_rows = _collect_anchor_rows(config, paths, warnings, problems)
    metric_rows = [*anchor_rows, *sdv_rows]
    headline_comparison_rows = _headline_comparison_rows(
        metric_rows,
        _same_name_final_row(metric_rows, _best_sdv_by_model_val(metric_rows)),
    )
    comparison_rows = _all_variant_comparison_rows(metric_rows)
    binary_rows = _flatten_binary_rows(metric_rows)
    answers = _answer_payload(metric_rows)
    pd10_summary = _read_json(paths["pd10_final_report_json"])
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary_json": str(output_dir / HLT_SDV_REPORT_SUMMARY_JSON),
        "report_json": str(output_dir / HLT_SDV_REPORT_JSON),
        "report_md": str(output_dir / HLT_SDV_REPORT_MD),
        "run_report_json": str(output_dir / HLT_SDV_REPORT_RUN_JSON),
        "metric_table_csv": str(output_dir / HLT_SDV_METRIC_TABLE_CSV),
        "comparison_table_csv": str(output_dir / HLT_SDV_COMPARISON_TABLE_CSV),
        "binary_projection_table_csv": str(output_dir / HLT_SDV_BINARY_TABLE_CSV),
    }
    report = {
        "ok": not problems,
        "contract": HLT_SDV_REPORT_CONTRACT,
        "experiment_name": HLT_SDV_EXPERIMENT_NAME,
        "experiment_step": HLT_SDV_STEP7_EXPERIMENT_STEP,
        "config": asdict(config),
        "paths": {key: str(value) for key, value in paths.items()},
        "answers": answers,
        "metric_rows": metric_rows,
        "headline_comparison_rows": headline_comparison_rows,
        "comparison_rows": comparison_rows,
        "binary_projection_rows": binary_rows,
        "pd10_anchor_report_loaded": isinstance(pd10_summary, Mapping),
        "pd10_anchor_report_answers": None if not isinstance(pd10_summary, Mapping) else pd10_summary.get("answers"),
        "warnings": warnings,
        "problems": problems,
        "outputs": outputs,
    }
    save_json(output_dir / HLT_SDV_REPORT_JSON, _jsonable(report))
    save_json(output_dir / HLT_SDV_REPORT_SUMMARY_JSON, _jsonable(report))
    save_json(output_dir / HLT_SDV_REPORT_RUN_JSON, _jsonable(report))
    (output_dir / HLT_SDV_REPORT_MD).write_text(render_hlt_sdv_report_markdown(report), encoding="utf-8")
    _write_csv(output_dir / HLT_SDV_METRIC_TABLE_CSV, metric_rows)
    _write_csv(output_dir / HLT_SDV_COMPARISON_TABLE_CSV, comparison_rows)
    _write_csv(output_dir / HLT_SDV_BINARY_TABLE_CSV, binary_rows)
    return _jsonable(report)


__all__ = [
    "HLTSDVReportConfig",
    "HLT_SDV_BINARY_TABLE_CSV",
    "HLT_SDV_COMPARISON_TABLE_CSV",
    "HLT_SDV_METRIC_TABLE_CSV",
    "HLT_SDV_REPORT_CONTRACT",
    "HLT_SDV_REPORT_JSON",
    "HLT_SDV_REPORT_MD",
    "HLT_SDV_REPORT_RUN_JSON",
    "HLT_SDV_REPORT_SUMMARY_JSON",
    "HLT_SDV_STEP7_EXPERIMENT_STEP",
    "default_hlt_sdv_report_config",
    "render_hlt_sdv_report_markdown",
    "write_hlt_sdv_report",
]
