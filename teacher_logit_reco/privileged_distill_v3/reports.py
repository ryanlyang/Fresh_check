"""Final report builder for PDV3 AV10-adapter privileged distillation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from jetclass_fresh.hlt_baseline import save_json

from teacher_logit_reco.set_matching.train import source_metadata

from .config import PDV3_LABEL_NAMES
from .students import (
    PDV3_STUDENT_DEFAULT_VARIANTS,
    PDV3_STUDENT_HLT_PART_CE,
    normalize_pdv3_student_variant,
    pdv3_student_variant_spec,
)


PDV3_STEP5_REPORT_STEP = "pdv3_step5_final_report"
PDV3_REPORT_CONTRACT = "pdv3_av10_adapter_privileged_distillation_report_v1"
PDV3_REPORT_JSON = "pdv3_report.json"
PDV3_REPORT_MD = "pdv3_report.md"
PDV3_REPORT_TABLES = {
    "student_metrics": "student_metrics.csv",
    "comparison_table": "comparison_table.csv",
    "per_class_metrics": "per_class_metrics.csv",
    "confusion_matrix": "confusion_matrix.csv",
    "parameter_accounting": "parameter_accounting.csv",
    "adapter_diagnostics": "adapter_diagnostics.csv",
    "kd_diagnostics": "kd_diagnostics.csv",
    "runtime_table": "runtime_table.csv",
}
PDV3_REPORT_SPLITS = ("model_val", "final_test")
PDV3_DATASET_CONSISTENCY_FIELDS = (
    "source_manifest_hash",
    "hlt_profile",
    "hlt_degradation_strength",
    "hlt_content_hash",
    "jet_identity_hash",
    "source_hlt_jet_identity_hash",
)
PDV3_REQUIRED_DATASET_CONSISTENCY_FIELDS = (
    "source_manifest_hash",
    "hlt_profile",
    "hlt_degradation_strength",
    "hlt_content_hash",
    "jet_identity_hash",
)


@dataclass(frozen=True)
class PDV3ReportConfig:
    """Locations and policy for the Step 5 PDV3 report."""

    output_dir: str
    students_dir: str
    student_variants: tuple[str, ...] = field(default_factory=lambda: tuple(PDV3_STUDENT_DEFAULT_VARIANTS))
    baseline_variant: str = PDV3_STUDENT_HLT_PART_CE
    require_all_students: bool = True
    confirm_final_test: bool = False

    def __post_init__(self) -> None:
        if not bool(self.confirm_final_test):
            raise ValueError("PDV3 final report requires confirm_final_test=True")
        variants = tuple(normalize_pdv3_student_variant(variant) for variant in self.student_variants)
        if not variants:
            variants = tuple(PDV3_STUDENT_DEFAULT_VARIANTS)
        if len(set(variants)) != len(variants):
            raise ValueError("student_variants contains duplicates")
        baseline = normalize_pdv3_student_variant(self.baseline_variant)
        if baseline not in variants:
            variants = (baseline, *variants)
        object.__setattr__(self, "student_variants", variants)
        object.__setattr__(self, "baseline_variant", baseline)


def _read_json(path: str | Path | None) -> Any | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_jsonable(value), sort_keys=True)
    return _jsonable(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                keys.append(str(key))
    if not keys:
        rows = [{"available": False}]
        keys = ["available"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _metrics_for_split(report: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    if split == "model_val":
        keys = ("best_model_val_metrics", "model_val_metrics")
    elif split == "final_test":
        keys = ("final_test_metrics",)
    else:
        keys = (f"{split}_metrics",)
    for key in keys:
        value = report.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _metric(metrics: Mapping[str, Any] | None, name: str) -> float | None:
    if not isinstance(metrics, Mapping):
        return None
    value = _float(metrics.get(name))
    if value is not None:
        return value
    binary = metrics.get("binary_metrics")
    if isinstance(binary, Mapping):
        return _float(binary.get(name))
    return None


def _relative_error_reduction(accuracy: float | None, baseline_accuracy: float | None) -> float | None:
    if accuracy is None or baseline_accuracy is None:
        return None
    baseline_error = 1.0 - float(baseline_accuracy)
    if baseline_error <= 0.0:
        return None
    return (float(accuracy) - float(baseline_accuracy)) / baseline_error


def _load_student_reports(config: PDV3ReportConfig, problems: list[str]) -> dict[str, Mapping[str, Any]]:
    reports: dict[str, Mapping[str, Any]] = {}
    root = Path(config.students_dir)
    for variant in config.student_variants:
        report_path = root / variant / "run_report.json"
        payload = _read_json(report_path)
        if not isinstance(payload, Mapping):
            problem = f"missing or invalid PDV3 student report for {variant}: {report_path}"
            problems.append(problem)
            continue
        if normalize_pdv3_student_variant(str(payload.get("student_variant", variant))) != variant:
            problems.append(f"student_variant mismatch in {report_path}")
            continue
        enriched = dict(payload)
        enriched["_run_report_path"] = str(report_path)
        reports[variant] = enriched
    if config.require_all_students:
        missing = [variant for variant in config.student_variants if variant not in reports]
        if missing:
            problems.append(f"missing required PDV3 students: {' '.join(missing)}")
    if config.baseline_variant not in reports:
        problems.append(f"missing required baseline student {config.baseline_variant}")
    return reports


def _nested_value(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _check_same_value(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
    path: Sequence[str],
    problems: list[str],
    required: bool = True,
    require_when_any_present: bool = False,
) -> None:
    observed: dict[str, Any] = {}
    missing: list[str] = []
    for variant, report in reports.items():
        value = _nested_value(report, path)
        if value in (None, ""):
            missing.append(variant)
            if required:
                problems.append(f"{variant} is missing {label}")
            continue
        observed[variant] = value
    if bool(require_when_any_present) and observed and missing:
        problems.append(f"some PDV3 student reports are missing {label}: {' '.join(sorted(missing))}")
    if len({json.dumps(_jsonable(value), sort_keys=True) for value in observed.values()}) > 1:
        details = ", ".join(f"{variant}={value}" for variant, value in sorted(observed.items()))
        problems.append(f"PDV3 student reports disagree on {label}: {details}")


def _effective_baseline_identity(report: Mapping[str, Any], label: str) -> Any:
    """Return the baseline identity used for consistency checks.

    A from-scratch ``pdv3_hlt_part_ce`` fallback has no upstream baseline
    checkpoint. Its own produced checkpoint is the baseline identity that later
    warm-started students should match.
    """

    if label == "baseline checkpoint hash":
        value = report.get("baseline_checkpoint_hash")
        if value not in (None, ""):
            return value
        if bool(report.get("baseline_from_scratch")):
            return report.get("checkpoint_hash")
        return None
    if label == "baseline split manifest hash":
        value = report.get("baseline_checkpoint_split_manifest_hash")
        if value not in (None, ""):
            return value
        if bool(report.get("baseline_from_scratch")):
            return _nested_value(report, ("manifest", "manifest_hash"))
        return None
    raise AssertionError(f"Unhandled baseline identity label {label!r}")


def _check_effective_baseline_identity(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
    problems: list[str],
) -> None:
    observed: dict[str, Any] = {}
    missing: list[str] = []
    for variant, report in reports.items():
        value = _effective_baseline_identity(report, label)
        if value in (None, ""):
            missing.append(variant)
            continue
        observed[variant] = value
    if observed and missing:
        problems.append(f"some PDV3 student reports are missing {label}: {' '.join(sorted(missing))}")
    if len({json.dumps(_jsonable(value), sort_keys=True) for value in observed.values()}) > 1:
        details = ", ".join(f"{variant}={value}" for variant, value in sorted(observed.items()))
        problems.append(f"PDV3 student reports disagree on {label}: {details}")


def _check_student_report_consistency(reports: Mapping[str, Mapping[str, Any]], problems: list[str]) -> None:
    """Reject mixed reports from different splits/caches/baseline checkpoints."""

    if not reports:
        return
    _check_same_value(reports, label="split manifest hash", path=("manifest", "manifest_hash"), problems=problems)
    _check_effective_baseline_identity(reports, label="baseline checkpoint hash", problems=problems)
    _check_effective_baseline_identity(reports, label="baseline split manifest hash", problems=problems)
    for split_key in ("train_dataset", "val_dataset", "final_test_dataset"):
        for field in PDV3_DATASET_CONSISTENCY_FIELDS:
            _check_same_value(
                reports,
                label=f"{split_key}.{field}",
                path=(split_key, field),
                problems=problems,
                required=field in PDV3_REQUIRED_DATASET_CONSISTENCY_FIELDS,
            )
    for variant, report in reports.items():
        manifest_hash = _nested_value(report, ("manifest", "manifest_hash"))
        if manifest_hash in (None, ""):
            continue
        for split_key in ("train_dataset", "val_dataset", "final_test_dataset"):
            dataset_manifest_hash = _nested_value(report, (split_key, "source_manifest_hash"))
            if dataset_manifest_hash in (None, ""):
                continue
            if str(dataset_manifest_hash) != str(manifest_hash):
                problems.append(
                    f"{variant} {split_key}.source_manifest_hash does not match manifest.manifest_hash: "
                    f"{dataset_manifest_hash} != {manifest_hash}"
                )
    for variant, report in reports.items():
        if bool(report.get("final_test_uses_teacher_logits")) or bool(report.get("final_test_uses_teacher_representations")):
            problems.append(
                f"{variant} final_test evaluation used privileged teacher tensors; "
                "PDV3 final-test evaluation must be teacher-free"
            )
        final_meta = report.get("final_test_dataset")
        if isinstance(final_meta, Mapping):
            if bool(final_meta.get("teacher_logits_train_time_only")) or bool(
                final_meta.get("teacher_representations_train_time_only")
            ):
                problems.append(
                    f"{variant} final_test_dataset loaded privileged teacher caches; "
                    "PDV3 final-test evaluation must be teacher-free"
                )


def _student_label(variant: str) -> str:
    spec = pdv3_student_variant_spec(variant)
    return spec.description or variant


def _student_metric_rows(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    baseline_variant: str,
) -> list[dict[str, Any]]:
    baseline = reports.get(baseline_variant)
    baseline_metrics = {
        split: _metrics_for_split(baseline, split) if isinstance(baseline, Mapping) else None
        for split in PDV3_REPORT_SPLITS
    }
    rows: list[dict[str, Any]] = []
    for variant, report in reports.items():
        spec = pdv3_student_variant_spec(variant)
        hlt_contract = report.get("hlt_input_contract")
        if not isinstance(hlt_contract, Mapping):
            hlt_contract = {}
        for split in PDV3_REPORT_SPLITS:
            metrics = _metrics_for_split(report, split)
            baseline_split = baseline_metrics.get(split)
            accuracy = _metric(metrics, "accuracy")
            baseline_accuracy = _metric(baseline_split, "accuracy")
            loss = _metric(metrics, "loss")
            ce_loss = _metric(metrics, "ce_loss")
            row = {
                "variant": variant,
                "display_name": _student_label(variant),
                "split": split,
                "student_family": spec.student_family,
                "teacher_family": spec.teacher_family,
                "loss_mode": spec.loss_mode,
                "training_schedule": spec.training_schedule,
                "freeze_policy": spec.freeze_policy,
                "combined_adapter": bool(spec.combined_adapter),
                "architecture_view_variant": spec.architecture_view_variant,
                "hlt_input_profile": hlt_contract.get("profile"),
                "hlt_input_degradation_strength": hlt_contract.get("degradation_strength"),
                "hlt_input_contract_label": hlt_contract.get("label"),
                "accuracy": accuracy,
                "baseline_accuracy": baseline_accuracy,
                "accuracy_gain_vs_baseline": None
                if accuracy is None or baseline_accuracy is None
                else accuracy - baseline_accuracy,
                "relative_error_reduction_vs_baseline": _relative_error_reduction(accuracy, baseline_accuracy),
                "loss": loss,
                "ce_loss": ce_loss,
                "macro_per_class_accuracy": _metric(metrics, "macro_per_class_accuracy"),
                "n_jets": _metric(metrics, "n_jets"),
                "kd_loss": _metric(metrics, "kd_loss"),
                "rep_loss": _metric(metrics, "rep_loss"),
                "teacher_student_logit_kl": _metric(metrics, "teacher_student_logit_kl"),
                "teacher_student_representation_cosine": _metric(
                    metrics,
                    "teacher_student_representation_cosine",
                ),
                "teacher_student_top1_agreement": _metric(metrics, "teacher_student_top1_agreement"),
                "teacher_entropy_mean": _metric(metrics, "teacher_entropy_mean"),
                "student_entropy_mean_with_teacher": _metric(metrics, "student_entropy_mean_with_teacher"),
                "teacher_confidence_mean": _metric(metrics, "teacher_confidence_mean"),
                "student_confidence_mean_with_teacher": _metric(metrics, "student_confidence_mean_with_teacher"),
                "student_accuracy_when_teacher_student_agree": _metric(
                    metrics,
                    "student_accuracy_when_teacher_student_agree",
                ),
                "student_accuracy_when_teacher_student_disagree": _metric(
                    metrics,
                    "student_accuracy_when_teacher_student_disagree",
                ),
                "effective_kd_alpha": _metric(metrics, "effective_kd_alpha"),
                "effective_representation_beta": _metric(metrics, "effective_representation_beta"),
                "representation_delta_l2_loss": _metric(metrics, "representation_delta_l2_loss"),
                "representation_delta_l2_mean": _metric(metrics, "representation_delta_l2_mean"),
                "representation_delta_l2_weight": _metric(metrics, "representation_delta_l2_weight"),
                "nonfinite_batches_skipped": _metric(metrics, "nonfinite_batches_skipped"),
                "best_epoch": report.get("best_epoch"),
                "run_report": report.get("_run_report_path"),
            }
            rows.append(row)
    return rows


def _comparison_rows(metric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    final_rows = [row for row in metric_rows if row.get("split") == "final_test"]
    sorted_rows = sorted(
        final_rows,
        key=lambda row: _float(row.get("accuracy")) if _float(row.get("accuracy")) is not None else float("-inf"),
        reverse=True,
    )
    output: list[dict[str, Any]] = []
    for rank, row in enumerate(sorted_rows, start=1):
        output.append(
            {
                "rank_by_final_test_accuracy": rank,
                "variant": row.get("variant"),
                "display_name": row.get("display_name"),
                "student_family": row.get("student_family"),
                "teacher_family": row.get("teacher_family"),
                "loss_mode": row.get("loss_mode"),
                "training_schedule": row.get("training_schedule"),
                "freeze_policy": row.get("freeze_policy"),
                "combined_adapter": row.get("combined_adapter"),
                "final_test_accuracy": row.get("accuracy"),
                "final_test_accuracy_gain_vs_baseline": row.get("accuracy_gain_vs_baseline"),
                "final_test_relative_error_reduction_vs_baseline": row.get(
                    "relative_error_reduction_vs_baseline"
                ),
                "final_test_loss": row.get("loss"),
                "best_epoch": row.get("best_epoch"),
            }
        )
    return output


def _per_class_rows(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    baseline_variant: str,
) -> list[dict[str, Any]]:
    baseline_by_split_class: dict[tuple[str, int], float | None] = {}
    baseline = reports.get(baseline_variant)
    if isinstance(baseline, Mapping):
        for split in PDV3_REPORT_SPLITS:
            metrics = _metrics_for_split(baseline, split)
            per_class = metrics.get("per_class_accuracy") if isinstance(metrics, Mapping) else None
            if isinstance(per_class, Sequence) and not isinstance(per_class, (str, bytes)):
                for item in per_class:
                    if isinstance(item, Mapping):
                        baseline_by_split_class[(split, int(item.get("class_index", -1)))] = _float(
                            item.get("accuracy")
                        )
    rows: list[dict[str, Any]] = []
    for variant, report in reports.items():
        for split in PDV3_REPORT_SPLITS:
            metrics = _metrics_for_split(report, split)
            per_class = metrics.get("per_class_accuracy") if isinstance(metrics, Mapping) else None
            if not isinstance(per_class, Sequence) or isinstance(per_class, (str, bytes)):
                continue
            for item in per_class:
                if not isinstance(item, Mapping):
                    continue
                class_index = int(item.get("class_index", -1))
                accuracy = _float(item.get("accuracy"))
                baseline_accuracy = baseline_by_split_class.get((split, class_index))
                rows.append(
                    {
                        "variant": variant,
                        "split": split,
                        "class_index": class_index,
                        "class_name": item.get("class_name")
                        or (PDV3_LABEL_NAMES[class_index] if 0 <= class_index < len(PDV3_LABEL_NAMES) else None),
                        "support": item.get("support"),
                        "correct": item.get("correct"),
                        "accuracy": accuracy,
                        "baseline_accuracy": baseline_accuracy,
                        "accuracy_gain_vs_baseline": None
                        if accuracy is None or baseline_accuracy is None
                        else accuracy - baseline_accuracy,
                    }
                )
    return rows


def _confusion_matrix_rows(reports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, report in reports.items():
        for split in PDV3_REPORT_SPLITS:
            metrics = _metrics_for_split(report, split)
            matrix = metrics.get("confusion_matrix") if isinstance(metrics, Mapping) else None
            if not isinstance(matrix, Sequence) or isinstance(matrix, (str, bytes)):
                continue
            for true_index, values in enumerate(matrix):
                if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                    continue
                for pred_index, count in enumerate(values):
                    true_name = PDV3_LABEL_NAMES[true_index] if 0 <= true_index < len(PDV3_LABEL_NAMES) else None
                    pred_name = PDV3_LABEL_NAMES[pred_index] if 0 <= pred_index < len(PDV3_LABEL_NAMES) else None
                    rows.append(
                        {
                            "variant": variant,
                            "split": split,
                            "true_class_index": true_index,
                            "true_class_name": true_name,
                            "pred_class_index": pred_index,
                            "pred_class_name": pred_name,
                            "count": count,
                        }
                    )
    return rows


def _parameter_rows(reports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, report in reports.items():
        spec = pdv3_student_variant_spec(variant)
        accounting = report.get("parameter_accounting")
        if not isinstance(accounting, Mapping):
            accounting = {}
        projector = report.get("representation_projector_config")
        rows.append(
            {
                "variant": variant,
                "student_family": spec.student_family,
                "architecture_view_variant": spec.architecture_view_variant,
                "training_schedule": spec.training_schedule,
                "freeze_policy": spec.freeze_policy,
                "combined_adapter": bool(spec.combined_adapter),
                "total_params": accounting.get("total_params"),
                "trainable_params": accounting.get("trainable_params"),
                "part_params": accounting.get("part_params"),
                "trainable_part_params": accounting.get("trainable_part_params"),
                "adapter_params": accounting.get("adapter_params"),
                "trainable_adapter_params": accounting.get("trainable_adapter_params"),
                "dormant_adapter_params": accounting.get("dormant_adapter_params"),
                "active_adapter_module_names": " ".join(
                    str(name) for name in accounting.get("active_adapter_module_names", [])
                )
                if isinstance(accounting.get("active_adapter_module_names"), list)
                else accounting.get("active_adapter_module_names"),
                "representation_projector_source": projector.get("source") if isinstance(projector, Mapping) else None,
                "representation_projector_contract": projector.get("contract") if isinstance(projector, Mapping) else None,
                "representation_projector_residual_form": projector.get("residual_form")
                if isinstance(projector, Mapping)
                else None,
                "representation_projector_zero_init_delta_projection": projector.get("zero_init_delta_projection")
                if isinstance(projector, Mapping)
                else None,
                "representation_projector_input_dim": projector.get("input_dim")
                if isinstance(projector, Mapping)
                else None,
                "representation_projector_output_dim": projector.get("output_dim")
                if isinstance(projector, Mapping)
                else None,
            }
        )
    return rows


def _diagnostic_rows(reports: Mapping[str, Mapping[str, Any]], *, kind: str) -> list[dict[str, Any]]:
    adapter_needles = ("delta", "gate", "adapter", "feature", "grad_norm", "embedding")
    kd_needles = ("kd", "teacher", "representation", "logit_kl", "rep_loss", "entropy", "confidence", "agreement")
    needles = adapter_needles if kind == "adapter" else kd_needles
    rows: list[dict[str, Any]] = []

    def add_metric_rows(variant: str, split: str, metrics: Mapping[str, Any], *, diagnostic_only: bool = False) -> None:
        for key in (
            "kd_loss",
            "rep_loss",
            "teacher_student_logit_kl",
            "teacher_student_representation_cosine",
            "effective_kd_alpha",
            "effective_representation_beta",
            "teacher_student_top1_agreement",
            "teacher_entropy_mean",
            "student_entropy_mean_with_teacher",
            "teacher_confidence_mean",
            "student_confidence_mean_with_teacher",
            "student_accuracy_when_teacher_student_agree",
            "student_accuracy_when_teacher_student_disagree",
            "delta_l2_loss",
            "delta_l2_mean",
            "representation_delta_l2_loss",
            "representation_delta_l2_mean",
            "representation_delta_l2_weight",
        ):
            if any(needle in key for needle in needles):
                rows.append(
                    {
                        "variant": variant,
                        "split": split,
                        "diagnostic": key,
                        "value": _metric(metrics, key),
                        "diagnostic_only": bool(diagnostic_only),
                    }
                )
        diagnostics = metrics.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            return
        for key, value in diagnostics.items():
            key_text = str(key)
            if any(needle in key_text for needle in needles):
                rows.append(
                    {
                        "variant": variant,
                        "split": split,
                        "diagnostic": key_text,
                        "value": _float(value),
                        "raw_value": value,
                        "diagnostic_only": bool(diagnostic_only),
                    }
                )

    for variant, report in reports.items():
        for split in PDV3_REPORT_SPLITS:
            metrics = _metrics_for_split(report, split)
            if not isinstance(metrics, Mapping):
                continue
            add_metric_rows(variant, split, metrics)
        final_teacher_metrics = report.get("final_test_teacher_diagnostic_metrics")
        if isinstance(final_teacher_metrics, Mapping):
            add_metric_rows(variant, "final_test_teacher_diagnostics", final_teacher_metrics, diagnostic_only=True)
    return rows


def _runtime_rows(reports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, report in reports.items():
        runtime = report.get("runtime") if isinstance(report.get("runtime"), Mapping) else {}
        rows.append(
            {
                "variant": variant,
                "epochs_completed": report.get("epochs_completed"),
                "best_epoch": report.get("best_epoch"),
                "elapsed_seconds": runtime.get("elapsed_seconds"),
                "elapsed_minutes": runtime.get("elapsed_minutes"),
                "final_test_loaded_during_training": report.get("final_test_loaded_during_training"),
                "inference_consumes_hlt_only": report.get("inference_consumes_hlt_only"),
            }
        )
    return rows


def _markdown_summary(report: Mapping[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    lines = [
        "# PDV3 AV10 Adapter Privileged Distillation Report",
        "",
        f"- ok: {report.get('ok')}",
        f"- baseline: {summary.get('baseline_variant')}",
        f"- best final-test student: {summary.get('best_final_test_variant')}",
        f"- best final-test accuracy: {summary.get('best_final_test_accuracy')}",
        f"- best gain over baseline: {summary.get('best_final_test_accuracy_gain_vs_baseline')}",
        f"- best relative error reduction: {summary.get('best_final_test_relative_error_reduction_vs_baseline')}",
        "",
        "## Problems",
        "",
    ]
    problems = report.get("problems") or []
    if problems:
        lines.extend([f"- {problem}" for problem in problems])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )
    outputs = report.get("outputs") if isinstance(report.get("outputs"), Mapping) else {}
    for key, value in outputs.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def build_pdv3_report(config: PDV3ReportConfig) -> dict[str, Any]:
    """Build and write the Step 5 PDV3 report."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    reports = _load_student_reports(config, problems)
    _check_student_report_consistency(reports, problems)
    metric_rows = _student_metric_rows(reports, baseline_variant=config.baseline_variant)
    comparison_rows = _comparison_rows(metric_rows)
    per_class_rows = _per_class_rows(reports, baseline_variant=config.baseline_variant)
    confusion_rows = _confusion_matrix_rows(reports)
    parameter_rows = _parameter_rows(reports)
    adapter_diagnostic_rows = _diagnostic_rows(reports, kind="adapter")
    kd_diagnostic_rows = _diagnostic_rows(reports, kind="kd")
    runtime_rows = _runtime_rows(reports)
    best_row = comparison_rows[0] if comparison_rows else {}
    baseline_final = next(
        (
            row
            for row in metric_rows
            if row.get("variant") == config.baseline_variant and row.get("split") == "final_test"
        ),
        {},
    )
    summary = {
        "baseline_variant": config.baseline_variant,
        "baseline_final_test_accuracy": baseline_final.get("accuracy"),
        "best_final_test_variant": best_row.get("variant"),
        "best_final_test_accuracy": best_row.get("final_test_accuracy"),
        "best_final_test_accuracy_gain_vs_baseline": best_row.get("final_test_accuracy_gain_vs_baseline"),
        "best_final_test_relative_error_reduction_vs_baseline": best_row.get(
            "final_test_relative_error_reduction_vs_baseline"
        ),
        "did_any_student_beat_baseline": any(
            (_float(row.get("final_test_accuracy_gain_vs_baseline")) or 0.0) > 0.0
            for row in comparison_rows
            if row.get("variant") != config.baseline_variant
        ),
        "n_student_reports": len(reports),
        "n_requested_students": len(config.student_variants),
        "macro_auc_available": False,
        "macro_auc_note": "The shared 10-class metrics helper does not compute macro AUC; binary projections remain nested in per-run metrics.",
    }
    outputs = {
        "report_json": str(output_dir / PDV3_REPORT_JSON),
        "report_md": str(output_dir / PDV3_REPORT_MD),
        **{key: str(output_dir / filename) for key, filename in PDV3_REPORT_TABLES.items()},
    }
    report = {
        "experiment_step": PDV3_STEP5_REPORT_STEP,
        "report_contract": PDV3_REPORT_CONTRACT,
        "ok": not problems,
        "summary": summary,
        "problems": problems,
        "config": asdict(config),
        "student_variants": list(config.student_variants),
        "baseline_variant": config.baseline_variant,
        "student_reports": {variant: report.get("_run_report_path") for variant, report in reports.items()},
        "tables": {
            "student_metrics": metric_rows,
            "comparison_table": comparison_rows,
            "per_class_metrics": per_class_rows,
            "confusion_matrix": confusion_rows,
            "parameter_accounting": parameter_rows,
            "adapter_diagnostics": adapter_diagnostic_rows,
            "kd_diagnostics": kd_diagnostic_rows,
            "runtime_table": runtime_rows,
        },
        "outputs": outputs,
        "source": source_metadata(),
    }
    _write_csv(output_dir / PDV3_REPORT_TABLES["student_metrics"], metric_rows)
    _write_csv(output_dir / PDV3_REPORT_TABLES["comparison_table"], comparison_rows)
    _write_csv(output_dir / PDV3_REPORT_TABLES["per_class_metrics"], per_class_rows)
    _write_csv(output_dir / PDV3_REPORT_TABLES["confusion_matrix"], confusion_rows)
    _write_csv(output_dir / PDV3_REPORT_TABLES["parameter_accounting"], parameter_rows)
    _write_csv(output_dir / PDV3_REPORT_TABLES["adapter_diagnostics"], adapter_diagnostic_rows)
    _write_csv(output_dir / PDV3_REPORT_TABLES["kd_diagnostics"], kd_diagnostic_rows)
    _write_csv(output_dir / PDV3_REPORT_TABLES["runtime_table"], runtime_rows)
    save_json(output_dir / PDV3_REPORT_JSON, _jsonable(report))
    (output_dir / PDV3_REPORT_MD).write_text(_markdown_summary(report), encoding="utf-8")
    return report


__all__ = [
    "PDV3_REPORT_CONTRACT",
    "PDV3_REPORT_JSON",
    "PDV3_REPORT_MD",
    "PDV3_REPORT_TABLES",
    "PDV3_STEP5_REPORT_STEP",
    "PDV3ReportConfig",
    "build_pdv3_report",
]
