"""Report builder for target-conditioned denoising ParT experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .tagger import (
    TARGET_DENOISING_TAGGER_VARIANTS,
    TARGET_DENOISING_VARIANT_HLT_PART_BASELINE,
)


TARGET_DENOISING_STEP5 = "target_conditioned_denoising_part_step5_reports"
TARGET_DENOISING_REPORT_CONTRACT = "target_conditioned_denoising_part_report_v1"
TARGET_DENOISING_REPORT_SPLITS = ("model_val", "stack_val", "final_test")


@dataclass(frozen=True)
class TargetDenoisingReportConfig:
    """Inputs for the Step 5 report builder."""

    output_dir: str
    tagger_root: str | None = None
    denoiser_report: str | None = None
    hlt_baseline_report: str | None = None
    offline_baseline_report: str | None = None
    tagger_report_paths: tuple[str, ...] = ()
    variants: tuple[str, ...] = field(default_factory=lambda: TARGET_DENOISING_TAGGER_VARIANTS)
    require_variants: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "tagger_report_paths", tuple(str(path) for path in self.tagger_report_paths))
        object.__setattr__(self, "variants", tuple(str(variant) for variant in self.variants))
        object.__setattr__(self, "require_variants", bool(self.require_variants))


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


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
        keys = ["empty"]
        rows = [{"empty": ""}]
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


def _first_path_value(report: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    for path in paths:
        value: Any = report
        ok = True
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                ok = False
                break
            value = value[key]
        if ok:
            return value
    return None


def _metrics_for_split(report: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    if split == "model_val":
        keys = (
            "best_model_val_metrics",
            "model_val_metrics",
            "model_val",
            "model_val_report",
        )
    elif split == "stack_val":
        keys = ("stack_val_metrics", "best_stack_val_metrics", "stack_val")
    elif split == "final_test":
        keys = ("final_test_metrics", "final_test", "final_test_report")
    else:
        keys = (f"{split}_metrics", split)
    for key in keys:
        value = report.get(key)
        if isinstance(value, Mapping):
            metrics = value.get("metrics") if isinstance(value.get("metrics"), Mapping) else value
            if isinstance(metrics, Mapping):
                return metrics
    return None


def _metric(metrics: Mapping[str, Any] | None, report: Mapping[str, Any], split: str, name: str) -> float | None:
    if isinstance(metrics, Mapping):
        direct = _float(metrics.get(name))
        if direct is not None:
            return direct
        nested = metrics.get("binary_metrics")
        if isinstance(nested, Mapping):
            value = _float(nested.get(name))
            if value is not None:
                return value
    if split == "model_val":
        return _float(report.get(f"best_model_val_{name}")) or _float(report.get(f"model_val_{name}"))
    if split == "final_test":
        return _float(report.get(f"final_test_{name}"))
    return _float(report.get(f"{split}_{name}"))


def _split_group(split: str) -> str:
    if split == "model_val":
        return "model_val_selection"
    if split == "final_test":
        return "final_test_one_shot"
    return "validation_diagnostic"


def _variant_from_report(path: Path, report: Mapping[str, Any], fallback: str | None = None) -> str:
    value = _first_path_value(
        report,
        (
            ("variant",),
            ("config", "variant"),
            ("model_config", "config", "variant"),
            ("model_config", "variant"),
        ),
    )
    if value:
        return str(value)
    if fallback:
        return str(fallback)
    return path.parent.name


def _load_tagger_reports(config: TargetDenoisingReportConfig, problems: list[str]) -> dict[str, Mapping[str, Any]]:
    reports: dict[str, Mapping[str, Any]] = {}
    root = None if config.tagger_root is None else Path(config.tagger_root)
    if root is not None:
        for variant in config.variants:
            path = root / str(variant) / "run_report.json"
            payload = _read_json(path)
            if isinstance(payload, Mapping):
                reports[str(variant)] = {"path": str(path), **dict(payload)}
            elif bool(config.require_variants):
                problems.append(f"missing tagger report for variant {variant}: {path}")
    for path_text in config.tagger_report_paths:
        path = Path(path_text)
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            problems.append(f"missing explicit tagger report: {path}")
            continue
        variant = _variant_from_report(path, payload)
        reports[variant] = {"path": str(path), **dict(payload)}
    return reports


def _source_rows(
    *,
    source_kind: str,
    variant: str,
    report: Mapping[str, Any],
    path: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in TARGET_DENOISING_REPORT_SPLITS:
        metrics = _metrics_for_split(report, split)
        if metrics is None:
            continue
        row = {
            "source_kind": source_kind,
            "variant": variant,
            "split": split,
            "split_group": _split_group(split),
            "path": path,
            "accuracy": _metric(metrics, report, split, "accuracy"),
            "loss": _metric(metrics, report, split, "loss"),
            "cross_entropy": _metric(metrics, report, split, "cross_entropy"),
            "auc": _metric(metrics, report, split, "auc"),
            "normalized_rmse": _metric(metrics, report, split, "normalized_rmse"),
            "nll_loss": _metric(metrics, report, split, "nll_loss"),
            "smooth_l1_loss": _metric(metrics, report, split, "smooth_l1_loss"),
            "n_jets": _metric(metrics, report, split, "n_jets"),
            "selection_metric": report.get("selection_metric"),
            "best_epoch": report.get("best_epoch"),
        }
        rows.append(row)
    return rows


def _per_class_rows(*, source_kind: str, variant: str, split: str, metrics: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(metrics, Mapping):
        return []
    value = (
        metrics.get("per_class_accuracy")
        or metrics.get("class_accuracy")
        or metrics.get("per_class_acc")
        or metrics.get("per_class")
    )
    if isinstance(value, Mapping):
        return [
            {
                "source_kind": source_kind,
                "variant": variant,
                "split": split,
                "class": str(label),
                "accuracy": _float(acc),
            }
            for label, acc in value.items()
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            {
                "source_kind": source_kind,
                "variant": variant,
                "split": split,
                "class": str(index),
                "accuracy": _float(acc),
            }
            for index, acc in enumerate(value)
        ]
    return []


def _confusion_entry(*, source_kind: str, variant: str, split: str, metrics: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metrics, Mapping):
        return None
    matrix = metrics.get("confusion_matrix") or metrics.get("confusion")
    if matrix is None:
        return None
    return {
        "source_kind": source_kind,
        "variant": variant,
        "split": split,
        "confusion_matrix": _jsonable(matrix),
    }


def _diagnostic_row(*, source_kind: str, variant: str, report: Mapping[str, Any], path: str | None) -> dict[str, Any]:
    model_config = report.get("model_config") if isinstance(report.get("model_config"), Mapping) else {}
    behavior = report.get("variant_behavior") or model_config.get("variant_behavior") if isinstance(model_config, Mapping) else None
    diagnostics = report.get("diagnostics") if isinstance(report.get("diagnostics"), Mapping) else {}
    if not diagnostics:
        diagnostics = report.get("model_val_diagnostics") if isinstance(report.get("model_val_diagnostics"), Mapping) else {}
    row = {
        "source_kind": source_kind,
        "variant": variant,
        "path": path,
        "variant_behavior": behavior,
        "parameter_accounting": report.get("parameter_accounting") or model_config.get("parameter_accounting")
        if isinstance(model_config, Mapping)
        else None,
        "diagnostics": diagnostics,
    }
    adapter = report.get("adapter_diagnostics")
    if isinstance(adapter, Mapping):
        row.update({f"adapter.{key}": value for key, value in adapter.items()})
    injection = report.get("injection_summary")
    if isinstance(injection, Mapping):
        row.update({f"injection.{key}": value for key, value in injection.items()})
    return row


def _diagnostic_only_rows(report: Mapping[str, Any], *, source_kind: str, variant: str, path: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in (
        "diagnostic_only_teacher_offline",
        "teacher_offline_diagnostics",
        "final_test_teacher_diagnostics",
        "offline_teacher_diagnostics",
    ):
        value = report.get(key)
        if isinstance(value, Mapping):
            rows.append(
                {
                    "source_kind": source_kind,
                    "variant": variant,
                    "split": value.get("split"),
                    "split_group": "diagnostic_only_teacher_offline",
                    "diagnostic_name": key,
                    "path": path,
                    "metrics": value,
                }
            )
    return rows


def _mechanism_rows(metric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    baseline_by_split: dict[str, Mapping[str, Any]] = {}
    for row in metric_rows:
        if row.get("variant") == TARGET_DENOISING_VARIANT_HLT_PART_BASELINE and row.get("source_kind") in {
            "tagger",
            "hlt_baseline",
        }:
            baseline_by_split[str(row.get("split"))] = row
    rows: list[dict[str, Any]] = []
    for row in metric_rows:
        split = str(row.get("split"))
        baseline = baseline_by_split.get(split)
        if baseline is None:
            continue
        accuracy = _float(row.get("accuracy"))
        base_accuracy = _float(baseline.get("accuracy"))
        loss = _float(row.get("loss"))
        base_loss = _float(baseline.get("loss"))
        rows.append(
            {
                "variant": row.get("variant"),
                "source_kind": row.get("source_kind"),
                "split": split,
                "split_group": row.get("split_group"),
                "baseline_variant": TARGET_DENOISING_VARIANT_HLT_PART_BASELINE,
                "accuracy": accuracy,
                "baseline_accuracy": base_accuracy,
                "accuracy_delta_vs_hlt": None if accuracy is None or base_accuracy is None else accuracy - base_accuracy,
                "loss": loss,
                "baseline_loss": base_loss,
                "loss_delta_vs_hlt": None if loss is None or base_loss is None else loss - base_loss,
            }
        )
    return rows


def write_target_denoising_report(config: TargetDenoisingReportConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Write the Step 5 comparison report and return its JSON payload."""

    if not isinstance(config, TargetDenoisingReportConfig):
        config = TargetDenoisingReportConfig(**dict(config))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []

    metric_rows: list[dict[str, Any]] = []
    denoising_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_entries: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    diagnostic_only_rows: list[dict[str, Any]] = []

    tagger_reports = _load_tagger_reports(config, problems)
    for variant, report in sorted(tagger_reports.items()):
        path = str(report.get("path")) if report.get("path") else None
        metric_rows.extend(_source_rows(source_kind="tagger", variant=variant, report=report, path=path))
        diagnostic_rows.append(_diagnostic_row(source_kind="tagger", variant=variant, report=report, path=path))
        diagnostic_only_rows.extend(_diagnostic_only_rows(report, source_kind="tagger", variant=variant, path=path))
        for split in TARGET_DENOISING_REPORT_SPLITS:
            metrics = _metrics_for_split(report, split)
            per_class_rows.extend(_per_class_rows(source_kind="tagger", variant=variant, split=split, metrics=metrics))
            entry = _confusion_entry(source_kind="tagger", variant=variant, split=split, metrics=metrics)
            if entry is not None:
                confusion_entries.append(entry)

    for source_kind, variant, path_text in (
        ("denoiser", "target_conditioned_denoiser", config.denoiser_report),
        ("hlt_baseline", TARGET_DENOISING_VARIANT_HLT_PART_BASELINE, config.hlt_baseline_report),
        ("offline_baseline", "offline_part_reference", config.offline_baseline_report),
    ):
        if not path_text:
            continue
        path = Path(path_text)
        report = _read_json(path)
        if not isinstance(report, Mapping):
            problems.append(f"missing {source_kind} report: {path}")
            continue
        source_rows = _source_rows(source_kind=source_kind, variant=variant, report=report, path=str(path))
        if source_kind == "denoiser":
            denoising_rows.extend(source_rows)
        else:
            metric_rows.extend(source_rows)
        diagnostic_rows.append(_diagnostic_row(source_kind=source_kind, variant=variant, report=report, path=str(path)))
        diagnostic_only_rows.extend(_diagnostic_only_rows(report, source_kind=source_kind, variant=variant, path=str(path)))
        for split in TARGET_DENOISING_REPORT_SPLITS:
            metrics = _metrics_for_split(report, split)
            per_class_rows.extend(_per_class_rows(source_kind=source_kind, variant=variant, split=split, metrics=metrics))
            entry = _confusion_entry(source_kind=source_kind, variant=variant, split=split, metrics=metrics)
            if entry is not None:
                confusion_entries.append(entry)

    mechanism_rows = _mechanism_rows(metric_rows)
    _write_csv(output_dir / "tagger_metrics.csv", metric_rows)
    _write_csv(output_dir / "denoising_metrics.csv", denoising_rows)
    _write_csv(output_dir / "per_class_accuracy.csv", per_class_rows)
    _write_csv(output_dir / "adapter_attention_diagnostics.csv", diagnostic_rows)
    _write_csv(output_dir / "diagnostic_only_teacher_offline.csv", diagnostic_only_rows)
    _write_csv(output_dir / "mechanism_ablation_metrics.csv", mechanism_rows)
    _write_json(output_dir / "confusion_matrices.json", {"entries": confusion_entries})

    summary = {
        "ok": not problems,
        "contract": TARGET_DENOISING_REPORT_CONTRACT,
        "step": TARGET_DENOISING_STEP5,
        "config": asdict(config),
        "problems": problems,
        "tagger_report_count": int(len(tagger_reports)),
        "metric_row_count": int(len(metric_rows)),
        "denoising_metric_row_count": int(len(denoising_rows)),
        "per_class_row_count": int(len(per_class_rows)),
        "confusion_matrix_count": int(len(confusion_entries)),
        "diagnostic_row_count": int(len(diagnostic_rows)),
        "diagnostic_only_row_count": int(len(diagnostic_only_rows)),
        "mechanism_row_count": int(len(mechanism_rows)),
        "outputs": {
            "tagger_metrics_csv": str(output_dir / "tagger_metrics.csv"),
            "denoising_metrics_csv": str(output_dir / "denoising_metrics.csv"),
            "per_class_accuracy_csv": str(output_dir / "per_class_accuracy.csv"),
            "adapter_attention_diagnostics_csv": str(output_dir / "adapter_attention_diagnostics.csv"),
            "diagnostic_only_teacher_offline_csv": str(output_dir / "diagnostic_only_teacher_offline.csv"),
            "mechanism_ablation_metrics_csv": str(output_dir / "mechanism_ablation_metrics.csv"),
            "confusion_matrices_json": str(output_dir / "confusion_matrices.json"),
            "summary_json": str(output_dir / "summary.json"),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


__all__ = [
    "TARGET_DENOISING_REPORT_CONTRACT",
    "TARGET_DENOISING_REPORT_SPLITS",
    "TARGET_DENOISING_STEP5",
    "TargetDenoisingReportConfig",
    "write_target_denoising_report",
]
