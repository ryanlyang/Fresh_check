"""Real-vs-shuffled audit report for reliability-gated dual-view ParT runs."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from jetclass_fresh.hlt_baseline import save_json

from .config import (
    DUALVIEW_PART_PRIMARY_METRIC,
    DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL,
    DUALVIEW_PART_VARIANT_SHUFFLED_PN_CONTROL,
)
from .training import DUALVIEW_PART_STEP10


DUALVIEW_PART_REPORT_STEP = "reliability_gated_dualview_part_step10_real_vs_shuffled_report"
DUALVIEW_PART_REPORT_CONTRACT = "dualview_part_real_vs_shuffled_report_v1"


@dataclass(frozen=True)
class DualViewPartReportConfig:
    output_dir: str
    experiment_dir: str
    tagger_root: str | None = None
    variants: tuple[str, ...] = (
        DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL,
        DUALVIEW_PART_VARIANT_SHUFFLED_PN_CONTROL,
    )
    real_variant: str = DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL
    shuffled_variant: str = DUALVIEW_PART_VARIANT_SHUFFLED_PN_CONTROL
    primary_metric: str = DUALVIEW_PART_PRIMARY_METRIC
    comparison_split: str = "final_test"
    confirm_final_test: bool = True
    require_real_beats_shuffled: bool = True

    def __post_init__(self) -> None:
        if not self.output_dir:
            raise ValueError("output_dir is required")
        if not self.experiment_dir:
            raise ValueError("experiment_dir is required")
        if not self.variants:
            raise ValueError("at least one variant is required")
        if self.comparison_split not in {"stack_val", "final_test"}:
            raise ValueError("comparison_split must be stack_val or final_test")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _metric(metrics: Mapping[str, Any] | None, name: str) -> float | None:
    if not isinstance(metrics, Mapping):
        return None
    value = _float_or_none(metrics.get(name))
    if value is not None:
        return value
    binary = metrics.get("binary_metrics")
    if isinstance(binary, Mapping):
        return _float_or_none(binary.get(name))
    return None


def _split_metrics(report: Mapping[str, Any] | None, split: str) -> Mapping[str, Any] | None:
    if not isinstance(report, Mapping):
        return None
    if split == "final_test":
        metrics = report.get("final_test_metrics")
    elif split == "stack_val":
        metrics = report.get("best_stack_val_metrics")
    else:
        metrics = None
    return metrics if isinstance(metrics, Mapping) else None


def _metric_direction(metric_name: str) -> str:
    if str(metric_name).startswith("fpr_at_signal_eff_") or str(metric_name) == "loss":
        return "lower"
    return "higher"


def _score(metric_value: float | None, *, direction: str) -> float | None:
    if metric_value is None:
        return None
    return -float(metric_value) if direction == "lower" else float(metric_value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(str(key))
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _format_value(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return ""
    return f"{number:.8g}"


def _write_markdown(path: Path, report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison = report.get("real_vs_shuffled", {})
    lines = [
        "# Dual-View ParT Real-vs-Shuffled Report",
        "",
        f"- ok: `{report.get('ok')}`",
        f"- comparison split: `{comparison.get('split')}`",
        f"- primary metric: `{comparison.get('metric')}` ({comparison.get('direction')})",
        f"- real beats shuffled: `{comparison.get('real_beats_shuffled')}`",
        "",
        "| variant | shuffle | init ok | final FPR@50 | final AUC | final acc | stack FPR@50 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('variant')} | "
            f"{row.get('shuffle_pn_view')} | "
            f"{row.get('initialization_check_passed')} | "
            f"{_format_value(row.get('final_test_fpr_at_signal_eff_0p50'))} | "
            f"{_format_value(row.get('final_test_auc'))} | "
            f"{_format_value(row.get('final_test_accuracy'))} | "
            f"{_format_value(row.get('stack_val_fpr_at_signal_eff_0p50'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row_from_report(variant: str, path: Path, report: Mapping[str, Any] | None) -> dict[str, Any]:
    stack = _split_metrics(report, "stack_val")
    final = _split_metrics(report, "final_test")
    row = {
        "variant": str(variant),
        "run_report": str(path),
        "present": report is not None,
        "shuffle_pn_view": None if report is None else bool(report.get("shuffle_pn_view", False)),
        "initialization_check_passed": None if report is None else report.get("initialization_check_passed"),
        "best_epoch": None if report is None else report.get("best_epoch"),
        "final_test_evaluated": None if report is None else report.get("final_test_evaluated"),
        "selection_metric": None if report is None else report.get("selection_metric"),
        "stack_val_accuracy": _metric(stack, "accuracy"),
        "stack_val_auc": _metric(stack, "auc"),
        "stack_val_fpr_at_signal_eff_0p30": _metric(stack, "fpr_at_signal_eff_0p30"),
        "stack_val_fpr_at_signal_eff_0p50": _metric(stack, "fpr_at_signal_eff_0p50"),
        "final_test_accuracy": _metric(final, "accuracy"),
        "final_test_auc": _metric(final, "auc"),
        "final_test_fpr_at_signal_eff_0p30": _metric(final, "fpr_at_signal_eff_0p30"),
        "final_test_fpr_at_signal_eff_0p50": _metric(final, "fpr_at_signal_eff_0p50"),
    }
    return row


def build_dualview_part_report(config: DualViewPartReportConfig) -> dict[str, Any]:
    experiment_dir = Path(config.experiment_dir)
    tagger_root = Path(config.tagger_root) if config.tagger_root else experiment_dir / "taggers"
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for variant in config.variants:
        report_path = tagger_root / variant / "run_report.json"
        payload = _read_json(report_path)
        if payload is None:
            problems.append(f"missing run_report for {variant}: {report_path}")
        rows.append(_row_from_report(variant, report_path, payload))

    direction = _metric_direction(config.primary_metric)
    metric_key = f"{config.comparison_split}_{config.primary_metric}"
    for row in rows:
        row["comparison_metric"] = metric_key
        row["comparison_direction"] = direction
        row["comparison_value"] = row.get(metric_key)
        row["comparison_score"] = _score(_float_or_none(row.get(metric_key)), direction=direction)

    row_by_variant = {str(row.get("variant")): row for row in rows}
    real = row_by_variant.get(config.real_variant)
    shuffled = row_by_variant.get(config.shuffled_variant)
    real_value = _float_or_none(real.get(metric_key)) if real else None
    shuffled_value = _float_or_none(shuffled.get(metric_key)) if shuffled else None
    real_beats_shuffled = None
    margin = None
    if real_value is None:
        problems.append(f"missing real comparison metric {metric_key} for {config.real_variant}")
    if shuffled_value is None:
        problems.append(f"missing shuffled comparison metric {metric_key} for {config.shuffled_variant}")
    if real_value is not None and shuffled_value is not None:
        if direction == "lower":
            real_beats_shuffled = bool(real_value < shuffled_value)
            margin = float(shuffled_value - real_value)
        else:
            real_beats_shuffled = bool(real_value > shuffled_value)
            margin = float(real_value - shuffled_value)
        if bool(config.require_real_beats_shuffled) and not real_beats_shuffled:
            problems.append(
                f"real PN did not beat shuffled PN on {metric_key}: real={real_value}, shuffled={shuffled_value}"
            )

    if bool(config.confirm_final_test):
        for row in rows:
            if row.get("final_test_evaluated") is not True:
                problems.append(f"{row.get('variant')} did not evaluate final_test")
    for row in rows:
        if row.get("initialization_check_passed") is False:
            problems.append(f"{row.get('variant')} failed initialization check")

    report = {
        "ok": len(problems) == 0,
        "problems": problems,
        "experiment_step": DUALVIEW_PART_REPORT_STEP,
        "smoke_step": DUALVIEW_PART_STEP10,
        "output_contract": DUALVIEW_PART_REPORT_CONTRACT,
        "experiment_dir": str(experiment_dir),
        "tagger_root": str(tagger_root),
        "output_dir": str(output_dir),
        "variants": list(config.variants),
        "real_vs_shuffled": {
            "real_variant": config.real_variant,
            "shuffled_variant": config.shuffled_variant,
            "split": config.comparison_split,
            "metric": config.primary_metric,
            "metric_key": metric_key,
            "direction": direction,
            "real_value": real_value,
            "shuffled_value": shuffled_value,
            "margin_positive_means_real_better": margin,
            "real_beats_shuffled": real_beats_shuffled,
            "require_real_beats_shuffled": bool(config.require_real_beats_shuffled),
        },
        "rows": rows,
        "config": {
            "output_dir": config.output_dir,
            "experiment_dir": config.experiment_dir,
            "tagger_root": config.tagger_root,
            "variants": list(config.variants),
            "real_variant": config.real_variant,
            "shuffled_variant": config.shuffled_variant,
            "primary_metric": config.primary_metric,
            "comparison_split": config.comparison_split,
            "confirm_final_test": bool(config.confirm_final_test),
            "require_real_beats_shuffled": bool(config.require_real_beats_shuffled),
        },
    }
    save_json(output_dir / "dualview_part_report.json", report)
    _write_csv(output_dir / "metric_table.csv", rows)
    _write_markdown(output_dir / "dualview_part_report.md", report, rows)
    return report


__all__ = [
    "DUALVIEW_PART_REPORT_CONTRACT",
    "DUALVIEW_PART_REPORT_STEP",
    "DualViewPartReportConfig",
    "build_dualview_part_report",
]
