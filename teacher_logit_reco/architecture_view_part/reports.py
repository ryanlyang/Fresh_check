"""Final reports for Architecture-View Residual ParT variants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from jetclass_fresh.hlt_baseline import save_json

from teacher_logit_reco.set_matching.train import source_metadata

from .config import (
    ARCHITECTURE_VIEW_DEFAULT_PILOT_VARIANTS,
    ARCHITECTURE_VIEW_PART_CONTRACT,
    ARCHITECTURE_VIEW_PRIMARY_METRIC,
    ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    normalize_architecture_view_variant,
)
from .train import (
    ARCHITECTURE_VIEW_LOWER_IS_BETTER_SELECTION_METRICS,
    ARCHITECTURE_VIEW_SELECTION_METRICS,
)


ARCHITECTURE_VIEW_REPORT_STEP = "architecture_view_part_step3_reports"
ARCHITECTURE_VIEW_REPORT_CONTRACT = f"{ARCHITECTURE_VIEW_PART_CONTRACT}_report_v1"
ARCHITECTURE_VIEW_REPORT_SPLITS = ("model_val", "stack_val", "final_test")
ARCHITECTURE_VIEW_VALIDATION_THRESHOLD_METRIC = "validation_threshold_final_test_fpr_at_signal_eff_0p50"
ARCHITECTURE_VIEW_REPORT_BINARY_METRICS = (
    "auc",
    "fpr_at_signal_eff_0p30",
    "fpr_at_signal_eff_0p50",
    "background_rejection_at_signal_eff_0p30",
    "background_rejection_at_signal_eff_0p50",
)


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


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_jsonable(value), sort_keys=True)
    return _jsonable(value)


def _float_or_none(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


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


def _metric(payload: Mapping[str, Any] | None, path: Sequence[str]) -> Any:
    cursor: Any = payload
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _resolve_report_path(experiment_dir: Path, value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("run_report", "run_report_path", "report_path", "path"):
            if key in value:
                return _resolve_report_path(experiment_dir, value[key])
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidate = experiment_dir / path
    if candidate.exists():
        return candidate
    if path.parts and path.parts[0] == experiment_dir.name:
        sibling_candidate = experiment_dir.parent / path
        if sibling_candidate.exists():
            return sibling_candidate
    if path.exists():
        return path
    return candidate


def _variant_from_report(report_path: Path, report: Mapping[str, Any] | None) -> str:
    if isinstance(report, Mapping):
        for key in ("variant", "name"):
            value = report.get(key)
            if value:
                return normalize_architecture_view_variant(str(value))
        config = report.get("config")
        if isinstance(config, Mapping) and config.get("variant"):
            return normalize_architecture_view_variant(str(config["variant"]))
    return normalize_architecture_view_variant(report_path.parent.name)


def _load_root_report(experiment_dir: Path) -> tuple[Path | None, Mapping[str, Any] | None, list[str]]:
    problems: list[str] = []
    for name in ("variant_suite_report.json", "run_report.json"):
        path = experiment_dir / name
        payload = _read_json(path)
        if payload is None:
            continue
        if not isinstance(payload, Mapping):
            problems.append(f"root report is not a JSON object: {path}")
            return path, None, problems
        return path, payload, problems
    return None, None, problems


def _load_child_report_paths(
    experiment_dir: Path,
    *,
    root_report: Mapping[str, Any] | None,
    variants: Sequence[str],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    child_reports = root_report.get("child_reports") if isinstance(root_report, Mapping) else None
    if isinstance(child_reports, Mapping):
        for variant, payload in child_reports.items():
            path = _resolve_report_path(experiment_dir, payload)
            if path is not None:
                paths[normalize_architecture_view_variant(str(variant))] = path

    run_reports = root_report.get("run_reports") if isinstance(root_report, Mapping) else None
    if isinstance(run_reports, Sequence) and not isinstance(run_reports, (str, bytes, bytearray)):
        for payload in run_reports:
            path = _resolve_report_path(experiment_dir, payload)
            if path is not None:
                paths.setdefault(_variant_from_report(path, None), path)

    requested = [normalize_architecture_view_variant(variant) for variant in variants if str(variant)]
    if requested:
        for variant in requested:
            paths.setdefault(variant, experiment_dir / variant / "run_report.json")
    elif not paths:
        for variant in ARCHITECTURE_VIEW_DEFAULT_PILOT_VARIANTS:
            paths.setdefault(variant, experiment_dir / variant / "run_report.json")
        for report_path in sorted(experiment_dir.glob("*/run_report.json")):
            if report_path.parent.name != "final_report":
                paths.setdefault(_variant_from_report(report_path, None), report_path)
    return paths


def _metrics_for_split(report: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    if split == "model_val":
        keys = ("best_model_val_metrics", "model_val_metrics")
    elif split == "stack_val":
        keys = ("stack_val_metrics", "best_stack_val_metrics")
    elif split == "final_test":
        keys = ("final_test_metrics",)
    else:
        keys = (f"{split}_metrics",)
    for key in keys:
        metrics = report.get(key)
        if isinstance(metrics, Mapping):
            return metrics
    nested = _metric(report, ("evaluations", split, "metrics"))
    return nested if isinstance(nested, Mapping) else None


def _lookup_metric(metrics: Mapping[str, Any] | None, metric_name: str) -> float | None:
    if metrics is None:
        return None
    value = _float_or_none(metrics.get(metric_name))
    if value is not None:
        return value
    binary = metrics.get("binary_metrics")
    if isinstance(binary, Mapping):
        return _float_or_none(binary.get(metric_name))
    return None


def _metric_direction(metric_name: str) -> str:
    return "minimize" if metric_name in ARCHITECTURE_VIEW_LOWER_IS_BETTER_SELECTION_METRICS else "maximize"


def _prediction_arrays(metrics: Mapping[str, Any] | None) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(metrics, Mapping):
        return None
    arrays = metrics.get("_prediction_arrays") or metrics.get("prediction_arrays")
    if not isinstance(arrays, Mapping):
        return None
    labels = arrays.get("labels")
    logits = arrays.get("logits")
    scores = arrays.get("scores") or arrays.get("signal_scores")
    if labels is None:
        return None
    labels_np = np.asarray(labels, dtype=np.int64).reshape(-1)
    if logits is not None:
        logits_np = np.asarray(logits, dtype=np.float64)
        if logits_np.ndim != 2 or logits_np.shape[1] != 2 or logits_np.shape[0] != labels_np.shape[0]:
            return None
        shifted = logits_np - np.max(logits_np, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.clip(exp.sum(axis=1, keepdims=True), 1.0e-300, None)
        return labels_np, probs[:, 1]
    if scores is not None:
        scores_np = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores_np.shape[0] != labels_np.shape[0]:
            return None
        return labels_np, scores_np
    return None


def _threshold_at_signal_efficiency(labels: np.ndarray, scores: np.ndarray, target_efficiency: float) -> float | None:
    positives = np.asarray(labels, dtype=np.int64).reshape(-1) == 1
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not bool(positives.any()):
        return None
    positive_scores = np.sort(scores[positives])[::-1]
    threshold_index = min(
        max(int(np.ceil(float(target_efficiency) * int(positive_scores.size))) - 1, 0),
        int(positive_scores.size) - 1,
    )
    return float(positive_scores[threshold_index])


def _apply_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = labels == 1
    negatives = labels == 0
    selected = scores >= float(threshold)
    signal_efficiency = float(np.mean(selected[positives])) if bool(positives.any()) else None
    false_positive_rate = float(np.mean(selected[negatives])) if bool(negatives.any()) else None
    return {
        "threshold": float(threshold),
        "signal_efficiency": signal_efficiency,
        "false_positive_rate": false_positive_rate,
        "background_rejection": None
        if false_positive_rate is None
        else (float("inf") if false_positive_rate == 0.0 else float(1.0 / false_positive_rate)),
    }


def _validation_threshold_final_test(report: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("validation_threshold_final_test", "validation_threshold_metrics"):
        value = report.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    model_val_arrays = _prediction_arrays(_metrics_for_split(report, "model_val"))
    final_test_arrays = _prediction_arrays(_metrics_for_split(report, "final_test"))
    if model_val_arrays is None or final_test_arrays is None:
        return {
            "available": False,
            "metric": ARCHITECTURE_VIEW_VALIDATION_THRESHOLD_METRIC,
            "reason": "prediction arrays are not stored in child run_report metrics",
            "target_signal_efficiency": 0.50,
            "threshold": None,
            "false_positive_rate": None,
            "signal_efficiency": None,
            "background_rejection": None,
        }
    val_labels, val_scores = model_val_arrays
    final_labels, final_scores = final_test_arrays
    threshold = _threshold_at_signal_efficiency(val_labels, val_scores, 0.50)
    if threshold is None:
        return {
            "available": False,
            "metric": ARCHITECTURE_VIEW_VALIDATION_THRESHOLD_METRIC,
            "reason": "model_val has no signal labels",
            "target_signal_efficiency": 0.50,
            "threshold": None,
            "false_positive_rate": None,
            "signal_efficiency": None,
            "background_rejection": None,
        }
    return {
        "available": True,
        "metric": ARCHITECTURE_VIEW_VALIDATION_THRESHOLD_METRIC,
        "source_split": "model_val",
        "target_signal_efficiency": 0.50,
        **_apply_threshold(final_labels, final_scores, threshold),
    }


def _flatten_metrics_for_row(row: dict[str, Any], report: Mapping[str, Any]) -> None:
    for split in ARCHITECTURE_VIEW_REPORT_SPLITS:
        metrics = _metrics_for_split(report, split)
        prefix = f"{split}_"
        if not isinstance(metrics, Mapping):
            row[f"{prefix}available"] = False
            continue
        row[f"{prefix}available"] = True
        for key in ("n_jets", "loss", "accuracy", "macro_per_class_accuracy"):
            row[f"{prefix}{key}"] = metrics.get(key)
        for key, value in metrics.items():
            if key in {"binary_metrics", "diagnostics", "_prediction_arrays", "prediction_arrays"}:
                continue
            if isinstance(value, (int, float, str, bool)) or value is None:
                row[f"{prefix}{key}"] = value
        binary = metrics.get("binary_metrics")
        if isinstance(binary, Mapping):
            for key in ARCHITECTURE_VIEW_REPORT_BINARY_METRICS:
                row[f"{prefix}{key}"] = binary.get(key)


def _diagnostic_rows(variant: str, report_path: Path, report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ARCHITECTURE_VIEW_REPORT_SPLITS:
        metrics = _metrics_for_split(report, split)
        diagnostics = metrics.get("diagnostics") if isinstance(metrics, Mapping) else None
        if not isinstance(diagnostics, Mapping):
            continue
        for key, value in diagnostics.items():
            rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "diagnostic": key,
                    "value": _float_or_none(value),
                    "raw_value": value,
                    "report_path": str(report_path),
                }
            )
    return rows


def _runtime_row(variant: str, report_path: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    runtime = report.get("runtime") if isinstance(report.get("runtime"), Mapping) else {}
    return {
        "variant": variant,
        "report_path": str(report_path),
        "best_epoch": report.get("best_epoch"),
        "epochs_completed": report.get("epochs_completed"),
        "walltime_seconds": report.get("walltime_seconds") or runtime.get("elapsed_seconds"),
        "elapsed_seconds": runtime.get("elapsed_seconds"),
        "elapsed_minutes": runtime.get("elapsed_minutes"),
        "seconds_per_completed_epoch": runtime.get("seconds_per_completed_epoch"),
    }


def _add_baseline_comparison(rows: list[dict[str, Any]], *, baseline_variant: str, direction: str) -> None:
    baseline = next((row for row in rows if row.get("variant") == baseline_variant), None)
    baseline_value = _float_or_none(baseline.get("primary_metric_value")) if baseline is not None else None
    for row in rows:
        row["baseline_variant"] = baseline_variant if baseline is not None else None
        row["baseline_primary_metric_value"] = baseline_value
        value = _float_or_none(row.get("primary_metric_value"))
        if baseline_value is None or value is None:
            row["primary_metric_delta_vs_baseline"] = None
            row["primary_metric_improvement_vs_baseline"] = None
            row["beats_baseline"] = None
            continue
        delta = value - baseline_value
        improvement = baseline_value - value if direction == "minimize" else value - baseline_value
        row["primary_metric_delta_vs_baseline"] = delta
        row["primary_metric_improvement_vs_baseline"] = improvement
        row["beats_baseline"] = improvement > 0.0


def _best_row(rows: Sequence[Mapping[str, Any]], *, direction: str) -> Mapping[str, Any] | None:
    scored: list[tuple[float, Mapping[str, Any]]] = []
    for row in rows:
        value = _float_or_none(row.get("primary_metric_value"))
        if value is not None:
            scored.append((-value if direction == "minimize" else value, row))
    return max(scored, key=lambda item: item[0])[1] if scored else None


def _sort_rows(rows: list[dict[str, Any]], *, direction: str) -> list[dict[str, Any]]:
    def key(row: Mapping[str, Any]) -> tuple[int, float, str]:
        value = _float_or_none(row.get("primary_metric_value"))
        if value is None:
            return (1, 0.0, str(row.get("variant")))
        return (0, value if direction == "minimize" else -value, str(row.get("variant")))

    return sorted(rows, key=key)


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = report.get("comparison_summary", {})
    rows = report.get("metric_table", [])
    lines = [
        "# Architecture-View Residual ParT Report",
        "",
        f"- ok: {report.get('ok')}",
        f"- comparison split: {summary.get('comparison_split')}",
        f"- primary metric: {summary.get('primary_metric')} ({summary.get('primary_metric_direction')})",
        f"- best variant: {summary.get('best_variant')}",
        f"- best metric value: {summary.get('best_metric_value')}",
        f"- baseline variant: {summary.get('baseline_variant')}",
        "",
        "## Metrics",
        "",
    ]
    if rows:
        lines.append("| variant | final fpr50 | final auc | final acc | val-threshold fpr | beats baseline |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for row in rows:
            lines.append(
                "| {variant} | {fpr50} | {auc} | {acc} | {val_fpr} | {beats} |".format(
                    variant=row.get("variant"),
                    fpr50=row.get("final_test_oracle_fpr_at_signal_eff_0p50"),
                    auc=row.get("final_test_auc"),
                    acc=row.get("final_test_accuracy"),
                    val_fpr=row.get(ARCHITECTURE_VIEW_VALIDATION_THRESHOLD_METRIC),
                    beats=row.get("beats_baseline"),
                )
            )
    else:
        lines.append("No metric rows were found.")
    problems = report.get("problems") or []
    if problems:
        lines.extend(["", "## Problems", ""])
        lines.extend(f"- {problem}" for problem in problems)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ArchitectureViewPartReportConfig:
    """Configuration for the architecture-view report builder."""

    output_dir: str
    experiment_dir: str
    primary_metric: str | None = None
    comparison_split: str | None = None
    variants: tuple[str, ...] = ()
    baseline_variant: str = ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK
    confirm_final_test: bool = False

    def __post_init__(self) -> None:
        if self.primary_metric is not None and self.primary_metric not in ARCHITECTURE_VIEW_SELECTION_METRICS:
            raise ValueError(f"primary_metric must be one of {ARCHITECTURE_VIEW_SELECTION_METRICS}")
        if self.comparison_split is not None and self.comparison_split not in ARCHITECTURE_VIEW_REPORT_SPLITS:
            raise ValueError(f"comparison_split must be one of {ARCHITECTURE_VIEW_REPORT_SPLITS}")
        if not str(self.output_dir):
            raise ValueError("output_dir is required")
        if not str(self.experiment_dir):
            raise ValueError("experiment_dir is required")


def build_architecture_view_part_report(config: ArchitectureViewPartReportConfig) -> dict[str, Any]:
    """Build summary tables from architecture-view child run reports."""

    experiment_dir = Path(config.experiment_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    root_report_path, root_report, root_problems = _load_root_report(experiment_dir)
    problems.extend(root_problems)

    child_paths = _load_child_report_paths(experiment_dir, root_report=root_report, variants=config.variants)
    child_payloads: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for requested_variant, report_path in child_paths.items():
        payload = _read_json(report_path)
        if payload is None:
            problems.append(f"missing child run_report for {requested_variant}: {report_path}")
            continue
        if not isinstance(payload, Mapping):
            problems.append(f"child run_report is not a JSON object for {requested_variant}: {report_path}")
            continue
        child_payloads[_variant_from_report(report_path, payload)] = (report_path, payload)

    primary_metric = config.primary_metric or ARCHITECTURE_VIEW_PRIMARY_METRIC
    comparison_split = config.comparison_split or "final_test"
    direction = _metric_direction(primary_metric)
    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for variant, (report_path, payload) in child_payloads.items():
        metrics = _metrics_for_split(payload, comparison_split)
        final_test_metrics = _metrics_for_split(payload, "final_test")
        validation_threshold = _validation_threshold_final_test(payload)
        row = {
            "variant": variant,
            "source_type": "exact_hlt_part_baseline"
            if variant == ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK
            else "architecture_view_residual_embedding_adapter",
            "report_path": str(report_path),
            "checkpoint": payload.get("checkpoint"),
            "experiment_step": payload.get("experiment_step"),
            "output_contract": payload.get("output_contract"),
            "variant_behavior": payload.get("variant_behavior"),
            "best_epoch": payload.get("best_epoch"),
            "epochs_completed": payload.get("epochs_completed"),
            "selection_metric": payload.get("selection_metric"),
            "selection_metric_direction": payload.get("selection_metric_direction"),
            "model_val_selected_metric": payload.get("best_model_selection_metric_value"),
            "comparison_split": comparison_split,
            "primary_metric": primary_metric,
            "primary_metric_direction": direction,
            "primary_metric_value": _lookup_metric(metrics, primary_metric),
            "num_classes": payload.get("num_classes"),
            "label_names": payload.get("label_names"),
            "label_filter": payload.get("label_filter"),
            "inference_consumes_hlt_only": payload.get("inference_consumes_hlt_only"),
            "final_test_auc": _lookup_metric(final_test_metrics, "auc"),
            "final_test_accuracy": _lookup_metric(final_test_metrics, "accuracy"),
            "final_test_oracle_fpr_at_signal_eff_0p50": _lookup_metric(
                final_test_metrics,
                "fpr_at_signal_eff_0p50",
            ),
            "final_test_background_rejection_at_signal_eff_0p50": _lookup_metric(
                final_test_metrics,
                "background_rejection_at_signal_eff_0p50",
            ),
            ARCHITECTURE_VIEW_VALIDATION_THRESHOLD_METRIC: validation_threshold.get("false_positive_rate"),
            "validation_threshold_final_test_signal_efficiency": validation_threshold.get("signal_efficiency"),
            "validation_threshold_final_test_background_rejection": validation_threshold.get("background_rejection"),
            "validation_threshold_final_test_threshold": validation_threshold.get("threshold"),
            "validation_threshold_final_test_available": validation_threshold.get("available"),
            "validation_threshold_final_test_reason": validation_threshold.get("reason"),
        }
        _flatten_metrics_for_row(row, payload)
        metric_rows.append(row)
        diagnostic_rows.extend(_diagnostic_rows(variant, report_path, payload))
        runtime_rows.append(_runtime_row(variant, report_path, payload))

    _add_baseline_comparison(metric_rows, baseline_variant=str(config.baseline_variant), direction=direction)
    metric_rows = _sort_rows(metric_rows, direction=direction)
    best = _best_row(metric_rows, direction=direction)
    baseline = next((row for row in metric_rows if row.get("variant") == config.baseline_variant), None)
    if best is None:
        problems.append(f"no rows had comparison metric {comparison_split}_{primary_metric}")
    if baseline is None:
        problems.append(f"baseline variant was not found: {config.baseline_variant}")
    if config.confirm_final_test:
        missing_final = [row["variant"] for row in metric_rows if not bool(row.get("final_test_available"))]
        if missing_final:
            problems.append(f"confirm_final_test=True but final_test metrics are missing for: {missing_final}")

    output_paths = {
        "report_json": str(output_dir / "architecture_view_part_final_report.json"),
        "report_markdown": str(output_dir / "architecture_view_part_final_report.md"),
        "metric_table_csv": str(output_dir / "metric_table.csv"),
        "baseline_comparison_csv": str(output_dir / "baseline_comparison.csv"),
        "diagnostics_csv": str(output_dir / "diagnostics.csv"),
        "runtime_summary_csv": str(output_dir / "runtime_summary.csv"),
        "run_report": str(output_dir / "run_report.json"),
    }
    report = {
        "experiment_step": ARCHITECTURE_VIEW_REPORT_STEP,
        "output_contract": ARCHITECTURE_VIEW_REPORT_CONTRACT,
        "ok": len(problems) == 0,
        "problems": problems,
        "experiment_dir": str(experiment_dir),
        "root_report": str(root_report_path) if root_report_path is not None else None,
        "config": asdict(config),
        "source": source_metadata(),
        "comparison_summary": {
            "comparison_split": comparison_split,
            "primary_metric": primary_metric,
            "primary_metric_direction": direction,
            "best_variant": best.get("variant") if best is not None else None,
            "best_metric_value": best.get("primary_metric_value") if best is not None else None,
            "baseline_variant": config.baseline_variant if baseline is not None else None,
            "baseline_metric_value": baseline.get("primary_metric_value") if baseline is not None else None,
            "best_improvement_vs_baseline": best.get("primary_metric_improvement_vs_baseline")
            if best is not None
            else None,
        },
        "metric_table": metric_rows,
        "diagnostics": diagnostic_rows,
        "runtime_summary": runtime_rows,
        "child_reports": {variant: str(path) for variant, (path, _payload) in child_payloads.items()},
        "outputs": output_paths,
    }
    _write_csv(Path(output_paths["metric_table_csv"]), metric_rows)
    _write_csv(Path(output_paths["baseline_comparison_csv"]), metric_rows)
    _write_csv(Path(output_paths["diagnostics_csv"]), diagnostic_rows)
    _write_csv(Path(output_paths["runtime_summary_csv"]), runtime_rows)
    save_json(Path(output_paths["report_json"]), report)
    save_json(Path(output_paths["run_report"]), report)
    _write_markdown(Path(output_paths["report_markdown"]), report)
    return report


__all__ = [
    "ARCHITECTURE_VIEW_REPORT_CONTRACT",
    "ARCHITECTURE_VIEW_REPORT_STEP",
    "ARCHITECTURE_VIEW_VALIDATION_THRESHOLD_METRIC",
    "ArchitectureViewPartReportConfig",
    "build_architecture_view_part_report",
]
