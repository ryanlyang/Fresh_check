"""Step 15 final reports for subtoken Particle Transformer comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any

from jetclass_fresh.hlt_baseline import save_json

from teacher_logit_reco.set_matching.train import source_metadata

from .config import SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE
from .train import SUBTOKEN_PART_LOWER_IS_BETTER_SELECTION_METRICS, SUBTOKEN_PART_SELECTION_METRICS


SUBTOKEN_PART_REPORT_STEP = "subtoken_part_step15_report_builder"
SUBTOKEN_PART_REPORT_CONTRACT = "subtoken_part_final_report_v1"
SUBTOKEN_PART_DEFAULT_BINARY_PRIMARY_METRIC = "fpr_at_signal_eff_0p50"
SUBTOKEN_PART_DEFAULT_MULTICLASS_PRIMARY_METRIC = "accuracy"
SUBTOKEN_PART_REPORT_SPLITS = ("model_val", "stack_val", "final_test")
SUBTOKEN_PART_REPORT_BINARY_METRICS = (
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


def _metric(payload: Mapping[str, Any] | None, path: Sequence[str]) -> Any:
    cursor: Any = payload
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


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
    if path.parts and experiment_dir.name and path.parts[0] == experiment_dir.name:
        sibling_candidate = experiment_dir.parent / path
        if sibling_candidate.exists():
            return sibling_candidate
        return path
    if path.exists():
        return path
    return candidate


def _variant_from_report(report_path: Path, report: Mapping[str, Any] | None) -> str:
    if isinstance(report, Mapping):
        for key in ("variant", "name"):
            value = report.get(key)
            if value:
                return str(value)
    return report_path.parent.name


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
                paths[str(variant)] = path

    requested = [str(variant) for variant in variants if str(variant)]
    if requested:
        for variant in requested:
            paths.setdefault(variant, experiment_dir / variant / "run_report.json")
    elif not paths:
        for report_path in sorted(experiment_dir.glob("*/run_report.json")):
            variant = report_path.parent.name
            if variant != "final_report":
                paths[variant] = report_path
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
    return "minimize" if metric_name in SUBTOKEN_PART_LOWER_IS_BETTER_SELECTION_METRICS else "maximize"


def _infer_num_classes(root_report: Mapping[str, Any] | None, child_reports: Sequence[Mapping[str, Any]]) -> int | None:
    value = _float_or_none(root_report.get("num_classes") if isinstance(root_report, Mapping) else None)
    if value is not None:
        return int(value)
    for report in child_reports:
        value = _float_or_none(report.get("num_classes"))
        if value is not None:
            return int(value)
    return None


def _infer_primary_metric(
    *,
    requested_metric: str | None,
    root_report: Mapping[str, Any] | None,
    child_reports: Sequence[Mapping[str, Any]],
    num_classes: int | None,
) -> str:
    if requested_metric:
        if requested_metric not in SUBTOKEN_PART_SELECTION_METRICS:
            raise ValueError(f"primary_metric must be one of {SUBTOKEN_PART_SELECTION_METRICS}")
        return requested_metric
    if num_classes == 2:
        return SUBTOKEN_PART_DEFAULT_BINARY_PRIMARY_METRIC
    root_metric = root_report.get("primary_metric") if isinstance(root_report, Mapping) else None
    if isinstance(root_metric, str) and root_metric in SUBTOKEN_PART_SELECTION_METRICS:
        return root_metric
    for report in child_reports:
        metric = report.get("selection_metric")
        if isinstance(metric, str) and metric in SUBTOKEN_PART_SELECTION_METRICS:
            return metric
    return SUBTOKEN_PART_DEFAULT_MULTICLASS_PRIMARY_METRIC


def _infer_comparison_split(
    *,
    requested_split: str | None,
    root_report: Mapping[str, Any] | None,
    child_reports: Sequence[Mapping[str, Any]],
) -> str:
    if requested_split:
        if requested_split not in SUBTOKEN_PART_REPORT_SPLITS:
            raise ValueError(f"comparison_split must be one of {SUBTOKEN_PART_REPORT_SPLITS}")
        return requested_split
    root_split = root_report.get("comparison_split") if isinstance(root_report, Mapping) else None
    if root_split in SUBTOKEN_PART_REPORT_SPLITS:
        return str(root_split)
    for split in ("final_test", "stack_val", "model_val"):
        if any(_metrics_for_split(report, split) is not None for report in child_reports):
            return split
    return "model_val"


def _resolve_checkpoint_path(report_path: Path, checkpoint: Any) -> Path | None:
    if not checkpoint:
        return None
    path = Path(str(checkpoint))
    if not path.is_absolute():
        path = report_path.parent / path
    return path


def _checkpoint_parameter_count(checkpoint_path: Path | None) -> dict[str, Any]:
    if checkpoint_path is None:
        return {"parameter_count": None, "checkpoint_size_bytes": None, "parameter_count_error": "missing checkpoint"}
    size = checkpoint_path.stat().st_size if checkpoint_path.exists() else None
    if not checkpoint_path.exists():
        return {
            "parameter_count": None,
            "checkpoint_size_bytes": size,
            "parameter_count_error": f"checkpoint does not exist: {checkpoint_path}",
        }
    try:
        import torch  # type: ignore

        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = payload.get("model_state_dict") if isinstance(payload, Mapping) else payload
        if not isinstance(state_dict, Mapping):
            raise ValueError("checkpoint payload does not contain a model_state_dict mapping")
        count = 0
        for value in state_dict.values():
            if hasattr(value, "numel"):
                count += int(value.numel())
        return {
            "parameter_count": int(count),
            "checkpoint_size_bytes": int(size) if size is not None else None,
            "parameter_count_error": None,
        }
    except Exception as exc:  # pragma: no cover - environment/checkpoint dependent
        return {
            "parameter_count": None,
            "checkpoint_size_bytes": int(size) if size is not None else None,
            "parameter_count_error": str(exc),
        }


def _flatten_metrics_for_row(row: dict[str, Any], report: Mapping[str, Any]) -> None:
    for split in SUBTOKEN_PART_REPORT_SPLITS:
        metrics = _metrics_for_split(report, split)
        prefix = f"{split}_"
        if not isinstance(metrics, Mapping):
            row[f"{prefix}available"] = False
            continue
        row[f"{prefix}available"] = True
        for key, value in metrics.items():
            if key in {"binary_metrics", "diagnostics"}:
                continue
            if isinstance(value, (int, float, str, bool)) or value is None:
                row[f"{prefix}{key}"] = value
        for key in ("n_jets", "loss", "accuracy", "macro_per_class_accuracy"):
            row[f"{prefix}{key}"] = metrics.get(key)
        binary = metrics.get("binary_metrics")
        if isinstance(binary, Mapping):
            for key, value in binary.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    row[f"{prefix}{key}"] = value
            for key in SUBTOKEN_PART_REPORT_BINARY_METRICS:
                row.setdefault(f"{prefix}{key}", binary.get(key))


def _gate_diagnostic_rows(variant: str, report_path: Path, report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SUBTOKEN_PART_REPORT_SPLITS:
        metrics = _metrics_for_split(report, split)
        diagnostics = metrics.get("diagnostics") if isinstance(metrics, Mapping) else None
        if not isinstance(diagnostics, Mapping):
            continue
        for key, value in diagnostics.items():
            if "gate" not in str(key):
                continue
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                for index, item in enumerate(value):
                    rows.append(
                        {
                            "variant": variant,
                            "split": split,
                            "diagnostic": f"{key}[{index}]",
                            "value": _float_or_none(item),
                            "raw_value": item,
                            "report_path": str(report_path),
                        }
                    )
            else:
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
    run_config_path = report_path.parent / "slurm_run_config.json"
    run_config = _read_json(run_config_path)
    if not isinstance(run_config, Mapping):
        run_config = {}
    env = run_config.get("environment") if isinstance(run_config.get("environment"), Mapping) else run_config
    runtime = report.get("runtime") if isinstance(report.get("runtime"), Mapping) else {}
    slurm_job_id = run_config.get("slurm_job_id") or run_config.get("job_id")
    slurm_job_name = run_config.get("slurm_job_name") or run_config.get("job_name")
    if isinstance(env, Mapping):
        slurm_job_id = slurm_job_id or env.get("SLURM_JOB_ID")
        slurm_job_name = slurm_job_name or env.get("SLURM_JOB_NAME")
    return {
        "variant": variant,
        "report_path": str(report_path),
        "run_config_path": str(run_config_path) if run_config_path.exists() else None,
        "slurm_job_id": slurm_job_id,
        "slurm_job_name": slurm_job_name,
        "job_kind": run_config.get("job_kind") or run_config.get("RUN_CONFIG_JOB_KIND"),
        "hostname": run_config.get("hostname"),
        "requested_cpus": env.get("SLURM_CPUS_PER_TASK") if isinstance(env, Mapping) else None,
        "requested_memory": env.get("SLURM_MEM_PER_NODE") if isinstance(env, Mapping) else None,
        "requested_gpus": env.get("SLURM_GPUS") if isinstance(env, Mapping) else None,
        "epochs_completed": report.get("epochs_completed"),
        "best_epoch": report.get("best_epoch"),
        "walltime_seconds": report.get("walltime_seconds") or runtime.get("elapsed_seconds"),
        "elapsed_seconds": runtime.get("elapsed_seconds"),
        "elapsed_minutes": runtime.get("elapsed_minutes"),
        "seconds_per_completed_epoch": runtime.get("seconds_per_completed_epoch"),
        "walltime_note": (
            "Elapsed time comes from the child Python report when available; Slurm queue/wait time "
            "is not persisted in slurm_run_config.json."
        ),
    }


def _add_baseline_comparison(
    rows: list[dict[str, Any]],
    *,
    baseline_variant: str,
    primary_metric: str,
    direction: str,
) -> list[dict[str, Any]]:
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
    return rows


def _best_row(rows: Sequence[Mapping[str, Any]], *, direction: str) -> Mapping[str, Any] | None:
    scored: list[tuple[float, Mapping[str, Any]]] = []
    for row in rows:
        value = _float_or_none(row.get("primary_metric_value"))
        if value is None:
            continue
        score = -value if direction == "minimize" else value
        scored.append((score, row))
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = report.get("comparison_summary", {})
    rows = report.get("metric_table", [])
    lines = [
        "# Subtoken Particle Transformer Final Report",
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
        metric = summary.get("primary_metric")
        split = summary.get("comparison_split")
        lines.append("| variant | primary | accuracy | auc | fpr50 | beats baseline |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for row in rows:
            lines.append(
                "| {variant} | {primary} | {accuracy} | {auc} | {fpr50} | {beats} |".format(
                    variant=row.get("variant"),
                    primary=row.get("primary_metric_value"),
                    accuracy=row.get(f"{split}_accuracy"),
                    auc=row.get(f"{split}_auc"),
                    fpr50=row.get(f"{split}_fpr_at_signal_eff_0p50"),
                    beats=row.get("beats_baseline"),
                )
            )
        lines.append("")
        lines.append(f"Primary metric key: `{split}_{metric}`.")
    else:
        lines.append("No metric rows were found.")
    problems = report.get("problems") or []
    if problems:
        lines.extend(["", "## Problems", ""])
        lines.extend(f"- {problem}" for problem in problems)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class SubtokenPartReportConfig:
    """Configuration for the Step 15 report builder."""

    output_dir: str
    experiment_dir: str
    primary_metric: str | None = None
    comparison_split: str | None = None
    variants: tuple[str, ...] = ()
    baseline_variant: str = SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE
    include_parameter_counts: bool = True
    confirm_final_test: bool = False

    def __post_init__(self) -> None:
        if self.primary_metric is not None and self.primary_metric not in SUBTOKEN_PART_SELECTION_METRICS:
            raise ValueError(f"primary_metric must be one of {SUBTOKEN_PART_SELECTION_METRICS}")
        if self.comparison_split is not None and self.comparison_split not in SUBTOKEN_PART_REPORT_SPLITS:
            raise ValueError(f"comparison_split must be one of {SUBTOKEN_PART_REPORT_SPLITS}")
        if not str(self.output_dir):
            raise ValueError("output_dir is required")
        if not str(self.experiment_dir):
            raise ValueError("experiment_dir is required")


def build_subtoken_part_final_report(config: SubtokenPartReportConfig) -> dict[str, Any]:
    """Build Step 15 final tables from a Step 13 compatibility experiment directory."""

    experiment_dir = Path(config.experiment_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []

    root_report_path = experiment_dir / "run_report.json"
    root_report = _read_json(root_report_path)
    if root_report is not None and not isinstance(root_report, Mapping):
        problems.append(f"root run_report is not a JSON object: {root_report_path}")
        root_report = None

    child_paths = _load_child_report_paths(
        experiment_dir,
        root_report=root_report,
        variants=config.variants,
    )
    child_payloads: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for requested_variant, report_path in child_paths.items():
        payload = _read_json(report_path)
        if payload is None:
            problems.append(f"missing child run_report for {requested_variant}: {report_path}")
            continue
        if not isinstance(payload, Mapping):
            problems.append(f"child run_report is not a JSON object for {requested_variant}: {report_path}")
            continue
        variant = _variant_from_report(report_path, payload)
        child_payloads[variant] = (report_path, payload)

    child_reports = [payload for _path, payload in child_payloads.values()]
    num_classes = _infer_num_classes(root_report, child_reports)
    primary_metric = _infer_primary_metric(
        requested_metric=config.primary_metric,
        root_report=root_report,
        child_reports=child_reports,
        num_classes=num_classes,
    )
    comparison_split = _infer_comparison_split(
        requested_split=config.comparison_split,
        root_report=root_report,
        child_reports=child_reports,
    )
    if config.confirm_final_test and comparison_split != "final_test":
        problems.append("confirm_final_test=True but final_test metrics were not selected/available")
    direction = _metric_direction(primary_metric)

    metric_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for variant, (report_path, payload) in child_payloads.items():
        metrics = _metrics_for_split(payload, comparison_split)
        primary_value = _lookup_metric(metrics, primary_metric)
        row = {
            "variant": variant,
            "source_type": "hlt_part_baseline" if variant == config.baseline_variant else "subtoken_tagger",
            "report_path": str(report_path),
            "checkpoint": payload.get("checkpoint"),
            "experiment_step": payload.get("experiment_step"),
            "output_contract": payload.get("output_contract"),
            "best_epoch": payload.get("best_epoch"),
            "epochs_completed": payload.get("epochs_completed"),
            "selection_metric": payload.get("selection_metric"),
            "selection_metric_direction": payload.get("selection_metric_direction"),
            "best_model_selection_metric_value": payload.get("best_model_selection_metric_value"),
            "comparison_split": comparison_split,
            "primary_metric": primary_metric,
            "primary_metric_direction": direction,
            "primary_metric_value": primary_value,
            "num_classes": payload.get("num_classes", num_classes),
            "label_names": payload.get("label_names"),
            "label_filter": payload.get("label_filter"),
            "inference_consumes_hlt_only": payload.get("inference_consumes_hlt_only"),
        }
        _flatten_metrics_for_row(row, payload)
        metric_rows.append(row)
        gate_rows.extend(_gate_diagnostic_rows(variant, report_path, payload))

        checkpoint_path = _resolve_checkpoint_path(report_path, payload.get("checkpoint"))
        parameter_row = {
            "variant": variant,
            "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
            "report_path": str(report_path),
        }
        if config.include_parameter_counts:
            parameter_row.update(_checkpoint_parameter_count(checkpoint_path))
        else:
            parameter_row.update(
                {
                    "parameter_count": None,
                    "checkpoint_size_bytes": checkpoint_path.stat().st_size
                    if checkpoint_path is not None and checkpoint_path.exists()
                    else None,
                    "parameter_count_error": "disabled",
                }
            )
        parameter_rows.append(parameter_row)
        runtime_rows.append(_runtime_row(variant, report_path, payload))

    _add_baseline_comparison(
        metric_rows,
        baseline_variant=str(config.baseline_variant),
        primary_metric=primary_metric,
        direction=direction,
    )
    best = _best_row(metric_rows, direction=direction)
    if best is None:
        problems.append(f"no rows had comparison metric {comparison_split}_{primary_metric}")
    baseline = next((row for row in metric_rows if row.get("variant") == config.baseline_variant), None)
    if baseline is None:
        problems.append(f"baseline variant was not found: {config.baseline_variant}")

    output_paths = {
        "report_json": str(output_dir / "subtoken_part_final_report.json"),
        "report_markdown": str(output_dir / "subtoken_part_final_report.md"),
        "metric_table_csv": str(output_dir / "metric_table.csv"),
        "gate_diagnostics_csv": str(output_dir / "gate_diagnostics.csv"),
        "parameter_counts_csv": str(output_dir / "parameter_counts.csv"),
        "runtime_summary_csv": str(output_dir / "runtime_summary.csv"),
        "baseline_comparison_csv": str(output_dir / "baseline_comparison.csv"),
        "run_report": str(output_dir / "run_report.json"),
    }
    report = {
        "experiment_step": SUBTOKEN_PART_REPORT_STEP,
        "output_contract": SUBTOKEN_PART_REPORT_CONTRACT,
        "ok": len(problems) == 0,
        "problems": problems,
        "experiment_dir": str(experiment_dir),
        "root_report": str(root_report_path) if root_report_path.exists() else None,
        "config": asdict(config),
        "source": source_metadata(),
        "num_classes": num_classes,
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
            "binary_default_rule": (
                "Binary reports default to fpr_at_signal_eff_0p50 with lower-is-better semantics "
                "unless --primary-metric overrides it."
            ),
        },
        "metric_table": metric_rows,
        "gate_diagnostics": gate_rows,
        "parameter_counts": parameter_rows,
        "runtime_summary": runtime_rows,
        "child_reports": {variant: str(path) for variant, (path, _payload) in child_payloads.items()},
        "outputs": output_paths,
    }

    _write_csv(Path(output_paths["metric_table_csv"]), metric_rows)
    _write_csv(Path(output_paths["gate_diagnostics_csv"]), gate_rows)
    _write_csv(Path(output_paths["parameter_counts_csv"]), parameter_rows)
    _write_csv(Path(output_paths["runtime_summary_csv"]), runtime_rows)
    _write_csv(Path(output_paths["baseline_comparison_csv"]), metric_rows)
    save_json(Path(output_paths["report_json"]), report)
    save_json(Path(output_paths["run_report"]), report)
    _write_markdown(Path(output_paths["report_markdown"]), report)
    return report


__all__ = [
    "SUBTOKEN_PART_DEFAULT_BINARY_PRIMARY_METRIC",
    "SUBTOKEN_PART_DEFAULT_MULTICLASS_PRIMARY_METRIC",
    "SUBTOKEN_PART_REPORT_CONTRACT",
    "SUBTOKEN_PART_REPORT_STEP",
    "SubtokenPartReportConfig",
    "build_subtoken_part_final_report",
]
