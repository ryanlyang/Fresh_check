"""Step 11 final reports for multi-scale subjet Particle Transformer runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any

from jetclass_fresh.hlt_baseline import save_json
from jetclass_fresh.hlt_cache import fixed_hlt_params_dict, fixed_hlt_params_from_strength

from teacher_logit_reco.set_matching.train import source_metadata

from .protocol import (
    MULTISCALE_SUBJET_BASELINE_VARIANT,
    MULTISCALE_SUBJET_CONTRACT,
    MULTISCALE_SUBJET_DEFAULT_VARIANTS,
    MULTISCALE_SUBJET_HLT_DEGRADATION_STRENGTH,
    MULTISCALE_SUBJET_PRIMARY_METRIC,
    MULTISCALE_SUBJET_PRIMARY_VARIANT,
    MULTISCALE_SUBJET_PROTOCOL_STEP,
    default_multiscale_subjet_part_protocol,
    multiscale_subjet_part_protocol_manifest,
)
from .train import MULTISCALE_SUBJET_LOWER_IS_BETTER_SELECTION_METRICS, MULTISCALE_SUBJET_SELECTION_METRICS


MULTISCALE_SUBJET_REPORT_STEP = "multiscale_subjet_part_step11_report_builder"
MULTISCALE_SUBJET_REPORT_CONTRACT = "multiscale_subjet_part_final_report_v1"
MULTISCALE_SUBJET_REPORT_SPLITS = ("model_val", "stack_val", "final_test")
MULTISCALE_SUBJET_REPORT_BINARY_METRICS = (
    "auc",
    "fpr_at_signal_eff_0p30",
    "fpr_at_signal_eff_0p50",
    "background_rejection_at_signal_eff_0p30",
    "background_rejection_at_signal_eff_0p50",
    "validation_threshold_final_test_fpr",
    "validation_threshold_final_test_signal_efficiency",
)
MULTISCALE_SUBJET_HLT_PARAM_TOLERANCE = 1.0e-9


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

    # Relative child paths are run artifacts first.  Falling back to cwd only
    # after experiment-dir resolution prevents an unrelated local file from
    # shadowing a real child report.
    candidate = experiment_dir / path
    if candidate.exists():
        return candidate
    if path.parts and experiment_dir.name and path.parts[0] == experiment_dir.name:
        sibling_candidate = experiment_dir.parent / path
        if sibling_candidate.exists():
            return sibling_candidate
        return sibling_candidate
    if path.exists():
        return path
    return candidate


def _variant_from_report(report_path: Path, report: Mapping[str, Any] | None) -> str:
    if isinstance(report, Mapping):
        for key in ("profile", "ablation_profile", "variant", "model_variant", "name"):
            value = report.get(key)
            if value:
                return str(value)
        config = report.get("config")
        if isinstance(config, Mapping) and config.get("ablation_profile"):
            return str(config["ablation_profile"])
        if isinstance(config, Mapping) and config.get("variant"):
            return str(config["variant"])
    return report_path.parent.name


def _load_child_report_paths(
    experiment_dir: Path,
    *,
    root_report: Mapping[str, Any] | None,
    child_reports: Sequence[Any] | Mapping[str, Any] | None,
    variants: Sequence[str],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    root_children = root_report.get("child_reports") if isinstance(root_report, Mapping) else None
    for payload in (root_children, child_reports):
        if isinstance(payload, Mapping):
            for variant, child in payload.items():
                path = _resolve_report_path(experiment_dir, child)
                if path is not None:
                    paths[str(variant)] = path
        elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            for child in payload:
                path = _resolve_report_path(experiment_dir, child)
                if path is not None:
                    paths[path.parent.name] = path

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

    nested_candidates = (
        ("evaluations", split, "metrics"),
        ("evaluations", split),
        ("metrics", split),
    )
    for path in nested_candidates:
        nested = _metric(report, path)
        if isinstance(nested, Mapping):
            return nested
    return None


def _lookup_metric(metrics: Mapping[str, Any] | None, metric_name: str) -> float | None:
    if metrics is None:
        return None
    value = _float_or_none(metrics.get(metric_name))
    if value is not None:
        return value
    for nested_key in ("binary_metrics", "binary", "classification_metrics"):
        nested = metrics.get(nested_key)
        if isinstance(nested, Mapping):
            value = _float_or_none(nested.get(metric_name))
            if value is not None:
                return value
    return None


def _metric_direction(metric_name: str) -> str:
    return "minimize" if metric_name in MULTISCALE_SUBJET_LOWER_IS_BETTER_SELECTION_METRICS else "maximize"


def _infer_comparison_split(
    *,
    requested_split: str | None,
    root_report: Mapping[str, Any] | None,
    child_reports: Sequence[Mapping[str, Any]],
) -> str:
    if requested_split:
        if requested_split not in MULTISCALE_SUBJET_REPORT_SPLITS:
            raise ValueError(f"comparison_split must be one of {MULTISCALE_SUBJET_REPORT_SPLITS}")
        return requested_split
    root_split = root_report.get("comparison_split") if isinstance(root_report, Mapping) else None
    if root_split in MULTISCALE_SUBJET_REPORT_SPLITS:
        return str(root_split)
    for split in ("final_test", "stack_val", "model_val"):
        if any(_metrics_for_split(report, split) is not None for report in child_reports):
            return split
    return "model_val"


def _resolve_checkpoint_path(report_path: Path, checkpoint: Any) -> Path | None:
    if not checkpoint:
        return None
    path = Path(str(checkpoint))
    if path.is_absolute():
        return path
    candidate = report_path.parent / path
    if candidate.exists():
        return candidate
    if path.exists():
        return path
    return candidate


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
            "note": "Checkpoint state_dict parameter count; frozen/trainable split is unavailable post-save.",
        }
    except Exception as exc:  # pragma: no cover - checkpoint/environment dependent
        return {
            "parameter_count": None,
            "checkpoint_size_bytes": int(size) if size is not None else None,
            "parameter_count_error": str(exc),
        }


def _flatten_metrics_for_row(row: dict[str, Any], report: Mapping[str, Any]) -> None:
    for split in MULTISCALE_SUBJET_REPORT_SPLITS:
        metrics = _metrics_for_split(report, split)
        prefix = f"{split}_"
        if not isinstance(metrics, Mapping):
            row[f"{prefix}available"] = False
            continue
        row[f"{prefix}available"] = True
        for key, value in metrics.items():
            if key in {"binary_metrics", "diagnostics", "hlt_degradation_slice_metrics"}:
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
        for key in MULTISCALE_SUBJET_REPORT_BINARY_METRICS:
            row.setdefault(f"{prefix}{key}", _lookup_metric(metrics, key))


def _diagnostic_category(key: str) -> str:
    clean = str(key).lower()
    if any(token in clean for token in ("assignment", "cluster", "subjet_valid", "valid_subjet", "entropy")):
        return "assignment"
    if any(token in clean for token in ("scale", "radius", "single_scale")):
        return "scale"
    if any(token in clean for token in ("readback", "delta", "gamma", "adapted_feature", "feature_delta")):
        return "residual_readback"
    if any(token in clean for token in ("fusion", "branch", "attention", "cls", "latent")):
        return "fusion"
    if any(token in clean for token in ("pair_bias", "mass", "pt_fraction", "cluster_radius")):
        return "physics_bias"
    return "model"


def _diagnostic_rows(variant: str, report_path: Path, report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in MULTISCALE_SUBJET_REPORT_SPLITS:
        metrics = _metrics_for_split(report, split)
        diagnostics = metrics.get("diagnostics") if isinstance(metrics, Mapping) else None
        if not isinstance(diagnostics, Mapping):
            continue
        for key, value in diagnostics.items():
            category = _diagnostic_category(str(key))
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                for index, item in enumerate(value):
                    rows.append(
                        {
                            "variant": variant,
                            "split": split,
                            "category": category,
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
                        "category": category,
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
        "runtime_note": "Elapsed time is child Python runtime when available; scheduler queue time is not included.",
    }


def _metadata_mapping(report: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    value = report.get(key)
    return value if isinstance(value, Mapping) else None


def _expected_hlt_params() -> dict[str, float]:
    return fixed_hlt_params_dict(fixed_hlt_params_from_strength(MULTISCALE_SUBJET_HLT_DEGRADATION_STRENGTH))


def _hlt_params_match_protocol(params: Any) -> bool | None:
    if not isinstance(params, Mapping):
        return None
    expected = _expected_hlt_params()
    for key, expected_value in expected.items():
        if key not in params:
            return False
        try:
            actual_value = float(params[key])
        except (TypeError, ValueError):
            return False
        if abs(actual_value - float(expected_value)) > MULTISCALE_SUBJET_HLT_PARAM_TOLERANCE:
            return False
    return True


def _degradation_rows(variant: str, report_path: Path, report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, key in (
        ("model_train", "train_dataset"),
        ("model_val", "val_dataset"),
        ("stack_val", "stack_val_dataset"),
        ("final_test", "final_test_dataset"),
    ):
        metadata = _metadata_mapping(report, key)
        if not isinstance(metadata, Mapping):
            continue
        params = metadata.get("hlt_params") if isinstance(metadata.get("hlt_params"), Mapping) else {}
        audit = metadata.get("hlt_protocol_audit") if isinstance(metadata.get("hlt_protocol_audit"), Mapping) else {}
        row = {
            "row_type": "metadata",
            "variant": variant,
            "split": split,
            "dataset_key": key,
            "report_path": str(report_path),
            "n_jets": metadata.get("n_jets"),
            "label_counts": metadata.get("label_counts"),
            "hlt_content_hash": metadata.get("hlt_content_hash"),
            "jet_identity_hash": metadata.get("jet_identity_hash"),
            "source_view": metadata.get("source_view"),
            "protocol_hlt_degradation_strength": MULTISCALE_SUBJET_HLT_DEGRADATION_STRENGTH,
            "hlt_params_match_protocol": _hlt_params_match_protocol(params),
            "hlt_protocol_audit_ok": audit.get("ok") if isinstance(audit, Mapping) else None,
            "hlt_protocol_audit_problems": audit.get("problems") if isinstance(audit, Mapping) else None,
            "hlt_params": params,
            "hlt_diagnostics_summary": metadata.get("hlt_diagnostics_summary"),
            "slice_note": (
                "Metadata-level HLT0.6 cache row. Behavioral per-jet degradation slices are included "
                "only when child metrics persist hlt_degradation_slice_metrics."
            ),
        }
        if isinstance(params, Mapping):
            for param_key, param_value in params.items():
                if isinstance(param_value, (int, float, str, bool)) or param_value is None:
                    row[f"hlt_param_{param_key}"] = param_value
        rows.append(row)
    return rows


def _degradation_slice_rows(variant: str, report_path: Path, report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in MULTISCALE_SUBJET_REPORT_SPLITS:
        metrics = _metrics_for_split(report, split)
        if not isinstance(metrics, Mapping):
            continue
        slice_metrics = metrics.get("hlt_degradation_slice_metrics")
        if not isinstance(slice_metrics, Sequence) or isinstance(slice_metrics, (str, bytes, bytearray)):
            continue
        for item in slice_metrics:
            if not isinstance(item, Mapping):
                continue
            row = {
                "row_type": "behavioral_slice",
                "variant": variant,
                "split": split,
                "dataset_key": f"{split}_metrics",
                "report_path": str(report_path),
                "protocol_hlt_degradation_strength": MULTISCALE_SUBJET_HLT_DEGRADATION_STRENGTH,
                "slice_note": "Behavioral metric slice computed from persisted per-jet HLT degradation diagnostics.",
            }
            row.update(dict(item))
            rows.append(row)
    return rows


def _add_baseline_comparison(
    rows: list[dict[str, Any]],
    *,
    baseline_variant: str,
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
            row["primary_metric_relative_improvement_vs_baseline"] = None
            row["beats_baseline"] = None
            continue
        delta = value - baseline_value
        improvement = baseline_value - value if direction == "minimize" else value - baseline_value
        row["primary_metric_delta_vs_baseline"] = delta
        row["primary_metric_improvement_vs_baseline"] = improvement
        row["primary_metric_relative_improvement_vs_baseline"] = (
            improvement / abs(baseline_value) if baseline_value != 0.0 else None
        )
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


def _strict_protocol_problems(
    *,
    primary_metric: str,
    comparison_split: str,
    baseline_variant: str,
    primary_variant: str,
    confirm_final_test: bool,
    metric_rows: Sequence[Mapping[str, Any]],
    degradation_rows: Sequence[Mapping[str, Any]],
    require_all_default_variants: bool,
    require_hlt_degradation_slices: bool,
) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    warnings: list[str] = []
    protocol = default_multiscale_subjet_part_protocol()
    if primary_metric != MULTISCALE_SUBJET_PRIMARY_METRIC:
        problems.append(f"strict protocol requires primary_metric={MULTISCALE_SUBJET_PRIMARY_METRIC}, got {primary_metric}")
    if comparison_split != "final_test":
        problems.append(f"strict protocol requires comparison_split=final_test, got {comparison_split}")
    if baseline_variant != MULTISCALE_SUBJET_BASELINE_VARIANT:
        problems.append(f"strict protocol requires baseline_variant={MULTISCALE_SUBJET_BASELINE_VARIANT}, got {baseline_variant}")
    if primary_variant != MULTISCALE_SUBJET_PRIMARY_VARIANT:
        problems.append(f"strict protocol requires primary_variant={MULTISCALE_SUBJET_PRIMARY_VARIANT}, got {primary_variant}")
    if not bool(confirm_final_test):
        problems.append("strict protocol requires confirm_final_test=True")

    present_variants = {str(row.get("variant")) for row in metric_rows}
    missing_default = [variant for variant in protocol.required_variant_names if variant not in present_variants]
    if missing_default:
        message = f"missing required protocol variants: {missing_default}"
        if bool(require_all_default_variants):
            problems.append(message)
        else:
            warnings.append(message)
    for variant in (baseline_variant, primary_variant):
        if variant not in present_variants:
            problems.append(f"strict protocol requires variant in metric rows: {variant}")

    for variant in (baseline_variant, primary_variant):
        row = next((item for item in metric_rows if item.get("variant") == variant), None)
        if row is not None and _float_or_none(row.get("primary_metric_value")) is None:
            problems.append(f"{variant} is missing {comparison_split}_{primary_metric}")

    metadata_rows = [row for row in degradation_rows if row.get("row_type") == "metadata"]
    if not metadata_rows:
        problems.append("strict protocol could not verify any HLT cache metadata rows")
    mismatched_hlt_rows = [
        row
        for row in metadata_rows
        if row.get("hlt_params_match_protocol") is False or row.get("hlt_protocol_audit_ok") is False
    ]
    unknown_hlt_rows = [
        row
        for row in metadata_rows
        if row.get("hlt_params_match_protocol") is None and row.get("hlt_protocol_audit_ok") is None
    ]
    if mismatched_hlt_rows:
        variants = sorted({str(row.get("variant")) for row in mismatched_hlt_rows})
        problems.append(f"strict protocol found HLT parameter rows that do not match degradation 0.6: {variants}")
    if unknown_hlt_rows:
        variants = sorted({str(row.get("variant")) for row in unknown_hlt_rows})
        problems.append(f"strict protocol could not verify HLT degradation parameters for rows: {variants}")
    behavioral_rows = [row for row in degradation_rows if row.get("row_type") == "behavioral_slice"]
    if bool(require_hlt_degradation_slices) and not behavioral_rows:
        problems.append("strict protocol requested behavioral HLT degradation slices, but no slice rows were found")
    elif not behavioral_rows:
        warnings.append("no behavioral HLT degradation slice rows found; report includes metadata-level HLT0.6 checks only")
    return problems, warnings


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    summary = report.get("comparison_summary", {})
    rows = report.get("metric_table", [])
    lines = [
        "# Multi-Scale Subjet Particle Transformer Report",
        "",
        f"- ok: {report.get('ok')}",
        f"- comparison split: {summary.get('comparison_split')}",
        f"- primary metric: {summary.get('primary_metric')} ({summary.get('primary_metric_direction')})",
        f"- best variant: {summary.get('best_variant')}",
        f"- best metric value: {summary.get('best_metric_value')}",
        f"- baseline variant: {summary.get('baseline_variant')}",
        f"- primary variant: {summary.get('primary_variant')}",
        "",
        "## Metrics",
        "",
    ]
    if rows:
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
        lines.append(f"Primary metric key: `{split}_{summary.get('primary_metric')}`.")
    else:
        lines.append("No metric rows were found.")
    problems = report.get("problems") or []
    if problems:
        lines.extend(["", "## Problems", ""])
        lines.extend(f"- {problem}" for problem in problems)
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class MultiScaleSubjetReportConfig:
    """Configuration for the Step 11 multi-scale subjet report builder."""

    output_dir: str
    experiment_dir: str
    primary_metric: str | None = None
    comparison_split: str | None = None
    variants: tuple[str, ...] = ()
    child_reports: tuple[str, ...] = ()
    baseline_variant: str = MULTISCALE_SUBJET_BASELINE_VARIANT
    primary_variant: str = MULTISCALE_SUBJET_PRIMARY_VARIANT
    include_parameter_counts: bool = True
    confirm_final_test: bool = True
    strict_protocol: bool = True
    require_all_default_variants: bool = True
    require_hlt_degradation_slices: bool = False

    def __post_init__(self) -> None:
        if self.primary_metric is not None and self.primary_metric not in MULTISCALE_SUBJET_SELECTION_METRICS:
            raise ValueError(f"primary_metric must be one of {MULTISCALE_SUBJET_SELECTION_METRICS}")
        if self.comparison_split is not None and self.comparison_split not in MULTISCALE_SUBJET_REPORT_SPLITS:
            raise ValueError(f"comparison_split must be one of {MULTISCALE_SUBJET_REPORT_SPLITS}")
        if not str(self.output_dir):
            raise ValueError("output_dir is required")
        if not str(self.experiment_dir):
            raise ValueError("experiment_dir is required")


def build_multiscale_subjet_part_report(config: MultiScaleSubjetReportConfig) -> dict[str, Any]:
    """Build Step 11 summary tables from child multi-scale subjet run reports."""

    experiment_dir = Path(config.experiment_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    warnings: list[str] = []

    root_report_path = experiment_dir / "run_report.json"
    root_report = _read_json(root_report_path)
    if root_report is not None and not isinstance(root_report, Mapping):
        problems.append(f"root run_report is not a JSON object: {root_report_path}")
        root_report = None

    child_paths = _load_child_report_paths(
        experiment_dir,
        root_report=root_report,
        child_reports=config.child_reports,
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
    primary_metric = config.primary_metric or MULTISCALE_SUBJET_PRIMARY_METRIC
    if primary_metric not in MULTISCALE_SUBJET_SELECTION_METRICS:
        raise ValueError(f"primary_metric must be one of {MULTISCALE_SUBJET_SELECTION_METRICS}")
    comparison_split = _infer_comparison_split(
        requested_split=config.comparison_split,
        root_report=root_report,
        child_reports=child_reports,
    )
    if config.confirm_final_test and comparison_split != "final_test":
        problems.append("confirm_final_test=True but final_test metrics were not selected/available")
    direction = _metric_direction(primary_metric)

    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    degradation_rows: list[dict[str, Any]] = []

    for variant, (report_path, payload) in child_payloads.items():
        metrics = _metrics_for_split(payload, comparison_split)
        primary_value = _lookup_metric(metrics, primary_metric)
        row = {
            "variant": variant,
            "ablation_profile": payload.get("profile") or _metric(payload, ("config", "ablation_profile")),
            "model_variant": payload.get("variant"),
            "source_type": "hlt_part_baseline" if variant == config.baseline_variant else "multiscale_subjet_variant",
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
            "num_classes": payload.get("num_classes"),
            "label_names": payload.get("label_names"),
            "label_filter": payload.get("label_filter"),
            "inference_consumes_hlt_only": payload.get("inference_consumes_hlt_only"),
            "uses_reference_part_backbone": _metric(payload, ("model_config", "uses_reference_part_backbone")),
            "baseline_recoverable_at_zero_gamma": _metric(payload, ("model_config", "baseline_recoverable_at_zero_gamma")),
            "query_mode": _metric(payload, ("model_config", "query_mode")),
            "scale_profile": _metric(payload, ("model_config", "scale_profile")),
            "use_assignment_scale_embedding": _metric(payload, ("model_config", "use_assignment_scale_embedding")),
            "use_token_scale_embedding": _metric(payload, ("model_config", "use_token_scale_embedding")),
            "use_scale_pair_embedding": _metric(payload, ("model_config", "use_scale_pair_embedding")),
            "effective_use_subjet_pair_bias": _metric(payload, ("model_config", "effective_use_subjet_pair_bias")),
        }
        _flatten_metrics_for_row(row, payload)
        metric_rows.append(row)
        diagnostic_rows.extend(_diagnostic_rows(variant, report_path, payload))

        checkpoint_path = _resolve_checkpoint_path(report_path, payload.get("checkpoint"))
        parameter_row = {
            "variant": variant,
            "model_variant": payload.get("variant"),
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
        degradation_rows.extend(_degradation_rows(variant, report_path, payload))
        degradation_rows.extend(_degradation_slice_rows(variant, report_path, payload))

    _add_baseline_comparison(metric_rows, baseline_variant=str(config.baseline_variant), direction=direction)
    best = _best_row(metric_rows, direction=direction)
    if best is None:
        problems.append(f"no rows had comparison metric {comparison_split}_{primary_metric}")
    baseline = next((row for row in metric_rows if row.get("variant") == config.baseline_variant), None)
    primary = next((row for row in metric_rows if row.get("variant") == config.primary_variant), None)
    if baseline is None:
        problems.append(f"baseline variant was not found: {config.baseline_variant}")
    if primary is None:
        problems.append(f"primary variant was not found: {config.primary_variant}")

    strict_problems, strict_warnings = _strict_protocol_problems(
        primary_metric=primary_metric,
        comparison_split=comparison_split,
        baseline_variant=str(config.baseline_variant),
        primary_variant=str(config.primary_variant),
        confirm_final_test=bool(config.confirm_final_test),
        metric_rows=metric_rows,
        degradation_rows=degradation_rows,
        require_all_default_variants=bool(config.require_all_default_variants),
        require_hlt_degradation_slices=bool(config.require_hlt_degradation_slices),
    )
    warnings.extend(strict_warnings)
    if bool(config.strict_protocol):
        problems.extend(strict_problems)

    diagnostic_category_counts: dict[str, int] = {}
    for row in diagnostic_rows:
        category = str(row.get("category"))
        diagnostic_category_counts[category] = diagnostic_category_counts.get(category, 0) + 1
    runtime_seconds = [_float_or_none(row.get("walltime_seconds")) for row in runtime_rows]
    runtime_seconds = [value for value in runtime_seconds if value is not None]
    parameter_counts = [_float_or_none(row.get("parameter_count")) for row in parameter_rows]
    parameter_counts = [value for value in parameter_counts if value is not None]
    behavioral_slice_rows = [row for row in degradation_rows if row.get("row_type") == "behavioral_slice"]

    output_paths = {
        "report_json": str(output_dir / "multiscale_subjet_part_report.json"),
        "report_markdown": str(output_dir / "multiscale_subjet_part_report.md"),
        "metric_table_csv": str(output_dir / "metric_table.csv"),
        "diagnostics_csv": str(output_dir / "diagnostics.csv"),
        "parameter_counts_csv": str(output_dir / "parameter_counts.csv"),
        "runtime_csv": str(output_dir / "runtime.csv"),
        "hlt_degradation_csv": str(output_dir / "hlt_degradation.csv"),
    }
    comparison_summary = {
        "comparison_split": comparison_split,
        "primary_metric": primary_metric,
        "primary_metric_direction": direction,
        "best_variant": None if best is None else best.get("variant"),
        "best_metric_value": None if best is None else best.get("primary_metric_value"),
        "baseline_variant": config.baseline_variant,
        "baseline_metric_value": None if baseline is None else baseline.get("primary_metric_value"),
        "primary_variant": config.primary_variant,
        "primary_metric_value": None if primary is None else primary.get("primary_metric_value"),
        "primary_beats_baseline": None if primary is None else primary.get("beats_baseline"),
        "primary_improvement_vs_baseline": None if primary is None else primary.get("primary_metric_improvement_vs_baseline"),
        "primary_relative_improvement_vs_baseline": None
        if primary is None
        else primary.get("primary_metric_relative_improvement_vs_baseline"),
        "metric_note": "For the QCD/Hgg HLT0.6 protocol, lower FPR@50 is better.",
    }
    report = {
        "experiment_step": MULTISCALE_SUBJET_REPORT_STEP,
        "contract": MULTISCALE_SUBJET_REPORT_CONTRACT,
        "protocol_step": MULTISCALE_SUBJET_PROTOCOL_STEP,
        "protocol_contract": MULTISCALE_SUBJET_CONTRACT,
        "protocol": multiscale_subjet_part_protocol_manifest(),
        "source": source_metadata(),
        "ok": not problems,
        "problems": problems,
        "warnings": warnings,
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "root_report_path": str(root_report_path) if root_report_path.exists() else None,
        "child_report_paths": {variant: str(path) for variant, (path, _payload) in child_payloads.items()},
        "metric_table": metric_rows,
        "comparison_summary": comparison_summary,
        "best_row": None if best is None else dict(best),
        "diagnostic_rows": diagnostic_rows,
        "diagnostic_summary": {
            "n_rows": len(diagnostic_rows),
            "category_counts": diagnostic_category_counts,
            "assignment_rows": diagnostic_category_counts.get("assignment", 0),
            "scale_rows": diagnostic_category_counts.get("scale", 0),
            "residual_readback_rows": diagnostic_category_counts.get("residual_readback", 0),
            "fusion_rows": diagnostic_category_counts.get("fusion", 0),
        },
        "parameter_rows": parameter_rows,
        "parameter_summary": {
            "n_rows": len(parameter_rows),
            "min_parameter_count": min(parameter_counts) if parameter_counts else None,
            "max_parameter_count": max(parameter_counts) if parameter_counts else None,
        },
        "runtime_rows": runtime_rows,
        "runtime_summary": {
            "n_rows": len(runtime_rows),
            "total_walltime_seconds": sum(runtime_seconds) if runtime_seconds else None,
            "max_walltime_seconds": max(runtime_seconds) if runtime_seconds else None,
        },
        "hlt_degradation_rows": degradation_rows,
        "hlt_degradation_summary": {
            "n_rows": len(degradation_rows),
            "metadata_rows": sum(1 for row in degradation_rows if row.get("row_type") == "metadata"),
            "behavioral_slice_rows": len(behavioral_slice_rows),
            "protocol_hlt_degradation_strength": MULTISCALE_SUBJET_HLT_DEGRADATION_STRENGTH,
        },
        "outputs": output_paths,
        "config": {
            "primary_metric": config.primary_metric,
            "comparison_split": config.comparison_split,
            "variants": list(config.variants),
            "child_reports": list(config.child_reports),
            "baseline_variant": config.baseline_variant,
            "primary_variant": config.primary_variant,
            "include_parameter_counts": config.include_parameter_counts,
            "confirm_final_test": config.confirm_final_test,
            "strict_protocol": config.strict_protocol,
            "require_all_default_variants": config.require_all_default_variants,
            "require_hlt_degradation_slices": config.require_hlt_degradation_slices,
        },
    }
    save_json(output_dir / "multiscale_subjet_part_report.json", _jsonable(report))
    save_json(output_dir / "run_report.json", _jsonable(report))
    _write_markdown(output_dir / "multiscale_subjet_part_report.md", report)
    _write_csv(output_dir / "metric_table.csv", metric_rows)
    _write_csv(output_dir / "diagnostics.csv", diagnostic_rows)
    _write_csv(output_dir / "parameter_counts.csv", parameter_rows)
    _write_csv(output_dir / "runtime.csv", runtime_rows)
    _write_csv(output_dir / "hlt_degradation.csv", degradation_rows)
    return _jsonable(report)


build_multiscale_subjet_report = build_multiscale_subjet_part_report
MultiscaleSubjetReportConfig = MultiScaleSubjetReportConfig


__all__ = [
    "MULTISCALE_SUBJET_REPORT_CONTRACT",
    "MULTISCALE_SUBJET_REPORT_STEP",
    "MULTISCALE_SUBJET_REPORT_SPLITS",
    "MultiScaleSubjetReportConfig",
    "MultiscaleSubjetReportConfig",
    "build_multiscale_subjet_part_report",
    "build_multiscale_subjet_report",
]
