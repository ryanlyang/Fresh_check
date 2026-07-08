"""Aggregate reporting for HLT multiview source/fusion experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import save_json

from .config import (
    HLTMVExperimentConfig,
    HLTMVExperimentLayout,
    HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
    HLT_MV_EXPERIMENT_NAME,
    HLT_MV_FUSION_HLT_RANDOM_4SEED,
    HLT_MV_FUSION_PRETRAINED_DUALVIEW_4MODEL,
    HLT_MV_FUSION_SCRATCH_DUALVIEW_4MODEL,
    HLT_MV_FUSION_SOURCE_5VIEW,
    HLT_MV_TRIVIEW_MODEL_NAME,
    default_hlt_mv_experiment_config,
    default_hlt_mv_experiment_layout,
)
from .logit_fusion import (
    HLT_MV_LOGIT_FUSION_UNIFORM_METHOD,
    HLT_MV_LOGIT_FUSION_WEIGHTED_METHOD,
)
from .particle_dualview import (
    HLT_MV_PRETRAINED_DUALVIEW_REPORT,
    HLT_MV_SCRATCH_DUALVIEW_REPORT,
)
from .triview import HLT_MV_TRIVIEW_REPORT


HLT_MV_FINAL_REPORT_EXPERIMENT_STEP = "hlt_mv_step8_final_report"
HLT_MV_FINAL_REPORT_CONTRACT = "hlt_multiview_source_fusion_final_report_v1"
HLT_MV_FINAL_REPORT_JSON = "hlt_multiview_source_fusion_report.json"
HLT_MV_FINAL_REPORT_MD = "hlt_multiview_source_fusion_report.md"
HLT_MV_FINAL_REPORT_SUMMARY_JSON = "summary.json"
HLT_MV_FINAL_REPORT_RUN_JSON = "run_report.json"
HLT_MV_FINAL_REPORT_METRIC_TABLE_CSV = "metric_table.csv"


@dataclass(frozen=True)
class HLTMVFinalReportConfig:
    """Configuration for the aggregate HLT-MV final report."""

    output_dir: str
    output_root: str = "checkpoints"
    pdv3_experiment_name: str = HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME
    allow_missing: bool = False
    require_triview: bool = False
    overwrite: bool = False


def _read_json(path: Path, *, allow_missing: bool) -> dict[str, Any] | None:
    if not path.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(f"Missing HLT-MV report artifact: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _nested_metric(metrics: Mapping[str, Any] | None, key: str) -> Any:
    if not isinstance(metrics, Mapping):
        return None
    if key in metrics:
        return metrics.get(key)
    aliases = {
        "cross_entropy": ("ce", "loss"),
        "accuracy": ("acc",),
    }
    for alias in aliases.get(key, ()):
        if alias in metrics:
            return metrics.get(alias)
    return None


def _metrics_from_report(report: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    if split == "model_val":
        for key in ("model_val_prediction_metrics", "model_val_metrics", "best_model_val_training_metrics"):
            value = report.get(key)
            if isinstance(value, Mapping):
                return value
    if split == "final_test":
        for key in ("final_test_metrics",):
            value = report.get(key)
            if isinstance(value, Mapping):
                return value
        final_report = report.get("final_test_report")
        if isinstance(final_report, Mapping):
            nested = final_report.get("metrics")
            return nested if isinstance(nested, Mapping) else final_report
    return None


def _model_row(
    *,
    family: str,
    model_name: str,
    report_path: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    model_val = _metrics_from_report(report, "model_val")
    final_test = _metrics_from_report(report, "final_test")
    return {
        "family": family,
        "model_name": model_name,
        "method": "direct",
        "report_path": report_path.as_posix(),
        "ok": bool(report.get("ok")),
        "model_val_accuracy": _nested_metric(model_val, "accuracy"),
        "model_val_cross_entropy": _nested_metric(model_val, "cross_entropy"),
        "final_test_accuracy": _nested_metric(final_test, "accuracy"),
        "final_test_cross_entropy": _nested_metric(final_test, "cross_entropy"),
    }


def _fusion_rows(
    *,
    family: str,
    fusion_name: str,
    report_path: Path,
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods = report.get("methods")
    if not isinstance(methods, Mapping):
        return rows
    for method_name in (HLT_MV_LOGIT_FUSION_UNIFORM_METHOD, HLT_MV_LOGIT_FUSION_WEIGHTED_METHOD):
        payload = methods.get(method_name)
        if not isinstance(payload, Mapping):
            continue
        metrics = payload.get("metrics")
        if not isinstance(metrics, Mapping):
            metrics = {}
        model_val = metrics.get("model_val") if isinstance(metrics.get("model_val"), Mapping) else None
        final_test = metrics.get("final_test") if isinstance(metrics.get("final_test"), Mapping) else None
        rows.append(
            {
                "family": family,
                "model_name": fusion_name,
                "method": method_name,
                "report_path": report_path.as_posix(),
                "ok": bool(report.get("ok")),
                "model_val_accuracy": _nested_metric(model_val, "accuracy"),
                "model_val_cross_entropy": _nested_metric(model_val, "cross_entropy"),
                "final_test_accuracy": _nested_metric(final_test, "accuracy"),
                "final_test_cross_entropy": _nested_metric(final_test, "cross_entropy"),
            }
        )
    return rows


def _add_model_rows(
    rows: list[dict[str, Any]],
    missing: list[str],
    *,
    family: str,
    names: tuple[str, ...],
    path_for_name,
    report_filename: str = "run_report.json",
    allow_missing: bool,
) -> None:
    for name in names:
        report_path = Path(path_for_name(name)) / report_filename
        report = _read_json(report_path, allow_missing=allow_missing)
        if report is None:
            missing.append(report_path.as_posix())
            continue
        rows.append(_model_row(family=family, model_name=name, report_path=report_path, report=report))


def _best_row_by_model_val(rows: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    scored = [
        row
        for row in rows
        if row.get("model_val_cross_entropy") is not None or row.get("model_val_accuracy") is not None
    ]
    if not scored:
        return None
    return min(
        scored,
        key=lambda row: (
            float(row.get("model_val_cross_entropy") if row.get("model_val_cross_entropy") is not None else float("inf")),
            -float(row.get("model_val_accuracy") if row.get("model_val_accuracy") is not None else -1.0),
            -float(row.get("final_test_accuracy") if row.get("final_test_accuracy") is not None else -1.0),
        ),
    )


def _posthoc_best_row_by_final_test(rows: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    scored = [row for row in rows if row.get("final_test_accuracy") is not None]
    if not scored:
        return None
    return max(
        scored,
        key=lambda row: (
            float(row.get("final_test_accuracy") if row.get("final_test_accuracy") is not None else -1.0),
            float(row.get("model_val_accuracy") if row.get("model_val_accuracy") is not None else -1.0),
        ),
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(str(key))
                seen.add(str(key))
    if not keys:
        keys = ["available"]
        rows = [{"available": False}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _render_markdown(report: Mapping[str, Any]) -> str:
    rows = report.get("metric_rows")
    lines = [
        "# HLT Multiview Source/Fusion Report",
        "",
        f"- ok: `{report.get('ok')}`",
        f"- pdv3_experiment: `{report.get('pdv3_experiment_name')}`",
        f"- best_by_model_val: `{(report.get('best_by_model_val') or {}).get('model_name')}`",
        f"- posthoc_best_by_final_test: `{(report.get('posthoc_best_by_final_test') or {}).get('model_name')}`",
        f"- triview_required: `{report.get('triview_required')}`",
        "",
        "| family | model | method | val acc | test acc |",
        "|---|---|---:|---:|---:|",
    ]
    if isinstance(rows, list):
        for row in rows:
            lines.append(
                "| {family} | {model_name} | {method} | {val} | {test} |".format(
                    family=row.get("family"),
                    model_name=row.get("model_name"),
                    method=row.get("method"),
                    val="" if row.get("model_val_accuracy") is None else f"{float(row['model_val_accuracy']):.6f}",
                    test="" if row.get("final_test_accuracy") is None else f"{float(row['final_test_accuracy']):.6f}",
                )
            )
    return "\n".join(lines) + "\n"


def write_hlt_mv_final_report(config: HLTMVFinalReportConfig) -> dict[str, Any]:
    """Write an aggregate final report for the HLT-MV run graph."""

    output_dir = Path(config.output_dir)
    report_path = output_dir / HLT_MV_FINAL_REPORT_JSON
    if report_path.exists() and not bool(config.overwrite):
        raise FileExistsError(f"HLT-MV final report already exists: {report_path}")

    cfg = default_hlt_mv_experiment_config(pdv3_experiment_name=config.pdv3_experiment_name)
    layout = default_hlt_mv_experiment_layout(
        output_root=config.output_root,
        pdv3_experiment_name=config.pdv3_experiment_name,
    )
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    optional_missing: list[str] = []

    _add_model_rows(
        rows,
        missing,
        family="source_model",
        names=cfg.source_model_names,
        path_for_name=layout.source_model_dir,
        allow_missing=config.allow_missing,
    )
    _add_model_rows(
        rows,
        missing,
        family="random_hlt_source_model",
        names=cfg.random_hlt_source_names,
        path_for_name=layout.random_hlt_source_dir,
        allow_missing=config.allow_missing,
    )
    _add_model_rows(
        rows,
        missing,
        family="pretrained_particle_dualview",
        names=cfg.pretrained_dualview_names,
        path_for_name=layout.pretrained_dualview_model_dir,
        report_filename=HLT_MV_PRETRAINED_DUALVIEW_REPORT,
        allow_missing=config.allow_missing,
    )
    _add_model_rows(
        rows,
        missing,
        family="scratch_particle_dualview",
        names=cfg.scratch_dualview_names,
        path_for_name=layout.scratch_dualview_model_dir,
        report_filename=HLT_MV_SCRATCH_DUALVIEW_REPORT,
        allow_missing=config.allow_missing,
    )
    _add_model_rows(
        rows,
        missing,
        family="control",
        names=cfg.control_names,
        path_for_name=layout.control_dir,
        allow_missing=config.allow_missing,
    )
    triview_missing: list[str] = []
    _add_model_rows(
        rows,
        triview_missing,
        family="triview_particle_fusion",
        names=(HLT_MV_TRIVIEW_MODEL_NAME,),
        path_for_name=layout.triview_model_dir,
        report_filename=HLT_MV_TRIVIEW_REPORT,
        allow_missing=bool(config.allow_missing or not config.require_triview),
    )
    if triview_missing:
        if bool(config.require_triview):
            missing.extend(triview_missing)
        else:
            optional_missing.extend(triview_missing)
    for fusion_name in (
        HLT_MV_FUSION_SOURCE_5VIEW,
        HLT_MV_FUSION_HLT_RANDOM_4SEED,
        HLT_MV_FUSION_PRETRAINED_DUALVIEW_4MODEL,
        HLT_MV_FUSION_SCRATCH_DUALVIEW_4MODEL,
    ):
        report_path_for_fusion = layout.logit_fusion_dir(fusion_name) / "run_report.json"
        report = _read_json(report_path_for_fusion, allow_missing=config.allow_missing)
        if report is None:
            missing.append(report_path_for_fusion.as_posix())
            continue
        rows.extend(
            _fusion_rows(
                family="logit_fusion",
                fusion_name=fusion_name,
                report_path=report_path_for_fusion,
                report=report,
            )
        )

    best_by_model_val = _best_row_by_model_val(rows)
    posthoc_best_by_final_test = _posthoc_best_row_by_final_test(rows)
    report = {
        "ok": not missing or bool(config.allow_missing),
        "contract": HLT_MV_FINAL_REPORT_CONTRACT,
        "experiment_name": HLT_MV_EXPERIMENT_NAME,
        "experiment_step": HLT_MV_FINAL_REPORT_EXPERIMENT_STEP,
        "pdv3_experiment_name": config.pdv3_experiment_name,
        "allow_missing": bool(config.allow_missing),
        "triview_required": bool(config.require_triview),
        "missing_artifacts": missing,
        "optional_missing_artifacts": optional_missing,
        "n_rows": int(len(rows)),
        "metric_rows": rows,
        "best_by_model_val": None if best_by_model_val is None else dict(best_by_model_val),
        "best_overall": None if best_by_model_val is None else dict(best_by_model_val),
        "best_overall_ranking": "model_val_cross_entropy_then_model_val_accuracy",
        "posthoc_best_by_final_test": None if posthoc_best_by_final_test is None else dict(posthoc_best_by_final_test),
        "posthoc_best_by_final_test_ranking": "final_test_accuracy_then_model_val_accuracy",
        "outputs": {
            "report_json": str(output_dir / HLT_MV_FINAL_REPORT_JSON),
            "summary_json": str(output_dir / HLT_MV_FINAL_REPORT_SUMMARY_JSON),
            "run_report_json": str(output_dir / HLT_MV_FINAL_REPORT_RUN_JSON),
            "metric_table_csv": str(output_dir / HLT_MV_FINAL_REPORT_METRIC_TABLE_CSV),
            "report_md": str(output_dir / HLT_MV_FINAL_REPORT_MD),
        },
    }
    summary = {
        "ok": report["ok"],
        "contract": HLT_MV_FINAL_REPORT_CONTRACT,
        "pdv3_experiment_name": config.pdv3_experiment_name,
        "n_rows": int(len(rows)),
        "missing_artifacts": missing,
        "optional_missing_artifacts": optional_missing,
        "triview_required": bool(config.require_triview),
        "best_by_model_val": report["best_by_model_val"],
        "best_overall": report["best_overall"],
        "best_overall_ranking": report["best_overall_ranking"],
        "posthoc_best_by_final_test": report["posthoc_best_by_final_test"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / HLT_MV_FINAL_REPORT_JSON, report)
    save_json(output_dir / HLT_MV_FINAL_REPORT_SUMMARY_JSON, summary)
    save_json(output_dir / HLT_MV_FINAL_REPORT_RUN_JSON, report)
    _write_csv(output_dir / HLT_MV_FINAL_REPORT_METRIC_TABLE_CSV, rows)
    (output_dir / HLT_MV_FINAL_REPORT_MD).write_text(_render_markdown(report), encoding="utf-8")
    return report


__all__ = [
    "HLTMVFinalReportConfig",
    "HLT_MV_FINAL_REPORT_CONTRACT",
    "HLT_MV_FINAL_REPORT_EXPERIMENT_STEP",
    "HLT_MV_FINAL_REPORT_JSON",
    "HLT_MV_FINAL_REPORT_MD",
    "HLT_MV_FINAL_REPORT_METRIC_TABLE_CSV",
    "HLT_MV_FINAL_REPORT_RUN_JSON",
    "HLT_MV_FINAL_REPORT_SUMMARY_JSON",
    "write_hlt_mv_final_report",
]
