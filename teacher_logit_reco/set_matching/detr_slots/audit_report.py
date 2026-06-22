"""Audit and final-report helpers for DETR/free-slot experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from jetclass_fresh.hlt_baseline import save_json

from teacher_logit_reco.set_matching.train import source_metadata

from .experiment import DETR_SLOT_ENCODER_ARCHITECTURES
from .five_view import DETR_SLOT_FIVE_VIEW_TAGGER_VARIANTS


DETR_SLOT_AUDIT_REPORT_STEP = "detr_free_slot_step15_audits_final_report"
DETR_SLOT_DEFAULT_REPORT_NAME = "detr_slot_final_report"
DETR_SLOT_SINGLE_VIEW_VARIANTS: tuple[str, ...] = tuple(
    f"hlt_plus_{architecture}" for architecture in DETR_SLOT_ENCODER_ARCHITECTURES
)
DETR_SLOT_FULL_FIVE_VIEW_VARIANTS: tuple[str, ...] = (
    "five_view_plain",
    "five_view_geometry",
    "five_view_no_confidence",
)


@dataclass(frozen=True)
class DetrSlotAuditReportConfig:
    """Filesystem layout and expectations for Step 15 report generation."""

    output_dir: str
    experiment_dir: str
    reconstructor_dir: str | None = None
    reconstructed_view_dir: str | None = None
    tagger_root: str | None = None
    offline_reference_dir: str | None = None
    hlt_reference_report: str | None = None
    five_view_audit_dir: str | None = None
    architectures: tuple[str, ...] = DETR_SLOT_ENCODER_ARCHITECTURES
    tagger_variants: tuple[str, ...] = DETR_SLOT_FIVE_VIEW_TAGGER_VARIANTS
    confirm_final_test: bool = False

    def __post_init__(self) -> None:
        if not self.output_dir:
            raise ValueError("output_dir is required")
        if not self.experiment_dir:
            raise ValueError("experiment_dir is required")
        if not self.architectures:
            raise ValueError("at least one architecture is required")
        if not self.tagger_variants:
            raise ValueError("at least one tagger variant is required")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _metric(payload: Mapping[str, Any] | None, path: Sequence[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _split_metrics(payload: Mapping[str, Any] | None, split: str) -> Mapping[str, Any] | None:
    """Read metrics from either the current nested or legacy top-level shape."""

    metrics = _metric(payload, ("evaluations", split, "metrics"))
    if metrics is None:
        metrics = _metric(payload, (f"{split}_metrics",))
    return metrics if isinstance(metrics, Mapping) else None


def _split_metric(payload: Mapping[str, Any] | None, split: str, path: Sequence[str]) -> Any:
    return _metric(_split_metrics(payload, split), path)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


_COMPARISON_METRIC_ORDER: tuple[tuple[str, str], ...] = (
    ("final_test_fpr_at_signal_eff_0p50", "lower"),
    ("final_test_background_rejection_at_signal_eff_0p50", "higher"),
    ("final_test_auc", "higher"),
    ("final_test_accuracy", "higher"),
    ("stack_val_fpr_at_signal_eff_0p50", "lower"),
    ("stack_val_background_rejection_at_signal_eff_0p50", "higher"),
    ("stack_val_auc", "higher"),
    ("stack_val_accuracy", "higher"),
)


def _comparison_direction(metric_name: str | None) -> str | None:
    if metric_name is None:
        return None
    for key, direction in _COMPARISON_METRIC_ORDER:
        if key == metric_name:
            return direction
    return None


def _preferred_metric_for_rows(rows: Sequence[Mapping[str, Any]]) -> str | None:
    for key, _direction in _COMPARISON_METRIC_ORDER:
        if any(_float_or_none(row.get(key)) is not None for row in rows):
            return key
    return None


def _score_details(
    row: Mapping[str, Any],
    *,
    preferred_metric: str | None = None,
) -> tuple[str | None, float | None, str | None, float | None]:
    if preferred_metric is not None:
        direction = _comparison_direction(preferred_metric)
        value = _float_or_none(row.get(preferred_metric))
        if direction is not None and value is not None:
            score = -value if direction == "lower" else value
            return preferred_metric, value, direction, score
        return preferred_metric, None, direction, None
    for key, direction in _COMPARISON_METRIC_ORDER:
        value = _float_or_none(row.get(key))
        if value is not None:
            score = -value if direction == "lower" else value
            return key, value, direction, score
    return None, None, None, None


def _score_value(
    row: Mapping[str, Any],
    *,
    preferred_metric: str | None = None,
) -> tuple[str | None, float | None]:
    metric_name, _raw_value, _direction, score = _score_details(row, preferred_metric=preferred_metric)
    return metric_name, score


def _attach_comparison_fields(row: dict[str, Any]) -> dict[str, Any]:
    metric_name, raw_value, direction, score = _score_details(row)
    row["comparison_metric"] = metric_name
    row["comparison_value"] = raw_value
    row["comparison_direction"] = direction
    row["comparison_score"] = score
    return row


def _binary_metrics_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    operating_rows: list[dict[str, Any]] = []
    for row in rows:
        name = row.get("variant") or row.get("name") or row.get("source")
        source_type = row.get("source_type", "tagger")
        for split in ("stack_val", "final_test"):
            operating_rows.append(
                {
                    "source_type": source_type,
                    "name": name,
                    "split": split,
                    "accuracy": row.get(f"{split}_accuracy"),
                    "auc": row.get(f"{split}_auc"),
                    "fpr_at_signal_eff_0p30": row.get(f"{split}_fpr_at_signal_eff_0p30"),
                    "fpr_at_signal_eff_0p50": row.get(f"{split}_fpr_at_signal_eff_0p50"),
                    "background_rejection_at_signal_eff_0p30": row.get(
                        f"{split}_background_rejection_at_signal_eff_0p30"
                    ),
                    "background_rejection_at_signal_eff_0p50": row.get(
                        f"{split}_background_rejection_at_signal_eff_0p50"
                    ),
                }
            )
    return operating_rows


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / float(len(xs))
    mean_y = sum(ys) / float(len(ys))
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom_x = math.sqrt(sum(value * value for value in dx))
    denom_y = math.sqrt(sum(value * value for value in dy))
    if denom_x == 0.0 or denom_y == 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / (denom_x * denom_y)


def _load_reconstructor_rows(
    reconstructor_dir: Path,
    architectures: Sequence[str],
    problems: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for architecture in architectures:
        report_path = reconstructor_dir / architecture / "run_report.json"
        payload = _read_json(report_path)
        if payload is None:
            problems.append(f"missing DETR reconstructor report: {report_path}")
        metrics = _metric(payload, ("best_model_val_metrics",))
        rows.append(
            {
                "architecture": architecture,
                "source_type": "detr_reconstructor",
                "exists": payload is not None,
                "report_path": str(report_path),
                "best_epoch": _metric(payload, ("best_epoch",)),
                "epochs_completed": _metric(payload, ("epochs_completed",)),
                "best_model_val_total_loss": _metric(payload, ("best_model_val_total_loss",)),
                "model_val_total": _metric(payload, ("best_model_val_metrics", "total")),
                "model_val_matched_core_loss": _metric(payload, ("best_model_val_metrics", "matched_core_loss")),
                "model_val_matched_aux_loss": _metric(payload, ("best_model_val_metrics", "matched_aux_loss")),
                "model_val_existence_loss": _metric(payload, ("best_model_val_metrics", "existence_loss")),
                "model_val_count_loss": _metric(payload, ("best_model_val_metrics", "count_loss")),
                "model_val_count_mae": _metric(payload, ("best_model_val_metrics", "metric_count_mae")),
                "model_val_existence_precision": _metric(
                    payload,
                    ("best_model_val_metrics", "metric_existence_precision"),
                ),
                "model_val_existence_recall": _metric(payload, ("best_model_val_metrics", "metric_existence_recall")),
                "model_val_matched_delta_r_p90": _metric(
                    payload,
                    ("best_model_val_metrics", "metric_matched_delta_r_p90"),
                ),
                "model_val_jet_sum_pt_relative_error_mean": _metric(
                    payload,
                    ("best_model_val_metrics", "metric_jet_sum_pt_relative_error_mean"),
                ),
                "model_val_jet_sum_energy_relative_error_mean": _metric(
                    payload,
                    ("best_model_val_metrics", "metric_jet_sum_energy_relative_error_mean"),
                ),
                "uses_aux_logits_for_bce": _metric(payload, ("uses_aux_logits_for_bce",)),
                "checkpoint": _metric(payload, ("checkpoint",)),
                "raw_best_model_val_metrics": metrics,
            }
        )
    return rows


def _load_cache_rows(
    reconstructed_view_dir: Path,
    architectures: Sequence[str],
    problems: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, str]] = []
    for architecture in architectures:
        report_path = reconstructed_view_dir / architecture / "cache_report.json"
        payload = _read_json(report_path)
        if payload is None:
            problems.append(f"missing DETR reconstructed-view cache report: {report_path}")
        summary_path = reconstructed_view_dir / architecture / "cache_summary.csv"
        for row in _read_csv_rows(summary_path):
            row = dict(row)
            row.setdefault("architecture", architecture)
            csv_rows.append(row)
        split_reports = _metric(payload, ("split_reports",)) if payload is not None else None
        for split in ("stack_train", "stack_val", "final_test"):
            split_payload = split_reports.get(split) if isinstance(split_reports, Mapping) else None
            rows.append(
                {
                    "architecture": architecture,
                    "source_type": "detr_cache",
                    "split": split,
                    "exists": payload is not None and isinstance(split_payload, Mapping),
                    "report_path": str(report_path),
                    "array_path": _metric(split_payload, ("array_path",)),
                    "n_jets": _metric(split_payload, ("n_jets",)),
                    "n_candidates": _metric(split_payload, ("n_candidates",)),
                    "candidate_count_mean": _metric(split_payload, ("candidate_count_summary", "mean")),
                    "exported_tokens_mean": _metric(split_payload, ("exported_tokens_summary", "mean")),
                    "top_existence_score_mean": _metric(split_payload, ("top_existence_score_mean",)),
                    "nonfinite_count": _metric(split_payload, ("nonfinite_count",)),
                    "heldout_total_loss": _metric(split_payload, ("heldout_detr_slot_metrics", "total")),
                    "heldout_matched_core_loss": _metric(
                        split_payload,
                        ("heldout_detr_slot_metrics", "matched_core_loss"),
                    ),
                    "heldout_matched_aux_loss": _metric(
                        split_payload,
                        ("heldout_detr_slot_metrics", "matched_aux_loss"),
                    ),
                    "heldout_count_mae": _metric(split_payload, ("heldout_detr_slot_metrics", "metric_count_mae")),
                    "heldout_existence_precision": _metric(
                        split_payload,
                        ("heldout_detr_slot_metrics", "metric_existence_precision"),
                    ),
                    "heldout_existence_recall": _metric(
                        split_payload,
                        ("heldout_detr_slot_metrics", "metric_existence_recall"),
                    ),
                    "heldout_matched_delta_r_p90": _metric(
                        split_payload,
                        ("heldout_detr_slot_metrics", "metric_matched_delta_r_p90"),
                    ),
                }
            )
    return rows, csv_rows


def _load_tagger_rows(tagger_root: Path, variants: Sequence[str], problems: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        report_path = tagger_root / variant / "run_report.json"
        payload = _read_json(report_path)
        if payload is None:
            problems.append(f"missing DETR five-view tagger report: {report_path}")
        elif _metric(payload, ("final_test_metrics",)) is None:
            problems.append(f"missing final_test metrics in tagger report: {report_path}")
        row = {
            "variant": variant,
            "source_type": "detr_tagger",
            "exists": payload is not None,
            "report_path": str(report_path),
            "best_epoch": _metric(payload, ("best_epoch",)),
            "selection_metric": _metric(payload, ("selection_metric",)),
            "selection_metric_direction": _metric(payload, ("selection_metric_direction",)),
            "best_model_selection_metric_value": _metric(payload, ("best_model_selection_metric_value",)),
            "stack_val_accuracy": _metric(payload, ("best_stack_val_metrics", "accuracy")),
            "stack_val_auc": _metric(payload, ("best_stack_val_metrics", "binary_metrics", "auc")),
            "stack_val_fpr_at_signal_eff_0p30": _metric(
                payload,
                ("best_stack_val_metrics", "binary_metrics", "fpr_at_signal_eff_0p30"),
            ),
            "stack_val_fpr_at_signal_eff_0p50": _metric(
                payload,
                ("best_stack_val_metrics", "binary_metrics", "fpr_at_signal_eff_0p50"),
            ),
            "stack_val_background_rejection_at_signal_eff_0p30": _metric(
                payload,
                ("best_stack_val_metrics", "binary_metrics", "background_rejection_at_signal_eff_0p30"),
            ),
            "stack_val_background_rejection_at_signal_eff_0p50": _metric(
                payload,
                ("best_stack_val_metrics", "binary_metrics", "background_rejection_at_signal_eff_0p50"),
            ),
            "final_test_accuracy": _metric(payload, ("final_test_metrics", "accuracy")),
            "final_test_auc": _metric(payload, ("final_test_metrics", "binary_metrics", "auc")),
            "final_test_fpr_at_signal_eff_0p30": _metric(
                payload,
                ("final_test_metrics", "binary_metrics", "fpr_at_signal_eff_0p30"),
            ),
            "final_test_fpr_at_signal_eff_0p50": _metric(
                payload,
                ("final_test_metrics", "binary_metrics", "fpr_at_signal_eff_0p50"),
            ),
            "final_test_background_rejection_at_signal_eff_0p30": _metric(
                payload,
                ("final_test_metrics", "binary_metrics", "background_rejection_at_signal_eff_0p30"),
            ),
            "final_test_background_rejection_at_signal_eff_0p50": _metric(
                payload,
                ("final_test_metrics", "binary_metrics", "background_rejection_at_signal_eff_0p50"),
            ),
            "final_test_evaluated": bool(_metric(payload, ("final_test_evaluated",))),
            "drop_views": _metric(payload, ("config", "drop_views")),
            "use_confidence": _metric(payload, ("config", "use_confidence")),
            "use_geometry_attention": _metric(payload, ("config", "use_geometry_attention")),
            "shuffle_view_labels": _metric(payload, ("config", "shuffle_view_labels")),
        }
        rows.append(_attach_comparison_fields(row))
    return rows


def _load_offline_reference_rows(reference_dir: Path | None, problems: list[str]) -> list[dict[str, Any]]:
    if reference_dir is None:
        return []
    if not reference_dir.exists():
        problems.append(f"explicit offline reference directory does not exist: {reference_dir}")
        return []
    rows: list[dict[str, Any]] = []
    for report_path in sorted(reference_dir.glob("*/run_report.json")) + sorted(reference_dir.glob("run_report.json")):
        payload = _read_json(report_path)
        if payload is None:
            continue
        row = {
            "name": report_path.parent.name,
            "source_type": "offline_reference",
            "report_path": str(report_path),
            "best_epoch": _metric(payload, ("best_epoch",)),
            "model_val_accuracy": (
                _metric(payload, ("model_val_metrics", "accuracy"))
                if _metric(payload, ("model_val_metrics", "accuracy")) is not None
                else _metric(payload, ("best_model_val_accuracy",))
            ),
            "stack_val_accuracy": _split_metric(payload, "stack_val", ("accuracy",)),
            "stack_val_auc": _split_metric(payload, "stack_val", ("binary_metrics", "auc")),
            "stack_val_fpr_at_signal_eff_0p30": _metric(
                _split_metrics(payload, "stack_val"),
                ("binary_metrics", "fpr_at_signal_eff_0p30"),
            ),
            "stack_val_fpr_at_signal_eff_0p50": _metric(
                _split_metrics(payload, "stack_val"),
                ("binary_metrics", "fpr_at_signal_eff_0p50"),
            ),
            "stack_val_background_rejection_at_signal_eff_0p30": _metric(
                _split_metrics(payload, "stack_val"),
                ("binary_metrics", "background_rejection_at_signal_eff_0p30"),
            ),
            "stack_val_background_rejection_at_signal_eff_0p50": _metric(
                _split_metrics(payload, "stack_val"),
                ("binary_metrics", "background_rejection_at_signal_eff_0p50"),
            ),
            "final_test_accuracy": _split_metric(payload, "final_test", ("accuracy",)),
            "final_test_auc": _split_metric(payload, "final_test", ("binary_metrics", "auc")),
            "final_test_fpr_at_signal_eff_0p30": _metric(
                _split_metrics(payload, "final_test"),
                ("binary_metrics", "fpr_at_signal_eff_0p30"),
            ),
            "final_test_fpr_at_signal_eff_0p50": _metric(
                _split_metrics(payload, "final_test"),
                ("binary_metrics", "fpr_at_signal_eff_0p50"),
            ),
            "final_test_background_rejection_at_signal_eff_0p30": _metric(
                _split_metrics(payload, "final_test"),
                ("binary_metrics", "background_rejection_at_signal_eff_0p30"),
            ),
            "final_test_background_rejection_at_signal_eff_0p50": _metric(
                _split_metrics(payload, "final_test"),
                ("binary_metrics", "background_rejection_at_signal_eff_0p50"),
            ),
        }
        rows.append(_attach_comparison_fields(row))
    return rows


def _load_hlt_reference_row(path: Path | None, problems: list[str]) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _read_json(path)
    if payload is None:
        problems.append(f"explicit HLT reference report does not exist: {path}")
        return None
    row = {
        "name": path.parent.name,
        "source_type": "hlt_reference",
        "report_path": str(path),
        "stack_val_accuracy": _split_metric(payload, "stack_val", ("accuracy",)),
        "stack_val_auc": _split_metric(payload, "stack_val", ("binary_metrics", "auc")),
        "stack_val_fpr_at_signal_eff_0p50": _split_metric(
            payload,
            "stack_val",
            ("binary_metrics", "fpr_at_signal_eff_0p50"),
        ),
        "stack_val_background_rejection_at_signal_eff_0p50": _split_metric(
            payload,
            "stack_val",
            ("binary_metrics", "background_rejection_at_signal_eff_0p50"),
        ),
        "final_test_accuracy": _split_metric(payload, "final_test", ("accuracy",)),
        "final_test_auc": _split_metric(payload, "final_test", ("binary_metrics", "auc")),
        "final_test_fpr_at_signal_eff_0p50": _split_metric(
            payload,
            "final_test",
            ("binary_metrics", "fpr_at_signal_eff_0p50"),
        ),
        "final_test_background_rejection_at_signal_eff_0p50": _split_metric(
            payload,
            "final_test",
            ("binary_metrics", "background_rejection_at_signal_eff_0p50"),
        ),
    }
    return _attach_comparison_fields(row)


def _load_five_view_audit_summary(audit_dir: Path | None, problems: list[str]) -> dict[str, Any] | None:
    if audit_dir is None:
        return None
    run_report_path = audit_dir / "run_report.json"
    summary_csv_path = audit_dir / "summary.csv"
    run_report = _read_json(run_report_path)
    summary_rows = _read_csv_rows(summary_csv_path)
    if run_report is None and not summary_rows:
        if problems is not None:
            problems.append(f"explicit five-view audit artifacts are missing under: {audit_dir}")
        return None
    return {
        "exists": True,
        "audit_dir": str(audit_dir),
        "run_report_path": str(run_report_path),
        "summary_csv_path": str(summary_csv_path),
        "summary_rows": summary_rows,
        "run_report": run_report,
        "ok": _metric(run_report, ("ok",)),
        "n_summary_rows": len(summary_rows),
        "skipped_specs": (
            _metric(run_report, ("skipped_specs",))
            if _metric(run_report, ("skipped_specs",)) is not None
            else _metric(run_report, ("skipped",))
        ),
        "source": "generic_five_view_ablation_audit",
    }


def _best_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    preferred_metric: str | None = None,
) -> Mapping[str, Any] | None:
    scored = [(row, _score_value(row, preferred_metric=preferred_metric)[1]) for row in rows]
    scored = [(row, score) for row, score in scored if score is not None]
    if not scored:
        return None
    return max(scored, key=lambda item: float(item[1]))[0]


def _build_comparison_summary(
    taggers: Sequence[Mapping[str, Any]],
    offline_references: Sequence[Mapping[str, Any]],
    hlt_reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    by_variant = {str(row.get("variant")): row for row in taggers}
    hlt_only = by_variant.get("hlt_only")
    single_view_rows = [by_variant[variant] for variant in DETR_SLOT_SINGLE_VIEW_VARIANTS if variant in by_variant]
    five_view_rows = [by_variant[variant] for variant in DETR_SLOT_FULL_FIVE_VIEW_VARIANTS if variant in by_variant]
    non_hlt_rows = [
        row
        for row in taggers
        if row.get("variant") != "hlt_only" and row.get("variant") != "view_label_shuffle_control"
    ]

    real_tagger_rows = [row for row in taggers if row.get("variant") != "view_label_shuffle_control"]
    metric_name = _preferred_metric_for_rows(real_tagger_rows)
    metric_name, hlt_value, metric_direction, hlt_score = _score_details(
        hlt_only or {},
        preferred_metric=metric_name,
    )
    if metric_name is None:
        metric_name, hlt_value, metric_direction, hlt_score = _score_details(hlt_only or {})
    best_non_hlt = _best_row(non_hlt_rows, preferred_metric=metric_name)
    best_single = _best_row(single_view_rows, preferred_metric=metric_name)
    best_five = _best_row(five_view_rows, preferred_metric=metric_name)
    best_overall = _best_row(real_tagger_rows, preferred_metric=metric_name)
    best_metric_name, _best_raw, best_metric_direction, _best_score = _score_details(
        best_overall or {},
        preferred_metric=metric_name,
    )

    single_deltas = []
    for row in single_view_rows:
        row_metric, raw_value, direction, score = _score_details(row, preferred_metric=metric_name)
        single_deltas.append(
            {
                "variant": row.get("variant"),
                "architecture": str(row.get("variant", "")).removeprefix("hlt_plus_"),
                "comparison_metric": row_metric,
                "comparison_value": raw_value,
                "comparison_direction": direction,
                "score": score,
                "delta_vs_hlt_only": None if hlt_score is None or score is None else float(score) - float(hlt_score),
            }
        )

    _, best_single_score = _score_value(best_single or {}, preferred_metric=metric_name)
    _, best_five_score = _score_value(best_five or {}, preferred_metric=metric_name)
    _, best_non_hlt_score = _score_value(best_non_hlt or {}, preferred_metric=metric_name)
    _, best_overall_score = _score_value(best_overall or {}, preferred_metric=metric_name)
    _, hlt_reference_score = _score_value(hlt_reference or {}, preferred_metric=metric_name)
    best_offline = _best_row(offline_references, preferred_metric=metric_name)
    _, best_offline_score = _score_value(best_offline or {}, preferred_metric=metric_name)

    return {
        "comparison_metric": metric_name or best_metric_name or (best_overall or {}).get("comparison_metric"),
        "comparison_direction": metric_direction or best_metric_direction or (best_overall or {}).get("comparison_direction"),
        "comparison_value_note": "For lower-is-better metrics such as FPR, comparison_score is the negative raw value.",
        "hlt_only": hlt_only,
        "best_non_hlt": best_non_hlt,
        "best_single_view": best_single,
        "best_five_view": best_five,
        "best_overall_tagger": best_overall,
        "best_offline_reference": best_offline,
        "external_hlt_reference": hlt_reference,
        "single_view_deltas_vs_hlt_only": single_deltas,
        "free_slot_improved_over_hlt_only": (
            None if hlt_score is None or best_non_hlt_score is None else best_non_hlt_score > hlt_score
        ),
        "best_non_hlt_delta_vs_hlt_only": (
            None if hlt_score is None or best_non_hlt_score is None else best_non_hlt_score - hlt_score
        ),
        "five_view_beat_every_single_view": (
            None if best_single_score is None or best_five_score is None else best_five_score > best_single_score
        ),
        "best_five_view_delta_vs_best_single_view": (
            None if best_single_score is None or best_five_score is None else best_five_score - best_single_score
        ),
        "best_overall_score": best_overall_score,
        "best_offline_reference_score": best_offline_score,
        "external_hlt_reference_score": hlt_reference_score,
        "hlt_only_comparison_value": hlt_value,
        "answered_questions": {
            "did_free_slot_reconstruction_improve_over_hlt_only": (
                None if hlt_score is None or best_non_hlt_score is None else best_non_hlt_score > hlt_score
            ),
            "which_encoder_helped_most": (best_single or {}).get("variant"),
            "did_five_view_beat_every_single_view": (
                None if best_single_score is None or best_five_score is None else best_five_score > best_single_score
            ),
        },
    }


def _build_correlation_summary(
    reconstructors: Sequence[Mapping[str, Any]],
    caches: Sequence[Mapping[str, Any]],
    taggers: Sequence[Mapping[str, Any]],
    architectures: Sequence[str],
) -> dict[str, Any]:
    reco_by_arch = {str(row.get("architecture")): row for row in reconstructors}
    final_cache_by_arch = {
        str(row.get("architecture")): row for row in caches if str(row.get("split")) == "final_test"
    }
    tagger_by_arch = {
        str(row.get("variant", "")).removeprefix("hlt_plus_"): row
        for row in taggers
        if str(row.get("variant", "")).startswith("hlt_plus_")
    }
    joined_rows: list[dict[str, Any]] = []
    for architecture in architectures:
        tagger = tagger_by_arch.get(architecture, {})
        reco = reco_by_arch.get(architecture, {})
        cache = final_cache_by_arch.get(architecture, {})
        joined_rows.append(
            {
                "architecture": architecture,
                "tagger_final_test_accuracy": tagger.get("final_test_accuracy"),
                "tagger_final_test_auc": tagger.get("final_test_auc"),
                "tagger_final_test_fpr_at_signal_eff_0p30": tagger.get("final_test_fpr_at_signal_eff_0p30"),
                "tagger_final_test_fpr_at_signal_eff_0p50": tagger.get("final_test_fpr_at_signal_eff_0p50"),
                "tagger_comparison_metric": tagger.get("comparison_metric"),
                "tagger_comparison_direction": tagger.get("comparison_direction"),
                "tagger_comparison_value": tagger.get("comparison_value"),
                "tagger_comparison_score": tagger.get("comparison_score"),
                "reco_model_val_total": reco.get("model_val_total") or reco.get("best_model_val_total_loss"),
                "reco_model_val_matched_core_loss": reco.get("model_val_matched_core_loss"),
                "reco_model_val_matched_aux_loss": reco.get("model_val_matched_aux_loss"),
                "reco_model_val_count_mae": reco.get("model_val_count_mae"),
                "cache_final_heldout_total_loss": cache.get("heldout_total_loss"),
                "cache_final_heldout_matched_core_loss": cache.get("heldout_matched_core_loss"),
                "cache_final_heldout_count_mae": cache.get("heldout_count_mae"),
                "cache_final_exported_tokens_mean": cache.get("exported_tokens_mean"),
            }
        )

    y_key = "tagger_comparison_score"
    if sum(_float_or_none(row.get(y_key)) is not None for row in joined_rows) < 2:
        y_key = "tagger_final_test_auc"
    if sum(_float_or_none(row.get(y_key)) is not None for row in joined_rows) < 2:
        y_key = "tagger_final_test_accuracy"
    correlations = []
    for x_key in (
        "reco_model_val_total",
        "reco_model_val_matched_core_loss",
        "reco_model_val_matched_aux_loss",
        "reco_model_val_count_mae",
        "cache_final_heldout_total_loss",
        "cache_final_heldout_matched_core_loss",
        "cache_final_heldout_count_mae",
        "cache_final_exported_tokens_mean",
    ):
        pairs = [
            (_float_or_none(row.get(x_key)), _float_or_none(row.get(y_key)))
            for row in joined_rows
        ]
        xs = [float(x) for x, y in pairs if x is not None and y is not None]
        ys = [float(y) for x, y in pairs if x is not None and y is not None]
        correlations.append(
            {
                "x_metric": x_key,
                "y_metric": y_key,
                "n": len(xs),
                "pearson_r": _pearson(xs, ys),
                "note": "negative r means lower reconstruction loss is associated with higher tagger score"
                if "loss" in x_key or "mae" in x_key
                else None,
            }
        )
    return {"joined_rows": joined_rows, "correlations": correlations}


def render_detr_slot_final_report_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact Markdown summary for Step 15."""

    comparisons = report.get("comparison_summary", {})
    answered = comparisons.get("answered_questions", {}) if isinstance(comparisons, Mapping) else {}
    lines = [
        "# DETR Free-Slot Final Report",
        "",
        f"ok: {report.get('ok')}",
        f"experiment_dir: {report.get('experiment_dir')}",
        "",
        "## Direct Answers",
        "",
        f"- Did free-slot reconstruction improve over HLT-only? {answered.get('did_free_slot_reconstruction_improve_over_hlt_only')}",
        f"- Which single encoder helped most? {answered.get('which_encoder_helped_most')}",
        f"- Did five-view beat every single view? {answered.get('did_five_view_beat_every_single_view')}",
        f"- Comparison metric: {comparisons.get('comparison_metric') if isinstance(comparisons, Mapping) else None} "
        f"({comparisons.get('comparison_direction') if isinstance(comparisons, Mapping) else None}-is-better)",
        f"- Best non-HLT delta vs HLT-only: {comparisons.get('best_non_hlt_delta_vs_hlt_only') if isinstance(comparisons, Mapping) else None}",
        f"- Best five-view delta vs best single-view: {comparisons.get('best_five_view_delta_vs_best_single_view') if isinstance(comparisons, Mapping) else None}",
        "",
        "## Tagger Summary",
        "",
        "| variant | metric | value | score | stack_val_auc | final_test_auc | final_test_accuracy | fpr@30 | fpr@50 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("tagger_summary", []):
        lines.append(
            f"| {row.get('variant')} | {row.get('comparison_metric')} | {row.get('comparison_value')} | "
            f"{row.get('comparison_score')} | {row.get('stack_val_auc')} | "
            f"{row.get('final_test_auc')} | {row.get('final_test_accuracy')} | "
            f"{row.get('final_test_fpr_at_signal_eff_0p30')} | {row.get('final_test_fpr_at_signal_eff_0p50')} |"
        )

    lines += [
        "",
        "## Reconstructor Summary",
        "",
        "| arch | best_epoch | total_loss | core_loss | aux_loss | count_mae | deltaR_p90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("reconstructor_summary", []):
        lines.append(
            f"| {row.get('architecture')} | {row.get('best_epoch')} | {row.get('model_val_total')} | "
            f"{row.get('model_val_matched_core_loss')} | {row.get('model_val_matched_aux_loss')} | "
            f"{row.get('model_val_count_mae')} | {row.get('model_val_matched_delta_r_p90')} |"
        )

    operating_rows = report.get("best_binary_operating_point_table", [])
    if operating_rows:
        lines += [
            "",
            "## Binary Operating Points",
            "",
            "| source | split | auc | fpr@30 | fpr@50 | rej@30 | rej@50 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in operating_rows:
            lines.append(
                f"| {row.get('name')} | {row.get('split')} | {row.get('auc')} | "
                f"{row.get('fpr_at_signal_eff_0p30')} | {row.get('fpr_at_signal_eff_0p50')} | "
                f"{row.get('background_rejection_at_signal_eff_0p30')} | "
                f"{row.get('background_rejection_at_signal_eff_0p50')} |"
            )

    problems = report.get("problems", [])
    if problems:
        lines += ["", "## Problems", ""]
        lines.extend(f"- {problem}" for problem in problems)
    return "\n".join(lines) + "\n"


def build_detr_slot_audit_final_report(config: DetrSlotAuditReportConfig) -> dict[str, Any]:
    """Read DETR run artifacts and write Step 15 final report outputs."""

    experiment_dir = Path(config.experiment_dir)
    output_dir = Path(config.output_dir)
    reconstructor_dir = Path(config.reconstructor_dir) if config.reconstructor_dir else experiment_dir / "reconstructors"
    reconstructed_view_dir = (
        Path(config.reconstructed_view_dir) if config.reconstructed_view_dir else experiment_dir / "reconstructed_views"
    )
    tagger_root = Path(config.tagger_root) if config.tagger_root else experiment_dir / "taggers"
    offline_reference_dir = (
        Path(config.offline_reference_dir)
        if config.offline_reference_dir
        else experiment_dir / "offline_teacher_reference"
    )
    hlt_reference_report = Path(config.hlt_reference_report) if config.hlt_reference_report else None
    five_view_audit_dir = (
        Path(config.five_view_audit_dir)
        if config.five_view_audit_dir
        else experiment_dir / "ablations" / "five_view_ablation_eval"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    reconstructors = _load_reconstructor_rows(reconstructor_dir, config.architectures, problems)
    caches, cache_csv_rows = _load_cache_rows(reconstructed_view_dir, config.architectures, problems)
    taggers = _load_tagger_rows(tagger_root, config.tagger_variants, problems)
    offline_references = _load_offline_reference_rows(offline_reference_dir, problems if config.offline_reference_dir else [])
    hlt_reference = _load_hlt_reference_row(hlt_reference_report, problems)
    five_view_audit = _load_five_view_audit_summary(
        five_view_audit_dir,
        problems if config.five_view_audit_dir else [],
    )

    if bool(config.confirm_final_test):
        for row in taggers:
            if not bool(row.get("final_test_evaluated")):
                problems.append(f"tagger did not mark final_test_evaluated=True: {row.get('variant')}")

    comparison_summary = _build_comparison_summary(taggers, offline_references, hlt_reference)
    correlation_summary = _build_correlation_summary(reconstructors, caches, taggers, config.architectures)
    operating_point_rows = _binary_metrics_rows([*taggers, *offline_references, *([hlt_reference] if hlt_reference else [])])

    csv_paths = {
        "reconstructor_summary": str(output_dir / "reconstructor_summary.csv"),
        "cache_export_summary": str(output_dir / "cache_export_summary.csv"),
        "cache_summary_raw": str(output_dir / "cache_summary_raw.csv"),
        "tagger_summary": str(output_dir / "tagger_summary.csv"),
        "binary_operating_points": str(output_dir / "binary_operating_points.csv"),
        "reco_tagger_correlation_rows": str(output_dir / "reco_tagger_correlation_rows.csv"),
        "reco_tagger_correlations": str(output_dir / "reco_tagger_correlations.csv"),
        "five_view_audit_summary": str(output_dir / "five_view_audit_summary.csv"),
    }
    _write_csv(Path(csv_paths["reconstructor_summary"]), reconstructors)
    _write_csv(Path(csv_paths["cache_export_summary"]), caches)
    _write_csv(Path(csv_paths["cache_summary_raw"]), cache_csv_rows)
    _write_csv(Path(csv_paths["tagger_summary"]), taggers)
    _write_csv(Path(csv_paths["binary_operating_points"]), operating_point_rows)
    _write_csv(Path(csv_paths["reco_tagger_correlation_rows"]), correlation_summary["joined_rows"])
    _write_csv(Path(csv_paths["reco_tagger_correlations"]), correlation_summary["correlations"])
    _write_csv(
        Path(csv_paths["five_view_audit_summary"]),
        five_view_audit.get("summary_rows", []) if isinstance(five_view_audit, Mapping) else [],
    )

    report = {
        "ok": len(problems) == 0,
        "experiment_step": DETR_SLOT_AUDIT_REPORT_STEP,
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "confirm_final_test": bool(config.confirm_final_test),
        "config": asdict(config),
        "problems": problems,
        "paths": {
            "reconstructor_dir": str(reconstructor_dir),
            "reconstructed_view_dir": str(reconstructed_view_dir),
            "tagger_root": str(tagger_root),
            "offline_reference_dir": str(offline_reference_dir),
            "hlt_reference_report": str(hlt_reference_report) if hlt_reference_report else None,
            "five_view_audit_dir": str(five_view_audit_dir),
            "csv": csv_paths,
        },
        "reconstructor_summary": reconstructors,
        "cache_export_summary": caches,
        "cache_summary_raw_rows": cache_csv_rows,
        "tagger_summary": taggers,
        "offline_reference_comparison": offline_references,
        "hlt_only_comparison": comparison_summary.get("hlt_only"),
        "external_hlt_reference_comparison": hlt_reference,
        "five_view_audit_summary": five_view_audit,
        "best_binary_operating_point_table": operating_point_rows,
        "comparison_summary": comparison_summary,
        "reconstruction_tagging_correlation": correlation_summary,
        "source": source_metadata(),
    }
    json_path = output_dir / f"{DETR_SLOT_DEFAULT_REPORT_NAME}.json"
    md_path = output_dir / f"{DETR_SLOT_DEFAULT_REPORT_NAME}.md"
    save_json(json_path, report)
    md_path.write_text(render_detr_slot_final_report_markdown(report), encoding="utf-8")
    report["report_json"] = str(json_path)
    report["report_markdown"] = str(md_path)
    save_json(json_path, report)
    return report


__all__ = [
    "DETR_SLOT_AUDIT_REPORT_STEP",
    "DETR_SLOT_DEFAULT_REPORT_NAME",
    "DETR_SLOT_FULL_FIVE_VIEW_VARIANTS",
    "DETR_SLOT_SINGLE_VIEW_VARIANTS",
    "DetrSlotAuditReportConfig",
    "build_detr_slot_audit_final_report",
    "render_detr_slot_final_report_markdown",
]
