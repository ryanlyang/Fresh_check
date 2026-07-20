"""Final campaign reporting for local particle residual-field runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .curriculum import load_selected_consumer_record
from .curriculum_report import (
    A0_LABEL,
    O0_LABEL,
    ALPHA_DIAGNOSTIC_IDS,
    RUNTIME_COLUMNS,
    baseline_columns,
    classify_result_row,
    consumer_selection_rows,
    curriculum_student_rows,
    curriculum_training_rows,
    load_diagnostic_rows,
    oracle_curve_rows,
    paired_tables,
    pilot_base_run_id,
    scan_curriculum_reports,
    validate_stage1b_reports,
)


LOCAL_RESIDUAL_FIELD_REPORT_CONTRACT = "local_particle_residual_field_report_v1"
LOCAL_RESIDUAL_FIELD_REPORT_STEP = "local_particle_residual_field_step9_final_report"

DEFAULT_REQUIRED_TAGGER_RUN_IDS: tuple[str, ...] = (
    "A0",
    "A1",
    "A2",
    "B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "D0",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "E0",
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "E6",
    "F0",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
)
DEFAULT_REQUIRED_RECONSTRUCTOR_RUN_IDS: tuple[str, ...] = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
DEFAULT_REQUIRED_FUSION_GROUPS: tuple[str, ...] = ("G0", "G1", "G2", "G3")

REPORT_SPLITS: tuple[str, ...] = ("model_val", "stack_val", "final_test")
ORACLE_RUN_IDS: tuple[str, ...] = ("B0", "B1", "B2", "B3", "B4")
CONTROL_RUN_IDS: tuple[str, ...] = ("F0", "F1", "F2", "F3", "F4", "F5")
FIELD_IMPORTANCE_RUN_IDS: tuple[str, ...] = ("E0", "E1", "E2", "E3", "E4", "E5", "E6")
MIN_TAGGER_METRIC_VALID_FRACTION = 0.99


@dataclass(frozen=True)
class LocalResidualFieldReportConfig:
    """Inputs for the final local residual-field campaign report."""

    output_dir: str
    tagger_root: str
    reconstructor_root: str
    fusion_dir: str | None = None
    prediction_dir: str | None = None
    target_cache_dir: str | None = None
    curriculum_root: str | None = None
    oracle_diagnostics_root: str | None = None
    curriculum_diagnostics_root: str | None = None
    selected_consumer_json: str | None = None
    required_tagger_run_ids: tuple[str, ...] = DEFAULT_REQUIRED_TAGGER_RUN_IDS
    required_reconstructor_run_ids: tuple[str, ...] = DEFAULT_REQUIRED_RECONSTRUCTOR_RUN_IDS
    required_fusion_groups: tuple[str, ...] = DEFAULT_REQUIRED_FUSION_GROUPS
    required_curriculum_run_ids: tuple[str, ...] = field(default_factory=tuple)
    require_fusion: bool = False
    allow_missing_runs: bool = False
    confirm_final_test: bool = False
    require_final_test_provenance: bool = False
    require_curriculum: bool = False
    paired_consumer_mode: bool = False
    summary_title: str = "Local Particle Residual Field Report"

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_tagger_run_ids", _clean_ids(self.required_tagger_run_ids))
        object.__setattr__(self, "required_reconstructor_run_ids", _clean_ids(self.required_reconstructor_run_ids))
        object.__setattr__(self, "required_fusion_groups", _clean_ids(self.required_fusion_groups))
        object.__setattr__(self, "required_curriculum_run_ids", _clean_ids(self.required_curriculum_run_ids))


def _clean_ids(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(value).strip() for value in values if str(value).strip())


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _maybe_load_json(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    return _load_json(path)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return "" if not math.isfinite(value) else value
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_jsonable(value), sort_keys=True)
    return str(value)


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered: list[str] = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(str(key))
    if not ordered:
        ordered = ["run_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in ordered})


def _path_value(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _first_path_value(payload: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    for path in paths:
        value = _path_value(payload, path)
        if value is not None:
            return value
    return None


def _stable_value(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_jsonable(value), sort_keys=True)
    return str(value)


def _float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _scan_run_reports(root: str | Path | None) -> dict[str, dict[str, Any]]:
    if not root:
        return {}
    root_path = Path(root)
    reports: dict[str, dict[str, Any]] = {}
    if not root_path.exists():
        return reports
    for path in sorted(root_path.glob("*/run_report.json")):
        reports[path.parent.name] = _load_json(path)
        reports[path.parent.name]["_report_path"] = str(path)
    return reports


def _load_required_reports(
    root: str | Path,
    *,
    required_ids: Sequence[str],
    family: str,
    problems: list[str],
    allow_missing_runs: bool,
) -> dict[str, dict[str, Any]]:
    reports = _scan_run_reports(root)
    for run_id in required_ids:
        if run_id not in reports:
            message = f"missing required {family} run_report for {run_id}"
            if allow_missing_runs:
                problems.append(f"allowed_missing: {message}")
            else:
                problems.append(message)
    for run_id, report in reports.items():
        if report.get("ok") is not True:
            problems.append(f"{family} run {run_id} is not ok: ok={report.get('ok')!r}")
    return reports


def _split_metrics(report: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    candidates: list[Any]
    if split == "model_val":
        candidates = [
            report.get("best_model_val"),
            report.get("best_model_val_metrics"),
            _path_value(report, ("metrics", "model_val")),
        ]
    elif split == "stack_val":
        candidates = [
            report.get("stack_val"),
            report.get("stack_val_metrics"),
            _path_value(report, ("metrics", "stack_val")),
        ]
    elif split == "final_test":
        candidates = [
            report.get("final_test"),
            report.get("final_test_metrics"),
            _path_value(report, ("metrics", "final_test")),
        ]
    else:
        candidates = [report.get(split)]
    for value in candidates:
        if isinstance(value, Mapping):
            if value.get("available") is False:
                return None
            return value
    return None


def _metric_value(metrics: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def _int_value(value: Any) -> int | None:
    try:
        output = int(value)
    except (TypeError, ValueError):
        return None
    return output if output >= 0 else None


def _expected_n_jets(report: Mapping[str, Any], metrics: Mapping[str, Any], split: str) -> int | None:
    for value in (
        _path_value(report, ("dataset_metadata", split, "n_jets")),
        _path_value(report, ("dataset_metadata", split, "selected_n_jets")),
        metrics.get("selection_expected_n_jets"),
        metrics.get("attempted_jets"),
    ):
        output = _int_value(value)
        if output is not None and output > 0:
            return output
    return None


def _tagger_metric_coverage_problem(
    *,
    run_id: str,
    report: Mapping[str, Any],
    split: str,
    metrics: Mapping[str, Any],
) -> str | None:
    expected = _expected_n_jets(report, metrics, split)
    if expected is None:
        return f"tagger {run_id} {split} metrics are missing expected jet-count provenance"
    seen = _int_value(metrics.get("n_jets"))
    if seen is None:
        return f"tagger {run_id} {split} metrics are missing n_jets"
    min_seen = int(math.ceil(float(expected) * MIN_TAGGER_METRIC_VALID_FRACTION))
    if seen < min_seen:
        return (
            f"tagger {run_id} {split} finite metric coverage {seen}/{expected} "
            f"is below required {min_seen} ({MIN_TAGGER_METRIC_VALID_FRACTION:.4f})"
        )
    if metrics.get("valid_for_selection") is False:
        reason = metrics.get("selection_rejection_reason") or "valid_for_selection=false"
        return f"tagger {run_id} {split} metrics were rejected during training: {reason}"
    if _float(metrics.get("loss")) is None and _float(metrics.get("cross_entropy")) is None:
        return f"tagger {run_id} {split} loss/cross_entropy is missing or nonfinite"
    if _float(metrics.get("accuracy")) is None:
        return f"tagger {run_id} {split} accuracy is missing or nonfinite"
    return None


def _prediction_metadata_path(prediction_dir: str | Path | None, run_id: str, split: str) -> Path | None:
    if not prediction_dir:
        return None
    return Path(prediction_dir) / str(run_id) / f"{split}_predictions_metadata.json"


def _prediction_metadata(prediction_dir: str | Path | None, run_id: str, split: str) -> Mapping[str, Any] | None:
    path = _prediction_metadata_path(prediction_dir, run_id, split)
    if path is None or not path.exists():
        return None
    metadata = _load_json(path)
    return metadata


def _prediction_final_test_metrics(prediction_dir: str | Path | None, run_id: str) -> Mapping[str, Any] | None:
    metadata = _prediction_metadata(prediction_dir, run_id, "final_test")
    if not isinstance(metadata, Mapping):
        return None
    metrics = metadata.get("metrics")
    return metrics if isinstance(metrics, Mapping) else None


def _with_prediction_dataset_metadata(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    prediction_dir: str | Path | None,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for run_id, report in reports.items():
        merged = dict(report)
        dataset_metadata = dict(report.get("dataset_metadata") or {})
        for split in ("stack_val", "final_test"):
            metadata = _prediction_metadata(prediction_dir, run_id, split)
            if isinstance(metadata, Mapping) and isinstance(metadata.get("dataset_metadata"), Mapping):
                prediction_dataset = metadata["dataset_metadata"]
                if split != "final_test" and prediction_dataset.get("target_fields_present") is False:
                    continue
                dataset_metadata[split] = prediction_dataset
        if dataset_metadata:
            merged["dataset_metadata"] = dataset_metadata
        output[run_id] = merged
    return output


def _tagger_metric_rows(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    prediction_dir: str | Path | None,
    confirm_final_test: bool,
    problems: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id, report in sorted(reports.items()):
        for split in ("model_val", "stack_val"):
            metrics = _split_metrics(report, split)
            if not isinstance(metrics, Mapping):
                continue
            coverage_problem = _tagger_metric_coverage_problem(
                run_id=run_id,
                report=report,
                split=split,
                metrics=metrics,
            )
            if coverage_problem:
                problems.append(coverage_problem)
                continue
            rows.append(_tagger_metric_row(run_id, report, split, metrics, deployable_final_test=False))
        final_metrics = _prediction_final_test_metrics(prediction_dir, run_id) or _split_metrics(report, "final_test")
        if final_metrics:
            if not bool(confirm_final_test):
                problems.append(f"final_test metrics found for tagger {run_id} but confirm_final_test is false")
                continue
            coverage_problem = _tagger_metric_coverage_problem(
                run_id=run_id,
                report=report,
                split="final_test",
                metrics=final_metrics,
            )
            if coverage_problem:
                problems.append(coverage_problem)
                continue
            rows.append(_tagger_metric_row(run_id, report, "final_test", final_metrics, deployable_final_test=True))
    return rows


def _tagger_metric_row(
    run_id: str,
    report: Mapping[str, Any],
    split: str,
    metrics: Mapping[str, Any],
    *,
    deployable_final_test: bool,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "split": split,
        "field_source": report.get("field_source"),
        "accuracy": _metric_value(metrics, "accuracy"),
        "cross_entropy": _metric_value(metrics, "cross_entropy", "ce_loss"),
        "loss": _metric_value(metrics, "loss"),
        "macro_per_class_accuracy": _metric_value(metrics, "macro_per_class_accuracy"),
        "macro_auc": _metric_value(metrics, "macro_auc", "auc_macro"),
        "n_jets": _metric_value(metrics, "n_jets"),
        "attempted_jets": _metric_value(metrics, "attempted_jets"),
        "valid_fraction": _metric_value(metrics, "valid_fraction"),
        "total_batches": _metric_value(metrics, "total_batches"),
        "finite_batches": _metric_value(metrics, "finite_batches"),
        "nonfinite_batches": _metric_value(metrics, "nonfinite_batches"),
        "nonfinite_grad_batches": _metric_value(metrics, "nonfinite_grad_batches"),
        "nonfinite_fraction": _metric_value(metrics, "nonfinite_fraction"),
        "valid_for_selection": _metric_value(metrics, "valid_for_selection"),
        "selection_expected_n_jets": _metric_value(metrics, "selection_expected_n_jets"),
        "selection_valid_fraction_required": _metric_value(metrics, "selection_valid_fraction_required"),
        "selection_rejection_reason": _metric_value(metrics, "selection_rejection_reason"),
        "best_epoch": report.get("best_epoch"),
        "selection_metric": report.get("selection_metric"),
        "best_model_selection_metric_value": report.get("best_model_selection_metric_value"),
        "selected_field_names": report.get("selected_field_names"),
        "checkpoint": report.get("checkpoint"),
        "deployable_final_test": bool(deployable_final_test),
        "per_class_accuracy": metrics.get("per_class_accuracy"),
        "confusion_matrix": metrics.get("confusion_matrix"),
    }


def _reconstructor_metric_rows(reports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id, report in sorted(reports.items()):
        for split in ("model_val", "stack_val"):
            metrics = _split_metrics(report, split)
            if not isinstance(metrics, Mapping):
                continue
            mae = _float(metrics.get("mae"))
            mse = _float(metrics.get("mse"))
            zero_mae = _float(metrics.get("zero_baseline_mae"))
            zero_mse = _float(metrics.get("zero_baseline_mse"))
            rows.append(
                {
                    "run_id": run_id,
                    "variant": report.get("variant"),
                    "split": split,
                    "mae": mae,
                    "mse": mse,
                    "zero_baseline_mae": zero_mae,
                    "zero_baseline_mse": zero_mse,
                    "relative_mae_improvement_over_zero": None
                    if mae is None or zero_mae in (None, 0.0)
                    else 1.0 - mae / float(zero_mae),
                    "relative_mse_improvement_over_zero": None
                    if mse is None or zero_mse in (None, 0.0)
                    else 1.0 - mse / float(zero_mse),
                    "relative_mae_vs_zero": metrics.get("relative_mae_vs_zero"),
                    "weighted_reconstruction_score": metrics.get("weighted_reconstruction_score"),
                    "consistency_loss": metrics.get("consistency_loss"),
                    "n_jets": metrics.get("n_jets"),
                    "best_epoch": report.get("best_epoch"),
                    "selection_metric": report.get("selection_metric"),
                    "selected_field_names": report.get("selected_field_names"),
                    "per_field_mae": metrics.get("per_field_mae"),
                    "per_group": metrics.get("per_group"),
                    "checkpoint": report.get("checkpoint"),
                }
            )
    return rows


def _rows_by_run_split(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    output: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        run_id = str(row.get("run_id"))
        split = str(row.get("split"))
        output[(run_id, split)] = row
    return output


def _gap_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_run: str,
    candidate_run_ids: Sequence[str],
    output_kind: str,
) -> list[dict[str, Any]]:
    by_key = _rows_by_run_split(rows)
    output: list[dict[str, Any]] = []
    for run_id in candidate_run_ids:
        for split in REPORT_SPLITS:
            candidate = by_key.get((run_id, split))
            baseline = by_key.get((baseline_run, split))
            if not candidate or not baseline:
                continue
            acc = _float(candidate.get("accuracy"))
            base_acc = _float(baseline.get("accuracy"))
            ce = _float(candidate.get("cross_entropy"))
            base_ce = _float(baseline.get("cross_entropy"))
            output.append(
                {
                    "kind": output_kind,
                    "run_id": run_id,
                    "baseline_run": baseline_run,
                    "split": split,
                    "accuracy": acc,
                    "baseline_accuracy": base_acc,
                    "accuracy_gap": None if acc is None or base_acc is None else acc - base_acc,
                    "cross_entropy": ce,
                    "baseline_cross_entropy": base_ce,
                    "cross_entropy_gap": None if ce is None or base_ce is None else ce - base_ce,
                    "field_source": candidate.get("field_source"),
                }
            )
    return output


def _field_importance_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = _rows_by_run_split(rows)
    output: list[dict[str, Any]] = []
    for run_id in FIELD_IMPORTANCE_RUN_IDS:
        for split in REPORT_SPLITS:
            candidate = by_key.get((run_id, split))
            all_fields = by_key.get(("E6", split))
            baseline = by_key.get(("A0", split))
            if not candidate:
                continue
            acc = _float(candidate.get("accuracy"))
            all_acc = _float(all_fields.get("accuracy")) if all_fields else None
            base_acc = _float(baseline.get("accuracy")) if baseline else None
            output.append(
                {
                    "run_id": run_id,
                    "split": split,
                    "accuracy": acc,
                    "accuracy_gap_vs_all_fields_E6": None if acc is None or all_acc is None else acc - all_acc,
                    "accuracy_gap_vs_A0": None if acc is None or base_acc is None else acc - base_acc,
                    "selected_field_names": candidate.get("selected_field_names"),
                    "field_source": candidate.get("field_source"),
                }
            )
    return output


def _validate_required_fusion_groups(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_fusion_groups: Sequence[str],
    require_fusion: bool,
    problems: list[str],
) -> None:
    if not bool(require_fusion):
        return
    required = set(_clean_ids(required_fusion_groups))
    if not required:
        return
    present = {str(row.get("group")) for row in rows if row.get("group") not in (None, "")}
    missing = sorted(required - present)
    if missing:
        problems.append(f"missing required fusion groups: {' '.join(missing)}")


def _fusion_rows(
    fusion_dir: str | Path | None,
    *,
    require_fusion: bool,
    required_fusion_groups: Sequence[str],
    problems: list[str],
) -> list[dict[str, Any]]:
    if not fusion_dir:
        if require_fusion:
            problems.append("fusion_dir is required but was not provided")
        return []
    fusion_path = Path(fusion_dir)
    report_path = fusion_path / "fusion_report.json"
    csv_path = fusion_path / "fusion_metrics.csv"
    if not report_path.exists():
        if require_fusion:
            problems.append(f"missing required fusion report: {report_path}")
        return []
    report = _load_json(report_path)
    if report.get("ok") is not True:
        problems.append(f"fusion report is not ok: ok={report.get('ok')!r}")
    rows: list[dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
        _validate_required_fusion_groups(
            rows,
            required_fusion_groups=required_fusion_groups,
            require_fusion=bool(require_fusion),
            problems=problems,
        )
        return rows
    for group, group_report in dict(report.get("groups") or {}).items():
        if not isinstance(group_report, Mapping):
            continue
        members = group_report.get("members")
        for mode, mode_report in dict(group_report.get("fusion_modes") or {}).items():
            if not isinstance(mode_report, Mapping):
                continue
            if mode_report.get("available") is False:
                problems.append(f"fusion group {group}/{mode} is unavailable")
                continue
            for split, metrics in dict(mode_report.get("metrics") or {}).items():
                if not isinstance(metrics, Mapping):
                    continue
                rows.append(
                    {
                        "group": group,
                        "mode": mode,
                        "split": split,
                        "accuracy": metrics.get("accuracy"),
                        "cross_entropy": metrics.get("cross_entropy"),
                        "macro_per_class_accuracy": metrics.get("macro_per_class_accuracy"),
                        "macro_auc": metrics.get("macro_auc"),
                        "n_jets": metrics.get("n_jets"),
                        "members": members,
                        "weights": _path_value(mode_report, ("fit", "weights")),
                    }
                )
    _validate_required_fusion_groups(
        rows,
        required_fusion_groups=required_fusion_groups,
        require_fusion=bool(require_fusion),
        problems=problems,
    )
    return rows


PROVENANCE_FIELDS: tuple[str, ...] = (
    "source_manifest_hash",
    "hlt_content_hash",
    "offline_content_hash",
    "target_content_hash",
    "jet_identity_hash",
    "hlt_profile",
    "hlt_profile_version",
    "hlt_degradation_strength",
    "target_field_dim",
    "field_names",
)
HLT_ONLY_PROVENANCE_FIELDS: tuple[str, ...] = (
    "source_manifest_hash",
    "hlt_content_hash",
    "jet_identity_hash",
    "hlt_profile",
    "hlt_profile_version",
    "hlt_degradation_strength",
)
TARGET_PROVENANCE_FIELDS: tuple[str, ...] = tuple(
    field for field in PROVENANCE_FIELDS if field not in HLT_ONLY_PROVENANCE_FIELDS
)

FINAL_TEST_REQUIRED_PROVENANCE_FIELDS: tuple[str, ...] = (
    "source_manifest_hash",
    "hlt_content_hash",
    "jet_identity_hash",
    "hlt_profile",
    "hlt_profile_version",
    "hlt_degradation_strength",
)


def _required_provenance_fields_for_split(split: str) -> tuple[str, ...]:
    if str(split) == "final_test":
        return FINAL_TEST_REQUIRED_PROVENANCE_FIELDS
    return PROVENANCE_FIELDS


def _report_uses_target_fields(report: Mapping[str, Any]) -> bool:
    """Return whether a run report should be bound to target/offline field caches."""

    if str(report.get("field_source") or "") == "hlt_only":
        return False
    selected_names = report.get("selected_field_names")
    selected_indices = report.get("selected_field_indices")
    if isinstance(selected_names, Sequence) and not isinstance(selected_names, (str, bytes)):
        if len(selected_names) == 0:
            return False
    if isinstance(selected_indices, Sequence) and not isinstance(selected_indices, (str, bytes)):
        if len(selected_indices) == 0:
            return False
    return True


def _provenance_fields_for_report(report: Mapping[str, Any], split: str) -> tuple[str, ...]:
    base_fields = _required_provenance_fields_for_split(split)
    if _report_uses_target_fields(report):
        return base_fields
    return tuple(field for field in base_fields if field in HLT_ONLY_PROVENANCE_FIELDS)


def _dataset_metadata_for_split(report: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    value = _path_value(report, ("dataset_metadata", split))
    return value if isinstance(value, Mapping) else None


def _dataset_provenance_value(split_metadata: Mapping[str, Any], field: str) -> Any:
    if field == "source_manifest_hash":
        return _first_path_value(
            split_metadata,
            (("alignment_report", "source_manifest_hash"), ("hlt_metadata", "source_manifest_hash")),
        )
    if field == "hlt_content_hash":
        return _first_path_value(
            split_metadata,
            (("alignment_report", "hlt_content_hash"), ("hlt_metadata", "hlt_content_hash")),
        )
    if field == "offline_content_hash":
        return _first_path_value(
            split_metadata,
            (("alignment_report", "offline_content_hash"), ("target_metadata", "offline_content_hash")),
        )
    if field == "target_content_hash":
        return _first_path_value(
            split_metadata,
            (("alignment_report", "target_content_hash"), ("target_metadata", "target_content_hash")),
        )
    if field == "jet_identity_hash":
        return _first_path_value(
            split_metadata,
            (("alignment_report", "jet_identity_hash"), ("hlt_metadata", "jet_identity_hash")),
        )
    if field in {"hlt_profile", "hlt_profile_version", "hlt_degradation_strength"}:
        return _path_value(split_metadata, ("hlt_metadata", field))
    return split_metadata.get(field)


def _target_cache_provenance(target_cache_dir: str | Path | None) -> dict[str, Any]:
    if not target_cache_dir:
        return {}
    root = Path(target_cache_dir)
    output: dict[str, Any] = {}
    for meta_path in sorted(root.glob("*_local_particle_residual_fields_metadata.json")):
        split = meta_path.name.removesuffix("_local_particle_residual_fields_metadata.json")
        try:
            metadata = _load_json(meta_path)
        except Exception:
            continue
        output[split] = {
            field: _target_metadata_value(metadata, field)
            for field in PROVENANCE_FIELDS
            if _target_metadata_value(metadata, field) is not None
        }
    return output


def _target_metadata_value(metadata: Mapping[str, Any], field: str) -> Any:
    if field == "source_manifest_hash":
        return metadata.get("source_manifest_hash")
    if field == "hlt_content_hash":
        return metadata.get("hlt_content_hash")
    if field == "offline_content_hash":
        return metadata.get("offline_content_hash")
    if field == "target_content_hash":
        return metadata.get("target_content_hash")
    if field == "jet_identity_hash":
        return metadata.get("jet_identity_hash")
    if field == "target_field_dim":
        return metadata.get("field_dim")
    if field == "hlt_profile":
        return metadata.get("hlt_profile")
    if field == "hlt_profile_version":
        return metadata.get("hlt_profile_version")
    if field == "hlt_degradation_strength":
        return metadata.get("hlt_degradation_strength")
    return metadata.get(field)


def _provenance_audit(
    *,
    tagger_reports: Mapping[str, Mapping[str, Any]],
    reconstructor_reports: Mapping[str, Mapping[str, Any]],
    required_tagger_run_ids: Sequence[str],
    required_reconstructor_run_ids: Sequence[str],
    target_cache_dir: str | Path | None,
    require_final_test: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    all_reports: dict[str, Mapping[str, Any]] = {
        **{f"tagger:{run_id}": report for run_id, report in tagger_reports.items()},
        **{f"reconstructor:{run_id}": report for run_id, report in reconstructor_reports.items()},
    }
    required_report_names = {
        **{f"tagger:{run_id}": run_id for run_id in required_tagger_run_ids if run_id in tagger_reports},
        **{f"reconstructor:{run_id}": run_id for run_id in required_reconstructor_run_ids if run_id in reconstructor_reports},
    }
    target_cache = _target_cache_provenance(target_cache_dir)
    for split in REPORT_SPLITS:
        split_values: dict[str, dict[str, Any]] = {}
        required_fields_by_run: dict[str, set[str]] = {}
        if split in target_cache:
            split_values["target_cache"] = target_cache[split]
        for run_name, report in all_reports.items():
            report_fields = set(_provenance_fields_for_report(report, split))
            if run_name in required_report_names:
                required_fields_by_run[run_name] = report_fields
            metadata = _dataset_metadata_for_split(report, split)
            if not isinstance(metadata, Mapping):
                if run_name in required_report_names and split != "final_test":
                    problems.append(f"{run_name} missing dataset_metadata for {split}")
                continue
            split_values[run_name] = {
                field: _dataset_provenance_value(metadata, field)
                for field in report_fields
                if _dataset_provenance_value(metadata, field) is not None
            }
        for field in PROVENANCE_FIELDS:
            values: dict[str, str] = {}
            raw_values: dict[str, Any] = {}
            missing_required: list[str] = []
            for run_name, value_map in split_values.items():
                value = value_map.get(field)
                raw_values[run_name] = value
                stable = _stable_value(value)
                if stable is None:
                    if field in required_fields_by_run.get(run_name, set()):
                        missing_required.append(run_name)
                else:
                    values[run_name] = stable
            unique_values = sorted(set(values.values()))
            conflict = len(unique_values) > 1
            required_missing = bool(missing_required)
            if conflict:
                problems.append(f"provenance mismatch for {split}.{field}: {values}")
            if required_missing:
                problems.append(f"missing required provenance {split}.{field}: {' '.join(sorted(missing_required))}")
            rows.append(
                {
                    "split": split,
                    "field": field,
                    "ok": not conflict and not required_missing,
                    "unique_value_count": len(unique_values),
                    "values_by_run": raw_values,
                    "missing_required_runs": missing_required,
                }
            )
    return {"ok": not problems, "problems": problems, "rows": rows, "target_cache": target_cache}


def _best_row(rows: Sequence[Mapping[str, Any]], *, split: str, metric: str, higher_is_better: bool) -> Mapping[str, Any] | None:
    best: Mapping[str, Any] | None = None
    best_value: float | None = None
    for row in rows:
        if row.get("split") != split:
            continue
        value = _float(row.get(metric))
        if value is None:
            continue
        if best is None or best_value is None:
            best = row
            best_value = value
            continue
        if (higher_is_better and value > best_value) or ((not higher_is_better) and value < best_value):
            best = row
            best_value = value
    return best


def _annotate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    reports: Mapping[str, Mapping[str, Any]] | None = None,
    family: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    report_map = dict(reports or {})
    for row in rows:
        run_id = str(row.get("run_id") or row.get("group") or "")
        output.append(classify_result_row(row, report=report_map.get(run_id), family=family))
    return output


def _diagnostic_annotation(rows: Sequence[Mapping[str, Any]], *, reports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = _annotate_rows(rows, reports=reports, family="diagnostic")
    for row in output:
        row["deployable"] = False
        row["selection_allowed"] = False
        row["result_category"] = "derived_diagnostic"
    return output


def _strict_deployable_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if row.get("deployable") is True
        and row.get("runtime_inputs") == "HLT_only"
        and row.get("uses_true_fields") is False
        and row.get("uses_offline_particles") is False
        and row.get("uses_teacher_logits_at_runtime") is False
    ]


def _with_a0_gaps(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    baseline_by_split = {
        str(row.get("split")): row
        for row in rows
        if str(row.get("run_id")) == "A0"
    }
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        baseline = baseline_by_split.get(str(row.get("split")))
        accuracy = _float(row.get("accuracy"))
        baseline_accuracy = _float(baseline.get("accuracy")) if baseline else None
        cross_entropy = _float(row.get("cross_entropy"))
        baseline_ce = _float(baseline.get("cross_entropy")) if baseline else None
        row["headline_baseline_run_id"] = "A0"
        row["headline_baseline_label"] = A0_LABEL
        row["accuracy_gap_vs_A0"] = (
            None if accuracy is None or baseline_accuracy is None else accuracy - baseline_accuracy
        )
        row["cross_entropy_gap_vs_A0"] = (
            None if cross_entropy is None or baseline_ce is None else cross_entropy - baseline_ce
        )
        output.append(row)
    return output


def _summary_markdown(
    *,
    config: LocalResidualFieldReportConfig,
    report: Mapping[str, Any],
    tagger_rows: Sequence[Mapping[str, Any]],
    reconstructor_rows: Sequence[Mapping[str, Any]],
    fusion_rows: Sequence[Mapping[str, Any]],
    deployable_rows: Sequence[Mapping[str, Any]] = (),
    oracle_diagnostic_rows: Sequence[Mapping[str, Any]] = (),
    offline_reference_rows: Sequence[Mapping[str, Any]] = (),
) -> str:
    lines = [
        f"# {config.summary_title}",
        "",
        f"- Contract: `{LOCAL_RESIDUAL_FIELD_REPORT_CONTRACT}`",
        f"- OK: `{bool(report.get('ok'))}`",
        f"- Problems: `{len(report.get('problems') or [])}`",
        f"- `A0`: {A0_LABEL} (headline denominator)",
        f"- `O0`: {O0_LABEL} (not an HLT floor)",
        "",
    ]
    if report.get("problems"):
        lines.append("## Problems")
        lines.extend(f"- {problem}" for problem in report.get("problems") or [])
        lines.append("")
    primary_rows = list(deployable_rows or tagger_rows)
    best_stack = _best_row(primary_rows, split="stack_val", metric="accuracy", higher_is_better=True)
    best_final = _best_row(primary_rows, split="final_test", metric="accuracy", higher_is_better=True)
    best_reco = _best_row(reconstructor_rows, split="stack_val", metric="mae", higher_is_better=False)
    best_fusion = _best_row(fusion_rows, split="final_test", metric="accuracy", higher_is_better=True)
    lines.append("## Main Signals")
    if best_stack:
        lines.append(
            f"- Best stack-val tagger: `{best_stack.get('run_id')}` "
            f"accuracy={best_stack.get('accuracy')} CE={best_stack.get('cross_entropy')}"
        )
    if best_final:
        lines.append(
            f"- Best final-test tagger: `{best_final.get('run_id')}` "
            f"accuracy={best_final.get('accuracy')} CE={best_final.get('cross_entropy')}"
        )
    if best_reco:
        lines.append(
            f"- Best stack-val reconstructor: `{best_reco.get('run_id')}` "
            f"MAE={best_reco.get('mae')} zero_MAE={best_reco.get('zero_baseline_mae')}"
        )
    if best_fusion:
        lines.append(
            f"- Best final-test fusion: `{best_fusion.get('group')}/{best_fusion.get('mode')}` "
            f"accuracy={best_fusion.get('accuracy')}"
        )
    if not any((best_stack, best_final, best_reco, best_fusion)):
        lines.append("- No metric rows were available.")
    lines.append("")
    lines.append("## Automatic Result Separation")
    lines.append(f"- Deployable leaderboard rows: `{len(deployable_rows)}`")
    lines.append(f"- Oracle diagnostic rows: `{len(oracle_diagnostic_rows)}`")
    lines.append(f"- Offline-particle reference rows: `{len(offline_reference_rows)}`")
    lines.append("- Individual checkpoints are selectable on `model_val` only; final-test rows are never selectable.")
    lines.append("")
    lines.append("## Outputs")
    for name, path in dict(report.get("outputs") or {}).items():
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


def build_local_residual_field_report(config: LocalResidualFieldReportConfig) -> dict[str, Any]:
    """Build the final Step 9 report and write all expected tables."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []

    tagger_reports = _load_required_reports(
        config.tagger_root,
        required_ids=config.required_tagger_run_ids,
        family="tagger",
        problems=problems,
        allow_missing_runs=bool(config.allow_missing_runs),
    )
    tagger_reports = _with_prediction_dataset_metadata(tagger_reports, prediction_dir=config.prediction_dir)
    reconstructor_reports = _load_required_reports(
        config.reconstructor_root,
        required_ids=config.required_reconstructor_run_ids,
        family="reconstructor",
        problems=problems,
        allow_missing_runs=bool(config.allow_missing_runs),
    )
    tagger_rows = _tagger_metric_rows(
        tagger_reports,
        prediction_dir=config.prediction_dir,
        confirm_final_test=bool(config.confirm_final_test),
        problems=problems,
    )
    tagger_rows = _annotate_rows(tagger_rows, reports=tagger_reports, family="tagger")
    reconstructor_rows = _annotate_rows(
        _reconstructor_metric_rows(reconstructor_reports),
        reports=reconstructor_reports,
        family="training",
    )
    for row in reconstructor_rows:
        row["deployable"] = False
        row["selection_allowed"] = False
        row["result_category"] = "reconstructor_training_diagnostic"
    fusion_rows = _annotate_rows(_fusion_rows(
        config.fusion_dir,
        require_fusion=bool(config.require_fusion),
        required_fusion_groups=tuple(config.required_fusion_groups),
        problems=problems,
    ), family="fusion")

    curriculum_reports = scan_curriculum_reports(config.curriculum_root)
    if bool(config.require_curriculum) and not config.curriculum_root:
        problems.append("curriculum_root is required when require_curriculum=true")
    for run_id in config.required_curriculum_run_ids:
        if run_id not in curriculum_reports:
            problems.append(f"missing required curriculum run_report for {run_id}")
    for run_id, curriculum_report in sorted(curriculum_reports.items()):
        if curriculum_report.get("ok") is not True:
            problems.append(f"curriculum run {run_id} is not ok: ok={curriculum_report.get('ok')!r}")
    selected = validate_stage1b_reports(
        curriculum_reports,
        selected_consumer_json=config.selected_consumer_json,
        paired_consumer_mode=bool(config.paired_consumer_mode),
        problems=problems,
    )
    if selected is None and config.selected_consumer_json and not bool(config.paired_consumer_mode):
        try:
            selected = load_selected_consumer_record(config.selected_consumer_json)
        except Exception as exc:
            problems.append(f"invalid selected_consumer.json: {exc}")
    curriculum_rows = curriculum_student_rows(
        curriculum_reports,
        confirm_final_test=bool(config.confirm_final_test),
        problems=problems,
    )
    curriculum_training = curriculum_training_rows(curriculum_reports)
    oracle_teacher_rows = oracle_curve_rows(
        config.oracle_diagnostics_root or config.curriculum_root,
        selected=selected,
    )
    selection_rows = consumer_selection_rows(oracle_teacher_rows, selected=selected)
    stage1b_present = any(
        pilot_base_run_id(run_id) in ("P2", "P4", "P7a", "P7b", "Q0", "Q3")
        for run_id in curriculum_reports
    )
    selection_sources = {str(row.get("run_id")) for row in selection_rows}
    if (stage1b_present or bool(config.require_curriculum)) and not bool(config.paired_consumer_mode):
        missing_diagnostics = [run_id for run_id in ALPHA_DIAGNOSTIC_IDS if run_id not in selection_sources]
        if missing_diagnostics:
            problems.append(
                "consumer_selection.csv is missing Stage 1a diagnostics: " + " ".join(missing_diagnostics)
            )

    diagnostic_tables = {
        name: load_diagnostic_rows(
            config.curriculum_diagnostics_root or config.curriculum_root,
            name,
            reports=curriculum_reports,
        )
        for name in (
            "alpha_mix_diagnostics",
            "teacher_student_agreement",
            "field_error_where_oracle_helps",
            "gate_calibration",
        )
    }

    oracle_gap_rows = _diagnostic_annotation(
        _gap_rows(tagger_rows, baseline_run="A0", candidate_run_ids=ORACLE_RUN_IDS, output_kind="oracle"),
        reports=tagger_reports,
    )
    control_rows = []
    control_rows.extend(_gap_rows(tagger_rows, baseline_run="A0", candidate_run_ids=CONTROL_RUN_IDS, output_kind="control_vs_A0"))
    control_rows.extend(_gap_rows(tagger_rows, baseline_run="D5", candidate_run_ids=CONTROL_RUN_IDS, output_kind="control_vs_D5"))
    control_rows = _diagnostic_annotation(control_rows, reports=tagger_reports)
    field_rows = _diagnostic_annotation(_field_importance_rows(tagger_rows), reports=tagger_reports)

    all_primary_rows = [*tagger_rows, *curriculum_rows, *fusion_rows]
    deployable_rows = _with_a0_gaps(_strict_deployable_rows(all_primary_rows))
    oracle_diagnostic_rows = [
        dict(row) for row in [*tagger_rows, *curriculum_rows, *fusion_rows, *oracle_teacher_rows]
        if row.get("result_category") == "oracle_diagnostic"
    ]
    offline_reference_rows = [
        dict(row) for row in [*tagger_rows, *curriculum_rows, *fusion_rows]
        if row.get("result_category") == "offline_reference"
    ]
    provenance = _provenance_audit(
        tagger_reports=tagger_reports,
        reconstructor_reports=reconstructor_reports,
        required_tagger_run_ids=config.required_tagger_run_ids,
        required_reconstructor_run_ids=config.required_reconstructor_run_ids,
        target_cache_dir=config.target_cache_dir,
        require_final_test=bool(config.require_final_test_provenance),
    )
    problems.extend(str(problem) for problem in provenance.get("problems", []))

    outputs = {
        "tagger_metrics_csv": str(output_dir / "tagger_metrics.csv"),
        "reconstructor_metrics_csv": str(output_dir / "reconstructor_metrics.csv"),
        "oracle_gap_csv": str(output_dir / "oracle_gap.csv"),
        "control_gap_csv": str(output_dir / "control_gap.csv"),
        "field_importance_csv": str(output_dir / "field_importance.csv"),
        "fusion_metrics_csv": str(output_dir / "fusion_metrics.csv"),
        "oracle_teacher_curve_csv": str(output_dir / "oracle_teacher_curve.csv"),
        "curriculum_student_metrics_csv": str(output_dir / "curriculum_student_metrics.csv"),
        "alpha_mix_diagnostics_csv": str(output_dir / "alpha_mix_diagnostics.csv"),
        "teacher_student_agreement_csv": str(output_dir / "teacher_student_agreement.csv"),
        "field_error_where_oracle_helps_csv": str(output_dir / "field_error_where_oracle_helps.csv"),
        "gate_calibration_csv": str(output_dir / "gate_calibration.csv"),
        "deployable_leaderboard_csv": str(output_dir / "deployable_leaderboard.csv"),
        "oracle_diagnostics_csv": str(output_dir / "oracle_diagnostics.csv"),
        "offline_reference_csv": str(output_dir / "offline_reference.csv"),
        "curriculum_training_diagnostics_csv": str(output_dir / "curriculum_training_diagnostics.csv"),
        "consumer_selection_csv": str(output_dir / "consumer_selection.csv"),
        "provenance_audit_json": str(output_dir / "provenance_audit.json"),
        "summary_md": str(output_dir / "summary.md"),
        "run_report_json": str(output_dir / "run_report.json"),
    }
    report = {
        "ok": not problems,
        "contract": LOCAL_RESIDUAL_FIELD_REPORT_CONTRACT,
        "step": LOCAL_RESIDUAL_FIELD_REPORT_STEP,
        "config": _jsonable(config.__dict__),
        "n_tagger_reports": len(tagger_reports),
        "n_reconstructor_reports": len(reconstructor_reports),
        "n_fusion_rows": len(fusion_rows),
        "n_curriculum_reports": len(curriculum_reports),
        "n_deployable_rows": len(deployable_rows),
        "n_oracle_diagnostic_rows": len(oracle_diagnostic_rows),
        "n_offline_reference_rows": len(offline_reference_rows),
        "selected_consumer_id": None if selected is None else selected.selected_consumer_id,
        "selected_alpha_endpoint": None if selected is None else selected.selected_alpha_endpoint,
        "problems": problems,
        "outputs": outputs,
    }

    common_fields = ["run_id", "baseline_run_id", "baseline_label", "run_label", "O0_label", *RUNTIME_COLUMNS]
    _write_csv(output_dir / "tagger_metrics.csv", tagger_rows, fieldnames=common_fields)
    _write_csv(output_dir / "reconstructor_metrics.csv", reconstructor_rows, fieldnames=common_fields)
    _write_csv(output_dir / "oracle_gap.csv", oracle_gap_rows, fieldnames=common_fields)
    _write_csv(output_dir / "control_gap.csv", control_rows, fieldnames=common_fields)
    _write_csv(output_dir / "field_importance.csv", field_rows, fieldnames=common_fields)
    _write_csv(output_dir / "fusion_metrics.csv", fusion_rows, fieldnames=common_fields)
    _write_csv(output_dir / "oracle_teacher_curve.csv", oracle_teacher_rows, fieldnames=common_fields)
    _write_csv(output_dir / "curriculum_student_metrics.csv", curriculum_rows, fieldnames=common_fields)
    _write_csv(
        output_dir / "alpha_mix_diagnostics.csv",
        diagnostic_tables["alpha_mix_diagnostics"],
        fieldnames=common_fields,
    )
    _write_csv(
        output_dir / "teacher_student_agreement.csv",
        diagnostic_tables["teacher_student_agreement"],
        fieldnames=common_fields,
    )
    _write_csv(
        output_dir / "field_error_where_oracle_helps.csv",
        diagnostic_tables["field_error_where_oracle_helps"],
        fieldnames=common_fields,
    )
    _write_csv(output_dir / "gate_calibration.csv", diagnostic_tables["gate_calibration"], fieldnames=common_fields)
    _write_csv(output_dir / "deployable_leaderboard.csv", deployable_rows, fieldnames=common_fields)
    _write_csv(output_dir / "oracle_diagnostics.csv", oracle_diagnostic_rows, fieldnames=common_fields)
    _write_csv(output_dir / "offline_reference.csv", offline_reference_rows, fieldnames=common_fields)
    _write_csv(
        output_dir / "curriculum_training_diagnostics.csv",
        curriculum_training,
        fieldnames=common_fields,
    )
    _write_csv(output_dir / "consumer_selection.csv", selection_rows, fieldnames=common_fields)

    if bool(config.paired_consumer_mode):
        paired_student = paired_tables(curriculum_rows)
        paired_deployable = paired_tables(deployable_rows)
        comparison_path = output_dir / "paired_consumer_comparison.csv"
        _write_csv(comparison_path, curriculum_rows, fieldnames=common_fields)
        outputs["paired_consumer_comparison_csv"] = str(comparison_path)
        for consumer_id in ("Ofull", "Orobust_light"):
            safe_name = consumer_id
            student_path = output_dir / f"curriculum_student_metrics_{safe_name}.csv"
            leaderboard_path = output_dir / f"deployable_leaderboard_{safe_name}.csv"
            _write_csv(student_path, paired_student[consumer_id], fieldnames=common_fields)
            _write_csv(leaderboard_path, paired_deployable[consumer_id], fieldnames=common_fields)
            outputs[f"curriculum_student_metrics_{safe_name}_csv"] = str(student_path)
            outputs[f"deployable_leaderboard_{safe_name}_csv"] = str(leaderboard_path)
    _write_json(output_dir / "provenance_audit.json", provenance)
    (output_dir / "summary.md").write_text(
        _summary_markdown(
            config=config,
            report=report,
            tagger_rows=tagger_rows,
            reconstructor_rows=reconstructor_rows,
            fusion_rows=fusion_rows,
            deployable_rows=deployable_rows,
            oracle_diagnostic_rows=oracle_diagnostic_rows,
            offline_reference_rows=offline_reference_rows,
        ),
        encoding="utf-8",
    )
    _write_json(output_dir / "run_report.json", report)
    return report


__all__ = [
    "LOCAL_RESIDUAL_FIELD_REPORT_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_REPORT_STEP",
    "DEFAULT_REQUIRED_TAGGER_RUN_IDS",
    "DEFAULT_REQUIRED_RECONSTRUCTOR_RUN_IDS",
    "DEFAULT_REQUIRED_FUSION_GROUPS",
    "LocalResidualFieldReportConfig",
    "build_local_residual_field_report",
]
