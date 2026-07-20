"""Step 9 reporting helpers for the residual-field curriculum campaign.

This module deliberately consumes saved artifacts rather than reconstructing a
recipe from a run name.  A report therefore fails closed when the selector,
schedule, or runtime-input provenance needed to classify a row is absent.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .curriculum import load_selected_consumer_record


STAGE1B_RUN_IDS = ("P2", "P4", "P7a", "P7b", "Q0", "Q3")
CURRICULUM_RUN_IDS = ("P0",) + STAGE1B_RUN_IDS
CONSUMER_IDS = ("Ofull", "Orobust_light")
ALPHA_DIAGNOSTIC_IDS = ("D_alpha_eval_Ofull", "D_alpha_eval_Orobust")

A0_LABEL = "clean HLT ParT baseline"
O0_LABEL = "zero-field oracle-consumer diagnostic"

RUNTIME_COLUMNS = (
    "runtime_inputs",
    "uses_true_fields",
    "uses_offline_particles",
    "uses_teacher_logits_at_runtime",
    "deployable",
    "split",
    "selection_allowed",
)

DIAGNOSTIC_FILENAMES = {
    "alpha_mix_diagnostics": "alpha_mix_diagnostics.csv",
    "teacher_student_agreement": "teacher_student_agreement.csv",
    "field_error_where_oracle_helps": "field_error_where_oracle_helps.csv",
    "gate_calibration": "gate_calibration.csv",
}


def pilot_base_run_id(run_id: Any) -> str:
    value = str(run_id or "")
    for candidate in sorted(CURRICULUM_RUN_IDS, key=len, reverse=True):
        if value == candidate or value.startswith(candidate + "_"):
            return candidate
    return value


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n", ""}:
            return False
    return bool(default)


def _float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _close(left: Any, right: Any) -> bool:
    left_value = _float(left)
    right_value = _float(right)
    return left_value is not None and right_value is not None and math.isclose(
        left_value, right_value, rel_tol=0.0, abs_tol=1.0e-8
    )


def scan_curriculum_reports(root: str | Path | None) -> dict[str, dict[str, Any]]:
    """Find named pilot reports and attach adjacent metadata artifacts."""

    if not root or not Path(root).exists():
        return {}
    reports: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(root).rglob("run_report.json")):
        report = _load_json(path)
        run_id = str(report.get("run_id") or path.parent.name)
        if pilot_base_run_id(run_id) not in CURRICULUM_RUN_IDS:
            continue
        report["_report_path"] = str(path)
        source_path = path.parent / "source_metadata.json"
        if source_path.is_file():
            source = _load_json(source_path)
            report.setdefault("dataset_metadata", source.get("dataset_metadata"))
            report["_source_metadata"] = source
        reports[run_id] = report
    return reports


def _metric_block(report: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    keys = {
        "model_val": ("best_model_val", "best_model_val_metrics"),
        "stack_val": ("stack_val", "stack_val_metrics"),
        "final_test": ("final_test", "final_test_metrics"),
    }.get(split, (split,))
    for key in keys:
        value = report.get(key)
        if isinstance(value, Mapping) and value.get("available") is not False:
            return value
    metrics = report.get("metrics")
    value = metrics.get(split) if isinstance(metrics, Mapping) else None
    return value if isinstance(value, Mapping) else None


def baseline_columns(run_id: Any = None) -> dict[str, Any]:
    run_key = str(run_id or "")
    return {
        "baseline_run_id": "A0",
        "baseline_label": A0_LABEL,
        "run_label": O0_LABEL if run_key == "O0" else (A0_LABEL if run_key == "A0" else run_key),
        "O0_label": O0_LABEL,
    }


def classify_result_row(
    row: Mapping[str, Any],
    *,
    report: Mapping[str, Any] | None = None,
    family: str = "tagger",
) -> dict[str, Any]:
    """Attach the mutually exclusive Step 9 runtime/result classification."""

    source = dict(report or {})
    evidence = {**dict(row), **source}
    run_id = str(row.get("run_id") or row.get("group") or source.get("run_id") or "")
    split = str(row.get("split") or "")
    members = row.get("members")
    if isinstance(members, str):
        member_ids = {item for item in members.replace(",", " ").split() if item}
    elif isinstance(members, Sequence):
        member_ids = {str(item) for item in members}
    else:
        member_ids = set()

    declared_runtime = str(evidence.get("runtime_inputs") or "")
    offline = (
        run_id == "A4"
        or run_id.startswith("offline")
        or "A4" in member_ids
        or "offline" in declared_runtime.lower()
    )
    oracle = (
        run_id == "O0"
        or run_id.startswith("Ofull")
        or run_id.startswith("Orobust")
        or run_id.startswith("D_alpha_eval_")
        or run_id in {"B0", "B1", "B2", "B3", "B4"}
        or any(
            item.startswith(("O0", "Ofull", "Orobust", "D_alpha_eval_"))
            or item in {"B0", "B1", "B2", "B3", "B4"}
            for item in member_ids
        )
    )
    true_fields = _as_bool(
        evidence.get("uses_true_fields"),
        (oracle and run_id != "O0") or "true_fields" in declared_runtime.lower(),
    )
    offline = _as_bool(evidence.get("uses_offline_particles"), offline)
    teacher_logits = _as_bool(evidence.get("uses_teacher_logits_at_runtime"), False)
    default_runtime = (
        "offline_particles"
        if offline
        else ("HLT_plus_zero_residual_fields" if run_id == "O0" else ("HLT_plus_true_fields" if true_fields else "HLT_only"))
    )
    runtime_inputs = str(evidence.get("runtime_inputs") or default_runtime)
    explicit_deployable = _as_bool(evidence.get("deployable"), not oracle and not offline and family != "oracle")
    deployable = bool(
        explicit_deployable
        and runtime_inputs == "HLT_only"
        and not true_fields
        and not offline
        and not teacher_logits
        and not oracle
    )
    if family == "fusion":
        selection_allowed = split in {"stack_train", "stack_val"}
    elif family == "oracle":
        selection_allowed = split in {"model_val", "stack_val"}
    else:
        selection_allowed = bool(deployable and split == "model_val")
    category = "deployable" if deployable else ("offline_reference" if offline else "oracle_diagnostic")
    return {
        **dict(row),
        **baseline_columns(run_id),
        "runtime_inputs": runtime_inputs,
        "uses_true_fields": bool(true_fields),
        "uses_offline_particles": bool(offline),
        "uses_teacher_logits_at_runtime": bool(teacher_logits),
        "deployable": bool(deployable),
        "split": split,
        "selection_allowed": bool(selection_allowed),
        "result_category": category,
    }


def curriculum_student_rows(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    confirm_final_test: bool,
    problems: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id, report in sorted(reports.items()):
        for split in ("model_val", "stack_val", "final_test"):
            metrics = _metric_block(report, split)
            if not metrics:
                continue
            if split == "final_test" and not confirm_final_test:
                problems.append(f"final_test metrics found for curriculum student {run_id} but confirm_final_test is false")
                continue
            if _float(metrics.get("accuracy")) is None:
                problems.append(f"curriculum student {run_id} {split} accuracy is missing or nonfinite")
                continue
            if _float(metrics.get("loss")) is None and _float(metrics.get("cross_entropy")) is None:
                problems.append(f"curriculum student {run_id} {split} loss/cross_entropy is missing or nonfinite")
                continue
            row = {
                "run_id": run_id,
                "split": split,
                "accuracy": metrics.get("accuracy"),
                "cross_entropy": metrics.get("cross_entropy", metrics.get("ce_loss")),
                "loss": metrics.get("loss"),
                "macro_per_class_accuracy": metrics.get("macro_per_class_accuracy"),
                "macro_auc": metrics.get("macro_auc", metrics.get("auc_macro")),
                "n_jets": metrics.get("n_jets"),
                "attempted_jets": metrics.get("attempted_jets"),
                "valid_fraction": metrics.get("valid_fraction"),
                "total_batches": metrics.get("total_batches"),
                "finite_batches": metrics.get("finite_batches"),
                "nonfinite_batches": metrics.get("nonfinite_batches"),
                "nonfinite_grad_batches": metrics.get("nonfinite_grad_batches"),
                "nonfinite_fraction": metrics.get("nonfinite_fraction"),
                "valid_for_selection": metrics.get("valid_for_selection"),
                "best_epoch": report.get("best_epoch"),
                "checkpoint": report.get("checkpoint"),
                "selected_consumer_id": report.get("selected_consumer_id"),
                "selected_alpha_endpoint": report.get("selected_alpha_endpoint"),
                "teacher_used_during_training": report.get("teacher_used_during_training"),
                "scientific_recipe_equivalent": report.get("scientific_recipe_equivalent"),
                "per_class_accuracy": metrics.get("per_class_accuracy"),
                "confusion_matrix": metrics.get("confusion_matrix"),
            }
            rows.append(classify_result_row(row, report=report, family="curriculum"))
    return rows


def _artifact_path(report: Mapping[str, Any], filename: str) -> Path | None:
    report_path = Path(str(report.get("_report_path"))) if report.get("_report_path") else None
    if report_path is not None:
        adjacent = report_path.parent / filename
        if adjacent.is_file():
            return adjacent
    declared = report.get(filename.removesuffix(".json"))
    if declared and Path(str(declared)).is_file():
        return Path(str(declared))
    return None


def curriculum_training_rows(reports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id, report in sorted(reports.items()):
        path = _artifact_path(report, "training_curves.json")
        if path is None:
            continue
        payload = _load_json(path)
        for epoch_row in payload.get("epochs") or []:
            if not isinstance(epoch_row, Mapping):
                continue
            schedule = epoch_row.get("schedule") if isinstance(epoch_row.get("schedule"), Mapping) else {}
            weights = schedule.get("loss_weights") if isinstance(schedule.get("loss_weights"), Mapping) else {}
            freeze = epoch_row.get("freeze") if isinstance(epoch_row.get("freeze"), Mapping) else {}
            for split in ("train", "model_val"):
                metrics = epoch_row.get(split)
                if not isinstance(metrics, Mapping):
                    continue
                row = {
                    "run_id": run_id,
                    "epoch": epoch_row.get("epoch"),
                    "split": split,
                    "alpha": schedule.get("alpha"),
                    "active_consumer_id": schedule.get("active_consumer_id"),
                    "freeze_phase": freeze.get("phase"),
                    "loss_weights": weights,
                    "accuracy": metrics.get("accuracy"),
                    "cross_entropy": metrics.get("cross_entropy", metrics.get("ce_loss")),
                    "loss": metrics.get("loss"),
                    "field_loss": metrics.get("field_loss"),
                    "oracle_path_loss": metrics.get("oracle_path_loss"),
                    "student_kd_loss": metrics.get("student_kd_loss"),
                    "gate_loss": metrics.get("gate_loss"),
                    "n_jets": metrics.get("n_jets"),
                    "nonfinite_batches": metrics.get("nonfinite_batches"),
                }
                annotated = classify_result_row(row, report=report, family="training")
                annotated["deployable"] = False
                annotated["selection_allowed"] = bool(split == "model_val")
                annotated["result_category"] = "curriculum_training_diagnostic"
                rows.append(annotated)
    return rows


def _schedule(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = _artifact_path(report, "curriculum_schedule.json")
    payload = _load_json(path) if path is not None else {}
    epochs = payload.get("epochs")
    return [dict(item) for item in epochs or [] if isinstance(item, Mapping)]


def _expected_piecewise_alpha(epoch: int, endpoint: float) -> float:
    if endpoint <= 0.25:
        return endpoint
    if endpoint <= 0.50:
        return 0.25 if epoch < 3 else endpoint
    if epoch < 3:
        return 0.25
    if epoch < 6:
        return 0.50
    return endpoint


def _weights(report: Mapping[str, Any]) -> dict[str, float]:
    resolved = report.get("resolved_run")
    raw = resolved.get("loss_weights") if isinstance(resolved, Mapping) else {}
    return {str(key): float(value) for key, value in dict(raw or {}).items() if _float(value) is not None}


def _resolved_value(report: Mapping[str, Any], key: str) -> Any:
    resolved = report.get("resolved_run")
    if isinstance(resolved, Mapping) and key in resolved:
        return resolved.get(key)
    return report.get(key)


def validate_stage1b_reports(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    selected_consumer_json: str | Path | None,
    paired_consumer_mode: bool,
    problems: list[str],
) -> Any:
    stage_reports = {
        run_id: report for run_id, report in reports.items()
        if pilot_base_run_id(run_id) in STAGE1B_RUN_IDS
    }
    if not stage_reports:
        return None
    selected = None
    if not paired_consumer_mode:
        if not selected_consumer_json:
            problems.append("Stage 1b rows require selected_consumer.json; refusing to accept a guessed consumer")
            return None
        try:
            selected = load_selected_consumer_record(selected_consumer_json)
        except Exception as exc:
            problems.append(f"invalid selected_consumer.json: {exc}")
            return None

    found_consumers: set[str] = set()
    for run_id, report in sorted(stage_reports.items()):
        recipe_id = pilot_base_run_id(run_id)
        consumer = str(_resolved_value(report, "selected_consumer_id") or "")
        endpoint = _float(_resolved_value(report, "selected_alpha_endpoint"))
        paired = _as_bool(_resolved_value(report, "paired_consumer_mode"), False)
        if consumer:
            found_consumers.add(consumer)
        if consumer not in CONSUMER_IDS:
            problems.append(f"Stage 1b run {run_id} has invalid or missing selected_consumer_id={consumer!r}")
        if endpoint is None or not 0.0 <= endpoint <= 1.0:
            problems.append(f"Stage 1b run {run_id} has invalid or missing selected_alpha_endpoint")
            continue
        if paired_consumer_mode:
            if not paired:
                problems.append(f"paired-consumer report received unpaired Stage 1b run {run_id}")
        else:
            assert selected is not None
            if paired:
                problems.append(f"default two-stage report received paired-consumer run {run_id}")
            if consumer != selected.selected_consumer_id:
                problems.append(
                    f"Stage 1b run {run_id} consumer {consumer!r} does not match selected_consumer_id "
                    f"{selected.selected_consumer_id!r}"
                )
            if not _close(endpoint, selected.selected_alpha_endpoint):
                problems.append(
                    f"Stage 1b run {run_id} endpoint {endpoint} does not match selected_alpha_endpoint "
                    f"{selected.selected_alpha_endpoint}"
                )
        if report.get("scientific_recipe_equivalent") is False:
            problems.append(f"Stage 1b run {run_id} is marked scientific_recipe_equivalent=false")
        if not _as_bool(report.get("deployable"), False) or str(report.get("runtime_inputs") or "") != "HLT_only":
            problems.append(f"Stage 1b run {run_id} does not declare a deployable HLT-only runtime contract")
        if any(_as_bool(report.get(name), False) for name in (
            "uses_true_fields", "uses_offline_particles", "uses_teacher_logits_at_runtime"
        )):
            problems.append(f"Stage 1b run {run_id} declares privileged inference inputs")

        states = _schedule(report)
        if not states:
            problems.append(f"Stage 1b run {run_id} is missing curriculum_schedule.json epoch provenance")
            continue
        if not paired_consumer_mode:
            assert selected is not None
            selector_hashes = {str(state.get("selected_consumer_hash") or "") for state in states}
            if selector_hashes != {str(selected.source_hash)}:
                problems.append(
                    f"Stage 1b run {run_id} did not persist the selected_consumer.json hash "
                    f"{selected.source_hash} in every schedule epoch"
                )
        for state in states:
            epoch = int(state.get("epoch", 0))
            active = str(state.get("active_consumer_id") or state.get("selected_consumer_id") or "")
            if active != consumer:
                problems.append(f"Stage 1b run {run_id} epoch {epoch} uses consumer {active!r}, expected {consumer!r}")
                break
        if recipe_id == "P2":
            if any(not _close(state.get("alpha"), 0.25) for state in states):
                problems.append("P2 must use fixed alpha=0.25 for every epoch")
        elif recipe_id == "Q3":
            if any(not _close(state.get("alpha"), endpoint) for state in states):
                problems.append("Q3 must use selected_alpha_endpoint from epoch 0 with no curriculum ramp")
        else:
            for state in states:
                epoch = int(state.get("epoch", 0))
                if not _close(state.get("alpha"), _expected_piecewise_alpha(epoch, endpoint)):
                    problems.append(f"{run_id} does not use the selected pilot schedule at epoch {epoch}")
                    break

    if not paired_consumer_mode and len(found_consumers) > 1:
        problems.append(
            "default two-stage report contains mixed Stage 1b consumers: " + " ".join(sorted(found_consumers))
        )

    q0_reports = [(run_id, report) for run_id, report in stage_reports.items() if pilot_base_run_id(run_id) == "Q0"]
    for q0_id, q0 in q0_reports:
        q_consumer = str(_resolved_value(q0, "selected_consumer_id") or "")
        matching_p7a = [
            report for run_id, report in stage_reports.items()
            if pilot_base_run_id(run_id) == "P7a"
            and str(_resolved_value(report, "selected_consumer_id") or "") == q_consumer
        ]
        if not matching_p7a:
            problems.append(f"{q0_id} cannot be accepted without matching P7a recipe provenance")
        else:
            p7a = matching_p7a[0]
            q_weights = _weights(q0)
            p_weights = _weights(p7a)
            names = set(q_weights) | set(p_weights)
            for name in sorted(names):
                expected = 0.0 if name == "oracle_path" else p_weights.get(name)
                if expected is None or not _close(q_weights.get(name), expected):
                    problems.append(f"Q0 does not match selected P7a loss recipe for {name}")
            for key in ("gate_mode", "student_init_source"):
                if _resolved_value(q0, key) != _resolved_value(p7a, key):
                    problems.append(f"Q0 does not match selected P7a recipe for {key}")
            q_states = _schedule(q0)
            p_states = _schedule(p7a)
            q_alpha = [state.get("alpha") for state in q_states]
            p_alpha = [state.get("alpha") for state in p_states]
            if len(q_alpha) != len(p_alpha) or any(not _close(a, b) for a, b in zip(q_alpha, p_alpha)):
                problems.append("Q0 alpha schedule does not exactly match selected P7a")
    return selected


def _curve_rows(run_id: str, consumer_id: str, split: str, curve: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alpha_key, value in curve.items():
        alpha = _float(alpha_key)
        if alpha is None:
            continue
        metrics = dict(value) if isinstance(value, Mapping) else {"accuracy": value}
        row = {
            "run_id": run_id,
            "consumer_id": consumer_id,
            "split": split,
            "alpha": alpha,
            "accuracy": metrics.get("accuracy"),
            "cross_entropy": metrics.get("cross_entropy", metrics.get("ce")),
            "teacher_entropy": metrics.get("teacher_entropy", metrics.get("entropy")),
            "teacher_confidence": metrics.get("teacher_confidence", metrics.get("confidence")),
            "per_class_accuracy": metrics.get("per_class_accuracy"),
            "n_jets": metrics.get("n_jets"),
        }
        rows.append(classify_result_row(row, report={"uses_true_fields": True, "deployable": False}, family="oracle"))
    return rows


def oracle_curve_rows(
    root: str | Path | None,
    *,
    selected: Any = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if root and Path(root).exists():
        for path in sorted(Path(root).rglob("run_report.json")):
            report = _load_json(path)
            run_id = str(report.get("run_id") or path.parent.name)
            if run_id not in ALPHA_DIAGNOSTIC_IDS and run_id not in {"O0", "Ofull", "Orobust_light"}:
                continue
            consumer = "Ofull" if "Ofull" in run_id else ("Orobust_light" if "Orobust" in run_id else run_id)
            found_curve = False
            for split in ("model_val", "stack_val", "final_test"):
                curve = report.get(f"{split}_alpha_curve")
                if isinstance(curve, Mapping):
                    rows.extend(_curve_rows(run_id, consumer, split, curve))
                    found_curve = True
            if found_curve:
                continue
            for split in ("model_val", "stack_val", "final_test"):
                metrics = _metric_block(report, split)
                if not metrics:
                    continue
                alpha = report.get("alpha", 0.0 if run_id == "O0" else 1.0)
                rows.extend(_curve_rows(run_id, consumer, split, {str(alpha): metrics}))
    if not rows and selected is not None:
        rows.extend(_curve_rows(
            f"D_alpha_eval_{'Ofull' if selected.selected_consumer_id == 'Ofull' else 'Orobust'}",
            selected.selected_consumer_id,
            "model_val",
            selected.model_val_alpha_curve,
        ))
        rows.extend(_curve_rows(
            f"D_alpha_eval_{'Ofull' if selected.selected_consumer_id == 'Ofull' else 'Orobust'}",
            selected.selected_consumer_id,
            "stack_val",
            selected.stack_val_alpha_curve,
        ))
    return rows


def consumer_selection_rows(oracle_rows: Sequence[Mapping[str, Any]], selected: Any = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in oracle_rows:
        if str(row.get("run_id")) not in ALPHA_DIAGNOSTIC_IDS:
            continue
        consumer = str(row.get("consumer_id") or "")
        output = dict(row)
        output.update({
            "selected": bool(selected is not None and consumer == selected.selected_consumer_id),
            "selected_alpha_endpoint": None if selected is None else selected.selected_alpha_endpoint,
            "selection_source": None if selected is None else selected.selection_source,
            "selection_reason": None if selected is None else selected.selection_reason,
        })
        rows.append(output)
    return rows


def load_diagnostic_rows(
    root: str | Path | None,
    table_name: str,
    *,
    reports: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    filename = DIAGNOSTIC_FILENAMES[table_name]
    search_roots: list[Path] = []
    if root and Path(root).exists():
        search_roots.append(Path(root))
    for report in reports.values():
        if report.get("_report_path"):
            search_roots.append(Path(str(report["_report_path"])).parent)
    paths: set[Path] = set()
    for search_root in search_roots:
        paths.update(search_root.rglob(filename))
        paths.update(search_root.rglob(filename.removesuffix(".csv") + ".json"))
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        if path.suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                raw_rows = list(csv.DictReader(handle))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                raw_rows = payload
            elif isinstance(payload, Mapping):
                raw_rows = payload.get("rows") or payload.get(table_name) or []
            else:
                raw_rows = []
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                continue
            run_id = raw.get("run_id") or path.parent.name
            report = reports.get(str(run_id), {})
            annotated = classify_result_row({**dict(raw), "run_id": run_id}, report=report, family="diagnostic")
            annotated["deployable"] = False
            annotated["selection_allowed"] = False
            annotated["result_category"] = table_name
            rows.append(annotated)
    return rows


def paired_tables(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {consumer: [] for consumer in CONSUMER_IDS}
    for row in rows:
        consumer = str(row.get("selected_consumer_id") or row.get("consumer_id") or "")
        if consumer in output:
            output[consumer].append(dict(row))
    return output


__all__ = [
    "A0_LABEL",
    "O0_LABEL",
    "RUNTIME_COLUMNS",
    "STAGE1B_RUN_IDS",
    "ALPHA_DIAGNOSTIC_IDS",
    "baseline_columns",
    "classify_result_row",
    "consumer_selection_rows",
    "curriculum_student_rows",
    "curriculum_training_rows",
    "load_diagnostic_rows",
    "oracle_curve_rows",
    "paired_tables",
    "pilot_base_run_id",
    "scan_curriculum_reports",
    "validate_stage1b_reports",
]
