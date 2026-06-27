"""Step 8 diagnostics for the reliability-gated dual-view ParT branch."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DUALVIEW_PART_STEP8 = "reliability_gated_dualview_part_step8_diagnostics"
DUALVIEW_PART_DIAGNOSTICS_CONTRACT = "dualview_part_residual_gate_diagnostics_v1"
DUALVIEW_PART_HLT_CONFIDENCE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("[0.00,0.50)", 0.00, 0.50),
    ("[0.50,0.70)", 0.50, 0.70),
    ("[0.70,0.85)", 0.70, 0.85),
    ("[0.85,0.95)", 0.85, 0.95),
    ("[0.95,1.00]", 0.95, 1.0000001),
)
DUALVIEW_PART_CASE_FILE_COLUMNS: tuple[str, ...] = (
    "split",
    "case_type",
    "dataset_index",
    "jet_file",
    "jet_entry",
    "jet_label",
    "true_label",
    "true_label_name",
    "hlt_pred",
    "hlt_pred_name",
    "final_pred",
    "final_pred_name",
    "hlt_correct",
    "final_correct",
    "hlt_confidence",
    "final_confidence",
    "gate_mean",
    "gate_max",
    "delta_l2",
    "residual_l2",
    "hlt_logits_json",
    "final_logits_json",
    "delta_logits_json",
    "residual_logits_json",
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float(default)
    return output if np.isfinite(output) else float(default)


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else 0.0


def _count_distribution(values: np.ndarray, *, names: Sequence[str]) -> dict[str, int]:
    output: dict[str, int] = {}
    for index, name in enumerate(names):
        output[str(name)] = int(np.sum(values == int(index)))
    return output


def _name_for_label(index: int, names: Sequence[str]) -> str:
    idx = int(index)
    if 0 <= idx < len(names):
        return str(names[idx])
    return str(idx)


def _json_float_list(values: np.ndarray) -> str:
    return json.dumps([round(float(value), 8) for value in np.asarray(values, dtype=np.float64).tolist()])


def _jet_id_field(jet_id: Any, field_name: str, default: Any = "") -> Any:
    if jet_id is None:
        return default
    if isinstance(jet_id, Mapping):
        return jet_id.get(field_name, default)
    return getattr(jet_id, field_name, default)


def build_residual_case_rows(
    *,
    split: str,
    logits: np.ndarray,
    hlt_logits: np.ndarray,
    labels: np.ndarray,
    gate: np.ndarray,
    delta_logits: np.ndarray,
    residual_logits: np.ndarray,
    sample_indices: Sequence[int] | np.ndarray | None = None,
    jet_ids: Sequence[Any] | None = None,
    label_names: Sequence[str] | None = None,
    max_cases_per_type: int | None = 1000,
) -> list[dict[str, Any]]:
    """Build inspectable per-jet rows where the residual fixes or breaks HLT.

    Rows are sorted by residual norm within each case type, so capped exports
    keep the most intervention-heavy examples.
    """

    logits = np.asarray(logits, dtype=np.float64)
    hlt_logits = np.asarray(hlt_logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    gate = np.asarray(gate, dtype=np.float64)
    delta_logits = np.asarray(delta_logits, dtype=np.float64)
    residual_logits = np.asarray(residual_logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [jets, classes]")
    n_jets, n_classes = int(logits.shape[0]), int(logits.shape[1])
    if hlt_logits.shape != logits.shape or delta_logits.shape != logits.shape or residual_logits.shape != logits.shape:
        raise ValueError("hlt/delta/residual logits must match logits shape")
    if labels.shape != (n_jets,):
        raise ValueError("labels must have shape [jets]")
    if gate.ndim == 1:
        gate = gate[:, None]
    if gate.ndim != 2 or int(gate.shape[0]) != n_jets:
        raise ValueError("gate must have shape [jets] or [jets, gate_dim]")
    names = tuple(label_names or [str(index) for index in range(n_classes)])
    if len(names) != n_classes:
        raise ValueError("label_names length must match logits class dimension")
    if sample_indices is None:
        indices = np.arange(n_jets, dtype=np.int64)
    else:
        indices = np.asarray(sample_indices, dtype=np.int64)
        if indices.shape != (n_jets,):
            raise ValueError("sample_indices must have shape [jets]")
    jet_id_list = list(jet_ids or [None] * n_jets)
    if len(jet_id_list) != n_jets:
        raise ValueError("jet_ids length must match number of jets")

    hlt_probs = _softmax(hlt_logits)
    final_probs = _softmax(logits)
    hlt_pred = np.argmax(hlt_logits, axis=1).astype(np.int64)
    final_pred = np.argmax(logits, axis=1).astype(np.int64)
    hlt_correct = hlt_pred == labels
    final_correct = final_pred == labels
    gate_scalar = gate.mean(axis=1)
    gate_max = gate.max(axis=1)
    delta_l2 = np.linalg.norm(delta_logits, axis=1)
    residual_l2 = np.linalg.norm(residual_logits, axis=1)

    rows: list[dict[str, Any]] = []
    case_masks = {
        "fix": (~hlt_correct) & final_correct,
        "break": hlt_correct & (~final_correct),
    }
    cap = None if max_cases_per_type is None else max(0, int(max_cases_per_type))
    for case_type, mask in case_masks.items():
        case_indices = np.flatnonzero(mask)
        if case_indices.size:
            order = np.argsort(-residual_l2[case_indices])
            case_indices = case_indices[order]
        if cap is not None:
            case_indices = case_indices[:cap]
        for row_index in case_indices:
            jet_id = jet_id_list[int(row_index)]
            true_label = int(labels[row_index])
            hlt_label = int(hlt_pred[row_index])
            final_label = int(final_pred[row_index])
            rows.append(
                {
                    "split": str(split),
                    "case_type": case_type,
                    "dataset_index": int(indices[row_index]),
                    "jet_file": str(_jet_id_field(jet_id, "file", "")),
                    "jet_entry": _jet_id_field(jet_id, "entry", ""),
                    "jet_label": _jet_id_field(jet_id, "label", ""),
                    "true_label": true_label,
                    "true_label_name": _name_for_label(true_label, names),
                    "hlt_pred": hlt_label,
                    "hlt_pred_name": _name_for_label(hlt_label, names),
                    "final_pred": final_label,
                    "final_pred_name": _name_for_label(final_label, names),
                    "hlt_correct": bool(hlt_correct[row_index]),
                    "final_correct": bool(final_correct[row_index]),
                    "hlt_confidence": float(hlt_probs[row_index, hlt_label]),
                    "final_confidence": float(final_probs[row_index, final_label]),
                    "gate_mean": float(gate_scalar[row_index]),
                    "gate_max": float(gate_max[row_index]),
                    "delta_l2": float(delta_l2[row_index]),
                    "residual_l2": float(residual_l2[row_index]),
                    "hlt_logits_json": _json_float_list(hlt_logits[row_index]),
                    "final_logits_json": _json_float_list(logits[row_index]),
                    "delta_logits_json": _json_float_list(delta_logits[row_index]),
                    "residual_logits_json": _json_float_list(residual_logits[row_index]),
                }
            )
    return rows


def _subset_summary(
    *,
    mask: np.ndarray,
    gate_scalar: np.ndarray,
    hlt_confidence: np.ndarray,
    delta_l2: np.ndarray,
    residual_l2: np.ndarray,
    final_correct: np.ndarray,
    hlt_correct: np.ndarray,
    changed: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    n_total = int(mask.shape[0])
    n = int(np.sum(mask))
    if n == 0:
        return {
            "n": 0,
            "fraction": 0.0,
            "gate_mean": 0.0,
            "gate_std": 0.0,
            "hlt_confidence_mean": 0.0,
            "delta_l2_mean": 0.0,
            "residual_l2_mean": 0.0,
            "hlt_accuracy": 0.0,
            "final_accuracy": 0.0,
            "prediction_change_fraction": 0.0,
            "fix_count": 0,
            "break_count": 0,
        }
    fixes = mask & (~hlt_correct) & final_correct
    breaks = mask & hlt_correct & (~final_correct)
    return {
        "n": n,
        "fraction": float(n / n_total) if n_total else 0.0,
        "gate_mean": _mean(gate_scalar[mask]),
        "gate_std": _std(gate_scalar[mask]),
        "hlt_confidence_mean": _mean(hlt_confidence[mask]),
        "delta_l2_mean": _mean(delta_l2[mask]),
        "residual_l2_mean": _mean(residual_l2[mask]),
        "hlt_accuracy": _mean(hlt_correct[mask].astype(np.float64)),
        "final_accuracy": _mean(final_correct[mask].astype(np.float64)),
        "prediction_change_fraction": _mean(changed[mask].astype(np.float64)),
        "fix_count": int(np.sum(fixes)),
        "break_count": int(np.sum(breaks)),
    }


def summarize_residual_behavior(
    *,
    logits: np.ndarray,
    hlt_logits: np.ndarray,
    labels: np.ndarray,
    gate: np.ndarray,
    delta_logits: np.ndarray,
    residual_logits: np.ndarray,
    label_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Summarize whether the PN residual branch is doing useful work."""

    logits = np.asarray(logits, dtype=np.float64)
    hlt_logits = np.asarray(hlt_logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    gate = np.asarray(gate, dtype=np.float64)
    delta_logits = np.asarray(delta_logits, dtype=np.float64)
    residual_logits = np.asarray(residual_logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [jets, classes]")
    if hlt_logits.shape != logits.shape:
        raise ValueError("hlt_logits shape must match logits")
    if delta_logits.shape != logits.shape or residual_logits.shape != logits.shape:
        raise ValueError("delta/residual logits must match logits shape")
    if labels.shape != (int(logits.shape[0]),):
        raise ValueError("labels must have shape [jets]")
    if gate.ndim == 1:
        gate = gate[:, None]
    if gate.ndim != 2 or int(gate.shape[0]) != int(logits.shape[0]):
        raise ValueError("gate must have shape [jets] or [jets, gate_dim]")
    names = tuple(label_names or [str(index) for index in range(int(logits.shape[1]))])
    if len(names) != int(logits.shape[1]):
        raise ValueError("label_names length must match logits class dimension")

    probs_hlt = _softmax(hlt_logits)
    hlt_confidence = probs_hlt.max(axis=1)
    hlt_pred = np.argmax(hlt_logits, axis=1).astype(np.int64)
    final_pred = np.argmax(logits, axis=1).astype(np.int64)
    hlt_correct = hlt_pred == labels
    final_correct = final_pred == labels
    changed = final_pred != hlt_pred
    fixes = (~hlt_correct) & final_correct
    breaks = hlt_correct & (~final_correct)
    both_correct = hlt_correct & final_correct
    both_wrong = (~hlt_correct) & (~final_correct)
    gate_scalar = gate.mean(axis=1)
    delta_l2 = np.linalg.norm(delta_logits, axis=1)
    residual_l2 = np.linalg.norm(residual_logits, axis=1)
    n_jets = int(labels.shape[0])

    by_class = []
    for index, name in enumerate(names):
        summary = _subset_summary(
            mask=labels == int(index),
            gate_scalar=gate_scalar,
            hlt_confidence=hlt_confidence,
            delta_l2=delta_l2,
            residual_l2=residual_l2,
            final_correct=final_correct,
            hlt_correct=hlt_correct,
            changed=changed,
        )
        summary.update({"class_index": int(index), "class_name": str(name)})
        by_class.append(summary)

    by_hlt_confidence_bucket = []
    for bucket_name, low, high in DUALVIEW_PART_HLT_CONFIDENCE_BUCKETS:
        mask = (hlt_confidence >= float(low)) & (hlt_confidence < float(high))
        summary = _subset_summary(
            mask=mask,
            gate_scalar=gate_scalar,
            hlt_confidence=hlt_confidence,
            delta_l2=delta_l2,
            residual_l2=residual_l2,
            final_correct=final_correct,
            hlt_correct=hlt_correct,
            changed=changed,
        )
        summary.update({"bucket": bucket_name, "low": float(low), "high": 1.0 if high > 1.0 else float(high)})
        by_hlt_confidence_bucket.append(summary)

    by_hlt_correctness = []
    for name, mask in (("hlt_correct", hlt_correct), ("hlt_wrong", ~hlt_correct)):
        summary = _subset_summary(
            mask=mask,
            gate_scalar=gate_scalar,
            hlt_confidence=hlt_confidence,
            delta_l2=delta_l2,
            residual_l2=residual_l2,
            final_correct=final_correct,
            hlt_correct=hlt_correct,
            changed=changed,
        )
        summary["bucket"] = name
        by_hlt_correctness.append(summary)

    prediction_changes = {
        "n_jets": n_jets,
        "changed_count": int(np.sum(changed)),
        "changed_fraction": float(np.mean(changed)) if n_jets else 0.0,
        "fix_count": int(np.sum(fixes)),
        "fix_fraction": float(np.mean(fixes)) if n_jets else 0.0,
        "break_count": int(np.sum(breaks)),
        "break_fraction": float(np.mean(breaks)) if n_jets else 0.0,
        "both_correct_count": int(np.sum(both_correct)),
        "both_wrong_count": int(np.sum(both_wrong)),
        "hlt_only_correct_count": int(np.sum(breaks)),
        "final_only_correct_count": int(np.sum(fixes)),
        "net_correct_gain_count": int(np.sum(fixes) - np.sum(breaks)),
        "hlt_accuracy": float(np.mean(hlt_correct)) if n_jets else 0.0,
        "final_accuracy": float(np.mean(final_correct)) if n_jets else 0.0,
        "fix_final_pred_distribution": _count_distribution(final_pred[fixes], names=names),
        "break_final_pred_distribution": _count_distribution(final_pred[breaks], names=names),
        "fix_true_label_distribution": _count_distribution(labels[fixes], names=names),
        "break_true_label_distribution": _count_distribution(labels[breaks], names=names),
        "fix_gate_mean": _mean(gate_scalar[fixes]),
        "break_gate_mean": _mean(gate_scalar[breaks]),
        "fix_hlt_confidence_mean": _mean(hlt_confidence[fixes]),
        "break_hlt_confidence_mean": _mean(hlt_confidence[breaks]),
    }

    return {
        "experiment_step": DUALVIEW_PART_STEP8,
        "output_contract": DUALVIEW_PART_DIAGNOSTICS_CONTRACT,
        "n_jets": n_jets,
        "gate_shape": list(gate.shape),
        "gate_mean": _mean(gate_scalar),
        "gate_std": _std(gate_scalar),
        "gate_min": float(np.min(gate_scalar)) if n_jets else 0.0,
        "gate_max": float(np.max(gate_scalar)) if n_jets else 0.0,
        "delta_logit_l2_mean": _mean(delta_l2),
        "delta_logit_l2_std": _std(delta_l2),
        "residual_logit_l2_mean": _mean(residual_l2),
        "residual_logit_l2_std": _std(residual_l2),
        "prediction_changes": prediction_changes,
        "gate_by_class": by_class,
        "gate_by_hlt_confidence_bucket": by_hlt_confidence_bucket,
        "gate_by_hlt_correctness": by_hlt_correctness,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _prefixed_rows(split: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        payload = {"split": str(split)}
        payload.update(dict(row))
        output.append(payload)
    return output


def write_residual_diagnostics(
    diagnostics_dir: str | Path,
    analyses_by_split: Mapping[str, Mapping[str, Any]],
    case_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, str]:
    """Write Step 8 diagnostics JSON and CSV summaries."""

    diagnostics_dir = Path(diagnostics_dir)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_step": DUALVIEW_PART_STEP8,
        "output_contract": DUALVIEW_PART_DIAGNOSTICS_CONTRACT,
        "splits": {str(split): dict(analysis) for split, analysis in analyses_by_split.items() if analysis},
        "case_exports": {},
    }
    case_rows_by_split = case_rows_by_split or {}
    for split, rows in case_rows_by_split.items():
        typed = [dict(row) for row in rows]
        payload["case_exports"][str(split)] = {
            "total_rows": len(typed),
            "fix_rows": sum(1 for row in typed if row.get("case_type") == "fix"),
            "break_rows": sum(1 for row in typed if row.get("case_type") == "break"),
        }
    json_path = diagnostics_dir / "residual_diagnostics.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    class_rows: list[dict[str, Any]] = []
    conf_rows: list[dict[str, Any]] = []
    correctness_rows: list[dict[str, Any]] = []
    change_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for split, analysis in payload["splits"].items():
        class_rows.extend(_prefixed_rows(split, analysis.get("gate_by_class", [])))
        conf_rows.extend(_prefixed_rows(split, analysis.get("gate_by_hlt_confidence_bucket", [])))
        correctness_rows.extend(_prefixed_rows(split, analysis.get("gate_by_hlt_correctness", [])))
        changes = {"split": split}
        changes.update(dict(analysis.get("prediction_changes", {})))
        for key, value in list(changes.items()):
            if isinstance(value, Mapping):
                changes[key] = json.dumps(value, sort_keys=True)
        change_rows.append(changes)
    for split, rows in case_rows_by_split.items():
        for row in rows:
            payload_row = dict(row)
            payload_row.setdefault("split", str(split))
            case_rows.append(payload_row)

    common_fields = [
        "split",
        "n",
        "fraction",
        "gate_mean",
        "gate_std",
        "hlt_confidence_mean",
        "delta_l2_mean",
        "residual_l2_mean",
        "hlt_accuracy",
        "final_accuracy",
        "prediction_change_fraction",
        "fix_count",
        "break_count",
    ]
    _write_csv(
        diagnostics_dir / "gate_by_class.csv",
        class_rows,
        fieldnames=["split", "class_index", "class_name", *common_fields[1:]],
    )
    _write_csv(
        diagnostics_dir / "gate_by_hlt_confidence.csv",
        conf_rows,
        fieldnames=["split", "bucket", "low", "high", *common_fields[1:]],
    )
    _write_csv(
        diagnostics_dir / "gate_by_hlt_correctness.csv",
        correctness_rows,
        fieldnames=["split", "bucket", *common_fields[1:]],
    )
    _write_csv(
        diagnostics_dir / "prediction_change_summary.csv",
        change_rows,
        fieldnames=[
            "split",
            "n_jets",
            "changed_count",
            "changed_fraction",
            "fix_count",
            "fix_fraction",
            "break_count",
            "break_fraction",
            "both_correct_count",
            "both_wrong_count",
            "hlt_only_correct_count",
            "final_only_correct_count",
            "net_correct_gain_count",
            "hlt_accuracy",
            "final_accuracy",
            "fix_gate_mean",
            "break_gate_mean",
            "fix_hlt_confidence_mean",
            "break_hlt_confidence_mean",
            "fix_final_pred_distribution",
            "break_final_pred_distribution",
            "fix_true_label_distribution",
            "break_true_label_distribution",
        ],
    )
    fix_rows = [row for row in case_rows if row.get("case_type") == "fix"]
    break_rows = [row for row in case_rows if row.get("case_type") == "break"]
    _write_csv(diagnostics_dir / "fix_break_cases.csv", case_rows, fieldnames=DUALVIEW_PART_CASE_FILE_COLUMNS)
    _write_csv(diagnostics_dir / "fix_cases.csv", fix_rows, fieldnames=DUALVIEW_PART_CASE_FILE_COLUMNS)
    _write_csv(diagnostics_dir / "break_cases.csv", break_rows, fieldnames=DUALVIEW_PART_CASE_FILE_COLUMNS)
    return {
        "residual_diagnostics_json": str(json_path),
        "gate_by_class_csv": str(diagnostics_dir / "gate_by_class.csv"),
        "gate_by_hlt_confidence_csv": str(diagnostics_dir / "gate_by_hlt_confidence.csv"),
        "gate_by_hlt_correctness_csv": str(diagnostics_dir / "gate_by_hlt_correctness.csv"),
        "prediction_change_summary_csv": str(diagnostics_dir / "prediction_change_summary.csv"),
        "fix_break_cases_csv": str(diagnostics_dir / "fix_break_cases.csv"),
        "fix_cases_csv": str(diagnostics_dir / "fix_cases.csv"),
        "break_cases_csv": str(diagnostics_dir / "break_cases.csv"),
    }


__all__ = [
    "DUALVIEW_PART_DIAGNOSTICS_CONTRACT",
    "DUALVIEW_PART_CASE_FILE_COLUMNS",
    "DUALVIEW_PART_HLT_CONFIDENCE_BUCKETS",
    "DUALVIEW_PART_STEP8",
    "build_residual_case_rows",
    "summarize_residual_behavior",
    "write_residual_diagnostics",
]
