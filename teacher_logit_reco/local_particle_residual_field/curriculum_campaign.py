"""Campaign utilities used by the Step 10 curriculum Slurm jobs."""

from __future__ import annotations

from dataclasses import replace
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import amp_autocast_context, require_torch, resolve_device, save_json
from jetclass_fresh.jetclass_data import LABEL_NAMES

from .curriculum import LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_CONTRACT
from .data import (
    LocalParticleResidualFieldDatasetConfig,
    load_local_particle_residual_field_dataset,
    make_local_particle_residual_field_loader,
)
from .fusion import (
    _collect_prediction_logits,
    _metrics_from_logits,
    load_local_residual_field_tagger_from_checkpoint,
)
from .tagger import ORACLE_RESIDUAL_FIELD_SOURCES


LOCAL_RESIDUAL_FIELD_ALPHA_EVAL_CONTRACT = "local_residual_field_fixed_consumer_alpha_eval_v1"
LOCAL_RESIDUAL_FIELD_STUDENT_SELECTION_CONTRACT = "local_residual_field_selected_curriculum_student_v1"
LOCAL_RESIDUAL_FIELD_PILOT_GATE_CONTRACT = "local_residual_field_curriculum_pilot_gate_v1"
ALPHA_EVAL_RUN_IDS = {
    "Ofull": "D_alpha_eval_Ofull",
    "Orobust_light": "D_alpha_eval_Orobust",
}
FIRST_STAGE_P_RUN_IDS = ("P2", "P4", "P7a", "P7b")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _finite(value: Any, *, name: str) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(output):
        raise ValueError(f"{name} must be finite")
    return output


def _curve_metric(curve: Mapping[str, Any], alpha: float, metric: str = "accuracy") -> float:
    for key, value in curve.items():
        try:
            matches = math.isclose(float(key), float(alpha), rel_tol=0.0, abs_tol=1.0e-8)
        except (TypeError, ValueError):
            continue
        if not matches:
            continue
        raw = value.get(metric) if isinstance(value, Mapping) else value
        return _finite(raw, name=f"alpha={alpha}.{metric}")
    raise ValueError(f"curve does not contain alpha={alpha}")


def _sorted_curve_points(curve: Mapping[str, Any]) -> list[tuple[float, Mapping[str, Any]]]:
    points: list[tuple[float, Mapping[str, Any]]] = []
    for raw_alpha, raw_metrics in curve.items():
        alpha = _finite(raw_alpha, name="curve alpha")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"curve alpha {alpha} is outside [0, 1]")
        metrics = dict(raw_metrics) if isinstance(raw_metrics, Mapping) else {"accuracy": raw_metrics}
        _finite(metrics.get("accuracy"), name=f"alpha={alpha}.accuracy")
        points.append((alpha, metrics))
    points.sort(key=lambda item: item[0])
    if not points or not math.isclose(points[0][0], 0.0, abs_tol=1.0e-8):
        raise ValueError("alpha curve must contain alpha=0.0")
    if len({alpha for alpha, _ in points}) != len(points):
        raise ValueError("alpha curve contains duplicate alpha values")
    return points


def curve_shape_metrics(curve: Mapping[str, Any], *, drop_tolerance: float = 0.002) -> dict[str, float]:
    points = _sorted_curve_points(curve)
    accuracies = np.asarray([float(metrics["accuracy"]) for _, metrics in points], dtype=np.float64)
    if len(accuracies) <= 1:
        monotonicity = 1.0
        roughness = 0.0
    else:
        deltas = np.diff(accuracies)
        monotonicity = float(np.mean(deltas >= -float(drop_tolerance)))
        roughness = float(np.mean(np.abs(np.diff(deltas)))) if len(deltas) > 1 else 0.0
    return {
        "monotonicity_score": monotonicity,
        "smoothing_score": 1.0 / (1.0 + 100.0 * roughness),
        "roughness": roughness,
    }


def _teacher_distribution_metrics(logits: np.ndarray) -> dict[str, float]:
    shifted = logits.astype(np.float64) - np.max(logits.astype(np.float64), axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1.0e-12)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1.0e-12))).sum(axis=1)
    return {
        "teacher_entropy": float(np.mean(entropy)),
        "teacher_confidence": float(np.mean(np.max(probabilities, axis=1))),
    }


def evaluate_fixed_consumer_alpha_curve(
    *,
    checkpoint: str | Path,
    consumer_id: str,
    output_dir: str | Path,
    hlt_cache_dir: str | Path,
    target_cache_dir: str | Path,
    manifest_path: str | Path | None,
    alphas: Sequence[float] = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0),
    splits: Sequence[str] = ("model_val", "stack_val"),
    baseline_report: str | Path | None = None,
    batch_size: int = 128,
    num_workers: int = 0,
    max_jets: int | None = None,
    device: str = "auto",
    amp: bool = True,
    verify_hash: bool = True,
    require_manifest_match: bool = True,
) -> dict[str, Any]:
    """Evaluate one frozen oracle consumer over scaled true fields."""

    if consumer_id not in ALPHA_EVAL_RUN_IDS:
        raise ValueError(f"consumer_id must be one of {tuple(ALPHA_EVAL_RUN_IDS)}, got {consumer_id!r}")
    normalized_alphas = tuple(sorted({_finite(alpha, name="alpha") for alpha in alphas}))
    if not normalized_alphas or normalized_alphas[0] < 0.0 or normalized_alphas[-1] > 1.0:
        raise ValueError("alphas must be non-empty and contained in [0, 1]")
    if 0.0 not in normalized_alphas or 0.25 not in normalized_alphas or 0.75 not in normalized_alphas:
        raise ValueError("the first-stage alpha sweep requires 0.0, 0.25, and 0.75")
    normalized_splits = tuple(str(split) for split in splits)
    if set(normalized_splits) != {"model_val", "stack_val"}:
        raise ValueError("fixed-consumer selection diagnostics must evaluate exactly model_val and stack_val")

    torch = require_torch()
    selected_device = resolve_device(device)
    amp_enabled = bool(amp and getattr(selected_device, "type", str(selected_device)) == "cuda")
    model, payload = load_local_residual_field_tagger_from_checkpoint(checkpoint, device=selected_device)
    if not hasattr(model, "config") or model.config.field_source not in ORACLE_RESIDUAL_FIELD_SOURCES:
        raise ValueError("fixed-consumer alpha evaluation requires an oracle residual-field checkpoint")
    teacher_config_path = Path(checkpoint).with_name("teacher_config.json")
    if teacher_config_path.is_file():
        teacher_config = _load_json(teacher_config_path)
        configured_id = str(teacher_config.get("teacher_id") or "")
        if configured_id and configured_id != consumer_id:
            raise ValueError(f"teacher_config consumer {configured_id!r} != requested {consumer_id!r}")

    baseline = _load_json(baseline_report) if baseline_report else {}
    output: dict[str, dict[str, Any]] = {}
    dataset_metadata: dict[str, Any] = {}
    with torch.no_grad():
        for split in normalized_splits:
            dataset = load_local_particle_residual_field_dataset(
                LocalParticleResidualFieldDatasetConfig(
                    hlt_cache_dir=str(hlt_cache_dir),
                    target_cache_dir=str(target_cache_dir),
                    split=split,
                    manifest_path=None if not manifest_path else str(manifest_path),
                    max_jets=max_jets,
                    include_oracle_fields=True,
                    verify_hash=bool(verify_hash),
                    require_manifest_match=bool(require_manifest_match),
                )
            )
            loader = make_local_particle_residual_field_loader(
                dataset,
                batch_size=int(batch_size),
                shuffle=False,
                num_workers=int(num_workers),
                seed=0,
            )
            dataset_metadata[split] = dataset.metadata
            split_curve: dict[str, Any] = {}
            baseline_metrics = baseline.get("best_model_val" if split == "model_val" else "stack_val")
            baseline_accuracy = (
                _finite(baseline_metrics.get("accuracy"), name=f"A0.{split}.accuracy")
                if isinstance(baseline_metrics, Mapping) and baseline_metrics.get("accuracy") is not None
                else None
            )
            for alpha in normalized_alphas:
                model.config = replace(model.config, oracle_field_alpha=float(alpha))
                model.eval()
                logits, labels = _collect_prediction_logits(
                    model,
                    loader,
                    device=selected_device,
                    amp_enabled=amp_enabled,
                )
                metrics = _metrics_from_logits(logits, labels, label_names=LABEL_NAMES)
                metrics.update(_teacher_distribution_metrics(logits))
                metrics["alpha"] = float(alpha)
                metrics["consumer_id"] = consumer_id
                metrics["split"] = split
                metrics["accuracy_gap_vs_A0"] = (
                    None if baseline_accuracy is None else float(metrics["accuracy"]) - baseline_accuracy
                )
                split_curve[f"{alpha:g}"] = metrics
            full_accuracy = _curve_metric(split_curve, 1.0) if 1.0 in normalized_alphas else None
            for metrics in split_curve.values():
                metrics["accuracy_gap_vs_alpha_1"] = (
                    None if full_accuracy is None else float(metrics["accuracy"]) - full_accuracy
                )
            shape = curve_shape_metrics(split_curve)
            for metrics in split_curve.values():
                metrics.update(shape)
            output[split] = split_curve

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = ALPHA_EVAL_RUN_IDS[consumer_id]
    report = {
        "ok": True,
        "contract": LOCAL_RESIDUAL_FIELD_ALPHA_EVAL_CONTRACT,
        "run_id": run_id,
        "canonical_config_id": (
            f"D_alpha-response_consumer-{consumer_id}_alphas-" + "-".join(f"{alpha:.2f}" for alpha in normalized_alphas)
        ),
        "consumer_id": consumer_id,
        "checkpoint": str(checkpoint),
        "checkpoint_hash": _sha256_file(checkpoint),
        "alphas": list(normalized_alphas),
        "selection_primary_split": "model_val",
        "selection_confirmation_split": "stack_val",
        "model_val_alpha_curve": output["model_val"],
        "stack_val_alpha_curve": output["stack_val"],
        "dataset_metadata": dataset_metadata,
        "runtime_inputs": "HLT_plus_true_fields",
        "uses_true_fields": True,
        "uses_offline_particles": False,
        "uses_teacher_logits_at_runtime": False,
        "deployable": False,
        "selection_allowed": True,
        "final_test_evaluated": False,
        "checkpoint_payload_contract": payload.get("contract") if isinstance(payload, Mapping) else None,
    }
    save_json(output_dir / "run_report.json", report)
    rows: list[dict[str, Any]] = []
    for split in normalized_splits:
        rows.extend(dict(metrics) for metrics in output[split].values())
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "alpha_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return report


def _stable_endpoint(
    curve: Mapping[str, Any],
    *,
    minimum_gain: float,
    drop_tolerance: float,
) -> tuple[float, float]:
    points = _sorted_curve_points(curve)
    baseline = float(points[0][1]["accuracy"])
    stable: list[tuple[float, float]] = []
    previous = baseline
    for alpha, metrics in points[1:]:
        accuracy = float(metrics["accuracy"])
        if accuracy + float(drop_tolerance) >= previous and accuracy - baseline >= float(minimum_gain):
            stable.append((alpha, accuracy))
        previous = accuracy
    if not stable:
        raise ValueError("consumer has no useful stable non-zero alpha endpoint on model_val")
    return stable[-1]


def select_curriculum_consumer(
    reports: Sequence[str | Path | Mapping[str, Any]],
    *,
    output_path: str | Path,
    minimum_gain: float = 0.002,
    close_accuracy_tolerance: float = 0.001,
    drop_tolerance: float = 0.002,
    stack_brittleness_tolerance: float = 0.003,
) -> dict[str, Any]:
    """Select the consumer using model-val first and stack-val only for stability/ties."""

    loaded = [dict(item) if isinstance(item, Mapping) else _load_json(item) for item in reports]
    by_consumer = {str(report.get("consumer_id") or ""): report for report in loaded}
    if set(by_consumer) != set(ALPHA_EVAL_RUN_IDS):
        raise ValueError("selector requires exactly D_alpha_eval_Ofull and D_alpha_eval_Orobust")
    candidates: list[dict[str, Any]] = []
    useful_weak_response = False
    for consumer_id in ALPHA_EVAL_RUN_IDS:
        report = by_consumer[consumer_id]
        if report.get("ok") is not True:
            raise ValueError(f"alpha diagnostic for {consumer_id} is not ok")
        model_curve = report.get("model_val_alpha_curve")
        stack_curve = report.get("stack_val_alpha_curve")
        if not isinstance(model_curve, Mapping) or not isinstance(stack_curve, Mapping):
            raise ValueError(f"alpha diagnostic for {consumer_id} is missing model/stack curves")
        baseline = _curve_metric(model_curve, 0.0)
        weak_gains = []
        for alpha in (0.25, 0.75):
            try:
                weak_gains.append(_curve_metric(model_curve, alpha) - baseline)
            except ValueError:
                pass
        useful_weak_response = useful_weak_response or any(gain >= float(minimum_gain) for gain in weak_gains)
        try:
            endpoint, model_accuracy = _stable_endpoint(
                model_curve,
                minimum_gain=float(minimum_gain),
                drop_tolerance=float(drop_tolerance),
            )
        except ValueError:
            continue
        stack_gain = _curve_metric(stack_curve, endpoint) - _curve_metric(stack_curve, 0.0)
        model_gain = model_accuracy - baseline
        brittle = stack_gain < model_gain - float(stack_brittleness_tolerance)
        shape = curve_shape_metrics(model_curve, drop_tolerance=float(drop_tolerance))
        candidates.append(
            {
                "consumer_id": consumer_id,
                "endpoint": endpoint,
                "model_accuracy": model_accuracy,
                "model_gain": model_gain,
                "stack_gain": stack_gain,
                "brittle": brittle,
                **shape,
                "report": report,
            }
        )
    if not useful_weak_response:
        raise ValueError(
            "pilot hard stop: neither alpha diagnostic improves at alpha=0.25 or 0.75 on model_val"
        )
    if not candidates:
        raise ValueError("pilot hard stop: neither consumer has a useful stable model_val endpoint")
    candidates.sort(key=lambda item: float(item["model_accuracy"]), reverse=True)
    best = candidates[0]
    close = [
        item for item in candidates
        if float(best["model_accuracy"]) - float(item["model_accuracy"]) <= float(close_accuracy_tolerance)
    ]
    if len(close) > 1:
        non_brittle = [item for item in close if not bool(item["brittle"])] or close
        non_brittle.sort(
            key=lambda item: (
                float(item["monotonicity_score"]),
                float(item["smoothing_score"]),
                float(item["stack_gain"]),
                float(item["model_accuracy"]),
            ),
            reverse=True,
        )
        best = non_brittle[0]
    selected_report = best["report"]
    selection_reason = (
        "model_val-primary strongest stable endpoint; close model-val responses were resolved by "
        "monotonicity/smoothness and stack_val stability confirmation"
    )
    artifact = {
        "contract": LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_CONTRACT,
        "selected_consumer_id": best["consumer_id"],
        "selected_alpha_endpoint": float(best["endpoint"]),
        "selection_source": "D_alpha_eval_Ofull,D_alpha_eval_Orobust",
        "selection_reason": selection_reason,
        "model_val_alpha_curve": selected_report["model_val_alpha_curve"],
        "stack_val_alpha_curve": selected_report["stack_val_alpha_curve"],
        "selection_primary_split": "model_val",
        "selection_confirmation_split": "stack_val",
        "candidate_summary": [
            {key: value for key, value in candidate.items() if key != "report"}
            for candidate in candidates
        ],
    }
    save_json(output_path, artifact)
    return artifact


def select_best_curriculum_student(
    reports: Sequence[str | Path | Mapping[str, Any]],
    *,
    output_path: str | Path,
    close_accuracy_tolerance: float = 0.001,
) -> dict[str, Any]:
    """Select the G0 P member on model_val; stack_val only breaks close ties."""

    candidates: list[dict[str, Any]] = []
    for item in reports:
        report = dict(item) if isinstance(item, Mapping) else _load_json(item)
        run_id = str(report.get("run_id") or "")
        if run_id not in FIRST_STAGE_P_RUN_IDS:
            raise ValueError(f"G0 selection accepts only {FIRST_STAGE_P_RUN_IDS}; got {run_id!r}")
        if report.get("ok") is not True or report.get("deployable") is not True:
            raise ValueError(f"curriculum student {run_id} is not a successful deployable run")
        if str(report.get("runtime_inputs") or "") != "HLT_only" or any(
            bool(report.get(key)) for key in (
                "uses_true_fields", "uses_offline_particles", "uses_teacher_logits_at_runtime"
            )
        ):
            raise ValueError(f"curriculum student {run_id} has privileged inference inputs")
        model_metrics = report.get("best_model_val")
        stack_metrics = report.get("stack_val")
        if not isinstance(model_metrics, Mapping) or not isinstance(stack_metrics, Mapping):
            raise ValueError(f"curriculum student {run_id} is missing model_val/stack_val metrics")
        candidates.append(
            {
                "run_id": run_id,
                "model_val_accuracy": _finite(model_metrics.get("accuracy"), name=f"{run_id}.model_val.accuracy"),
                "stack_val_accuracy": _finite(stack_metrics.get("accuracy"), name=f"{run_id}.stack_val.accuracy"),
                "checkpoint": report.get("checkpoint"),
                "checkpoint_hash": report.get("checkpoint_hash"),
                "selected_consumer_id": report.get("selected_consumer_id"),
                "selected_alpha_endpoint": report.get("selected_alpha_endpoint"),
            }
        )
    if {candidate["run_id"] for candidate in candidates} != set(FIRST_STAGE_P_RUN_IDS):
        raise ValueError(f"G0 selection requires all first-stage P runs: {FIRST_STAGE_P_RUN_IDS}")
    candidates.sort(key=lambda item: float(item["model_val_accuracy"]), reverse=True)
    model_best = float(candidates[0]["model_val_accuracy"])
    close = [
        candidate for candidate in candidates
        if model_best - float(candidate["model_val_accuracy"]) <= float(close_accuracy_tolerance)
    ]
    close.sort(key=lambda item: (float(item["stack_val_accuracy"]), float(item["model_val_accuracy"])), reverse=True)
    selected = close[0]
    artifact = {
        "contract": LOCAL_RESIDUAL_FIELD_STUDENT_SELECTION_CONTRACT,
        "selected_run_id": selected["run_id"],
        "selection_primary_split": "model_val",
        "selection_confirmation_split": "stack_val",
        "selection_reason": "highest model_val accuracy; stack_val used only to break a close model_val tie",
        "selected": selected,
        "candidates": candidates,
        "runtime_inputs": "HLT_only",
        "deployable": True,
    }
    save_json(output_path, artifact)
    return artifact


def evaluate_pilot_gate(
    report_dir: str | Path,
    *,
    student_uplift_threshold: float = 0.003,
    fusion_uplift_threshold: float = 0.005,
    minimum_validation_coverage: float = 0.99,
    maximum_nonfinite_fraction: float = 0.01,
) -> dict[str, Any]:
    """Require the Step 10 high-data promotion criteria from frozen report tables."""

    report_dir = Path(report_dir)
    report = _load_json(report_dir / "run_report.json")
    if report.get("ok") is not True:
        raise ValueError("pilot run_report.json is not ok")
    provenance = _load_json(report_dir / "provenance_audit.json")
    if provenance.get("ok") is not True:
        raise ValueError("pilot provenance_audit.json is not ok")
    with (report_dir / "deployable_leaderboard.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (report_dir / "curriculum_student_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        curriculum_rows = list(csv.DictReader(handle))
    validation_rows = [
        row for row in curriculum_rows
        if row.get("run_id") in FIRST_STAGE_P_RUN_IDS and row.get("split") == "stack_val"
    ]
    if {row.get("run_id") for row in validation_rows} != set(FIRST_STAGE_P_RUN_IDS):
        raise ValueError("pilot gate requires stack_val diagnostics for P2, P4, P7a, and P7b")
    for row in validation_rows:
        run_id = str(row.get("run_id"))
        valid_fraction = row.get("valid_fraction")
        if valid_fraction in (None, ""):
            seen = _finite(row.get("n_jets"), name=f"{run_id}.n_jets")
            attempted = _finite(row.get("attempted_jets"), name=f"{run_id}.attempted_jets")
            valid_fraction = seen / attempted if attempted > 0.0 else 0.0
        else:
            valid_fraction = _finite(valid_fraction, name=f"{run_id}.valid_fraction")
        nonfinite_fraction = _finite(
            row.get("nonfinite_fraction"), name=f"{run_id}.nonfinite_fraction"
        )
        if valid_fraction < float(minimum_validation_coverage):
            raise ValueError(f"pilot validation coverage for {run_id} is below {minimum_validation_coverage}")
        if nonfinite_fraction > float(maximum_nonfinite_fraction):
            raise ValueError(f"pilot nonfinite fraction for {run_id} exceeds {maximum_nonfinite_fraction}")
        if str(row.get("valid_for_selection") or "").strip().lower() in {"false", "0", "no"}:
            raise ValueError(f"pilot stack_val metrics for {run_id} were rejected for selection")
    a0 = [row for row in rows if row.get("run_id") == "A0" and row.get("split") == "stack_val"]
    if len(a0) != 1:
        raise ValueError("pilot leaderboard must contain exactly one A0 stack_val row")
    baseline = _finite(a0[0].get("accuracy"), name="A0.stack_val.accuracy")
    students = [
        row for row in rows
        if row.get("run_id") in FIRST_STAGE_P_RUN_IDS and row.get("split") == "stack_val"
    ]
    fusions = [row for row in rows if row.get("group") == "G0" and row.get("split") == "stack_val"]
    best_student = max((_finite(row.get("accuracy"), name="student accuracy") for row in students), default=-math.inf)
    best_fusion = max((_finite(row.get("accuracy"), name="fusion accuracy") for row in fusions), default=-math.inf)
    student_uplift = best_student - baseline
    fusion_uplift = best_fusion - baseline
    ok = student_uplift >= float(student_uplift_threshold) or fusion_uplift >= float(fusion_uplift_threshold)
    result = {
        "ok": ok,
        "contract": LOCAL_RESIDUAL_FIELD_PILOT_GATE_CONTRACT,
        "baseline_A0_stack_val_accuracy": baseline,
        "best_student_stack_val_accuracy": None if not math.isfinite(best_student) else best_student,
        "best_fusion_stack_val_accuracy": None if not math.isfinite(best_fusion) else best_fusion,
        "student_uplift": None if not math.isfinite(student_uplift) else student_uplift,
        "fusion_uplift": None if not math.isfinite(fusion_uplift) else fusion_uplift,
        "student_uplift_threshold": float(student_uplift_threshold),
        "fusion_uplift_threshold": float(fusion_uplift_threshold),
        "minimum_validation_coverage": float(minimum_validation_coverage),
        "maximum_nonfinite_fraction": float(maximum_nonfinite_fraction),
    }
    if not ok:
        raise ValueError(
            f"pilot uplift gate failed: student={student_uplift:.6g} fusion={fusion_uplift:.6g}"
        )
    return result


__all__ = [
    "ALPHA_EVAL_RUN_IDS",
    "FIRST_STAGE_P_RUN_IDS",
    "LOCAL_RESIDUAL_FIELD_ALPHA_EVAL_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_STUDENT_SELECTION_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_PILOT_GATE_CONTRACT",
    "curve_shape_metrics",
    "evaluate_fixed_consumer_alpha_curve",
    "evaluate_pilot_gate",
    "select_best_curriculum_student",
    "select_curriculum_consumer",
]
