"""Leakage-safe L0--L5 late fusion for the P7b fusion campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.fusion import PredictionBlock, load_prediction_block, validate_prediction_alignment
from jetclass_fresh.hlt_baseline import require_torch, resolve_device
from jetclass_fresh.jetclass_data import LABEL_NAMES

from .fusion_atomic import publish_temporary_file
from .fusion_campaign import (
    FUSION_FAMILY_LATE,
    FUSION_FIT_SPLIT,
    FUSION_SELECTION_SPLIT,
    default_fusion_candidate_specs,
    default_fusion_group_specs,
    stable_fusion_json_hash,
)
from .fusion_features import require_development_prediction_sources
from .fusion_metrics import (
    local_residual_field_binary_projection_metrics,
    local_residual_field_multiclass_metrics,
)


LOCAL_RESIDUAL_FIELD_LATE_FUSION_FIT_CONTRACT = "local_residual_field_late_fusion_fit_v1"
LATE_FUSION_SPLITS = (FUSION_FIT_SPLIT, FUSION_SELECTION_SPLIT)
G0_DEVELOPMENT_REFERENCE = {
    FUSION_FIT_SPLIT: {"n_jets": 300_000, "accuracy": 0.7566766666666667, "cross_entropy": 0.6766777643455919},
    FUSION_SELECTION_SPLIT: {"n_jets": 150_000, "accuracy": 0.75762, "cross_entropy": 0.673884991266452},
}


def _softmax(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    probabilities = _softmax(logits)
    return float(-np.mean(np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1.0e-12, 1.0))))


def _validate_pair(block_a: PredictionBlock, block_b: PredictionBlock) -> None:
    validate_prediction_alignment([block_a, block_b])
    if block_a.split != block_b.split:
        raise ValueError("fusion member blocks must belong to the same split")
    if block_a.logits.shape[1] != len(LABEL_NAMES):
        raise ValueError("fusion logits do not use the locked ten-class schema")


def _golden_temperature(logits: np.ndarray, labels: np.ndarray, bounds: tuple[float, float]) -> float:
    lower, upper = map(float, bounds)
    if not (0.0 < lower < upper):
        raise ValueError("temperature bounds must be finite, positive, and ordered")
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    f_left = _cross_entropy(logits / left, labels)
    f_right = _cross_entropy(logits / right, labels)
    for _ in range(80):
        if f_left <= f_right:
            upper, right, f_right = right, left, f_left
            left = upper - ratio * (upper - lower)
            f_left = _cross_entropy(logits / left, labels)
        else:
            lower, left, f_left = left, right, f_right
            right = lower + ratio * (upper - lower)
            f_right = _cross_entropy(logits / right, labels)
    candidates = (float(bounds[0]), 0.5 * (lower + upper), float(bounds[1]))
    return min(candidates, key=lambda temperature: _cross_entropy(logits / temperature, labels))


def _classwise_weights(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float,
    steps: int = 600,
) -> np.ndarray:
    """Projected Adam for ten convex weights, fit only on stack-train."""

    weights = np.full(logits_a.shape[1], 0.5, dtype=np.float64)
    first = np.zeros_like(weights)
    second = np.zeros_like(weights)
    one_hot = np.eye(logits_a.shape[1], dtype=np.float64)[labels]
    difference = np.asarray(logits_a, dtype=np.float64) - np.asarray(logits_b, dtype=np.float64)
    for step in range(1, int(steps) + 1):
        fused = logits_b + weights[None, :] * difference
        gradient = np.mean((_softmax(fused) - one_hot) * difference, axis=0)
        gradient += 2.0 * float(l2) * (weights - 0.5)
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        corrected_first = first / (1.0 - 0.9**step)
        corrected_second = second / (1.0 - 0.999**step)
        weights = np.clip(weights - 0.03 * corrected_first / (np.sqrt(corrected_second) + 1.0e-8), 0.0, 1.0)
    return weights


def _stacker_features(logits_a: np.ndarray, logits_b: np.ndarray, mode: str) -> np.ndarray:
    if mode == "logits":
        return np.concatenate([logits_a, logits_b], axis=1).astype(np.float32)
    probabilities = np.concatenate([_softmax(logits_a), _softmax(logits_b)], axis=1)
    if mode == "probabilities":
        return probabilities.astype(np.float32)
    if mode == "logits+probabilities":
        return np.concatenate([logits_a, logits_b, probabilities], axis=1).astype(np.float32)
    raise ValueError(f"unsupported stacker feature mode {mode!r}")


def _fit_linear_stacker(
    train_x: np.ndarray,
    labels: np.ndarray,
    *,
    c_value: float,
    device: str,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torch = require_torch()
    active_device = resolve_device(device)
    mean = np.mean(train_x, axis=0, dtype=np.float64).astype(np.float32)
    centered = np.asarray(train_x, dtype=np.float32) - mean
    x_tensor = torch.as_tensor(centered, device=active_device)
    y_tensor = torch.as_tensor(labels, dtype=torch.long, device=active_device)
    layer = torch.nn.Linear(centered.shape[1], len(LABEL_NAMES), device=active_device)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()
    optimizer = torch.optim.LBFGS(
        layer.parameters(), lr=1.0, max_iter=int(max_steps), tolerance_grad=1.0e-8,
        tolerance_change=1.0e-10, history_size=20, line_search_fn="strong_wolfe",
    )
    penalty = 1.0 / max(2.0 * float(c_value) * max(len(labels), 1), 1.0e-12)

    def closure() -> Any:
        optimizer.zero_grad(set_to_none=True)
        output = layer(x_tensor)
        loss = torch.nn.functional.cross_entropy(output, y_tensor)
        loss = loss + penalty * torch.sum(layer.weight * layer.weight)
        loss.backward()
        return loss

    optimizer.step(closure)
    weight = layer.weight.detach().cpu().numpy().astype(np.float64)
    bias = layer.bias.detach().cpu().numpy().astype(np.float64)
    return mean.astype(np.float64), weight, bias


def _predict_stacker(x: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
    mean = np.asarray(parameters["feature_mean"], dtype=np.float64)
    weight = np.asarray(parameters["weight"], dtype=np.float64)
    bias = np.asarray(parameters["bias"], dtype=np.float64)
    return (np.asarray(x, dtype=np.float64) - mean) @ weight.T + bias


@dataclass(frozen=True)
class LateFusionFitResult:
    candidate_id: str
    parameters: Mapping[str, Any]
    train_logits: np.ndarray
    validation_logits: np.ndarray
    train_cross_entropy: float
    validation_cross_entropy: float


def fit_late_fusion_candidate(
    candidate_id: str,
    train_a: PredictionBlock,
    train_b: PredictionBlock,
    validation_a: PredictionBlock,
    validation_b: PredictionBlock,
    *,
    device: str = "cpu",
    stacker_max_steps: int = 80,
    classwise_steps: int = 600,
) -> LateFusionFitResult:
    """Fit one locked L0--L5 candidate using only explicit development blocks."""

    if (
        train_a.split != FUSION_FIT_SPLIT
        or train_b.split != FUSION_FIT_SPLIT
        or validation_a.split != FUSION_SELECTION_SPLIT
        or validation_b.split != FUSION_SELECTION_SPLIT
    ):
        raise ValueError(f"late fusion accepts only {LATE_FUSION_SPLITS}, in that order")
    _validate_pair(train_a, train_b)
    _validate_pair(validation_a, validation_b)
    specs = {spec.candidate_id: spec for spec in default_fusion_candidate_specs() if spec.family == FUSION_FAMILY_LATE}
    if candidate_id not in specs:
        raise ValueError(f"candidate_id must identify a locked late candidate, got {candidate_id!r}")
    spec = specs[candidate_id]
    za, zb = np.asarray(train_a.logits, dtype=np.float64), np.asarray(train_b.logits, dtype=np.float64)
    va, vb = np.asarray(validation_a.logits, dtype=np.float64), np.asarray(validation_b.logits, dtype=np.float64)
    labels, val_labels = train_a.labels, validation_a.labels
    parameters: dict[str, Any] = {}

    if candidate_id == "L0_mean_logits":
        train_logits, val_logits = 0.5 * (za + zb), 0.5 * (va + vb)
    elif candidate_id == "L1_mean_probs":
        train_logits = np.log(np.clip(0.5 * (_softmax(za) + _softmax(zb)), 1.0e-12, 1.0))
        val_logits = np.log(np.clip(0.5 * (_softmax(va) + _softmax(vb)), 1.0e-12, 1.0))
    elif candidate_id == "L2_temp_mean_logits":
        bounds = tuple(spec.hyperparameter_grid["temperature_bounds"])
        temperatures = [_golden_temperature(za, labels, bounds), _golden_temperature(zb, labels, bounds)]
        parameters = {"temperatures": temperatures, "temperature_bounds": list(bounds)}
        train_logits = 0.5 * (za / temperatures[0] + zb / temperatures[1])
        val_logits = 0.5 * (va / temperatures[0] + vb / temperatures[1])
    elif candidate_id == "L3_scalar_simplex_logits":
        rows = []
        for weight in spec.hyperparameter_grid["weight_grid"]:
            logits = float(weight) * za + (1.0 - float(weight)) * zb
            rows.append((_cross_entropy(logits, labels), float(weight)))
        _, selected = min(rows)
        parameters = {"weight": selected, "weight_grid": spec.hyperparameter_grid["weight_grid"]}
        train_logits = selected * za + (1.0 - selected) * zb
        val_logits = selected * va + (1.0 - selected) * vb
    elif candidate_id == "L4_classwise_simplex_logits":
        trials = []
        for l2 in spec.hyperparameter_grid["l2"]:
            weights = _classwise_weights(za, zb, labels, l2=float(l2), steps=classwise_steps)
            val_candidate = weights[None, :] * va + (1.0 - weights[None, :]) * vb
            trials.append((_cross_entropy(val_candidate, val_labels), float(l2), weights))
        _, selected_l2, selected_weights = min(trials, key=lambda row: (row[0], row[1]))
        parameters = {
            "weights": selected_weights.tolist(), "selected_l2": selected_l2,
            "l2_grid": spec.hyperparameter_grid["l2"], "selection_source": FUSION_SELECTION_SPLIT,
        }
        train_logits = selected_weights[None, :] * za + (1.0 - selected_weights[None, :]) * zb
        val_logits = selected_weights[None, :] * va + (1.0 - selected_weights[None, :]) * vb
    else:
        trials = []
        for mode in spec.hyperparameter_grid["feature_modes"]:
            train_x = _stacker_features(za, zb, mode)
            val_x = _stacker_features(va, vb, mode)
            for c_value in spec.hyperparameter_grid["C"]:
                mean, weight, bias = _fit_linear_stacker(
                    train_x, labels, c_value=float(c_value), device=device, max_steps=stacker_max_steps,
                )
                candidate_parameters = {
                    "feature_mode": mode, "C": float(c_value), "feature_mean": mean.tolist(),
                    "weight": weight.tolist(), "bias": bias.tolist(),
                }
                candidate_logits = _predict_stacker(val_x, candidate_parameters)
                trials.append((_cross_entropy(candidate_logits, val_labels), mode, float(c_value), candidate_parameters))
        _, selected_mode, selected_c, parameters = min(trials, key=lambda row: (row[0], row[1], row[2]))
        parameters = {
            **parameters, "feature_modes": spec.hyperparameter_grid["feature_modes"],
            "C_grid": spec.hyperparameter_grid["C"], "selection_source": FUSION_SELECTION_SPLIT,
        }
        train_logits = _predict_stacker(_stacker_features(za, zb, selected_mode), parameters)
        val_logits = _predict_stacker(_stacker_features(va, vb, selected_mode), parameters)

    return LateFusionFitResult(
        candidate_id=candidate_id,
        parameters=parameters,
        train_logits=np.asarray(train_logits, dtype=np.float32),
        validation_logits=np.asarray(val_logits, dtype=np.float32),
        train_cross_entropy=_cross_entropy(train_logits, labels),
        validation_cross_entropy=_cross_entropy(val_logits, val_labels),
    )


def apply_late_fusion_candidate(
    candidate_id: str,
    parameters: Mapping[str, Any],
    logits_a: np.ndarray,
    logits_b: np.ndarray,
) -> np.ndarray:
    """Apply a frozen L0--L5 recipe without fitting or split access."""

    za = np.asarray(logits_a, dtype=np.float64)
    zb = np.asarray(logits_b, dtype=np.float64)
    if za.shape != zb.shape or za.ndim != 2 or za.shape[1] != len(LABEL_NAMES):
        raise ValueError("late-fusion inference requires aligned ten-class member logits")
    if candidate_id == "L0_mean_logits":
        output = 0.5 * (za + zb)
    elif candidate_id == "L1_mean_probs":
        output = np.log(np.clip(0.5 * (_softmax(za) + _softmax(zb)), 1.0e-12, 1.0))
    elif candidate_id == "L2_temp_mean_logits":
        temperatures = tuple(float(value) for value in parameters.get("temperatures", ()))
        if len(temperatures) != 2 or any(value <= 0.0 for value in temperatures):
            raise ValueError("L2 requires two positive frozen temperatures")
        output = 0.5 * (za / temperatures[0] + zb / temperatures[1])
    elif candidate_id == "L3_scalar_simplex_logits":
        weight = float(parameters["weight"])
        if not 0.0 <= weight <= 1.0:
            raise ValueError("L3 frozen weight is not convex")
        output = weight * za + (1.0 - weight) * zb
    elif candidate_id == "L4_classwise_simplex_logits":
        weights = np.asarray(parameters["weights"], dtype=np.float64)
        if weights.shape != (len(LABEL_NAMES),) or np.any((weights < 0.0) | (weights > 1.0)):
            raise ValueError("L4 frozen classwise weights are not convex ten-class weights")
        output = weights[None, :] * za + (1.0 - weights[None, :]) * zb
    elif candidate_id == "L5_linear_stacker":
        mode = str(parameters["feature_mode"])
        output = _predict_stacker(_stacker_features(za, zb, mode), parameters)
    else:
        raise ValueError(f"candidate is not a locked late-fusion recipe: {candidate_id!r}")
    if not np.isfinite(output).all():
        raise ValueError("late-fusion inference produced non-finite logits")
    return np.asarray(output, dtype=np.float32)


def replay_late_fusion_recipe(
    candidate_id: str,
    source_parameters: Mapping[str, Any],
    train_a: PredictionBlock,
    train_b: PredictionBlock,
    validation_a: PredictionBlock,
    validation_b: PredictionBlock,
    *,
    device: str = "cpu",
    stacker_max_steps: int = 80,
    classwise_steps: int = 600,
) -> LateFusionFitResult:
    """Refit one frozen method recipe on a new group without hyperparameter search."""

    if (
        train_a.split != FUSION_FIT_SPLIT or train_b.split != FUSION_FIT_SPLIT
        or validation_a.split != FUSION_SELECTION_SPLIT or validation_b.split != FUSION_SELECTION_SPLIT
    ):
        raise ValueError("recipe replay accepts only stack_train then stack_val")
    _validate_pair(train_a, train_b)
    _validate_pair(validation_a, validation_b)
    za, zb = np.asarray(train_a.logits, dtype=np.float64), np.asarray(train_b.logits, dtype=np.float64)
    va, vb = np.asarray(validation_a.logits, dtype=np.float64), np.asarray(validation_b.logits, dtype=np.float64)
    labels = train_a.labels
    parameters: dict[str, Any] = {}
    if candidate_id in {"L0_mean_logits", "L1_mean_probs"}:
        parameters = {}
    elif candidate_id == "L2_temp_mean_logits":
        bounds = tuple(float(value) for value in source_parameters["temperature_bounds"])
        parameters = {
            "temperatures": [_golden_temperature(za, labels, bounds), _golden_temperature(zb, labels, bounds)],
            "temperature_bounds": list(bounds),
        }
    elif candidate_id == "L3_scalar_simplex_logits":
        grid = [float(value) for value in source_parameters["weight_grid"]]
        selected = min(grid, key=lambda weight: _cross_entropy(weight * za + (1.0 - weight) * zb, labels))
        parameters = {"weight": selected, "weight_grid": grid}
    elif candidate_id == "L4_classwise_simplex_logits":
        selected_l2 = float(source_parameters["selected_l2"])
        weights = _classwise_weights(za, zb, labels, l2=selected_l2, steps=classwise_steps)
        parameters = {
            "weights": weights.tolist(), "selected_l2": selected_l2,
            "l2_grid": list(source_parameters.get("l2_grid", [selected_l2])),
            "selection_source": "F_method_recipe_replay",
        }
    elif candidate_id == "L5_linear_stacker":
        mode = str(source_parameters["feature_mode"])
        c_value = float(source_parameters["C"])
        mean, weight, bias = _fit_linear_stacker(
            _stacker_features(za, zb, mode), labels,
            c_value=c_value, device=device, max_steps=stacker_max_steps,
        )
        parameters = {
            "feature_mode": mode, "C": c_value, "feature_mean": mean.tolist(),
            "weight": weight.tolist(), "bias": bias.tolist(),
            "feature_modes": list(source_parameters.get("feature_modes", [mode])),
            "C_grid": list(source_parameters.get("C_grid", [c_value])),
            "selection_source": "F_method_recipe_replay",
        }
    else:
        raise ValueError(f"recipe replay requires a locked late candidate, got {candidate_id!r}")
    train_logits = apply_late_fusion_candidate(candidate_id, parameters, za, zb)
    validation_logits = apply_late_fusion_candidate(candidate_id, parameters, va, vb)
    return LateFusionFitResult(
        candidate_id=candidate_id, parameters=parameters,
        train_logits=train_logits, validation_logits=validation_logits,
        train_cross_entropy=_cross_entropy(train_logits, train_a.labels),
        validation_cross_entropy=_cross_entropy(validation_logits, validation_a.labels),
    )


def _atomic_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        publish_temporary_file(temporary, path, overwrite=overwrite)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass
class LateFusionCampaignFitConfig:
    campaign_id: str
    group_id: str
    candidate_id: str
    prediction_sources: str
    source_artifact_audit: str
    output_path: str
    device: str = "cpu"
    stacker_max_steps: int = 80
    classwise_steps: int = 600
    enforce_g0_reproduction: bool = True
    g0_reproduction_tolerance: float = 2.0e-6
    overwrite: bool = False


def audit_g0_development_reproduction(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: float = 2.0e-6,
) -> dict[str, Any]:
    """Compare L0/F_method to the immutable completed-pilot G0 stack reference."""

    differences: dict[str, Any] = {}
    problems: list[str] = []
    for split, reference in G0_DEVELOPMENT_REFERENCE.items():
        actual = metrics.get(split, {}).get("multiclass", {})
        row = {
            "reference": dict(reference),
            "actual": {key: actual.get(key) for key in reference},
            "accuracy_absolute_difference": abs(float(actual.get("accuracy", float("nan"))) - reference["accuracy"]),
            "cross_entropy_absolute_difference": abs(
                float(actual.get("cross_entropy", float("nan"))) - reference["cross_entropy"]
            ),
        }
        differences[split] = row
        if actual.get("n_jets") != reference["n_jets"]:
            problems.append(f"{split} n_jets={actual.get('n_jets')!r}, expected {reference['n_jets']}")
        for key in ("accuracy_absolute_difference", "cross_entropy_absolute_difference"):
            if not np.isfinite(row[key]) or row[key] > float(tolerance):
                problems.append(f"{split} {key}={row[key]!r} exceeds tolerance {tolerance}")
    return {
        "ok": not problems,
        "reference_id": "completed_curriculum_pilot_G0_uniform_logit_mean",
        "tolerance": float(tolerance),
        "splits": differences,
        "problems": problems,
    }


def fit_late_fusion_campaign_candidate(config: LateFusionCampaignFitConfig) -> dict[str, Any]:
    """Resolve audited stack-only blocks, fit one candidate, and write an immutable report."""

    registry = require_development_prediction_sources(
        config.prediction_sources, source_artifact_audit=config.source_artifact_audit,
    )
    groups = {group.group_id: group for group in default_fusion_group_specs()}
    if config.group_id not in groups:
        raise ValueError(f"unknown fusion group {config.group_id!r}")
    member_a, member_b = groups[config.group_id].member_ids

    def block(member: str, split: str) -> PredictionBlock:
        return load_prediction_block(registry["members"][member]["prediction_root"], member, split, verify_hash=True)

    result = fit_late_fusion_candidate(
        config.candidate_id,
        block(member_a, FUSION_FIT_SPLIT), block(member_b, FUSION_FIT_SPLIT),
        block(member_a, FUSION_SELECTION_SPLIT), block(member_b, FUSION_SELECTION_SPLIT),
        device=config.device, stacker_max_steps=config.stacker_max_steps, classwise_steps=config.classwise_steps,
    )
    labels = {
        FUSION_FIT_SPLIT: block(member_a, FUSION_FIT_SPLIT).labels,
        FUSION_SELECTION_SPLIT: block(member_a, FUSION_SELECTION_SPLIT).labels,
    }
    logits = {FUSION_FIT_SPLIT: result.train_logits, FUSION_SELECTION_SPLIT: result.validation_logits}
    candidate_spec = next(spec for spec in default_fusion_candidate_specs() if spec.candidate_id == config.candidate_id)
    metrics = {
        split: {
            "multiclass": local_residual_field_multiclass_metrics(logits[split], labels[split], label_names=LABEL_NAMES),
            "binary_projection": local_residual_field_binary_projection_metrics(logits[split], labels[split], label_names=LABEL_NAMES),
        }
        for split in LATE_FUSION_SPLITS
    }
    trainable_parameter_count = 0
    if config.candidate_id == "L2_temp_mean_logits":
        trainable_parameter_count = len(result.parameters["temperatures"])
    elif config.candidate_id == "L3_scalar_simplex_logits":
        trainable_parameter_count = 1
    elif config.candidate_id == "L4_classwise_simplex_logits":
        trainable_parameter_count = len(result.parameters["weights"])
    elif config.candidate_id == "L5_linear_stacker":
        trainable_parameter_count = int(np.asarray(result.parameters["weight"]).size + np.asarray(result.parameters["bias"]).size)
    report: dict[str, Any] = {
        "ok": True,
        "contract": LOCAL_RESIDUAL_FIELD_LATE_FUSION_FIT_CONTRACT,
        "campaign_id": str(config.campaign_id),
        "group_id": config.group_id,
        "member_ids": [member_a, member_b],
        "candidate_id": config.candidate_id,
        "candidate_spec": candidate_spec.to_dict(),
        "candidate_spec_hash": stable_fusion_json_hash(candidate_spec.to_dict()),
        "parameters": dict(result.parameters),
        "fit_split": FUSION_FIT_SPLIT,
        "selection_split": FUSION_SELECTION_SPLIT,
        "development_splits": list(LATE_FUSION_SPLITS),
        "final_test_opened": False,
        "metrics": metrics,
        "source_artifact_audit_hash": registry["source_artifact_audit_hash"],
        "prediction_sources_hash": registry["manifest_hash"],
        "prediction_source_hashes": {
            member: {
                split: registry["members"][member]["splits"][split]["prediction_sha256"]
                for split in LATE_FUSION_SPLITS
            }
            for member in (member_a, member_b)
        },
        "trainable_parameter_count": trainable_parameter_count,
        "runtime_inputs": "HLT_only",
        "uses_true_fields": False,
        "uses_offline_particles": False,
        "uses_teacher_logits_at_runtime": False,
        "deployable": True,
        "config": asdict(config),
    }
    if config.group_id == "F_method" and config.candidate_id == "L0_mean_logits":
        reproduction = audit_g0_development_reproduction(
            metrics, tolerance=float(config.g0_reproduction_tolerance),
        )
        report["g0_development_reproduction"] = reproduction
        if bool(config.enforce_g0_reproduction) and not reproduction["ok"]:
            raise ValueError(f"L0/F_method failed G0 development reproduction: {reproduction['problems']}")
    report["artifact_hash"] = stable_fusion_json_hash(report)
    _atomic_json(Path(config.output_path), report, overwrite=bool(config.overwrite))
    return report


__all__ = [
    "LOCAL_RESIDUAL_FIELD_LATE_FUSION_FIT_CONTRACT",
    "LATE_FUSION_SPLITS",
    "G0_DEVELOPMENT_REFERENCE",
    "LateFusionFitResult",
    "LateFusionCampaignFitConfig",
    "fit_late_fusion_candidate",
    "apply_late_fusion_candidate",
    "replay_late_fusion_recipe",
    "fit_late_fusion_campaign_candidate",
    "audit_g0_development_reproduction",
]
