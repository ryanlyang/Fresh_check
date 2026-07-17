"""Fail-closed three-epoch runtime-profile selection for C2F Step 8."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .runtime import build_runtime_profile, collect_code_environment


RUNTIME_CANDIDATE_CONTRACT = "constrained_c2f_accelerated_candidate_v1"
REQUIRED_VARIANTS = ("C1", "C5-B3", "C6", "C4")


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


@dataclass(frozen=True)
class RuntimeAcceptanceConfig:
    """Predeclared Step 8 gates. Values are persisted in the candidate closure."""

    component_relative_tolerance: float = 0.01
    gradient_relative_tolerance: float = 0.05
    reconstruction_relative_tolerance: float = 0.01
    diagnostic_relative_tolerance: float = 0.02
    absolute_tolerance: float = 1.0e-6
    gradient_absolute_tolerance: float = 1.0e-5
    gpu_memory_bytes: int = 96 * 1024**3
    gpu_reserved_fraction_cap: float = 0.80
    worker_throughput_fraction: float = 0.99
    accelerated_max_epochs: int = 10
    accelerated_min_epochs: int = 5
    accelerated_early_stop_patience: int = 2
    accelerated_warmup_fraction: float = 0.10
    accelerated_min_lr_ratio: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.component_relative_tolerance < 1.0:
            raise ValueError("component_relative_tolerance must be in (0, 1)")
        if not 0.0 < self.reconstruction_relative_tolerance < 1.0:
            raise ValueError("reconstruction_relative_tolerance must be in (0, 1)")
        if not 0.0 < self.diagnostic_relative_tolerance < 1.0:
            raise ValueError("diagnostic_relative_tolerance must be in (0, 1)")
        if self.gpu_memory_bytes <= 0 or not 0.0 < self.gpu_reserved_fraction_cap <= 1.0:
            raise ValueError("invalid GPU memory safety cap")
        if not 0.0 < self.worker_throughput_fraction <= 1.0:
            raise ValueError("worker_throughput_fraction must be in (0, 1]")
        if not 0 < self.accelerated_min_epochs <= self.accelerated_max_epochs:
            raise ValueError("accelerated epoch bounds are invalid")


def _metrics(record: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    final_epoch = record.get("final_epoch")
    if not isinstance(final_epoch, Mapping):
        return {}
    numeric = final_epoch.get(f"{split}_numeric_metrics")
    if isinstance(numeric, Mapping):
        return numeric
    fallback = final_epoch.get(split)
    return fallback if isinstance(fallback, Mapping) else {}


def _nonfinite_batches(record: Mapping[str, Any]) -> float | None:
    return _finite(record.get("resource", {}).get("nonfinite_batches_skipped")) if isinstance(record.get("resource"), Mapping) else None


def _comparison(
    *,
    candidate: float | None,
    reference: float | None,
    relative_tolerance: float,
    absolute_tolerance: float,
    higher_is_better: bool = False,
) -> dict[str, Any]:
    if candidate is None or reference is None:
        return {"ok": False, "reason": "missing_or_nonfinite", "candidate": candidate, "reference": reference}
    if abs(reference) <= absolute_tolerance:
        allowed = absolute_tolerance
        ok = candidate >= reference - allowed if higher_is_better else candidate <= reference + allowed
        mode = "absolute"
    else:
        allowed = abs(reference) * relative_tolerance
        ok = candidate >= reference - allowed if higher_is_better else candidate <= reference + allowed
        mode = "relative"
    return {
        "ok": bool(ok),
        "mode": mode,
        "candidate": candidate,
        "reference": reference,
        "allowed_degradation": allowed,
        "higher_is_better": higher_is_better,
    }


def _all_component_gate(
    candidate: Mapping[str, Any], reference: Mapping[str, Any], config: RuntimeAcceptanceConfig
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    component_keys = sorted(
        key
        for key in reference
        if key.startswith(("loss.", "hierarchy.component.", "slot.component."))
    )
    if "loss.total" not in component_keys:
        component_keys.append("loss.total")
    if not component_keys:
        return {"ok": False, "reason": "no_named_loss_components", "comparisons": comparisons}
    for key in component_keys:
        comparisons[key] = _comparison(
            candidate=_finite(candidate.get(key)),
            reference=_finite(reference.get(key)),
            relative_tolerance=config.component_relative_tolerance,
            absolute_tolerance=config.absolute_tolerance,
        )
    return {"ok": all(item["ok"] for item in comparisons.values()), "comparisons": comparisons}


def _bf16_gate(candidate: Mapping[str, Any], reference: Mapping[str, Any], config: RuntimeAcceptanceConfig) -> dict[str, Any]:
    candidate_train = _metrics(candidate, "train")
    reference_train = _metrics(reference, "train")
    candidate_val = _metrics(candidate, "model_val")
    reference_val = _metrics(reference, "model_val")
    train_components = _all_component_gate(candidate_train, reference_train, config)
    val_components = _all_component_gate(candidate_val, reference_val, config)
    gradient = _comparison(
        candidate=_finite(candidate_train.get("train.grad_norm_before_clip")),
        reference=_finite(reference_train.get("train.grad_norm_before_clip")),
        relative_tolerance=config.gradient_relative_tolerance,
        absolute_tolerance=config.gradient_absolute_tolerance,
    )
    cardinality = {
        "train_n_jets": candidate_train.get("n_jets") == reference_train.get("n_jets"),
        "model_val_n_jets": candidate_val.get("n_jets") == reference_val.get("n_jets"),
    }
    no_nonfinite = _nonfinite_batches(candidate) == 0.0 and _nonfinite_batches(reference) == 0.0
    return {
        "ok": bool(train_components["ok"] and val_components["ok"] and gradient["ok"] and all(cardinality.values()) and no_nonfinite),
        "train_components": train_components,
        "model_val_components": val_components,
        "gradient_norm": gradient,
        "cardinality": cardinality,
        "no_nonfinite_batches": no_nonfinite,
    }


def _reconstruction_gate(
    candidate: Mapping[str, Any], reference: Mapping[str, Any], config: RuntimeAcceptanceConfig
) -> dict[str, Any]:
    candidate_metrics = _metrics(candidate, "model_val")
    reference_metrics = _metrics(reference, "model_val")
    comparisons: dict[str, Any] = {}
    comparisons["selection.reconstruction_score"] = _comparison(
        candidate=_finite(candidate_metrics.get("selection.reconstruction_score")),
        reference=_finite(reference_metrics.get("selection.reconstruction_score")),
        relative_tolerance=config.reconstruction_relative_tolerance,
        absolute_tolerance=config.absolute_tolerance,
    )
    for key in ("slot.metric.matched_pT_mae", "slot.metric.matched_eta_mae", "slot.metric.matched_phi_mae"):
        comparisons[key] = _comparison(
            candidate=_finite(candidate_metrics.get(key)),
            reference=_finite(reference_metrics.get(key)),
            relative_tolerance=config.diagnostic_relative_tolerance,
            absolute_tolerance=config.absolute_tolerance,
        )
    comparisons["slot.metric.matched_pid_accuracy"] = _comparison(
        candidate=_finite(candidate_metrics.get("slot.metric.matched_pid_accuracy")),
        reference=_finite(reference_metrics.get("slot.metric.matched_pid_accuracy")),
        relative_tolerance=config.diagnostic_relative_tolerance,
        absolute_tolerance=config.absolute_tolerance,
        higher_is_better=True,
    )
    hierarchy_keys = sorted(
        key
        for key in reference_metrics
        if key.startswith("hierarchy.metric.")
        and ("accounting" in key or "parent_child_consistency" in key or key.endswith("global_total_pT_relative_mae"))
    )
    if not hierarchy_keys:
        comparisons["hierarchy_metrics"] = {"ok": False, "reason": "missing_accounting_or_parent_child_metrics"}
    else:
        for key in hierarchy_keys:
            comparisons[key] = _comparison(
                candidate=_finite(candidate_metrics.get(key)),
                reference=_finite(reference_metrics.get(key)),
                relative_tolerance=config.diagnostic_relative_tolerance,
                absolute_tolerance=config.absolute_tolerance,
            )
    peak_reserved = _finite(candidate.get("resource", {}).get("peak_cuda_reserved_bytes")) if isinstance(candidate.get("resource"), Mapping) else None
    memory_cap = int(config.gpu_memory_bytes * config.gpu_reserved_fraction_cap)
    memory = {"ok": peak_reserved is not None and peak_reserved <= memory_cap, "peak_reserved_bytes": peak_reserved, "cap_bytes": memory_cap}
    no_nonfinite = _nonfinite_batches(candidate) == 0.0
    return {
        "ok": bool(all(bool(item.get("ok")) for item in comparisons.values()) and memory["ok"] and no_nonfinite),
        "comparisons": comparisons,
        "memory": memory,
        "no_nonfinite_batches": no_nonfinite,
    }


def _record_index(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    index: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        variant = record.get("variant")
        matrix_case = record.get("matrix_case")
        if not isinstance(variant, str) or not isinstance(matrix_case, str):
            continue
        index.setdefault((variant, matrix_case), []).append(record)
    return index


def _one(records: Sequence[Mapping[str, Any]], label: str) -> Mapping[str, Any]:
    if len(records) != 1:
        raise ValueError(f"expected exactly one {label} benchmark record, found {len(records)}")
    return records[0]


def _same_code_environment(records: Sequence[Mapping[str, Any]], current: Mapping[str, Any]) -> dict[str, Any]:
    current_hash = current.get("code_environment_hash")
    if not bool(current.get("source_tree_clean")) or not isinstance(current_hash, str):
        raise ValueError("candidate creation requires a clean current source tree and code_environment_hash")
    for record in records:
        environment = record.get("code_environment")
        if not isinstance(environment, Mapping) or environment.get("code_environment_hash") != current_hash:
            raise ValueError(f"benchmark code/environment mismatch for {record.get('run_id')}")
        if not bool(environment.get("source_tree_clean")):
            raise ValueError(f"benchmark record {record.get('run_id')} was not produced from a clean source tree")
    return dict(current)


def select_accelerated_candidate(
    benchmark_report: Mapping[str, Any],
    *,
    config: RuntimeAcceptanceConfig,
    code_environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the Step 8 three-epoch gates and return an immutable candidate payload."""

    if not bool(benchmark_report.get("ok")):
        raise ValueError("runtime benchmark report is not complete and valid")
    records_raw = benchmark_report.get("records")
    if not isinstance(records_raw, list):
        raise ValueError("runtime benchmark report has no records")
    records = [record for record in records_raw if isinstance(record, Mapping)]
    index = _record_index(records)
    references = {variant: _one(index.get((variant, "A"), []), f"{variant} A") for variant in REQUIRED_VARIANTS}
    b_records = {
        variant: _one(index.get((variant, "B_serial" if variant == "C4" else "B"), []), f"{variant} B")
        for variant in REQUIRED_VARIANTS
    }
    bf16_gates = {variant: _bf16_gate(b_records[variant], references[variant], config) for variant in REQUIRED_VARIANTS}
    if not all(gate["ok"] for gate in bf16_gates.values()):
        failed = [variant for variant, gate in bf16_gates.items() if not gate["ok"]]
        raise ValueError(f"BF16 numerical gate failed for {failed}")

    c4_thread_gates: dict[int, dict[str, Any]] = {}
    for row in index.get(("C4", "B_thread"), []):
        requested = row.get("requested_execution")
        if not isinstance(requested, Mapping):
            continue
        workers = int(requested.get("hungarian_workers", 0))
        c4_thread_gates[workers] = _bf16_gate(row, references["C4"], config)

    gate_rows: dict[str, dict[str, Any]] = {}
    by_lr: dict[float, dict[str, list[Mapping[str, Any]]]] = {}
    for record in records:
        if record.get("matrix_case") != "D" or record.get("variant") not in REQUIRED_VARIANTS:
            continue
        requested = record.get("requested_execution")
        if not isinstance(requested, Mapping):
            continue
        variant = str(record["variant"])
        learning_rate = _finite(requested.get("learning_rate"))
        if learning_rate is None:
            continue
        gate = _reconstruction_gate(record, references[variant], config)
        if variant == "C4":
            gate["threaded_hungarian_gate"] = c4_thread_gates.get(int(requested.get("hungarian_workers", 0)))
            gate["ok"] = bool(gate["ok"] and gate["threaded_hungarian_gate"] and gate["threaded_hungarian_gate"]["ok"])
        gate_rows[str(record.get("run_id"))] = gate
        if gate["ok"]:
            by_lr.setdefault(learning_rate, {}).setdefault(variant, []).append(record)

    eligible_profiles: list[dict[str, Any]] = []
    for learning_rate, rows_by_variant in by_lr.items():
        if any(variant not in rows_by_variant for variant in REQUIRED_VARIANTS):
            continue
        selected: dict[str, Mapping[str, Any]] = {}
        throughput_score = 0.0
        for variant in REQUIRED_VARIANTS:
            candidates = rows_by_variant[variant]
            rates = [
                _finite(row.get("resource", {}).get("train_jets_per_second_weighted"))
                if isinstance(row.get("resource"), Mapping)
                else None
                for row in candidates
            ]
            if any(rate is None for rate in rates):
                raise ValueError(f"passing {variant} candidates are missing throughput telemetry")
            best_rate = max(rate for rate in rates if rate is not None)
            near_best = [
                row
                for row, rate in zip(candidates, rates)
                if rate is not None and rate >= best_rate * config.worker_throughput_fraction
            ]
            def selection_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
                execution = row["requested_execution"]
                assert isinstance(execution, Mapping)
                return (
                    int(execution.get("num_workers", 0)),
                    int(execution.get("hungarian_workers", 1)),
                    int(execution.get("train_batch_size", 0)),
                    str(row.get("run_id")),
                )
            selected_row = min(near_best, key=selection_key)
            selected[variant] = selected_row
            reference_rate = _finite(references[variant].get("resource", {}).get("train_jets_per_second_weighted"))
            chosen_rate = _finite(selected_row.get("resource", {}).get("train_jets_per_second_weighted"))
            if reference_rate is None or reference_rate <= 0.0 or chosen_rate is None:
                raise ValueError(f"invalid throughput telemetry for {variant}")
            throughput_score += chosen_rate / reference_rate
        eligible_profiles.append({"learning_rate": learning_rate, "selected": selected, "throughput_score": throughput_score})
    if not eligible_profiles:
        raise ValueError("no one-LR accelerated profile passed all representative three-epoch gates")
    chosen_profile = max(eligible_profiles, key=lambda row: (float(row["throughput_score"]), -float(row["learning_rate"])))
    selected = chosen_profile["selected"]
    assert isinstance(selected, Mapping)
    hlt_scales = {
        _finite(row["requested_execution"].get("hlt_encoder_lr_scale"))
        for row in selected.values()
        if isinstance(row.get("requested_execution"), Mapping)
    }
    if len(hlt_scales) != 1 or None in hlt_scales:
        raise ValueError("selected variants do not share one HLT encoder LR scale")
    hlt_scale = next(iter(hlt_scales))
    assert hlt_scale is not None
    environments = (
        list(references.values())
        + list(b_records.values())
        + [row for row in index.get(("C4", "B_thread"), [])]
        + list(selected.values())
    )
    environment = _same_code_environment(environments, code_environment or collect_code_environment())
    calibration_inputs = benchmark_report.get("calibration_input_provenance")
    if not isinstance(calibration_inputs, Mapping) or set(calibration_inputs) != {"model_train", "model_val"}:
        raise ValueError("benchmark report lacks calibration input provenance")
    for split, row in calibration_inputs.items():
        if not isinstance(row, Mapping) or any(row.get(key) in (None, "") for key in (
            "hlt_content_hash", "offline_content_hash", "target_content_hash", "jet_identity_hash"
        )):
            raise ValueError(f"benchmark report has incomplete {split} calibration provenance")
    execution_by_variant: dict[str, Any] = {}
    runtime_profiles: dict[str, Any] = {}
    for variant, record in selected.items():
        requested = record["requested_execution"]
        assert isinstance(requested, Mapping)
        profile = build_runtime_profile(
            profile="accelerated_candidate_v1",
            precision_mode="bf16_forward_fp32_loss",
            batch_size=int(requested["train_batch_size"]),
            eval_batch_size=int(requested["eval_batch_size"]),
            num_workers=int(requested["num_workers"]),
            prefetch_factor=requested.get("prefetch_factor"),
            learning_rate=float(chosen_profile["learning_rate"]),
            hlt_encoder_lr_scale=float(hlt_scale),
            weight_decay=1.0e-4,
            grad_clip_norm=1.0,
            lr_schedule="warmup_cosine",
            warmup_fraction=config.accelerated_warmup_fraction,
            min_lr_ratio=config.accelerated_min_lr_ratio,
            min_epochs=config.accelerated_min_epochs,
            early_stop_patience=config.accelerated_early_stop_patience,
            fixed_horizon=False,
            max_epochs=config.accelerated_max_epochs,
            hungarian_workers=int(requested["hungarian_workers"]),
            hungarian_executor=str(requested["hungarian_executor"]),
        )
        execution_by_variant[variant] = {
            "benchmark_run_id": record["run_id"],
            "checkpoint_sha256": record.get("checkpoint_sha256"),
            "runtime_profile_hash": profile["runtime_profile_hash"],
            "execution": profile,
        }
        runtime_profiles[variant] = profile
    payload: dict[str, Any] = {
        "contract": RUNTIME_CANDIDATE_CONTRACT,
        "status": "accelerated_candidate_v1",
        "selection_stage": "three_epoch_calibration",
        "benchmark_plan_hash": benchmark_report.get("benchmark_plan_hash"),
        "benchmark_report_calibration_manifest_hash": benchmark_report.get("calibration_manifest_hash"),
        "benchmark_report_calibration_validation_sha256": benchmark_report.get("calibration_validation_sha256"),
        "calibration_input_provenance": calibration_inputs,
        "shared_optimizer": {
            "learning_rate": float(chosen_profile["learning_rate"]),
            "hlt_encoder_lr_scale": float(hlt_scale),
            "weight_decay": 1.0e-4,
            "grad_clip_norm": 1.0,
            "lr_schedule": "warmup_cosine",
            "warmup_fraction": config.accelerated_warmup_fraction,
            "min_lr_ratio": config.accelerated_min_lr_ratio,
            "max_epochs": config.accelerated_max_epochs,
            "min_epochs": config.accelerated_min_epochs,
            "early_stop_patience": config.accelerated_early_stop_patience,
        },
        "execution_by_variant": execution_by_variant,
        "runtime_profiles_by_variant": runtime_profiles,
        "thresholds": asdict(config),
        "bf16_numerical_gates": bf16_gates,
        "three_epoch_reconstruction_gates": gate_rows,
        "selection_candidates": [
            {
                "learning_rate": row["learning_rate"],
                "throughput_score": row["throughput_score"],
                "selected_run_ids": {variant: record["run_id"] for variant, record in row["selected"].items()},
            }
            for row in eligible_profiles
        ],
        "code_environment": environment,
        "code_environment_hash": environment["code_environment_hash"],
        "queue_rights": {"full_pilot": True, "high_data": False, "final_claims": False},
    }
    payload["candidate_profile_hash"] = _canonical_hash(payload)
    return payload


def write_accelerated_candidate(
    *,
    benchmark_report_path: str | Path,
    output_path: str | Path,
    config: RuntimeAcceptanceConfig,
    code_environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report_path = Path(benchmark_report_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable candidate artifact: {output}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("benchmark report must be a JSON object")
    candidate = select_accelerated_candidate(report, config=config, code_environment=code_environment)
    candidate["benchmark_report_path"] = str(report_path.resolve())
    candidate["benchmark_report_sha256"] = _file_hash(report_path)
    candidate["candidate_profile_hash"] = _canonical_hash({key: value for key, value in candidate.items() if key != "candidate_profile_hash"})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return candidate


__all__ = [
    "REQUIRED_VARIANTS",
    "RUNTIME_CANDIDATE_CONTRACT",
    "RuntimeAcceptanceConfig",
    "select_accelerated_candidate",
    "write_accelerated_candidate",
]
