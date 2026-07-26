"""Fail-closed Step-10 runtime, screening, and campaign promotion gates."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import canonical_hash
from .convergence_schedule import (
    ABPH_ACCELERATED_STAGE_BUDGETS,
    ABPH_EXTENSION_COMPARISON_CONTRACT,
)
from .runtime_batch import RuntimeBatchContract
from .runtime_profile import ABPH_RUNTIME_PROFILE_CONTRACT


ABPH_DDP_SMOKE_CONTRACT = "adaptive_binary_ddp_acceptance_smoke_v1"
ABPH_RUNTIME_ACCEPTANCE_CONTRACT = "adaptive_binary_runtime_acceptance_v1"
ABPH_RUNTIME_BENCHMARK_CONTRACT = "adaptive_binary_pseudooffline_runtime_reference_v2"
ABPH_RUNTIME_BENCHMARK_VALIDATION_POLICY = (
    "one_fixed_model_val_subset_at_fixed_update_v1"
)
ABPH_SINGLE_PATH_ACCEPTANCE_CONTRACT = "adaptive_binary_single_path_acceptance_v2"
ABPH_RUNTIME_REPRESENTATIVE_VARIANTS = (
    "B1_semantic_query_root",
    "D1_kt32_mh4_particles",
)
ABPH_RUNTIME_ACCEPTANCE_THRESHOLDS = {
    "deep_ddp_speedup_minimum": 1.8,
    "communication_fraction_maximum": 0.35,
    "validation_relative_loss_tolerance": 0.01,
    "parameter_absolute_tolerance": 2.0e-6,
    "parameter_relative_tolerance": 2.0e-5,
    "memory_utilization_maximum": 0.90,
    "deep_projected_wall_seconds_maximum": 48.0 * 3600.0,
}


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"acceptance artifact {source} must be a JSON object")
    return dict(payload)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_content_hash(payload: Mapping[str, Any], field: str) -> None:
    saved = payload.get(field)
    if not isinstance(saved, str) or not saved:
        raise ValueError(f"artifact lacks {field}")
    observed = canonical_hash({key: value for key, value in payload.items() if key != field})
    if observed != saved:
        raise ValueError(f"artifact {field} does not match its content")


def _artifact_row(path: str | Path) -> dict[str, str]:
    source = Path(path).resolve()
    return {"path": str(source), "sha256": _sha256_file(source)}


def _verify_referenced_artifacts(value: Any, *, location: str = "acceptance") -> None:
    """Re-hash every path/hash pair so a stale compiled gate cannot be reused."""

    if isinstance(value, Mapping):
        if set(value) == {"path", "sha256"}:
            path = Path(str(value["path"]))
            expected = str(value["sha256"])
            if not path.is_file() or _sha256_file(path) != expected:
                raise ValueError(f"runtime acceptance artifact changed after review: {location}")
            return
        for key, item in value.items():
            _verify_referenced_artifacts(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _verify_referenced_artifacts(item, location=f"{location}[{index}]")


def _runtime_profile(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("contract") != ABPH_RUNTIME_PROFILE_CONTRACT:
        raise ValueError(f"runtime profile has the wrong contract: {path}")
    _verify_content_hash(payload, "profile_content_hash")
    if payload.get("ok") is not True:
        raise ValueError(f"runtime profile is not successful: {path}")
    return payload


def _benchmark_evidence(
    run_dir: str | Path,
    *,
    expected_variant: str,
    expected_world_size: int,
) -> dict[str, Any]:
    directory = Path(run_dir).resolve()
    report_path = directory / "run_report.json"
    curves_path = directory / "training_curves.json"
    profile_path = directory / "runtime_profile.json"
    report = _read_json(report_path)
    profile = _runtime_profile(profile_path)
    curves = _read_json(curves_path)
    benchmark = report.get("runtime_reference_benchmark")
    if report.get("ok") is not True or not isinstance(benchmark, Mapping):
        raise ValueError(f"runtime benchmark is incomplete: {directory}")
    if benchmark.get("contract") != ABPH_RUNTIME_BENCHMARK_CONTRACT:
        raise ValueError(f"runtime benchmark contract mismatch: {directory}")
    if (
        benchmark.get("validation_policy")
        != ABPH_RUNTIME_BENCHMARK_VALIDATION_POLICY
        or int(benchmark.get("fixed_model_val_evaluations", -1)) != 1
        or int(benchmark.get("curriculum_transition_validations", -1)) != 0
    ):
        raise ValueError(
            f"runtime benchmark did not execute exactly one fixed validation: {directory}"
        )
    if report.get("variant_name") != expected_variant:
        raise ValueError(f"runtime benchmark variant mismatch: {directory}")
    runtime = report.get("distributed_runtime")
    if not isinstance(runtime, Mapping) or int(runtime.get("world_size", -1)) != int(
        expected_world_size
    ):
        raise ValueError(f"runtime benchmark world-size mismatch: {directory}")
    declared_profile = report.get("runtime_profile")
    if not isinstance(declared_profile, Mapping) or declared_profile.get(
        "profile_content_hash"
    ) != profile.get("profile_content_hash"):
        raise ValueError(f"runtime benchmark profile hash mismatch: {directory}")
    evaluations = curves.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise ValueError(f"runtime benchmark has no full validation: {directory}")
    validation = evaluations[-1].get("model_val_rollout")
    if not isinstance(validation, Mapping):
        raise ValueError(f"runtime benchmark validation is malformed: {directory}")
    selection_score = float(validation.get("selection_score", math.nan))
    n_jets = int(validation.get("n_jets", -1))
    if not math.isfinite(selection_score) or n_jets <= 0:
        raise ValueError(f"runtime benchmark validation is incomplete: {directory}")
    if int(benchmark.get("validation_jets", -1)) != n_jets:
        raise ValueError(f"runtime benchmark validation count mismatch: {directory}")
    coverage = validation.get("validation_coverage")
    if not isinstance(coverage, Mapping) or coverage.get("selection_eligible") is not True:
        raise ValueError(f"runtime benchmark lacks validation identity coverage: {directory}")
    coverage_base = {
        key: coverage[key]
        for key in ("contract", "split", "n_jets", "world_size", "ordered_ranges")
    }
    if canonical_hash(coverage_base) != coverage.get("validation_coverage_hash"):
        raise ValueError(f"runtime benchmark validation coverage hash mismatch: {directory}")
    if canonical_hash(coverage["ordered_ranges"]) != coverage.get(
        "expected_ordered_row_hash"
    ):
        raise ValueError(f"runtime benchmark validation range hash mismatch: {directory}")
    if int(coverage.get("n_jets", -1)) != n_jets or int(
        coverage.get("world_size", -1)
    ) != expected_world_size:
        raise ValueError(f"runtime benchmark validation coverage identity mismatch: {directory}")
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError(f"runtime benchmark lacks input provenance: {directory}")
    buckets = profile.get("buckets")
    stages = profile.get("stages")
    summary = profile.get("summary")
    device = profile.get("device")
    if not all(isinstance(value, Mapping) for value in (buckets, stages, summary, device)):
        raise ValueError(f"runtime benchmark telemetry is malformed: {directory}")
    update_seconds = float(
        buckets["optimizer_update_total"].get(
            "synchronized_wall_total_seconds", 0.0
        )
    )
    sampled_jets = sum(int(row.get("sampled_jets", 0)) for row in stages.values())
    sampled_updates = sum(int(row.get("sampled_updates", 0)) for row in stages.values())
    validation_seconds = float(
        buckets["full_validation"].get(
            "synchronized_wall_total_seconds", 0.0
        )
    )
    peak = int(summary.get("peak_reserved_bytes", 0))
    total_memory = int(device.get("total_memory_bytes", 0))
    if (
        update_seconds <= 0.0
        or sampled_jets <= 0
        or sampled_updates <= 0
        or validation_seconds <= 0.0
    ):
        raise ValueError(f"runtime benchmark lacks measured throughput: {directory}")
    if total_memory <= 0 or peak < 0:
        raise ValueError(f"runtime benchmark lacks device-memory telemetry: {directory}")
    return {
        "variant": expected_variant,
        "world_size": int(expected_world_size),
        "run_dir": str(directory),
        "selection_score": selection_score,
        "validation_n_jets": n_jets,
        "validation_seconds": validation_seconds,
        "sampled_jets": sampled_jets,
        "sampled_updates": sampled_updates,
        "sampled_update_seconds": update_seconds,
        "seconds_per_update": update_seconds / sampled_updates,
        "jets_per_second": sampled_jets / update_seconds,
        "communication_fraction": summary.get("communication_fraction"),
        "peak_reserved_bytes": peak,
        "total_device_memory_bytes": total_memory,
        "memory_utilization": peak / total_memory,
        "profile_content_hash": profile["profile_content_hash"],
        "input_provenance_hash": canonical_hash(provenance),
        "validation_coverage_hash": coverage["validation_coverage_hash"],
        "validation_expected_ordered_row_hash": coverage[
            "expected_ordered_row_hash"
        ],
        "artifacts": {
            "run_report": _artifact_row(report_path),
            "training_curves": _artifact_row(curves_path),
            "runtime_profile": _artifact_row(profile_path),
        },
    }


def _load_smoke(path: str | Path, *, expected_world_size: int) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("contract") != ABPH_DDP_SMOKE_CONTRACT:
        raise ValueError("DDP smoke contract mismatch")
    _verify_content_hash(payload, "smoke_content_hash")
    if payload.get("ok") is not True or int(payload.get("world_size", -1)) != int(
        expected_world_size
    ):
        raise ValueError("DDP smoke is incomplete or has the wrong world size")
    checks = payload.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(checks.values()):
        raise ValueError("DDP smoke has a failed required check")
    state_path = Path(str(payload.get("state_path", "")))
    if not state_path.is_file() or _sha256_file(state_path) != payload.get("state_sha256"):
        raise ValueError("DDP smoke state artifact hash mismatch")
    return payload


def compare_ddp_smokes(
    single_path: str | Path,
    ddp4_path: str | Path,
    *,
    parameter_atol: float = ABPH_RUNTIME_ACCEPTANCE_THRESHOLDS[
        "parameter_absolute_tolerance"
    ],
    parameter_rtol: float = ABPH_RUNTIME_ACCEPTANCE_THRESHOLDS[
        "parameter_relative_tolerance"
    ],
) -> dict[str, Any]:
    import torch

    single = _load_smoke(single_path, expected_world_size=1)
    ddp4 = _load_smoke(ddp4_path, expected_world_size=4)
    if single.get("global_batch_identity_hash") != ddp4.get(
        "global_batch_identity_hash"
    ):
        raise ValueError("single/DDP4 smoke global batch identities differ")

    def load_state(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            return torch.load(payload["state_path"], map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - older research PyTorch
            return torch.load(payload["state_path"], map_location="cpu")

    left = load_state(single)
    right = load_state(ddp4)
    maximum_absolute = 0.0
    maximum_relative = 0.0
    compared = 0
    for section in ("model_state_dict", "ema_state_dict"):
        left_rows = left.get(section)
        right_rows = right.get(section)
        if not isinstance(left_rows, Mapping) or set(left_rows) != set(right_rows or {}):
            raise ValueError(f"single/DDP4 smoke {section} membership differs")
        for name in sorted(left_rows):
            lhs = left_rows[name].detach().to(torch.float64)
            rhs = right_rows[name].detach().to(torch.float64)
            if lhs.shape != rhs.shape:
                raise ValueError(f"single/DDP4 smoke tensor shape differs for {name}")
            delta = (lhs - rhs).abs()
            maximum_absolute = max(maximum_absolute, float(delta.max().item()))
            scale = torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1.0e-12)
            maximum_relative = max(
                maximum_relative, float((delta / scale).max().item())
            )
            compared += 1
    validation_delta = abs(
        float(single["validation_mean_loss"]) - float(ddp4["validation_mean_loss"])
    )
    training_loss_delta = abs(
        float(single["global_training_mean_loss"])
        - float(ddp4["global_training_mean_loss"])
    )
    gradient_norm_delta = abs(
        float(single["preclip_gradient_l2_norm"])
        - float(ddp4["preclip_gradient_l2_norm"])
    )
    checks = {
        "same_global_batch_identities": True,
        "state_membership_and_shapes_match": compared > 0,
        "post_step_parameters_within_tolerance": maximum_absolute <= parameter_atol
        or maximum_relative <= parameter_rtol,
        "validation_aggregate_within_tolerance": validation_delta <= parameter_atol,
        "raw_weighted_training_loss_within_tolerance": training_loss_delta
        <= parameter_atol,
        "preclip_gradient_norm_within_tolerance": gradient_norm_delta
        <= parameter_atol,
        "single_resume_passed": bool(single["checks"].get("checkpoint_resume")),
        "ddp4_resume_passed": bool(ddp4["checks"].get("checkpoint_resume")),
        "ddp4_failure_consensus_passed": bool(
            ddp4["checks"].get("forward_failure_consensus")
        ),
        "ddp4_wrapper_rebuild_passed": bool(ddp4["checks"].get("wrapper_rebuild")),
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "maximum_parameter_absolute_difference": maximum_absolute,
        "maximum_parameter_relative_difference": maximum_relative,
        "validation_absolute_difference": validation_delta,
        "training_loss_absolute_difference": training_loss_delta,
        "gradient_norm_absolute_difference": gradient_norm_delta,
        "compared_tensor_count": compared,
        "single_smoke": _artifact_row(single_path),
        "ddp4_smoke": _artifact_row(ddp4_path),
    }


def _runtime_batch_contract(path: str | Path, *, variant: str) -> dict[str, Any]:
    payload = _read_json(path)
    contract = RuntimeBatchContract.from_dict(payload)
    if contract.variant_name != variant or contract.requested_world_size != 4:
        raise ValueError(f"DDP4 runtime batch contract identity mismatch for {variant}")
    if set(contract.selections) != {"root_hierarchy", "renderer_distribution"}:
        raise ValueError(f"runtime batch contract stage membership mismatch for {variant}")
    return {
        "variant": variant,
        "contract_hash": contract.contract_hash,
        "artifact": _artifact_row(path),
        "selections": {
            name: value.to_dict() for name, value in contract.selections.items()
        },
    }


def _extension_report(path: str | Path, *, variant: str) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("contract") != ABPH_EXTENSION_COMPARISON_CONTRACT:
        raise ValueError(f"extension comparison contract mismatch for {variant}")
    if payload.get("variant_name") != variant or payload.get("evaluation_split") != "model_val":
        raise ValueError(f"extension comparison identity mismatch for {variant}")
    if payload.get("ok") is not True or payload.get("final_test_loaded") is not False:
        raise ValueError(f"extension comparison is not scientifically usable for {variant}")
    if "report_content_hash" in payload:
        _verify_content_hash(payload, "report_content_hash")
    return {
        "variant": variant,
        "schedule_truncated": bool(payload.get("schedule_truncated")),
        "screening_checkpoint_policy": payload.get("screening_checkpoint_policy"),
        "material_reconstruction_improvement": bool(
            payload.get("material_reconstruction_improvement")
        ),
        "tagging_conclusion_changed": bool(
            dict(payload.get("categories", {})).get("tagging_conclusion_changed")
        ),
        "artifact": _artifact_row(path),
    }


def _pilot_report(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    saved = payload.get("report_content_hash")
    if isinstance(saved, str):
        _verify_content_hash(payload, "report_content_hash")
    screening = payload.get("schedule_screening")
    if payload.get("ok") is not True or not isinstance(screening, Mapping):
        raise ValueError("optimized pilot report is incomplete")
    return {
        "ok": True,
        "automatic_highdata_promotion_allowed": bool(
            screening.get("automatic_highdata_promotion_allowed")
        ),
        "negative_mechanism_conclusion_valid": bool(
            screening.get("negative_mechanism_conclusion_valid")
        ),
        "artifact": _artifact_row(path),
    }


def _single_path_report(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("contract") != ABPH_SINGLE_PATH_ACCEPTANCE_CONTRACT:
        raise ValueError("single-path acceptance contract mismatch")
    _verify_content_hash(payload, "report_content_hash")
    checks = payload.get("checks")
    if payload.get("ok") is not True or not isinstance(checks, Mapping) or not all(
        checks.values()
    ):
        raise ValueError("single-path acceleration evidence is not approved")
    source_artifacts = payload.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping) or not source_artifacts:
        raise ValueError("single-path acceleration evidence lacks bound sources")
    _verify_referenced_artifacts(source_artifacts, location="single_path.source_artifacts")
    return {
        "ok": True,
        "instrumentation_overhead_fraction": float(
            payload["instrumentation_overhead_fraction"]
        ),
        "deep_training_speedup": float(payload["deep_training_speedup"]),
        "profiler_explanation": payload.get("profiler_explanation"),
        "checks": dict(checks),
        "source_artifacts": dict(source_artifacts),
        "artifact": _artifact_row(path),
    }


def build_runtime_acceptance_report(
    *,
    single_run_dirs: Mapping[str, str | Path],
    ddp4_run_dirs: Mapping[str, str | Path],
    single_smoke_path: str | Path,
    ddp4_smoke_path: str | Path,
    ddp4_batch_contracts: Mapping[str, str | Path],
    single_path_acceptance: str | Path,
    extension_reports: Mapping[str, str | Path] | None = None,
    optimized_pilot_report: str | Path | None = None,
    expected_validation_jets: int = 4_096,
) -> dict[str, Any]:
    variants = ABPH_RUNTIME_REPRESENTATIVE_VARIANTS
    if set(single_run_dirs) != set(variants) or set(ddp4_run_dirs) != set(variants):
        raise ValueError("runtime acceptance requires exact root/deep benchmark membership")
    if set(ddp4_batch_contracts) != set(variants):
        raise ValueError("runtime acceptance requires exact root/deep DDP4 batch contracts")
    single = {
        variant: _benchmark_evidence(
            single_run_dirs[variant], expected_variant=variant, expected_world_size=1
        )
        for variant in variants
    }
    ddp4 = {
        variant: _benchmark_evidence(
            ddp4_run_dirs[variant], expected_variant=variant, expected_world_size=4
        )
        for variant in variants
    }
    smoke = compare_ddp_smokes(single_smoke_path, ddp4_smoke_path)
    batch_contracts = {
        variant: _runtime_batch_contract(ddp4_batch_contracts[variant], variant=variant)
        for variant in variants
    }
    single_path = _single_path_report(single_path_acceptance)
    comparisons: dict[str, Any] = {}
    trajectory_checks: list[bool] = []
    coverage_checks: list[bool] = []
    memory_checks: list[bool] = []
    for variant in variants:
        one = single[variant]
        four = ddp4[variant]
        relative_loss_delta = abs(four["selection_score"] - one["selection_score"]) / max(
            abs(one["selection_score"]), 1.0e-12
        )
        row = {
            "single_jets_per_second": one["jets_per_second"],
            "ddp4_jets_per_second": four["jets_per_second"],
            "speedup": four["jets_per_second"] / one["jets_per_second"],
            "single_validation_seconds": one["validation_seconds"],
            "ddp4_validation_seconds": four["validation_seconds"],
            "ddp4_validation_speedup": one["validation_seconds"]
            / four["validation_seconds"],
            "single_selection_score": one["selection_score"],
            "ddp4_selection_score": four["selection_score"],
            "relative_selection_score_difference": relative_loss_delta,
            "ddp4_communication_fraction": four["communication_fraction"],
            "ddp4_memory_utilization": four["memory_utilization"],
            "same_input_provenance": one["input_provenance_hash"]
            == four["input_provenance_hash"],
        }
        comparisons[variant] = row
        trajectory_checks.append(
            relative_loss_delta
            <= ABPH_RUNTIME_ACCEPTANCE_THRESHOLDS[
                "validation_relative_loss_tolerance"
            ]
        )
        coverage_checks.append(
            one["validation_n_jets"] == expected_validation_jets
            and four["validation_n_jets"] == expected_validation_jets
            and row["same_input_provenance"]
        )
        memory_checks.append(
            four["memory_utilization"]
            <= ABPH_RUNTIME_ACCEPTANCE_THRESHOLDS["memory_utilization_maximum"]
        )
    deep = comparisons["D1_kt32_mh4_particles"]
    communication = deep["ddp4_communication_fraction"]
    deep_nominal_updates = sum(
        budget.nominal_updates
        for budget in ABPH_ACCELERATED_STAGE_BUDGETS["pilot"].values()
    )
    deep_projected_wall_seconds = (
        ddp4["D1_kt32_mh4_particles"]["seconds_per_update"]
        * deep_nominal_updates
    )
    runtime_checks = {
        "single_vs_ddp4_transport_parity": bool(smoke["ok"]),
        "exact_model_val_identity_counts": all(coverage_checks),
        "representative_validation_trajectories_compatible": all(trajectory_checks),
        "deep_ddp4_speedup_at_least_1p8": deep["speedup"]
        >= ABPH_RUNTIME_ACCEPTANCE_THRESHOLDS["deep_ddp_speedup_minimum"],
        "deep_communication_fraction_below_35_percent": communication is not None
        and float(communication)
        < ABPH_RUNTIME_ACCEPTANCE_THRESHOLDS["communication_fraction_maximum"],
        "deep_distributed_validation_faster": deep["ddp4_validation_speedup"] > 1.0,
        "all_ddp4_memory_contracts_pass": all(memory_checks),
        "production_topology_batch_contracts_present": len(batch_contracts) == 2,
        "instrumentation_and_single_gpu_path_approved": bool(single_path["ok"]),
        "deep_projected_wall_below_48_hours": deep_projected_wall_seconds
        < ABPH_RUNTIME_ACCEPTANCE_THRESHOLDS[
            "deep_projected_wall_seconds_maximum"
        ],
    }
    ddp4_approved = all(runtime_checks.values())

    extension_rows: dict[str, Any] = {}
    extension_complete = extension_reports is not None and set(extension_reports) == set(
        variants
    )
    if extension_complete:
        extension_rows = {
            variant: _extension_report(extension_reports[variant], variant=variant)
            for variant in variants
        }
    screening_not_truncated = bool(
        extension_complete
        and all(not row["schedule_truncated"] for row in extension_rows.values())
    )
    representative_screening_approved = bool(ddp4_approved and screening_not_truncated)
    pilot = None if optimized_pilot_report is None else _pilot_report(optimized_pilot_report)
    pilot_approved = bool(
        representative_screening_approved
        and pilot is not None
        and pilot["automatic_highdata_promotion_allowed"]
        and pilot["negative_mechanism_conclusion_valid"]
    )
    report: dict[str, Any] = {
        "contract": ABPH_RUNTIME_ACCEPTANCE_CONTRACT,
        "ok": ddp4_approved,
        "status": "ddp4_runtime_approved" if ddp4_approved else "runtime_gate_failed",
        "final_test_loaded": False,
        "thresholds": dict(ABPH_RUNTIME_ACCEPTANCE_THRESHOLDS),
        "expected_validation_jets": int(expected_validation_jets),
        "runtime_gate": {
            "approved": ddp4_approved,
            "checks": runtime_checks,
            "comparisons": comparisons,
            "deep_projected_wall_seconds": deep_projected_wall_seconds,
            "deep_nominal_update_projection": deep_nominal_updates,
            "transport_smoke": smoke,
            "single_runs": single,
            "ddp4_runs": ddp4,
            "ddp4_batch_contracts": batch_contracts,
            "single_path_acceptance": single_path,
        },
        "representative_screening_gate": {
            "complete": extension_complete,
            "approved": representative_screening_approved,
            "schedule_not_truncated": screening_not_truncated,
            "extensions": extension_rows,
        },
        "optimized_pilot_gate": {
            "complete": pilot is not None,
            "approved": pilot_approved,
            "pilot_report": pilot,
        },
        "promotion": {
            "ddp4_runtime_approved": ddp4_approved,
            "optimized_pilot_submission_allowed": representative_screening_approved,
            "highdata_submission_allowed": pilot_approved,
            "production_reconstructor_parallelism": "ddp4" if ddp4_approved else "single",
        },
    }
    report["acceptance_content_hash"] = canonical_hash(report)
    return report


def require_runtime_acceptance(
    path: str | Path,
    *,
    scope: str = "ddp4_runtime",
) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("contract") != ABPH_RUNTIME_ACCEPTANCE_CONTRACT:
        raise ValueError("runtime acceptance contract mismatch")
    _verify_content_hash(payload, "acceptance_content_hash")
    _verify_referenced_artifacts(
        {key: value for key, value in payload.items() if key != "acceptance_content_hash"}
    )
    promotion = payload.get("promotion")
    runtime_gate = payload.get("runtime_gate")
    if (
        payload.get("ok") is not True
        or not isinstance(promotion, Mapping)
        or not isinstance(runtime_gate, Mapping)
        or runtime_gate.get("approved") is not True
        or not all(dict(runtime_gate.get("checks", {})).values())
    ):
        raise ValueError("runtime acceptance artifact is not approved")
    field = {
        "ddp4_runtime": "ddp4_runtime_approved",
        "optimized_pilot": "optimized_pilot_submission_allowed",
        "highdata": "highdata_submission_allowed",
    }.get(scope)
    if field is None:
        raise ValueError(f"unknown runtime acceptance scope {scope!r}")
    if promotion.get(field) is not True:
        raise PermissionError(f"runtime acceptance does not approve {scope}")
    return payload


__all__ = [
    "ABPH_DDP_SMOKE_CONTRACT",
    "ABPH_RUNTIME_ACCEPTANCE_CONTRACT",
    "ABPH_RUNTIME_ACCEPTANCE_THRESHOLDS",
    "ABPH_RUNTIME_BENCHMARK_CONTRACT",
    "ABPH_RUNTIME_BENCHMARK_VALIDATION_POLICY",
    "ABPH_RUNTIME_REPRESENTATIVE_VARIANTS",
    "ABPH_SINGLE_PATH_ACCEPTANCE_CONTRACT",
    "build_runtime_acceptance_report",
    "compare_ddp_smokes",
    "require_runtime_acceptance",
]
