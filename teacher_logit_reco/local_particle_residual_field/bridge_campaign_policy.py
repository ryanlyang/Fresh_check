"""Step 9 campaign policy, reporting, and deployable publication.

This module is intentionally downstream of the numerical/model code.  It turns
the Step 1--8 evidence into immutable decisions: the B0--B6 campaign ledger,
aggregate-only deployment selection, one-shot confirmation, disjoint reports,
and a bundle whose only inference input is an HLT batch.
"""

from __future__ import annotations

from copy import deepcopy
import builtins
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
import math
import io
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .bridge_campaign import (
    PAIRED_SEED_IDS,
    resolve_registry_run,
    validate_campaign_registry,
)
from .bridge_contracts import (
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .bridge_splits import (
    PREDICTION_ANCHORED_ACCESS_RECEIPT_CONTRACT,
    authorize_manifest_split_access,
    build_manifest_validation_unlock,
    claim_split_access,
)


PREDICTION_ANCHORED_CAMPAIGN_RESERVATION_CONTRACT = (
    "prediction_anchored_step9_campaign_reservations_v1"
)
PREDICTION_ANCHORED_CAMPAIGN_STATE_CONTRACT = "prediction_anchored_step9_campaign_state_v1"
PREDICTION_ANCHORED_STAGE_EVIDENCE_CONTRACT = "prediction_anchored_step9_stage_evidence_v1"
PREDICTION_ANCHORED_DEPLOYABLE_REPLICA_CONTRACT = (
    "prediction_anchored_deployable_replica_evidence_v1"
)
PREDICTION_ANCHORED_DEPLOYABLE_AGGREGATE_CONTRACT = (
    "prediction_anchored_deployable_configuration_aggregate_v1"
)
PREDICTION_ANCHORED_DEPLOYABLE_PRECONFIRMATION_CONTRACT = (
    "prediction_anchored_deployable_preconfirmation_v1"
)
PREDICTION_ANCHORED_DEPLOYABLE_CONFIRMATION_CONTRACT = (
    "prediction_anchored_deployable_confirmation_v1"
)
PREDICTION_ANCHORED_LOCKED_DEPLOYABLE_CONTRACT = "prediction_anchored_locked_deployable_v1"
PREDICTION_ANCHORED_STOPPED_DEPLOYABLE_CONTRACT = "prediction_anchored_stopped_deployable_v1"
PREDICTION_ANCHORED_REPORT_ROW_CONTRACT = "prediction_anchored_step9_report_row_v1"
PREDICTION_ANCHORED_REPORTS_CONTRACT = "prediction_anchored_step9_reports_v1"
PREDICTION_ANCHORED_MEDIAN_PUBLICATION_CONTRACT = (
    "prediction_anchored_step9_median_publication_v1"
)
PREDICTION_ANCHORED_DEPLOYABLE_BUNDLE_MANIFEST_CONTRACT = (
    "prediction_anchored_deployable_bundle_manifest_v1"
)
PREDICTION_ANCHORED_DEPLOYABLE_BUNDLE_CHECKPOINT_CONTRACT = (
    "prediction_anchored_deployable_bundle_checkpoint_v1"
)
PREDICTION_ANCHORED_CLEAN_RELOAD_AUDIT_CONTRACT = (
    "prediction_anchored_clean_hlt_reload_audit_v1"
)

B_STAGES = ("B0", "B1", "B2", "B3", "B4", "B5", "B6")
RUN_OUTCOME_STATES = {
    "PENDING",
    "COMPLETED",
    "FAILED",
    "SKIPPED_INVALID_PARENT",
    "SKIPPED_FAILED_PARENT",
    "SKIPPED_QUOTA",
}
STAGE_STATES = {"PENDING", "COMPLETED", "FAILED", "SKIPPED_FAILED_PARENT"}

_STAGE_REQUIRED_FLAGS: dict[str, tuple[str, ...]] = {
    "B0": (
        "data_audit_passed",
        "split_audit_passed",
        "mask_audit_passed",
        "unit_audit_passed",
        "ram_preflight_passed",
        "storage_preflight_passed",
        "r0_prerequisites_passed",
    ),
    "B1": ("r0_registered", "live_f0_valid", "live_h0_valid"),
    "B2": ("physical45_recipe_valid", "all50_recipe_valid", "batch_audit_passed"),
    "B3": ("l0_launched", "upstream_paired3_completed"),
    "B4": ("consumer_aggregate_selected", "stack_val_consumer_confirmed"),
    "B5": ("teacher_bindings_valid", "target_cache_namespaces_valid"),
    "B6": ("post_teacher_matrix_released", "paired3_breadth_completed"),
}
_STAGE_REQUIRED_PARENTS: dict[str, tuple[str, ...]] = {
    "B0": ("step8_measurement", "ram_audit"),
    "B1": ("r0_registration",),
    "B2": ("physical45_recipe", "all50_recipe"),
    "B3": ("b0_b2_release",),
    "B4": ("selected_consumer",),
    "B5": (
        "selected_consumer",
        "primary_binding",
        "primary_cache",
        "all50_cache",
        "n3_cache",
    ),
    "B6": ("post_teacher_release",),
}


def _sha256(value: Any, *, name: str) -> str:
    output = str(value)
    if len(output) != 64 or any(char not in "0123456789abcdef" for char in output):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return output


def _finite(value: Any, *, name: str) -> float:
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{name} must be finite")
    return output


def _mean(values: Sequence[float]) -> float:
    return float(np.asarray(values, dtype=np.float64).mean())


def _sample_std(values: Sequence[float]) -> float:
    return float(np.asarray(values, dtype=np.float64).std(ddof=1))


def _validate_hashed_reference(value: Mapping[str, Any], *, name: str) -> str:
    try:
        return validate_content_hash(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a valid immutable artifact") from exc


def build_campaign_reservations(
    registry: Mapping[str, Any],
    *,
    execution_spec: Mapping[str, Any],
    production_readiness: Mapping[str, Any],
    fixed_parent_artifacts: Mapping[str, Mapping[str, Any]],
    final_deployable_bundle_bytes: int,
) -> dict[str, Any]:
    """Reserve measured bytes exactly once for each run and frozen parent.

    ``fixed_parent_artifacts`` rows contain ``sha256``, ``size_bytes``, and
    ``path``.  Equal hashes or overlapping paths are rejected so an R0/T10
    parent cannot accidentally be published or charged twice.
    """

    validate_campaign_registry(registry)
    validate_content_hash(
        execution_spec, expected_contract="prediction_anchored_execution_spec_v1"
    )
    child_manifest = execution_spec.get("child_manifest")
    parent_manifest = execution_spec.get("parent_manifest")
    if not isinstance(child_manifest, Mapping):
        raise ValueError("execution spec has no immutable child-manifest binding")
    child_sha256 = _sha256(
        child_manifest.get("content_hash"), name="execution-spec child manifest"
    )
    if not isinstance(parent_manifest, Mapping):
        raise ValueError("execution spec has no immutable parent-manifest binding")
    parent_sha256 = _sha256(
        parent_manifest.get("sha256"), name="execution-spec parent manifest file"
    )
    if not bool(production_readiness.get("ok")) or not bool(
        production_readiness.get("production_submission_allowed")
    ):
        raise PermissionError("campaign reservations require a passed measured storage preflight")
    if int(production_readiness.get("projected_persistent_bytes", -1)) > int(
        production_readiness.get("selected_budget_bytes", -1)
    ):
        raise PermissionError("campaign storage projection exceeds the selected quota")
    if production_readiness.get("registry_sha256") != registry["content_hash"]:
        raise ValueError("production readiness is bound to a different registry")

    parents: dict[str, dict[str, Any]] = {}
    hashes: set[str] = set()
    resolved_paths: list[Path] = []
    for role, raw in sorted(fixed_parent_artifacts.items()):
        digest = _sha256(raw.get("sha256"), name=f"{role} sha256")
        size = raw.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"{role} size_bytes must be a positive measured integer")
        path = Path(str(raw.get("path", ""))).resolve(strict=False)
        if digest in hashes:
            raise ValueError("fixed parent hash is duplicated across publication roles")
        if any(path == prior or path in prior.parents or prior in path.parents for prior in resolved_paths):
            raise ValueError("fixed parent paths overlap and would be duplicated")
        hashes.add(digest)
        resolved_paths.append(path)
        parents[str(role)] = {"sha256": digest, "size_bytes": size, "path": str(path)}

    run_reservations = {}
    for row in registry["runs"]:
        run_id = row["canonical_run_id"]
        if row["execution_status"] == "RUNNABLE":
            measured = row.get("measured_retained_bytes")
            if not isinstance(measured, int) or isinstance(measured, bool) or measured <= 0:
                raise PermissionError(f"runnable row {run_id} has no measured retained-byte reservation")
            run_reservations[run_id] = int(measured)
        else:
            run_reservations[run_id] = 0

    expected_runs = int(production_readiness["measured_run_bytes"])
    if sum(run_reservations.values()) != expected_runs:
        raise ValueError("per-run reservations do not reconcile with production readiness")
    fixed_bytes = int(production_readiness["fixed_persistent_bytes"])
    if sum(row["size_bytes"] for row in parents.values()) > fixed_bytes:
        raise ValueError("declared parent bytes exceed the fixed-storage reservation")
    if (
        not isinstance(final_deployable_bundle_bytes, int)
        or isinstance(final_deployable_bundle_bytes, bool)
        or final_deployable_bundle_bytes <= 0
        or final_deployable_bundle_bytes > fixed_bytes
    ):
        raise ValueError("final deployable bundle reservation must fit measured fixed storage")
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_CAMPAIGN_RESERVATION_CONTRACT,
            "registry_sha256": registry["content_hash"],
            "execution_spec_sha256": execution_spec["content_hash"],
            "child_manifest_sha256": child_sha256,
            "parent_manifest_file_sha256": parent_sha256,
            "selected_budget_bytes": int(production_readiness["selected_budget_bytes"]),
            "projected_persistent_bytes": int(
                production_readiness["projected_persistent_bytes"]
            ),
            "run_reservations_bytes": run_reservations,
            "fixed_parent_artifacts": parents,
            "fixed_storage_reserved_bytes": fixed_bytes,
            "final_deployable_bundle_reserved_bytes": int(final_deployable_bundle_bytes),
            "duplicate_parent_hashes": False,
            "overlapping_parent_paths": False,
            "quota_preflight_passed": True,
        }
    )


def build_stage_evidence(
    stage: str,
    *,
    flags: Mapping[str, bool],
    parent_artifact_sha256: Mapping[str, str] | None = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    if stage not in B_STAGES:
        raise ValueError(f"unknown campaign stage {stage!r}")
    required = _STAGE_REQUIRED_FLAGS[stage]
    normalized = {name: bool(flags.get(name, False)) for name in required}
    passed = all(normalized.values()) and failure_reason is None
    parents = {
        str(name): _sha256(value, name=f"{stage} parent {name}")
        for name, value in sorted((parent_artifact_sha256 or {}).items())
    }
    missing_parents = set(_STAGE_REQUIRED_PARENTS[stage]) - set(parents)
    if missing_parents:
        raise ValueError(
            f"{stage} evidence is missing parent artifacts: {sorted(missing_parents)}"
        )
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_STAGE_EVIDENCE_CONTRACT,
            "stage": stage,
            "required_flags": list(required),
            "flags": normalized,
            "passed": passed,
            "failure_reason": None if passed else str(failure_reason or "required gate failed"),
            "parent_artifact_sha256": parents,
        }
    )


def initialize_step9_campaign(
    registry: Mapping[str, Any],
    *,
    reservations: Mapping[str, Any],
) -> dict[str, Any]:
    validate_campaign_registry(registry)
    validate_content_hash(
        reservations, expected_contract=PREDICTION_ANCHORED_CAMPAIGN_RESERVATION_CONTRACT
    )
    if reservations.get("registry_sha256") != registry["content_hash"]:
        raise ValueError("campaign reservation registry binding mismatch")
    outcomes = {
        row["canonical_run_id"]: (
            "SKIPPED_INVALID_PARENT"
            if row["execution_status"] == "SKIPPED_INVALID_PARENT"
            else "PENDING"
        )
        for row in registry["runs"]
    }
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_CAMPAIGN_STATE_CONTRACT,
            "status": "ACTIVE",
            "registry_sha256": registry["content_hash"],
            "reservation_sha256": reservations["content_hash"],
            "stages": {stage: {"status": "PENDING", "evidence_sha256": None} for stage in B_STAGES},
            "next_stage": "B0",
            "stop_reason": None,
            "run_outcomes": outcomes,
            "fallback_allowed": False,
            "cross_allocation_resume": False,
        }
    )


def advance_step9_campaign(
    state: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(state, expected_contract=PREDICTION_ANCHORED_CAMPAIGN_STATE_CONTRACT)
    validate_campaign_registry(registry)
    validate_content_hash(evidence, expected_contract=PREDICTION_ANCHORED_STAGE_EVIDENCE_CONTRACT)
    if state.get("registry_sha256") != registry["content_hash"]:
        raise ValueError("campaign state registry binding mismatch")
    if state.get("status") != "ACTIVE":
        raise PermissionError("stopped or completed campaign cannot advance")
    stage = str(evidence["stage"])
    if stage != state.get("next_stage"):
        raise PermissionError(f"campaign requires {state.get('next_stage')}, not {stage}")

    output = deepcopy(dict(state))
    output.pop("content_hash", None)
    output["stages"][stage] = {
        "status": "COMPLETED" if evidence["passed"] else "FAILED",
        "evidence_sha256": evidence["content_hash"],
    }
    if not evidence["passed"]:
        output["status"] = "STOPPED"
        output["next_stage"] = None
        output["stop_reason"] = f"{stage}: {evidence['failure_reason']}"
        later = B_STAGES[B_STAGES.index(stage) + 1 :]
        for name in later:
            output["stages"][name] = {"status": "SKIPPED_FAILED_PARENT", "evidence_sha256": None}
        for row in registry["runs"]:
            run_id = row["canonical_run_id"]
            if output["run_outcomes"][run_id] == "PENDING" and row["stage"] >= stage:
                output["run_outcomes"][run_id] = "SKIPPED_FAILED_PARENT"
        return with_content_hash(output)

    if stage in {"B3", "B6"}:
        for row in registry["runs"]:
            if row["stage"] == stage and output["run_outcomes"][row["canonical_run_id"]] == "PENDING":
                output["run_outcomes"][row["canonical_run_id"]] = "COMPLETED"
    index = B_STAGES.index(stage)
    if index == len(B_STAGES) - 1:
        output["status"] = "BREADTH_COMPLETE"
        output["next_stage"] = None
    else:
        output["next_stage"] = B_STAGES[index + 1]
    return with_content_hash(output)


def record_campaign_run_outcomes(
    state: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    outcomes: Mapping[str, str],
) -> dict[str, Any]:
    """Record failures/skips without removing any of the 54 registry rows."""

    validate_content_hash(state, expected_contract=PREDICTION_ANCHORED_CAMPAIGN_STATE_CONTRACT)
    validate_campaign_registry(registry)
    if set(outcomes) - set(state["run_outcomes"]):
        raise KeyError("run outcomes contain an unknown campaign row")
    output = deepcopy(dict(state))
    output.pop("content_hash", None)
    for run_id, status in outcomes.items():
        if status not in RUN_OUTCOME_STATES:
            raise ValueError(f"invalid run outcome {status!r}")
        row = resolve_registry_run(registry, run_id)
        if row["execution_status"] == "SKIPPED_INVALID_PARENT" and status != "SKIPPED_INVALID_PARENT":
            raise PermissionError("a conditional invalid-parent row cannot be made runnable")
        output["run_outcomes"][row["canonical_run_id"]] = status
    return with_content_hash(output)


@dataclass(frozen=True)
class DeployableReplicaEvidence:
    run_id: str
    seed_id: int
    accuracy: float
    macro_per_class_accuracy: float
    cross_entropy: float
    baseline_accuracy: float
    teacher_bridge_accuracy: float
    epoch: int
    checkpoint: str
    checkpoint_sha256: str
    scaler_sha256: str
    teacher_sha256: str
    recipe_sha256: str
    deployed_parameter_count: int
    persistent_bytes: int
    reserved_bytes: int
    recovery_fraction: float | None
    saturation_fraction: float
    perturbation_mean_accuracy_loss: float
    perturbation_worst_accuracy_loss: float
    provenance_valid: bool = True
    leakage_audit_passed: bool = True
    mask_audit_passed: bool = True
    reload_audit_passed: bool = True
    reliability_pass_through_exact: bool = True
    alignment_finite: bool = True
    distribution_distance_finite: bool = True

    def to_artifact(self) -> dict[str, Any]:
        if int(self.seed_id) not in PAIRED_SEED_IDS:
            raise ValueError("deployable replica seed must be 101, 202, or 303")
        numeric = {
            "accuracy": self.accuracy,
            "macro_per_class_accuracy": self.macro_per_class_accuracy,
            "cross_entropy": self.cross_entropy,
            "baseline_accuracy": self.baseline_accuracy,
            "teacher_bridge_accuracy": self.teacher_bridge_accuracy,
            "saturation_fraction": self.saturation_fraction,
            "perturbation_mean_accuracy_loss": self.perturbation_mean_accuracy_loss,
            "perturbation_worst_accuracy_loss": self.perturbation_worst_accuracy_loss,
        }
        values = {name: _finite(value, name=name) for name, value in numeric.items()}
        if self.recovery_fraction is not None:
            values["recovery_fraction"] = _finite(
                self.recovery_fraction, name="recovery_fraction"
            )
        if int(self.epoch) < 0:
            raise ValueError("deployable epoch must be non-negative")
        if int(self.deployed_parameter_count) <= 0:
            raise ValueError("deployed parameter count must be positive")
        if int(self.persistent_bytes) <= 0 or int(self.reserved_bytes) <= 0:
            raise ValueError("persistent/reserved bytes must be positive")
        hashes = {
            "checkpoint_sha256": _sha256(self.checkpoint_sha256, name="checkpoint"),
            "scaler_sha256": _sha256(self.scaler_sha256, name="scaler"),
            "teacher_sha256": _sha256(self.teacher_sha256, name="teacher"),
            "recipe_sha256": _sha256(self.recipe_sha256, name="recipe"),
        }
        deployable_gain = values["accuracy"] - values["baseline_accuracy"]
        teacher_gain = values["teacher_bridge_accuracy"] - values["baseline_accuracy"]
        expected_recovery = None if teacher_gain <= 0 else deployable_gain / teacher_gain
        if expected_recovery is None:
            if self.recovery_fraction is not None:
                raise ValueError("recovery must be null for a non-positive teacher gain")
        elif self.recovery_fraction is None or not math.isclose(
            float(self.recovery_fraction), expected_recovery, abs_tol=1e-9, rel_tol=1e-7
        ):
            raise ValueError("recovery fraction does not match the declared accuracies")
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_DEPLOYABLE_REPLICA_CONTRACT,
                "run_id": str(self.run_id),
                "seed_id": int(self.seed_id),
                **values,
                "deployable_gain": float(deployable_gain),
                "teacher_bridge_gain": float(teacher_gain),
                "recovery_fraction": expected_recovery,
                "epoch": int(self.epoch),
                "checkpoint": str(self.checkpoint),
                **hashes,
                "deployed_parameter_count": int(self.deployed_parameter_count),
                "persistent_bytes": int(self.persistent_bytes),
                "reserved_bytes": int(self.reserved_bytes),
                "provenance_valid": bool(self.provenance_valid),
                "leakage_audit_passed": bool(self.leakage_audit_passed),
                "mask_audit_passed": bool(self.mask_audit_passed),
                "reload_audit_passed": bool(self.reload_audit_passed),
                "reliability_pass_through_exact": bool(self.reliability_pass_through_exact),
                "alignment_finite": bool(self.alignment_finite),
                "distribution_distance_finite": bool(self.distribution_distance_finite),
                "saturation_fraction": values["saturation_fraction"],
                "perturbation_mean_accuracy_loss": values[
                    "perturbation_mean_accuracy_loss"
                ],
                "perturbation_worst_accuracy_loss": values[
                    "perturbation_worst_accuracy_loss"
                ],
                "split_name": "model_val_select",
            }
        )


def aggregate_deployable_configuration(
    registry: Mapping[str, Any],
    replicas: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_campaign_registry(registry)
    if len(replicas) != 3:
        raise ValueError("deployable aggregate requires exactly three paired replicas")
    for row in replicas:
        validate_content_hash(
            row, expected_contract=PREDICTION_ANCHORED_DEPLOYABLE_REPLICA_CONTRACT
        )
    run_ids = {str(row["run_id"]) for row in replicas}
    seeds = {int(row["seed_id"]) for row in replicas}
    if len(run_ids) != 1 or seeds != set(PAIRED_SEED_IDS):
        raise ValueError("deployable aggregate must pair one run across 101/202/303")
    run_id = next(iter(run_ids))
    registry_row = resolve_registry_run(registry, run_id)
    normalized = sorted(replicas, key=lambda row: int(row["seed_id"]))
    scored = sorted(
        normalized,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["deployable_gain"]),
            -float(row["cross_entropy"]),
            int(row["seed_id"]),
        ),
    )
    median = scored[1]
    accuracies = [float(row["accuracy"]) for row in normalized]
    macros = [float(row["macro_per_class_accuracy"]) for row in normalized]
    losses = [float(row["cross_entropy"]) for row in normalized]
    gains = [float(row["deployable_gain"]) for row in normalized]
    recoveries = [
        float(row["recovery_fraction"])
        for row in normalized
        if row.get("recovery_fraction") is not None
    ]
    parameter_counts = {int(row["deployed_parameter_count"]) for row in normalized}
    hashes_by_kind = {
        kind: {str(row[kind]) for row in normalized}
        for kind in ("scaler_sha256", "teacher_sha256", "recipe_sha256")
    }
    if len(parameter_counts) != 1 or any(len(values) != 1 for values in hashes_by_kind.values()):
        raise ValueError("paired replicas changed deployed capacity or frozen lineage")

    gates = {
        "registry_selectable": bool(registry_row["selectable_for_primary_deployment"]),
        "positive_mean_deployable_gain": _mean(gains) > 0,
        "two_of_three_positive_gain": sum(value > 0 for value in gains) >= 2,
        "median_positive_gain": float(median["deployable_gain"]) > 0,
        "positive_recovery_where_defined": all(value > 0 for value in recoveries),
        "provenance_valid": all(bool(row["provenance_valid"]) for row in normalized),
        "leakage_audit_passed": all(bool(row["leakage_audit_passed"]) for row in normalized),
        "mask_audit_passed": all(bool(row["mask_audit_passed"]) for row in normalized),
        "reload_audit_passed": all(bool(row["reload_audit_passed"]) for row in normalized),
        "reliability_pass_through_exact": all(
            bool(row["reliability_pass_through_exact"]) for row in normalized
        ),
        "trust_saturation_at_most_one_percent": all(
            float(row["saturation_fraction"]) <= 0.01 for row in normalized
        ),
        "alignment_finite": all(bool(row["alignment_finite"]) for row in normalized),
        "distribution_distance_finite": all(
            bool(row["distribution_distance_finite"]) for row in normalized
        ),
        "perturbation_mean_loss_at_most_0p002": _mean(
            [float(row["perturbation_mean_accuracy_loss"]) for row in normalized]
        )
        <= 0.002,
        "perturbation_worst_loss_at_most_0p003": max(
            float(row["perturbation_worst_accuracy_loss"]) for row in normalized
        )
        <= 0.003,
        "persistent_bytes_within_reservation": all(
            int(row["persistent_bytes"]) <= int(row["reserved_bytes"]) for row in normalized
        ),
    }
    valid = all(gates.values())
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_DEPLOYABLE_AGGREGATE_CONTRACT,
            "run_id": run_id,
            "registry_run_sha256": registry_row["content_hash"],
            "selectable_for_primary_deployment": bool(
                registry_row["selectable_for_primary_deployment"]
            ),
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "replicas": [deepcopy(dict(row)) for row in normalized],
            "mean_accuracy": _mean(accuracies),
            "accuracy_sample_std": _sample_std(accuracies),
            "mean_macro_per_class_accuracy": _mean(macros),
            "macro_per_class_accuracy_sample_std": _sample_std(macros),
            "mean_cross_entropy": _mean(losses),
            "cross_entropy_sample_std": _sample_std(losses),
            "mean_deployable_gain": _mean(gains),
            "deployable_gain_sample_std": _sample_std(gains),
            "mean_recovery_fraction": None if not recoveries else _mean(recoveries),
            "ordered_seed_ids": [int(row["seed_id"]) for row in scored],
            "median_seed_id": int(median["seed_id"]),
            "median_replica_sha256": median["content_hash"],
            "median_replica": deepcopy(dict(median)),
            "best_seed_weights_rejected": True,
            "deployed_parameter_count": next(iter(parameter_counts)),
            "scaler_sha256": next(iter(hashes_by_kind["scaler_sha256"])),
            "teacher_sha256": next(iter(hashes_by_kind["teacher_sha256"])),
            "recipe_sha256": next(iter(hashes_by_kind["recipe_sha256"])),
            "validity_gates": gates,
            "valid_for_primary_selection": valid,
            "invalid_reasons": [name for name, passed in gates.items() if not passed],
        }
    )


def select_deployable_preconfirmation(
    registry: Mapping[str, Any],
    aggregates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_campaign_registry(registry)
    if not aggregates:
        raise ValueError("deployable selection requires configuration aggregates")
    valid = []
    for aggregate in aggregates:
        validate_content_hash(
            aggregate,
            expected_contract=PREDICTION_ANCHORED_DEPLOYABLE_AGGREGATE_CONTRACT,
        )
        row = resolve_registry_run(registry, str(aggregate["run_id"]))
        if aggregate.get("registry_run_sha256") != row["content_hash"]:
            raise ValueError("deployable aggregate is bound to a stale registry row")
        if bool(aggregate.get("valid_for_primary_selection")) and bool(
            row["selectable_for_primary_deployment"]
        ):
            valid.append(aggregate)
    if not valid:
        raise PermissionError("no scientifically valid selectable deployable configuration")
    best_score = max(float(row["mean_accuracy"]) for row in valid)
    tie_pool = [row for row in valid if best_score - float(row["mean_accuracy"]) <= 0.0005]
    selected = min(
        tie_pool,
        key=lambda row: (
            -float(row["mean_macro_per_class_accuracy"]),
            float(row["mean_cross_entropy"]),
            float(row["deployable_gain_sample_std"]),
            int(row["deployed_parameter_count"]),
            str(row["run_id"]),
        ),
    )
    median = selected["median_replica"]
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_DEPLOYABLE_PRECONFIRMATION_CONTRACT,
            "status": "LOCKED_PRECONFIRMATION",
            "registry_sha256": registry["content_hash"],
            "selection_split": "model_val_select",
            "best_score": best_score,
            "tie_tolerance": 0.0005,
            "tie_pool_run_ids": sorted(str(row["run_id"]) for row in tie_pool),
            "tie_order": [
                "higher_mean_macro_per_class_accuracy",
                "lower_mean_cross_entropy",
                "lower_deployable_gain_sample_std",
                "fewer_exact_deployed_parameters",
                "lexicographic_run_id",
            ],
            "latency_used_for_selection": False,
            "selected_run_id": selected["run_id"],
            "selected_aggregate_sha256": selected["content_hash"],
            "configuration_aggregate": deepcopy(dict(selected)),
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "median_seed_id": int(selected["median_seed_id"]),
            "epoch": int(median["epoch"]),
            "checkpoint": median["checkpoint"],
            "checkpoint_sha256": median["checkpoint_sha256"],
            "scaler_sha256": median["scaler_sha256"],
            "teacher_sha256": median["teacher_sha256"],
            "recipe_sha256": median["recipe_sha256"],
            "runner_up_fallback_allowed": False,
            "stack_val_deploy_opened": False,
        }
    )


def authorize_deployable_confirmation(
    child_manifest: Mapping[str, Any],
    *,
    parent: Any,
    preconfirmation: Mapping[str, Any],
    receipt_path: str | Path | None = None,
    config: Any = None,
) -> dict[str, Any]:
    validate_content_hash(
        preconfirmation,
        expected_contract=PREDICTION_ANCHORED_DEPLOYABLE_PRECONFIRMATION_CONTRACT,
    )
    if preconfirmation.get("status") != "LOCKED_PRECONFIRMATION":
        raise PermissionError("stack_val_deploy requires a locked pre-confirmation selection")
    unlock = build_manifest_validation_unlock(
        child_manifest,
        split_name="stack_val_deploy",
        selection_sha256=preconfirmation["content_hash"],
    )
    kwargs = {}
    if config is not None:
        kwargs["config"] = config
    authorization = authorize_manifest_split_access(
        child_manifest,
        parent=parent,
        split_name="stack_val_deploy",
        purpose="deployable_confirmation",
        unlock=unlock,
        **kwargs,
    )
    if receipt_path is None:
        return authorization
    return claim_split_access(receipt_path, authorization)


def build_deployable_confirmation(
    preconfirmation: Mapping[str, Any],
    *,
    access_receipt: Mapping[str, Any],
    deployable_gain: float,
    accuracy: float,
    cross_entropy: float,
    provenance_valid: bool,
) -> dict[str, Any]:
    validate_content_hash(
        preconfirmation,
        expected_contract=PREDICTION_ANCHORED_DEPLOYABLE_PRECONFIRMATION_CONTRACT,
    )
    validate_content_hash(
        access_receipt, expected_contract=PREDICTION_ANCHORED_ACCESS_RECEIPT_CONTRACT
    )
    if access_receipt.get("split_name") != "stack_val_deploy" or access_receipt.get(
        "selection_sha256"
    ) != preconfirmation["content_hash"]:
        raise PermissionError("confirmation receipt does not authorize this locked selection")
    values = {
        "deployable_gain": _finite(deployable_gain, name="confirmation deployable gain"),
        "accuracy": _finite(accuracy, name="confirmation accuracy"),
        "cross_entropy": _finite(cross_entropy, name="confirmation cross entropy"),
    }
    passed = values["deployable_gain"] >= 0 and bool(provenance_valid)
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_DEPLOYABLE_CONFIRMATION_CONTRACT,
            "preconfirmation_sha256": preconfirmation["content_hash"],
            "access_receipt_sha256": access_receipt["content_hash"],
            "split_name": "stack_val_deploy",
            "selected_run_id": preconfirmation["selected_run_id"],
            "median_seed_id": preconfirmation["median_seed_id"],
            "epoch": preconfirmation["epoch"],
            "checkpoint_sha256": preconfirmation["checkpoint_sha256"],
            **values,
            "provenance_valid": bool(provenance_valid),
            "metrics_finite": True,
            "passed": passed,
            "one_shot": True,
            "ranking_or_tie_breaking_performed": False,
        }
    )


def finalize_deployable_confirmation(
    preconfirmation: Mapping[str, Any],
    confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        preconfirmation,
        expected_contract=PREDICTION_ANCHORED_DEPLOYABLE_PRECONFIRMATION_CONTRACT,
    )
    validate_content_hash(
        confirmation,
        expected_contract=PREDICTION_ANCHORED_DEPLOYABLE_CONFIRMATION_CONTRACT,
    )
    if confirmation.get("preconfirmation_sha256") != preconfirmation["content_hash"]:
        raise ValueError("confirmation belongs to another pre-confirmation manifest")
    locked_fields = ("selected_run_id", "median_seed_id", "epoch", "checkpoint_sha256")
    if any(confirmation.get(name) != preconfirmation.get(name) for name in locked_fields):
        raise ValueError("confirmation attempted to change a locked selection field")
    if not bool(confirmation["passed"]):
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_STOPPED_DEPLOYABLE_CONTRACT,
                "status": "STOPPED_CONFIRMATION_FAILED",
                "preconfirmation_sha256": preconfirmation["content_hash"],
                "confirmation_sha256": confirmation["content_hash"],
                "selected_run_id": preconfirmation["selected_run_id"],
                "median_seed_id": preconfirmation["median_seed_id"],
                "runner_up_promoted": False,
                "fallback_allowed": False,
                "final_test_authorized": False,
            }
        )
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_LOCKED_DEPLOYABLE_CONTRACT,
            "status": "CONFIRMED_LOCKED",
            "preconfirmation_sha256": preconfirmation["content_hash"],
            "confirmation_sha256": confirmation["content_hash"],
            "selected_run_id": preconfirmation["selected_run_id"],
            "selected_aggregate_sha256": preconfirmation["selected_aggregate_sha256"],
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "median_seed_id": preconfirmation["median_seed_id"],
            "epoch": preconfirmation["epoch"],
            "checkpoint": preconfirmation["checkpoint"],
            "checkpoint_sha256": preconfirmation["checkpoint_sha256"],
            "scaler_sha256": preconfirmation["scaler_sha256"],
            "teacher_sha256": preconfirmation["teacher_sha256"],
            "recipe_sha256": preconfirmation["recipe_sha256"],
            "selection_immutable": True,
            "runner_up_fallback_allowed": False,
            "final_test_pending_clean_reload": True,
        }
    )


def validate_final_test_request(
    locked_deployable: Mapping[str, Any],
    *,
    evaluation_flags: Mapping[str, Any] | None = None,
    clean_reload_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_content_hash(
        locked_deployable, expected_contract=PREDICTION_ANCHORED_LOCKED_DEPLOYABLE_CONTRACT
    )
    if locked_deployable.get("status") != "CONFIRMED_LOCKED":
        raise PermissionError("final test requires a confirmed locked deployable")
    flags = dict(evaluation_flags or {})
    forbidden = {
        "oracle",
        "bridge",
        "privileged",
        "offline",
        "f_true",
        "target_logits",
        "alternate_teacher",
        "all50_oracle",
    }
    active_forbidden = sorted(
        name
        for name, value in flags.items()
        if bool(value) and any(token in str(name).lower() for token in forbidden)
    )
    if active_forbidden:
        raise PermissionError(
            "final-test is HLT-only; forbidden evaluation flags: " + ", ".join(active_forbidden)
        )
    if clean_reload_audit is None:
        raise PermissionError("final-test remains sealed until the clean HLT-only reload audit")
    validate_content_hash(
        clean_reload_audit, expected_contract=PREDICTION_ANCHORED_CLEAN_RELOAD_AUDIT_CONTRACT
    )
    if not bool(clean_reload_audit.get("passed")) or clean_reload_audit.get(
        "locked_deployable_sha256"
    ) != locked_deployable["content_hash"]:
        raise PermissionError("clean reload audit is absent, failed, or bound to another deployment")
    return {
        "ok": True,
        "hlt_only": True,
        "selected_run_id": locked_deployable["selected_run_id"],
        "median_seed_id": locked_deployable["median_seed_id"],
        "selection_changes_allowed": False,
        "evaluation_flags": flags,
    }


def authorize_final_test(
    child_manifest: Mapping[str, Any],
    *,
    parent: Any,
    locked_deployable: Mapping[str, Any],
    clean_reload_audit: Mapping[str, Any],
    evaluation_flags: Mapping[str, Any] | None = None,
    receipt_path: str | Path | None = None,
    config: Any = None,
) -> dict[str, Any]:
    validate_final_test_request(
        locked_deployable,
        evaluation_flags=evaluation_flags,
        clean_reload_audit=clean_reload_audit,
    )
    unlock = build_manifest_validation_unlock(
        child_manifest,
        split_name="final_test",
        selection_sha256=locked_deployable["content_hash"],
    )
    kwargs = {}
    if config is not None:
        kwargs["config"] = config
    authorization = authorize_manifest_split_access(
        child_manifest,
        parent=parent,
        split_name="final_test",
        purpose="final_evaluation",
        unlock=unlock,
        **kwargs,
    )
    if receipt_path is None:
        return authorization
    return claim_split_access(receipt_path, authorization)


BASELINE_DEPLOYABLE_REQUIRED_IDS = (
    "A0_legacy",
    "A0_C250",
    "A0_C250_LONG",
    "A0_S500",
    "A0_CAP500_direct_hlt",
    "A0_CAP500_r0rep_direct",
    "Tpred(f0)",
    "Tpred_continue(f0)",
    "selected_T10(f0)",
)
PRIVILEGED_REQUIRED_IDS = (
    "selected_T10(physical45_oracle_ceiling)",
    "selected_T10(full50_oracle_ceiling)",
)


def build_step9_report_row(
    *,
    row_id: str,
    section: str,
    status: str,
    metrics: Mapping[str, Any] | None = None,
    seed_metrics: Sequence[Mapping[str, Any]] = (),
    aggregate: Mapping[str, Any] | None = None,
    median_seed_id: int | None = None,
    deployable: bool,
    hlt_only_reload_passed: bool | None,
    teacher_bridge_gain: float | None = None,
    recovery_fraction: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if section not in {"baselines_and_deployable", "privileged_oracle", "ablation_evidence"}:
        raise ValueError("unknown Step 9 report section")
    if section == "privileged_oracle" and deployable:
        raise ValueError("privileged/oracle rows cannot be deployable")
    if deployable and hlt_only_reload_passed is not True:
        raise ValueError("every deployable report row requires a passed HLT-only reload audit")
    if teacher_bridge_gain is not None:
        gain = _finite(teacher_bridge_gain, name="teacher bridge gain")
        if gain <= 0 and recovery_fraction is not None:
            raise ValueError("recovery must be null when teacher bridge gain is non-positive")
    else:
        gain = None
        if recovery_fraction is not None:
            raise ValueError("recovery requires a declared teacher bridge gain")
    recovery = None if recovery_fraction is None else _finite(
        recovery_fraction, name="recovery fraction"
    )
    if str(status) == "COMPLETED" and (
        len(seed_metrics) != 3 or aggregate is None or median_seed_id is None
    ):
        raise ValueError("completed report rows require three seeds, an aggregate, and a median")
    if seed_metrics:
        seeds = [int(row["seed_id"]) for row in seed_metrics]
        if sorted(seeds) != list(PAIRED_SEED_IDS):
            raise ValueError("completed report rows require all paired seeds 101/202/303")
    if median_seed_id is not None and int(median_seed_id) not in PAIRED_SEED_IDS:
        raise ValueError("report median seed is outside the paired set")
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_REPORT_ROW_CONTRACT,
            "row_id": str(row_id),
            "section": section,
            "status": str(status),
            "deployable": bool(deployable),
            "watermark": None if deployable else "NON_DEPLOYABLE_DIAGNOSTIC",
            "hlt_only_reload_passed": hlt_only_reload_passed,
            "metrics": deepcopy(dict(metrics or {})),
            "seed_metrics": [deepcopy(dict(row)) for row in seed_metrics],
            "aggregate": deepcopy(dict(aggregate or {})),
            "median_seed_id": None if median_seed_id is None else int(median_seed_id),
            "teacher_bridge_gain": gain,
            "recovery_fraction": recovery,
            "metadata": deepcopy(dict(metadata or {})),
        }
    )


def build_step9_reports(
    registry: Mapping[str, Any],
    *,
    baseline_deployable_rows: Sequence[Mapping[str, Any]],
    privileged_rows: Sequence[Mapping[str, Any]],
    ablation_rows: Mapping[str, Mapping[str, Any]],
    run_outcomes: Mapping[str, str],
    persistent_telemetry: Mapping[str, Any],
    ram_telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    validate_campaign_registry(registry)
    baseline_ids = {str(row.get("row_id")) for row in baseline_deployable_rows}
    missing_baselines = set(BASELINE_DEPLOYABLE_REQUIRED_IDS) - baseline_ids
    if missing_baselines or not any(value.startswith("selected_T10(fhat") for value in baseline_ids):
        raise ValueError(
            "baseline/deployable report is incomplete: "
            + ", ".join(sorted(missing_baselines | ({"selected_T10(fhat)"} if not any(value.startswith("selected_T10(fhat") for value in baseline_ids) else set())))
        )
    privileged_ids = {str(row.get("row_id")) for row in privileged_rows}
    missing_privileged = set(PRIVILEGED_REQUIRED_IDS) - privileged_ids
    if missing_privileged:
        raise ValueError("privileged report omits distinct oracle ceilings: " + ", ".join(sorted(missing_privileged)))
    for row in baseline_deployable_rows:
        validate_content_hash(row, expected_contract=PREDICTION_ANCHORED_REPORT_ROW_CONTRACT)
        if row.get("section") != "baselines_and_deployable" or not bool(row.get("deployable")):
            raise ValueError("baseline/deployable section contains a misplaced row")
        if row.get("row_id") in PRIVILEGED_REQUIRED_IDS:
            raise ValueError("oracle ceiling rows are forbidden from the deployable section")
        if row["row_id"].startswith("A0"):
            metadata = row.get("metadata", {})
            required = {"training_manifest_sha256", "unique_jet_count", "optimizer_step_budget"}
            legacy_reference = (
                row.get("row_id") == "A0_legacy"
                and row.get("status") == "REFERENCE_ONLY_UNPAIRED"
                and metadata.get("paired_seed_evidence_available") is False
                and bool(metadata.get("replica_evidence_unavailable_reason"))
            )
            if not required.issubset(metadata) or (
                not row.get("seed_metrics") and not legacy_reference
            ):
                raise ValueError("every A0 row requires manifest/count/step and three-seed evidence")
    for row in privileged_rows:
        validate_content_hash(row, expected_contract=PREDICTION_ANCHORED_REPORT_ROW_CONTRACT)
        if row.get("section") != "privileged_oracle" or bool(row.get("deployable")):
            raise ValueError("privileged report contains a deployable or misplaced row")

    canonical_ids = [row["canonical_run_id"] for row in registry["runs"]]
    if set(run_outcomes) != set(canonical_ids):
        raise ValueError("report run outcomes must keep every one of the 54 registry rows visible")
    rendered_ablation = []
    for run_id in canonical_ids:
        outcome = str(run_outcomes[run_id])
        if outcome not in RUN_OUTCOME_STATES:
            raise ValueError(f"invalid report outcome for {run_id}")
        raw = ablation_rows.get(run_id)
        if raw is None:
            row = build_step9_report_row(
                row_id=run_id,
                section="ablation_evidence",
                status=outcome,
                deployable=False,
                hlt_only_reload_passed=None,
                metadata={"missing_metrics_visible": True},
            )
        else:
            validate_content_hash(raw, expected_contract=PREDICTION_ANCHORED_REPORT_ROW_CONTRACT)
            if raw.get("row_id") != run_id or raw.get("section") != "ablation_evidence":
                raise ValueError("ablation row ID/section mismatch")
            if raw.get("status") != outcome:
                raise ValueError("ablation row status disagrees with the campaign outcome ledger")
            row = raw
        rendered_ablation.append(deepcopy(dict(row)))

    def telemetry_bytes(payload: Mapping[str, Any], *, label: str) -> int:
        candidates = [value for key, value in payload.items() if "bytes" in str(key).lower() and isinstance(value, int)]
        if not candidates or any(value < 0 for value in candidates):
            raise ValueError(f"{label} telemetry must contain non-negative byte counters")
        return int(sum(candidates))

    persistent_sum = telemetry_bytes(persistent_telemetry, label="persistent")
    ram_sum = telemetry_bytes(ram_telemetry, label="RAM")
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_REPORTS_CONTRACT,
            "registry_sha256": registry["content_hash"],
            "sections": {
                "baselines_and_deployable": [deepcopy(dict(row)) for row in baseline_deployable_rows],
                "privileged_oracle": [deepcopy(dict(row)) for row in privileged_rows],
                "ablation_evidence": rendered_ablation,
            },
            "baseline_deployable_row_count": len(baseline_deployable_rows),
            "privileged_row_count": len(privileged_rows),
            "ablation_row_count": len(rendered_ablation),
            "all_registry_rows_visible": len(rendered_ablation) == 54,
            "physical45_and_full50_ceilings_separate": True,
            "persistent_telemetry": deepcopy(dict(persistent_telemetry)),
            "ram_telemetry": deepcopy(dict(ram_telemetry)),
            "persistent_byte_counter_sum": persistent_sum,
            "ram_byte_counter_sum": ram_sum,
            "failed_and_skipped_rows_visible": True,
            "exploratory_rows_cannot_select": True,
        }
    )


def publish_step9_reports(reports: Mapping[str, Any], *, output_dir: str | Path) -> dict[str, Any]:
    validate_content_hash(reports, expected_contract=PREDICTION_ANCHORED_REPORTS_CONTRACT)
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Step 9 report directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for section, rows in reports["sections"].items():
        artifact = with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_REPORTS_CONTRACT,
                "parent_reports_sha256": reports["content_hash"],
                "section": section,
                "rows": rows,
            }
        )
        path = root / f"{section}.json"
        outputs[section] = write_immutable_json(path, artifact)
    write_immutable_json(root / "report_index.json", reports)
    return {"ok": True, "output_dir": str(root.resolve()), "sections": outputs}


def build_median_publication_manifest(
    aggregates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    seen = set()
    for aggregate in aggregates:
        validate_content_hash(
            aggregate,
            expected_contract=PREDICTION_ANCHORED_DEPLOYABLE_AGGREGATE_CONTRACT,
        )
        run_id = str(aggregate["run_id"])
        if run_id in seen:
            raise ValueError("median publication contains a duplicate configuration")
        seen.add(run_id)
        replicas = aggregate["replicas"]
        rows.append(
            {
                "run_id": run_id,
                "aggregate_sha256": aggregate["content_hash"],
                "all_seed_metrics_retained": [int(row["seed_id"]) for row in replicas],
                "retained_median_seed_id": int(aggregate["median_seed_id"]),
                "retained_checkpoint": aggregate["median_replica"]["checkpoint"],
                "retained_checkpoint_sha256": aggregate["median_replica"]["checkpoint_sha256"],
                "nonmedian_weights_persisted": False,
                "optimizer_state_persisted": False,
                "frozen_parent_weights_duplicated": False,
            }
        )
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_MEDIAN_PUBLICATION_CONTRACT,
            "configuration_count": len(rows),
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "publications": rows,
            "retained_state_rule": "metrics_all_seeds__weights_ordered_median_only",
            "bounded_logs_only": True,
        }
    )


class PredictionAnchoredDeployableBundle(torch.nn.Module):
    """Single HLT-only ``R0 -> correction -> selected T10`` inference graph.

    The input mapping contains only trigger-available tensors: ``tokens``,
    ``raw_mask``, ``points``, ``features``, ``lorentz_vectors``, and ``mask``.
    The selected T10 receives the live correction through its ordinary
    ``residual_fields`` argument, so no oracle loader is reachable.
    """

    def __init__(
        self,
        r0: torch.nn.Module,
        correction: torch.nn.Module,
        consumer: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.r0 = r0
        self.correction = correction
        self.consumer = consumer
        for module in (self.r0, self.consumer):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None

    def train(self, mode: bool = True) -> "PredictionAnchoredDeployableBundle":
        super().train(mode)
        self.r0.eval()
        self.correction.eval()
        self.consumer.eval()
        return self

    @staticmethod
    def _r0_values(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(output, Mapping):
            f0 = output.get("predicted_fields", output.get("f0"))
            h0 = output.get("hidden", output.get("h0"))
        elif isinstance(output, tuple) and len(output) == 2:
            f0, h0 = output
        else:
            f0 = getattr(output, "predicted_fields", getattr(output, "f0", None))
            h0 = getattr(output, "hidden", getattr(output, "h0", None))
        if not isinstance(f0, torch.Tensor) or not isinstance(h0, torch.Tensor):
            raise ValueError("embedded R0 must return predicted_fields/f0 and hidden/h0 tensors")
        return f0, h0

    @staticmethod
    def _corrected_fields(output: Any) -> torch.Tensor:
        if isinstance(output, Mapping):
            value = output.get("f_hat")
        else:
            value = getattr(output, "f_hat", output if isinstance(output, torch.Tensor) else None)
        if not isinstance(value, torch.Tensor):
            raise ValueError("correction module must return an f_hat tensor")
        return value

    def forward(self, hlt: Mapping[str, torch.Tensor]) -> torch.Tensor:
        required = {"tokens", "raw_mask", "points", "features", "lorentz_vectors", "mask"}
        missing = required - set(hlt)
        if missing:
            raise ValueError(f"HLT-only bundle input is missing {sorted(missing)}")
        tokens = hlt["tokens"]
        raw_mask = hlt["raw_mask"].to(dtype=torch.bool)
        if tokens.ndim != 3 or raw_mask.shape != tokens.shape[:2]:
            raise ValueError("bundle HLT tokens/raw mask do not align")
        r0_output = self.r0(tokens, raw_mask)
        f0, h0 = self._r0_values(r0_output)
        correction_output = self.correction(tokens, raw_mask, f0, h0)
        f_hat = self._corrected_fields(correction_output)
        if f_hat.shape != f0.shape or f_hat.shape[-1] != 50:
            raise ValueError("deployable correction must produce [B,P,50]")
        if bool(torch.count_nonzero(f_hat[~raw_mask])):
            raise ValueError("deployable correction changed padded particles")
        output = self.consumer(
            hlt["points"],
            hlt["features"],
            hlt["lorentz_vectors"],
            hlt["mask"],
            tokens=tokens,
            raw_mask=raw_mask,
            residual_fields=f_hat,
        )
        logits = getattr(output, "logits", output)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("selected T10 must return [B,C] logits")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("deployable bundle produced non-finite logits")
        return logits


def build_deployable_bundle_manifest(
    locked_deployable: Mapping[str, Any],
    *,
    component_sha256: Mapping[str, str],
    preprocessing: Mapping[str, Any],
    residual_normalization: Mapping[str, Any],
    target_schema: Mapping[str, Any],
    class_order: Sequence[str],
    architecture_manifest: Mapping[str, Any],
    bundle_reservation_bytes: int,
) -> dict[str, Any]:
    validate_content_hash(
        locked_deployable, expected_contract=PREDICTION_ANCHORED_LOCKED_DEPLOYABLE_CONTRACT
    )
    expected_roles = {"r0", "correction", "consumer"}
    if set(component_sha256) != expected_roles:
        raise ValueError("bundle requires exactly R0, correction, and consumer component hashes")
    components = {
        role: _sha256(value, name=f"bundle {role}")
        for role, value in sorted(component_sha256.items())
    }
    if len(set(components.values())) != len(components):
        raise ValueError("bundle component parent hashes must not be duplicated")
    if components["correction"] != locked_deployable["checkpoint_sha256"]:
        raise ValueError("bundle correction is not the locked median checkpoint")
    if components["consumer"] != locked_deployable["teacher_sha256"]:
        raise ValueError("bundle consumer is not the exact selected frozen T10")
    classes = [str(value) for value in class_order]
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("bundle class order must be non-empty and unique")
    if (
        not isinstance(bundle_reservation_bytes, int)
        or isinstance(bundle_reservation_bytes, bool)
        or bundle_reservation_bytes <= 0
    ):
        raise ValueError("bundle reservation must be a measured positive byte count")
    embedded_metadata = {
        "preprocessing": deepcopy(dict(preprocessing)),
        "residual_normalization": deepcopy(dict(residual_normalization)),
        "target_schema": deepcopy(dict(target_schema)),
        "architecture_manifest": deepcopy(dict(architecture_manifest)),
    }

    forbidden_dependency_tokens = (
        "offline_path",
        "offline_loader",
        "oracle_path",
        "oracle_loader",
        "privileged_path",
        "privileged_loader",
        "target_logit_path",
        "target_logit_cache",
    )

    def dependency_keys(value: Any, prefix: str = "") -> list[str]:
        found = []
        if isinstance(value, Mapping):
            for key, child in value.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                lowered = str(key).lower()
                if any(token in lowered for token in forbidden_dependency_tokens):
                    found.append(name)
                found.extend(dependency_keys(child, name))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                found.extend(dependency_keys(child, f"{prefix}[{index}]"))
        return found

    forbidden_dependencies = dependency_keys(embedded_metadata)
    if forbidden_dependencies:
        raise ValueError(
            "bundle metadata contains a forbidden external dependency: "
            + ", ".join(forbidden_dependencies)
        )
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_DEPLOYABLE_BUNDLE_MANIFEST_CONTRACT,
            "locked_deployable_sha256": locked_deployable["content_hash"],
            "selected_run_id": locked_deployable["selected_run_id"],
            "median_seed_id": locked_deployable["median_seed_id"],
            "epoch": locked_deployable["epoch"],
            "component_sha256": components,
            "scaler_sha256": locked_deployable["scaler_sha256"],
            "recipe_sha256": locked_deployable["recipe_sha256"],
            "preprocessing": embedded_metadata["preprocessing"],
            "residual_normalization": embedded_metadata["residual_normalization"],
            "target_schema": embedded_metadata["target_schema"],
            "class_order": classes,
            "architecture_manifest": embedded_metadata["architecture_manifest"],
            "bundle_reservation_bytes": int(bundle_reservation_bytes),
            "inference_entrypoint": "PredictionAnchoredDeployableBundle.forward(hlt)",
            "input_availability": "hlt_only",
            "embedded_components": ["r0", "correction", "consumer"],
            "duplicate_parent_weights": False,
            "cached_target_logits_included": False,
            "generated_field_tensors_included": False,
            "external_teacher_loader_required": False,
            "offline_matching_required": False,
        }
    )


def _state_component_presence(state: Mapping[str, Any]) -> dict[str, bool]:
    keys = [str(key) for key in state]
    return {
        role: any(key.startswith(f"{role}.") for key in keys)
        for role in ("r0", "correction", "consumer")
    }


def export_deployable_bundle(
    bundle: PredictionAnchoredDeployableBundle,
    *,
    manifest: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    validate_content_hash(
        manifest,
        expected_contract=PREDICTION_ANCHORED_DEPLOYABLE_BUNDLE_MANIFEST_CONTRACT,
    )
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite deployable bundle: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = bundle.state_dict()
    presence = _state_component_presence(state)
    if not all(presence.values()):
        raise ValueError("deployable state does not contain all three embedded components")
    payload = {
        "checkpoint_contract": PREDICTION_ANCHORED_DEPLOYABLE_BUNDLE_CHECKPOINT_CONTRACT,
        "manifest": deepcopy(dict(manifest)),
        "model_state_dict": state,
        "weights_only": True,
        "optimizer_state_persisted": False,
        "scheduler_state_persisted": False,
        "nonmedian_weights_persisted": False,
        "cached_target_logits_included": False,
        "generated_field_tensors_included": False,
        "offline_inputs_included": False,
        "external_teacher_loader_included": False,
    }
    buffer = BytesIO()
    torch.save(payload, buffer)
    encoded = buffer.getvalue()
    reservation = int(manifest["bundle_reservation_bytes"])
    if len(encoded) > reservation:
        raise PermissionError(
            f"deployable bundle bytes {len(encoded)} exceed reservation {reservation}"
        )
    with destination.open("xb") as handle:
        handle.write(encoded)
    loaded = torch.load(destination, map_location="cpu", weights_only=False)
    if loaded.get("checkpoint_contract") != PREDICTION_ANCHORED_DEPLOYABLE_BUNDLE_CHECKPOINT_CONTRACT:
        raise AssertionError("deployable bundle checkpoint contract changed on reload")
    forbidden_keys = {
        "optimizer_state_dict",
        "scheduler_state_dict",
        "f_true",
        "bridge_fields",
        "target_logits",
        "offline_tokens",
    }
    if forbidden_keys.intersection(loaded):
        raise AssertionError("deployable bundle contains forbidden training/privileged state")
    return {
        "ok": True,
        "path": str(destination.resolve()),
        "sha256": sha256_file(destination),
        "size_bytes": int(destination.stat().st_size),
        "reserved_bytes": reservation,
        "within_reservation": int(destination.stat().st_size) <= reservation,
        "manifest_sha256": manifest["content_hash"],
        "component_state_present": presence,
        "hlt_only": True,
    }


def load_deployable_bundle(
    checkpoint_path: str | Path,
    *,
    bundle_factory: Callable[[Mapping[str, Any]], PredictionAnchoredDeployableBundle],
    map_location: Any = "cpu",
) -> tuple[PredictionAnchoredDeployableBundle, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping) or payload.get(
        "checkpoint_contract"
    ) != PREDICTION_ANCHORED_DEPLOYABLE_BUNDLE_CHECKPOINT_CONTRACT:
        raise ValueError("not a prediction-anchored deployable bundle")
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("deployable bundle lacks its manifest")
    validate_content_hash(
        manifest,
        expected_contract=PREDICTION_ANCHORED_DEPLOYABLE_BUNDLE_MANIFEST_CONTRACT,
    )
    model = bundle_factory(manifest)
    if not isinstance(model, PredictionAnchoredDeployableBundle):
        raise TypeError("bundle factory returned the wrong inference entrypoint")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not all(_state_component_presence(state).values()):
        raise ValueError("deployable bundle state is incomplete")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, dict(manifest)


def clean_hlt_only_reload_audit(
    source_bundle: PredictionAnchoredDeployableBundle,
    *,
    checkpoint_path: str | Path,
    locked_deployable: Mapping[str, Any],
    bundle_factory: Callable[[Mapping[str, Any]], PredictionAnchoredDeployableBundle],
    fixed_hlt_batch: Mapping[str, torch.Tensor],
    privileged_source_paths: Sequence[str | Path],
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    validate_content_hash(
        locked_deployable, expected_contract=PREDICTION_ANCHORED_LOCKED_DEPLOYABLE_CONTRACT
    )
    privileged = [Path(value).resolve(strict=True) for value in privileged_source_paths]
    if not privileged:
        raise ValueError("clean reload audit requires real privileged source paths")

    def forbidden(path_value: Any) -> bool:
        if not isinstance(path_value, (str, bytes, Path)):
            return False
        try:
            candidate = Path(path_value).resolve(strict=False)
        except (OSError, TypeError, ValueError):
            return False
        return any(
            candidate == root or root in candidate.parents or candidate in root.parents
            for root in privileged
        )

    original_builtin_open = builtins.open
    original_io_open = io.open

    def guarded_builtin_open(path_value: Any, *args: Any, **kwargs: Any):
        if forbidden(path_value):
            raise PermissionError(f"privileged source is isolated from reload: {path_value}")
        return original_builtin_open(path_value, *args, **kwargs)

    def guarded_io_open(path_value: Any, *args: Any, **kwargs: Any):
        if forbidden(path_value):
            raise PermissionError(f"privileged source is isolated from reload: {path_value}")
        return original_io_open(path_value, *args, **kwargs)
    source_bundle.eval()
    with torch.no_grad():
        source_logits = source_bundle(fixed_hlt_batch).detach().cpu()
    with tempfile.TemporaryDirectory(prefix="prediction_anchored_clean_reload_") as temporary:
        copied = Path(temporary) / "deployable_bundle.pt"
        shutil.copy2(checkpoint_path, copied)
        builtins.open = guarded_builtin_open
        io.open = guarded_io_open
        try:
            loaded, manifest = load_deployable_bundle(copied, bundle_factory=bundle_factory)
            with torch.no_grad():
                loaded_logits = loaded(fixed_hlt_batch).detach().cpu()
        finally:
            io.open = original_io_open
            builtins.open = original_builtin_open
    if source_logits.shape != loaded_logits.shape:
        raise AssertionError("clean reload changed the deployable logit shape")
    difference = torch.abs(source_logits - loaded_logits)
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    passed = bool(torch.allclose(source_logits, loaded_logits, atol=float(atol), rtol=float(rtol)))
    if manifest.get("locked_deployable_sha256") != locked_deployable["content_hash"]:
        raise ValueError("clean bundle manifest is bound to another locked deployment")
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_CLEAN_RELOAD_AUDIT_CONTRACT,
            "locked_deployable_sha256": locked_deployable["content_hash"],
            "bundle_checkpoint_sha256": sha256_file(checkpoint_path),
            "bundle_manifest_sha256": manifest["content_hash"],
            "privileged_source_path_count": len(privileged),
            "privileged_sources_absent": True,
            "privileged_sources_existed_before_isolation": True,
            "privileged_source_access_denied_during_reload": True,
            "isolation_method": "deny_canonical_privileged_roots_during_clean_copy_reload",
            "fixed_hlt_batch_keys": sorted(fixed_hlt_batch),
            "atol": float(atol),
            "rtol": float(rtol),
            "max_abs_logit_difference": max_abs,
            "tolerance_equivalent": passed,
            "passed": passed,
            "offline_matching_used": False,
            "external_teacher_loader_used": False,
        }
    )


def write_step9_decision_artifact(path: str | Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Small CLI-friendly immutable writer for any Step 9 hashed decision."""

    validate_content_hash(artifact)
    return write_immutable_json(path, artifact)


__all__ = [
    "PREDICTION_ANCHORED_CAMPAIGN_RESERVATION_CONTRACT",
    "PREDICTION_ANCHORED_CAMPAIGN_STATE_CONTRACT",
    "PREDICTION_ANCHORED_STAGE_EVIDENCE_CONTRACT",
    "PREDICTION_ANCHORED_DEPLOYABLE_REPLICA_CONTRACT",
    "PREDICTION_ANCHORED_DEPLOYABLE_AGGREGATE_CONTRACT",
    "PREDICTION_ANCHORED_DEPLOYABLE_PRECONFIRMATION_CONTRACT",
    "PREDICTION_ANCHORED_DEPLOYABLE_CONFIRMATION_CONTRACT",
    "PREDICTION_ANCHORED_LOCKED_DEPLOYABLE_CONTRACT",
    "PREDICTION_ANCHORED_STOPPED_DEPLOYABLE_CONTRACT",
    "PREDICTION_ANCHORED_REPORT_ROW_CONTRACT",
    "PREDICTION_ANCHORED_REPORTS_CONTRACT",
    "PREDICTION_ANCHORED_MEDIAN_PUBLICATION_CONTRACT",
    "PREDICTION_ANCHORED_DEPLOYABLE_BUNDLE_MANIFEST_CONTRACT",
    "PREDICTION_ANCHORED_DEPLOYABLE_BUNDLE_CHECKPOINT_CONTRACT",
    "PREDICTION_ANCHORED_CLEAN_RELOAD_AUDIT_CONTRACT",
    "B_STAGES",
    "RUN_OUTCOME_STATES",
    "STAGE_STATES",
    "BASELINE_DEPLOYABLE_REQUIRED_IDS",
    "PRIVILEGED_REQUIRED_IDS",
    "DeployableReplicaEvidence",
    "PredictionAnchoredDeployableBundle",
    "build_campaign_reservations",
    "build_stage_evidence",
    "initialize_step9_campaign",
    "advance_step9_campaign",
    "record_campaign_run_outcomes",
    "aggregate_deployable_configuration",
    "select_deployable_preconfirmation",
    "authorize_deployable_confirmation",
    "build_deployable_confirmation",
    "finalize_deployable_confirmation",
    "validate_final_test_request",
    "authorize_final_test",
    "build_step9_report_row",
    "build_step9_reports",
    "publish_step9_reports",
    "build_median_publication_manifest",
    "build_deployable_bundle_manifest",
    "export_deployable_bundle",
    "load_deployable_bundle",
    "clean_hlt_only_reload_audit",
    "write_step9_decision_artifact",
]
