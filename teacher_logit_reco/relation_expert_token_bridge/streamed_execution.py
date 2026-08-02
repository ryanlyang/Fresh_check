"""Authenticated task-local execution policy for RETB Stages D--N.

The policy deliberately separates scientific artifacts from recomputable
execution material.  It is also used by the compact streamed smoke, whose
artifacts are explicitly ineligible as production evidence.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator, Mapping, Sequence

from .contracts import require_sha256, validate_content_hash, with_content_hash


FULL_STREAMED_PROFILE = "full_streamed"
STREAMED_SMOKE_PROFILE = "streamed_smoke"
STREAMED_PROFILES = (
    "offline_abc_streamed",
    FULL_STREAMED_PROFILE,
    STREAMED_SMOKE_PROFILE,
)
STREAMED_EXECUTION_CONTRACT = "retb_streamed_execution_profile_v1"
STREAMED_TASK_RECEIPT_CONTRACT = "retb_streamed_task_lifecycle_receipt_v1"
STREAMED_STORAGE_PROJECTION_CONTRACT = "retb_streamed_storage_projection_v1"
STREAMED_SMOKE_PLAN_CONTRACT = "retb_compact_streamed_smoke_plan_v1"
STREAMED_SMOKE_PHASE_CONTRACT = "retb_compact_streamed_smoke_phase_v1"
STREAMED_SMOKE_LEDGER_CONTRACT = "retb_compact_streamed_smoke_ledger_v1"


# These classes are semantic policy, not a glob-based deletion list.  A worker
# may only delete a path that it created below its authenticated task root.
TRANSIENT_CLASSES = (
    "single_command_preparation_arrays",
    "internal_phase_raw_outputs_consumed_before_phase_completion",
    "final_test_raw_logits_before_seal",
    "successful_resume_state",
)
ROLLING_CLASSES = (
    "native_fusion_token_banks_until_native_fusion_completion",
    "target_caches_until_joint_and_consumer_completion",
    "predictor_prepared_arrays_until_joint_completion",
    "joint_dataset_arrays_until_deployable_export_completion",
    "scale_refit_caches_until_scale_graph_completion",
)
DURABLE_CLASSES = (
    "selected_checkpoints",
    "checkpoint_registrations",
    "compact_deployable_logits",
    "metrics_and_uncertainty",
    "selection_locks",
    "lineage_and_identity_manifests",
    "semantic_and_robustness_evidence",
    "final_test_input_and_execution_seals",
    "final_reports",
    "task_and_phase_lifecycle_receipts",
)


SMOKE_PHASES: tuple[dict[str, Any], ...] = (
    {"phase_id": "a_inputs", "stage": "A", "resource": "cpu", "kind": "input_cache"},
    {"phase_id": "a_relations", "stage": "A", "resource": "cpu", "kind": "relation_cache"},
    {"phase_id": "b_expert", "stage": "B", "resource": "gpu", "kind": "expert_training"},
    {"phase_id": "c_fusion", "stage": "C", "resource": "gpu", "kind": "streamed_fusion"},
    {"phase_id": "d_native", "stage": "D", "resource": "gpu", "kind": "native_hlt_expert"},
    {"phase_id": "e_bridge", "stage": "E", "resource": "gpu", "kind": "bridge_training"},
    {"phase_id": "f_targets", "stage": "F", "resource": "cpu", "kind": "target_cache"},
    {"phase_id": "g_predictor", "stage": "G", "resource": "gpu", "kind": "predictor_training"},
    {"phase_id": "h_bundle", "stage": "H", "resource": "cpu", "kind": "bundle_selection"},
    {"phase_id": "i_joint", "stage": "I", "resource": "gpu", "kind": "joint_training"},
    {"phase_id": "j_consumer", "stage": "J", "resource": "gpu", "kind": "final_consumer"},
    {"phase_id": "k_semantics", "stage": "K", "resource": "gpu", "kind": "semantic_evidence"},
    {"phase_id": "l_confirmation", "stage": "L", "resource": "gpu", "kind": "confirmation"},
    {"phase_id": "m_scale", "stage": "M", "resource": "gpu", "kind": "scale_refit"},
    {"phase_id": "n_sealed_final", "stage": "N", "resource": "gpu", "kind": "sealed_final_test"},
    {"phase_id": "n_report", "stage": "N", "resource": "cpu", "kind": "final_reporting"},
)


def is_streamed_profile(value: object) -> bool:
    return str(value) in STREAMED_PROFILES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_task_local_parent(environment: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environment is None else environment
    candidates = (
        values.get("RETB_STREAM_ROOT"),
        values.get("SLURM_TMPDIR"),
        "/dev/shm",
    )
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_dir() and os.access(path, os.W_OK):
            return path.resolve()
    return Path(tempfile.gettempdir()).resolve()


@contextmanager
def task_local_workspace(
    *, campaign_id: str, node_id: str, task_index: int,
    environment: Mapping[str, str] | None = None,
) -> Iterator[Path]:
    """Yield a unique workspace and remove it on success and failure."""

    parent = select_task_local_parent(environment)
    prefix = f"retb_{campaign_id[-18:]}_{node_id}_{int(task_index)}_"
    workspace = Path(tempfile.mkdtemp(prefix=prefix, dir=parent)).resolve()
    try:
        yield workspace
    finally:
        # The resolved child check makes the cleanup target fail closed.
        if workspace.parent != parent or not workspace.name.startswith(prefix):
            raise RuntimeError("refusing unsafe RETB task-local cleanup")
        shutil.rmtree(workspace, ignore_errors=False)


def build_streamed_execution_profile(
    *, campaign_id: str, campaign_root: str | Path,
    source: Mapping[str, Any], profile: str,
) -> dict[str, Any]:
    if profile not in {FULL_STREAMED_PROFILE, STREAMED_SMOKE_PROFILE}:
        raise ValueError("streamed D-N execution profile differs")
    smoke = profile == STREAMED_SMOKE_PROFILE
    return with_content_hash({
        "contract": STREAMED_EXECUTION_CONTRACT,
        "schema_version": 1,
        "campaign_id": str(campaign_id),
        "campaign_root": str(Path(campaign_root)),
        "source": dict(source),
        "execution_profile": profile,
        "scientific_evidence_eligible": not smoke,
        "smoke_artifacts_forbidden_as_production_evidence": smoke,
        "stages": list("ABCDEFGHIJKLMN"),
        "task_local_root_precedence": ["RETB_STREAM_ROOT", "SLURM_TMPDIR", "/dev/shm"],
        "cleanup_on_success": True,
        "cleanup_on_failure": True,
        "transient_artifact_classes": list(TRANSIENT_CLASSES),
        "rolling_authenticated_artifact_classes": list(ROLLING_CLASSES),
        "durable_artifact_classes": list(DURABLE_CLASSES),
        "successful_resume_state_removed_after_attestation": True,
        "failed_resume_state_outside_task_root_may_be_retained": True,
        "scientific_underperformance_blocks_continuation": False,
        "runtime_or_lineage_failure_blocks_dependents": True,
        "final_test_requires_input_and_execution_locks": True,
        "production_array_concurrency_policy": "unthrottled_up_to_global_high_limits",
    })


def validate_streamed_execution_profile(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(payload, expected_contract=STREAMED_EXECUTION_CONTRACT)
    expected = build_streamed_execution_profile(
        campaign_id=str(payload["campaign_id"]),
        campaign_root=str(payload["campaign_root"]),
        source=payload["source"], profile=str(payload["execution_profile"]),
    )
    if dict(payload) != expected:
        raise ValueError("streamed execution profile semantics differ")
    return digest


def build_task_lifecycle_receipt(
    *, campaign_spec_sha256: str, task_manifest_sha256: str,
    node_id: str, task_index: int, status: str,
    workspace_parent: str | Path, workspace_removed: bool,
    output_paths: Sequence[str | Path], source: Mapping[str, Any],
) -> dict[str, Any]:
    if status not in {"completed", "failed"}:
        raise ValueError("streamed task status differs")
    outputs = []
    for raw in output_paths:
        path = Path(raw)
        if status == "completed" and not path.is_file():
            raise FileNotFoundError(path)
        if path.is_file():
            outputs.append({"path": str(path), "sha256": _sha256(path)})
    return with_content_hash({
        "contract": STREAMED_TASK_RECEIPT_CONTRACT,
        "schema_version": 1,
        "campaign_spec_sha256": require_sha256(campaign_spec_sha256, name="campaign_spec_sha256"),
        "task_manifest_sha256": require_sha256(task_manifest_sha256, name="task_manifest_sha256"),
        "node_id": str(node_id), "task_index": int(task_index), "status": status,
        "workspace_parent": str(Path(workspace_parent)),
        "workspace_removed_before_publication": bool(workspace_removed),
        "persistent_outputs": outputs,
        "persistent_output_count": len(outputs),
        "performance_sign_used_as_completion_gate": False,
        "source": dict(source),
    })


def validate_task_lifecycle_receipt(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STREAMED_TASK_RECEIPT_CONTRACT
    )
    for field in ("campaign_spec_sha256", "task_manifest_sha256"):
        require_sha256(payload.get(field), name=field)
    if payload.get("status") not in {"completed", "failed"}:
        raise ValueError("streamed task receipt status differs")
    if payload.get("workspace_removed_before_publication") is not True:
        raise ValueError("streamed task workspace was not removed")
    outputs = payload.get("persistent_outputs")
    if not isinstance(outputs, list) or int(payload.get("persistent_output_count", -1)) != len(outputs):
        raise ValueError("streamed task output coverage differs")
    for row in outputs:
        require_sha256(row.get("sha256"), name="persistent_output.sha256")
        if not str(row.get("path", "")):
            raise ValueError("streamed task output path is absent")
    if payload.get("performance_sign_used_as_completion_gate") is not False:
        raise ValueError("streamed task receipt gained a performance gate")
    return digest


def build_streamed_smoke_plan(
    *, campaign_spec_sha256: str, production_graph_sha256: str,
    campaign_id: str, source: Mapping[str, Any],
) -> dict[str, Any]:
    phases = []
    for index, raw in enumerate(SMOKE_PHASES):
        phases.append({**raw, "sequence_index": index,
                       "dependency_phase_id": None if index == 0 else SMOKE_PHASES[index - 1]["phase_id"]})
    physical_allocations = 2 + len(phases)  # real split/bootstrap + phases
    if physical_allocations > 30:
        raise RuntimeError("compact streamed smoke exceeds allocation bound")
    return with_content_hash({
        "contract": STREAMED_SMOKE_PLAN_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": require_sha256(campaign_spec_sha256, name="campaign_spec_sha256"),
        "production_graph_sha256": require_sha256(production_graph_sha256, name="production_graph_sha256"),
        "campaign_id": str(campaign_id), "execution_profile": STREAMED_SMOKE_PROFILE,
        "phases": phases, "phase_count": len(phases),
        "bootstrap_allocation_count": 2,
        "physical_allocation_count": physical_allocations,
        "physical_allocation_limit": 30,
        "scientific_grid_execution": "one_representative_coordinate_per_material_execution_family",
        "complete_scientific_grid_queued": False,
        "underperformance_blocks_continuation": False,
        "runtime_and_lineage_fail_closed": True,
        "final_test_is_miniature_and_doubly_sealed": True,
        "production_evidence_eligible": False,
        "source": dict(source),
    })


def validate_streamed_smoke_plan(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(payload, expected_contract=STREAMED_SMOKE_PLAN_CONTRACT)
    expected = build_streamed_smoke_plan(
        campaign_spec_sha256=str(payload["campaign_spec_sha256"]),
        production_graph_sha256=str(payload["production_graph_sha256"]),
        campaign_id=str(payload["campaign_id"]), source=payload["source"],
    )
    if dict(payload) != expected:
        raise ValueError("compact streamed smoke plan differs")
    return digest


def build_streamed_storage_projection(
    *, storage_measurements_sha256: str,
    persistent_classes: Mapping[str, int],
    rolling_classes: Mapping[str, int],
    transient_classes: Mapping[str, int],
    maximum_concurrent_allocations: int,
    serialized_reserve_bytes: int,
    available_storage_bytes: int,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    persistent = {str(k): int(v) for k, v in sorted(persistent_classes.items())}
    rolling = {str(k): int(v) for k, v in sorted(rolling_classes.items())}
    transient = {str(k): int(v) for k, v in sorted(transient_classes.items())}
    if not persistent or not rolling or not transient or any(
        v < 0 for v in (*persistent.values(), *rolling.values(), *transient.values())
    ):
        raise ValueError("streamed storage classes differ")
    concurrency = int(maximum_concurrent_allocations)
    reserve = int(serialized_reserve_bytes)
    available = int(available_storage_bytes)
    if concurrency <= 0 or reserve < 0 or available < 0:
        raise ValueError("streamed storage bounds differ")
    persistent_peak = sum(persistent.values()) + sum(rolling.values()) + reserve
    per_allocation_transient_peak = max(transient.values())
    cluster_transient_peak = per_allocation_transient_peak * concurrency
    return with_content_hash({
        "contract": STREAMED_STORAGE_PROJECTION_CONTRACT,
        "schema_version": 1,
        "storage_measurements_sha256": require_sha256(
            storage_measurements_sha256, name="storage_measurements_sha256"
        ),
        "persistent_classes": persistent,
        "rolling_authenticated_classes": rolling,
        "transient_classes": transient,
        "persistent_peak_bytes": persistent_peak,
        "per_allocation_transient_peak_bytes": per_allocation_transient_peak,
        "maximum_concurrent_allocations": concurrency,
        "cluster_transient_peak_bytes": cluster_transient_peak,
        "serialized_reserve_bytes": reserve,
        "available_persistent_storage_bytes": available,
        "persistent_storage_admitted": persistent_peak <= available,
        "transient_storage_location": "allocation_local_not_shared_persistent_storage",
        "transient_cluster_sum_is_not_charged_to_shared_persistent_storage": True,
        "performance_measurements_select_scientific_models": False,
        "source": dict(source),
    })


def validate_streamed_storage_projection(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STREAMED_STORAGE_PROJECTION_CONTRACT
    )
    expected = build_streamed_storage_projection(
        storage_measurements_sha256=str(payload["storage_measurements_sha256"]),
        persistent_classes=payload["persistent_classes"],
        rolling_classes=payload["rolling_authenticated_classes"],
        transient_classes=payload["transient_classes"],
        maximum_concurrent_allocations=int(payload["maximum_concurrent_allocations"]),
        serialized_reserve_bytes=int(payload["serialized_reserve_bytes"]),
        available_storage_bytes=int(payload["available_persistent_storage_bytes"]),
        source=payload["source"],
    )
    if dict(payload) != expected:
        raise ValueError("streamed storage projection differs")
    return digest


__all__ = [name for name in globals() if name.startswith("STREAMED_") or name in {
    "FULL_STREAMED_PROFILE", "DURABLE_CLASSES", "TRANSIENT_CLASSES",
    "ROLLING_CLASSES",
    "build_streamed_execution_profile", "build_streamed_smoke_plan",
    "build_streamed_storage_projection",
    "build_task_lifecycle_receipt", "is_streamed_profile",
    "select_task_local_parent", "task_local_workspace",
    "validate_streamed_execution_profile", "validate_streamed_smoke_plan",
    "validate_streamed_storage_projection",
    "validate_task_lifecycle_receipt",
}]
