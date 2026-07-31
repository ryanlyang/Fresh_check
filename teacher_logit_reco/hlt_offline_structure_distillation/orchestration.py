"""Production DAG, restart monitor, miniature evidence, and authorization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    CAMPAIGN_MONITOR_CONTRACT,
    FULL_AUTHORIZATION_CONTRACT,
    MINIATURE_ACCEPTANCE_CONTRACT,
    MINIATURE_CHECK_RECEIPT_CONTRACT,
    PRODUCTION_EXECUTION_PLAN_CONTRACT,
    RESOURCE_MEASUREMENTS_CONTRACT,
    RESOURCE_PREFLIGHT_CONTRACT,
    NODE_FACTORY_REGISTRY_CONTRACT,
    RUNTIME_MANIFEST_CONTRACT,
    SLURM_SUBMISSION_LEDGER_CONTRACT,
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .registry import validate_stage_job_registry


RESOURCE_CLASSES = {
    "cpu": {
        "partition": "tigris",
        "cpus": 16,
        "memory": "192G",
        "time": "24:00:00",
        "gres": None,
    },
    "gpu": {
        "partition": "tigris",
        "cpus": 16,
        "memory": "220G",
        "time": "2-00:00:00",
        "gres": "gpu:gh200:1",
    },
}

REQUIRED_SCALE_MEMORY_PROJECTION_NODES = frozenset(
    {
        "scale_input_prepare",
        "scale_tree_build",
        "scale_target_build",
        "scale_teacher_target_inference",
        "scale_graph_train",
    }
)

MINIATURE_RESOURCE_CLASSES = {
    "cpu": {
        "partition": "tigris",
        "cpus": 4,
        "memory": "32G",
        "time": "04:00:00",
        "gres": None,
    },
    "gpu": {
        "partition": "tigris",
        "cpus": 4,
        "memory": "64G",
        "time": "08:00:00",
        "gres": "gpu:gh200:1",
    },
}


def build_resource_measurements(
    *,
    miniature_execution_plan_sha256: str,
    scheduler_evidence_sha256: str,
    requests_by_class: Mapping[str, Mapping[str, Any]],
    projected_target_extraction_seconds: int,
    projected_gpu_hours_by_stage: Mapping[str, float],
    maximum_concurrent_jobs: int,
    checkpoint_bytes: int,
    export_bytes: int,
    scale_resident_layout_ledger: Mapping[str, Any],
    scale_resident_memory_projections: Mapping[str, Mapping[str, Any]],
    measurement_evidence_sha256: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate production requests derived from the real miniature."""

    if set(requests_by_class) != {"cpu", "gpu"}:
        raise ValueError("resource request class coverage differs")
    checked = {}
    for kind, raw in requests_by_class.items():
        request = dict(raw)
        if (
            set(request)
            != {"partition", "cpus", "memory", "time", "gres"}
            or int(request["cpus"]) <= 0
            or not str(request["memory"]).endswith("G")
            or not str(request["time"])
        ):
            raise ValueError("measured Slurm resource request differs")
        if kind == "gpu" and request["gres"] != "gpu:gh200:1":
            raise ValueError("measured production GPU resource is not GH200")
        if kind == "cpu" and request["gres"] is not None:
            raise ValueError("measured CPU resource unexpectedly requests a GPU")
        checked[kind] = request
    if (
        int(projected_target_extraction_seconds) <= 0
        or int(maximum_concurrent_jobs) <= 0
        or int(checkpoint_bytes) <= 0
        or int(export_bytes) <= 0
        or not projected_gpu_hours_by_stage
    ):
        raise ValueError("resource projections must be positive and complete")
    layout_ledger = dict(scale_resident_layout_ledger)
    validate_content_hash(
        layout_ledger, expected_contract="hosd_scale_resident_layout_ledger_v5"
    )
    source_sha256 = canonical_sha256(source)
    if (
        layout_ledger.get("source_sha256") != source_sha256
        or not isinstance(layout_ledger.get("scale_execution_plan_sha256"), str)
        or len(layout_ledger["scale_execution_plan_sha256"]) != 64
        or set(layout_ledger.get("active_completion_hashes", {}))
        != {"scale_inputs", "scale_trees", "scale_targets", "scale_teacher_outputs"}
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in layout_ledger.get("active_completion_hashes", {}).values()
        )
        or int(layout_ledger.get("production_tree_shard_events", 0))
        != 10_000
        or int(layout_ledger.get("production_target_shard_events", 0))
        != 2_048
        or int(
            layout_ledger.get(
                "production_tree_shard_decoded_bytes_upper_bound", 0
            )
        )
        <= 0
        or int(
            layout_ledger.get(
                "production_target_shard_decoded_bytes_upper_bound", 0
            )
        )
        <= 0
    ):
        raise ValueError("scale resident layout source differs")
    projections = {
        str(node_id): dict(row)
        for node_id, row in scale_resident_memory_projections.items()
    }
    if set(projections) != set(REQUIRED_SCALE_MEMORY_PROJECTION_NODES):
        raise ValueError("scale resident-memory projection coverage differs")
    for node_id, projection in projections.items():
        validate_content_hash(
            projection,
            expected_contract="hosd_scale_resident_memory_projection_v5",
        )
        model = projection.get("projection_model")
        fixed = int(projection.get("fixed_resident_bytes", -1))
        unit_bytes = int(
            projection.get("production_resident_unit_bytes_upper_bound", 0)
        )
        required_unit_events = {
            "scale_input_prepare": 1,
            "scale_tree_build": 10_000,
            "scale_target_build": 2_048,
            "scale_teacher_target_inference": 2_048,
            "scale_graph_train": 10_000,
        }[node_id]
        population = int(projection.get("production_population", 0))
        expected_projected = (
            fixed + unit_bytes * population
            if model == "fixed_plus_authenticated_per_event_population_v2"
            else fixed + unit_bytes
            if model == "fixed_plus_authenticated_single_shard_v2"
            else -1
        )
        if (
            projection.get("pilot_node_id") != node_id
            or projection.get("source_sha256") != source_sha256
            or projection.get("layout_evidence_sha256")
            != layout_ledger["content_hash"]
            or not projection.get("coordinate_type")
            or projection.get("resource_class") not in RESOURCE_CLASSES
            or not projection.get("loader_storage_contracts")
            or not projection.get("representative_real_task_completed")
            or fixed < 0
            or unit_bytes <= 0
            or int(projection.get("miniature_resident_unit_bytes_upper_bound", 0))
            <= 0
            or int(projection.get("production_resident_unit_events", 0))
            != required_unit_events
            or int(projection.get("miniature_resident_unit_events", 0)) <= 0
            or population != 3_000_000
            or int(projection.get("projected_resident_bytes", 0))
            != expected_projected
            or int(projection.get("registered_tigris_limit_bytes", 0))
            != int(
                str(
                    RESOURCE_CLASSES[projection["resource_class"]]["memory"]
                ).removesuffix("G")
            )
            * 1024**3
            or projection.get("within_registered_tigris_limit") is not True
            or expected_projected
            > int(projection["registered_tigris_limit_bytes"])
            or expected_projected
            > int(str(checked[projection["resource_class"]]["memory"]).removesuffix("G"))
            * 1024**3
        ):
            raise ValueError(
                f"{node_id} resident-memory projection differs"
            )
    return with_content_hash(
        {
            "contract": RESOURCE_MEASUREMENTS_CONTRACT,
            "schema_version": 8,
            "source": dict(source),
            "miniature_execution_plan_sha256": require_sha256(
                miniature_execution_plan_sha256,
                name="miniature_execution_plan_sha256",
            ),
            "scheduler_evidence_sha256": require_sha256(
                scheduler_evidence_sha256,
                name="scheduler_evidence_sha256",
            ),
            "measurement_evidence_sha256": require_sha256(
                measurement_evidence_sha256,
                name="measurement_evidence_sha256",
            ),
            "requests_by_class": checked,
            "projected_target_extraction_seconds": int(
                projected_target_extraction_seconds
            ),
            "projected_gpu_hours_by_stage": {
                str(stage): float(hours)
                for stage, hours in sorted(projected_gpu_hours_by_stage.items())
            },
            "maximum_concurrent_jobs": int(maximum_concurrent_jobs),
            "checkpoint_bytes": int(checkpoint_bytes),
            "export_bytes": int(export_bytes),
            "scale_resident_layout_ledger": layout_ledger,
            "scale_resident_memory_projections": projections,
            "derived_from_real_miniature": True,
            "hand_authored_measurements_allowed": False,
            "performance_results_not_read": True,
        }
    )


def build_production_execution_plan(
    *,
    stage_job_registry: Mapping[str, Any],
    commands_by_node: Mapping[str, Sequence[Sequence[str]]],
    campaign_spec_sha256: str,
    source: Mapping[str, Any],
    profile: str,
    resource_measurements: Mapping[str, Any] | None = None,
    node_factory_registry: Mapping[str, Any] | None = None,
    runtime_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_stage_job_registry(stage_job_registry)
    if profile not in {"miniature_test", "production_500k_scale3m"}:
        raise ValueError("unknown HOSD execution profile")
    nodes = list(stage_job_registry["nodes"])
    if node_factory_registry is None or runtime_manifest is None:
        raise ValueError("execution plan requires registered node factories")
    validate_content_hash(
        node_factory_registry,
        expected_contract=NODE_FACTORY_REGISTRY_CONTRACT,
    )
    validate_content_hash(
        runtime_manifest, expected_contract=RUNTIME_MANIFEST_CONTRACT
    )
    if (
        runtime_manifest.get("execution_ready") is not True
        or runtime_manifest.get("missing_required_options_by_node")
    ):
        raise ValueError(
            "execution runtime manifest is incomplete: "
            f"{runtime_manifest.get('missing_required_options_by_node')}"
        )
    if (
        node_factory_registry.get("stage_job_registry_sha256")
        != stage_job_registry["content_hash"]
        or node_factory_registry.get("source") != dict(source)
        or runtime_manifest.get("source") != dict(source)
        or runtime_manifest.get("campaign_spec_sha256")
        != campaign_spec_sha256
    ):
        raise ValueError("execution node-factory lineage differs")
    expected = {node["node_id"] for node in nodes}
    if set(commands_by_node) != expected:
        raise ValueError("production command coverage differs from stage registry")
    if profile == "production_500k_scale3m":
        if resource_measurements is None:
            raise ValueError(
                "production execution requires authenticated miniature resources"
            )
        validate_content_hash(
            resource_measurements,
            expected_contract=RESOURCE_MEASUREMENTS_CONTRACT,
        )
        if (
            resource_measurements.get("source") != dict(source)
            or not resource_measurements.get("derived_from_real_miniature")
        ):
            raise ValueError("production resource measurement lineage differs")
        resource_classes = resource_measurements["requests_by_class"]
        resource_measurements_sha256 = resource_measurements["content_hash"]
    else:
        if resource_measurements is not None:
            raise ValueError("miniature execution cannot consume production resources")
        resource_classes = MINIATURE_RESOURCE_CLASSES
        resource_measurements_sha256 = None
    rows = []
    for node in nodes:
        commands = [
            [str(value) for value in command]
            for command in commands_by_node[node["node_id"]]
        ]
        if not commands or any(
            not command
            or command[0] != "python"
            or any(not value for value in command)
            for command in commands
        ):
            raise ValueError(f"node command list differs: {node['node_id']}")
        expected_entrypoint = str(
            node_factory_registry["factory_entrypoint"]
        ).replace("\\", "/")
        if any(
            len(command) < 2
            or command[1].replace("\\", "/") != expected_entrypoint
            for command in commands
        ):
            raise ValueError(
                f"node commands bypass registered entrypoint: {node['node_id']}"
            )
        resource = resource_classes[node["resource"]]
        rows.append(
            {
                "node_id": node["node_id"],
                "stage": node["stage"],
                "dependencies": list(node["dependencies"]),
                "outputs": list(node["outputs"]),
                "commands": commands,
                "resource": dict(resource),
                "scientific_underperformance_can_fail_node": False,
                "integrity_or_runtime_failure_blocks_dependents": True,
            }
        )
    return with_content_hash(
        {
            "contract": PRODUCTION_EXECUTION_PLAN_CONTRACT,
            "schema_version": 2,
            "source": dict(source),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "profile": profile,
            "stage_job_registry_sha256": stage_job_registry["content_hash"],
            "node_factory_registry_sha256": node_factory_registry[
                "content_hash"
            ],
            "runtime_manifest_sha256": runtime_manifest["content_hash"],
            "manual_command_bundle_consumed": False,
            "resource_measurements_sha256": resource_measurements_sha256,
            "resources_are_authenticated_miniature_measurements": (
                profile == "production_500k_scale3m"
            ),
            "nodes": rows,
            "node_count": len(rows),
            "all_stage_nodes_have_commands": True,
            "dependency_policy": "afterok_integrity_only",
            "performance_based_termination": False,
            "negative_result_continuation": True,
            "restart_policy": (
                "reuse_only_complete_source_and_lineage_validated_artifacts"
            ),
        }
    )


def node_execution(
    plan: Mapping[str, Any], *, node_id: str
) -> Mapping[str, Any]:
    validate_content_hash(
        plan, expected_contract=PRODUCTION_EXECUTION_PLAN_CONTRACT
    )
    rows = [row for row in plan["nodes"] if row["node_id"] == node_id]
    if len(rows) != 1:
        raise ValueError("production node is absent or duplicated")
    return rows[0]


def build_campaign_monitor(
    *,
    execution_plan: Mapping[str, Any],
    node_states: Mapping[str, Mapping[str, Any]],
    artifact_validity: Mapping[str, bool],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        execution_plan, expected_contract=PRODUCTION_EXECUTION_PLAN_CONTRACT
    )
    expected = {row["node_id"] for row in execution_plan["nodes"]}
    if set(node_states) != expected or set(artifact_validity) != expected:
        raise ValueError("campaign monitor node coverage differs")
    allowed = {
        "pending",
        "running",
        "completed",
        "failed",
        "cancelled",
        "timeout",
    }
    rows = []
    for node in execution_plan["nodes"]:
        state = str(node_states[node["node_id"]]["state"]).lower()
        if state not in allowed:
            raise ValueError("campaign monitor state differs")
        valid = bool(artifact_validity[node["node_id"]])
        reusable = state == "completed" and valid
        rows.append(
            {
                "node_id": node["node_id"],
                "state": state,
                "artifact_valid": valid,
                "reusable": reusable,
                "needs_repair": state
                in {"failed", "cancelled", "timeout"}
                or (state == "completed" and not valid),
                "scientific_result_sign_used": False,
            }
        )
    repair = {row["node_id"] for row in rows if row["needs_repair"]}
    changed = True
    while changed:
        changed = False
        for node in execution_plan["nodes"]:
            if (
                node["node_id"] not in repair
                and not next(
                    row["reusable"]
                    for row in rows
                    if row["node_id"] == node["node_id"]
                )
                and any(dependency in repair for dependency in node["dependencies"])
            ):
                repair.add(node["node_id"])
                changed = True
    return with_content_hash(
        {
            "contract": CAMPAIGN_MONITOR_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "execution_plan_sha256": execution_plan["content_hash"],
            "nodes": rows,
            "complete": all(row["reusable"] for row in rows),
            "restart_nodes": [
                row["node_id"] for row in rows if row["needs_repair"]
            ],
            "recovery_submission_nodes": [
                node["node_id"]
                for node in execution_plan["nodes"]
                if node["node_id"] in repair
            ],
            "performance_based_cancellation_detected": False,
        }
    )


def build_slurm_submission_ledger(
    *,
    execution_plan: Mapping[str, Any],
    jobs: Mapping[str, str | None],
    submission_mode: str,
    attempt: int,
    selected_node_ids: Sequence[str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        execution_plan, expected_contract=PRODUCTION_EXECUTION_PLAN_CONTRACT
    )
    selected = tuple(str(value) for value in selected_node_ids)
    if (
        submission_mode not in {"dry_run", "smoke_submit", "full_submit", "resume_submit"}
        or int(attempt) <= 0
        or len(selected) != len(set(selected))
        or set(jobs) != set(selected)
    ):
        raise ValueError("Slurm submission ledger declaration differs")
    known = {node["node_id"] for node in execution_plan["nodes"]}
    if not set(selected).issubset(known):
        raise ValueError("Slurm submission contains an unknown node")
    if submission_mode == "dry_run":
        if any(value is not None for value in jobs.values()):
            raise ValueError("dry-run ledger contains submitted jobs")
    elif any(value is None or not str(value).isdigit() for value in jobs.values()):
        raise ValueError("submitted ledger contains a nonnumeric job ID")
    return with_content_hash(
        {
            "contract": SLURM_SUBMISSION_LEDGER_CONTRACT,
            "schema_version": 2,
            "source": dict(source),
            "campaign_spec_sha256": execution_plan["campaign_spec_sha256"],
            "execution_plan_sha256": execution_plan["content_hash"],
            "profile": execution_plan["profile"],
            "submission_mode": submission_mode,
            "attempt": int(attempt),
            "selected_node_ids": list(selected),
            "jobs": {key: jobs[key] for key in selected},
            "dependency_policy": "afterok_integrity_only",
            "performance_based_cancellation": False,
            "negative_scientific_result_can_cancel": False,
        }
    )


MINIATURE_CHECK_IDS = frozenset(
    {
        "real_jetclass_hlt_v3",
        "all_current_family_groups",
        "baseline_probe_aux_feedback_trained",
        "all_negative_selector_continued",
        "target_shard_interrupt_resume",
        "training_row_interrupt_resume",
        "hlt_only_export_validated",
        "stack_and_two_locks_traversed",
        "no_manual_artifact_injection",
    }
)


def build_miniature_check_receipt(
    *,
    execution_plan: Mapping[str, Any],
    check_id: str,
    evidence_hashes: Mapping[str, str],
    verifier: str,
    verifier_source_sha256: str | None = None,
    verified_predicates: Mapping[str, bool] | None = None,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        execution_plan, expected_contract=PRODUCTION_EXECUTION_PLAN_CONTRACT
    )
    if (
        execution_plan["profile"] != "miniature_test"
        or check_id not in MINIATURE_CHECK_IDS
        or not evidence_hashes
        or verifier != "hosd_miniature_semantic_verifier_v1"
        or verifier_source_sha256 is None
        or not verified_predicates
        or not all(value is True for value in verified_predicates.values())
    ):
        raise ValueError("miniature check receipt declaration differs")
    return with_content_hash(
        {
            "contract": MINIATURE_CHECK_RECEIPT_CONTRACT,
            "schema_version": 2,
            "source": dict(source),
            "execution_plan_sha256": execution_plan["content_hash"],
            "check_id": check_id,
            "evidence_hashes": {
                key: require_sha256(value, name=f"evidence.{key}")
                for key, value in sorted(evidence_hashes.items())
            },
            "verifier": verifier,
            "verifier_source_sha256": require_sha256(
                verifier_source_sha256, name="verifier_source_sha256"
            ),
            "verified_predicates": {
                key: bool(value)
                for key, value in sorted(verified_predicates.items())
            },
            "semantic_verification": True,
            "observed_on_research_compute": True,
            "synthetic_evidence": False,
            "passed": True,
        }
    )


def build_miniature_acceptance(
    *,
    execution_plan: Mapping[str, Any],
    check_receipts: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        execution_plan, expected_contract=PRODUCTION_EXECUTION_PLAN_CONTRACT
    )
    if execution_plan["profile"] != "miniature_test":
        raise ValueError("miniature acceptance requires miniature execution plan")
    receipts = {}
    for receipt in check_receipts:
        validate_content_hash(
            receipt, expected_contract=MINIATURE_CHECK_RECEIPT_CONTRACT
        )
        check_id = str(receipt["check_id"])
        if (
            check_id in receipts
            or receipt.get("execution_plan_sha256")
            != execution_plan["content_hash"]
            or receipt.get("source") != dict(source)
            or not receipt.get("passed")
            or receipt.get("synthetic_evidence")
            or receipt.get("semantic_verification") is not True
            or receipt.get("verifier")
            != "hosd_miniature_semantic_verifier_v1"
            or not receipt.get("verified_predicates")
            or not all(receipt["verified_predicates"].values())
        ):
            raise ValueError("miniature check receipt lineage differs")
        receipts[check_id] = receipt
    if set(receipts) != set(MINIATURE_CHECK_IDS):
        raise ValueError("real miniature acceptance checks are incomplete")
    return with_content_hash(
        {
            "contract": MINIATURE_ACCEPTANCE_CONTRACT,
            "schema_version": 2,
            "source": dict(source),
            "execution_plan_sha256": execution_plan["content_hash"],
            "checks": {key: True for key in sorted(MINIATURE_CHECK_IDS)},
            "check_receipt_hashes": {
                key: receipts[key]["content_hash"] for key in sorted(receipts)
            },
            "real_research_compute": True,
            "synthetic_dag_is_not_acceptance": True,
            "passed": True,
        }
    )


def build_full_authorization(
    *,
    production_plan: Mapping[str, Any],
    miniature_acceptance: Mapping[str, Any],
    resource_preflight: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        production_plan, expected_contract=PRODUCTION_EXECUTION_PLAN_CONTRACT
    )
    validate_content_hash(
        miniature_acceptance, expected_contract=MINIATURE_ACCEPTANCE_CONTRACT
    )
    validate_content_hash(
        resource_preflight, expected_contract=RESOURCE_PREFLIGHT_CONTRACT
    )
    if (
        production_plan["profile"] != "production_500k_scale3m"
        or not miniature_acceptance.get("passed")
        or production_plan.get("source") != dict(source)
        or miniature_acceptance.get("source") != dict(source)
        or resource_preflight.get("source") != dict(source)
        or resource_preflight.get("profile") != "production_500k_scale3m"
        or not resource_preflight.get("runtime_ready")
        or not isinstance(
            resource_preflight.get("storage_measurements_sha256"), str
        )
        or len(resource_preflight["storage_measurements_sha256"]) != 64
        or resource_preflight.get("resource_measurements_sha256")
        != production_plan.get("resource_measurements_sha256")
    ):
        raise ValueError("full campaign authorization prerequisites differ")
    storage_measurements_sha256 = require_sha256(
        resource_preflight["storage_measurements_sha256"],
        name="resource_preflight.storage_measurements_sha256",
    )
    return with_content_hash(
        {
            "contract": FULL_AUTHORIZATION_CONTRACT,
            "schema_version": 8,
            "source": dict(source),
            "production_execution_plan_sha256": production_plan["content_hash"],
            "miniature_acceptance_sha256": miniature_acceptance["content_hash"],
            "resource_preflight_sha256": resource_preflight["content_hash"],
            "storage_measurements_sha256": storage_measurements_sha256,
            "full_campaign_submission_authorized": True,
            "all_registered_rows_must_run": True,
            "scientific_underperformance_can_stop_campaign": False,
        }
    )


__all__ = [
    "RESOURCE_CLASSES",
    "MINIATURE_RESOURCE_CLASSES",
    "build_campaign_monitor",
    "build_full_authorization",
    "build_miniature_acceptance",
    "build_miniature_check_receipt",
    "build_production_execution_plan",
    "build_resource_measurements",
    "build_slurm_submission_ledger",
    "node_execution",
]
