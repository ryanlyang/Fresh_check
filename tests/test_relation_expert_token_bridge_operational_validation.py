from __future__ import annotations

from pathlib import Path
import sys

import pytest

from teacher_logit_reco.relation_expert_token_bridge import (
    MANIFEST_PLAN_FACTORY_REGISTRY_CONTRACT,
    audit_manifest_producer_invocations,
    build_full_submission_authorization,
    build_job_ledger,
    build_task_manifest,
    build_manifest_materialization_plan,
    build_production_dry_run_evidence,
    build_production_graph,
    build_tigris_smoke_evidence,
    materialize_downstream_manifests,
    publish_manifest_materialization_plan,
    publish_task_row_completion,
    run_local_synthetic_dag,
    source_snapshot,
    validate_full_submission_authorization,
    validate_stale_cancellation_request,
)
from teacher_logit_reco.relation_expert_token_bridge.task_completion import (
    build_task_manifest_completion,
    task_manifest_completion_path,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    bind_source,
    build_campaign_spec,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.storage import (
    REQUIRED_MEASUREMENTS,
    build_storage_measurements,
)
from teacher_logit_reco.relation_expert_token_bridge.manifest_orchestration import (
    manifest_plan_path,
)


ROOT = Path(__file__).resolve().parents[1]
PARENTS = (
    "artifact_layout",
    "final_select_label_manifest",
    "global_determinism",
    "hlt_replica_manifest",
    "raw_input_schema",
    "scale_train_manifest",
    "split_audit",
    "split_manifest",
    "storage_measurements",
    "validation_partition_manifest",
)


def _campaign_graph(
    root: Path, snapshot: dict, *, miniature: bool
) -> tuple[dict, dict]:
    profile = "miniature_test" if miniature else "production_500k_scale3m"
    campaign = build_campaign_spec(
        campaign_id=root.name,
        campaign_profile=profile,
        source_snapshot=snapshot,
        parent_artifact_hashes={
            name: f"{index + 1:064x}"
            for index, name in enumerate(PARENTS)
        },
        run_registry_hashes={"operational-test": "f" * 64},
    )
    graph = build_production_graph(
        campaign_root=root,
        campaign_id=root.name,
        source_commit=str(snapshot["source_commit"]),
        source_status_sha256=str(snapshot["source_status_sha256"]),
        storage_measurements_sha256=campaign[
            "parent_artifact_hashes"
        ]["storage_measurements"],
        miniature=miniature,
    )
    return campaign, graph


def _jobs(graph: dict, start: int = 90_000) -> dict[str, str]:
    jobs = {
        node["node_id"]: str(start + index)
        for index, node in enumerate(graph["nodes"])
    }
    jobs["rejection_finalist_selector"] = jobs[
        "accuracy_finalist_selector"
    ]
    jobs["locked_scale_finalists"] = jobs[
        "accuracy_finalist_selector"
    ]
    return jobs


def test_local_synthetic_dag_traverses_every_stage_and_recovery_path(
    tmp_path: Path,
) -> None:
    report = run_local_synthetic_dag(
        campaign_root=tmp_path / "local-retb", repo_root=ROOT
    )
    assert report["node_count"] == 84
    assert report["worker_interfaces"]["wrapper_count"] == 84
    assert report["worker_interfaces"]["python_cli_help_probed"] is True
    assert report["worker_interfaces"]["python_cli_help_probe_count"] > 50
    assert report["checks"] == {
        "all_graph_nodes_traversed": True,
        "all_stages_A_through_N_traversed": True,
        "synthetic_outputs_are_not_worker_execution_evidence": True,
        "incomplete_array_completion_rejected": True,
        "authenticated_row_reuse_verified": True,
        "source_drift_rejected": True,
        "interrupted_selector_continuation_rejected_then_recovered": True,
        "failed_node_restart_frontier_verified": True,
        "completed_ledger_bound": True,
        "smoke_simulation_resolved_all_nodes": True,
        "stale_cancellation_requires_explicit_drifted_lineage": True,
        "performance_threshold_abort_observed": False,
    }
    invocation_audit = report["manifest_producer_invocations"]
    assert invocation_audit[
        "registration_alone_counted_as_execution_evidence"
    ] is False
    assert invocation_audit[
        "shared_hook_counted_as_factory_evidence"
    ] is False
    assert invocation_audit[
        "synthetic_dag_counted_as_execution_evidence"
    ] is False
    assert invocation_audit["manifest_target_count"] == 63
    assert invocation_audit["bootstrap_prepublished_target_count"] == 13
    assert invocation_audit["downstream_plan_factory_target_count"] == 50
    assert invocation_audit["registered_plan_factory_count"] == 50
    # Closure Blocks 1/2, the first two Stage-F factories, all ten
    # Stage-K--M factories, and all eight Stage-N factories are wired.
    assert invocation_audit["implemented_producer_count"] == 63
    assert invocation_audit["execution_complete_producer_count"] == 0
    assert invocation_audit[
        "missing_execution_complete_producer_count"
    ] == 63
    assert invocation_audit[
        "full_submission_producer_gate_passed"
    ] is False
    assert report["full_submission_eligible"] is False
    assert report[
        "synthetic_control_plane_execution_is_sufficient"
    ] is False


def test_plan_factory_registry_is_exhaustive_and_missing_factory_fails_gate(
    tmp_path: Path,
) -> None:
    snapshot = source_snapshot(ROOT)
    _, graph = _campaign_graph(
        tmp_path / "factory-registry", snapshot, miniature=True
    )
    registry = graph["manifest_plan_factory_registry"]
    assert registry["contract"] == MANIFEST_PLAN_FACTORY_REGISTRY_CONTRACT
    assert registry["entry_count"] == 50
    assert len({
        entry["target_node_id"] for entry in registry["entries"]
    }) == 50
    assert all(
        entry["performance_independent_continuation_required"] is True
        and entry[
            "scientific_underperformance_blocks_continuation"
        ]
        is False
        and entry["allowed_worker_entrypoints"]
        and entry["expected_output_contracts"]
        for entry in registry["entries"]
    )

    missing_one = {
        **registry,
        "entries": registry["entries"][1:],
        "entry_count": registry["entry_count"] - 1,
    }
    audit = audit_manifest_producer_invocations(
        production_graph=graph,
        repo_root=ROOT,
        plan_factory_registry=missing_one,
    )
    assert audit["shared_materialization_hook_present"] is True
    assert audit["missing_plan_factory_registration_count"] == 1
    assert audit["full_submission_producer_gate_passed"] is False


def test_stage_b_c_manifest_ownership_follows_complete_metric_production(
    tmp_path: Path,
) -> None:
    snapshot = source_snapshot(ROOT)
    _, graph = _campaign_graph(
        tmp_path / "stage-b-c-order", snapshot, miniature=True
    )
    nodes = {
        str(node["node_id"]): node for node in graph["nodes"]
    }
    order = [
        str(node["node_id"]) for node in graph["nodes"]
    ]
    assert order.index("offline_expert_training") < order.index(
        "offline_expert_confirmation"
    )
    assert order.index("offline_expert_confirmation") < order.index(
        "offline_fusion_cache"
    )
    assert order.index("offline_fusion_cache") < order.index(
        "offline_fusion_training"
    )
    assert order.index("offline_fusion_training") < order.index(
        "offline_shape_selector"
    )
    assert nodes["step5_offline_fusion_contracts"]["dependencies"] == [
        "step4_offline_training_contracts"
    ]
    assert nodes["offline_fusion_training"]["dependencies"] == [
        "step5_offline_fusion_contracts",
        "offline_fusion_cache",
    ]
    assert nodes["offline_expert_confirmation"]["dependencies"] == [
        "step5_offline_fusion_contracts",
        "offline_expert_training",
    ]
    assert nodes["offline_fusion_cache"]["dependencies"] == [
        "offline_expert_confirmation"
    ]
    assert nodes["offline_shape_selector"]["dependencies"] == [
        "offline_fusion_training"
    ]
    assert set(nodes["step6_native_hlt_contracts"]["dependencies"]) == {
        "offline_shape_selector",
        "offline_optimization_selector",
        "offline_complementarity",
        "offline_capacity_controls",
        "input_audit",
    }
    registry = {
        str(entry["node_id"]): entry
        for entry in graph["node_execution_registry"]["entries"]
    }
    assert registry["offline_shape_selector"]["manifest_producer"][
        "node_id"
    ] == "offline_fusion_training"
    assert registry["offline_optimization_selector"]["manifest_producer"][
        "node_id"
    ] == "offline_expert_training"


def test_stale_cancellation_is_explicit_bound_and_source_drift_only(
    tmp_path: Path,
) -> None:
    snapshot = source_snapshot(ROOT)
    _, graph = _campaign_graph(tmp_path / "stale", snapshot, miniature=True)
    ledger = build_job_ledger(
        production_graph=graph,
        jobs=_jobs(graph),
        submission_mode="smoke_submitted",
    )
    bound = ledger["jobs"]["campaign_bootstrap"]
    assert validate_stale_cancellation_request(
        job_ledger=ledger,
        stale_job_ids=[bound, bound],
        source_validated=False,
    ) == [bound]
    with pytest.raises(ValueError, match="matching source"):
        validate_stale_cancellation_request(
            job_ledger=ledger,
            stale_job_ids=[bound],
            source_validated=True,
        )
    with pytest.raises(ValueError, match="ledger-bound"):
        validate_stale_cancellation_request(
            job_ledger=ledger,
            stale_job_ids=["999999"],
            source_validated=False,
        )


def test_post_completion_materializer_is_fail_closed_idempotent_and_negative_safe(
    tmp_path: Path,
) -> None:
    snapshot = source_snapshot(ROOT)
    campaign, graph = _campaign_graph(
        tmp_path / "materializer", snapshot, miniature=True
    )
    root = tmp_path / "materializer"
    write_immutable_json(root / "campaign_spec.json", campaign)
    write_immutable_json(
        root / "job_ledgers" / "production_graph.json", graph
    )
    with pytest.raises(FileNotFoundError, match="required manifest plan"):
        materialize_downstream_manifests(
            campaign_root=root,
            repo_root=ROOT,
            producer_node_id="offline_expert_training",
            campaign=campaign,
            production_graph=graph,
        )
    trigger = with_content_hash(
        {
            "contract": "retb_negative_selector_fixture_v1",
            "schema_version": 1,
            "result": "all_candidates_worse_than_baseline",
            "source": campaign["source"],
        }
    )
    trigger_path = root / "selection" / "negative_optimization.json"
    write_immutable_json(trigger_path, trigger)
    producer_manifest = build_task_manifest(
        campaign_spec_sha256=campaign["content_hash"],
        production_graph_sha256=graph["content_hash"],
        node_id="offline_expert_training",
        rows=[
            {
                "task_id": "offline_expert_training:0",
                "argv": [sys.executable, "scripts/write_retb_synthetic_output.py"],
                "environment": {},
                "expected_outputs": [str(trigger_path.resolve())],
                "input_artifact_hashes": {
                    "campaign_spec": campaign["content_hash"],
                    "production_graph": graph["content_hash"],
                },
            }
        ],
        maximum_concurrent_tasks=1,
    )
    publish_task_row_completion(
        campaign_root=root,
        campaign=campaign,
        task_manifest=producer_manifest,
        task_index=0,
    )
    producer_completion = build_task_manifest_completion(
        campaign_root=root,
        campaign=campaign,
        task_manifest=producer_manifest,
    )
    write_immutable_json(
        task_manifest_completion_path(
            root, node_id="offline_expert_training"
        ),
        producer_completion,
    )
    output = root / "selection" / "optimization_selector_done.json"
    plan = build_manifest_materialization_plan(
        campaign=campaign,
        production_graph=graph,
        producer_node_id="offline_expert_training",
        target_node_id="offline_optimization_selector",
        trigger_artifact_path=trigger_path,
        trigger_artifact_sha256=trigger["content_hash"],
        rows=[
            {
                "task_id": "offline_optimization_selector:0",
                "argv": [
                    sys.executable,
                    "scripts/write_retb_synthetic_output.py",
                    "--campaign-root",
                    str(root),
                    "--node-id",
                    "offline_optimization_selector",
                    "--task-index",
                    "0",
                    "--output",
                    str(output),
                ],
                "environment": {},
                "expected_outputs": [str(output)],
                "input_artifact_hashes": {
                    "negative_selection": trigger["content_hash"],
                    "producer_completion": producer_completion["content_hash"],
                },
            }
        ],
    )
    publish_manifest_materialization_plan(
        campaign_root=root,
        plan=plan,
        campaign=campaign,
        production_graph=graph,
    )
    first = materialize_downstream_manifests(
        campaign_root=root,
        repo_root=ROOT,
        producer_node_id="offline_expert_training",
        campaign=campaign,
        production_graph=graph,
    )
    authenticated_plan_path = manifest_plan_path(
        root, target_node_id="offline_optimization_selector"
    )
    authenticated_plan_path.unlink()
    with pytest.raises(FileNotFoundError, match="authenticated producer plan"):
        materialize_downstream_manifests(
            campaign_root=root,
            repo_root=ROOT,
            producer_node_id="offline_expert_training",
            campaign=campaign,
            production_graph=graph,
        )
    write_immutable_json(authenticated_plan_path, plan)
    second = materialize_downstream_manifests(
        campaign_root=root,
        repo_root=ROOT,
        producer_node_id="offline_expert_training",
        campaign=campaign,
        production_graph=graph,
    )
    assert first["target_count"] == 1
    assert first["manifest_hashes"] == second["manifest_hashes"]
    assert second["publications"]["offline_optimization_selector"] == {
        "status": "already_present",
        "producer_plan_revalidated": True,
        "producer_completion_revalidated": True,
    }


def test_real_smoke_and_authenticated_dry_run_gate_full_submission(
    tmp_path: Path,
) -> None:
    snapshot = source_snapshot(ROOT)
    local = run_local_synthetic_dag(
        campaign_root=tmp_path / "local", repo_root=ROOT
    )
    smoke_campaign, smoke_graph = _campaign_graph(
        tmp_path / "smoke", snapshot, miniature=True
    )
    smoke_ledger = build_job_ledger(
        production_graph=smoke_graph,
        jobs=_jobs(smoke_graph),
        submission_mode="completed",
        completion_artifact_hashes={
            "locked_scale_finalists": "1" * 64,
            "final_test_execution_lock": "2" * 64,
            "sealed_final_test_evaluation": "3" * 64,
            "final_report": "4" * 64,
        },
    )
    smoke = build_tigris_smoke_evidence(
        campaign_spec=smoke_campaign,
        production_graph=smoke_graph,
        completed_ledger=smoke_ledger,
        source_snapshot=snapshot,
    )
    evidence_file = tmp_path / "storage-source.bin"
    evidence_file.write_bytes(b"authenticated representative storage")
    measurements = {
        name: 1 for name in REQUIRED_MEASUREMENTS
    }
    measurements["available_storage_bytes"] = 10
    storage = build_storage_measurements(
        measurements=measurements,
        source_evidence={
            "representative": {
                "path": str(evidence_file),
                "purpose": "test",
            }
        },
        measurement_profile="production_source_evidence",
    )
    bound_storage = bind_source(storage, source_snapshot=snapshot)
    production_graph = build_production_graph(
        campaign_root=tmp_path / "production-dry",
        campaign_id="production-dry",
        source_commit=str(snapshot["source_commit"]),
        source_status_sha256=str(snapshot["source_status_sha256"]),
        storage_measurements_sha256=bound_storage["content_hash"],
        miniature=False,
    )
    dry_ledger = build_job_ledger(
        production_graph=production_graph,
        jobs={
            node["node_id"]: None for node in production_graph["nodes"]
        },
        submission_mode="dry_run",
    )
    dry = build_production_dry_run_evidence(
        production_graph=production_graph,
        dry_run_ledger=dry_ledger,
        storage_measurements=storage,
        source_snapshot=snapshot,
    )
    with pytest.raises(
        ValueError, match="execution-complete genuine producer evidence"
    ):
        build_full_submission_authorization(
            local_report=local,
            tigris_smoke_evidence=smoke,
            production_dry_run_evidence=dry,
            source_snapshot=snapshot,
        )


def test_full_submitter_requires_operational_authorization_before_sbatch() -> None:
    source = (
        ROOT / "sbatch" / "submit_retb_tigris_full.sh"
    ).read_text(encoding="utf-8")
    gate = source.index("verify-authorization")
    submission = source.index("sbatch --parsable")
    assert gate < submission
    assert "RETB_OPERATIONAL_AUTHORIZATION" in source
    assert "Complete local validation" in source
