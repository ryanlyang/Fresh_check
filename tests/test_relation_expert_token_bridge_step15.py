from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
from scripts.measure_retb_streamed_abc_storage import (
    main as measure_streamed_storage_main,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    with_content_hash,
)

from teacher_logit_reco.relation_expert_token_bridge import (
    DEFAULT_CONCURRENCY,
    TASK_MANIFEST_PRODUCER_NODES,
    build_offline_submission_scope,
    build_streamed_offline_submission_scope,
    build_job_ledger,
    build_node_execution_registry,
    build_production_graph,
    build_resource_probe,
    build_resume_plan,
    build_step15_contract_bundle,
    build_target_shard_plan,
    build_task_manifest,
    offline_submission_node_ids,
    source_snapshot,
    validate_job_ledger,
    validate_node_execution_registry,
    validate_offline_submission_scope,
    validate_streamed_offline_submission_scope,
    validate_production_graph,
    validate_production_campaign_binding,
    validate_resource_probe,
    validate_step15_contract_bundle,
    validate_target_shard_plan,
    validate_task_manifest,
    validate_task_manifest_for_graph,
)


ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source() -> dict[str, object]:
    return {
        "source_commit": "a" * 40,
        "source_status_sha256": "b" * 64,
        "source_dirty": False,
    }


def _graph(*, miniature: bool = False) -> dict:
    return build_production_graph(
        campaign_root="/campaign/retb_test",
        campaign_id="retb_test",
        source_commit="a" * 40,
        source_status_sha256="b" * 64,
        storage_measurements_sha256="c" * 64,
        miniature=miniature,
    )


def test_production_graph_contract_and_schema_validator_are_consistent() -> None:
    graph = _graph(miniature=True)
    assert graph["contract"] == "retb_tigris_production_graph_v40"
    assert graph["schema_version"] == 37
    assert validate_production_graph(graph) == graph["content_hash"]


def test_detached_campaign_source_is_independent_of_mutable_checkout(
    tmp_path: Path,
) -> None:
    mutable = tmp_path / "mutable"
    frozen = tmp_path / "frozen"
    mutable.mkdir()
    _git(mutable, "init")
    _git(mutable, "config", "user.name", "RETB Test")
    _git(mutable, "config", "user.email", "retb@example.invalid")
    tracked = mutable / "worker.py"
    tracked.write_text("VERSION = 1\n", encoding="utf-8")
    _git(mutable, "add", "worker.py")
    _git(mutable, "commit", "-m", "campaign source")
    campaign_commit = _git(mutable, "rev-parse", "HEAD")
    _git(
        mutable,
        "worktree",
        "add",
        "--detach",
        str(frozen),
        campaign_commit,
    )
    before = source_snapshot(frozen)
    tracked.write_text("VERSION = 2\n", encoding="utf-8")
    _git(mutable, "add", "worker.py")
    _git(mutable, "commit", "-m", "unrelated later checkout change")
    after = source_snapshot(frozen)
    assert after == before
    assert after["source_commit"] == campaign_commit
    assert after["source_dirty"] is False
    assert (frozen / "worker.py").read_text(encoding="utf-8") == (
        "VERSION = 1\n"
    )


def test_complete_graph_covers_stages_selectors_scale_and_two_locks() -> None:
    graph = _graph()
    validate_production_graph(graph)
    assert {row["stage"] for row in graph["nodes"]} == set("ABCDEFGHIJKLMN")
    nodes = {row["node_id"]: row for row in graph["nodes"]}
    assert graph["two_stage_n_selectors"] == [
        "accuracy_finalist_selector",
        "rejection_finalist_selector",
    ]
    assert nodes["scale_graph_training"]["stage"] == "M"
    assert nodes["compiled_region_backend"]["stage"] == "A"
    assert "compiled_region_backend" in nodes["input_audit"]["dependencies"]
    assert nodes["stack_val_inference"]["dependencies"] == ["scale_completion"]
    assert set(nodes["locked_scale_finalists"]["dependencies"]) == {
        "accuracy_finalist_selector",
        "rejection_finalist_selector",
    }
    assert nodes["rejection_finalist_selector"]["virtual_alias_of"] == (
        "accuracy_finalist_selector"
    )
    assert nodes["sealed_final_test"]["dependencies"] == [
        "final_test_execution_lock"
    ]
    assert set(nodes["stage_n_evidence_join"]["dependencies"]) == {
        "postlock_oracle_targets",
        "finalist_controls",
        "prelock_final_inputs",
    }
    assert nodes["final_test_execution_lock"]["dependencies"] == [
        "stage_n_evidence_join"
    ]
    assert nodes["prelock_final_inputs"]["dataset_access"] == (
        "checkpoint_free_final_input_preparation"
    )
    contract_barriers = {
        node_id
        for node_id in nodes
        if node_id.startswith("step") and node_id.endswith("_contracts")
    }
    assert contract_barriers == {
        "step3_architecture_contracts",
        "step4_offline_training_contracts",
        "step5_offline_fusion_contracts",
        "step6_native_hlt_contracts",
        "step7_bridge_contracts",
        "step8_target_cache_contracts",
        "step9_predictor_contracts",
        "step10_predictor_bundle_contracts",
        "step11_joint_bridge_contracts",
        "step12_final_consumer_contracts",
        "step13_confirmation_contracts",
        "step14_scale_final_contracts",
    }
    assert graph["negative_campaign_continues_to_final_report"] is True
    assert graph["performance_based_termination"] is False
    expected_sizes = {
        "model_train": 500_000,
        "model_val": 100_000,
        "stack_train": 0,
        "stack_val": 50_000,
        "final_test": 300_000,
        "scale_train": 3_000_000,
    }
    assert graph["split_sizes"] == expected_sizes
    for node in graph["nodes"]:
        assert node["dependency_mode"] == "afterok"
        assert node["performance_warning_blocks_dependency"] is False
        assert node["scientific_underperformance_blocks_dependency"] is False
        assert node["provenance_or_execution_failure_blocks_dependency"] is True
        if node["array"] is not None:
            assert node["array"]["maximum_concurrent_tasks"] > 0
            assert node["array"]["maximum_tasks"] >= node["array"]["smoke_tasks"]


def test_miniature_graph_sizes_match_the_genuine_step1_identity_profile() -> None:
    graph = _graph(miniature=True)
    assert graph["split_sizes"] == {
        "model_train": 20,
        "model_val": 20,
        "stack_train": 0,
        "stack_val": 10,
        "final_test": 20,
        "scale_train": 40,
    }
    drifted = dict(graph)
    drifted["split_sizes"] = {**graph["split_sizes"], "model_train": 400}
    drifted.pop("content_hash")
    drifted = with_content_hash(drifted)
    with pytest.raises(ValueError, match="scientific controls"):
        validate_production_graph(drifted)


def test_hosd_miniature_graph_has_source_bound_double_validation_profile() -> None:
    from teacher_logit_reco.relation_expert_token_bridge.production import (
        HOSD_MINIATURE_SPLIT_PROFILE,
    )

    graph = build_production_graph(
        campaign_root="/tmp/shared_retb_parent_campaign",
        campaign_id="shared_retb_parent_campaign",
        source_commit="a" * 40,
        source_status_sha256="b" * 64,
        storage_measurements_sha256="c" * 64,
        miniature=True,
        miniature_split_profile=HOSD_MINIATURE_SPLIT_PROFILE,
        split_profile_parent_sha256="d" * 64,
    )
    assert graph["split_sizes"] == {
        "model_train": 20,
        "model_val": 40,
        "stack_train": 0,
        "stack_val": 10,
        "final_test": 20,
        "scale_train": 40,
    }
    assert graph["split_profile_parent_sha256"] == "d" * 64
    validate_production_graph(graph)


def test_node_execution_registry_covers_every_worker_and_manifest_producer() -> None:
    graph = _graph()
    registry = graph["node_execution_registry"]
    validate_node_execution_registry(registry, nodes=graph["nodes"])
    entries = {
        row["node_id"]: row for row in registry["entries"]
    }
    nodes = {row["node_id"]: row for row in graph["nodes"]}
    assert set(entries) == set(nodes)
    assert registry["node_count"] == len(nodes)
    assert registry["missing_manifest_producers"] == []
    for node_id, entry in entries.items():
        node = nodes[node_id]
        assert entry["worker"] == node["worker"]
        assert entry["resource"] == node["resource"]
        assert entry["required_inputs"][:2] == [
            "campaign_spec",
            "production_graph",
        ]
        if entry["manifest_required"]:
            assert entry["task_manifest_path"]
            assert entry["manifest_producer"]["node_id"]
            assert entry["manifest_producer"]["entrypoint"]
            assert entry["row_resolution"] in {"static", "dynamic"}
        else:
            assert entry["task_manifest_path"] is None
            assert entry["manifest_producer"] is None
            assert entry["row_resolution"] == "not_applicable"
    assert entries["input_audit"]["manifest_producer"]["node_id"] == (
        "campaign_bootstrap"
    )
    assert entries["offline_input_cache"]["manifest_producer"] == {
        "node_id": "campaign_bootstrap",
        "entrypoint": "scripts/bootstrap_retb_input_tasks.py",
        "publication_mode": "campaign_bootstrap",
    }


def test_manifest_driven_node_without_producer_fails_closed() -> None:
    graph = _graph()
    incomplete = dict(TASK_MANIFEST_PRODUCER_NODES)
    incomplete.pop("input_audit")
    with pytest.raises(
        ValueError, match="producer coverage.*input_audit"
    ):
        build_node_execution_registry(
            nodes=graph["nodes"],
            manifest_producer_nodes=incomplete,
        )

    registry = graph["node_execution_registry"]
    entries = [dict(row) for row in registry["entries"]]
    target = next(
        row for row in entries if row["node_id"] == "input_audit"
    )
    target["manifest_producer"] = None
    bad_registry = dict(registry)
    bad_registry.pop("content_hash")
    bad_registry["entries"] = entries
    bad_registry = with_content_hash(bad_registry)
    with pytest.raises(ValueError, match="automatic manifest producer"):
        validate_node_execution_registry(
            bad_registry, nodes=graph["nodes"]
        )

    bad_graph = dict(graph)
    bad_graph.pop("content_hash")
    bad_graph["node_execution_registry"] = bad_registry
    bad_graph = with_content_hash(bad_graph)
    with pytest.raises(ValueError, match="automatic manifest producer"):
        validate_production_graph(bad_graph)


def test_execution_registry_rejects_worker_resource_and_io_drift() -> None:
    graph = _graph()
    registry = graph["node_execution_registry"]
    entries = [dict(row) for row in registry["entries"]]
    target = next(
        row
        for row in entries
        if row["node_id"] == "offline_expert_training"
    )
    target["worker"] = "run_unregistered_worker.sh"
    bad = dict(registry)
    bad.pop("content_hash")
    bad["entries"] = entries
    bad = with_content_hash(bad)
    with pytest.raises(ValueError, match="registry semantics"):
        validate_node_execution_registry(bad, nodes=graph["nodes"])


def test_production_graph_binds_exact_campaign_source_storage_and_profile() -> None:
    graph = _graph()
    campaign = {
        "campaign_id": "retb_test",
        "campaign_profile": "production_500k_scale3m",
        "source": {
            "commit": "a" * 40,
            "status_sha256": "b" * 64,
        },
        "parent_artifact_hashes": {"storage_measurements": "c" * 64},
    }
    validate_production_campaign_binding(graph, campaign)
    campaign["parent_artifact_hashes"]["storage_measurements"] = "d" * 64
    with pytest.raises(ValueError, match="campaign binding"):
        validate_production_campaign_binding(graph, campaign)


def test_graph_rejects_performance_gate_and_premature_final_test() -> None:
    graph = _graph()
    graph["nodes"][10]["scientific_underperformance_blocks_dependency"] = True
    graph.pop("content_hash")
    graph = with_content_hash(graph)
    with pytest.raises(ValueError, match="failure semantics"):
        validate_production_graph(graph)

    graph = _graph()
    by_id = {row["node_id"]: row for row in graph["nodes"]}
    by_id["sealed_final_test"]["dependencies"] = ["locked_scale_finalists"]
    graph.pop("content_hash")
    graph = with_content_hash(graph)
    with pytest.raises(
        ValueError,
        match="manifest producer is not an ancestor|lacks ancestors",
    ):
        validate_production_graph(graph)


def test_job_ledger_binds_virtual_selectors_and_rejects_unknown_jobs() -> None:
    graph = _graph(miniature=True)
    jobs = {
        row["node_id"]: str(10_000 + index)
        for index, row in enumerate(graph["nodes"])
    }
    selector = jobs["accuracy_finalist_selector"]
    jobs["rejection_finalist_selector"] = selector
    jobs["locked_scale_finalists"] = selector
    ledger = build_job_ledger(
        production_graph=graph,
        jobs=jobs,
        submission_mode="smoke_submitted",
    )
    validate_job_ledger(ledger, production_graph=graph)
    assert ledger["all_nodes_bound"] is True
    assert ledger["performance_based_cancellation_allowed"] is False
    with pytest.raises(ValueError, match="unknown nodes"):
        build_job_ledger(
            production_graph=graph,
            jobs={"not_a_node": "123"},
            submission_mode="production_submitted",
        )


def test_offline_submission_scope_is_exact_dependency_closed_and_real() -> None:
    graph = _graph()
    node_ids = offline_submission_node_ids(graph)
    selected = set(node_ids)
    by_id = {row["node_id"]: row for row in graph["nodes"]}
    assert {by_id[node_id]["stage"] for node_id in selected} == set("ABC")
    assert all(
        set(by_id[node_id]["dependencies"]).issubset(selected)
        for node_id in selected
    )
    assert "offline_capacity_controls" in selected
    assert "step6_native_hlt_contracts" not in selected
    assert "sealed_final_test" not in selected

    scope = build_offline_submission_scope(production_graph=graph)
    validate_offline_submission_scope(scope, production_graph=graph)
    assert scope["submitted_node_ids"] == node_ids
    assert scope["excluded_stages"] == list("DEFGHIJKLMN")
    assert scope["final_test_jobs_submitted"] is False
    assert scope["full_campaign_operational_authorization_claimed"] is False

    jobs = {
        node_id: str(30_000 + index)
        for index, node_id in enumerate(node_ids)
    }
    ledger = build_job_ledger(
        production_graph=graph,
        jobs=jobs,
        submission_mode="offline_production_submitted",
    )
    validate_job_ledger(ledger, production_graph=graph)
    assert ledger["submission_scope"] == "offline_abc"
    assert ledger["submission_scope_sha256"] == scope["content_hash"]
    assert ledger["all_scope_nodes_bound"] is True
    assert ledger["all_nodes_bound"] is False
    assert ledger["submitted_node_count"] == len(node_ids)
    assert all(
        ledger["jobs"][node_id] is None
        for node_id in scope["excluded_node_ids"]
    )

    missing = dict(jobs)
    missing.pop(node_ids[-1])
    with pytest.raises(ValueError, match="exactly every A-C node"):
        build_job_ledger(
            production_graph=graph,
            jobs=missing,
            submission_mode="offline_production_submitted",
        )
    with pytest.raises(ValueError, match="exactly every A-C node"):
        build_job_ledger(
            production_graph=graph,
            jobs={**jobs, "step6_native_hlt_contracts": "39999"},
            submission_mode="offline_production_submitted",
        )


def test_offline_submission_scope_rejects_miniature_graph() -> None:
    with pytest.raises(ValueError, match="requires a production graph"):
        offline_submission_node_ids(_graph(miniature=True))


def test_streamed_offline_scope_and_ledger_are_explicit() -> None:
    graph = build_production_graph(
        campaign_root="/campaign/retb_streamed",
        campaign_id="retb_streamed",
        source_commit="a" * 40,
        source_status_sha256="b" * 64,
        storage_measurements_sha256="c" * 64,
        execution_profile="offline_abc_streamed",
    )
    scope = build_streamed_offline_submission_scope(
        production_graph=graph
    )
    validate_streamed_offline_submission_scope(
        scope, production_graph=graph
    )
    assert scope["frozen_token_banks_task_local"] is True
    assert scope["scientific_matrix_changed"] is False
    node_ids = offline_submission_node_ids(graph)
    ledger = build_job_ledger(
        production_graph=graph,
        jobs={node_id: str(40_000 + i) for i, node_id in enumerate(node_ids)},
        submission_mode="offline_streamed_production_submitted",
    )
    validate_job_ledger(ledger, production_graph=graph)
    assert ledger["submission_scope"] == "offline_abc_streamed"
    assert ledger["submission_scope_sha256"] == scope["content_hash"]


def test_streamed_storage_projection_is_source_evidence_bound(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    for role in ("model_train", "val_stop", "val_design"):
        directory = evidence / "inputs" / "offline" / role
        directory.mkdir(parents=True)
        (directory / "offline_inputs.npz").write_bytes(b"x" * 100)
        (directory / "offline_input_manifest.json").write_text(
            json.dumps({"event_count": 100}), encoding="utf-8"
        )
    hlt = (
        evidence
        / "inputs"
        / "hlt_v3"
        / "model_train"
        / "replica_0"
        / "R_MULTI"
        / "D_NOMINAL"
    )
    hlt.mkdir(parents=True)
    (hlt / "hlt_v3_arrays.npz").write_bytes(b"h" * 100)
    (hlt / "hlt_v3_metadata.json").write_text(
        json.dumps({"shape": [100, 128, 14]}), encoding="utf-8"
    )
    tree = (
        evidence
        / "inputs"
        / "region_tree"
        / "offline"
        / "model_train_exclusive_ca_v1"
        / "shards"
    )
    tree.mkdir(parents=True)
    (tree / "shard_00000.npz").write_bytes(b"t" * 100)
    (tree / "shard_00000.metadata.json").write_text(
        json.dumps({"jet_count": 100}), encoding="utf-8"
    )
    checkpoint = evidence / "runs" / "stage_c" / "run"
    checkpoint.mkdir(parents=True)
    (checkpoint / "best_model_val.pt").write_bytes(b"c" * 100)
    output = tmp_path / "bootstrap" / "storage_measurements.json"
    assert (
        measure_streamed_storage_main(
            [
                "--evidence-campaign-root",
                str(evidence),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    projection = json.loads(
        output.with_name(f"{output.stem}_projection.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["measurement_profile"] == "production_source_evidence"
    assert projection["storage_measurements_sha256"] == payload["content_hash"]
    assert projection["ephemeral_frozen_token_banks_included_in_persistent_peak"] is False


def test_offline_submission_plan_prints_only_authenticated_scope(
    tmp_path: Path,
) -> None:
    graph = _graph()
    graph_path = tmp_path / "production_graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "print_retb_submission_plan.py"),
            "--production-graph",
            str(graph_path),
            "--submission-scope",
            "offline_abc",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    printed = [line.split("|", 1)[0] for line in completed.stdout.splitlines()]
    assert printed == offline_submission_node_ids(graph)


def test_resource_probe_and_storage_admission_are_deterministic() -> None:
    accepted = build_resource_probe(
        campaign_spec_sha256="a" * 64,
        storage_measurements_sha256="b" * 64,
        resource_kind="gpu",
        node_name="gh-a-001",
        available_memory_bytes=300,
        available_storage_bytes=500,
        measured_items_per_second=42.5,
        compiler_backend_parity_passed=True,
        requested_memory_bytes=200,
        projected_peak_storage_bytes=400,
    )
    validate_resource_probe(accepted)
    assert accepted["resource_admitted"] is True
    rejected = build_resource_probe(
        campaign_spec_sha256="a" * 64,
        storage_measurements_sha256="b" * 64,
        resource_kind="gpu",
        node_name="gh-a-001",
        available_memory_bytes=199,
        available_storage_bytes=500,
        measured_items_per_second=42.5,
        compiler_backend_parity_passed=True,
        requested_memory_bytes=200,
        projected_peak_storage_bytes=400,
    )
    assert rejected["resource_admitted"] is False
    assert rejected["throughput_changes_scientific_selection"] is False


def test_target_shard_plan_is_bounded_contiguous_and_resumable() -> None:
    plan = build_target_shard_plan(
        campaign_spec_sha256="a" * 64,
        target_cache_specification_sha256="b" * 64,
        identity_order_sha256="c" * 64,
        event_count=5001,
        shard_size=2048,
        maximum_concurrent_tasks=2,
    )
    validate_target_shard_plan(plan)
    assert plan["shard_count"] == 3
    assert plan["slurm_array"] == "0-2%2"
    assert [
        (row["start_index"], row["stop_index_exclusive"])
        for row in plan["rows"]
    ] == [(0, 2048), (2048, 4096), (4096, 5001)]
    assert plan["resume_rule"] == (
        "reuse_only_after_manifest_and_npz_hash_validation"
    )
    assert plan["partial_npz_without_manifest_fails_closed"] is True


def test_task_manifest_has_exact_rows_and_no_performance_skipping() -> None:
    rows = [
        {
            "task_id": "predictor_training:0",
            "argv": [sys.executable, "worker.py", "--row", "0"],
            "environment": {"RETB_RUN_ID": "row0"},
            "expected_outputs": ["/campaign/output/row0.json"],
            "input_artifact_hashes": {"campaign": "a" * 64},
        },
        {
            "task_id": "predictor_training:1",
            "argv": [sys.executable, "worker.py", "--row", "1"],
            "environment": {"RETB_RUN_ID": "row1"},
            "expected_outputs": ["/campaign/output/row1.json"],
            "input_artifact_hashes": {"campaign": "a" * 64},
        },
    ]
    manifest = build_task_manifest(
        campaign_spec_sha256="b" * 64,
        production_graph_sha256="c" * 64,
        node_id="predictor_training",
        rows=rows,
        maximum_concurrent_tasks=1,
    )
    validate_task_manifest(manifest)
    assert manifest["slurm_array"] == "0-1%1"
    assert manifest["performance_based_row_skipping"] is False
    bad = [dict(rows[0])]
    bad[0]["task_id"] = "another:0"
    with pytest.raises(ValueError, match="row semantics"):
        build_task_manifest(
            campaign_spec_sha256="b" * 64,
            production_graph_sha256="c" * 64,
            node_id="predictor_training",
            rows=bad,
            maximum_concurrent_tasks=1,
        )


def test_task_manifest_cannot_escape_campaign_or_change_degradation() -> None:
    graph = _graph()
    declaration = {
        row["node_id"]: row for row in graph["nodes"]
    }["hlt_v3_cache"]["array"]
    row = {
        "task_id": "hlt_v3_cache:0",
        "argv": [
            sys.executable,
            "scripts/build_retb_hlt_v3_from_offline_cache.py",
            "--profile-id",
            "D_NOMINAL",
        ],
        "environment": {},
        "expected_outputs": [
            "/campaign/retb_test/inputs/hlt_v3/metadata.json"
        ],
        "input_artifact_hashes": {"campaign": "a" * 64},
    }
    manifest = build_task_manifest(
        campaign_spec_sha256="b" * 64,
        production_graph_sha256=graph["content_hash"],
        node_id="hlt_v3_cache",
        rows=[row],
        maximum_concurrent_tasks=int(
            declaration["maximum_concurrent_tasks"]
        ),
    )
    validate_task_manifest_for_graph(
        manifest,
        production_graph=graph,
        campaign_root="/campaign/retb_test",
        repo_root=ROOT,
    )
    bad_row = {**row, "argv": [*row["argv"][:-1], "D_TRACK_UP"]}
    bad = build_task_manifest(
        campaign_spec_sha256="b" * 64,
        production_graph_sha256=graph["content_hash"],
        node_id="hlt_v3_cache",
        rows=[bad_row],
        maximum_concurrent_tasks=int(
            declaration["maximum_concurrent_tasks"]
        ),
    )
    with pytest.raises(ValueError, match="degradation profile"):
        validate_task_manifest_for_graph(
            bad,
            production_graph=graph,
            campaign_root="/campaign/retb_test",
            repo_root=ROOT,
        )


def test_resume_reuses_only_declared_completed_ancestors() -> None:
    graph = _graph(miniature=True)
    jobs = {
        row["node_id"]: str(20_000 + index)
        for index, row in enumerate(graph["nodes"])
    }
    ledger = build_job_ledger(
        production_graph=graph,
        jobs=jobs,
        submission_mode="smoke_submitted",
    )
    completed = [
        "split_build",
        "campaign_bootstrap",
        "compiled_region_backend",
        "cpu_resource_probe",
        "gpu_resource_probe",
    ]
    resume = build_resume_plan(
        production_graph=graph,
        previous_ledger=ledger,
        completed_nodes={name: "d" * 64 for name in completed},
        failed_nodes=["offline_input_cache"],
    )
    assert [
        row["node_id"] for row in resume["reusable_completed_nodes"]
    ] == sorted(completed)
    assert all(
        row["output_artifact_sha256"] == "d" * 64
        for row in resume["reusable_completed_nodes"]
    )
    ready = {row["node_id"] for row in resume["ready_to_resubmit"]}
    assert "offline_input_cache" in ready
    assert "hlt_v3_cache" not in ready
    assert resume["performance_based_resubmission"] is False
    assert resume["final_test_may_run_more_than_once"] is False


def test_step15_bundle_and_shell_contracts_cover_production_interfaces() -> None:
    graph = _graph(miniature=True)
    ledger = build_job_ledger(
        production_graph=graph,
        jobs={row["node_id"]: None for row in graph["nodes"]},
        submission_mode="dry_run",
    )
    bundle = build_step15_contract_bundle(
        production_graph=graph,
        dry_run_ledger=ledger,
        source_snapshot=_source(),
    )
    validate_step15_contract_bundle(bundle)
    assert bundle["step15_bundle"]["stage_coverage"] == list("ABCDEFGHIJKLMN")
    step15 = bundle["step15_bundle"]
    checks = bundle["step15_preflight_report"]["checks"]
    assert checks[
        "negative_campaign_reaches_final_report"
    ]
    assert checks["all_63_manifest_targets_require_genuine_execution"]
    assert checks["all_50_nonbootstrap_targets_require_plan_factories"]
    assert checks["shortlisted_500k_controls_are_real_three_seed_training"]
    assert checks["complete_section_28_semantic_control_matrix_required"]
    assert checks["stage_M_training_split_into_bounded_component_continuations"]
    assert checks["sealed_final_test_rows_resume_under_one_immutable_claim"]
    assert checks["campaign_executes_from_clean_detached_worktree"]
    assert checks["mutable_submission_checkout_may_change_after_launch"]
    assert step15["stage_k_m_completion_contracts"][
        "shortlisted_500k_controls"
    ] == "retb_shortlisted_500k_controls_v3"
    assert step15["stage_n_completion_contracts"][
        "final_test_execution_claim"
    ] == "retb_final_test_execution_claim_v3"
    assert step15["source_execution_policy"] == {
        "submission_checkout_role": "mutable_control_plane_only",
        "campaign_checkout": (
            "detached_git_worktree_at_bound_source_commit"
        ),
        "campaign_checkout_must_remain_clean": True,
        "main_checkout_changes_after_submission_allowed": True,
        "all_slurm_jobs_export_frozen_project_dir": True,
        "uncommitted_submission_checkout_changes_executed": False,
    }
    assert step15["current_source_authorization_scope"] == (
        "frozen_campaign_checkout_not_mutable_submission_checkout"
    )
    assert step15["offline_abc_submission"] == {
        "supported": True,
        "scope_contract": "retb_offline_submission_scope_v1",
        "job_ledger_mode": "offline_production_submitted",
        "submitted_stages": list("ABC"),
        "complete_graph_remains_authoritative": True,
        "exact_dependency_closed_node_set_required": True,
        "real_data_and_authenticated_storage_required": True,
        "full_campaign_authorization_claimed": False,
        "stages_D_through_N_submitted": False,
        "final_test_jobs_submitted": False,
        "later_reuse_requires_authenticated_validation": True,
    }

    submitter = (ROOT / "sbatch" / "submit_retb_tigris_full.sh").read_text()
    common = (ROOT / "sbatch" / "retb_common.sh").read_text()
    array_launcher = (
        ROOT / "sbatch" / "run_retb_array_launcher.sh"
    ).read_text()
    split_worker = (
        ROOT / "sbatch" / "run_retb_build_splits.sh"
    ).read_text()
    resource_probe_worker = (
        ROOT / "sbatch" / "run_retb_resource_probe.sh"
    ).read_text()
    assert "--dry-run|--smoke-simulate|--smoke-submit" in submitter
    assert "--offline-submit" in submitter
    assert 'RETB_SUBMISSION_SCOPE="offline_abc"' in submitter
    assert '--submission-scope "${RETB_SUBMISSION_SCOPE}"' in submitter
    assert 'submission_mode="offline_production_submitted"' in submitter
    assert "offline_submission_scope.json" in submitter
    assert "print_retb_submission_plan.py" in submitter
    assert "dispatch_mode" in submitter
    assert "initial_submission_ledger.json" in submitter
    assert "storage projection:" in submitter
    assert "HLT-v3 cache hashes:" in submitter
    assert "monitor_retb_campaign.py" in submitter
    assert "D_NOMINAL" in submitter
    assert "git -C \"${submission_project_dir}\" worktree add --detach" in submitter
    assert "RETB_FROZEN_REENTRY=1" in submitter
    assert "RETB_FROZEN_SOURCE_COMMIT" in submitter
    assert "RETB_SUBMISSION_PROJECT_DIR" in submitter
    assert "PROJECT_DIR=${PROJECT_DIR},CAMPAIGN_ROOT=" in submitter
    assert submitter.index("worktree add --detach") < submitter.index(
        "scripts/submit_retb_graph.py \"${graph_arguments[@]}\""
    )
    assert submitter.count("CONCURRENCY:=64") == 5
    assert submitter.count("export RETB_") >= 5
    assert DEFAULT_CONCURRENCY == {
        "cpu_cache": 64,
        "gpu_expert": 64,
        "gpu_predictor": 64,
        "gpu_scale": 64,
        "gpu_final": 64,
    }
    assert "performance" not in submitter.lower()
    assert 'source "${PROJECT_DIR}/sbatch/retb_common.sh"' in array_launcher
    assert "sbatch --parsable --wait" in array_launcher
    assert "PYTHONNOUSERSITE=1" in common
    assert ': "${RETB_C_COMPILER:=/usr/bin/gcc}"' in common
    assert ': "${RETB_CXX_COMPILER:=/usr/bin/c++}"' in common
    assert 'export CC="${RETB_C_COMPILER}"' in common
    assert 'export CXX="${RETB_CXX_COMPILER}"' in common
    assert "Pinned RETB C compiler is absent" in common
    assert "Pinned RETB C++ compiler is absent" in common
    assert graph["tigris_defaults"]["c_compiler"] == "/usr/bin/gcc"
    assert graph["tigris_defaults"]["cxx_compiler"] == "/usr/bin/c++"
    assert graph["node_execution_registry"]["runtime_environment"] == {
        "c_compiler": "/usr/bin/gcc",
        "cxx_compiler": "/usr/bin/c++",
    }
    assert 'cd "${PROJECT_DIR}"' in common
    assert "retb_validate_frozen_source" in common
    assert "Frozen RETB source checkout became dirty" in common
    assert 'json.load(open(sys.argv[1]))["split_sizes"]' in split_worker
    assert (
        'job_ledgers/resource_probes/${RETB_RESOURCE_KIND}.json'
        in resource_probe_worker
    )
    assert "job_ledgers/resource_probe_${RETB_RESOURCE_KIND}.json" not in (
        resource_probe_worker
    )
    assert "sizes=(20 20 0 10 20)" not in split_worker
    assert "/var/spool" not in common + submitter + array_launcher

    workers = {row["worker"] for row in graph["nodes"]}
    assert all((ROOT / "sbatch" / worker).is_file() for worker in workers)


def test_submission_graph_cli_prints_complete_miniature_dag() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "submit_retb_graph.py"),
            "--miniature",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["production_submission_performed"] is False
    assert {row["stage"] for row in payload["nodes"]} == set("ABCDEFGHIJKLMN")
    node_ids = {row["node_id"] for row in payload["nodes"]}
    assert {
        "accuracy_finalist_selector",
        "rejection_finalist_selector",
        "scale_graph_training",
        "sealed_final_test",
    } <= node_ids
    assert payload["degradation_profile"] == "D_NOMINAL"
    assert payload["bounded_concurrency"] == DEFAULT_CONCURRENCY


def test_submission_graph_cli_resolves_real_offline_scope() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "submit_retb_graph.py"),
            "--dry-run",
            "--submission-scope",
            "offline_abc",
            "--storage-measurements-sha256",
            "c" * 64,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["submission_scope"] == "offline_abc"
    assert (
        payload["submission_node_count"]
        < payload["complete_graph_node_count"]
    )
    by_id = {row["node_id"]: row for row in payload["nodes"]}
    assert {
        by_id[node_id]["stage"] for node_id in payload["submission_node_ids"]
    } == set("ABC")
    assert "step6_native_hlt_contracts" not in payload["submission_node_ids"]
