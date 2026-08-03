from __future__ import annotations

from pathlib import Path
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from teacher_logit_reco.hlt_offline_structure_distillation import (
    AuthenticatedTreeSplit,
    compatible_artifact_content_hashes,
    LoadedTargetCache,
    fit_sharded_target_normalizer,
    build_campaign_monitor,
    build_full_authorization,
    build_miniature_acceptance,
    build_miniature_check_receipt,
    build_production_execution_plan,
    build_node_factory_registry,
    build_registered_command_matrix,
    build_runtime_manifest,
    build_resource_measurements,
    build_stage_job_registry,
    REQUIRED_INFRASTRUCTURE_OPTIONS_BY_NODE,
    REQUIRED_INFRASTRUCTURE_OPTION_MIN_COUNTS,
    REQUIRED_INFRASTRUCTURE_OPTION_KEYS,
    DIRECTORY_INFRASTRUCTURE_OPTIONS,
    NODE_COORDINATE_LIMITS,
    resolve_tree_parent_lineage,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (
    canonical_sha256,
    with_content_hash,
)
from teacher_logit_reco.hlt_offline_structure_distillation.node_runtime import (
    _row_scoped_arguments,
    resolve_node_argv,
)
from teacher_logit_reco.relational_part import (
    build_reference_tree,
    finalize_tree_split,
    write_tree_shard,
)


SOURCE = {
    "commit": "a" * 40,
    "status_sha256": "b" * 64,
    "dirty": True,
    "status_hash_policy": "test",
}
REPO_ROOT = Path(__file__).resolve().parents[1]


def _ready_arguments():
    output = {}
    for node_id, options in REQUIRED_INFRASTRUCTURE_OPTIONS_BY_NODE.items():
        values = []
        minimums = REQUIRED_INFRASTRUCTURE_OPTION_MIN_COUNTS.get(
            node_id, {}
        )
        for option in options:
            keyed = sorted(
                REQUIRED_INFRASTRUCTURE_OPTION_KEYS.get(node_id, {}).get(
                    option, ()
                )
            )
            count = max(minimums.get(option, 1), len(keyed))
            for index in range(count):
                payload = (
                    (
                        f"{keyed[index]}="
                        f"{{{'directory' if option in DIRECTORY_INFRASTRUCTURE_OPTIONS else 'file'}_test}}"
                    )
                    if index < len(keyed)
                    else (
                        "1000000"
                        if option == "--available-storage-bytes"
                        else "32"
                        if option == "--production-batch-size"
                        else "locked_measured_clocks"
                        if option == "--clock-power-mode"
                        else f"{{{'directory' if option in DIRECTORY_INFRASTRUCTURE_OPTIONS else 'file'}_test}}"
                    )
                )
                values.extend(
                    [option, payload]
                )
        output[node_id] = values
    return output


def _runtime_bindings(file_path=None):
    return (
        {"test": file_path or Path(__file__).resolve()},
        {"test": REPO_ROOT},
    )


def _load_submit_module():
    path = REPO_ROOT / "scripts" / "submit_hosd_slurm.py"
    spec = importlib.util.spec_from_file_location("submit_hosd_slurm", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_resource_measurement_module():
    path = REPO_ROOT / "scripts" / "measure_hosd_miniature_resources.py"
    spec = importlib.util.spec_from_file_location(
        "measure_hosd_miniature_resources", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan(profile, *, production_tree_unit=10_000, production_target_unit=2_048):
    registry = build_stage_job_registry(source=SOURCE)
    files, directories = _runtime_bindings()
    runtime = build_runtime_manifest(
        campaign_spec_sha256="c" * 64,
        files=files,
        directories=directories,
        infrastructure_arguments_by_node=_ready_arguments(),
        source=SOURCE,
    )
    factories = build_node_factory_registry(
        stage_job_registry=registry,
        source=SOURCE,
        execution_profile=profile,
    )
    commands = build_registered_command_matrix(
        stage_job_registry=registry,
        factory_registry=factories,
        runtime_manifest=runtime,
        campaign_root="/authenticated/campaign",
    )
    layout_ledger = with_content_hash(
        {
            "contract": "hosd_scale_resident_layout_ledger_v7",
            "source_sha256": canonical_sha256(SOURCE),
            "scale_execution_plan_sha256": "0" * 64,
            "production_tree_shard_events": production_tree_unit,
            "production_target_shard_events": production_target_unit,
            "production_tree_shard_decoded_bytes_upper_bound": 1,
            "production_target_shard_decoded_bytes_upper_bound": 1,
            "active_completion_hashes": {
                "scale_inputs": "1" * 64,
                "scale_trees": "2" * 64,
                "scale_targets": "3" * 64,
                "scale_teacher_outputs": "4" * 64,
                "scale_native_relations": "5" * 64,
            },
            "scale_graph_store_multiplicities": [
                {
                    "graph_id": "G_WORST",
                    "hlt_replica_input_store_instances": 4,
                    "graph_global_resident_tree_shards": 1,
                    "resident_target_shard_budget": 1,
                    "target_store_instances": 8,
                    "production_resident_bytes": 1,
                }
            ],
            "worst_case_scale_graph": {
                "graph_id": "G_WORST",
                "hlt_replica_input_store_instances": 4,
                "graph_global_resident_tree_shards": 1,
                "resident_target_shard_budget": 1,
                "target_store_instances": 8,
                "production_resident_bytes": 1,
            },
            "production_byte_accounting": {"scale_graph_train": 1},
            "scale_training_sampler_contract": "hosd_scale_shard_aware_sampler_v1",
            "scale_sampler_maximum_locality_window_events": 2_048,
            "scale_decode_complexity_contract": "O(locality_segments_times_static_target_coordinates_plus_tree_replica_groups)",
            "test_layout": True,
        }
    )
    projections = {}
    for node_id in (
        "scale_input_prepare",
        "scale_tree_build",
        "scale_target_build",
        "scale_teacher_target_inference",
        "scale_graph_train",
    ):
        population_model = node_id == "scale_input_prepare"
        resource = "gpu" if node_id in {
            "scale_graph_train", "scale_teacher_target_inference"
        } else "cpu"
        limit = (220 if resource == "gpu" else 192) * 1024**3
        unit_bytes = 1024
        fixed = 4096
        projected = fixed + unit_bytes * (3_000_000 if population_model else 1)
        projections[node_id] = with_content_hash(
            {
                "contract": "hosd_scale_resident_memory_projection_v7",
                "pilot_node_id": node_id,
                "pilot_job_id": "123",
                "coordinate_type": f"test_{node_id}",
                "resource_class": resource,
                "source_sha256": canonical_sha256(SOURCE),
                "layout_evidence_sha256": layout_ledger["content_hash"],
                "fixed_resident_bytes": fixed,
                "miniature_resident_unit_events": 7,
                "miniature_resident_unit_bytes_upper_bound": unit_bytes,
                "production_resident_unit_events": (
                    1
                    if node_id == "scale_input_prepare"
                    else production_target_unit
                    if node_id in {"scale_target_build", "scale_teacher_target_inference"}
                    else production_tree_unit
                ),
                "production_resident_unit_bytes_upper_bound": unit_bytes,
                "production_population": 3_000_000,
                "projected_resident_bytes": projected,
                "projection_model": (
                    "fixed_plus_authenticated_per_event_population_v2"
                    if population_model
                    else "fixed_plus_authenticated_single_shard_v2"
                ),
                "loader_storage_contracts": ["hosd_npy_mmap_store_v2"],
                "registered_tigris_limit_bytes": limit,
                "within_registered_tigris_limit": True,
                "representative_real_task_completed": True,
            }
        )
    resources = (
        build_resource_measurements(
            miniature_execution_plan_sha256="1" * 64,
            scheduler_evidence_sha256="2" * 64,
            requests_by_class={
                "cpu": {
                    "partition": "tigris",
                    "cpus": 8,
                    "memory": "96G",
                    "time": "12:00:00",
                    "gres": None,
                },
                "gpu": {
                    "partition": "tigris",
                    "cpus": 8,
                    "memory": "128G",
                    "time": "1-00:00:00",
                    "gres": "gpu:gh200:1",
                },
            },
            projected_target_extraction_seconds=3600,
            projected_gpu_hours_by_stage={"C": 10.0},
            maximum_concurrent_jobs=8,
            checkpoint_bytes=1024,
            export_bytes=512,
            scale_resident_layout_ledger=layout_ledger,
            scale_resident_memory_projections=projections,
            measurement_evidence_sha256="3" * 64,
            source=SOURCE,
        )
        if profile == "production_500k_scale3m"
        else None
    )
    return build_production_execution_plan(
        stage_job_registry=registry,
        commands_by_node=commands,
        campaign_spec_sha256="c" * 64,
        source=SOURCE,
        profile=profile,
        resource_measurements=resources,
        node_factory_registry=factories,
        runtime_manifest=runtime,
    )


def test_execution_plan_covers_every_stage_node_and_never_performance_stops():
    plan = _plan("production_500k_scale3m")
    assert plan["node_count"] == len(plan["nodes"])
    assert plan["all_stage_nodes_have_commands"]
    assert plan["dependency_policy"] == "afterok_integrity_only"
    assert plan["performance_based_termination"] is False
    assert plan["negative_result_continuation"] is True
    incomplete_runtime = build_runtime_manifest(
        campaign_spec_sha256="c" * 64,
        files={},
        directories={},
        infrastructure_arguments_by_node={},
        source=SOURCE,
    )
    assert incomplete_runtime["execution_ready"] is False
    with pytest.raises(ValueError, match="runtime manifest is incomplete"):
        registry = build_stage_job_registry(source=SOURCE)
        factories = build_node_factory_registry(
            stage_job_registry=registry,
            source=SOURCE,
            execution_profile="miniature_test",
        )
        build_production_execution_plan(
            stage_job_registry=registry,
            commands_by_node=build_registered_command_matrix(
                stage_job_registry=registry,
                factory_registry=factories,
                runtime_manifest=incomplete_runtime,
                campaign_root="/authenticated/campaign",
            ),
            campaign_spec_sha256="c" * 64,
            source=SOURCE,
            profile="miniature_test",
            node_factory_registry=factories,
            runtime_manifest=incomplete_runtime,
        )


def test_miniature_batches_every_logical_coordinate_into_bounded_allocations():
    registry = build_stage_job_registry(source=SOURCE)
    production = build_node_factory_registry(
        stage_job_registry=registry,
        source=SOURCE,
        execution_profile="production_500k_scale3m",
    )
    miniature = build_node_factory_registry(
        stage_job_registry=registry,
        source=SOURCE,
        execution_profile="miniature_test",
    )
    assert sum(row["coordinate_limit"] for row in miniature["entries"]) == 1251
    assert sum(row["scheduled_coordinate_count"] for row in miniature["entries"]) == 139
    assert all(row["coordinate_span"] == 1 for row in production["entries"])
    assert all(
        1 <= row["coordinate_span"] <= 16 for row in miniature["entries"]
    )
    runtime = _ready_runtime()
    commands = build_registered_command_matrix(
        stage_job_registry=registry,
        factory_registry=miniature,
        runtime_manifest=runtime,
        campaign_root="/authenticated/campaign",
    )
    probe_commands = commands["probe_input_materialization"]
    assert len(probe_commands) == 11
    assert probe_commands[0][-4:] == [
        "--coordinate-start",
        "0",
        "--coordinate-stop",
        "16",
    ]
    assert probe_commands[-1][-4:] == [
        "--coordinate-start",
        "160",
        "--coordinate-stop",
        "162",
    ]


def test_production_walltime_is_measured_projected_and_policy_bounded():
    module = _load_resource_measurement_module()
    requests = module.derive_requests_from_measurements(
        maximum_projected_seconds_by_class={"cpu": 3600, "gpu": 7200},
        maximum_rss_bytes_by_class={
            "cpu": 8 * 1024**3,
            "gpu": 40 * 1024**3,
        },
    )
    assert requests["cpu"]["time"] == "01:30:00"
    assert requests["gpu"]["time"] == "03:00:00"
    assert requests["cpu"]["memory"] == "12G"
    assert requests["gpu"]["memory"] == "60G"
    with pytest.raises(ValueError, match="walltime"):
        module.derive_requests_from_measurements(
            maximum_projected_seconds_by_class={
                "cpu": 24 * 3600,
                "gpu": 3600,
            },
            maximum_rss_bytes_by_class={"cpu": 1, "gpu": 1},
        )


def test_worst_scale_graph_multiplicities_cover_eight_members_and_residual_replicas():
    module = _load_resource_measurement_module()
    members = [
        {
            "target_id": "T_HLT_REGION_PAIR_8",
            "parameterization": "ABS",
        },
        *[
            {
                "target_id": f"T_OFFLINE_RESIDUAL_{index}",
                "parameterization": "RES",
            }
            for index in range(6)
        ],
        {
            "target_id": "T_OFFLINE_LOGITS_O_BASE",
            "parameterization": "KD",
        },
    ]
    rows = module._scale_graph_store_multiplicities(
        {
            "graph_definitions": [
                {
                    "graph_id": "C_ALL_BEST",
                    "graph_definition": {
                        "graph_kind": "COMBINATION",
                        "graph": {
                            "members": members,
                            "native_relation_auxiliary": {"weight": 1.0},
                        },
                    },
                }
            ]
        }
    )
    assert rows == [
        {
            "graph_id": "C_ALL_BEST",
            "simultaneous_member_count": 8,
            "hlt_replica_input_store_instances": 4,
            "tree_store_instances": 4,
            "graph_global_resident_tree_shards": 1,
            "target_store_instances": 25,
            "resident_target_shard_budget": 7,
            "native_relation_mmap_store_instances": 4,
            "kd_logit_mmap_store_instances": 1,
        }
    ]


def test_standalone_residual_graph_budgets_one_resident_of_four_stores():
    module = _load_resource_measurement_module()
    rows = module._scale_graph_store_multiplicities(
        {
            "graph_definitions": [
                {
                    "graph_id": "AUX_RES",
                    "graph_definition": {
                        "graph_kind": "AUXILIARY",
                        "row": {
                            "target_id": "T_OFFLINE_TRACK_32",
                            "parameterization": "RES",
                        },
                    },
                }
            ]
        }
    )
    assert rows[0]["target_store_instances"] == 4
    assert rows[0]["resident_target_shard_budget"] == 1


def test_resource_authorization_requires_every_stage_j_memory_projection():
    ledger = with_content_hash(
        {
            "contract": "hosd_scale_resident_layout_ledger_v7",
            "source_sha256": canonical_sha256(SOURCE),
            "scale_execution_plan_sha256": "0" * 64,
            "production_tree_shard_events": 10_000,
            "production_target_shard_events": 2_048,
            "production_tree_shard_decoded_bytes_upper_bound": 1,
            "production_target_shard_decoded_bytes_upper_bound": 1,
            "active_completion_hashes": {
                "scale_inputs": "1" * 64,
                "scale_trees": "2" * 64,
                "scale_targets": "3" * 64,
                "scale_teacher_outputs": "4" * 64,
                "scale_native_relations": "5" * 64,
            },
            "scale_graph_store_multiplicities": [
                {
                    "graph_id": "G_WORST",
                    "hlt_replica_input_store_instances": 4,
                    "graph_global_resident_tree_shards": 1,
                    "resident_target_shard_budget": 1,
                    "target_store_instances": 8,
                    "production_resident_bytes": 1,
                }
            ],
            "worst_case_scale_graph": {
                "graph_id": "G_WORST",
                "hlt_replica_input_store_instances": 4,
                "graph_global_resident_tree_shards": 1,
                "resident_target_shard_budget": 1,
                "target_store_instances": 8,
                "production_resident_bytes": 1,
            },
            "production_byte_accounting": {"scale_graph_train": 1},
            "scale_training_sampler_contract": "hosd_scale_shard_aware_sampler_v1",
            "scale_sampler_maximum_locality_window_events": 2_048,
            "scale_decode_complexity_contract": "O(locality_segments_times_static_target_coordinates_plus_tree_replica_groups)",
            "test_layout": True,
        }
    )
    with pytest.raises(ValueError, match="projection coverage"):
        build_resource_measurements(
            miniature_execution_plan_sha256="1" * 64,
            scheduler_evidence_sha256="2" * 64,
            requests_by_class={
                "cpu": {
                    "partition": "tigris",
                    "cpus": 8,
                    "memory": "96G",
                    "time": "12:00:00",
                    "gres": None,
                },
                "gpu": {
                    "partition": "tigris",
                    "cpus": 8,
                    "memory": "128G",
                    "time": "1-00:00:00",
                    "gres": "gpu:gh200:1",
                },
            },
            projected_target_extraction_seconds=1,
            projected_gpu_hours_by_stage={"J": 1.0},
            maximum_concurrent_jobs=1,
            checkpoint_bytes=1,
            export_bytes=1,
            scale_resident_layout_ledger=ledger,
            scale_resident_memory_projections={},
            measurement_evidence_sha256="3" * 64,
            source=SOURCE,
        )


def test_small_miniature_cannot_shrink_production_resident_shard_units():
    with pytest.raises(ValueError, match="layout source differs"):
        _plan(
            "production_500k_scale3m",
            production_tree_unit=7,
            production_target_unit=7,
        )


def test_resource_layout_accounting_uses_authenticated_array_shapes(tmp_path):
    module = _load_resource_measurement_module()
    path = tmp_path / "layout.npz"
    first = np.zeros((7, 3), dtype=np.float32)
    second = np.zeros((7, 2), dtype=np.int16)
    np.savez_compressed(path, first=first, second=second)
    assert module._npz_layout_bytes(path) == first.nbytes + second.nbytes


def test_production_execution_plan_requires_command_coverage():
    with pytest.raises(ValueError, match="coverage"):
        registry = build_stage_job_registry(source=SOURCE)
        build_production_execution_plan(
            stage_job_registry=registry,
            commands_by_node={},
            campaign_spec_sha256="c" * 64,
            source=SOURCE,
            profile="production_500k_scale3m",
            node_factory_registry=build_node_factory_registry(
                stage_job_registry=registry, source=SOURCE
            ),
            runtime_manifest=_ready_runtime(),
        )


def _ready_runtime():
    files, directories = _runtime_bindings()
    return build_runtime_manifest(
        campaign_spec_sha256="c" * 64,
        files=files,
        directories=directories,
        infrastructure_arguments_by_node=_ready_arguments(),
        source=SOURCE,
    )


def test_submitter_topologically_orders_same_stage_forward_dependencies():
    submit = _load_submit_module()
    nodes = [
        {"node_id": "consumer", "dependencies": ["producer"]},
        {"node_id": "independent", "dependencies": []},
        {"node_id": "producer", "dependencies": []},
    ]
    ordered = submit._topological_nodes(nodes)
    positions = {row["node_id"]: index for index, row in enumerate(ordered)}
    assert positions["producer"] < positions["consumer"]
    with pytest.raises(ValueError, match="cyclic"):
        submit._topological_nodes(
            [
                {"node_id": "a", "dependencies": ["b"]},
                {"node_id": "b", "dependencies": ["a"]},
            ]
        )


def test_miniature_resume_does_not_require_impossible_full_authorization():
    submit = _load_submit_module()
    args = SimpleNamespace(
        smoke_submit=False,
        dry_run=False,
        resume_submit=True,
    )
    profile, mode, authorization = submit._submission_profile_and_mode(
        args, {"profile": "miniature_test"}
    )
    assert (profile, mode, authorization) == (
        "miniature_test",
        "resume_submit",
        False,
    )
    profile, mode, authorization = submit._submission_profile_and_mode(
        args, {"profile": "production_500k_scale3m"}
    )
    assert authorization is True


def test_submitter_accepts_a_frozen_campaign_source_root_for_launcher_recovery(
    tmp_path,
):
    submit = _load_submit_module()
    source_root = tmp_path / "frozen-source"
    args = submit._parser().parse_args(
        [
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--smoke-submit",
            "--attempt",
            "2",
            "--campaign-source-root",
            str(source_root),
        ]
    )
    assert args.campaign_source_root == source_root
    assert args.attempt == 2


def test_registered_runner_routes_parent_lock_through_idempotent_launcher(
    tmp_path, monkeypatch
):
    path = REPO_ROOT / "scripts" / "run_hosd_registered_node.py"
    spec = importlib.util.spec_from_file_location("hosd_registered_recovery", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    campaign_root = tmp_path / "campaign"
    source_root = tmp_path / "frozen-source"
    campaign = {"source": SOURCE}
    plan = {"source": SOURCE}
    monkeypatch.setattr(
        module, "load_and_validate_campaign", lambda *_args, **_kwargs: campaign
    )
    monkeypatch.setattr(module, "load_hashed_json", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(
        module,
        "node_execution",
        lambda *_args, **_kwargs: {"commands": [["python", "stale-lock.py"]]},
    )
    calls = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: (
            calls.append((list(command), kwargs))
            or SimpleNamespace(returncode=0)
        ),
    )
    assert (
        module.main(
            [
                "--campaign-root",
                str(campaign_root),
                "--campaign-source-root",
                str(source_root),
                "--node-id",
                "resolved_parent_lock",
            ]
        )
        == 0
    )
    command, kwargs = calls[0]
    assert Path(command[2]).name == "lock_hosd_inherited_parents.py"
    assert command[-2:] == ["--campaign-source-root", str(source_root.resolve())]
    assert kwargs["cwd"] == source_root.resolve()


def test_monitor_repairs_runtime_failures_not_negative_science():
    plan = _plan("miniature_test")
    states = {
        row["node_id"]: {"state": "completed"} for row in plan["nodes"]
    }
    validity = {row["node_id"]: True for row in plan["nodes"]}
    failed = plan["nodes"][5]["node_id"]
    states[failed] = {"state": "timeout"}
    dependent = next(
        row["node_id"]
        for row in plan["nodes"]
        if failed in row["dependencies"]
    )
    states[dependent] = {"state": "pending"}
    validity[dependent] = False
    monitor = build_campaign_monitor(
        execution_plan=plan,
        node_states=states,
        artifact_validity=validity,
        source=SOURCE,
    )
    assert monitor["restart_nodes"] == [failed]
    assert failed in monitor["recovery_submission_nodes"]
    assert dependent in monitor["recovery_submission_nodes"]
    assert monitor["performance_based_cancellation_detected"] is False
    assert all(not row["scientific_result_sign_used"] for row in monitor["nodes"])


def test_real_miniature_is_required_before_full_authorization():
    miniature_plan = _plan("miniature_test")
    check_ids = (
            "real_jetclass_hlt_v3",
            "all_current_family_groups",
            "baseline_probe_aux_feedback_trained",
            "all_negative_selector_continued",
            "target_shard_interrupt_resume",
            "training_row_interrupt_resume",
            "hlt_only_export_validated",
            "stack_and_two_locks_traversed",
            "no_manual_artifact_injection",
    )
    receipts = [
        build_miniature_check_receipt(
            execution_plan=miniature_plan,
            check_id=key,
            evidence_hashes={"run": f"{index + 1:x}"[-1] * 64},
            verifier="hosd_miniature_semantic_verifier_v1",
            verifier_source_sha256="f" * 64,
            verified_predicates={"semantic_evidence_checked": True},
            source=SOURCE,
        )
        for index, key in enumerate(check_ids)
    ]
    with pytest.raises(ValueError, match="incomplete"):
        build_miniature_acceptance(
            execution_plan=miniature_plan,
            check_receipts=receipts[:-1],
            source=SOURCE,
        )
    acceptance = build_miniature_acceptance(
        execution_plan=miniature_plan,
        check_receipts=receipts,
        source=SOURCE,
    )
    production = _plan("production_500k_scale3m")
    preflight = with_content_hash(
        {
                "contract": "hosd_resource_preflight_v11",
                "schema_version": 11,
            "source": SOURCE,
            "profile": "production_500k_scale3m",
            "storage_measurements_sha256": "4" * 64,
            "resource_measurements_sha256": production[
                "resource_measurements_sha256"
            ],
            "runtime_ready": True,
        }
    )
    authorization = build_full_authorization(
        production_plan=production,
        miniature_acceptance=acceptance,
        resource_preflight=preflight,
        source=SOURCE,
    )
    assert authorization["full_campaign_submission_authorized"]
    assert authorization["scientific_underperformance_can_stop_campaign"] is False
    assert authorization["storage_measurements_sha256"] == "4" * 64
    assert production["resources_are_authenticated_miniature_measurements"]


def test_all_required_slurm_entrypoints_are_present_and_fail_closed():
    names = (
        "hosd_common.sh",
        "run_hosd_bootstrap.sh",
        "run_hosd_hlt_cache.sh",
        "run_hosd_tree_shards.sh",
        "run_hosd_tree_finalize.sh",
        "run_hosd_relation_normalization.sh",
        "run_hosd_target_build.sh",
        "run_hosd_teacher_wave.sh",
        "run_hosd_baseline_array.sh",
        "run_hosd_probe_array.sh",
        "run_hosd_auxiliary_array.sh",
        "run_hosd_feedback_array.sh",
        "run_hosd_combination_array.sh",
        "run_hosd_controls_array.sh",
        "run_hosd_robustness_array.sh",
        "run_hosd_confirmation_array.sh",
        "run_hosd_scale_array.sh",
        "run_hosd_stack_val.sh",
        "run_hosd_final_test.sh",
        "run_hosd_registered_node.sh",
        "submit_hosd_tigris_full.sh",
    )
    for name in names:
        text = (REPO_ROOT / "sbatch" / name).read_text(encoding="utf-8")
        assert "set -euo pipefail" in text
    submit = (REPO_ROOT / "sbatch" / "submit_hosd_tigris_full.sh").read_text(
        encoding="utf-8"
    )
    assert "submit_hosd_slurm.py" in submit
    workers = sorted((REPO_ROOT / "sbatch").glob("run_hosd_*.sh"))
    assert workers
    for worker in workers:
        text = worker.read_text(encoding="utf-8")
        assert "BASH_SOURCE" not in text
        assert ': "${PROJECT_DIR:?PROJECT_DIR is required}"' in text
        assert ': "${HOSD_LAUNCHER_ROOT:=${PROJECT_DIR}}"' in text
        assert 'source "${HOSD_LAUNCHER_ROOT}/sbatch/hosd_common.sh"' in text
    submitter = (REPO_ROOT / "scripts" / "submit_hosd_slurm.py").read_text(
        encoding="utf-8"
    )
    assert (
        'f"PROJECT_DIR={campaign_source_root},HOSD_LAUNCHER_ROOT={REPO_ROOT},"'
        in submitter
    )
    assert "repo_root=campaign_source_root" in submitter
    common = (REPO_ROOT / "sbatch" / "hosd_common.sh").read_text(encoding="utf-8")
    assert "PYTHONNOUSERSITE=1" in common
    assert "load_and_validate_campaign" in common
    compiler = (
        REPO_ROOT / "scripts" / "build_hosd_execution_plan.py"
    ).read_text(encoding="utf-8")
    assert "--commands-json" not in compiler
    assert "build_registered_command_matrix" in compiler


def test_stage_j_builds_scale_inputs_trees_normalizers_and_adapters_in_dag():
    nodes = {
        row["node_id"]: row for row in build_stage_job_registry(source=SOURCE)[
            "nodes"
        ]
    }
    expected_chain = (
        ("scale_input_prepare", ("scale_plan_compile",)),
        ("scale_tree_build", ("scale_input_prepare",)),
        ("scale_normalization", ("scale_tree_build",)),
        ("scale_teacher_train", ("scale_normalization",)),
        ("scale_teacher_lock", ("scale_teacher_train",)),
        ("scale_teacher_adapter_compile", ("scale_teacher_lock",)),
        (
            "scale_teacher_target_inference",
            ("scale_teacher_adapter_compile",),
        ),
        ("scale_target_build", ("scale_teacher_target_inference",)),
        ("scale_native_relation_build", ("scale_target_build",)),
        ("scale_graph_train", ("scale_native_relation_build",)),
        ("scale_finalize", ("scale_graph_train",)),
    )
    for node_id, dependencies in expected_chain:
        assert tuple(nodes[node_id]["dependencies"]) == dependencies
    assert NODE_COORDINATE_LIMITS["scale_input_prepare"] == 5
    assert NODE_COORDINATE_LIMITS["scale_tree_build"] == 5
    assert NODE_COORDINATE_LIMITS["scale_native_relation_build"] == 4
    assert "scale_input_prepare" not in REQUIRED_INFRASTRUCTURE_OPTIONS_BY_NODE
    # These are generated by upstream Stage-J jobs and must never be requested
    # as manually injected future artifacts in the runtime manifest.
    assert "scale_target_build" not in REQUIRED_INFRASTRUCTURE_OPTIONS_BY_NODE
    scale_graph_external = set(
        REQUIRED_INFRASTRUCTURE_OPTIONS_BY_NODE.get("scale_graph_train", ())
    )
    assert not {
        "--scale-train-cache",
        "--scale-train-tree",
        "--scale-relation-normalizer",
        "--teacher-adapter-config",
    } & scale_graph_external
    for path in (
        "scripts/prepare_hosd_scale_inputs.py",
        "scripts/build_hosd_scale_tree.py",
        "scripts/fit_hosd_scale_normalizers.py",
        "scripts/compile_hosd_scale_teacher_adapters.py",
        "scripts/build_hosd_scale_native_relations.py",
    ):
        assert (REPO_ROOT / path).is_file()


def test_dynamic_stage_plans_are_published_by_completed_predecessors():
    nodes = {
        row["node_id"]: row
        for row in build_stage_job_registry(source=SOURCE)["nodes"]
    }
    assert "job_ledgers/stage_d_execution_plan.json" in nodes[
        "predictability_aggregate"
    ]["outputs"]
    assert "job_ledgers/stage_e_execution_plan.json" in nodes[
        "single_family_select"
    ]["outputs"]
    assert "job_ledgers/stage_f_execution_plan.json" in nodes[
        "feedback_select"
    ]["outputs"]

    source_requirements = {
        "scripts/aggregate_hosd_predictability.py": (
            "build_stage_d_plan",
            "stage_d_execution_plan.json",
        ),
        "scripts/select_hosd_single_targets.py": (
            "build_stage_e_plan",
            "stage_e_execution_plan.json",
        ),
        "scripts/select_hosd_feedback.py": (
            "build_stage_f_plan",
            "stage_f_execution_plan.json",
        ),
    }
    for relative, required in source_requirements.items():
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        for token in required:
            assert token in source


def test_stage_d_boundary_contracts_cover_current_wave_and_design_subroles():
    trainer = (REPO_ROOT / "scripts" / "train_hosd_auxiliary.py").read_text(
        encoding="utf-8"
    )
    assert "AUXILIARY_COMPLETION_CONTRACT" in trainer
    assert 'expected_contract="hosd_auxiliary_completion_v1"' not in trainer

    runtime = (
        REPO_ROOT
        / "teacher_logit_reco"
        / "hlt_offline_structure_distillation"
        / "node_runtime.py"
    ).read_text(encoding="utf-8")
    controls = (
        REPO_ROOT / "scripts" / "execute_hosd_control_wave.py"
    ).read_text(encoding="utf-8")
    for role in ("design_select", "design_confirm"):
        assert role in runtime
        assert role in controls
    assert "hosd_target_control_wave_v2" in (
        REPO_ROOT
        / "teacher_logit_reco"
        / "hlt_offline_structure_distillation"
        / "contracts.py"
    ).read_text(encoding="utf-8")


def test_stage_j_tree_and_target_producers_are_bounded_resident():
    tree_source = (REPO_ROOT / "scripts" / "build_hosd_scale_tree.py").read_text(
        encoding="utf-8"
    )
    target_source = (REPO_ROOT / "scripts" / "build_hosd_targets.py").read_text(
        encoding="utf-8"
    )
    teacher_target_source = (
        REPO_ROOT / "scripts" / "infer_hosd_teacher_targets.py"
    ).read_text(encoding="utf-8")
    cache_source = (
        REPO_ROOT
        / "teacher_logit_reco"
        / "hlt_offline_structure_distillation"
        / "target_cache.py"
    ).read_text(encoding="utf-8")
    assert "load_materialized_input_view(" in tree_source
    assert "np.load(input_path" not in tree_source
    assert "load_materialized_input_view(" in target_source
    assert "tree_by_identity" not in target_source
    assert "publish_target_cache_shard(" in target_source
    assert "canonicalize_identities(" in target_source
    assert "canonical_to_source=canonical_to_source" in target_source
    assert "raw_tokens = raw_tokens[canonical_to_source]" not in target_source
    assert "mask = mask[canonical_to_source]" not in target_source
    assert "identities_are_canonical=False" in teacher_target_source
    assert "publish_target_cache(" in teacher_target_source
    normalization_source = (
        REPO_ROOT / "scripts" / "execute_hosd_normalization_wave.py"
    ).read_text(encoding="utf-8")
    statistics_source = (
        REPO_ROOT / "scripts" / "fit_hosd_teacher_statistics.py"
    ).read_text(encoding="utf-8")
    assert "align_conditional_context_to_cache(" in normalization_source
    assert "align_conditional_context_to_cache(" in statistics_source
    assert "raw = raw[" not in normalization_source
    assert "mask = mask[" not in normalization_source
    assert "all_identities" not in cache_source
    teacher_source = (
        REPO_ROOT
        / "teacher_logit_reco"
        / "hlt_offline_structure_distillation"
        / "teacher_inference_runtime.py"
    ).read_text(encoding="utf-8")
    scale_source = (REPO_ROOT / "scripts" / "execute_hosd_scale_row.py").read_text(
        encoding="utf-8"
    )
    residual_source = (
        REPO_ROOT / "scripts" / "build_hosd_target_derivatives.py"
    ).read_text(encoding="utf-8")
    pair_source = (
        REPO_ROOT
        / "teacher_logit_reco"
        / "hlt_offline_structure_distillation"
        / "scale_runtime.py"
    ).read_text(encoding="utf-8")
    assert "tree_by_identity" not in teacher_source
    assert "load_materialized_input_view(" in teacher_source
    assert "load_target_cache_sharded" in scale_source
    assert "np.concatenate(" not in scale_source
    assert "load_target_cache_sharded" in residual_source
    assert "publish_target_cache_shard(" in residual_source
    assert "_tree_lookup" not in pair_source


def test_source_bound_tree_parent_alias_authenticates_raw_builder_hash():
    raw_backend = with_content_hash(
        {"contract": "test_backend", "schema_version": 1, "compiler": "gcc"}
    )
    bound_backend = with_content_hash(
        {
            "contract": "test_backend",
            "schema_version": 1,
            "compiler": "gcc",
            "source": SOURCE,
        }
    )
    resource = with_content_hash(
        {"contract": "test_resource", "schema_version": 1, "source": SOURCE}
    )
    assert raw_backend["content_hash"] in compatible_artifact_content_hashes(
        bound_backend
    )
    parents = {
        "hlt_content_sha256": "a" * 64,
        "tree_resource_sha256": resource["content_hash"],
        "backend_manifest_sha256": raw_backend["content_hash"],
    }
    assert resolve_tree_parent_lineage(
        parents,
        hlt_content_sha256="a" * 64,
        tree_resource=resource,
        tree_backend=bound_backend,
    ) == parents
    with pytest.raises(ValueError, match="backend parent differs"):
        resolve_tree_parent_lineage(
            {**parents, "backend_manifest_sha256": "f" * 64},
            hlt_content_sha256="a" * 64,
            tree_resource=resource,
            tree_backend=bound_backend,
        )


def test_all_tree_consumers_use_shared_fail_closed_split_authentication(
    tmp_path,
):
    # Tree shards preserve the authenticated source order; that order is not
    # required to be lexicographic.  Consumers resolve their requested role
    # identities explicitly instead of silently treating source order as a
    # canonical sort order.
    identities = ["jet-b", "jet-a"]
    trees = []
    for offset in (0.0, 0.2):
        tokens = np.zeros((4, 14), dtype=np.float32)
        tokens[:2, 0] = (2.0, 1.0)
        tokens[:2, 1] = (offset, offset + 0.1)
        tokens[:2, 2] = (0.0, 0.1)
        tokens[:2, 3] = (2.2, 1.2)
        mask = np.asarray([True, True, False, False])
        vectors = np.zeros((4, 4), dtype=np.float32)
        vectors[:2] = np.asarray(
            [[2.0, 0.0, 0.0, 2.2], [0.995, 0.1, 0.1, 1.2]],
            dtype=np.float32,
        )
        trees.append(build_reference_tree(vectors, tokens, mask))
    shard = tmp_path / "tree" / "shards" / "shard_00000.npz"
    write_tree_shard(
        shard,
        trees,
        identities,
        hlt_content_sha256="a" * 64,
        tree_resource_sha256="b" * 64,
        backend_manifest_sha256="c" * 64,
    )
    finalize_tree_split(
        tmp_path / "tree" / "manifest.json",
        [shard.with_suffix(".metadata.json")],
        split="scale_train",
        expected_jet_count=2,
        hlt_content_sha256="a" * 64,
        tree_resource_sha256="b" * 64,
        backend_manifest_sha256="c" * 64,
    )
    authenticated = AuthenticatedTreeSplit(
        tmp_path / "tree",
        expected_identities=identities,
        expected_parents={
            "hlt_content_sha256": "a" * 64,
            "tree_resource_sha256": "b" * 64,
            "backend_manifest_sha256": "c" * 64,
        },
    )
    assert len(authenticated.load_shard(0)[1]) == 2
    assert len(
        authenticated.load_event_rows([1], expected_identities=["jet-a"])
    ) == 1
    selected = authenticated.event_indices_for_identities(["jet-a", "jet-b"])
    assert selected.tolist() == [1, 0]
    assert len(
        authenticated.load_event_rows(
            selected, expected_identities=["jet-a", "jet-b"]
        )
    ) == 2
    with pytest.raises(ValueError, match="parents differ"):
        AuthenticatedTreeSplit(
            tmp_path / "tree",
            expected_identities=identities,
            expected_parents={
                "hlt_content_sha256": "d" * 64,
                "tree_resource_sha256": "b" * 64,
                "backend_manifest_sha256": "c" * 64,
            },
        )

    original = shard.read_bytes()
    shard.write_bytes(original + b"drift")
    with pytest.raises(ValueError, match="changed after split authentication"):
        authenticated.load_shard(0)
    with pytest.raises(ValueError, match="shard attestation differs"):
        AuthenticatedTreeSplit(tmp_path / "tree")
    shard.write_bytes(original)

    target_source = (REPO_ROOT / "scripts" / "build_hosd_targets.py").read_text(
        encoding="utf-8"
    )
    graph_source = (
        REPO_ROOT
        / "teacher_logit_reco"
        / "hlt_offline_structure_distillation"
        / "stage_d_data_factory.py"
    ).read_text(encoding="utf-8")
    measurement_source = (
        REPO_ROOT / "scripts" / "measure_hosd_miniature_resources.py"
    ).read_text(encoding="utf-8")
    assert "AuthenticatedTreeSplit(" in target_source
    assert "AuthenticatedTreeSplit(" in graph_source
    assert "AuthenticatedTreeSplit(" in measurement_source
    assert "expected_parents={" in measurement_source
    assert '"scale_inputs": input_completion["content_hash"]' in measurement_source
    assert '"scale_trees": tree_completion["content_hash"]' in measurement_source
    assert '"scale_targets": target_completion["content_hash"]' in measurement_source
    assert (
        '"scale_teacher_outputs": teacher_completion["content_hash"]'
        in measurement_source
    )
    scale_layout_source = measurement_source.split(
        "def _scale_layout_ledger", 1
    )[1].split("def _latest_jobs", 1)[0]
    assert ".rglob(" not in scale_layout_source


def test_stage_j_normalizer_consumes_lazy_shards_without_population_coercion(tmp_path):
    class LazyRows:
        def __init__(self, values):
            self._values = values
            self.shape = values.shape

        def __getitem__(self, index):
            return self._values[index]

        def __array__(self, *args, **kwargs):
            raise AssertionError("population-wide coercion is forbidden")

    values = np.arange(30, dtype=np.float32).reshape(10, 3)
    masks = np.ones_like(values, dtype=bool)
    cache = LoadedTargetCache(
        identities=tuple(f"jet-{index}" for index in range(10)),
        values={"T_OFFLINE_JET_12": LazyRows(values)},
        masks={"T_OFFLINE_JET_12": LazyRows(masks)},
        manifest={
            "split": "scale_train",
            "source": SOURCE,
            "content_hash": "a" * 64,
            "canonical_identity_order_sha256": "b" * 64,
        },
    )
    artifact = fit_sharded_target_normalizer(
        [cache],
        target_id="T_OFFLINE_JET_12",
        fitting_population="target_scale",
        source=SOURCE,
        component_kinds=("continuous",) * 3,
        workspace=tmp_path,
        batch_size=4,
    )
    assert artifact["bounded_statistics"] == "disk_backed_one_component_one_shard_v1"
    assert artifact["event_count"] == 10
    assert artifact["targets"][0]["components"][1]["q50"] == pytest.approx(14.5)


def test_runtime_manifest_cannot_inject_scientific_rows_or_seeds():
    with pytest.raises(ValueError, match="scientific row injection"):
        build_runtime_manifest(
            campaign_spec_sha256="c" * 64,
            files={},
            directories={},
            infrastructure_arguments_by_node={
                "auxiliary_train": ["--row-id", "invented"]
            },
            source=SOURCE,
        )
    with pytest.raises(ValueError, match="scientific row injection"):
        build_runtime_manifest(
            campaign_spec_sha256="c" * 64,
            files={},
            directories={},
            infrastructure_arguments_by_node={
                "confirmation_train": ["--seed", "999"]
            },
            source=SOURCE,
        )


def test_runtime_manifest_rejects_unresolved_or_unbound_templates():
    files, directories = _runtime_bindings()
    arguments = _ready_arguments()
    arguments["storage_measurement"][1] = "__REQUIRED_FILE_EVIDENCE__"
    with pytest.raises(ValueError, match="unresolved placeholder"):
        build_runtime_manifest(
            campaign_spec_sha256="c" * 64,
            files=files,
            directories=directories,
            infrastructure_arguments_by_node=arguments,
            source=SOURCE,
        )
    arguments = _ready_arguments()
    arguments["storage_measurement"][1] = "{file_missing}"
    with pytest.raises(ValueError, match="unbound placeholders"):
        build_runtime_manifest(
            campaign_spec_sha256="c" * 64,
            files=files,
            directories=directories,
            infrastructure_arguments_by_node=arguments,
            source=SOURCE,
        )


def test_runtime_manifest_rejects_literal_paths_and_detects_bound_file_drift(
    tmp_path,
):
    literal = _ready_arguments()
    literal["storage_measurement"].extend(
        ["--probe-input", str(tmp_path / "literal.json")]
    )
    files, directories = _runtime_bindings()
    with pytest.raises(ValueError, match="unauthorized option"):
        build_runtime_manifest(
            campaign_spec_sha256="c" * 64,
            files=files,
            directories=directories,
            infrastructure_arguments_by_node=literal,
            source=SOURCE,
        )

    evidence = tmp_path / "evidence.json"
    evidence.write_text("original\n", encoding="utf-8")
    files, directories = _runtime_bindings(evidence)
    runtime = build_runtime_manifest(
        campaign_spec_sha256="c" * 64,
        files=files,
        directories=directories,
        infrastructure_arguments_by_node=_ready_arguments(),
        source=SOURCE,
    )
    evidence.write_text("drifted\n", encoding="utf-8")
    # Campaign-owned paths are derived by the factory; an unreferenced legacy
    # file binding is not part of this node's execution surface.
    command, _ = resolve_node_argv(
        node={
            "node_id": "storage_measurement",
            "entrypoint": "scripts/measure_hosd_storage.py",
        },
        runtime_manifest=runtime,
        campaign_root=tmp_path,
        coordinate=0,
    )
    assert "--probe-input" in command


def test_node_factory_stream_checks_only_that_nodes_bound_inputs(tmp_path):
    used = tmp_path / "used.json"
    irrelevant = tmp_path / "irrelevant.json"
    used.write_text("used\n", encoding="utf-8")
    irrelevant.write_text("original\n", encoding="utf-8")
    arguments = _ready_arguments()
    runtime = build_runtime_manifest(
        campaign_spec_sha256="c" * 64,
        files={"test": used, "irrelevant": irrelevant},
        directories={"test": REPO_ROOT},
        infrastructure_arguments_by_node=arguments,
        source=SOURCE,
    )
    irrelevant.write_text("drifted\n", encoding="utf-8")
    command, row = resolve_node_argv(
        node={
            "node_id": "storage_measurement",
            "entrypoint": "scripts/measure_hosd_storage.py",
        },
        runtime_manifest=runtime,
        campaign_root=tmp_path,
        coordinate=0,
    )
    assert row is None
    assert command is not None
    second, _ = resolve_node_argv(
        node={
            "node_id": "scale_teacher_adapter_compile",
            "entrypoint": "scripts/compile_hosd_scale_teacher_adapters.py",
        },
        runtime_manifest=runtime,
        campaign_root=tmp_path,
        coordinate=0,
    )
    assert "--screening-registry" in second


def test_pair_probe_materialization_never_persists_tap_states(
    tmp_path, monkeypatch
):
    from teacher_logit_reco.hlt_offline_structure_distillation import node_runtime

    rows = [
        {
            "row_id": f"{target}__{kind}"
            + (f"__{tap}" if tap is not None else ""),
            "target_id": target,
            "probe_kind": kind,
            "tap": tap,
        }
        for target in ("T_HLT_TRACK_PAIR_13", "T_HLT_REGION_PAIR_8")
        for kind, tap in (
            ("P_STATISTICAL_REFERENCES", None),
            ("P_LINEAR", "TAP_EARLY"),
            ("P_SHALLOW", "TAP_MID"),
            ("P_TARGET_TO_CLASS_ORACLE", None),
        )
    ]
    monkeypatch.setattr(
        node_runtime,
        "_rows",
        lambda root, node_id: rows
        if node_id == "probe_input_materialization"
        else None,
    )
    files, directories = _runtime_bindings()
    runtime = build_runtime_manifest(
        campaign_spec_sha256="c" * 64,
        files=files,
        directories=directories,
        infrastructure_arguments_by_node=_ready_arguments(),
        source=SOURCE,
    )
    node = {
        "node_id": "probe_input_materialization",
        "entrypoint": "scripts/materialize_hosd_pair_probe_inputs.py",
    }
    for coordinate, expected in enumerate(rows):
        command, observed = resolve_node_argv(
            node=node,
            runtime_manifest=runtime,
            campaign_root=tmp_path,
            coordinate=coordinate,
        )
        assert observed == expected
        rendered = " ".join(command)
        assert "__None.npz" not in rendered
        assert "--tap-cache" not in command
        assert "frozen_taps" not in rendered


def test_learned_probe_training_streams_exact_tap_from_native_inputs(
    tmp_path, monkeypatch
):
    from teacher_logit_reco.hlt_offline_structure_distillation import node_runtime

    row = {
        "row_id": "T_OFFLINE_JET_10__P_LINEAR__TAP_EARLY",
        "target_id": "T_OFFLINE_JET_10",
        "probe_kind": "P_LINEAR",
        "tap": "TAP_EARLY",
    }
    monkeypatch.setattr(
        node_runtime,
        "_rows",
        lambda root, node_id: [row] if node_id == "probe_train" else None,
    )
    runtime = _ready_runtime()
    command, observed = resolve_node_argv(
        node={"node_id": "probe_train", "entrypoint": "scripts/train_hosd_probe.py"},
        runtime_manifest=runtime,
        campaign_root=tmp_path,
        coordinate=0,
    )
    assert observed == row
    assert command.count("--train-cache") == 4
    assert command.count("--val-stop-cache") == 1
    assert command.count("--design-select-cache") == 1
    assert "--baseline-checkpoint" in command
    assert "--probe-encoder-lock" in command
    assert "--tap-cache" not in command


def test_streamed_probe_tap_ram_projection_fits_registered_gpu_request() -> None:
    module = _load_resource_measurement_module()
    assert module.PROBE_TAP_BYTES_PER_EVENT_REPLICA == 65_664
    assert module.PROBE_TAP_RESIDENT_EVENT_REPLICAS == 2_075_000
    projected = module.PROBE_TAP_PROJECTED_RESIDENT_BYTES
    assert projected == 136_252_800_000
    assert projected * module.MEMORY_SAFETY_FACTOR < 220 * 1024**3


def test_scale_graph_command_does_not_bind_all_tree_or_native_populations(
    tmp_path, monkeypatch
):
    from teacher_logit_reco.hlt_offline_structure_distillation import node_runtime

    row = {"graph_id": "H_BASE", "seed": 202}
    monkeypatch.setattr(
        node_runtime,
        "_rows",
        lambda root, node_id: [row] if node_id == "scale_graph_train" else None,
    )
    files, directories = _runtime_bindings()
    runtime = build_runtime_manifest(
        campaign_spec_sha256="c" * 64,
        files=files,
        directories=directories,
        infrastructure_arguments_by_node=_ready_arguments(),
        source=SOURCE,
    )
    command, observed = resolve_node_argv(
        node={
            "node_id": "scale_graph_train",
            "entrypoint": "scripts/execute_hosd_scale_graph.py",
        },
        runtime_manifest=runtime,
        campaign_root=tmp_path,
        coordinate=0,
    )
    assert observed == row
    assert command.count("--scale-train-cache") == 4
    assert "--scale-train-tree" not in command
    assert "--scale-native-relation" not in command
    assert "--design-confirm-native-relation" not in command


def test_runtime_manifest_rejects_bad_scalars_and_nonexact_key_coverage():
    files, directories = _runtime_bindings()
    arguments = _ready_arguments()
    storage = arguments["storage_measurement"]
    storage[storage.index("--available-storage-bytes") + 1] = "not-an-int"
    runtime = build_runtime_manifest(
        campaign_spec_sha256="c" * 64,
        files=files,
        directories=directories,
        infrastructure_arguments_by_node=arguments,
        source=SOURCE,
    )
    assert not runtime["execution_ready"]
    assert "--available-storage-bytes" in " ".join(
        runtime["missing_required_options_by_node"]["storage_measurement"]
    )

    arguments = _ready_arguments()
    arguments["scale_efficiency"].extend(
        ["--production-batch-size", "64"]
    )
    runtime = build_runtime_manifest(
        campaign_spec_sha256="c" * 64,
        files=files,
        directories=directories,
        infrastructure_arguments_by_node=arguments,
        source=SOURCE,
    )
    assert not runtime["execution_ready"]
    assert "one positive integer" in " ".join(
        runtime["missing_required_options_by_node"]["scale_efficiency"]
    )


def test_runtime_config_template_covers_exact_required_keyed_bindings():
    path = REPO_ROOT / "scripts" / "write_hosd_runtime_config_template.py"
    spec = importlib.util.spec_from_file_location("hosd_runtime_template", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    template = module.build_template()
    arguments = template["infrastructure_arguments_by_node"]
    for node_id, by_option in REQUIRED_INFRASTRUCTURE_OPTION_KEYS.items():
        for option, required_keys in by_option.items():
            values = arguments[node_id]
            observed = {
                values[index + 1].split("=", 1)[0]
                for index, value in enumerate(values[:-1])
                if value == option
            }
            assert observed == set(required_keys)
    assert (
        REPO_ROOT / "scripts" / "prepare_hosd_execution.py"
    ).is_file()
    materializer_path = (
        REPO_ROOT / "scripts" / "materialize_hosd_runtime_inputs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hosd_runtime_inputs", materializer_path
    )
    assert spec is not None and spec.loader is not None
    materializer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(materializer)
    assert set(materializer.HLT_COORDINATES) == {
        *(("model_train", replica, "R_MULTI") for replica in range(4)),
        ("val_stop", 0, "R_FIXED"),
        ("val_design", 0, "R_FIXED"),
    }


def test_campaign_owned_paths_are_derived_per_scientific_coordinate(
    tmp_path, monkeypatch
):
    from teacher_logit_reco.hlt_offline_structure_distillation import node_runtime

    row = {
        "split": "model_train",
        "artifact_kind": "canonical",
        "row_id": "canonical_model_train",
        "target_ids": ["T_OFFLINE_JET_12"],
    }
    monkeypatch.setattr(
        node_runtime,
        "_rows",
        lambda root, node_id: [row]
        if node_id == "canonical_target_build"
        else None,
    )
    command, observed = resolve_node_argv(
        node={
            "node_id": "canonical_target_build",
            "entrypoint": "scripts/build_hosd_targets.py",
        },
        runtime_manifest=_ready_runtime(),
        campaign_root=tmp_path,
        coordinate=0,
    )
    assert observed == row
    rendered = " ".join(command)
    assert str(
        tmp_path
        / "inputs"
        / "hosd_views"
        / "offline"
        / "model_train.npz"
    ) in rendered
    assert str(
        tmp_path
        / "inputs"
        / "shared_retb_parent_campaign"
        / "inputs"
        / "region_tree"
        / "offline"
        / "model_train_exclusive_ca_v1"
    ) in rendered
    assert "__REQUIRED_" not in rendered


def test_design_roles_resolve_to_disjoint_authenticated_subrole_inputs(tmp_path):
    from teacher_logit_reco.hlt_offline_structure_distillation import node_runtime

    baseline = " ".join(
        node_runtime._derived_infrastructure(tmp_path, "baseline_train", {})
    )
    confirmation = " ".join(
        node_runtime._derived_infrastructure(tmp_path, "confirmation_train", {})
    )
    robustness = " ".join(
        node_runtime._derived_infrastructure(tmp_path, "robustness_evaluation", {})
    )
    assert str(
        tmp_path
        / "inputs"
        / "shared_retb_parent_campaign"
        / "inputs"
        / "hlt_v3"
        / "model_train"
        / "replica_0"
        / "R_MULTI"
        / "D_NOMINAL"
    ) in baseline
    assert str(
        tmp_path
        / "inputs"
        / "shared_retb_parent_campaign"
        / "inputs"
        / "hlt_v3"
        / "val_design"
        / "replica_0"
        / "R_FIXED"
        / "D_NOMINAL"
    ) in baseline
    assert str(
        tmp_path
        / "inputs"
        / "hosd_views"
        / "hlt"
        / "design_select"
        / "replica_0.npz"
    ) not in baseline
    assert "design_select_identity_labels.npz" in baseline
    assert str(tmp_path / "inputs" / "hosd_views" / "hlt" / "design_confirm" / "replica_0.npz") in confirmation
    assert "design_confirm_identity_labels.npz" in confirmation
    assert "design_confirm_identity_labels.npz" in robustness
    assert str(tmp_path / "inputs" / "hosd_views" / "hlt" / "val_design" / "replica_0.npz") not in baseline
    assert str(tmp_path / "inputs" / "hosd_views" / "hlt" / "val_design" / "replica_0.npz") not in confirmation


def test_kd_baseline_aligns_compact_teacher_logits_by_exact_identity(
    tmp_path, monkeypatch
):
    path = REPO_ROOT / "scripts" / "train_hosd_baseline.py"
    spec = importlib.util.spec_from_file_location("hosd_baseline_kd_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cache_root = tmp_path / "teacher_cache"
    cache_root.mkdir()
    canonical_logits = np.asarray(
        [[10.0] * 10, [20.0] * 10], dtype=np.float32
    )

    class ContiguousSliceOnly:
        shape = canonical_logits.shape

        def __getitem__(self, index):
            assert isinstance(index, slice)
            assert index == slice(None)
            return canonical_logits[index]

    cache = LoadedTargetCache(
        identities=("jet-a", "jet-b"),
        values={
            "T_OFFLINE_LOGITS_O_BASE": ContiguousSliceOnly()
        },
        masks={
            "T_OFFLINE_LOGITS_O_BASE": np.ones((2, 10), dtype=bool)
        },
        manifest={"content_hash": "a" * 64},
    )
    monkeypatch.setattr(module, "load_hashed_json", lambda *args, **kwargs: {})
    import teacher_logit_reco.hlt_offline_structure_distillation as hosd

    monkeypatch.setattr(hosd, "load_target_cache", lambda *args, **kwargs: cache)

    aligned = module._privileged(
        cache_root, ("jet-b", "jet-a"), "logits"
    )
    assert aligned.shape == (2, 10)
    assert aligned[:, 0].tolist() == [20.0, 10.0]

    with pytest.raises(ValueError, match="identities differ"):
        module._privileged(cache_root, ("jet-a", "jet-c"), "logits")


def test_native_baseline_uses_identity_index_without_copying_population(tmp_path):
    path = REPO_ROOT / "scripts" / "train_hosd_baseline.py"
    spec = importlib.util.spec_from_file_location("hosd_baseline_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Base:
        replicas = {0: {}}
        identities = ("jet-a", "jet-b")
        logical_role = "model_train"
        realization_policy = "R_MULTI"
        metadata = {0: {}}

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {"replica_id": 0, "identity": self.identities[index]}

        def set_epoch(self, epoch):
            self.epoch = epoch

    targets = np.zeros((2, 545), dtype=np.float32)
    targets[:, 0] = [20.0, 10.0]
    wrapped = module._ReplicaNativeTargetDataset(
        Base(),
        {
            0: {
                "targets": targets,
                "target_mask": np.ones_like(targets, dtype=bool),
                "availability": np.ones((2, 7), dtype=np.float32),
                "source_indices": np.asarray([1, 0], dtype=np.int64),
            }
        },
    )
    assert wrapped[0]["offline_target_tokens"][0, 0] == 10.0
    assert wrapped[1]["offline_target_tokens"][0, 0] == 20.0
    assert wrapped.targets_by_replica[0]["targets"] is targets


def test_prepare_execution_can_build_minimal_runtime_without_template(
    tmp_path, monkeypatch
):
    path = REPO_ROOT / "scripts" / "prepare_hosd_execution.py"
    spec = importlib.util.spec_from_file_location("hosd_prepare_execution", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    campaign = {"content_hash": "c" * 64, "source": SOURCE}
    captured = {}
    monkeypatch.setattr(module, "load_and_validate_campaign", lambda *a, **k: campaign)
    monkeypatch.setattr(module, "parity_main", lambda argv: 0)
    monkeypatch.setattr(
        module,
        "publish_runtime_support",
        lambda **kwargs: {"content_hash": "d" * 64},
    )

    def runtime_builder(**kwargs):
        captured.update(kwargs)
        return {
            "content_hash": "e" * 64,
            "execution_ready": True,
            "missing_required_options_by_node": {},
            "source": SOURCE,
        }

    monkeypatch.setattr(module, "build_runtime_manifest", runtime_builder)
    monkeypatch.setattr(
        module,
        "load_hashed_json",
        lambda *a, **k: {"content_hash": "f" * 64},
    )
    monkeypatch.setattr(
        module,
        "build_node_factory_registry",
        lambda **kwargs: {"content_hash": "1" * 64},
    )
    monkeypatch.setattr(module, "build_registered_command_matrix", lambda **kwargs: {})
    monkeypatch.setattr(
        module,
        "build_production_execution_plan",
        lambda **kwargs: {"content_hash": "2" * 64, "node_count": 78},
    )
    monkeypatch.setattr(
        module,
        "write_immutable_json",
        lambda *a, **k: {"status": "published"},
    )
    assert (
        module.main(
            [
                "--campaign-root",
                str(tmp_path),
                "--profile",
                "miniature_test",
                "--available-storage-bytes",
                "123456",
            ]
        )
        == 0
    )
    assert captured["files"] == {}
    assert captured["directories"] == {}
    assert captured["infrastructure_arguments_by_node"] == {
        "storage_measurement": ["--available-storage-bytes", "123456"],
        "scale_efficiency": [
            "--production-batch-size",
            "32",
            "--clock-power-mode",
            "miniature_control_plane_unmeasured",
        ],
    }


def test_miniature_bootstrap_controller_is_resumable_and_performance_blind(
    tmp_path, monkeypatch
):
    path = REPO_ROOT / "scripts" / "bootstrap_hosd_miniature_execution.py"
    spec = importlib.util.spec_from_file_location("hosd_mini_bootstrap", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "load_and_validate_campaign",
        lambda *a, **k: {"campaign_profile": "miniature_test"},
    )
    (tmp_path / "campaign_spec.json").write_text("{}\n", encoding="utf-8")
    commands = []
    monkeypatch.setattr(
        module,
        "_run",
        lambda argv, **kwargs: commands.append((list(argv), kwargs)),
    )
    assert module.main(["--campaign-root", str(tmp_path)]) == 0
    rendered = [" ".join(argv) for argv, _ in commands]
    assert any("build_hosd_shared_hlt_parents.py" in row for row in rendered)
    assert any("build_hosd_tree_parents.py" in row for row in rendered)
    assert any("fit_hosd_relation_normalizers.py" in row for row in rendered)
    assert any("lock_hosd_inherited_parents.py" in row for row in rendered)
    assert any("materialize_hosd_runtime_inputs.py" in row for row in rendered)
    assert any("prepare_hosd_execution.py" in row for row in rendered)
    assert rendered[-1].endswith("--smoke-submit")
    source = path.read_text(encoding="utf-8")
    assert "performance blind" in source
    assert "accuracy" not in source.lower()


def test_miniature_bootstrap_reuses_authenticated_parent_lock(tmp_path, monkeypatch):
    path = REPO_ROOT / "scripts" / "bootstrap_hosd_miniature_execution.py"
    spec = importlib.util.spec_from_file_location("hosd_mini_bootstrap_restart", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = {
        "commit": "a" * 40,
        "dirty": False,
        "status_hash_policy": "test",
        "status_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        module,
        "load_and_validate_campaign",
        lambda *a, **k: {"campaign_profile": "miniature_test", "source": source},
    )
    monkeypatch.setattr(
        module,
        "load_hashed_json",
        lambda *a, **k: {
            "source": source,
            "all_stage_b_parents_reusable": True,
        },
    )
    (tmp_path / "campaign_spec.json").write_text("{}\n", encoding="utf-8")
    lock_path = tmp_path / "inputs" / "resolved_inherited_parent_lock.json"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("{}\n", encoding="utf-8")
    commands = []
    monkeypatch.setattr(
        module,
        "_run",
        lambda argv, **kwargs: commands.append((list(argv), kwargs)),
    )
    assert (
        module.main(
            ["--campaign-root", str(tmp_path), "--prepare-only"]
        )
        == 0
    )
    rendered = [" ".join(argv) for argv, _ in commands]
    assert not any("lock_hosd_inherited_parents.py" in row for row in rendered)
    assert any("materialize_hosd_runtime_inputs.py" in row for row in rendered)
    assert any("prepare_hosd_execution.py" in row for row in rendered)


def test_miniature_bootstrap_can_create_fresh_source_bound_campaign(
    tmp_path, monkeypatch
):
    path = REPO_ROOT / "scripts" / "bootstrap_hosd_miniature_execution.py"
    spec = importlib.util.spec_from_file_location("hosd_mini_create", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent = tmp_path / "split_manifest.json.gz"
    parent.write_bytes(b"split")
    root = tmp_path / "new-campaign"
    commands = []
    monkeypatch.setattr(
        module,
        "_run",
        lambda argv, **kwargs: commands.append((list(argv), kwargs)),
    )
    monkeypatch.setattr(
        module,
        "load_and_validate_campaign",
        lambda *a, **k: {"campaign_profile": "miniature_test"},
    )
    assert module.main(
        [
            "--campaign-root",
            str(root),
            "--parent-manifest",
            str(parent),
            "--prepare-only",
        ]
    ) == 0
    first = " ".join(commands[0][0])
    assert "build_hosd_campaign.py" in first
    assert "--miniature" in first
    assert str(parent.resolve()) in first
    assert not any("--smoke-submit" in " ".join(row[0]) for row in commands)


def test_authoritative_parity_fixture_uses_physical_four_vectors():
    path = REPO_ROOT / "scripts" / "validate_hosd_weaver_parity.py"
    spec = importlib.util.spec_from_file_location("hosd_parity_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    batch = module._batch(2718)
    vectors = batch["lorentz_vectors"]
    mask = batch["mask"][:, 0]
    momentum_squared = vectors[:, :3].square().sum(dim=1)
    mass_squared = vectors[:, 3].square() - momentum_squared
    assert np.isfinite(vectors.numpy()).all()
    assert bool((mass_squared[mask] > 0).all())
    assert bool((vectors[:, 3][mask] > 0).all())
    assert bool((vectors.masked_select(~batch["mask"]).eq(0)).all())


def test_parent_lock_consumers_use_the_versioned_contract_constant():
    consumers = (
        REPO_ROOT
        / "teacher_logit_reco"
        / "hlt_offline_structure_distillation"
        / "parent_submission.py",
        REPO_ROOT / "scripts" / "materialize_hosd_runtime_inputs.py",
        REPO_ROOT / "scripts" / "build_hosd_robustness_cache.py",
        REPO_ROOT / "scripts" / "prepare_hosd_scale_inputs.py",
    )
    for path in consumers:
        source = path.read_text(encoding="utf-8")
        assert "hosd_parent_status_v1" not in source
        assert "PARENT_STATUS_CONTRACT" in source


def test_target_array_factory_selects_exact_split_and_replica_bindings():
    canonical = _row_scoped_arguments(
        "canonical_target_build",
        [
            "--input-npz",
            "model_train=/train.npz",
            "--input-npz",
            "val_stop=/stop.npz",
            "--input-npz",
            "val_design=/design.npz",
            "--tree-cache-dir",
            "model_train=/tree/train",
            "--tree-cache-dir",
            "val_stop=/tree/stop",
            "--tree-cache-dir",
            "val_design=/tree/design",
            "--relation-normalizer",
            "/normalizer.json",
        ],
        {"split": "val_stop"},
    )
    assert canonical == [
        "--input-npz",
        "/stop.npz",
        "--tree-cache-dir",
        "/tree/stop",
        "--relation-normalizer",
        "/normalizer.json",
    ]
    hlt = _row_scoped_arguments(
        "hlt_analogue_target_build",
        [
            "--input-npz",
            "model_train:0=/wrong.npz",
            "--input-npz",
            "val_design:3=/right.npz",
            "--tree-cache-dir",
            "model_train:0=/wrong-tree",
            "--tree-cache-dir",
            "val_design:3=/right-tree",
        ],
        {"split": "val_design", "replica": 3},
    )
    assert hlt == [
        "--input-npz",
        "/right.npz",
        "--tree-cache-dir",
        "/right-tree",
    ]


def test_terminal_ledger_rejects_one_failed_array_coordinate():
    path = REPO_ROOT / "scripts" / "complete_hosd_job_ledger.py"
    spec = importlib.util.spec_from_file_location("hosd_completed_ledger", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {"job_id_raw": "42", "state": "COMPLETED", "exit_code": "0:0"},
        {"job_id_raw": "42_0", "state": "COMPLETED", "exit_code": "0:0"},
        {"job_id_raw": "42_1", "state": "FAILED", "exit_code": "1:0"},
    ]
    with pytest.raises(RuntimeError, match="not completely successful"):
        module._completed_rows_for_job(rows, "42")
    rows[-1] = {
        "job_id_raw": "42_1",
        "state": "COMPLETED",
        "exit_code": "0:0",
    }
    assert len(module._completed_rows_for_job(rows, "42")) == 3


def test_resource_measurement_helpers_are_exact_and_fail_closed():
    path = REPO_ROOT / "scripts" / "measure_hosd_miniature_resources.py"
    spec = importlib.util.spec_from_file_location("hosd_resource_measurement", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._bytes("1.5G") == int(1.5 * 1024**3)
    assert module._slurm_seconds("2-00:00:00") == 2 * 24 * 3600
    assert module._slurm_time(2 * 24 * 3600) == "2-00:00:00"
    rows = [
        {
            "job_id_raw": "17",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "elapsed_seconds": "5",
        },
        {
            "job_id_raw": "17_0",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "elapsed_seconds": "3",
        },
        {
            "job_id_raw": "17_1",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "elapsed_seconds": "4",
        },
    ]
    assert [
        row["job_id_raw"] for row in module._allocation_rows(rows, "17")
    ] == ["17_0", "17_1"]
    rows[-1]["state"] = "FAILED"
    with pytest.raises(ValueError, match="not completely measured"):
        module._allocation_rows(rows, "17")
