from __future__ import annotations

from pathlib import Path
import importlib.util
from types import SimpleNamespace

import pytest

from teacher_logit_reco.hlt_offline_structure_distillation import (
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
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (
    with_content_hash,
)
from teacher_logit_reco.hlt_offline_structure_distillation.node_runtime import (
    _row_scoped_arguments,
    resolve_node_argv,
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


def _plan(profile):
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
        stage_job_registry=registry, source=SOURCE
    )
    commands = build_registered_command_matrix(
        stage_job_registry=registry,
        factory_registry=factories,
        runtime_manifest=runtime,
        campaign_root="/authenticated/campaign",
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
            stage_job_registry=registry, source=SOURCE
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
            "contract": "hosd_resource_preflight_v3",
            "schema_version": 3,
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
        "submit_hosd_tigris_full.sh",
    )
    for name in names:
        text = (REPO_ROOT / "sbatch" / name).read_text(encoding="utf-8")
        assert "set -euo pipefail" in text
    submit = (REPO_ROOT / "sbatch" / "submit_hosd_tigris_full.sh").read_text(
        encoding="utf-8"
    )
    assert "submit_hosd_slurm.py" in submit
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
    assert set(
        REQUIRED_INFRASTRUCTURE_OPTIONS_BY_NODE["scale_input_prepare"]
    ) == {"--scale-offline-cache", "--scale-hlt-cache"}
    assert REQUIRED_INFRASTRUCTURE_OPTION_KEYS["scale_input_prepare"][
        "--scale-hlt-cache"
    ] == frozenset({"0", "1", "2", "3"})
    # These are generated by upstream Stage-J jobs and must never be requested
    # as manually injected future artifacts in the runtime manifest.
    assert "scale_target_build" not in REQUIRED_INFRASTRUCTURE_OPTIONS_BY_NODE
    scale_graph_external = set(
        REQUIRED_INFRASTRUCTURE_OPTIONS_BY_NODE["scale_graph_train"]
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
    literal["storage_measurement"][1] = str(tmp_path / "literal.json")
    files, directories = _runtime_bindings()
    with pytest.raises(ValueError, match="not authenticated"):
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
    with pytest.raises(ValueError, match="drifted"):
        resolve_node_argv(
            node={
                "node_id": "storage_measurement",
                "entrypoint": "scripts/measure_hosd_storage.py",
            },
            runtime_manifest=runtime,
            campaign_root=tmp_path,
            coordinate=0,
        )


def test_node_factory_stream_checks_only_that_nodes_bound_inputs(tmp_path):
    used = tmp_path / "used.json"
    irrelevant = tmp_path / "irrelevant.json"
    used.write_text("used\n", encoding="utf-8")
    irrelevant.write_text("original\n", encoding="utf-8")
    arguments = _ready_arguments()
    arguments["scale_teacher_adapter_compile"][
        arguments["scale_teacher_adapter_compile"].index(
            "--screening-registry"
        )
        + 1
    ] = "{file_irrelevant}"
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
    with pytest.raises(ValueError, match="drifted"):
        resolve_node_argv(
            node={
                "node_id": "scale_teacher_adapter_compile",
                "entrypoint": "scripts/compile_hosd_scale_teacher_adapters.py",
            },
            runtime_manifest=runtime,
            campaign_root=tmp_path,
            coordinate=0,
        )


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
    arguments["target_normalization"].extend(
        ["--model-train-hlt", "4={file_test}"]
    )
    runtime = build_runtime_manifest(
        campaign_spec_sha256="c" * 64,
        files=files,
        directories=directories,
        infrastructure_arguments_by_node=arguments,
        source=SOURCE,
    )
    assert not runtime["execution_ready"]
    assert "exact keys" in " ".join(
        runtime["missing_required_options_by_node"]["target_normalization"]
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
