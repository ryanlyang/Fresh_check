from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from teacher_logit_reco.local_particle_residual_field.particle_view import (
    LOGICAL_NODE_LAYOUT,
    NODE_WRAPPERS,
    PARTICLE_VIEW_PRODUCTION_GRAPH_CONTRACT,
    CONFIRMATION_ROLE_IDS,
    CONSUMER_SCREEN_IDS,
    EXPECTED_LOW_DATA_CATEGORY_COUNTS,
    FOCUSED_INTERACTION_IDS,
    ParticleViewRunSpec,
    TARGET_SCREEN_IDS,
    aggregate_quality_warnings,
    build_node_completion,
    build_low_data_campaign_inventory,
    build_low_data_campaign_registry,
    build_particle_view_production_graph,
    build_particle_view_registry,
    build_unified_split_manifest,
    build_quality_warning,
    build_runtime_command_catalog,
    build_runtime_execution_manifest,
    build_runtime_handler_catalog,
    build_runtime_task_result,
    build_scientific_handler_commands,
    build_scientific_task_catalog,
    capture_clean_source_checkout,
    plan_particle_view_submissions,
    reconcile_particle_view_production_graph,
    submit_particle_view_graph,
    validate_particle_view_production_graph,
    load_quality_warning_jsonl,
    miniature_parent_manifest,
    miniature_split_config,
    validate_low_data_campaign_registry,
    execute_runtime_node,
    execute_scientific_task,
    sha256_file,
    validate_runtime_execution_manifest,
    validate_scientific_task_catalog,
    write_quality_warning_jsonl,
    write_immutable_json,
    with_content_hash,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _registry():
    specs = []
    parent = None
    for index, (_, stages, _) in enumerate(LOGICAL_NODE_LAYOUT):
        stage = stages[0]
        run_id = f"run_{index:02d}_{stage}"
        inference = stage in {"source", "stack", "report_export", "final_test"}
        final = stage == "final_test"
        specs.append(
            ParticleViewRunSpec(
                run_id=run_id,
                stage=stage,
                scientific_role=f"test_{stage}",
                selection_family=(
                    "pre_stage_g_deployable" if final else "infrastructure"
                ),
                seed_ids=(101, 202, 303)
                if stage in {"confirmation", "fairness"}
                else (101,),
                parent_run_ids=(() if parent is None else (parent,)),
                uses_labels=not inference,
                train_split=None if inference else "train",
                selectable=final,
                final_test_eligible=final,
            )
        )
        parent = run_id
    return build_particle_view_registry(
        unified_split_manifest_sha256=_sha("manifest"),
        train_identity_sha256=_sha("train"),
        run_specs=specs,
        campaign_id="step10_test",
    )


def _graph():
    registry = _registry()
    catalog = {
        node_id: ["python", "-u", "worker.py", "--node", node_id]
        for node_id, _, _ in LOGICAL_NODE_LAYOUT
    }
    graph = build_particle_view_production_graph(
        registry=registry,
        artifact_root="/campaign/pview",
        source_commit="a" * 40,
        command_catalog=catalog,
    )
    return registry, graph


def _low_data_registry(**kwargs):
    parent = miniature_parent_manifest()
    config = miniature_split_config()
    unified = build_unified_split_manifest(parent, config=config)
    return unified, build_low_data_campaign_registry(
        unified_split_manifest=unified,
        **kwargs,
    )


def _runtime_registry():
    return build_particle_view_registry(
        unified_split_manifest_sha256=_sha("runtime-manifest"),
        train_identity_sha256=_sha("runtime-train"),
        run_specs=[
            ParticleViewRunSpec(
                run_id="runtime_source",
                stage="source",
                scientific_role="source:runtime_fixture",
                selection_family="infrastructure",
                uses_labels=False,
                train_split=None,
            ),
            ParticleViewRunSpec(
                run_id="runtime_baseline",
                stage="baseline",
                scientific_role="baseline:runtime_fixture",
                selection_family="infrastructure",
                seed_ids=(101, 202, 303),
                parent_run_ids=("runtime_source",),
            ),
        ],
        campaign_id="runtime_fixture",
    )


def _runtime_manifest(tmp_path):
    registry = _runtime_registry()
    handlers = build_runtime_handler_catalog(
        {
            category: [
                sys.executable,
                "worker.py",
                "--run-id",
                "{run_id}",
                "--seed",
                "{seed}",
                "--output-dir",
                "{task_output_dir}",
            ]
            for category in ("baseline", "source")
        }
    )
    return registry, build_runtime_execution_manifest(
        registry=registry,
        registry_path=str(tmp_path / "registry.json"),
        handler_catalog=handlers,
        handler_catalog_path=str(tmp_path / "handlers.json"),
        artifact_root=str(tmp_path / "campaign"),
    )


def test_step10_authoritative_low_data_registry_has_the_locked_full_inventory():
    _, registry = _low_data_registry()
    audit = validate_low_data_campaign_registry(registry)
    inventory = build_low_data_campaign_inventory(registry)
    assert len(TARGET_SCREEN_IDS) == 36
    assert len(CONSUMER_SCREEN_IDS) == 12
    assert len(FOCUSED_INTERACTION_IDS) == 22
    assert len(CONFIRMATION_ROLE_IDS) == 13
    assert inventory["category_counts"] == EXPECTED_LOW_DATA_CATEGORY_COUNTS
    assert inventory["declared_run_count"] == 230
    assert inventory["seed_expanded_replica_count"] == 300
    assert audit["single_training_pool"] is True
    assert inventory["quality_gates"] is False
    by_id = {row["run_id"]: row for row in registry["runs"]}
    assert by_id["VGEN_MEMORY_HLT"]["seed_ids"] == [101, 202, 303]
    assert by_id["VGEN_MEMORY_HLT"]["selection_family"] == (
        "pre_stage_g_deployable"
    )
    assert by_id["VGEN_MEMORY_HLT_SELFMASK"]["diagnostic"] is True
    assert by_id["FINAL_PRIVILEGED_SCIENTIFIC"]["final_test_eligible"] is True
    assert by_id["FINAL_PRE_STAGE_G_DEPLOYABLE"]["final_test_eligible"] is True


def test_step10_optional_teacher_compatibility_changes_permissions_not_inventory():
    _, default = _low_data_registry()
    _, compatible = _low_data_registry(
        existing_teacher_compatible=True,
        teacher_mix_compatible=True,
    )
    default_rows = {row["run_id"]: row for row in default["runs"]}
    compatible_rows = {row["run_id"]: row for row in compatible["runs"]}
    assert set(default_rows) == set(compatible_rows)
    assert default_rows["VGEN_TEACHER_EXISTING"]["diagnostic"] is True
    assert default_rows["VGEN_TEACHER_MIX2"]["diagnostic"] is True
    assert compatible_rows["VGEN_TEACHER_EXISTING"]["selectable"] is True
    assert compatible_rows["VGEN_TEACHER_MIX2"]["selectable"] is True


def test_step10_low_data_validator_rejects_an_incomplete_generic_registry():
    incomplete = build_particle_view_registry(
        unified_split_manifest_sha256=_sha("manifest"),
        train_identity_sha256=_sha("train"),
        run_specs=[
            ParticleViewRunSpec(
                run_id="only_source",
                stage="source",
                scientific_role="source:incomplete",
                selection_family="infrastructure",
                uses_labels=False,
                train_split=None,
            )
        ],
        campaign_id="incomplete",
    )
    with pytest.raises(ValueError, match="category inventory mismatch"):
        validate_low_data_campaign_registry(incomplete)


def test_step10_full_low_data_registry_reconciles_to_all_logical_nodes():
    _, registry = _low_data_registry()
    catalog = {
        node_id: ["python", "-u", "worker.py", "--node", node_id]
        for node_id, _, _ in LOGICAL_NODE_LAYOUT
    }
    graph = build_particle_view_production_graph(
        registry=registry,
        artifact_root="/campaign/pview",
        source_commit="a" * 40,
        command_catalog=catalog,
    )
    reconciliation = reconcile_particle_view_production_graph(
        graph=graph,
        registry=registry,
    )
    assert reconciliation["reconciled"] is True
    assert reconciliation["counts"]["declared_runs"] == 230
    assert reconciliation["counts"]["declared_seed_replicas"] == 300
    assert reconciliation["counts"]["generated_seed_replicas"] == 300
    assert reconciliation["invalid_parent_edges"] == []
    assert sum(
        node["seed_expanded_task_count"] for node in graph["nodes"]
    ) == 300
    assert all(node["array_max_concurrency"] == 16 for node in graph["nodes"])
    final_node = next(
        node for node in graph["nodes"]
        if node["node_id"] == "pv10_hlt_only_final_test"
    )
    assert all(len(wave) == 1 for wave in final_node["task_waves"])
    run_by_id = {row["run_id"]: row for row in registry["runs"]}
    for control_id in (
        "A0_VIEW_LONG_DEPLOY",
        "A0_VIEW_TOTAL_LABEL_BUDGET",
        "SELECTED_PARAMETER_MATCH",
        "SELECTED_FLOP_MATCH",
    ):
        parents = run_by_id[
            f"FAIR_PRIVILEGED_SCIENTIFIC_{control_id}"
        ]["parent_run_ids"]
        assert f"FAIR_PRE_STAGE_G_DEPLOYABLE_{control_id}" in parents


def test_step10_runtime_manifest_expands_seeds_and_generates_all_node_commands(
    tmp_path,
):
    registry, manifest = _runtime_manifest(tmp_path)
    audit = validate_runtime_execution_manifest(manifest, registry=registry)
    assert audit["task_count"] == 4
    assert audit["node_task_counts"] == {
        "pv00_source": 1,
        "pv01_baselines": 3,
    }
    tasks = {row["task_id"]: row for row in manifest["tasks"]}
    for seed in (101, 202, 303):
        task = tasks[f"runtime_baseline__seed_{seed}"]
        assert task["parent_task_ids"] == ["runtime_source__seed_101"]
        assert str(seed) in task["command"]
    commands = build_runtime_command_catalog(
        execution_manifest_path=str(tmp_path / "execution.json"),
    )
    assert set(commands) == {row[0] for row in LOGICAL_NODE_LAYOUT}
    assert all(
        "scripts/run_particle_view_campaign_node.py" in command
        for command in commands.values()
    )


def test_step10_runtime_manifest_rejects_missing_handler_categories(tmp_path):
    registry = _runtime_registry()
    incomplete = build_runtime_handler_catalog(
        {
            "source": [
                "worker",
                "{run_id}",
                "{seed}",
                "{task_output_dir}",
            ]
        }
    )
    with pytest.raises(ValueError, match="handler coverage mismatch"):
        build_runtime_execution_manifest(
            registry=registry,
            registry_path=str(tmp_path / "registry.json"),
            handler_catalog=incomplete,
            handler_catalog_path=str(tmp_path / "handlers.json"),
            artifact_root=str(tmp_path / "campaign"),
        )


def test_step10_runtime_node_executes_resumes_and_authenticates_results(tmp_path):
    _, manifest = _runtime_manifest(tmp_path)
    calls = []

    def successful_runner(command, **kwargs):
        calls.append(command)
        environment = kwargs["env"]
        output = Path(environment["PARTICLE_VIEW_TASK_OUTPUT_DIR"])
        artifact = output / "payload.txt"
        artifact.write_text(
            environment["PARTICLE_VIEW_TASK_ID"],
            encoding="utf-8",
        )
        result = build_runtime_task_result(
            task_id=environment["PARTICLE_VIEW_TASK_ID"],
            artifacts=[
                {"path": str(artifact), "sha256": _sha(artifact.read_text())}
            ],
        )
        write_immutable_json(output / "task_result.json", result)
        return subprocess.CompletedProcess(command, 0)

    source = execute_runtime_node(
        manifest=manifest,
        node_id="pv00_source",
        runner=successful_runner,
    )
    assert source["completed_count"] == 1
    baseline = execute_runtime_node(
        manifest=manifest,
        node_id="pv01_baselines",
        runner=successful_runner,
    )
    assert baseline["completed_count"] == 3
    assert len(calls) == 4
    assert not any(
        path.name.startswith(".") and ".attempt_" in path.name
        for path in (tmp_path / "campaign" / "runtime_tasks").iterdir()
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("completed tasks must be resumed, not rerun")

    resumed = execute_runtime_node(
        manifest=manifest,
        node_id="pv01_baselines",
        runner=forbidden,
    )
    assert resumed["completed_count"] == 3
    assert {row["action"] for row in resumed["records"]} == {
        "reuse_complete"
    }


def test_step10_runtime_rejects_embedded_attempt_path(tmp_path):
    _, manifest = _runtime_manifest(tmp_path)

    def leaking_runner(command, **kwargs):
        environment = kwargs["env"]
        output = Path(environment["PARTICLE_VIEW_TASK_OUTPUT_DIR"])
        artifact = output / "confirmation_replica.json"
        write_immutable_json(
            artifact,
            with_content_hash(
                {
                    "contract": "production_shaped_confirmation_fixture_v1",
                    "bundle_path": str(output / "selected_confirmation_bundle.pt"),
                }
            ),
        )
        result = build_runtime_task_result(
            task_id=environment["PARTICLE_VIEW_TASK_ID"],
            artifacts=[
                {"path": str(artifact), "sha256": sha256_file(artifact)}
            ],
        )
        write_immutable_json(output / "task_result.json", result)
        return subprocess.CompletedProcess(command, 0)

    report = execute_runtime_node(
        manifest=manifest,
        node_id="pv00_source",
        runner=leaking_runner,
    )
    assert report["failed_count"] == 1
    assert report["records"][0]["action"] == "invalid_result"
    assert "transaction-attempt path leaked" in report["records"][0]["error"]


def test_step10_source_checkout_is_derived_and_dirty_tree_is_rejected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Particle View Test"],
        check=True,
    )
    source = repo / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    checkout = capture_clean_source_checkout(repo)
    assert checkout["source_checkout_clean"]
    assert len(checkout["source_commit"]) == 40
    assert len(checkout["source_tree_git_oid"]) == 40
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean checkout"):
        capture_clean_source_checkout(repo)


def test_step10_teacher_training_operation_is_locked_to_real_trainer():
    from teacher_logit_reco.local_particle_residual_field.particle_view import scientific_tasks
    from teacher_logit_reco.local_particle_residual_field.particle_view.teacher_train import (
        train_particle_view_teacher,
    )

    assert (
        scientific_tasks._operation_callable("teacher_training")
        is train_particle_view_teacher
    )


def test_step10_runtime_zero_exit_without_result_is_a_hard_failure(tmp_path):
    _, manifest = _runtime_manifest(tmp_path)

    def no_result(command, **kwargs):
        return subprocess.CompletedProcess(command, 0)

    report = execute_runtime_node(
        manifest=manifest,
        node_id="pv00_source",
        runner=no_result,
    )
    assert report["exit_code"] == 1
    assert report["failed_count"] == 1
    assert report["records"][0]["action"] == "invalid_result"


def test_step10_runtime_resume_rejects_a_tampered_completed_artifact(tmp_path):
    _, manifest = _runtime_manifest(tmp_path)

    def successful_runner(command, **kwargs):
        environment = kwargs["env"]
        output = Path(environment["PARTICLE_VIEW_TASK_OUTPUT_DIR"])
        artifact = output / "payload.txt"
        artifact.write_text("original", encoding="utf-8")
        result = build_runtime_task_result(
            task_id=environment["PARTICLE_VIEW_TASK_ID"],
            artifacts=[
                {"path": str(artifact), "sha256": _sha("original")}
            ],
        )
        write_immutable_json(output / "task_result.json", result)
        return subprocess.CompletedProcess(command, 0)

    execute_runtime_node(
        manifest=manifest,
        node_id="pv00_source",
        runner=successful_runner,
    )
    source_task = next(
        task for task in manifest["tasks"] if task["node_id"] == "pv00_source"
    )
    artifact = Path(source_task["output_dir"]) / "payload.txt"
    artifact.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        execute_runtime_node(
            manifest=manifest,
            node_id="pv00_source",
            runner=successful_runner,
        )


def test_step10_scientific_catalog_requires_exact_coverage_and_executes_action(
    tmp_path,
    monkeypatch,
):
    registry = build_particle_view_registry(
        unified_split_manifest_sha256=_sha("scientific-manifest"),
        train_identity_sha256=_sha("scientific-train"),
        run_specs=[
            ParticleViewRunSpec(
                run_id="scientific_source",
                stage="source",
                scientific_role="source:preflight",
                selection_family="infrastructure",
                uses_labels=False,
                train_split=None,
            )
        ],
        campaign_id="scientific_fixture",
    )
    config = with_content_hash(
        {"contract": "scientific_factory_fixture_v1", "value": 7}
    )
    config_path = tmp_path / "factory_config.json"
    write_immutable_json(config_path, config)
    module_path = tmp_path / "scientific_fixture_factory.py"
    module_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "def build(**context):",
                "    artifact = Path(context['output_dir']) / 'source.json'",
                "    def action(*, path):",
                "        Path(path).parent.mkdir(parents=True, exist_ok=True)",
                "        Path(path).write_text('source-ok', encoding='utf-8')",
                "    return {",
                "        'kwargs': {'path': str(artifact)},",
                "        'artifact_paths': [str(artifact)],",
                "        'action': action,",
                "    }",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    task_specs = {
        "scientific_source": {
            "operation": "source_preflight",
            "factory": "scientific_fixture_factory:build",
            "factory_config_path": str(config_path),
            "factory_config_sha256": sha256_file(config_path),
        }
    }
    catalog = build_scientific_task_catalog(
        registry=registry,
        task_specs=task_specs,
    )
    assert validate_scientific_task_catalog(
        catalog,
        registry=registry,
    )["run_count"] == 1
    handler_commands = build_scientific_handler_commands(
        catalog=catalog,
        catalog_path=str(tmp_path / "scientific_catalog.json"),
    )
    assert set(handler_commands) == {"source"}
    assert "{registry_path}" in handler_commands["source"]
    assert "scripts/execute_particle_view_scientific_task.py" in (
        handler_commands["source"]
    )
    output = tmp_path / "task"
    result = execute_scientific_task(
        catalog=catalog,
        registry=registry,
        run_id="scientific_source",
        seed=101,
        task_id="scientific_source__seed_101",
        output_dir=output,
    )
    assert result["status"] == "complete"
    assert (output / "source.json").read_text(encoding="utf-8") == "source-ok"
    assert (output / "task_result.json").is_file()
    with pytest.raises(ValueError, match="coverage mismatch"):
        build_scientific_task_catalog(registry=registry, task_specs={})


def test_step10_scientific_paths_and_legacy_warnings_publish_non_gating(
    tmp_path,
    monkeypatch,
):
    registry = build_particle_view_registry(
        unified_split_manifest_sha256=_sha("warning-manifest"),
        train_identity_sha256=_sha("warning-train"),
        run_specs=[
            ParticleViewRunSpec(
                run_id="warning_source",
                stage="source",
                scientific_role="source:warning_fixture",
                selection_family="infrastructure",
                uses_labels=False,
                train_split=None,
            )
        ],
        campaign_id="warning_fixture",
    )
    config = with_content_hash(
        {"contract": "warning_factory_fixture_v1", "value": 1}
    )
    config_path = tmp_path / "warning_factory_config.json"
    write_immutable_json(config_path, config)
    module_path = tmp_path / "warning_fixture_factory.py"
    module_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "from teacher_logit_reco.local_particle_residual_field.particle_view import with_content_hash, write_immutable_json",
                "def build(**context):",
                "    output = Path(context['output_dir'])",
                "    artifact = output / 'confirmation_replica.json'",
                "    def action(*, path, bundle_path):",
                "        Path(path).parent.mkdir(parents=True, exist_ok=True)",
                "        write_immutable_json(path, with_content_hash({",
                "            'contract': 'warning_confirmation_fixture_v1',",
                "            'bundle_path': bundle_path,",
                "            'quality_warnings': [",
                "                {'warning_code': 'WARN_SCIENTIFIC', 'severity': 'scientific', 'declared_warning_threshold': 0.01},",
                "                {'warning_code': 'WARN_SCIENTIFIC_WARNING', 'severity': 'scientific_warning'},",
                "            ],",
                "        }))",
                "    return {",
                "        'kwargs': {'path': str(artifact), 'bundle_path': str(output / 'selected_confirmation_bundle.pt')},",
                "        'artifact_paths': [str(artifact)],",
                "        'action': action,",
                "    }",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    catalog = build_scientific_task_catalog(
        registry=registry,
        task_specs={
            "warning_source": {
                "operation": "source_preflight",
                "factory": "warning_fixture_factory:build",
                "factory_config_path": str(config_path),
                "factory_config_sha256": sha256_file(config_path),
            }
        },
    )
    attempt = tmp_path / ".warning_task.attempt_123"
    final = tmp_path / "warning_task"
    monkeypatch.setenv("PARTICLE_VIEW_TASK_OUTPUT_DIR", str(attempt))
    monkeypatch.setenv("PARTICLE_VIEW_TASK_FINAL_OUTPUT_DIR", str(final))
    result = execute_scientific_task(
        catalog=catalog,
        registry=registry,
        run_id="warning_source",
        seed=101,
        task_id="warning_source__seed_101",
        output_dir=attempt,
    )
    payload = json.loads(
        (attempt / "confirmation_replica.json").read_text(encoding="utf-8")
    )
    assert payload["bundle_path"] == str(
        final / "selected_confirmation_bundle.pt"
    )
    warnings = load_quality_warning_jsonl(attempt / "quality_warnings.jsonl")
    assert result["status"] == "complete"
    assert len(warnings) == 2
    assert {row["severity"] for row in warnings} == {"warning"}
    threshold = next(
        row for row in warnings if row["warning_code"] == "WARN_SCIENTIFIC"
    )
    assert threshold["warning_threshold"] == 0.01
    assert all(row["exit_code"] == 0 and row["non_gating"] for row in warnings)


def test_step10_graph_reconciles_every_registry_run_and_locked_tigris_contract():
    registry, graph = _graph()
    audit = validate_particle_view_production_graph(graph)
    reconciliation = reconcile_particle_view_production_graph(
        graph=graph, registry=registry
    )
    assert graph["contract"] == PARTICLE_VIEW_PRODUCTION_GRAPH_CONTRACT
    assert reconciliation["reconciled"]
    assert reconciliation["counts"]["declared_runs"] == len(LOGICAL_NODE_LAYOUT)
    assert reconciliation["counts"]["generated_run_assignments"] == len(
        LOGICAL_NODE_LAYOUT
    )
    assert reconciliation["counts"]["seed_expanded_replicas"] == 15
    assert reconciliation["counts"]["declared_seed_replicas"] == 15
    assert reconciliation["counts"]["generated_seed_replicas"] == 15
    assert reconciliation["invalid_parent_edges"] == []
    assert audit["topological_order"] == [
        node_id for node_id, _, _ in LOGICAL_NODE_LAYOUT
    ]
    for node in graph["nodes"]:
        assert node["tigris"] == {
            "account": "reu-aisocial",
            "conda_env": "atlas_kd_tigris",
            "conda_base": "/home/ryreu/miniforge3-aarch64",
            "python_no_user_site": "1",
        }
        assert node["scientific_warning_dependency"] is False
        assert node["sbatch_script"] == NODE_WRAPPERS[node["node_id"]]


def test_step10_reconciliation_rejects_registry_parent_edges_that_run_backwards():
    registry = build_particle_view_registry(
        unified_split_manifest_sha256=_sha("manifest"),
        train_identity_sha256=_sha("train"),
        run_specs=[
            ParticleViewRunSpec(
                run_id="early_baseline",
                stage="baseline",
                scientific_role="invalid_early_child",
                selection_family="infrastructure",
                parent_run_ids=("late_predictor",),
            ),
            ParticleViewRunSpec(
                run_id="late_predictor",
                stage="predictor",
                scientific_role="invalid_late_parent",
                selection_family="infrastructure",
            ),
        ],
        campaign_id="step10_backward_parent",
    )
    catalog = {
        node_id: ["python", "-u", "worker.py", "--node", node_id]
        for node_id, _, _ in LOGICAL_NODE_LAYOUT
    }
    with pytest.raises(ValueError, match="did not reconcile"):
        build_particle_view_production_graph(
            registry=registry,
            artifact_root="/campaign/pview",
            source_commit="a" * 40,
            command_catalog=catalog,
        )


def test_step10_clean_start_submission_uses_afterok_and_no_warning_dependency():
    _, graph = _graph()
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout=f"{12000 + len(calls)}\n", stderr=""
        )

    ledger = submit_particle_view_graph(
        graph=graph,
        graph_path="/campaign/graph.json",
        mode="execute",
        runner=runner,
    )
    assert ledger["submitted_count"] > 11
    assert any(
        any(token.startswith("--array=") for token in command)
        for command in calls
    )
    assert ledger["scientific_warning_dependency_count"] == 0
    assert "--dependency=afterok:12001" in calls[1]
    assert all(
        not any("warning" in token.lower() for token in command)
        for command in calls
    )
    assert all(
        any("PARTICLE_VIEW_NODE_ID=" in token for token in command)
        for command in calls
    )


def test_step10_execute_publishes_progress_before_a_later_sbatch_failure():
    _, graph = _graph()
    snapshots = []
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if len(calls) == 3:
            return subprocess.CompletedProcess(
                command, 1, stdout="", stderr="synthetic submission failure"
            )
        return subprocess.CompletedProcess(
            command, 0, stdout=f"{13000 + len(calls)}\n", stderr=""
        )

    with pytest.raises(RuntimeError, match="synthetic submission failure"):
        submit_particle_view_graph(
            graph=graph,
            graph_path="/campaign/graph.json",
            mode="execute",
            runner=runner,
            progress_callback=lambda records: snapshots.append(list(records)),
        )
    assert [len(snapshot) for snapshot in snapshots] == [1, 2]
    assert snapshots[-1][-1]["node_id"] == "pv00_source"
    assert snapshots[-1][-1]["job_id"] == "13002"


def test_step10_logical_recovery_reuses_completed_and_active_jobs():
    _, graph = _graph()
    node_ids = [node_id for node_id, _, _ in LOGICAL_NODE_LAYOUT]
    existing = {
        node_ids[0]: {"job_id": "9001", "state": "COMPLETED"},
        node_ids[1]: {"job_id": "9002", "state": "RUNNING"},
        node_ids[2]: {"job_id": "9003", "state": "FAILED"},
    }
    with pytest.raises(ValueError, match="no authenticated node completion"):
        plan_particle_view_submissions(
            graph=graph,
            graph_path="/campaign/graph.json",
            existing_jobs=existing,
        )


def test_step10_active_recovery_is_disabled_without_progress_ledger():
    _, graph = _graph()
    node_id = LOGICAL_NODE_LAYOUT[0][0]
    with pytest.raises(ValueError, match="active Slurm jobs cannot be reused"):
        plan_particle_view_submissions(
            graph=graph,
            graph_path="/campaign/graph.json",
            existing_jobs={
                node_id: {"job_id": "9002", "state": "RUNNING"}
            },
        )


def test_step10_structured_warnings_exit_zero_and_do_not_block_completion():
    _, graph = _graph()
    warning = build_quality_warning(
        warning_code="WARN_NEGATIVE_STACK_GAIN",
        severity="high",
        graph_node="pv08_sealed_stack_fusion",
        configuration_id="Dview",
        seed=101,
        split="stack_val",
        observed_value=-0.001,
        reference_value=0.0,
        warning_threshold=0.0,
        interpretation="The selected model has a negative stack gain.",
        suggested_diagnostic="Inspect seed and per-class rows.",
        supporting_artifacts=[
            {"path": "reports/Dview.json", "sha256": _sha("report")}
        ],
        source_commit="b" * 40,
        timestamp_utc="2026-07-26T00:00:00Z",
    )
    summary, markdown = aggregate_quality_warnings([warning])
    assert warning["exit_code"] == 0
    assert warning["non_gating"] is True
    assert summary["aggregate_exit_code"] == 0
    assert summary["counts_by_code"] == {"WARN_NEGATIVE_STACK_GAIN": 1}
    assert "never gate submission" in markdown
    completion = build_node_completion(
        graph=graph,
        node_id="pv08_sealed_stack_fusion",
        output_artifacts=[
            {"path": "reports/Dview.json", "sha256": _sha("report")}
        ],
        warning_sha256=[warning["content_hash"]],
    )
    assert completion["integrity_status"] == "complete"
    assert completion["exit_code"] == 0
    assert completion["quality_warning_count"] == 1


def test_step10_warning_jsonl_is_immutable_and_accepts_empty_streams(tmp_path):
    warning = build_quality_warning(
        warning_code="WARN_WEAK_OR_NONPOSITIVE_RECOVERY",
        severity="warning",
        graph_node="pv05_predictor_loss_packs",
        configuration_id="weak",
        seed=101,
        split="model_val_select",
        observed_value=0.0,
        reference_value=0.1,
        warning_threshold=0.01,
        interpretation="Recovery was weak.",
        suggested_diagnostic="Inspect the representation losses.",
        supporting_artifacts=[
            {"path": "weak.json", "sha256": _sha("weak")}
        ],
        source_commit="c" * 40,
        timestamp_utc="2026-07-26T00:00:00Z",
    )
    path = tmp_path / "quality_warnings.jsonl"
    write_quality_warning_jsonl(path, [warning])
    assert load_quality_warning_jsonl(path) == [warning]
    write_quality_warning_jsonl(path, [warning])
    with pytest.raises(FileExistsError):
        write_quality_warning_jsonl(path, [])
    empty = tmp_path / "empty.jsonl"
    write_quality_warning_jsonl(empty, [])
    assert load_quality_warning_jsonl(empty) == []


def test_step10_miniature_filesystem_rehearsal_and_warning_recovery(tmp_path):
    command = [
        str(Path(".venv/Scripts/python.exe")),
        "scripts/rehearse_particle_view_campaign.py",
        "--output-dir",
        str(tmp_path),
    ]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(
        (tmp_path / "rehearsal_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "PASS"
    assert report["logical_node_count"] == 11
    assert report["completion_count"] == 11
    assert report["clean_start_planned_count"] > 11
    assert report["recovery_reused_completed_count"] == 6
    assert report["recovery_planned_count"] > 5
    assert report["warning_count"] == 1
    assert report["warning_did_not_block_descendants"] is True
    assert len(list((tmp_path / "node_completions").glob("*.json"))) == 11
    assert len(
        list((tmp_path / "quality_warnings").glob("*/quality_warnings.jsonl"))
    ) == 11
    assert (tmp_path / "quality_warning_summary.md").is_file()


def test_step10_all_tigris_wrappers_use_full_account_miniforge_and_no_user_site():
    common = Path("sbatch/common.sh").read_text(encoding="utf-8")
    assert 'export PYTHONDONTWRITEBYTECODE' in common
    wrappers = set(NODE_WRAPPERS.values())
    wrappers.add("sbatch/run_particle_view_controls.sh")
    wrappers.add("sbatch/submit_particle_view_full_pilot.sh")
    for name in sorted(wrappers):
        text = Path(name).read_text(encoding="utf-8")
        if name != "sbatch/submit_particle_view_full_pilot.sh":
            assert "#SBATCH --account=reu-aisocial" in text
        assert "export PYTHONNOUSERSITE=1" in text
        assert "atlas_kd_tigris" in text
        assert "miniforge3-aarch64" in text
        assert "reu-aisoc\n" not in text


def test_step10_print_and_dry_run_never_call_sbatch():
    _, graph = _graph()

    def forbidden(*args, **kwargs):
        raise AssertionError("runner must not be called")

    for mode in ("dry_run", "print_only"):
        ledger = submit_particle_view_graph(
            graph=graph,
            graph_path="/campaign/graph.json",
            mode=mode,
            runner=forbidden,
        )
        assert ledger["submitted_count"] == 0
        assert ledger["planned_submit_count"] > 11


def test_step10_one_command_bootstrap_builds_graph_reconciliation_and_ledger(
    tmp_path,
):
    registry = _registry()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    catalog = {
        node_id: ["python", "-c", f"print('{node_id}')"]
        for node_id, _, _ in LOGICAL_NODE_LAYOUT
    }
    catalog_path = tmp_path / "commands.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    artifact_root = tmp_path / "campaign"
    result = subprocess.run(
        [
            str(Path(".venv/Scripts/python.exe")),
            "scripts/submit_particle_view_full_pilot.py",
            "--registry",
            str(registry_path),
            "--command-catalog",
            str(catalog_path),
            "--artifact-root",
            str(artifact_root),
            "--source-commit",
            "d" * 40,
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (artifact_root / "preflight" / "production_graph.json").is_file()
    assert (artifact_root / "preflight" / "graph_reconciliation.json").is_file()
    ledgers = list((artifact_root / "job_ledgers").glob("*.json"))
    assert len(ledgers) == 1
    ledger = json.loads(ledgers[0].read_text(encoding="utf-8"))
    assert ledger["mode"] == "dry_run"
    assert ledger["planned_submit_count"] > 11


def test_step10_low_data_registry_cli_and_unified_submission_bootstrap(tmp_path):
    parent = miniature_parent_manifest()
    unified = build_unified_split_manifest(
        parent,
        config=miniature_split_config(),
    )
    unified_path = tmp_path / "unified.json"
    unified_path.write_text(json.dumps(unified), encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    inventory_path = tmp_path / "inventory.json"
    build_result = subprocess.run(
        [
            str(Path(".venv/Scripts/python.exe")),
            "scripts/build_particle_view_low_data_registry.py",
            "--unified-manifest",
            str(unified_path),
            "--output",
            str(registry_path),
            "--inventory-output",
            str(inventory_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build_result.returncode == 0, build_result.stderr
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert len(registry["runs"]) == 230
    assert inventory["seed_expanded_replica_count"] == 300

    catalog = {
        node_id: ["python", "-c", f"print('{node_id}')"]
        for node_id, _, _ in LOGICAL_NODE_LAYOUT
    }
    catalog_path = tmp_path / "commands.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    artifact_root = tmp_path / "campaign"
    submit_result = subprocess.run(
        [
            str(Path(".venv/Scripts/python.exe")),
            "scripts/submit_particle_view_full_pilot.py",
            "--unified-manifest",
            str(unified_path),
            "--command-catalog",
            str(catalog_path),
            "--artifact-root",
            str(artifact_root),
            "--source-commit",
            "e" * 40,
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert submit_result.returncode == 0, submit_result.stderr
    preflight = artifact_root / "preflight"
    assert (preflight / "low_data_campaign_registry.json").is_file()
    assert (preflight / "low_data_campaign_inventory.json").is_file()
    reconciliation = json.loads(
        (preflight / "graph_reconciliation.json").read_text(encoding="utf-8")
    )
    assert reconciliation["reconciled"] is True
    assert reconciliation["counts"]["declared_runs"] == 230

    handler_commands = {
        category: [
            "python",
            "role_worker.py",
            "--run-id",
            "{run_id}",
            "--seed",
            "{seed}",
            "--output-dir",
            "{task_output_dir}",
        ]
        for category in EXPECTED_LOW_DATA_CATEGORY_COUNTS
    }
    handler_path = tmp_path / "handler_commands.json"
    handler_path.write_text(json.dumps(handler_commands), encoding="utf-8")
    runtime_root = tmp_path / "runtime_campaign"
    runtime_result = subprocess.run(
        [
            str(Path(".venv/Scripts/python.exe")),
            "scripts/submit_particle_view_full_pilot.py",
            "--unified-manifest",
            str(unified_path),
            "--handler-commands",
            str(handler_path),
            "--artifact-root",
            str(runtime_root),
            "--source-commit",
            "f" * 40,
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert runtime_result.returncode == 0, runtime_result.stderr
    runtime_preflight = runtime_root / "preflight"
    execution = json.loads(
        (runtime_preflight / "runtime_execution_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    graph = json.loads(
        (runtime_preflight / "production_graph.json").read_text(
            encoding="utf-8"
        )
    )
    assert execution["seed_expanded_task_count"] == 300
    assert all(
        "scripts/run_particle_view_campaign_node.py" in node["command"]
        for node in graph["nodes"]
    )
