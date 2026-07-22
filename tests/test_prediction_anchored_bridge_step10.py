from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from teacher_logit_reco.local_particle_residual_field import (
    PAIRED_SEED_IDS,
    TIGRIS_ACCOUNT,
    build_allocation_launch_manifest,
    build_campaign_registry,
    build_campaign_reservations,
    build_prediction_anchored_job_ledger,
    build_prediction_anchored_tigris_graph,
    record_registry_measurements,
    rehearse_prediction_anchored_campaign_cpu,
    render_tigris_sbatch_commands,
    require_production_ready,
    simulate_prediction_anchored_scheduler,
    validate_prediction_anchored_tigris_graph,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (
    with_content_hash,
    write_immutable_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH_FILES = (
    "run_prepare_prediction_anchored_bridge_ram.sh",
    "run_train_prediction_anchored_bridge_consumer.sh",
    "run_cache_prediction_anchored_bridge_logits.sh",
    "run_train_prediction_anchored_bridge_reconstructor.sh",
    "submit_prediction_anchored_bridge_pilot.sh",
)


def _registry(*, alternate: bool = False, measured: bool = True):
    value = build_campaign_registry(alternate_teacher_valid=alternate)
    if measured:
        value = record_registry_measurements(
            value, {row["canonical_run_id"]: 1024 for row in value["runs"]}
        )
    return value


def _reservations(registry, tmp_path: Path, *, budget_gib: int = 5):
    readiness = require_production_ready(
        registry,
        fixed_persistent_bytes=4096,
        selected_budget_bytes=budget_gib * 1024**3,
    )
    return build_campaign_reservations(
        registry,
        production_readiness=readiness,
        fixed_parent_artifacts={
            "r0": {"sha256": "1" * 64, "size_bytes": 512, "path": tmp_path / "r0.pt"},
            "consumer": {"sha256": "2" * 64, "size_bytes": 512, "path": tmp_path / "t10.pt"},
            "metadata": {"sha256": "3" * 64, "size_bytes": 512, "path": tmp_path / "metadata"},
        },
    )


def _graph(tmp_path: Path, *, alternate: bool = False, budget_gib: int = 5):
    registry = _registry(alternate=alternate)
    return build_prediction_anchored_tigris_graph(
        registry,
        reservations=_reservations(registry, tmp_path, budget_gib=budget_gib),
        artifact_root=str(tmp_path / "campaign"),
    )


def _by_id(graph):
    return {row["node_id"]: row for row in graph["nodes"]}


def test_registry_rendered_graph_covers_54_46_45_and_conditional_skip(tmp_path):
    graph = _graph(tmp_path)
    validation = validate_prediction_anchored_tigris_graph(graph)
    assert validation["configuration_count"] == 54
    assert validation["reconstruction_breadth_count"] == 46
    assert validation["post_teacher_configuration_count"] == 45
    assert graph["runnable_configuration_count"] == 53
    assert graph["covered_runnable_configuration_count"] == 53
    assert graph["conditional_skips"] == [
        {
            "run_id": "D10_TALT_A3",
            "status": "SKIPPED_INVALID_PARENT",
            "reason": "conditional parent is invalid",
        }
    ]
    covered = [run_id for node in graph["nodes"] for run_id in node["configuration_run_ids"]]
    assert len(covered) == len(set(covered)) == 53


def test_alternate_teacher_adds_binding_cache_and_exactly_54_runnable_rows(tmp_path):
    graph = _graph(tmp_path, alternate=True)
    assert graph["runnable_configuration_count"] == 54
    assert graph["conditional_skips"] == []
    nodes = _by_id(graph)
    assert "b5_cache_alternate" in nodes
    assert nodes["b5_release_postteacher"]["dependencies"] == [
        "b5_cache_primary", "b5_cache_all50", "b5_cache_n3", "b5_cache_alternate"
    ]


def test_graph_refuses_unmeasured_and_accepts_measured_five_and_six_gib_modes(tmp_path):
    unmeasured = _registry(measured=False)
    fake_reservations = with_content_hash(
        {
            "contract": "prediction_anchored_step9_campaign_reservations_v1",
            "registry_sha256": unmeasured["content_hash"],
            "projected_persistent_bytes": 1,
            "selected_budget_bytes": 5 * 1024**3,
            "run_reservations_bytes": {
                row["canonical_run_id"]: 1 for row in unmeasured["runs"]
            },
            "fixed_storage_reserved_bytes": 1,
        }
    )
    with pytest.raises(PermissionError, match="UNMEASURED"):
        build_prediction_anchored_tigris_graph(
            unmeasured, reservations=fake_reservations, artifact_root=str(tmp_path)
        )
    five = _graph(tmp_path / "five", budget_gib=5)
    six = _graph(tmp_path / "six", budget_gib=6)
    assert five["selected_budget_bytes"] == 5 * 1024**3
    assert six["selected_budget_bytes"] == 6 * 1024**3
    assert five["projected_persistent_bytes"] <= five["selected_budget_bytes"]
    assert six["projected_persistent_bytes"] <= six["selected_budget_bytes"]


def test_packing_is_single_node_shared_source_paired3_and_median_only(tmp_path):
    graph = _graph(tmp_path)
    for node in graph["nodes"]:
        assert node["resources"]["nodes"] == 1
        assert node["resources"]["host_memory_gib"] >= 64
        assert node["allocation_packing"]["allocation_leader_rank"] == 0
        assert node["allocation_packing"]["one_persistent_source_open_by_leader"] is True
        assert node["allocation_packing"]["shared_allocation_ram_ledger"] is True
        assert node["allocation_packing"]["cross_allocation_resume"] is False
        assert node["allocation_packing"]["preemption_policy"] == "restart_whole_configuration_pack"
        assert node["persistent_dense_field_output_paths"] == []
        if node["configuration_run_ids"]:
            assert node["paired_seed_ids"] == list(PAIRED_SEED_IDS)
            assert node["publication_policy"] == "metrics_all_seeds__weights_ordered_median_only"
    b6_packs = [node for node in graph["nodes"] if node["node_id"].startswith("b6_") and "pack" in node["node_id"]]
    assert all(1 <= len(node["configuration_run_ids"]) <= 4 for node in b6_packs)
    assert all(len({node["teacher_namespace"]}) == 1 for node in b6_packs)


def test_selected_consumer_and_sealed_confirmations_are_ordered_fail_closed(tmp_path):
    graph = _graph(tmp_path)
    nodes = _by_id(graph)
    assert nodes["b4_confirm_consumer"]["requires_selected_consumer"] is False
    assert nodes["b5_bind_teachers"]["dependencies"] == ["b4_confirm_consumer"]
    assert nodes["b5_bind_teachers"]["requires_selected_consumer"] is True
    assert nodes["b6_confirm_deployable"]["dependencies"] == ["b6_aggregate_select_deployable"]
    assert nodes["final_test_hlt_only"]["protected_final_test"] is True
    assert graph["final_test_automatic_submission"] is False
    assert graph["final_test_hlt_only"] is True
    assert graph["final_test_privileged_environment_scrub_required"] is True
    assert graph["stage1b_guessed_consumer_allowed"] is False


def test_allocation_preflight_enforces_leader_memory_usersite_and_selection(tmp_path):
    graph = _graph(tmp_path)
    base = {
        "SLURM_NNODES": "1",
        "SLURM_PROCID": "0",
        "SLURM_JOB_ID": "12345",
        "SLURM_MEM_PER_NODE": str(512 * 1024),
        "PYTHONNOUSERSITE": "1",
    }
    selected = with_content_hash(
        {"contract": "selected_bridge_consumer_v2", "status": "CONFIRMED_LOCKED"}
    )
    launch = build_allocation_launch_manifest(
        graph,
        node_id="b5_bind_teachers",
        environment=base,
        ram_root=str(tmp_path / "dry-ram"),
        selected_consumer=selected,
        dry_run=True,
    )
    assert launch["allocation_leader_rank"] == 0
    assert launch["one_source_open_by_rank0"] is True
    assert launch["all_workers_join_same_ram_ledger"] is True
    assert launch["persistent_dense_field_output_paths"] == []
    assert "scripts/cache_prediction_anchored_bridge_logits.py" in launch["connected_command_surfaces"]
    with pytest.raises(PermissionError, match="guessing is forbidden"):
        build_allocation_launch_manifest(
            graph, node_id="b5_bind_teachers", environment=base,
            ram_root=str(tmp_path), dry_run=True,
        )
    for key, value, match in (
        ("SLURM_NNODES", "2", "NNODES=1"),
        ("SLURM_PROCID", "1", "leader rank 0"),
        ("SLURM_MEM_PER_NODE", "1", "below requested"),
        ("PYTHONNOUSERSITE", "0", "PYTHONNOUSERSITE=1"),
    ):
        changed = {**base, key: value}
        with pytest.raises(PermissionError, match=match):
            build_allocation_launch_manifest(
                graph, node_id="b0_validate_preflight", environment=changed,
                ram_root=str(tmp_path), dry_run=True,
            )


def test_scheduler_success_consumer_failure_confirmation_failure_and_preemption(tmp_path):
    graph = _graph(tmp_path)
    success = simulate_prediction_anchored_scheduler(graph)
    assert success["statuses"]["b6_report_export_reload"] == "COMPLETED"
    assert success["statuses"]["final_test_hlt_only"] == "NOT_SUBMITTED_PROTECTED"
    consumer_failure = simulate_prediction_anchored_scheduler(
        graph, requested_outcomes={"b4_confirm_consumer": "FAILED"}
    )
    assert consumer_failure["cache_started"] is False
    assert consumer_failure["b6_training_started"] is False
    assert consumer_failure["statuses"]["b5_bind_teachers"] == "DEPENDENCY_NEVER_SATISFIED"
    deploy_failure = simulate_prediction_anchored_scheduler(
        graph, requested_outcomes={"b6_confirm_deployable": "FAILED"}
    )
    assert deploy_failure["statuses"]["b6_report_export_reload"] == "DEPENDENCY_NEVER_SATISFIED"
    first_pack = next(node["node_id"] for node in graph["nodes"] if node["node_id"].startswith("b6_") and "pack" in node["node_id"])
    preempted = simulate_prediction_anchored_scheduler(
        graph, requested_outcomes={first_pack: "PREEMPTED"}
    )
    row = next(item for item in preempted["rows"] if item["node_id"] == first_pack)
    assert row["status"] == "PREEMPTED_RESTART_WHOLE_CONFIGURATION_PACK"
    assert row["partial_replica_resume_allowed"] is False
    assert row["restart_scope"] == "whole_configuration_pack"


def test_render_and_local_cpu_rehearsal_never_submit(tmp_path):
    graph = _graph(tmp_path)
    commands = render_tigris_sbatch_commands(graph)
    assert commands["submission_executed"] is False
    assert all(command["submission_executed"] is False for command in commands["commands"])
    assert all("final_test_hlt_only" != command["node_id"] for command in commands["commands"])
    assert all("--account=reu-aisocial" in command["argv"] for command in commands["commands"])
    assert all("--nodes=1" in command["argv"] for command in commands["commands"])
    rehearsal = rehearse_prediction_anchored_campaign_cpu(graph)
    assert rehearsal["submission_executed"] is False
    assert rehearsal["final_test_submitted"] is False
    assert rehearsal["dense_field_output_paths_present"] is False
    assert "PREDICTION_ANCHORED_EXECUTE=1" in rehearsal["explicit_execution_command"]


def test_dry_job_ledger_records_every_afterok_dependency(tmp_path):
    graph = _graph(tmp_path)
    submitted = [node for node in graph["nodes"] if not node["protected_final_test"]]
    ids = {node["node_id"]: f"DRYRUN_{index}" for index, node in enumerate(submitted, start=1)}
    ledger = build_prediction_anchored_job_ledger(
        graph, job_ids=ids, include_final_test=False
    )
    assert ledger["job_count"] == len(submitted)
    assert ledger["immutable_after_submission"] is True
    for row in ledger["jobs"]:
        assert row["dependency_job_ids"] == [ids[value] for value in row["dependency_node_ids"]]


def test_submit_cli_is_non_submitting_without_explicit_execute(tmp_path):
    graph_path = tmp_path / "graph.json"
    write_immutable_json(graph_path, _graph(tmp_path))
    completed = subprocess.run(
        [sys.executable, "scripts/submit_prediction_anchored_bridge_graph.py", "--graph", str(graph_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["submission_executed"] is False
    assert payload["rendered"]["submission_executed"] is False
    assert "submitted_jobs" not in payload


def test_execute_refuses_unconfigured_scientific_executors_before_sbatch(tmp_path):
    graph_path = tmp_path / "graph.json"
    write_immutable_json(graph_path, _graph(tmp_path))
    environment = os.environ.copy()
    for name in (
        "PAB_CONSUMER_EXECUTOR",
        "PAB_RECONSTRUCTOR_EXECUTOR",
        "PAB_TEACHER_FORWARD_EXECUTOR",
        "PAB_DEPLOYABLE_EXPORT_EXECUTOR",
        "PAB_FINAL_TEST_EXECUTOR",
    ):
        environment.pop(name, None)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/submit_prediction_anchored_bridge_graph.py",
            "--graph", str(graph_path),
            "--execute",
            "--ledger-output", str(tmp_path / "ledger.json"),
            "--sbatch-bin", "definitely-not-a-real-sbatch",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, env=environment,
    )
    assert completed.returncode != 0
    assert "PAB_CONSUMER_EXECUTOR" in completed.stderr
    assert "definitely-not-a-real-sbatch" not in completed.stderr


def test_required_clis_help_and_tigris_shell_contracts():
    clis = (
        "audit_prediction_anchored_bridge_inputs.py",
        "write_prediction_anchored_bridge_recipe.py",
        "train_prediction_anchored_bridge_consumer.py",
        "select_prediction_anchored_bridge_consumer.py",
        "cache_prediction_anchored_bridge_logits.py",
        "train_prediction_anchored_bridge_reconstructor.py",
        "run_prediction_anchored_bridge_campaign.py",
        "evaluate_prediction_anchored_bridge_campaign.py",
        "submit_prediction_anchored_bridge_graph.py",
        "run_prediction_anchored_bridge_allocation.py",
        "inspect_prediction_anchored_bridge_graph.py",
    )
    for name in clis:
        completed = subprocess.run(
            [sys.executable, f"scripts/{name}", "--help"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert completed.returncode == 0, (name, completed.stderr)
    for name in SBATCH_FILES:
        text = (REPO_ROOT / "sbatch" / name).read_text(encoding="utf-8")
        assert "PYTHONNOUSERSITE=1" in text or "prediction_anchored_bridge_common.sh" in text
        assert "reu-aisocial" in text or name == "submit_prediction_anchored_bridge_pilot.sh"
        if name != "submit_prediction_anchored_bridge_pilot.sh":
            assert "#SBATCH --nodes=1" in text
            assert "#SBATCH --mem=" in text
        assert "reu-aisoc\n" not in text
    common = (REPO_ROOT / "sbatch" / "prediction_anchored_bridge_common.sh").read_text(encoding="utf-8")
    assert "Only allocation leader rank 0" in common
    assert "restart" in common.lower() and "whole" in common.lower()
    assert "/dev/shm/prediction_anchored_bridge/" in common
    prepare = (REPO_ROOT / "sbatch" / "run_prepare_prediction_anchored_bridge_ram.sh").read_text(encoding="utf-8")
    assert "fresh_run env" in prepare and "-u PAB_OFFLINE_NPZ" in prepare
    bash = shutil.which("bash") if os.name != "nt" else None
    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    if os.name == "nt" and git_bash.is_file():
        bash = str(git_bash)
    if bash:
        for name in (*SBATCH_FILES, "prediction_anchored_bridge_common.sh"):
            subprocess.run([bash, "-n", str(REPO_ROOT / "sbatch" / name)], check=True)
