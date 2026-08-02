from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import pytest

from scripts.run_retb_streamed_smoke_phase import _tiny_gpu_step

from teacher_logit_reco.relation_expert_token_bridge.production import build_production_graph
from teacher_logit_reco.relation_expert_token_bridge.contracts import write_immutable_json
from teacher_logit_reco.relation_expert_token_bridge.provenance import source_snapshot
from teacher_logit_reco.relation_expert_token_bridge.storage import (
    build_storage_measurements, miniature_storage_measurements,
)
from teacher_logit_reco.relation_expert_token_bridge.streamed_execution import (
    DURABLE_CLASSES, FULL_STREAMED_PROFILE, SMOKE_PHASES,
    STAGE_ARTIFACT_POLICY,
    STREAMED_SMOKE_PROFILE, TRANSIENT_CLASSES,
    build_streamed_execution_profile, build_streamed_smoke_plan,
    build_streamed_smoke_phase_control_evidence,
    build_streamed_storage_projection, build_task_lifecycle_receipt,
    task_local_workspace, validate_streamed_execution_profile,
    validate_streamed_smoke_plan, validate_streamed_storage_projection,
    validate_task_lifecycle_receipt,
)


SOURCE = {"source_commit": "a" * 40, "source_status_sha256": "b" * 64}


def _graph(*, miniature: bool, profile: str):
    return build_production_graph(
        campaign_root="/campaign/streamed", campaign_id="streamed",
        source_commit="a" * 40, source_status_sha256="b" * 64,
        storage_measurements_sha256="c" * 64,
        miniature=miniature, execution_profile=profile,
    )


def test_full_and_smoke_graph_profiles_are_distinct_and_fail_closed() -> None:
    full = _graph(miniature=False, profile=FULL_STREAMED_PROFILE)
    smoke = _graph(miniature=True, profile=STREAMED_SMOKE_PROFILE)
    assert full["scientific_results_allowed"] is True
    assert full["task_local_stage_d_through_n_intermediates"] is True
    assert smoke["scientific_results_allowed"] is False
    assert smoke["smoke_evidence_eligible_for_production"] is False
    with pytest.raises(ValueError):
        _graph(miniature=False, profile=STREAMED_SMOKE_PROFILE)


def test_streamed_profile_freezes_transient_and_durable_classes() -> None:
    profile = build_streamed_execution_profile(
        campaign_id="c", campaign_root="/campaign/c", source=SOURCE,
        profile=FULL_STREAMED_PROFILE,
    )
    validate_streamed_execution_profile(profile)
    assert profile["transient_artifact_classes"] == list(TRANSIENT_CLASSES)
    assert profile["durable_artifact_classes"] == list(DURABLE_CLASSES)
    assert set(profile["stage_d_through_n_artifact_policy"]) == set("DEFGHIJKLMN")
    assert profile["stage_d_through_n_artifact_policy"] == {
        stage: {kind: list(values) for kind, values in policy.items()}
        for stage, policy in STAGE_ARTIFACT_POLICY.items()
    }
    assert all(
        set(policy) == {"transient", "rolling_authenticated", "durable"}
        and all(policy.values())
        for policy in STAGE_ARTIFACT_POLICY.values()
    )
    assert profile["scientific_underperformance_blocks_continuation"] is False
    tampered = dict(profile)
    tampered["cleanup_on_failure"] = False
    with pytest.raises(ValueError):
        validate_streamed_execution_profile(tampered)


def test_task_local_workspace_is_removed_on_success_and_failure(tmp_path: Path) -> None:
    env = {"RETB_STREAM_ROOT": str(tmp_path)}
    with task_local_workspace(
        campaign_id="campaign", node_id="node", task_index=0,
        environment=env,
    ) as success:
        (success / "large.bin").write_bytes(b"x" * 32)
    assert not success.exists()
    with pytest.raises(RuntimeError, match="fixture"):
        with task_local_workspace(
            campaign_id="campaign", node_id="node", task_index=1,
            environment=env,
        ) as failed:
            (failed / "large.bin").write_bytes(b"x" * 32)
            raise RuntimeError("fixture")
    assert not failed.exists()


def test_lifecycle_receipt_hashes_outputs_and_rejects_tampering(tmp_path: Path) -> None:
    output = tmp_path / "model.pt"
    output.write_bytes(b"checkpoint")
    receipt = build_task_lifecycle_receipt(
        campaign_spec_sha256="a" * 64, task_manifest_sha256="b" * 64,
        node_id="train", task_index=2, status="completed",
        workspace_parent=tmp_path, workspace_removed=True,
        output_paths=[output], source=SOURCE,
    )
    validate_task_lifecycle_receipt(receipt)
    output.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="output bytes"):
        validate_task_lifecycle_receipt(receipt)
    output.write_bytes(b"checkpoint")
    tampered = dict(receipt)
    tampered["workspace_removed_before_publication"] = False
    with pytest.raises(ValueError):
        validate_task_lifecycle_receipt(tampered)


def test_compact_smoke_has_exact_a_to_n_coverage_and_under_30_allocations() -> None:
    plan = build_streamed_smoke_plan(
        campaign_spec_sha256="a" * 64,
        production_graph_sha256="b" * 64,
        campaign_id="c", source=SOURCE,
    )
    validate_streamed_smoke_plan(plan)
    assert plan["physical_allocation_count"] == 18
    assert plan["physical_allocation_count"] <= 30
    assert set(row["stage"] for row in SMOKE_PHASES) == set("ABCDEFGHIJKLMN")
    assert plan["complete_scientific_grid_queued"] is False
    assert plan["production_evidence_eligible"] is False


def test_compact_smoke_locks_semantics_and_final_test_without_performance_gate() -> None:
    common = {
        "previous_phase_sha256": "a" * 64,
        "split_manifest_sha256": "b" * 64,
        "production_graph_sha256": "c" * 64,
        "execution_logit_sha256": "d" * 64,
    }
    semantics = build_streamed_smoke_phase_control_evidence(
        phase_id="k_semantics", **common
    )
    assert len(semantics["semantic_evidence_bundle_sha256"]) == 64
    assert semantics["scientific_underperformance_blocks_continuation"] is False
    sealed = build_streamed_smoke_phase_control_evidence(
        phase_id="n_sealed_final", **common
    )
    assert sealed["final_test_seal"]["both_locks_present_before_inference"] is True
    assert sealed["final_test_seal"]["oracle_inputs_consumed"] is False
    tampered = dict(sealed)
    tampered["final_test_seal"] = {
        **sealed["final_test_seal"], "oracle_inputs_consumed": True,
    }
    assert tampered != build_streamed_smoke_phase_control_evidence(
        phase_id="n_sealed_final", **common
    )


@pytest.mark.parametrize(
    ("phase_id", "component"),
    (
        ("b_expert", "retb_token_only_expert_head"),
        ("c_fusion", "retb_token_transformer_fusion"),
        ("g_predictor", "retb_a1_resmlp_token_predictor"),
        ("j_consumer", "retb_unrestricted_hlt_fusion"),
    ),
)
def test_compact_smoke_executes_registered_retb_components(
    monkeypatch: pytest.MonkeyPatch, phase_id: str, component: str
) -> None:
    monkeypatch.setenv("RETB_SMOKE_ALLOW_CPU", "1")
    result = _tiny_gpu_step(7300, require_cuda=False, phase_id=phase_id)
    assert result["architecture_component"] == component
    assert result["gradient_step_completed"] is True
    assert result["masking_exact"] is True
    assert result["token_shape"] == [4, 128]


def test_storage_projection_separates_persistent_and_transient() -> None:
    projection = build_streamed_storage_projection(
        storage_measurements_sha256="a" * 64,
        persistent_classes={"checkpoints": 100, "reports": 20},
        rolling_classes={"recomputable_cache": 15},
        transient_classes={"tokens": 400, "targets": 300},
        maximum_concurrent_allocations=64,
        serialized_reserve_bytes=10,
        available_storage_bytes=200,
        source=SOURCE,
    )
    validate_streamed_storage_projection(projection)
    assert projection["persistent_peak_bytes"] == 145
    assert projection["per_allocation_transient_peak_bytes"] == 400
    assert projection["cluster_transient_peak_bytes"] == 25_600
    assert projection["persistent_storage_admitted"] is True


def test_storage_projection_uses_lifetime_peak_not_sum() -> None:
    projection = build_streamed_storage_projection(
        storage_measurements_sha256="a" * 64,
        persistent_classes={"durable": 100},
        rolling_classes={"stage_f_to_l": 70, "stage_m": 90},
        transient_classes={"allocation": 300},
        maximum_concurrent_allocations=8,
        serialized_reserve_bytes=10,
        available_storage_bytes=200,
        source=SOURCE,
    )
    validate_streamed_storage_projection(projection)
    assert projection["rolling_authenticated_peak_bytes"] == 90
    assert projection["persistent_peak_bytes"] == 200
    assert projection["rolling_lifetimes_dependency_serialized"] is True


def test_full_streamed_storage_validator_authenticates_adjacent_projection(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    source_measurement = miniature_storage_measurements()
    source = source_snapshot(root)
    projection = build_streamed_storage_projection(
        storage_measurements_sha256=source_measurement["content_hash"],
        persistent_classes={"durable": 100},
        rolling_classes={"rolling": 20},
        transient_classes={"transient": 300},
        maximum_concurrent_allocations=4,
        serialized_reserve_bytes=10,
        available_storage_bytes=200,
        source=source,
    )
    values = dict(source_measurement["measurements"])
    values["projected_peak_concurrent_bytes"] = projection["persistent_peak_bytes"]
    values["available_storage_bytes"] = projection[
        "available_persistent_storage_bytes"
    ]
    evidence_path = tmp_path / "measured.bin"
    evidence_path.write_bytes(b"authenticated measurement evidence")
    measurement = build_storage_measurements(
        measurements=values,
        evidence_hashes={
            "full_streamed_storage_projection": projection["content_hash"]
        },
        source_evidence={
            "fixture": {"path": evidence_path, "purpose": "unit test"}
        },
        measurement_profile="production_source_evidence",
    )
    measurement_path = tmp_path / "storage_measurements.json"
    write_immutable_json(measurement_path, measurement)
    write_immutable_json(
        tmp_path / "full_streamed_storage_projection.json", projection
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "validate_retb_full_streamed_storage.py"),
            "--measurements",
            str(measurement_path),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["status"] == (
        "full_streamed_storage_admitted"
    )


def test_shell_exposes_real_compact_smoke_and_full_streamed_commands() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "sbatch" / "submit_retb_tigris_full.sh").read_text()
    worker = (root / "sbatch" / "run_retb_streamed_smoke_phase.sh").read_text()
    phase_worker = (root / "scripts" / "run_retb_streamed_smoke_phase.py").read_text()
    assert "--streamed-smoke-submit" in launcher
    assert "--streamed-submit" in launcher
    assert "streamed_smoke_submission_ledger.json" in launcher
    assert "run_retb_streamed_smoke_phase.py" in worker
    for node_id in (
        "offline_input_cache", "hlt_v3_cache", "region_tree_cache",
        "region_tree_finalize", "normalizers_500k", "input_audit",
        "offline_expert_training", "native_hlt_expert_training",
    ):
        assert f'node_id="{node_id}"' in phase_worker
    for builder in (
        "build_retb_step3_contracts.py", "build_retb_step4_contracts.py",
        "build_retb_step5_contracts.py", "build_retb_step6_contracts.py",
    ):
        assert builder in phase_worker
    assert "validate_task_lifecycle_receipt" in phase_worker
    assert "offline_parent_alias" in phase_worker


def test_graph_cli_reports_only_18_physical_smoke_allocations(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable, str(root / "scripts" / "submit_retb_graph.py"),
            "--miniature", "--submission-scope", "streamed_smoke",
            "--dry-run", "--campaign-id", "compact_smoke",
            "--campaign-root", str(tmp_path / "compact_smoke"),
        ], cwd=root, check=True, capture_output=True, text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["submission_node_count"] == 18
    assert payload["complete_graph_node_count"] == 87
    assert payload["production_submission_performed"] is False
