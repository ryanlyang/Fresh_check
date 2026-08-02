from __future__ import annotations

from pathlib import Path

import pytest

from teacher_logit_reco.relation_expert_token_bridge.production import build_production_graph
from teacher_logit_reco.relation_expert_token_bridge.streamed_execution import (
    DURABLE_CLASSES, FULL_STREAMED_PROFILE, SMOKE_PHASES,
    STREAMED_SMOKE_PROFILE, TRANSIENT_CLASSES,
    build_streamed_execution_profile, build_streamed_smoke_plan,
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


def test_shell_exposes_real_compact_smoke_and_full_streamed_commands() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "sbatch" / "submit_retb_tigris_full.sh").read_text()
    worker = (root / "sbatch" / "run_retb_streamed_smoke_phase.sh").read_text()
    assert "--streamed-smoke-submit" in launcher
    assert "--streamed-submit" in launcher
    assert "streamed_smoke_submission_ledger.json" in launcher
    assert "run_retb_streamed_smoke_phase.py" in worker
