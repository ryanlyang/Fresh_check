from __future__ import annotations

from pathlib import Path
import sys

import pytest

from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    build_campaign_spec,
)
from teacher_logit_reco.relation_expert_token_bridge.production import (
    build_production_graph,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_d_matrix_smoke import (
    STAGE_D_MATRIX_SMOKE_COUNTS,
    STAGE_D_MATRIX_SMOKE_TERMINAL_NODE,
    build_stage_d_matrix_smoke_ledger,
    build_stage_d_matrix_smoke_scope,
    stage_d_matrix_smoke_node_ids,
    summarize_stage_d_matrix,
    validate_stage_d_matrix_smoke_scope,
)
from teacher_logit_reco.relation_expert_token_bridge.static_experiments import (
    build_static_experiment_bundle,
)


def _source() -> dict[str, object]:
    return {
        "source_commit": "a" * 40,
        "source_status_sha256": "b" * 64,
        "source_dirty": False,
    }


def _campaign_and_graph(root: Path, *, miniature: bool = True) -> tuple[dict, dict]:
    parent_names = (
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
    campaign = build_campaign_spec(
        campaign_id=root.name,
        campaign_profile=(
            "miniature_test" if miniature else "production_500k_scale3m"
        ),
        source_snapshot=_source(),
        parent_artifact_hashes={
            name: f"{index + 1:064x}"
            for index, name in enumerate(parent_names)
        },
        run_registry_hashes={"stage-d-matrix-smoke-test": "f" * 64},
    )
    graph = build_production_graph(
        campaign_root=root,
        campaign_id=root.name,
        source_commit="a" * 40,
        source_status_sha256="b" * 64,
        storage_measurements_sha256=campaign["parent_artifact_hashes"][
            "storage_measurements"
        ],
        miniature=miniature,
    )
    return campaign, graph


def test_stage_d_matrix_scope_is_exact_dependency_closed_production_prefix(
    tmp_path: Path,
) -> None:
    _, graph = _campaign_and_graph(tmp_path / "miniature")
    node_ids = stage_d_matrix_smoke_node_ids(graph)
    scope = build_stage_d_matrix_smoke_scope(production_graph=graph)
    selected = set(node_ids)
    assert node_ids[-1] == STAGE_D_MATRIX_SMOKE_TERMINAL_NODE
    assert scope["submitted_node_ids"] == node_ids
    assert scope["submitted_stages"] == list("ABCD")
    assert scope["expected_static_matrix_counts"] == STAGE_D_MATRIX_SMOKE_COUNTS
    assert scope["all_541_native_hlt_configurations_required"] is True
    assert scope["scientific_underperformance_blocks_continuation"] is False
    assert scope["production_evidence_eligible"] is False
    for node in graph["nodes"]:
        if node["node_id"] in selected:
            assert node["stage"] in "ABCD"
            assert set(node["dependencies"]).issubset(selected)
    validate_stage_d_matrix_smoke_scope(scope, production_graph=graph)


def test_stage_d_matrix_scope_rejects_production_data(tmp_path: Path) -> None:
    _, graph = _campaign_and_graph(tmp_path / "production", miniature=False)
    with pytest.raises(ValueError, match="requires the miniature graph"):
        build_stage_d_matrix_smoke_scope(production_graph=graph)


def test_stage_d_matrix_ledger_requires_every_exact_graph_binding(
    tmp_path: Path,
) -> None:
    _, graph = _campaign_and_graph(tmp_path / "ledger")
    node_ids = stage_d_matrix_smoke_node_ids(graph)
    jobs = {node_id: str(10_000 + index) for index, node_id in enumerate(node_ids)}
    ledger = build_stage_d_matrix_smoke_ledger(
        production_graph=graph,
        jobs=jobs,
        report_job_id="99999",
    )
    assert ledger["jobs"] == dict(sorted(jobs.items()))
    assert ledger["terminal_dependency_job_id"] == jobs[
        STAGE_D_MATRIX_SMOKE_TERMINAL_NODE
    ]
    broken = dict(jobs)
    broken.pop(node_ids[0])
    with pytest.raises(ValueError, match="job bindings differ"):
        build_stage_d_matrix_smoke_ledger(
            production_graph=graph,
            jobs=broken,
            report_job_id="99999",
        )


def test_stage_d_matrix_summary_covers_all_real_static_coordinates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "matrix"
    campaign, graph = _campaign_and_graph(root)
    plan = build_static_experiment_bundle(
        campaign=campaign,
        production_graph=graph,
        campaign_root=root,
        python_executable=sys.executable,
    )["static_experiment_plan"]
    summary = summarize_stage_d_matrix(plan)
    assert summary["static_matrix_counts"] == STAGE_D_MATRIX_SMOKE_COUNTS
    assert summary["native_hlt_run_count"] == 541
    assert summary["realization_policies"] == ["R_FIXED", "R_MULTI", "R_RANDOM"]
    assert summary["measurement_embedding_values"] == [False, True]
    assert summary["matched_controls"] == ["H_BASE", "H_WIDE"]
    assert summary["all_rows_non_performance_gated"] is True


def test_stage_d_matrix_submission_shell_uses_full_matrix_and_smoke_resources() -> None:
    launcher = Path("sbatch/submit_retb_tigris_full.sh").read_text(
        encoding="utf-8"
    )
    array_launcher = Path("sbatch/run_retb_array_launcher.sh").read_text(
        encoding="utf-8"
    )
    assert "--stage-d-matrix-smoke-submit" in launcher
    assert 'RETB_STAGE_D_MATRIX_CONCURRENCY:=128' in launcher
    assert "run_finalize_retb_stage_d_matrix_smoke.sh" in launcher
    assert "write_retb_stage_d_matrix_smoke_ledger.py" in launcher
    assert '== "stage_d_matrix_smoke"' in array_launcher
    assert "RETB_SMOKE_GPU_CPUS_PER_TASK" in array_launcher
    assert "RETB_SMOKE_GPU_MEM" in array_launcher
