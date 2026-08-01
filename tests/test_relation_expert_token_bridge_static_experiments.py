from __future__ import annotations

from pathlib import Path
import sys

import pytest

from teacher_logit_reco.relation_expert_token_bridge import (
    STATIC_EXPERIMENT_MANIFEST_NODES,
    STATIC_MANIFEST_NODES,
    build_production_graph,
    build_static_experiment_bundle,
    validate_static_experiment_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    build_campaign_spec,
    canonical_sha256,
)


def _source() -> dict[str, object]:
    return {
        "source_commit": "a" * 40,
        "source_status_sha256": "b" * 64,
        "source_dirty": False,
    }


def _campaign_and_graph(root: Path, *, miniature: bool) -> tuple[dict, dict]:
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
        run_registry_hashes={"static-test": "f" * 64},
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


def test_full_static_matrix_is_exact_deduplicated_and_fully_bound(
    tmp_path: Path,
) -> None:
    root = tmp_path / "retb_static_production"
    campaign, graph = _campaign_and_graph(root, miniature=False)
    bundle = build_static_experiment_bundle(
        campaign=campaign,
        production_graph=graph,
        campaign_root=root,
        python_executable=sys.executable,
    )
    validate_static_experiment_bundle(
        bundle,
        campaign=campaign,
        production_graph=graph,
        campaign_root=root,
        python_executable=sys.executable,
    )
    plan = bundle["static_experiment_plan"]
    assert tuple(plan["groups"]) == STATIC_MANIFEST_NODES
    assert plan["full_matrix_counts"] == {
        "offline_expert_training": 147,
        "offline_expert_confirmation": 147,
        "offline_fusion_cache": 63,
        "offline_fusion_training": 49,
        "native_hlt_expert_training": 541,
        "native_hlt_fusion_training": 30,
        "bridge_pilot_training": 105,
    }
    assert plan["execution_counts"] == plan["full_matrix_counts"]
    assert plan["physical_run_deduplication"] == (
        "first_registry_membership_order"
    )
    assert plan["selector_dependent_rows_included"] is False
    for node_id, rows in plan["groups"].items():
        assert len({row["static_id"] for row in rows}) == len(rows)
        manifest = bundle["task_manifests"][node_id]
        assert manifest["task_count"] == len(rows)
        assert manifest["performance_based_row_skipping"] is False
        for record, task in zip(rows, manifest["rows"]):
            assert record["seed"] >= 0
            assert record["configuration_sha256"] == canonical_sha256(
                record["configuration"]
            )
            assert record["command_sha256"] == canonical_sha256(
                record["argv"]
            )
            assert task["argv"] == record["argv"]
            assert task["expected_outputs"] == record["expected_artifacts"]
            assert task["input_artifact_hashes"]["static_row"] == record[
                "row_sha256"
            ]
            assert all(
                Path(path).resolve().is_relative_to(root.resolve())
                for path in record["expected_artifacts"]
            )


def test_streamed_static_matrix_groups_cache_lifetime_without_dropping_runs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "retb_static_streamed"
    campaign, _ = _campaign_and_graph(root, miniature=False)
    graph = build_production_graph(
        campaign_root=root,
        campaign_id=root.name,
        source_commit="a" * 40,
        source_status_sha256="b" * 64,
        storage_measurements_sha256=campaign["parent_artifact_hashes"][
            "storage_measurements"
        ],
        execution_profile="offline_abc_streamed",
    )
    bundle = build_static_experiment_bundle(
        campaign=campaign,
        production_graph=graph,
        campaign_root=root,
        python_executable=sys.executable,
    )
    validate_static_experiment_bundle(
        bundle,
        campaign=campaign,
        production_graph=graph,
        campaign_root=root,
        python_executable=sys.executable,
    )
    plan = bundle["static_experiment_plan"]
    assert plan["scientific_matrix_counts"]["offline_fusion_cache"] == 63
    assert plan["scientific_matrix_counts"]["offline_fusion_training"] == 49
    assert plan["execution_counts"]["offline_fusion_cache"] == 21
    assert plan["execution_counts"]["offline_fusion_training"] == 49
    assert plan["persistent_frozen_token_cache_rows"] == 0
    waves = plan["groups"]["offline_fusion_cache"]
    assert len(
        {
            (
                row["configuration"]["shape_id"],
                row["configuration"]["pipeline_seed"],
            )
            for row in waves
        }
    ) == 21
    run_ids = {
        run_id
        for wave in waves
        for run_id in wave["configuration"]["fusion_run_ids"]
    }
    assert len(run_ids) == 49
    assert all(
        row["argv"][1] == "scripts/run_retb_streamed_fusion_wave.py"
        for row in waves
    )
    assert all(
        row["argv"][1] == "scripts/verify_retb_streamed_fusion_output.py"
        for row in plan["groups"]["offline_fusion_training"]
    )


def test_static_rows_route_controls_fusions_and_deferred_pilots(
    tmp_path: Path,
) -> None:
    root = tmp_path / "retb_static_routing"
    campaign, graph = _campaign_and_graph(root, miniature=False)
    plan = build_static_experiment_bundle(
        campaign=campaign,
        production_graph=graph,
        campaign_root=root,
        python_executable=sys.executable,
    )["static_experiment_plan"]

    stage_b = plan["groups"]["offline_expert_training"]
    assert any(
        row["configuration"]["tokenizer_mode"] == "TOK_MASKED_MEAN"
        for row in stage_b
    )
    assert any(
        row["configuration"]["topology"] == "B_DUAL_GATED"
        for row in stage_b
    )
    assert all(row["run_id"] is not None for row in stage_b)
    kd_rows = [
        row
        for row in stage_b
        if row["configuration"]["loss_id"] != "ELOSS_CE"
    ]
    assert kd_rows
    assert all(
        "/inputs/teacher_logits/" in row["argv"][
            row["argv"].index("--train-npz") + 1
        ].replace("\\", "/")
        and "/inputs/teacher_logits/" in row["argv"][
            row["argv"].index("--val-stop-npz") + 1
        ].replace("\\", "/")
        for row in kd_rows
    )
    assert all(
        any(
            value["producer"] == "offline_teacher_prerequisites"
            and value["expected_contract"]
            == "retb_teacher_logits_manifest_v1"
            for value in row["deferred_inputs"]
        )
        for row in kd_rows
    )
    warm_rows = [
        row
        for row in stage_b
        if row["configuration"]["initialization"] != "INIT_SCRATCH"
    ]
    assert warm_rows
    assert all(
        any(
            value["producer"] == "offline_teacher_obase"
            and value["role"] == "ordinary_particle_backbone_initialization"
            for value in row["deferred_inputs"]
        )
        for row in warm_rows
    )

    stage_c = plan["groups"]["offline_fusion_training"]
    assert {
        row["configuration"]["fusion_variant"] for row in stage_c
    } == {
        "F_BEST_SINGLE",
        "F_UNIFORM_LOGIT_MEAN",
        "F_TRAINED_LOGIT_LINEAR",
        "F_POOLED_MLP",
        "F_TOKEN_TRANSFORMER",
    }
    assert any(
        row["argv"][1] == "scripts/evaluate_retb_offline_fusion_control.py"
        for row in stage_c
    )
    assert any(
        row["argv"][1] == "scripts/train_retb_offline_fusion.py"
        for row in stage_c
    )
    parameter_free = [
        row
        for row in stage_c
        if row["configuration"]["fusion_variant"]
        in {"F_BEST_SINGLE", "F_UNIFORM_LOGIT_MEAN"}
    ]
    assert len(parameter_free) == 14
    assert all("--val-design-cache" in row["argv"] for row in parameter_free)
    for row in parameter_free:
        expected_names = {
            "val_stop_parameter_free_evaluation.json",
            "val_design_parameter_free_evaluation.json",
        }
        if row["configuration"]["fusion_variant"] == "F_BEST_SINGLE":
            expected_names.add("best_single_selection.json")
        assert {Path(path).name for path in row["expected_artifacts"]} == (
            expected_names
        )
        assert {
            Path(value["path"]).parent.name
            for value in row["deferred_inputs"]
        } == {"val_stop", "val_design"}
        assert all(
            value["producer"] == "offline_fusion_cache"
            for value in row["deferred_inputs"]
        )
    confirmations = plan["groups"]["offline_expert_confirmation"]
    assert len(confirmations) == 147
    assert {
        int(row["seed"]) for row in confirmations
    } == {101, 202, 303}
    assert all(
        row["argv"][1] == "scripts/train_retb_offline_expert.py"
        and row["argv"][row["argv"].index("--registry-stage") + 1] == "C"
        for row in confirmations
    )
    caches = plan["groups"]["offline_fusion_cache"]
    assert len(caches) == 63
    assert {
        row["configuration"]["split"] for row in caches
    } == {"model_train", "val_stop", "val_design"}
    assert all(
        row["argv"][1] == "scripts/build_retb_frozen_token_cache.py"
        and sum(
            value == "--expert-registration" for value in row["argv"]
        )
        == 7
        and sum(value == "--expert-checkpoint" for value in row["argv"])
        == 7
        for row in caches
    )

    stage_d = plan["groups"]["native_hlt_expert_training"]
    assert sum(
        row["configuration"].get("kind") == "NATIVE_HLT_MATCHED_CONTROL"
        for row in stage_d
    ) == 2
    assert any(
        row["configuration"].get("mode") == "HE_DUAL_OBJECTIVE"
        for row in stage_d
    )
    assert len(plan["groups"]["native_hlt_fusion_training"]) == 30

    pilots = plan["groups"]["bridge_pilot_training"]
    assert all(row["run_id"] is None for row in pilots)
    assert all(
        row["argv"][1] == "scripts/train_retb_bridge_pilot.py"
        for row in pilots
    )
    assert all(
        {
            "checkpoint_registration.json",
            "best_model_val.pt",
            "model_train_coordinate_arrays.npz",
            "val_stop_coordinate_arrays.npz",
            "val_design_coordinate_arrays.npz",
        }
        == {Path(path).name for path in row["expected_artifacts"]}
        for row in pilots
    )
    assert all("--val-design-dataset" in row["argv"] for row in pilots)
    assert all(
        row["run_id_resolution"]
        == "materialize_after_parent_checkpoint_hashes_are_immutable"
        for row in pilots
    )
    assert all(row["deferred_inputs"] for row in pilots)


def test_miniature_uses_complete_matrix_and_graph_marks_static(
    tmp_path: Path,
) -> None:
    root = tmp_path / "retb_static_miniature"
    campaign, graph = _campaign_and_graph(root, miniature=True)
    bundle = build_static_experiment_bundle(
        campaign=campaign,
        production_graph=graph,
        campaign_root=root,
        python_executable=sys.executable,
    )
    plan = bundle["static_experiment_plan"]
    assert (
        plan["miniature_policy"]
        == "complete_scientific_matrix_on_miniature_populations"
    )
    nodes = {row["node_id"]: row for row in graph["nodes"]}
    entries = {
        row["node_id"]: row
        for row in graph["node_execution_registry"]["entries"]
    }
    for node_id in STATIC_MANIFEST_NODES:
        expected = plan["full_matrix_counts"][node_id]
        assert plan["execution_counts"][node_id] == expected
        assert bundle["task_manifests"][node_id]["task_count"] == expected
        assert nodes[node_id]["dynamic_continuation"] is False
        assert entries[node_id]["row_resolution"] == "static"
        assert entries[node_id]["manifest_producer"] == {
            "node_id": "campaign_bootstrap",
            "entrypoint": (
                "scripts/compile_retb_static_experiment_manifests.py"
            ),
            "publication_mode": "campaign_bootstrap",
        }
    assert STATIC_EXPERIMENT_MANIFEST_NODES == frozenset(
        STATIC_MANIFEST_NODES
    )


def test_static_bundle_rejects_configuration_or_output_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "retb_static_tamper"
    campaign, graph = _campaign_and_graph(root, miniature=True)
    bundle = build_static_experiment_bundle(
        campaign=campaign,
        production_graph=graph,
        campaign_root=root,
        python_executable=sys.executable,
    )
    bundle["static_experiment_plan"]["groups"][
        "offline_expert_training"
    ][0]["configuration"]["shape_id"] = "S16_128"
    with pytest.raises(ValueError, match="semantics differ"):
        validate_static_experiment_bundle(
            bundle,
            campaign=campaign,
            production_graph=graph,
            campaign_root=root,
            python_executable=sys.executable,
        )
