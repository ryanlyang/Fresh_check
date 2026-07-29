from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

import teacher_logit_reco.relation_expert_token_bridge.late_continuation as late
from teacher_logit_reco.relation_expert_token_bridge import (
    LATE_CONTINUATION_MANIFEST_NODES,
    LATE_NODE_ENTRYPOINTS,
    LATE_WAVE_PREREQUISITES,
    build_late_continuation,
    build_production_graph,
    build_task_manifest,
    publish_late_continuation,
    publish_task_manifest_completion,
    publish_task_row_completion,
    validate_late_continuation,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    build_campaign_spec,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.confirmation import (
    SEED_COMPONENT_KEYS,
)
from teacher_logit_reco.relation_expert_token_bridge.confirmation_execution import (
    CONFIRMATION_EXECUTION_PLAN_CONTRACT,
    validate_confirmation_execution_plan,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_execution import (
    SCALE_GRAPH_EXECUTION_PLAN_CONTRACT,
    SCALE_REFIT_EXECUTION_PLAN_CONTRACT,
    validate_scale_graph_execution_plan,
    validate_scale_refit_execution_plan,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_up import (
    SCALE_COMPONENT_KEYS,
    SCALE_REFIT_KEYS,
)


ROOT = Path(__file__).resolve().parents[1]


def _source() -> dict[str, object]:
    return {
        "source_commit": "a" * 40,
        "source_status_sha256": "b" * 64,
        "source_dirty": False,
    }


def _campaign_graph(root: Path) -> tuple[dict, dict]:
    parents = (
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
        campaign_profile="miniature_test",
        source_snapshot=_source(),
        parent_artifact_hashes={
            name: f"{index + 1:064x}"
            for index, name in enumerate(parents)
        },
        run_registry_hashes={"late-test": "f" * 64},
    )
    graph = build_production_graph(
        campaign_root=root,
        campaign_id=root.name,
        source_commit="a" * 40,
        source_status_sha256="b" * 64,
        storage_measurements_sha256=campaign[
            "parent_artifact_hashes"
        ]["storage_measurements"],
        miniature=True,
    )
    return campaign, graph


def _trigger(campaign: dict, identity: str) -> dict:
    return with_content_hash(
        {
            "contract": f"retb_test_{identity}_v1",
            "schema_version": 1,
            "identity": identity,
            "source": campaign["source"],
        }
    )


def _rows(root: Path, node: str, count: int = 1) -> list[dict]:
    entrypoint = LATE_NODE_ENTRYPOINTS[node][0]
    return [
        {
            "task_id": "canonicalized",
            "argv": [
                sys.executable,
                entrypoint,
                "--campaign-root",
                str(root),
            ],
            "environment": {"RETB_TEST_ROW": str(index)},
            "expected_outputs": [
                str(root / "outputs" / node / f"row_{index}.json")
            ],
            "input_artifact_hashes": {
                "configuration": f"{index + 41:064x}"
            },
        }
        for index in range(count)
    ]


def _completion_pair(
    root: Path,
    campaign: dict,
    graph: dict,
    node: str,
    *,
    count: int = 1,
) -> dict:
    graph_node = next(row for row in graph["nodes"] if row["node_id"] == node)
    concurrency = (
        1
        if graph_node["array"] is None
        else graph_node["array"]["maximum_concurrent_tasks"]
    )
    rows = (
        [
            {
                "task_id": "deployable_export:0",
                "argv": [
                    sys.executable,
                    "scripts/export_retb_deployable_graph.py",
                    "--campaign-root",
                    str(root),
                ],
                "environment": {},
                "expected_outputs": [
                    str(root / "outputs" / node / "row_0.json")
                ],
                "input_artifact_hashes": {"configuration": "9" * 64},
            }
        ]
        if node == "deployable_export"
        else _rows(root, node, count=count)
    )
    manifest = build_task_manifest(
        campaign_spec_sha256=campaign["content_hash"],
        production_graph_sha256=graph["content_hash"],
        node_id=node,
        rows=rows,
        maximum_concurrent_tasks=concurrency,
    )
    for row in manifest["rows"]:
        write_immutable_json(
            Path(row["expected_outputs"][0]),
            with_content_hash(
                {
                    "contract": "retb_test_late_output_v1",
                    "schema_version": 1,
                    "task_id": row["task_id"],
                    "accuracy": 0.0,
                    "source": campaign["source"],
                }
            ),
        )
        publish_task_row_completion(
            campaign_root=root,
            campaign=campaign,
            task_manifest=manifest,
            task_index=row["task_index"],
        )
    completion = publish_task_manifest_completion(
        campaign_root=root,
        campaign=campaign,
        task_manifest=manifest,
    )["artifact"]
    return {"task_manifest": manifest, "completion": completion}


def test_stage_k_m_registry_is_dynamic_and_automatic() -> None:
    _, graph = _campaign_graph(Path("C:/campaign/retb_late"))
    nodes = {row["node_id"]: row for row in graph["nodes"]}
    entries = {
        row["node_id"]: row
        for row in graph["node_execution_registry"]["entries"]
    }
    assert set(LATE_WAVE_PREREQUISITES) == set(
        LATE_CONTINUATION_MANIFEST_NODES
    )
    for node in LATE_CONTINUATION_MANIFEST_NODES:
        assert nodes[node]["dynamic_continuation"] is True
        assert entries[node]["row_resolution"] == "dynamic"
        assert entries[node]["manifest_producer"]["entrypoint"] == (
            "scripts/continue_retb_stage_k_m.py"
        )
        assert all((ROOT / path).is_file() for path in LATE_NODE_ENTRYPOINTS[node])


def test_negative_controls_publish_and_join_only_after_both_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "late-controls").resolve()
    root.mkdir()
    campaign, graph = _campaign_graph(root)
    deploy = _completion_pair(
        root, campaign, graph, "deployable_export"
    )
    monkeypatch.setattr(
        late,
        "_load_prerequisites",
        lambda **_: {"deployable_export": deploy},
    )
    for node in ("robustness_controls", "semantic_controls"):
        payload = build_late_continuation(
            campaign=campaign,
            production_graph=graph,
            campaign_root=root,
            node_id=node,
            trigger_artifact=_trigger(campaign, f"{node}-trigger"),
            rows=_rows(root, node),
        )
        validate_late_continuation(
            payload,
            campaign=campaign,
            production_graph=graph,
            campaign_root=root,
            trigger_artifact=_trigger(campaign, f"{node}-trigger"),
            rows=_rows(root, node),
        )
        publish_late_continuation(campaign_root=root, payload=payload)
        manifest = payload["dynamic_continuation"]["task_manifest"]
        write_immutable_json(
            Path(manifest["rows"][0]["expected_outputs"][0]),
            with_content_hash(
                {
                    "contract": "retb_test_late_output_v1",
                    "schema_version": 1,
                    "task_id": manifest["rows"][0]["task_id"],
                    "accuracy": 0.0,
                    "source": campaign["source"],
                }
            ),
        )
        publish_task_row_completion(
            campaign_root=root,
            campaign=campaign,
            task_manifest=manifest,
            task_index=0,
        )
        publish_task_manifest_completion(
            campaign_root=root,
            campaign=campaign,
            task_manifest=manifest,
        )
    monkeypatch.undo()
    joined = build_late_continuation(
        campaign=campaign,
        production_graph=graph,
        campaign_root=root,
        node_id="stage_l_graph_registration",
        trigger_artifact=_trigger(campaign, "step13"),
        rows=_rows(root, "stage_l_graph_registration"),
    )
    assert joined["gate"]["prerequisite_node_order"] == [
        "robustness_controls",
        "semantic_controls",
    ]
    assert joined["gate"][
        "all_declared_rows_required_even_when_metrics_are_negative"
    ]
    assert joined["gate"]["scientific_underperformance_used_as_gate"] is False

    completion_path = (
        root
        / "job_ledgers"
        / "completions"
        / "semantic_controls"
        / "manifest_completion.json"
    )
    completion_path.unlink()
    with pytest.raises(FileNotFoundError, match="semantic_controls"):
        build_late_continuation(
            campaign=campaign,
            production_graph=graph,
            campaign_root=root,
            node_id="stage_l_graph_registration",
            trigger_artifact=_trigger(campaign, "step13"),
            rows=_rows(root, "stage_l_graph_registration"),
        )


def _step(root: Path, entrypoint: str) -> list[dict]:
    return [
        {
            "step_id": "step_000",
            "argv": [sys.executable, entrypoint, "--campaign-root", str(root)],
            "expected_outputs": [str(root / "work" / "output.json")],
        }
    ]


def test_scale_execution_plans_require_real_workers_and_complete_coverage(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    source = {
        "commit": "a" * 40,
        "status_sha256": "b" * 64,
        "dirty": False,
        "status_hash_policy": (
            "git_diff_binary_HEAD_plus_sorted_untracked_file_bytes_v2"
        ),
    }
    operations = {}
    for name in SCALE_REFIT_KEYS:
        operations[name] = {
            "population": (
                "shared_hlt_scale"
                if name.startswith("shared_HLT_")
                else "scale_train_offline_targets"
                if name == "target_token"
                else "val_design_label_free"
                if name == "uncertainty_calibrator"
                else "offline_scale"
            ),
            "identity_manifest_sha256": "1" * 64,
            "recipe_sha256": "2" * 64,
            "replica_ids": (
                [0, 1, 2, 3]
                if name.startswith("shared_HLT_")
                else []
            ),
            "steps": _step(root, "scripts/fit_retb_normalizers.py"),
            "output_artifact": str(root / "refits" / f"{name}.json"),
        }
    refit = with_content_hash(
        {
            "contract": SCALE_REFIT_EXECUTION_PLAN_CONTRACT,
            "schema_version": 1,
            "graph_id": "graph",
            "pipeline_seed": 101,
            "locked_scale_shortlist_sha256": "3" * 64,
            "scale_train_manifest_sha256": "4" * 64,
            "val_design_identity_manifest_sha256": "5" * 64,
            "five_hundred_k_artifact_hashes": ["6" * 64],
            "operations": operations,
            "source": source,
        }
    )
    validate_scale_refit_execution_plan(
        refit,
        campaign_source=source,
        campaign_root=root,
        repo_root=ROOT,
    )
    components = {
        name: str(root / "components" / f"{name}.json")
        for name in SCALE_COMPONENT_KEYS
    }
    graph = with_content_hash(
        {
            "contract": SCALE_GRAPH_EXECUTION_PLAN_CONTRACT,
            "schema_version": 1,
            "graph_id": "graph",
            "pipeline_seed": 101,
            "locked_scale_shortlist_sha256": "3" * 64,
            "scale_refit_bundle_sha256": "7" * 64,
            "steps": _step(root, "scripts/train_retb_final_consumer.py"),
            "component_artifacts": components,
            "training_summary": str(root / "summary.json"),
            "pre_stack_metrics": str(root / "metrics.json"),
            "source": source,
        }
    )
    validate_scale_graph_execution_plan(
        graph,
        campaign_source=source,
        campaign_root=root,
        repo_root=ROOT,
    )
    incomplete = dict(graph)
    incomplete.pop("content_hash")
    incomplete["component_artifacts"] = dict(components)
    incomplete["component_artifacts"].pop("training_curve")
    with pytest.raises(ValueError, match="semantics differ"):
        validate_scale_graph_execution_plan(
            with_content_hash(incomplete),
            campaign_source=source,
            campaign_root=root,
            repo_root=ROOT,
        )
    registration_only = dict(graph)
    registration_only.pop("content_hash")
    registration_only["steps"] = _step(
        root, "scripts/train_retb_scale_shortlist.py"
    )
    with pytest.raises(ValueError, match="not a training/refit worker"):
        validate_scale_graph_execution_plan(
            with_content_hash(registration_only),
            campaign_source=source,
            campaign_root=root,
            repo_root=ROOT,
        )


def test_confirmation_execution_plan_requires_real_complete_training(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    source = {
        "source_commit": "a" * 40,
        "source_status_sha256": "b" * 64,
        "source_dirty": False,
    }
    plan = with_content_hash(
        {
            "contract": CONFIRMATION_EXECUTION_PLAN_CONTRACT,
            "schema_version": 1,
            "graph_id": "graph",
            "pipeline_seed": 101,
            "stage_l_graph_registry_sha256": "1" * 64,
            "steps": _step(
                root, "scripts/train_retb_final_consumer.py"
            ),
            "component_artifacts": {
                name: str(root / "components" / f"{name}.json")
                for name in SEED_COMPONENT_KEYS
            },
            "training_summary": str(root / "training_summary.json"),
            "val_design_label_manifest_sha256": "2" * 64,
            "source": source,
        }
    )
    validate_confirmation_execution_plan(
        plan,
        campaign_source=source,
        campaign_root=root,
        repo_root=ROOT,
    )
    registration_only = dict(plan)
    registration_only.pop("content_hash")
    registration_only["steps"] = _step(
        root, "scripts/register_retb_500k_seed_confirmation.py"
    )
    with pytest.raises(ValueError, match="registration-only"):
        validate_confirmation_execution_plan(
            with_content_hash(registration_only),
            campaign_source=source,
            campaign_root=root,
            repo_root=ROOT,
        )


def test_task6_entrypoints_and_no_performance_gate() -> None:
    for path in (
        "scripts/continue_retb_stage_k_m.py",
        "scripts/execute_retb_500k_seed_confirmation.py",
        "scripts/execute_retb_scale_refits.py",
        "scripts/execute_retb_scale_graph_training.py",
    ):
        assert (ROOT / path).is_file()
    source = (ROOT / "scripts/continue_retb_stage_k_m.py").read_text()
    assert "build_late_continuation(" in source
    assert "validate_late_continuation(" in source
    assert "publish_late_continuation(" in source
