from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

import teacher_logit_reco.relation_expert_token_bridge.final_continuation as final
from teacher_logit_reco.relation_expert_token_bridge import (
    FINAL_CONTINUATION_MANIFEST_NODES,
    FINAL_NODE_ENTRYPOINTS,
    FINAL_WAVE_PREREQUISITES,
    build_final_continuation,
    build_job_ledger,
    build_production_graph,
    build_task_manifest,
    publish_task_manifest_completion,
    publish_task_row_completion,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    build_campaign_spec,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_execution import (
    SEALED_FINAL_TEST_EXECUTION_PLAN_CONTRACT,
    STACK_INFERENCE_EXECUTION_PLAN_CONTRACT,
    publish_deployable_inference_input,
    validate_deployable_inference_input,
    validate_sealed_final_test_execution_plan,
    validate_stack_inference_execution_plan,
)
from scripts.run_retb_deployable_inference import run_deployable_inference


ROOT = Path(__file__).resolve().parents[1]


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
    source = {
        "source_commit": "a" * 40,
        "source_status_sha256": "b" * 64,
        "source_dirty": False,
    }
    campaign = build_campaign_spec(
        campaign_id=root.name,
        campaign_profile="miniature_test",
        source_snapshot=source,
        parent_artifact_hashes={
            name: f"{index + 1:064x}"
            for index, name in enumerate(parents)
        },
        run_registry_hashes={"stage-n-test": "f" * 64},
    )
    graph = build_production_graph(
        campaign_root=root,
        campaign_id=root.name,
        source_commit=source["source_commit"],
        source_status_sha256=source["source_status_sha256"],
        storage_measurements_sha256=campaign[
            "parent_artifact_hashes"
        ]["storage_measurements"],
        miniature=True,
    )
    return campaign, graph


def _rows(root: Path, node: str, count: int = 1) -> list[dict]:
    return [
        {
            "task_id": "canonicalized",
            "argv": [
                sys.executable,
                FINAL_NODE_ENTRYPOINTS[node][0],
                "--campaign-root",
                str(root),
            ],
            "environment": {},
            "expected_outputs": [
                str(root / "outputs" / node / f"row_{index}.json")
            ],
            "input_artifact_hashes": {
                "configuration": f"{index + 31:064x}"
            },
        }
        for index in range(count)
    ]


def _completion_pair(
    root: Path, campaign: dict, graph: dict, node: str
) -> dict:
    graph_node = next(
        row for row in graph["nodes"] if row["node_id"] == node
    )
    manifest = build_task_manifest(
        campaign_spec_sha256=campaign["content_hash"],
        production_graph_sha256=graph["content_hash"],
        node_id=node,
        rows=[
            {
                "task_id": f"{node}:0",
                "argv": [
                    sys.executable,
                    (
                        "scripts/aggregate_retb_scale_completion.py"
                        if node == "scale_completion"
                        else FINAL_NODE_ENTRYPOINTS[node][0]
                    ),
                    "--campaign-root",
                    str(root),
                ],
                "environment": {},
                "expected_outputs": [
                    str(root / "outputs" / node / "parent.json")
                ],
                "input_artifact_hashes": {"configuration": "9" * 64},
            }
        ],
        maximum_concurrent_tasks=(
            1
            if graph_node["array"] is None
            else graph_node["array"]["maximum_concurrent_tasks"]
        ),
    )
    output = with_content_hash(
        {
            "contract": "retb_stage_n_test_parent_v1",
            "schema_version": 1,
            "accuracy": 0.0,
            "mean_log_rejection": -999.0,
            "source": campaign["source"],
        }
    )
    write_immutable_json(manifest["rows"][0]["expected_outputs"][0], output)
    publish_task_row_completion(
        campaign_root=root,
        campaign=campaign,
        task_manifest=manifest,
        task_index=0,
    )
    completion = publish_task_manifest_completion(
        campaign_root=root,
        campaign=campaign,
        task_manifest=manifest,
    )["artifact"]
    return {"task_manifest": manifest, "completion": completion}


def _step(root: Path, entrypoint: str) -> list[dict]:
    return [
        {
            "step_id": "step_000",
            "argv": [
                sys.executable,
                entrypoint,
                "--campaign-root",
                str(root),
            ],
            "expected_outputs": [str(root / "work" / "output.npz")],
        }
    ]


def test_stage_n_registry_is_dynamic_automatic_and_single_shot() -> None:
    _, graph = _campaign_graph(Path("C:/campaign/retb_stage_n"))
    nodes = {row["node_id"]: row for row in graph["nodes"]}
    entries = {
        row["node_id"]: row
        for row in graph["node_execution_registry"]["entries"]
    }
    assert set(FINAL_WAVE_PREREQUISITES) == set(
        FINAL_CONTINUATION_MANIFEST_NODES
    )
    for node in FINAL_CONTINUATION_MANIFEST_NODES:
        assert nodes[node]["dynamic_continuation"] is True
        assert entries[node]["manifest_producer"]["entrypoint"] == (
            "scripts/continue_retb_stage_n.py"
        )
        assert all(
            (ROOT / path).is_file() for path in FINAL_NODE_ENTRYPOINTS[node]
        )
    assert nodes["sealed_final_test"]["array"]["maximum_tasks"] == 1
    assert nodes["finalist_controls"]["array"]["maximum_tasks"] == 1


def test_negative_science_does_not_block_stage_n_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "stage-n").resolve()
    root.mkdir()
    campaign, graph = _campaign_graph(root)
    parent = _completion_pair(
        root, campaign, graph, "scale_completion"
    )
    monkeypatch.setattr(
        final,
        "_load_prerequisites",
        lambda **_: {"scale_completion": parent},
    )
    trigger = with_content_hash(
        {
            "contract": "retb_negative_scale_completion_v1",
            "schema_version": 1,
            "all_candidates_worse_than_baseline": True,
            "source": campaign["source"],
        }
    )
    payload = build_final_continuation(
        campaign=campaign,
        production_graph=graph,
        campaign_root=root,
        node_id="stack_val_inference",
        trigger_artifact=trigger,
        rows=_rows(root, "stack_val_inference"),
    )
    assert payload["gate"]["scientific_performance_used_as_gate"] is False
    assert payload["final_continuation_bundle"][
        "performance_threshold_abort_allowed"
    ] is False


def test_stage_n_execution_plans_forbid_registration_only_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    source = {
        "source_commit": "a" * 40,
        "source_status_sha256": "b" * 64,
        "source_dirty": False,
    }
    stack = with_content_hash(
        {
            "contract": STACK_INFERENCE_EXECUTION_PLAN_CONTRACT,
            "schema_version": 1,
            "graph_id": "graph",
            "pipeline_seed": 101,
            "locked_scale_shortlist_sha256": "1" * 64,
            "scale_completion_sha256": "2" * 64,
            "parent_hashes": {},
            "steps": [
                {
                    "step_id": "step_000",
                    "argv": [
                        sys.executable,
                        "scripts/run_retb_deployable_inference.py",
                        "--campaign-root",
                        str(root),
                        "--split",
                        "stack_val",
                        "--graph-id",
                        "graph",
                        "--pipeline-seed",
                        "101",
                        "--scale-completion",
                        str(root / "scale_completion.json"),
                        "--output",
                        str(root / "stack.npz"),
                    ],
                    "expected_outputs": [
                        str(root / "stack.npz")
                    ],
                }
            ],
            "inference_output_npz": str(root / "stack.npz"),
            "source": source,
        }
    )
    validate_stack_inference_execution_plan(
        stack,
        campaign_source=source,
        campaign_root=root,
        repo_root=ROOT,
    )
    bad = dict(stack)
    bad.pop("content_hash")
    bad["steps"] = _step(root, "scripts/infer_retb_scale_stack_val.py")
    with pytest.raises(ValueError, match="training/refit worker"):
        validate_stack_inference_execution_plan(
            with_content_hash(bad),
            campaign_source=source,
            campaign_root=root,
            repo_root=ROOT,
        )


def test_sealed_final_plan_requires_every_locked_row_once(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    source = {
        "source_commit": "a" * 40,
        "source_status_sha256": "b" * 64,
        "source_dirty": False,
    }
    expected = {
        "row": {
            "row_id": "row",
            "graph_id": "graph",
            "pipeline_seed": 101,
            "checkpoint_sha256": "3" * 64,
        }
    }
    lock = with_content_hash(
        {
            "contract": "retb_final_test_execution_lock_v1",
            "schema_version": 1,
            "eligible_evaluation_rows": expected,
            "source": source,
        }
    )
    plan = with_content_hash(
        {
            "contract": SEALED_FINAL_TEST_EXECUTION_PLAN_CONTRACT,
            "schema_version": 1,
            "final_test_execution_lock_sha256": lock["content_hash"],
            "steps": [
                {
                    "step_id": "step_000",
                    "argv": [
                        sys.executable,
                        "scripts/run_retb_deployable_inference.py",
                        "--campaign-root",
                        str(root),
                        "--split",
                        "final_test",
                        "--graph-id",
                        "graph",
                        "--pipeline-seed",
                        "101",
                        "--execution-lock",
                        str(root / "lock.json"),
                        "--execution-claim",
                        str(root / "claim.json"),
                        "--execution-plan",
                        str(root / "plan.json"),
                        "--output",
                        str(root / "prediction.npz"),
                    ],
                    "expected_outputs": [
                        str(root / "prediction.npz")
                    ],
                }
            ],
            "final_labels_manifest": str(root / "labels.json"),
            "prediction_rows": [
                {
                    **expected["row"],
                    "inference_output_npz": str(root / "prediction.npz"),
                    "prediction_manifest_output": str(
                        root / "prediction.json"
                    ),
                }
            ],
            "source": source,
        }
    )
    validate_sealed_final_test_execution_plan(
        plan,
        execution_lock=lock,
        campaign_source=source,
        campaign_root=root,
        repo_root=ROOT,
    )
    missing = dict(plan)
    missing.pop("content_hash")
    missing["prediction_rows"] = []
    with pytest.raises(ValueError, match="semantics differ"):
        validate_sealed_final_test_execution_plan(
            with_content_hash(missing),
            execution_lock=lock,
            campaign_source=source,
            campaign_root=root,
            repo_root=ROOT,
        )


def test_completed_ledger_requires_all_jobs_and_final_lineage() -> None:
    _, graph = _campaign_graph(Path("C:/campaign/retb_completed"))
    jobs = {
        row["node_id"]: str(40_000 + index)
        for index, row in enumerate(graph["nodes"])
    }
    with pytest.raises(ValueError, match="final artifacts"):
        build_job_ledger(
            production_graph=graph,
            jobs=jobs,
            submission_mode="completed",
        )
    ledger = build_job_ledger(
        production_graph=graph,
        jobs=jobs,
        submission_mode="completed",
        completion_artifact_hashes={
            "locked_scale_finalists": "1" * 64,
            "final_test_execution_lock": "2" * 64,
            "sealed_final_test_evaluation": "3" * 64,
            "final_report": "4" * 64,
        },
    )
    assert ledger["completed_after_final_report"] is True
    assert ledger["all_nodes_bound"] is True


def test_execution_claim_precedes_final_inference_and_no_threshold_abort() -> None:
    source = (
        ROOT / "scripts" / "execute_retb_sealed_final_test.py"
    ).read_text(encoding="utf-8")
    assert source.index("claim_publication =") < source.index(
        "receipts = execute_plan_steps("
    )
    assert "refusing repeat access" in source
    assert "performance" not in source.lower()


class _NumericalDeployable(torch.nn.Module):
    def forward(self, *, hlt_inputs):
        values = hlt_inputs["features"].float()
        return {"logits": values[:, :1].repeat(1, 10)}


def test_deployable_worker_runs_real_batched_hlt_only_inference() -> None:
    identities = ["a", "b", "c"]
    logits, probabilities = run_deployable_inference(
        graph=_NumericalDeployable(),
        hlt_inputs={
            "features": torch.tensor(
                [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
            )
        },
        identities=identities,
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert logits.shape == (3, 10)
    assert probabilities.shape == (3, 10)
    assert logits[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert torch.allclose(
        torch.from_numpy(probabilities.sum(axis=1)),
        torch.ones(3),
    )


def test_deployable_input_publication_is_label_free_and_authenticated(
    tmp_path: Path,
) -> None:
    source = {
        "source_commit": "a" * 40,
        "source_status_sha256": "b" * 64,
        "source_dirty": False,
    }
    publication = publish_deployable_inference_input(
        output_dir=tmp_path,
        split="stack_val",
        graph_id="GRAPH_A",
        pipeline_seed=101,
        identities=["a", "b"],
        hlt_inputs={"features": torch.ones(2, 3)},
        source_snapshot=source,
    )
    manifest = publication["manifest"]
    assert manifest["contains_labels"] is False
    validate_deployable_inference_input(
        manifest,
        manifest_path=tmp_path / "stack_val_GRAPH_A_seed101.json",
    )
    with pytest.raises(ValueError, match="privileged field"):
        publish_deployable_inference_input(
            output_dir=tmp_path / "bad",
            split="stack_val",
            graph_id="GRAPH_B",
            pipeline_seed=101,
            identities=["a", "b"],
            hlt_inputs={"offline_targets": torch.ones(2, 3)},
            source_snapshot=source,
        )
