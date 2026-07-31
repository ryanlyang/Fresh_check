from __future__ import annotations

import hashlib
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
    publish_deployable_inference_input_binding,
    publish_shared_deployable_inference_payload,
    validate_deployable_inference_input,
    validate_deployable_inference_input_binding,
    validate_shared_deployable_inference_payload,
    validate_sealed_final_test_execution_plan,
    validate_stack_inference_execution_plan,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_capacity_controls import (
    build_hlt_capacity_control_row,
    publish_hlt_capacity_control_export,
    validate_hlt_capacity_control_row,
)
from scripts.run_retb_deployable_inference import run_deployable_inference
from scripts.train_retb_scale_finalist_control import _long_exposure_ledger


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
    assert nodes["finalist_controls"]["array"]["maximum_tasks"] == 18


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


class _DirectLogitControl(torch.nn.Module):
    def forward(self, *, features, **_):
        return features[:, :1].float().repeat(1, 10)


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
    direct, _ = run_deployable_inference(
        graph=_DirectLogitControl(),
        hlt_inputs={"features": torch.tensor([[4.0], [5.0]])},
        identities=["a", "b"],
        batch_size=1,
        device=torch.device("cpu"),
        call_interface="particle_batch",
    )
    assert direct[:, 0].tolist() == [4.0, 5.0]


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
    control = publish_deployable_inference_input(
        output_dir=tmp_path / "control",
        split="final_test",
        graph_id="H_MONO_PARAM::GRAPH_A",
        pipeline_seed=101,
        identities=["a", "b"],
        hlt_inputs={"features": torch.ones(2, 3)},
        source_snapshot=source,
    )
    validate_deployable_inference_input(
        control["manifest"],
        manifest_path=(
            tmp_path
            / "control"
            / "final_test_H_MONO_PARAM__GRAPH_A_seed101.json"
        ),
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


def test_shared_deployable_payload_is_written_once_and_bound_per_graph(
    tmp_path: Path,
) -> None:
    source = {
        "source_commit": "a" * 40,
        "source_status_sha256": "b" * 64,
        "source_dirty": False,
    }
    shared = publish_shared_deployable_inference_payload(
        output_dir=tmp_path,
        split="final_test",
        identities=["a", "b"],
        hlt_inputs={"features": torch.ones(2, 3)},
        source_snapshot=source,
    )
    shared_path = tmp_path / "retb_final_test_shared_HLT_inputs.json"
    validate_shared_deployable_inference_payload(
        shared["manifest"], manifest_path=shared_path
    )
    first = publish_deployable_inference_input_binding(
        output_dir=tmp_path,
        shared_payload_manifest=shared["manifest"],
        shared_payload_manifest_path=shared_path,
        graph_id="GRAPH_A",
        pipeline_seed=101,
    )
    second = publish_deployable_inference_input_binding(
        output_dir=tmp_path,
        shared_payload_manifest=shared["manifest"],
        shared_payload_manifest_path=shared_path,
        graph_id="H_MONO_PARAM::GRAPH_A",
        pipeline_seed=202,
    )
    assert first["manifest"]["payload_filename"] == second["manifest"][
        "payload_filename"
    ]
    assert len(list(tmp_path.glob("*.pt"))) == 1
    validate_deployable_inference_input_binding(
        first["manifest"],
        manifest_path=tmp_path / "final_test_GRAPH_A_seed101.json",
    )
    tampered = dict(first["manifest"])
    tampered["shared_payload_manifest_sha256"] = "0" * 64
    tampered.pop("content_hash")
    tampered = with_content_hash(tampered)
    with pytest.raises(ValueError, match="shared-payload lineage"):
        validate_deployable_inference_input_binding(
            tampered,
            manifest_path=tmp_path / "final_test_GRAPH_A_seed101.json",
        )


def test_hlt_capacity_control_row_attests_actual_fixed_budget() -> None:
    row = build_hlt_capacity_control_row(
        owner_finalist_graph_id="GRAPH_A",
        control_kind="H_MONO_PARAM",
        pipeline_seed=101,
        checkpoint_sha256="1" * 64,
        deployable_export_sha256="2" * 64,
        training_registration_sha256="3" * 64,
        optimizer_updates_completed=8,
        labeled_example_presentations=1024,
        capacity_selection_sha256="4" * 64,
    )
    assert validate_hlt_capacity_control_row(row) == row["content_hash"]
    with pytest.raises(ValueError, match="incomplete"):
        build_hlt_capacity_control_row(
            owner_finalist_graph_id="GRAPH_A",
            control_kind="H_BASE_LONG",
            pipeline_seed=101,
            checkpoint_sha256="1" * 64,
            deployable_export_sha256="2" * 64,
            training_registration_sha256="3" * 64,
            optimizer_updates_completed=0,
            labeled_example_presentations=1024,
            capacity_selection_sha256="4" * 64,
        )


def test_hlt_capacity_export_uses_relative_checkpoint_path(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "training" / "best_model_val.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    export = publish_hlt_capacity_control_export(
        output=tmp_path / "deployable_control.json",
        owner_finalist_graph_id="GRAPH_A",
        control_kind="H_MONO_PARAM",
        pipeline_seed=101,
        configuration=(128, 4, 8, 8, 2),
        checkpoint_path=checkpoint,
        checkpoint_sha256=hashlib.sha256(b"checkpoint").hexdigest(),
        training_registration_sha256="1" * 64,
        capacity_selection_sha256="2" * 64,
        source_snapshot={
            "source_commit": "a" * 40,
            "source_status_sha256": "b" * 64,
            "source_dirty": False,
        },
    )
    assert export["checkpoint_path"].replace("\\", "/") == (
        "training/best_model_val.pt"
    )
    assert not Path(export["checkpoint_path"]).is_absolute()


def test_h_base_long_ledger_excludes_zero_ce_predictor(
    tmp_path: Path,
) -> None:
    def publish(path: Path, payload: dict) -> None:
        write_immutable_json(path, with_content_hash(payload))

    role = (
        tmp_path
        / "runs"
        / "scale"
        / "refits"
        / "seed_101"
        / "roles"
        / "ROLE"
    )
    graph = (
        tmp_path
        / "runs"
        / "scale"
        / "graphs"
        / "GRAPH_A"
        / "seed_101"
    )
    for expert, objective, updates in (
        ("PT", "W_TOKEN_ONLY", 2),
        ("TRACK", "W_CANONICAL", 3),
    ):
        phase = role / "predictors" / expert
        publish(
            phase / "training_curves.json",
            {
                "contract": "test_predictor_curves_v1",
                "schema_version": 1,
                "rows": [{"optimizer_update_ordinal": updates}],
                "planned_update_counts": {
                    "total_optimizer_updates": updates
                },
                "fixed_budget_completed": True,
            },
        )
        publish(
            phase / "registration.json",
            {
                "contract": "test_predictor_registration_v1",
                "schema_version": 1,
                "objective_id": objective,
            },
        )
    publish(
        role / "native_hlt" / "BASE4" / "training_curves.json",
        {
            "contract": "test_native_curves_v1",
            "schema_version": 1,
            "rows": [{"epoch": 1}],
            "optimizer_update_counts": {
                "total_optimizer_updates": 4
            },
            "fixed_budget_completed": True,
        },
    )
    publish(
        graph / "joint" / "training_curves.json",
        {
            "contract": "test_joint_curves_v1",
            "schema_version": 1,
            "rows": [{"optimizer_update_ordinal": 5}],
            "planned_update_counts": {"total_optimizer_updates": 5},
            "fixed_budget_completed": True,
        },
    )
    publish(
        graph / "joint" / "registration.json",
        {
            "contract": "test_joint_registration_v1",
            "schema_version": 1,
            "variant": "J5_END_TO_END",
        },
    )
    ledger = _long_exposure_ledger(
        root=tmp_path,
        owner_graph_id="GRAPH_A",
        seed=101,
        scale_train_events=100,
    )
    assert ledger["total_labeled_example_presentations"] == 1200
    assert len(ledger["component_rows"]) == 3
    assert len(ledger["excluded_zero_CE_component_rows"]) == 1
    assert (
        ledger["excluded_zero_CE_component_rows"][0][
            "ground_truth_CE_evidence"
        ]["objective_id"]
        == "W_TOKEN_ONLY"
    )
