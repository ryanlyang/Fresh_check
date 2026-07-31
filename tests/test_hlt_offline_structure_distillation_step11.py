from __future__ import annotations

import numpy as np
import pytest

from teacher_logit_reco.hlt_offline_structure_distillation import (
    build_final_evaluation,
    build_final_row_result,
    build_final_execution_lock,
    build_final_input_preparation,
    build_finalist_lock,
    build_postlock_oracle_manifest,
    build_stack_prediction_manifest,
    consume_final_execution_claim,
    load_final_execution_claim,
    select_stack_finalists,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (
    with_content_hash,
)


SOURCE = {
    "commit": "a" * 40,
    "status_sha256": "b" * 64,
    "dirty": True,
    "status_hash_policy": "test",
}


def _stack():
    identities = [f"jet-{index:03d}" for index in range(30)]
    labels = np.tile(np.arange(10), 3)
    rows = []
    for graph_index, graph in enumerate(("A_GRAPH", "B_GRAPH")):
        for seed in (202, 303, 404):
            logits = np.zeros((30, 10), dtype=np.float32)
            if graph_index:
                logits[np.arange(30), labels] = 0.1
            rows.append(
                build_stack_prediction_manifest(
                    graph_id=graph,
                    seed=seed,
                    identities=identities,
                    logits=logits,
                    checkpoint_sha256=f"{graph_index + 1}" * 64,
                    export_sha256=f"{graph_index + 3}" * 64,
                    lineage_hashes={"hlt_cache": "a" * 64},
                    source=SOURCE,
                )
            )
    capacity = {
        graph: {
            "deployable": True,
            "export_parity_validated": True,
            "inference_flops": 10.0 + index,
            "deployed_parameters": 100 + index,
            "lineage_hashes": {"target": "b" * 64},
        }
        for index, graph in enumerate(("A_GRAPH", "B_GRAPH"))
    }
    return identities, labels, rows, capacity


def test_stack_predictions_are_label_free_and_selector_requires_three_seeds():
    identities, labels, rows, capacity = _stack()
    assert all(not row["contains_labels"] and not row["contains_targets"] for row in rows)
    with pytest.raises(ValueError, match="incomplete"):
        select_stack_finalists(
            predictions=rows[:-1],
            label_identities=identities,
            labels=labels,
            label_manifest_sha256="c" * 64,
            capacity_by_graph=capacity,
            source=SOURCE,
        )
    trace = select_stack_finalists(
        predictions=rows,
        label_identities=identities,
        labels=labels,
        label_manifest_sha256="c" * 64,
        capacity_by_graph=capacity,
        source=SOURCE,
    )
    assert trace["accuracy_finalist_graph_id"] == "B_GRAPH"
    assert trace["minimum_improvement_required"] is False
    assert trace["final_test_consumed"] is False


def test_two_locks_and_exactly_once_claim_fail_closed(tmp_path):
    identities, labels, rows, capacity = _stack()
    trace = select_stack_finalists(
        predictions=rows,
        label_identities=identities,
        labels=labels,
        label_manifest_sha256="c" * 64,
        capacity_by_graph=capacity,
        source=SOURCE,
    )
    required = {
        key: f"{index:x}"[-1] * 64
        for index, key in enumerate(
            (
                "split_manifest",
                "validation_partition",
                "scale_pool",
                "hlt_profile",
                "hlt_replica_manifest",
                "hlt_cache_audit",
                "target_capability_audit",
                "target_registry",
                "confirmation_summary",
                "scale_shortlist",
                "scale_export_audit",
                "label_manifest",
                "selector_metrics",
                "selector_trace",
            ),
            start=1,
        )
    }
    finalist = build_finalist_lock(
        selector_trace=trace,
        campaign_spec_sha256="d" * 64,
        required_lineage_hashes=required,
        source=SOURCE,
    )
    assert finalist["final_test_inference_authorized"] is False
    oracle = build_postlock_oracle_manifest(
        finalist_lock=finalist,
        diagnostic_hashes={
            "stack_val_offline_teacher_agreement": "e" * 64
        },
        source=SOURCE,
    )
    prepared = build_final_input_preparation(
        finalist_lock=finalist,
        input_hashes={"final_hlt": "f" * 64},
        source=SOURCE,
    )
    assert prepared["model_inference_performed"] is False
    execution = build_final_execution_lock(
        finalist_lock=finalist,
        postlock_oracle=oracle,
        prepared_inputs=prepared,
        finalist_control_hashes={
            "capacity_controls": "1" * 64,
            "semantic_controls": "2" * 64,
            "export_parity": "3" * 64,
            "latency_controls": "5" * 64,
            "matched_baselines": "4" * 64,
        },
        finalist_control_rows=[
            {
                "row_id": "FINAL_CONTROL_KD",
                "graph_id": "H_KD_LOGIT_O_BASE",
                "seed": 202,
                "checkpoint_sha256": "5" * 64,
                "export_sha256": "6" * 64,
                "export_path": "/exports/kd.pt",
                "control_family": "H_KD_O_BASE",
            }
        ],
        source=SOURCE,
    )
    assert any(
        row["row_id"] == "FINAL_CONTROL_KD"
        for row in execution["final_rows"]
    )
    assert execution["finalist_control_row_count"] == 1
    claim_path = tmp_path / "claim.json"
    claim = consume_final_execution_claim(
        execution_lock=execution, claim_path=claim_path, source=SOURCE
    )
    with pytest.raises(RuntimeError, match="already consumed"):
        consume_final_execution_claim(
            execution_lock=execution, claim_path=claim_path, source=SOURCE
        )
    resumed_claim = load_final_execution_claim(
        execution_lock=execution, claim_path=claim_path, source=SOURCE
    )
    assert resumed_claim == claim
    results = [
        build_final_row_result(
            execution_lock=execution,
            consumed_claim=claim,
            row_id=row["row_id"],
            graph_id=row["graph_id"],
            seed=row["seed"],
            export_sha256=row["export_sha256"],
            checkpoint_sha256=row["checkpoint_sha256"],
            classification_metrics={"macro_per_class_accuracy": 0.0},
            source=SOURCE,
        )
        for row in execution["final_rows"]
    ]
    final = build_final_evaluation(
        execution_lock=execution,
        consumed_claim=claim,
        row_results=results,
        source=SOURCE,
    )
    assert final["final_test_executions"] == 1
    assert final["final_metrics_used_for_selection"] is False
    assert final["negative_results_reported"]
