from __future__ import annotations

import pytest

from teacher_logit_reco.relation_expert_token_bridge.final_plan_factories import (
    FINAL_PLAN_FACTORIES,
    FINAL_PLAN_FACTORY_TARGETS,
    build_final_factory_input,
)
from teacher_logit_reco.relation_expert_token_bridge.final_seal import (
    CONTROL_KINDS,
)
from teacher_logit_reco.relation_expert_token_bridge.production import (
    FINAL_NODE_ENTRYPOINTS,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SOURCE = {
    "source_commit": "1" * 40,
    "source_status_sha256": "2" * 64,
    "source_dirty": True,
}


def _environment(target: str) -> dict[str, str]:
    common = {
        "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
    }
    additions = {
        "prelock_final_inputs": {
            "RETB_PRELOCK_MODEL_OUTPUTS_EMITTED": "0"
        },
        "stack_val_inference": {
            "RETB_INFERENCE_SPLIT": "stack_val",
            "RETB_PREDICTION_SHARDS_CONTAIN_LABELS": "0",
        },
        "accuracy_finalist_selector": {
            "RETB_SELECTOR_SPLIT": "stack_val",
            "RETB_LABEL_JOIN_LOCATION": "selector_only",
        },
        "postlock_oracle_targets": {
            "RETB_CREATED_AFTER_FINALIST_LOCK": "1",
            "RETB_SELECTION_ELIGIBLE": "0",
        },
        "finalist_controls": {
            "RETB_CREATED_AFTER_FINALIST_LOCK": "1",
            "RETB_CONTROLS_MAY_REPLACE_FINALIST": "0",
        },
        "final_test_execution_lock": {
            "RETB_ALL_POSTLOCK_EVIDENCE_COMPLETE": "1",
            "RETB_FINAL_TEST_MODEL_OUTPUT_AUTHORIZED": "1",
        },
        "sealed_final_test": {
            "RETB_EXACTLY_ONCE_FINAL_TEST": "1",
            "RETB_BOTH_FINAL_LOCKS_REQUIRED": "1",
        },
        "final_report": {
            "RETB_TEST_RESULT_MAY_REPLACE_FINALIST": "0"
        },
    }
    return {**common, **additions[target]}


def _row(
    target: str,
    index: int,
    *,
    environment: dict[str, str] | None = None,
    extra_argv: list[str] | None = None,
) -> dict:
    target_argv = (
        [
            "--configuration",
            "/campaign/inputs/stage_n/prelock_input_configuration.json",
        ]
        if target == "prelock_final_inputs" and extra_argv is None
        else list(extra_argv or [])
    )
    expected_outputs = (
        [
            "/campaign/inputs/stage_n/prelock_final_inputs.json",
            *[
                f"/campaign/inputs/stage_n/shared/retb_{split}_shared_HLT_inputs{suffix}"
                for split in ("stack_val", "final_test")
                for suffix in (".json", ".pt")
            ],
        ]
        if target == "prelock_final_inputs"
        else [f"/campaign/outputs/{target}/{index}.json"]
    )
    return {
        "task_id": f"{target}:{index}",
        "argv": [
            "python",
            FINAL_NODE_ENTRYPOINTS[target][0],
            "--campaign-root",
            "/campaign",
            *target_argv,
            "--output",
            str(expected_outputs[0]),
        ],
        "expected_outputs": expected_outputs,
        "input_artifact_hashes": {
            "campaign_spec": SHA_A,
            "production_graph": SHA_B,
        },
        "environment": {
            **_environment(target),
            **dict(environment or {}),
        },
    }


def _coverage(**extra) -> dict:
    return {
        "all_predeclared_rows_present": True,
        "scientific_metric_used_for_membership": False,
        "incomplete_wave_permitted": False,
        **extra,
    }


def _build(target: str, rows: list[dict], coverage: dict) -> dict:
    return build_final_factory_input(
        target_node_id=target,
        producer_node_id="producer",
        campaign_spec_sha256=SHA_A,
        production_graph_sha256=SHA_B,
        producer_task_manifest_sha256=SHA_C,
        rows=rows,
        coverage=coverage,
        source=SOURCE,
    )


def test_all_eight_final_factories_are_callable() -> None:
    assert tuple(FINAL_PLAN_FACTORIES) == FINAL_PLAN_FACTORY_TARGETS
    assert all(callable(value) for value in FINAL_PLAN_FACTORIES.values())
    assert FINAL_NODE_ENTRYPOINTS["prelock_final_inputs"] == (
        "scripts/prepare_retb_final_test_inputs.py",
    )


def test_prelock_input_factory_forbids_model_derived_outputs() -> None:
    artifact = _build(
        "prelock_final_inputs",
        [_row("prelock_final_inputs", 0)],
        _coverage(),
    )
    assert artifact["row_count"] == 1
    bad = _row(
        "prelock_final_inputs",
        0,
        extra_argv=["--checkpoint", "/campaign/model.pt"],
    )
    with pytest.raises(ValueError, match="prelock final-input"):
        _build("prelock_final_inputs", [bad], _coverage())


def test_prelock_input_factory_allows_authenticated_checkpoint_root() -> None:
    row = _row("prelock_final_inputs", 0)
    row["argv"] = [
        value.replace("/campaign", "/srv/checkpoints/retb_campaign")
        for value in row["argv"]
    ]
    row["expected_outputs"] = [
        value.replace("/campaign", "/srv/checkpoints/retb_campaign")
        for value in row["expected_outputs"]
    ]
    artifact = _build("prelock_final_inputs", [row], _coverage())
    assert artifact["row_count"] == 1


def test_stack_val_factory_requires_every_shortlisted_graph_seed() -> None:
    graph_ids = ["g_a", "g_b"]
    coordinates = [
        (graph_id, seed)
        for graph_id in graph_ids
        for seed in (101, 202, 303)
    ]
    rows = [
        _row(
            "stack_val_inference",
            index,
            environment={
                "RETB_GRAPH_ID": graph_id,
                "RETB_PIPELINE_SEED": str(seed),
            },
        )
        for index, (graph_id, seed) in enumerate(coordinates)
    ]
    artifact = _build(
        "stack_val_inference",
        rows,
        _coverage(shortlisted_graph_ids=graph_ids),
    )
    assert artifact["row_count"] == 6
    with pytest.raises(ValueError, match="graph/seed coverage"):
        _build(
            "stack_val_inference",
            rows[:-1],
            _coverage(shortlisted_graph_ids=graph_ids),
        )


def test_selector_is_singleton_and_requires_complete_predictions() -> None:
    row = _row("accuracy_finalist_selector", 0)
    _build(
        "accuracy_finalist_selector",
        [row],
        _coverage(prediction_graph_seed_coverage_complete=True),
    )
    with pytest.raises(ValueError, match="complete predictions"):
        _build(
            "accuracy_finalist_selector",
            [row],
            _coverage(prediction_graph_seed_coverage_complete=False),
        )


def test_postlock_target_factory_requires_both_splits_for_all_seeds() -> None:
    graph_ids = ["g_a", "g_b"]
    coordinates = [
        (graph_id, seed, split)
        for graph_id in graph_ids
        for seed in (101, 202, 303)
        for split in ("stack_val", "final_test")
    ]
    rows = [
        _row(
            "postlock_oracle_targets",
            index,
            environment={
                "RETB_GRAPH_ID": graph_id,
                "RETB_PIPELINE_SEED": str(seed),
                "RETB_TARGET_SPLIT": split,
            },
        )
        for index, (graph_id, seed, split) in enumerate(coordinates)
    ]
    assert _build(
        "postlock_oracle_targets",
        rows,
        _coverage(finalist_graph_ids=graph_ids),
    )["row_count"] == 12
    with pytest.raises(ValueError, match="graph/seed/split coverage"):
        _build(
            "postlock_oracle_targets",
            rows[:-1],
            _coverage(finalist_graph_ids=graph_ids),
        )


def test_controls_lock_and_sealed_execution_require_complete_evidence() -> None:
    graph_ids = ["g_a", "g_b"]
    coordinates = [
        (graph_id, kind, seed)
        for graph_id in graph_ids
        for kind in ("H_MONO_PARAM", "H_MONO_FLOP", "H_BASE_LONG")
        for seed in (101, 202, 303)
    ]
    control_rows = [
        _row(
            "finalist_controls",
            index,
            environment={
                "RETB_OWNER_FINALIST_GRAPH_ID": graph_id,
                "RETB_CONTROL_KIND": kind,
                "RETB_PIPELINE_SEED": str(seed),
            },
        )
        for index, (graph_id, kind, seed) in enumerate(coordinates)
    ]
    _build(
        "finalist_controls",
        control_rows,
        _coverage(
            required_control_kinds=list(CONTROL_KINDS),
            finalist_graph_ids=graph_ids,
        ),
    )
    with pytest.raises(ValueError, match="kind coverage"):
        _build(
            "finalist_controls",
            control_rows,
            _coverage(
                required_control_kinds=["FINALIST"],
                finalist_graph_ids=graph_ids,
            ),
        )
    with pytest.raises(ValueError, match="graph/kind/seed coverage"):
        _build(
            "finalist_controls",
            control_rows[:-1],
            _coverage(
                required_control_kinds=list(CONTROL_KINDS),
                finalist_graph_ids=graph_ids,
            ),
        )
    _build(
        "final_test_execution_lock",
        [_row("final_test_execution_lock", 0)],
        _coverage(
            prelock_inputs_complete=True,
            postlock_targets_complete=True,
            finalist_controls_complete=True,
        ),
    )
    _build(
        "sealed_final_test",
        [_row("sealed_final_test", 0)],
        _coverage(
            task_count=1,
            execution_claim_precedes_model_access=True,
        ),
    )
    with pytest.raises(ValueError, match="exactly-once coverage"):
        _build(
            "sealed_final_test",
            [_row("sealed_final_test", 0)],
            _coverage(
                task_count=2,
                execution_claim_precedes_model_access=True,
            ),
        )


def test_final_report_cannot_change_locked_selection() -> None:
    artifact = _build(
        "final_report", [_row("final_report", 0)], _coverage()
    )
    assert artifact["rows"][0]["environment"][
        "RETB_TEST_RESULT_MAY_REPLACE_FINALIST"
    ] == "0"
