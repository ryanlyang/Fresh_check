from __future__ import annotations

import pytest

from teacher_logit_reco.relation_expert_token_bridge.late_plan_factories import (
    LATE_PLAN_FACTORIES,
    LATE_PLAN_FACTORY_TARGETS,
    ROBUSTNESS_PROFILES,
    ROBUSTNESS_REPLICAS,
    SEMANTIC_CONTROL_KINDS,
    build_late_factory_input,
)
from teacher_logit_reco.relation_expert_token_bridge.production import (
    LATE_NODE_ENTRYPOINTS,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SOURCE = {
    "source_commit": "1" * 40,
    "source_status_sha256": "2" * 64,
    "source_dirty": True,
}


def _row(
    target: str,
    index: int,
    *,
    environment: dict[str, str] | None = None,
) -> dict:
    return {
        "task_id": f"{target}:{index}",
        "argv": [
            "python",
            LATE_NODE_ENTRYPOINTS[target][0],
            "--campaign-root",
            "/campaign",
            "--output",
            f"/campaign/outputs/{target}/{index}.json",
        ],
        "expected_outputs": [
            f"/campaign/outputs/{target}/{index}.json"
        ],
        "input_artifact_hashes": {
            "campaign_spec": SHA_A,
            "production_graph": SHA_B,
        },
        "environment": {
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
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
    return build_late_factory_input(
        target_node_id=target,
        producer_node_id="producer",
        campaign_spec_sha256=SHA_A,
        production_graph_sha256=SHA_B,
        producer_task_manifest_sha256=SHA_C,
        rows=rows,
        coverage=coverage,
        source=SOURCE,
    )


def test_all_ten_late_factories_are_callable() -> None:
    assert tuple(LATE_PLAN_FACTORIES) == LATE_PLAN_FACTORY_TARGETS
    assert all(callable(value) for value in LATE_PLAN_FACTORIES.values())


def test_robustness_factory_input_requires_every_profile_replica() -> None:
    coordinates = [
        (profile, replica)
        for profile in ROBUSTNESS_PROFILES
        for replica in ROBUSTNESS_REPLICAS
    ]
    rows = [
        _row(
            "robustness_controls",
            index,
            environment={
                "RETB_DEGRADATION_PROFILE": profile,
                "RETB_DEGRADATION_REPLICA": str(replica),
            },
        )
        for index, (profile, replica) in enumerate(coordinates)
    ]
    artifact = _build(
        "robustness_controls", rows, _coverage()
    )
    assert artifact["row_count"] == len(coordinates)
    with pytest.raises(ValueError, match="profile/replica coverage"):
        _build("robustness_controls", rows[:-1], _coverage())


def test_semantic_factory_input_requires_all_control_kinds() -> None:
    rows = [
        _row(
            "semantic_controls",
            index,
            environment={"RETB_SEMANTIC_CONTROL_KIND": kind},
        )
        for index, kind in enumerate(SEMANTIC_CONTROL_KINDS)
    ]
    assert _build(
        "semantic_controls", rows, _coverage()
    )["row_count"] == len(SEMANTIC_CONTROL_KINDS)
    controller = _build(
        "semantic_controls",
        [_row("semantic_controls", 0)],
        _coverage(
            required_semantic_control_kinds=list(SEMANTIC_CONTROL_KINDS)
        ),
    )
    assert controller["row_count"] == 1
    with pytest.raises(ValueError, match="semantic-control coverage"):
        _build("semantic_controls", rows[:-1], _coverage())


@pytest.mark.parametrize(
    "target", ["confirmation_500k", "scale_graph_training"]
)
def test_graph_seed_waves_are_exact_and_performance_independent(
    target: str,
) -> None:
    graph_ids = ["g_a", "g_b"]
    coordinates = [
        (graph_id, seed)
        for graph_id in graph_ids
        for seed in (101, 202, 303)
    ]
    rows = [
        _row(
            target,
            index,
            environment={
                "RETB_GRAPH_ID": graph_id,
                "RETB_PIPELINE_SEED": str(seed),
                "RETB_SCIENTIFIC_RESULT": "negative",
            },
        )
        for index, (graph_id, seed) in enumerate(coordinates)
    ]
    artifact = _build(
        target, rows, _coverage(required_graph_ids=graph_ids)
    )
    assert artifact["row_count"] == 6
    assert all(
        row["environment"][
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION"
        ]
        == "0"
        for row in artifact["rows"]
    )
    with pytest.raises(ValueError, match="graph/seed coverage"):
        _build(
            target,
            rows[:-1],
            _coverage(required_graph_ids=graph_ids),
        )


def test_scale_refits_are_shared_once_per_seed() -> None:
    rows = [
        _row(
            "scale_refits",
            index,
            environment={
                "RETB_PIPELINE_SEED": str(seed),
                "RETB_SCIENTIFIC_RESULT": "negative",
            },
        )
        for index, seed in enumerate((101, 202, 303))
    ]
    artifact = _build("scale_refits", rows, _coverage())
    assert artifact["row_count"] == 3
    with pytest.raises(ValueError, match="shared refit seed coverage"):
        _build("scale_refits", rows[:-1], _coverage())


def test_singleton_lock_and_aggregation_waves_reject_extra_rows() -> None:
    for target in (
        "stage_l_graph_registration",
        "confirmation_summary",
        "bridge_shape_selector",
        "scale_shortlist_selector",
        "scale_completion",
    ):
        _build(target, [_row(target, 0)], _coverage())
        rows = [_row(target, 0), _row(target, 1)]
        with pytest.raises(ValueError, match="exactly one"):
            _build(target, rows, _coverage())
