from __future__ import annotations

from pathlib import Path

import pytest

from scripts.execute_retb_predictor_campaign import _hlt_cache

from teacher_logit_reco.relation_expert_token_bridge.predictor_campaign import (
    FINAL_CONSUMER_PHASE_ORDER,
    JOINT_PHASE_ORDER,
    PREDICTOR_PHASE_ORDER,
    build_phased_controller_topology,
    build_stage_f_optimizer_followup_registry,
    select_stage_f_architecture_families,
    select_stage_f_optimizer_configurations,
    select_stage_g_configurations,
)
from teacher_logit_reco.relation_expert_token_bridge.step9 import (
    build_stage_f_registry,
    build_stage_g_registry,
)


def test_confirmation_native_hlt_commands_use_cache_directories() -> None:
    root = Path("/campaign")
    assert _hlt_cache(root, "model_train", 3) == (
        root
        / "inputs"
        / "hlt_v3"
        / "model_train"
        / "replica_3"
        / "R_MULTI"
        / "D_NOMINAL"
    )
    assert _hlt_cache(root, "val_design", 0).name == "D_NOMINAL"


def _result(identity_field: str, identity: str, index: int) -> dict:
    return {
        identity_field: identity,
        "val_design_accuracy": 0.01 - index * 1.0e-7,
        "normalized_token_error": 10.0 + index,
        "parameter_count": 1000 + index,
        "result_sha256": f"{index + 1:064x}",
    }


def test_stage_f_phases_select_deterministically_when_every_row_is_bad() -> None:
    registry = build_stage_f_registry()
    results = [
        _result("run_id", row["run_id"], index)
        for index, row in enumerate(registry["rows"])
    ]
    selection = select_stage_f_architecture_families(results)
    assert selection["complete_result_count"] == registry[
        "membership_count"
    ]
    assert selection[
        "scientific_underperformance_blocks_continuation"
    ] is False
    followup = build_stage_f_optimizer_followup_registry(selection)
    assert followup["membership_count"] == 48
    followup_results = [
        _result("run_id", row["run_id"], index)
        for index, row in enumerate(followup["rows"])
    ]
    optimizer = select_stage_f_optimizer_configurations(
        registry=followup, results=followup_results
    )
    assert optimizer["complete_result_count"] == 48
    assert set(optimizer["selected_families"]) == {
        "selected_direct",
        "selected_gated",
    }


def test_stage_f_selection_rejects_incomplete_coverage() -> None:
    registry = build_stage_f_registry()
    results = [
        _result("run_id", row["run_id"], index)
        for index, row in enumerate(registry["rows"][:-1])
    ]
    with pytest.raises(ValueError, match="coverage"):
        select_stage_f_architecture_families(results)


def test_stage_g_selection_consumes_every_registered_template() -> None:
    registry = build_stage_g_registry()
    results = [
        _result("template_id", row["template_id"], index)
        for index, row in enumerate(registry["templates"])
    ]
    selected = select_stage_g_configurations(results)
    assert selected["complete_result_count"] == registry[
        "membership_count"
    ]
    assert set(selected["selected_families"]) == {
        "selected_direct",
        "selected_gated",
    }
    assert selected[
        "scientific_underperformance_blocks_continuation"
    ] is False


def test_public_targets_wrap_ordered_internal_scientific_controllers() -> None:
    predictor = build_phased_controller_topology("predictor_campaign")
    joint = build_phased_controller_topology("joint_predictor_campaign")
    consumers = build_phased_controller_topology(
        "final_consumer_campaign"
    )
    assert predictor["public_target"] == "predictor_training"
    assert predictor["internal_phase_order"] == list(PREDICTOR_PHASE_ORDER)
    assert predictor["selection_edges"]["H_CONFIRMATION"] == (
        "H_EVIDENCE_FUSION_CONFIRM"
    )
    assert joint["public_target"] == "joint_predictor_training"
    assert joint["internal_phase_order"] == list(JOINT_PHASE_ORDER)
    assert joint["selection_edges"]["J5_END_TO_END"] == "J4_BLOCK_SELECT"
    assert consumers["public_target"] == "final_consumer_training"
    assert consumers["internal_phase_order"] == list(
        FINAL_CONSUMER_PHASE_ORDER
    )
    assert consumers["selection_edges"] == {
        "TOKEN_REFINER_WAVE": "FINAL_DATASET_PREP",
        "TOKEN_REFINER_SELECT": "TOKEN_REFINER_WAVE",
        "FINAL_CONSUMER_WAVE": "TOKEN_REFINER_SELECT",
    }
    for artifact in (predictor, joint, consumers):
        assert artifact["public_target_count_preserved"]
        assert not artifact["selection_result_sign_blocks_continuation"]
        assert artifact["incomplete_or_stale_phase_blocks_continuation"]
