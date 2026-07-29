from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    bind_source,
)
from teacher_logit_reco.relation_expert_token_bridge.oracle_substitutions import (
    build_stage_i_policy,
    evaluate_stage_i_substitutions,
    validate_stage_i_evaluation,
    within_class_wrong_event_indices,
    wrong_event_indices,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_bundle import (
    BEAM_WIDTH,
    build_bundle_search_policy,
    build_locked_target_coordinate,
    build_predictor_cache_index,
    build_predictor_candidate,
    score_frozen_bundle,
    select_joint_predictor_bundle,
    shared_predictor_configuration_id,
    validate_locked_target_coordinate,
    validate_predictor_bundle_selection,
    validate_predictor_candidate,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import EXPERT_ORDER
from teacher_logit_reco.relation_expert_token_bridge.step10 import (
    build_step10_bundle,
    validate_step10_bundle,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SOURCE = {
    "source_commit": "1" * 40,
    "source_status_sha256": "2" * 64,
    "source_dirty": True,
}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _coordinate():
    return build_locked_target_coordinate(
        coordinate_id="COORD_T0",
        target_modes={expert: "T0_PURE" for expert in EXPERT_ORDER},
        allocation={expert: [2, 64] for expert in EXPERT_ORDER},
        fusion_checkpoint_hashes={
            seed: _digest(f"fusion-checkpoint-{seed}")
            for seed in (101, 202, 303)
        },
        fusion_registration_hashes={
            seed: _digest(f"fusion-registration-{seed}")
            for seed in (101, 202, 303)
        },
        stage_e_coordinate_sha256=SHA_A,
    )


def _candidate(expert: str, variant: str):
    canonical = variant == "A"
    architecture = (
        "A3_SLOT_DECODER_DIRECT"
        if canonical
        else "A4_SLOT_DECODER_GATED"
    )
    objective_id = "W_CANONICAL" if canonical else "W_TOKEN_HEAVY"
    return build_predictor_candidate(
        candidate_id=f"{expert}_{variant}",
        expert_id=expert,
        coordinate_id="COORD_T0",
        target_mode="T0_PURE",
        shape_id="SHAPE_HIGH",
        token_count=2,
        token_dimension=64,
        architecture=architecture,
        context="C2_ALL",
        objective_id=objective_id,
        uncertainty_head="U_SLOT",
        normalization_mode="N_UNCLIPPED",
        learning_rate=5.0e-4,
        dropout=0.0,
        hlt_evidence_mode="HE_OFFLINE_INIT",
        shared_configuration_id=shared_predictor_configuration_id(
            architecture=architecture,
            context="C2_ALL",
            objective_id=objective_id,
            uncertainty_head="U_SLOT",
            normalization_mode="N_UNCLIPPED",
            learning_rate=5.0e-4,
            dropout=0.0,
            hlt_evidence_mode="HE_OFFLINE_INIT",
        ),
        normalized_token_error_by_seed={
            seed: 1.0 if canonical else 2.0 for seed in (101, 202, 303)
        },
        hybrid_accuracy_by_seed={
            seed: 0.1 for seed in (101, 202, 303)
        },
        hybrid_cross_entropy_by_seed={
            seed: 2.3 for seed in (101, 202, 303)
        },
        inference_flops=100 if canonical else 200,
        parameter_count=50 if canonical else 100,
        seed_artifacts={
            seed: {
                "predictor_registration": _digest(
                    f"{expert}-{variant}-{seed}-registration"
                ),
                "predictor_checkpoint": _digest(
                    f"{expert}-{variant}-{seed}-checkpoint"
                ),
                "inference_manifest": _digest(
                    f"{expert}-{variant}-{seed}-inference"
                ),
                "uncertainty_calibration": _digest(
                    f"{expert}-{variant}-{seed}-calibration"
                ),
                "capacity_report": _digest(
                    f"{expert}-{variant}-{seed}-capacity"
                ),
                "identity_order_sha256": _digest(
                    f"val-design-identity-order-{seed}"
                ),
            }
            for seed in (101, 202, 303)
        },
        materialized_run_hashes={
            seed: _digest(f"{expert}-{variant}-{seed}-run")
            for seed in (101, 202, 303)
        },
    )


def test_candidates_coordinates_and_cache_index_are_exact() -> None:
    candidates = [
        _candidate(expert, variant)
        for expert in EXPERT_ORDER
        for variant in ("A", "B")
    ]
    coordinate = _coordinate()
    for candidate in candidates:
        assert (
            validate_predictor_candidate(candidate)
            == candidate["content_hash"]
        )
    assert (
        validate_locked_target_coordinate(coordinate)
        == coordinate["content_hash"]
    )
    index = build_predictor_cache_index(
        candidates=candidates,
        coordinates=[coordinate],
        step9_bundle_sha256=SHA_A,
    )
    assert index["candidate_count"] == 14
    assert index["all_three_seed_inference_caches_bound"]


def test_width32_bundle_search_is_deterministic_on_all_negative_fixture() -> None:
    candidates = [
        _candidate(expert, variant)
        for expert in EXPERT_ORDER
        for variant in ("A", "B")
    ]
    coordinate = _coordinate()
    index = build_predictor_cache_index(
        candidates=candidates,
        coordinates=[coordinate],
        step9_bundle_sha256=SHA_A,
    )
    candidate_map = {row["candidate_id"]: row for row in candidates}

    def score(names):
        rows = [candidate_map[name] for name in names]
        return {
            "candidate_tuple": list(names),
            "coordinate_id": "COORD_T0",
            "metrics_by_seed": {
                str(seed): {"accuracy": 0.1, "cross_entropy": 2.3}
                for seed in (101, 202, 303)
            },
            "mean_accuracy": 0.1,
            "mean_cross_entropy": 2.3,
            "mean_normalized_token_error": float(
                np.mean([row["mean_normalized_token_error"] for row in rows])
            ),
            "inference_flops": sum(row["inference_flops"] for row in rows),
            "parameter_count": sum(row["parameter_count"] for row in rows),
        }

    kwargs = {
        "candidates": candidates,
        "coordinates": [coordinate],
        "score_tuple": score,
        "predictor_cache_index_sha256": index["content_hash"],
        "label_manifest_hashes_by_seed": {
            seed: _digest(f"labels-{seed}") for seed in (101, 202, 303)
        },
        "label_payload_hashes_by_seed": {
            seed: _digest(f"label-payload-{seed}")
            for seed in (101, 202, 303)
        },
    }
    first = select_joint_predictor_bundle(**kwargs)
    second = select_joint_predictor_bundle(**kwargs)
    assert first == second
    assert (
        validate_predictor_bundle_selection(first)
        == first["predictor_bundle_lock"]["content_hash"]
    )
    assert first["search"]["selected_tuple"] == [
        f"{expert}_A" for expert in EXPERT_ORDER
    ]
    assert set(
        first["predictor_bundle_lock"][
            "selected_candidate_descriptors"
        ]
    ) == set(EXPERT_ORDER)
    assert len(first["search"]["final_beam"]) <= BEAM_WIDTH
    assert first["search"]["all_negative_campaign_completed"]
    assert first["search"]["controls"]["GLOBAL_SHARED_CONFIGURATION"] is not None
    assert first["search"]["controls"]["ALL_W_CANONICAL"] is not None


def test_cross_coordinate_candidate_cannot_enter_locked_tuple() -> None:
    candidates = [
        _candidate(expert, "A") for expert in EXPERT_ORDER
    ]
    broken = dict(candidates[-1])
    broken.pop("content_hash")
    broken["coordinate_id"] = "UNKNOWN"
    from teacher_logit_reco.relation_expert_token_bridge.contracts import (
        with_content_hash,
    )

    broken = with_content_hash(broken)
    with pytest.raises(ValueError, match="unknown coordinate"):
        build_predictor_cache_index(
            candidates=[*candidates[:-1], broken],
            coordinates=[_coordinate()],
            step9_bundle_sha256=SHA_A,
        )


class _TinyFusion(torch.nn.Module):
    def forward(self, *, token_banks):
        pooled = torch.stack(
            [token_banks[name].mean(dim=(1, 2)) for name in EXPERT_ORDER],
            dim=1,
        )
        total = pooled.sum(dim=1, keepdim=True)
        offsets = torch.arange(10, device=total.device).float()[None] * 0.01
        return total + offsets


def test_complete_tuple_scoring_runs_all_three_exact_frozen_fusions() -> None:
    candidates = [_candidate(expert, "A") for expert in EXPERT_ORDER]
    candidate_map = {row["candidate_id"]: row for row in candidates}
    names = tuple(row["candidate_id"] for row in candidates)
    rng = np.random.default_rng(1009)
    banks = {
        (row["candidate_id"], seed): rng.normal(
            size=(20, 2, 64)
        ).astype(np.float32)
        for row in candidates
        for seed in (101, 202, 303)
    }
    score = score_frozen_bundle(
        names,
        candidates_by_id=candidate_map,
        predicted_banks=banks,
        labels_by_seed={
            seed: np.arange(20, dtype=np.int64) % 10
            for seed in (101, 202, 303)
        },
        fusion_by_seed={seed: _TinyFusion() for seed in (101, 202, 303)},
        batch_size=6,
    )
    assert score["candidate_tuple"] == list(names)
    assert set(score["metrics_by_seed"]) == {"101", "202", "303"}
    assert 0.0 <= score["mean_accuracy"] <= 1.0


def test_stage_i_oracle_substitutions_and_negatives_are_complete() -> None:
    rng = np.random.default_rng(1010)
    count = 20
    identities = [f"jet-{index:03d}" for index in range(count)]
    labels = np.arange(count, dtype=np.int64) % 10
    oracle = {
        expert: rng.normal(size=(count, 2, 64)).astype(np.float32)
        for expert in EXPERT_ORDER
    }
    predicted = {
        expert: oracle[expert]
        + rng.normal(scale=0.1, size=oracle[expert].shape).astype(np.float32)
        for expert in EXPERT_ORDER
    }
    identity_hlt = {
        expert: oracle[expert] * 0.9 for expert in EXPERT_ORDER
    }
    no_reconstruction = rng.normal(size=(count, 10)).astype(np.float32)
    artifact = evaluate_stage_i_substitutions(
        identities=identities,
        labels=labels,
        predicted_banks=predicted,
        oracle_banks=oracle,
        identity_projected_hlt_banks=identity_hlt,
        no_reconstruction_logits=no_reconstruction,
        frozen_offline_fusion=_TinyFusion(),
        pipeline_seed=101,
        stage_i_policy_sha256=build_stage_i_policy()["content_hash"],
        stage_i_input_payload_sha256=SHA_C,
        predictor_bundle_lock_sha256=SHA_A,
        frozen_fusion_checkpoint_sha256=SHA_B,
        identity_manifest_sha256=SHA_C,
        label_manifest_sha256=SHA_A,
        predicted_cache_hashes={
            expert: _digest(f"cache-{expert}") for expert in EXPERT_ORDER
        },
        oracle_target_cache_sha256=SHA_B,
        hlt_cache_sha256=SHA_C,
        identity_projected_hlt_cache_sha256=SHA_A,
        no_reconstruction_prediction_sha256=SHA_B,
        batch_size=7,
    )
    assert validate_stage_i_evaluation(artifact) == artifact["content_hash"]
    assert artifact["condition_count"] == 37
    assert not artifact["wrong_event_controls_entered_selection"]
    assert not artifact["stack_val_consumed"]
    assert (
        artifact["conditions"]["ALL_PREDICTED"]["metrics"]["event_count"]
        == count
    )


def test_negative_event_permutations_are_exact_and_nonself() -> None:
    assert wrong_event_indices(4).tolist() == [1, 2, 3, 0]
    labels = np.asarray([0, 1, 0, 1] * 5, dtype=np.int64)
    # Expand to all ten classes with two entries each.
    labels = np.repeat(np.arange(10, dtype=np.int64), 2)
    indices = within_class_wrong_event_indices(labels)
    assert np.array_equal(labels[indices], labels)
    assert not np.any(indices == np.arange(len(labels)))


def test_step10_contract_bundle_is_source_bound_and_fail_closed() -> None:
    bundle = build_step10_bundle(
        campaign_spec_sha256=SHA_A,
        step9_bundle_sha256=SHA_B,
        global_determinism_sha256=SHA_C,
        source_snapshot=SOURCE,
    )
    assert validate_step10_bundle(bundle) == bundle["step10_bundle"][
        "content_hash"
    ]
    assert build_bundle_search_policy()["beam_width"] == 32
    assert not build_stage_i_policy()["selection_use_permitted"]
    assert len({repr(row["source"]) for row in bundle.values()}) == 1


def test_source_binding_changes_candidate_hash_without_semantic_drift() -> None:
    candidate = _candidate("PT", "A")
    bound = bind_source(candidate, source_snapshot=SOURCE)
    assert bound["content_hash"] != candidate["content_hash"]
    assert validate_predictor_candidate(bound) == bound["content_hash"]


def test_selection_lock_rejects_selected_lineage_substitution() -> None:
    candidates = [
        _candidate(expert, variant)
        for expert in EXPERT_ORDER
        for variant in ("A", "B")
    ]
    coordinate = _coordinate()
    index = build_predictor_cache_index(
        candidates=candidates,
        coordinates=[coordinate],
        step9_bundle_sha256=SHA_A,
    )
    candidate_map = {row["candidate_id"]: row for row in candidates}

    def score(names):
        rows = [candidate_map[name] for name in names]
        return {
            "candidate_tuple": list(names),
            "coordinate_id": "COORD_T0",
            "metrics_by_seed": {
                str(seed): {"accuracy": 0.1, "cross_entropy": 2.3}
                for seed in (101, 202, 303)
            },
            "mean_accuracy": 0.1,
            "mean_cross_entropy": 2.3,
            "mean_normalized_token_error": float(
                np.mean([row["mean_normalized_token_error"] for row in rows])
            ),
            "inference_flops": sum(row["inference_flops"] for row in rows),
            "parameter_count": sum(row["parameter_count"] for row in rows),
        }

    result = select_joint_predictor_bundle(
        candidates=candidates,
        coordinates=[coordinate],
        score_tuple=score,
        predictor_cache_index_sha256=index["content_hash"],
        label_manifest_hashes_by_seed={
            seed: _digest(f"labels-{seed}") for seed in (101, 202, 303)
        },
        label_payload_hashes_by_seed={
            seed: _digest(f"payload-{seed}") for seed in (101, 202, 303)
        },
    )
    from teacher_logit_reco.relation_expert_token_bridge.contracts import (
        with_content_hash,
    )

    tampered = dict(result)
    tampered_lock = dict(result["predictor_bundle_lock"])
    tampered_lock.pop("content_hash")
    tampered_lock["candidate_hashes"] = dict(tampered_lock["candidate_hashes"])
    tampered_lock["candidate_hashes"]["PT"] = SHA_C
    tampered["predictor_bundle_lock"] = with_content_hash(tampered_lock)
    with pytest.raises(ValueError, match="lineage differs"):
        validate_predictor_bundle_selection(tampered)


def test_step10_production_entrypoints_are_wired() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "scripts/build_retb_step10_contracts.py": "build_step10_bundle",
        "scripts/materialize_retb_predictor_bundle_inputs.py": (
            "build_predictor_candidate"
        ),
        "scripts/select_retb_joint_predictor_bundle.py": (
            "select_joint_predictor_bundle"
        ),
        "scripts/evaluate_retb_stage_i_substitutions.py": (
            "evaluate_stage_i_substitutions"
        ),
        "sbatch/run_retb_build_step10_contracts.sh": (
            "build_retb_step10_contracts.py"
        ),
        "sbatch/run_retb_materialize_predictor_bundle_inputs.sh": (
            "materialize_retb_predictor_bundle_inputs.py"
        ),
        "sbatch/run_retb_select_predictor_bundle.sh": (
            "select_retb_joint_predictor_bundle.py"
        ),
        "sbatch/run_retb_stage_i_substitutions.sh": (
            "evaluate_retb_stage_i_substitutions.py"
        ),
    }
    for relative, needle in expected.items():
        path = root / relative
        assert path.is_file()
        assert needle in path.read_text("utf-8")
