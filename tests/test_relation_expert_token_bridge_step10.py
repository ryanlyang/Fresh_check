from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.evaluate_retb_relation_predictor_semantics import _groups
from scripts.execute_retb_semantic_control_campaign import (
    PHASES,
    Planner,
    semantic_controls_bundle_path,
)
from scripts.execute_retb_stage_l_registration import _load_semantic_controls
from scripts import finalize_retb_semantic_control_campaign as semantic_finalizer

from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    SEMANTIC_CONTROL_POLICY,
    bind_source,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.late_plan_factories import (
    SEMANTIC_CONTROL_KINDS,
)
from teacher_logit_reco.relation_expert_token_bridge.joint_bridge import (
    JointBridgeGraph,
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
from teacher_logit_reco.relation_expert_token_bridge.semantic_evidence import (
    validate_stage_k_semantic_controls,
)
from teacher_logit_reco.relation_expert_token_bridge.step7 import STAGE_E_SHAPES
from tests.retb_semantic_test_support import build_valid_semantic_controls


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
    # The v2 semantic-control contract adds one global- and one same-class
    # wrong-event replacement for each of the seven expert banks.
    assert artifact["condition_count"] == 51
    for expert in EXPERT_ORDER:
        assert f"WRONG_EVENT_BANK__{expert}" in artifact["conditions"]
        assert (
            f"WITHIN_CLASS_WRONG_EVENT_BANK__{expert}"
            in artifact["conditions"]
        )
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


def test_semantic_shuffle_batches_preserve_population_and_multiplicity() -> None:
    mask = np.zeros((8, 1, 4), dtype=bool)
    mask[:3, 0, :1] = True
    mask[3:, 0, :2] = True
    groups = _groups(mask, matched=True, batch_size=4)
    assert sorted(np.concatenate(groups).tolist()) == list(range(8))
    assert all(len(group) >= 2 for group in groups)
    assert all(
        len(set(mask[group, 0].sum(axis=1).tolist())) == 1
        for group in groups
    )
    assert [len(group) for group in _groups(mask[:5], matched=False, batch_size=4)] == [5]

    singleton = mask[:4].copy()
    singleton[3, 0, :] = True
    eligible = _groups(singleton, matched=True, batch_size=4)
    assert np.concatenate(eligible).tolist() == [0, 1, 2]
    assert 3 not in np.concatenate(eligible).tolist()


def test_relation_zero_can_target_each_biased_expert_independently() -> None:
    calls = {expert: [] for expert in EXPERT_ORDER}

    class Encoder:
        def __init__(self, expert: str) -> None:
            self.expert = expert

        def set_semantic_relation_transform(self, mode: str) -> None:
            calls[self.expert].append(mode)

    graph = object.__new__(JointBridgeGraph)
    object.__setattr__(
        graph,
        "hlt_experts",
        {
            expert: SimpleNamespace(particle_encoder=Encoder(expert))
            for expert in EXPERT_ORDER
        },
    )
    graph.set_semantic_relation_transform("zero", expert_id="TRACK")
    assert calls["TRACK"] == ["zero"]
    assert all(
        calls[expert] == ["active"]
        for expert in EXPERT_ORDER
        if expert != "TRACK"
    )


def test_real_stage_k_output_path_is_the_stage_l_input(tmp_path: Path) -> None:
    source = dict(SOURCE)
    campaign = with_content_hash(
        {"contract": "fixture_campaign_v1", "schema_version": 1,
         "source": source,
         "semantic_control_policy": dict(SEMANTIC_CONTROL_POLICY)}
    )
    graph = with_content_hash(
        {"contract": "fixture_graph_v1", "schema_version": 1,
         "source": source}
    )
    planner = Planner(root=tmp_path, campaign=campaign, graph=graph)
    plan = planner.phase_plan(PHASES[-1], len(PHASES) - 1, {})
    canonical = semantic_controls_bundle_path(tmp_path)
    assert plan["rows"][0]["expected_outputs"] == [str(canonical)]
    artifact = build_valid_semantic_controls(SOURCE)
    write_immutable_json(canonical, artifact)
    assert _load_semantic_controls(tmp_path, campaign=campaign) == artifact
    arbitrary_root = tmp_path / "arbitrary"
    write_immutable_json(
        semantic_controls_bundle_path(arbitrary_root),
        with_content_hash({
            "contract": "fixture_semantics_v1",
            "schema_version": 1,
            "source": source,
        }),
    )
    with pytest.raises(ValueError, match="contract"):
        _load_semantic_controls(arbitrary_root, campaign=campaign)


def _rehash_semantic(payload: dict) -> dict:
    unhashed = copy.deepcopy(payload)
    unhashed.pop("content_hash", None)
    return with_content_hash(unhashed)


def test_stage_k_rejects_placeholder_metric_records() -> None:
    artifact = build_valid_semantic_controls(SOURCE)
    artifact["rows"][0]["metric_records"] = [{"condition_id": "fixture"}]
    artifact["rows"][0]["metric_record_count"] = 1
    with pytest.raises(ValueError, match="schema differs"):
        validate_stage_k_semantic_controls(
            _rehash_semantic(artifact), expected_source=SOURCE
        )


@pytest.mark.parametrize("tamper", ("nonfinite", "delta", "hash", "coverage"))
def test_stage_k_rejects_tampered_semantic_evidence(tamper: str) -> None:
    artifact = build_valid_semantic_controls(SOURCE)
    row = artifact["rows"][0]
    if tamper == "nonfinite":
        row["metric_records"][0]["metrics"]["accuracy"] = float("inf")
    elif tamper == "delta":
        row["metric_records"][0]["metric_deltas"][
            "accuracy_control_minus_reference"
        ] = 0.0
    elif tamper == "hash":
        row["metric_records"][0]["source_artifact_sha256"] = "not-a-hash"
    else:
        row["metric_records"].pop()
        row["metric_record_count"] -= 1
    with pytest.raises(ValueError):
        validate_stage_k_semantic_controls(
            _rehash_semantic(artifact), expected_source=SOURCE
        )


def test_stage_k_rejects_predictor_selector_lineage_tampering() -> None:
    artifact = build_valid_semantic_controls(SOURCE)
    artifact["predictor_architecture_evidence"]["direct_evaluations"][0][
        "candidate_manifest_sha256"
    ] = "f" * 64
    with pytest.raises(ValueError, match="hash differs|coverage differs"):
        validate_stage_k_semantic_controls(
            _rehash_semantic(artifact), expected_source=SOURCE
        )


def test_predictor_architecture_evidence_authenticates_selector_inputs(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        semantic_finalizer, "validate_predictor_candidate", lambda row: row["content_hash"]
    )
    monkeypatch.setattr(
        semantic_finalizer, "validate_locked_target_coordinate", lambda row: row["content_hash"]
    )
    monkeypatch.setattr(
        semantic_finalizer, "validate_materialized_predictor_run", lambda row: row["content_hash"]
    )
    selection = tmp_path / "selection" / "predictor_bundle"
    candidates, run_paths, inference_paths = [], {}, {}
    candidate_specs = (
        ("direct", "A3_SLOT_DECODER_DIRECT", "W_CANONICAL"),
        ("gated", "A4_SLOT_DECODER_GATED", "W_CANONICAL"),
        ("logit", "A0_AFFINE", "W_LOGIT_ONLY"),
    )
    for candidate_id, architecture, objective in candidate_specs:
        seed_artifacts, run_hashes = {}, {}
        run_paths[candidate_id], inference_paths[candidate_id] = {}, {}
        metrics_by_seed = {
            "hybrid_accuracy": {}, "hybrid_cross_entropy": {},
            "normalized_token_error": {},
        }
        for seed in (101, 202, 303):
            run = with_content_hash({
                "contract": "fixture_run_v1", "source": SOURCE,
                "run_id": f"{candidate_id}:{seed}", "pipeline_seed": seed,
                "expert_id": "PT", "architecture": architecture,
                "objective_id": objective,
            })
            run_path = tmp_path / "runs" / candidate_id / str(seed) / "run.json"
            write_immutable_json(run_path, run)
            registration = _digest(f"registration:{candidate_id}:{seed}")
            checkpoint = _digest(f"checkpoint:{candidate_id}:{seed}")
            capacity = _digest(f"capacity:{candidate_id}:{seed}")
            inference = with_content_hash({
                "contract": "retb_predictor_inference_cache_v1", "source": SOURCE,
                "parents": {"predictor_registration": registration,
                            "predictor_checkpoint": checkpoint},
            })
            inference_path = (
                tmp_path / "runs" / candidate_id / str(seed)
                / "val_design" / "predictor_outputs_manifest.json"
            )
            write_immutable_json(inference_path, inference)
            accuracy = 0.5 + seed / 1_000_000
            cross_entropy = 1.0 - seed / 1_000_000
            token_error = 0.25 + seed / 1_000_000
            metric = with_content_hash({
                "contract": "retb_predictor_val_design_metrics_v1",
                "source": SOURCE, "run_sha256": run["content_hash"],
                "run_id": run["run_id"], "pipeline_seed": seed,
                "expert_id": "PT", "capacity_report_sha256": capacity,
                "predictor_registration_sha256": registration,
                "val_design_accuracy": accuracy, "cross_entropy": cross_entropy,
                "normalized_token_error": token_error,
            })
            write_immutable_json(inference_path.parent / "val_design_metrics.json", metric)
            seed_artifacts[str(seed)] = {
                "predictor_registration": registration,
                "predictor_checkpoint": checkpoint,
                "inference_manifest": inference["content_hash"],
                "capacity_report": capacity,
            }
            run_hashes[str(seed)] = run["content_hash"]
            metrics_by_seed["hybrid_accuracy"][str(seed)] = accuracy
            metrics_by_seed["hybrid_cross_entropy"][str(seed)] = cross_entropy
            metrics_by_seed["normalized_token_error"][str(seed)] = token_error
            run_paths[candidate_id][str(seed)] = str(run_path)
            inference_paths[candidate_id][str(seed)] = str(inference_path)
        candidate = with_content_hash({
            "contract": "fixture_candidate_v1", "source": SOURCE,
            "candidate_id": candidate_id, "expert_id": "PT",
            "configuration": {"architecture": architecture, "objective_id": objective},
            "seed_artifacts": seed_artifacts,
            "materialized_run_hashes": run_hashes,
            "metrics_by_seed": metrics_by_seed,
        })
        candidate_path = selection / "inputs" / "candidates" / f"{candidate_id}.json"
        write_immutable_json(candidate_path, candidate)
        candidates.append((candidate_path, candidate))
    coordinate = with_content_hash({
        "contract": "fixture_coordinate_v1", "source": SOURCE,
        "coordinate_id": "coordinate",
    })
    coordinate_path = selection / "inputs" / "coordinates" / "coordinate.json"
    write_immutable_json(coordinate_path, coordinate)
    configuration = {
        "candidate_manifest_paths": [str(path) for path, _ in candidates],
        "coordinate_manifest_paths": [str(coordinate_path)],
        "materialized_run_paths": run_paths,
        "inference_manifest_paths": inference_paths,
        "calibration_artifact_paths": {},
        "capacity_report_paths": {},
        "fusion_checkpoint_paths": {},
        "label_npz_paths": {},
        "label_manifest_paths_by_seed": {},
        "label_manifest_hashes_by_seed": {},
        "label_npz_hashes_by_seed": {},
    }
    configuration_path = selection / "inputs" / "selector_configuration.json"
    configuration_path.parent.mkdir(parents=True, exist_ok=True)
    configuration_path.write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    index = with_content_hash({
        "contract": "retb_predictor_bundle_input_index_v1", "source": SOURCE,
        "schema_version": 1,
        "candidate_count": len(candidates), "coordinate_count": 1,
        "candidate_manifest_hashes": [row["content_hash"] for _, row in candidates],
        "coordinate_manifest_hashes": [coordinate["content_hash"]],
        "selector_configuration_sha256": hashlib.sha256(
            configuration_path.read_bytes()
        ).hexdigest(),
        "predictor_phase_plan_sha256": _digest("phase-plan"),
        "scientific_underperformance_blocks_continuation": False,
    })
    write_immutable_json(selection / "bundle_input_index.json", index)
    campaign = {"source": SOURCE}
    evidence = semantic_finalizer._predictor_architecture_evidence(
        tmp_path, campaign=campaign
    )
    assert evidence["bundle_input_index_sha256"] == index["content_hash"]
    configuration_path.write_text(
        configuration_path.read_text("utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="input-index lineage"):
        semantic_finalizer._predictor_architecture_evidence(
            tmp_path, campaign=campaign
        )


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
