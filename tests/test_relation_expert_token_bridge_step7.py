from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from teacher_logit_reco.relation_expert_token_bridge.bridge_certification import (
    build_bridge_candidate_eligibility,
    certify_bridge_content,
    certify_offline_noninferiority,
    effective_rank,
    validate_bridge_candidate_eligibility,
    validate_bridge_content_certification,
    validate_bridge_noninferiority,
)
from teacher_logit_reco.relation_expert_token_bridge.bridge_selection import (
    select_joint_bridge_coordinates,
)
from teacher_logit_reco.relation_expert_token_bridge.bridge_targets import (
    BridgeOfflineTarget,
    BridgeProjection,
    PilotSlotDecoderDirect,
    alternating_bridge_update,
    bridge_target_objective,
    build_bridge_target_contract,
    build_pilot_architecture_contract,
    deterministic_within_class_negatives,
    fit_bridge_token_normalizer,
    pilot_t0_objective,
    relative_slot_covariance_loss,
    within_class_retrieval_loss,
)
from teacher_logit_reco.relation_expert_token_bridge.bridge_training import (
    BridgeCandidateTrainingConfig,
    BridgePilotDataset,
    PilotTrainingConfig,
    make_bridge_pilot_loader,
    train_pilot_t0,
    train_bridge_candidate,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    with_content_hash,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import EXPERT_ORDER
from teacher_logit_reco.relation_expert_token_bridge.replicas import replica_for
from teacher_logit_reco.relation_expert_token_bridge.step7 import (
    build_stage_e_template_registry,
    build_step7_bundle,
    execute_miniature_stage_e,
    materialize_stage_e_run,
    publish_step7_bundle,
    validate_stage_e_template_registry,
    validate_step7_bundle,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _evidence(batch: int = 4, k: int = 2, d: int = 64):
    torch.manual_seed(7001)
    return (
        {name: torch.randn(batch, k, d) for name in EXPERT_ORDER},
        torch.randn(batch, 5, d),
        torch.ones(batch, 5, dtype=torch.bool),
    )


def test_pilot_a3_c2_shapes_uncertainty_and_gradients() -> None:
    queries = torch.randn(2, 64)
    model = PilotSlotDecoderDirect(
        token_count=2,
        token_dimension=64,
        target_expert_id="TRACK",
        offline_slot_queries=queries,
        dropout=0.0,
    )
    assert model.target_queries.data_ptr() != queries.data_ptr()
    banks, particles, mask = _evidence()
    output = model(
        hlt_token_banks=banks,
        unbiased_particle_states=particles,
        particle_mask=mask,
    )
    assert output["predicted_tokens"].shape == (4, 2, 64)
    assert output["log_variance"].shape == (4, 2, 1)
    assert output["gate"] is None
    output["predicted_tokens"].square().mean().backward()
    assert model.target_queries.grad is not None


def test_t2_projection_is_dimension_correct_and_decodable() -> None:
    values = torch.randn(3, 2, 128, requires_grad=True)
    same = BridgeProjection(128, 128)
    bridge = same(values)
    assert bridge.shape == values.shape
    assert same.decode(bridge).shape == values.shape
    changed = BridgeProjection(128, 64)
    compressed = changed(values)
    assert compressed.shape == (3, 2, 64)
    assert changed.decode(compressed).shape == values.shape
    changed.decode(compressed).square().mean().backward()
    assert values.grad is not None


class _FakeOfflineExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = torch.nn.Linear(64, 64)
        self.head = _TinyHead(64)
        self.particle_encoder = torch.nn.Module()
        self.particle_encoder.mod = torch.nn.Module()
        self.particle_encoder.mod.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(64, 64) for _ in range(8)]
        )

    def forward(self, *, return_details=False, values):
        tokens = self.tokenizer(values)
        logits = self.head(tokens)
        details = {"tokens": tokens, "logits": logits}
        return details if return_details else logits


def test_moving_target_trainability_and_t2_consumers_are_explicit() -> None:
    expert = _FakeOfflineExpert()
    t1 = BridgeOfflineTarget(
        target_mode="T1_ANCHORED_BRIDGE",
        target_expert_id="TRACK",
        expert_model=expert,
        candidate_fusion=_TinyFusion(64),
    )
    names = t1.configure_bridge_trainability(
        unfreeze_final_two_blocks=False
    )
    assert any(name.startswith("expert_model.tokenizer.") for name in names)
    assert not any("particle_encoder" in name for name in names)
    names = t1.configure_bridge_trainability(unfreeze_final_two_blocks=True)
    assert any(".blocks.6." in name for name in names)
    t2 = BridgeOfflineTarget(
        target_mode="T2_PROJECT",
        target_expert_id="TRACK",
        expert_model=_FakeOfflineExpert(),
        candidate_fusion=_TinyFusion(64),
        projection=BridgeProjection(64, 64),
        projected_expert_head=_TinyHead(64),
    )
    t2_names = t2.configure_bridge_trainability(
        unfreeze_final_two_blocks=False
    )
    assert any(name.startswith("projection.") for name in t2_names)
    assert not any("expert_model." in name for name in t2_names)


def test_pilot_and_target_objective_semantics_are_separate() -> None:
    torch.manual_seed(7002)
    predicted = torch.randn(4, 2, 64, requires_grad=True)
    target = torch.randn_like(predicted)
    logits = torch.randn(4, 10, requires_grad=True)
    teacher = torch.randn_like(logits)
    total, pieces = pilot_t0_objective(
        predicted_tokens=predicted,
        target_tokens=target,
        log_variance=torch.zeros(4, 2, 1, requires_grad=True),
        predicted_expert_logits=logits,
        target_expert_logits=teacher,
        predicted_hybrid_logits=logits,
        target_hybrid_logits=teacher,
        labels=torch.arange(4),
    )
    assert set(pieces) == {
        "token",
        "cosine",
        "relation",
        "expertKD",
        "swapKD",
        "CE",
    }
    total.backward(retain_graph=True)
    scalar = logits.square().mean()
    with pytest.raises(ValueError, match="lacks content"):
        bridge_target_objective(
            target_mode="T1_ANCHORED_BRIDGE",
            offline_expert_loss=scalar,
            token_prediction_loss=scalar,
            offline_fusion_loss=scalar,
            t0_logit_loss=scalar,
            lambda_pred=0.10,
        )
    t2, _ = bridge_target_objective(
        target_mode="T2_PROJECT",
        offline_expert_loss=scalar,
        token_prediction_loss=scalar,
        offline_fusion_loss=scalar,
        t0_logit_loss=scalar,
        lambda_pred=0.10,
        t0_project_loss=scalar,
        decoded_t0_logit_loss=scalar,
    )
    assert torch.isfinite(t2)


def test_retrieval_and_covariance_conventions() -> None:
    rings = {label: [f"class-{label}-{index:03d}" for index in range(40)] for label in range(10)}
    first = deterministic_within_class_negatives(
        identity="class-3-000",
        class_label=3,
        class_rings=rings,
        pipeline_seed=101,
        certification=False,
    )
    second = deterministic_within_class_negatives(
        identity="class-3-000",
        class_label=3,
        class_rings=rings,
        pipeline_seed=101,
        certification=False,
    )
    certified = deterministic_within_class_negatives(
        identity="class-3-000",
        class_label=3,
        class_rings=rings,
        pipeline_seed=101,
        certification=True,
    )
    assert first == second and len(first) == len(set(first)) == 31
    assert first != certified and "class-3-000" not in first
    query = torch.randn(3, 2, 64, requires_grad=True)
    candidates = torch.randn(3, 32, 2, 64)
    candidates[:, 0] = query.detach()
    loss = within_class_retrieval_loss(query, candidates)
    loss.backward()
    assert query.grad is not None
    assert relative_slot_covariance_loss(query, query.detach()).item() == pytest.approx(0.0)
    with pytest.raises(ValueError, match="effective batch"):
        relative_slot_covariance_loss(query[:1], query[:1])


def test_alternating_update_enforces_detached_opposite_graph() -> None:
    predictor = torch.nn.Linear(3, 3)
    target = torch.nn.Linear(3, 3)
    predictor_optimizer = torch.optim.SGD(predictor.parameters(), lr=0.1)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1)
    values = torch.randn(4, 3)
    result = alternating_bridge_update(
        phase="predictor",
        predictor_loss=predictor(values).square().mean(),
        target_loss=target(values).square().mean(),
        predictor_optimizer=predictor_optimizer,
        target_optimizer=target_optimizer,
    )
    assert result["opposite_graph_detached"] is True
    leaked = predictor(values).mean() + target(values).mean()
    with pytest.raises(RuntimeError, match="leaked"):
        alternating_bridge_update(
            phase="offline_target",
            predictor_loss=predictor(values).square().mean(),
            target_loss=leaked,
            predictor_optimizer=predictor_optimizer,
            target_optimizer=target_optimizer,
        )


def _certification_fixture(events: int = 320):
    rng = np.random.default_rng(7003)
    pure = rng.normal(size=(events, 2, 64)).astype(np.float32)
    logits = rng.normal(size=(events, 10)).astype(np.float32)
    identities = [f"jet-{index:04d}" for index in range(events)]
    return pure, logits, identities


def test_content_certification_and_noninferiority_fail_closed() -> None:
    pure, logits, identities = _certification_fixture()
    labels = np.arange(len(pure)) % 10
    passed = certify_bridge_content(
        target_mode="T1_ANCHORED_BRIDGE",
        expert_id="TRACK",
        shape_id="SHAPE_COMPACT",
        pipeline_seed=101,
        moving_tokens=pure.copy(),
        t0_tokens=pure,
        predicted_hlt_tokens=pure.copy(),
        frozen_moving_logits={"expert": logits.copy(), "fusion": logits.copy()},
        frozen_t0_logits={"expert": logits, "fusion": logits},
        identities=identities,
        labels=labels,
        candidate_checkpoint_sha256=SHA_A,
        t0_checkpoint_sha256=SHA_B,
        identity_manifest_sha256=SHA_C,
        coordinate_normalizer_sha256=SHA_A,
        t0_normalizer_sha256=SHA_A,
    )
    assert passed["bridge_content_certified"] is True
    assert validate_bridge_content_certification(passed) == passed["content_hash"]
    collapsed = certify_bridge_content(
        target_mode="T1_TASK_BRIDGE",
        expert_id="TRACK",
        shape_id="SHAPE_COMPACT",
        pipeline_seed=101,
        moving_tokens=np.zeros_like(pure),
        t0_tokens=pure,
        predicted_hlt_tokens=np.zeros_like(pure),
        frozen_moving_logits={"expert": logits, "fusion": logits},
        frozen_t0_logits={"expert": logits, "fusion": logits},
        identities=identities,
        labels=labels,
        candidate_checkpoint_sha256=SHA_A,
        t0_checkpoint_sha256=SHA_B,
        identity_manifest_sha256=SHA_C,
        coordinate_normalizer_sha256=SHA_A,
        t0_normalizer_sha256=SHA_A,
    )
    assert collapsed["bridge_content_certified"] is False
    rows = [
        {
            "seed": seed,
            "accuracy": 0.80,
            "cross_entropy": 0.50,
            "per_class_efficiency": {str(index): 0.75 for index in range(10)},
        }
        for seed in (101, 202, 303)
    ]
    noninferior = certify_offline_noninferiority(
        target_mode="T1_ANCHORED_BRIDGE",
        candidate_rows=rows,
        t0_rows=rows,
        candidate_bundle_sha256=SHA_A,
        t0_bundle_sha256=SHA_B,
    )
    assert noninferior["offline_noninferior"] is True
    assert validate_bridge_noninferiority(noninferior) == noninferior["content_hash"]
    certifications = []
    for seed in (101, 202, 303):
        certification = certify_bridge_content(
            target_mode="T1_ANCHORED_BRIDGE",
            expert_id="TRACK",
            shape_id="SHAPE_COMPACT",
            pipeline_seed=seed,
            moving_tokens=pure,
            t0_tokens=pure,
            predicted_hlt_tokens=pure,
            frozen_moving_logits={"expert": logits, "fusion": logits},
            frozen_t0_logits={"expert": logits, "fusion": logits},
            identities=identities,
            labels=labels,
            candidate_checkpoint_sha256=SHA_A,
            t0_checkpoint_sha256=SHA_B,
            identity_manifest_sha256=SHA_C,
            coordinate_normalizer_sha256=SHA_A,
            t0_normalizer_sha256=SHA_A,
        )
        certifications.append(certification)
    eligibility = build_bridge_candidate_eligibility(
        target_mode="T1_ANCHORED_BRIDGE",
        expert_id="TRACK",
        shape_id="SHAPE_COMPACT",
        checkpoint_hashes_by_seed={
            101: SHA_A,
            202: SHA_B,
            303: SHA_C,
        },
        noninferiority=noninferior,
        content_certifications=certifications,
    )
    assert eligibility["maximum_performance_eligible"] is True
    assert eligibility["representation_preserving_claim_eligible"] is True
    assert (
        validate_bridge_candidate_eligibility(eligibility)
        == eligibility["content_hash"]
    )
    with pytest.raises(ValueError, match="identity"):
        build_bridge_candidate_eligibility(
            target_mode="T3_LOGIT",
            expert_id="TRACK",
            shape_id="SHAPE_COMPACT",
            checkpoint_hashes_by_seed={
                101: SHA_A,
                202: SHA_B,
                303: SHA_C,
            },
        )
    assert effective_rank(pure)["effective_rank"] > 0


def test_bridge_normalizer_is_train_bound_and_unclipped() -> None:
    values = np.arange(4 * 2 * 64, dtype=np.float32).reshape(4, 2, 64)
    artifact = fit_bridge_token_normalizer(
        values,
        expert_id="TRACK",
        shape_id="SHAPE_COMPACT",
        target_checkpoint_sha256=SHA_A,
        token_cache_sha256=SHA_B,
        identity_manifest_sha256=SHA_C,
    )
    assert artifact["fit_split"] == "model_train"
    assert artifact["primary_mode"] == "N_UNCLIPPED"
    assert np.asarray(artifact["mean"]).shape == (2, 64)


def test_joint_coordinate_beam_locks_own_fusions_even_all_negative() -> None:
    eligible = {
        name: ["T0_PURE", "T1_ANCHORED_BRIDGE"] for name in EXPERT_ORDER
    }
    metrics = {
        name: {
            mode: {
                "accuracy": 0.4 if mode == "T0_PURE" else 0.3,
                "cross_entropy": 1.0,
                "normalized_token_error": 0.0 if mode == "T0_PURE" else 1.0,
            }
            for mode in eligible[name]
        }
        for name in EXPERT_ORDER
    }

    def pooled(target_tuple, seed):
        assert seed == 41703
        return {
            "accuracy": 0.2 - 0.001 * target_tuple.count("T1_ANCHORED_BRIDGE"),
            "cross_entropy": 2.0,
            "readout_sha256": hashlib.sha256(
                ("pooled:" + ":".join(target_tuple)).encode()
            ).hexdigest(),
        }

    def transformer(target_tuple, seed):
        assert seed == 41703
        key = "-".join(target_tuple)
        return {
            "accuracy": 0.1,
            "cross_entropy": 2.5,
            "fusion_sha256": hashlib.sha256(
                ("fusion:" + key).encode()
            ).hexdigest(),
            "normalizer_set_sha256": hashlib.sha256(
                ("normalizer:" + key).encode()
            ).hexdigest(),
            "target_cache_namespace": key,
        }

    selection = select_joint_bridge_coordinates(
        eligible_modes=eligible,
        default_metrics=metrics,
        pooled_scorer=pooled,
        transformer_scorer=transformer,
        shape_id="SHAPE_COMPACT",
        eligibility_hashes={
            name: {mode: SHA_A for mode in eligible[name]}
            for name in EXPERT_ORDER
        },
    )
    assert selection["selected_target_tuple"] == ["T0_PURE"] * 7
    assert len(selection["locked_coordinate_systems"]) >= 2
    assert all(
        row["target_cache_namespace"]
        for row in selection["locked_coordinate_systems"]
    )


class _TinyHead(torch.nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.linear = torch.nn.Linear(d, 10)

    def forward(self, tokens):
        return self.linear(tokens.mean(dim=1))


class _TinyFusion(torch.nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.linear = torch.nn.Linear(d, 10)

    def forward(self, *, token_banks):
        pooled = torch.stack(
            [token_banks[name].mean(dim=1) for name in EXPERT_ORDER], dim=0
        ).mean(dim=0)
        return self.linear(pooled)


def _pilot_dataset(split: str):
    rng = np.random.default_rng(7004 if split == "model_train" else 7005)
    n, k, d = 20, 2, 64
    target = rng.normal(size=(n, k, d)).astype(np.float32)
    return BridgePilotDataset(
        identities=[f"{split}-{index:04d}" for index in range(n)],
        labels=np.arange(n) % 10,
        hlt_token_banks={
            name: rng.normal(size=(n, k, d)).astype(np.float32)
            for name in EXPERT_ORDER
        },
        unbiased_particle_states=rng.normal(size=(n, 5, d)).astype(np.float32),
        particle_mask=np.ones((n, 5), dtype=bool),
        target_tokens=target,
        token_mean=np.zeros((k, d), dtype=np.float32),
        token_standard_deviation=np.ones((k, d), dtype=np.float32),
        target_expert_logits=rng.normal(size=(n, 10)).astype(np.float32),
        target_hybrid_logits=rng.normal(size=(n, 10)).astype(np.float32),
        other_t0_banks={
            name: rng.normal(size=(n, k, d)).astype(np.float32)
            for name in EXPERT_ORDER
            if name != "TRACK"
        },
        target_expert_id="TRACK",
        split=split,
        lineage_hashes={
            "T0_checkpoint": SHA_A,
            "HLT_encoder_checkpoint": SHA_B,
            "unbiased_HLT_particle_encoder_checkpoint": SHA_B,
            "target_normalizer": SHA_C,
            "T0_fusion": SHA_A,
        },
    )


def test_pilot_r_multi_uses_identity_dependent_epoch_cycle() -> None:
    n, k, d = 8, 2, 64
    identities = [f"pilot-cycle-{index}" for index in range(n)]
    replicas = np.stack(
        [
            np.full((n, k, d), replica, dtype=np.float32)
            for replica in range(4)
        ]
    )
    states = np.stack(
        [
            np.full((n, 3, d), replica, dtype=np.float32)
            for replica in range(4)
        ]
    )
    masks = np.ones((4, n, 3), dtype=bool)
    dataset = BridgePilotDataset(
        identities=identities,
        labels=np.arange(n) % 10,
        hlt_token_banks={name: replicas.copy() for name in EXPERT_ORDER},
        unbiased_particle_states=states,
        particle_mask=masks,
        target_tokens=np.zeros((n, k, d), dtype=np.float32),
        token_mean=np.zeros((k, d), dtype=np.float32),
        token_standard_deviation=np.ones((k, d), dtype=np.float32),
        target_expert_logits=np.zeros((n, 10), dtype=np.float32),
        target_hybrid_logits=np.zeros((n, 10), dtype=np.float32),
        other_t0_banks={
            name: np.zeros((n, k, d), dtype=np.float32)
            for name in EXPERT_ORDER
            if name != "TRACK"
        },
        target_expert_id="TRACK",
        split="model_train",
        lineage_hashes={
            "T0_checkpoint": SHA_A,
            "HLT_encoder_checkpoint": SHA_B,
            "unbiased_HLT_particle_encoder_checkpoint": SHA_B,
            "target_normalizer": SHA_C,
            "T0_fusion": SHA_A,
        },
    )
    for one_based_epoch in (1, 2, 3, 4):
        dataset.set_epoch(one_based_epoch)
        selected = [
            int(dataset[index]["hlt_token_banks"]["BASE4"][0, 0])
            for index in range(n)
        ]
        assert selected == [
            replica_for(
                policy="R_MULTI",
                logical_role="model_train",
                epoch=one_based_epoch - 1,
                canonical_identity=identity,
            )
            for identity in identities
        ]
    assert {
        int(dataset[index]["hlt_token_banks"]["BASE4"][0, 0])
        for index in range(n)
    } != {0}


def test_stage_e_registry_pilot_training_bundle_and_miniature(tmp_path: Path) -> None:
    registry = build_stage_e_template_registry()
    assert validate_stage_e_template_registry(registry) == registry["content_hash"]
    assert registry["pilot_membership_count"] == 105
    assert registry["candidate_template_count_per_expert_shape_seed"] == 19
    assert registry["candidate_membership_count"] == 1995
    pilot_run = materialize_stage_e_run(
        template_registry=registry,
        pipeline_seed=101,
        expert_id="TRACK",
        shape_id="SHAPE_COMPACT",
        target_mode="T0_PURE",
        lambda_pred=0.0,
        bridge_dimension=None,
        unfreeze_final_two_blocks=False,
        t0_checkpoint_sha256=SHA_A,
        hlt_encoder_checkpoint_sha256=SHA_B,
        unbiased_particle_encoder_checkpoint_sha256=SHA_B,
        pilot_checkpoint_sha256=None,
    )
    candidate = materialize_stage_e_run(
        template_registry=registry,
        pipeline_seed=101,
        expert_id="TRACK",
        shape_id="SHAPE_COMPACT",
        target_mode="T1_ANCHORED_BRIDGE",
        lambda_pred=0.10,
        bridge_dimension=None,
        unfreeze_final_two_blocks=False,
        t0_checkpoint_sha256=SHA_A,
        hlt_encoder_checkpoint_sha256=SHA_B,
        unbiased_particle_encoder_checkpoint_sha256=SHA_B,
        pilot_checkpoint_sha256=SHA_C,
    )
    task_candidate = materialize_stage_e_run(
        template_registry=registry,
        pipeline_seed=101,
        expert_id="TRACK",
        shape_id="SHAPE_COMPACT",
        target_mode="T1_TASK_BRIDGE",
        lambda_pred=0.10,
        bridge_dimension=None,
        unfreeze_final_two_blocks=False,
        t0_checkpoint_sha256=SHA_A,
        hlt_encoder_checkpoint_sha256=SHA_B,
        unbiased_particle_encoder_checkpoint_sha256=SHA_B,
        pilot_checkpoint_sha256=SHA_C,
    )
    project_candidate = materialize_stage_e_run(
        template_registry=registry,
        pipeline_seed=101,
        expert_id="TRACK",
        shape_id="SHAPE_COMPACT",
        target_mode="T2_PROJECT",
        lambda_pred=0.10,
        bridge_dimension=64,
        unfreeze_final_two_blocks=False,
        t0_checkpoint_sha256=SHA_A,
        hlt_encoder_checkpoint_sha256=SHA_B,
        unbiased_particle_encoder_checkpoint_sha256=SHA_B,
        pilot_checkpoint_sha256=SHA_C,
    )
    logit_candidate = materialize_stage_e_run(
        template_registry=registry,
        pipeline_seed=101,
        expert_id="TRACK",
        shape_id="SHAPE_COMPACT",
        target_mode="T3_LOGIT",
        lambda_pred=0.0,
        bridge_dimension=None,
        unfreeze_final_two_blocks=False,
        t0_checkpoint_sha256=SHA_A,
        hlt_encoder_checkpoint_sha256=SHA_B,
        unbiased_particle_encoder_checkpoint_sha256=SHA_B,
        pilot_checkpoint_sha256=SHA_C,
    )
    with pytest.raises(ValueError, match="pilot_checkpoint"):
        materialize_stage_e_run(
            template_registry=registry,
            pipeline_seed=101,
            expert_id="TRACK",
            shape_id="SHAPE_COMPACT",
            target_mode="T1_TASK_BRIDGE",
            lambda_pred=0.10,
            bridge_dimension=None,
            unfreeze_final_two_blocks=False,
            t0_checkpoint_sha256=SHA_A,
            hlt_encoder_checkpoint_sha256=SHA_B,
            unbiased_particle_encoder_checkpoint_sha256=SHA_B,
            pilot_checkpoint_sha256=None,
        )
    assert pilot_run["run_id"] != candidate["run_id"]
    assert len(
        {
            pilot_run["run_id"],
            candidate["run_id"],
            task_candidate["run_id"],
            project_candidate["run_id"],
            logit_candidate["run_id"],
        }
    ) == 5
    train = _pilot_dataset("model_train")
    val = _pilot_dataset("val_stop")
    model = PilotSlotDecoderDirect(
        token_count=2,
        token_dimension=64,
        target_expert_id="TRACK",
        offline_slot_queries=torch.randn(2, 64),
        dropout=0.0,
    )
    registration = train_pilot_t0(
        model=model,
        train_loader=make_bridge_pilot_loader(
            train, batch_size=10, seed=101, training=True
        ),
        val_stop_loader=make_bridge_pilot_loader(
            val, batch_size=10, seed=0, training=False
        ),
        expert_head=_TinyHead(64),
        hybrid_fusion=_TinyFusion(64),
        target_expert_id="TRACK",
        output_dir=tmp_path / "pilot",
        materialized_run=pilot_run,
        pilot_architecture_sha256=build_pilot_architecture_contract()[
            "content_hash"
        ],
        global_determinism_sha256=SHA_A,
        config=PilotTrainingConfig(
            seed=101,
            maximum_epochs=2,
            batch_size=10,
            campaign_profile="miniature_test",
        ),
    )
    assert registration["epochs_completed"] == 2
    values = torch.randn(8, 4)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(values), batch_size=4, shuffle=False
    )
    predictor = torch.nn.Linear(4, 4)
    moving_target = torch.nn.Linear(4, 4)

    def phase_loss(batch, predictor_model, target_model, phase):
        inputs = batch[0]
        if phase == "predictor":
            return (
                predictor_model(inputs) - target_model(inputs).detach()
            ).square().mean()
        return (
            target_model(inputs) - predictor_model(inputs).detach()
        ).square().mean()

    candidate_registration = train_bridge_candidate(
        predictor=predictor,
        offline_target=moving_target,
        train_loader=loader,
        val_stop_evaluator=lambda p, t, _device: {
            "accuracy": float(
                -((p(values) - t(values)) ** 2).mean().detach()
            ),
            "cross_entropy": float(
                ((p(values) - t(values)) ** 2).mean().detach()
            ),
        },
        phase_loss_builder=phase_loss,
        output_dir=tmp_path / "candidate",
        materialized_run=candidate,
        pilot_checkpoint=with_content_hash(
            {
                "contract": "retb_pilot_t0_registration_v1",
                "schema_version": 1,
                "checkpoint_sha256": SHA_C,
                "pipeline_seed": 101,
                "expert_id": "TRACK",
                "shape_id": "SHAPE_COMPACT",
                "dataset_lineage_hashes": {
                    "T0_checkpoint": SHA_A,
                    "HLT_encoder_checkpoint": SHA_B,
                    "unbiased_HLT_particle_encoder_checkpoint": SHA_B,
                    "target_normalizer": SHA_C,
                    "T0_fusion": SHA_A,
                },
            }
        ),
        global_determinism_sha256=SHA_A,
        config=BridgeCandidateTrainingConfig(
            seed=101,
            target_mode="T1_ANCHORED_BRIDGE",
            maximum_epochs=2,
            effective_batch_size=4,
            campaign_profile="miniature_test",
        ),
    )
    assert candidate_registration["epochs_completed"] == 2
    assert candidate_registration["alternating_detach_verified_every_update"] is True
    pilot_parent = with_content_hash(
        {
            "contract": "retb_pilot_t0_registration_v1",
            "schema_version": 1,
            "checkpoint_sha256": SHA_C,
            "pipeline_seed": 101,
            "expert_id": "TRACK",
            "shape_id": "SHAPE_COMPACT",
            "dataset_lineage_hashes": {
                "T0_checkpoint": SHA_A,
                "HLT_encoder_checkpoint": SHA_B,
                "unbiased_HLT_particle_encoder_checkpoint": SHA_B,
                "target_normalizer": SHA_C,
                "T0_fusion": SHA_A,
            },
        }
    )
    logit_registration = train_bridge_candidate(
        predictor=torch.nn.Linear(4, 10),
        offline_target=None,
        train_loader=loader,
        val_stop_evaluator=lambda p, _t, _device: {
            "accuracy": 0.1,
            "cross_entropy": 2.3,
        },
        phase_loss_builder=lambda batch, p, _t, phase: (
            p(batch[0]).square().mean()
            if phase == "predictor"
            else pytest.fail("T3 requested an offline-target phase")
        ),
        output_dir=tmp_path / "logit_candidate",
        materialized_run=logit_candidate,
        pilot_checkpoint=pilot_parent,
        global_determinism_sha256=SHA_A,
        config=BridgeCandidateTrainingConfig(
            seed=101,
            target_mode="T3_LOGIT",
            maximum_epochs=2,
            effective_batch_size=4,
            campaign_profile="miniature_test",
        ),
    )
    assert logit_registration["token_target_trained"] is False
    assert logit_registration["token_fidelity_claim"] is False
    completion = execute_miniature_stage_e(
        [pilot_run, candidate],
        executor=lambda _row: {
            "status": "completed",
            "performance_based_termination": False,
        },
    )
    assert completion["completed"] == {"BRIDGE_PILOT": 1, "BRIDGE_TARGET": 1}
    snapshot = source_snapshot(Path(__file__).resolve().parents[1])
    bundle = build_step7_bundle(
        campaign_spec_sha256=SHA_A,
        step6_bundle_sha256=SHA_B,
        global_determinism_sha256=SHA_C,
        source_snapshot=snapshot,
    )
    assert validate_step7_bundle(bundle) == bundle["step7_bundle"]["content_hash"]
    publication = publish_step7_bundle(
        campaign_root=tmp_path / "campaign", bundle=bundle
    )
    assert publication["step7_bundle_sha256"] == bundle["step7_bundle"][
        "content_hash"
    ]
    assert build_bridge_target_contract()["modes"]["T3_LOGIT"][
        "token_fidelity_claim"
    ] is False
