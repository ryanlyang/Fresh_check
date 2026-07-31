from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    with_content_hash,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_cache import (
    calibrate_predictor_inference_cache,
    load_predictor_inference_cache,
    publish_predictor_inference_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_losses import (
    DeterministicGradNorm,
    apply_uncertainty_calibration,
    fit_uncertainty_calibration,
    inverse_normalize_tokens,
    normalize_tokens,
    predictor_objective,
    directional_agreement_loss,
    temperature_two_kl,
    token_relation_loss,
    token_tail_diagnostics,
    uncertainty_weighted_token_loss,
    validate_uncertainty_calibration,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_training import (
    PredictorDataset,
    PredictorTrainingConfig,
    evaluate_predictor,
    make_predictor_loader,
    train_predictor,
)
from teacher_logit_reco.relation_expert_token_bridge.target_coordinates import (
    target_slot_queries,
)
from teacher_logit_reco.relation_expert_token_bridge.bridge_targets import (
    BridgeProjection,
)
from teacher_logit_reco.relation_expert_token_bridge.predictors import (
    RetbTokenPredictor,
    TypedHLTEvidence,
    build_predictor_capacity_report,
    predictor_analytical_flops,
    select_widened_resmlp_width,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import EXPERT_ORDER
from teacher_logit_reco.relation_expert_token_bridge.step9 import (
    build_stage_f_registry,
    build_stage_g_registry,
    build_stage_h_policy,
    build_step9_bundle,
    materialize_predictor_run,
    validate_stage_f_registry,
    validate_stage_g_registry,
    validate_materialized_predictor_run,
    validate_step9_bundle,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SOURCE = {
    "source_commit": "1" * 40,
    "source_status_sha256": "2" * 64,
    "source_dirty": True,
}


def _evidence(batch: int = 3, k: int = 2, d: int = 64):
    generator = torch.Generator().manual_seed(9001)
    banks = {
        expert: torch.randn(batch, k, d, generator=generator)
        for expert in EXPERT_ORDER
    }
    particles = torch.randn(batch, 5, d, generator=generator)
    mask = torch.ones(batch, 5, dtype=torch.bool)
    relation = {
        name: torch.randn(batch, 4, d, generator=generator)
        for name in ("PT", "TRACK", "REGION")
    }
    relation_masks = {
        name: torch.ones(batch, 4, dtype=torch.bool) for name in relation
    }
    return banks, particles, mask, relation, relation_masks


@pytest.mark.parametrize(
    ("architecture", "context"),
    (
        ("A0_AFFINE", "C0_SELF"),
        ("A1_RESMLP", "C0_SELF"),
        ("A2_TOKEN_ENCODER", "C0_SELF"),
        ("A3_SLOT_DECODER_DIRECT", "C0_SELF"),
        ("A3_SLOT_DECODER_DIRECT", "C1_NATIVE"),
        ("A3_SLOT_DECODER_DIRECT", "C2_ALL"),
        ("A3_SLOT_DECODER_DIRECT", "C3_ALL_PARTICLE"),
        ("A4_SLOT_DECODER_GATED", "C2_ALL"),
    ),
)
def test_all_predictor_architectures_and_contexts_backpropagate(
    architecture: str, context: str
) -> None:
    banks, particles, mask, relation, relation_masks = _evidence()
    queries = torch.randn(2, 64)
    model = RetbTokenPredictor(
        architecture=architecture,
        context=context,
        target_expert_id="PT",
        token_count=2,
        token_dimension=64,
        offline_slot_queries=queries,
        uncertainty_head="U_SLOT",
        dropout=0.0,
    )
    output = model(
        corresponding_hlt_tokens=banks["PT"],
        hlt_token_banks=banks,
        unbiased_particle_states=particles,
        particle_mask=mask,
        relation_particle_states=relation,
        relation_particle_masks=relation_masks,
    )
    assert output["predicted_tokens"].shape == (3, 2, 64)
    assert output["log_variance"].shape == (3, 2, 1)
    assert torch.all(output["log_variance"] >= -8)
    assert torch.all(output["log_variance"] <= 4)
    if architecture == "A4_SLOT_DECODER_GATED":
        assert torch.allclose(
            output["gate"],
            torch.full_like(output["gate"], torch.sigmoid(torch.tensor(-2.0))),
        )
    else:
        assert output["gate"] is None
    output["predicted_tokens"].square().mean().backward()
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if architecture.startswith("A3") or architecture.startswith("A4"):
        assert model.target_queries.data_ptr() != queries.data_ptr()


@pytest.mark.parametrize(
    ("head", "width"), (("U_SLOT", 1), ("U_GROUP4", 4), ("U_DIAGONAL", 64))
)
def test_uncertainty_heads_have_exact_width(head: str, width: int) -> None:
    banks, particles, mask, relation, relation_masks = _evidence()
    model = RetbTokenPredictor(
        architecture="A3_SLOT_DECODER_DIRECT",
        context="C2_ALL",
        target_expert_id="TRACK",
        token_count=2,
        token_dimension=64,
        offline_slot_queries=torch.randn(2, 64),
        uncertainty_head=head,
        dropout=0.0,
    )
    output = model(
        corresponding_hlt_tokens=banks["TRACK"],
        hlt_token_banks=banks,
        unbiased_particle_states=particles,
        particle_mask=mask,
        relation_particle_states=relation,
        relation_particle_masks=relation_masks,
    )
    assert output["log_variance"].shape == (3, 2, width)


def test_typed_evidence_policies_and_zero_control_are_nontrivial() -> None:
    banks, particles, mask, relation, relation_masks = _evidence()
    lengths = {"C0_SELF": 2, "C1_NATIVE": 7, "C2_ALL": 21, "C3_ALL_PARTICLE": 31}
    for context, expected_length in lengths.items():
        builder = TypedHLTEvidence(
            token_dimension=64, target_expert_id="PT", context=context
        )
        values, padding = builder(
            corresponding_hlt_tokens=banks["PT"],
            hlt_token_banks=banks,
            unbiased_particle_states=particles,
            particle_mask=mask,
            relation_particle_states=relation,
            relation_particle_masks=relation_masks,
        )
        assert values.shape == (3, expected_length, 64)
        assert padding.shape == (3, expected_length)
        zero, _ = builder(
            corresponding_hlt_tokens=banks["PT"],
            hlt_token_banks=banks,
            unbiased_particle_states=particles,
            particle_mask=mask,
            relation_particle_states=relation,
            relation_particle_masks=relation_masks,
            zero_evidence=True,
        )
        assert torch.count_nonzero(zero) == 0
        assert sum(parameter.numel() for parameter in builder.parameters()) > 0


def test_typed_evidence_projects_real_128_wide_particles_and_mixed_banks() -> None:
    batch, k = 2, 2
    generator = torch.Generator().manual_seed(9017)
    banks = {
        expert: torch.randn(
            batch,
            k,
            64 if index % 2 == 0 else 128,
            generator=generator,
        )
        for index, expert in enumerate(EXPERT_ORDER)
    }
    particles = torch.randn(batch, 5, 128, generator=generator)
    mask = torch.ones(batch, 5, dtype=torch.bool)
    relation = {
        name: torch.randn(batch, 4, 128, generator=generator)
        for name in ("PT", "TRACK", "REGION")
    }
    relation_masks = {
        name: torch.ones(batch, 4, dtype=torch.bool) for name in relation
    }
    builder = TypedHLTEvidence(
        token_dimension=64,
        target_expert_id="BASE4",
        context="C3_ALL_PARTICLE",
    )
    values, padding = builder(
        corresponding_hlt_tokens=banks["BASE4"],
        hlt_token_banks=banks,
        unbiased_particle_states=particles,
        particle_mask=mask,
        relation_particle_states=relation,
        relation_particle_masks=relation_masks,
    )
    assert values.shape == (batch, 7 * k + 5 + 3 * 4, 64)
    assert padding.shape == values.shape[:2]
    values.square().mean().backward()
    assert all(
        projection.weight.grad is not None
        for projection in builder.particle_projections.values()
    )


def test_gated_formula_is_anchor_plus_sigmoid_gate_times_delta() -> None:
    banks, particles, mask, relation, relation_masks = _evidence()
    model = RetbTokenPredictor(
        architecture="A4_SLOT_DECODER_GATED",
        context="C0_SELF",
        target_expert_id="BASE4",
        token_count=2,
        token_dimension=64,
        offline_slot_queries=torch.randn(2, 64),
        dropout=0.0,
    )
    captured = {}

    def hook(_module, _inputs, output):
        captured["delta"] = output

    handle = model.output_norm.register_forward_hook(hook)
    result = model(corresponding_hlt_tokens=banks["BASE4"])
    handle.remove()
    anchor = model.anchor_map(model.anchor_norm(banks["BASE4"]))
    assert torch.allclose(
        result["predicted_tokens"],
        anchor + result["gate"] * captured["delta"],
    )


def test_wrong_event_evidence_shuffle_is_nontrivial() -> None:
    banks, particles, mask, relation, relation_masks = _evidence()
    model = RetbTokenPredictor(
        architecture="A3_SLOT_DECODER_DIRECT",
        context="C2_ALL",
        target_expert_id="PT",
        token_count=2,
        token_dimension=64,
        offline_slot_queries=torch.randn(2, 64),
        dropout=0.0,
    )
    model.eval()
    kwargs = {
        "corresponding_hlt_tokens": banks["PT"],
        "hlt_token_banks": banks,
        "unbiased_particle_states": particles,
        "particle_mask": mask,
        "relation_particle_states": relation,
        "relation_particle_masks": relation_masks,
    }
    ordinary = model(**kwargs)["predicted_tokens"]
    shuffled = model(
        **kwargs, evidence_batch_permutation=torch.tensor([1, 2, 0])
    )["predicted_tokens"]
    assert not torch.allclose(ordinary, shuffled)


def test_resmlp_slots_do_not_communicate() -> None:
    model = RetbTokenPredictor(
        architecture="A1_RESMLP",
        context="C0_SELF",
        target_expert_id="PT",
        token_count=2,
        token_dimension=64,
        offline_slot_queries=torch.randn(2, 64),
        dropout=0.0,
    )
    tokens = torch.randn(2, 2, 64)
    changed = tokens.clone()
    changed[:, 0] += 100.0
    ordinary = model(corresponding_hlt_tokens=tokens)["predicted_tokens"]
    perturbed = model(corresponding_hlt_tokens=changed)["predicted_tokens"]
    assert torch.equal(ordinary[:, 1], perturbed[:, 1])
    assert not torch.equal(ordinary[:, 0], perturbed[:, 0])


def test_normalization_losses_and_tail_controls_match_hand_fixtures() -> None:
    values = torch.tensor([[[0.0, 20.0, -20.0, 2.0]]])
    mean = torch.zeros(1, 4)
    std = torch.ones(1, 4)
    assert torch.equal(
        normalize_tokens(values, mean=mean, standard_deviation=std, mode="N_UNCLIPPED"),
        values,
    )
    clipped = normalize_tokens(
        values, mean=mean, standard_deviation=std, mode="N_CLIP8"
    )
    assert clipped.tolist() == [[[0.0, 8.0, -8.0, 2.0]]]
    assert torch.equal(
        inverse_normalize_tokens(clipped, mean=mean, standard_deviation=std),
        clipped,
    )
    target = torch.zeros_like(values)
    log_variance = torch.zeros(1, 1, 1)
    expected_huber = torch.nn.functional.huber_loss(
        values, target, delta=0.5, reduction="none"
    ).mean()
    assert torch.allclose(
        uncertainty_weighted_token_loss(
            values, target, log_variance, uncertainty_head="U_SLOT"
        ),
        expected_huber,
    )
    identical = torch.randn(2, 3, 8)
    assert token_relation_loss(identical, identical).item() == pytest.approx(0)
    tails = token_tail_diagnostics(
        values.numpy(), labels=np.array([3], dtype=np.int64)
    )
    assert tails["8"]["element_count"] == 2
    assert tails["8"]["event_count_by_class"][3] == 1


def test_fixed_objectives_read_only_declared_information() -> None:
    prediction = torch.randn(2, 2, 8, requires_grad=True)
    target = torch.randn(2, 2, 8)
    log_variance = torch.zeros(2, 2, 1, requires_grad=True)
    total, details = predictor_objective(
        weight_id="W_TOKEN_ONLY",
        uncertainty_head="U_SLOT",
        predicted_tokens=prediction,
        target_tokens=target,
        log_variance=log_variance,
    )
    assert set(details["terms"]) == {"token"}
    total.backward()
    assert prediction.grad is not None and log_variance.grad is not None
    logits = torch.randn(2, 10, requires_grad=True)
    teacher = torch.randn(2, 10)
    hybrid = torch.randn(2, 10, requires_grad=True)
    total, details = predictor_objective(
        weight_id="W_LOGIT_ONLY",
        uncertainty_head="U_SLOT",
        predicted_tokens=prediction.detach(),
        target_tokens=None,
        log_variance=log_variance.detach(),
        predicted_expert_logits=logits,
        target_expert_logits=teacher,
        predicted_hybrid_logits=hybrid,
        target_hybrid_logits=teacher,
        labels=torch.tensor([0, 1]),
    )
    assert set(details["terms"]) == {"expertKD", "swapKD", "CE"}
    assert not details["faithful_token_recovery_claim_eligible"]
    assert torch.isfinite(total)


def test_directional_kl_and_ce_terms_match_direct_formulas() -> None:
    prediction = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    assert directional_agreement_loss(prediction, target).item() == pytest.approx(
        0.5
    )
    student = torch.tensor([[0.2, -0.1, *([0.0] * 8)]], requires_grad=True)
    teacher = torch.tensor([[0.5, 0.0, *([0.0] * 8)]])
    expected_kl = torch.nn.functional.kl_div(
        torch.log_softmax(student / 2.0, dim=-1),
        torch.softmax(teacher / 2.0, dim=-1),
        reduction="batchmean",
    ) * 4.0
    assert torch.allclose(temperature_two_kl(student, teacher), expected_kl)
    hybrid = torch.tensor([[1.0, 0.0, *([0.0] * 8)]], requires_grad=True)
    total, details = predictor_objective(
        weight_id="W_LOGIT_ONLY",
        uncertainty_head="U_SLOT",
        predicted_tokens=prediction,
        target_tokens=None,
        log_variance=torch.zeros(1, 2, 1),
        predicted_expert_logits=student,
        target_expert_logits=teacher,
        predicted_hybrid_logits=hybrid,
        target_hybrid_logits=teacher,
        labels=torch.tensor([1]),
    )
    expected = (
        temperature_two_kl(student, teacher)
        + temperature_two_kl(hybrid, teacher)
        + 0.5 * torch.nn.functional.cross_entropy(hybrid, torch.tensor([1]))
    )
    assert torch.allclose(total, expected)
    assert torch.allclose(
        details["terms"]["CE"],
        torch.nn.functional.cross_entropy(hybrid, torch.tensor([1])),
    )


def test_gradnorm_is_model_train_only_clipped_and_resumable() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    terms = {
        name: (index + 1) * parameter.square()
        for index, name in enumerate(
            ("token", "cosine", "relation", "expertKD", "swapKD", "CE")
        )
    }
    balancer = DeterministicGradNorm()
    weights = balancer.update(
        terms=terms, shared_parameters=[parameter], split="model_train"
    )
    assert sum(weights.values()) == pytest.approx(2.6)
    for name, initial in balancer.initial.items():
        assert 0.1 * initial <= weights[name] <= 10.0 * initial
    restored = DeterministicGradNorm()
    restored.load_state_dict(balancer.state_dict())
    assert restored.current == weights
    with pytest.raises(ValueError, match="model_train"):
        balancer.update(
            terms=terms, shared_parameters=[parameter], split="val_design"
        )


def test_uncertainty_calibration_is_label_free_and_deterministic() -> None:
    rng = np.random.default_rng(9002)
    target = rng.normal(size=(20, 2, 64)).astype(np.float32)
    prediction = target + 0.2
    log_variance = np.zeros((20, 2, 4), dtype=np.float32)
    artifact = fit_uncertainty_calibration(
        expert_id="PT",
        uncertainty_head="U_GROUP4",
        predicted_tokens=prediction,
        target_tokens=target,
        log_variance=log_variance,
        predictor_checkpoint_sha256=SHA_A,
        predictor_registration_sha256=SHA_B,
        predictor_inference_manifest_sha256=SHA_C,
        target_cache_manifest_sha256=SHA_A,
        identity_manifest_sha256=SHA_B,
    )
    validate_uncertainty_calibration(artifact)
    assert not artifact["labels_consumed"]
    assert len(artifact["coverage_error_curve"]) == 10
    calibrated = apply_uncertainty_calibration(
        torch.from_numpy(log_variance), artifact
    )
    assert calibrated.shape == torch.Size([20, 2, 4])


def test_uncertainty_calibration_minimizes_the_clipped_objective() -> None:
    target = np.zeros((6, 1, 4), dtype=np.float32)
    prediction = np.asarray(
        [
            [[0.0, 0.0, 0.0, 0.0]],
            [[0.01, 0.01, 0.01, 0.01]],
            [[0.1, 0.1, 0.1, 0.1]],
            [[1.0, 1.0, 1.0, 1.0]],
            [[10.0, 10.0, 10.0, 10.0]],
            [[100.0, 100.0, 100.0, 100.0]],
        ],
        dtype=np.float32,
    )
    log_variance = np.asarray(
        [[[-8.0]], [[-4.0]], [[0.0]], [[2.0]], [[4.0]], [[4.0]]],
        dtype=np.float32,
    )
    artifact = fit_uncertainty_calibration(
        expert_id="PT",
        uncertainty_head="U_SLOT",
        predicted_tokens=prediction,
        target_tokens=target,
        log_variance=log_variance,
        predictor_checkpoint_sha256=SHA_A,
        predictor_registration_sha256=SHA_B,
        predictor_inference_manifest_sha256=SHA_C,
        target_cache_manifest_sha256=SHA_A,
        identity_manifest_sha256=SHA_B,
    )
    selected = artifact["additive_offset_by_group"][0]

    def objective(offset):
        calibrated = np.clip(log_variance + offset, -8.0, 4.0)
        return float(
            (
                np.exp(-calibrated) * np.square(prediction - target)
                + calibrated
            ).mean(dtype=np.float64)
        )

    grid = np.linspace(-12.0, 12.0, 20_001)
    assert objective(selected) <= min(map(objective, grid)) + 1.0e-8


def test_capacity_controls_and_stage_registries_are_exact() -> None:
    stage_f = build_stage_f_registry()
    stage_g = build_stage_g_registry()
    assert validate_stage_f_registry(stage_f) == stage_f["content_hash"]
    assert validate_stage_g_registry(stage_g) == stage_g["content_hash"]
    assert stage_f["membership_count"] == 72
    assert stage_g["membership_count"] == 432
    assert not build_stage_h_policy()["performance_based_termination"]
    width = select_widened_resmlp_width(
        token_dimension=64, target_incremental_parameters=50_000
    )
    assert width["hidden_width"] >= 64
    assert (
        abs(width["incremental_parameter_count"] - 50_000)
        == width["incremental_parameter_mismatch"]
    )
    flops = predictor_analytical_flops(
        architecture="A3_SLOT_DECODER_DIRECT",
        batch_size=8,
        token_count=2,
        token_dimension=64,
        evidence_token_count=21,
        uncertainty_width_value=1,
        evidence_projection_flops=1234,
    )
    assert flops > 1234
    profile = {
        "parameter_count": 50_000,
        "analytical_flops": flops,
        "measured_latency_seconds_mean": 0.01,
        "measured_iterations": 5,
        "peak_memory_bytes": None,
        "latency_used_for_selection": False,
    }
    report = build_predictor_capacity_report(
        run_id="run",
        architecture="A3_SLOT_DECODER_DIRECT",
        token_dimension=64,
        selected_profile=profile,
        affine_baseline_parameter_count=5_000,
        zero_evidence_profile=profile,
    )
    assert not report["zero_evidence_decoder"]["parameters_removed_or_frozen"]


class _TinyHead(torch.nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.linear = torch.nn.Linear(dimension, 10)

    def forward(self, tokens):
        return self.linear(tokens.mean(dim=1))


class _TinyFusion(torch.nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.linear = torch.nn.Linear(len(EXPERT_ORDER) * dimension, 10)

    def forward(self, token_banks=None):
        pooled = [token_banks[name].mean(dim=1) for name in EXPERT_ORDER]
        return self.linear(torch.cat(pooled, dim=-1))


def _training_fixture(split: str, events: int = 20):
    rng = np.random.default_rng(9003 if split == "model_train" else 9004)
    k, d = 2, 64
    banks = {
        expert: rng.normal(size=(events, k, d)).astype(np.float32)
        for expert in EXPERT_ORDER
    }
    target = banks["PT"] + rng.normal(
        scale=0.05, size=(events, k, d)
    ).astype(np.float32)
    head = _TinyHead(d)
    fusion = _TinyFusion(d)
    with torch.no_grad():
        target_logits = head(torch.from_numpy(target)).numpy()
        oracle = dict(banks)
        oracle["PT"] = target
        hybrid_logits = fusion(
            {name: torch.from_numpy(value) for name, value in oracle.items()}
        ).numpy()
    dataset = PredictorDataset(
        identities=[f"{split}-{index}" for index in range(events)],
        labels=np.arange(events, dtype=np.int64) % 10,
        hlt_token_banks=banks,
        unbiased_particle_states=rng.normal(
            size=(events, 4, d)
        ).astype(np.float32),
        particle_mask=np.ones((events, 4), dtype=bool),
        target_tokens=target,
        target_expert_logits=target_logits,
        target_hybrid_logits=hybrid_logits,
        other_oracle_banks={
            expert: value for expert, value in oracle.items() if expert != "PT"
        },
        target_expert_id="PT",
        token_mean=np.zeros((k, d), dtype=np.float32),
        token_standard_deviation=np.ones((k, d), dtype=np.float32),
        normalization_mode="N_UNCLIPPED",
        split=split,
        lineage_hashes={"split": SHA_A if split == "model_train" else SHA_B},
    )
    return dataset, head, fusion


def test_predictor_dataset_uses_identity_epoch_bound_common_r_multi_replica():
    events, k, d = 8, 2, 64
    ids = [f"identity-{index}" for index in range(events)]
    replicas = np.stack(
        [
            np.full((events, k, d), replica, dtype=np.float32)
            for replica in range(4)
        ]
    )
    states = np.stack(
        [
            np.full((events, 3, d), replica, dtype=np.float32)
            for replica in range(4)
        ]
    )
    masks = np.ones((4, events, 3), dtype=bool)
    dataset = PredictorDataset(
        identities=ids,
        labels=np.arange(events, dtype=np.int64) % 10,
        hlt_token_banks={expert: replicas for expert in EXPERT_ORDER},
        unbiased_particle_states=states,
        particle_mask=masks,
        target_tokens=np.zeros((events, k, d), dtype=np.float32),
        target_expert_logits=np.zeros((events, 10), dtype=np.float32),
        target_hybrid_logits=np.zeros((events, 10), dtype=np.float32),
        other_oracle_banks={
            expert: np.zeros((events, k, d), dtype=np.float32)
            for expert in EXPERT_ORDER
            if expert != "PT"
        },
        target_expert_id="PT",
        token_mean=np.zeros((k, d), dtype=np.float32),
        token_standard_deviation=np.ones((k, d), dtype=np.float32),
        normalization_mode="N_UNCLIPPED",
        split="model_train",
        lineage_hashes={"split": SHA_A},
        relation_particle_states={
            name: states for name in ("PT", "TRACK", "REGION")
        },
        relation_particle_masks={
            name: masks for name in ("PT", "TRACK", "REGION")
        },
        realization_policy="R_MULTI",
    )
    first = [dataset[index]["replica_id"] for index in range(events)]
    for index, replica in enumerate(first):
        row = dataset[index]
        assert set(
            float(values[0, 0])
            for values in row["hlt_token_banks"].values()
        ) == {float(replica)}
        assert float(row["unbiased_particle_states"][0, 0]) == replica
        assert set(
            float(values[0, 0])
            for values in row["relation_particle_states"].values()
        ) == {float(replica)}
    dataset.set_epoch(2)
    second = [dataset[index]["replica_id"] for index in range(events)]
    assert second == [(value + 1) % 4 for value in first]


def test_t2_predictor_queries_are_projected_into_target_coordinates(
    tmp_path: Path,
) -> None:
    torch.manual_seed(41)
    queries = torch.randn(4, 128)
    projection = BridgeProjection(128, 64)
    state = {
        "expert_model.tokenizer.slot_queries": queries,
        **{
            f"projection.{name}": value
            for name, value in projection.state_dict().items()
        },
    }
    checkpoint = tmp_path / "t2.pt"
    torch.save({"offline_target_state_dict": state}, checkpoint)
    with torch.no_grad():
        expected = projection(queries).numpy()
    actual = target_slot_queries(
        checkpoint, target_mode="T2_PROJECT"
    )
    assert actual.shape == (4, 64)
    np.testing.assert_array_equal(actual, expected)


def test_miniature_predictor_trains_fixed_budget_and_reuses(
    tmp_path: Path,
) -> None:
    train, head, fusion = _training_fixture("model_train")
    validation, _, _ = _training_fixture("val_stop")
    # The cached target logits must use the exact frozen consumers passed to
    # training, so rebuild validation targets through the train fixture heads.
    with torch.no_grad():
        validation.target_expert_logits = head(
            torch.from_numpy(validation.target_tokens_original)
        ).numpy()
        banks = dict(validation.other_oracle_banks)
        banks["PT"] = validation.target_tokens_original
        validation.target_hybrid_logits = fusion(
            {name: torch.from_numpy(value) for name, value in banks.items()}
        ).numpy()
    train_loader = make_predictor_loader(
        train, batch_size=10, seed=101, training=True
    )
    val_loader = make_predictor_loader(
        validation, batch_size=10, seed=101, training=False
    )
    model = RetbTokenPredictor(
        architecture="A1_RESMLP",
        context="C0_SELF",
        target_expert_id="PT",
        token_count=2,
        token_dimension=64,
        offline_slot_queries=torch.randn(2, 64),
        uncertainty_head="U_SLOT",
        dropout=0.0,
    )
    run = with_content_hash(
        {
            "contract": "test_predictor_run_v1",
            "schema_version": 1,
            "run_id": "mini-predictor",
            "pipeline_seed": 101,
            "expert_id": "PT",
            "token_count": 2,
            "token_dimension": 64,
            "architecture": "A1_RESMLP",
            "context": "C0_SELF",
            "objective_id": "W_CANONICAL",
            "uncertainty_head": "U_SLOT",
            "normalization_mode": "N_UNCLIPPED",
            "learning_rate": 5.0e-4,
            "dropout": 0.0,
        }
    )
    config = PredictorTrainingConfig(
        seed=101,
        architecture="A1_RESMLP",
        context="C0_SELF",
        maximum_epochs=2,
        microbatch_size=10,
        effective_batch_size=10,
        dropout=0.0,
        campaign_profile="miniature_test",
    )
    lineage = {
        "model_train_target_cache": SHA_A,
        "val_stop_target_cache": SHA_B,
        "val_design_target_cache": SHA_C,
        "target_normalizer": SHA_C,
        "slot_queries": SHA_B,
        "offline_target_checkpoint": SHA_A,
        "offline_fusion": SHA_B,
        "native_hlt_expert": SHA_C,
        "model_train_hlt_evidence_cache": SHA_A,
        "val_stop_hlt_evidence_cache": SHA_B,
        "val_design_hlt_evidence_cache": SHA_C,
        "model_train_identity_manifest": SHA_C,
        "val_stop_identity_manifest": SHA_A,
        "val_design_identity_manifest": SHA_B,
    }
    class _InterruptAfterOneEpoch:
        def __init__(self, loader):
            self.loader = loader
            self.dataset = loader.dataset
            self.sampler = loader.sampler
            self.calls = 0

        def __len__(self):
            return len(self.loader)

        def __iter__(self):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated predictor interruption")
            return iter(self.loader)

    with pytest.raises(RuntimeError, match="simulated predictor interruption"):
        train_predictor(
            model=model,
            train_loader=_InterruptAfterOneEpoch(train_loader),
            val_stop_loader=val_loader,
            frozen_expert_head=head,
            frozen_fusion=fusion,
            token_mean=np.zeros((2, 64), dtype=np.float32),
            token_standard_deviation=np.ones((2, 64), dtype=np.float32),
            output_dir=tmp_path / "run",
            run_record=run,
            step9_bundle_sha256=SHA_A,
            global_determinism_sha256=SHA_B,
            lineage_hashes=lineage,
            config=config,
        )
    assert (tmp_path / "run" / "last_state.pt").is_file()
    registration = train_predictor(
        model=model,
        train_loader=train_loader,
        val_stop_loader=val_loader,
        frozen_expert_head=head,
        frozen_fusion=fusion,
        token_mean=np.zeros((2, 64), dtype=np.float32),
        token_standard_deviation=np.ones((2, 64), dtype=np.float32),
        output_dir=tmp_path / "run",
        run_record=run,
        step9_bundle_sha256=SHA_A,
        global_determinism_sha256=SHA_B,
        lineage_hashes=lineage,
        config=config,
    )
    assert registration["fixed_budget_completed"]
    assert not registration["performance_based_termination"]
    assert all(parameter.grad is None for parameter in head.parameters())
    assert all(parameter.grad is None for parameter in fusion.parameters())
    assert all(not parameter.requires_grad for parameter in head.parameters())
    assert all(not parameter.requires_grad for parameter in fusion.parameters())
    reused = train_predictor(
        model=RetbTokenPredictor(
            architecture="A1_RESMLP",
            context="C0_SELF",
            target_expert_id="PT",
            token_count=2,
            token_dimension=64,
            offline_slot_queries=torch.randn(2, 64),
            uncertainty_head="U_SLOT",
            dropout=0.0,
        ),
        train_loader=train_loader,
        val_stop_loader=val_loader,
        frozen_expert_head=head,
        frozen_fusion=fusion,
        token_mean=np.zeros((2, 64), dtype=np.float32),
        token_standard_deviation=np.ones((2, 64), dtype=np.float32),
        output_dir=tmp_path / "run",
        run_record=run,
        step9_bundle_sha256=SHA_A,
        global_determinism_sha256=SHA_B,
        lineage_hashes=lineage,
        config=config,
    )
    assert reused == registration


def test_clip_control_applies_to_targets_and_predictions_before_consumers() -> None:
    dataset, head, fusion = _training_fixture("val_stop", events=4)
    dataset.normalization_mode = "N_CLIP8"
    dataset.target_tokens_unclipped = dataset.target_tokens.copy()
    dataset.target_tokens = np.clip(
        dataset.target_tokens_unclipped, -8.0, 8.0
    )
    model = RetbTokenPredictor(
        architecture="A0_AFFINE",
        context="C0_SELF",
        target_expert_id="PT",
        token_count=2,
        token_dimension=64,
        offline_slot_queries=torch.randn(2, 64),
        uncertainty_head="U_SLOT",
        dropout=0.0,
    )
    with torch.no_grad():
        model.affine.weight.zero_()
        model.affine.bias.fill_(20.0)
    result = evaluate_predictor(
        model=model,
        loader=make_predictor_loader(
            dataset, batch_size=2, seed=101, training=False
        ),
        frozen_expert_head=head,
        frozen_fusion=fusion,
        token_mean=np.zeros((2, 64), dtype=np.float32),
        token_standard_deviation=np.ones((2, 64), dtype=np.float32),
        objective_id="W_TOKEN_ONLY",
        normalization_mode="N_CLIP8",
        device=torch.device("cpu"),
    )
    assert np.max(result["predicted_tokens"]) == 8.0
    diagnostics = result["metrics"]["normalization_tail_diagnostics"]
    assert diagnostics["prediction_clipped_element_count"] == 4 * 2 * 64
    assert (
        diagnostics["prediction_after_control_clip"]["8"]["element_count"]
        == 0
    )


def test_predictor_cache_and_calibration_reject_cross_seed(tmp_path: Path) -> None:
    rng = np.random.default_rng(9005)
    identities = [f"jet-{index}" for index in range(12)]
    tokens = rng.normal(size=(12, 2, 64)).astype(np.float32)
    variance = np.zeros((12, 2, 1), dtype=np.float32)
    logits = rng.normal(size=(12, 10)).astype(np.float32)
    manifest = publish_predictor_inference_cache(
        output_dir=tmp_path,
        split="val_design",
        pipeline_seed=101,
        expert_id="PT",
        uncertainty_head="U_SLOT",
        identities=identities,
        predicted_tokens=tokens,
        normalized_predicted_tokens=tokens / 2.0,
        log_variance=variance,
        expert_logits=logits,
        hybrid_logits=logits,
        predictor_registration_sha256=SHA_A,
        predictor_checkpoint_sha256=SHA_B,
        target_cache_manifest_sha256=SHA_C,
        target_normalizer_sha256=SHA_C,
        identity_manifest_sha256=SHA_A,
        source_snapshot=SOURCE,
    )
    loaded, arrays = load_predictor_inference_cache(
        tmp_path / "predictor_outputs_manifest.json",
        expected_pipeline_seed=101,
        expected_registration_sha256=SHA_A,
    )
    assert loaded["labels_present"] is False
    assert np.array_equal(arrays["predicted_tokens"], tokens)
    assert np.array_equal(
        arrays["normalized_predicted_tokens"], tokens / 2.0
    )
    with pytest.raises(ValueError, match="seed/registration"):
        load_predictor_inference_cache(
            tmp_path / "predictor_outputs_manifest.json",
            expected_pipeline_seed=202,
            expected_registration_sha256=SHA_A,
        )
    calibration = calibrate_predictor_inference_cache(
        manifest_path=tmp_path / "predictor_outputs_manifest.json",
        expected_pipeline_seed=101,
        expected_registration_sha256=SHA_A,
        target_tokens=tokens / 2.0 + 0.1,
        identity_order_sha256=manifest["identity_order_sha256"],
        source_snapshot=SOURCE,
    )
    validate_uncertainty_calibration(calibration)


def test_step9_bundle_is_source_bound_and_has_no_performance_gate() -> None:
    bundle = build_step9_bundle(
        campaign_spec_sha256=SHA_A,
        step8_bundle_sha256=SHA_B,
        global_determinism_sha256=SHA_C,
        source_snapshot=SOURCE,
    )
    assert validate_step9_bundle(bundle) == bundle["step9_bundle"][
        "content_hash"
    ]
    assert not bundle["step9_bundle"]["performance_based_termination"]


def test_materialized_run_binds_every_seed_teacher_and_cache_parent() -> None:
    parents = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in (
            "step9_bundle",
            "model_train_target_cache",
            "val_stop_target_cache",
            "val_design_target_cache",
            "target_normalizer",
            "slot_queries",
            "offline_target_checkpoint",
            "offline_fusion",
            "native_hlt_expert",
            "model_train_hlt_evidence_cache",
            "val_stop_hlt_evidence_cache",
            "val_design_hlt_evidence_cache",
            "model_train_identity_manifest",
            "val_stop_identity_manifest",
            "val_design_identity_manifest",
        )
    }
    run = materialize_predictor_run(
        run_id="run",
        stage="F",
        pipeline_seed=101,
        expert_id="PT",
        shape_id="S8_128",
        token_count=8,
        token_dimension=128,
        architecture="A3_SLOT_DECODER_DIRECT",
        context="C2_ALL",
        objective_id="W_CANONICAL",
        uncertainty_head="U_SLOT",
        normalization_mode="N_UNCLIPPED",
        target_mode="T0_PURE",
        hlt_evidence_mode="selected_native_HLT_evidence",
        learning_rate=5.0e-4,
        dropout=0.1,
        role="scientific_candidate",
        parent_hashes=parents,
    )
    assert validate_materialized_predictor_run(run) == run["content_hash"]
    assert run["parent_hashes"] == dict(sorted(parents.items()))
    assert not run["performance_based_termination"]
    swapped = dict(parents)
    swapped.pop("val_design_target_cache")
    with pytest.raises(ValueError, match="semantics"):
        materialize_predictor_run(
            **{
                key: value
                for key, value in {
                    "run_id": "run",
                    "stage": "F",
                    "pipeline_seed": 101,
                    "expert_id": "PT",
                    "shape_id": "S8_128",
                    "token_count": 8,
                    "token_dimension": 128,
                    "architecture": "A3_SLOT_DECODER_DIRECT",
                    "context": "C2_ALL",
                    "objective_id": "W_CANONICAL",
                    "uncertainty_head": "U_SLOT",
                    "normalization_mode": "N_UNCLIPPED",
                    "target_mode": "T0_PURE",
                    "hlt_evidence_mode": "selected_native_HLT_evidence",
                    "learning_rate": 5.0e-4,
                    "dropout": 0.1,
                    "role": "scientific_candidate",
                    "parent_hashes": swapped,
                }.items()
            }
        )


def test_step9_production_entrypoints_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = (
        "scripts/build_retb_step9_contracts.py",
        "scripts/materialize_retb_predictor_run.py",
        "scripts/train_retb_predictor.py",
        "sbatch/run_retb_build_step9_contracts.sh",
        "sbatch/run_retb_materialize_predictor_run.sh",
        "sbatch/run_retb_train_predictor.sh",
    )
    for relative in expected:
        assert (root / relative).is_file()
    training_cli = (root / "scripts/train_retb_predictor.py").read_text()
    assert "stack_val" not in training_cli
    assert "load_and_validate_campaign_source" in training_cli
