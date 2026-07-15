from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest
import torch

from jetclass_fresh.jetclass_data import JetIdentity, RAW_TOKEN_DIM
from teacher_logit_reco.adaptive_binary_pseudooffline import (
    ABPH_CHARGE_SUPPORT_MIN,
    ABPH_PID_CATEGORIES,
    ROOT_FEATURE_INDEX,
    ROOT_RESIDUAL_CHANNEL_NAMES,
    ROOT_SHAPE_FEATURE_NAMES,
    AdaptiveBinaryHierarchyLayout,
    RootLossWeights,
    SemanticRootPredictor,
    SemanticRootPredictorConfig,
    allocate_integer_type_counts,
    build_adaptive_binary_targets,
    build_root_residual_targets,
    compile_root_state,
    compute_root_losses,
    compute_root_metrics,
    feasible_charge_mask,
    fit_root_normalization_stats,
    minimum_mass_budget,
    root_acceptance_report,
    root_compiler_manifest,
    root_head_gradient_norms,
    root_residual_to_physical,
    summarize_hlt_root,
    wrap_phi,
    wrap_phi_tensor,
)


_MASS = (0.13957, 0.0, 0.0, 0.000511, 0.105658, 0.0)


def _token(
    pt: float,
    eta: float,
    phi: float,
    *,
    pid: int,
    charge: float,
) -> np.ndarray:
    result = np.zeros(RAW_TOKEN_DIM, dtype=np.float32)
    momentum = pt * np.cosh(eta)
    energy = np.sqrt(momentum * momentum + _MASS[pid] ** 2)
    result[:5] = (pt, eta, phi, energy, charge)
    if pid < 5:
        result[5 + pid] = 1.0
    result[10:] = (0.01, 0.002, -0.03, 0.004)
    return result


def _toy_root_batch(
    n_jets: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[JetIdentity, ...]]:
    hlt = np.zeros((n_jets, 128, RAW_TOKEN_DIM), dtype=np.float32)
    offline = np.zeros_like(hlt)
    hlt_mask = np.zeros((n_jets, 128), dtype=bool)
    offline_mask = np.zeros_like(hlt_mask)
    identities: list[JetIdentity] = []
    for jet_index in range(n_jets):
        offline_count = 7 + jet_index
        hlt_count = 4 + jet_index
        hlt_mask[jet_index, :hlt_count] = True
        offline_mask[jet_index, :offline_count] = True
        for particle_index in range(offline_count):
            pid = particle_index % len(ABPH_PID_CATEGORIES)
            if pid in (0, 3, 4):
                charge = 1.0 if particle_index % 2 == 0 else -1.0
            elif pid == 5:
                charge = float((particle_index % 3) - 1)
            else:
                charge = 0.0
            token = _token(
                42.0 - 1.7 * particle_index + 0.8 * jet_index,
                -0.34 + 0.065 * particle_index + 0.01 * jet_index,
                float(wrap_phi(3.08 + 0.057 * particle_index + 0.02 * jet_index)),
                pid=pid,
                charge=charge,
            )
            offline[jet_index, particle_index] = token
            if particle_index < hlt_count:
                degraded = token.copy()
                degraded[0] *= 0.96
                degraded[3] *= 0.96
                degraded[1] += 0.003
                degraded[2] = wrap_phi(float(degraded[2]) - 0.004)
                hlt[jet_index, particle_index] = degraded
        identities.append(
            JetIdentity(file=f"HToBB_root_{jet_index:03d}.root", entry=100 + jet_index, label=1)
        )
    target_batch = build_adaptive_binary_targets(
        hlt,
        hlt_mask,
        offline,
        offline_mask,
        jet_ids=tuple(identities),
        layout=AdaptiveBinaryHierarchyLayout(grouping="exclusive_kt"),
    )
    return (
        torch.from_numpy(hlt),
        torch.from_numpy(hlt_mask),
        torch.from_numpy(target_batch.root_features),
        tuple(identities),
    )


def _small_predictor(input_dim: int = 16) -> SemanticRootPredictor:
    return SemanticRootPredictor(
        SemanticRootPredictorConfig(
            input_dim=input_dim,
            jet_input_dim=input_dim,
            d_model=32,
            num_heads=4,
            query_blocks=3,
            ffn_dim=64,
            dropout=0.0,
            attention_dropout=0.0,
        )
    )


def _prediction_for_hlt(
    hlt_tokens: torch.Tensor,
    hlt_mask: torch.Tensor,
    *,
    input_dim: int = 16,
) -> tuple[SemanticRootPredictor, object]:
    torch.manual_seed(19)
    model = _small_predictor(input_dim)
    embeddings = torch.randn(hlt_tokens.shape[0], 128, input_dim)
    embeddings = embeddings * hlt_mask.unsqueeze(-1)
    jet_embedding = (
        embeddings.sum(dim=1)
        / hlt_mask.sum(dim=1, keepdim=True).clamp_min(1)
    )
    return model, model(embeddings, hlt_mask, jet_embedding)


def test_root_physical_transform_exactly_inverts_target_preprocessing():
    hlt_tokens, hlt_mask, root_ledger, _ = _toy_root_batch()
    targets = build_root_residual_targets(hlt_tokens, hlt_mask, root_ledger)
    hlt = summarize_hlt_root(hlt_tokens, hlt_mask)
    reconstructed = root_residual_to_physical(
        hlt.kinematics, targets.p4_residuals, targets.minimum_mass_budget
    )
    torch.testing.assert_close(reconstructed.pt, targets.physical.pt, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(reconstructed.eta, targets.physical.eta, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        wrap_phi_tensor(reconstructed.phi - targets.physical.phi),
        torch.zeros_like(reconstructed.phi),
        rtol=0.0,
        atol=2e-5,
    )
    torch.testing.assert_close(reconstructed.mass, targets.physical.mass, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(
        reconstructed.four_vector(), targets.physical.four_vector(), rtol=2e-4, atol=2e-4
    )


def test_semantic_predictor_has_six_typed_queries_and_no_offline_input():
    hlt_tokens, hlt_mask, _, _ = _toy_root_batch(2)
    model, prediction = _prediction_for_hlt(hlt_tokens, hlt_mask)
    assert prediction.query_tokens.shape == (2, 6, 32)
    assert prediction.count_logits.shape == (2, 128)
    assert prediction.type_count_logits.shape == (2, 6)
    assert prediction.charge_logits.shape == (2, 257)
    assert prediction.diagnostics["offline_inputs_loaded"] is False
    assert prediction.diagnostics["teacher_logits_loaded"] is False
    assert "offline" not in inspect.signature(model.forward).parameters
    assert SemanticRootPredictorConfig().to_dict()["input_semantics"]["offline_inputs"] is False


def test_exact_type_allocator_closes_for_random_counts_and_simplexes():
    torch.manual_seed(8)
    probabilities = torch.rand(256, len(ABPH_PID_CATEGORIES)).softmax(dim=-1)
    counts = torch.randint(1, 129, (256,))
    hard, relaxed = allocate_integer_type_counts(probabilities, counts)
    assert torch.equal(hard.sum(dim=-1), counts)
    assert bool((hard >= 0).all())
    torch.testing.assert_close(relaxed.detach(), hard.float())


def test_charge_feasibility_respects_particle_type_parity_and_other_support():
    type_counts = torch.tensor(
        (
            (1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1),
            (0, 2, 0, 0, 0, 0),
        )
    )
    mask, lower, upper = feasible_charge_mask(type_counts)
    zero = -ABPH_CHARGE_SUPPORT_MIN
    assert mask[0, zero - 1] and mask[0, zero + 1] and not mask[0, zero]
    assert bool(mask[1, zero - 1 : zero + 2].all())
    assert mask[2, zero] and int(mask[2].sum()) == 1
    assert lower.tolist() == [-1, -1, 0]
    assert upper.tolist() == [1, 1, 0]


def test_oracle_root_prediction_compiles_exact_target_p4_count_type_and_charge():
    hlt_tokens, hlt_mask, root_ledger, _ = _toy_root_batch()
    targets = build_root_residual_targets(hlt_tokens, hlt_mask, root_ledger)
    _, random_prediction = _prediction_for_hlt(hlt_tokens, hlt_mask)
    count_logits = torch.full_like(random_prediction.count_logits, -30.0)
    count_logits.scatter_(1, targets.count_index[:, None], 30.0)
    type_logits = targets.type_count_fractions.clamp_min(1.0e-6).log()
    charge_logits = torch.full_like(random_prediction.charge_logits, -30.0)
    charge_logits.scatter_(
        1, (targets.integer_charge - ABPH_CHARGE_SUPPORT_MIN)[:, None], 30.0
    )
    prediction = replace(
        random_prediction,
        p4_residual_mean=targets.p4_residuals,
        count_logits=count_logits,
        type_count_logits=type_logits,
        type_pt_logits=targets.type_pt_fractions.clamp_min(1.0e-6).log(),
        type_energy_logits=targets.type_energy_fractions.clamp_min(1.0e-6).log(),
        charge_logits=charge_logits,
    )
    compiled = compile_root_state(
        prediction,
        hlt_tokens,
        hlt_mask,
        count_override=targets.count_index + 1,
        type_count_override=targets.type_counts,
        charge_override=targets.integer_charge,
    )
    torch.testing.assert_close(
        compiled.kinematics.four_vector(), targets.physical.four_vector(), rtol=3e-4, atol=3e-4
    )
    assert torch.equal(compiled.constituent_count, targets.count_index + 1)
    assert torch.equal(compiled.type_counts, targets.type_counts)
    assert torch.equal(compiled.integer_charge, targets.integer_charge)
    assert torch.equal(compiled.type_counts.sum(dim=-1), compiled.constituent_count)
    torch.testing.assert_close(
        compiled.root_ledger[:, :4], compiled.kinematics.four_vector(), rtol=1e-6, atol=1e-6
    )
    assert root_acceptance_report(prediction, compiled)["ok"]


def test_model_train_normalization_round_trips_and_compiler_denormalizes_heads():
    hlt_tokens, hlt_mask, root_ledger, _ = _toy_root_batch()
    targets = build_root_residual_targets(hlt_tokens, hlt_mask, root_ledger)
    stats = fit_root_normalization_stats(targets)
    restored = type(stats).from_dict(stats.to_dict())
    assert restored.normalization_hash == stats.normalization_hash
    matrix = targets.normalization_matrix()
    torch.testing.assert_close(stats.denormalize(stats.normalize(matrix)), matrix)
    subset = targets.p4_residuals
    torch.testing.assert_close(
        stats.denormalize_named(
            stats.normalize_named(subset, ROOT_RESIDUAL_CHANNEL_NAMES[:4]),
            ROOT_RESIDUAL_CHANNEL_NAMES[:4],
        ),
        subset,
    )
    _, random_prediction = _prediction_for_hlt(hlt_tokens, hlt_mask)
    prediction = replace(
        random_prediction,
        p4_residual_mean=stats.normalize_named(
            targets.p4_residuals, ROOT_RESIDUAL_CHANNEL_NAMES[:4]
        ),
    )
    compiled = compile_root_state(
        prediction,
        hlt_tokens,
        hlt_mask,
        count_override=targets.count_index + 1,
        type_count_override=targets.type_counts,
        charge_override=targets.integer_charge,
        normalization=stats,
    )
    torch.testing.assert_close(
        compiled.kinematics.four_vector(), targets.physical.four_vector(), rtol=3e-4, atol=3e-4
    )
    assert compiled.diagnostics["normalization_hash"] == stats.normalization_hash


def test_root_losses_are_finite_and_every_typed_head_receives_gradient():
    hlt_tokens, hlt_mask, root_ledger, _ = _toy_root_batch()
    targets = build_root_residual_targets(hlt_tokens, hlt_mask, root_ledger)
    model, prediction = _prediction_for_hlt(hlt_tokens, hlt_mask)
    compiled = compile_root_state(prediction, hlt_tokens, hlt_mask)
    losses = compute_root_losses(prediction, compiled, targets)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    gradient_norms = root_head_gradient_norms(model)
    assert set(gradient_norms) == {
        "semantic_queries", "p4", "count", "composition", "shape", "charge", "uncertainty"
    }
    assert all(value > 0.0 and np.isfinite(value) for value in gradient_norms.values())
    assert all(value >= 1.0e-4 for value in RootLossWeights().__dict__.values())


def test_root_metrics_report_physical_units_and_exact_compiler_closure():
    hlt_tokens, hlt_mask, root_ledger, _ = _toy_root_batch()
    targets = build_root_residual_targets(hlt_tokens, hlt_mask, root_ledger)
    _, prediction = _prediction_for_hlt(hlt_tokens, hlt_mask)
    compiled = compile_root_state(prediction, hlt_tokens, hlt_mask)
    losses = compute_root_losses(prediction, compiled, targets)
    metrics = compute_root_metrics(prediction, compiled, targets, losses=losses)
    assert metrics["n_jets"] == 4
    assert np.isfinite(metrics["physical"]["pt_mae_gev"])
    assert np.isfinite(metrics["physical"]["mass_mae_gev"])
    assert len(metrics["count"]["confusion_matrix"]) == 128
    assert metrics["compiler"]["type_count_closure_max"] == 0
    assert metrics["compiler"]["all_fractions_valid"] is True
    assert metrics["compiler"]["compiler_hash"] == root_compiler_manifest()["compiler_hash"]


def test_compiler_rejects_nonclosing_type_override():
    hlt_tokens, hlt_mask, _, _ = _toy_root_batch(2)
    _, prediction = _prediction_for_hlt(hlt_tokens, hlt_mask)
    bad_types = torch.zeros((2, 6), dtype=torch.long)
    with pytest.raises(ValueError, match="sum exactly"):
        compile_root_state(
            prediction,
            hlt_tokens,
            hlt_mask,
            count_override=torch.tensor((5, 6)),
            type_count_override=bad_types,
        )


def test_minimum_mass_table_matches_compiled_root_ledger():
    hlt_tokens, hlt_mask, _, _ = _toy_root_batch(2)
    _, prediction = _prediction_for_hlt(hlt_tokens, hlt_mask)
    compiled = compile_root_state(prediction, hlt_tokens, hlt_mask)
    expected = minimum_mass_budget(compiled.type_counts)
    torch.testing.assert_close(compiled.minimum_mass_budget, expected)
    torch.testing.assert_close(
        compiled.root_ledger[:, ROOT_FEATURE_INDEX["minimum_mass_budget"]], expected
    )
    for fractions in (
        compiled.type_count_fractions,
        compiled.type_pt_fractions,
        compiled.type_energy_fractions,
    ):
        assert bool((fractions >= 0.0).all())
        torch.testing.assert_close(fractions.sum(dim=-1), torch.ones(fractions.shape[0]))


def test_primary_semantic_dimensions_are_locked_but_tests_may_use_narrow_inputs():
    primary = SemanticRootPredictorConfig()
    assert (primary.input_dim, primary.d_model, primary.num_heads, primary.query_blocks) == (
        192, 256, 8, 4
    )
    assert primary.ffn_dim == 1024
    with pytest.raises(ValueError, match="at least three"):
        SemanticRootPredictorConfig(query_blocks=2)
