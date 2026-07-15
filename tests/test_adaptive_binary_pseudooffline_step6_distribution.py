from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from jetclass_fresh.jetclass_data import JetIdentity, RAW_TOKEN_DIM
from teacher_logit_reco.adaptive_binary_pseudooffline.binary_accounting import AccountingState
from teacher_logit_reco.adaptive_binary_pseudooffline.conditional_latent import (
    ConditionalLatentConfig,
    ConditionalSplinePrior,
    VariationalLatentSample,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.hierarchy_alignment import (
    hierarchy_targets_to_tensors,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.hierarchy_decoder import (
    RecursiveHierarchyDecoder,
    RecursiveHierarchyDecoderConfig,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.hypothesis_distribution import (
    ConditionalHierarchyHypothesisModel,
    DistributionLossWeights,
    MultiHypothesisHierarchyReconstructor,
    conditional_distribution_weight,
    compute_distribution_losses,
    deployment_hypothesis_indices,
    model_val_oracle_best_hypothesis,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.schemas import (
    ABPH_MAX_PARTICLES,
    ABPH_PID_CATEGORIES,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.targets import (
    AdaptiveBinaryHierarchyLayout,
    ROOT_FEATURE_NAMES,
    build_adaptive_binary_targets,
)


_MASS = (0.13957039, 0.0, 0.0, 0.00051099895, 0.1056583755, 0.0)


def _token(pt: float, eta: float, phi: float, pid: int, charge: float) -> np.ndarray:
    row = np.zeros(RAW_TOKEN_DIM, dtype=np.float32)
    momentum = pt * np.cosh(eta)
    row[:5] = (pt, eta, phi, np.sqrt(momentum * momentum + _MASS[pid] ** 2), charge)
    if pid < 5:
        row[5 + pid] = 1.0
    row[10:] = (0.01, 0.002, -0.02, 0.003)
    return row


def _target_batch():
    hlt = np.zeros((1, ABPH_MAX_PARTICLES, RAW_TOKEN_DIM), dtype=np.float32)
    offline = np.zeros_like(hlt)
    hlt_mask = np.zeros((1, ABPH_MAX_PARTICLES), dtype=bool)
    offline_mask = np.zeros_like(hlt_mask)
    hlt_mask[0, :5] = True
    offline_mask[0, :7] = True
    for particle in range(7):
        pid = particle % len(ABPH_PID_CATEGORIES)
        charge = float((-1) ** particle) if pid in (0, 3, 4) else 0.0
        row = _token(
            30.0 - particle,
            -0.21 + 0.06 * particle,
            -1.1 + 0.31 * particle,
            pid,
            charge,
        )
        offline[0, particle] = row
        if particle < 5:
            hlt[0, particle] = row
    return build_adaptive_binary_targets(
        hlt,
        hlt_mask,
        offline,
        offline_mask,
        jet_ids=(JetIdentity(file="HToBB_010.root", entry=3, label=0),),
        layout=AdaptiveBinaryHierarchyLayout(grouping="exclusive_kt"),
    )


def _latent_config() -> ConditionalLatentConfig:
    return ConditionalLatentConfig(
        hlt_evidence_dim=8,
        hlt_jet_dim=16,
        root_semantic_dim=16,
        context_dim=16,
        latent_dim=8,
        num_context_queries=2,
        context_heads=4,
        context_blocks=1,
        context_ffn_dim=32,
        posterior_blocks=1,
        mean_quadrature_samples=8,
        spline_layers=2,
        spline_bins=4,
        spline_hidden_dim=32,
        dropout=0.0,
    )


def _decoder() -> RecursiveHierarchyDecoder:
    return RecursiveHierarchyDecoder(
        RecursiveHierarchyDecoderConfig(
            hlt_input_dims=(8, 6),
            d_model=16,
            num_heads=4,
            ffn_dim=32,
            blocks_per_level=1,
            dropout=0.0,
            attention_dropout=0.0,
            root_semantic_dim=16,
            latent_dim=8,
        )
    )


def _deployment_inputs(targets):
    torch.manual_seed(911)
    root = AccountingState.from_ledger(torch.as_tensor(targets.root_features))
    first = torch.randn(1, 6, 8)
    second = torch.randn(1, 6, 6)
    mask = torch.tensor(((True, True, True, True, True, False),))
    support = torch.zeros(1, 6, 9)
    support[:, :, 0] = torch.linspace(-0.3, 0.3, 6)
    support[:, :, 1] = torch.linspace(-2.0, 2.0, 6)
    support[:, :, 2] = torch.linspace(1.0, 2.0, 6)
    support[:, :, 3] = 1.0
    hidden = torch.randn(1, 16)
    semantics = torch.randn(1, 4, 16)
    return root, hidden, semantics, (first, second), mask, support


def _deployment_context_inputs():
    torch.manual_seed(128)
    return (
        torch.randn(2, 5, 8),
        torch.tensor(((True, True, True, False, False), (True, True, True, True, False))),
        torch.randn(2, 16),
        torch.randn(2, 4, 16),
        torch.randn(2, len(ROOT_FEATURE_NAMES)),
    )


def test_conditional_spline_is_invertible_and_density_jacobians_cancel():
    torch.manual_seed(42)
    config = _latent_config()
    prior = ConditionalSplinePrior(config)
    base = torch.randn(7, config.latent_dim).clamp(-3.0, 3.0)
    context = torch.randn(7, config.context_dim)
    latent, forward_logdet = prior.transform(base, context)
    recovered, inverse_logdet = prior.inverse(latent, context)
    torch.testing.assert_close(recovered, base, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(
        forward_logdet + inverse_logdet,
        torch.zeros_like(forward_logdet),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    assert torch.isfinite(prior.log_prob(latent, context)).all()


def test_fixed_seed_reproduces_mean_and_four_distinct_reported_hypotheses():
    model = ConditionalHierarchyHypothesisModel(_latent_config()).eval()
    context = model.encode_deployment_context(*_deployment_context_inputs())
    first = model.deployment_hypotheses(context, evaluation_seed=701)
    repeated = model.deployment_hypotheses(context, evaluation_seed=701)
    changed = model.deployment_hypotheses(context, evaluation_seed=702)
    assert first.values.shape == (2, 5, 8)
    assert torch.equal(first.values, repeated.values)
    assert torch.equal(first.values[:, 0], changed.values[:, 0])
    assert not torch.equal(first.values[:, 1:], changed.values[:, 1:])
    assert first.identities[0].kind == "deterministic_mean"
    assert all(item.kind == "stochastic" for item in first.identities[1:])
    assert len({item.report_identity for item in first.identities}) == 5
    assert first.diagnostics["offline_target_consumed"] is False


def test_training_posterior_uses_offline_hierarchy_but_deployment_api_does_not():
    model = ConditionalHierarchyHypothesisModel(_latent_config())
    context = model.encode_deployment_context(*_deployment_context_inputs())
    targets = hierarchy_targets_to_tensors(_target_batch())
    sample = model.training_posterior_sample(context[:1], targets, seed=91)
    assert sample.latent.shape == (1, 8)
    assert sample.diagnostics["posterior_used_offline_target"] is True
    assert sample.diagnostics["posterior_deployable"] is False
    assert torch.isfinite(sample.monte_carlo_kl).all()
    (sample.monte_carlo_kl.mean() + sample.latent.square().mean()).backward()
    posterior_gradient = model.training_only_posterior.head[-1].weight.grad
    prior_gradient = model.prior.base_head[-1].weight.grad
    assert posterior_gradient is not None and float(posterior_gradient.abs().sum()) > 0.0
    assert prior_gradient is not None and float(prior_gradient.abs().sum()) > 0.0
    signature = inspect.signature(model.deployment_hypotheses)
    assert all("target" not in name for name in signature.parameters)


def test_all_hypotheses_and_hierarchy_definitions_share_one_exact_hard_root():
    targets = _target_batch()
    latent_model = ConditionalHierarchyHypothesisModel(_latent_config())
    model = MultiHypothesisHierarchyReconstructor(
        {"exclusive_kt": _decoder(), "cambridge_aachen": _decoder()},
        latent_model,
    ).eval()
    output = model(*_deployment_inputs(targets), evaluation_seed=312)
    assert len(output.hypotheses) == 5
    assert output.diagnostics[
        "exact_root_identity_across_all_hypotheses_and_hierarchies"
    ]
    assert output.diagnostics["root_hard_maximum_variance"] == 0.0
    assert output.diagnostics["root_sampled"] is False
    assert output.diagnostics["root_compiler_calls_inside_hypothesis_model"] == 0
    for hypothesis in output.hypotheses:
        assert set(hypothesis.hierarchy_outputs) == {"exclusive_kt", "cambridge_aachen"}
        for hierarchy in hypothesis.hierarchy_outputs.values():
            assert torch.equal(hierarchy.root_frontier.ledger[:, 0], output.shared_root_ledger)
            assert hierarchy.diagnostics["coherent_latent_injected_at_every_level"]
            assert hierarchy.diagnostics["node_local_noise_used"] is False
            assert all(
                level.diagnostics["coherent_jet_latent_injected_directly"]
                for level in hierarchy.levels
            )


def test_offline_target_cannot_select_deployment_hypotheses():
    model = ConditionalHierarchyHypothesisModel(_latent_config()).eval()
    context = model.encode_deployment_context(*_deployment_context_inputs())
    hypotheses = model.deployment_hypotheses(context)
    assert deployment_hypothesis_indices(hypotheses) == (0, 1, 2, 3, 4)
    with pytest.raises(ValueError, match="may not select"):
        deployment_hypothesis_indices(hypotheses, offline_target_scores=torch.zeros(2, 5))
    with pytest.raises(ValueError, match="model_val analysis only"):
        model_val_oracle_best_hypothesis(
            torch.zeros(2, 5), split="final_test", analysis_only=True
        )
    diagnostic = model_val_oracle_best_hypothesis(
        torch.rand(2, 5), split="model_val", analysis_only=True
    )
    assert diagnostic["deployable"] is False
    assert diagnostic["analysis_only"] is True


def test_distribution_objective_has_proper_scores_calibration_and_bounded_collapse():
    torch.manual_seed(19)
    posterior = VariationalLatentSample(
        latent=torch.randn(3, 8),
        posterior_log_prob=torch.ones(3),
        prior_log_prob=torch.zeros(3),
        monte_carlo_kl=torch.ones(3),
        sampling_seed=19,
        diagnostics={},
    )
    target = torch.randn(3, 6)
    diverse = (target[:, None, :] + 0.3 * torch.randn(3, 4, 6)).requires_grad_()
    collapsed = target[:, None, :].expand(-1, 4, -1).clone().requires_grad_()
    weights = DistributionLossWeights(minimum_observable_pair_distance=0.10)
    diverse_loss = compute_distribution_losses(
        posterior,
        diverse,
        target,
        split_negative_log_likelihood=torch.tensor((0.7, 0.8, 0.9)),
        particle_negative_log_likelihood=torch.tensor((1.1, 1.0, 0.9)),
        weights=weights,
    )
    collapsed_loss = compute_distribution_losses(
        posterior,
        collapsed,
        target,
        split_negative_log_likelihood=0.8,
        particle_negative_log_likelihood=1.0,
        weights=weights,
    )
    assert torch.isfinite(diverse_loss.total)
    assert float(collapsed_loss.anti_collapse_penalty.detach()) > float(
        diverse_loss.anti_collapse_penalty.detach()
    )
    assert "coverage_50" in diverse_loss.metrics
    assert "coverage_90" in diverse_loss.metrics
    assert diverse_loss.metrics["anti_collapse_scope"].startswith("caller_supplied")
    diverse_loss.total.backward()
    assert diverse.grad is not None and torch.isfinite(diverse.grad).all()


def test_locked_distribution_weight_warmup_is_explicit():
    assert conditional_distribution_weight(0.0) == pytest.approx(0.05)
    assert conditional_distribution_weight(0.05) == pytest.approx(0.15)
    assert conditional_distribution_weight(0.10) == pytest.approx(0.25)
    assert conditional_distribution_weight(1.0) == pytest.approx(0.25)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        conditional_distribution_weight(1.1)
