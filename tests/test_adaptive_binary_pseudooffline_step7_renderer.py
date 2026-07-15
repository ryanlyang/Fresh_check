from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from jetclass_fresh.jetclass_data import JetIdentity, RAW_TOKEN_DIM
from teacher_logit_reco.adaptive_binary_pseudooffline.binary_accounting import AccountingState
from teacher_logit_reco.adaptive_binary_pseudooffline.hierarchy_alignment import (
    RendererTargetMap,
    align_recursive_hierarchy,
    hierarchy_targets_to_tensors,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.hierarchy_decoder import (
    RecursiveHierarchyDecoder,
    RecursiveHierarchyDecoderConfig,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.particle_matching import (
    compute_particle_auxiliary_losses,
    compute_local_particle_matching_loss,
    compute_particle_observables,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.particle_renderer import (
    ConstrainedParticleRenderer,
    ParticleRendererConfig,
    allocate_particle_charges,
    allocate_particle_types,
    exact_particle_slot_layout,
    project_n_body_phase_space,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.root_transforms import ROOT_FEATURE_INDEX
from teacher_logit_reco.adaptive_binary_pseudooffline.schemas import (
    ABPH_MAX_PARTICLES,
    ABPH_PID_CATEGORIES,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.targets import (
    AdaptiveBinaryHierarchyLayout,
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
    hlt_mask[0, :6] = True
    offline_mask[0, :8] = True
    for particle in range(8):
        pid = particle % len(ABPH_PID_CATEGORIES)
        charge = float((-1) ** particle) if pid in (0, 3, 4) else 0.0
        row = _token(
            34.0 - 1.5 * particle,
            -0.25 + 0.07 * particle,
            -1.25 + 0.30 * particle,
            pid,
            charge,
        )
        offline[0, particle] = row
        if particle < 6:
            hlt[0, particle] = row
    return build_adaptive_binary_targets(
        hlt,
        hlt_mask,
        offline,
        offline_mask,
        jet_ids=(JetIdentity(file="HToBB_010.root", entry=7, label=0),),
        layout=AdaptiveBinaryHierarchyLayout(grouping="exclusive_kt"),
    )


def _decoder() -> RecursiveHierarchyDecoder:
    model = RecursiveHierarchyDecoder(
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
    model.eval()
    return model


def _renderer() -> ConstrainedParticleRenderer:
    model = ConstrainedParticleRenderer(
        ParticleRendererConfig(
            hlt_input_dims=(8, 6),
            d_model=16,
            num_heads=4,
            ffn_dim=32,
            blocks=1,
            dropout=0.0,
            attention_dropout=0.0,
            root_semantic_dim=16,
            latent_dim=8,
            type_sinkhorn_iterations=60,
            phase_space_iterations=72,
        )
    )
    model.eval()
    return model


def _rollout_inputs(targets):
    torch.manual_seed(442)
    root = AccountingState.from_ledger(torch.as_tensor(targets.root_features))
    first = torch.randn(1, 7, 8, requires_grad=True)
    second = torch.randn(1, 7, 6, requires_grad=True)
    mask = torch.tensor(((True, True, True, True, True, True, False),))
    support = torch.zeros(1, 7, 9)
    support[:, :, 0] = torch.linspace(-0.35, 0.35, 7)
    support[:, :, 1] = torch.linspace(-2.0, 2.0, 7)
    support[:, :, 2] = torch.linspace(1.0, 2.0, 7)
    support[:, :, 3] = 1.0
    hidden = torch.randn(1, 16)
    semantics = torch.randn(1, 4, 16)
    latent = torch.randn(1, 8)
    return root, hidden, semantics, (first, second), mask, support, latent


def _render_complete(targets):
    decoder = _decoder()
    renderer = _renderer()
    root, hidden, semantics, embeddings, mask, support, latent = _rollout_inputs(targets)
    hierarchy = decoder(
        root,
        hidden,
        semantics,
        embeddings,
        mask,
        support,
        mode="rollout",
        hypothesis_latent=latent,
    )
    rendered = renderer(
        hierarchy,
        semantics,
        embeddings,
        mask,
        support,
        latent,
        torch.as_tensor(targets.hlt_axis_eta),
        torch.as_tensor(targets.hlt_axis_phi),
        hypothesis_index=2,
    )
    return rendered, hierarchy, renderer, embeddings


def _identity_target_map(rendered, hierarchy) -> RendererTargetMap:
    weights = torch.zeros(
        1, hierarchy.final_frontier.capacity, ABPH_MAX_PARTICLES
    )
    for slot in torch.nonzero(rendered.mask[0], as_tuple=False).flatten().tolist():
        group = int(rendered.group_indices[0, slot])
        weights[0, group, slot] = 1.0
    return RendererTargetMap(
        target_particle_indices=torch.arange(ABPH_MAX_PARTICLES)[None, None, :].expand_as(weights),
        target_particle_weights=weights,
        target_particle_mask=weights > 0,
        predicted_null_mass=torch.zeros(1, hierarchy.final_frontier.capacity),
        target_null_particle_weight=torch.zeros(1, ABPH_MAX_PARTICLES),
        hard_particle_assignment=rendered.group_indices.clone(),
        terminal_mask=hierarchy.final_frontier.mask.clone(),
        diagnostics={"identity": True},
    )


def test_n_body_phase_space_is_differentiable_and_closes_massive_parent():
    parent = torch.tensor((20.0, 3.0, -2.0, 4.0), dtype=torch.float64)
    raw = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    masses = torch.tensor((0.14, 0.0, 0.0, 0.01, 0.1), dtype=torch.float64)
    fractions = torch.randn(5, dtype=torch.float64, requires_grad=True)
    particles, diagnostics = project_n_body_phase_space(parent, raw, masses, fractions)
    torch.testing.assert_close(particles.sum(dim=0), parent, atol=2.0e-7, rtol=2.0e-7)
    on_shell = torch.sqrt(
        (particles[:, 0].square() - particles[:, 1:].square().sum(dim=-1)).clamp_min(0.0)
    )
    torch.testing.assert_close(on_shell, masses, atol=2.0e-7, rtol=2.0e-7)
    assert diagnostics["branch"] == "massive_rest_frame"
    particles[:, 0].square().sum().backward()
    assert raw.grad is not None and float(raw.grad.abs().sum()) > 0.0


def test_massless_phase_space_branch_closes_without_a_fake_mass_floor():
    parent = torch.tensor((9.0, 9.0, 0.0, 0.0), dtype=torch.float64)
    particles, diagnostics = project_n_body_phase_space(
        parent,
        torch.zeros(3, 3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        torch.tensor((0.1, -0.3, 0.2), dtype=torch.float64),
    )
    torch.testing.assert_close(particles.sum(dim=0), parent)
    assert diagnostics["branch"] == "massless_collinear"
    with pytest.raises(ValueError, match="infeasible|massive rendered"):
        project_n_body_phase_space(
            parent,
            torch.zeros(2, 3),
            torch.tensor((0.1, 0.0)),
            torch.zeros(2),
        )


def test_exact_slots_pid_and_charge_follow_compiled_group_budgets():
    targets = _target_batch()
    decoder = _decoder()
    root, hidden, semantics, embeddings, mask, support, latent = _rollout_inputs(targets)
    hierarchy = decoder(
        root, hidden, semantics, embeddings, mask, support, mode="rollout", hypothesis_latent=latent
    )
    layout = exact_particle_slot_layout(hierarchy.final_frontier, root)
    groups = hierarchy.final_frontier.capacity
    type_counts = torch.stack(
        tuple(
            hierarchy.final_frontier.ledger[:, :, ROOT_FEATURE_INDEX[f"count_{name}"]]
            for name in ABPH_PID_CATEGORIES
        ),
        dim=-1,
    ).round().long()
    allocation = allocate_particle_types(
        torch.randn(1, ABPH_MAX_PARTICLES, len(ABPH_PID_CATEGORIES)),
        layout,
        type_counts,
        temperature=0.5,
        sinkhorn_iterations=80,
    )
    group_charge = hierarchy.final_frontier.ledger[:, :, ROOT_FEATURE_INDEX["integer_charge"]]
    charges = allocate_particle_charges(
        torch.randn(1, ABPH_MAX_PARTICLES, 3),
        layout,
        allocation.hard_indices,
        group_charge,
        temperature=0.5,
    )
    assert layout.mask.sum().item() == int(root.constituent_count.item())
    assert allocation.diagnostics["hard_type_counts_close_exactly"]
    assert allocation.diagnostics["soft_quota_max_residual"] < 2.0e-3
    assert charges.diagnostics["hard_group_charge_closes_exactly"]
    assert charges.diagnostics["hard_charges_respect_pid_support"]
    assert groups == 32


def test_complete_renderer_closes_every_local_parent_and_shared_root():
    targets = _target_batch()
    rendered, hierarchy, renderer, embeddings = _render_complete(targets)
    assert rendered.mask.sum().item() == 8
    assert rendered.diagnostics["count_closes_exactly"]
    assert rendered.diagnostics["types_close_exactly"]
    assert rendered.diagnostics["charges_close_exactly"]
    assert rendered.diagnostics["hlt_only_deployment_inputs"]
    assert rendered.diagnostics["offline_inputs_consumed"] is False
    assert rendered.canonical_features.shape[-1] == targets.particle_targets.shape[-1]
    for group in torch.unique(rendered.group_indices[0, rendered.mask[0]]):
        group_index = int(group)
        member = rendered.mask[0] & (rendered.group_indices[0] == group_index)
        parent = hierarchy.final_frontier.ledger[0, group_index, :4]
        torch.testing.assert_close(
            rendered.four_vector[0, member].sum(dim=0), parent, atol=2.0e-4, rtol=2.0e-5
        )
    root = hierarchy.root_frontier.ledger[:, 0, :4]
    torch.testing.assert_close(rendered.four_vector.sum(dim=1), root, atol=3.0e-4, rtol=2.0e-5)
    objective = (
        rendered.four_vector.square().mean()
        + rendered.track_features.square().mean()
        + rendered.soft_pid_probabilities.square().mean()
    )
    objective.backward()
    assert renderer.rest_spatial_head.weight.grad is not None
    assert float(renderer.rest_spatial_head.weight.grad.abs().sum()) > 0.0
    assert embeddings[0].grad is not None and torch.isfinite(embeddings[0].grad).all()


def test_topology_matched_local_targets_use_hungarian_without_crossing_groups():
    targets = _target_batch()
    rendered, hierarchy, _, _ = _render_complete(targets)
    target_map = _identity_target_map(rendered, hierarchy)
    target_features = rendered.canonical_features.detach().clone()
    loss = compute_local_particle_matching_loss(
        rendered,
        target_features,
        rendered.mask,
        target_map,
    )
    assert torch.isfinite(loss.total)
    assert loss.diagnostics["method_counts"]["hungarian"] > 0
    assert loss.diagnostics["method_counts"]["unbalanced_ot"] == 0
    assert loss.diagnostics["matching_crosses_group_boundaries"] is False
    for assignment in loss.assignments:
        predicted_groups = rendered.group_indices[
            assignment.batch_index, assignment.predicted_slot_indices
        ]
        target_groups = target_map.hard_particle_assignment[
            assignment.batch_index, assignment.target_particle_indices
        ]
        assert bool((predicted_groups == assignment.group_index).all())
        assert bool((target_groups == assignment.group_index).all())


def test_rollout_weighted_targets_use_ot_and_false_split_capacity_hits_null_sink():
    targets = _target_batch()
    rendered, hierarchy, _, _ = _render_complete(targets)
    identity = _identity_target_map(rendered, hierarchy)
    group = int(rendered.group_indices[0, rendered.mask[0]][0])
    slots = torch.nonzero(
        rendered.mask[0] & (rendered.group_indices[0] == group), as_tuple=False
    ).flatten()
    if int(slots.numel()) < 2:
        pytest.skip("random rollout produced no multi-particle terminal")
    weights = identity.target_particle_weights.clone()
    weights[0, group, slots[-1]] = 0.0
    predicted_null = identity.predicted_null_mass.clone()
    predicted_null[0, group] = 1.0
    soft_map = replace(
        identity,
        target_particle_weights=weights,
        target_particle_mask=weights > 0,
        predicted_null_mass=predicted_null,
        diagnostics={"false_split": True},
    )
    loss = compute_local_particle_matching_loss(
        rendered,
        rendered.canonical_features.detach(),
        rendered.mask,
        soft_map,
    )
    assert loss.diagnostics["method_counts"]["unbalanced_ot"] >= 1
    assert float(loss.null_sink_penalty.detach()) > 0.0
    assert loss.diagnostics["teacher_forced_topology_used"] is False

    tensors = hierarchy_targets_to_tensors(targets)
    alignment = align_recursive_hierarchy(hierarchy, tensors)
    rollout_loss = compute_local_particle_matching_loss(
        rendered,
        torch.as_tensor(targets.particle_targets),
        torch.as_tensor(targets.particle_mask),
        alignment.levels[-1].renderer_target_map,
    )
    assert torch.isfinite(rollout_loss.total)
    assert rollout_loss.diagnostics["teacher_forced_topology_used"] is False


def test_particle_observables_recompute_hard_and_shape_diagnostics():
    targets = _target_batch()
    rendered, _, _, _ = _render_complete(targets)
    observables = compute_particle_observables(rendered)
    assert observables["jet_four_vector"].shape == (1, 4)
    assert observables["radial_pt_profile"].shape == (1, 6)
    assert observables["leading_pt_fractions"].shape == (1, 4)
    assert torch.isfinite(observables["energy_correlation_2"]).all()
    torch.testing.assert_close(
        observables["radial_pt_profile"].sum(dim=-1), torch.ones(1), atol=1.0e-5, rtol=1.0e-5
    )
    auxiliary = compute_particle_auxiliary_losses(
        rendered,
        rendered.canonical_features.detach(),
        rendered.mask,
    )
    assert torch.isfinite(auxiliary.total)
    assert set(auxiliary.components) == {
        "jet_four_vector",
        "radial_profile",
        "leading_pt",
        "energy_correlation",
        "n_subjettiness",
        "track_summary",
        "particle_energy_distance",
    }
    assert auxiliary.diagnostics["hard_quantities_audited_not_soft_repaired"]
