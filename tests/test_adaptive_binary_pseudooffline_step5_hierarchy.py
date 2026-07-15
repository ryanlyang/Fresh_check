from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from jetclass_fresh.jetclass_data import JetIdentity, RAW_TOKEN_DIM
from teacher_logit_reco.adaptive_binary_pseudooffline.binary_accounting import AccountingState
from teacher_logit_reco.adaptive_binary_pseudooffline.hierarchy_alignment import (
    align_frontier,
    align_recursive_hierarchy,
    build_teacher_parent_frontiers,
    compute_teacher_forced_level_supervision,
    hierarchy_targets_to_tensors,
    local_sibling_match,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.hierarchy_decoder import (
    ABPH_GROUP_SUPPORT_DIM,
    HierarchyFrontier,
    RecursiveHierarchyDecoder,
    RecursiveHierarchyDecoderConfig,
    support_from_ledger,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.root_compiler import minimum_mass_budget
from teacher_logit_reco.adaptive_binary_pseudooffline.root_transforms import ROOT_FEATURE_INDEX
from teacher_logit_reco.adaptive_binary_pseudooffline.schemas import (
    ABPH_MAX_PARTICLES,
    ABPH_PID_CATEGORIES,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.targets import (
    ABPH_LEVEL_CAPACITIES,
    ROOT_FEATURE_NAMES,
    TOPOLOGY_ACTIVE_SPLIT,
    TOPOLOGY_ACTIVE_TERMINAL,
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
    batch = 2
    hlt = np.zeros((batch, ABPH_MAX_PARTICLES, RAW_TOKEN_DIM), dtype=np.float32)
    offline = np.zeros_like(hlt)
    hlt_mask = np.zeros((batch, ABPH_MAX_PARTICLES), dtype=bool)
    offline_mask = np.zeros_like(hlt_mask)
    identities = []
    for jet in range(batch):
        count = 7 + jet
        hlt_count = count - 2
        offline_mask[jet, :count] = True
        hlt_mask[jet, :hlt_count] = True
        for particle in range(count):
            pid = particle % len(ABPH_PID_CATEGORIES)
            charge = float((-1) ** particle) if pid in (0, 3, 4) else 0.0
            row = _token(
                35.0 - particle + 2.0 * jet,
                -0.24 + 0.07 * particle,
                -1.2 + 0.29 * particle,
                pid,
                charge,
            )
            offline[jet, particle] = row
            if particle < hlt_count:
                hlt[jet, particle] = row
        identities.append(JetIdentity(file=f"class_{jet}.root", entry=jet, label=jet))
    return build_adaptive_binary_targets(
        hlt,
        hlt_mask,
        offline,
        offline_mask,
        jet_ids=tuple(identities),
        layout=AdaptiveBinaryHierarchyLayout(grouping="exclusive_kt"),
    )


def _small_model() -> RecursiveHierarchyDecoder:
    model = RecursiveHierarchyDecoder(
        RecursiveHierarchyDecoderConfig(
            hlt_input_dims=(16, 12),
            d_model=32,
            num_heads=4,
            ffn_dim=64,
            blocks_per_level=1,
            dropout=0.0,
            attention_dropout=0.0,
            root_semantic_dim=32,
            latent_dim=8,
        )
    )
    model.eval()
    return model


def _model_inputs(targets):
    torch.manual_seed(123)
    batch = targets.n_jets
    n_hlt = 10
    first = torch.randn(batch, n_hlt, 16, requires_grad=True)
    second = torch.randn(batch, n_hlt, 12, requires_grad=True)
    mask = torch.zeros(batch, n_hlt, dtype=torch.bool)
    mask[:, :6] = True
    support = torch.zeros(batch, n_hlt, 9)
    support[:, :, 0] = torch.linspace(-0.4, 0.4, n_hlt)
    support[:, :, 1] = torch.linspace(-2.5, 2.5, n_hlt)
    support[:, :, 2] = torch.linspace(1.0, 3.0, n_hlt)
    support[:, :, 3] = 1.0
    root = AccountingState.from_ledger(torch.as_tensor(targets.root_features))
    hidden = torch.randn(batch, 32)
    semantics = torch.randn(batch, 6, 32)
    return root, hidden, semantics, (first, second), mask, support


def _ledger(
    p4: tuple[float, float, float, float],
    type_counts: tuple[int, ...],
    charge: int = 0,
) -> torch.Tensor:
    row = torch.zeros(len(ROOT_FEATURE_NAMES))
    for index, name in enumerate(("energy", "px", "py", "pz")):
        row[ROOT_FEATURE_INDEX[name]] = p4[index]
    count = sum(type_counts)
    row[ROOT_FEATURE_INDEX["constituent_count"]] = count
    for index, name in enumerate(ABPH_PID_CATEGORIES):
        row[ROOT_FEATURE_INDEX[f"count_{name}"]] = type_counts[index]
    row[ROOT_FEATURE_INDEX["integer_charge"]] = charge
    row[ROOT_FEATURE_INDEX["minimum_mass_budget"]] = minimum_mass_budget(
        torch.tensor((type_counts,))
    )[0]
    row[ROOT_FEATURE_INDEX["feasible_charge_min"]] = -count
    row[ROOT_FEATURE_INDEX["feasible_charge_max"]] = count
    row[ROOT_FEATURE_INDEX["scalar_sum_pt"]] = math.hypot(p4[1], p4[2])
    return row


def _frontier(
    ledgers: torch.Tensor,
    topology: tuple[int, ...],
    parent_indices: tuple[int, ...],
) -> HierarchyFrontier:
    ledgers = ledgers[None, :, :]
    capacity = ledgers.shape[1]
    return HierarchyFrontier(
        ledger=ledgers,
        hidden=torch.zeros(1, capacity, 8),
        support=support_from_ledger(ledgers[0])[:, None, :].transpose(0, 1),
        uncertainty=torch.zeros(1, capacity),
        mask=torch.ones(1, capacity, dtype=torch.bool),
        topology=torch.tensor((topology,), dtype=torch.long),
        parent_indices=torch.tensor((parent_indices,), dtype=torch.long),
        source_child_indices=torch.zeros(1, capacity, dtype=torch.long),
    )


def test_recursive_rollout_uses_fixed_capacities_masks_padding_and_every_hlt_depth():
    targets = _target_batch()
    model = _small_model()
    root, hidden, semantics, embeddings, hlt_mask, support = _model_inputs(targets)
    output = model(root, hidden, semantics, embeddings, hlt_mask, support, mode="rollout")
    assert output.mode == "rollout"
    assert [level.next_frontier.capacity for level in output.levels] == list(
        ABPH_LEVEL_CAPACITIES
    )
    assert output.diagnostics["all_levels_accessed_original_hlt"] is True
    for level in output.levels:
        frontier = level.next_frontier
        assert torch.equal(frontier.hidden[~frontier.mask], torch.zeros_like(frontier.hidden[~frontier.mask]))
        assert not bool((frontier.parent_indices[~frontier.mask] != -1).any())
        assert torch.isfinite(level.support_attention_bias).all()
        assert not bool((level.support_attention_bias[~level.parent_frontier.mask] != 0).any())


def test_all_levels_backpropagate_to_original_multidepth_hlt_evidence():
    targets = _target_batch()
    model = _small_model()
    root, hidden, semantics, embeddings, hlt_mask, support = _model_inputs(targets)
    output = model(root, hidden, semantics, embeddings, hlt_mask, support, mode="rollout")
    objective = sum(level.parent_context.square().mean() for level in output.levels)
    objective.backward()
    for source in embeddings:
        assert source.grad is not None
        assert torch.isfinite(source.grad).all()
        assert float(source.grad.abs().sum()) > 0.0
    for level in model.levels:
        gradient = level.blocks[0].global_hlt_attention.in_proj_weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()


def test_rollout_terminal_decision_is_absorbing_across_later_levels():
    targets = _target_batch()
    model = _small_model()
    with torch.no_grad():
        for depth, level in enumerate(model.levels):
            final = level.topology_head.network[-1]
            final.weight.zero_()
            final.bias.copy_(
                torch.tensor((10.0, -10.0))
                if depth == 0
                else torch.tensor((-10.0, 10.0))
            )
    root, hidden, semantics, embeddings, hlt_mask, support = _model_inputs(targets)
    output = model(root, hidden, semantics, embeddings, hlt_mask, support, mode="rollout")
    assert all(level.next_frontier.mask.sum(dim=1).tolist() == [1, 1] for level in output.levels)
    assert all(
        bool(
            (
                level.next_frontier.topology[level.next_frontier.mask]
                == int(TOPOLOGY_ACTIVE_TERMINAL)
            ).all()
        )
        for level in output.levels
    )
    assert all(level.diagnostics["carried_terminal_reopened"] is False for level in output.levels)


def test_teacher_forcing_is_explicit_and_separately_reported():
    targets = _target_batch()
    tensors = hierarchy_targets_to_tensors(targets)
    teachers = build_teacher_parent_frontiers(tensors, d_model=32)
    model = _small_model()
    root, hidden, semantics, embeddings, hlt_mask, support = _model_inputs(targets)
    teacher = model(
        root,
        hidden,
        semantics,
        embeddings,
        hlt_mask,
        support,
        mode="teacher_forced",
        teacher_parent_frontiers=teachers,
    )
    assert teacher.diagnostics["teacher_inputs_consumed"] is True
    assert teacher.diagnostics["offline_inputs_consumed_in_rollout"] is False
    teacher_alignment = align_recursive_hierarchy(teacher, tensors)
    assert teacher_alignment.diagnostics["teacher_forced_report"] is True
    assert teacher_alignment.diagnostics["rollout_report"] is False
    rollout = model(root, hidden, semantics, embeddings, hlt_mask, support, mode="rollout")
    assert rollout.diagnostics["teacher_inputs_consumed"] is False
    with pytest.raises(ValueError, match="cannot consume teacher"):
        model(
            root,
            hidden,
            semantics,
            embeddings,
            hlt_mask,
            support,
            mode="rollout",
            teacher_parent_frontiers=teachers,
        )


def test_teacher_forced_local_losses_ignore_padding_and_match_only_siblings():
    targets = _target_batch()
    tensors = hierarchy_targets_to_tensors(targets)
    teachers = build_teacher_parent_frontiers(tensors, d_model=32)
    model = _small_model()
    root, hidden, semantics, embeddings, hlt_mask, support = _model_inputs(targets)
    output = model(
        root,
        hidden,
        semantics,
        embeddings,
        hlt_mask,
        support,
        mode="teacher_forced",
        teacher_parent_frontiers=teachers,
    )
    for depth, level in enumerate(output.levels):
        parent_topology = teachers[depth].topology
        supervision = compute_teacher_forced_level_supervision(
            level,
            tensors.level_ledgers[depth],
            tensors.level_supports[depth],
            tensors.level_masks[depth],
            tensors.level_parent_indices[depth],
            parent_topology,
        )
        assert torch.isfinite(supervision.accounting_loss.total)
        assert torch.isfinite(supervision.total_loss)
        assert supervision.diagnostics["padding_contributes_loss"] is False
        matched = supervision.sibling_match.matched_mask
        indices = supervision.sibling_match.target_indices[matched]
        if int(indices.numel()):
            cursor = 0
            for batch_index in range(matched.shape[0]):
                slots = torch.nonzero(matched[batch_index], as_tuple=False).flatten()
                count = int(slots.numel())
                predicted_parents = supervision.sibling_match.predicted_parent_indices[
                    batch_index, slots
                ]
                target_parents = tensors.level_parent_indices[depth][batch_index, indices[cursor : cursor + count]]
                assert torch.equal(predicted_parents, target_parents)
                cursor += count
        changed = tensors.level_ledgers[depth].clone()
        changed[~tensors.level_masks[depth]] = 1.0e9
        repeated = compute_teacher_forced_level_supervision(
            level,
            changed,
            tensors.level_supports[depth],
            tensors.level_masks[depth],
            tensors.level_parent_indices[depth],
            parent_topology,
        )
        torch.testing.assert_close(
            supervision.accounting_loss.total,
            repeated.accounting_loss.total,
        )


def test_parent_local_sibling_matching_cannot_cross_parent_boundaries():
    child_a = _ledger((6.0, 1.0, 0.0, 0.0), (0, 1, 0, 0, 0, 0))
    child_b = _ledger((4.0, -1.0, 0.0, 0.0), (0, 1, 0, 0, 0, 0))
    predicted = torch.stack((child_a, child_b, child_b, child_a))[None, :, :]
    target = torch.stack((child_b, child_a, child_a, child_b))[None, :, :]
    support_pred = support_from_ledger(predicted[0])[None, :, :]
    support_target = support_from_ledger(target[0])[None, :, :]
    parents = torch.tensor(((0, 0, 1, 1),))
    result = local_sibling_match(
        predicted,
        support_pred,
        torch.ones(1, 4, dtype=torch.bool),
        parents,
        target,
        support_target,
        torch.ones(1, 4, dtype=torch.bool),
        parents,
        parent_mask=torch.ones(1, 2, dtype=torch.bool),
        parent_split_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    assert result.diagnostics["parent_local_only"] is True
    for slot in range(4):
        assert int(parents[0, slot]) == int(parents[0, result.target_indices[0, slot]])


def test_false_stop_collapses_target_children_into_complete_renderer_measure():
    parent = _ledger((10.0, 0.0, 0.0, 0.0), (0, 4, 0, 0, 0, 0))
    predicted = _frontier(
        parent[None, :],
        (int(TOPOLOGY_ACTIVE_TERMINAL),),
        (0,),
    )
    targets = torch.stack(
        (
            _ledger((5.0, 3.0, 0.0, 0.0), (0, 2, 0, 0, 0, 0)),
            _ledger((5.0, -3.0, 0.0, 0.0), (0, 2, 0, 0, 0, 0)),
        )
    )[None, :, :]
    membership = torch.zeros(1, 2, ABPH_MAX_PARTICLES, dtype=torch.bool)
    membership[0, 0, :2] = True
    membership[0, 1, 2:4] = True
    particle_mask = torch.zeros(1, ABPH_MAX_PARTICLES, dtype=torch.bool)
    particle_mask[0, :4] = True
    result = align_frontier(
        predicted,
        targets,
        support_from_ledger(targets[0])[None, :, :],
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 2, dtype=torch.long),
        membership,
        particle_mask,
        mode="rollout",
    )
    assert result.diagnostics["unequal_cardinality_examples"] == 1
    assert result.renderer_target_map.diagnostics["all_predicted_terminals_have_maps"]
    assert result.renderer_target_map.diagnostics["particle_measure_max_residual"] < 1.0e-4
    assert float(result.renderer_target_map.target_particle_weights[0, 0, :4].sum()) > 3.9
    assert torch.isfinite(result.loss)


def test_false_split_uses_explicit_null_mass_without_dropping_target_particle():
    predicted_ledgers = torch.stack(
        (
            _ledger((3.0, 1.0, 0.0, 0.0), (0, 1, 0, 0, 0, 0)),
            _ledger((3.0, -1.0, 0.0, 0.0), (0, 1, 0, 0, 0, 0)),
        )
    )
    predicted = _frontier(
        predicted_ledgers,
        (int(TOPOLOGY_ACTIVE_TERMINAL), int(TOPOLOGY_ACTIVE_TERMINAL)),
        (0, 0),
    )
    target = _ledger((4.0, 0.0, 0.0, 0.0), (0, 1, 0, 0, 0, 0))[None, None, :]
    membership = torch.zeros(1, 1, ABPH_MAX_PARTICLES, dtype=torch.bool)
    membership[0, 0, 0] = True
    particle_mask = torch.zeros(1, ABPH_MAX_PARTICLES, dtype=torch.bool)
    particle_mask[0, 0] = True
    result = align_frontier(
        predicted,
        target,
        support_from_ledger(target[0])[None, :, :],
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.long),
        membership,
        particle_mask,
        mode="rollout",
    )
    assert result.diagnostics["unmatched_nodes_silently_dropped"] is False
    assert float(result.predicted_to_null.sum()) > 0.9
    assert result.renderer_target_map.diagnostics["particle_measure_max_residual"] < 1.0e-4
    assert result.renderer_target_map.diagnostics["predicted_measure_max_residual"] < 1.0e-4


def test_ancestry_constraints_block_cross_parent_transport():
    ledger = torch.stack(
        (
            _ledger((4.0, 1.0, 0.0, 0.0), (0, 1, 0, 0, 0, 0)),
            _ledger((4.0, -1.0, 0.0, 0.0), (0, 1, 0, 0, 0, 0)),
        )
    )
    predicted = _frontier(
        ledger,
        (int(TOPOLOGY_ACTIVE_TERMINAL), int(TOPOLOGY_ACTIVE_TERMINAL)),
        (0, 1),
    )
    membership = torch.zeros(1, 2, ABPH_MAX_PARTICLES, dtype=torch.bool)
    membership[0, 0, 0] = True
    membership[0, 1, 1] = True
    particle_mask = torch.zeros(1, ABPH_MAX_PARTICLES, dtype=torch.bool)
    particle_mask[0, :2] = True
    previous = torch.zeros(1, 2, 2)
    previous[0, 0, 0] = 1.0
    previous[0, 1, 1] = 1.0
    result = align_frontier(
        predicted,
        ledger[None, :, :],
        support_from_ledger(ledger)[None, :, :],
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor(((0, 1),)),
        membership,
        particle_mask,
        mode="rollout",
        previous_real_transport=previous,
    )
    assert not bool(result.allowed_ancestry[0, 0, 1])
    assert not bool(result.allowed_ancestry[0, 1, 0])
    assert float(result.real_transport[0, 0, 1]) == 0.0
    assert float(result.real_transport[0, 1, 0]) == 0.0


def test_complete_recursive_alignment_reports_rollout_separately():
    targets = _target_batch()
    tensors = hierarchy_targets_to_tensors(targets)
    model = _small_model()
    root, hidden, semantics, embeddings, hlt_mask, support = _model_inputs(targets)
    output = model(root, hidden, semantics, embeddings, hlt_mask, support, mode="rollout")
    aligned = align_recursive_hierarchy(output, tensors)
    assert aligned.mode == "rollout"
    assert aligned.diagnostics["rollout_report"] is True
    assert aligned.diagnostics["teacher_forced_report"] is False
    assert len(aligned.levels) == 5
    assert torch.isfinite(aligned.total_frontier_loss)
    assert all(
        level.diagnostics["unmatched_nodes_silently_dropped"] is False
        for level in aligned.levels
    )
