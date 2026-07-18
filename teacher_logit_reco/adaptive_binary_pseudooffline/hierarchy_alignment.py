"""Local sibling supervision and ancestry-constrained rollout alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch

from .binary_accounting import AccountingState, BinarySplitPrediction, compile_binary_split
from .binary_objectives import (
    BinaryAccountingLossOutput,
    BinaryAccountingLossWeights,
    compute_binary_accounting_losses,
)
from .hierarchy_decoder import (
    ABPH_GROUP_SUPPORT_DIM,
    HierarchyFrontier,
    HierarchyLevelOutput,
    RecursiveHierarchyOutput,
    support_from_ledger,
)
from .root_transforms import ROOT_FEATURE_INDEX, ROOT_SHAPE_FEATURE_NAMES, wrap_phi_tensor
from .schemas import ABPH_MAX_PARTICLES, ABPH_PID_CATEGORIES
from .targets import (
    GROUP_FEATURE_NAMES,
    ROOT_FEATURE_NAMES,
    TOPOLOGY_ACTIVE_SPLIT,
    TOPOLOGY_ACTIVE_TERMINAL,
    AdaptiveBinaryTargetBatch,
)


ABPH_HIERARCHY_ALIGNMENT_CONTRACT = "adaptive_binary_pseudooffline_hierarchy_alignment_v1"
ABPH_FRONTIER_SINKHORN_EPSILON = 0.20
ABPH_FRONTIER_SINKHORN_ITERATIONS = 128
ABPH_FRONTIER_NULL_COST = 4.0
_GROUP_INDEX = {name: index for index, name in enumerate(GROUP_FEATURE_NAMES)}


@dataclass(frozen=True)
class HierarchyTargetTensors:
    root_ledger: Any
    level_ledgers: tuple[Any, ...]
    level_supports: tuple[Any, ...]
    level_masks: tuple[Any, ...]
    level_topology: tuple[Any, ...]
    level_parent_indices: tuple[Any, ...]
    level_membership: tuple[Any, ...]
    particle_mask: Any

    @property
    def batch_size(self) -> int:
        return int(self.root_ledger.shape[0])


@dataclass(frozen=True)
class SiblingMatchResult:
    target_indices: Any
    matched_mask: Any
    predicted_parent_indices: Any
    permutation: Any
    cost: Any
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class RendererTargetMap:
    target_particle_indices: Any
    target_particle_weights: Any
    target_particle_mask: Any
    predicted_null_mass: Any
    target_null_particle_weight: Any
    hard_particle_assignment: Any
    terminal_mask: Any
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class FrontierAlignmentResult:
    mode: str
    real_transport: Any
    predicted_to_null: Any
    null_to_target: Any
    allowed_ancestry: Any
    pairwise_cost: Any
    loss: Any
    renderer_target_map: RendererTargetMap
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class HierarchyLevelSupervision:
    sibling_match: SiblingMatchResult
    accounting_loss: BinaryAccountingLossOutput
    total_loss: Any
    supervised_compiled: Any
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class HierarchyAlignmentOutput:
    mode: str
    levels: tuple[FrontierAlignmentResult, ...]
    total_frontier_loss: Any
    diagnostics: Mapping[str, Any]


def hierarchy_targets_to_tensors(
    targets: AdaptiveBinaryTargetBatch,
    *,
    device: Any | None = None,
) -> HierarchyTargetTensors:
    torch = require_torch()
    ledgers: list[Any] = []
    supports: list[Any] = []
    masks: list[Any] = []
    topology: list[Any] = []
    parents: list[Any] = []
    membership: list[Any] = []
    for depth in range(len(targets.level_features)):
        features = torch.as_tensor(targets.level_features[depth], device=device).float()
        ledgers.append(features[:, :, : len(ROOT_FEATURE_NAMES)])
        supports.append(group_support_from_target_features(features))
        masks.append(torch.as_tensor(targets.level_masks[depth], device=device).bool())
        topology.append(torch.as_tensor(targets.level_topology[depth], device=device).to(torch.long))
        parents.append(
            torch.as_tensor(targets.level_parent_indices[depth], device=device).to(torch.long)
        )
        membership.append(
            torch.as_tensor(targets.level_membership[depth], device=device).bool()
        )
    return HierarchyTargetTensors(
        root_ledger=torch.as_tensor(targets.root_features, device=device).float(),
        level_ledgers=tuple(ledgers),
        level_supports=tuple(supports),
        level_masks=tuple(masks),
        level_topology=tuple(topology),
        level_parent_indices=tuple(parents),
        level_membership=tuple(membership),
        particle_mask=torch.as_tensor(targets.particle_mask, device=device).bool(),
    )


def group_support_from_target_features(features: Any) -> Any:
    torch = require_torch()
    values = torch.as_tensor(features)
    required = (
        "centroid_eta",
        "centroid_phi",
        *(f"support_covariance_cholesky[{index}]" for index in range(3)),
        *(f"support_radial_quantiles[{index}]" for index in range(3)),
        "maximum_member_radius",
        "principal_axis_sin",
        "principal_axis_cos",
    )
    if values.ndim != 3 or values.shape[-1] != len(GROUP_FEATURE_NAMES):
        raise ValueError("target group features have the wrong shape")
    result = torch.stack(tuple(values[:, :, _GROUP_INDEX[name]] for name in required), dim=-1)
    result[:, :, 1] = wrap_phi_tensor(result[:, :, 1])
    return result


def build_teacher_parent_frontiers(
    targets: HierarchyTargetTensors,
    *,
    d_model: int,
) -> tuple[HierarchyFrontier, ...]:
    """Create the root plus L1-L4 parent frontiers used by teacher forcing."""

    torch = require_torch()
    if int(d_model) <= 0:
        raise ValueError("d_model must be positive")
    device = targets.root_ledger.device
    batch = targets.batch_size
    root_count = targets.root_ledger[:, ROOT_FEATURE_INDEX["constituent_count"]].round()
    root_topology = torch.where(
        root_count > 1,
        torch.full((batch,), int(TOPOLOGY_ACTIVE_SPLIT), dtype=torch.long, device=device),
        torch.full((batch,), int(TOPOLOGY_ACTIVE_TERMINAL), dtype=torch.long, device=device),
    )[:, None]
    frontiers: list[HierarchyFrontier] = [
        HierarchyFrontier(
            ledger=targets.root_ledger[:, None, :],
            hidden=torch.zeros((batch, 1, int(d_model)), device=device),
            support=support_from_ledger(targets.root_ledger)[:, None, :],
            uncertainty=torch.zeros((batch, 1), device=device),
            mask=torch.ones((batch, 1), dtype=torch.bool, device=device),
            topology=root_topology,
            parent_indices=torch.full((batch, 1), -1, dtype=torch.long, device=device),
            source_child_indices=torch.full((batch, 1), -1, dtype=torch.long, device=device),
        )
    ]
    for depth in range(4):
        mask = targets.level_masks[depth]
        frontiers.append(
            HierarchyFrontier(
                ledger=targets.level_ledgers[depth],
                hidden=torch.zeros((*mask.shape, int(d_model)), device=device),
                support=targets.level_supports[depth],
                uncertainty=torch.zeros(mask.shape, device=device),
                mask=mask,
                topology=targets.level_topology[depth],
                parent_indices=targets.level_parent_indices[depth],
                source_child_indices=torch.full_like(targets.level_parent_indices[depth], -1),
            )
        )
    return tuple(frontiers)


def _physical_pairwise_cost(
    predicted_ledger: Any,
    predicted_support: Any,
    target_ledger: Any,
    target_support: Any,
) -> Any:
    torch = require_torch()
    pred = torch.as_tensor(predicted_ledger).float()
    target = torch.as_tensor(target_ledger, device=pred.device).float()
    pred_support = torch.as_tensor(predicted_support, device=pred.device).float()
    target_support = torch.as_tensor(target_support, device=pred.device).float()
    pred_p4 = torch.stack(
        tuple(pred[..., ROOT_FEATURE_INDEX[name]] for name in ("energy", "px", "py", "pz")),
        dim=-1,
    )
    target_p4 = torch.stack(
        tuple(target[..., ROOT_FEATURE_INDEX[name]] for name in ("energy", "px", "py", "pz")),
        dim=-1,
    )
    scale = target_p4[..., 0].abs().clamp_min(1.0)
    p4_cost = ((pred_p4[:, None, :] - target_p4[None, :, :]) / scale[None, :, None]).square().mean(dim=-1)
    pred_count = pred[..., ROOT_FEATURE_INDEX["constituent_count"]]
    target_count = target[..., ROOT_FEATURE_INDEX["constituent_count"]]
    count_scale = target_count.abs().clamp_min(1.0)
    count_cost = ((pred_count[:, None] - target_count[None, :]) / count_scale[None, :]).square()
    pred_types = torch.stack(
        tuple(pred[..., ROOT_FEATURE_INDEX[f"count_{name}"]] for name in ABPH_PID_CATEGORIES),
        dim=-1,
    ) / pred_count.clamp_min(1.0)[..., None]
    target_types = torch.stack(
        tuple(target[..., ROOT_FEATURE_INDEX[f"count_{name}"]] for name in ABPH_PID_CATEGORIES),
        dim=-1,
    ) / target_count.clamp_min(1.0)[..., None]
    type_cost = (pred_types[:, None, :] - target_types[None, :, :]).square().mean(dim=-1)
    delta_eta = pred_support[:, None, 0] - target_support[None, :, 0]
    delta_phi = wrap_phi_tensor(pred_support[:, None, 1] - target_support[None, :, 1])
    geometry_cost = delta_eta.square() + delta_phi.square()
    shape_cost = (
        pred_support[:, None, 2:9] - target_support[None, :, 2:9]
    ).square().mean(dim=-1)
    return p4_cost + 0.5 * count_cost + 0.5 * type_cost + 0.35 * geometry_cost + 0.20 * shape_cost


def local_sibling_match(
    predicted_ledger: Any,
    predicted_support: Any,
    predicted_mask: Any,
    predicted_parent_indices: Any,
    target_ledger: Any,
    target_support: Any,
    target_mask: Any,
    target_parent_indices: Any,
    *,
    parent_mask: Any,
    parent_split_mask: Any,
) -> SiblingMatchResult:
    """Choose one of two sibling permutations independently inside each parent."""

    torch = require_torch()
    pred_mask = torch.as_tensor(predicted_mask).bool()
    target_valid = torch.as_tensor(target_mask, device=pred_mask.device).bool()
    pred_parents = torch.as_tensor(predicted_parent_indices, device=pred_mask.device).to(torch.long)
    target_parents = torch.as_tensor(target_parent_indices, device=pred_mask.device).to(torch.long)
    active_parents = torch.as_tensor(parent_mask, device=pred_mask.device).bool()
    split_parents = torch.as_tensor(parent_split_mask, device=pred_mask.device).bool()
    batch, pred_capacity = pred_mask.shape
    target_indices = torch.full((batch, pred_capacity), -1, dtype=torch.long, device=pred_mask.device)
    matched = torch.zeros_like(pred_mask)
    permutation = torch.full(active_parents.shape, -1, dtype=torch.long, device=pred_mask.device)
    zero = torch.as_tensor(predicted_ledger).sum() * 0.0
    total_cost = zero
    matched_parent_count = 0
    problems: list[str] = []
    for batch_index in range(batch):
        for parent_index in torch.nonzero(active_parents[batch_index] & split_parents[batch_index], as_tuple=False).flatten().tolist():
            pred_children = torch.nonzero(
                pred_mask[batch_index] & (pred_parents[batch_index] == parent_index),
                as_tuple=False,
            ).flatten()
            target_children = torch.nonzero(
                target_valid[batch_index] & (target_parents[batch_index] == parent_index),
                as_tuple=False,
            ).flatten()
            if int(pred_children.numel()) != 2 or int(target_children.numel()) != 2:
                problems.append(
                    f"batch {batch_index} parent {parent_index}: expected two predicted and target siblings"
                )
                continue
            costs = _physical_pairwise_cost(
                predicted_ledger[batch_index, pred_children],
                predicted_support[batch_index, pred_children],
                target_ledger[batch_index, target_children],
                target_support[batch_index, target_children],
            )
            direct = costs[0, 0] + costs[1, 1]
            swapped = costs[0, 1] + costs[1, 0]
            use_swap = bool((swapped.detach() < direct.detach()).item())
            selected = target_children.flip(0) if use_swap else target_children
            target_indices[batch_index, pred_children] = selected
            matched[batch_index, pred_children] = True
            permutation[batch_index, parent_index] = int(use_swap)
            total_cost = total_cost + (swapped if use_swap else direct)
            matched_parent_count += 1
    normalized = total_cost / max(matched_parent_count, 1)
    return SiblingMatchResult(
        target_indices=target_indices,
        matched_mask=matched,
        predicted_parent_indices=pred_parents,
        permutation=permutation,
        cost=normalized,
        diagnostics={
            "contract": ABPH_HIERARCHY_ALIGNMENT_CONTRACT,
            "matched_parent_count": matched_parent_count,
            "parent_local_only": True,
            "problems": problems,
        },
    )


def _slice_prediction(prediction: BinarySplitPrediction, indices: Any) -> BinarySplitPrediction:
    return BinarySplitPrediction(
        **{name: getattr(prediction, name)[indices] for name in prediction.__dataclass_fields__}
    )


def compute_teacher_forced_level_supervision(
    output: HierarchyLevelOutput,
    target_child_ledger: Any,
    target_child_support: Any,
    target_child_mask: Any,
    target_child_parent_indices: Any,
    target_parent_topology: Any,
    *,
    weights: BinaryAccountingLossWeights | None = None,
) -> HierarchyLevelSupervision:
    """Deep supervision with exact local sibling ordering and no padding loss."""

    torch = require_torch()
    parent_mask = output.parent_frontier.mask
    target_topology = torch.as_tensor(target_parent_topology, device=parent_mask.device).to(torch.long)
    if target_topology.shape != parent_mask.shape:
        raise ValueError("target parent topology must match the parent frontier")
    flat_topology = target_topology[parent_mask]
    parent_state = AccountingState.from_ledger(output.parent_frontier.ledger[parent_mask])
    supervised_compiled = compile_binary_split(
        parent_state,
        output.prediction,
        topology_override=flat_topology,
    )
    # The teacher-forced forward already used this exact topology. Reuse its
    # packed child support so centroid/radius/orientation heads receive direct
    # local matching gradients; terminal carry slots remain excluded.
    batch, child_capacity = target_child_mask.shape
    if output.next_frontier.mask.shape != (batch, child_capacity):
        raise ValueError("teacher-forced output and target child capacities differ")
    pred_ledger = output.next_frontier.ledger
    pred_support = output.next_frontier.support
    pred_mask = output.next_frontier.mask & (
        output.next_frontier.source_child_indices >= 0
    )
    pred_parents = output.next_frontier.parent_indices
    match = local_sibling_match(
        pred_ledger,
        pred_support,
        pred_mask,
        pred_parents,
        target_child_ledger,
        target_child_support,
        target_child_mask,
        target_child_parent_indices,
        parent_mask=parent_mask,
        parent_split_mask=target_topology == int(TOPOLOGY_ACTIVE_SPLIT),
    )
    target_rows = torch.zeros(
        (int(output.flat_parent_indices.shape[0]), 2, len(ROOT_FEATURE_NAMES)),
        dtype=target_child_ledger.dtype,
        device=target_child_ledger.device,
    )
    target_rows_mask = torch.zeros(
        (int(output.flat_parent_indices.shape[0]), 2),
        dtype=torch.bool,
        device=target_child_ledger.device,
    )
    for flat_index, pair in enumerate(output.flat_parent_indices.tolist()):
        batch_index, parent_index = int(pair[0]), int(pair[1])
        children = torch.nonzero(
            target_child_mask[batch_index]
            & (target_child_parent_indices[batch_index] == parent_index),
            as_tuple=False,
        ).flatten()
        if int(flat_topology[flat_index]) == int(TOPOLOGY_ACTIVE_SPLIT):
            if int(children.numel()) != 2:
                raise ValueError("split target parent must have exactly two local children")
            # Find the predicted slots for this parent and reuse their selected permutation.
            pred_slots = torch.nonzero(
                pred_mask[batch_index] & (pred_parents[batch_index] == parent_index),
                as_tuple=False,
            ).flatten()
            selected = match.target_indices[batch_index, pred_slots]
            if bool((selected < 0).any()):
                raise RuntimeError("local sibling matching did not cover a split parent")
            target_rows[flat_index] = target_child_ledger[batch_index, selected]
            target_rows_mask[flat_index] = True
        elif int(children.numel()) == 1:
            target_rows[flat_index, 0] = target_child_ledger[batch_index, children[0]]
            target_rows_mask[flat_index, 0] = True
    accounting = compute_binary_accounting_losses(
        output.prediction,
        supervised_compiled,
        parent_state,
        target_rows,
        target_rows_mask,
        flat_topology,
        weights=weights,
    )
    total_loss = accounting.total + match.cost
    return HierarchyLevelSupervision(
        sibling_match=match,
        accounting_loss=accounting,
        total_loss=total_loss,
        supervised_compiled=supervised_compiled,
        diagnostics={
            "mode": "teacher_forced",
            "padding_parent_count": int((~parent_mask).sum().detach().cpu()),
            "padding_contributes_loss": False,
            "local_sibling_cost": float(match.cost.detach().cpu()),
            "total_accounting_loss": float(accounting.total.detach().cpu()),
            "total_level_loss": float(total_loss.detach().cpu()),
        },
    )


def _log_sinkhorn_with_dustbins(
    cost: Any,
    allowed: Any,
    source_mass: Any,
    target_mass: Any,
    *,
    epsilon: float,
    iterations: int,
    null_cost: float,
) -> Any:
    torch = require_torch()
    pred_count, target_count = int(cost.shape[0]), int(cost.shape[1])
    total_source = source_mass.sum()
    total_target = target_mass.sum()
    augmented_source = torch.cat((source_mass, total_target[None]))
    augmented_target = torch.cat((target_mass, total_source[None]))
    augmented_cost = torch.full(
        (pred_count + 1, target_count + 1),
        float(null_cost),
        dtype=cost.dtype,
        device=cost.device,
    )
    augmented_cost[:pred_count, :target_count] = cost
    augmented_allowed = torch.ones_like(augmented_cost, dtype=torch.bool)
    augmented_allowed[:pred_count, :target_count] = allowed
    # The null-to-null corner carries measure displaced by useful real-real
    # matches. Forbidding it would algebraically force every real row and column
    # through a dustbin even when the physical match is excellent.
    augmented_cost[-1, -1] = 0.0
    log_kernel = -augmented_cost / float(epsilon)
    log_kernel = log_kernel.masked_fill(~augmented_allowed, -1.0e9)
    log_source = torch.log(augmented_source.clamp_min(1.0e-12))
    log_target = torch.log(augmented_target.clamp_min(1.0e-12))
    u = torch.zeros_like(log_source)
    v = torch.zeros_like(log_target)
    for _ in range(int(iterations)):
        u = log_source - torch.logsumexp(log_kernel + v[None, :], dim=1)
        v = log_target - torch.logsumexp(log_kernel + u[:, None], dim=0)
    transport = torch.exp(log_kernel + u[:, None] + v[None, :])
    return transport * augmented_allowed.to(transport.dtype)


def align_frontier(
    predicted: HierarchyFrontier,
    target_ledger: Any,
    target_support: Any,
    target_mask: Any,
    target_parent_indices: Any,
    target_membership: Any,
    target_particle_mask: Any,
    *,
    mode: str,
    previous_real_transport: Any | None = None,
    epsilon: float = ABPH_FRONTIER_SINKHORN_EPSILON,
    iterations: int = ABPH_FRONTIER_SINKHORN_ITERATIONS,
    null_cost: float = ABPH_FRONTIER_NULL_COST,
) -> FrontierAlignmentResult:
    """Align complete frontiers with ancestry constraints and explicit null mass."""

    torch = require_torch()
    resolved_mode = str(mode).strip().lower().replace("-", "_")
    if resolved_mode not in {"teacher_forced", "rollout"}:
        raise ValueError("alignment mode must be teacher_forced or rollout")
    target_valid = torch.as_tensor(target_mask, device=predicted.mask.device).bool()
    target_parents = torch.as_tensor(target_parent_indices, device=predicted.mask.device).long()
    membership = torch.as_tensor(target_membership, device=predicted.mask.device).bool()
    particle_mask = torch.as_tensor(target_particle_mask, device=predicted.mask.device).bool()
    batch, pred_capacity = predicted.mask.shape
    target_capacity = int(target_valid.shape[1])
    real_transport = torch.zeros(
        (batch, pred_capacity, target_capacity), dtype=predicted.ledger.dtype, device=predicted.ledger.device
    )
    pred_null = torch.zeros((batch, pred_capacity), dtype=predicted.ledger.dtype, device=predicted.ledger.device)
    target_null = torch.zeros((batch, target_capacity), dtype=predicted.ledger.dtype, device=predicted.ledger.device)
    allowed_full = torch.zeros((batch, pred_capacity, target_capacity), dtype=torch.bool, device=predicted.ledger.device)
    costs_full = torch.zeros_like(real_transport)
    particle_weights = torch.zeros(
        (batch, pred_capacity, ABPH_MAX_PARTICLES), dtype=predicted.ledger.dtype, device=predicted.ledger.device
    )
    target_null_particles = torch.zeros(
        (batch, ABPH_MAX_PARTICLES), dtype=predicted.ledger.dtype, device=predicted.ledger.device
    )
    losses: list[Any] = []
    unequal_cardinality = 0
    ancestry_blocked_pairs = 0
    for batch_index in range(batch):
        pred_indices = torch.nonzero(predicted.mask[batch_index], as_tuple=False).flatten()
        target_indices = torch.nonzero(target_valid[batch_index], as_tuple=False).flatten()
        if not int(pred_indices.numel()) or not int(target_indices.numel()):
            raise ValueError("frontier alignment requires nonempty predicted and target frontiers")
        unequal_cardinality += int(pred_indices.numel() != target_indices.numel())
        pair_cost = _physical_pairwise_cost(
            predicted.ledger[batch_index, pred_indices],
            predicted.support[batch_index, pred_indices],
            target_ledger[batch_index, target_indices],
            target_support[batch_index, target_indices],
        )
        if previous_real_transport is None:
            allowed = torch.ones_like(pair_cost, dtype=torch.bool)
        else:
            previous = torch.as_tensor(previous_real_transport, device=pair_cost.device)
            pred_parent = predicted.parent_indices[batch_index, pred_indices]
            target_parent = target_parents[batch_index, target_indices]
            if bool((pred_parent < 0).any()) or bool((target_parent < 0).any()):
                raise ValueError("non-root ancestry alignment requires valid parent indices")
            allowed = previous[batch_index][pred_parent[:, None], target_parent[None, :]] > 1.0e-10
        ancestry_blocked_pairs += int((~allowed).sum().detach().cpu())
        pred_mass = predicted.ledger[
            batch_index, pred_indices, ROOT_FEATURE_INDEX["constituent_count"]
        ].clamp_min(1.0)
        target_mass = target_ledger[
            batch_index, target_indices, ROOT_FEATURE_INDEX["constituent_count"]
        ].clamp_min(1.0)
        transport = _log_sinkhorn_with_dustbins(
            pair_cost,
            allowed,
            pred_mass,
            target_mass,
            epsilon=epsilon,
            iterations=iterations,
            null_cost=null_cost,
        )
        real = transport[:-1, :-1]
        to_null = transport[:-1, -1]
        from_null = transport[-1, :-1]
        real_transport[batch_index][pred_indices[:, None], target_indices[None, :]] = real.to(
            real_transport.dtype
        )
        pred_null[batch_index, pred_indices] = to_null.to(pred_null.dtype)
        target_null[batch_index, target_indices] = from_null.to(target_null.dtype)
        allowed_full[batch_index][pred_indices[:, None], target_indices[None, :]] = allowed
        costs_full[batch_index][pred_indices[:, None], target_indices[None, :]] = pair_cost.to(
            costs_full.dtype
        )
        normalization = (pred_mass.sum() + target_mass.sum()).clamp_min(1.0)
        losses.append(
            (
                (real * pair_cost).sum()
                + float(null_cost) * (to_null.sum() + from_null.sum())
            )
            / normalization
        )
        target_members = membership[batch_index, target_indices].to(real.dtype)
        per_particle = real / target_mass[None, :].clamp_min(1.0)
        particle_weights[batch_index, pred_indices] = (
            per_particle @ target_members
        ).to(particle_weights.dtype)
        target_null_particles[batch_index] = (
            from_null / target_mass.clamp_min(1.0)
        ).matmul(target_members).to(target_null_particles.dtype)
    loss = torch.stack(losses).mean()
    hard_assignment = torch.full(
        (batch, ABPH_MAX_PARTICLES), -1, dtype=torch.long, device=predicted.ledger.device
    )
    for batch_index in range(batch):
        maximum, owner = particle_weights[batch_index].max(dim=0)
        use_real = particle_mask[batch_index] & (maximum >= target_null_particles[batch_index])
        hard_assignment[batch_index, use_real] = owner[use_real]
    particle_indices = torch.arange(ABPH_MAX_PARTICLES, device=predicted.ledger.device)
    particle_indices = particle_indices[None, None, :].expand(batch, pred_capacity, -1)
    renderer = RendererTargetMap(
        target_particle_indices=particle_indices,
        target_particle_weights=particle_weights,
        target_particle_mask=particle_weights > 1.0e-10,
        predicted_null_mass=pred_null,
        target_null_particle_weight=target_null_particles,
        hard_particle_assignment=hard_assignment,
        terminal_mask=predicted.mask & (predicted.topology == int(TOPOLOGY_ACTIVE_TERMINAL)),
        diagnostics={
            "all_predicted_terminals_have_maps": bool(
                ((particle_weights.sum(dim=-1) + pred_null) > 0.0)[
                    predicted.mask & (predicted.topology == int(TOPOLOGY_ACTIVE_TERMINAL))
                ].all()
            ),
            "particle_measure_max_residual": float(
                (
                    particle_weights.sum(dim=1)
                    + target_null_particles
                    - particle_mask.to(particle_weights.dtype)
                ).abs().max().detach().cpu()
            ),
            "predicted_measure_max_residual": float(
                (
                    particle_weights.sum(dim=-1)
                    + pred_null
                    - predicted.ledger[:, :, ROOT_FEATURE_INDEX["constituent_count"]]
                    * predicted.mask
                ).abs().max().detach().cpu()
            ),
            "null_mass_is_explicit": True,
        },
    )
    return FrontierAlignmentResult(
        mode=resolved_mode,
        real_transport=real_transport,
        predicted_to_null=pred_null,
        null_to_target=target_null,
        allowed_ancestry=allowed_full,
        pairwise_cost=costs_full,
        loss=loss,
        renderer_target_map=renderer,
        diagnostics={
            "contract": ABPH_HIERARCHY_ALIGNMENT_CONTRACT,
            "mode": resolved_mode,
            "predicted_cardinality": predicted.mask.sum(dim=1).tolist(),
            "target_cardinality": target_valid.sum(dim=1).tolist(),
            "unequal_cardinality_examples": unequal_cardinality,
            "ancestry_blocked_pairs": ancestry_blocked_pairs,
            "predicted_null_mass": float(pred_null.sum().detach().cpu()),
            "target_null_mass": float(target_null.sum().detach().cpu()),
            "unmatched_nodes_silently_dropped": False,
        },
    )


def align_recursive_hierarchy(
    output: RecursiveHierarchyOutput,
    targets: HierarchyTargetTensors,
) -> HierarchyAlignmentOutput:
    if len(output.levels) != len(targets.level_ledgers):
        raise ValueError("decoder and target hierarchy depths differ")
    previous = None
    alignments: list[FrontierAlignmentResult] = []
    for depth, level in enumerate(output.levels):
        predicted = level.next_frontier
        if depth + 1 < len(output.levels) and output.mode == "rollout":
            # The next decoder's hard decision is the topology of this frontier.
            next_level = output.levels[depth + 1]
            topology = predicted.topology.clone()
            topology[next_level.parent_frontier.mask] = next_level.compiled.topology
            predicted = HierarchyFrontier(
                ledger=predicted.ledger,
                hidden=predicted.hidden,
                support=predicted.support,
                uncertainty=predicted.uncertainty,
                mask=predicted.mask,
                topology=topology,
                parent_indices=predicted.parent_indices,
                source_child_indices=predicted.source_child_indices,
            )
        ancestry_transport = previous
        if output.mode == "teacher_forced" and depth > 0:
            previous_mask = targets.level_masks[depth - 1]
            previous_count = targets.level_ledgers[depth - 1][
                :, :, ROOT_FEATURE_INDEX["constituent_count"]
            ]
            ancestry_transport = require_torch().diag_embed(
                previous_count * previous_mask.to(previous_count.dtype)
            )
        alignment = align_frontier(
            predicted,
            targets.level_ledgers[depth],
            targets.level_supports[depth],
            targets.level_masks[depth],
            targets.level_parent_indices[depth],
            targets.level_membership[depth],
            targets.particle_mask,
            mode=output.mode,
            previous_real_transport=ancestry_transport,
        )
        alignments.append(alignment)
        previous = alignment.real_transport
    total = sum(value.loss for value in alignments) / len(alignments)
    return HierarchyAlignmentOutput(
        mode=output.mode,
        levels=tuple(alignments),
        total_frontier_loss=total,
        diagnostics={
            "contract": ABPH_HIERARCHY_ALIGNMENT_CONTRACT,
            "mode": output.mode,
            "n_levels": len(alignments),
            "teacher_forced_report": output.mode == "teacher_forced",
            "rollout_report": output.mode == "rollout",
            "all_levels_have_renderer_maps": len(alignments) == len(targets.level_ledgers),
            "total_predicted_null_mass": sum(
                float(value.predicted_to_null.sum().detach().cpu()) for value in alignments
            ),
            "total_target_null_mass": sum(
                float(value.null_to_target.sum().detach().cpu()) for value in alignments
            ),
        },
    )


__all__ = [
    "ABPH_FRONTIER_NULL_COST",
    "ABPH_FRONTIER_SINKHORN_EPSILON",
    "ABPH_FRONTIER_SINKHORN_ITERATIONS",
    "ABPH_HIERARCHY_ALIGNMENT_CONTRACT",
    "FrontierAlignmentResult",
    "HierarchyAlignmentOutput",
    "HierarchyLevelSupervision",
    "HierarchyTargetTensors",
    "RendererTargetMap",
    "SiblingMatchResult",
    "align_frontier",
    "align_recursive_hierarchy",
    "build_teacher_parent_frontiers",
    "compute_teacher_forced_level_supervision",
    "group_support_from_target_features",
    "hierarchy_targets_to_tensors",
    "local_sibling_match",
]
