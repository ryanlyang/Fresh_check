"""Exact local particle rendering for adaptive binary pseudo-offline trees."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .binary_accounting import AccountingState
from .hierarchy_decoder import (
    ABPH_GROUP_SUPPORT_DIM,
    ABPH_HLT_SUPPORT_FEATURE_NAMES,
    RecursiveHierarchyOutput,
)
from .root_transforms import ROOT_FEATURE_INDEX, wrap_phi_tensor
from .schemas import ABPH_EFFECTIVE_MASS_GEV, ABPH_MAX_PARTICLES, ABPH_PID_CATEGORIES
from .targets import PARTICLE_TARGET_NAMES, ROOT_FEATURE_NAMES, TOPOLOGY_ACTIVE_TERMINAL


try:  # Keep schemas importable without the training environment.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


ABPH_PARTICLE_RENDERER_CONTRACT = "adaptive_binary_pseudooffline_particle_renderer_v1"
ABPH_RENDERED_SIDE_CHANNEL_NAMES: tuple[str, ...] = (
    "prediction_uncertainty",
    "hypothesis_index",
    "microgroup_confidence",
    "ancestor_depth_confidence",
    "predicted_observed_support_score",
    "source_is_pseudo",
)
_PID_MASSES = tuple(float(ABPH_EFFECTIVE_MASS_GEV[name]) for name in ABPH_PID_CATEGORIES)
_CHARGE_SUPPORT = (-1, 0, 1)


@dataclass(frozen=True)
class ParticleRendererConfig:
    hlt_input_dims: tuple[int, ...] = (192,)
    d_model: int = 256
    num_heads: int = 8
    ffn_dim: int = 1024
    blocks: int = 4
    dropout: float = 0.10
    attention_dropout: float = 0.10
    root_semantic_dim: int = 256
    latent_dim: int = 64
    maximum_particles: int = ABPH_MAX_PARTICLES
    type_sinkhorn_iterations: int = 40
    type_temperature: float = 0.50
    charge_temperature: float = 0.50
    phase_space_iterations: int = 64
    phase_space_mass_epsilon: float = 1.0e-5
    near_massless_threshold: float = 1.0e-6
    maximum_local_attention_bias: float = 5.0
    exact_nbody_projection: bool = True
    local_matching: bool = True

    def __post_init__(self) -> None:
        if not self.hlt_input_dims or any(int(value) <= 0 for value in self.hlt_input_dims):
            raise ValueError("hlt_input_dims must contain positive dimensions")
        for name in (
            "d_model",
            "num_heads",
            "ffn_dim",
            "blocks",
            "root_semantic_dim",
            "latent_dim",
            "type_sinkhorn_iterations",
            "phase_space_iterations",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.d_model) % int(self.num_heads):
            raise ValueError("d_model must be divisible by num_heads")
        if int(self.maximum_particles) != ABPH_MAX_PARTICLES:
            raise ValueError(f"maximum_particles must be exactly {ABPH_MAX_PARTICLES}")
        for name in ("dropout", "attention_dropout"):
            if not 0.0 <= float(getattr(self, name)) < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
        for name in (
            "type_temperature",
            "charge_temperature",
            "phase_space_mass_epsilon",
            "near_massless_threshold",
            "maximum_local_attention_bias",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "contract": ABPH_PARTICLE_RENDERER_CONTRACT,
                "slot_count_source": "compiled_microgroup_integer_count",
                "particle_count_threshold": False,
                "local_self_attention_crosses_groups": False,
                "node_local_noise": False,
                "pid_hard_allocation": "exact_minimum_cost_quota_assignment",
                "charge_hard_allocation": "exact_feasible_dynamic_program",
                "phase_space": "differentiable_bracketed_n_body_rest_frame",
            }
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["config_hash"] = hashlib.sha256(encoded).hexdigest()
        return payload


@dataclass(frozen=True)
class ParticleSlotLayout:
    mask: Any
    group_indices: Any
    local_slot_indices: Any
    group_counts: Any
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class TypeAllocation:
    soft_probabilities: Any
    hard_indices: Any
    hard_one_hot: Any
    straight_through_one_hot: Any
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class ChargeAllocation:
    soft_probabilities: Any
    hard_charges: Any
    straight_through_charges: Any
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class RenderedParticleBatch:
    four_vector: Any
    mass: Any
    mask: Any
    group_indices: Any
    local_slot_indices: Any
    soft_pid_probabilities: Any
    hard_pid_indices: Any
    hard_pid_one_hot: Any
    charges: Any
    hard_charges: Any
    track_features: Any
    uncertainty: Any
    canonical_features: Any
    side_channels: Any
    slot_hidden: Any
    diagnostics: Mapping[str, Any]


def exact_particle_slot_layout(final_frontier: Any, root_state: AccountingState) -> ParticleSlotLayout:
    """Allocate exactly one contiguous output slot per compiled constituent."""

    torch = require_torch()
    mask = torch.as_tensor(final_frontier.mask).bool()
    topology = torch.as_tensor(final_frontier.topology, device=mask.device).long()
    counts = final_frontier.ledger[:, :, ROOT_FEATURE_INDEX["constituent_count"]].round().long()
    if bool((mask & (topology != int(TOPOLOGY_ACTIVE_TERMINAL))).any()):
        raise ValueError("particle rendering requires an all-terminal final frontier")
    if bool((counts[mask] <= 0).any()):
        raise ValueError("active microgroups must have positive constituent counts")
    batch, _ = mask.shape
    active_counts = torch.where(mask, counts, torch.zeros_like(counts))
    cumulative = active_counts.cumsum(dim=1)
    root_count = root_state.constituent_count.round().long()
    if bool((root_count < 0).any()) or bool(
        (root_count > ABPH_MAX_PARTICLES).any()
    ):
        raise RuntimeError("compiled root count exceeds the 128-slot contract")
    if not bool((active_counts.sum(dim=1) == root_count).all()):
        raise RuntimeError("final microgroup counts do not close to the root count")
    slot = torch.arange(ABPH_MAX_PARTICLES, device=mask.device).expand(batch, -1)
    slot_mask = slot < root_count[:, None]
    # right=True skips repeated cumulative boundaries from inactive groups.
    resolved_group = torch.searchsorted(
        cumulative.contiguous(), slot.contiguous(), right=True
    ).clamp_max(mask.shape[1] - 1)
    group_indices = torch.where(
        slot_mask, resolved_group, torch.full_like(resolved_group, -1)
    )
    preceding = torch.cat(
        (torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]), dim=1
    )
    group_start = preceding.gather(1, resolved_group)
    local_indices = torch.where(
        slot_mask, slot - group_start, torch.full_like(slot, -1)
    )
    return ParticleSlotLayout(
        mask=slot_mask,
        group_indices=group_indices,
        local_slot_indices=local_indices,
        group_counts=counts,
        diagnostics={
            "contract": ABPH_PARTICLE_RENDERER_CONTRACT,
            "slot_count_source": "compiled_integer_ledger",
            "exact_root_count_closure": True,
            "existence_threshold_used": False,
            "truncation_used": False,
            "rendered_counts": slot_mask.sum(dim=1).tolist(),
        },
    )


def _minimum_cost_square_assignment_rows(detached: list[list[float]]) -> list[int]:
    """Exact O(N^3) Hungarian assignment over already-host-resident rows."""

    n = len(detached)
    if any(len(row) != n for row in detached):
        raise ValueError("minimum-cost assignment requires a square cost matrix")
    if any(not math.isfinite(float(value)) for row in detached for value in row):
        raise ValueError("minimum-cost assignment received nonfinite costs")
    if n == 0:
        return []
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for row in range(1, n + 1):
        p[0] = row
        column0 = 0
        minimum = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, n + 1):
                if used[column]:
                    continue
                current = detached[row0 - 1][column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            if not math.isfinite(delta):
                raise RuntimeError("minimum-cost assignment has no feasible solution")
            for column in range(n + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = [-1] * n
    for column in range(1, n + 1):
        if p[column]:
            assignment[p[column] - 1] = column - 1
    if any(value < 0 for value in assignment):
        raise RuntimeError("minimum-cost assignment did not cover every row")
    return assignment


def _minimum_cost_square_assignment(cost: Any) -> Any:
    """Exact O(N^3) Hungarian assignment without an optional SciPy dependency."""

    torch = require_torch()
    values = torch.as_tensor(cost)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("minimum-cost assignment requires a square cost matrix")
    detached = values.detach().to(torch.float64).cpu().tolist()
    assignment = _minimum_cost_square_assignment_rows(detached)
    return torch.tensor(assignment, dtype=torch.long, device=values.device)


def _quota_sinkhorn(logits: Any, quotas: Any, *, temperature: float, iterations: int) -> Any:
    torch = require_torch()
    # Reductions such as logsumexp are promoted by AMP. Keep the complete
    # transport in FP32 so indexed publication cannot cross BF16/FP32 dtypes.
    scores = torch.as_tensor(logits).float()
    counts = torch.as_tensor(quotas, device=scores.device).long()
    n_slots = int(scores.shape[0])
    if counts.shape != (len(ABPH_PID_CATEGORIES),) or int(counts.sum()) != n_slots:
        raise ValueError("type quotas must close exactly to the local slot count")
    active = counts > 0
    log_transport = scores[:, active] / float(temperature)
    log_column_mass = counts[active].to(scores.dtype).log()
    for _ in range(int(iterations)):
        log_transport = log_transport - torch.logsumexp(log_transport, dim=1, keepdim=True)
        log_transport = (
            log_transport
            - torch.logsumexp(log_transport, dim=0, keepdim=True)
            + log_column_mass[None, :]
        )
    log_transport = log_transport - torch.logsumexp(log_transport, dim=1, keepdim=True)
    result = torch.zeros_like(scores)
    result[:, active] = log_transport.exp()
    return result


def _active_group_rows(layout: ParticleSlotLayout, group_count: int) -> tuple[Any, Any, Any]:
    """Return active (batch, group) pairs and their fixed-width slot masks."""

    torch = require_torch()
    group_axis = torch.arange(
        int(group_count), device=layout.mask.device
    ).view(1, -1, 1)
    membership = (
        layout.mask[:, None, :]
        & (layout.group_indices[:, None, :] == group_axis)
    )
    pairs = torch.nonzero(membership.any(dim=-1), as_tuple=False)
    return pairs[:, 0], pairs[:, 1], membership[pairs[:, 0], pairs[:, 1]]


def _batched_quota_sinkhorn(
    logits: Any,
    slot_mask: Any,
    quotas: Any,
    *,
    temperature: float,
    iterations: int,
) -> Any:
    """Solve all group quota transports in one fixed-width GPU batch."""

    torch = require_torch()
    scores = torch.as_tensor(logits).float()
    valid = torch.as_tensor(slot_mask, device=scores.device).bool()
    counts = torch.as_tensor(quotas, device=scores.device).long()
    if scores.ndim != 3 or valid.shape != scores.shape[:2]:
        raise ValueError("batched quota transport shapes are inconsistent")
    if counts.shape != (scores.shape[0], scores.shape[2]):
        raise ValueError("batched quota counts have the wrong shape")
    if not bool((counts.sum(dim=-1) == valid.sum(dim=-1)).all()):
        raise ValueError("batched type quotas do not close to slot counts")
    active_columns = counts > 0
    support = valid[:, :, None] & active_columns[:, None, :]
    log_transport = (scores / float(temperature)).masked_fill(
        ~support, float("-inf")
    )
    log_column_mass = counts.clamp_min(1).to(scores.dtype).log()
    for _ in range(int(iterations)):
        row_normalizer = torch.logsumexp(log_transport, dim=-1, keepdim=True)
        row_normalizer = torch.where(
            valid[:, :, None],
            row_normalizer,
            torch.zeros_like(row_normalizer),
        )
        log_transport = torch.where(
            support,
            log_transport - row_normalizer,
            torch.full_like(log_transport, float("-inf")),
        )
        column_normalizer = torch.logsumexp(log_transport, dim=1, keepdim=True)
        column_normalizer = torch.where(
            active_columns[:, None, :],
            column_normalizer,
            torch.zeros_like(column_normalizer),
        )
        log_transport = torch.where(
            support,
            log_transport
            - column_normalizer
            + log_column_mass[:, None, :],
            torch.full_like(log_transport, float("-inf")),
        )
    row_normalizer = torch.logsumexp(log_transport, dim=-1, keepdim=True)
    row_normalizer = torch.where(
        valid[:, :, None], row_normalizer, torch.zeros_like(row_normalizer)
    )
    return torch.where(
        support,
        (log_transport - row_normalizer).exp(),
        torch.zeros_like(log_transport),
    )


def allocate_particle_types(
    logits: Any,
    layout: ParticleSlotLayout,
    group_type_counts: Any,
    *,
    temperature: float,
    sinkhorn_iterations: int,
) -> TypeAllocation:
    """Relaxed quota transport plus exact hard minimum-cost type allocation."""

    torch = require_torch()
    scores = torch.as_tensor(logits)
    counts = torch.as_tensor(group_type_counts, device=scores.device).long()
    if scores.shape != (*layout.mask.shape, len(ABPH_PID_CATEGORIES)):
        raise ValueError("particle type logits have the wrong shape")
    batch_rows, group_rows, slot_rows = _active_group_rows(
        layout, counts.shape[1]
    )
    grouped_scores = scores[batch_rows]
    grouped_quotas = counts[batch_rows, group_rows]
    grouped_soft = _batched_quota_sinkhorn(
        grouped_scores,
        slot_rows,
        grouped_quotas,
        temperature=temperature,
        iterations=sinkhorn_iterations,
    )
    soft = torch.zeros(scores.shape, dtype=torch.float32, device=scores.device)
    batch_grid = batch_rows[:, None].expand_as(slot_rows)
    slot_grid = torch.arange(
        scores.shape[1], device=scores.device
    )[None, :].expand_as(slot_rows)
    soft[batch_grid[slot_rows], slot_grid[slot_rows]] = grouped_soft[slot_rows]

    # Hard quota assignment is nondifferentiable. Transfer every active group
    # once, solve the small exact Hungarian problems on the host, then publish
    # all selected types with one device transfer.
    host_scores = grouped_scores.detach().to(torch.float64).cpu()
    host_slots = slot_rows.cpu()
    host_quotas = grouped_quotas.cpu()
    grouped_hard = torch.full(
        slot_rows.shape, -1, dtype=torch.long, device="cpu"
    )
    for row_index in range(slot_rows.shape[0]):
        active_slots = torch.nonzero(host_slots[row_index], as_tuple=False).flatten()
        quotas = host_quotas[row_index]
        expanded_types = torch.repeat_interleave(
            torch.arange(len(ABPH_PID_CATEGORIES)), quotas
        )
        local_cost = (
            -host_scores[row_index, active_slots][:, expanded_types]
        ).tolist()
        assignment = _minimum_cost_square_assignment_rows(local_cost)
        grouped_hard[row_index, active_slots] = expanded_types[
            torch.tensor(assignment, dtype=torch.long)
        ]
    grouped_hard = grouped_hard.to(scores.device)
    hard_indices = torch.full(
        layout.mask.shape, -1, dtype=torch.long, device=scores.device
    )
    hard_indices[batch_grid[slot_rows], slot_grid[slot_rows]] = grouped_hard[
        slot_rows
    ]
    hard = torch.nn.functional.one_hot(
        hard_indices.clamp_min(0), num_classes=len(ABPH_PID_CATEGORIES)
    ).to(soft.dtype) * layout.mask.unsqueeze(-1)
    # Parenthesize the zero-valued gradient carrier so the forward value is
    # exactly the hard quota assignment rather than a rounded (hard + soft)
    # subtraction under mixed precision.
    straight_through = hard + (soft - soft.detach())
    grouped_hard_one_hot = torch.nn.functional.one_hot(
        grouped_hard.clamp_min(0), num_classes=len(ABPH_PID_CATEGORIES)
    ).to(soft.dtype) * slot_rows.unsqueeze(-1)
    grouped_hard_counts = grouped_hard_one_hot.sum(dim=1).long()
    grouped_expected_counts = grouped_soft.sum(dim=1)
    hard_counts = torch.zeros_like(counts)
    expected_counts = torch.zeros_like(counts, dtype=soft.dtype)
    hard_counts[batch_rows, group_rows] = grouped_hard_counts
    expected_counts[batch_rows, group_rows] = grouped_expected_counts
    if not bool((hard_counts == counts).all()):
        raise RuntimeError("hard PID assignment failed exact type-count closure")
    return TypeAllocation(
        soft_probabilities=soft,
        hard_indices=hard_indices,
        hard_one_hot=hard,
        straight_through_one_hot=straight_through,
        diagnostics={
            "hard_type_counts_close_exactly": True,
            "soft_row_max_residual": float(
                (soft.sum(dim=-1) - layout.mask.to(soft.dtype)).abs().max().detach().cpu()
            ),
            "soft_quota_max_residual": float(
                (expected_counts - counts.to(soft.dtype)).abs().max().detach().cpu()
            ),
            "hard_assignment": "minimum_cost_quota_expansion_hungarian",
        },
    )


def _type_conditioned_minimum_mass(type_allocation: TypeAllocation, mass_table: Any) -> Any:
    """Return exact hard PID masses with straight-through type gradients."""

    torch = require_torch()
    hard = torch.as_tensor(type_allocation.hard_one_hot).float()
    straight_through = torch.as_tensor(type_allocation.straight_through_one_hot).float()
    table = torch.as_tensor(mass_table, device=hard.device).float()
    # A BF16 matmul can accumulate a material mass-floor error over 128 slots.
    # Elementwise FP32 accounting keeps the physical forward value identical
    # to the discrete PID assignment while retaining the relaxed gradient.
    with torch.autocast(device_type=hard.device.type, enabled=False):
        hard_mass = (hard * table).sum(dim=-1)
        relaxed_mass = (straight_through * table).sum(dim=-1)
        return hard_mass + (relaxed_mass - relaxed_mass.detach())


def _allocate_local_particle_masses(
    parent_mass: Any,
    local_minimum: Any,
    mass_logits: Any,
    group_fraction: Any,
    *,
    phase_space_mass_epsilon: float,
    near_massless_threshold: float,
) -> Any:
    """Allocate optional rest mass without violating the massless limit."""

    torch = require_torch()
    parent = torch.as_tensor(parent_mass).to(torch.float64)
    minimum = torch.as_tensor(local_minimum, device=parent.device).to(torch.float64)
    logits = torch.as_tensor(mass_logits, device=parent.device).float()
    fraction = torch.as_tensor(group_fraction, device=parent.device).to(torch.float64)
    available = (
        (1.0 - float(phase_space_mass_epsilon)) * parent - minimum.sum()
    ).clamp_min(0.0)
    optional_mass = (
        available
        * fraction
        * logits.softmax(dim=0).to(torch.float64)
    )
    # A lightlike parent has no rest-frame energy available for learned child
    # masses. Keep only immutable PID floors; a nonzero floor then fails the
    # projector's physical feasibility guard with useful diagnostics.
    optional_mass = torch.where(
        parent > float(near_massless_threshold),
        optional_mass,
        torch.zeros_like(optional_mass),
    )
    return minimum + optional_mass


def _allowed_charge_mask(hard_pid_indices: Any) -> Any:
    torch = require_torch()
    pid = torch.as_tensor(hard_pid_indices).long()
    allowed = torch.zeros((*pid.shape, 3), dtype=torch.bool, device=pid.device)
    fixed_charged = (pid == 0) | (pid == 3) | (pid == 4)
    other = pid == 5
    neutral = (pid == 1) | (pid == 2)
    allowed[..., 0] = fixed_charged | other
    allowed[..., 1] = neutral | other
    allowed[..., 2] = fixed_charged | other
    return allowed


def _minimum_cost_charge_sequence_rows(
    detached: Any, permitted_cpu: Any, target_charge: int
) -> tuple[int, ...]:
    """Exact charge DP over host-resident rows."""

    n_slots = int(detached.shape[0])
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for slot in range(n_slots):
        updated: dict[int, tuple[float, tuple[int, ...]]] = {}
        for running, (cost, path) in states.items():
            for charge_index, charge in enumerate(_CHARGE_SUPPORT):
                if not bool(permitted_cpu[slot, charge_index]):
                    continue
                candidate_charge = running + int(charge)
                candidate = cost - float(detached[slot, charge_index])
                previous = updated.get(candidate_charge)
                candidate_path = path + (int(charge),)
                if previous is None or candidate < previous[0]:
                    updated[candidate_charge] = (candidate, candidate_path)
        states = updated
    if int(target_charge) not in states:
        raise RuntimeError("compiled charge budget is infeasible for allocated particle types")
    return states[int(target_charge)][1]


def _minimum_cost_charge_sequence(logits: Any, allowed: Any, target_charge: int) -> Any:
    torch = require_torch()
    scores = torch.as_tensor(logits)
    permitted = torch.as_tensor(allowed, device=scores.device).bool()
    sequence = _minimum_cost_charge_sequence_rows(
        scores.detach().to(torch.float64).cpu(),
        permitted.cpu(),
        target_charge,
    )
    return torch.tensor(sequence, dtype=torch.long, device=scores.device)


def _charge_soft_probabilities(
    logits: Any,
    allowed: Any,
    target_charge: int,
    *,
    temperature: float,
) -> Any:
    torch = require_torch()
    # The constrained expectation solve is a numerical projection, not a
    # neural layer. FP32 also makes its indexed caller dtype-stable under AMP.
    scores = torch.as_tensor(logits).float()
    permitted = torch.as_tensor(allowed, device=scores.device).bool()
    support = scores.new_tensor(_CHARGE_SUPPORT)
    lower = scores.new_tensor(-40.0)
    upper = scores.new_tensor(40.0)
    for _ in range(64):
        middle = 0.5 * (lower + upper)
        tilted = (scores + middle * support) / float(temperature)
        probabilities = tilted.masked_fill(~permitted, float("-inf")).softmax(dim=-1)
        expectation = (probabilities * support).sum()
        lower = torch.where(expectation < float(target_charge), middle, lower)
        upper = torch.where(expectation < float(target_charge), upper, middle)
    multiplier = 0.5 * (lower + upper)
    return ((scores + multiplier * support) / float(temperature)).masked_fill(
        ~permitted, float("-inf")
    ).softmax(dim=-1)


def _batched_charge_soft_probabilities(
    logits: Any,
    allowed: Any,
    slot_mask: Any,
    target_charge: Any,
    *,
    temperature: float,
) -> Any:
    """Project all local expected charges with one batched multiplier solve."""

    torch = require_torch()
    scores = torch.as_tensor(logits).float()
    permitted = torch.as_tensor(allowed, device=scores.device).bool()
    valid = torch.as_tensor(slot_mask, device=scores.device).bool()
    targets = torch.as_tensor(target_charge, device=scores.device).float()
    support = scores.new_tensor(_CHARGE_SUPPORT)
    lower = scores.new_full((scores.shape[0],), -40.0)
    upper = scores.new_full((scores.shape[0],), 40.0)
    active = permitted & valid[:, :, None]
    for _ in range(64):
        middle = 0.5 * (lower + upper)
        tilted = (
            scores + middle[:, None, None] * support
        ) / float(temperature)
        tilted = tilted.masked_fill(~permitted, float("-inf"))
        tilted = torch.where(
            valid[:, :, None], tilted, torch.zeros_like(tilted)
        )
        probabilities = tilted.softmax(dim=-1)
        probabilities = torch.where(
            valid[:, :, None], probabilities, torch.zeros_like(probabilities)
        )
        expectation = (probabilities * support).sum(dim=(1, 2))
        below = expectation < targets
        lower = torch.where(below, middle, lower)
        upper = torch.where(below, upper, middle)
    multiplier = 0.5 * (lower + upper)
    final_logits = (
        scores + multiplier[:, None, None] * support
    ) / float(temperature)
    final_logits = final_logits.masked_fill(~permitted, float("-inf"))
    final_logits = torch.where(
        valid[:, :, None], final_logits, torch.zeros_like(final_logits)
    )
    result = final_logits.softmax(dim=-1)
    return torch.where(valid[:, :, None], result, torch.zeros_like(result))


def allocate_particle_charges(
    logits: Any,
    layout: ParticleSlotLayout,
    hard_pid_indices: Any,
    group_integer_charge: Any,
    *,
    temperature: float,
) -> ChargeAllocation:
    """Assign legal per-type charges with exact local integer-charge closure."""

    torch = require_torch()
    scores = torch.as_tensor(logits)
    targets = torch.as_tensor(group_integer_charge, device=scores.device).round().long()
    if scores.shape != (*layout.mask.shape, 3):
        raise ValueError("charge logits must have shape [B, 128, 3]")
    allowed_all = _allowed_charge_mask(hard_pid_indices)
    batch_rows, group_rows, slot_rows = _active_group_rows(
        layout, targets.shape[1]
    )
    grouped_scores = scores[batch_rows]
    grouped_allowed = allowed_all[batch_rows]
    grouped_targets = targets[batch_rows, group_rows]
    grouped_soft = _batched_charge_soft_probabilities(
        grouped_scores,
        grouped_allowed,
        slot_rows,
        grouped_targets,
        temperature=temperature,
    )
    batch_grid = batch_rows[:, None].expand_as(slot_rows)
    slot_grid = torch.arange(
        scores.shape[1], device=scores.device
    )[None, :].expand_as(slot_rows)
    soft = torch.zeros(scores.shape, dtype=torch.float32, device=scores.device)
    soft[batch_grid[slot_rows], slot_grid[slot_rows]] = grouped_soft[slot_rows]

    host_scores = grouped_scores.detach().to(torch.float64).cpu()
    host_allowed = grouped_allowed.cpu()
    host_slots = slot_rows.cpu()
    host_targets = grouped_targets.cpu()
    grouped_hard = torch.zeros(slot_rows.shape, dtype=torch.long, device="cpu")
    for row_index in range(slot_rows.shape[0]):
        active_slots = torch.nonzero(host_slots[row_index], as_tuple=False).flatten()
        sequence = _minimum_cost_charge_sequence_rows(
            host_scores[row_index, active_slots],
            host_allowed[row_index, active_slots],
            int(host_targets[row_index]),
        )
        grouped_hard[row_index, active_slots] = torch.tensor(
            sequence, dtype=torch.long
        )
    grouped_hard = grouped_hard.to(scores.device)
    hard = torch.zeros(layout.mask.shape, dtype=torch.long, device=scores.device)
    hard[batch_grid[slot_rows], slot_grid[slot_rows]] = grouped_hard[slot_rows]
    expected = (soft * scores.new_tensor(_CHARGE_SUPPORT)).sum(dim=-1)
    straight_through = hard.to(soft.dtype) + expected - expected.detach()
    hard_group_charge = torch.zeros_like(targets)
    hard_group_charge[batch_rows, group_rows] = (
        grouped_hard * slot_rows
    ).sum(dim=1)
    if not bool((hard_group_charge == targets).all()):
        raise RuntimeError("hard particle charges failed exact group closure")
    selected_allowed = allowed_all.gather(
        -1, (hard + 1).clamp(0, 2).unsqueeze(-1)
    ).squeeze(-1)
    if not bool(selected_allowed[layout.mask].all()):
        raise RuntimeError("hard particle charge violates its selected PID support")
    return ChargeAllocation(
        soft_probabilities=soft,
        hard_charges=hard,
        straight_through_charges=straight_through,
        diagnostics={
            "hard_group_charge_closes_exactly": True,
            "hard_charges_respect_pid_support": True,
            "soft_expected_charge_max_residual": float(
                (
                    (grouped_soft * scores.new_tensor(_CHARGE_SUPPORT)).sum(
                        dim=(1, 2)
                    )
                    - grouped_targets
                )
                .abs()
                .max()
                .detach()
                .cpu()
            ),
        },
    )


def _deterministic_rest_directions(count: int, *, device: Any, dtype: Any) -> Any:
    torch = require_torch()
    index = torch.arange(count, device=device, dtype=dtype)
    z = 1.0 - 2.0 * (index + 0.5) / max(count, 1)
    angle = index * (math.pi * (3.0 - math.sqrt(5.0)))
    radius = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
    return torch.stack((radius * torch.cos(angle), radius * torch.sin(angle), z), dim=-1)


def _boost_rest_to_lab(rest_four_vector: Any, parent_four_vector: Any, parent_mass: Any) -> Any:
    torch = require_torch()
    rest = torch.as_tensor(rest_four_vector)
    parent = torch.as_tensor(parent_four_vector, device=rest.device, dtype=rest.dtype)
    mass = torch.as_tensor(parent_mass, device=rest.device, dtype=rest.dtype)
    beta = parent[1:] / parent[0].clamp_min(1.0e-12)
    beta2 = beta.square().sum().clamp_max(1.0 - 1.0e-12)
    gamma = parent[0] / mass.clamp_min(1.0e-12)
    beta_dot = (rest[:, 1:] * beta[None, :]).sum(dim=-1)
    coefficient = torch.where(
        beta2 > 1.0e-16,
        (gamma - 1.0) * beta_dot / beta2.clamp_min(1.0e-16) + gamma * rest[:, 0],
        rest[:, 0],
    )
    energy = gamma * (rest[:, 0] + beta_dot)
    spatial = rest[:, 1:] + coefficient[:, None] * beta[None, :]
    return torch.cat((energy[:, None], spatial), dim=-1)


def _stable_nonnegative_sqrt(values: Any, *, epsilon: float = 1.0e-12) -> Any:
    torch = require_torch()
    values = torch.as_tensor(values)
    # Do not quantize small positive values to zero.  The phase-space solver
    # operates in float64 and legitimately encounters squared momenta below
    # 1e-12 for low-energy, highly boosted groups.  Dropping those energies
    # prevents the rest-frame children from closing to their parent and the
    # boost magnifies the discrepancy.  Replacing only the non-positive input
    # before sqrt keeps that branch finite without changing positive values.
    del epsilon  # Retained in the signature for compatibility with callers.
    positive = values > 0.0
    safe_values = torch.where(positive, values, torch.ones_like(values))
    rooted = torch.sqrt(safe_values)
    return torch.where(positive, rooted, torch.zeros_like(rooted))


def _invariant_mass_float64(four_vector: Any) -> Any:
    """Return invariant mass using one cancellation-resistant representation."""

    torch = require_torch()
    p4 = torch.as_tensor(four_vector)
    if p4.shape[-1] != 4:
        raise ValueError("invariant mass expects a four-vector ending in [4]")
    work = p4.to(torch.float64)
    return _stable_nonnegative_sqrt(
        work[..., 0].square() - work[..., 1:].square().sum(dim=-1)
    )


def project_n_body_phase_space(
    parent_four_vector: Any,
    raw_rest_spatial: Any,
    particle_masses: Any,
    energy_fraction_logits: Any,
    *,
    iterations: int = 64,
    near_massless_threshold: float = 1.0e-6,
) -> tuple[Any, Mapping[str, Any]]:
    """Project one local unordered set to exact parent four-momentum."""

    torch = require_torch()
    parent_input = torch.as_tensor(parent_four_vector)
    output_dtype = parent_input.dtype
    parent = parent_input.to(torch.float64)
    raw = torch.as_tensor(raw_rest_spatial, device=parent.device).to(torch.float64)
    masses = torch.as_tensor(particle_masses, device=parent.device).to(torch.float64)
    fractions_raw = torch.as_tensor(energy_fraction_logits, device=parent.device).to(torch.float64)
    if parent.shape != (4,) or raw.ndim != 2 or raw.shape[-1] != 3:
        raise ValueError("N-body projection expects parent [4] and raw momenta [N, 3]")
    count = int(raw.shape[0])
    if count <= 0 or masses.shape != (count,) or fractions_raw.shape != (count,):
        raise ValueError("N-body particle dimensions are inconsistent")
    parent_mass = _invariant_mass_float64(parent)
    feasibility_tolerance = 2.0e-5 * parent_mass.abs().clamp_min(1.0)
    minimum_mass = masses.min()
    mass_sum = masses.sum()
    if bool(minimum_mass < 0.0) or bool(
        mass_sum > parent_mass + feasibility_tolerance
    ):
        raise ValueError(
            "particle masses are infeasible for the parent invariant mass: "
            f"minimum_particle_mass={float(minimum_mass.detach().cpu()):.8e}, "
            f"particle_mass_sum={float(mass_sum.detach().cpu()):.8e}, "
            f"parent_mass={float(parent_mass.detach().cpu()):.8e}, "
            f"tolerance={float(feasibility_tolerance.detach().cpu()):.8e}, "
            f"particle_count={count}"
        )
    if count == 1:
        return parent[None, :].to(output_dtype), {
            "branch": "single_particle_exact_parent",
            "scale": 0.0,
            "closure_max_residual": 0.0,
        }
    if float(parent_mass.detach().cpu()) <= float(near_massless_threshold):
        if bool((masses > 1.0e-9).any()):
            raise ValueError(
                "near-massless parent cannot carry massive rendered particles: "
                f"parent_mass={float(parent_mass.detach().cpu()):.8e}, "
                f"maximum_particle_mass={float(masses.max().detach().cpu()):.8e}, "
                f"particle_mass_sum={float(masses.sum().detach().cpu()):.8e}, "
                f"threshold={float(near_massless_threshold):.8e}, "
                f"particle_count={count}"
            )
        fractions = fractions_raw.softmax(dim=0)
        result = fractions[:, None] * parent[None, :]
        residual = (result.sum(dim=0) - parent).abs().max()
        return result.to(output_dtype), {
            "branch": "massless_collinear",
            "scale": 0.0,
            "closure_max_residual": float(residual.detach().cpu()),
        }
    centered = raw - raw.mean(dim=0, keepdim=True)
    if float(centered.square().sum().detach().cpu()) < 1.0e-18:
        centered = _deterministic_rest_directions(
            count, device=raw.device, dtype=raw.dtype
        )
        centered = centered - centered.mean(dim=0, keepdim=True)
    lower = parent_mass.new_zeros(())
    mean_norm = centered.norm(dim=-1).mean().clamp_min(1.0e-8)
    upper = parent_mass / mean_norm
    for _ in range(16):
        upper_energy = _stable_nonnegative_sqrt(
            (upper * centered).square().sum(dim=-1) + masses.square()
        ).sum()
        upper = torch.where(upper_energy < parent_mass, 2.0 * upper, upper)
    for _ in range(int(iterations)):
        middle = 0.5 * (lower + upper)
        energy = _stable_nonnegative_sqrt(
            (middle * centered).square().sum(dim=-1) + masses.square()
        ).sum()
        lower = torch.where(energy < parent_mass, middle, lower)
        upper = torch.where(energy < parent_mass, upper, middle)
    scale = 0.5 * (lower + upper)
    spatial = scale * centered
    rest_energy = _stable_nonnegative_sqrt(
        spatial.square().sum(dim=-1) + masses.square()
    )
    rest = torch.cat((rest_energy[:, None], spatial), dim=-1)
    lab = _boost_rest_to_lab(rest, parent, parent_mass)
    raw_residual_vector = parent - lab.sum(dim=0)
    raw_residual = raw_residual_vector.abs().max()

    # A large Lorentz boost can amplify float64 summation error enough to miss
    # the component-space closure tolerance even though the rest-frame
    # solution is valid.  Complete the numerical projection by assigning the
    # residual to the highest-energy child.  This is differentiable with
    # respect to the selected branch and minimizes its relative perturbation.
    anchor = lab[:, 0].abs().argmax()
    anchor_weight = torch.nn.functional.one_hot(anchor, num_classes=count).to(
        device=lab.device, dtype=lab.dtype
    )
    lab = lab + anchor_weight[:, None] * raw_residual_vector[None, :]
    residual = (lab.sum(dim=0) - parent).abs().max()
    rendered_masses = _invariant_mass_float64(lab)
    mass_shell_residual = (rendered_masses - masses).abs().max()
    parent_energy_scale = max(float(parent[0].abs().detach().cpu()), 1.0)
    closure_tolerance = 5.0e-6 * parent_energy_scale
    mass_shell_tolerance = 2.0e-5 * parent_energy_scale
    if (
        float(residual.detach().cpu()) > closure_tolerance
        or float(mass_shell_residual.detach().cpu()) > mass_shell_tolerance
    ):
        raise RuntimeError(
            "N-body phase-space projection failed guarded closure: "
            f"corrected_residual={float(residual.detach().cpu()):.8e}, "
            f"raw_residual={float(raw_residual.detach().cpu()):.8e}, "
            f"mass_shell_residual={float(mass_shell_residual.detach().cpu()):.8e}, "
            f"parent_energy={float(parent[0].detach().cpu()):.8e}, "
            f"parent_mass={float(parent_mass.detach().cpu()):.8e}, "
            f"particle_count={count}"
        )
    return lab.to(output_dtype), {
        "branch": "massive_rest_frame",
        "scale": float(scale.detach().cpu()),
        "closure_max_residual": float(residual.detach().cpu()),
        "raw_closure_max_residual": float(raw_residual.detach().cpu()),
        "mass_shell_max_residual": float(mass_shell_residual.detach().cpu()),
        "closure_anchor_index": int(anchor.detach().cpu()),
    }


def _boost_rest_to_lab_batched(
    rest_four_vector: Any, parent_four_vector: Any, parent_mass: Any
) -> Any:
    torch = require_torch()
    rest = torch.as_tensor(rest_four_vector)
    parent = torch.as_tensor(
        parent_four_vector, device=rest.device, dtype=rest.dtype
    )
    mass = torch.as_tensor(parent_mass, device=rest.device, dtype=rest.dtype)
    beta = parent[:, 1:] / parent[:, :1].clamp_min(1.0e-12)
    beta2 = beta.square().sum(dim=-1).clamp_max(1.0 - 1.0e-12)
    gamma = parent[:, 0] / mass.clamp_min(1.0e-12)
    beta_dot = (rest[..., 1:] * beta[:, None, :]).sum(dim=-1)
    coefficient = torch.where(
        beta2[:, None] > 1.0e-16,
        ((gamma - 1.0)[:, None] * beta_dot / beta2[:, None])
        + gamma[:, None] * rest[..., 0],
        rest[..., 0],
    )
    boosted_spatial = rest[..., 1:] + coefficient[..., None] * beta[:, None, :]
    boosted_energy = gamma[:, None] * (rest[..., 0] + beta_dot)
    return torch.cat((boosted_energy[..., None], boosted_spatial), dim=-1)


def _allocate_local_particle_masses_batched(
    parent_mass: Any,
    local_minimum: Any,
    mass_logits: Any,
    group_fraction: Any,
    slot_mask: Any,
    *,
    phase_space_mass_epsilon: float,
    near_massless_threshold: float,
) -> Any:
    torch = require_torch()
    parent = torch.as_tensor(parent_mass).to(torch.float64)
    minimum = torch.as_tensor(
        local_minimum, device=parent.device
    ).to(torch.float64)
    logits = torch.as_tensor(
        mass_logits, device=parent.device
    ).to(torch.float64)
    fraction = torch.as_tensor(
        group_fraction, device=parent.device
    ).to(torch.float64)
    valid = torch.as_tensor(slot_mask, device=parent.device).bool()
    minimum = minimum * valid
    available = (
        (1.0 - float(phase_space_mass_epsilon)) * parent
        - minimum.sum(dim=-1)
    ).clamp_min(0.0)
    masked_logits = logits.masked_fill(~valid, float("-inf"))
    weights = masked_logits.softmax(dim=-1)
    optional = available[:, None] * fraction[:, None] * weights
    optional = torch.where(
        (parent > float(near_massless_threshold))[:, None] & valid,
        optional,
        torch.zeros_like(optional),
    )
    result = minimum + optional
    single = valid.sum(dim=-1) == 1
    result = torch.where(
        single[:, None] & valid, parent[:, None], result
    )
    return result * valid


def project_batched_n_body_phase_space(
    parent_four_vector: Any,
    raw_rest_spatial: Any,
    particle_masses: Any,
    energy_fraction_logits: Any,
    slot_mask: Any,
    local_slot_indices: Any,
    *,
    iterations: int = 64,
    near_massless_threshold: float = 1.0e-6,
) -> tuple[Any, Mapping[str, Any]]:
    """Vectorized exact projection for every active terminal group in a batch."""

    torch = require_torch()
    parent_input = torch.as_tensor(parent_four_vector)
    output_dtype = parent_input.dtype
    parent = parent_input.to(torch.float64)
    raw = torch.as_tensor(
        raw_rest_spatial, device=parent.device
    ).to(torch.float64)
    masses = torch.as_tensor(
        particle_masses, device=parent.device
    ).to(torch.float64)
    fractions_raw = torch.as_tensor(
        energy_fraction_logits, device=parent.device
    ).to(torch.float64)
    valid = torch.as_tensor(slot_mask, device=parent.device).bool()
    local_index = torch.as_tensor(
        local_slot_indices, device=parent.device
    ).to(torch.float64)
    if (
        parent.ndim != 2
        or parent.shape[-1] != 4
        or raw.shape != (*valid.shape, 3)
        or masses.shape != valid.shape
        or fractions_raw.shape != valid.shape
    ):
        raise ValueError("batched N-body particle dimensions are inconsistent")
    counts = valid.sum(dim=-1)
    if bool((counts <= 0).any()):
        raise ValueError("batched N-body projection received an empty group")
    parent_mass = _invariant_mass_float64(parent)
    tolerance = 2.0e-5 * parent_mass.abs().clamp_min(1.0)
    minimum = masses.masked_fill(~valid, float("inf")).min(dim=-1).values
    mass_sum = (masses * valid).sum(dim=-1)
    if bool((minimum < 0.0).any()) or bool(
        (mass_sum > parent_mass + tolerance).any()
    ):
        raise ValueError("batched particle masses are infeasible for a parent")

    result = torch.zeros(
        (*valid.shape, 4), dtype=torch.float64, device=parent.device
    )
    single = counts == 1
    massless = (
        (parent_mass <= float(near_massless_threshold)) & ~single
    )
    massive = ~(single | massless)
    if bool((massless[:, None] & valid & (masses > 1.0e-9)).any()):
        raise ValueError("near-massless parent cannot carry massive rendered particles")

    if bool(single.any()):
        rows = torch.nonzero(single, as_tuple=False).flatten()
        columns = valid[rows].to(torch.long).argmax(dim=-1)
        result[rows, columns] = parent[rows]

    if bool(massless.any()):
        rows = torch.nonzero(massless, as_tuple=False).flatten()
        local_valid = valid[rows]
        logits = fractions_raw[rows].masked_fill(~local_valid, float("-inf"))
        fractions = logits.softmax(dim=-1)
        result[rows] = fractions[..., None] * parent[rows, None, :]

    scale_all = parent_mass.new_zeros(parent_mass.shape)
    if bool(massive.any()):
        rows = torch.nonzero(massive, as_tuple=False).flatten()
        local_valid = valid[rows]
        local_raw = raw[rows]
        local_masses = masses[rows]
        local_counts = counts[rows].to(torch.float64)
        mean = (local_raw * local_valid[..., None]).sum(dim=1) / local_counts[:, None]
        centered = (local_raw - mean[:, None, :]) * local_valid[..., None]
        degenerate = centered.square().sum(dim=(1, 2)) < 1.0e-18
        if bool(degenerate.any()):
            local_position = local_index[rows].clamp_min(0.0)
            z = 1.0 - 2.0 * (
                local_position + 0.5
            ) / local_counts[:, None]
            angle = local_position * (math.pi * (3.0 - math.sqrt(5.0)))
            radius = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
            deterministic = torch.stack(
                (radius * torch.cos(angle), radius * torch.sin(angle), z),
                dim=-1,
            ) * local_valid[..., None]
            deterministic = deterministic - (
                deterministic.sum(dim=1) / local_counts[:, None]
            )[:, None, :]
            centered = torch.where(
                degenerate[:, None, None], deterministic, centered
            )
        local_parent_mass = parent_mass[rows]
        lower = torch.zeros_like(local_parent_mass)
        mean_norm = (
            centered.norm(dim=-1).sum(dim=-1) / local_counts
        ).clamp_min(1.0e-8)
        upper = local_parent_mass / mean_norm
        for _ in range(16):
            upper_energy = (
                _stable_nonnegative_sqrt(
                    (upper[:, None, None] * centered).square().sum(dim=-1)
                    + local_masses.square()
                )
                * local_valid
            ).sum(dim=-1)
            upper = torch.where(
                upper_energy < local_parent_mass, 2.0 * upper, upper
            )
        for _ in range(int(iterations)):
            middle = 0.5 * (lower + upper)
            energy = (
                _stable_nonnegative_sqrt(
                    (middle[:, None, None] * centered).square().sum(dim=-1)
                    + local_masses.square()
                )
                * local_valid
            ).sum(dim=-1)
            below = energy < local_parent_mass
            lower = torch.where(below, middle, lower)
            upper = torch.where(below, upper, middle)
        scale = 0.5 * (lower + upper)
        spatial = scale[:, None, None] * centered
        rest_energy = _stable_nonnegative_sqrt(
            spatial.square().sum(dim=-1) + local_masses.square()
        ) * local_valid
        rest = torch.cat((rest_energy[..., None], spatial), dim=-1)
        lab = _boost_rest_to_lab_batched(
            rest, parent[rows], local_parent_mass
        ) * local_valid[..., None]
        residual_vector = parent[rows] - lab.sum(dim=1)
        anchor = lab[..., 0].abs().masked_fill(~local_valid, -1.0).argmax(dim=-1)
        anchor_weight = torch.nn.functional.one_hot(
            anchor, num_classes=valid.shape[1]
        ).to(lab.dtype)
        lab = lab + anchor_weight[..., None] * residual_vector[:, None, :]
        result[rows] = lab
        scale_all[rows] = scale

    residual = (result.sum(dim=1) - parent).abs().max(dim=-1).values
    rendered_mass = _invariant_mass_float64(result)
    shell_residual = (
        (rendered_mass - masses).abs().masked_fill(~valid, 0.0).max(dim=-1).values
    )
    energy_scale = parent[:, 0].abs().clamp_min(1.0)
    if bool(
        (
            (residual > 5.0e-6 * energy_scale)
            | (shell_residual > 2.0e-5 * energy_scale)
        ).any()
    ):
        raise RuntimeError("batched N-body phase-space projection failed guarded closure")
    branch_counts = {
        "single_particle_exact_parent": int(single.sum().detach().cpu()),
        "massless_collinear": int(massless.sum().detach().cpu()),
        "massive_rest_frame": int(massive.sum().detach().cpu()),
    }
    return result.to(output_dtype), {
        "branch_counts": {
            key: value for key, value in branch_counts.items() if value
        },
        "maximum_scale": float(scale_all.max().detach().cpu()),
        "closure_max_residual": float(residual.max().detach().cpu()),
        "mass_shell_max_residual": float(shell_residual.max().detach().cpu()),
    }


def _gather_slots(values: Any, group_indices: Any) -> Any:
    torch = require_torch()
    tensor = torch.as_tensor(values)
    indices = torch.as_tensor(group_indices, device=tensor.device).clamp_min(0)
    suffix = tensor.shape[2:]
    gather_shape = (*indices.shape, *suffix)
    view_shape = (*indices.shape, *((1,) * len(suffix)))
    expanded = indices.reshape(view_shape).expand(gather_shape)
    return tensor.gather(1, expanded)


def _ancestor_path(output: RecursiveHierarchyOutput, group_indices: Any, slot_mask: Any) -> Any:
    torch = require_torch()
    indices = torch.as_tensor(group_indices).clamp_min(0)
    if not output.levels:
        root_indices = torch.zeros_like(indices)
        root_hidden = _gather_slots(output.root_frontier.hidden, root_indices)
        return (
            root_hidden.unsqueeze(2)
            * torch.as_tensor(slot_mask).unsqueeze(-1).unsqueeze(-1)
        )
    paths = []
    for depth in reversed(range(len(output.levels))):
        frontier = output.levels[depth].next_frontier
        paths.append(_gather_slots(frontier.hidden, indices))
        indices = _gather_slots(frontier.parent_indices[:, :, None], indices).squeeze(-1).clamp_min(0)
    return torch.stack(tuple(reversed(paths)), dim=2) * torch.as_tensor(slot_mask).unsqueeze(-1).unsqueeze(-1)


class _RendererBlock(_ModuleBase):
    def __init__(self, config: ParticleRendererConfig) -> None:
        torch = require_torch()
        super().__init__()
        kwargs = {
            "embed_dim": config.d_model,
            "num_heads": config.num_heads,
            "dropout": config.attention_dropout,
            "batch_first": True,
        }
        self.config = config
        self.self_attention = torch.nn.MultiheadAttention(**kwargs)
        self.global_hlt_attention = torch.nn.MultiheadAttention(**kwargs)
        self.local_hlt_attention = torch.nn.MultiheadAttention(**kwargs)
        self.root_attention = torch.nn.MultiheadAttention(**kwargs)
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(config.d_model) for _ in range(5)])
        self.gates = torch.nn.Parameter(torch.full((5,), 1.0e-3))
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(config.d_model, config.ffn_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.ffn_dim, config.d_model),
            torch.nn.Dropout(config.dropout),
        )

    def _repeat_heads(self, mask: Any) -> Any:
        return mask.repeat_interleave(self.config.num_heads, dim=0)

    def forward(
        self,
        slots: Any,
        slot_mask: Any,
        group_indices: Any,
        hlt: Any,
        hlt_mask: Any,
        local_hlt_bias: Any,
        root_memory: Any,
    ) -> Any:
        torch = require_torch()
        valid_queries = slot_mask[:, :, None]
        same_group = group_indices[:, :, None] == group_indices[:, None, :]
        allowed = same_group & slot_mask[:, None, :]
        allowed = allowed | (~slot_mask[:, :, None] & slot_mask[:, None, :])
        self_mask = torch.zeros(allowed.shape, dtype=slots.dtype, device=slots.device)
        self_mask = self_mask.masked_fill(~allowed, float("-inf"))
        slot_padding_bias = torch.zeros(
            slot_mask.shape, dtype=slots.dtype, device=slots.device
        ).masked_fill(~slot_mask, float("-inf"))
        hlt_padding_bias = torch.zeros(
            hlt_mask.shape, dtype=slots.dtype, device=slots.device
        ).masked_fill(~hlt_mask, float("-inf"))
        normalized = self.norms[0](slots)
        update, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=slot_padding_bias,
            attn_mask=self._repeat_heads(self_mask),
            need_weights=False,
        )
        slots = slots + self.gates[0] * update * valid_queries
        update, _ = self.global_hlt_attention(
            self.norms[1](slots), hlt, hlt, key_padding_mask=~hlt_mask, need_weights=False
        )
        slots = slots + self.gates[1] * update * valid_queries
        update, _ = self.local_hlt_attention(
            self.norms[2](slots),
            hlt,
            hlt,
            key_padding_mask=hlt_padding_bias,
            attn_mask=self._repeat_heads(local_hlt_bias),
            need_weights=False,
        )
        slots = slots + self.gates[2] * update * valid_queries
        update, _ = self.root_attention(
            self.norms[3](slots), root_memory, root_memory, need_weights=False
        )
        slots = slots + self.gates[3] * update * valid_queries
        slots = slots + self.gates[4] * self.ffn(self.norms[4](slots)) * valid_queries
        return slots * valid_queries


class ConstrainedParticleRenderer(_ModuleBase):
    """Performance-first local renderer with exact discrete and P4 accounting."""

    def __init__(self, config: ParticleRendererConfig | Mapping[str, Any] | None = None) -> None:
        torch = require_torch()
        super().__init__()
        if config is None:
            resolved = ParticleRendererConfig()
        elif isinstance(config, ParticleRendererConfig):
            resolved = config
        else:
            resolved = ParticleRendererConfig(**dict(config))
        self.config = resolved
        self.hlt_projections = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.LayerNorm(input_dim),
                    torch.nn.Linear(input_dim, resolved.d_model),
                )
                for input_dim in resolved.hlt_input_dims
            ]
        )
        self.hlt_source_embedding = torch.nn.Parameter(
            torch.zeros(len(resolved.hlt_input_dims), resolved.d_model)
        )
        torch.nn.init.trunc_normal_(self.hlt_source_embedding, std=0.02)
        self.group_ledger_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(len(ROOT_FEATURE_NAMES)),
            torch.nn.Linear(len(ROOT_FEATURE_NAMES), resolved.d_model),
        )
        self.group_support_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(ABPH_GROUP_SUPPORT_DIM),
            torch.nn.Linear(ABPH_GROUP_SUPPORT_DIM, resolved.d_model),
        )
        self.root_semantic_projection = torch.nn.Linear(
            resolved.root_semantic_dim, resolved.d_model
        )
        self.latent_projection = torch.nn.Linear(resolved.latent_dim, resolved.d_model)
        self.slot_embedding = torch.nn.Embedding(ABPH_MAX_PARTICLES, resolved.d_model)
        self.path_depth_embedding = torch.nn.Embedding(5, resolved.d_model)
        self.path_query = torch.nn.Linear(resolved.d_model, resolved.d_model)
        self.path_key = torch.nn.Linear(resolved.d_model, resolved.d_model)
        self.path_value = torch.nn.Linear(resolved.d_model, resolved.d_model)
        self.blocks = torch.nn.ModuleList(
            [_RendererBlock(resolved) for _ in range(resolved.blocks)]
        )
        self.type_head = torch.nn.Linear(resolved.d_model, len(ABPH_PID_CATEGORIES))
        self.charge_head = torch.nn.Linear(resolved.d_model, 3)
        self.rest_spatial_head = torch.nn.Linear(resolved.d_model, 3)
        self.energy_fraction_head = torch.nn.Linear(resolved.d_model, 1)
        self.mass_weight_head = torch.nn.Linear(resolved.d_model, 1)
        self.group_mass_fraction_head = torch.nn.Linear(resolved.d_model, 1)
        self.track_head = torch.nn.Linear(resolved.d_model, 4)
        self.uncertainty_head = torch.nn.Linear(resolved.d_model, 1)
        self.support_score_head = torch.nn.Linear(resolved.d_model, 1)

    def _project_hlt(self, embeddings: Any, mask: Any) -> Any:
        torch = require_torch()
        sources = (embeddings,) if not isinstance(embeddings, (tuple, list)) else tuple(embeddings)
        if len(sources) != len(self.hlt_projections):
            raise ValueError("renderer HLT embedding source count mismatch")
        projected = []
        prefix = None
        for index, (values, projection, expected_dim) in enumerate(
            zip(sources, self.hlt_projections, self.config.hlt_input_dims)
        ):
            tensor = torch.as_tensor(values)
            if tensor.ndim != 3 or tensor.shape[-1] != expected_dim:
                raise ValueError(f"renderer HLT source {index} has the wrong shape")
            prefix = tensor.shape[:2] if prefix is None else prefix
            if tensor.shape[:2] != prefix:
                raise ValueError("all renderer HLT sources must share particle axes")
            projected.append(projection(tensor) + self.hlt_source_embedding[index])
        hlt_mask = torch.as_tensor(mask, device=projected[0].device).bool()
        if hlt_mask.shape != prefix or not bool(hlt_mask.any(dim=1).all()):
            raise ValueError("renderer HLT mask is invalid")
        return torch.stack(projected, dim=0).mean(dim=0) * hlt_mask.unsqueeze(-1)

    def _local_hlt_bias(self, slot_support: Any, hlt_support: Any, slot_mask: Any) -> Any:
        torch = require_torch()
        hlt_values = torch.as_tensor(hlt_support, device=slot_support.device)
        if hlt_values.shape[-1] != len(ABPH_HLT_SUPPORT_FEATURE_NAMES):
            raise ValueError("renderer HLT support has the wrong feature dimension")
        delta_eta = slot_support[:, :, None, 0] - hlt_values[:, None, :, 0]
        delta_phi = wrap_phi_tensor(
            slot_support[:, :, None, 1] - hlt_values[:, None, :, 1]
        )
        scale = slot_support[:, :, None, 8].abs().clamp_min(0.05)
        bias = -self.config.maximum_local_attention_bias * torch.tanh(
            (delta_eta.square() + delta_phi.square()) / scale.square()
        )
        return bias * slot_mask.unsqueeze(-1)

    def _slot_hidden(
        self,
        hierarchy: RecursiveHierarchyOutput,
        layout: ParticleSlotLayout,
        root_semantic_tokens: Any,
        hlt_particle_embeddings: Any,
        hlt_particle_mask: Any,
        hlt_support_features: Any,
        hypothesis_latent: Any,
    ) -> tuple[Any, Any, Any, Any]:
        torch = require_torch()
        final = hierarchy.final_frontier
        hlt = self._project_hlt(hlt_particle_embeddings, hlt_particle_mask)
        hlt_mask = torch.as_tensor(hlt_particle_mask, device=hlt.device).bool()
        group_hidden = _gather_slots(final.hidden, layout.group_indices)
        group_ledger = _gather_slots(final.ledger, layout.group_indices)
        group_support = _gather_slots(final.support, layout.group_indices)
        local_indices = layout.local_slot_indices.clamp_min(0)
        latent = torch.as_tensor(
            hypothesis_latent, device=hlt.device, dtype=hlt.dtype
        )
        if latent.shape != (hlt.shape[0], self.config.latent_dim):
            raise ValueError("renderer hypothesis latent has the wrong shape")
        slots = (
            group_hidden
            + self.group_ledger_projection(group_ledger)
            + self.group_support_projection(group_support)
            + self.slot_embedding(local_indices)
            + self.latent_projection(latent)[:, None, :]
        )
        path = _ancestor_path(hierarchy, layout.group_indices, layout.mask)
        path = path + self.path_depth_embedding.weight[
            None, None, : path.shape[2], :
        ]
        query = self.path_query(slots)[:, :, None, :]
        score = (query * self.path_key(path)).sum(dim=-1) / math.sqrt(self.config.d_model)
        path_weights = score.softmax(dim=-1)
        path_context = (path_weights.unsqueeze(-1) * self.path_value(path)).sum(dim=2)
        slots = (slots + path_context) * layout.mask.unsqueeze(-1)
        root_memory = self.root_semantic_projection(
            torch.as_tensor(root_semantic_tokens, device=hlt.device, dtype=hlt.dtype)
        )
        local_bias = self._local_hlt_bias(
            group_support, hlt_support_features, layout.mask
        )
        for block in self.blocks:
            slots = block(
                slots,
                layout.mask,
                layout.group_indices,
                hlt,
                hlt_mask,
                local_bias,
                root_memory,
            )
        return slots, group_ledger, group_support, path_weights

    def forward(
        self,
        hierarchy: RecursiveHierarchyOutput,
        root_semantic_tokens: Any,
        hlt_particle_embeddings: Any,
        hlt_particle_mask: Any,
        hlt_support_features: Any,
        hypothesis_latent: Any,
        hlt_axis_eta: Any,
        hlt_axis_phi: Any,
        *,
        hypothesis_index: int,
    ) -> RenderedParticleBatch:
        torch = require_torch()
        if hierarchy.mode != "rollout":
            raise ValueError("deployable particle rendering requires rollout hierarchy input")
        root_state = AccountingState.from_ledger(hierarchy.root_frontier.ledger[:, 0])
        layout = exact_particle_slot_layout(hierarchy.final_frontier, root_state)
        slots, group_ledger, group_support, path_weights = self._slot_hidden(
            hierarchy,
            layout,
            root_semantic_tokens,
            hlt_particle_embeddings,
            hlt_particle_mask,
            hlt_support_features,
            hypothesis_latent,
        )
        final = hierarchy.final_frontier
        group_type_counts = torch.stack(
            tuple(
                final.ledger[:, :, ROOT_FEATURE_INDEX[f"count_{name}"]]
                for name in ABPH_PID_CATEGORIES
            ),
            dim=-1,
        ).round().long()
        group_charge = final.ledger[:, :, ROOT_FEATURE_INDEX["integer_charge"]]
        type_allocation = allocate_particle_types(
            self.type_head(slots),
            layout,
            group_type_counts,
            temperature=self.config.type_temperature,
            sinkhorn_iterations=self.config.type_sinkhorn_iterations,
        )
        charge_allocation = allocate_particle_charges(
            self.charge_head(slots),
            layout,
            type_allocation.hard_indices,
            group_charge,
            temperature=self.config.charge_temperature,
        )
        mass_table = torch.tensor(
            _PID_MASSES, dtype=torch.float32, device=slots.device
        )
        minimum_mass = _type_conditioned_minimum_mass(type_allocation, mass_table)
        raw_spatial = self.rest_spatial_head(slots)
        energy_fraction_logits = self.energy_fraction_head(slots).squeeze(-1)
        mass_logits = self.mass_weight_head(slots).squeeze(-1)
        batch_rows, group_rows, slot_rows = _active_group_rows(
            layout, final.mask.shape[1]
        )
        grouped_parent = final.ledger[batch_rows, group_rows]
        grouped_parent_p4 = torch.stack(
            tuple(
                grouped_parent[:, ROOT_FEATURE_INDEX[name]]
                for name in ("energy", "px", "py", "pz")
            ),
            dim=-1,
        )
        grouped_parent_mass = _invariant_mass_float64(grouped_parent_p4)
        grouped_fraction = torch.sigmoid(
            self.group_mass_fraction_head(final.hidden[batch_rows, group_rows])
        ).squeeze(-1)
        grouped_mass = _allocate_local_particle_masses_batched(
            grouped_parent_mass,
            minimum_mass[batch_rows],
            mass_logits[batch_rows],
            grouped_fraction,
            slot_rows,
            phase_space_mass_epsilon=self.config.phase_space_mass_epsilon,
            near_massless_threshold=self.config.near_massless_threshold,
        )
        if self.config.exact_nbody_projection:
            grouped_p4, phase = project_batched_n_body_phase_space(
                grouped_parent_p4,
                raw_spatial[batch_rows],
                grouped_mass,
                energy_fraction_logits[batch_rows],
                slot_rows,
                layout.local_slot_indices[batch_rows],
                iterations=self.config.phase_space_iterations,
                near_massless_threshold=self.config.near_massless_threshold,
            )
            phase_branches = dict(phase["branch_counts"])
            maximum_local_residual = float(phase["closure_max_residual"])
        else:
            direction = torch.nn.functional.normalize(
                raw_spatial[batch_rows], dim=-1, eps=1.0e-8
            )
            energy = (
                torch.nn.functional.softplus(
                    energy_fraction_logits[batch_rows]
                )
                + grouped_mass
            ) * slot_rows
            momentum = _stable_nonnegative_sqrt(
                energy.square() - grouped_mass.square()
            ) * slot_rows
            grouped_p4 = torch.cat(
                (energy[..., None], direction * momentum[..., None]), dim=-1
            ) * slot_rows[..., None]
            phase_branches = {
                "unconstrained_no_nbody_projection": int(
                    slot_rows.shape[0]
                )
            }
            maximum_local_residual = float(
                (grouped_p4.sum(dim=1) - grouped_parent_p4)
                .abs()
                .max()
                .detach()
                .cpu()
            )
        # Publish every group in one indexed operation. Group slot masks are
        # disjoint by construction, so each particle slot is written exactly once.
        batch_grid = batch_rows[:, None].expand_as(slot_rows)
        slot_grid = torch.arange(
            layout.mask.shape[1], device=slots.device
        )[None, :].expand_as(slot_rows)
        four_vector = torch.zeros(
            (*layout.mask.shape, 4), dtype=torch.float32, device=slots.device
        )
        mass = torch.zeros(
            layout.mask.shape, dtype=torch.float32, device=slots.device
        )
        four_vector[batch_grid[slot_rows], slot_grid[slot_rows]] = grouped_p4[
            slot_rows
        ].to(four_vector.dtype)
        mass[batch_grid[slot_rows], slot_grid[slot_rows]] = grouped_mass[
            slot_rows
        ].to(mass.dtype)
        track_raw = self.track_head(slots)
        track = torch.stack(
            (
                track_raw[..., 0],
                torch.nn.functional.softplus(track_raw[..., 1]),
                track_raw[..., 2],
                torch.nn.functional.softplus(track_raw[..., 3]),
            ),
            dim=-1,
        ) * layout.mask.unsqueeze(-1)
        uncertainty = torch.nn.functional.softplus(
            self.uncertainty_head(slots).squeeze(-1)
        ) * layout.mask
        pt = torch.linalg.vector_norm(four_vector[..., 1:3], dim=-1)
        eta = torch.asinh(four_vector[..., 3] / pt.clamp_min(1.0e-8))
        phi = torch.atan2(four_vector[..., 2], four_vector[..., 1])
        axis_eta = torch.as_tensor(hlt_axis_eta, device=slots.device, dtype=slots.dtype)
        axis_phi = torch.as_tensor(hlt_axis_phi, device=slots.device, dtype=slots.dtype)
        if axis_eta.shape != (slots.shape[0],) or axis_phi.shape != (slots.shape[0],):
            raise ValueError("HLT axis tensors must have shape [B]")
        canonical = torch.cat(
            (
                pt.unsqueeze(-1),
                (eta - axis_eta[:, None]).unsqueeze(-1),
                wrap_phi_tensor(phi - axis_phi[:, None]).unsqueeze(-1),
                four_vector[..., :1],
                charge_allocation.straight_through_charges.unsqueeze(-1),
                type_allocation.soft_probabilities,
                track,
                torch.full_like(pt.unsqueeze(-1), -1.0),
                layout.group_indices.to(slots.dtype).unsqueeze(-1),
            ),
            dim=-1,
        ) * layout.mask.unsqueeze(-1)
        if canonical.shape[-1] != len(PARTICLE_TARGET_NAMES):
            raise RuntimeError("rendered canonical feature order does not match the target schema")
        group_uncertainty = _gather_slots(final.uncertainty[:, :, None], layout.group_indices).squeeze(-1)
        side_channels = torch.stack(
            (
                uncertainty,
                torch.full_like(uncertainty, float(hypothesis_index)),
                torch.exp(-group_uncertainty.clamp_min(0.0)),
                path_weights.max(dim=-1).values,
                torch.sigmoid(self.support_score_head(slots).squeeze(-1)),
                torch.ones_like(uncertainty),
            ),
            dim=-1,
        ) * layout.mask.unsqueeze(-1)
        root_p4 = root_state.four_vector
        rendered_p4 = four_vector.sum(dim=1)
        root_residual = (rendered_p4 - root_p4).abs()
        hard_type_totals = (type_allocation.hard_one_hot * layout.mask.unsqueeze(-1)).sum(dim=1).long()
        hard_charge_total = (charge_allocation.hard_charges * layout.mask).sum(dim=1)
        if not bool((hard_type_totals == root_state.type_counts).all()):
            raise RuntimeError("rendered particle types do not close to the root")
        if not bool((hard_charge_total == root_state.integer_charge).all()):
            raise RuntimeError("rendered particle charges do not close to the root")
        relative_scale = root_p4[:, 0].abs().clamp_min(1.0)
        root_relative = root_residual.max(dim=-1).values / relative_scale
        if (
            self.config.exact_nbody_projection
            and float(root_relative.max().detach().cpu()) > 2.0e-5
        ):
            raise RuntimeError("complete rendered pseudo jet does not close to the root four-vector")
        return RenderedParticleBatch(
            four_vector=four_vector,
            mass=mass,
            mask=layout.mask,
            group_indices=layout.group_indices,
            local_slot_indices=layout.local_slot_indices,
            soft_pid_probabilities=type_allocation.soft_probabilities,
            hard_pid_indices=type_allocation.hard_indices,
            hard_pid_one_hot=type_allocation.hard_one_hot,
            charges=charge_allocation.straight_through_charges,
            hard_charges=charge_allocation.hard_charges,
            track_features=track,
            uncertainty=uncertainty,
            canonical_features=canonical,
            side_channels=side_channels,
            slot_hidden=slots,
            diagnostics={
                "contract": ABPH_PARTICLE_RENDERER_CONTRACT,
                "offline_inputs_consumed": False,
                "hlt_only_deployment_inputs": True,
                "hypothesis_index": int(hypothesis_index),
                "slot_layout": layout.diagnostics,
                "type_allocation": type_allocation.diagnostics,
                "charge_allocation": charge_allocation.diagnostics,
                "phase_space_branches": phase_branches,
                "local_p4_maximum_residual": maximum_local_residual,
                "root_p4_maximum_absolute_residual": float(root_residual.max().detach().cpu()),
                "root_p4_maximum_relative_residual": float(root_relative.max().detach().cpu()),
                "exact_nbody_projection": bool(self.config.exact_nbody_projection),
                "local_matching_objective": bool(self.config.local_matching),
                "count_closes_exactly": bool(
                    (layout.mask.sum(dim=1) == root_state.constituent_count).all()
                ),
                "types_close_exactly": True,
                "charges_close_exactly": True,
                "group_local_self_attention": bool(hierarchy.levels),
                "renderer_grouping": (
                    "hierarchy_local" if hierarchy.levels else "single_global_root_set"
                ),
                "source_flag": "pseudo",
                "side_channel_names": list(ABPH_RENDERED_SIDE_CHANNEL_NAMES),
            },
        )


__all__ = [
    "ABPH_PARTICLE_RENDERER_CONTRACT",
    "ABPH_RENDERED_SIDE_CHANNEL_NAMES",
    "ChargeAllocation",
    "ConstrainedParticleRenderer",
    "ParticleRendererConfig",
    "ParticleSlotLayout",
    "RenderedParticleBatch",
    "TypeAllocation",
    "allocate_particle_charges",
    "allocate_particle_types",
    "exact_particle_slot_layout",
    "project_batched_n_body_phase_space",
    "project_n_body_phase_space",
]
