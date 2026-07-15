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
    batch, groups = mask.shape
    slot_mask = torch.zeros((batch, ABPH_MAX_PARTICLES), dtype=torch.bool, device=mask.device)
    group_indices = torch.full(
        (batch, ABPH_MAX_PARTICLES), -1, dtype=torch.long, device=mask.device
    )
    local_indices = torch.full_like(group_indices, -1)
    for batch_index in range(batch):
        cursor = 0
        for group_index in torch.nonzero(mask[batch_index], as_tuple=False).flatten().tolist():
            count = int(counts[batch_index, group_index].detach().cpu())
            if cursor + count > ABPH_MAX_PARTICLES:
                raise RuntimeError("compiled microgroup counts exceed the 128-slot contract")
            slot_mask[batch_index, cursor : cursor + count] = True
            group_indices[batch_index, cursor : cursor + count] = int(group_index)
            local_indices[batch_index, cursor : cursor + count] = torch.arange(
                count, device=mask.device
            )
            cursor += count
        root_count = int(root_state.constituent_count[batch_index].detach().cpu())
        if cursor != root_count:
            raise RuntimeError(
                f"final microgroup counts {cursor} do not close to root count {root_count}"
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


def _minimum_cost_square_assignment(cost: Any) -> Any:
    """Exact O(N^3) Hungarian assignment without an optional SciPy dependency."""

    torch = require_torch()
    values = torch.as_tensor(cost)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("minimum-cost assignment requires a square cost matrix")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("minimum-cost assignment received nonfinite costs")
    n = int(values.shape[0])
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=values.device)
    detached = values.detach().to(torch.float64).cpu().tolist()
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
    return torch.tensor(assignment, dtype=torch.long, device=values.device)


def _quota_sinkhorn(logits: Any, quotas: Any, *, temperature: float, iterations: int) -> Any:
    torch = require_torch()
    scores = torch.as_tensor(logits)
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
    soft = torch.zeros_like(scores)
    hard_indices = torch.full(layout.mask.shape, -1, dtype=torch.long, device=scores.device)
    for batch_index in range(scores.shape[0]):
        groups = torch.unique(layout.group_indices[batch_index, layout.mask[batch_index]])
        for group_index_tensor in groups:
            group_index = int(group_index_tensor)
            slots = torch.nonzero(
                layout.mask[batch_index]
                & (layout.group_indices[batch_index] == group_index),
                as_tuple=False,
            ).flatten()
            quotas = counts[batch_index, group_index]
            probabilities = _quota_sinkhorn(
                scores[batch_index, slots],
                quotas,
                temperature=temperature,
                iterations=sinkhorn_iterations,
            )
            soft[batch_index, slots] = probabilities
            expanded_types = torch.repeat_interleave(
                torch.arange(len(ABPH_PID_CATEGORIES), device=scores.device), quotas
            )
            assignment = _minimum_cost_square_assignment(
                -scores[batch_index, slots][:, expanded_types]
            )
            hard_indices[batch_index, slots] = expanded_types[assignment]
    hard = torch.nn.functional.one_hot(
        hard_indices.clamp_min(0), num_classes=len(ABPH_PID_CATEGORIES)
    ).to(scores.dtype) * layout.mask.unsqueeze(-1)
    straight_through = hard + soft - soft.detach()
    hard_counts = torch.zeros_like(counts)
    for group_index in range(counts.shape[1]):
        member = layout.mask & (layout.group_indices == group_index)
        hard_counts[:, group_index] = (hard * member.unsqueeze(-1)).sum(dim=1).long()
    expected_counts = torch.zeros_like(counts, dtype=soft.dtype)
    for group_index in range(counts.shape[1]):
        member = layout.mask & (layout.group_indices == group_index)
        expected_counts[:, group_index] = (soft * member.unsqueeze(-1)).sum(dim=1)
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


def _minimum_cost_charge_sequence(logits: Any, allowed: Any, target_charge: int) -> Any:
    torch = require_torch()
    scores = torch.as_tensor(logits)
    permitted = torch.as_tensor(allowed, device=scores.device).bool()
    n_slots = int(scores.shape[0])
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    detached = scores.detach().to(torch.float64).cpu()
    permitted_cpu = permitted.cpu()
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
    return torch.tensor(states[int(target_charge)][1], dtype=torch.long, device=scores.device)


def _charge_soft_probabilities(
    logits: Any,
    allowed: Any,
    target_charge: int,
    *,
    temperature: float,
) -> Any:
    torch = require_torch()
    scores = torch.as_tensor(logits)
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
    soft = torch.zeros_like(scores)
    hard = torch.zeros(layout.mask.shape, dtype=torch.long, device=scores.device)
    for batch_index in range(scores.shape[0]):
        groups = torch.unique(layout.group_indices[batch_index, layout.mask[batch_index]])
        for group_index_tensor in groups:
            group_index = int(group_index_tensor)
            slots = torch.nonzero(
                layout.mask[batch_index]
                & (layout.group_indices[batch_index] == group_index),
                as_tuple=False,
            ).flatten()
            target = int(targets[batch_index, group_index].detach().cpu())
            local_allowed = allowed_all[batch_index, slots]
            soft[batch_index, slots] = _charge_soft_probabilities(
                scores[batch_index, slots],
                local_allowed,
                target,
                temperature=temperature,
            )
            hard[batch_index, slots] = _minimum_cost_charge_sequence(
                scores[batch_index, slots], local_allowed, target
            )
    expected = (soft * scores.new_tensor(_CHARGE_SUPPORT)).sum(dim=-1)
    straight_through = hard.to(scores.dtype) + expected - expected.detach()
    hard_group_charge = torch.zeros_like(targets)
    for group_index in range(targets.shape[1]):
        member = layout.mask & (layout.group_indices == group_index)
        hard_group_charge[:, group_index] = (hard * member).sum(dim=1)
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
                max(
                    (
                        (expected * (layout.group_indices == group_index) * layout.mask).sum(dim=1)
                        - targets[:, group_index]
                    ).abs().max().detach().cpu()
                    for group_index in range(targets.shape[1])
                )
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
    mass_squared = parent[0].square() - parent[1:].square().sum()
    parent_mass = torch.sqrt(mass_squared.clamp_min(0.0))
    feasibility_tolerance = 2.0e-5 * parent_mass.abs().clamp_min(1.0)
    if bool((masses < 0.0).any()) or bool(
        masses.sum() > parent_mass + feasibility_tolerance
    ):
        raise ValueError("particle masses are infeasible for the parent invariant mass")
    if count == 1:
        return parent[None, :].to(output_dtype), {
            "branch": "single_particle_exact_parent",
            "scale": 0.0,
            "closure_max_residual": 0.0,
        }
    if float(parent_mass.detach().cpu()) <= float(near_massless_threshold):
        if bool((masses > 1.0e-9).any()):
            raise ValueError("near-massless parent cannot carry massive rendered particles")
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
        upper_energy = torch.sqrt((upper * centered).square().sum(dim=-1) + masses.square()).sum()
        upper = torch.where(upper_energy < parent_mass, 2.0 * upper, upper)
    for _ in range(int(iterations)):
        middle = 0.5 * (lower + upper)
        energy = torch.sqrt((middle * centered).square().sum(dim=-1) + masses.square()).sum()
        lower = torch.where(energy < parent_mass, middle, lower)
        upper = torch.where(energy < parent_mass, upper, middle)
    scale = 0.5 * (lower + upper)
    spatial = scale * centered
    rest_energy = torch.sqrt(spatial.square().sum(dim=-1) + masses.square())
    rest = torch.cat((rest_energy[:, None], spatial), dim=-1)
    lab = _boost_rest_to_lab(rest, parent, parent_mass)
    residual = (lab.sum(dim=0) - parent).abs().max()
    if float(residual.detach().cpu()) > 5.0e-6 * max(float(parent[0].abs().detach().cpu()), 1.0):
        raise RuntimeError("N-body phase-space projection failed parent closure")
    return lab.to(output_dtype), {
        "branch": "massive_rest_frame",
        "scale": float(scale.detach().cpu()),
        "closure_max_residual": float(residual.detach().cpu()),
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
        path = path + self.path_depth_embedding.weight[None, None, :, :]
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
        mass_table = slots.new_tensor(_PID_MASSES)
        minimum_mass = type_allocation.straight_through_one_hot @ mass_table
        mass = torch.zeros(layout.mask.shape, dtype=slots.dtype, device=slots.device)
        four_vector = torch.zeros(
            (*layout.mask.shape, 4), dtype=slots.dtype, device=slots.device
        )
        raw_spatial = self.rest_spatial_head(slots)
        energy_fraction_logits = self.energy_fraction_head(slots).squeeze(-1)
        mass_logits = self.mass_weight_head(slots).squeeze(-1)
        phase_branches: dict[str, int] = {}
        maximum_local_residual = 0.0
        for batch_index in range(slots.shape[0]):
            groups = torch.unique(layout.group_indices[batch_index, layout.mask[batch_index]])
            for group_index_tensor in groups:
                group_index = int(group_index_tensor)
                member = torch.nonzero(
                    layout.mask[batch_index]
                    & (layout.group_indices[batch_index] == group_index),
                    as_tuple=False,
                ).flatten()
                parent = final.ledger[batch_index, group_index]
                parent_p4 = torch.stack(
                    tuple(parent[ROOT_FEATURE_INDEX[name]] for name in ("energy", "px", "py", "pz"))
                )
                parent_mass = torch.sqrt(
                    (parent_p4[0].square() - parent_p4[1:].square().sum()).clamp_min(0.0)
                )
                local_minimum = minimum_mass[batch_index, member]
                if int(member.numel()) == 1:
                    local_mass = parent_mass[None]
                else:
                    available = (
                        (1.0 - self.config.phase_space_mass_epsilon) * parent_mass
                        - local_minimum.sum()
                    ).clamp_min(0.0)
                    group_fraction = torch.sigmoid(
                        self.group_mass_fraction_head(final.hidden[batch_index, group_index])
                    ).squeeze(-1)
                    local_mass = local_minimum + (
                        available
                        * group_fraction
                        * mass_logits[batch_index, member].softmax(dim=0)
                    )
                local_p4, phase = project_n_body_phase_space(
                    parent_p4,
                    raw_spatial[batch_index, member],
                    local_mass,
                    energy_fraction_logits[batch_index, member],
                    iterations=self.config.phase_space_iterations,
                    near_massless_threshold=self.config.near_massless_threshold,
                )
                four_vector[batch_index, member] = local_p4
                mass[batch_index, member] = local_mass
                phase_branches[phase["branch"]] = phase_branches.get(phase["branch"], 0) + 1
                maximum_local_residual = max(
                    maximum_local_residual, float(phase["closure_max_residual"])
                )
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
        if float(root_relative.max().detach().cpu()) > 2.0e-5:
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
                "count_closes_exactly": bool(
                    (layout.mask.sum(dim=1) == root_state.constituent_count).all()
                ),
                "types_close_exactly": True,
                "charges_close_exactly": True,
                "group_local_self_attention": True,
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
    "project_n_body_phase_space",
]
