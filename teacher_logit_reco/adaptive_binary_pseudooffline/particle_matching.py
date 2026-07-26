"""Group-local Hungarian and unbalanced-OT losses for rendered particles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .hierarchy_alignment import RendererTargetMap
from .particle_renderer import (
    RenderedParticleBatch,
    _minimum_cost_square_assignment,
    _minimum_cost_square_assignment_rows,
)
from .schemas import ABPH_MAX_PARTICLES, ABPH_PID_CATEGORIES
from .targets import PARTICLE_TARGET_NAMES
from .root_transforms import wrap_phi_tensor


ABPH_PARTICLE_MATCHING_CONTRACT = "adaptive_binary_pseudooffline_particle_matching_v1"
_PARTICLE_INDEX = {name: index for index, name in enumerate(PARTICLE_TARGET_NAMES)}


@dataclass(frozen=True)
class ParticleMatchingConfig:
    log_pt_weight: float = 1.0
    angular_weight: float = 1.0
    four_vector_weight: float = 0.75
    pid_weight: float = 0.50
    charge_weight: float = 0.25
    track_weight: float = 0.20
    uncertainty_weight: float = 0.10
    sinkhorn_epsilon: float = 0.08
    sinkhorn_iterations: int = 50
    null_sink_cost: float = 1.0
    ordinary_tolerance: float = 1.0e-5

    def __post_init__(self) -> None:
        for name in (
            "log_pt_weight",
            "angular_weight",
            "four_vector_weight",
            "pid_weight",
            "charge_weight",
            "track_weight",
            "uncertainty_weight",
            "sinkhorn_epsilon",
            "null_sink_cost",
            "ordinary_tolerance",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if float(self.sinkhorn_epsilon) <= 0.0 or int(self.sinkhorn_iterations) <= 0:
            raise ValueError("Sinkhorn epsilon and iterations must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalParticleAssignment:
    batch_index: int
    group_index: int
    method: str
    predicted_slot_indices: Any
    target_particle_indices: Any
    transport: Any
    source_to_null: Any
    null_to_target: Any
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class ParticleMatchingLossOutput:
    total: Any
    real_particle_loss: Any
    null_sink_penalty: Any
    null_source_penalty: Any
    component_losses: Mapping[str, Any]
    assignments: tuple[LocalParticleAssignment, ...]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class ParticleAuxiliaryLossWeights:
    jet_four_vector: float = 1.0
    radial_profile: float = 0.50
    leading_pt: float = 0.50
    energy_correlation: float = 0.25
    n_subjettiness: float = 0.25
    track_summary: float = 0.20
    particle_energy_distance: float = 0.25

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class ParticleAuxiliaryLossOutput:
    total: Any
    components: Mapping[str, Any]
    rendered_observables: Mapping[str, Any]
    target_observables: Mapping[str, Any]
    diagnostics: Mapping[str, Any]


def _target_columns(targets: Any) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    torch = require_torch()
    values = torch.as_tensor(targets)
    pt = values[..., _PARTICLE_INDEX["pt"]]
    eta = values[..., _PARTICLE_INDEX["eta_hlt_relative"]]
    phi = values[..., _PARTICLE_INDEX["phi_hlt_relative"]]
    energy = values[..., _PARTICLE_INDEX["energy"]]
    charge = values[..., _PARTICLE_INDEX["charge"]]
    pid = torch.stack(
        tuple(values[..., _PARTICLE_INDEX[f"pid_{name}"]] for name in ABPH_PID_CATEGORIES),
        dim=-1,
    )
    track = torch.stack(
        tuple(
            values[..., _PARTICLE_INDEX[name]]
            for name in ("d0_value", "d0_error", "dz_value", "dz_error")
        ),
        dim=-1,
    )
    return pt, eta, phi, energy, charge, pid, track


def pairwise_particle_cost(
    rendered: RenderedParticleBatch,
    batch_index: int,
    predicted_indices: Any,
    target_rows: Any,
    config: ParticleMatchingConfig | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Return the complete local pair cost and named component matrices."""

    torch = require_torch()
    resolved = config or ParticleMatchingConfig()
    pred = rendered.canonical_features[batch_index, predicted_indices]
    target = torch.as_tensor(target_rows, device=pred.device, dtype=pred.dtype)
    target_pt, target_eta, target_phi, target_energy, target_charge, target_pid, target_track = (
        _target_columns(target)
    )
    pred_pt = pred[:, _PARTICLE_INDEX["pt"]]
    pred_eta = pred[:, _PARTICLE_INDEX["eta_hlt_relative"]]
    pred_phi = pred[:, _PARTICLE_INDEX["phi_hlt_relative"]]
    pred_energy = pred[:, _PARTICLE_INDEX["energy"]]
    pred_charge = rendered.charges[batch_index, predicted_indices]
    pred_pid = rendered.soft_pid_probabilities[batch_index, predicted_indices]
    pred_track = rendered.track_features[batch_index, predicted_indices]
    log_pt = (
        torch.log(pred_pt.clamp_min(1.0e-6))[:, None]
        - torch.log(target_pt.clamp_min(1.0e-6))[None, :]
    ).square()
    delta_eta = pred_eta[:, None] - target_eta[None, :]
    delta_phi = wrap_phi_tensor(pred_phi[:, None] - target_phi[None, :])
    angular = delta_eta.square() + delta_phi.square()
    target_scale = target_energy.abs().clamp_min(1.0)
    pred_proxy = torch.stack(
        (
            pred_energy,
            pred_pt * torch.cos(pred_phi),
            pred_pt * torch.sin(pred_phi),
            pred_pt * torch.sinh(pred_eta),
        ),
        dim=-1,
    )
    target_proxy = torch.stack(
        (
            target_energy,
            target_pt * torch.cos(target_phi),
            target_pt * torch.sin(target_phi),
            target_pt * torch.sinh(target_eta),
        ),
        dim=-1,
    )
    four_vector = (
        (pred_proxy[:, None, :] - target_proxy[None, :, :])
        / target_scale[None, :, None]
    ).square().mean(dim=-1)
    pid = -(target_pid[None, :, :] * pred_pid[:, None, :].clamp_min(1.0e-8).log()).sum(dim=-1)
    charge = (pred_charge[:, None] - target_charge[None, :]).square()
    track_scale = target_track.abs().mean(dim=0).clamp_min(1.0)
    track = (
        (pred_track[:, None, :] - target_track[None, :, :]) / track_scale[None, None, :]
    ).square().mean(dim=-1)
    uncertainty = rendered.uncertainty[batch_index, predicted_indices].clamp_min(1.0e-4)
    kinematic = log_pt + angular + four_vector
    uncertainty_normalized = kinematic / uncertainty[:, None] + uncertainty[:, None].log()
    components = {
        "log_pt": log_pt,
        "angular": angular,
        "four_vector": four_vector,
        "pid": pid,
        "charge": charge,
        "track": track,
        "uncertainty_normalized": uncertainty_normalized,
    }
    total = (
        resolved.log_pt_weight * log_pt
        + resolved.angular_weight * angular
        + resolved.four_vector_weight * four_vector
        + resolved.pid_weight * pid
        + resolved.charge_weight * charge
        + resolved.track_weight * track
        + resolved.uncertainty_weight * uncertainty_normalized
    )
    return total, components


def _batched_pairwise_particle_cost(
    rendered: RenderedParticleBatch,
    batch_indices: Any,
    predicted_indices: Any,
    target_rows: Any,
    config: ParticleMatchingConfig,
) -> tuple[Any, Mapping[str, Any]]:
    """Vectorized pair costs for groups with identical local cardinalities."""

    torch = require_torch()
    batch = torch.as_tensor(
        batch_indices, device=rendered.four_vector.device
    ).long()
    predicted = torch.as_tensor(
        predicted_indices, device=rendered.four_vector.device
    ).long()
    target = torch.as_tensor(
        target_rows,
        device=rendered.four_vector.device,
        dtype=rendered.four_vector.dtype,
    )
    pred = rendered.canonical_features[batch[:, None], predicted]
    target_pt, target_eta, target_phi, target_energy, target_charge, target_pid, target_track = (
        _target_columns(target)
    )
    pred_pt = pred[..., _PARTICLE_INDEX["pt"]]
    pred_eta = pred[..., _PARTICLE_INDEX["eta_hlt_relative"]]
    pred_phi = pred[..., _PARTICLE_INDEX["phi_hlt_relative"]]
    pred_energy = pred[..., _PARTICLE_INDEX["energy"]]
    pred_charge = rendered.charges[batch[:, None], predicted]
    pred_pid = rendered.soft_pid_probabilities[batch[:, None], predicted]
    pred_track = rendered.track_features[batch[:, None], predicted]
    log_pt = (
        torch.log(pred_pt.clamp_min(1.0e-6))[:, :, None]
        - torch.log(target_pt.clamp_min(1.0e-6))[:, None, :]
    ).square()
    delta_eta = pred_eta[:, :, None] - target_eta[:, None, :]
    delta_phi = wrap_phi_tensor(pred_phi[:, :, None] - target_phi[:, None, :])
    angular = delta_eta.square() + delta_phi.square()
    target_scale = target_energy.abs().clamp_min(1.0)
    pred_proxy = torch.stack(
        (
            pred_energy,
            pred_pt * torch.cos(pred_phi),
            pred_pt * torch.sin(pred_phi),
            pred_pt * torch.sinh(pred_eta),
        ),
        dim=-1,
    )
    target_proxy = torch.stack(
        (
            target_energy,
            target_pt * torch.cos(target_phi),
            target_pt * torch.sin(target_phi),
            target_pt * torch.sinh(target_eta),
        ),
        dim=-1,
    )
    four_vector = (
        (pred_proxy[:, :, None, :] - target_proxy[:, None, :, :])
        / target_scale[:, None, :, None]
    ).square().mean(dim=-1)
    pid = -(
        target_pid[:, None, :, :]
        * pred_pid[:, :, None, :].clamp_min(1.0e-8).log()
    ).sum(dim=-1)
    charge = (pred_charge[:, :, None] - target_charge[:, None, :]).square()
    track_scale = target_track.abs().mean(dim=1).clamp_min(1.0)
    track = (
        (pred_track[:, :, None, :] - target_track[:, None, :, :])
        / track_scale[:, None, None, :]
    ).square().mean(dim=-1)
    uncertainty = rendered.uncertainty[
        batch[:, None], predicted
    ].clamp_min(1.0e-4)
    kinematic = log_pt + angular + four_vector
    uncertainty_normalized = (
        kinematic / uncertainty[:, :, None]
        + uncertainty[:, :, None].log()
    )
    components = {
        "log_pt": log_pt,
        "angular": angular,
        "four_vector": four_vector,
        "pid": pid,
        "charge": charge,
        "track": track,
        "uncertainty_normalized": uncertainty_normalized,
    }
    total = (
        config.log_pt_weight * log_pt
        + config.angular_weight * angular
        + config.four_vector_weight * four_vector
        + config.pid_weight * pid
        + config.charge_weight * charge
        + config.track_weight * track
        + config.uncertainty_weight * uncertainty_normalized
    )
    return total, components


def _log_sinkhorn_with_nulls(
    cost: Any,
    source_mass: Any,
    target_mass: Any,
    *,
    epsilon: float,
    iterations: int,
    null_cost: float,
) -> tuple[Any, Any, Any]:
    torch = require_torch()
    values = torch.as_tensor(cost)
    source = torch.as_tensor(source_mass, device=values.device, dtype=values.dtype)
    target = torch.as_tensor(target_mass, device=values.device, dtype=values.dtype)
    source_total = source.sum()
    target_total = target.sum()
    augmented_source = torch.cat((source, target_total[None]))
    augmented_target = torch.cat((target, source_total[None]))
    augmented_cost = torch.full(
        (values.shape[0] + 1, values.shape[1] + 1),
        float(null_cost),
        dtype=values.dtype,
        device=values.device,
    )
    augmented_cost[:-1, :-1] = values
    augmented_cost[-1, -1] = 0.0
    log_kernel = -augmented_cost / float(epsilon)
    log_source = augmented_source.clamp_min(1.0e-12).log()
    log_target = augmented_target.clamp_min(1.0e-12).log()
    u = torch.zeros_like(log_source)
    v = torch.zeros_like(log_target)
    for _ in range(int(iterations)):
        u = log_source - torch.logsumexp(log_kernel + v[None, :], dim=1)
        v = log_target - torch.logsumexp(log_kernel + u[:, None], dim=0)
    transport = torch.exp(log_kernel + u[:, None] + v[None, :])
    return transport[:-1, :-1], transport[:-1, -1], transport[-1, :-1]


def _batched_log_sinkhorn_with_nulls(
    cost: Any,
    source_mass: Any,
    target_mass: Any,
    *,
    epsilon: float,
    iterations: int,
    null_cost: float,
) -> tuple[Any, Any, Any]:
    torch = require_torch()
    values = torch.as_tensor(cost)
    source = torch.as_tensor(
        source_mass, device=values.device, dtype=values.dtype
    )
    target = torch.as_tensor(
        target_mass, device=values.device, dtype=values.dtype
    )
    source_total = source.sum(dim=-1)
    target_total = target.sum(dim=-1)
    augmented_source = torch.cat((source, target_total[:, None]), dim=-1)
    augmented_target = torch.cat((target, source_total[:, None]), dim=-1)
    augmented_cost = torch.full(
        (values.shape[0], values.shape[1] + 1, values.shape[2] + 1),
        float(null_cost),
        dtype=values.dtype,
        device=values.device,
    )
    augmented_cost[:, :-1, :-1] = values
    augmented_cost[:, -1, -1] = 0.0
    log_kernel = -augmented_cost / float(epsilon)
    log_source = augmented_source.clamp_min(1.0e-12).log()
    log_target = augmented_target.clamp_min(1.0e-12).log()
    u = torch.zeros_like(log_source)
    v = torch.zeros_like(log_target)
    for _ in range(int(iterations)):
        u = log_source - torch.logsumexp(
            log_kernel + v[:, None, :], dim=2
        )
        v = log_target - torch.logsumexp(
            log_kernel + u[:, :, None], dim=1
        )
    transport = torch.exp(log_kernel + u[:, :, None] + v[:, None, :])
    return transport[:, :-1, :-1], transport[:, :-1, -1], transport[:, -1, :-1]


def _weighted_component_mean(component: Any, transport: Any, normalization: Any) -> Any:
    return (component * transport).sum() / normalization.clamp_min(1.0)


def compute_local_particle_matching_loss(
    rendered: RenderedParticleBatch,
    target_particle_features: Any,
    target_particle_mask: Any,
    target_map: RendererTargetMap,
    *,
    config: ParticleMatchingConfig | None = None,
    return_assignments: bool = True,
) -> ParticleMatchingLossOutput:
    """Match every local set only to its rollout-aligned local particle measure."""

    torch = require_torch()
    resolved = config or ParticleMatchingConfig()
    targets = torch.as_tensor(
        target_particle_features,
        device=rendered.four_vector.device,
        dtype=rendered.four_vector.dtype,
    )
    target_valid = torch.as_tensor(
        target_particle_mask, device=rendered.four_vector.device
    ).bool()
    if targets.shape != (
        rendered.mask.shape[0],
        ABPH_MAX_PARTICLES,
        len(PARTICLE_TARGET_NAMES),
    ):
        raise ValueError("particle target features have the wrong shape")
    if target_valid.shape != rendered.mask.shape:
        raise ValueError("particle target mask has the wrong shape")
    if target_map.target_particle_weights.shape[:2] != target_map.terminal_mask.shape:
        raise ValueError("renderer target-map shapes are inconsistent")
    group_count = target_map.terminal_mask.shape[1]
    group_axis = torch.arange(
        group_count, device=rendered.mask.device
    ).view(1, -1, 1)
    predicted_membership = (
        rendered.mask[:, None, :]
        & (rendered.group_indices[:, None, :] == group_axis)
    )
    active_pairs = torch.nonzero(
        predicted_membership.any(dim=-1), as_tuple=False
    )
    batch_rows = active_pairs[:, 0]
    group_rows = active_pairs[:, 1]
    predicted_masks = predicted_membership[batch_rows, group_rows]
    if not bool(target_map.terminal_mask[batch_rows, group_rows].all()):
        raise ValueError("rendered group does not have a terminal RendererTargetMap")
    grouped_weights = target_map.target_particle_weights[
        batch_rows, group_rows
    ].to(rendered.four_vector.dtype)
    target_masks = target_valid[batch_rows] & (grouped_weights > 1.0e-10)
    predicted_counts = predicted_masks.sum(dim=-1)
    target_counts = target_masks.sum(dim=-1)
    weight_residual = (
        (grouped_weights - 1.0).abs() * target_masks
    ).max(dim=-1).values
    grouped_target_null = target_map.target_null_particle_weight[batch_rows]
    target_null_residual = (
        grouped_target_null.abs() * target_masks
    ).max(dim=-1).values
    ordinary = (
        (predicted_counts == target_counts)
        & (target_counts > 0)
        & (
            target_map.predicted_null_mass[batch_rows, group_rows]
            <= resolved.ordinary_tolerance
        )
        & (weight_residual <= resolved.ordinary_tolerance)
        & (target_null_residual <= resolved.ordinary_tolerance)
    )
    metadata = torch.stack(
        (ordinary.long(), predicted_counts, target_counts), dim=-1
    ).detach().cpu()
    unique_metadata = torch.unique(metadata, dim=0).tolist()

    assignments: list[LocalParticleAssignment] = []
    real_terms: list[Any] = []
    sink_terms: list[Any] = []
    source_terms: list[Any] = []
    component_totals: dict[str, list[Any]] = {}
    method_counts = {"hungarian": 0, "unbalanced_ot": 0}
    for ordinary_value, predicted_count, target_count in unique_metadata:
        selected_cpu = (
            (metadata[:, 0] == ordinary_value)
            & (metadata[:, 1] == predicted_count)
            & (metadata[:, 2] == target_count)
        )
        selected = torch.nonzero(
            selected_cpu.to(rendered.mask.device), as_tuple=False
        ).flatten()
        local_batch = batch_rows[selected]
        local_group = group_rows[selected]
        local_predicted_mask = predicted_masks[selected]
        predicted_indices = torch.nonzero(
            local_predicted_mask, as_tuple=False
        )[:, 1].reshape(-1, int(predicted_count))
        local_target_mask = target_masks[selected]
        target_indices = (
            torch.empty(
                (selected.numel(), 0),
                dtype=torch.long,
                device=rendered.mask.device,
            )
            if int(target_count) == 0
            else torch.nonzero(
                local_target_mask, as_tuple=False
            )[:, 1].reshape(-1, int(target_count))
        )
        if int(target_count) == 0:
            zero = (
                rendered.four_vector[
                    local_batch[:, None], predicted_indices
                ].sum(dim=(1, 2))
                * 0.0
            )
            to_null = torch.ones(
                (selected.numel(), int(predicted_count)),
                device=rendered.four_vector.device,
                dtype=rendered.four_vector.dtype,
            )
            from_null = torch.empty(
                (selected.numel(), 0),
                device=rendered.four_vector.device,
                dtype=rendered.four_vector.dtype,
            )
            transport = torch.empty(
                (selected.numel(), int(predicted_count), 0),
                device=rendered.four_vector.device,
                dtype=rendered.four_vector.dtype,
            )
            real_terms.append(zero)
            sink_terms.append(zero + float(resolved.null_sink_cost))
            source_terms.append(zero)
            method = "unbalanced_ot"
        else:
            target_rows = targets[
                local_batch[:, None], target_indices
            ]
            target_weights = grouped_weights[selected].gather(
                1, target_indices
            )
            cost, components = _batched_pairwise_particle_cost(
                rendered,
                local_batch,
                predicted_indices,
                target_rows,
                resolved,
            )
            if bool(ordinary_value):
                host_cost = cost.detach().to(torch.float64).cpu().tolist()
                host_assignment = [
                    _minimum_cost_square_assignment_rows(rows)
                    for rows in host_cost
                ]
                columns = torch.tensor(
                    host_assignment, dtype=torch.long, device=cost.device
                )
                transport = torch.zeros_like(cost)
                row_index = torch.arange(
                    int(predicted_count), device=cost.device
                ).view(1, -1).expand(selected.numel(), -1)
                batch_index = torch.arange(
                    selected.numel(), device=cost.device
                ).view(-1, 1).expand_as(row_index)
                transport[batch_index, row_index, columns] = 1.0
                to_null = torch.zeros(
                    (selected.numel(), int(predicted_count)),
                    device=cost.device,
                    dtype=cost.dtype,
                )
                from_null = torch.zeros(
                    (selected.numel(), int(target_count)),
                    device=cost.device,
                    dtype=cost.dtype,
                )
                method = "hungarian"
            else:
                transport, to_null, from_null = (
                    _batched_log_sinkhorn_with_nulls(
                        cost,
                        torch.ones(
                            (selected.numel(), int(predicted_count)),
                            device=cost.device,
                            dtype=cost.dtype,
                        ),
                        target_weights,
                        epsilon=resolved.sinkhorn_epsilon,
                        iterations=resolved.sinkhorn_iterations,
                        null_cost=resolved.null_sink_cost,
                    )
                )
                method = "unbalanced_ot"
            normalization = target_weights.sum(dim=-1).clamp_min(1.0)
            real_terms.append(
                (transport * cost).sum(dim=(1, 2)) / normalization
            )
            sink_terms.append(
                resolved.null_sink_cost
                * to_null.sum(dim=1)
                / normalization
            )
            source_terms.append(
                resolved.null_sink_cost
                * from_null.sum(dim=1)
                / normalization
            )
            for name, values in components.items():
                component_totals.setdefault(name, []).append(
                    (values * transport).sum(dim=(1, 2)) / normalization
                )
        method_counts[method] += int(selected.numel())
        if return_assignments:
            target_weight_rows = (
                torch.zeros(selected.numel(), device=rendered.mask.device)
                if int(target_count) == 0
                else grouped_weights[selected].gather(
                    1, target_indices
                ).sum(dim=1)
            )
            diagnostic_rows = torch.stack(
                (
                    target_weight_rows,
                    to_null.sum(dim=1),
                    from_null.sum(dim=1),
                ),
                dim=-1,
            ).detach().cpu()
            local_batch_cpu = local_batch.detach().cpu().tolist()
            local_group_cpu = local_group.detach().cpu().tolist()
            for row_index, (batch_index, group_index) in enumerate(
                zip(local_batch_cpu, local_group_cpu)
            ):
                assignments.append(
                    LocalParticleAssignment(
                        batch_index=int(batch_index),
                        group_index=int(group_index),
                        method=method,
                        predicted_slot_indices=predicted_indices[row_index],
                        target_particle_indices=target_indices[row_index],
                        transport=transport[row_index],
                        source_to_null=to_null[row_index],
                        null_to_target=from_null[row_index],
                        diagnostics={
                            "group_local_only": True,
                            "target_weight": float(diagnostic_rows[row_index, 0]),
                            "predicted_slots": int(predicted_count),
                            "source_to_null": float(diagnostic_rows[row_index, 1]),
                            "null_to_target": float(diagnostic_rows[row_index, 2]),
                            "all_null_target": int(target_count) == 0,
                        },
                    )
                )
    if not real_terms:
        raise ValueError("particle matching received no active local groups")
    real_loss = torch.cat(real_terms).mean()
    null_sink = torch.cat(sink_terms).mean()
    null_source = torch.cat(source_terms).mean()
    total = real_loss + null_sink + null_source
    components_mean = {
        name: torch.cat(values).mean() for name, values in component_totals.items()
    }
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("nonfinite local particle matching loss")
    return ParticleMatchingLossOutput(
        total=total,
        real_particle_loss=real_loss,
        null_sink_penalty=null_sink,
        null_source_penalty=null_source,
        component_losses=components_mean,
        assignments=tuple(assignments),
        diagnostics={
            "contract": ABPH_PARTICLE_MATCHING_CONTRACT,
            "method_counts": method_counts,
            "group_count": int(active_pairs.shape[0]),
            "cross_group_assignments": 0,
            "matching_crosses_group_boundaries": False,
            "teacher_forced_topology_used": False,
            "weighted_targets_use_unbalanced_ot": True,
            "ordinary_targets_use_hungarian": True,
            "batched_by_local_cardinality": True,
            "assignment_records_materialized": bool(return_assignments),
            "target_null_particle_weight_total": float(
                target_map.target_null_particle_weight.sum().detach().cpu()
            ),
            "config": resolved.to_dict(),
        },
    )


def compute_global_particle_matching_loss(
    rendered: RenderedParticleBatch,
    target_particle_features: Any,
    target_particle_mask: Any,
    *,
    config: ParticleMatchingConfig | None = None,
) -> ParticleMatchingLossOutput:
    """D3 control: match complete predicted and offline sets without group locality."""

    torch = require_torch()
    resolved = config or ParticleMatchingConfig()
    targets = torch.as_tensor(
        target_particle_features,
        device=rendered.four_vector.device,
        dtype=rendered.four_vector.dtype,
    )
    target_valid = torch.as_tensor(
        target_particle_mask, device=rendered.four_vector.device
    ).bool()
    assignments = []
    real_terms = []
    sink_terms = []
    source_terms = []
    component_totals: dict[str, list[Any]] = {}
    methods = {"hungarian": 0, "unbalanced_ot": 0}
    for batch_index in range(rendered.mask.shape[0]):
        pred_indices = torch.nonzero(rendered.mask[batch_index], as_tuple=False).flatten()
        target_indices = torch.nonzero(target_valid[batch_index], as_tuple=False).flatten()
        if not int(pred_indices.numel()) or not int(target_indices.numel()):
            raise ValueError("global particle matching requires nonempty predicted and target sets")
        cost, components = pairwise_particle_cost(
            rendered,
            batch_index,
            pred_indices,
            targets[batch_index, target_indices],
            resolved,
        )
        if int(pred_indices.numel()) == int(target_indices.numel()):
            columns = _minimum_cost_square_assignment(cost)
            transport = torch.zeros_like(cost)
            transport[torch.arange(pred_indices.numel(), device=cost.device), columns] = 1.0
            to_null = torch.zeros(pred_indices.numel(), device=cost.device, dtype=cost.dtype)
            from_null = torch.zeros(target_indices.numel(), device=cost.device, dtype=cost.dtype)
            method = "hungarian"
        else:
            transport, to_null, from_null = _log_sinkhorn_with_nulls(
                cost,
                torch.ones(pred_indices.numel(), device=cost.device, dtype=cost.dtype),
                torch.ones(target_indices.numel(), device=cost.device, dtype=cost.dtype),
                epsilon=resolved.sinkhorn_epsilon,
                iterations=resolved.sinkhorn_iterations,
                null_cost=resolved.null_sink_cost,
            )
            method = "unbalanced_ot"
        normalization = float(max(int(target_indices.numel()), 1))
        real_terms.append((transport * cost).sum() / normalization)
        sink_terms.append(float(resolved.null_sink_cost) * to_null.sum() / normalization)
        source_terms.append(float(resolved.null_sink_cost) * from_null.sum() / normalization)
        for name, values in components.items():
            component_totals.setdefault(name, []).append(
                _weighted_component_mean(values, transport, cost.new_tensor(normalization))
            )
        methods[method] += 1
        assignments.append(
            LocalParticleAssignment(
                batch_index=batch_index,
                group_index=-1,
                method=method,
                predicted_slot_indices=pred_indices,
                target_particle_indices=target_indices,
                transport=transport,
                source_to_null=to_null,
                null_to_target=from_null,
                diagnostics={
                    "group_local_only": False,
                    "global_set_control": True,
                    "predicted_slots": int(pred_indices.numel()),
                    "target_particles": int(target_indices.numel()),
                },
            )
        )
    real_loss = torch.stack(real_terms).mean()
    null_sink = torch.stack(sink_terms).mean()
    null_source = torch.stack(source_terms).mean()
    total = real_loss + null_sink + null_source
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("nonfinite global particle matching loss")
    return ParticleMatchingLossOutput(
        total=total,
        real_particle_loss=real_loss,
        null_sink_penalty=null_sink,
        null_source_penalty=null_source,
        component_losses={
            name: torch.stack(values).mean() for name, values in component_totals.items()
        },
        assignments=tuple(assignments),
        diagnostics={
            "contract": ABPH_PARTICLE_MATCHING_CONTRACT,
            "method_counts": methods,
            "group_count": len(assignments),
            "cross_group_assignments": "allowed",
            "matching_crosses_group_boundaries": True,
            "teacher_forced_topology_used": False,
            "weighted_targets_use_unbalanced_ot": True,
            "ordinary_targets_use_hungarian": True,
            "global_set_control": True,
            "config": resolved.to_dict(),
        },
    )


def compute_particle_observables(rendered: RenderedParticleBatch) -> Mapping[str, Any]:
    """Differentiable jet/set summaries used for auxiliary losses and reports."""

    torch = require_torch()
    mask = rendered.mask
    p4 = rendered.four_vector * mask.unsqueeze(-1)
    pt = torch.linalg.vector_norm(p4[..., 1:3], dim=-1)
    eta = torch.asinh(p4[..., 3] / pt.clamp_min(1.0e-8))
    phi = torch.atan2(p4[..., 2], p4[..., 1])
    total_pt = pt.sum(dim=1).clamp_min(1.0e-8)
    weights = pt / total_pt[:, None]
    eta_axis = (weights * eta).sum(dim=1)
    sin_axis = (weights * torch.sin(phi)).sum(dim=1)
    cos_axis = (weights * torch.cos(phi)).sum(dim=1)
    phi_axis = torch.atan2(sin_axis, cos_axis)
    delta_eta = eta - eta_axis[:, None]
    delta_phi = wrap_phi_tensor(phi - phi_axis[:, None])
    radius = torch.sqrt(delta_eta.square() + delta_phi.square() + 1.0e-12)
    radial_edges = (0.05, 0.10, 0.20, 0.40, 0.80)
    radial_profile = []
    previous = 0.0
    for edge in radial_edges:
        radial_profile.append(
            (pt * mask * ((radius >= previous) & (radius < edge))).sum(dim=1) / total_pt
        )
        previous = edge
    radial_profile.append((pt * mask * (radius >= previous)).sum(dim=1) / total_pt)
    pair_delta_eta = eta[:, :, None] - eta[:, None, :]
    pair_delta_phi = wrap_phi_tensor(phi[:, :, None] - phi[:, None, :])
    pair_radius = torch.sqrt(
        pair_delta_eta.square() + pair_delta_phi.square() + 1.0e-12
    )
    pair_mask = mask[:, :, None] & mask[:, None, :]
    upper = torch.triu(
        torch.ones((ABPH_MAX_PARTICLES, ABPH_MAX_PARTICLES), dtype=torch.bool, device=mask.device),
        diagonal=1,
    )
    ecf2 = (
        pt[:, :, None] * pt[:, None, :] * pair_radius * pair_mask * upper
    ).sum(dim=(1, 2)) / total_pt.square()
    sorted_pt = torch.sort(pt.masked_fill(~mask, 0.0), dim=1, descending=True).values
    leading = sorted_pt[:, :4] / total_pt[:, None]
    tau1 = (pt * radius * mask).sum(dim=1) / total_pt
    top_indices = torch.topk(pt.masked_fill(~mask, -1.0), k=2, dim=1).indices
    axis_eta_two = eta.gather(1, top_indices)
    axis_phi_two = phi.gather(1, top_indices)
    distance_two = torch.stack(
        tuple(
            torch.sqrt(
                (eta - axis_eta_two[:, index : index + 1]).square()
                + wrap_phi_tensor(phi - axis_phi_two[:, index : index + 1]).square()
                + 1.0e-12
            )
            for index in range(2)
        ),
        dim=-1,
    ).min(dim=-1).values
    tau2 = (pt * distance_two * mask).sum(dim=1) / total_pt
    return {
        "jet_four_vector": p4.sum(dim=1),
        "constituent_count": mask.sum(dim=1),
        "type_counts": (rendered.hard_pid_one_hot * mask.unsqueeze(-1)).sum(dim=1),
        "integer_charge": (rendered.hard_charges * mask).sum(dim=1),
        "absolute_charge_sum": (rendered.hard_charges.abs() * mask).sum(dim=1),
        "radial_pt_profile": torch.stack(radial_profile, dim=-1),
        "leading_pt_fractions": leading,
        "energy_correlation_2": ecf2,
        "tau1": tau1,
        "tau2": tau2,
        "track_absolute_mean": (
            rendered.track_features.abs() * mask.unsqueeze(-1)
        ).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1),
    }


def _canonical_observables(features: Any, mask: Any) -> Mapping[str, Any]:
    torch = require_torch()
    values = torch.as_tensor(features)
    valid = torch.as_tensor(mask, device=values.device).bool()
    pt, eta, phi, energy, charge, pid, track = _target_columns(values)
    pt = pt * valid
    energy = energy * valid
    total_pt = pt.sum(dim=1).clamp_min(1.0e-8)
    weights = pt / total_pt[:, None]
    eta_axis = (weights * eta).sum(dim=1)
    phi_axis = torch.atan2(
        (weights * torch.sin(phi)).sum(dim=1),
        (weights * torch.cos(phi)).sum(dim=1),
    )
    delta_eta = eta - eta_axis[:, None]
    delta_phi = wrap_phi_tensor(phi - phi_axis[:, None])
    radius = torch.sqrt(delta_eta.square() + delta_phi.square() + 1.0e-12)
    radial = []
    lower = 0.0
    for upper in (0.05, 0.10, 0.20, 0.40, 0.80):
        radial.append(
            (pt * ((radius >= lower) & (radius < upper))).sum(dim=1) / total_pt
        )
        lower = upper
    radial.append((pt * (radius >= lower)).sum(dim=1) / total_pt)
    sorted_pt = torch.sort(pt, dim=1, descending=True).values
    leading = sorted_pt[:, :4] / total_pt[:, None]
    pair_delta_eta = eta[:, :, None] - eta[:, None, :]
    pair_delta_phi = wrap_phi_tensor(phi[:, :, None] - phi[:, None, :])
    pair_radius = torch.sqrt(
        pair_delta_eta.square() + pair_delta_phi.square() + 1.0e-12
    )
    pair_valid = valid[:, :, None] & valid[:, None, :]
    upper_triangle = torch.triu(
        torch.ones(
            (ABPH_MAX_PARTICLES, ABPH_MAX_PARTICLES),
            dtype=torch.bool,
            device=values.device,
        ),
        diagonal=1,
    )
    ecf2 = (
        pt[:, :, None] * pt[:, None, :] * pair_radius * pair_valid * upper_triangle
    ).sum(dim=(1, 2)) / total_pt.square()
    tau1 = (pt * radius).sum(dim=1) / total_pt
    top_indices = torch.topk(pt.masked_fill(~valid, -1.0), k=2, dim=1).indices
    axes_eta = eta.gather(1, top_indices)
    axes_phi = phi.gather(1, top_indices)
    distance_two = torch.stack(
        tuple(
            torch.sqrt(
                (eta - axes_eta[:, index : index + 1]).square()
                + wrap_phi_tensor(phi - axes_phi[:, index : index + 1]).square()
                + 1.0e-12
            )
            for index in range(2)
        ),
        dim=-1,
    ).min(dim=-1).values
    tau2 = (pt * distance_two).sum(dim=1) / total_pt
    proxy_p4 = torch.stack(
        (
            energy.sum(dim=1),
            (pt * torch.cos(phi)).sum(dim=1),
            (pt * torch.sin(phi)).sum(dim=1),
            (pt * torch.sinh(eta)).sum(dim=1),
        ),
        dim=-1,
    )
    return {
        "jet_four_vector_proxy": proxy_p4,
        "constituent_count": valid.sum(dim=1),
        "type_counts": (pid * valid.unsqueeze(-1)).sum(dim=1),
        "integer_charge": (charge * valid).sum(dim=1),
        "absolute_charge_sum": (charge.abs() * valid).sum(dim=1),
        "radial_pt_profile": torch.stack(radial, dim=-1),
        "leading_pt_fractions": leading,
        "energy_correlation_2": ecf2,
        "tau1": tau1,
        "tau2": tau2,
        "track_absolute_mean": (track.abs() * valid.unsqueeze(-1)).sum(dim=1)
        / valid.sum(dim=1).clamp_min(1).unsqueeze(-1),
    }


def _particle_energy_distance(
    predicted_features: Any,
    predicted_mask: Any,
    target_features: Any,
    target_mask: Any,
) -> Any:
    torch = require_torch()
    pred = torch.as_tensor(predicted_features)
    target = torch.as_tensor(target_features, device=pred.device, dtype=pred.dtype)
    losses = []
    for batch_index in range(pred.shape[0]):
        pred_rows = pred[batch_index, predicted_mask[batch_index]]
        target_rows = target[batch_index, target_mask[batch_index]]

        def coordinates(rows: Any) -> Any:
            return torch.stack(
                (
                    torch.log(rows[:, _PARTICLE_INDEX["pt"]].clamp_min(1.0e-6)),
                    rows[:, _PARTICLE_INDEX["eta_hlt_relative"]],
                    rows[:, _PARTICLE_INDEX["phi_hlt_relative"]],
                ),
                dim=-1,
            )

        pred_coordinates = coordinates(pred_rows)
        target_coordinates = coordinates(target_rows)

        def distance(first: Any, second: Any) -> Any:
            delta_log_pt = first[:, None, 0] - second[None, :, 0]
            delta_eta = first[:, None, 1] - second[None, :, 1]
            delta_phi = wrap_phi_tensor(first[:, None, 2] - second[None, :, 2])
            return torch.sqrt(
                delta_log_pt.square() + delta_eta.square() + delta_phi.square() + 1.0e-12
            )

        cross = distance(pred_coordinates, target_coordinates).mean()
        pred_self = distance(pred_coordinates, pred_coordinates).mean()
        target_self = distance(target_coordinates, target_coordinates).mean()
        losses.append(2.0 * cross - pred_self - target_self)
    return torch.stack(losses).mean()


def compute_particle_auxiliary_losses(
    rendered: RenderedParticleBatch,
    target_particle_features: Any,
    target_particle_mask: Any,
    *,
    weights: ParticleAuxiliaryLossWeights | None = None,
) -> ParticleAuxiliaryLossOutput:
    """Supervise post-projection jet/set summaries without relaxing hard closure."""

    torch = require_torch()
    resolved = weights or ParticleAuxiliaryLossWeights()
    target_features = torch.as_tensor(
        target_particle_features,
        device=rendered.canonical_features.device,
        dtype=rendered.canonical_features.dtype,
    )
    target_mask = torch.as_tensor(
        target_particle_mask, device=rendered.canonical_features.device
    ).bool()
    rendered_observables = _canonical_observables(
        rendered.canonical_features, rendered.mask
    )
    rendered_hard_observables = compute_particle_observables(rendered)
    for name in (
        "constituent_count",
        "type_counts",
        "integer_charge",
        "absolute_charge_sum",
    ):
        rendered_observables[name] = rendered_hard_observables[name]
    target_observables = _canonical_observables(target_features, target_mask)
    components = {
        "jet_four_vector": (
            (
                rendered_observables["jet_four_vector_proxy"]
                - target_observables["jet_four_vector_proxy"]
            )
            / target_observables["jet_four_vector_proxy"][:, :1].abs().clamp_min(1.0)
        ).square().mean(),
        "radial_profile": (
            rendered_observables["radial_pt_profile"]
            - target_observables["radial_pt_profile"]
        ).square().mean(),
        "leading_pt": (
            rendered_observables["leading_pt_fractions"]
            - target_observables["leading_pt_fractions"]
        ).square().mean(),
        "energy_correlation": (
            rendered_observables["energy_correlation_2"]
            - target_observables["energy_correlation_2"]
        ).square().mean(),
        "n_subjettiness": 0.5
        * (
            (
                rendered_observables["tau1"] - target_observables["tau1"]
            ).square().mean()
            + (
                rendered_observables["tau2"] - target_observables["tau2"]
            ).square().mean()
        ),
        "track_summary": (
            rendered_observables["track_absolute_mean"]
            - target_observables["track_absolute_mean"]
        ).square().mean(),
        "particle_energy_distance": _particle_energy_distance(
            rendered.canonical_features,
            rendered.mask,
            target_features,
            target_mask,
        ),
    }
    total = sum(
        float(getattr(resolved, name)) * value for name, value in components.items()
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("nonfinite particle auxiliary objective")
    return ParticleAuxiliaryLossOutput(
        total=total,
        components=components,
        rendered_observables=rendered_observables,
        target_observables=target_observables,
        diagnostics={
            "contract": ABPH_PARTICLE_MATCHING_CONTRACT,
            "hard_count_residual": (
                rendered_observables["constituent_count"]
                - target_observables["constituent_count"]
            ),
            "hard_type_count_residual": (
                rendered_observables["type_counts"]
                - target_observables["type_counts"]
            ),
            "hard_charge_residual": (
                rendered_observables["integer_charge"]
                - target_observables["integer_charge"]
            ),
            "hard_quantities_audited_not_soft_repaired": True,
            "weights": asdict(resolved),
        },
    )


__all__ = [
    "ABPH_PARTICLE_MATCHING_CONTRACT",
    "LocalParticleAssignment",
    "ParticleMatchingConfig",
    "ParticleMatchingLossOutput",
    "ParticleAuxiliaryLossOutput",
    "ParticleAuxiliaryLossWeights",
    "compute_particle_auxiliary_losses",
    "compute_local_particle_matching_loss",
    "compute_global_particle_matching_loss",
    "compute_particle_observables",
    "pairwise_particle_cost",
]
