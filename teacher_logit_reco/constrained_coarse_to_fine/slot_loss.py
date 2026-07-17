"""Cell-local set matching and accounting losses for particle slots."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import itertools
import math
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from .layout import ACCOUNTING_FIELD_NAMES, PID_CATEGORY_NAMES
from .slots import ParticleSlotDecoderOutput


PARTICLE_SLOT_LOSS_CONTRACT = "constrained_coarse_to_fine_particle_slot_loss_v1"


@dataclass(frozen=True)
class CellSlotTargets:
    """Offline particles grouped into terminal hierarchy cells."""

    local_coordinates: torch.Tensor
    pt: torch.Tensor
    energy: torch.Tensor
    pid_index: torch.Tensor
    charge_index: torch.Tensor
    mask: torch.Tensor
    terminal_cell_indices: torch.Tensor
    terminal_level: int

    @property
    def num_cells(self) -> int:
        return int(self.mask.shape[1])

    @property
    def max_targets_per_cell(self) -> int:
        return int(self.mask.shape[2])


@dataclass(frozen=True)
class ParticleSlotLossConfig:
    matching_mode: str = "sinkhorn"
    log_pt_weight: float = 1.0
    coordinate_weight: float = 4.0
    pid_weight: float = 0.75
    charge_weight: float = 0.20
    log_energy_weight: float = 0.35
    existence_weight: float = 0.50
    count_weight: float = 0.20
    pid_consistency_weight: float = 0.10
    accounting_consistency_weight: float = 0.50
    dust_weight: float = 0.10
    missing_target_weight: float = 1.0
    # The matching losses remain primary.  This bounded calibration term is
    # intentionally auxiliary so it cannot destabilize slot reconstruction.
    uncertainty_weight: float = 0.05
    reliability_weight: float = 0.05
    sinkhorn_temperature: float = 0.12
    sinkhorn_iterations: int = 30
    huber_beta: float = 0.20
    epsilon: float = 1.0e-8
    brute_force_limit: int = 8
    hungarian_workers: int = 1
    hungarian_executor: str = "serial"

    def __post_init__(self) -> None:
        if self.matching_mode not in {"ordered", "sinkhorn", "hungarian"}:
            raise ValueError("matching_mode must be ordered, sinkhorn, or hungarian")
        for name, value in asdict(self).items():
            if name in {"matching_mode", "sinkhorn_iterations", "brute_force_limit", "hungarian_workers", "hungarian_executor"}:
                continue
            if float(value) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if int(self.sinkhorn_iterations) <= 0 or int(self.brute_force_limit) <= 0:
            raise ValueError("iteration and fallback limits must be positive")
        if int(self.hungarian_workers) <= 0:
            raise ValueError("hungarian_workers must be positive")
        if str(self.hungarian_executor).strip().lower() not in {"serial", "thread"}:
            raise ValueError("hungarian_executor must be serial or thread")
        if float(self.sinkhorn_temperature) <= 0.0 or float(self.epsilon) <= 0.0:
            raise ValueError("sinkhorn_temperature and epsilon must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract"] = PARTICLE_SLOT_LOSS_CONTRACT
        return payload


@dataclass(frozen=True)
class SlotAssignment:
    batch_index: int
    view_index: int
    cell_index: int
    pred_indices: torch.Tensor
    target_indices: torch.Tensor
    method: str


@dataclass(frozen=True)
class ParticleSlotLossOutput:
    loss: torch.Tensor
    components: Mapping[str, torch.Tensor]
    metrics: Mapping[str, torch.Tensor]
    assignments: tuple[SlotAssignment, ...]

    def detached_summary(self) -> dict[str, Any]:
        return {
            "contract": PARTICLE_SLOT_LOSS_CONTRACT,
            "loss": float(self.loss.detach().cpu().item()),
            "components": {
                name: float(value.detach().cpu().item()) for name, value in self.components.items()
            },
            "metrics": {
                name: float(value.detach().cpu().item()) for name, value in self.metrics.items()
            },
            "assignment_count": len(self.assignments),
        }

def _wrap_phi(value: torch.Tensor) -> torch.Tensor:
    return torch.remainder(value + math.pi, 2.0 * math.pi) - math.pi


def prepare_cell_slot_targets(
    offline_tokens: torch.Tensor,
    offline_mask: torch.Tensor,
    final_cell_indices: torch.Tensor,
    reference_eta: torch.Tensor,
    reference_phi: torch.Tensor,
    *,
    terminal_level: int,
    coordinate_extent: float = 0.8,
) -> CellSlotTargets:
    """Group aligned offline particles by the requested terminal depth."""

    tokens = offline_tokens.to(dtype=torch.float32)
    if tokens.ndim != 3 or int(tokens.shape[-1]) != RAW_TOKEN_DIM:
        raise ValueError(f"offline_tokens must have shape [B, P, {RAW_TOKEN_DIM}]")
    batch, particles, _ = tokens.shape
    mask = offline_mask.to(device=tokens.device, dtype=torch.bool)
    if tuple(mask.shape) != (batch, particles):
        raise ValueError("offline_mask does not align with offline_tokens")
    fine_cells = final_cell_indices.to(device=tokens.device, dtype=torch.long)
    if tuple(fine_cells.shape) != (batch, particles):
        raise ValueError("final_cell_indices does not align with offline_tokens")
    if int(terminal_level) not in (1, 2, 3):
        raise ValueError("terminal_level must be 1, 2, or 3")
    reference_eta = torch.as_tensor(reference_eta, device=tokens.device, dtype=tokens.dtype)
    reference_phi = torch.as_tensor(reference_phi, device=tokens.device, dtype=tokens.dtype)
    if tuple(reference_eta.shape) != (batch,) or tuple(reference_phi.shape) != (batch,):
        raise ValueError("reference_eta/reference_phi must contain one value per jet")
    divisor = 4 ** (3 - int(terminal_level))
    cells = torch.div(fine_cells, divisor, rounding_mode="floor")
    num_cells = (8, 32, 128)[int(terminal_level) - 1]
    finite = torch.isfinite(tokens[..., :10]).all(dim=-1)
    valid = mask & finite & (tokens[..., 0] > 0.0) & (fine_cells >= 0)
    if bool(valid.any()):
        counts = torch.zeros(batch, num_cells, dtype=torch.long, device=tokens.device)
        for batch_index in range(batch):
            valid_cells = cells[batch_index, valid[batch_index]]
            if int(valid_cells.numel()):
                counts[batch_index].scatter_add_(
                    0, valid_cells, torch.ones_like(valid_cells, dtype=torch.long)
                )
        max_targets = max(1, int(counts.max().detach().cpu().item()))
    else:
        max_targets = 1
    coordinates = tokens.new_zeros(batch, num_cells, max_targets, 2)
    pt = tokens.new_zeros(batch, num_cells, max_targets)
    energy = tokens.new_zeros(batch, num_cells, max_targets)
    pid_index = torch.zeros(batch, num_cells, max_targets, dtype=torch.long, device=tokens.device)
    charge_index = torch.ones(batch, num_cells, max_targets, dtype=torch.long, device=tokens.device)
    target_mask = torch.zeros(batch, num_cells, max_targets, dtype=torch.bool, device=tokens.device)
    extent = float(coordinate_extent)
    upper = math.nextafter(extent, -math.inf)
    for batch_index in range(batch):
        valid_indices = torch.nonzero(valid[batch_index], as_tuple=False).flatten()
        if not int(valid_indices.numel()):
            continue
        # Stable pT ordering gives the ordered control a deterministic target order.
        order = torch.argsort(tokens[batch_index, valid_indices, 0], descending=True, stable=True)
        valid_indices = valid_indices.index_select(0, order)
        offsets = torch.zeros(num_cells, dtype=torch.long, device=tokens.device)
        for particle_index_tensor in valid_indices:
            particle_index = int(particle_index_tensor.detach().cpu().item())
            cell_index = int(cells[batch_index, particle_index].detach().cpu().item())
            slot_index = int(offsets[cell_index].detach().cpu().item())
            offsets[cell_index] += 1
            row = tokens[batch_index, particle_index]
            coordinates[batch_index, cell_index, slot_index, 0] = torch.clamp(
                row[1] - reference_eta[batch_index], min=-extent, max=upper
            )
            coordinates[batch_index, cell_index, slot_index, 1] = torch.clamp(
                _wrap_phi(row[2] - reference_phi[batch_index]), min=-extent, max=upper
            )
            pt[batch_index, cell_index, slot_index] = row[0].clamp_min(0.0)
            energy[batch_index, cell_index, slot_index] = row[3].clamp_min(0.0)
            pid_scores = row[5:10]
            pid_index[batch_index, cell_index, slot_index] = (
                torch.argmax(pid_scores) if bool(pid_scores.sum() > 0.0) else 1
            )
            charge_index[batch_index, cell_index, slot_index] = torch.sign(row[4]).to(torch.long) + 1
            target_mask[batch_index, cell_index, slot_index] = True
    return CellSlotTargets(
        local_coordinates=coordinates,
        pt=pt,
        energy=energy,
        pid_index=pid_index,
        charge_index=charge_index,
        mask=target_mask,
        terminal_cell_indices=cells,
        terminal_level=int(terminal_level),
    )


def _linear_sum_assignment_cpu(cost: np.ndarray, brute_force_limit: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Solve one CPU-resident hard assignment with the legacy tie policy."""

    if cost.ndim != 2:
        raise ValueError(f"Hungarian cost must be 2D, got {cost.shape}")
    rows_count, columns_count = int(cost.shape[0]), int(cost.shape[1])
    if rows_count == 0 or columns_count == 0:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty, "hungarian_empty"
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        rows, columns = linear_sum_assignment(cost)
        method = "hungarian_scipy"
    except ImportError:
        smaller = min(rows_count, columns_count)
        larger = max(rows_count, columns_count)
        if smaller > int(brute_force_limit):
            raise ImportError(
                "scipy is required for Hungarian matching above the configured brute-force limit"
            )
        best: tuple[float, tuple[int, ...], tuple[int, ...]] | None = None
        if rows_count <= columns_count:
            row_choices = tuple(range(rows_count))
            for columns_choice in itertools.permutations(range(columns_count), rows_count):
                value = float(cost[row_choices, columns_choice].sum())
                if best is None or value < best[0]:
                    best = (value, row_choices, columns_choice)
        else:
            column_choices = tuple(range(columns_count))
            for row_choice in itertools.permutations(range(rows_count), columns_count):
                value = float(cost[row_choice, column_choices].sum())
                if best is None or value < best[0]:
                    best = (value, row_choice, column_choices)
        assert best is not None
        rows, columns = np.asarray(best[1]), np.asarray(best[2])
        method = "hungarian_bruteforce"
    return np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64), method


def _linear_sum_assignment(cost: torch.Tensor, brute_force_limit: int) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Scalar-reference tensor wrapper retained for hard-assignment parity tests."""

    rows, columns, method = _linear_sum_assignment_cpu(
        cost.detach().cpu().contiguous().numpy(),
        brute_force_limit,
    )
    return (
        torch.as_tensor(rows, dtype=torch.long, device=cost.device),
        torch.as_tensor(columns, dtype=torch.long, device=cost.device),
        method,
    )


def _solve_hungarian_cost_collection(
    costs: np.ndarray,
    config: ParticleSlotLossConfig,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Solve a packed cost collection, returning results in original pair order."""

    if costs.ndim != 3:
        raise ValueError(f"packed Hungarian costs must have shape [pairs, slots, targets], got {costs.shape}")
    pairs = int(costs.shape[0])

    def solve(index: int) -> tuple[np.ndarray, np.ndarray, str]:
        return _linear_sum_assignment_cpu(costs[index], int(config.brute_force_limit))

    workers = min(int(config.hungarian_workers), pairs)
    if str(config.hungarian_executor).strip().lower() != "thread" or workers <= 1:
        return [solve(index) for index in range(pairs)]
    # executor.map yields in input order even when individual SciPy calls
    # finish out of order, preserving the scalar branch's deterministic pair
    # ordering for tied costs and all downstream metric aggregation.
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="c2f-hungarian") as executor:
        return list(executor.map(solve, range(pairs)))


def _sinkhorn_square(cost: torch.Tensor, config: ParticleSlotLossConfig) -> torch.Tensor:
    log_transport = -cost / float(config.sinkhorn_temperature)
    for _ in range(int(config.sinkhorn_iterations)):
        log_transport = log_transport - torch.logsumexp(log_transport, dim=-1, keepdim=True)
        log_transport = log_transport - torch.logsumexp(log_transport, dim=-2, keepdim=True)
    return torch.exp(log_transport)


def _pairwise_components(
    output: ParticleSlotDecoderOutput,
    targets: CellSlotTargets,
    batch_index: int,
    view_index: int,
    cell_index: int,
    target_indices: torch.Tensor,
    config: ParticleSlotLossConfig,
) -> dict[str, torch.Tensor]:
    pred_pt = output.total_pt[batch_index, view_index, cell_index]
    pred_energy = output.total_energy[batch_index, view_index, cell_index]
    pred_coordinates = output.local_coordinates[batch_index, view_index, cell_index]
    pred_pid = output.pid_probabilities[batch_index, view_index, cell_index].clamp_min(config.epsilon)
    pred_charge = F.log_softmax(output.charge_logits[batch_index, view_index, cell_index], dim=-1)
    target_pt = targets.pt[batch_index, cell_index].index_select(0, target_indices)
    target_energy = targets.energy[batch_index, cell_index].index_select(0, target_indices)
    target_coordinates = targets.local_coordinates[batch_index, cell_index].index_select(0, target_indices)
    target_pid = targets.pid_index[batch_index, cell_index].index_select(0, target_indices)
    target_charge = targets.charge_index[batch_index, cell_index].index_select(0, target_indices)
    log_pt = (torch.log1p(pred_pt)[:, None] - torch.log1p(target_pt)[None, :]).abs()
    deta = (pred_coordinates[:, None, 0] - target_coordinates[None, :, 0]).abs()
    dphi = _wrap_phi(pred_coordinates[:, None, 1] - target_coordinates[None, :, 1]).abs()
    coordinate = torch.sqrt(deta.square() + dphi.square() + 1.0e-12)
    log_energy = (torch.log1p(pred_energy)[:, None] - torch.log1p(target_energy)[None, :]).abs()
    pid = -torch.log(
        pred_pid[:, None, :].expand(-1, int(target_indices.numel()), -1).gather(
            -1,
            target_pid[None, :, None].expand(int(pred_pt.shape[0]), -1, -1),
        ).squeeze(-1)
    )
    charge = -pred_charge[:, None, :].expand(-1, int(target_indices.numel()), -1).gather(
        -1,
        target_charge[None, :, None].expand(int(pred_pt.shape[0]), -1, -1),
    ).squeeze(-1)
    total = (
        config.log_pt_weight * log_pt
        + config.coordinate_weight * coordinate
        + config.pid_weight * pid
        + config.charge_weight * charge
        + config.log_energy_weight * log_energy
    )
    return {
        "total": total,
        "log_pt": log_pt,
        "deta": deta,
        "dphi": dphi,
        "coordinate": coordinate,
        "pid": pid,
        "charge": charge,
        "log_energy": log_energy,
    }


def _pairwise_components_batched(
    *,
    pred_pt: torch.Tensor,
    pred_energy: torch.Tensor,
    pred_coordinates: torch.Tensor,
    pred_pid: torch.Tensor,
    pred_charge_logits: torch.Tensor,
    target_pt: torch.Tensor,
    target_energy: torch.Tensor,
    target_coordinates: torch.Tensor,
    target_pid: torch.Tensor,
    target_charge: torch.Tensor,
    config: ParticleSlotLossConfig,
) -> dict[str, torch.Tensor]:
    """Return pair costs for a same-target-count batch of cell/view pairs.

    All tensors use a leading ``[N, ...]`` axis, where each row is one
    ``(jet, view, terminal-cell)`` pairing.  Keeping this path batched avoids
    building thousands of tiny Sinkhorn autograd graphs in every C-tier batch.
    """

    if pred_pt.ndim != 2 or target_pt.ndim != 2:
        raise ValueError("batched slot pT tensors must have shape [N, slots/targets]")
    pairs, slots = pred_pt.shape
    if tuple(pred_energy.shape) != (pairs, slots) or tuple(pred_coordinates.shape) != (pairs, slots, 2):
        raise ValueError("batched predicted slot tensors do not align")
    targets_count = int(target_pt.shape[1])
    if tuple(target_energy.shape) != (pairs, targets_count) or tuple(target_coordinates.shape) != (
        pairs,
        targets_count,
        2,
    ):
        raise ValueError("batched target slot tensors do not align")
    if tuple(pred_pid.shape[:2]) != (pairs, slots) or tuple(pred_charge_logits.shape[:2]) != (pairs, slots):
        raise ValueError("batched categorical slot tensors do not align")
    if tuple(target_pid.shape) != (pairs, targets_count) or tuple(target_charge.shape) != (
        pairs,
        targets_count,
    ):
        raise ValueError("batched categorical targets do not align")

    pred_pid = pred_pid.clamp_min(config.epsilon)
    pred_charge = F.log_softmax(pred_charge_logits, dim=-1)
    log_pt = (torch.log1p(pred_pt)[:, :, None] - torch.log1p(target_pt)[:, None, :]).abs()
    deta = (pred_coordinates[:, :, None, 0] - target_coordinates[:, None, :, 0]).abs()
    dphi = _wrap_phi(pred_coordinates[:, :, None, 1] - target_coordinates[:, None, :, 1]).abs()
    coordinate = torch.sqrt(deta.square() + dphi.square() + 1.0e-12)
    log_energy = (torch.log1p(pred_energy)[:, :, None] - torch.log1p(target_energy)[:, None, :]).abs()
    pid = -torch.log(
        torch.gather(
            pred_pid[:, :, None, :].expand(-1, -1, targets_count, -1),
            -1,
            target_pid[:, None, :, None].expand(-1, slots, -1, -1),
        ).squeeze(-1)
    )
    charge = -torch.gather(
        pred_charge[:, :, None, :].expand(-1, -1, targets_count, -1),
        -1,
        target_charge[:, None, :, None].expand(-1, slots, -1, -1),
    ).squeeze(-1)
    total = (
        config.log_pt_weight * log_pt
        + config.coordinate_weight * coordinate
        + config.pid_weight * pid
        + config.charge_weight * charge
        + config.log_energy_weight * log_energy
    )
    return {
        "total": total,
        "log_pt": log_pt,
        "deta": deta,
        "dphi": dphi,
        "coordinate": coordinate,
        "pid": pid,
        "charge": charge,
        "log_energy": log_energy,
    }


def _packed_target_indices(mask: torch.Tensor) -> torch.Tensor:
    """Return valid target positions first, preserving their original order."""

    if mask.ndim != 3:
        raise ValueError("cell target mask must have shape [B, cells, targets]")
    width = int(mask.shape[-1])
    positions = torch.arange(width, device=mask.device, dtype=torch.long)
    positions = positions.view(1, 1, width).expand_as(mask)
    # Valid entries receive keys [0, width); padded entries [width, 2 * width),
    # so sorting provides packed valid positions without assuming the cache is
    # already compacted.
    return torch.argsort((~mask).to(torch.long) * width + positions, dim=-1)


def _weighted_mean(matrix: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
    return (matrix * weights).sum() / weights.sum().clamp_min(epsilon)


def _append_hungarian_rows_scalar(
    output: ParticleSlotDecoderOutput,
    targets: CellSlotTargets,
    config: ParticleSlotLossConfig,
    *,
    return_assignments: bool,
    component_rows: dict[str, list[torch.Tensor]],
    existence_rows: list[torch.Tensor],
    count_rows: list[torch.Tensor],
    missing_rows: list[torch.Tensor],
    reliability_rows: list[torch.Tensor],
    assignments: list[SlotAssignment],
) -> torch.Tensor:
    """Retain the original per-cell C4 implementation as a parity reference."""

    batch, views, cells, slots = output.total_pt.shape
    matched_count = output.total_pt.sum() * 0.0
    for batch_index in range(batch):
        for cell_index in range(cells):
            target_indices = torch.nonzero(targets.mask[batch_index, cell_index], as_tuple=False).flatten()
            target_count = int(target_indices.numel())
            for view_index in range(views):
                existence_target = torch.zeros(slots, device=output.total_pt.device, dtype=output.total_pt.dtype)
                if target_count > 0:
                    pair = _pairwise_components(
                        output,
                        targets,
                        batch_index,
                        view_index,
                        cell_index,
                        target_indices,
                        config,
                    )
                    pred_indices, target_local, method = _linear_sum_assignment(
                        pair["total"], config.brute_force_limit
                    )
                    existence_target[pred_indices] = 1.0
                    for name in ("log_pt", "coordinate", "pid", "charge", "log_energy"):
                        component_rows[name].append(pair[name][pred_indices, target_local].mean())
                    missing_rows.append(pair["total"].new_tensor(float(max(0, target_count - slots))))
                    matched_count = matched_count + float(min(slots, target_count))
                    if return_assignments:
                        assignments.append(
                            SlotAssignment(
                                batch_index=batch_index,
                                view_index=view_index,
                                cell_index=cell_index,
                                pred_indices=pred_indices.detach(),
                                target_indices=target_indices.index_select(0, target_local).detach(),
                                method=method,
                            )
                        )
                logits = output.existence_logits[batch_index, view_index, cell_index]
                existence_rows.append(F.binary_cross_entropy_with_logits(logits, existence_target))
                count_rows.append((torch.sigmoid(logits).sum() - float(target_count)).abs())
                realized_error = (
                    torch.sigmoid(logits).sum() - float(target_count)
                ).abs().detach() / max(1.0, float(target_count))
                reliability_rows.append(
                    F.smooth_l1_loss(
                        output.reliability[batch_index, view_index, cell_index].mean(),
                        (1.0 - realized_error.clamp(0.0, 1.0)),
                    )
                )
    return matched_count


def _append_hungarian_rows_packed(
    output: ParticleSlotDecoderOutput,
    targets: CellSlotTargets,
    config: ParticleSlotLossConfig,
    *,
    return_assignments: bool,
    component_rows: dict[str, list[torch.Tensor]],
    existence_rows: list[torch.Tensor],
    count_rows: list[torch.Tensor],
    missing_rows: list[torch.Tensor],
    reliability_rows: list[torch.Tensor],
    assignments: list[SlotAssignment],
    target_counts: torch.Tensor,
) -> torch.Tensor:
    """Run C4's exact hard assignments with packed CPU cost transfer.

    Pair costs remain FP32 GPU tensors.  Each same-target-count collection is
    copied to CPU once, then SciPy assignments run serially or in a bounded
    thread pool.  Only integer assignment indices return to the GPU; selected
    component losses retain their original autograd path through ``pair``.
    """

    batch, views, cells, slots = output.total_pt.shape
    device = output.total_pt.device
    packed_indices = _packed_target_indices(targets.mask)
    flat_batch = torch.arange(batch, device=device).view(batch, 1, 1).expand(batch, cells, views).reshape(-1)
    flat_cell = torch.arange(cells, device=device).view(1, cells, 1).expand(batch, cells, views).reshape(-1)
    flat_view = torch.arange(views, device=device).view(1, 1, views).expand(batch, cells, views).reshape(-1)
    flat_counts = target_counts.to(dtype=torch.long)[:, :, None].expand(-1, -1, views).reshape(-1)
    records: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, str, dict[str, torch.Tensor]] | None] = [
        None
    ] * int(flat_counts.numel())
    max_targets = int(flat_counts.max().detach().cpu().item()) if int(flat_counts.numel()) else 0

    for target_count in range(1, max_targets + 1):
        selected = torch.nonzero(flat_counts == target_count, as_tuple=False).flatten()
        if not int(selected.numel()):
            continue
        batch_indices = flat_batch.index_select(0, selected)
        cell_indices = flat_cell.index_select(0, selected)
        view_indices = flat_view.index_select(0, selected)
        target_indices = packed_indices[batch_indices, cell_indices, :target_count]
        pair = _pairwise_components_batched(
            pred_pt=output.total_pt[batch_indices, view_indices, cell_indices],
            pred_energy=output.total_energy[batch_indices, view_indices, cell_indices],
            pred_coordinates=output.local_coordinates[batch_indices, view_indices, cell_indices],
            pred_pid=output.pid_probabilities[batch_indices, view_indices, cell_indices],
            pred_charge_logits=output.charge_logits[batch_indices, view_indices, cell_indices],
            target_pt=targets.pt[batch_indices, cell_indices].gather(-1, target_indices),
            target_energy=targets.energy[batch_indices, cell_indices].gather(-1, target_indices),
            target_coordinates=targets.local_coordinates[batch_indices, cell_indices].gather(
                1,
                target_indices[:, :, None].expand(-1, -1, 2),
            ),
            target_pid=targets.pid_index[batch_indices, cell_indices].gather(-1, target_indices),
            target_charge=targets.charge_index[batch_indices, cell_indices].gather(-1, target_indices),
            config=config,
        )
        # One synchronization/copy per target-count group replaces thousands
        # of per-cell GPU-to-CPU transfers in the scalar C4 path.
        solutions = _solve_hungarian_cost_collection(
            pair["total"].detach().to(device="cpu").contiguous().numpy(),
            config,
        )
        original_positions = selected.detach().cpu().tolist()
        for row, (pred_cpu, target_cpu, method) in enumerate(solutions):
            pred_indices = torch.as_tensor(pred_cpu, device=device, dtype=torch.long)
            target_local = torch.as_tensor(target_cpu, device=device, dtype=torch.long)
            selected_components = {
                name: pair[name][row, pred_indices, target_local].mean()
                for name in ("log_pt", "coordinate", "pid", "charge", "log_energy")
            }
            records[int(original_positions[row])] = (
                int(target_count),
                target_indices[row],
                pred_indices,
                target_local,
                method,
                selected_components,
            )

    matched_count = output.total_pt.sum() * 0.0
    for flat_index, record in enumerate(records):
        batch_index = flat_index // (cells * views)
        within_batch = flat_index % (cells * views)
        cell_index = within_batch // views
        view_index = within_batch % views
        target_count = 0 if record is None else record[0]
        existence_target = torch.zeros(slots, device=device, dtype=output.total_pt.dtype)
        if record is not None:
            _, target_indices, pred_indices, target_local, method, selected_components = record
            existence_target[pred_indices] = 1.0
            for name, value in selected_components.items():
                component_rows[name].append(value)
            missing_rows.append(existence_target.new_tensor(float(max(0, target_count - slots))))
            matched_count = matched_count + float(min(slots, target_count))
            if return_assignments:
                assignments.append(
                    SlotAssignment(
                        batch_index=batch_index,
                        view_index=view_index,
                        cell_index=cell_index,
                        pred_indices=pred_indices.detach(),
                        target_indices=target_indices.index_select(0, target_local).detach(),
                        method=method,
                    )
                )
        logits = output.existence_logits[batch_index, view_index, cell_index]
        existence_rows.append(F.binary_cross_entropy_with_logits(logits, existence_target))
        count_rows.append((torch.sigmoid(logits).sum() - float(target_count)).abs())
        realized_error = (torch.sigmoid(logits).sum() - float(target_count)).abs().detach() / max(
            1.0, float(target_count)
        )
        reliability_rows.append(
            F.smooth_l1_loss(
                output.reliability[batch_index, view_index, cell_index].mean(),
                (1.0 - realized_error.clamp(0.0, 1.0)),
            )
        )
    return matched_count


def compute_particle_slot_loss(
    output: ParticleSlotDecoderOutput,
    targets: CellSlotTargets,
    config: ParticleSlotLossConfig | Mapping[str, Any] | None = None,
    *,
    return_assignments: bool = True,
    _hungarian_execution: str = "packed",
) -> ParticleSlotLossOutput:
    """Compute differentiable cell-local matching and consistency losses."""

    if config is None:
        config = ParticleSlotLossConfig(matching_mode=str(output.diagnostics["matching_mode"]))
    elif not isinstance(config, ParticleSlotLossConfig):
        config = ParticleSlotLossConfig(**dict(config))
    if int(targets.terminal_level) != int(output.terminal_level):
        raise ValueError("target and decoder terminal hierarchy levels differ")
    batch, views, cells, slots = output.total_pt.shape
    if tuple(targets.mask.shape[:2]) != (batch, cells):
        raise ValueError("cell targets do not align with slot output")
    zero = output.total_pt.sum() * 0.0
    component_rows: dict[str, list[torch.Tensor]] = {
        name: []
        for name in ("log_pt", "coordinate", "pid", "charge", "log_energy", "uncertainty")
    }
    existence_rows: list[torch.Tensor] = []
    count_rows: list[torch.Tensor] = []
    missing_rows: list[torch.Tensor] = []
    reliability_rows: list[torch.Tensor] = []
    assignments: list[SlotAssignment] = []
    matched_count = zero
    target_counts = targets.mask.sum(dim=-1).to(dtype=output.total_pt.dtype)
    target_count_total = target_counts.sum()
    existence_targets = torch.zeros_like(output.existence_logits)

    if config.matching_mode == "hungarian":
        if _hungarian_execution == "scalar_reference":
            matched_count = _append_hungarian_rows_scalar(
                output,
                targets,
                config,
                return_assignments=return_assignments,
                component_rows=component_rows,
                existence_rows=existence_rows,
                count_rows=count_rows,
                missing_rows=missing_rows,
                reliability_rows=reliability_rows,
                assignments=assignments,
            )
        elif _hungarian_execution == "packed":
            matched_count = _append_hungarian_rows_packed(
                output,
                targets,
                config,
                return_assignments=return_assignments,
                component_rows=component_rows,
                existence_rows=existence_rows,
                count_rows=count_rows,
                missing_rows=missing_rows,
                reliability_rows=reliability_rows,
                assignments=assignments,
                target_counts=target_counts,
            )
        else:
            raise ValueError(f"unknown Hungarian execution mode {_hungarian_execution!r}")
    else:
        packed_indices = _packed_target_indices(targets.mask)
        flat_batch = torch.arange(batch, device=output.total_pt.device).view(batch, 1, 1).expand(
            batch, cells, views
        ).reshape(-1)
        flat_cell = torch.arange(cells, device=output.total_pt.device).view(1, cells, 1).expand(
            batch, cells, views
        ).reshape(-1)
        flat_view = torch.arange(views, device=output.total_pt.device).view(1, 1, views).expand(
            batch, cells, views
        ).reshape(-1)
        flat_counts = target_counts.to(dtype=torch.long)[:, :, None].expand(-1, -1, views).reshape(-1)
        max_targets = int(flat_counts.max().detach().cpu().item()) if int(flat_counts.numel()) else 0
        for target_count in range(1, max_targets + 1):
            selected = torch.nonzero(flat_counts == target_count, as_tuple=False).flatten()
            if not int(selected.numel()):
                continue
            batch_indices = flat_batch.index_select(0, selected)
            cell_indices = flat_cell.index_select(0, selected)
            view_indices = flat_view.index_select(0, selected)
            target_indices = packed_indices[batch_indices, cell_indices, :target_count]
            target_pt = targets.pt[batch_indices, cell_indices].gather(-1, target_indices)
            target_energy = targets.energy[batch_indices, cell_indices].gather(-1, target_indices)
            target_coordinates = targets.local_coordinates[batch_indices, cell_indices].gather(
                1,
                target_indices[:, :, None].expand(-1, -1, 2),
            )
            target_pid = targets.pid_index[batch_indices, cell_indices].gather(-1, target_indices)
            target_charge = targets.charge_index[batch_indices, cell_indices].gather(-1, target_indices)
            pred_pt = output.total_pt[batch_indices, view_indices, cell_indices]
            pred_energy = output.total_energy[batch_indices, view_indices, cell_indices]
            pred_coordinates = output.local_coordinates[batch_indices, view_indices, cell_indices]
            pred_pid = output.pid_probabilities[batch_indices, view_indices, cell_indices]
            pred_charge_logits = output.charge_logits[batch_indices, view_indices, cell_indices]
            pair = _pairwise_components_batched(
                pred_pt=pred_pt,
                pred_energy=pred_energy,
                pred_coordinates=pred_coordinates,
                pred_pid=pred_pid,
                pred_charge_logits=pred_charge_logits,
                target_pt=target_pt,
                target_energy=target_energy,
                target_coordinates=target_coordinates,
                target_pid=target_pid,
                target_charge=target_charge,
                config=config,
            )
            pairs = int(selected.numel())
            if config.matching_mode == "sinkhorn":
                size = max(slots, target_count)
                square = pair["total"].new_zeros(pairs, size, size)
                square[:, :slots, :target_count] = pair["total"]
                if target_count < size:
                    square[:, :slots, target_count:] = F.softplus(
                        output.existence_logits[batch_indices, view_indices, cell_indices]
                    )[:, :, None]
                if slots < size:
                    square[:, slots:, :target_count] = float(config.missing_target_weight)
                transport = _sinkhorn_square(square, config) * float(size)
                real_transport = transport[:, :slots, :target_count]
                existence_target = real_transport.sum(dim=-1).clamp(0.0, 1.0).detach()
                existence_targets[batch_indices, view_indices, cell_indices] = existence_target
                for name in ("log_pt", "coordinate", "pid", "charge", "log_energy"):
                    component_rows[name].append(
                        (pair[name] * real_transport).sum(dim=(-2, -1))
                        / real_transport.sum(dim=(-2, -1)).clamp_min(config.epsilon)
                    )
                missing_rows.append(
                    transport[:, slots:, :target_count].sum(dim=(-2, -1)) / float(max(1, target_count))
                )
                if output.log_sigma is not None:
                    sigma = output.log_sigma[batch_indices, view_indices, cell_indices]
                    error_vector = torch.stack(
                        (
                            pair["log_pt"],
                            pair["deta"],
                            pair["dphi"],
                            pair["log_energy"],
                            pair["pid"],
                        ),
                        dim=-1,
                    )
                    log_sigma = sigma[:, :, None, :].expand_as(error_vector)
                    nll = 0.5 * error_vector.square() * torch.exp(-2.0 * log_sigma) + log_sigma
                    component_rows["uncertainty"].append(
                        (nll.mean(dim=-1) * real_transport).sum(dim=(-2, -1))
                        / real_transport.sum(dim=(-2, -1)).clamp_min(config.epsilon)
                    )
                if return_assignments:
                    for row in range(pairs):
                        pred_indices, target_local = torch.nonzero(
                            real_transport[row] > (0.5 / float(size)), as_tuple=True
                        )
                        assignments.append(
                            SlotAssignment(
                                batch_index=int(batch_indices[row].detach().cpu().item()),
                                view_index=int(view_indices[row].detach().cpu().item()),
                                cell_index=int(cell_indices[row].detach().cpu().item()),
                                pred_indices=pred_indices.detach(),
                                target_indices=target_indices[row].index_select(0, target_local).detach(),
                                method="sinkhorn",
                            )
                        )
            else:
                matched = min(slots, target_count)
                diagonal = torch.arange(matched, device=output.total_pt.device)
                existence_target = torch.zeros_like(pred_pt)
                existence_target[:, :matched] = 1.0
                existence_targets[batch_indices, view_indices, cell_indices] = existence_target
                for name in ("log_pt", "coordinate", "pid", "charge", "log_energy"):
                    component_rows[name].append(pair[name][:, diagonal, diagonal].mean(dim=-1))
                missing_rows.append(pair["total"].new_full((pairs,), float(max(0, target_count - slots))))
                if return_assignments:
                    for row in range(pairs):
                        assignments.append(
                            SlotAssignment(
                                batch_index=int(batch_indices[row].detach().cpu().item()),
                                view_index=int(view_indices[row].detach().cpu().item()),
                                cell_index=int(cell_indices[row].detach().cpu().item()),
                                pred_indices=diagonal.detach(),
                                target_indices=target_indices[row, :matched].detach(),
                                method="ordered",
                            )
                        )
            matched_count = matched_count + float(min(slots, target_count) * pairs)

        existence_rows.append(
            F.binary_cross_entropy_with_logits(output.existence_logits, existence_targets)
        )
        expected_counts = torch.sigmoid(output.existence_logits).sum(dim=-1)
        target_counts_by_view = target_counts[:, None, :].expand(-1, views, -1)
        count_rows.append((expected_counts - target_counts_by_view).abs().mean())
        realized_error = (expected_counts - target_counts_by_view).abs().detach() / target_counts_by_view.clamp_min(1.0)
        reliability_rows.append(
            F.smooth_l1_loss(
                output.reliability.mean(dim=-1),
                1.0 - realized_error.clamp(0.0, 1.0),
            )
        )

    def mean(rows: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([row.reshape(-1) for row in rows]).mean() if rows else zero

    pid_raw = F.log_softmax(output.raw_pid_logits, dim=-1)
    constrained_pid = output.pid_probabilities.detach().clamp_min(config.epsilon)
    pid_consistency = F.kl_div(pid_raw, constrained_pid, reduction="none").sum(dim=-1).mean()
    accounting_scale = torch.log1p(output.terminal_accounting.clamp_min(0.0))[:, None, :, :]
    rendered_scale = torch.log1p(output.rendered_accounting.clamp_min(0.0))
    accounting_consistency = F.smooth_l1_loss(
        rendered_scale,
        accounting_scale.expand_as(rendered_scale),
        beta=float(config.huber_beta),
    )
    if output.dust_total_pt is None:
        dust = zero
        dust_fraction = zero
    else:
        cell_pt = output.terminal_accounting[..., ACCOUNTING_FIELD_NAMES.index("total_pT")]
        dust_fraction_tensor = output.dust_total_pt / cell_pt[:, None, :].clamp_min(config.epsilon)
        dust = dust_fraction_tensor.mean()
        dust_fraction = dust_fraction_tensor.detach().mean()
    components = {
        "matched_log_pt": mean(component_rows["log_pt"]),
        "matched_coordinate": mean(component_rows["coordinate"]),
        "matched_pid": mean(component_rows["pid"]),
        "matched_charge": mean(component_rows["charge"]),
        "matched_log_energy": mean(component_rows["log_energy"]),
        "existence": mean(existence_rows),
        "count": mean(count_rows),
        "pid_consistency": pid_consistency,
        "accounting_consistency": accounting_consistency,
        "dust": dust,
        "missing_target": mean(missing_rows),
        "uncertainty_nll": mean(component_rows["uncertainty"]),
        "reliability": mean(reliability_rows),
    }
    loss = (
        config.log_pt_weight * components["matched_log_pt"]
        + config.coordinate_weight * components["matched_coordinate"]
        + config.pid_weight * components["matched_pid"]
        + config.charge_weight * components["matched_charge"]
        + config.log_energy_weight * components["matched_log_energy"]
        + config.existence_weight * components["existence"]
        + config.count_weight * components["count"]
        + config.pid_consistency_weight * components["pid_consistency"]
        + config.accounting_consistency_weight * components["accounting_consistency"]
        + config.dust_weight * components["dust"]
        + config.missing_target_weight * components["missing_target"]
        + config.uncertainty_weight * components["uncertainty_nll"]
        + config.reliability_weight * components["reliability"]
    )
    metrics = {
        "matched_slots": matched_count.detach(),
        "target_particles": target_count_total.detach(),
        "matched_fraction": (matched_count / target_count_total.clamp_min(1.0)).detach(),
        "dust_pt_fraction": dust_fraction,
        "rendered_accounting_log_mae": (rendered_scale - accounting_scale).abs().mean().detach(),
        "mean_expected_slot_count": output.expected_count.detach().mean(),
    }
    if not torch.isfinite(loss):
        raise FloatingPointError("particle-slot loss is non-finite")
    return ParticleSlotLossOutput(
        loss=loss,
        components=components,
        metrics=metrics,
        assignments=tuple(assignments),
    )


def _compute_particle_slot_loss_hungarian_scalar_reference(
    output: ParticleSlotDecoderOutput,
    targets: CellSlotTargets,
    config: ParticleSlotLossConfig | Mapping[str, Any] | None = None,
    *,
    return_assignments: bool = True,
) -> ParticleSlotLossOutput:
    """Test-only scalar C4 reference for packed/threaded parity checks."""

    if config is None:
        resolved = ParticleSlotLossConfig(matching_mode="hungarian")
    elif isinstance(config, ParticleSlotLossConfig):
        resolved = config
    else:
        resolved = ParticleSlotLossConfig(**dict(config))
    if resolved.matching_mode != "hungarian":
        raise ValueError("Hungarian scalar reference requires matching_mode='hungarian'")
    return compute_particle_slot_loss(
        output,
        targets,
        resolved,
        return_assignments=return_assignments,
        _hungarian_execution="scalar_reference",
    )


@torch.no_grad()
def hungarian_slot_diagnostic(
    output: ParticleSlotDecoderOutput,
    targets: CellSlotTargets,
    config: ParticleSlotLossConfig | Mapping[str, Any] | None = None,
) -> ParticleSlotLossOutput:
    """Evaluation-only hard assignment diagnostic independent of train matcher."""

    if config is None:
        resolved = ParticleSlotLossConfig(matching_mode="hungarian")
    elif isinstance(config, ParticleSlotLossConfig):
        resolved = ParticleSlotLossConfig(**{**asdict(config), "matching_mode": "hungarian"})
    else:
        resolved = ParticleSlotLossConfig(**{**dict(config), "matching_mode": "hungarian"})
    return compute_particle_slot_loss(output, targets, resolved)
