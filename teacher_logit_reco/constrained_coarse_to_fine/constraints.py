"""Differentiable accounting constraints for coarse-to-fine reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .layout import (
    ACCOUNTING_FIELD_NAMES,
    MOMENT_FIELD_NAMES,
    PID_CATEGORY_NAMES,
    PID_COUNT_FIELD_NAMES,
    PID_PT_FIELD_NAMES,
    PRIMITIVE_ACCOUNTING_FIELD_NAMES,
)


CONSTRAINED_ACCOUNTING_LAYER_CONTRACT = "constrained_coarse_to_fine_accounting_layers_v1"
DEFAULT_POSITIVE_FLOOR = 1.0e-8

ACCOUNTING_INDEX = {name: index for index, name in enumerate(ACCOUNTING_FIELD_NAMES)}
PRIMITIVE_TO_FULL_INDICES: tuple[int, ...] = tuple(
    ACCOUNTING_INDEX[name] for name in PRIMITIVE_ACCOUNTING_FIELD_NAMES
)
PID_PT_INDICES: tuple[int, ...] = tuple(ACCOUNTING_INDEX[name] for name in PID_PT_FIELD_NAMES)
PID_COUNT_INDICES: tuple[int, ...] = tuple(ACCOUNTING_INDEX[name] for name in PID_COUNT_FIELD_NAMES)
TOTAL_PT_INDEX = ACCOUNTING_INDEX["total_pT"]
TOTAL_ENERGY_INDEX = ACCOUNTING_INDEX["total_energy"]
TOTAL_COUNT_INDEX = ACCOUNTING_INDEX["expected_constituent_count"]

SIGNED_MOMENT_NAMES: tuple[str, ...] = (
    "sum_pT_deta",
    "sum_pT_dphi",
    "axis_deta",
    "axis_dphi",
    "width_eta",
    "width_phi",
    "mean_r",
    "r_rms",
)


def _require_last_dim(tensor: torch.Tensor, expected: int, *, name: str) -> None:
    if tensor.ndim < 1 or int(tensor.shape[-1]) != int(expected):
        raise ValueError(f"{name} last dimension must be {expected}, got {tuple(tensor.shape)}")


def _index_tensor(indices: Sequence[int], reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(tuple(int(index) for index in indices), dtype=torch.long, device=reference.device)


def primitive_accounting(accounting: torch.Tensor) -> torch.Tensor:
    """Select the independently predicted nonnegative accounting channels."""

    _require_last_dim(accounting, len(ACCOUNTING_FIELD_NAMES), name="accounting")
    return torch.index_select(accounting, -1, _index_tensor(PRIMITIVE_TO_FULL_INDICES, accounting))


def assemble_accounting(primitive: torch.Tensor) -> torch.Tensor:
    """Assemble full accounting and derive pT/count totals from categories."""

    _require_last_dim(primitive, len(PRIMITIVE_ACCOUNTING_FIELD_NAMES), name="primitive")
    output = primitive.new_zeros((*primitive.shape[:-1], len(ACCOUNTING_FIELD_NAMES)))
    output[..., _index_tensor(PRIMITIVE_TO_FULL_INDICES, primitive)] = primitive
    output[..., TOTAL_PT_INDEX] = torch.sum(
        output[..., _index_tensor(PID_PT_INDICES, output)], dim=-1
    )
    output[..., TOTAL_COUNT_INDEX] = torch.sum(
        output[..., _index_tensor(PID_COUNT_INDICES, output)], dim=-1
    )
    return output


def canonicalize_accounting(accounting: torch.Tensor) -> torch.Tensor:
    """Discard free derived totals and reconstruct the coupled full vector."""

    return assemble_accounting(primitive_accounting(accounting))


class PositiveAccountingParameterization(nn.Module):
    """Map unconstrained primitive logits to a coupled nonnegative vector."""

    def __init__(
        self,
        *,
        minimum: float = DEFAULT_POSITIVE_FLOOR,
        beta: float = 1.0,
        threshold: float = 20.0,
    ) -> None:
        super().__init__()
        if float(minimum) < 0.0:
            raise ValueError("minimum must be nonnegative")
        if float(beta) <= 0.0:
            raise ValueError("beta must be positive")
        self.minimum = float(minimum)
        self.beta = float(beta)
        self.threshold = float(threshold)

    @property
    def input_dim(self) -> int:
        return len(PRIMITIVE_ACCOUNTING_FIELD_NAMES)

    @property
    def output_dim(self) -> int:
        return len(ACCOUNTING_FIELD_NAMES)

    def forward(self, raw_primitive: torch.Tensor) -> torch.Tensor:
        _require_last_dim(raw_primitive, self.input_dim, name="raw_primitive")
        positive = F.softplus(raw_primitive, beta=self.beta, threshold=self.threshold)
        if self.minimum:
            positive = positive + self.minimum
        return assemble_accounting(positive)


@dataclass(frozen=True)
class AccountingAllocationOutput:
    children: torch.Tensor
    primitive_fractions: torch.Tensor


class SoftmaxAccountingAllocator(nn.Module):
    """Allocate every primitive parent channel across a fixed child set."""

    def __init__(self, num_children: int, *, temperature: float = 1.0) -> None:
        super().__init__()
        if int(num_children) <= 0:
            raise ValueError("num_children must be positive")
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be positive")
        self.num_children = int(num_children)
        self.temperature = float(temperature)

    @property
    def logit_dim(self) -> int:
        return len(PRIMITIVE_ACCOUNTING_FIELD_NAMES)

    def forward(
        self,
        parent_accounting: torch.Tensor,
        allocation_logits: torch.Tensor,
    ) -> AccountingAllocationOutput:
        _require_last_dim(parent_accounting, len(ACCOUNTING_FIELD_NAMES), name="parent_accounting")
        expected_shape = (*parent_accounting.shape[:-1], self.num_children, self.logit_dim)
        if tuple(allocation_logits.shape) != expected_shape:
            raise ValueError(
                f"allocation_logits shape must be {expected_shape}, got {tuple(allocation_logits.shape)}"
            )
        parent_primitive = primitive_accounting(parent_accounting)
        fractions = torch.softmax(allocation_logits / self.temperature, dim=-2)
        child_primitive = parent_primitive.unsqueeze(-2) * fractions
        return AccountingAllocationOutput(
            children=assemble_accounting(child_primitive),
            primitive_fractions=fractions,
        )


@dataclass(frozen=True)
class CategorySlotAllocationOutput:
    category_per_slot: torch.Tensor
    total_per_slot: torch.Tensor
    category_probabilities: torch.Tensor
    fractions: torch.Tensor


def _category_slot_allocation(
    category_totals: torch.Tensor,
    logits: torch.Tensor,
    *,
    epsilon: float,
) -> CategorySlotAllocationOutput:
    _require_last_dim(category_totals, len(PID_CATEGORY_NAMES), name="category_totals")
    if logits.ndim != category_totals.ndim + 1:
        raise ValueError("category slot logits must add exactly one slot dimension")
    if logits.shape[:-2] != category_totals.shape[:-1] or logits.shape[-1] != len(PID_CATEGORY_NAMES):
        raise ValueError(
            "category slot logits must have shape [..., K, num_pid_categories] aligned with category totals"
        )
    if int(logits.shape[-2]) <= 0:
        raise ValueError("category slot logits must contain at least one slot")
    fractions = torch.softmax(logits, dim=-2)
    category_per_slot = category_totals.unsqueeze(-2) * fractions
    total_per_slot = torch.sum(category_per_slot, dim=-1)
    cell_total = torch.sum(category_totals, dim=-1, keepdim=True)
    prior = category_totals / torch.clamp(cell_total, min=float(epsilon))
    uniform = torch.full_like(prior, 1.0 / float(len(PID_CATEGORY_NAMES)))
    prior = torch.where(cell_total > float(epsilon), prior, uniform)
    normalized = category_per_slot / torch.clamp(total_per_slot.unsqueeze(-1), min=float(epsilon))
    category_probabilities = torch.where(
        total_per_slot.unsqueeze(-1) > float(epsilon),
        normalized,
        prior.unsqueeze(-2).expand_as(normalized),
    )
    return CategorySlotAllocationOutput(
        category_per_slot=category_per_slot,
        total_per_slot=total_per_slot,
        category_probabilities=category_probabilities,
        fractions=fractions,
    )


class CategoryPtSlotAllocator(nn.Module):
    """Allocate each PID-category pT total over slots, including dust."""

    def __init__(self, *, epsilon: float = DEFAULT_POSITIVE_FLOOR) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, cell_accounting: torch.Tensor, category_pt_logits: torch.Tensor) -> CategorySlotAllocationOutput:
        canonical = canonicalize_accounting(cell_accounting)
        category_totals = torch.index_select(canonical, -1, _index_tensor(PID_PT_INDICES, canonical))
        return _category_slot_allocation(category_totals, category_pt_logits, epsilon=self.epsilon)


class CategoryCountSlotAllocator(nn.Module):
    """Optional stronger allocator for per-category expected slot counts."""

    def __init__(self, *, epsilon: float = DEFAULT_POSITIVE_FLOOR) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(
        self,
        cell_accounting: torch.Tensor,
        category_count_logits: torch.Tensor,
    ) -> CategorySlotAllocationOutput:
        canonical = canonicalize_accounting(cell_accounting)
        category_totals = torch.index_select(canonical, -1, _index_tensor(PID_COUNT_INDICES, canonical))
        return _category_slot_allocation(category_totals, category_count_logits, epsilon=self.epsilon)


@dataclass(frozen=True)
class SlotCountAllocationOutput:
    expected_count_per_slot: torch.Tensor
    fractions: torch.Tensor


class SlotCountNormalizer(nn.Module):
    """Normalize sigmoid slot weights to the cell's expected total count."""

    def __init__(self, *, epsilon: float = DEFAULT_POSITIVE_FLOOR) -> None:
        super().__init__()
        if float(epsilon) <= 0.0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)

    def forward(self, cell_accounting: torch.Tensor, raw_count_weights: torch.Tensor) -> SlotCountAllocationOutput:
        canonical = canonicalize_accounting(cell_accounting)
        if raw_count_weights.ndim != canonical.ndim:
            raise ValueError("raw_count_weights must replace the accounting field dimension with a slot dimension")
        if raw_count_weights.shape[:-1] != canonical.shape[:-1]:
            raise ValueError("raw_count_weights leading dimensions must align with cell_accounting")
        if int(raw_count_weights.shape[-1]) <= 0:
            raise ValueError("raw_count_weights must contain at least one slot")
        weights = torch.sigmoid(raw_count_weights).clamp_min(self.epsilon)
        fractions = weights / torch.sum(weights, dim=-1, keepdim=True)
        total_count = canonical[..., TOTAL_COUNT_INDEX]
        expected = total_count.unsqueeze(-1) * fractions
        return SlotCountAllocationOutput(expected_count_per_slot=expected, fractions=fractions)


@dataclass(frozen=True)
class ConstrainedSlotOutput:
    category_pt_per_slot: torch.Tensor
    total_pt_per_slot: torch.Tensor
    pid_probabilities: torch.Tensor
    expected_count_per_slot: torch.Tensor
    category_pt_fractions: torch.Tensor
    count_fractions: torch.Tensor


class ConstrainedSlotNormalizer(nn.Module):
    """Joint pT-composition and expected-count normalization for slots."""

    def __init__(self, *, epsilon: float = DEFAULT_POSITIVE_FLOOR) -> None:
        super().__init__()
        self.pt_allocator = CategoryPtSlotAllocator(epsilon=epsilon)
        self.count_allocator = SlotCountNormalizer(epsilon=epsilon)

    def forward(
        self,
        cell_accounting: torch.Tensor,
        category_pt_logits: torch.Tensor,
        raw_count_weights: torch.Tensor,
    ) -> ConstrainedSlotOutput:
        pt_output = self.pt_allocator(cell_accounting, category_pt_logits)
        count_output = self.count_allocator(cell_accounting, raw_count_weights)
        if pt_output.total_per_slot.shape != count_output.expected_count_per_slot.shape:
            raise ValueError("category_pt_logits and raw_count_weights must describe the same slot count")
        return ConstrainedSlotOutput(
            category_pt_per_slot=pt_output.category_per_slot,
            total_pt_per_slot=pt_output.total_per_slot,
            pid_probabilities=pt_output.category_probabilities,
            expected_count_per_slot=count_output.expected_count_per_slot,
            category_pt_fractions=pt_output.fractions,
            count_fractions=count_output.fractions,
        )


@dataclass(frozen=True)
class MomentReconstructionOutput:
    values: torch.Tensor
    names: tuple[str, ...] = SIGNED_MOMENT_NAMES

    def field(self, name: str) -> torch.Tensor:
        try:
            index = self.names.index(str(name))
        except ValueError as exc:
            raise KeyError(f"unknown reconstructed moment {name!r}") from exc
        return self.values[..., index]


class PositiveNegativeMomentReconstructor(nn.Module):
    """Derive signed moments, axes, and widths from positive accounting pieces."""

    def __init__(self, *, epsilon: float = DEFAULT_POSITIVE_FLOOR) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, accounting: torch.Tensor) -> MomentReconstructionOutput:
        canonical = canonicalize_accounting(accounting)
        total_pt = canonical[..., TOTAL_PT_INDEX]
        safe_pt = torch.clamp(total_pt, min=self.epsilon)
        signed_deta = (
            canonical[..., ACCOUNTING_INDEX["sum_pT_abs_deta_pos"]]
            - canonical[..., ACCOUNTING_INDEX["sum_pT_abs_deta_neg"]]
        )
        signed_dphi = (
            canonical[..., ACCOUNTING_INDEX["sum_pT_abs_dphi_pos"]]
            - canonical[..., ACCOUNTING_INDEX["sum_pT_abs_dphi_neg"]]
        )
        axis_deta = signed_deta / safe_pt
        axis_dphi = signed_dphi / safe_pt
        width_eta = torch.clamp(
            canonical[..., ACCOUNTING_INDEX["sum_pT_deta2"]] / safe_pt - axis_deta.square(),
            min=0.0,
        )
        width_phi = torch.clamp(
            canonical[..., ACCOUNTING_INDEX["sum_pT_dphi2"]] / safe_pt - axis_dphi.square(),
            min=0.0,
        )
        mean_r = canonical[..., ACCOUNTING_INDEX["sum_pT_r"]] / safe_pt
        r_rms = torch.sqrt(
            torch.clamp(canonical[..., ACCOUNTING_INDEX["sum_pT_r2"]] / safe_pt, min=0.0)
        )
        values = torch.stack(
            (signed_deta, signed_dphi, axis_deta, axis_dphi, width_eta, width_phi, mean_r, r_rms),
            dim=-1,
        )
        values = torch.where((total_pt > self.epsilon).unsqueeze(-1), values, torch.zeros_like(values))
        return MomentReconstructionOutput(values=values)


class CellBoundCoordinateTransform(nn.Module):
    """Map unconstrained local coordinates into each cell's eta/phi rectangle."""

    def forward(self, raw_coordinates: torch.Tensor, cell_bounds: torch.Tensor) -> torch.Tensor:
        _require_last_dim(raw_coordinates, 2, name="raw_coordinates")
        _require_last_dim(cell_bounds, 4, name="cell_bounds")
        if cell_bounds.ndim < raw_coordinates.ndim:
            # Bounds normally omit the slot dimension: [B, C, 4] or [C, 4]
            # accompanies coordinates [B, C, K, 2]. Standard broadcasting can
            # supply any omitted batch axes after this one explicit slot axis.
            cell_bounds = cell_bounds.unsqueeze(-2)
        try:
            leading_shape = torch.broadcast_shapes(raw_coordinates.shape[:-1], cell_bounds.shape[:-1])
            raw_coordinates = raw_coordinates.expand(*leading_shape, 2)
            cell_bounds = cell_bounds.expand(*leading_shape, 4)
        except RuntimeError as exc:
            raise ValueError("raw_coordinates and cell_bounds are not broadcast-compatible") from exc
        eta_min, eta_max, phi_min, phi_max = cell_bounds.unbind(dim=-1)
        if torch.any(eta_max < eta_min) or torch.any(phi_max < phi_min):
            raise ValueError("cell bounds must satisfy max >= min")
        unit = torch.sigmoid(raw_coordinates)
        eta = eta_min + unit[..., 0] * (eta_max - eta_min)
        phi = phi_min + unit[..., 1] * (phi_max - phi_min)
        return torch.stack((eta, phi), dim=-1)
