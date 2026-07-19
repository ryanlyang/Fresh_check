"""Exact differentiable binary accounting and two-body phase-space compiler."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .root_compiler import (
    feasible_charge_mask,
    minimum_mass_budget,
)
from .root_transforms import (
    ROOT_FEATURE_INDEX,
    ROOT_SHAPE_FEATURE_NAMES,
)
from .root_compiler import compile_shape_features
from .schemas import ABPH_EFFECTIVE_MASS_GEV, ABPH_MAX_PARTICLES, ABPH_PID_CATEGORIES
from .targets import (
    ROOT_FEATURE_NAMES,
    TOPOLOGY_ACTIVE_SPLIT,
    TOPOLOGY_ACTIVE_TERMINAL,
)


ABPH_BINARY_ACCOUNTING_CONTRACT = "adaptive_binary_pseudooffline_binary_accounting_v1"
ABPH_BINARY_ACCOUNTING_VERSION = "v1"
ABPH_BINARY_COUNT_SUPPORT = ABPH_MAX_PARTICLES - 1
ABPH_BINARY_P4_ABS_TOLERANCE = 2.5e-4
ABPH_BINARY_P4_REL_TOLERANCE = 3.0e-6
ABPH_NEAR_MASSLESS_THRESHOLD_GEV = 1.0e-5
ABPH_MASS_PRECISION_ABS_TOLERANCE = 3.0e-5
# Float32 (E, p) loses low invariant masses through E^2-p^2 cancellation for
# highly boosted particles. The ledger floor remains exact; this tolerance is
# only for auditing the p4 representation of that already-compiled state.
# Invariant mass is recovered from E^2-|p|^2, so float32 component error is
# amplified by cancellation for boosted states. Its resolvable mass scale is
# O(sqrt(eps_float32) * E), not O(eps_float32 * E). This bound is used only as
# a numerical feasibility tolerance; compiled masses are still hard-bounded by
# the discrete type budget before p4 construction.
ABPH_MASS_PRECISION_ENERGY_FACTOR = 5.0e-4
ABPH_AUXILIARY_ADDITIVE_NAMES: tuple[str, ...] = (
    "scalar_sum_pt",
    *(f"energy_{name}" for name in ABPH_PID_CATEGORIES),
    *(f"scalar_pt_{name}" for name in ABPH_PID_CATEGORIES),
    "absolute_charge_sum",
)


def binary_accounting_manifest() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contract": ABPH_BINARY_ACCOUNTING_CONTRACT,
        "version": ABPH_BINARY_ACCOUNTING_VERSION,
        "compiler_order": [
            "topology",
            "child_count",
            "child_type_counts",
            "child_charge",
            "child_minimum_mass",
            "two_body_four_vector",
            "soft_auxiliary_accounting",
        ],
        "effective_mass_gev": dict(ABPH_EFFECTIVE_MASS_GEV),
        "near_massless_threshold_gev": ABPH_NEAR_MASSLESS_THRESHOLD_GEV,
        "near_massless_branch": "positive_collinear_fraction",
        "integer_type_allocator": "exact_linear_minimum_cost_with_capped_sigmoid_relaxation",
        "integer_charge_allocator": "masked_exact_parent_conserving_map",
        "hard_tolerances": {
            "p4_absolute": ABPH_BINARY_P4_ABS_TOLERANCE,
            "p4_relative": ABPH_BINARY_P4_REL_TOLERANCE,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["compiler_hash"] = hashlib.sha256(encoded).hexdigest()
    return payload


@dataclass(frozen=True)
class AccountingState:
    four_vector: Any
    constituent_count: Any
    type_counts: Any
    integer_charge: Any
    minimum_mass_budget: Any
    scalar_sum_pt: Any
    type_energy: Any
    type_scalar_pt: Any
    absolute_charge_sum: Any
    shape_features: Any
    ledger: Any

    @property
    def batch_size(self) -> int:
        return int(self.constituent_count.shape[0])

    @classmethod
    def from_ledger(cls, ledger: Any, *, validate: bool = True) -> "AccountingState":
        torch = require_torch()
        raw_values = torch.as_tensor(ledger)
        dtype = torch.float64 if raw_values.dtype == torch.float64 else torch.float32
        values = raw_values.to(dtype)
        if values.ndim != 2 or values.shape[-1] < len(ROOT_FEATURE_NAMES):
            raise ValueError(
                f"accounting ledger must have shape [B, >= {len(ROOT_FEATURE_NAMES)}]"
            )
        values = values[:, : len(ROOT_FEATURE_NAMES)]
        state = cls(
            four_vector=torch.stack(
                tuple(values[:, ROOT_FEATURE_INDEX[name]] for name in ("energy", "px", "py", "pz")),
                dim=-1,
            ),
            constituent_count=values[:, ROOT_FEATURE_INDEX["constituent_count"]].round().to(torch.long),
            type_counts=torch.stack(
                tuple(
                    values[:, ROOT_FEATURE_INDEX[f"count_{name}"]].round().to(torch.long)
                    for name in ABPH_PID_CATEGORIES
                ),
                dim=-1,
            ),
            integer_charge=values[:, ROOT_FEATURE_INDEX["integer_charge"]].round().to(torch.long),
            minimum_mass_budget=values[:, ROOT_FEATURE_INDEX["minimum_mass_budget"]],
            scalar_sum_pt=values[:, ROOT_FEATURE_INDEX["scalar_sum_pt"]],
            type_energy=torch.stack(
                tuple(values[:, ROOT_FEATURE_INDEX[f"energy_{name}"]] for name in ABPH_PID_CATEGORIES),
                dim=-1,
            ),
            type_scalar_pt=torch.stack(
                tuple(values[:, ROOT_FEATURE_INDEX[f"scalar_pt_{name}"]] for name in ABPH_PID_CATEGORIES),
                dim=-1,
            ),
            absolute_charge_sum=values[:, ROOT_FEATURE_INDEX["absolute_charge_sum"]],
            shape_features=torch.stack(
                tuple(values[:, ROOT_FEATURE_INDEX[name]] for name in ROOT_SHAPE_FEATURE_NAMES),
                dim=-1,
            ),
            ledger=values,
        )
        if validate:
            require_accounting_state(state)
        return state


@dataclass(frozen=True)
class BinarySplitPrediction:
    topology_logits: Any
    count_logits: Any
    type_allocation_logits: Any
    charge_logits: Any
    mass_allocation_logits: Any
    direction_raw: Any
    collinear_fraction_raw: Any
    auxiliary_fraction_logits: Any
    child_shape_raw: Any

    @property
    def batch_size(self) -> int:
        return int(self.topology_logits.shape[0])


@dataclass(frozen=True)
class CompiledBinarySplit:
    topology: Any
    relaxed_split_probability: Any
    split_mask: Any
    child_mask: Any
    child_four_vector: Any
    child_constituent_count: Any
    relaxed_child_constituent_count: Any
    child_type_counts: Any
    relaxed_child_type_counts: Any
    child_integer_charge: Any
    relaxed_child_charge: Any
    child_minimum_mass_budget: Any
    child_scalar_sum_pt: Any
    child_type_energy: Any
    child_type_scalar_pt: Any
    child_absolute_charge_sum: Any
    child_shape_features: Any
    child_ledger: Any
    diagnostics: Mapping[str, Any]

    def child_state(self, child_index: int) -> AccountingState:
        index = int(child_index)
        if index not in (0, 1):
            raise IndexError("binary child index must be zero or one")
        return AccountingState.from_ledger(self.child_ledger[:, index], validate=False)


def _stable_nonnegative_sqrt(values: Any, *, epsilon: float = 1.0e-12) -> Any:
    """Square root with finite zero-boundary gradients for physical invariants."""

    torch = require_torch()
    values = torch.as_tensor(values)
    rooted = torch.sqrt(values.clamp_min(float(epsilon)))
    return torch.where(values > float(epsilon), rooted, torch.zeros_like(rooted))


def _invariant_mass(four_vector: Any) -> Any:
    torch = require_torch()
    p4 = torch.as_tensor(four_vector)
    return _stable_nonnegative_sqrt(
        (
            p4[..., 0].square()
            - p4[..., 1].square()
            - p4[..., 2].square()
            - p4[..., 3].square()
        )
    )


def accounting_state_audit(state: AccountingState) -> dict[str, Any]:
    torch = require_torch()
    problems: list[str] = []
    if state.four_vector.ndim != 2 or state.four_vector.shape[-1] != 4:
        problems.append("four-vector shape is not [B, 4]")
    if bool(((state.constituent_count < 1) | (state.constituent_count > ABPH_MAX_PARTICLES)).any()):
        problems.append("constituent count lies outside [1, 128]")
    if bool((state.type_counts < 0).any()) or not bool(
        (state.type_counts.sum(dim=-1) == state.constituent_count).all()
    ):
        problems.append("type counts do not close to total count")
    expected_floor = minimum_mass_budget(state.type_counts)
    floor_residual = (expected_floor - state.minimum_mass_budget).abs()
    if float(floor_residual.max().detach().cpu()) > 2.0e-6:
        problems.append("minimum-mass ledger does not match type counts")
    charge_mask, _, _ = feasible_charge_mask(state.type_counts)
    charge_index = state.integer_charge + ABPH_MAX_PARTICLES
    in_support = (charge_index >= 0) & (charge_index < charge_mask.shape[-1])
    safe_index = charge_index.clamp(0, charge_mask.shape[-1] - 1)
    charge_valid = charge_mask.gather(1, safe_index[:, None]).squeeze(-1)
    if not bool((in_support & charge_valid).all()):
        problems.append("integer charge is infeasible for the type counts")
    mass = _invariant_mass(state.four_vector)
    mass_tolerance = (
        ABPH_MASS_PRECISION_ABS_TOLERANCE
        + ABPH_MASS_PRECISION_ENERGY_FACTOR * state.four_vector[:, 0].abs()
    )
    if bool((mass + mass_tolerance < expected_floor).any()):
        problems.append("four-vector mass lies below the minimum-mass budget")
    finite = (
        state.four_vector,
        state.minimum_mass_budget,
        state.scalar_sum_pt,
        state.type_energy,
        state.type_scalar_pt,
        state.absolute_charge_sum,
        state.shape_features,
    )
    if not all(bool(torch.isfinite(value).all()) for value in finite):
        problems.append("accounting state contains nonfinite values")
    return {
        "ok": not problems,
        "problems": problems,
        "max_minimum_mass_ledger_residual": float(floor_residual.max().detach().cpu()),
        "minimum_mass_margin_min": float((mass - expected_floor).min().detach().cpu()),
    }


def require_accounting_state(state: AccountingState) -> None:
    report = accounting_state_audit(state)
    if not report["ok"]:
        raise ValueError("invalid accounting state: " + "; ".join(report["problems"]))


def _capped_sigmoid_allocation(logits: Any, capacities: Any, total: Any) -> Any:
    """Differentiable allocation in [0, capacity] with the requested row sum."""

    torch = require_torch()
    scores = torch.as_tensor(logits).float()
    caps = torch.as_tensor(capacities, device=scores.device).float()
    target = torch.as_tensor(total, device=scores.device).float()
    low = torch.full_like(target, -40.0)
    high = torch.full_like(target, 40.0)
    for _ in range(48):
        midpoint = 0.5 * (low + high)
        allocation = caps * torch.sigmoid(scores + midpoint[:, None])
        below = allocation.sum(dim=-1) < target
        low = torch.where(below, midpoint, low)
        high = torch.where(below, high, midpoint)
    allocation = caps * torch.sigmoid(scores + (0.5 * (low + high))[:, None])
    residual = target - allocation.sum(dim=-1)
    room = (caps - allocation).clamp_min(0.0)
    add_weights = room / room.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    remove_weights = allocation / allocation.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    allocation = allocation + residual.clamp_min(0.0)[:, None] * add_weights
    allocation = allocation + residual.clamp_max(0.0)[:, None] * remove_weights
    return allocation.clamp_min(0.0)


def allocate_child_type_counts(
    logits: Any,
    parent_type_counts: Any,
    child_one_count: Any,
) -> tuple[Any, Any]:
    """Exact minimum-cost bounded type allocation plus a straight-through relaxation."""

    torch = require_torch()
    scores = torch.as_tensor(logits).float()
    capacities = torch.as_tensor(parent_type_counts, device=scores.device).to(torch.long)
    total = torch.as_tensor(child_one_count, device=scores.device).to(torch.long)
    if scores.ndim != 2 or scores.shape[-1] != len(ABPH_PID_CATEGORIES):
        raise ValueError("type-allocation logits must have shape [B, 6]")
    if capacities.shape != scores.shape or total.shape != scores.shape[:1]:
        raise ValueError("type-allocation inputs have incompatible shapes")
    parent_total = capacities.sum(dim=-1)
    if bool((capacities < 0).any()) or bool(((total < 1) | (total >= parent_total)).any()):
        raise ValueError("child-one count must be in [1, parent_count - 1]")
    hard_rows: list[Any] = []
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    for batch_index in range(int(scores.shape[0])):
        remaining = int(total[batch_index].item())
        allocation = torch.zeros_like(capacities[batch_index])
        for type_index in order[batch_index].tolist():
            amount = min(remaining, int(capacities[batch_index, type_index].item()))
            allocation[type_index] = amount
            remaining -= amount
            if remaining == 0:
                break
        if remaining:
            raise RuntimeError("exact child type allocator exhausted parent capacities")
        hard_rows.append(allocation)
    hard = torch.stack(hard_rows, dim=0)
    relaxed_soft = _capped_sigmoid_allocation(scores, capacities, total)
    relaxed = hard.to(relaxed_soft.dtype) + relaxed_soft - relaxed_soft.detach()
    if not bool((hard.sum(dim=-1) == total).all()) or bool((hard > capacities).any()):
        raise RuntimeError("child type allocator violated exact closure")
    return hard, relaxed


def _allocate_child_charge(
    logits: Any,
    parent_charge: Any,
    child_type_counts: Any,
    *,
    child_one_charge_override: Any | None = None,
) -> tuple[Any, Any, Any]:
    torch = require_torch()
    scores = torch.as_tensor(logits).float()
    if scores.ndim != 2 or scores.shape[-1] != 2 * ABPH_MAX_PARTICLES + 1:
        raise ValueError("child charge logits must have shape [B, 257]")
    parent = torch.as_tensor(parent_charge, device=scores.device).to(torch.long)
    types = torch.as_tensor(child_type_counts, device=scores.device).to(torch.long)
    mask_one, _, _ = feasible_charge_mask(types[:, 0])
    mask_two, _, _ = feasible_charge_mask(types[:, 1])
    support = torch.arange(-ABPH_MAX_PARTICLES, ABPH_MAX_PARTICLES + 1, device=scores.device)
    second_charge = parent[:, None] - support[None, :]
    second_index = second_charge + ABPH_MAX_PARTICLES
    in_support = (second_index >= 0) & (second_index < mask_two.shape[-1])
    safe_index = second_index.clamp(0, mask_two.shape[-1] - 1)
    second_feasible = mask_two.gather(1, safe_index)
    joint_mask = mask_one & in_support & second_feasible
    if not bool(joint_mask.any(dim=-1).all()):
        raise ValueError("no jointly feasible child charges conserve the parent charge")
    masked_logits = scores.masked_fill(~joint_mask, float("-inf"))
    if child_one_charge_override is None:
        charge_one = support[masked_logits.argmax(dim=-1)]
    else:
        charge_one = torch.as_tensor(child_one_charge_override, device=scores.device).to(torch.long)
        if charge_one.shape != parent.shape:
            raise ValueError("child-one charge override has the wrong shape")
        index = charge_one + ABPH_MAX_PARTICLES
        valid_support = (index >= 0) & (index < joint_mask.shape[-1])
        selected = joint_mask.gather(1, index.clamp(0, joint_mask.shape[-1] - 1)[:, None]).squeeze(-1)
        if not bool((valid_support & selected).all()):
            raise ValueError("child-one charge override is not jointly feasible")
    probabilities = masked_logits.softmax(dim=-1)
    expected_one = (probabilities * support.to(probabilities.dtype)).sum(dim=-1)
    relaxed_one = charge_one.to(expected_one.dtype) + expected_one - expected_one.detach()
    charge_two = parent - charge_one
    relaxed_two = parent.to(relaxed_one.dtype) - relaxed_one
    return torch.stack((charge_one, charge_two), dim=1), torch.stack((relaxed_one, relaxed_two), dim=1), joint_mask


def _inverse_boost_spatial(child: Any, parent: Any) -> Any:
    torch = require_torch()
    beta = parent[:, 1:] / parent[:, :1].clamp_min(1.0e-12)
    beta_squared = beta.square().sum(dim=-1).clamp(max=1.0 - 1.0e-12)
    gamma = 1.0 / torch.sqrt((1.0 - beta_squared).clamp_min(1.0e-12))
    beta_dot_p = (beta * child[:, 1:]).sum(dim=-1)
    coefficient = torch.where(
        beta_squared > 1.0e-14,
        (gamma - 1.0) * beta_dot_p / beta_squared.clamp_min(1.0e-14)
        - gamma * child[:, 0],
        torch.zeros_like(beta_squared),
    )
    return child[:, 1:] + coefficient[:, None] * beta


def two_body_phase_space_split(
    parent_four_vector: Any,
    child_masses: Any,
    direction_raw: Any,
    *,
    collinear_fraction: Any,
    near_massless_threshold: float = ABPH_NEAR_MASSLESS_THRESHOLD_GEV,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Conserve one parent p4 exactly, using a safe collinear lightlike branch."""

    torch = require_torch()
    parent_input = torch.as_tensor(parent_four_vector)
    work_dtype = torch.float64 if parent_input.dtype == torch.float64 else torch.float32
    parent = parent_input.to(work_dtype)
    masses = torch.as_tensor(child_masses, device=parent.device, dtype=work_dtype)
    direction_values = torch.as_tensor(direction_raw, device=parent.device, dtype=work_dtype)
    fraction = torch.as_tensor(collinear_fraction, device=parent.device, dtype=work_dtype)
    if parent.ndim != 2 or parent.shape[-1] != 4 or masses.shape != (parent.shape[0], 2):
        raise ValueError("two-body inputs require parent [B,4] and child masses [B,2]")
    if direction_values.shape != (parent.shape[0], 3) or fraction.shape != parent.shape[:1]:
        raise ValueError("two-body direction/fraction shapes do not match the parent batch")
    if bool((parent[:, 0] <= 0.0).any()) or not bool(torch.isfinite(parent).all()):
        raise ValueError("parent four-vectors must have finite positive energy")
    parent_mass = _invariant_mass(parent)
    if bool((masses < 0.0).any()) or bool((masses.sum(dim=-1) > parent_mass + 3.0e-5).any()):
        raise ValueError("child masses are infeasible for the parent invariant mass")
    near_massless = parent_mass <= float(near_massless_threshold)
    if bool((near_massless & (masses.max(dim=-1).values > 3.0e-5)).any()):
        raise ValueError("near-massless parents cannot carry massive children")

    direction_norm = torch.linalg.vector_norm(direction_values, dim=-1, keepdim=True)
    fallback = torch.zeros_like(direction_values)
    fallback[:, 0] = 1.0
    direction = torch.where(
        direction_norm > 1.0e-10,
        direction_values / direction_norm.clamp_min(1.0e-10),
        fallback,
    )
    safe_mass = parent_mass.clamp_min(float(near_massless_threshold))
    mass_one, mass_two = masses.unbind(dim=-1)
    kallen = (
        (safe_mass.square() - (mass_one + mass_two).square())
        * (safe_mass.square() - (mass_one - mass_two).square())
    ).clamp_min(0.0)
    momentum = _stable_nonnegative_sqrt(kallen) / (2.0 * safe_mass)
    energy_one_rest = (safe_mass.square() + mass_one.square() - mass_two.square()) / (2.0 * safe_mass)
    momentum_one_rest = momentum[:, None] * direction

    beta = parent[:, 1:] / parent[:, :1].clamp_min(1.0e-12)
    beta_squared = beta.square().sum(dim=-1).clamp(max=1.0 - 1.0e-12)
    gamma = 1.0 / torch.sqrt((1.0 - beta_squared).clamp_min(1.0e-12))
    beta_dot_p = (beta * momentum_one_rest).sum(dim=-1)
    energy_one_lab = gamma * (energy_one_rest + beta_dot_p)
    coefficient = torch.where(
        beta_squared > 1.0e-14,
        (gamma - 1.0) * beta_dot_p / beta_squared.clamp_min(1.0e-14)
        + gamma * energy_one_rest,
        torch.zeros_like(beta_squared),
    )
    spatial_one_lab = momentum_one_rest + coefficient[:, None] * beta
    regular_one = torch.cat((energy_one_lab[:, None], spatial_one_lab), dim=-1)

    alpha = fraction.clamp(1.0e-6, 1.0 - 1.0e-6)
    collinear_one = parent * alpha[:, None]
    child_one = torch.where(near_massless[:, None], collinear_one, regular_one)
    child_two = parent - child_one
    closure = (child_one + child_two - parent).abs().amax(dim=-1)
    diagnostics = {
        "near_massless_mask": near_massless,
        "near_massless_count": int(near_massless.sum().detach().cpu()),
        "branch": "two_body_with_collinear_safe_lightlike_limit",
        "max_four_vector_residual": float(closure.max().detach().cpu()),
    }
    return child_one, child_two, diagnostics


def _prediction_shapes(prediction: BinarySplitPrediction, batch: int) -> None:
    expected = {
        "topology_logits": (batch, 2),
        "count_logits": (batch, ABPH_BINARY_COUNT_SUPPORT),
        "type_allocation_logits": (batch, len(ABPH_PID_CATEGORIES)),
        "charge_logits": (batch, 2 * ABPH_MAX_PARTICLES + 1),
        "mass_allocation_logits": (batch, 3),
        "direction_raw": (batch, 3),
        "collinear_fraction_raw": (batch,),
        "auxiliary_fraction_logits": (batch, len(ABPH_AUXILIARY_ADDITIVE_NAMES)),
        "child_shape_raw": (batch, 2, len(ROOT_SHAPE_FEATURE_NAMES) + 1),
    }
    for name, shape in expected.items():
        if tuple(getattr(prediction, name).shape) != shape:
            raise ValueError(f"{name} shape {tuple(getattr(prediction, name).shape)} != {shape}")


def _empty_children(parent: AccountingState, prediction: BinarySplitPrediction) -> dict[str, Any]:
    torch = require_torch()
    batch = parent.batch_size
    dtype = parent.four_vector.dtype
    device = parent.four_vector.device
    zero_anchor = sum(value.sum() for value in (
        prediction.count_logits,
        prediction.type_allocation_logits,
        prediction.charge_logits,
        prediction.mass_allocation_logits,
        prediction.direction_raw,
        prediction.auxiliary_fraction_logits,
        prediction.child_shape_raw,
    )) * 0.0
    return {
        "p4": torch.zeros((batch, 2, 4), dtype=dtype, device=device) + zero_anchor,
        "count": torch.zeros((batch, 2), dtype=torch.long, device=device),
        "relaxed_count": torch.zeros((batch, 2), dtype=dtype, device=device) + zero_anchor,
        "types": torch.zeros((batch, 2, len(ABPH_PID_CATEGORIES)), dtype=torch.long, device=device),
        "relaxed_types": torch.zeros((batch, 2, len(ABPH_PID_CATEGORIES)), dtype=dtype, device=device) + zero_anchor,
        "charge": torch.zeros((batch, 2), dtype=torch.long, device=device),
        "relaxed_charge": torch.zeros((batch, 2), dtype=dtype, device=device) + zero_anchor,
        "floor": torch.zeros((batch, 2), dtype=dtype, device=device) + zero_anchor,
        "scalar_pt": torch.zeros((batch, 2), dtype=dtype, device=device) + zero_anchor,
        "type_energy": torch.zeros((batch, 2, len(ABPH_PID_CATEGORIES)), dtype=dtype, device=device) + zero_anchor,
        "type_pt": torch.zeros((batch, 2, len(ABPH_PID_CATEGORIES)), dtype=dtype, device=device) + zero_anchor,
        "abs_charge": torch.zeros((batch, 2), dtype=dtype, device=device) + zero_anchor,
        "shape": torch.zeros((batch, 2, len(ROOT_SHAPE_FEATURE_NAMES)), dtype=dtype, device=device) + zero_anchor,
    }


def _index_copy(base: Any, indices: Any, values: Any) -> Any:
    return base.index_copy(0, indices, values.to(dtype=base.dtype, device=base.device))


def _child_ledgers(values: Mapping[str, Any], child_charge_bounds: Any) -> Any:
    torch = require_torch()
    p4 = values["p4"]
    ledger = torch.zeros(
        (p4.shape[0], 2, len(ROOT_FEATURE_NAMES)), dtype=p4.dtype, device=p4.device
    )

    def put(name: str, value: Any) -> None:
        ledger[:, :, ROOT_FEATURE_INDEX[name]] = value.to(ledger.dtype)

    for index, name in enumerate(("energy", "px", "py", "pz")):
        put(name, p4[:, :, index])
    put("constituent_count", values["count"])
    for index, name in enumerate(ABPH_PID_CATEGORIES):
        put(f"count_{name}", values["types"][:, :, index])
        put(f"energy_{name}", values["type_energy"][:, :, index])
        put(f"scalar_pt_{name}", values["type_pt"][:, :, index])
    put("integer_charge", values["charge"])
    put("minimum_mass_budget", values["floor"])
    put("feasible_charge_min", child_charge_bounds[:, :, 0])
    put("feasible_charge_max", child_charge_bounds[:, :, 1])
    put("scalar_sum_pt", values["scalar_pt"])
    put("absolute_charge_sum", values["abs_charge"])
    for index, name in enumerate(ROOT_SHAPE_FEATURE_NAMES):
        put(name, values["shape"][:, :, index])
    return ledger


def compile_binary_split(
    parent: AccountingState,
    prediction: BinarySplitPrediction,
    *,
    topology_override: Any | None = None,
    child_one_count_override: Any | None = None,
    child_one_type_counts_override: Any | None = None,
    child_one_charge_override: Any | None = None,
    child_four_vector_override: Any | None = None,
) -> CompiledBinarySplit:
    """Compile topology through p4 in one fail-closed, parent-conserving operation."""

    torch = require_torch()
    require_accounting_state(parent)
    batch = parent.batch_size
    _prediction_shapes(prediction, batch)
    device = parent.four_vector.device
    if topology_override is None:
        topology = torch.where(
            parent.constituent_count == 1,
            torch.full_like(parent.constituent_count, int(TOPOLOGY_ACTIVE_TERMINAL)),
            prediction.topology_logits.argmax(dim=-1).to(torch.long) + 1,
        )
    else:
        topology = torch.as_tensor(topology_override, device=device).to(torch.long)
        if topology.shape != (batch,) or not bool(
            ((topology == int(TOPOLOGY_ACTIVE_TERMINAL)) | (topology == int(TOPOLOGY_ACTIVE_SPLIT))).all()
        ):
            raise ValueError("topology override must contain only terminal or split states")
        if bool(((parent.constituent_count == 1) & (topology == int(TOPOLOGY_ACTIVE_SPLIT))).any()):
            raise ValueError("a singleton parent cannot be split")
    split_mask = topology == int(TOPOLOGY_ACTIVE_SPLIT)
    learned_split_probability = prediction.topology_logits.float().softmax(dim=-1)[:, 1]
    learned_split_probability = torch.where(
        parent.constituent_count == 1,
        torch.zeros_like(learned_split_probability),
        learned_split_probability,
    )
    relaxed_split_probability = (
        split_mask.to(learned_split_probability.dtype)
        + learned_split_probability
        - learned_split_probability.detach()
    )
    split_indices = torch.nonzero(split_mask, as_tuple=False).squeeze(-1)
    values = _empty_children(parent, prediction)
    near_massless_full = torch.zeros(batch, dtype=torch.bool, device=device)
    joint_charge_mask_full = torch.zeros(
        (batch, 2 * ABPH_MAX_PARTICLES + 1), dtype=torch.bool, device=device
    )
    if int(split_indices.numel()):
        parent_count = parent.constituent_count[split_indices]
        count_logits = prediction.count_logits[split_indices].float()
        support = torch.arange(1, ABPH_BINARY_COUNT_SUPPORT + 1, device=device)
        count_mask = support[None, :] < parent_count[:, None]
        if not bool(count_mask.any(dim=-1).all()):
            raise RuntimeError("split parent has no valid child count")
        masked_count_logits = count_logits.masked_fill(~count_mask, float("-inf"))
        if child_one_count_override is None:
            count_one = masked_count_logits.argmax(dim=-1).to(torch.long) + 1
        else:
            supplied = torch.as_tensor(child_one_count_override, device=device).to(torch.long)
            if supplied.shape == (batch,):
                supplied = supplied[split_indices]
            if supplied.shape != parent_count.shape or bool(
                ((supplied < 1) | (supplied >= parent_count)).any()
            ):
                raise ValueError("child-one count override is outside the parent budget")
            count_one = supplied
        count_probabilities = masked_count_logits.softmax(dim=-1)
        expected_one = (count_probabilities * support.to(count_probabilities.dtype)).sum(dim=-1)
        relaxed_one = count_one.to(expected_one.dtype) + expected_one - expected_one.detach()
        count_two = parent_count - count_one
        relaxed_two = parent_count.to(expected_one.dtype) - relaxed_one
        hard_count = torch.stack((count_one, count_two), dim=1)
        relaxed_count = torch.stack((relaxed_one, relaxed_two), dim=1)

        parent_types = parent.type_counts[split_indices]
        if child_one_type_counts_override is None:
            type_one, relaxed_type_one = allocate_child_type_counts(
                prediction.type_allocation_logits[split_indices], parent_types, count_one
            )
        else:
            supplied_types = torch.as_tensor(child_one_type_counts_override, device=device).to(torch.long)
            if supplied_types.shape == (batch, len(ABPH_PID_CATEGORIES)):
                supplied_types = supplied_types[split_indices]
            if supplied_types.shape != parent_types.shape:
                raise ValueError("child-one type override has the wrong shape")
            if bool((supplied_types < 0).any()) or bool((supplied_types > parent_types).any()) or not bool(
                (supplied_types.sum(dim=-1) == count_one).all()
            ):
                raise ValueError("child-one type override violates count or parent capacities")
            type_one = supplied_types
            soft_one = _capped_sigmoid_allocation(
                prediction.type_allocation_logits[split_indices], parent_types, count_one
            )
            relaxed_type_one = type_one.to(soft_one.dtype) + soft_one - soft_one.detach()
        type_two = parent_types - type_one
        relaxed_type_two = parent_types.to(relaxed_type_one.dtype) - relaxed_type_one
        hard_types = torch.stack((type_one, type_two), dim=1)
        relaxed_types = torch.stack((relaxed_type_one, relaxed_type_two), dim=1)

        charge_override = child_one_charge_override
        if charge_override is not None:
            charge_override = torch.as_tensor(charge_override, device=device)
            if charge_override.shape == (batch,):
                charge_override = charge_override[split_indices]
        hard_charge, relaxed_charge, joint_charge_mask = _allocate_child_charge(
            prediction.charge_logits[split_indices],
            parent.integer_charge[split_indices],
            hard_types,
            child_one_charge_override=charge_override,
        )
        floors = minimum_mass_budget(hard_types.reshape(-1, len(ABPH_PID_CATEGORIES))).reshape(-1, 2)
        parent_p4 = parent.four_vector[split_indices]
        parent_mass = _invariant_mass(parent_p4)
        floor_sum = floors.sum(dim=-1)
        available = parent_mass - floor_sum
        parent_mass_tolerance = (
            ABPH_MASS_PRECISION_ABS_TOLERANCE
            + ABPH_MASS_PRECISION_ENERGY_FACTOR * parent_p4[:, 0].abs()
        )
        if bool((available < -parent_mass_tolerance).any()):
            raise ValueError("child type allocations imply mass floors above the parent mass")
        # A highly boosted FP32 p4 can recover an invariant mass a few ulps
        # below its exact additive floor. The audit explicitly accepts that
        # representational error, so the phase-space input must use the same
        # contract rather than pass an impossible floor sum to a stricter
        # downstream check. Only tolerance-accepted rows are projected, and
        # their hard minimum-mass ledger remains unchanged for supervision.
        representable_scale = torch.where(
            floor_sum > parent_mass,
            parent_mass / floor_sum.clamp_min(1.0e-12),
            torch.ones_like(parent_mass),
        )
        phase_space_floors = floors * representable_scale[:, None]
        available = (parent_mass - phase_space_floors.sum(dim=-1)).clamp_min(0.0)
        mass_fractions = prediction.mass_allocation_logits[split_indices].float().softmax(dim=-1)
        masses = phase_space_floors + available[:, None] * mass_fractions[:, :2]
        direction = prediction.direction_raw[split_indices]
        collinear_fraction = torch.sigmoid(
            prediction.collinear_fraction_raw[split_indices].float()
        )
        target_p4 = None
        if child_four_vector_override is not None:
            target_p4 = torch.as_tensor(
                child_four_vector_override,
                device=device,
                dtype=parent_p4.dtype,
            )
            if target_p4.shape == (batch, 2, 4):
                target_p4 = target_p4[split_indices]
            if target_p4.shape != (len(split_indices), 2, 4):
                raise ValueError("child four-vector override has the wrong shape")
            closure = (target_p4.sum(dim=1) - parent_p4).abs()
            tolerance = ABPH_BINARY_P4_ABS_TOLERANCE + ABPH_BINARY_P4_REL_TOLERANCE * parent_p4.abs()
            if bool((closure > tolerance).any()):
                raise ValueError("child four-vector override does not conserve the parent")
            target_masses = _invariant_mass(target_p4)
            target_mass_tolerance = (
                ABPH_MASS_PRECISION_ABS_TOLERANCE
                + ABPH_MASS_PRECISION_ENERGY_FACTOR * target_p4[:, :, 0].abs()
            )
            if bool((target_masses + target_mass_tolerance < floors).any()) or bool(
                (target_masses.sum(dim=-1) > parent_mass + parent_mass_tolerance).any()
            ):
                raise ValueError("child four-vector override violates mass feasibility")
        if target_p4 is None:
            p4_one, p4_two, phase_diagnostics = two_body_phase_space_split(
                parent_p4,
                masses,
                direction,
                collinear_fraction=collinear_fraction,
            )
            child_p4 = torch.stack((p4_one, p4_two), dim=1)
        else:
            # An override is used only by the mandatory target-replay audit. It
            # must preserve the already validated target p4 exactly; converting
            # a boosted target through inverse/forward Lorentz transforms can
            # catastrophically lose its small invariant mass through numerical
            # cancellation and no longer tests the supplied target.
            child_p4 = target_p4
            target_mass = _invariant_mass(parent_p4)
            phase_diagnostics = {
                "near_massless_mask": target_mass <= float(ABPH_NEAR_MASSLESS_THRESHOLD_GEV),
                "near_massless_count": int(
                    (target_mass <= float(ABPH_NEAR_MASSLESS_THRESHOLD_GEV)).sum().detach().cpu()
                ),
                "branch": "validated_exact_target_override",
                "max_four_vector_residual": float(
                    (child_p4.sum(dim=1) - parent_p4).abs().max().detach().cpu()
                ),
            }

        parent_additive = torch.stack(
            tuple(parent.ledger[split_indices, ROOT_FEATURE_INDEX[name]] for name in ABPH_AUXILIARY_ADDITIVE_NAMES),
            dim=-1,
        )
        auxiliary_fraction = torch.sigmoid(
            prediction.auxiliary_fraction_logits[split_indices].float()
        )
        additive_children = torch.stack(
            (parent_additive * auxiliary_fraction, parent_additive * (1.0 - auxiliary_fraction)),
            dim=1,
        )
        shape = compile_shape_features(
            prediction.child_shape_raw[split_indices].reshape(-1, len(ROOT_SHAPE_FEATURE_NAMES) + 1)
        ).reshape(-1, 2, len(ROOT_SHAPE_FEATURE_NAMES))
        scalar_pt = additive_children[:, :, 0]
        type_energy = additive_children[:, :, 1 : 1 + len(ABPH_PID_CATEGORIES)]
        type_pt_start = 1 + len(ABPH_PID_CATEGORIES)
        type_pt = additive_children[:, :, type_pt_start : type_pt_start + len(ABPH_PID_CATEGORIES)]
        abs_charge = additive_children[:, :, -1]

        for name, subvalue in (
            ("p4", child_p4),
            ("count", hard_count),
            ("relaxed_count", relaxed_count),
            ("types", hard_types),
            ("relaxed_types", relaxed_types),
            ("charge", hard_charge),
            ("relaxed_charge", relaxed_charge),
            ("floor", floors),
            ("scalar_pt", scalar_pt),
            ("type_energy", type_energy),
            ("type_pt", type_pt),
            ("abs_charge", abs_charge),
            ("shape", shape),
        ):
            values[name] = _index_copy(values[name], split_indices, subvalue)
        near_massless_full = near_massless_full.index_copy(
            0, split_indices, phase_diagnostics["near_massless_mask"]
        )
        joint_charge_mask_full = joint_charge_mask_full.index_copy(
            0, split_indices, joint_charge_mask
        )

    child_mask = split_mask[:, None].expand(-1, 2)
    child_bounds = torch.zeros((batch, 2, 2), dtype=parent.four_vector.dtype, device=device)
    if int(split_indices.numel()):
        child_types_selected = values["types"][split_indices]
        bounds_rows = []
        for child_index in range(2):
            _, lower, upper = feasible_charge_mask(child_types_selected[:, child_index])
            bounds_rows.append(torch.stack((lower, upper), dim=-1))
        child_bounds = _index_copy(child_bounds, split_indices, torch.stack(bounds_rows, dim=1))
    child_ledger = _child_ledgers(values, child_bounds)
    diagnostics = binary_accounting_audit(
        parent=parent,
        topology=topology,
        split_mask=split_mask,
        child_four_vector=values["p4"],
        child_count=values["count"],
        child_type_counts=values["types"],
        child_charge=values["charge"],
        child_minimum_mass=values["floor"],
        child_scalar_pt=values["scalar_pt"],
        child_type_energy=values["type_energy"],
        child_type_pt=values["type_pt"],
        near_massless_mask=near_massless_full,
    )
    diagnostics["joint_charge_support_nonempty"] = bool(
        joint_charge_mask_full[split_mask].any(dim=-1).all()
    ) if bool(split_mask.any()) else True
    diagnostics["phase_space_branch"] = (
        phase_diagnostics["branch"] if int(split_indices.numel()) else "no_split"
    )
    if not diagnostics["ok"]:
        raise RuntimeError("binary accounting compiler failed closed: " + "; ".join(diagnostics["problems"]))
    return CompiledBinarySplit(
        topology=topology,
        relaxed_split_probability=relaxed_split_probability,
        split_mask=split_mask,
        child_mask=child_mask,
        child_four_vector=values["p4"],
        child_constituent_count=values["count"],
        relaxed_child_constituent_count=values["relaxed_count"],
        child_type_counts=values["types"],
        relaxed_child_type_counts=values["relaxed_types"],
        child_integer_charge=values["charge"],
        relaxed_child_charge=values["relaxed_charge"],
        child_minimum_mass_budget=values["floor"],
        child_scalar_sum_pt=values["scalar_pt"],
        child_type_energy=values["type_energy"],
        child_type_scalar_pt=values["type_pt"],
        child_absolute_charge_sum=values["abs_charge"],
        child_shape_features=values["shape"],
        child_ledger=child_ledger,
        diagnostics=diagnostics,
    )


def binary_accounting_audit(
    *,
    parent: AccountingState,
    topology: Any,
    split_mask: Any,
    child_four_vector: Any,
    child_count: Any,
    child_type_counts: Any,
    child_charge: Any,
    child_minimum_mass: Any,
    child_scalar_pt: Any,
    child_type_energy: Any,
    child_type_pt: Any,
    near_massless_mask: Any,
) -> dict[str, Any]:
    torch = require_torch()
    split = torch.as_tensor(split_mask).bool()
    problems: list[str] = []
    if bool(((parent.constituent_count == 1) & split).any()):
        problems.append("singleton parent was split")
    if bool(split.any()):
        p4_residual = child_four_vector[split].sum(dim=1) - parent.four_vector[split]
        count_residual = child_count[split].sum(dim=1) - parent.constituent_count[split]
        type_residual = child_type_counts[split].sum(dim=1) - parent.type_counts[split]
        charge_residual = child_charge[split].sum(dim=1) - parent.integer_charge[split]
        child_mass = _invariant_mass(child_four_vector[split])
        mass_floor_margin = child_mass - child_minimum_mass[split]
        parent_mass = _invariant_mass(parent.four_vector[split])
        mass_sum_margin = parent_mass - child_mass.sum(dim=-1)
        p4_tolerance = ABPH_BINARY_P4_ABS_TOLERANCE + ABPH_BINARY_P4_REL_TOLERANCE * parent.four_vector[split].abs()
        if bool((p4_residual.abs() > p4_tolerance).any()):
            problems.append("child four-vectors do not conserve the parent")
        if bool((count_residual != 0).any()):
            problems.append("child counts do not conserve the parent")
        if bool((type_residual != 0).any()):
            problems.append("child type counts do not conserve the parent")
        if bool((charge_residual != 0).any()):
            problems.append("child charges do not conserve the parent")
        child_mass_tolerance = (
            ABPH_MASS_PRECISION_ABS_TOLERANCE
            + ABPH_MASS_PRECISION_ENERGY_FACTOR * child_four_vector[split, :, 0].abs()
        )
        parent_mass_tolerance = (
            ABPH_MASS_PRECISION_ABS_TOLERANCE
            + ABPH_MASS_PRECISION_ENERGY_FACTOR * parent.four_vector[split, 0].abs()
        )
        if bool((mass_floor_margin < -child_mass_tolerance).any()):
            problems.append("child mass lies below its minimum budget")
        mass_sum_tolerance = (
            parent_mass_tolerance + child_mass_tolerance.sum(dim=-1)
        )
        if bool((mass_sum_margin < -mass_sum_tolerance).any()):
            problems.append("child masses exceed the parent mass")
        max_p4 = float(p4_residual.abs().max().detach().cpu())
        mean_p4 = float(p4_residual.abs().mean().detach().cpu())
        max_count = int(count_residual.abs().max().detach().cpu())
        max_type = int(type_residual.abs().max().detach().cpu())
        max_charge = int(charge_residual.abs().max().detach().cpu())
        minimum_floor_margin = float(mass_floor_margin.min().detach().cpu())
        minimum_mass_sum_margin = float(mass_sum_margin.min().detach().cpu())
        minimum_mass_sum_tolerance_margin = float(
            (mass_sum_margin + mass_sum_tolerance).min().detach().cpu()
        )
        scalar_pt_residual = child_scalar_pt[split].sum(dim=1) - parent.scalar_sum_pt[split]
        type_energy_residual = child_type_energy[split].sum(dim=1) - parent.type_energy[split]
        type_pt_residual = child_type_pt[split].sum(dim=1) - parent.type_scalar_pt[split]
    else:
        zero = parent.four_vector.sum() * 0.0
        max_p4 = mean_p4 = float(zero.detach().cpu())
        max_count = max_type = max_charge = 0
        minimum_floor_margin = minimum_mass_sum_margin = 0.0
        minimum_mass_sum_tolerance_margin = 0.0
        scalar_pt_residual = zero.reshape(1)
        type_energy_residual = zero.reshape(1)
        type_pt_residual = zero.reshape(1)
    finite_values = (
        child_four_vector,
        child_minimum_mass,
        child_scalar_pt,
        child_type_energy,
        child_type_pt,
    )
    if not all(bool(torch.isfinite(value).all()) for value in finite_values):
        problems.append("binary compiler produced nonfinite values")
    return {
        "ok": not problems,
        "contract": ABPH_BINARY_ACCOUNTING_CONTRACT,
        "compiler_hash": binary_accounting_manifest()["compiler_hash"],
        "problems": problems,
        "compiler_failure_count": int(bool(problems)),
        "n_split": int(split.sum().detach().cpu()),
        "n_terminal": int((topology == int(TOPOLOGY_ACTIVE_TERMINAL)).sum().detach().cpu()),
        "near_massless_count": int(torch.as_tensor(near_massless_mask).sum().detach().cpu()),
        "hard": {
            "max_four_vector_residual": max_p4,
            "mean_four_vector_residual": mean_p4,
            "max_count_residual": max_count,
            "max_type_count_residual": max_type,
            "max_charge_residual": max_charge,
            "minimum_mass_floor_margin": minimum_floor_margin,
            "minimum_parent_mass_remainder": minimum_mass_sum_margin,
            "minimum_parent_mass_tolerance_remainder": minimum_mass_sum_tolerance_margin,
        },
        "soft": {
            "scalar_pt_consistency_mae": float(scalar_pt_residual.abs().mean().detach().cpu()),
            "type_energy_consistency_mae": float(type_energy_residual.abs().mean().detach().cpu()),
            "type_pt_consistency_mae": float(type_pt_residual.abs().mean().detach().cpu()),
        },
    }


__all__ = [
    "ABPH_AUXILIARY_ADDITIVE_NAMES",
    "ABPH_BINARY_ACCOUNTING_CONTRACT",
    "ABPH_BINARY_ACCOUNTING_VERSION",
    "ABPH_BINARY_COUNT_SUPPORT",
    "ABPH_BINARY_P4_ABS_TOLERANCE",
    "ABPH_BINARY_P4_REL_TOLERANCE",
    "ABPH_NEAR_MASSLESS_THRESHOLD_GEV",
    "ABPH_MASS_PRECISION_ABS_TOLERANCE",
    "ABPH_MASS_PRECISION_ENERGY_FACTOR",
    "AccountingState",
    "BinarySplitPrediction",
    "CompiledBinarySplit",
    "accounting_state_audit",
    "allocate_child_type_counts",
    "binary_accounting_audit",
    "binary_accounting_manifest",
    "compile_binary_split",
    "require_accounting_state",
    "two_body_phase_space_split",
]
