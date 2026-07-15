"""Ordered exact compiler from probabilistic root heads to a feasible ledger."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .root_model import (
    ABPH_CHARGE_SUPPORT_MIN,
    ABPH_CHARGE_SUPPORT_SIZE,
    SemanticRootPrediction,
)
from .root_transforms import (
    ABPH_ROOT_EPSILON,
    ROOT_FEATURE_INDEX,
    ROOT_RESIDUAL_CHANNEL_NAMES,
    ROOT_SHAPE_FEATURE_NAMES,
    RootNormalizationStats,
    RootPhysicalKinematics,
    root_residual_to_physical,
    summarize_hlt_root,
)
from .schemas import ABPH_EFFECTIVE_MASS_GEV, ABPH_MAX_PARTICLES, ABPH_PID_CATEGORIES
from .targets import ROOT_FEATURE_NAMES


ABPH_ROOT_COMPILER_CONTRACT = "adaptive_binary_pseudooffline_root_compiler_v1"
ABPH_ROOT_COMPILER_VERSION = "v1"
def root_compiler_manifest() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contract": ABPH_ROOT_COMPILER_CONTRACT,
        "version": ABPH_ROOT_COMPILER_VERSION,
        "compiler_order": [
            "constituent_count",
            "integer_type_counts",
            "feasible_integer_charge",
            "type_conditioned_minimum_mass",
            "physical_four_vector",
            "auxiliary_ledger",
        ],
        "effective_mass_gev": dict(ABPH_EFFECTIVE_MASS_GEV),
        "charge_support": {
            "charged_hadron_electron_muon": [-1, 1],
            "neutral_hadron_photon": [0],
            "other": [-1, 0, 1],
        },
        "type_allocation": "deterministic_largest_remainder_with_straight_through_relaxation",
        "count_support": [1, ABPH_MAX_PARTICLES],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["compiler_hash"] = hashlib.sha256(encoded).hexdigest()
    return payload


@dataclass(frozen=True)
class CompiledRootState:
    kinematics: RootPhysicalKinematics
    constituent_count: Any
    relaxed_constituent_count: Any
    type_counts: Any
    relaxed_type_counts: Any
    integer_charge: Any
    relaxed_charge: Any
    feasible_charge_mask: Any
    feasible_charge_min: Any
    feasible_charge_max: Any
    minimum_mass_budget: Any
    relaxed_minimum_mass_budget: Any
    type_count_fractions: Any
    type_pt_fractions: Any
    type_energy_fractions: Any
    scalar_sum_pt: Any
    shape_features: Any
    root_ledger: Any
    diagnostics: Mapping[str, Any]

    def hard_four_vector(self) -> Any:
        return self.kinematics.four_vector()


def _validate_probabilities(probabilities: Any, *, label: str) -> None:
    torch = require_torch()
    if not bool(torch.isfinite(probabilities).all()):
        raise FloatingPointError(f"{label} probabilities are nonfinite")
    if bool((probabilities < 0.0).any()):
        raise ValueError(f"{label} probabilities are negative")
    sums = probabilities.sum(dim=-1)
    if not bool(torch.allclose(sums, torch.ones_like(sums), atol=2.0e-6, rtol=2.0e-6)):
        raise ValueError(f"{label} probabilities do not sum to one")


def allocate_integer_type_counts(
    probabilities: Any,
    total_count: Any,
) -> tuple[Any, Any]:
    """Largest-remainder allocation with exact sums and straight-through gradients."""

    torch = require_torch()
    probs = torch.as_tensor(probabilities).float()
    counts = torch.as_tensor(total_count, device=probs.device).to(torch.long)
    if probs.ndim != 2 or probs.shape[-1] != len(ABPH_PID_CATEGORIES):
        raise ValueError("type probabilities must have shape [B, 6]")
    if counts.shape != probs.shape[:1]:
        raise ValueError("total_count shape does not match type probabilities")
    if bool(((counts < 1) | (counts > ABPH_MAX_PARTICLES)).any()):
        raise ValueError("total counts must lie in [1, 128]")
    _validate_probabilities(probs, label="type")
    raw = probs * counts.to(probs.dtype).unsqueeze(-1)
    floors = torch.floor(raw).to(torch.long)
    hard = floors.clone()
    remainders = raw - floors.to(raw.dtype)
    missing = counts - floors.sum(dim=-1)
    order = torch.argsort(remainders, dim=-1, descending=True, stable=True)
    for batch_index in range(int(probs.shape[0])):
        amount = int(missing[batch_index].item())
        if amount < 0 or amount > len(ABPH_PID_CATEGORIES):
            raise RuntimeError("largest-remainder type allocation produced an invalid remainder")
        if amount:
            hard[batch_index, order[batch_index, :amount]] += 1
    if not bool((hard.sum(dim=-1) == counts).all()) or bool((hard < 0).any()):
        raise RuntimeError("integer type allocator failed exact count closure")
    relaxed = hard.to(raw.dtype) + raw - raw.detach()
    return hard, relaxed


def feasible_charge_mask(type_counts: Any) -> tuple[Any, Any, Any]:
    """Enumerate exact root charges realizable by the selected particle types."""

    torch = require_torch()
    counts = torch.as_tensor(type_counts).to(torch.long)
    if counts.ndim != 2 or counts.shape[-1] != len(ABPH_PID_CATEGORIES):
        raise ValueError("type_counts must have shape [B, 6]")
    if bool((counts < 0).any()):
        raise ValueError("type_counts cannot be negative")
    fixed = counts[:, 0] + counts[:, 3] + counts[:, 4]
    other = counts[:, 5]
    support = torch.arange(
        ABPH_CHARGE_SUPPORT_MIN,
        ABPH_CHARGE_SUPPORT_MIN + ABPH_CHARGE_SUPPORT_SIZE,
        device=counts.device,
        dtype=torch.long,
    )
    q = support.unsqueeze(0)
    lower = torch.maximum(-fixed.unsqueeze(-1), q - other.unsqueeze(-1))
    upper = torch.minimum(fixed.unsqueeze(-1), q + other.unsqueeze(-1))
    first_matching_parity = lower + torch.remainder(
        fixed.unsqueeze(-1) - lower, 2
    )
    feasible = first_matching_parity <= upper
    minimum = -(fixed + other)
    maximum = fixed + other
    if not bool(feasible.any(dim=-1).all()):
        raise RuntimeError("charge compiler produced an empty feasible support")
    return feasible, minimum, maximum


def select_feasible_charge(
    charge_logits: Any,
    type_counts: Any,
    *,
    charge_override: Any | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    """Select an exact feasible integer charge and a straight-through relaxation."""

    torch = require_torch()
    logits = torch.as_tensor(charge_logits).float()
    if logits.ndim != 2 or logits.shape[-1] != ABPH_CHARGE_SUPPORT_SIZE:
        raise ValueError(f"charge logits must have shape [B, {ABPH_CHARGE_SUPPORT_SIZE}]")
    feasible, minimum, maximum = feasible_charge_mask(type_counts)
    masked_logits = logits.masked_fill(~feasible, float("-inf"))
    support = torch.arange(
        ABPH_CHARGE_SUPPORT_MIN,
        ABPH_CHARGE_SUPPORT_MIN + ABPH_CHARGE_SUPPORT_SIZE,
        device=logits.device,
        dtype=torch.long,
    )
    if charge_override is None:
        hard = support[masked_logits.argmax(dim=-1)]
    else:
        hard = torch.as_tensor(charge_override, device=logits.device).to(torch.long)
        if hard.shape != logits.shape[:1]:
            raise ValueError("charge override shape does not match the batch")
        indices = hard - ABPH_CHARGE_SUPPORT_MIN
        in_support = (indices >= 0) & (indices < ABPH_CHARGE_SUPPORT_SIZE)
        safe_indices = indices.clamp(0, ABPH_CHARGE_SUPPORT_SIZE - 1)
        selected_feasible = feasible.gather(1, safe_indices.unsqueeze(-1)).squeeze(-1)
        if not bool((in_support & selected_feasible).all()):
            raise ValueError("charge override is infeasible for the selected type counts")
    probabilities = masked_logits.softmax(dim=-1)
    expected = (probabilities * support.to(probabilities.dtype)).sum(dim=-1)
    relaxed = hard.to(expected.dtype) + expected - expected.detach()
    return hard, relaxed, feasible, minimum, maximum


def minimum_mass_budget(type_counts: Any) -> Any:
    torch = require_torch()
    counts_raw = torch.as_tensor(type_counts)
    dtype = torch.float64 if counts_raw.dtype == torch.float64 else torch.float32
    counts = counts_raw.to(dtype)
    masses = torch.as_tensor(
        tuple(ABPH_EFFECTIVE_MASS_GEV[name] for name in ABPH_PID_CATEGORIES),
        dtype=counts.dtype,
        device=counts.device,
    )
    return (counts * masses).sum(dim=-1)


def compile_shape_features(shape_raw: Any) -> Any:
    """Map anonymous shape outputs to PSD, ordered, bounded physical features."""

    torch = require_torch()
    raw = torch.as_tensor(shape_raw).float()
    if raw.ndim != 2 or raw.shape[-1] != len(ROOT_SHAPE_FEATURE_NAMES) + 1:
        raise ValueError(
            f"shape_raw must have shape [B, {len(ROOT_SHAPE_FEATURE_NAMES) + 1}]"
        )
    eta_first = raw[:, 0]
    phi_first = torch.pi * torch.tanh(raw[:, 1])
    eta_second = torch.nn.functional.softplus(raw[:, 2])
    phi_second = torch.nn.functional.softplus(raw[:, 3])
    correlation = torch.tanh(raw[:, 4])
    cross = correlation * torch.sqrt((eta_second * phi_second).clamp_min(0.0))
    radial_first = torch.nn.functional.softplus(raw[:, 5])
    radial_second = radial_first.square() + torch.nn.functional.softplus(raw[:, 6])
    chol0 = torch.nn.functional.softplus(raw[:, 7])
    chol1 = raw[:, 8]
    chol2 = torch.nn.functional.softplus(raw[:, 9])
    radial_increments = torch.nn.functional.softplus(raw[:, 10:13])
    radial_quantiles = torch.cumsum(radial_increments, dim=-1)
    leading_distribution = raw[:, 13:18].softmax(dim=-1)
    leading_fractions = torch.sort(leading_distribution, dim=-1, descending=True).values[:, :4]
    return torch.cat(
        (
            eta_first[:, None],
            phi_first[:, None],
            eta_second[:, None],
            phi_second[:, None],
            cross[:, None],
            radial_first[:, None],
            radial_second[:, None],
            chol0[:, None],
            chol1[:, None],
            chol2[:, None],
            radial_quantiles,
            leading_fractions,
        ),
        dim=-1,
    )


def _fill_ledger(
    prediction: SemanticRootPrediction,
    kinematics: RootPhysicalKinematics,
    count: Any,
    type_counts: Any,
    charge: Any,
    minimum_mass: Any,
    charge_minimum: Any,
    charge_maximum: Any,
    scalar_sum_pt: Any,
    type_pt_fractions: Any,
    type_energy_fractions: Any,
    shape_features: Any,
) -> Any:
    torch = require_torch()
    batch = int(count.shape[0])
    ledger = torch.zeros(
        (batch, len(ROOT_FEATURE_NAMES)),
        dtype=kinematics.pt.dtype,
        device=kinematics.pt.device,
    )

    def put(name: str, value: Any) -> None:
        ledger[:, ROOT_FEATURE_INDEX[name]] = value.to(ledger.dtype)

    for name, value in zip(
        ("energy", "px", "py", "pz"),
        (kinematics.energy, kinematics.px, kinematics.py, kinematics.pz),
    ):
        put(name, value)
    put("constituent_count", count)
    for index, pid in enumerate(ABPH_PID_CATEGORIES):
        put(f"count_{pid}", type_counts[:, index])
    put("integer_charge", charge)
    put("minimum_mass_budget", minimum_mass)
    put("feasible_charge_min", charge_minimum)
    put("feasible_charge_max", charge_maximum)
    put("scalar_sum_pt", scalar_sum_pt)
    type_pt = scalar_sum_pt.unsqueeze(-1) * type_pt_fractions
    type_energy = kinematics.energy.unsqueeze(-1) * type_energy_fractions
    for index, pid in enumerate(ABPH_PID_CATEGORIES):
        put(f"energy_{pid}", type_energy[:, index])
        put(f"scalar_pt_{pid}", type_pt[:, index])
    absolute_charge = charge.abs().to(ledger.dtype) + torch.nn.functional.softplus(
        prediction.absolute_charge_mean
    )
    put("absolute_charge_sum", torch.minimum(absolute_charge, count.to(ledger.dtype)))
    for index, name in enumerate(ROOT_SHAPE_FEATURE_NAMES):
        put(name, shape_features[:, index])
    return ledger


def compile_root_state(
    prediction: SemanticRootPrediction,
    hlt_tokens: Any,
    hlt_mask: Any,
    *,
    count_override: Any | None = None,
    type_count_override: Any | None = None,
    charge_override: Any | None = None,
    normalization: RootNormalizationStats | None = None,
) -> CompiledRootState:
    """Compile one jointly feasible root in count/type/charge/mass/p4 order."""

    torch = require_torch()
    hlt = summarize_hlt_root(hlt_tokens, hlt_mask)
    batch = int(hlt.constituent_count.shape[0])
    if prediction.batch_size != batch:
        raise ValueError("prediction and HLT batch sizes differ")
    count_probabilities = prediction.count_probabilities()
    _validate_probabilities(count_probabilities, label="count")
    support = torch.arange(1, ABPH_MAX_PARTICLES + 1, device=count_probabilities.device)
    if count_override is None:
        count = count_probabilities.argmax(dim=-1).to(torch.long) + 1
    else:
        count = torch.as_tensor(count_override, device=count_probabilities.device).to(torch.long)
        if count.shape != (batch,) or bool(((count < 1) | (count > ABPH_MAX_PARTICLES)).any()):
            raise ValueError("count override must have shape [B] with values in [1, 128]")
    expected_count = (count_probabilities * support.to(count_probabilities.dtype)).sum(dim=-1)
    relaxed_count = count.to(expected_count.dtype) + expected_count - expected_count.detach()

    type_fractions = prediction.type_count_fractions()
    _validate_probabilities(type_fractions, label="type-count")
    if type_count_override is None:
        type_counts, relaxed_type_counts = allocate_integer_type_counts(type_fractions, count)
    else:
        type_counts = torch.as_tensor(
            type_count_override, device=count.device
        ).to(torch.long)
        if type_counts.shape != (batch, len(ABPH_PID_CATEGORIES)):
            raise ValueError("type-count override must have shape [B, 6]")
        if bool((type_counts < 0).any()) or not bool((type_counts.sum(dim=-1) == count).all()):
            raise ValueError("type-count override must be nonnegative and sum exactly to count")
        soft = type_fractions * relaxed_count.unsqueeze(-1)
        relaxed_type_counts = type_counts.to(soft.dtype) + soft - soft.detach()

    charge, relaxed_charge, charge_mask, charge_minimum, charge_maximum = select_feasible_charge(
        prediction.charge_logits, type_counts, charge_override=charge_override
    )
    hard_minimum_mass = minimum_mass_budget(type_counts)
    soft_minimum_mass = minimum_mass_budget(relaxed_type_counts)
    compiled_minimum_mass = hard_minimum_mass + soft_minimum_mass - soft_minimum_mass.detach()
    p4_residual = prediction.p4_residual_mean
    if normalization is not None:
        p4_residual = normalization.denormalize_named(
            p4_residual, ROOT_RESIDUAL_CHANNEL_NAMES[:4]
        )
    kinematics = root_residual_to_physical(
        hlt.kinematics, p4_residual, compiled_minimum_mass
    )
    type_pt_fractions = prediction.type_pt_fractions()
    type_energy_fractions = prediction.type_energy_fractions()
    _validate_probabilities(type_pt_fractions, label="type-pT")
    _validate_probabilities(type_energy_fractions, label="type-energy")
    scalar_sum_pt = kinematics.pt + torch.nn.functional.softplus(
        prediction.scalar_pt_excess_raw
    )
    shape_features = compile_shape_features(prediction.shape_raw)
    ledger = _fill_ledger(
        prediction,
        kinematics,
        count,
        type_counts,
        charge,
        hard_minimum_mass,
        charge_minimum,
        charge_maximum,
        scalar_sum_pt,
        type_pt_fractions,
        type_energy_fractions,
        shape_features,
    )
    diagnostics = root_compiler_audit(
        kinematics=kinematics,
        count=count,
        type_counts=type_counts,
        charge=charge,
        charge_mask=charge_mask,
        minimum_mass=hard_minimum_mass,
        type_count_fractions=type_fractions,
        type_pt_fractions=type_pt_fractions,
        type_energy_fractions=type_energy_fractions,
        shape_features=shape_features,
        ledger=ledger,
    )
    diagnostics["normalization_hash"] = (
        None if normalization is None else normalization.normalization_hash
    )
    if not diagnostics["ok"]:
        raise RuntimeError("root compiler failed closed: " + "; ".join(diagnostics["problems"]))
    return CompiledRootState(
        kinematics=kinematics,
        constituent_count=count,
        relaxed_constituent_count=relaxed_count,
        type_counts=type_counts,
        relaxed_type_counts=relaxed_type_counts,
        integer_charge=charge,
        relaxed_charge=relaxed_charge,
        feasible_charge_mask=charge_mask,
        feasible_charge_min=charge_minimum,
        feasible_charge_max=charge_maximum,
        minimum_mass_budget=hard_minimum_mass,
        relaxed_minimum_mass_budget=compiled_minimum_mass,
        type_count_fractions=type_fractions,
        type_pt_fractions=type_pt_fractions,
        type_energy_fractions=type_energy_fractions,
        scalar_sum_pt=scalar_sum_pt,
        shape_features=shape_features,
        root_ledger=ledger,
        diagnostics=diagnostics,
    )


def root_compiler_audit(
    *,
    kinematics: RootPhysicalKinematics,
    count: Any,
    type_counts: Any,
    charge: Any,
    charge_mask: Any,
    minimum_mass: Any,
    type_count_fractions: Any,
    type_pt_fractions: Any,
    type_energy_fractions: Any,
    shape_features: Any,
    ledger: Any,
) -> dict[str, Any]:
    torch = require_torch()
    problems: list[str] = []
    if not bool((type_counts.sum(dim=-1) == count).all()):
        problems.append("type counts do not sum to total count")
    charge_indices = charge - ABPH_CHARGE_SUPPORT_MIN
    selected_charge_valid = charge_mask.gather(1, charge_indices.unsqueeze(-1)).squeeze(-1)
    if not bool(selected_charge_valid.all()):
        problems.append("compiled charge is infeasible")
    if bool((kinematics.mass + 2.0e-6 < minimum_mass).any()):
        problems.append("compiled mass is below its type-conditioned floor")
    for label, fractions in (
        ("type-count", type_count_fractions),
        ("type-pT", type_pt_fractions),
        ("type-energy", type_energy_fractions),
    ):
        if not bool(torch.isfinite(fractions).all()) or bool((fractions < 0.0).any()):
            problems.append(f"{label} fractions are invalid")
        elif not bool(
            torch.allclose(
                fractions.sum(dim=-1),
                torch.ones_like(fractions.sum(dim=-1)),
                atol=2.0e-6,
                rtol=2.0e-6,
            )
        ):
            problems.append(f"{label} fractions do not sum to one")
    leading_start = ROOT_SHAPE_FEATURE_NAMES.index("leading_pt_fractions[0]")
    leading = shape_features[:, leading_start : leading_start + 4]
    if bool((leading < 0.0).any()) or bool((leading > 1.0).any()):
        problems.append("leading-pT fractions lie outside [0, 1]")
    if bool((leading[:, 1:] > leading[:, :-1] + 1.0e-7).any()):
        problems.append("leading-pT fractions are not ordered")
    if bool((leading.sum(dim=-1) > 1.0 + 1.0e-6).any()):
        problems.append("leading-pT fractions sum above one")
    finite_tensors = (
        kinematics.four_vector(),
        kinematics.mass,
        minimum_mass,
        shape_features,
        ledger,
    )
    if not all(bool(torch.isfinite(value).all()) for value in finite_tensors):
        problems.append("compiler produced nonfinite values")
    p4_from_ledger = torch.stack(
        tuple(ledger[:, ROOT_FEATURE_INDEX[name]] for name in ("energy", "px", "py", "pz")),
        dim=-1,
    )
    closure = (p4_from_ledger - kinematics.four_vector()).abs().amax()
    if float(closure.detach().cpu()) > 2.0e-5:
        problems.append("ledger four-vector differs from compiled root")
    return {
        "ok": not problems,
        "contract": ABPH_ROOT_COMPILER_CONTRACT,
        "version": ABPH_ROOT_COMPILER_VERSION,
        "compiler_hash": root_compiler_manifest()["compiler_hash"],
        "problems": problems,
        "compiler_failure_count": int(bool(problems)),
        "max_four_vector_ledger_residual": float(closure.detach().cpu()),
        "type_count_closure_max": int(
            (type_counts.sum(dim=-1) - count).abs().max().detach().cpu()
        ),
        "all_fractions_valid": not any("fraction" in problem for problem in problems),
    }


__all__ = [
    "ABPH_EFFECTIVE_MASS_GEV",
    "ABPH_ROOT_COMPILER_CONTRACT",
    "ABPH_ROOT_COMPILER_VERSION",
    "CompiledRootState",
    "allocate_integer_type_counts",
    "compile_root_state",
    "compile_shape_features",
    "feasible_charge_mask",
    "minimum_mass_budget",
    "root_compiler_audit",
    "root_compiler_manifest",
    "select_feasible_charge",
]
