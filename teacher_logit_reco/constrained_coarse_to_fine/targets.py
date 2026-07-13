"""Pure NumPy hierarchy-target construction for constrained pseudo-offline jets."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from .layout import (
    ACCOUNTING_FIELD_NAMES,
    DERIVED_DIAGNOSTIC_FIELD_NAMES,
    LEVEL_CELL_COUNTS,
    LEVEL_NAMES,
    MOMENT_FIELD_NAMES,
    PID_CATEGORY_NAMES,
    PID_COUNT_FIELD_NAMES,
    PID_PT_FIELD_NAMES,
    PID_TOKEN_SLICE,
    HierarchyTargetLayout,
)


HIERARCHY_TARGET_BUILDER_VERSION = "constrained_coarse_to_fine_target_builder_v1"
EPSILON = 1.0e-8


@dataclass(frozen=True)
class HierarchyTargetOutput:
    """One aligned batch of deterministic offline hierarchy targets."""

    global_accounting: np.ndarray
    level1_accounting: np.ndarray
    level2_accounting: np.ndarray
    level3_accounting: np.ndarray
    final_cell_indices: np.ndarray
    reference_eta: np.ndarray
    reference_phi: np.ndarray
    valid_hlt_counts: np.ndarray
    valid_offline_counts: np.ndarray
    unknown_pid_counts: np.ndarray
    clipped_particle_counts: np.ndarray
    layout: HierarchyTargetLayout
    diagnostics: dict[str, Any]

    def accounting(self, level: int | str) -> np.ndarray:
        aliases: Mapping[int | str, str] = {
            0: "global",
            1: "level1",
            2: "level2",
            3: "level3",
            "global": "global",
            "level1": "level1",
            "level2": "level2",
            "level3": "level3",
        }
        try:
            name = aliases[level]
        except KeyError as exc:
            raise KeyError(f"unknown hierarchy level {level!r}") from exc
        return {
            "global": self.global_accounting,
            "level1": self.level1_accounting,
            "level2": self.level2_accounting,
            "level3": self.level3_accounting,
        }[name]

    def derived_diagnostics(self, level: int | str) -> np.ndarray:
        return derive_accounting_diagnostics(self.accounting(level))


def wrap_phi(delta: np.ndarray) -> np.ndarray:
    return ((np.asarray(delta) + np.pi) % (2.0 * np.pi)) - np.pi


def _validate_particle_arrays(tokens: np.ndarray, mask: np.ndarray, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(tokens)
    valid = np.asarray(mask, dtype=bool)
    if arr.ndim != 3 or int(arr.shape[-1]) != RAW_TOKEN_DIM:
        raise ValueError(f"{name}_tokens must have shape [N, P, {RAW_TOKEN_DIM}], got {arr.shape}")
    if valid.shape != arr.shape[:2]:
        raise ValueError(f"{name}_mask shape {valid.shape} does not match tokens {arr.shape[:2]}")
    return np.asarray(arr, dtype=np.float32), valid


def hlt_reference_axis(tokens: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the pT-weighted eta and circular-phi axis of each HLT jet."""

    arr, supplied_mask = _validate_particle_arrays(tokens, mask, name="hlt")
    finite = np.isfinite(arr[:, :, :3]).all(axis=-1)
    valid = supplied_mask & finite & (arr[:, :, 0] > 0.0)
    pt = np.where(valid, np.maximum(arr[:, :, 0], 0.0), 0.0).astype(np.float64)
    denominator = pt.sum(axis=1)
    safe_denominator = np.maximum(denominator, EPSILON)
    eta = (pt * np.where(valid, arr[:, :, 1], 0.0)).sum(axis=1) / safe_denominator
    sin_sum = (pt * np.sin(np.where(valid, arr[:, :, 2], 0.0))).sum(axis=1)
    cos_sum = (pt * np.cos(np.where(valid, arr[:, :, 2], 0.0))).sum(axis=1)
    phi = np.arctan2(sin_sum, cos_sum)
    empty = denominator <= EPSILON
    eta[empty] = 0.0
    phi[empty] = 0.0
    return eta.astype(np.float32), phi.astype(np.float32), valid.sum(axis=1).astype(np.int32)


def fit_radial_boundary_from_hlt(
    hlt_tokens: np.ndarray,
    hlt_mask: np.ndarray,
    *,
    coordinate_extent: float = 0.8,
    histogram_bins: int = 4096,
    chunk_size: int = 8192,
) -> tuple[float, dict[str, Any]]:
    """Fit a deterministic approximate pT-weighted median HLT radius.

    A bounded histogram avoids materializing every constituent radius in the
    high-data model_train split. The fitted scalar is stored in the layout and
    reused unchanged for every campaign split.
    """

    tokens, mask = _validate_particle_arrays(hlt_tokens, hlt_mask, name="hlt")
    if int(histogram_bins) < 16:
        raise ValueError("histogram_bins must be at least 16")
    if not np.isfinite(coordinate_extent) or float(coordinate_extent) <= 0.0:
        raise ValueError("coordinate_extent must be finite and positive")
    chunk = max(1, int(chunk_size))
    max_radius = math.sqrt(2.0) * float(coordinate_extent)
    histogram = np.zeros(int(histogram_bins), dtype=np.float64)
    n_valid = 0
    n_clipped = 0

    for start in range(0, int(tokens.shape[0]), chunk):
        stop = min(start + chunk, int(tokens.shape[0]))
        part = tokens[start:stop]
        part_mask = mask[start:stop]
        eta_ref, phi_ref, _ = hlt_reference_axis(part, part_mask)
        finite = np.isfinite(part[:, :, :3]).all(axis=-1)
        valid = part_mask & finite & (part[:, :, 0] > 0.0)
        deta = part[:, :, 1] - eta_ref[:, None]
        dphi = wrap_phi(part[:, :, 2] - phi_ref[:, None])
        radius = np.sqrt(deta * deta + dphi * dphi)
        weights = np.maximum(part[:, :, 0], 0.0)
        values = radius[valid].astype(np.float64, copy=False)
        valid_weights = weights[valid].astype(np.float64, copy=False)
        if values.size == 0:
            continue
        n_valid += int(values.size)
        n_clipped += int(np.sum(values >= max_radius))
        values = np.clip(values, 0.0, np.nextafter(max_radius, 0.0))
        histogram += np.histogram(
            values,
            bins=int(histogram_bins),
            range=(0.0, max_radius),
            weights=valid_weights,
        )[0]

    total_weight = float(histogram.sum())
    if total_weight <= 0.0:
        raise ValueError("cannot fit radial boundary because model_train HLT has no valid positive-pT particles")
    median_bin = int(np.searchsorted(np.cumsum(histogram), 0.5 * total_weight, side="left"))
    bin_width = max_radius / float(histogram_bins)
    boundary = (median_bin + 0.5) * bin_width
    boundary = float(np.clip(boundary, bin_width, max_radius - bin_width))
    diagnostics = {
        "method": "hlt_model_train_pt_weighted_radius_histogram_median",
        "histogram_bins": int(histogram_bins),
        "coordinate_extent": float(coordinate_extent),
        "max_radius": float(max_radius),
        "n_valid_particles": int(n_valid),
        "n_radius_clipped": int(n_clipped),
        "total_pT_weight": total_weight,
        "radial_boundary": boundary,
    }
    return boundary, diagnostics


def assign_hierarchy_cells(
    deta: np.ndarray,
    dphi: np.ndarray,
    valid_mask: np.ndarray,
    *,
    layout: HierarchyTargetLayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign particles to nested 8/32/128 cells with stable integer IDs."""

    deta_arr = np.asarray(deta, dtype=np.float64)
    dphi_arr = np.asarray(dphi, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if deta_arr.shape != dphi_arr.shape or valid.shape != deta_arr.shape:
        raise ValueError("deta, dphi, and valid_mask must have identical shapes")

    extent = float(layout.coordinate_extent)
    upper = np.nextafter(1.0, 0.0)
    eta_unit = np.clip((deta_arr + extent) / (2.0 * extent), 0.0, upper)
    phi_unit = np.clip((dphi_arr + extent) / (2.0 * extent), 0.0, upper)
    eta_bin = np.floor(8.0 * eta_unit).astype(np.int16)
    phi_bin = np.floor(8.0 * phi_unit).astype(np.int16)
    radius = np.sqrt(deta_arr * deta_arr + dphi_arr * dphi_arr)
    radial_bin = (radius >= float(layout.radial_boundary)).astype(np.int16)

    eta_bit0 = eta_bin >> 2
    eta_bit1 = (eta_bin >> 1) & 1
    eta_bit2 = eta_bin & 1
    phi_bit0 = phi_bin >> 2
    phi_bit1 = (phi_bin >> 1) & 1
    phi_bit2 = phi_bin & 1

    level1 = radial_bin * 4 + eta_bit0 * 2 + phi_bit0
    level2 = level1 * 4 + eta_bit1 * 2 + phi_bit1
    level3 = level2 * 4 + eta_bit2 * 2 + phi_bit2
    clipped = valid & ((np.abs(deta_arr) >= extent) | (np.abs(dphi_arr) >= extent))
    invalid_value = np.int16(-1)
    return (
        np.where(valid, level1, invalid_value).astype(np.int16),
        np.where(valid, level2, invalid_value).astype(np.int16),
        np.where(valid, level3, invalid_value).astype(np.int16),
        clipped,
        radius.astype(np.float32),
    )


def _derive_coupled_totals(accounting: np.ndarray) -> None:
    pt_indices = [ACCOUNTING_FIELD_NAMES.index(name) for name in PID_PT_FIELD_NAMES]
    count_indices = [ACCOUNTING_FIELD_NAMES.index(name) for name in PID_COUNT_FIELD_NAMES]
    accounting[..., ACCOUNTING_FIELD_NAMES.index("total_pT")] = np.sum(
        accounting[..., pt_indices], axis=-1, dtype=np.float32
    )
    accounting[..., ACCOUNTING_FIELD_NAMES.index("expected_constituent_count")] = np.sum(
        accounting[..., count_indices], axis=-1, dtype=np.float32
    )


def _aggregate_children(children: np.ndarray, parent_count: int, *, dtype: np.dtype) -> np.ndarray:
    child_count = int(children.shape[-2])
    if child_count != int(parent_count) * 4:
        raise ValueError(f"cannot aggregate {child_count} children into {parent_count} parents")
    shape = (*children.shape[:-2], int(parent_count), 4, children.shape[-1])
    parent = np.sum(children.reshape(shape), axis=-2, dtype=np.float32).astype(dtype, copy=False)
    _derive_coupled_totals(parent)
    return parent


def derive_accounting_diagnostics(accounting: np.ndarray) -> np.ndarray:
    """Derive fractions, axes, widths, and ratios from additive channels."""

    values = np.asarray(accounting, dtype=np.float64)
    if values.shape[-1] != len(ACCOUNTING_FIELD_NAMES):
        raise ValueError(
            f"accounting last dimension must be {len(ACCOUNTING_FIELD_NAMES)}, got {values.shape[-1]}"
        )
    index = {name: ACCOUNTING_FIELD_NAMES.index(name) for name in ACCOUNTING_FIELD_NAMES}
    total_pt = values[..., index["total_pT"]]
    total_count = values[..., index["expected_constituent_count"]]
    safe_pt = np.maximum(total_pt, EPSILON)
    safe_count = np.maximum(total_count, EPSILON)
    axis_deta = (
        values[..., index["sum_pT_abs_deta_pos"]] - values[..., index["sum_pT_abs_deta_neg"]]
    ) / safe_pt
    axis_dphi = (
        values[..., index["sum_pT_abs_dphi_pos"]] - values[..., index["sum_pT_abs_dphi_neg"]]
    ) / safe_pt
    width_eta = np.maximum(values[..., index["sum_pT_deta2"]] / safe_pt - axis_deta * axis_deta, 0.0)
    width_phi = np.maximum(values[..., index["sum_pT_dphi2"]] / safe_pt - axis_dphi * axis_dphi, 0.0)
    result = np.stack(
        [
            *(values[..., index[name]] / safe_pt for name in PID_PT_FIELD_NAMES),
            axis_deta,
            axis_dphi,
            width_eta,
            width_phi,
            values[..., index["sum_pT_r"]] / safe_pt,
            np.sqrt(np.maximum(values[..., index["sum_pT_r2"]] / safe_pt, 0.0)),
            total_pt / safe_count,
            values[..., index["total_energy"]] / safe_pt,
        ],
        axis=-1,
    )
    result = np.where((total_pt > EPSILON)[..., None], result, 0.0)
    if result.shape[-1] != len(DERIVED_DIAGNOSTIC_FIELD_NAMES):
        raise AssertionError("derived diagnostic layout does not match field names")
    return result.astype(np.float32)


def build_hierarchy_targets(
    hlt_tokens: np.ndarray,
    hlt_mask: np.ndarray,
    offline_tokens: np.ndarray,
    offline_mask: np.ndarray,
    *,
    layout: HierarchyTargetLayout,
    target_dtype: str | np.dtype = "float32",
) -> HierarchyTargetOutput:
    """Build deterministic additive hierarchy targets for an aligned jet batch."""

    hlt, hlt_valid_mask = _validate_particle_arrays(hlt_tokens, hlt_mask, name="hlt")
    offline, offline_supplied_mask = _validate_particle_arrays(offline_tokens, offline_mask, name="offline")
    if hlt.shape[0] != offline.shape[0]:
        raise ValueError("HLT and offline batches must contain the same number of jets")
    dtype = np.dtype(target_dtype)
    if dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("target_dtype must be float16 or float32")

    reference_eta, reference_phi, valid_hlt_counts = hlt_reference_axis(hlt, hlt_valid_mask)
    finite_offline = np.isfinite(offline[:, :, :10]).all(axis=-1)
    valid_offline = offline_supplied_mask & finite_offline & (offline[:, :, 0] > 0.0)
    deta = offline[:, :, 1] - reference_eta[:, None]
    dphi = wrap_phi(offline[:, :, 2] - reference_phi[:, None])
    level1_ids, level2_ids, level3_ids, clipped, radius = assign_hierarchy_cells(
        deta,
        dphi,
        valid_offline,
        layout=layout,
    )

    n_jets = int(offline.shape[0])
    field_dim = len(ACCOUNTING_FIELD_NAMES)
    fine = np.zeros((n_jets, LEVEL_CELL_COUNTS[3], field_dim), dtype=np.float64)
    jet_indices, particle_indices = np.nonzero(valid_offline)
    cell_indices = level3_ids[jet_indices, particle_indices].astype(np.int64)
    token_rows = offline[jet_indices, particle_indices]
    pt = np.maximum(token_rows[:, 0], 0.0).astype(np.float64)
    energy = np.maximum(token_rows[:, 3], 0.0).astype(np.float64)
    local_deta = deta[jet_indices, particle_indices].astype(np.float64)
    local_dphi = dphi[jet_indices, particle_indices].astype(np.float64)
    local_radius = radius[jet_indices, particle_indices].astype(np.float64)

    pid_scores = token_rows[:, PID_TOKEN_SLICE]
    pid_score_sums = np.sum(pid_scores, axis=1)
    unknown_pid = pid_score_sums <= 0.0
    pid_indices = np.argmax(pid_scores, axis=1).astype(np.int64)
    pid_indices[unknown_pid] = PID_CATEGORY_NAMES.index("neutral_hadron")

    def add(field_name: str, values: np.ndarray) -> None:
        np.add.at(
            fine,
            (jet_indices, cell_indices, ACCOUNTING_FIELD_NAMES.index(field_name)),
            np.asarray(values, dtype=np.float64),
        )

    add("total_energy", energy)
    for category_index, category_name in enumerate(PID_CATEGORY_NAMES):
        category_mask = pid_indices == category_index
        if not np.any(category_mask):
            continue
        np.add.at(
            fine,
            (
                jet_indices[category_mask],
                cell_indices[category_mask],
                ACCOUNTING_FIELD_NAMES.index(f"{category_name}_pT"),
            ),
            pt[category_mask],
        )
        np.add.at(
            fine,
            (
                jet_indices[category_mask],
                cell_indices[category_mask],
                ACCOUNTING_FIELD_NAMES.index(f"{category_name}_count"),
            ),
            np.ones(int(np.sum(category_mask)), dtype=np.float64),
        )

    moment_values = {
        "sum_pT_abs_deta_pos": pt * np.maximum(local_deta, 0.0),
        "sum_pT_abs_deta_neg": pt * np.maximum(-local_deta, 0.0),
        "sum_pT_abs_dphi_pos": pt * np.maximum(local_dphi, 0.0),
        "sum_pT_abs_dphi_neg": pt * np.maximum(-local_dphi, 0.0),
        "sum_pT_deta2": pt * local_deta * local_deta,
        "sum_pT_dphi2": pt * local_dphi * local_dphi,
        "sum_pT_r": pt * local_radius,
        "sum_pT_r2": pt * local_radius * local_radius,
    }
    for field_name in MOMENT_FIELD_NAMES:
        add(field_name, moment_values[field_name])

    level3 = fine.astype(dtype, copy=False)
    _derive_coupled_totals(level3)
    level2 = _aggregate_children(level3, LEVEL_CELL_COUNTS[2], dtype=dtype)
    level1 = _aggregate_children(level2, LEVEL_CELL_COUNTS[1], dtype=dtype)
    global_accounting = np.sum(level1, axis=1, dtype=np.float32).astype(dtype, copy=False)
    _derive_coupled_totals(global_accounting)

    unknown_pid_counts = np.bincount(
        jet_indices[unknown_pid], minlength=n_jets
    ).astype(np.int32)
    clipped_particle_counts = clipped.sum(axis=1).astype(np.int32)
    output = HierarchyTargetOutput(
        global_accounting=global_accounting,
        level1_accounting=level1,
        level2_accounting=level2,
        level3_accounting=level3,
        final_cell_indices=level3_ids,
        reference_eta=reference_eta,
        reference_phi=reference_phi,
        valid_hlt_counts=valid_hlt_counts,
        valid_offline_counts=valid_offline.sum(axis=1).astype(np.int32),
        unknown_pid_counts=unknown_pid_counts,
        clipped_particle_counts=clipped_particle_counts,
        layout=layout,
        diagnostics={
            "builder_version": HIERARCHY_TARGET_BUILDER_VERSION,
            "all_finite": bool(
                np.isfinite(global_accounting).all()
                and np.isfinite(level1).all()
                and np.isfinite(level2).all()
                and np.isfinite(level3).all()
            ),
            "empty_hlt_jets": int(np.sum(valid_hlt_counts == 0)),
            "empty_offline_jets": int(np.sum(valid_offline.sum(axis=1) == 0)),
            "unknown_pid_particles": int(np.sum(unknown_pid)),
            "clipped_coordinate_particles": int(np.sum(clipped)),
        },
    )
    output.diagnostics.update(hierarchy_consistency_report(output))
    return output


def _max_abs(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(np.max(np.abs(np.asarray(values, dtype=np.float64))))


def hierarchy_consistency_report(output: HierarchyTargetOutput) -> dict[str, Any]:
    """Measure closure of every stored parent/child and coupled-total relation."""

    global_from_level1 = np.sum(output.level1_accounting, axis=1, dtype=np.float32)
    level1_from_level2 = np.sum(
        output.level2_accounting.reshape(output.level2_accounting.shape[0], 8, 4, -1),
        axis=2,
        dtype=np.float32,
    )
    level2_from_level3 = np.sum(
        output.level3_accounting.reshape(output.level3_accounting.shape[0], 32, 4, -1),
        axis=2,
        dtype=np.float32,
    )
    index = {name: ACCOUNTING_FIELD_NAMES.index(name) for name in ACCOUNTING_FIELD_NAMES}
    pt_indices = [index[name] for name in PID_PT_FIELD_NAMES]
    count_indices = [index[name] for name in PID_COUNT_FIELD_NAMES]

    closure = {
        "global_from_level1_max_abs": _max_abs(output.global_accounting - global_from_level1),
        "level1_from_level2_max_abs": _max_abs(output.level1_accounting - level1_from_level2),
        "level2_from_level3_max_abs": _max_abs(output.level2_accounting - level2_from_level3),
    }
    coupled: dict[str, float] = {}
    for level_name in LEVEL_NAMES:
        accounting = output.accounting(level_name)
        coupled[f"{level_name}_total_pT_max_abs"] = _max_abs(
            accounting[..., index["total_pT"]] - np.sum(accounting[..., pt_indices], axis=-1, dtype=np.float32)
        )
        coupled[f"{level_name}_total_count_max_abs"] = _max_abs(
            accounting[..., index["expected_constituent_count"]]
            - np.sum(accounting[..., count_indices], axis=-1, dtype=np.float32)
        )
    return {
        "parent_child_closure": closure,
        "coupled_total_closure": coupled,
        "minimum_accounting_value": float(
            min(
                np.min(output.global_accounting, initial=0.0),
                np.min(output.level1_accounting, initial=0.0),
                np.min(output.level2_accounting, initial=0.0),
                np.min(output.level3_accounting, initial=0.0),
            )
        ),
    }


def require_hierarchy_consistency(
    output: HierarchyTargetOutput,
    *,
    atol: float = 2.0e-4,
    rtol: float = 2.0e-6,
) -> None:
    """Fail closed when hierarchy targets do not conserve their parent totals."""

    pairs = (
        (
            output.global_accounting,
            np.sum(output.level1_accounting, axis=1, dtype=np.float32),
            "global <- level1",
        ),
        (
            output.level1_accounting,
            np.sum(
                output.level2_accounting.reshape(output.level2_accounting.shape[0], 8, 4, -1),
                axis=2,
                dtype=np.float32,
            ),
            "level1 <- level2",
        ),
        (
            output.level2_accounting,
            np.sum(
                output.level3_accounting.reshape(output.level3_accounting.shape[0], 32, 4, -1),
                axis=2,
                dtype=np.float32,
            ),
            "level2 <- level3",
        ),
    )
    for parent, child_sum, label in pairs:
        if not np.allclose(parent, child_sum, atol=float(atol), rtol=float(rtol)):
            raise ValueError(f"hierarchy target parent-child closure failed for {label}")

    index = {name: ACCOUNTING_FIELD_NAMES.index(name) for name in ACCOUNTING_FIELD_NAMES}
    pt_indices = [index[name] for name in PID_PT_FIELD_NAMES]
    count_indices = [index[name] for name in PID_COUNT_FIELD_NAMES]
    for level_name in LEVEL_NAMES:
        accounting = output.accounting(level_name)
        if not np.allclose(
            accounting[..., index["total_pT"]],
            np.sum(accounting[..., pt_indices], axis=-1, dtype=np.float32),
            atol=float(atol),
            rtol=float(rtol),
        ):
            raise ValueError(f"hierarchy target category-pT closure failed for {level_name}")
        if not np.allclose(
            accounting[..., index["expected_constituent_count"]],
            np.sum(accounting[..., count_indices], axis=-1, dtype=np.float32),
            atol=float(atol),
            rtol=float(rtol),
        ):
            raise ValueError(f"hierarchy target category-count closure failed for {level_name}")
    if not bool(output.diagnostics.get("all_finite", False)):
        raise ValueError("hierarchy targets contain nonfinite values")
    if float(output.diagnostics["minimum_accounting_value"]) < -float(atol):
        raise ValueError("hierarchy targets contain negative additive accounting values")

