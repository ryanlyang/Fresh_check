"""Deterministic accounting and geometry layout for pseudo-offline targets."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


HIERARCHY_LAYOUT_VERSION = "constrained_coarse_to_fine_layout_v1"

PID_CATEGORY_NAMES: tuple[str, ...] = (
    "charged",
    "neutral_hadron",
    "photon",
    "electron",
    "muon",
)
PID_TOKEN_SLICE = slice(5, 10)

PID_PT_FIELD_NAMES: tuple[str, ...] = tuple(f"{name}_pT" for name in PID_CATEGORY_NAMES)
PID_COUNT_FIELD_NAMES: tuple[str, ...] = tuple(f"{name}_count" for name in PID_CATEGORY_NAMES)
MOMENT_FIELD_NAMES: tuple[str, ...] = (
    "sum_pT_abs_deta_pos",
    "sum_pT_abs_deta_neg",
    "sum_pT_abs_dphi_pos",
    "sum_pT_abs_dphi_neg",
    "sum_pT_deta2",
    "sum_pT_dphi2",
    "sum_pT_r",
    "sum_pT_r2",
)

ACCOUNTING_FIELD_NAMES: tuple[str, ...] = (
    "total_pT",
    "total_energy",
    "expected_constituent_count",
    *PID_PT_FIELD_NAMES,
    *PID_COUNT_FIELD_NAMES,
    *MOMENT_FIELD_NAMES,
)

PRIMITIVE_ACCOUNTING_FIELD_NAMES: tuple[str, ...] = (
    "total_energy",
    *PID_PT_FIELD_NAMES,
    *PID_COUNT_FIELD_NAMES,
    *MOMENT_FIELD_NAMES,
)

DERIVED_ACCOUNTING_FIELD_NAMES: tuple[str, ...] = (
    "total_pT",
    "expected_constituent_count",
)

DERIVED_DIAGNOSTIC_FIELD_NAMES: tuple[str, ...] = (
    "charged_pT_fraction",
    "neutral_hadron_pT_fraction",
    "photon_pT_fraction",
    "electron_pT_fraction",
    "muon_pT_fraction",
    "axis_deta",
    "axis_dphi",
    "width_eta",
    "width_phi",
    "mean_r",
    "r_rms",
    "mean_pT_per_constituent",
    "energy_over_pT",
)

LEVEL_NAMES: tuple[str, ...] = ("global", "level1", "level2", "level3")
LEVEL_CELL_COUNTS: tuple[int, ...] = (1, 8, 32, 128)


def _decode_cell_bits(level: int, cell_id: int) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    if level not in (1, 2, 3):
        raise ValueError(f"level must be 1, 2, or 3, got {level}")
    if not 0 <= int(cell_id) < LEVEL_CELL_COUNTS[level]:
        raise ValueError(f"cell_id {cell_id} is outside level {level}")

    current = int(cell_id)
    child_digits: list[int] = []
    for _ in range(level - 1):
        child_digits.append(current % 4)
        current //= 4

    radial_bin = current // 4
    quadrant = current % 4
    eta_bits = [quadrant // 2]
    phi_bits = [quadrant % 2]
    for digit in reversed(child_digits):
        eta_bits.append(digit // 2)
        phi_bits.append(digit % 2)
    return radial_bin, tuple(eta_bits), tuple(phi_bits)


def _bits_to_bin(bits: tuple[int, ...]) -> int:
    value = 0
    for bit in bits:
        value = 2 * value + int(bit)
    return value


@dataclass(frozen=True)
class HierarchyTargetLayout:
    """Fixed hierarchy geometry shared by target caches and future models."""

    radial_boundary: float
    coordinate_extent: float = 0.8
    layout_version: str = HIERARCHY_LAYOUT_VERSION

    def __post_init__(self) -> None:
        if not np.isfinite(self.radial_boundary) or float(self.radial_boundary) <= 0.0:
            raise ValueError("radial_boundary must be finite and positive")
        if not np.isfinite(self.coordinate_extent) or float(self.coordinate_extent) <= 0.0:
            raise ValueError("coordinate_extent must be finite and positive")
        if float(self.radial_boundary) >= math.sqrt(2.0) * float(self.coordinate_extent):
            raise ValueError("radial_boundary must lie inside the configured eta/phi square")

    @property
    def field_names(self) -> tuple[str, ...]:
        return ACCOUNTING_FIELD_NAMES

    @property
    def field_dim(self) -> int:
        return len(ACCOUNTING_FIELD_NAMES)

    @property
    def level_cell_counts(self) -> tuple[int, ...]:
        return LEVEL_CELL_COUNTS

    def field_index(self, name: str) -> int:
        try:
            return ACCOUNTING_FIELD_NAMES.index(str(name))
        except ValueError as exc:
            raise KeyError(f"unknown accounting field {name!r}") from exc

    def parent_indices(self, level: int) -> np.ndarray:
        if level == 1:
            return np.zeros(LEVEL_CELL_COUNTS[1], dtype=np.int16)
        if level in (2, 3):
            return np.arange(LEVEL_CELL_COUNTS[level], dtype=np.int16) // 4
        raise ValueError(f"level must be 1, 2, or 3, got {level}")

    def cell_geometry(self, level: int) -> tuple[dict[str, Any], ...]:
        """Return stable rectangular bounds plus radial-shell metadata."""

        if level not in (1, 2, 3):
            raise ValueError(f"level must be 1, 2, or 3, got {level}")
        bins_per_axis = 2**level
        extent = float(self.coordinate_extent)
        width = 2.0 * extent / float(bins_per_axis)
        max_radius = math.sqrt(2.0) * extent
        rows: list[dict[str, Any]] = []
        parents = self.parent_indices(level)
        for cell_id in range(LEVEL_CELL_COUNTS[level]):
            radial_bin, eta_bits, phi_bits = _decode_cell_bits(level, cell_id)
            eta_bin = _bits_to_bin(eta_bits)
            phi_bin = _bits_to_bin(phi_bits)
            rows.append(
                {
                    "cell_id": int(cell_id),
                    "parent_id": int(parents[cell_id]),
                    "radial_bin": int(radial_bin),
                    "eta_bin": int(eta_bin),
                    "phi_bin": int(phi_bin),
                    "eta_min": float(-extent + eta_bin * width),
                    "eta_max": float(-extent + (eta_bin + 1) * width),
                    "phi_min": float(-extent + phi_bin * width),
                    "phi_max": float(-extent + (phi_bin + 1) * width),
                    "radial_min": 0.0 if radial_bin == 0 else float(self.radial_boundary),
                    "radial_max": float(self.radial_boundary) if radial_bin == 0 else max_radius,
                }
            )
        return tuple(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_version": str(self.layout_version),
            "radial_boundary": float(self.radial_boundary),
            "coordinate_extent": float(self.coordinate_extent),
            "reference_axis": "hlt_pt_weighted_eta_and_circular_phi",
            "field_names": list(ACCOUNTING_FIELD_NAMES),
            "primitive_field_names": list(PRIMITIVE_ACCOUNTING_FIELD_NAMES),
            "derived_accounting_field_names": list(DERIVED_ACCOUNTING_FIELD_NAMES),
            "derived_diagnostic_field_names": list(DERIVED_DIAGNOSTIC_FIELD_NAMES),
            "pid_category_names": list(PID_CATEGORY_NAMES),
            "level_names": list(LEVEL_NAMES),
            "level_cell_counts": list(LEVEL_CELL_COUNTS),
            "parent_indices": {
                f"level{level}": self.parent_indices(level).astype(int).tolist()
                for level in (1, 2, 3)
            },
            "cell_geometry": {
                f"level{level}": list(self.cell_geometry(level))
                for level in (1, 2, 3)
            },
        }


def default_hierarchy_target_layout(
    *,
    radial_boundary: float,
    coordinate_extent: float = 0.8,
) -> HierarchyTargetLayout:
    return HierarchyTargetLayout(
        radial_boundary=float(radial_boundary),
        coordinate_extent=float(coordinate_extent),
    )

