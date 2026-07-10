"""Deterministic Canonical Multi-Scale Jet State builder.

`Phi(jet)` is intentionally not learned.  It maps raw JetClass particle tokens
to the frozen CMS-JS token/field layout so HLT and offline views can later be
compared with a meaningful residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.jetclass_data import JetView, RAW_TOKEN_DIM

from .layout import (
    CANONICAL_STATE_FIELD_NAMES,
    CanonicalJetStateLayout,
    CanonicalStateTokenSpec,
    default_canonical_jet_state_layout,
)


CANONICAL_STATE_PHI_BUILDER_VERSION = "canonical_phi_builder_v1"
EPS = 1.0e-8

PT_INDEX = 0
ETA_INDEX = 1
PHI_INDEX = 2
ENERGY_INDEX = 3
CHARGE_INDEX = 4
PID_CHARGED_HADRON_INDEX = 5
PID_NEUTRAL_HADRON_INDEX = 6
PID_PHOTON_INDEX = 7
PID_ELECTRON_INDEX = 8
PID_MUON_INDEX = 9


@dataclass(frozen=True)
class CanonicalPhiOutput:
    """Output of the deterministic `Phi` builder."""

    phi_tokens: np.ndarray
    state_mask: np.ndarray
    layout_metadata: dict[str, Any]
    diagnostics: dict[str, Any]
    source_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phi_tokens_shape": list(self.phi_tokens.shape),
            "state_mask_shape": list(self.state_mask.shape),
            "layout_metadata": self.layout_metadata,
            "diagnostics": self.diagnostics,
            "source_metadata": self.source_metadata,
        }


def _wrap_phi(value: np.ndarray) -> np.ndarray:
    return ((value + np.pi) % (2.0 * np.pi)) - np.pi


def _as_batch_tokens(tokens: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(tokens, dtype=np.float32)
    squeezed = False
    if arr.ndim == 2:
        arr = arr[None, :, :]
        squeezed = True
    if arr.ndim != 3 or int(arr.shape[-1]) != RAW_TOKEN_DIM:
        raise ValueError(f"tokens must have shape [B, N, {RAW_TOKEN_DIM}], got {arr.shape}")
    return arr, squeezed


def _as_batch_mask(mask: np.ndarray | None, tokens: np.ndarray) -> np.ndarray:
    if mask is None:
        return np.isfinite(tokens).all(axis=-1) & (tokens[:, :, PT_INDEX] > 0.0)
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape != tokens.shape[:2]:
        raise ValueError(f"mask must have shape {tokens.shape[:2]}, got {arr.shape}")
    return arr


def _safe_weighted_sum(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(values * weights, axis=1)


def _field_index(name: str) -> int:
    return CANONICAL_STATE_FIELD_NAMES.index(name)


def _normalize_fields(raw: np.ndarray, layout: CanonicalJetStateLayout) -> np.ndarray:
    scales = np.asarray(layout.feature_scale_vector(), dtype=np.float32)
    normalized = raw / np.maximum(scales[None, :], EPS)
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _summarize_selection(
    tokens: np.ndarray,
    valid: np.ndarray,
    base_selected: np.ndarray,
    selected: np.ndarray,
    total_pt: np.ndarray,
    total_energy: np.ndarray,
    layout: CanonicalJetStateLayout,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = tokens.shape[0]
    raw = np.zeros((batch_size, layout.d_phi), dtype=np.float32)
    selected_valid = selected & valid
    selected_pt = np.where(selected_valid, np.clip(tokens[:, :, PT_INDEX], 0.0, None), 0.0).astype(np.float32)
    selected_energy = np.where(selected_valid, np.clip(tokens[:, :, ENERGY_INDEX], 0.0, None), 0.0).astype(np.float32)
    selected_count = selected_valid.sum(axis=1).astype(np.float32)
    token_mask = selected_count > 0
    safe_count = np.maximum(selected_count, 1.0)
    pt_sum = selected_pt.sum(axis=1)
    energy_sum = selected_energy.sum(axis=1)
    safe_pt_sum = np.maximum(pt_sum, EPS)
    safe_total_pt = np.maximum(total_pt, EPS)
    safe_total_energy = np.maximum(total_energy, EPS)

    eta = tokens[:, :, ETA_INDEX].astype(np.float32)
    phi = _wrap_phi(tokens[:, :, PHI_INDEX].astype(np.float32))
    r2 = eta * eta + phi * phi
    mean_eta = _safe_weighted_sum(eta, selected_pt) / safe_pt_sum
    mean_phi = _wrap_phi(_safe_weighted_sum(phi, selected_pt) / safe_pt_sum)
    var_eta = _safe_weighted_sum((eta - mean_eta[:, None]) ** 2, selected_pt) / safe_pt_sum
    phi_delta = _wrap_phi(phi - mean_phi[:, None])
    var_phi = _safe_weighted_sum(phi_delta * phi_delta, selected_pt) / safe_pt_sum
    width = np.sqrt(_safe_weighted_sum(r2, selected_pt) / safe_pt_sum)

    px = _safe_weighted_sum(np.cos(phi), selected_pt)
    py = _safe_weighted_sum(np.sin(phi), selected_pt)
    pz = _safe_weighted_sum(np.sinh(eta), selected_pt)
    mass2 = np.maximum(energy_sum * energy_sum - px * px - py * py - pz * pz, 0.0)
    mass_proxy = np.sqrt(mass2) / safe_total_pt

    charge = tokens[:, :, CHARGE_INDEX]
    charged_like = (np.abs(charge) > 0.5) | (tokens[:, :, PID_CHARGED_HADRON_INDEX] > 0.5)
    neutral_like = tokens[:, :, PID_NEUTRAL_HADRON_INDEX] > 0.5
    photon_like = tokens[:, :, PID_PHOTON_INDEX] > 0.5
    electron_like = tokens[:, :, PID_ELECTRON_INDEX] > 0.5
    muon_like = tokens[:, :, PID_MUON_INDEX] > 0.5
    hadron_like = charged_like | neutral_like

    def pid_fraction(pid_mask: np.ndarray) -> np.ndarray:
        return np.sum(np.where(selected_valid & pid_mask, selected_pt, 0.0), axis=1) / safe_pt_sum

    base_count = np.maximum(base_selected.sum(axis=1).astype(np.float32), 1.0)
    missing_fraction = 1.0 - selected_count / base_count

    raw[:, _field_index("sum_pt_frac")] = pt_sum / safe_total_pt
    raw[:, _field_index("sum_energy_frac")] = energy_sum / safe_total_energy
    raw[:, _field_index("log1p_count")] = np.log1p(selected_count)
    raw[:, _field_index("mean_pt_frac")] = (pt_sum / safe_count) / safe_total_pt
    raw[:, _field_index("max_pt_frac")] = np.max(selected_pt, axis=1) / safe_total_pt
    raw[:, _field_index("pt_weighted_mean_deta")] = mean_eta
    raw[:, _field_index("pt_weighted_mean_dphi")] = mean_phi
    raw[:, _field_index("pt_weighted_var_deta")] = var_eta
    raw[:, _field_index("pt_weighted_var_dphi")] = var_phi
    raw[:, _field_index("mass_proxy")] = mass_proxy
    raw[:, _field_index("width_proxy")] = width
    raw[:, _field_index("charged_pt_frac")] = pid_fraction(charged_like)
    raw[:, _field_index("neutral_pt_frac")] = pid_fraction(neutral_like)
    raw[:, _field_index("photon_pt_frac")] = pid_fraction(photon_like)
    raw[:, _field_index("electron_pt_frac")] = pid_fraction(electron_like)
    raw[:, _field_index("muon_pt_frac")] = pid_fraction(muon_like)
    raw[:, _field_index("hadron_pt_frac")] = pid_fraction(hadron_like)
    raw[:, _field_index("quality_or_missingness_proxy")] = np.clip(missing_fraction, 0.0, 1.0)
    raw = np.where(token_mask[:, None], raw, 0.0)
    return _normalize_fields(raw, layout), token_mask


def _topk_indices(pt: np.ndarray, valid: np.ndarray, max_k: int) -> tuple[np.ndarray, np.ndarray]:
    score = np.where(valid, pt, -np.inf)
    keep = min(int(max_k), int(pt.shape[1]))
    order = np.argsort(-score, axis=1, kind="mergesort")[:, :keep]
    available = np.zeros((pt.shape[0], int(max_k)), dtype=bool)
    if keep < int(max_k):
        padded = np.zeros((pt.shape[0], int(max_k)), dtype=np.int64)
        padded[:, :keep] = order
        order = padded
    available[:, :keep] = True
    return order, available


def _global_selection(
    spec: CanonicalStateTokenSpec,
    pt: np.ndarray,
    base_mask: np.ndarray,
    valid: np.ndarray,
    total_pt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base = base_mask.copy()
    if spec.name == "global_multiplicity_softness":
        pt_frac = pt / np.maximum(total_pt[:, None], EPS)
        selected = valid & (pt_frac <= 0.05)
        return base, selected
    if spec.name == "global_leading_structure":
        order, available = _topk_indices(pt, valid, min(2, pt.shape[1]))
        selected = np.zeros_like(valid)
        rows = np.arange(pt.shape[0])[:, None]
        selected[rows, order] = valid[rows, order] & available
        return base, selected
    return base, valid.copy()


def _radial_selection(
    spec: CanonicalStateTokenSpec,
    r: np.ndarray,
    base_mask: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inner = 0.0 if spec.radius_inner is None else float(spec.radius_inner)
    base = base_mask & (r >= inner)
    if spec.radius_outer is not None:
        base = base & (r < float(spec.radius_outer))
    return base, base & valid


def _angular_selection(
    spec: CanonicalStateTokenSpec,
    phi: np.ndarray,
    base_mask: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    width = float(spec.angular_width)
    lower = -np.pi + float(spec.sector_id) * width
    upper = lower + width
    if spec.sector_id == 7:
        base = base_mask & (phi >= lower) & (phi <= upper)
    else:
        base = base_mask & (phi >= lower) & (phi < upper)
    return base, base & valid


def build_canonical_jet_state_phi(
    tokens: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    layout: CanonicalJetStateLayout | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> CanonicalPhiOutput:
    """Build deterministic canonical state tokens for a batch of jets."""

    layout = default_canonical_jet_state_layout() if layout is None else layout
    tokens, squeezed = _as_batch_tokens(tokens)
    original_finite = np.isfinite(tokens).all(axis=-1)
    base_mask = _as_batch_mask(mask, tokens)
    safe_tokens = np.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    pt = np.clip(safe_tokens[:, :, PT_INDEX], 0.0, None)
    energy = np.clip(safe_tokens[:, :, ENERGY_INDEX], 0.0, None)
    eta = safe_tokens[:, :, ETA_INDEX].astype(np.float32)
    phi = _wrap_phi(safe_tokens[:, :, PHI_INDEX].astype(np.float32))
    r = np.sqrt(np.maximum(eta * eta + phi * phi, 0.0))
    valid = base_mask & original_finite & (pt > 0.0)
    total_pt = np.sum(np.where(valid, pt, 0.0), axis=1).astype(np.float32)
    total_energy = np.sum(np.where(valid, energy, 0.0), axis=1).astype(np.float32)

    phi_tokens = np.zeros((safe_tokens.shape[0], layout.k_state, layout.d_phi), dtype=np.float32)
    state_mask = np.zeros((safe_tokens.shape[0], layout.k_state), dtype=bool)

    max_anchor_count = max(int(value) for value in layout.config.anchor_counts.values())
    topk, topk_available = _topk_indices(pt, valid, max_anchor_count)
    rows = np.arange(safe_tokens.shape[0])[:, None]
    topk_valid = valid[rows, topk] & topk_available
    topk_eta = eta[rows, topk]
    topk_phi = phi[rows, topk]

    for spec in layout.token_specs:
        if spec.family == "global":
            base_selected, selected = _global_selection(spec, pt, base_mask, valid, total_pt)
        elif spec.family == "radial":
            base_selected, selected = _radial_selection(spec, r, base_mask, valid)
        elif spec.family == "angular":
            base_selected, selected = _angular_selection(spec, phi, base_mask, valid)
        elif spec.family.startswith("anchor_"):
            slot = int(spec.slot_id)
            anchor_valid = topk_valid[:, slot]
            anchor_eta = topk_eta[:, slot]
            anchor_phi = topk_phi[:, slot]
            deta = eta - anchor_eta[:, None]
            dphi = _wrap_phi(phi - anchor_phi[:, None])
            dr = np.sqrt(np.maximum(deta * deta + dphi * dphi, 0.0))
            base_selected = base_mask & anchor_valid[:, None] & (dr <= float(spec.anchor_radius))
            selected = valid & anchor_valid[:, None] & (dr <= float(spec.anchor_radius))
        else:  # pragma: no cover - guarded by layout validation
            raise ValueError(f"unknown canonical state family {spec.family!r}")
        features, token_mask = _summarize_selection(
            safe_tokens,
            valid,
            base_selected,
            selected,
            total_pt,
            total_energy,
            layout,
        )
        phi_tokens[:, spec.index, :] = features
        state_mask[:, spec.index] = token_mask

    diagnostics = {
        "builder_version": CANONICAL_STATE_PHI_BUILDER_VERSION,
        "n_jets": int(safe_tokens.shape[0]),
        "max_particles": int(safe_tokens.shape[1]),
        "k_state": int(layout.k_state),
        "d_phi": int(layout.d_phi),
        "valid_particle_counts": valid.sum(axis=1).astype(np.int64).tolist(),
        "masked_particle_counts": base_mask.sum(axis=1).astype(np.int64).tolist(),
        "finite_particle_fraction": (
            (original_finite & base_mask).sum(axis=1) / np.maximum(base_mask.sum(axis=1), 1)
        ).astype(np.float32).tolist(),
        "total_pt": total_pt.astype(np.float32).tolist(),
        "total_energy": total_energy.astype(np.float32).tolist(),
        "state_valid_counts": state_mask.sum(axis=1).astype(np.int64).tolist(),
        "family_valid_counts": {
            family: state_mask[:, start:end].sum(axis=1).astype(np.int64).tolist()
            for family, (start, end) in layout.family_slices().items()
        },
        "all_finite": bool(np.isfinite(phi_tokens).all()),
        "squeezed_input": bool(squeezed),
    }
    output = CanonicalPhiOutput(
        phi_tokens=phi_tokens,
        state_mask=state_mask,
        layout_metadata=layout.to_dict(),
        diagnostics=diagnostics,
        source_metadata=dict(source_metadata or {}),
    )
    if squeezed:
        return CanonicalPhiOutput(
            phi_tokens=output.phi_tokens[0],
            state_mask=output.state_mask[0],
            layout_metadata=output.layout_metadata,
            diagnostics=output.diagnostics,
            source_metadata=output.source_metadata,
        )
    return output


def build_canonical_jet_state_phi_from_view(
    view: JetView,
    *,
    layout: CanonicalJetStateLayout | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> CanonicalPhiOutput:
    metadata = dict(view.metadata)
    metadata.update(dict(source_metadata or {}))
    metadata.setdefault("split", view.split)
    return build_canonical_jet_state_phi(
        view.tokens,
        view.mask,
        layout=layout,
        source_metadata=metadata,
    )
