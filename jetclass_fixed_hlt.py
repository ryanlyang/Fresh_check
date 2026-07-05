#!/usr/bin/env python3
"""Standalone fixed JetClass HLT-style corruption used for the fresh-start check.

This module intentionally contains only the HLT generation logic needed for the
independent replication experiment. It mirrors the fixed m2-style HLT profile
used in the same-HLT JetClass runs:

  hlt_pt_threshold = 1.30
  merge_prob_scale = 1.35
  reassign_scale = 1.00
  smear_scale = 1.00
  eff_plateau_barrel = 0.99
  eff_plateau_endcap = 0.97
  eff_turnon_pt = 1.40
  eff_width_pt = 0.20

Input convention:
  tokens: float array [n_jets, max_constits, 14]
  mask:   bool array  [n_jets, max_constits]

Token columns are JetClass-style raw constituent columns:
  0 pt, 1 eta, 2 phi, 3 energy, 4 charge, 5:10 PID flags, 10:14 track attrs.

The output has the same shape as input tokens/mask and is sorted by descending
HLT pT per jet. All randomness is deterministic from the supplied seed and jet
index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

RAW_DIM = 14
IDX_PT = 0
IDX_ETA = 1
IDX_PHI = 2
IDX_E = 3

HLT_PROFILE_V1 = "fixed_hlt_v1"
HLT_PROFILE_V2_REALISTIC = "fixed_hlt_v2_realistic"
HLT_PROFILE_V2_REALISTIC_VERSION = "v1"


@dataclass(frozen=True)
class FixedHLTParams:
    """Fixed same-HLT profile used for the 7-model replication check."""

    hlt_pt_threshold: float = 1.30
    merge_prob_scale: float = 1.35
    reassign_scale: float = 1.00
    smear_scale: float = 1.00
    eff_plateau_barrel: float = 0.99
    eff_plateau_endcap: float = 0.97
    eff_turnon_pt: float = 1.40
    eff_width_pt: float = 0.20


@dataclass(frozen=True)
class FixedHLTV2Params:
    """Mild HLT profile whose strength=1 is the realistic target point.

    Unlike :class:`FixedHLTParams`, this profile is designed so all degradation
    terms scale away from the exact offline identity. The cache layer will add
    the chosen strength separately; these values are the already-scaled
    effective parameters used by the builder.
    """

    profile_name: str = HLT_PROFILE_V2_REALISTIC
    profile_version: str = HLT_PROFILE_V2_REALISTIC_VERSION
    hlt_pt_threshold: float = 0.20
    merge_radius: float = 0.0025
    merge_probability: float = 0.25
    eff_plateau_barrel: float = 0.999
    eff_plateau_endcap: float = 0.996
    eff_turnon_pt_barrel: float = 0.20
    eff_turnon_pt_endcap: float = 0.35
    eff_width_pt_barrel: float = 0.06
    eff_width_pt_endcap: float = 0.08
    density_loss_scale: float = 0.015
    jet_quality_sigma: float = 0.025
    smear_scale: float = 0.20
    tail_probability_base: float = 0.002
    tail_probability_eta: float = 0.0015
    tail_probability_density: float = 0.0015
    reassign_scale: float = 0.15


def scaled_fixed_hlt_params(strength: float = 1.0) -> FixedHLTParams:
    """Return a smoothly scaled version of the fixed HLT profile.

    ``strength=1`` is exactly the original fixed profile. Smaller values make
    the HLT view more offline-like while keeping the same degradation mechanism.
    This is an experimental control knob, not a detector recalibration.
    """

    strength = float(strength)
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("HLT degradation strength must be a finite nonnegative value")

    base = FixedHLTParams()
    return FixedHLTParams(
        hlt_pt_threshold=base.hlt_pt_threshold * strength,
        merge_prob_scale=base.merge_prob_scale * strength,
        reassign_scale=base.reassign_scale * strength,
        smear_scale=base.smear_scale * strength,
        eff_plateau_barrel=1.0 - strength * (1.0 - base.eff_plateau_barrel),
        eff_plateau_endcap=1.0 - strength * (1.0 - base.eff_plateau_endcap),
        eff_turnon_pt=base.eff_turnon_pt * strength,
        eff_width_pt=base.eff_width_pt * strength,
    )


def scaled_fixed_hlt_v2_realistic_params(strength: float = 1.0) -> FixedHLTV2Params:
    """Return the profile-aware mild HLT v2 parameters.

    The v2 strength convention is intentionally different from the old fixed
    profile: ``strength=1`` is the mild realistic target, and ``strength=0``
    represents exact offline identity. Values above one are stress tests.
    """

    strength = float(strength)
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("HLT v2 degradation strength must be a finite nonnegative value")

    base = FixedHLTV2Params()

    def plateau(target: float) -> float:
        return float(np.clip(1.0 - strength * (1.0 - float(target)), 0.0, 1.0))

    return FixedHLTV2Params(
        profile_name=base.profile_name,
        profile_version=base.profile_version,
        hlt_pt_threshold=base.hlt_pt_threshold * strength,
        merge_radius=base.merge_radius * strength,
        merge_probability=base.merge_probability * strength,
        eff_plateau_barrel=plateau(base.eff_plateau_barrel),
        eff_plateau_endcap=plateau(base.eff_plateau_endcap),
        eff_turnon_pt_barrel=base.eff_turnon_pt_barrel * strength,
        eff_turnon_pt_endcap=base.eff_turnon_pt_endcap * strength,
        eff_width_pt_barrel=base.eff_width_pt_barrel * strength,
        eff_width_pt_endcap=base.eff_width_pt_endcap * strength,
        density_loss_scale=base.density_loss_scale * strength,
        jet_quality_sigma=base.jet_quality_sigma * strength,
        smear_scale=base.smear_scale * strength,
        tail_probability_base=base.tail_probability_base * strength,
        tail_probability_eta=base.tail_probability_eta * strength,
        tail_probability_density=base.tail_probability_density * strength,
        reassign_scale=base.reassign_scale * strength,
    )


def wrap_phi_np(phi: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def compute_local_density_np(
    eta: np.ndarray,
    phi: np.ndarray,
    valid_idx: np.ndarray | None = None,
    radius: float = 0.04,
) -> np.ndarray:
    """Count nearby constituents inside a small delta-R cone for each token."""
    eta = np.asarray(eta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    if valid_idx is None:
        valid_idx = np.arange(len(eta), dtype=np.int64)
    valid_idx = np.asarray(valid_idx, dtype=np.int64)
    dens = np.zeros((len(eta),), dtype=np.float32)
    if len(valid_idx) <= 1:
        return dens
    eta_v = eta[valid_idx]
    phi_v = phi[valid_idx]
    for local_i, global_i in enumerate(valid_idx):
        deta = eta_v[local_i] - eta_v
        dphi = wrap_phi_np(phi_v[local_i] - phi_v)
        dr = np.sqrt(deta * deta + dphi * dphi)
        dens[global_i] = float(np.count_nonzero((dr < radius) & (dr > 0.0)))
    return dens


def merge_tokens_copy_dominant(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Merge kinematics while copying non-kinematic fields from dominant-energy token."""
    e1 = float(t1[IDX_E])
    e2 = float(t2[IDX_E])
    if e2 > e1 or (abs(e2 - e1) < 1e-8 and float(t2[IDX_PT]) > float(t1[IDX_PT])):
        dom = t2
    else:
        dom = t1

    out = dom.copy()
    pt1 = max(float(t1[IDX_PT]), 1e-8)
    pt2 = max(float(t2[IDX_PT]), 1e-8)
    pt_sum = pt1 + pt2
    w1 = pt1 / max(pt_sum, 1e-8)
    w2 = pt2 / max(pt_sum, 1e-8)

    eta = w1 * float(t1[IDX_ETA]) + w2 * float(t2[IDX_ETA])
    phi = math.atan2(
        w1 * math.sin(float(t1[IDX_PHI])) + w2 * math.sin(float(t2[IDX_PHI])),
        w1 * math.cos(float(t1[IDX_PHI])) + w2 * math.cos(float(t2[IDX_PHI])),
    )
    e = max(float(t1[IDX_E] + t2[IDX_E]), 1e-8)

    out[IDX_PT] = pt_sum
    out[IDX_ETA] = np.clip(eta, -5.0, 5.0)
    out[IDX_PHI] = wrap_phi_np(np.array([phi], dtype=np.float64))[0]
    out[IDX_E] = e
    return out.astype(np.float32)


def _empty_diag(n_offline: int = 0) -> Dict[str, float]:
    return {
        "n_offline": float(n_offline),
        "n_after_eff": float(n_offline),
        "n_after_threshold": float(n_offline),
        "n_after_merge": float(n_offline),
        "drop_eff": 0.0,
        "drop_threshold": 0.0,
        "drop_merge": 0.0,
        "drop_total": 0.0,
        "merge_count": 0.0,
    }


def _fixed_hlt_v2_is_identity(params: FixedHLTV2Params) -> bool:
    return (
        float(params.hlt_pt_threshold) <= 0.0
        and float(params.merge_radius) <= 0.0
        and float(params.merge_probability) <= 0.0
        and float(params.eff_plateau_barrel) >= 1.0
        and float(params.eff_plateau_endcap) >= 1.0
        and float(params.eff_turnon_pt_barrel) <= 0.0
        and float(params.eff_turnon_pt_endcap) <= 0.0
        and float(params.eff_width_pt_barrel) <= 0.0
        and float(params.eff_width_pt_endcap) <= 0.0
        and float(params.density_loss_scale) <= 0.0
        and float(params.jet_quality_sigma) <= 0.0
        and float(params.smear_scale) <= 0.0
        and float(params.tail_probability_base) <= 0.0
        and float(params.tail_probability_eta) <= 0.0
        and float(params.tail_probability_density) <= 0.0
        and float(params.reassign_scale) <= 0.0
    )


def apply_hlt_single_jet_m2style(
    tok: np.ndarray,
    msk: np.ndarray,
    params: FixedHLTParams,
    rng: np.random.RandomState,
    max_constits: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Apply the fixed m2-style HLT corruption to one JetClass jet."""
    diag = {
        "n_offline": 0.0,
        "n_after_eff": 0.0,
        "n_after_threshold": 0.0,
        "n_after_merge": 0.0,
        "drop_eff": 0.0,
        "drop_threshold": 0.0,
        "drop_merge": 0.0,
        "drop_total": 0.0,
        "merge_count": 0.0,
    }

    valid = tok[msk].copy()
    n0 = int(len(valid))
    diag["n_offline"] = float(n0)
    if n0 == 0:
        return (
            np.zeros((max_constits, RAW_DIM), dtype=np.float32),
            np.zeros((max_constits,), dtype=bool),
            diag,
        )

    # 1) Pre-threshold drop.
    keep_thr = valid[:, IDX_PT] >= float(params.hlt_pt_threshold)
    valid = valid[keep_thr]
    n_thr = int(len(valid))
    diag["n_after_threshold"] = float(n_thr)
    if n_thr == 0:
        diag["drop_threshold"] = float(n0)
        diag["drop_total"] = float(n0)
        return (
            np.zeros((max_constits, RAW_DIM), dtype=np.float32),
            np.zeros((max_constits,), dtype=bool),
            diag,
        )

    # 2) Type-agnostic local merging. merge_prob_scale acts as radius multiplier.
    merge_radius = 0.01 * float(max(0.05, params.merge_prob_scale))
    n_merged = 0
    if merge_radius > 0:
        to_remove: set[int] = set()
        for i in range(len(valid)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(valid)):
                if j in to_remove:
                    continue
                deta = float(valid[i, IDX_ETA] - valid[j, IDX_ETA])
                dphi = float(
                    math.atan2(
                        math.sin(float(valid[i, IDX_PHI] - valid[j, IDX_PHI])),
                        math.cos(float(valid[i, IDX_PHI] - valid[j, IDX_PHI])),
                    )
                )
                d_r = math.sqrt(deta * deta + dphi * dphi)
                if d_r >= merge_radius:
                    continue
                valid[i] = merge_tokens_copy_dominant(valid[i], valid[j])
                to_remove.add(j)
                n_merged += 1
        if to_remove:
            keep_idx = [k for k in range(len(valid)) if k not in to_remove]
            valid = valid[keep_idx]
    n_after_merge_raw = int(len(valid))

    # 3) Efficiency loss.
    if n_after_merge_raw > 0:
        eta = valid[:, IDX_ETA]
        phi = valid[:, IDX_PHI]
        pt = np.maximum(valid[:, IDX_PT], 1e-8)
        abs_eta = np.abs(eta)

        dens = compute_local_density_np(eta=eta, phi=phi, valid_idx=np.arange(len(valid)), radius=0.04)
        jet_quality = np.clip(rng.lognormal(mean=0.0, sigma=0.08), 0.75, 1.35)

        plateau = np.where(abs_eta < 1.5, float(params.eff_plateau_barrel), float(params.eff_plateau_endcap))
        pt50 = np.where(abs_eta < 1.5, float(params.eff_turnon_pt), float(params.eff_turnon_pt) + 0.30)
        width = np.where(abs_eta < 1.5, float(params.eff_width_pt), 1.25 * float(params.eff_width_pt))
        turn_on = 1.0 / (1.0 + np.exp(-(pt - pt50) / np.maximum(width, 1e-6)))
        density_term = np.exp(-0.055 * dens)
        q_eff = np.clip(jet_quality, 0.90, 1.06)

        eps = plateau * turn_on * density_term * q_eff
        eps = np.clip(eps, 0.02, 0.995)
        keep_eff = rng.random_sample(len(valid)) < eps
        valid = valid[keep_eff]
    n_eff = int(len(valid))
    diag["n_after_eff"] = float(n_eff)

    # 4) Smearing, tails, and local reassignment.
    if n_eff > 0:
        pt = np.maximum(valid[:, IDX_PT], 1e-8)
        eta = valid[:, IDX_ETA]
        phi = valid[:, IDX_PHI]
        abs_eta = np.abs(eta)
        dens = compute_local_density_np(eta=eta, phi=phi, valid_idx=np.arange(len(valid)), radius=0.04)
        q = float(np.clip(rng.lognormal(mean=0.0, sigma=0.08), 0.75, 1.35))

        smear_scale = float(max(0.0, params.smear_scale))
        reassign_scale = float(max(0.0, params.reassign_scale))

        sigma_rel = np.sqrt(
            ((0.35 * smear_scale) / np.sqrt(pt)) ** 2
            + (0.012 * smear_scale) ** 2
            + ((0.08 * smear_scale) / pt) ** 2
        )
        sigma_rel = sigma_rel * (1.0 + 0.08 * abs_eta) * q
        sigma_rel = np.clip(sigma_rel, 0.004, 0.40)

        tail_prob = 0.015 + 0.010 * abs_eta + 0.010 * dens
        tail_prob = np.clip(tail_prob, 0.0, 0.25)
        is_tail = rng.random_sample(len(valid)) < tail_prob

        ratio = rng.normal(loc=1.0, scale=sigma_rel)
        tail_sigma = 2.5 * sigma_rel + 0.015
        ratio_tail = rng.normal(loc=0.98, scale=tail_sigma)
        ratio[is_tail] = ratio_tail[is_tail]
        ratio = np.clip(ratio, 0.40, 1.60)
        pt_new = np.clip(pt * ratio, 1e-8, None)

        sigma_eta = (0.0008 * smear_scale + (0.010 * smear_scale) / np.sqrt(pt)) * (1.0 + 0.08 * abs_eta) * q
        sigma_phi = (0.0008 * smear_scale + (0.010 * smear_scale) / np.sqrt(pt)) * (1.0 + 0.08 * abs_eta) * q
        eta_new = eta + rng.normal(loc=0.0, scale=sigma_eta)
        phi_new = wrap_phi_np(phi + rng.normal(loc=0.0, scale=sigma_phi))

        if len(valid) > 1 and reassign_scale > 0.0:
            p_reassign = (0.01 + 0.006 * dens) * reassign_scale
            p_reassign = np.clip(p_reassign, 0.0, 0.08)
            do_reassign = rng.random_sample(len(valid)) < p_reassign
            for ii in np.where(do_reassign)[0]:
                deta = eta_new[ii] - eta_new
                dphi = wrap_phi_np(phi_new[ii] - phi_new)
                d_r = np.sqrt(deta * deta + dphi * dphi)
                d_r[ii] = 1e9
                nn = int(np.argmin(d_r))
                if d_r[nn] > 0.08:
                    continue
                lam = rng.uniform(0.20, 0.65)
                eta_new[ii] = (1.0 - lam) * eta_new[ii] + lam * eta_new[nn]
                phi_new[ii] = math.atan2(
                    (1.0 - lam) * math.sin(phi_new[ii]) + lam * math.sin(phi_new[nn]),
                    (1.0 - lam) * math.cos(phi_new[ii]) + lam * math.cos(phi_new[nn]),
                )

        eta_new = np.clip(eta_new, -5.0, 5.0)
        phi_new = wrap_phi_np(phi_new)
        e_new = pt_new * np.cosh(eta_new)

        valid[:, IDX_PT] = pt_new
        valid[:, IDX_ETA] = eta_new
        valid[:, IDX_PHI] = phi_new
        valid[:, IDX_E] = np.maximum(e_new, 1e-8)

    final = valid
    order = np.argsort(-final[:, IDX_PT]) if len(final) > 0 else np.array([], dtype=np.int64)
    final = final[order] if len(order) > 0 else final
    take = min(len(final), max_constits)
    out_tok = np.zeros((max_constits, RAW_DIM), dtype=np.float32)
    out_mask = np.zeros((max_constits,), dtype=bool)
    if take > 0:
        out_tok[:take] = final[:take]
        out_mask[:take] = True

    n_final_raw = int(len(final))
    diag["n_after_merge"] = float(n_final_raw)
    diag["drop_threshold"] = float(max(n0 - n_thr, 0))
    diag["drop_merge"] = float(max(n_thr - n_after_merge_raw, 0))
    diag["drop_eff"] = float(max(n_after_merge_raw - n_eff, 0))
    diag["drop_total"] = float(max(n0 - n_final_raw, 0))
    diag["merge_count"] = float(n_merged)
    return out_tok, out_mask, diag


def apply_hlt_single_jet_v2_realistic(
    tok: np.ndarray,
    msk: np.ndarray,
    params: FixedHLTV2Params,
    rng: np.random.RandomState,
    max_constits: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Apply the mild HLT v2 corruption to one JetClass jet."""

    if _fixed_hlt_v2_is_identity(params):
        out_tok = np.zeros((max_constits, RAW_DIM), dtype=np.float32)
        out_mask = np.zeros((max_constits,), dtype=bool)
        take = min(int(tok.shape[0]), max_constits)
        out_tok[:take] = np.asarray(tok[:take], dtype=np.float32)
        out_mask[:take] = np.asarray(msk[:take], dtype=bool)
        n0 = int(np.sum(out_mask))
        return out_tok, out_mask, _empty_diag(n0)

    valid = tok[msk].copy()
    n0 = int(len(valid))
    if n0 == 0:
        return (
            np.zeros((max_constits, RAW_DIM), dtype=np.float32),
            np.zeros((max_constits,), dtype=bool),
            _empty_diag(0),
        )

    diag = _empty_diag(n0)

    # 1) Mild low-pT threshold.
    threshold = float(max(0.0, params.hlt_pt_threshold))
    if threshold > 0.0:
        keep_thr = valid[:, IDX_PT] >= threshold
        valid = valid[keep_thr]
    n_thr = int(len(valid))
    diag["n_after_threshold"] = float(n_thr)
    if n_thr == 0:
        diag["n_after_eff"] = 0.0
        diag["n_after_merge"] = 0.0
        diag["drop_threshold"] = float(n0)
        diag["drop_total"] = float(n0)
        return (
            np.zeros((max_constits, RAW_DIM), dtype=np.float32),
            np.zeros((max_constits,), dtype=bool),
            diag,
        )

    # 2) Probabilistic local merging with no radius/probability floor.
    merge_radius = float(max(0.0, params.merge_radius))
    merge_probability = float(np.clip(params.merge_probability, 0.0, 1.0))
    n_merged = 0
    if merge_radius > 0.0 and merge_probability > 0.0 and len(valid) > 1:
        to_remove: set[int] = set()
        for i in range(len(valid)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(valid)):
                if j in to_remove:
                    continue
                deta = float(valid[i, IDX_ETA] - valid[j, IDX_ETA])
                dphi = float(
                    math.atan2(
                        math.sin(float(valid[i, IDX_PHI] - valid[j, IDX_PHI])),
                        math.cos(float(valid[i, IDX_PHI] - valid[j, IDX_PHI])),
                    )
                )
                d_r = math.sqrt(deta * deta + dphi * dphi)
                if d_r >= merge_radius or rng.random_sample() >= merge_probability:
                    continue
                valid[i] = merge_tokens_copy_dominant(valid[i], valid[j])
                to_remove.add(j)
                n_merged += 1
        if to_remove:
            keep_idx = [k for k in range(len(valid)) if k not in to_remove]
            valid = valid[keep_idx]
    n_after_merge_raw = int(len(valid))

    # 3) Efficiency loss. All non-identity terms are parameter controlled.
    if n_after_merge_raw > 0:
        eta = valid[:, IDX_ETA]
        phi = valid[:, IDX_PHI]
        pt = np.maximum(valid[:, IDX_PT], 1e-8)
        abs_eta = np.abs(eta)
        dens = compute_local_density_np(eta=eta, phi=phi, valid_idx=np.arange(len(valid)), radius=0.04)

        plateau = np.where(abs_eta < 1.5, float(params.eff_plateau_barrel), float(params.eff_plateau_endcap))
        pt50 = np.where(abs_eta < 1.5, float(params.eff_turnon_pt_barrel), float(params.eff_turnon_pt_endcap))
        width = np.where(abs_eta < 1.5, float(params.eff_width_pt_barrel), float(params.eff_width_pt_endcap))
        positive_width = width > 0.0
        turn_on = np.ones_like(pt, dtype=np.float64)
        if np.any(positive_width):
            turn_on[positive_width] = 1.0 / (
                1.0
                + np.exp(
                    -(pt[positive_width] - pt50[positive_width])
                    / np.maximum(width[positive_width], 1e-6)
                )
            )
        if np.any(~positive_width):
            turn_on[~positive_width] = (pt[~positive_width] >= pt50[~positive_width]).astype(np.float64)

        density_term = np.exp(-float(max(0.0, params.density_loss_scale)) * dens)
        quality_sigma = float(max(0.0, params.jet_quality_sigma))
        if quality_sigma > 0.0:
            jet_quality = np.clip(rng.lognormal(mean=0.0, sigma=quality_sigma), 0.75, 1.35)
            q_eff = np.clip(jet_quality, 0.90, 1.06)
        else:
            q_eff = 1.0

        eps = np.asarray(plateau * turn_on * density_term * q_eff, dtype=np.float64)
        eps = np.clip(eps, 0.0, 1.0)
        keep_eff = rng.random_sample(len(valid)) < eps
        valid = valid[keep_eff]
    n_eff = int(len(valid))
    diag["n_after_eff"] = float(n_eff)

    # 4) Mild pT/axis smearing, tails, and local reassignment.
    if n_eff > 0:
        pt = np.maximum(valid[:, IDX_PT], 1e-8)
        eta = valid[:, IDX_ETA]
        phi = valid[:, IDX_PHI]
        abs_eta = np.abs(eta)
        dens = compute_local_density_np(eta=eta, phi=phi, valid_idx=np.arange(len(valid)), radius=0.04)

        smear_scale = float(max(0.0, params.smear_scale))
        reassign_scale = float(max(0.0, params.reassign_scale))
        quality_sigma = float(max(0.0, params.jet_quality_sigma))
        q = float(np.clip(rng.lognormal(mean=0.0, sigma=quality_sigma), 0.75, 1.35)) if quality_sigma > 0 else 1.0

        if smear_scale > 0.0:
            sigma_rel = np.sqrt(
                ((0.35 * smear_scale) / np.sqrt(pt)) ** 2
                + (0.012 * smear_scale) ** 2
                + ((0.08 * smear_scale) / pt) ** 2
            )
            sigma_rel = sigma_rel * (1.0 + 0.08 * abs_eta) * q
            sigma_rel = np.clip(sigma_rel, 0.0, 0.25)

            tail_prob = (
                float(max(0.0, params.tail_probability_base))
                + float(max(0.0, params.tail_probability_eta)) * abs_eta
                + float(max(0.0, params.tail_probability_density)) * dens
            )
            tail_prob = np.clip(tail_prob, 0.0, 0.25)
            is_tail = rng.random_sample(len(valid)) < tail_prob

            ratio = rng.normal(loc=1.0, scale=sigma_rel)
            tail_sigma = 2.5 * sigma_rel + 0.015 * min(1.0, smear_scale)
            ratio_tail = rng.normal(loc=1.0 - 0.02 * min(1.0, smear_scale), scale=tail_sigma)
            ratio[is_tail] = ratio_tail[is_tail]
            ratio = np.clip(ratio, 0.55, 1.45)
            pt_new = np.clip(pt * ratio, 1e-8, None)

            sigma_eta = (0.0008 * smear_scale + (0.010 * smear_scale) / np.sqrt(pt)) * (1.0 + 0.08 * abs_eta) * q
            sigma_phi = (0.0008 * smear_scale + (0.010 * smear_scale) / np.sqrt(pt)) * (1.0 + 0.08 * abs_eta) * q
            eta_new = eta + rng.normal(loc=0.0, scale=sigma_eta)
            phi_new = wrap_phi_np(phi + rng.normal(loc=0.0, scale=sigma_phi))
        else:
            pt_new = pt
            eta_new = eta
            phi_new = phi

        if len(valid) > 1 and reassign_scale > 0.0:
            p_reassign = (0.01 + 0.006 * dens) * reassign_scale
            p_reassign = np.clip(p_reassign, 0.0, 0.08)
            do_reassign = rng.random_sample(len(valid)) < p_reassign
            for ii in np.where(do_reassign)[0]:
                deta = eta_new[ii] - eta_new
                dphi = wrap_phi_np(phi_new[ii] - phi_new)
                d_r = np.sqrt(deta * deta + dphi * dphi)
                d_r[ii] = 1e9
                nn = int(np.argmin(d_r))
                if d_r[nn] > 0.08:
                    continue
                lam = rng.uniform(0.20, 0.65)
                eta_new[ii] = (1.0 - lam) * eta_new[ii] + lam * eta_new[nn]
                phi_new[ii] = math.atan2(
                    (1.0 - lam) * math.sin(phi_new[ii]) + lam * math.sin(phi_new[nn]),
                    (1.0 - lam) * math.cos(phi_new[ii]) + lam * math.cos(phi_new[nn]),
                )

        eta_new = np.clip(eta_new, -5.0, 5.0)
        phi_new = wrap_phi_np(phi_new)
        e_new = pt_new * np.cosh(eta_new)

        valid[:, IDX_PT] = pt_new
        valid[:, IDX_ETA] = eta_new
        valid[:, IDX_PHI] = phi_new
        valid[:, IDX_E] = np.maximum(e_new, 1e-8)

    final = valid
    order = np.argsort(-final[:, IDX_PT]) if len(final) > 0 else np.array([], dtype=np.int64)
    final = final[order] if len(order) > 0 else final
    take = min(len(final), max_constits)
    out_tok = np.zeros((max_constits, RAW_DIM), dtype=np.float32)
    out_mask = np.zeros((max_constits,), dtype=bool)
    if take > 0:
        out_tok[:take] = final[:take]
        out_mask[:take] = True

    n_final_raw = int(len(final))
    diag["n_after_merge"] = float(n_final_raw)
    diag["drop_threshold"] = float(max(n0 - n_thr, 0))
    diag["drop_merge"] = float(max(n_thr - n_after_merge_raw, 0))
    diag["drop_eff"] = float(max(n_after_merge_raw - n_eff, 0))
    diag["drop_total"] = float(max(n0 - n_final_raw, 0))
    diag["merge_count"] = float(n_merged)
    return out_tok, out_mask, diag


def _validate_hlt_batch(tokens: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.asarray(tokens, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if tokens.ndim != 3 or tokens.shape[-1] != RAW_DIM:
        raise ValueError(f"tokens must have shape [n_jets, max_constits, {RAW_DIM}], got {tokens.shape}")
    if mask.shape != tokens.shape[:2]:
        raise ValueError(f"mask shape must match tokens[:2], got mask={mask.shape}, tokens={tokens.shape}")
    return tokens, mask


def build_fixed_hlt_view(
    tokens: np.ndarray,
    mask: np.ndarray,
    seed: int,
    params: FixedHLTParams | None = None,
    show_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Build the fixed same-HLT view for a batch of JetClass jets.

    Parameters
    ----------
    tokens:
        Raw JetClass constituent tokens, shape [n_jets, max_constits, 14].
    mask:
        Boolean valid-constituent mask, shape [n_jets, max_constits].
    seed:
        Split-specific seed. The original runs used base seeds like
        data_seed + 1001/1002/1003 for train/val/test.
    params:
        Optional override. Leave as None for the fixed profile.
    show_progress:
        If True, use tqdm if installed; otherwise prints no progress.

    Returns
    -------
    hlt_tokens, hlt_mask, diagnostics
    """
    params = params or FixedHLTParams()
    tokens, mask = _validate_hlt_batch(tokens, mask)

    n_jets = int(tokens.shape[0])
    max_constits = int(tokens.shape[1])
    out_tokens = np.zeros_like(tokens, dtype=np.float32)
    out_mask = np.zeros_like(mask, dtype=bool)
    diag_rows: List[Dict[str, float]] = []

    iterator = range(n_jets)
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Applying fixed JetClass HLT corruption")
        except Exception:
            pass

    for i in iterator:
        rng = np.random.RandomState(int(seed) + i * 37 + 11)
        ti, mi, di = apply_hlt_single_jet_m2style(tokens[i], mask[i], params, rng, max_constits)
        out_tokens[i] = ti
        out_mask[i] = mi
        diag_rows.append(di)

    keys = [
        "n_offline",
        "n_after_eff",
        "n_after_threshold",
        "n_after_merge",
        "drop_eff",
        "drop_threshold",
        "drop_merge",
        "drop_total",
        "merge_count",
    ]
    diagnostics = {k: np.array([row[k] for row in diag_rows], dtype=np.float32) for k in keys}
    return out_tokens, out_mask, diagnostics


def build_fixed_hlt_v2_realistic_view(
    tokens: np.ndarray,
    mask: np.ndarray,
    seed: int,
    params: FixedHLTV2Params | None = None,
    show_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Build the realistic mild HLT v2 view for a batch of JetClass jets."""

    params = params or FixedHLTV2Params()
    tokens, mask = _validate_hlt_batch(tokens, mask)
    n_jets = int(tokens.shape[0])
    max_constits = int(tokens.shape[1])

    if _fixed_hlt_v2_is_identity(params):
        counts = np.sum(mask, axis=1).astype(np.float32)
        diagnostics = {
            "n_offline": counts.copy(),
            "n_after_eff": counts.copy(),
            "n_after_threshold": counts.copy(),
            "n_after_merge": counts.copy(),
            "drop_eff": np.zeros((n_jets,), dtype=np.float32),
            "drop_threshold": np.zeros((n_jets,), dtype=np.float32),
            "drop_merge": np.zeros((n_jets,), dtype=np.float32),
            "drop_total": np.zeros((n_jets,), dtype=np.float32),
            "merge_count": np.zeros((n_jets,), dtype=np.float32),
        }
        return tokens.copy(), mask.copy(), diagnostics

    out_tokens = np.zeros_like(tokens, dtype=np.float32)
    out_mask = np.zeros_like(mask, dtype=bool)
    diag_rows: List[Dict[str, float]] = []

    iterator = range(n_jets)
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Applying realistic HLT v2 corruption")
        except Exception:
            pass

    for i in iterator:
        rng = np.random.RandomState(int(seed) + i * 37 + 11)
        ti, mi, di = apply_hlt_single_jet_v2_realistic(tokens[i], mask[i], params, rng, max_constits)
        out_tokens[i] = ti
        out_mask[i] = mi
        diag_rows.append(di)

    keys = [
        "n_offline",
        "n_after_eff",
        "n_after_threshold",
        "n_after_merge",
        "drop_eff",
        "drop_threshold",
        "drop_merge",
        "drop_total",
        "merge_count",
    ]
    diagnostics = {k: np.array([row[k] for row in diag_rows], dtype=np.float32) for k in keys}
    return out_tokens, out_mask, diagnostics


def summarize_hlt_diagnostics(diagnostics: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compact summary matching the counts used in logs."""
    n_off = np.maximum(diagnostics["n_offline"], 1.0)
    return {
        "mean_offline_constits": float(np.mean(diagnostics["n_offline"])),
        "mean_hlt_constits": float(np.mean(diagnostics["n_after_merge"])),
        "drop_eff_fraction": float(np.sum(diagnostics["drop_eff"]) / np.sum(n_off)),
        "drop_threshold_fraction": float(np.sum(diagnostics["drop_threshold"]) / np.sum(n_off)),
        "drop_merge_fraction": float(np.sum(diagnostics["drop_merge"]) / np.sum(n_off)),
        "drop_total_fraction": float(np.sum(diagnostics["drop_total"]) / np.sum(n_off)),
        "mean_merges_per_jet": float(np.mean(diagnostics["merge_count"])),
    }


# Backward-compatible alias for code that expects a generic name.
build_hlt_view = build_fixed_hlt_view
