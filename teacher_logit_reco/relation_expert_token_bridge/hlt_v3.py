"""Deterministic track-dominant HLT-v3 controlled proxy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import math
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fixed_hlt import (
    FixedHLTV2Params,
    compute_local_density_np,
    fixed_hlt_v2_efficiency_base_terms,
    fixed_hlt_v2_kinematic_base_terms,
    wrap_phi_np,
)

from .contracts import require_sha256, validate_content_hash, with_content_hash
from .replicas import RANDOM_MULTIPLIERS, event_rng_seed


HLT_V3_PROFILE_NAME = "fixed_hlt_v3_track_dominant_proxy"
HLT_V3_PROFILE_VERSION = "v1"
HLT_V3_PROFILE_CONTRACT = "retb_hlt_v3_profile_v1"
RAW_DIM = 14
INVALID_TRACK_SENTINEL = 0.0
PID_NAMES = (
    "charged_hadron",
    "neutral_hadron",
    "photon",
    "electron",
    "muon",
    "unknown",
)
CHARGED_PID = frozenset({0, 3, 4})
NEUTRAL_PID = frozenset({1, 2})
SUBSTREAM_IDS = {
    "merge": 11,
    "efficiency_quality": 21,
    "efficiency_loss": 22,
    "kinematic_quality": 31,
    "kinematic_core": 32,
    "kinematic_tail": 33,
    "angular_core": 34,
    "reassignment": 35,
    "pid_confusion": 41,
    "track_loss": 51,
    "track_error_scale": 52,
    "track_core": 53,
    "track_tail": 54,
    "charge_flip": 61,
}
OPERATION_ORDER = (
    "validate_offline_raw_schema",
    "mild_constituent_threshold",
    "type_aware_neutral_local_merging",
    "constituent_efficiency_loss",
    "mild_kinematic_response_and_local_reassignment",
    "pid_confusion",
    "charge_pid_consistency",
    "track_measurement_loss",
    "surviving_track_response",
    "rare_charge_flips",
    "mass_preserving_energy_recomputation",
    "zero_invalid_and_padded_rows",
    "stable_descending_pt_sort",
    "diagnostics_and_hashes",
)


@dataclass(frozen=True)
class HltV3Parameters:
    profile_name: str = HLT_V3_PROFILE_NAME
    profile_version: str = HLT_V3_PROFILE_VERSION
    hlt_pt_threshold: float = 0.10
    merge_radius: float = 0.0015
    merge_probability: float = 0.15
    eff_plateau_barrel: float = 0.9995
    eff_plateau_endcap: float = 0.9980
    eff_turnon_pt_barrel: float = 0.10
    eff_turnon_pt_endcap: float = 0.20
    eff_width_pt_barrel: float = 0.05
    eff_width_pt_endcap: float = 0.07
    density_loss_scale: float = 0.005
    jet_quality_sigma: float = 0.010
    kinematic_smear_scale: float = 0.080
    kinematic_tail_base: float = 0.0005
    kinematic_tail_eta: float = 0.0005
    kinematic_tail_density: float = 0.0005
    local_reassign_scale: float = 0.050

    def v2_base_parameters(self) -> FixedHLTV2Params:
        return FixedHLTV2Params(
            hlt_pt_threshold=self.hlt_pt_threshold,
            merge_radius=self.merge_radius,
            merge_probability=self.merge_probability,
            eff_plateau_barrel=self.eff_plateau_barrel,
            eff_plateau_endcap=self.eff_plateau_endcap,
            eff_turnon_pt_barrel=self.eff_turnon_pt_barrel,
            eff_turnon_pt_endcap=self.eff_turnon_pt_endcap,
            eff_width_pt_barrel=self.eff_width_pt_barrel,
            eff_width_pt_endcap=self.eff_width_pt_endcap,
            density_loss_scale=self.density_loss_scale,
            jet_quality_sigma=self.jet_quality_sigma,
            smear_scale=self.kinematic_smear_scale,
            tail_probability_base=self.kinematic_tail_base,
            tail_probability_eta=self.kinematic_tail_eta,
            tail_probability_density=self.kinematic_tail_density,
            reassign_scale=self.local_reassign_scale,
        )


@dataclass(frozen=True)
class DegradationProfile:
    profile_id: str
    strength: float
    threshold: bool
    merging: bool
    constituent_loss: bool
    kinematic_response: bool
    reassignment: bool
    pid_confusion: bool
    track_loss: bool
    track_response: bool
    charge_flip: bool
    legacy_profile: str | None = None


def _profile(
    profile_id: str,
    strength: float,
    *,
    kinematics: bool,
    constituent_missing: bool,
    track_missing: bool,
    track_response: bool,
    pid_charge: bool,
) -> DegradationProfile:
    return DegradationProfile(
        profile_id=profile_id,
        strength=strength,
        threshold=constituent_missing,
        merging=constituent_missing,
        constituent_loss=constituent_missing,
        kinematic_response=kinematics,
        reassignment=kinematics,
        pid_confusion=pid_charge,
        track_loss=track_missing,
        track_response=track_response,
        charge_flip=pid_charge,
    )


DEGRADATION_PROFILES = {
    "D_OFFLINE_IDENTITY": _profile(
        "D_OFFLINE_IDENTITY",
        0.0,
        kinematics=True,
        constituent_missing=True,
        track_missing=True,
        track_response=True,
        pid_charge=True,
    ),
    "D_KIN_ONLY": _profile(
        "D_KIN_ONLY",
        1.0,
        kinematics=True,
        constituent_missing=True,
        track_missing=False,
        track_response=False,
        pid_charge=False,
    ),
    "D_TRACK_ONLY": _profile(
        "D_TRACK_ONLY",
        1.0,
        kinematics=False,
        constituent_missing=False,
        track_missing=True,
        track_response=True,
        pid_charge=False,
    ),
    "D_MISSING_ONLY": _profile(
        "D_MISSING_ONLY",
        1.0,
        kinematics=False,
        constituent_missing=True,
        track_missing=True,
        track_response=False,
        pid_charge=False,
    ),
    "D_NOMINAL": _profile(
        "D_NOMINAL",
        1.0,
        kinematics=True,
        constituent_missing=True,
        track_missing=True,
        track_response=True,
        pid_charge=True,
    ),
    "D_MILD": _profile(
        "D_MILD",
        0.5,
        kinematics=True,
        constituent_missing=True,
        track_missing=True,
        track_response=True,
        pid_charge=True,
    ),
    "D_SEVERE": _profile(
        "D_SEVERE",
        1.5,
        kinematics=True,
        constituent_missing=True,
        track_missing=True,
        track_response=True,
        pid_charge=True,
    ),
    "D_LEGACY_V1": DegradationProfile(
        "D_LEGACY_V1",
        0.6,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        "fixed_hlt_v1",
    ),
    "D_LEGACY_V2": DegradationProfile(
        "D_LEGACY_V2",
        1.0,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        "fixed_hlt_v2_realistic",
    ),
}

TYPE_MULTIPLIERS = {
    "charged_hadron": (0.50, 0.45, 0.35, 0.00),
    "electron": (0.60, 0.55, 0.45, 0.00),
    "muon": (0.40, 0.35, 0.30, 0.00),
    "photon": (0.90, 0.85, 0.90, 1.00),
    "neutral_hadron": (1.35, 1.30, 1.25, 1.50),
    "unknown": (1.00, 1.00, 1.00, 1.00),
}
PID_TRANSITIONS = {
    0: ((3, 0.002), (4, 0.002)),
    3: ((0, 0.010), (4, 0.001)),
    4: ((0, 0.010), (3, 0.001)),
    1: ((2, 0.010),),
    2: ((1, 0.010),),
    5: (),
}


def degradation_profile(profile_id: str) -> DegradationProfile:
    try:
        profile = DEGRADATION_PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown degradation profile {profile_id!r}") from exc
    if profile.legacy_profile is not None:
        raise ValueError(
            f"{profile_id} is a comparison-only legacy profile, not an HLT-v3 mode"
        )
    return profile


def _pid_categories(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    flags = np.asarray(tokens[:, 5:10], dtype=np.float64)
    if np.any(np.abs(flags - np.rint(flags)) > 1.0e-6):
        raise ValueError("PID flags must be binary within tolerance")
    binary = np.rint(flags).astype(np.int8)
    counts = np.sum(binary, axis=1)
    if np.any(counts[mask] > 1):
        raise ValueError("multi-hot PID input is invalid")
    categories = np.full((len(tokens),), 5, dtype=np.int8)
    one_hot = counts == 1
    categories[one_hot] = np.argmax(binary[one_hot], axis=1).astype(np.int8)
    return categories


def _validate_raw_tokens(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if tokens.ndim != 2 or tokens.shape[1] != RAW_DIM:
        raise ValueError(f"single-jet tokens must have shape [N,{RAW_DIM}]")
    if mask.shape != (tokens.shape[0],):
        raise ValueError("single-jet mask shape mismatch")
    valid = np.asarray(mask, dtype=bool)
    if not bool(np.isfinite(tokens[valid]).all()):
        raise ValueError("valid raw-token values must be finite")
    if np.any(tokens[valid, 0] < 0.0) or np.any(tokens[valid, 3] < 0.0):
        raise ValueError("valid pT and energy must be nonnegative")
    charge = tokens[valid, 4]
    if np.any(np.min(np.abs(charge[:, None] - np.array([-1, 0, 1])), axis=1) > 1e-6):
        raise ValueError("charge must be one of -1, 0, +1")
    categories = _pid_categories(tokens, valid)
    neutral = np.isin(categories, list(NEUTRAL_PID))
    if np.any(np.abs(tokens[neutral & valid, 4]) > 1e-6):
        raise ValueError("neutral PID rows must have zero charge")
    return categories


def measurement_validity_states(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    tokens = np.asarray(tokens)
    mask = np.asarray(mask, dtype=bool)
    categories = _pid_categories(tokens, mask)
    states = np.zeros((len(tokens),), dtype=np.int8)
    charged = mask & np.isin(categories, list(CHARGED_PID))
    available = (
        charged
        & np.all(np.isfinite(tokens[:, 10:14]), axis=1)
        & (tokens[:, 11] > 0.0)
        & (tokens[:, 13] > 0.0)
    )
    states[charged] = 2
    states[available] = 1
    return states


def _substream_seed(base_seed: int, family: str) -> int:
    digest = hashlib.sha256()
    digest.update(b"retb_hlt_v3_substream_v1\0")
    digest.update(int(base_seed).to_bytes(8, "big", signed=False))
    digest.update(b"\0")
    digest.update(str(SUBSTREAM_IDS[family]).encode("ascii"))
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


def _rng(base_seed: int, family: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(_substream_seed(base_seed, family)))


def _four_vector(token: np.ndarray) -> np.ndarray:
    pt, eta, phi, energy = (float(token[index]) for index in range(4))
    return np.array(
        [pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta), energy],
        dtype=np.float64,
    )


def _mass_from_token(token: np.ndarray) -> float:
    px, py, pz, energy = _four_vector(token)
    return math.sqrt(max(energy * energy - px * px - py * py - pz * pz, 0.0))


def merge_equal_neutral_tokens(
    first: np.ndarray,
    second: np.ndarray,
    *,
    category: int,
) -> tuple[np.ndarray, float]:
    if category not in NEUTRAL_PID:
        raise ValueError("only neutral-hadron or photon rows may merge")
    vector = _four_vector(first) + _four_vector(second)
    px, py, pz, energy = (float(value) for value in vector)
    pt = math.hypot(px, py)
    phi = math.atan2(py, px)
    eta = math.asinh(pz / max(pt, 1.0e-12)) if pt > 0.0 else 0.0
    if not all(math.isfinite(value) for value in (pt, eta, phi, energy)):
        raise FloatingPointError("merged four-vector is nonfinite")
    mass = math.sqrt(max(energy * energy - px * px - py * py - pz * pz, 0.0))
    output = np.zeros((RAW_DIM,), dtype=np.float64)
    output[:4] = (pt, eta, phi, max(energy, 0.0))
    output[4] = 0.0
    output[5 + category] = 1.0
    output[10:14] = INVALID_TRACK_SENTINEL
    return output, mass


def _replica_multipliers(policy: str, replica_id: int) -> tuple[float, float, float, float]:
    if policy not in {"R_FIXED", "R_MULTI", "R_RANDOM"}:
        raise ValueError(f"unknown realization policy {policy!r}")
    if int(replica_id) not in range(4):
        raise ValueError("replica_id must be in [0,3]")
    if policy != "R_RANDOM":
        return (1.0, 1.0, 1.0, 1.0)
    row = RANDOM_MULTIPLIERS[str(int(replica_id))]
    return (
        float(row["kinematic"]),
        float(row["track_loss"]),
        float(row["track_core_noise"]),
        float(row["tail_probability"]),
    )


def _clip01(value: np.ndarray | float) -> np.ndarray:
    return np.clip(value, 0.0, 1.0)


def scale_mechanism_terms(
    base_terms: Mapping[str, np.ndarray | float],
    *,
    pid_category: int,
    strength: float,
    replica_multipliers: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
) -> dict[str, np.ndarray]:
    """Apply the locked type/strength/replica multiplication table.

    This deliberately contains no random draws.  It is the auditable adapter
    between the parity-tested v2 base-term helpers and the v3 generator.
    """

    if int(pid_category) not in range(len(PID_NAMES)):
        raise ValueError("pid_category lies outside the six-domain schema")
    if float(strength) < 0.0:
        raise ValueError("degradation strength must be nonnegative")
    if len(replica_multipliers) != 4:
        raise ValueError("four replica-family multipliers are required")
    r_kin, _r_track_loss, _r_track_core, r_tail = (
        float(value) for value in replica_multipliers
    )
    if min(r_kin, _r_track_loss, _r_track_core, r_tail) < 0.0:
        raise ValueError("replica-family multipliers must be nonnegative")
    a_loss, a_p, a_ang, a_reassign = TYPE_MULTIPLIERS[
        PID_NAMES[int(pid_category)]
    ]
    s = float(strength)

    def array(name: str) -> np.ndarray:
        if name not in base_terms:
            raise KeyError(f"v2 base terms lack {name!r}")
        value = np.asarray(base_terms[name], dtype=np.float64)
        if not bool(np.isfinite(value).all()):
            raise FloatingPointError(f"v2 base term {name!r} is nonfinite")
        return value

    output = {
        "reassignment_delta_scale": np.asarray(
            a_reassign * s * r_kin, dtype=np.float64
        ),
        "kinematic_tail_delta_scale": np.asarray(
            a_p * s * r_kin, dtype=np.float64
        ),
    }
    if "loss_probability" in base_terms:
        output["loss_probability"] = _clip01(
            array("loss_probability") * a_loss * s * r_kin
        )
    if "sigma_p" in base_terms:
        output["sigma_p"] = np.minimum(
            array("sigma_p") * a_p * s * r_kin, 0.25
        )
    if "tail_probability" in base_terms:
        output["kinematic_tail_probability"] = _clip01(
            array("tail_probability") * a_p * s * r_tail
        )
    if "sigma_eta" in base_terms:
        output["sigma_eta"] = np.minimum(
            array("sigma_eta") * a_ang * s * r_kin, 0.25
        )
    if "sigma_phi" in base_terms:
        output["sigma_phi"] = np.minimum(
            array("sigma_phi") * a_ang * s * r_kin, 0.25
        )
    if "reassignment_probability" in base_terms:
        output["reassignment_probability"] = _clip01(
            array("reassignment_probability") * a_reassign * s * r_kin
        )
    return output


def _stable_pt_order(tokens: np.ndarray, canonical_indices: np.ndarray) -> np.ndarray:
    return np.lexsort((canonical_indices, -tokens[:, 0]))


def apply_hlt_v3_single_jet(
    tokens: np.ndarray,
    mask: np.ndarray,
    *,
    canonical_identity: str,
    logical_role: str,
    replica_id: int,
    realization_policy: str = "R_MULTI",
    profile_id: str = "D_NOMINAL",
    parameters: HltV3Parameters | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    parameters = parameters or HltV3Parameters()
    profile = degradation_profile(profile_id)
    tokens = np.asarray(tokens)
    mask = np.asarray(mask, dtype=bool)
    categories = _validate_raw_tokens(tokens, mask)
    strength = float(profile.strength)
    if strength == 0.0:
        copied_tokens = np.array(tokens, copy=True)
        copied_mask = np.array(mask, copy=True)
        states = measurement_validity_states(copied_tokens, copied_mask)
        return copied_tokens, copied_mask, states, {
            "profile_id": profile_id,
            "strength": 0.0,
            "identity_short_circuit": True,
            "rng_constructed": False,
            "n_offline": int(np.sum(mask)),
            "n_output": int(np.sum(mask)),
            "operation_order": list(OPERATION_ORDER),
        }

    base_seed = event_rng_seed(
        logical_role=logical_role,
        replica_id=replica_id,
        canonical_identity=canonical_identity,
    )
    r_kin, r_track_loss, r_track_core, r_tail = _replica_multipliers(
        realization_policy, replica_id
    )
    valid_indices = np.flatnonzero(mask)
    rows = np.asarray(tokens[valid_indices], dtype=np.float64).copy()
    row_categories = categories[valid_indices].astype(np.int8, copy=True)
    canonical_indices = valid_indices.astype(np.int64, copy=True)
    source_masses = np.array([_mass_from_token(row) for row in rows], dtype=np.float64)
    original_density = compute_local_density_np(
        rows[:, 1],
        rows[:, 2],
        np.arange(len(rows), dtype=np.int64),
        radius=0.04,
    ).astype(np.float64)
    diagnostics: dict[str, Any] = {
        "profile_id": profile_id,
        "strength": strength,
        "identity_short_circuit": False,
        "rng_constructed": True,
        "n_offline": int(len(rows)),
        "operation_order": list(OPERATION_ORDER),
        "mechanism_counts": {
            "threshold_drop": 0,
            "merge": 0,
            "constituent_loss": 0,
            "reassignment": 0,
            "pid_transition": 0,
            "track_loss": 0,
            "track_tail": 0,
            "charge_flip": 0,
        },
        "probability_sums": {},
        "type_input_counts": {
            name: int(np.sum(row_categories == index))
            for index, name in enumerate(PID_NAMES)
        },
        "replica_multipliers": {
            "kinematic": r_kin,
            "track_loss": r_track_loss,
            "track_core": r_track_core,
            "tail": r_tail,
        },
    }
    if len(rows) == 0:
        return (
            np.zeros_like(tokens),
            np.zeros_like(mask),
            np.zeros_like(mask, dtype=np.int8),
            {**diagnostics, "n_output": 0},
        )

    if profile.threshold:
        threshold = parameters.hlt_pt_threshold * strength * r_kin
        keep = rows[:, 0] >= threshold
        diagnostics["mechanism_counts"]["threshold_drop"] = int(np.sum(~keep))
        rows = rows[keep]
        row_categories = row_categories[keep]
        canonical_indices = canonical_indices[keep]
        source_masses = source_masses[keep]
        original_density = original_density[keep]

    if profile.merging and len(rows) > 1:
        merge_rng = _rng(base_seed, "merge")
        radius = parameters.merge_radius * strength * r_kin
        probability = float(_clip01(parameters.merge_probability * strength * r_kin))
        removed: set[int] = set()
        for index in range(len(rows)):
            if index in removed or row_categories[index] not in NEUTRAL_PID:
                continue
            for other in range(index + 1, len(rows)):
                if other in removed or row_categories[other] != row_categories[index]:
                    continue
                deta = rows[index, 1] - rows[other, 1]
                dphi = float(wrap_phi_np(np.array([rows[index, 2] - rows[other, 2]]))[0])
                if math.hypot(deta, dphi) >= radius:
                    continue
                if merge_rng.random() >= probability:
                    continue
                rows[index], source_masses[index] = merge_equal_neutral_tokens(
                    rows[index],
                    rows[other],
                    category=int(row_categories[index]),
                )
                canonical_indices[index] = min(
                    canonical_indices[index], canonical_indices[other]
                )
                original_density[index] = max(
                    original_density[index], original_density[other]
                )
                removed.add(other)
                diagnostics["mechanism_counts"]["merge"] += 1
        if removed:
            keep = np.array(
                [index not in removed for index in range(len(rows))], dtype=bool
            )
            rows = rows[keep]
            row_categories = row_categories[keep]
            canonical_indices = canonical_indices[keep]
            source_masses = source_masses[keep]
            original_density = original_density[keep]

    if profile.constituent_loss and len(rows):
        quality = float(
            np.clip(
                _rng(base_seed, "efficiency_quality").lognormal(
                    mean=0.0, sigma=parameters.jet_quality_sigma
                ),
                0.75,
                1.35,
            )
        )
        base = fixed_hlt_v2_efficiency_base_terms(
            pt=rows[:, 0],
            eta=rows[:, 1],
            density=original_density,
            params=parameters.v2_base_parameters(),
            jet_quality=quality,
        )
        p_loss = np.empty((len(rows),), dtype=np.float64)
        for category in range(len(PID_NAMES)):
            selected = row_categories == category
            if np.any(selected):
                scaled = scale_mechanism_terms(
                    {"loss_probability": base["loss_probability"][selected]},
                    pid_category=category,
                    strength=strength,
                    replica_multipliers=(
                        r_kin,
                        r_track_loss,
                        r_track_core,
                        r_tail,
                    ),
                )
                p_loss[selected] = scaled["loss_probability"]
        diagnostics["probability_sums"]["constituent_loss"] = float(np.sum(p_loss))
        keep = _rng(base_seed, "efficiency_loss").random(len(rows)) >= p_loss
        diagnostics["mechanism_counts"]["constituent_loss"] = int(np.sum(~keep))
        rows = rows[keep]
        row_categories = row_categories[keep]
        canonical_indices = canonical_indices[keep]
        source_masses = source_masses[keep]
        original_density = original_density[keep]

    if profile.kinematic_response and len(rows):
        quality = float(
            np.clip(
                _rng(base_seed, "kinematic_quality").lognormal(
                    mean=0.0, sigma=parameters.jet_quality_sigma
                ),
                0.75,
                1.35,
            )
        )
        base = fixed_hlt_v2_kinematic_base_terms(
            pt=rows[:, 0],
            eta=rows[:, 1],
            density=original_density,
            params=parameters.v2_base_parameters(),
            jet_quality=quality,
        )
        sigma_p = np.empty((len(rows),), dtype=np.float64)
        p_tail = np.empty((len(rows),), dtype=np.float64)
        tail_delta_scale = np.empty((len(rows),), dtype=np.float64)
        sigma_eta = np.empty((len(rows),), dtype=np.float64)
        sigma_phi = np.empty((len(rows),), dtype=np.float64)
        p_reassign = np.empty((len(rows),), dtype=np.float64)
        reassign_delta_scale = np.empty((len(rows),), dtype=np.float64)
        for category in range(len(PID_NAMES)):
            selected = row_categories == category
            if not np.any(selected):
                continue
            scaled = scale_mechanism_terms(
                {name: value[selected] for name, value in base.items()},
                pid_category=category,
                strength=strength,
                replica_multipliers=(
                    r_kin,
                    r_track_loss,
                    r_track_core,
                    r_tail,
                ),
            )
            sigma_p[selected] = scaled["sigma_p"]
            p_tail[selected] = scaled["kinematic_tail_probability"]
            tail_delta_scale[selected] = scaled["kinematic_tail_delta_scale"]
            sigma_eta[selected] = scaled["sigma_eta"]
            sigma_phi[selected] = scaled["sigma_phi"]
            p_reassign[selected] = scaled["reassignment_probability"]
            reassign_delta_scale[selected] = scaled["reassignment_delta_scale"]
        core_rng = _rng(base_seed, "kinematic_core")
        log_ratio = core_rng.normal(size=len(rows)) * sigma_p
        tail_rng = _rng(base_seed, "kinematic_tail")
        tail_mask = tail_rng.random(len(rows)) < p_tail
        tail_delta = (
            (base["tail_mean"] - 1.0)
            + base["tail_sigma"] * tail_rng.normal(size=len(rows))
        ) * tail_delta_scale
        log_ratio[tail_mask] = tail_delta[tail_mask]
        rows[:, 0] *= np.clip(np.exp(log_ratio), 0.55, 1.45)
        angular_rng = _rng(base_seed, "angular_core")
        rows[:, 1] += angular_rng.normal(size=len(rows)) * sigma_eta
        rows[:, 1] = np.clip(rows[:, 1], -5.0, 5.0)
        rows[:, 2] = wrap_phi_np(
            rows[:, 2] + angular_rng.normal(size=len(rows)) * sigma_phi
        )

        if profile.reassignment and len(rows) > 1:
            reassign_rng = _rng(base_seed, "reassignment")
            selected = reassign_rng.random(len(rows)) < p_reassign
            for index in np.flatnonzero(selected):
                if row_categories[index] in CHARGED_PID:
                    raise AssertionError("charged rows may not be locally reassigned")
                deta = rows[index, 1] - rows[:, 1]
                dphi = wrap_phi_np(rows[index, 2] - rows[:, 2])
                distances = np.hypot(deta, dphi)
                distances[index] = np.inf
                nearest = int(np.argmin(distances))
                if distances[nearest] > 0.08:
                    continue
                amplitude = min(
                    1.0,
                    reassign_delta_scale[index],
                )
                fraction = reassign_rng.uniform(0.20, 0.65) * amplitude
                rows[index, 1] = (
                    (1.0 - fraction) * rows[index, 1]
                    + fraction * rows[nearest, 1]
                )
                rows[index, 2] = math.atan2(
                    (1.0 - fraction) * math.sin(rows[index, 2])
                    + fraction * math.sin(rows[nearest, 2]),
                    (1.0 - fraction) * math.cos(rows[index, 2])
                    + fraction * math.cos(rows[nearest, 2]),
                )
                diagnostics["mechanism_counts"]["reassignment"] += 1

    if profile.pid_confusion and len(rows):
        pid_rng = _rng(base_seed, "pid_confusion")
        draws = pid_rng.random(len(rows))
        for index, category in enumerate(row_categories.copy()):
            cumulative = 0.0
            for target, probability in PID_TRANSITIONS[int(category)]:
                cumulative += probability * strength
                if draws[index] < min(cumulative, 1.0):
                    row_categories[index] = target
                    diagnostics["mechanism_counts"]["pid_transition"] += 1
                    break
        rows[:, 5:10] = 0.0
        known = row_categories < 5
        rows[np.flatnonzero(known), 5 + row_categories[known]] = 1.0

    neutral = np.isin(row_categories, list(NEUTRAL_PID))
    rows[neutral, 4] = 0.0
    rows[neutral, 10:14] = INVALID_TRACK_SENTINEL
    track_states = measurement_validity_states(
        rows.astype(np.float32),
        np.ones((len(rows),), dtype=bool),
    )

    if profile.track_loss and len(rows):
        eligible = track_states == 1
        abs_eta = np.abs(rows[:, 1])
        sigmoid = 1.0 / (1.0 + np.exp(-(0.80 - rows[:, 0]) / 0.25))
        p_track_loss_base = np.clip(
            0.030
            + 0.030 * (abs_eta >= 1.5)
            + 0.080 * sigmoid
            + 0.020 * np.minimum(original_density / 8.0, 1.0),
            0.0,
            0.35,
        )
        p_track_loss = _clip01(
            p_track_loss_base * strength * r_track_loss
        )
        diagnostics["probability_sums"]["track_loss"] = float(
            np.sum(p_track_loss[eligible])
        )
        lost = (
            _rng(base_seed, "track_loss").random(len(rows)) < p_track_loss
        ) & eligible
        rows[lost, 10:14] = INVALID_TRACK_SENTINEL
        diagnostics["mechanism_counts"]["track_loss"] = int(np.sum(lost))
        track_states[lost] = 2

    if profile.track_response and len(rows):
        surviving = track_states == 1
        if np.any(surviving):
            original_d0_error = rows[:, 11].copy()
            original_dz_error = rows[:, 13].copy()
            error_rng = _rng(base_seed, "track_error_scale")
            error_z = error_rng.normal(size=(len(rows), 2))
            displacement = strength * r_track_core
            rows[surviving, 11] *= np.exp(
                displacement
                * (math.log(1.35) + 0.15 * error_z[surviving, 0])
            )
            rows[surviving, 13] *= np.exp(
                displacement
                * (math.log(1.30) + 0.15 * error_z[surviving, 1])
            )
            core_rng = _rng(base_seed, "track_core")
            z0 = core_rng.normal(size=len(rows))
            z1 = core_rng.normal(size=len(rows))
            correlated = 0.25 * z0 + math.sqrt(1.0 - 0.25**2) * z1
            abs_eta = np.abs(rows[:, 1])
            p_track_tail_base = np.clip(
                0.010
                + 0.005 * (abs_eta >= 1.5)
                + 0.002 * np.minimum(original_density, 5.0),
                0.0,
                0.08,
            )
            p_track_tail = _clip01(
                p_track_tail_base * strength * r_tail
            )
            tail = (
                _rng(base_seed, "track_tail").random(len(rows)) < p_track_tail
            ) & surviving
            tail_scale = np.where(tail, 4.0, 1.0)
            rows[surviving, 10] += (
                0.75
                * original_d0_error[surviving]
                * z0[surviving]
                * displacement
                * tail_scale[surviving]
            )
            rows[surviving, 12] += (
                0.65
                * original_dz_error[surviving]
                * correlated[surviving]
                * displacement
                * tail_scale[surviving]
            )
            diagnostics["mechanism_counts"]["track_tail"] = int(np.sum(tail))

    if profile.charge_flip and len(rows):
        eligible = np.isin(row_categories, list(CHARGED_PID)) & np.isin(
            np.rint(rows[:, 4]).astype(np.int8), [-1, 1]
        )
        p_flip = _clip01(
            (
                0.002
                + 0.002 * (np.abs(rows[:, 1]) >= 1.5)
                + 0.001 * np.minimum(rows[:, 0] / 100.0, 1.0)
            )
            * strength
        )
        flipped = (
            _rng(base_seed, "charge_flip").random(len(rows)) < p_flip
        ) & eligible
        rows[flipped, 4] *= -1.0
        diagnostics["mechanism_counts"]["charge_flip"] = int(np.sum(flipped))

    momentum = rows[:, 0] * np.cosh(rows[:, 1])
    rows[:, 3] = np.sqrt(np.maximum(momentum * momentum + source_masses**2, 0.0))
    if not bool(np.isfinite(rows).all()):
        raise FloatingPointError("HLT-v3 output contains nonfinite values")
    order = _stable_pt_order(rows, canonical_indices)
    rows = rows[order]
    canonical_indices = canonical_indices[order]
    take = min(len(rows), tokens.shape[0])
    output = np.zeros(tokens.shape, dtype=np.float32)
    output_mask = np.zeros(mask.shape, dtype=bool)
    output[:take] = rows[:take].astype(np.float32)
    output_mask[:take] = True
    states = measurement_validity_states(output, output_mask)
    diagnostics["n_output"] = int(take)
    diagnostics["canonical_output_indices"] = canonical_indices[:take].tolist()
    diagnostics["type_output_counts"] = {
        name: int(np.sum(_pid_categories(output, output_mask)[output_mask] == index))
        for index, name in enumerate(PID_NAMES)
    }
    diagnostics["measurement_states"] = {
        "not_track_domain": int(np.sum(states[output_mask] == 0)),
        "available": int(np.sum(states[output_mask] == 1)),
        "missing": int(np.sum(states[output_mask] == 2)),
    }
    return output, output_mask, states, diagnostics


def build_hlt_v3_view(
    tokens: np.ndarray,
    mask: np.ndarray,
    *,
    canonical_identities: Sequence[str],
    logical_role: str,
    replica_id: int,
    realization_policy: str = "R_MULTI",
    profile_id: str = "D_NOMINAL",
    parameters: HltV3Parameters | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    tokens = np.asarray(tokens)
    mask = np.asarray(mask, dtype=bool)
    if tokens.ndim != 3 or tokens.shape[-1] != RAW_DIM:
        raise ValueError(f"tokens must have shape [B,N,{RAW_DIM}]")
    if mask.shape != tokens.shape[:2]:
        raise ValueError("batch mask shape mismatch")
    if len(canonical_identities) != len(tokens):
        raise ValueError("canonical identity count differs from batch")
    outputs = np.empty_like(tokens)
    output_masks = np.empty_like(mask)
    states = np.empty_like(mask, dtype=np.int8)
    diagnostics: list[dict[str, Any]] = []
    for index, identity in enumerate(canonical_identities):
        output, output_mask, state, diagnostic = apply_hlt_v3_single_jet(
            tokens[index],
            mask[index],
            canonical_identity=str(identity),
            logical_role=logical_role,
            replica_id=replica_id,
            realization_policy=realization_policy,
            profile_id=profile_id,
            parameters=parameters,
        )
        outputs[index] = output
        output_masks[index] = output_mask
        states[index] = state
        diagnostics.append(diagnostic)
    return outputs, output_masks, states, diagnostics


def _function_sha256(function: Any) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def build_hlt_v3_profile_contract(
    *,
    raw_input_schema_sha256: str,
    hlt_replica_manifest_sha256: str,
) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": HLT_V3_PROFILE_CONTRACT,
            "schema_version": 1,
            "profile_name": HLT_V3_PROFILE_NAME,
            "profile_version": HLT_V3_PROFILE_VERSION,
            "proxy_claim": "HLT_like_controlled_proxy_not_real_HLT",
            "nominal_strength": 1.0,
            "parameters": asdict(HltV3Parameters()),
            "type_multiplier_order": [
                "constituent_loss",
                "momentum",
                "angular",
                "local_reassignment",
            ],
            "type_multipliers": TYPE_MULTIPLIERS,
            "pid_transitions": {
                PID_NAMES[source]: [
                    {"target": PID_NAMES[target], "probability": probability}
                    for target, probability in transitions
                ]
                for source, transitions in PID_TRANSITIONS.items()
            },
            "invalid_track_sentinel": [0.0, 0.0, 0.0, 0.0],
            "measurement_states": [
                "not_track_domain",
                "available",
                "missing",
            ],
            "operation_order": list(OPERATION_ORDER),
            "substream_ids": SUBSTREAM_IDS,
            "degradation_profiles": {
                name: asdict(profile)
                for name, profile in DEGRADATION_PROFILES.items()
            },
            "v2_base_term_helpers": {
                "efficiency": {
                    "qualified_name": (
                        "jetclass_fixed_hlt.fixed_hlt_v2_efficiency_base_terms"
                    ),
                    "source_sha256": _function_sha256(
                        fixed_hlt_v2_efficiency_base_terms
                    ),
                },
                "kinematic": {
                    "qualified_name": (
                        "jetclass_fixed_hlt.fixed_hlt_v2_kinematic_base_terms"
                    ),
                    "source_sha256": _function_sha256(
                        fixed_hlt_v2_kinematic_base_terms
                    ),
                },
            },
            "raw_input_schema_sha256": require_sha256(
                raw_input_schema_sha256, name="raw_input_schema_sha256"
            ),
            "hlt_replica_manifest_sha256": require_sha256(
                hlt_replica_manifest_sha256,
                name="hlt_replica_manifest_sha256",
            ),
            "fake_duplicate_split_constituents": False,
            "strength_zero_rng_constructed": False,
            "derived_features_rebuilt_from_degraded_view_only": True,
        }
    )


def validate_hlt_v3_profile_contract(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=HLT_V3_PROFILE_CONTRACT
    )
    helper_rows = payload["v2_base_term_helpers"]
    expected = {
        "efficiency": _function_sha256(fixed_hlt_v2_efficiency_base_terms),
        "kinematic": _function_sha256(fixed_hlt_v2_kinematic_base_terms),
    }
    for name, source_hash in expected.items():
        if helper_rows[name]["source_sha256"] != source_hash:
            raise ValueError(f"v2 base-term helper source drifted for {name}")
    return digest


__all__ = [
    "DEGRADATION_PROFILES",
    "HLT_V3_PROFILE_CONTRACT",
    "HLT_V3_PROFILE_NAME",
    "HLT_V3_PROFILE_VERSION",
    "HltV3Parameters",
    "INVALID_TRACK_SENTINEL",
    "OPERATION_ORDER",
    "PID_NAMES",
    "SUBSTREAM_IDS",
    "TYPE_MULTIPLIERS",
    "apply_hlt_v3_single_jet",
    "build_hlt_v3_profile_contract",
    "build_hlt_v3_view",
    "degradation_profile",
    "measurement_validity_states",
    "merge_equal_neutral_tokens",
    "scale_mechanism_terms",
    "validate_hlt_v3_profile_contract",
]
