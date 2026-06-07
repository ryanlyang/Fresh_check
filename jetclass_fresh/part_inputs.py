"""Build Particle Transformer input groups from any JetClass constituent view.

The central rule is that all derived jet axes, relative kinematics, and
four-vectors are computed from the supplied view only. Passing an HLT view or a
reconstructed view therefore cannot accidentally reuse offline jet kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

import numpy as np

from .jetclass_data import JetView, RAW_TOKEN_DIM


EPS = 1e-8

PF_POINT_NAMES = [
    "part_deta",
    "part_dphi",
]

PF_FEATURE_NAMES = [
    "part_pt_log",
    "part_e_log",
    "part_logptrel",
    "part_logerel",
    "part_deltaR",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
    "part_d0",
    "part_d0err",
    "part_dz",
    "part_dzerr",
    "part_deta",
    "part_dphi",
]

PF_VECTOR_NAMES = [
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
]

PF_MASK_NAMES = [
    "part_mask",
]

JET_FEATURE_NAMES = [
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "jet_energy",
    "jet_mass",
    "jet_nparticles",
]


@dataclass(frozen=True)
class ParticleTransformerInputs:
    """Canonical Particle Transformer arrays for one constituent view."""

    pf_points: np.ndarray
    pf_features: np.ndarray
    pf_vectors: np.ndarray
    pf_mask: np.ndarray
    labels: np.ndarray | None = None
    jet_features: np.ndarray | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_jets = self.pf_features.shape[0]
        n_constits = self.pf_features.shape[2]
        expected = {
            "pf_points": (n_jets, len(PF_POINT_NAMES), n_constits),
            "pf_features": (n_jets, len(PF_FEATURE_NAMES), n_constits),
            "pf_vectors": (n_jets, len(PF_VECTOR_NAMES), n_constits),
            "pf_mask": (n_jets, len(PF_MASK_NAMES), n_constits),
        }
        actual = {
            "pf_points": self.pf_points.shape,
            "pf_features": self.pf_features.shape,
            "pf_vectors": self.pf_vectors.shape,
            "pf_mask": self.pf_mask.shape,
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(f"{name} has shape {actual[name]}, expected {shape}")
        if self.labels is not None and len(self.labels) != n_jets:
            raise ValueError("labels length does not match number of jets")
        if self.jet_features is not None and self.jet_features.shape != (n_jets, len(JET_FEATURE_NAMES)):
            raise ValueError(
                f"jet_features has shape {self.jet_features.shape}, "
                f"expected {(n_jets, len(JET_FEATURE_NAMES))}"
            )

    def as_dict(self) -> Dict[str, np.ndarray]:
        """Return the four input groups using the Particle Transformer names."""

        return {
            "pf_points": self.pf_points,
            "pf_features": self.pf_features,
            "pf_vectors": self.pf_vectors,
            "pf_mask": self.pf_mask,
        }


def wrap_phi(phi: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""

    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def _apply_manual_transform(
    value: np.ndarray,
    *,
    subtract: float,
    multiply: float,
    clip_min: float = -5.0,
    clip_max: float = 5.0,
) -> np.ndarray:
    return np.clip((value - float(subtract)) * float(multiply), float(clip_min), float(clip_max))


def _safe_log(value: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(value, EPS))


def _prepare_tokens_and_mask(
    raw_tokens: np.ndarray,
    raw_mask: np.ndarray,
    *,
    candidate_weights: np.ndarray | None,
    fold_weights: bool,
    weight_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.asarray(raw_tokens, dtype=np.float32)
    mask = np.asarray(raw_mask, dtype=bool).copy()
    if tokens.ndim != 3 or tokens.shape[-1] != RAW_TOKEN_DIM:
        raise ValueError(f"view tokens must have last dimension {RAW_TOKEN_DIM}, got {tokens.shape}")

    if candidate_weights is None:
        prepared = tokens.copy()
    else:
        weights = np.asarray(candidate_weights, dtype=np.float32)
        if weights.shape != tokens.shape[:2]:
            raise ValueError(f"candidate_weights must have shape {tokens.shape[:2]}, got {weights.shape}")
        if not fold_weights:
            raise NotImplementedError("Canonical Particle Transformer inputs require folded candidate weights")
        weights = np.clip(weights, 0.0, None)
        mask &= weights > float(weight_threshold)
        prepared = tokens.copy()
        prepared[:, :, 0] *= weights
        prepared[:, :, 3] *= weights

    prepared *= mask[:, :, None]
    return prepared, mask


def compute_view_jet_features(
    tokens: np.ndarray,
    mask: np.ndarray,
) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    """Compute particle and jet kinematics using only the supplied tokens."""

    tokens = np.asarray(tokens, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)

    pt = np.where(mask, tokens[:, :, 0], 0.0).astype(np.float32)
    eta = np.where(mask, tokens[:, :, 1], 0.0).astype(np.float32)
    phi = np.where(mask, tokens[:, :, 2], 0.0).astype(np.float32)
    energy = np.where(mask, tokens[:, :, 3], 0.0).astype(np.float32)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    jet_px = np.sum(px, axis=1)
    jet_py = np.sum(py, axis=1)
    jet_pz = np.sum(pz, axis=1)
    jet_energy = np.sum(energy, axis=1)
    jet_pt = np.hypot(jet_px, jet_py)
    jet_phi = np.arctan2(jet_py, jet_px)
    jet_eta = np.arcsinh(jet_pz / np.maximum(jet_pt, EPS))
    jet_eta = np.where(jet_pt > EPS, jet_eta, 0.0)
    jet_phi = np.where(jet_pt > EPS, jet_phi, 0.0)
    mass2 = jet_energy * jet_energy - jet_px * jet_px - jet_py * jet_py - jet_pz * jet_pz
    jet_mass = np.sqrt(np.maximum(mass2, 0.0))
    jet_nparticles = np.sum(mask, axis=1).astype(np.float32)

    jet_features = np.stack(
        [
            jet_pt,
            jet_eta,
            jet_phi,
            jet_energy,
            jet_mass,
            jet_nparticles,
        ],
        axis=1,
    ).astype(np.float32)

    particle = {
        "pt": pt.astype(np.float32),
        "eta": eta.astype(np.float32),
        "phi": phi.astype(np.float32),
        "energy": energy.astype(np.float32),
        "px": px.astype(np.float32),
        "py": py.astype(np.float32),
        "pz": pz.astype(np.float32),
        "jet_pt": jet_pt.astype(np.float32),
        "jet_eta": jet_eta.astype(np.float32),
        "jet_phi": jet_phi.astype(np.float32),
        "jet_energy": jet_energy.astype(np.float32),
    }
    return particle, jet_features


def build_particle_transformer_inputs(
    view: JetView,
    *,
    candidate_weights: np.ndarray | None = None,
    fold_weights: bool = True,
    weight_threshold: float = 0.0,
) -> ParticleTransformerInputs:
    """Convert an offline, HLT, or reconstructed JetView to ParT input groups."""

    return build_particle_transformer_inputs_from_tokens(
        np.asarray(view.tokens, dtype=np.float32),
        np.asarray(view.mask, dtype=bool),
        labels=view.labels,
        split=view.split,
        source_view=view.metadata.get("view"),
        candidate_weights=candidate_weights,
        fold_weights=fold_weights,
        weight_threshold=weight_threshold,
    )


def build_particle_transformer_inputs_from_tokens(
    tokens: np.ndarray,
    mask: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    split: str | None = None,
    source_view: str | None = None,
    candidate_weights: np.ndarray | None = None,
    fold_weights: bool = True,
    weight_threshold: float = 0.0,
) -> ParticleTransformerInputs:
    """Convert raw constituent tokens to ParT input groups.

    All relative features are derived from `view.tokens` and `view.mask` after
    optional candidate-weight folding. Metadata on the view is never used for
    kinematic calculations.
    """

    labels_array = None if labels is None else np.asarray(labels, dtype=np.int64)

    tokens, mask = _prepare_tokens_and_mask(
        tokens,
        mask,
        candidate_weights=candidate_weights,
        fold_weights=fold_weights,
        weight_threshold=weight_threshold,
    )
    particle, jet_features = compute_view_jet_features(tokens, mask)

    pt = particle["pt"]
    eta = particle["eta"]
    phi = particle["phi"]
    energy = particle["energy"]
    jet_pt = particle["jet_pt"][:, None]
    jet_eta = particle["jet_eta"][:, None]
    jet_phi = particle["jet_phi"][:, None]
    jet_energy = particle["jet_energy"][:, None]

    eta_sign = np.sign(jet_eta)
    eta_sign = np.where(eta_sign == 0.0, 1.0, eta_sign)
    part_deta = (eta - jet_eta) * eta_sign
    part_dphi = wrap_phi(phi - jet_phi)
    part_delta_r = np.hypot(part_deta, part_dphi)

    feature_map = {
        "part_pt_log": _apply_manual_transform(_safe_log(pt), subtract=1.7, multiply=0.7),
        "part_e_log": _apply_manual_transform(_safe_log(energy), subtract=2.0, multiply=0.7),
        "part_logptrel": _apply_manual_transform(_safe_log(pt / np.maximum(jet_pt, EPS)), subtract=-4.7, multiply=0.7),
        "part_logerel": _apply_manual_transform(
            _safe_log(energy / np.maximum(jet_energy, EPS)),
            subtract=-4.7,
            multiply=0.7,
        ),
        "part_deltaR": _apply_manual_transform(part_delta_r, subtract=0.2, multiply=4.0),
        "part_charge": tokens[:, :, 4],
        "part_isChargedHadron": tokens[:, :, 5],
        "part_isNeutralHadron": tokens[:, :, 6],
        "part_isPhoton": tokens[:, :, 7],
        "part_isElectron": tokens[:, :, 8],
        "part_isMuon": tokens[:, :, 9],
        "part_d0": np.tanh(tokens[:, :, 10]),
        "part_d0err": _apply_manual_transform(tokens[:, :, 11], subtract=0.0, multiply=1.0, clip_min=0.0, clip_max=1.0),
        "part_dz": np.tanh(tokens[:, :, 12]),
        "part_dzerr": _apply_manual_transform(tokens[:, :, 13], subtract=0.0, multiply=1.0, clip_min=0.0, clip_max=1.0),
        "part_deta": part_deta,
        "part_dphi": part_dphi,
    }

    for key in feature_map:
        feature_map[key] = np.where(mask, feature_map[key], 0.0).astype(np.float32)

    pf_points = np.stack([feature_map[name] for name in PF_POINT_NAMES], axis=1).astype(np.float32)
    pf_features = np.stack([feature_map[name] for name in PF_FEATURE_NAMES], axis=1).astype(np.float32)
    pf_vectors = np.stack(
        [
            particle["px"],
            particle["py"],
            particle["pz"],
            particle["energy"],
        ],
        axis=1,
    ).astype(np.float32)
    pf_vectors *= mask[:, None, :].astype(np.float32)
    pf_mask = mask[:, None, :].astype(bool)

    return ParticleTransformerInputs(
        pf_points=pf_points,
        pf_features=pf_features,
        pf_vectors=pf_vectors,
        pf_mask=pf_mask,
        labels=None if labels_array is None else labels_array.astype(np.int64, copy=True),
        jet_features=jet_features,
        metadata={
            "source_split": split,
            "source_view": source_view,
            "feature_convention": "particle_transformer/data/JetClass/JetClass_full.yaml",
            "pf_point_names": list(PF_POINT_NAMES),
            "pf_feature_names": list(PF_FEATURE_NAMES),
            "pf_vector_names": list(PF_VECTOR_NAMES),
            "pf_mask_names": list(PF_MASK_NAMES),
            "jet_feature_names": list(JET_FEATURE_NAMES),
            "candidate_weights_folded": candidate_weights is not None,
            "weight_threshold": float(weight_threshold),
        },
    )


def summarize_particle_transformer_inputs(inputs: ParticleTransformerInputs) -> Dict[str, Any]:
    """Compact summary for audits and smoke tests."""

    counts = np.sum(inputs.pf_mask[:, 0, :], axis=1).astype(np.float32)
    return {
        "n_jets": int(inputs.pf_features.shape[0]),
        "n_constits": int(inputs.pf_features.shape[2]),
        "pf_points_shape": list(inputs.pf_points.shape),
        "pf_features_shape": list(inputs.pf_features.shape),
        "pf_vectors_shape": list(inputs.pf_vectors.shape),
        "pf_mask_shape": list(inputs.pf_mask.shape),
        "mean_valid_constits": float(np.mean(counts)) if counts.size else 0.0,
        "max_abs_feature": float(np.max(np.abs(inputs.pf_features))) if inputs.pf_features.size else 0.0,
        "has_nan": bool(
            np.isnan(inputs.pf_points).any()
            or np.isnan(inputs.pf_features).any()
            or np.isnan(inputs.pf_vectors).any()
        ),
        "source_view": inputs.metadata.get("source_view"),
    }
