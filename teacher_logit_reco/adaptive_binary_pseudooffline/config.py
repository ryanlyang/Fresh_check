"""Locked Step 1 configuration for the adaptive binary hierarchy campaign."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from jetclass_fresh.hlt_cache import (
    HLT_PROFILE_V2_REALISTIC,
    HLT_PROFILE_V2_REALISTIC_VERSION,
    fixed_hlt_params_dict,
    fixed_hlt_params_from_profile,
    normalize_hlt_profile,
)
from jetclass_fresh.jetclass_data import LABEL_NAMES, RAW_TOKEN_DIM, SPLIT_ORDER

from .schemas import ABPH_MAX_PARTICLES, schema_manifest


ABPH_EXPERIMENT_NAME = "adaptive_binary_pseudooffline_hierarchy_hltv2_s2p5_10class"
ABPH_EXPERIMENT_STEP = "adaptive_binary_pseudooffline_step1_inputs"
ABPH_INPUT_CONTRACT = "adaptive_binary_pseudooffline_input_contract_v1"
ABPH_RESOLVED_CONFIG_CONTRACT = "adaptive_binary_pseudooffline_resolved_config_v1"

ABPH_DATA_DIR = "/home/ryreu/atlas/PracticeTagging/data/jetclass_part1"
ABPH_HLT_PROFILE = HLT_PROFILE_V2_REALISTIC
ABPH_HLT_PROFILE_VERSION = HLT_PROFILE_V2_REALISTIC_VERSION
ABPH_HLT_DEGRADATION_STRENGTH = 2.5
ABPH_SPLIT_ORDER: tuple[str, ...] = tuple(str(row) for row in SPLIT_ORDER)
ABPH_LABEL_NAMES: tuple[str, ...] = tuple(str(row) for row in LABEL_NAMES)
ABPH_LABEL_FILTER: tuple[int, ...] = tuple(range(len(ABPH_LABEL_NAMES)))

ABPH_HIGHDATA_SPLIT_SIZES: Mapping[str, int] = {
    "model_train": 5_000_000,
    "model_val": 1_000_000,
    "stack_train": 2_000_000,
    "stack_val": 1_000_000,
    "final_test": 1_000_000,
}
ABPH_PILOT_SPLIT_SIZES: Mapping[str, int] = {
    "model_train": 500_000,
    "model_val": 150_000,
    "stack_train": 300_000,
    "stack_val": 150_000,
    "final_test": 150_000,
}
ABPH_CAMPAIGN_MODES: tuple[str, ...] = ("pilot", "highdata")
ABPH_OFFLINE_SUPERVISION_SPLITS: tuple[str, ...] = ("model_train", "model_val")
ABPH_DEPLOYABLE_EVAL_SPLITS: tuple[str, ...] = ("model_val", "stack_train", "stack_val", "final_test")


def canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def abph_hlt_params_dict() -> dict[str, Any]:
    return fixed_hlt_params_dict(
        fixed_hlt_params_from_profile(
            ABPH_HLT_PROFILE,
            ABPH_HLT_DEGRADATION_STRENGTH,
        )
    )


def abph_split_sizes(campaign_mode: str) -> dict[str, int]:
    mode = str(campaign_mode).strip().lower()
    if mode == "pilot":
        source = ABPH_PILOT_SPLIT_SIZES
    elif mode == "highdata":
        source = ABPH_HIGHDATA_SPLIT_SIZES
    else:
        raise ValueError(f"unknown ABPH campaign mode {campaign_mode!r}; expected {ABPH_CAMPAIGN_MODES}")
    return {split: int(source[split]) for split in ABPH_SPLIT_ORDER}


@dataclass(frozen=True)
class AdaptiveBinaryInputContractConfig:
    """Strict data contract shared by all ABPH variants."""

    campaign_mode: str = "highdata"
    data_dir: str = ABPH_DATA_DIR
    split_sizes: Mapping[str, int] | None = None
    label_names: tuple[str, ...] = ABPH_LABEL_NAMES
    label_filter: tuple[int, ...] = ABPH_LABEL_FILTER
    hlt_profile: str = ABPH_HLT_PROFILE
    hlt_profile_version: str = ABPH_HLT_PROFILE_VERSION
    hlt_degradation_strength: float = ABPH_HLT_DEGRADATION_STRENGTH
    raw_token_dim: int = RAW_TOKEN_DIM
    max_particles: int = ABPH_MAX_PARTICLES
    final_test_locked: bool = True

    def __post_init__(self) -> None:
        mode = str(self.campaign_mode).strip().lower()
        expected_sizes = abph_split_sizes(mode)
        source_sizes = expected_sizes if self.split_sizes is None else self.split_sizes
        if set(source_sizes) != set(ABPH_SPLIT_ORDER):
            raise ValueError(f"split size keys must be exactly {ABPH_SPLIT_ORDER}")
        sizes = {split: int(source_sizes[split]) for split in ABPH_SPLIT_ORDER}
        if sizes != expected_sizes:
            raise ValueError(f"{mode} split sizes must be exactly {expected_sizes}")
        if tuple(str(row) for row in self.label_names) != ABPH_LABEL_NAMES:
            raise ValueError(f"label_names must be the JetClass order {ABPH_LABEL_NAMES}")
        if tuple(int(row) for row in self.label_filter) != ABPH_LABEL_FILTER:
            raise ValueError(f"label_filter must be {ABPH_LABEL_FILTER}")
        profile = normalize_hlt_profile(self.hlt_profile)
        if profile != ABPH_HLT_PROFILE:
            raise ValueError(f"ABPH is locked to HLT profile {ABPH_HLT_PROFILE}")
        if str(self.hlt_profile_version) != ABPH_HLT_PROFILE_VERSION:
            raise ValueError(f"ABPH is locked to HLT profile version {ABPH_HLT_PROFILE_VERSION}")
        strength = float(self.hlt_degradation_strength)
        if abs(strength - ABPH_HLT_DEGRADATION_STRENGTH) > 1.0e-12:
            raise ValueError(f"ABPH is locked to HLT strength {ABPH_HLT_DEGRADATION_STRENGTH:g}")
        if int(self.raw_token_dim) != RAW_TOKEN_DIM:
            raise ValueError(f"raw_token_dim must be {RAW_TOKEN_DIM}")
        if int(self.max_particles) != ABPH_MAX_PARTICLES:
            raise ValueError(f"max_particles must be {ABPH_MAX_PARTICLES}")
        if not bool(self.final_test_locked):
            raise ValueError("Step 1 requires final_test_locked=True")
        object.__setattr__(self, "campaign_mode", mode)
        object.__setattr__(self, "data_dir", str(self.data_dir))
        object.__setattr__(self, "split_sizes", sizes)
        object.__setattr__(self, "label_names", ABPH_LABEL_NAMES)
        object.__setattr__(self, "label_filter", ABPH_LABEL_FILTER)
        object.__setattr__(self, "hlt_profile", profile)
        object.__setattr__(self, "hlt_profile_version", str(self.hlt_profile_version))
        object.__setattr__(self, "hlt_degradation_strength", strength)
        object.__setattr__(self, "raw_token_dim", int(self.raw_token_dim))
        object.__setattr__(self, "max_particles", int(self.max_particles))
        object.__setattr__(self, "final_test_locked", True)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "contract": ABPH_INPUT_CONTRACT,
            "experiment_name": ABPH_EXPERIMENT_NAME,
            "experiment_step": ABPH_EXPERIMENT_STEP,
            "campaign_mode": self.campaign_mode,
            "data_dir": self.data_dir,
            "split_order": list(ABPH_SPLIT_ORDER),
            "split_sizes": dict(self.split_sizes),
            "label_names": list(self.label_names),
            "label_filter": list(self.label_filter),
            "num_classes": len(self.label_names),
            "hlt_profile": self.hlt_profile,
            "hlt_profile_version": self.hlt_profile_version,
            "hlt_degradation_strength": self.hlt_degradation_strength,
            "hlt_params": abph_hlt_params_dict(),
            "raw_token_dim": self.raw_token_dim,
            "max_particles": self.max_particles,
            "final_test_locked": self.final_test_locked,
            "schema_manifest_hash": schema_manifest()["manifest_hash"],
        }
        payload["config_hash"] = canonical_hash(payload)
        return payload


@dataclass(frozen=True)
class AdaptiveBinaryExperimentLayout:
    """Stable path layout for an ABPH campaign root."""

    output_root: str | Path = "checkpoints"
    campaign_mode: str = "highdata"
    experiment_name: str = ABPH_EXPERIMENT_NAME

    def __post_init__(self) -> None:
        mode = str(self.campaign_mode).strip().lower()
        if mode not in ABPH_CAMPAIGN_MODES:
            raise ValueError(f"unknown campaign_mode {self.campaign_mode!r}")
        object.__setattr__(self, "campaign_mode", mode)

    @property
    def root(self) -> Path:
        return Path(self.output_root) / f"{self.experiment_name}_{self.campaign_mode}"

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def manifest_path(self) -> Path:
        return self.inputs_dir / "split_manifest" / "split_manifest.json.gz"

    @property
    def hlt_cache_dir(self) -> Path:
        return self.inputs_dir / "hlt_cache"

    @property
    def offline_cache_dir(self) -> Path:
        return self.inputs_dir / "offline_cache"

    @property
    def hierarchy_target_cache_dir(self) -> Path:
        return self.inputs_dir / "adaptive_binary_target_cache"

    @property
    def audit_dir(self) -> Path:
        return self.inputs_dir / "audits"

    def to_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "inputs_dir": str(self.inputs_dir),
            "manifest_path": str(self.manifest_path),
            "hlt_cache_dir": str(self.hlt_cache_dir),
            "offline_cache_dir": str(self.offline_cache_dir),
            "hierarchy_target_cache_dir": str(self.hierarchy_target_cache_dir),
            "audit_dir": str(self.audit_dir),
        }


def default_input_contract(campaign_mode: str = "highdata") -> AdaptiveBinaryInputContractConfig:
    return AdaptiveBinaryInputContractConfig(campaign_mode=campaign_mode)


__all__ = [
    "ABPH_CAMPAIGN_MODES",
    "ABPH_DATA_DIR",
    "ABPH_DEPLOYABLE_EVAL_SPLITS",
    "ABPH_EXPERIMENT_NAME",
    "ABPH_EXPERIMENT_STEP",
    "ABPH_HIGHDATA_SPLIT_SIZES",
    "ABPH_HLT_DEGRADATION_STRENGTH",
    "ABPH_HLT_PROFILE",
    "ABPH_HLT_PROFILE_VERSION",
    "ABPH_INPUT_CONTRACT",
    "ABPH_LABEL_FILTER",
    "ABPH_LABEL_NAMES",
    "ABPH_OFFLINE_SUPERVISION_SPLITS",
    "ABPH_PILOT_SPLIT_SIZES",
    "ABPH_RESOLVED_CONFIG_CONTRACT",
    "ABPH_SPLIT_ORDER",
    "AdaptiveBinaryExperimentLayout",
    "AdaptiveBinaryInputContractConfig",
    "abph_hlt_params_dict",
    "abph_split_sizes",
    "canonical_hash",
    "default_input_contract",
]
