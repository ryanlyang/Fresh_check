"""Step 1 input contract for Canonical Multi-Scale Jet State experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from jetclass_fresh.hlt_cache import (
    HLT_PROFILE_V2_REALISTIC,
    HLT_PROFILE_V2_REALISTIC_VERSION,
    fixed_hlt_params_dict,
    fixed_hlt_params_from_profile,
    normalize_hlt_profile,
)
from jetclass_fresh.jetclass_data import LABEL_NAMES, RAW_TOKEN_DIM


CANONICAL_STATE_EXPERIMENT_NAME = "canonical_multi_scale_jet_state_hltv2_s2p5_10class"
CANONICAL_STATE_EXPERIMENT_STEP = "canonical_state_step1_hltv2_s2p5_inputs"
CANONICAL_STATE_CONTRACT = "canonical_multi_scale_jet_state_hltv2_s2p5_10class_v1"
CANONICAL_STATE_INPUTS_CONTRACT = "canonical_state_hltv2_s2p5_inputs_v1"

CANONICAL_STATE_LABEL_NAMES: tuple[str, ...] = tuple(str(name) for name in LABEL_NAMES)
CANONICAL_STATE_LABEL_FILTER: tuple[int, ...] = tuple(range(len(CANONICAL_STATE_LABEL_NAMES)))
CANONICAL_STATE_NUM_CLASSES = len(CANONICAL_STATE_LABEL_NAMES)

CANONICAL_STATE_SPLIT_ORDER: tuple[str, ...] = (
    "model_train",
    "model_val",
    "stack_train",
    "stack_val",
    "final_test",
)
CANONICAL_STATE_HIGH_DATA_SPLIT_SIZES: dict[str, int] = {
    "model_train": 5_000_000,
    "model_val": 1_000_000,
    "stack_train": 3_000_000,
    "stack_val": 1_000_000,
    "final_test": 1_000_000,
}

CANONICAL_STATE_HLT_PROFILE = HLT_PROFILE_V2_REALISTIC
CANONICAL_STATE_HLT_PROFILE_VERSION = HLT_PROFILE_V2_REALISTIC_VERSION
CANONICAL_STATE_HLT_DEGRADATION_STRENGTH = 2.5

CANONICAL_STATE_STEP1_AUDIT_REPORT = "canonical_state_step1_input_audit_report.json"
CANONICAL_STATE_STEP1_AUDIT_SUMMARY = "canonical_state_step1_input_audit_summary.md"
CANONICAL_STATE_SPLIT_AUDIT_REPORT = "canonical_state_split_audit_report.json"
CANONICAL_STATE_HLT_AUDIT_REPORT = "canonical_state_hlt_cache_audit_report.json"


def canonical_state_split_sizes(expected_counts: Mapping[str, int] | None = None) -> dict[str, int]:
    source = CANONICAL_STATE_HIGH_DATA_SPLIT_SIZES if expected_counts is None else expected_counts
    return {split: int(source[split]) for split in CANONICAL_STATE_SPLIT_ORDER}


def canonical_state_hlt_params_dict() -> dict[str, Any]:
    return fixed_hlt_params_dict(
        fixed_hlt_params_from_profile(
            CANONICAL_STATE_HLT_PROFILE,
            CANONICAL_STATE_HLT_DEGRADATION_STRENGTH,
        )
    )


@dataclass(frozen=True)
class CanonicalStateInputContractConfig:
    """Strict Step 1 data contract for the CMS-JS campaign."""

    label_names: tuple[str, ...] = CANONICAL_STATE_LABEL_NAMES
    label_filter: tuple[int, ...] = CANONICAL_STATE_LABEL_FILTER
    num_classes: int = CANONICAL_STATE_NUM_CLASSES
    split_sizes: Mapping[str, int] = field(default_factory=lambda: dict(CANONICAL_STATE_HIGH_DATA_SPLIT_SIZES))
    hlt_profile: str = CANONICAL_STATE_HLT_PROFILE
    hlt_profile_version: str = CANONICAL_STATE_HLT_PROFILE_VERSION
    hlt_degradation_strength: float = CANONICAL_STATE_HLT_DEGRADATION_STRENGTH
    raw_token_dim: int = RAW_TOKEN_DIM
    confirm_final_test: bool = True

    def __post_init__(self) -> None:
        label_names = tuple(str(name) for name in self.label_names)
        if label_names != CANONICAL_STATE_LABEL_NAMES:
            raise ValueError(f"label_names must be the JetClass order {CANONICAL_STATE_LABEL_NAMES}")
        label_filter = tuple(int(value) for value in self.label_filter)
        if label_filter != CANONICAL_STATE_LABEL_FILTER:
            raise ValueError(f"label_filter must be {CANONICAL_STATE_LABEL_FILTER}")
        if int(self.num_classes) != CANONICAL_STATE_NUM_CLASSES:
            raise ValueError(f"num_classes must be {CANONICAL_STATE_NUM_CLASSES}")
        split_sizes = canonical_state_split_sizes(self.split_sizes)
        if tuple(split_sizes.keys()) != CANONICAL_STATE_SPLIT_ORDER:
            raise ValueError(f"split_sizes keys must be exactly {CANONICAL_STATE_SPLIT_ORDER}")
        if split_sizes != CANONICAL_STATE_HIGH_DATA_SPLIT_SIZES:
            raise ValueError(f"CMS-JS Step 1 is locked to {CANONICAL_STATE_HIGH_DATA_SPLIT_SIZES}")
        hlt_profile = normalize_hlt_profile(self.hlt_profile)
        if hlt_profile != CANONICAL_STATE_HLT_PROFILE:
            raise ValueError(f"CMS-JS Step 1 is locked to HLT profile {CANONICAL_STATE_HLT_PROFILE}")
        if str(self.hlt_profile_version) != CANONICAL_STATE_HLT_PROFILE_VERSION:
            raise ValueError(
                "CMS-JS Step 1 is locked to HLT profile version "
                f"{CANONICAL_STATE_HLT_PROFILE_VERSION}"
            )
        strength = float(self.hlt_degradation_strength)
        if abs(strength - CANONICAL_STATE_HLT_DEGRADATION_STRENGTH) > 1.0e-12:
            raise ValueError(
                "CMS-JS Step 1 is locked to HLT degradation strength "
                f"{CANONICAL_STATE_HLT_DEGRADATION_STRENGTH:g}"
            )
        if int(self.raw_token_dim) != RAW_TOKEN_DIM:
            raise ValueError(f"raw_token_dim must match repo RAW_TOKEN_DIM={RAW_TOKEN_DIM}")
        if not bool(self.confirm_final_test):
            raise ValueError("CMS-JS Step 1 requires explicit final-test confirmation")
        object.__setattr__(self, "label_names", label_names)
        object.__setattr__(self, "label_filter", label_filter)
        object.__setattr__(self, "split_sizes", split_sizes)
        object.__setattr__(self, "hlt_profile", hlt_profile)
        object.__setattr__(self, "hlt_profile_version", str(self.hlt_profile_version))
        object.__setattr__(self, "hlt_degradation_strength", strength)
        object.__setattr__(self, "raw_token_dim", int(self.raw_token_dim))
        object.__setattr__(self, "confirm_final_test", bool(self.confirm_final_test))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": CANONICAL_STATE_CONTRACT,
            "experiment_name": CANONICAL_STATE_EXPERIMENT_NAME,
            "experiment_step": CANONICAL_STATE_EXPERIMENT_STEP,
            "label_names": list(self.label_names),
            "label_filter": list(self.label_filter),
            "num_classes": int(self.num_classes),
            "split_order": list(CANONICAL_STATE_SPLIT_ORDER),
            "split_sizes": {split: int(self.split_sizes[split]) for split in CANONICAL_STATE_SPLIT_ORDER},
            "hlt_profile": self.hlt_profile,
            "hlt_profile_version": self.hlt_profile_version,
            "hlt_degradation_strength": float(self.hlt_degradation_strength),
            "hlt_params": canonical_state_hlt_params_dict(),
            "raw_token_dim": int(self.raw_token_dim),
            "confirm_final_test": bool(self.confirm_final_test),
        }


@dataclass(frozen=True)
class CanonicalStateExperimentLayout:
    """Path helper for shared CMS-JS Step 1 artifacts."""

    output_root: str | Path = "checkpoints"
    experiment_name: str = CANONICAL_STATE_EXPERIMENT_NAME

    @property
    def root(self) -> Path:
        return Path(self.output_root) / self.experiment_name

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def split_manifest_path(self) -> Path:
        return self.inputs_dir / "split_manifest.json.gz"

    @property
    def hlt_cache_dir(self) -> Path:
        return self.inputs_dir / "hlt_cache"

    @property
    def step1_audit_dir(self) -> Path:
        return self.inputs_dir / "audits"

    def to_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "inputs_dir": str(self.inputs_dir),
            "split_manifest_path": str(self.split_manifest_path),
            "hlt_cache_dir": str(self.hlt_cache_dir),
            "step1_audit_dir": str(self.step1_audit_dir),
        }


def default_canonical_state_input_contract_config() -> CanonicalStateInputContractConfig:
    return CanonicalStateInputContractConfig()


def default_canonical_state_experiment_layout(
    output_root: str | Path = "checkpoints",
) -> CanonicalStateExperimentLayout:
    return CanonicalStateExperimentLayout(output_root=output_root)


def canonical_state_config_manifest() -> dict[str, Any]:
    cfg = default_canonical_state_input_contract_config()
    layout = default_canonical_state_experiment_layout()
    return {"config": cfg.to_dict(), "layout": layout.to_dict(), "dataclass": asdict(cfg)}
