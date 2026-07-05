"""Configuration contract for PDV3 AV10-adapter privileged distillation.

Step 1 is deliberately data-contract only. It locks the new experiment family
to full 10-class JetClass and the 5M/1M/1M split sizes, then names the shared
split/cache artifacts later teacher and student stages must reuse.  The HLT
profile/strength are environment-configurable so the same PDV3 machinery can
run the original fixed-HLT v1 HLT0.2 setup or the newer realistic HLT v2 setup.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from jetclass_fresh.hlt_cache import HLT_PROFILE_V1, normalize_hlt_profile
from jetclass_fresh.jetclass_data import LABEL_NAMES, RAW_TOKEN_DIM


PDV3_EXPERIMENT_NAME = "privileged_distill_v3_av10_adapter_hlt0p2_5m"
PDV3_EXPERIMENT_STEP = "pdv3_step1_hlt0p2_paired_inputs"
PDV3_CONTRACT = "pdv3_av10_adapter_privileged_distill_hlt0p2_10class_5m_v1"

PDV3_LABEL_NAMES: tuple[str, ...] = tuple(str(name) for name in LABEL_NAMES)
PDV3_LABEL_FILTER: tuple[int, ...] = tuple(range(len(PDV3_LABEL_NAMES)))
PDV3_NUM_CLASSES = len(PDV3_LABEL_NAMES)

def _float_from_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _hlt_profile_from_env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return normalize_hlt_profile(default)
    return normalize_hlt_profile(raw)


PDV3_HLT_PROFILE = _hlt_profile_from_env("PDV3_HLT_PROFILE", HLT_PROFILE_V1)
PDV3_HLT_DEGRADATION_STRENGTH = _float_from_env("PDV3_HLT_DEGRADATION_STRENGTH", 0.2)

PDV3_MODEL_SPLIT_ORDER: tuple[str, ...] = ("model_train", "model_val", "final_test")
PDV3_MODEL_SPLIT_SIZES: dict[str, int] = {
    "model_train": 5_000_000,
    "model_val": 1_000_000,
    "final_test": 1_000_000,
}
PDV3_MANIFEST_SPLIT_ORDER: tuple[str, ...] = (
    "model_train",
    "model_val",
    "stack_train",
    "stack_val",
    "final_test",
)
PDV3_STACK_PLACEHOLDER_SPLIT_SIZES: dict[str, int] = {
    "stack_train": 10,
    "stack_val": 10,
}
PDV3_MANIFEST_SPLIT_SIZES: dict[str, int] = {
    "model_train": PDV3_MODEL_SPLIT_SIZES["model_train"],
    "model_val": PDV3_MODEL_SPLIT_SIZES["model_val"],
    "stack_train": PDV3_STACK_PLACEHOLDER_SPLIT_SIZES["stack_train"],
    "stack_val": PDV3_STACK_PLACEHOLDER_SPLIT_SIZES["stack_val"],
    "final_test": PDV3_MODEL_SPLIT_SIZES["final_test"],
}

PDV3_INPUTS_CONTRACT = "pdv3_hlt0p2_paired_hlt_offline_inputs_v1"
PDV3_STEP1_AUDIT_REPORT = "pdv3_step1_input_audit_report.json"
PDV3_STEP1_AUDIT_SUMMARY = "pdv3_step1_input_audit_summary.md"
PDV3_SPLIT_AUDIT_REPORT = "split_audit_report.json"
PDV3_HLT_AUDIT_REPORT = "hlt_cache_audit_report.json"
PDV3_OFFLINE_AUDIT_REPORT = "offline_cache_audit_report.json"


def normalize_pdv3_split_name(value: str) -> str:
    split = str(value).strip()
    if split not in PDV3_MODEL_SPLIT_ORDER:
        raise ValueError(f"Unknown PDV3 model split {value!r}; expected one of {PDV3_MODEL_SPLIT_ORDER}")
    return split


def pdv3_model_split_sizes(expected_counts: Mapping[str, int] | None = None) -> dict[str, int]:
    source = PDV3_MODEL_SPLIT_SIZES if expected_counts is None else expected_counts
    return {split: int(source[split]) for split in PDV3_MODEL_SPLIT_ORDER}


def pdv3_stack_placeholder_split_sizes(expected_counts: Mapping[str, int] | None = None) -> dict[str, int]:
    source = PDV3_STACK_PLACEHOLDER_SPLIT_SIZES if expected_counts is None else expected_counts
    return {split: int(source[split]) for split in PDV3_STACK_PLACEHOLDER_SPLIT_SIZES}


def pdv3_manifest_split_sizes(
    expected_counts: Mapping[str, int] | None = None,
    placeholder_counts: Mapping[str, int] | None = None,
) -> dict[str, int]:
    model_counts = pdv3_model_split_sizes(expected_counts)
    placeholder = pdv3_stack_placeholder_split_sizes(placeholder_counts)
    source = {**model_counts, **placeholder}
    return {split: int(source[split]) for split in PDV3_MANIFEST_SPLIT_ORDER}


@dataclass(frozen=True)
class PDV3InputContractConfig:
    """Strict data contract for one PDV3 HLT profile run."""

    label_names: tuple[str, ...] = PDV3_LABEL_NAMES
    label_filter: tuple[int, ...] = PDV3_LABEL_FILTER
    num_classes: int = PDV3_NUM_CLASSES
    model_split_sizes: Mapping[str, int] = field(default_factory=lambda: dict(PDV3_MODEL_SPLIT_SIZES))
    stack_placeholder_split_sizes: Mapping[str, int] = field(
        default_factory=lambda: dict(PDV3_STACK_PLACEHOLDER_SPLIT_SIZES)
    )
    hlt_profile: str = PDV3_HLT_PROFILE
    hlt_degradation_strength: float = PDV3_HLT_DEGRADATION_STRENGTH
    raw_token_dim: int = RAW_TOKEN_DIM
    require_offline_cache: bool = True
    confirm_final_test: bool = True

    def __post_init__(self) -> None:
        label_names = tuple(str(name) for name in self.label_names)
        if label_names != PDV3_LABEL_NAMES:
            raise ValueError(f"PDV3 label_names must be the JetClass order {PDV3_LABEL_NAMES}")
        label_filter = tuple(int(value) for value in self.label_filter)
        if label_filter != PDV3_LABEL_FILTER:
            raise ValueError(f"PDV3 label_filter must be {PDV3_LABEL_FILTER}")
        if int(self.num_classes) != PDV3_NUM_CLASSES:
            raise ValueError(f"PDV3 num_classes must be {PDV3_NUM_CLASSES}")
        model_split_sizes = pdv3_model_split_sizes(self.model_split_sizes)
        if tuple(model_split_sizes.keys()) != PDV3_MODEL_SPLIT_ORDER:
            raise ValueError(f"model_split_sizes keys must be exactly {PDV3_MODEL_SPLIT_ORDER}")
        if model_split_sizes != PDV3_MODEL_SPLIT_SIZES:
            raise ValueError(f"PDV3 first serious run is locked to {PDV3_MODEL_SPLIT_SIZES}")
        placeholders = pdv3_stack_placeholder_split_sizes(self.stack_placeholder_split_sizes)
        if placeholders != PDV3_STACK_PLACEHOLDER_SPLIT_SIZES:
            raise ValueError(f"PDV3 stack placeholders must be {PDV3_STACK_PLACEHOLDER_SPLIT_SIZES}")
        hlt_profile = normalize_hlt_profile(self.hlt_profile)
        if hlt_profile != PDV3_HLT_PROFILE:
            raise ValueError(f"PDV3 is locked to HLT profile {PDV3_HLT_PROFILE}")
        strength = float(self.hlt_degradation_strength)
        if abs(strength - PDV3_HLT_DEGRADATION_STRENGTH) > 1.0e-12:
            raise ValueError(f"PDV3 is locked to HLT degradation strength {PDV3_HLT_DEGRADATION_STRENGTH:g}")
        if int(self.raw_token_dim) != RAW_TOKEN_DIM:
            raise ValueError(f"raw_token_dim must match repo RAW_TOKEN_DIM={RAW_TOKEN_DIM}")
        if not bool(self.require_offline_cache):
            raise ValueError("PDV3 Step 1 requires a paired offline cache")
        if not bool(self.confirm_final_test):
            raise ValueError("PDV3 Step 1 requires explicit final-test confirmation")
        object.__setattr__(self, "label_names", label_names)
        object.__setattr__(self, "label_filter", label_filter)
        object.__setattr__(self, "model_split_sizes", model_split_sizes)
        object.__setattr__(self, "stack_placeholder_split_sizes", placeholders)
        object.__setattr__(self, "hlt_profile", hlt_profile)
        object.__setattr__(self, "hlt_degradation_strength", strength)
        object.__setattr__(self, "raw_token_dim", int(self.raw_token_dim))
        object.__setattr__(self, "require_offline_cache", bool(self.require_offline_cache))
        object.__setattr__(self, "confirm_final_test", bool(self.confirm_final_test))

    @property
    def manifest_split_sizes(self) -> dict[str, int]:
        return pdv3_manifest_split_sizes(self.model_split_sizes, self.stack_placeholder_split_sizes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": PDV3_CONTRACT,
            "experiment_name": PDV3_EXPERIMENT_NAME,
            "experiment_step": PDV3_EXPERIMENT_STEP,
            "label_names": list(self.label_names),
            "label_filter": list(self.label_filter),
            "num_classes": int(self.num_classes),
            "model_split_order": list(PDV3_MODEL_SPLIT_ORDER),
            "model_split_sizes": {split: int(self.model_split_sizes[split]) for split in PDV3_MODEL_SPLIT_ORDER},
            "manifest_split_order": list(PDV3_MANIFEST_SPLIT_ORDER),
            "manifest_split_sizes": {split: int(self.manifest_split_sizes[split]) for split in PDV3_MANIFEST_SPLIT_ORDER},
            "stack_placeholder_split_sizes": dict(self.stack_placeholder_split_sizes),
            "hlt_profile": self.hlt_profile,
            "hlt_degradation_strength": float(self.hlt_degradation_strength),
            "raw_token_dim": int(self.raw_token_dim),
            "require_offline_cache": bool(self.require_offline_cache),
            "confirm_final_test": bool(self.confirm_final_test),
        }


@dataclass(frozen=True)
class PDV3ExperimentLayout:
    """Path helper for the shared PDV3 Step 1 artifacts."""

    output_root: str | Path = "checkpoints"
    experiment_name: str = PDV3_EXPERIMENT_NAME

    @property
    def root(self) -> Path:
        return Path(self.output_root) / self.experiment_name

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def split_manifest_dir(self) -> Path:
        return self.inputs_dir / "split_manifest"

    @property
    def split_manifest_path(self) -> Path:
        return self.split_manifest_dir / "split_manifest.json.gz"

    @property
    def hlt_cache_dir(self) -> Path:
        return self.inputs_dir / "hlt_cache"

    @property
    def offline_cache_dir(self) -> Path:
        return self.inputs_dir / "offline_cache"

    @property
    def audits_dir(self) -> Path:
        return self.root / "audits"

    @property
    def step1_audit_dir(self) -> Path:
        return self.audits_dir / "step1_inputs"

    @property
    def teachers_dir(self) -> Path:
        return self.root / "teachers"

    @property
    def students_dir(self) -> Path:
        return self.root / "students"

    @property
    def final_report_dir(self) -> Path:
        return self.root / "final_report"

    def to_dict(self) -> dict[str, str]:
        return {
            "root": self.root.as_posix(),
            "inputs_dir": self.inputs_dir.as_posix(),
            "split_manifest_dir": self.split_manifest_dir.as_posix(),
            "split_manifest_path": self.split_manifest_path.as_posix(),
            "hlt_cache_dir": self.hlt_cache_dir.as_posix(),
            "offline_cache_dir": self.offline_cache_dir.as_posix(),
            "audits_dir": self.audits_dir.as_posix(),
            "step1_audit_dir": self.step1_audit_dir.as_posix(),
            "teachers_dir": self.teachers_dir.as_posix(),
            "students_dir": self.students_dir.as_posix(),
            "final_report_dir": self.final_report_dir.as_posix(),
        }


def default_pdv3_input_contract_config() -> PDV3InputContractConfig:
    return PDV3InputContractConfig()


def default_pdv3_experiment_layout(
    *,
    output_root: str | Path = "checkpoints",
    experiment_name: str = PDV3_EXPERIMENT_NAME,
) -> PDV3ExperimentLayout:
    return PDV3ExperimentLayout(output_root=output_root, experiment_name=experiment_name)


def pdv3_config_manifest(
    *,
    config: PDV3InputContractConfig | None = None,
    layout: PDV3ExperimentLayout | None = None,
) -> dict[str, Any]:
    cfg = config or default_pdv3_input_contract_config()
    lay = layout or default_pdv3_experiment_layout()
    return {
        "config": cfg.to_dict(),
        "layout": lay.to_dict(),
    }


__all__ = [
    "PDV3_CONTRACT",
    "PDV3_EXPERIMENT_NAME",
    "PDV3_EXPERIMENT_STEP",
    "PDV3_HLT_DEGRADATION_STRENGTH",
    "PDV3_HLT_PROFILE",
    "PDV3_HLT_AUDIT_REPORT",
    "PDV3_INPUTS_CONTRACT",
    "PDV3_LABEL_FILTER",
    "PDV3_LABEL_NAMES",
    "PDV3_MANIFEST_SPLIT_ORDER",
    "PDV3_MANIFEST_SPLIT_SIZES",
    "PDV3_MODEL_SPLIT_ORDER",
    "PDV3_MODEL_SPLIT_SIZES",
    "PDV3_NUM_CLASSES",
    "PDV3_OFFLINE_AUDIT_REPORT",
    "PDV3_SPLIT_AUDIT_REPORT",
    "PDV3_STACK_PLACEHOLDER_SPLIT_SIZES",
    "PDV3_STEP1_AUDIT_REPORT",
    "PDV3_STEP1_AUDIT_SUMMARY",
    "PDV3ExperimentLayout",
    "PDV3InputContractConfig",
    "default_pdv3_experiment_layout",
    "default_pdv3_input_contract_config",
    "normalize_pdv3_split_name",
    "pdv3_config_manifest",
    "pdv3_manifest_split_sizes",
    "pdv3_model_split_sizes",
    "pdv3_stack_placeholder_split_sizes",
]
