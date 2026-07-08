"""Configuration and layout helpers for deployable HLT multiview fusion.

Step 1 is deliberately config-only. It freezes the PDV3-rooted run grid and
path layout so the later training, logit-fusion, Slurm, and reporting steps do
not disagree about model names or output locations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from jetclass_fresh.hlt_cache import HLT_PROFILE_V2_REALISTIC

from teacher_logit_reco.hlt_self_dualview.config import (
    HLT_SDV_HLT2_PROFILE_NAME,
    HLT_SDV_HLT2_PROFILE_VERSION,
    hlt_sdv_dual_hlt2_variant_name,
    hlt_sdv_hlt2_cache_name,
    hlt_sdv_strength_tag,
    normalize_hlt_sdv_strength,
)
from teacher_logit_reco.privileged_distill_v3.config import (
    PDV3_MODEL_SPLIT_ORDER,
    PDV3_MODEL_SPLIT_SIZES,
    PDV3_NUM_CLASSES,
    PDV3ExperimentLayout,
    default_pdv3_experiment_layout,
)


HLT_MV_EXPERIMENT_NAME = "hlt_multiview_source_fusion_10class"
HLT_MV_EXPERIMENT_STEP = "hlt_mv_step1_config_layout"
HLT_MV_CONTRACT = "deployable_hlt_multiview_source_fusion_layout_v1"
HLT_MV_ROOT_DIRNAME = "hlt_multiview_source_fusion"
HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME = (
    "privileged_distill_v3_av10_adapter_fixed_hlt_v2_realistic_s1p0_highdata_20260705_190747"
)

HLT_MV_ALLOWED_INPUTS = "HLT_only"
HLT_MV_DEPLOYMENT_INPUTS = "HLT_plus_deterministic_HLT2_multiview"
HLT_MV_PDV3_HLT_PROFILE = HLT_PROFILE_V2_REALISTIC
HLT_MV_PDV3_HLT_DEGRADATION_STRENGTH = 1.0
HLT_MV_DEFAULT_STRENGTHS: tuple[float, ...] = (0.10, 0.20, 0.35, 1.00)
HLT_MV_DEFAULT_HLT_SOURCE_SEED = 8801
HLT_MV_DEFAULT_HLT2_SOURCE_SEEDS: dict[float, int] = {
    0.10: 8811,
    0.20: 8821,
    0.35: 8831,
    1.00: 8841,
}
HLT_MV_DEFAULT_HLT_RANDOM_SEEDS: tuple[int, ...] = (9101, 9102, 9103, 9104)

HLT_MV_CANONICAL_HLT_SOURCE = f"hlt_part_seed{HLT_MV_DEFAULT_HLT_SOURCE_SEED}"
HLT_MV_FUSION_SOURCE_5VIEW = "source_5view"
HLT_MV_FUSION_HLT_RANDOM_4SEED = "hlt_random_4seed"
HLT_MV_FUSION_PRETRAINED_DUALVIEW_4MODEL = "pretrained_dualview_4model"
HLT_MV_FUSION_SCRATCH_DUALVIEW_4MODEL = "scratch_dualview_4model"
HLT_MV_TRIVIEW_MODEL_NAME = "tri_hlt_hlt2_s0p35_s1p00"

HLT_MV_SOURCE_MODELS_DIRNAME = "source_models"
HLT_MV_RANDOM_HLT_CONTROLS_DIRNAME = "hlt_random_seed_controls"
HLT_MV_LOGIT_FUSIONS_DIRNAME = "logit_fusions"
HLT_MV_PRETRAINED_DUALVIEW_DIRNAME = "particle_dualview_pretrained"
HLT_MV_SCRATCH_DUALVIEW_DIRNAME = "particle_dualview_scratch"
HLT_MV_CONTROLS_DIRNAME = "controls"
HLT_MV_TRIVIEW_DIRNAME = "triview"
HLT_MV_REPORT_DIRNAME = "final_report"


def normalize_hlt_mv_strengths(values: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    """Normalize the HLT2 strength grid, excluding the identity-cache control."""

    strengths = tuple(normalize_hlt_sdv_strength(value) for value in values)
    if not strengths:
        raise ValueError("HLT-MV strength grid cannot be empty.")
    if len(set(strengths)) != len(strengths):
        raise ValueError(f"HLT-MV strengths must be unique, got {strengths!r}.")
    if any(strength <= 0.0 for strength in strengths):
        raise ValueError("HLT-MV model strengths must exclude the identity s0p00 cache.")
    return strengths


def _seed_for_strength(strength: float | int | str, seeds_by_strength: Mapping[float, int]) -> int:
    normalized = normalize_hlt_sdv_strength(strength)
    try:
        return int(seeds_by_strength[normalized])
    except KeyError as exc:
        raise ValueError(f"No HLT2 source seed configured for strength {normalized:.2f}.") from exc


def normalize_hlt_mv_source_name(name: str) -> str:
    value = str(name).strip()
    if not value:
        raise ValueError("HLT-MV source/model name cannot be empty.")
    if any(ch.isspace() for ch in value):
        raise ValueError(f"HLT-MV source/model name cannot contain whitespace: {name!r}.")
    return value


def hlt_mv_hlt2_source_name(
    strength: float | int | str,
    *,
    seeds_by_strength: Mapping[float, int] | None = None,
) -> str:
    seed_map = HLT_MV_DEFAULT_HLT2_SOURCE_SEEDS if seeds_by_strength is None else seeds_by_strength
    return f"hlt2_part_{hlt_sdv_strength_tag(strength)}_seed{_seed_for_strength(strength, seed_map)}"


def hlt_mv_random_hlt_source_name(seed: int) -> str:
    return f"hlt_part_seed{int(seed)}"


def hlt_mv_pretrained_dualview_name(strength: float | int | str) -> str:
    return hlt_sdv_dual_hlt2_variant_name(strength)


def hlt_mv_scratch_dualview_name(strength: float | int | str) -> str:
    return f"{hlt_sdv_dual_hlt2_variant_name(strength)}_scratch"


def hlt_mv_hlt2_only_name(strength: float | int | str) -> str:
    return f"hlt2_only_part_{hlt_sdv_strength_tag(strength)}"


def hlt_mv_tta_name(strength: float | int | str) -> str:
    return f"tta_hlt_part_hlt_plus_hlt2_{hlt_sdv_strength_tag(strength)}"


def build_hlt_mv_hlt2_source_names(
    strengths: tuple[float, ...] | list[float] = HLT_MV_DEFAULT_STRENGTHS,
    *,
    seeds_by_strength: Mapping[float, int] | None = None,
) -> tuple[str, ...]:
    normalized = normalize_hlt_mv_strengths(strengths)
    return tuple(hlt_mv_hlt2_source_name(strength, seeds_by_strength=seeds_by_strength) for strength in normalized)


def build_hlt_mv_random_hlt_source_names(
    seeds: tuple[int, ...] | list[int] = HLT_MV_DEFAULT_HLT_RANDOM_SEEDS,
) -> tuple[str, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    if len(normalized) != 4:
        raise ValueError("HLT-MV random HLT ensemble must contain exactly four seeds.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"HLT-MV random HLT seeds must be unique, got {normalized!r}.")
    return tuple(hlt_mv_random_hlt_source_name(seed) for seed in normalized)


def build_hlt_mv_pretrained_dualview_names(
    strengths: tuple[float, ...] | list[float] = HLT_MV_DEFAULT_STRENGTHS,
) -> tuple[str, ...]:
    return tuple(hlt_mv_pretrained_dualview_name(strength) for strength in normalize_hlt_mv_strengths(strengths))


def build_hlt_mv_scratch_dualview_names(
    strengths: tuple[float, ...] | list[float] = HLT_MV_DEFAULT_STRENGTHS,
) -> tuple[str, ...]:
    return tuple(hlt_mv_scratch_dualview_name(strength) for strength in normalize_hlt_mv_strengths(strengths))


def build_hlt_mv_control_names(
    strengths: tuple[float, ...] | list[float] = HLT_MV_DEFAULT_STRENGTHS,
) -> tuple[str, ...]:
    normalized = normalize_hlt_mv_strengths(strengths)
    return ("sdv_hlt_hlt_same_view",) + tuple(hlt_mv_tta_name(strength) for strength in normalized)


@dataclass(frozen=True)
class HLTMVExperimentConfig:
    """Immutable run-grid contract for HLT multiview source/fusion."""

    pdv3_experiment_name: str = HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME
    strengths: tuple[float, ...] = HLT_MV_DEFAULT_STRENGTHS
    hlt_source_seed: int = HLT_MV_DEFAULT_HLT_SOURCE_SEED
    hlt2_source_seeds: Mapping[float, int] = field(
        default_factory=lambda: dict(HLT_MV_DEFAULT_HLT2_SOURCE_SEEDS)
    )
    random_hlt_seeds: tuple[int, ...] = HLT_MV_DEFAULT_HLT_RANDOM_SEEDS
    hlt2_profile_name: str = HLT_SDV_HLT2_PROFILE_NAME
    hlt2_profile_version: str = HLT_SDV_HLT2_PROFILE_VERSION
    pdv3_hlt_profile: str = HLT_MV_PDV3_HLT_PROFILE
    pdv3_hlt_degradation_strength: float = HLT_MV_PDV3_HLT_DEGRADATION_STRENGTH
    allowed_inputs: str = HLT_MV_ALLOWED_INPUTS
    deployment_inputs: str = HLT_MV_DEPLOYMENT_INPUTS
    split_order: tuple[str, ...] = PDV3_MODEL_SPLIT_ORDER
    split_sizes: Mapping[str, int] = field(default_factory=lambda: dict(PDV3_MODEL_SPLIT_SIZES))
    num_classes: int = PDV3_NUM_CLASSES

    def __post_init__(self) -> None:
        strengths = normalize_hlt_mv_strengths(self.strengths)
        seed_map = {
            normalize_hlt_sdv_strength(strength): int(seed)
            for strength, seed in dict(self.hlt2_source_seeds).items()
        }
        missing = [strength for strength in strengths if strength not in seed_map]
        if missing:
            formatted = ", ".join(f"{strength:.2f}" for strength in missing)
            raise ValueError(f"Missing HLT2 source seeds for strengths: {formatted}")
        if int(self.hlt_source_seed) <= 0:
            raise ValueError("HLT-MV canonical HLT source seed must be positive.")
        random_seeds = tuple(int(seed) for seed in self.random_hlt_seeds)
        build_hlt_mv_random_hlt_source_names(random_seeds)
        if int(self.hlt_source_seed) in set(random_seeds):
            raise ValueError("Canonical HLT source seed must be distinct from random ensemble seeds.")
        if self.hlt2_profile_name != HLT_SDV_HLT2_PROFILE_NAME:
            raise ValueError(f"HLT-MV requires HLT2 profile {HLT_SDV_HLT2_PROFILE_NAME!r}.")
        if self.hlt2_profile_version != HLT_SDV_HLT2_PROFILE_VERSION:
            raise ValueError(f"HLT-MV requires HLT2 profile version {HLT_SDV_HLT2_PROFILE_VERSION!r}.")
        if self.pdv3_hlt_profile != HLT_MV_PDV3_HLT_PROFILE:
            raise ValueError(f"HLT-MV requires PDV3 HLT profile {HLT_MV_PDV3_HLT_PROFILE!r}.")
        if abs(float(self.pdv3_hlt_degradation_strength) - HLT_MV_PDV3_HLT_DEGRADATION_STRENGTH) > 1.0e-12:
            raise ValueError(
                f"HLT-MV requires PDV3 HLT degradation strength {HLT_MV_PDV3_HLT_DEGRADATION_STRENGTH:g}."
            )
        if self.allowed_inputs != HLT_MV_ALLOWED_INPUTS:
            raise ValueError(f"HLT-MV allowed inputs must be {HLT_MV_ALLOWED_INPUTS!r}.")
        if self.deployment_inputs != HLT_MV_DEPLOYMENT_INPUTS:
            raise ValueError(f"HLT-MV deployment inputs must be {HLT_MV_DEPLOYMENT_INPUTS!r}.")
        if tuple(self.split_order) != PDV3_MODEL_SPLIT_ORDER:
            raise ValueError(f"HLT-MV split order must be {PDV3_MODEL_SPLIT_ORDER}.")
        if dict(self.split_sizes) != dict(PDV3_MODEL_SPLIT_SIZES):
            raise ValueError(f"HLT-MV split sizes must be {PDV3_MODEL_SPLIT_SIZES}.")
        if int(self.num_classes) != int(PDV3_NUM_CLASSES):
            raise ValueError(f"HLT-MV num_classes must be {PDV3_NUM_CLASSES}.")
        object.__setattr__(self, "strengths", strengths)
        object.__setattr__(self, "hlt_source_seed", int(self.hlt_source_seed))
        object.__setattr__(self, "hlt2_source_seeds", seed_map)
        object.__setattr__(self, "random_hlt_seeds", random_seeds)
        object.__setattr__(self, "split_sizes", dict(self.split_sizes))

    @property
    def canonical_hlt_source_name(self) -> str:
        return hlt_mv_random_hlt_source_name(self.hlt_source_seed)

    @property
    def hlt2_source_names(self) -> tuple[str, ...]:
        return build_hlt_mv_hlt2_source_names(self.strengths, seeds_by_strength=self.hlt2_source_seeds)

    @property
    def source_model_names(self) -> tuple[str, ...]:
        return (self.canonical_hlt_source_name,) + self.hlt2_source_names

    @property
    def random_hlt_source_names(self) -> tuple[str, ...]:
        return build_hlt_mv_random_hlt_source_names(self.random_hlt_seeds)

    @property
    def pretrained_dualview_names(self) -> tuple[str, ...]:
        return build_hlt_mv_pretrained_dualview_names(self.strengths)

    @property
    def scratch_dualview_names(self) -> tuple[str, ...]:
        return build_hlt_mv_scratch_dualview_names(self.strengths)

    @property
    def control_names(self) -> tuple[str, ...]:
        return build_hlt_mv_control_names(self.strengths)

    @property
    def logit_fusion_names(self) -> tuple[str, ...]:
        return (
            HLT_MV_FUSION_SOURCE_5VIEW,
            HLT_MV_FUSION_HLT_RANDOM_4SEED,
            HLT_MV_FUSION_PRETRAINED_DUALVIEW_4MODEL,
            HLT_MV_FUSION_SCRATCH_DUALVIEW_4MODEL,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": HLT_MV_CONTRACT,
            "experiment_name": HLT_MV_EXPERIMENT_NAME,
            "experiment_step": HLT_MV_EXPERIMENT_STEP,
            "pdv3_experiment_name": self.pdv3_experiment_name,
            "strengths": list(self.strengths),
            "hlt_source_seed": int(self.hlt_source_seed),
            "hlt2_source_seeds": {hlt_sdv_strength_tag(k): int(v) for k, v in self.hlt2_source_seeds.items()},
            "random_hlt_seeds": list(self.random_hlt_seeds),
            "hlt2_profile_name": self.hlt2_profile_name,
            "hlt2_profile_version": self.hlt2_profile_version,
            "pdv3_hlt_profile": self.pdv3_hlt_profile,
            "pdv3_hlt_degradation_strength": float(self.pdv3_hlt_degradation_strength),
            "allowed_inputs": self.allowed_inputs,
            "deployment_inputs": self.deployment_inputs,
            "split_order": list(self.split_order),
            "split_sizes": dict(self.split_sizes),
            "num_classes": int(self.num_classes),
            "source_model_names": list(self.source_model_names),
            "random_hlt_source_names": list(self.random_hlt_source_names),
            "pretrained_dualview_names": list(self.pretrained_dualview_names),
            "scratch_dualview_names": list(self.scratch_dualview_names),
            "control_names": list(self.control_names),
            "logit_fusion_names": list(self.logit_fusion_names),
            "triview_model_name": HLT_MV_TRIVIEW_MODEL_NAME,
        }


@dataclass(frozen=True)
class HLTMVExperimentLayout:
    """Filesystem layout rooted under an existing PDV3 experiment directory."""

    output_root: str | Path = "checkpoints"
    pdv3_experiment_name: str = HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME
    root_dirname: str = HLT_MV_ROOT_DIRNAME

    @property
    def pdv3_layout(self) -> PDV3ExperimentLayout:
        return default_pdv3_experiment_layout(
            output_root=self.output_root,
            experiment_name=self.pdv3_experiment_name,
        )

    @property
    def pdv3_root(self) -> Path:
        return self.pdv3_layout.root

    @property
    def root(self) -> Path:
        return self.pdv3_root / self.root_dirname

    @property
    def split_manifest_path(self) -> Path:
        return self.pdv3_layout.split_manifest_path

    @property
    def hlt_cache_dir(self) -> Path:
        return self.pdv3_layout.hlt_cache_dir

    @property
    def hlt2_cache_root(self) -> Path:
        return self.pdv3_root / "hlt_self_dualview" / "hlt2_cache"

    @property
    def source_models_dir(self) -> Path:
        return self.root / HLT_MV_SOURCE_MODELS_DIRNAME

    @property
    def random_hlt_controls_dir(self) -> Path:
        return self.root / HLT_MV_RANDOM_HLT_CONTROLS_DIRNAME

    @property
    def logit_fusions_dir(self) -> Path:
        return self.root / HLT_MV_LOGIT_FUSIONS_DIRNAME

    @property
    def pretrained_dualview_dir(self) -> Path:
        return self.root / HLT_MV_PRETRAINED_DUALVIEW_DIRNAME

    @property
    def scratch_dualview_dir(self) -> Path:
        return self.root / HLT_MV_SCRATCH_DUALVIEW_DIRNAME

    @property
    def controls_dir(self) -> Path:
        return self.root / HLT_MV_CONTROLS_DIRNAME

    @property
    def triview_dir(self) -> Path:
        return self.root / HLT_MV_TRIVIEW_DIRNAME

    @property
    def final_report_dir(self) -> Path:
        return self.root / HLT_MV_REPORT_DIRNAME

    @property
    def final_report_path(self) -> Path:
        return self.final_report_dir / "summary.json"

    def hlt2_cache_dir(self, strength: float | int | str) -> Path:
        return self.hlt2_cache_root / hlt_sdv_hlt2_cache_name(strength)

    def source_model_dir(self, name: str) -> Path:
        return self.source_models_dir / normalize_hlt_mv_source_name(name)

    def random_hlt_source_dir(self, name: str) -> Path:
        return self.random_hlt_controls_dir / normalize_hlt_mv_source_name(name)

    def logit_fusion_dir(self, name: str) -> Path:
        return self.logit_fusions_dir / normalize_hlt_mv_source_name(name)

    def pretrained_dualview_model_dir(self, name: str) -> Path:
        return self.pretrained_dualview_dir / normalize_hlt_mv_source_name(name)

    def scratch_dualview_model_dir(self, name: str) -> Path:
        return self.scratch_dualview_dir / normalize_hlt_mv_source_name(name)

    def control_dir(self, name: str) -> Path:
        return self.controls_dir / normalize_hlt_mv_source_name(name)

    def triview_model_dir(self, name: str = HLT_MV_TRIVIEW_MODEL_NAME) -> Path:
        return self.triview_dir / normalize_hlt_mv_source_name(name)

    def to_dict(self, *, config: HLTMVExperimentConfig | None = None) -> dict[str, Any]:
        cfg = config or default_hlt_mv_experiment_config(pdv3_experiment_name=self.pdv3_experiment_name)
        return {
            "contract": HLT_MV_CONTRACT,
            "output_root": str(Path(self.output_root)),
            "pdv3_experiment_name": self.pdv3_experiment_name,
            "pdv3_root": self.pdv3_root.as_posix(),
            "root": self.root.as_posix(),
            "split_manifest_path": self.split_manifest_path.as_posix(),
            "hlt_cache_dir": self.hlt_cache_dir.as_posix(),
            "hlt2_cache_root": self.hlt2_cache_root.as_posix(),
            "hlt2_cache_dirs": {
                hlt_sdv_strength_tag(strength): self.hlt2_cache_dir(strength).as_posix()
                for strength in cfg.strengths
            },
            "source_models_dir": self.source_models_dir.as_posix(),
            "source_model_dirs": {
                name: self.source_model_dir(name).as_posix()
                for name in cfg.source_model_names
            },
            "random_hlt_controls_dir": self.random_hlt_controls_dir.as_posix(),
            "random_hlt_source_dirs": {
                name: self.random_hlt_source_dir(name).as_posix()
                for name in cfg.random_hlt_source_names
            },
            "logit_fusions_dir": self.logit_fusions_dir.as_posix(),
            "logit_fusion_dirs": {
                name: self.logit_fusion_dir(name).as_posix()
                for name in cfg.logit_fusion_names
            },
            "pretrained_dualview_dir": self.pretrained_dualview_dir.as_posix(),
            "pretrained_dualview_model_dirs": {
                name: self.pretrained_dualview_model_dir(name).as_posix()
                for name in cfg.pretrained_dualview_names
            },
            "scratch_dualview_dir": self.scratch_dualview_dir.as_posix(),
            "scratch_dualview_model_dirs": {
                name: self.scratch_dualview_model_dir(name).as_posix()
                for name in cfg.scratch_dualview_names
            },
            "controls_dir": self.controls_dir.as_posix(),
            "control_dirs": {
                name: self.control_dir(name).as_posix()
                for name in cfg.control_names
            },
            "triview_dir": self.triview_dir.as_posix(),
            "triview_model_dir": self.triview_model_dir().as_posix(),
            "final_report_dir": self.final_report_dir.as_posix(),
            "final_report_path": self.final_report_path.as_posix(),
        }


def default_hlt_mv_experiment_config(
    *,
    pdv3_experiment_name: str = HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
    strengths: tuple[float, ...] = HLT_MV_DEFAULT_STRENGTHS,
) -> HLTMVExperimentConfig:
    return HLTMVExperimentConfig(
        pdv3_experiment_name=pdv3_experiment_name,
        strengths=strengths,
    )


def default_hlt_mv_experiment_layout(
    *,
    output_root: str | Path = "checkpoints",
    pdv3_experiment_name: str = HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
) -> HLTMVExperimentLayout:
    return HLTMVExperimentLayout(
        output_root=output_root,
        pdv3_experiment_name=pdv3_experiment_name,
    )


def hlt_mv_config_manifest(
    *,
    config: HLTMVExperimentConfig | None = None,
    layout: HLTMVExperimentLayout | None = None,
) -> dict[str, Any]:
    cfg = config or default_hlt_mv_experiment_config()
    lay = layout or default_hlt_mv_experiment_layout(pdv3_experiment_name=cfg.pdv3_experiment_name)
    return {
        "contract": HLT_MV_CONTRACT,
        "config": cfg.to_dict(),
        "layout": lay.to_dict(config=cfg),
    }


__all__ = [
    "HLTMVExperimentConfig",
    "HLTMVExperimentLayout",
    "HLT_MV_ALLOWED_INPUTS",
    "HLT_MV_CANONICAL_HLT_SOURCE",
    "HLT_MV_CONTRACT",
    "HLT_MV_DEFAULT_HLT_RANDOM_SEEDS",
    "HLT_MV_DEFAULT_HLT_SOURCE_SEED",
    "HLT_MV_DEFAULT_HLT2_SOURCE_SEEDS",
    "HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME",
    "HLT_MV_DEFAULT_STRENGTHS",
    "HLT_MV_DEPLOYMENT_INPUTS",
    "HLT_MV_EXPERIMENT_NAME",
    "HLT_MV_EXPERIMENT_STEP",
    "HLT_MV_FUSION_HLT_RANDOM_4SEED",
    "HLT_MV_FUSION_PRETRAINED_DUALVIEW_4MODEL",
    "HLT_MV_FUSION_SCRATCH_DUALVIEW_4MODEL",
    "HLT_MV_FUSION_SOURCE_5VIEW",
    "HLT_MV_PDV3_HLT_DEGRADATION_STRENGTH",
    "HLT_MV_PDV3_HLT_PROFILE",
    "HLT_MV_ROOT_DIRNAME",
    "HLT_MV_TRIVIEW_MODEL_NAME",
    "build_hlt_mv_control_names",
    "build_hlt_mv_hlt2_source_names",
    "build_hlt_mv_pretrained_dualview_names",
    "build_hlt_mv_random_hlt_source_names",
    "build_hlt_mv_scratch_dualview_names",
    "default_hlt_mv_experiment_config",
    "default_hlt_mv_experiment_layout",
    "hlt_mv_config_manifest",
    "hlt_mv_hlt2_only_name",
    "hlt_mv_hlt2_source_name",
    "hlt_mv_pretrained_dualview_name",
    "hlt_mv_random_hlt_source_name",
    "hlt_mv_scratch_dualview_name",
    "hlt_mv_tta_name",
    "normalize_hlt_mv_source_name",
    "normalize_hlt_mv_strengths",
]
