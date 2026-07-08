"""Configuration and layout helpers for deployable HLT self-dualview runs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from teacher_logit_reco.privileged_distill_10class.config import (
    PD10_EXPERIMENT_NAME,
    PD10_SPLIT_ORDER,
    PD10_SPLIT_SIZES,
    PD10_TEACHER_HLT,
    default_pd10_experiment_layout,
)

HLT_SDV_EXPERIMENT_NAME = "hlt_self_dualview_10class"
HLT_SDV_EXPERIMENT_STEP = "hlt_sdv_step1_config_layout"
HLT_SDV_CONTRACT = "hlt_self_dualview_10class_layout_v1"
HLT_SDV_ROOT_DIRNAME = "hlt_self_dualview"

HLT_SDV_HLT2_PROFILE_NAME = "hlt_second_degrade_mild_v1"
HLT_SDV_HLT2_PROFILE_VERSION = "v1"
HLT_SDV_ALLOWED_INPUTS = "HLT_only"
HLT_SDV_DEPLOYMENT_INPUTS = "HLT_plus_deterministic_HLT2"

HLT_SDV_DEFAULT_STRENGTHS = (0.00, 0.10, 0.20, 0.35, 1.00)
HLT_SDV_IDENTITY_STRENGTH = 0.00
HLT_SDV_PRIMARY_STRENGTH = 0.20

HLT_SDV_VARIANT_SAME_VIEW = "sdv_hlt_hlt_same_view"
HLT_SDV_VARIANT_HLT2_ONLY = "hlt2_only_part_s0p20"
HLT_SDV_VARIANT_TTA = "tta_hlt_part_hlt_plus_hlt2_s0p20"
HLT_SDV_VARIANT_HLT2_PREFIX = "sdv_hlt_hlt2"

HLT_SDV_MODEL_DIRNAME = "models"
HLT_SDV_HLT2_CACHE_DIRNAME = "hlt2_cache"
HLT_SDV_AUDIT_DIRNAME = "audits"
HLT_SDV_REPORT_DIRNAME = "final_report"


def normalize_hlt_sdv_strength(strength: float | int | str) -> float:
    """Return a canonical rounded HLT2 degradation strength."""

    if isinstance(strength, str):
        value = strength.strip()
        if value.startswith("s"):
            value = value[1:]
        value = value.replace("p", ".").replace("m", "-")
        strength_value = float(value)
    else:
        strength_value = float(strength)
    if not math.isfinite(strength_value):
        raise ValueError(f"HLT self-dualview strength must be finite, got {strength!r}.")
    if strength_value < 0:
        raise ValueError(f"HLT self-dualview strength must be non-negative, got {strength!r}.")
    return round(strength_value, 2)


def hlt_sdv_strength_tag(strength: float | int | str) -> str:
    """Format strengths as stable path fragments, e.g. 0.20 -> s0p20."""

    strength_value = normalize_hlt_sdv_strength(strength)
    return f"s{strength_value:.2f}".replace(".", "p").replace("-", "m")


def hlt_sdv_hlt2_cache_name(strength: float | int | str) -> str:
    """Return the directory name for one deterministic second-HLT cache."""

    return f"{HLT_SDV_HLT2_PROFILE_NAME}_{hlt_sdv_strength_tag(strength)}"


def hlt_sdv_dual_hlt2_variant_name(strength: float | int | str) -> str:
    """Return the deployable dual-view model variant for one HLT2 strength."""

    return f"{HLT_SDV_VARIANT_HLT2_PREFIX}_{hlt_sdv_strength_tag(strength)}"


def hlt_sdv_strength_from_variant(variant: str, *, default: float | None = None) -> float | None:
    """Extract an HLT2 strength from a variant name when one is encoded."""

    value = normalize_hlt_sdv_variant(variant)
    prefix = f"{HLT_SDV_VARIANT_HLT2_PREFIX}_"
    if value.startswith(prefix):
        strength_tag = value.removeprefix(prefix)
        if strength_tag.endswith("_scratch"):
            strength_tag = strength_tag.removesuffix("_scratch")
        return normalize_hlt_sdv_strength(strength_tag)
    if value.startswith("hlt2_only_part_"):
        return normalize_hlt_sdv_strength(value.removeprefix("hlt2_only_part_"))
    if value.startswith("tta_hlt_part_hlt_plus_hlt2_"):
        return normalize_hlt_sdv_strength(value.removeprefix("tta_hlt_part_hlt_plus_hlt2_"))
    return default


def normalize_hlt_sdv_variant(variant: str) -> str:
    """Validate and normalize a model variant name."""

    value = str(variant).strip()
    if not value:
        raise ValueError("HLT self-dualview variant name cannot be empty.")
    if any(ch.isspace() for ch in value):
        raise ValueError(f"HLT self-dualview variant name cannot contain spaces: {variant!r}.")
    return value


def normalize_hlt_sdv_strengths(strengths: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    """Normalize a strength list while preserving the requested order."""

    normalized = tuple(normalize_hlt_sdv_strength(value) for value in strengths)
    if not normalized:
        raise ValueError("HLT self-dualview strength list cannot be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"HLT self-dualview strengths must be unique, got {normalized!r}.")
    if HLT_SDV_IDENTITY_STRENGTH not in normalized:
        raise ValueError("HLT self-dualview strengths must include the identity 0.00 control.")
    return normalized


def build_hlt_sdv_cache_strengths(
    strengths: tuple[float, ...] | list[float] = HLT_SDV_DEFAULT_STRENGTHS,
) -> tuple[float, ...]:
    """Return the HLT2 strengths that should have deterministic caches."""

    return normalize_hlt_sdv_strengths(strengths)


def build_hlt_sdv_required_variants(
    strengths: tuple[float, ...] | list[float] = HLT_SDV_DEFAULT_STRENGTHS,
    *,
    include_hlt2_only: bool = True,
    include_tta: bool = True,
) -> tuple[str, ...]:
    """Return the Step-1 model variant names implied by the strength grid."""

    normalized_strengths = build_hlt_sdv_cache_strengths(strengths)
    variants = [HLT_SDV_VARIANT_SAME_VIEW]
    variants.extend(
        hlt_sdv_dual_hlt2_variant_name(strength)
        for strength in normalized_strengths
        if strength != HLT_SDV_IDENTITY_STRENGTH
    )
    if include_hlt2_only:
        variants.append(HLT_SDV_VARIANT_HLT2_ONLY)
    if include_tta:
        variants.append(HLT_SDV_VARIANT_TTA)
    return tuple(variants)


@dataclass(frozen=True)
class HLTSDVExperimentConfig:
    """Small immutable config for the deployable HLT self-dualview experiment."""

    pd10_experiment_name: str = PD10_EXPERIMENT_NAME
    hlt2_strengths: tuple[float, ...] = HLT_SDV_DEFAULT_STRENGTHS
    primary_strength: float = HLT_SDV_PRIMARY_STRENGTH
    hlt2_profile_name: str = HLT_SDV_HLT2_PROFILE_NAME
    hlt2_profile_version: str = HLT_SDV_HLT2_PROFILE_VERSION
    reuse_pd10_split_manifest: bool = True
    reuse_pd10_hlt_cache: bool = True
    allowed_inputs: str = HLT_SDV_ALLOWED_INPUTS
    deployment_inputs: str = HLT_SDV_DEPLOYMENT_INPUTS
    split_order: tuple[str, ...] = PD10_SPLIT_ORDER
    split_sizes: Mapping[str, int] = field(default_factory=lambda: dict(PD10_SPLIT_SIZES))

    def __post_init__(self) -> None:
        strengths = build_hlt_sdv_cache_strengths(self.hlt2_strengths)
        primary = normalize_hlt_sdv_strength(self.primary_strength)
        if primary not in strengths:
            raise ValueError(
                f"Primary HLT self-dualview strength {primary:.2f} is not in {strengths!r}."
            )
        if self.hlt2_profile_name != HLT_SDV_HLT2_PROFILE_NAME:
            raise ValueError(
                f"Unexpected HLT2 profile {self.hlt2_profile_name!r}; "
                f"expected {HLT_SDV_HLT2_PROFILE_NAME!r}."
            )
        if self.hlt2_profile_version != HLT_SDV_HLT2_PROFILE_VERSION:
            raise ValueError(
                f"Unexpected HLT2 profile version {self.hlt2_profile_version!r}; "
                f"expected {HLT_SDV_HLT2_PROFILE_VERSION!r}."
            )
        if not self.reuse_pd10_split_manifest:
            raise ValueError("HLT self-dualview must reuse the existing PD10 split manifest.")
        if not self.reuse_pd10_hlt_cache:
            raise ValueError("HLT self-dualview must reuse the existing PD10 HLT cache.")
        if self.allowed_inputs != HLT_SDV_ALLOWED_INPUTS:
            raise ValueError(
                f"HLT self-dualview allowed_inputs must be {HLT_SDV_ALLOWED_INPUTS!r}."
            )
        if self.deployment_inputs != HLT_SDV_DEPLOYMENT_INPUTS:
            raise ValueError(
                f"HLT self-dualview deployment_inputs must be {HLT_SDV_DEPLOYMENT_INPUTS!r}."
            )
        if tuple(self.split_order) != PD10_SPLIT_ORDER:
            raise ValueError("HLT self-dualview split order must match the PD10 split order.")
        if dict(self.split_sizes) != dict(PD10_SPLIT_SIZES):
            raise ValueError("HLT self-dualview split sizes must match the PD10 split sizes.")
        object.__setattr__(self, "hlt2_strengths", strengths)
        object.__setattr__(self, "primary_strength", primary)
        object.__setattr__(self, "split_sizes", dict(self.split_sizes))

    @property
    def hlt2_cache_names(self) -> tuple[str, ...]:
        return tuple(hlt_sdv_hlt2_cache_name(strength) for strength in self.hlt2_strengths)

    @property
    def variants(self) -> tuple[str, ...]:
        return build_hlt_sdv_required_variants(self.hlt2_strengths)

    @property
    def primary_variant(self) -> str:
        return hlt_sdv_dual_hlt2_variant_name(self.primary_strength)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": HLT_SDV_CONTRACT,
            "experiment_name": HLT_SDV_EXPERIMENT_NAME,
            "pd10_experiment_name": self.pd10_experiment_name,
            "hlt2_profile_name": self.hlt2_profile_name,
            "hlt2_profile_version": self.hlt2_profile_version,
            "hlt2_strengths": list(self.hlt2_strengths),
            "primary_strength": self.primary_strength,
            "primary_variant": self.primary_variant,
            "hlt2_cache_names": list(self.hlt2_cache_names),
            "variants": list(self.variants),
            "reuse_pd10_split_manifest": self.reuse_pd10_split_manifest,
            "reuse_pd10_hlt_cache": self.reuse_pd10_hlt_cache,
            "allowed_inputs": self.allowed_inputs,
            "deployment_inputs": self.deployment_inputs,
            "split_order": list(self.split_order),
            "split_sizes": dict(self.split_sizes),
        }


@dataclass(frozen=True)
class HLTSDVExperimentLayout:
    """Filesystem layout rooted under an existing PD10 experiment directory."""

    output_root: str | Path = "checkpoints"
    pd10_experiment_name: str = PD10_EXPERIMENT_NAME
    root_dirname: str = HLT_SDV_ROOT_DIRNAME

    @property
    def pd10_layout(self):
        return default_pd10_experiment_layout(
            output_root=self.output_root,
            experiment_name=self.pd10_experiment_name,
        )

    @property
    def pd10_root(self) -> Path:
        return self.pd10_layout.root

    @property
    def root(self) -> Path:
        return self.pd10_root / self.root_dirname

    @property
    def split_manifest_path(self) -> Path:
        return self.pd10_layout.split_manifest_path

    @property
    def parent_hlt_cache_dir(self) -> Path:
        return self.pd10_layout.hlt_cache_dir

    @property
    def hlt_teacher_checkpoint(self) -> Path:
        return self.pd10_layout.teacher_checkpoint(PD10_TEACHER_HLT)

    @property
    def hlt2_cache_root(self) -> Path:
        return self.root / HLT_SDV_HLT2_CACHE_DIRNAME

    @property
    def models_dir(self) -> Path:
        return self.root / HLT_SDV_MODEL_DIRNAME

    @property
    def audits_dir(self) -> Path:
        return self.root / HLT_SDV_AUDIT_DIRNAME

    @property
    def final_report_dir(self) -> Path:
        return self.root / HLT_SDV_REPORT_DIRNAME

    @property
    def final_report_path(self) -> Path:
        return self.final_report_dir / "summary.json"

    def hlt2_cache_dir(self, strength: float | int | str) -> Path:
        return self.hlt2_cache_root / hlt_sdv_hlt2_cache_name(strength)

    def variant_dir(self, variant: str) -> Path:
        return self.models_dir / normalize_hlt_sdv_variant(variant)

    def variant_report_path(self, variant: str) -> Path:
        return self.variant_dir(variant) / "final_test_metrics.json"

    def to_dict(
        self,
        *,
        strengths: tuple[float, ...] | list[float] = HLT_SDV_DEFAULT_STRENGTHS,
        variants: tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, Any]:
        cache_strengths = build_hlt_sdv_cache_strengths(strengths)
        variant_names = tuple(variants) if variants is not None else build_hlt_sdv_required_variants(
            cache_strengths
        )
        return {
            "contract": HLT_SDV_CONTRACT,
            "output_root": str(Path(self.output_root)),
            "pd10_experiment_name": self.pd10_experiment_name,
            "pd10_root": self.pd10_root.as_posix(),
            "root": self.root.as_posix(),
            "split_manifest_path": self.split_manifest_path.as_posix(),
            "parent_hlt_cache_dir": self.parent_hlt_cache_dir.as_posix(),
            "hlt_teacher_checkpoint": self.hlt_teacher_checkpoint.as_posix(),
            "hlt2_cache_root": self.hlt2_cache_root.as_posix(),
            "hlt2_cache_dirs": {
                hlt_sdv_strength_tag(strength): self.hlt2_cache_dir(strength).as_posix()
                for strength in cache_strengths
            },
            "models_dir": self.models_dir.as_posix(),
            "variant_dirs": {
                normalize_hlt_sdv_variant(variant): self.variant_dir(variant).as_posix()
                for variant in variant_names
            },
            "audits_dir": self.audits_dir.as_posix(),
            "final_report_dir": self.final_report_dir.as_posix(),
            "final_report_path": self.final_report_path.as_posix(),
        }


def default_hlt_sdv_experiment_config(
    *,
    pd10_experiment_name: str = PD10_EXPERIMENT_NAME,
    hlt2_strengths: tuple[float, ...] = HLT_SDV_DEFAULT_STRENGTHS,
) -> HLTSDVExperimentConfig:
    return HLTSDVExperimentConfig(
        pd10_experiment_name=pd10_experiment_name,
        hlt2_strengths=hlt2_strengths,
    )


def default_hlt_sdv_experiment_layout(
    *,
    output_root: str | Path = "checkpoints",
    pd10_experiment_name: str = PD10_EXPERIMENT_NAME,
) -> HLTSDVExperimentLayout:
    return HLTSDVExperimentLayout(
        output_root=output_root,
        pd10_experiment_name=pd10_experiment_name,
    )


def hlt_sdv_config_manifest(
    *,
    config: HLTSDVExperimentConfig | None = None,
    layout: HLTSDVExperimentLayout | None = None,
) -> dict[str, Any]:
    cfg = config or default_hlt_sdv_experiment_config()
    ly = layout or default_hlt_sdv_experiment_layout(pd10_experiment_name=cfg.pd10_experiment_name)
    return {
        "contract": HLT_SDV_CONTRACT,
        "config": cfg.to_dict(),
        "layout": ly.to_dict(strengths=cfg.hlt2_strengths, variants=cfg.variants),
    }


__all__ = [
    "HLTSDVExperimentConfig",
    "HLTSDVExperimentLayout",
    "HLT_SDV_ALLOWED_INPUTS",
    "HLT_SDV_AUDIT_DIRNAME",
    "HLT_SDV_CONTRACT",
    "HLT_SDV_DEFAULT_STRENGTHS",
    "HLT_SDV_DEPLOYMENT_INPUTS",
    "HLT_SDV_EXPERIMENT_NAME",
    "HLT_SDV_EXPERIMENT_STEP",
    "HLT_SDV_HLT2_CACHE_DIRNAME",
    "HLT_SDV_HLT2_PROFILE_NAME",
    "HLT_SDV_HLT2_PROFILE_VERSION",
    "HLT_SDV_IDENTITY_STRENGTH",
    "HLT_SDV_MODEL_DIRNAME",
    "HLT_SDV_PRIMARY_STRENGTH",
    "HLT_SDV_REPORT_DIRNAME",
    "HLT_SDV_ROOT_DIRNAME",
    "HLT_SDV_VARIANT_HLT2_ONLY",
    "HLT_SDV_VARIANT_HLT2_PREFIX",
    "HLT_SDV_VARIANT_SAME_VIEW",
    "HLT_SDV_VARIANT_TTA",
    "build_hlt_sdv_cache_strengths",
    "build_hlt_sdv_required_variants",
    "default_hlt_sdv_experiment_config",
    "default_hlt_sdv_experiment_layout",
    "hlt_sdv_config_manifest",
    "hlt_sdv_dual_hlt2_variant_name",
    "hlt_sdv_hlt2_cache_name",
    "hlt_sdv_strength_from_variant",
    "hlt_sdv_strength_tag",
    "normalize_hlt_sdv_strength",
    "normalize_hlt_sdv_strengths",
    "normalize_hlt_sdv_variant",
]
