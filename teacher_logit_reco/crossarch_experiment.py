"""Fresh cross-architecture teacher-logit experiment definitions.

This module is deliberately configuration-only.  It defines the names, split
sizes, source grids, fusion groups, fuser list, and output layout for the
500k/150k/500k cross-architecture experiment without launching training or
loading model checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPERIMENT_NAME = "teacher_logit_reco_crossarch_500k"
EXPERIMENT_STEP = "teacher_logit_crossarch_step1_config"

SPLIT_SIZES: dict[str, int] = {
    "model_train": 500_000,
    "model_val": 150_000,
    "stack_train": 500_000,
    "stack_val": 150_000,
    "final_test": 500_000,
}
SPLIT_ORDER: tuple[str, ...] = ("model_train", "model_val", "stack_train", "stack_val", "final_test")

RECONSTRUCTOR_ARCHITECTURES: tuple[str, ...] = ("gt", "pn", "pfn", "pcnn")
RECONSTRUCTOR_IMPLEMENTATIONS: dict[str, str] = {
    "gt": "global_transformer",
    "pn": "particle_net",
    "pfn": "particle_flow",
    "pcnn": "particle_cnn",
}
RECONSTRUCTOR_ALIASES: dict[str, str] = {
    "global_transformer": "gt",
    "global": "gt",
    "transformer": "gt",
    "part": "gt",
    "part_reco": "gt",
    "ParticleNet": "pn",
    "particlenet": "pn",
    "particle_net": "pn",
    "ParticleFlow": "pfn",
    "particleflow": "pfn",
    "particle_flow": "pfn",
    "pf": "pfn",
    "pfc": "pfn",
    "P-CNN": "pcnn",
    "pcnn_reco": "pcnn",
    "particlecnn": "pcnn",
    "particle_cnn": "pcnn",
    "cnn": "pcnn",
}

TEACHER_ARCHITECTURES: tuple[str, ...] = ("part", "pn", "pfn", "pcnn")
TEACHER_ALIASES: dict[str, str] = {
    "particletransformer": "part",
    "particle_transformer": "part",
    "parT": "part",
    "ParticleTransformer": "part",
    "particle_net": "pn",
    "particleflow": "pfn",
    "particle_flow": "pfn",
    "pf": "pfn",
    "pfc": "pfn",
    "particle_cnn": "pcnn",
    "cnn": "pcnn",
}
DIRECT_HLT_ARCHITECTURES: tuple[str, ...] = TEACHER_ARCHITECTURES

SAME_FAMILY_RECO_TEACHER_PAIRS: tuple[tuple[str, str], ...] = (
    ("gt", "part"),
    ("pn", "pn"),
    ("pfn", "pfn"),
    ("pcnn", "pcnn"),
)
MIXED4_RECO_TEACHER_PAIRS: tuple[tuple[str, str], ...] = (
    ("gt", "pn"),
    ("pn", "pfn"),
    ("pfn", "pcnn"),
    ("pcnn", "part"),
)

REQUIRED_FUSION_GROUPS: tuple[str, ...] = ("all16", "cross12", "part_teacher4", "mixed4", "hlt4")
OPTIONAL_FUSION_GROUPS: tuple[str, ...] = (
    "all16_plus_hlt4",
    "cross12_plus_hlt4",
    "part_teacher4_plus_hlt_part",
)

DEFAULT_FUSERS: tuple[str, ...] = (
    "mean_logits",
    "mean_probs",
    "logistic_logits",
    "logistic_probs",
    "logistic_logits_probs",
    "uncertainty_logistic_logits_probs",
    "entropy_bin_gated_logistic",
    "margin_bin_gated_logistic",
    "multiplicity_bin_gated_logistic",
    "disagreement_bin_gated_logistic",
    "predicted_class_bin_gated_logistic",
)


def _normalize(value: str, *, allowed: tuple[str, ...], aliases: Mapping[str, str], kind: str) -> str:
    text = str(value).strip()
    lowered = text.lower()
    normalized = aliases.get(text, aliases.get(lowered, lowered))
    if normalized not in allowed:
        raise ValueError(f"Unknown {kind} {value!r}; expected one of {allowed}")
    return normalized


def normalize_reconstructor_architecture(value: str) -> str:
    return _normalize(
        value,
        allowed=RECONSTRUCTOR_ARCHITECTURES,
        aliases=RECONSTRUCTOR_ALIASES,
        kind="crossarch reconstructor architecture",
    )


def normalize_teacher_architecture(value: str) -> str:
    return _normalize(
        value,
        allowed=TEACHER_ARCHITECTURES,
        aliases=TEACHER_ALIASES,
        kind="crossarch teacher architecture",
    )


def reco_model_name(reco_architecture: str, teacher_architecture: str) -> str:
    reco = normalize_reconstructor_architecture(reco_architecture)
    teacher = normalize_teacher_architecture(teacher_architecture)
    return f"{reco}_reco_to_{teacher}_teacher"


def hlt_model_name(architecture: str) -> str:
    return f"hlt_{normalize_teacher_architecture(architecture)}"


@dataclass(frozen=True)
class CrossArchSourceSpec:
    """One prediction source in the cross-architecture experiment."""

    name: str
    source_kind: str
    reco_architecture: str | None = None
    teacher_architecture: str | None = None
    hlt_architecture: str | None = None
    reconstructor_implementation: str | None = None

    def __post_init__(self) -> None:
        if self.source_kind not in {"teacher_logit_reco", "direct_hlt"}:
            raise ValueError("source_kind must be teacher_logit_reco or direct_hlt")
        if self.source_kind == "teacher_logit_reco":
            if self.reco_architecture is None or self.teacher_architecture is None:
                raise ValueError("teacher_logit_reco sources require reco_architecture and teacher_architecture")
            reco = normalize_reconstructor_architecture(self.reco_architecture)
            teacher = normalize_teacher_architecture(self.teacher_architecture)
            expected_name = reco_model_name(reco, teacher)
            object.__setattr__(self, "reco_architecture", reco)
            object.__setattr__(self, "teacher_architecture", teacher)
            object.__setattr__(self, "hlt_architecture", None)
            object.__setattr__(self, "reconstructor_implementation", RECONSTRUCTOR_IMPLEMENTATIONS[reco])
            if self.name != expected_name:
                raise ValueError(f"Reco source name {self.name!r} should be {expected_name!r}")
        else:
            if self.hlt_architecture is None:
                raise ValueError("direct_hlt sources require hlt_architecture")
            hlt_arch = normalize_teacher_architecture(self.hlt_architecture)
            expected_name = hlt_model_name(hlt_arch)
            object.__setattr__(self, "hlt_architecture", hlt_arch)
            object.__setattr__(self, "reco_architecture", None)
            object.__setattr__(self, "teacher_architecture", None)
            object.__setattr__(self, "reconstructor_implementation", None)
            if self.name != expected_name:
                raise ValueError(f"HLT source name {self.name!r} should be {expected_name!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_kind": self.source_kind,
            "reco_architecture": self.reco_architecture,
            "teacher_architecture": self.teacher_architecture,
            "hlt_architecture": self.hlt_architecture,
            "reconstructor_implementation": self.reconstructor_implementation,
        }


@dataclass(frozen=True)
class FusionGroupSpec:
    """One named fusion group and its ordered model-name members."""

    name: str
    model_names: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.model_names)
        if len(names) != len(set(names)):
            raise ValueError(f"Fusion group {self.name!r} contains duplicate model names")
        if not names:
            raise ValueError(f"Fusion group {self.name!r} must contain at least one model")
        object.__setattr__(self, "model_names", names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_names": list(self.model_names),
            "n_models": len(self.model_names),
            "description": self.description,
        }


@dataclass(frozen=True)
class CrossArchExperimentLayout:
    """Path helper for the fresh cross-architecture output namespace."""

    output_root: str | Path = "checkpoints"
    experiment_name: str = EXPERIMENT_NAME

    @property
    def root(self) -> Path:
        return Path(self.output_root) / self.experiment_name

    @property
    def split_manifest_dir(self) -> Path:
        return self.root / "split_manifest"

    @property
    def split_manifest_path(self) -> Path:
        return self.split_manifest_dir / "split_manifest.json.gz"

    @property
    def hlt_cache_dir(self) -> Path:
        return self.root / "hlt_cache"

    @property
    def offline_teachers_dir(self) -> Path:
        return self.root / "offline_teachers"

    @property
    def hlt_baselines_dir(self) -> Path:
        return self.root / "hlt_baselines"

    @property
    def reco_models_dir(self) -> Path:
        return self.root / "reco_models"

    @property
    def prediction_runs_dir(self) -> Path:
        return self.root / "prediction_runs"

    @property
    def predictions_dir(self) -> Path:
        return self.root / "predictions"

    @property
    def fusion_dir(self) -> Path:
        return self.root / "fusion"

    @property
    def audits_dir(self) -> Path:
        return self.root / "audits"

    @property
    def final_report_dir(self) -> Path:
        return self.root / "final_report"

    def offline_teacher_dir(self, architecture: str) -> Path:
        return self.offline_teachers_dir / normalize_teacher_architecture(architecture)

    def offline_teacher_checkpoint(self, architecture: str) -> Path:
        return self.offline_teacher_dir(architecture) / "best_model_val.pt"

    def hlt_baseline_dir(self, architecture: str) -> Path:
        return self.hlt_baselines_dir / normalize_teacher_architecture(architecture)

    def hlt_baseline_checkpoint(self, architecture: str) -> Path:
        return self.hlt_baseline_dir(architecture) / "best_model_val.pt"

    def reco_model_dir(self, reco_architecture: str, teacher_architecture: str) -> Path:
        reco = normalize_reconstructor_architecture(reco_architecture)
        teacher = normalize_teacher_architecture(teacher_architecture)
        return self.reco_models_dir / reco / teacher

    def reco_model_checkpoint(self, reco_architecture: str, teacher_architecture: str) -> Path:
        return self.reco_model_dir(reco_architecture, teacher_architecture) / "best_model_val.pt"

    def prediction_source_dir(self, model_name: str) -> Path:
        return self.predictions_dir / str(model_name)

    def fusion_group_dir(self, group_name: str) -> Path:
        return self.fusion_dir / str(group_name)

    def to_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "split_manifest_dir": str(self.split_manifest_dir),
            "split_manifest_path": str(self.split_manifest_path),
            "hlt_cache_dir": str(self.hlt_cache_dir),
            "offline_teachers_dir": str(self.offline_teachers_dir),
            "hlt_baselines_dir": str(self.hlt_baselines_dir),
            "reco_models_dir": str(self.reco_models_dir),
            "prediction_runs_dir": str(self.prediction_runs_dir),
            "predictions_dir": str(self.predictions_dir),
            "fusion_dir": str(self.fusion_dir),
            "audits_dir": str(self.audits_dir),
            "final_report_dir": str(self.final_report_dir),
        }


@dataclass(frozen=True)
class CrossArchExperimentConfig:
    """Configuration-only descriptor for the 16x4 crossarch experiment."""

    output_root: str | Path = "checkpoints"
    experiment_name: str = EXPERIMENT_NAME
    split_sizes: Mapping[str, int] = field(default_factory=lambda: dict(SPLIT_SIZES))
    reconstructors: tuple[str, ...] = RECONSTRUCTOR_ARCHITECTURES
    teachers: tuple[str, ...] = TEACHER_ARCHITECTURES
    direct_hlt_architectures: tuple[str, ...] = DIRECT_HLT_ARCHITECTURES
    fusers: tuple[str, ...] = DEFAULT_FUSERS
    include_optional_groups: bool = False

    def __post_init__(self) -> None:
        split_sizes = {str(key): int(value) for key, value in self.split_sizes.items()}
        if tuple(split_sizes.keys()) != SPLIT_ORDER:
            raise ValueError(f"split_sizes keys must be exactly {SPLIT_ORDER} in order")
        if any(value <= 0 for value in split_sizes.values()):
            raise ValueError("split sizes must be positive")
        reconstructors = tuple(normalize_reconstructor_architecture(value) for value in self.reconstructors)
        teachers = tuple(normalize_teacher_architecture(value) for value in self.teachers)
        direct_hlt = tuple(normalize_teacher_architecture(value) for value in self.direct_hlt_architectures)
        if reconstructors != RECONSTRUCTOR_ARCHITECTURES:
            raise ValueError(f"reconstructors must be {RECONSTRUCTOR_ARCHITECTURES}")
        if teachers != TEACHER_ARCHITECTURES:
            raise ValueError(f"teachers must be {TEACHER_ARCHITECTURES}")
        if direct_hlt != DIRECT_HLT_ARCHITECTURES:
            raise ValueError(f"direct_hlt_architectures must be {DIRECT_HLT_ARCHITECTURES}")
        fusers = tuple(str(value) for value in self.fusers)
        if not fusers:
            raise ValueError("fusers must contain at least one fuser")
        object.__setattr__(self, "split_sizes", split_sizes)
        object.__setattr__(self, "reconstructors", reconstructors)
        object.__setattr__(self, "teachers", teachers)
        object.__setattr__(self, "direct_hlt_architectures", direct_hlt)
        object.__setattr__(self, "fusers", fusers)

    @property
    def layout(self) -> CrossArchExperimentLayout:
        return CrossArchExperimentLayout(output_root=self.output_root, experiment_name=self.experiment_name)

    @property
    def reco_sources(self) -> tuple[CrossArchSourceSpec, ...]:
        return build_reco_source_specs(self.reconstructors, self.teachers)

    @property
    def hlt_sources(self) -> tuple[CrossArchSourceSpec, ...]:
        return build_hlt_source_specs(self.direct_hlt_architectures)

    @property
    def all_sources(self) -> tuple[CrossArchSourceSpec, ...]:
        return self.reco_sources + self.hlt_sources

    @property
    def fusion_groups(self) -> dict[str, FusionGroupSpec]:
        return build_fusion_groups(include_optional=self.include_optional_groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_step": EXPERIMENT_STEP,
            "experiment_name": self.experiment_name,
            "split_sizes": dict(self.split_sizes),
            "reconstructors": list(self.reconstructors),
            "teachers": list(self.teachers),
            "direct_hlt_architectures": list(self.direct_hlt_architectures),
            "reco_sources": [source.to_dict() for source in self.reco_sources],
            "hlt_sources": [source.to_dict() for source in self.hlt_sources],
            "fusion_groups": {name: group.to_dict() for name, group in self.fusion_groups.items()},
            "fusers": list(self.fusers),
            "layout": self.layout.to_dict(),
        }


def build_reco_source_specs(
    reconstructors: Iterable[str] = RECONSTRUCTOR_ARCHITECTURES,
    teachers: Iterable[str] = TEACHER_ARCHITECTURES,
) -> tuple[CrossArchSourceSpec, ...]:
    specs = []
    for reco_architecture in reconstructors:
        reco = normalize_reconstructor_architecture(reco_architecture)
        for teacher_architecture in teachers:
            teacher = normalize_teacher_architecture(teacher_architecture)
            specs.append(
                CrossArchSourceSpec(
                    name=reco_model_name(reco, teacher),
                    source_kind="teacher_logit_reco",
                    reco_architecture=reco,
                    teacher_architecture=teacher,
                )
            )
    return tuple(specs)


def build_hlt_source_specs(
    architectures: Iterable[str] = DIRECT_HLT_ARCHITECTURES,
) -> tuple[CrossArchSourceSpec, ...]:
    return tuple(
        CrossArchSourceSpec(
            name=hlt_model_name(architecture),
            source_kind="direct_hlt",
            hlt_architecture=architecture,
        )
        for architecture in architectures
    )


def build_fusion_groups(*, include_optional: bool = False) -> dict[str, FusionGroupSpec]:
    reco_names = tuple(source.name for source in build_reco_source_specs())
    hlt_names = tuple(source.name for source in build_hlt_source_specs())
    same_family = {reco_model_name(reco, teacher) for reco, teacher in SAME_FAMILY_RECO_TEACHER_PAIRS}
    cross12 = tuple(name for name in reco_names if name not in same_family)
    part_teacher4 = tuple(reco_model_name(reco, "part") for reco in RECONSTRUCTOR_ARCHITECTURES)
    mixed4 = tuple(reco_model_name(reco, teacher) for reco, teacher in MIXED4_RECO_TEACHER_PAIRS)

    groups = {
        "all16": FusionGroupSpec(
            name="all16",
            model_names=reco_names,
            description="All sixteen teacher-logit reco/teacher combinations.",
        ),
        "cross12": FusionGroupSpec(
            name="cross12",
            model_names=cross12,
            description="All off-diagonal reco/teacher combinations, excluding same-family pairs.",
        ),
        "part_teacher4": FusionGroupSpec(
            name="part_teacher4",
            model_names=part_teacher4,
            description="All four reconstructors targeting the ParT offline teacher.",
        ),
        "mixed4": FusionGroupSpec(
            name="mixed4",
            model_names=mixed4,
            description="Cyclic mixed-bias quartet: gt->pn, pn->pfn, pfn->pcnn, pcnn->part.",
        ),
        "hlt4": FusionGroupSpec(
            name="hlt4",
            model_names=hlt_names,
            description="Four direct HLT tagger baselines.",
        ),
    }
    if include_optional:
        groups.update(
            {
                "all16_plus_hlt4": FusionGroupSpec(
                    name="all16_plus_hlt4",
                    model_names=reco_names + hlt_names,
                    description="All teacher-logit reco sources plus direct HLT baselines.",
                ),
                "cross12_plus_hlt4": FusionGroupSpec(
                    name="cross12_plus_hlt4",
                    model_names=cross12 + hlt_names,
                    description="Off-diagonal teacher-logit reco sources plus direct HLT baselines.",
                ),
                "part_teacher4_plus_hlt_part": FusionGroupSpec(
                    name="part_teacher4_plus_hlt_part",
                    model_names=part_teacher4 + (hlt_model_name("part"),),
                    description="ParT-teacher reconstructor quartet plus direct HLT ParT.",
                ),
            }
        )
    validate_fusion_groups(groups)
    return groups


def validate_fusion_groups(groups: Mapping[str, FusionGroupSpec]) -> None:
    required_counts = {
        "all16": 16,
        "cross12": 12,
        "part_teacher4": 4,
        "mixed4": 4,
        "hlt4": 4,
    }
    for name, expected_count in required_counts.items():
        if name not in groups:
            raise ValueError(f"Missing required fusion group {name!r}")
        actual_count = len(groups[name].model_names)
        if actual_count != expected_count:
            raise ValueError(f"Fusion group {name!r} has {actual_count} models; expected {expected_count}")

    all_reco_names = {source.name for source in build_reco_source_specs()}
    all_hlt_names = {source.name for source in build_hlt_source_specs()}
    known_names = all_reco_names | all_hlt_names
    for group in groups.values():
        unknown = sorted(set(group.model_names) - known_names)
        if unknown:
            raise ValueError(f"Fusion group {group.name!r} contains unknown models: {unknown}")


def default_crossarch_experiment_config() -> CrossArchExperimentConfig:
    return CrossArchExperimentConfig()


__all__ = [
    "DEFAULT_FUSERS",
    "DIRECT_HLT_ARCHITECTURES",
    "EXPERIMENT_NAME",
    "EXPERIMENT_STEP",
    "MIXED4_RECO_TEACHER_PAIRS",
    "OPTIONAL_FUSION_GROUPS",
    "RECONSTRUCTOR_ARCHITECTURES",
    "RECONSTRUCTOR_IMPLEMENTATIONS",
    "REQUIRED_FUSION_GROUPS",
    "SAME_FAMILY_RECO_TEACHER_PAIRS",
    "SPLIT_ORDER",
    "SPLIT_SIZES",
    "TEACHER_ARCHITECTURES",
    "CrossArchExperimentConfig",
    "CrossArchExperimentLayout",
    "CrossArchSourceSpec",
    "FusionGroupSpec",
    "build_fusion_groups",
    "build_hlt_source_specs",
    "build_reco_source_specs",
    "default_crossarch_experiment_config",
    "hlt_model_name",
    "normalize_reconstructor_architecture",
    "normalize_teacher_architecture",
    "reco_model_name",
    "validate_fusion_groups",
]
