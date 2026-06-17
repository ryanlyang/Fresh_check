"""Configuration boundary for the set-matching multi-view experiment.

This module deliberately stays configuration-only.  It defines names, view
contracts, split sizes, and output paths for the new branch without importing
teacher-logit losses or launching any reconstruction/training code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPERIMENT_NAME = "set_matching_multiview_500k"
EXPERIMENT_STEP = "set_matching_multiview_step1_config"

SPLIT_SIZES: dict[str, int] = {
    "model_train": 500_000,
    "model_val": 150_000,
    "stack_train": 500_000,
    "stack_val": 150_000,
    "final_test": 500_000,
}
SPLIT_ORDER: tuple[str, ...] = ("model_train", "model_val", "stack_train", "stack_val", "final_test")

SET_RECONSTRUCTOR_ARCHITECTURES: tuple[str, ...] = ("gt", "pn", "pfn", "pcnn")
SET_RECONSTRUCTOR_IMPLEMENTATIONS: dict[str, str] = {
    "gt": "set_matching_global_transformer",
    "pn": "set_matching_particle_net",
    "pfn": "set_matching_particle_flow",
    "pcnn": "set_matching_particle_cnn",
}
SET_RECONSTRUCTOR_ALIASES: dict[str, str] = {
    "global": "gt",
    "global_transformer": "gt",
    "globaltransformer": "gt",
    "gt_reco": "gt",
    "parT": "gt",
    "part": "gt",
    "particle_transformer": "gt",
    "particletransformer": "gt",
    "transformer": "gt",
    "edgeconv": "pn",
    "particle_net": "pn",
    "particlenet": "pn",
    "pn_reco": "pn",
    "deep_sets": "pfn",
    "deepsets": "pfn",
    "particle_flow": "pfn",
    "particleflow": "pfn",
    "pf": "pfn",
    "pfc": "pfn",
    "pfn_reco": "pfn",
    "cnn": "pcnn",
    "p_cnn": "pcnn",
    "particle_cnn": "pcnn",
    "particlecnn": "pcnn",
    "pcnn_reco": "pcnn",
}

HLT_VIEW_NAME = "hlt"
RECONSTRUCTED_VIEW_NAMES: tuple[str, ...] = ("gt_reco", "pn_reco", "pfn_reco", "pcnn_reco")
VIEW_NAMES: tuple[str, ...] = (HLT_VIEW_NAME,) + RECONSTRUCTED_VIEW_NAMES
FIVE_VIEW_GROUP_NAME = "hlt_plus_four_set_recos"

VIEW_KIND_ORIGINAL_HLT = "original_hlt"
VIEW_KIND_SET_RECONSTRUCTION = "set_reconstruction"
VIEW_KINDS: tuple[str, ...] = (VIEW_KIND_ORIGINAL_HLT, VIEW_KIND_SET_RECONSTRUCTION)

SOURCE_TYPE_ORIGINAL_HLT = "original_hlt"
SOURCE_TYPE_RECONSTRUCTED = "reconstructed"
SOURCE_TYPES: tuple[str, ...] = (SOURCE_TYPE_ORIGINAL_HLT, SOURCE_TYPE_RECONSTRUCTED)

DEFAULT_PARTICLE_FEATURE_DIM = 19
DEFAULT_MAX_SLOTS = 128
DEFAULT_MAX_TOKENS_PER_VIEW = 128
DEFAULT_MIN_TOKENS_PER_VIEW = 8
DEFAULT_CONFIDENCE_THRESHOLD = 0.05


def _alias_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def normalize_split_name(value: str) -> str:
    split = str(value).strip()
    if split not in SPLIT_ORDER:
        raise ValueError(f"Unknown set-matching split {value!r}; expected one of {SPLIT_ORDER}")
    return split


def normalize_set_reconstructor_architecture(value: str) -> str:
    text = str(value).strip()
    key = _alias_key(text)
    normalized = SET_RECONSTRUCTOR_ALIASES.get(text, SET_RECONSTRUCTOR_ALIASES.get(key, key))
    if normalized is None:
        normalized = key
    if normalized not in SET_RECONSTRUCTOR_ARCHITECTURES:
        raise ValueError(
            f"Unknown set-matching reconstructor architecture {value!r}; "
            f"expected one of {SET_RECONSTRUCTOR_ARCHITECTURES}"
        )
    return normalized


def view_name_for_reconstructor(architecture: str) -> str:
    return f"{normalize_set_reconstructor_architecture(architecture)}_reco"


def set_reconstructor_model_name(architecture: str) -> str:
    return f"setmatch_{normalize_set_reconstructor_architecture(architecture)}_reco"


def normalize_view_name(value: str) -> str:
    text = str(value).strip()
    key = _alias_key(text)
    if key in {"hlt", "hlt_view", "original_hlt", "fixed_hlt", "fixed_hlt_view"}:
        return HLT_VIEW_NAME
    try:
        candidate = view_name_for_reconstructor(key)
    except ValueError:
        candidate = key
    if candidate in VIEW_NAMES:
        return candidate
    if key in RECONSTRUCTED_VIEW_NAMES:
        return key
    raise ValueError(f"Unknown set-matching view name {value!r}; expected one of {VIEW_NAMES}")


@dataclass(frozen=True)
class SetMatchingViewSpec:
    """One input view in the set-matching multi-view branch."""

    name: str
    view_kind: str
    reconstructor_architecture: str | None = None
    source_type: str | None = None

    def __post_init__(self) -> None:
        view_kind = str(self.view_kind).strip()
        if view_kind not in VIEW_KINDS:
            raise ValueError(f"view_kind must be one of {VIEW_KINDS}")
        name = normalize_view_name(self.name)

        if view_kind == VIEW_KIND_ORIGINAL_HLT:
            if name != HLT_VIEW_NAME:
                raise ValueError(f"original HLT view must be named {HLT_VIEW_NAME!r}")
            if self.reconstructor_architecture is not None:
                raise ValueError("original HLT view cannot declare a reconstructor architecture")
            source_type = self.source_type or SOURCE_TYPE_ORIGINAL_HLT
            if source_type != SOURCE_TYPE_ORIGINAL_HLT:
                raise ValueError(f"original HLT source_type must be {SOURCE_TYPE_ORIGINAL_HLT!r}")
            object.__setattr__(self, "reconstructor_architecture", None)
        else:
            if self.reconstructor_architecture is None:
                raise ValueError("set-reconstruction views require reconstructor_architecture")
            architecture = normalize_set_reconstructor_architecture(self.reconstructor_architecture)
            expected_name = view_name_for_reconstructor(architecture)
            if name != expected_name:
                raise ValueError(f"set-reconstruction view {name!r} should be named {expected_name!r}")
            source_type = self.source_type or SOURCE_TYPE_RECONSTRUCTED
            if source_type != SOURCE_TYPE_RECONSTRUCTED:
                raise ValueError(f"set-reconstruction source_type must be {SOURCE_TYPE_RECONSTRUCTED!r}")
            object.__setattr__(self, "reconstructor_architecture", architecture)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "view_kind", view_kind)
        object.__setattr__(self, "source_type", source_type)

    @property
    def implementation(self) -> str | None:
        if self.reconstructor_architecture is None:
            return None
        return SET_RECONSTRUCTOR_IMPLEMENTATIONS[self.reconstructor_architecture]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "view_kind": self.view_kind,
            "source_type": self.source_type,
            "reconstructor_architecture": self.reconstructor_architecture,
            "implementation": self.implementation,
        }


@dataclass(frozen=True)
class FiveViewGroupSpec:
    """Ordered set of views used by the primary multi-view tagger."""

    name: str
    view_names: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        view_names = tuple(normalize_view_name(name) for name in self.view_names)
        if not view_names:
            raise ValueError(f"Five-view group {self.name!r} must contain at least one view")
        if len(view_names) != len(set(view_names)):
            raise ValueError(f"Five-view group {self.name!r} contains duplicate views")
        object.__setattr__(self, "view_names", view_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "view_names": list(self.view_names),
            "n_views": len(self.view_names),
            "description": self.description,
        }


@dataclass(frozen=True)
class SetMatchingMultiViewLayout:
    """Path helper for the isolated set-matching multi-view namespace."""

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
    def normalization_dir(self) -> Path:
        return self.root / "normalization"

    @property
    def normalization_path(self) -> Path:
        return self.normalization_dir / "feature_normalization.json"

    @property
    def reconstructors_dir(self) -> Path:
        return self.root / "reconstructors"

    @property
    def reconstructed_views_dir(self) -> Path:
        return self.root / "reconstructed_views"

    @property
    def five_view_cache_dir(self) -> Path:
        return self.root / "five_view_cache"

    @property
    def taggers_dir(self) -> Path:
        return self.root / "taggers"

    @property
    def ablations_dir(self) -> Path:
        return self.root / "ablations"

    @property
    def audits_dir(self) -> Path:
        return self.root / "audits"

    @property
    def final_report_dir(self) -> Path:
        return self.root / "final_report"

    def reconstructor_dir(self, architecture: str) -> Path:
        return self.reconstructors_dir / normalize_set_reconstructor_architecture(architecture)

    def reconstructor_checkpoint(self, architecture: str) -> Path:
        return self.reconstructor_dir(architecture) / "best_model_val.pt"

    def reconstructed_view_dir(self, architecture: str) -> Path:
        return self.reconstructed_views_dir / normalize_set_reconstructor_architecture(architecture)

    def reconstructed_view_cache_path(self, architecture: str, split: str) -> Path:
        normalized_split = normalize_split_name(split)
        return self.reconstructed_view_dir(architecture) / f"{normalized_split}_reconstructed_view.npz"

    def five_view_cache_path(self, split: str) -> Path:
        normalized_split = normalize_split_name(split)
        return self.five_view_cache_dir / f"{normalized_split}_five_view.npz"

    def tagger_dir(self, name: str = "five_view_tagger") -> Path:
        return self.taggers_dir / str(name)

    def tagger_checkpoint(self, name: str = "five_view_tagger") -> Path:
        return self.tagger_dir(name) / "best_model_val.pt"

    def ablation_dir(self, name: str) -> Path:
        return self.ablations_dir / str(name)

    def to_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "split_manifest_dir": str(self.split_manifest_dir),
            "split_manifest_path": str(self.split_manifest_path),
            "hlt_cache_dir": str(self.hlt_cache_dir),
            "normalization_dir": str(self.normalization_dir),
            "normalization_path": str(self.normalization_path),
            "reconstructors_dir": str(self.reconstructors_dir),
            "reconstructed_views_dir": str(self.reconstructed_views_dir),
            "five_view_cache_dir": str(self.five_view_cache_dir),
            "taggers_dir": str(self.taggers_dir),
            "ablations_dir": str(self.ablations_dir),
            "audits_dir": str(self.audits_dir),
            "final_report_dir": str(self.final_report_dir),
        }


@dataclass(frozen=True)
class SetMatchingMultiViewConfig:
    """Configuration-only descriptor for the new multi-view experiment."""

    output_root: str | Path = "checkpoints"
    experiment_name: str = EXPERIMENT_NAME
    split_sizes: Mapping[str, int] = field(default_factory=lambda: dict(SPLIT_SIZES))
    reconstructors: tuple[str, ...] = SET_RECONSTRUCTOR_ARCHITECTURES
    views: tuple[str, ...] = VIEW_NAMES
    particle_feature_dim: int = DEFAULT_PARTICLE_FEATURE_DIM
    max_slots: int = DEFAULT_MAX_SLOTS
    max_tokens_per_view: int = DEFAULT_MAX_TOKENS_PER_VIEW
    min_tokens_per_view: int = DEFAULT_MIN_TOKENS_PER_VIEW
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD

    def __post_init__(self) -> None:
        split_sizes = {str(key): int(value) for key, value in self.split_sizes.items()}
        if tuple(split_sizes.keys()) != SPLIT_ORDER:
            raise ValueError(f"split_sizes keys must be exactly {SPLIT_ORDER} in order")
        if any(value <= 0 for value in split_sizes.values()):
            raise ValueError("split sizes must be positive")

        reconstructors = tuple(normalize_set_reconstructor_architecture(value) for value in self.reconstructors)
        if reconstructors != SET_RECONSTRUCTOR_ARCHITECTURES:
            raise ValueError(f"reconstructors must be {SET_RECONSTRUCTOR_ARCHITECTURES}")

        views = tuple(normalize_view_name(value) for value in self.views)
        if views != VIEW_NAMES:
            raise ValueError(f"views must be {VIEW_NAMES}")

        if int(self.particle_feature_dim) <= 0:
            raise ValueError("particle_feature_dim must be positive")
        if int(self.max_slots) <= 0:
            raise ValueError("max_slots must be positive")
        if int(self.max_tokens_per_view) <= 0:
            raise ValueError("max_tokens_per_view must be positive")
        if int(self.min_tokens_per_view) < 0:
            raise ValueError("min_tokens_per_view cannot be negative")
        if int(self.min_tokens_per_view) > int(self.max_tokens_per_view):
            raise ValueError("min_tokens_per_view cannot exceed max_tokens_per_view")
        confidence_threshold = float(self.confidence_threshold)
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        object.__setattr__(self, "split_sizes", split_sizes)
        object.__setattr__(self, "reconstructors", reconstructors)
        object.__setattr__(self, "views", views)
        object.__setattr__(self, "particle_feature_dim", int(self.particle_feature_dim))
        object.__setattr__(self, "max_slots", int(self.max_slots))
        object.__setattr__(self, "max_tokens_per_view", int(self.max_tokens_per_view))
        object.__setattr__(self, "min_tokens_per_view", int(self.min_tokens_per_view))
        object.__setattr__(self, "confidence_threshold", confidence_threshold)

    @property
    def layout(self) -> SetMatchingMultiViewLayout:
        return SetMatchingMultiViewLayout(output_root=self.output_root, experiment_name=self.experiment_name)

    @property
    def view_specs(self) -> tuple[SetMatchingViewSpec, ...]:
        return build_view_specs(self.reconstructors)

    @property
    def five_view_group(self) -> FiveViewGroupSpec:
        return build_five_view_group()

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_step": EXPERIMENT_STEP,
            "experiment_name": self.experiment_name,
            "split_sizes": dict(self.split_sizes),
            "reconstructors": list(self.reconstructors),
            "views": list(self.views),
            "view_specs": [spec.to_dict() for spec in self.view_specs],
            "five_view_group": self.five_view_group.to_dict(),
            "particle_feature_dim": self.particle_feature_dim,
            "max_slots": self.max_slots,
            "max_tokens_per_view": self.max_tokens_per_view,
            "min_tokens_per_view": self.min_tokens_per_view,
            "confidence_threshold": self.confidence_threshold,
            "layout": self.layout.to_dict(),
        }


def build_reconstructed_view_specs(
    reconstructors: Iterable[str] = SET_RECONSTRUCTOR_ARCHITECTURES,
) -> tuple[SetMatchingViewSpec, ...]:
    specs = []
    for architecture in reconstructors:
        normalized = normalize_set_reconstructor_architecture(architecture)
        specs.append(
            SetMatchingViewSpec(
                name=view_name_for_reconstructor(normalized),
                view_kind=VIEW_KIND_SET_RECONSTRUCTION,
                reconstructor_architecture=normalized,
            )
        )
    return tuple(specs)


def build_view_specs(
    reconstructors: Iterable[str] = SET_RECONSTRUCTOR_ARCHITECTURES,
) -> tuple[SetMatchingViewSpec, ...]:
    return (
        SetMatchingViewSpec(name=HLT_VIEW_NAME, view_kind=VIEW_KIND_ORIGINAL_HLT),
    ) + build_reconstructed_view_specs(reconstructors)


def build_five_view_group(
    view_names: Iterable[str] = VIEW_NAMES,
    *,
    name: str = FIVE_VIEW_GROUP_NAME,
) -> FiveViewGroupSpec:
    group = FiveViewGroupSpec(
        name=name,
        view_names=tuple(view_names),
        description="Original fixed HLT plus four set-matching reconstruction hypotheses.",
    )
    if group.view_names != VIEW_NAMES:
        raise ValueError(f"primary five-view group must use views {VIEW_NAMES}")
    return group


def default_set_matching_multiview_config() -> SetMatchingMultiViewConfig:
    return SetMatchingMultiViewConfig()


__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_MAX_SLOTS",
    "DEFAULT_MAX_TOKENS_PER_VIEW",
    "DEFAULT_MIN_TOKENS_PER_VIEW",
    "DEFAULT_PARTICLE_FEATURE_DIM",
    "EXPERIMENT_NAME",
    "EXPERIMENT_STEP",
    "FIVE_VIEW_GROUP_NAME",
    "HLT_VIEW_NAME",
    "RECONSTRUCTED_VIEW_NAMES",
    "SET_RECONSTRUCTOR_ARCHITECTURES",
    "SET_RECONSTRUCTOR_IMPLEMENTATIONS",
    "SOURCE_TYPE_ORIGINAL_HLT",
    "SOURCE_TYPE_RECONSTRUCTED",
    "SPLIT_ORDER",
    "SPLIT_SIZES",
    "VIEW_KIND_ORIGINAL_HLT",
    "VIEW_KIND_SET_RECONSTRUCTION",
    "VIEW_NAMES",
    "FiveViewGroupSpec",
    "SetMatchingMultiViewConfig",
    "SetMatchingMultiViewLayout",
    "SetMatchingViewSpec",
    "build_five_view_group",
    "build_reconstructed_view_specs",
    "build_view_specs",
    "default_set_matching_multiview_config",
    "normalize_set_reconstructor_architecture",
    "normalize_split_name",
    "normalize_view_name",
    "set_reconstructor_model_name",
    "view_name_for_reconstructor",
]
