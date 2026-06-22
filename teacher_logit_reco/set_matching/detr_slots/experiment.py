"""Configuration boundary for DETR/free-slot set reconstruction.

This module is intentionally config-only for Step 1.  It names the branch,
defines the four encoder families, and establishes the view/path layout without
importing losses, training loops, or model code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM


DETR_SLOT_EXPERIMENT_NAME = "detr_slot_multiview"
DETR_SLOT_EXPERIMENT_STEP = "detr_free_slot_step1_namespace"
DETR_SLOT_RECONSTRUCTOR_CONTRACT = "hlt_to_free_slot_offline_set_v1"

DETR_SLOT_SPLIT_ORDER: tuple[str, ...] = (
    "model_train",
    "model_val",
    "stack_train",
    "stack_val",
    "final_test",
)
DETR_SLOT_SPLIT_SIZES: dict[str, int] = {
    "model_train": 500_000,
    "model_val": 150_000,
    "stack_train": 500_000,
    "stack_val": 150_000,
    "final_test": 500_000,
}

DETR_SLOT_ENCODER_ARCHITECTURES: tuple[str, ...] = ("gt", "pn", "pfn", "pcnn")
DETR_SLOT_ENCODER_IMPLEMENTATIONS: dict[str, str] = {
    "gt": "detr_slot_global_transformer_encoder",
    "pn": "detr_slot_particle_net_encoder",
    "pfn": "detr_slot_particle_flow_encoder",
    "pcnn": "detr_slot_particle_cnn_encoder",
}
DETR_SLOT_ENCODER_ALIASES: dict[str, str] = {
    "global": "gt",
    "global_transformer": "gt",
    "globaltransformer": "gt",
    "gt_reco": "gt",
    "part": "gt",
    "part_reco": "gt",
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

DETR_SLOT_HLT_VIEW_NAME = "hlt"
DETR_SLOT_RECONSTRUCTED_VIEW_NAMES: tuple[str, ...] = (
    "detr_gt",
    "detr_pn",
    "detr_pfn",
    "detr_pcnn",
)
DETR_SLOT_VIEW_NAMES: tuple[str, ...] = (DETR_SLOT_HLT_VIEW_NAME,) + DETR_SLOT_RECONSTRUCTED_VIEW_NAMES
DETR_SLOT_FIVE_VIEW_GROUP_NAME = "hlt_plus_four_detr_slot_recos"

DETR_SLOT_VIEW_KIND_ORIGINAL_HLT = "original_hlt"
DETR_SLOT_VIEW_KIND_RECONSTRUCTED = "detr_slot_reconstruction"
DETR_SLOT_VIEW_KINDS: tuple[str, ...] = (
    DETR_SLOT_VIEW_KIND_ORIGINAL_HLT,
    DETR_SLOT_VIEW_KIND_RECONSTRUCTED,
)
DETR_SLOT_SOURCE_TYPE_ORIGINAL_HLT = "original_hlt"
DETR_SLOT_SOURCE_TYPE_RECONSTRUCTED = "detr_slot_reconstructed"

DETR_SLOT_DEFAULT_PARTICLE_FEATURE_DIM = RAW_TOKEN_DIM
DETR_SLOT_DEFAULT_NUM_SLOTS = 160
DETR_SLOT_DEFAULT_EXPORT_MAX_TOKENS = 128
DETR_SLOT_DEFAULT_MIN_TOKENS_PER_VIEW = 8
DETR_SLOT_DEFAULT_CONFIDENCE_THRESHOLD = 0.05
DETR_SLOT_DEFAULT_EMBED_DIM = 128
DETR_SLOT_DEFAULT_DECODER_LAYERS = 4
DETR_SLOT_DEFAULT_DECODER_HEADS = 4


def _alias_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def normalize_detr_slot_split_name(value: str) -> str:
    split = str(value).strip()
    if split not in DETR_SLOT_SPLIT_ORDER:
        raise ValueError(f"Unknown DETR-slot split {value!r}; expected one of {DETR_SLOT_SPLIT_ORDER}")
    return split


def normalize_detr_slot_encoder_architecture(value: str) -> str:
    text = str(value).strip()
    key = _alias_key(text)
    normalized = DETR_SLOT_ENCODER_ALIASES.get(text, DETR_SLOT_ENCODER_ALIASES.get(key, key))
    if normalized not in DETR_SLOT_ENCODER_ARCHITECTURES:
        raise ValueError(
            f"Unknown DETR-slot encoder architecture {value!r}; "
            f"expected one of {DETR_SLOT_ENCODER_ARCHITECTURES}"
        )
    return normalized


def detr_slot_view_name_for_encoder(architecture: str) -> str:
    return f"detr_{normalize_detr_slot_encoder_architecture(architecture)}"


def detr_slot_model_name(architecture: str) -> str:
    return f"detr_slot_{normalize_detr_slot_encoder_architecture(architecture)}_reco"


def normalize_detr_slot_view_name(value: str) -> str:
    text = str(value).strip()
    key = _alias_key(text)
    if key in {"hlt", "hlt_view", "original_hlt", "fixed_hlt", "fixed_hlt_view"}:
        return DETR_SLOT_HLT_VIEW_NAME
    if key in DETR_SLOT_RECONSTRUCTED_VIEW_NAMES:
        return key
    if key.startswith("detr_"):
        architecture = key.removeprefix("detr_")
        return detr_slot_view_name_for_encoder(architecture)
    if key in DETR_SLOT_ENCODER_ARCHITECTURES:
        return detr_slot_view_name_for_encoder(key)
    raise ValueError(f"Unknown DETR-slot view name {value!r}; expected one of {DETR_SLOT_VIEW_NAMES}")


@dataclass(frozen=True)
class DetrSlotViewSpec:
    """One view in the DETR/free-slot multi-view branch."""

    name: str
    view_kind: str
    encoder_architecture: str | None = None
    source_type: str | None = None

    def __post_init__(self) -> None:
        view_kind = str(self.view_kind).strip()
        if view_kind not in DETR_SLOT_VIEW_KINDS:
            raise ValueError(f"view_kind must be one of {DETR_SLOT_VIEW_KINDS}")
        name = normalize_detr_slot_view_name(self.name)

        if view_kind == DETR_SLOT_VIEW_KIND_ORIGINAL_HLT:
            if name != DETR_SLOT_HLT_VIEW_NAME:
                raise ValueError(f"original HLT view must be named {DETR_SLOT_HLT_VIEW_NAME!r}")
            if self.encoder_architecture is not None:
                raise ValueError("original HLT view cannot declare a DETR encoder architecture")
            source_type = self.source_type or DETR_SLOT_SOURCE_TYPE_ORIGINAL_HLT
            if source_type != DETR_SLOT_SOURCE_TYPE_ORIGINAL_HLT:
                raise ValueError(
                    f"original HLT source_type must be {DETR_SLOT_SOURCE_TYPE_ORIGINAL_HLT!r}"
                )
            object.__setattr__(self, "encoder_architecture", None)
        else:
            if self.encoder_architecture is None:
                raise ValueError("DETR reconstructed views require encoder_architecture")
            architecture = normalize_detr_slot_encoder_architecture(self.encoder_architecture)
            expected_name = detr_slot_view_name_for_encoder(architecture)
            if name != expected_name:
                raise ValueError(f"DETR reconstructed view {name!r} should be named {expected_name!r}")
            source_type = self.source_type or DETR_SLOT_SOURCE_TYPE_RECONSTRUCTED
            if source_type != DETR_SLOT_SOURCE_TYPE_RECONSTRUCTED:
                raise ValueError(
                    f"DETR reconstructed source_type must be {DETR_SLOT_SOURCE_TYPE_RECONSTRUCTED!r}"
                )
            object.__setattr__(self, "encoder_architecture", architecture)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "view_kind", view_kind)
        object.__setattr__(self, "source_type", source_type)

    @property
    def implementation(self) -> str | None:
        if self.encoder_architecture is None:
            return None
        return DETR_SLOT_ENCODER_IMPLEMENTATIONS[self.encoder_architecture]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "view_kind": self.view_kind,
            "source_type": self.source_type,
            "encoder_architecture": self.encoder_architecture,
            "implementation": self.implementation,
        }


@dataclass(frozen=True)
class DetrSlotFiveViewGroupSpec:
    """Ordered set of views used by the primary DETR multi-view tagger."""

    name: str
    view_names: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        view_names = tuple(normalize_detr_slot_view_name(name) for name in self.view_names)
        if not view_names:
            raise ValueError(f"DETR view group {self.name!r} must contain at least one view")
        if len(view_names) != len(set(view_names)):
            raise ValueError(f"DETR view group {self.name!r} contains duplicate views")
        object.__setattr__(self, "view_names", view_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "view_names": list(self.view_names),
            "n_views": len(self.view_names),
            "description": self.description,
        }


@dataclass(frozen=True)
class DetrSlotExperimentLayout:
    """Path helper for the isolated DETR/free-slot namespace."""

    output_root: str | Path = "checkpoints"
    experiment_name: str = DETR_SLOT_EXPERIMENT_NAME

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
    def reconstructors_dir(self) -> Path:
        return self.root / "detr_slot_reconstructors"

    def reconstructor_dir(self, architecture: str) -> Path:
        return self.reconstructors_dir / normalize_detr_slot_encoder_architecture(architecture)

    @property
    def reconstructed_views_dir(self) -> Path:
        return self.root / "detr_slot_reconstructed_views"

    def reconstructed_view_dir(self, architecture: str) -> Path:
        return self.reconstructed_views_dir / normalize_detr_slot_encoder_architecture(architecture)

    @property
    def taggers_dir(self) -> Path:
        return self.root / "taggers"

    @property
    def ablations_dir(self) -> Path:
        return self.root / "ablations"

    @property
    def final_report_dir(self) -> Path:
        return self.root / "final_report"

    def to_dict(self) -> dict[str, str]:
        return {
            "root": self.root.as_posix(),
            "split_manifest_path": self.split_manifest_path.as_posix(),
            "hlt_cache_dir": self.hlt_cache_dir.as_posix(),
            "reconstructors_dir": self.reconstructors_dir.as_posix(),
            "reconstructed_views_dir": self.reconstructed_views_dir.as_posix(),
            "taggers_dir": self.taggers_dir.as_posix(),
            "ablations_dir": self.ablations_dir.as_posix(),
            "final_report_dir": self.final_report_dir.as_posix(),
        }


@dataclass(frozen=True)
class DetrSlotExperimentConfig:
    """Configuration defaults for the DETR/free-slot branch."""

    experiment_name: str = DETR_SLOT_EXPERIMENT_NAME
    split_sizes: Mapping[str, int] = field(default_factory=lambda: dict(DETR_SLOT_SPLIT_SIZES))
    encoder_architectures: tuple[str, ...] = DETR_SLOT_ENCODER_ARCHITECTURES
    particle_feature_dim: int = DETR_SLOT_DEFAULT_PARTICLE_FEATURE_DIM
    num_slots: int = DETR_SLOT_DEFAULT_NUM_SLOTS
    export_max_tokens: int = DETR_SLOT_DEFAULT_EXPORT_MAX_TOKENS
    min_tokens_per_view: int = DETR_SLOT_DEFAULT_MIN_TOKENS_PER_VIEW
    confidence_threshold: float = DETR_SLOT_DEFAULT_CONFIDENCE_THRESHOLD
    embed_dim: int = DETR_SLOT_DEFAULT_EMBED_DIM
    decoder_layers: int = DETR_SLOT_DEFAULT_DECODER_LAYERS
    decoder_heads: int = DETR_SLOT_DEFAULT_DECODER_HEADS

    def __post_init__(self) -> None:
        if not str(self.experiment_name).strip():
            raise ValueError("experiment_name must be non-empty")
        split_sizes = {normalize_detr_slot_split_name(k): int(v) for k, v in self.split_sizes.items()}
        missing = [split for split in DETR_SLOT_SPLIT_ORDER if split not in split_sizes]
        extra = [split for split in split_sizes if split not in DETR_SLOT_SPLIT_ORDER]
        if missing or extra:
            raise ValueError(f"split_sizes must contain exactly {DETR_SLOT_SPLIT_ORDER}")
        for split, size in split_sizes.items():
            if int(size) <= 0:
                raise ValueError(f"split size for {split} must be positive")
        architectures = tuple(normalize_detr_slot_encoder_architecture(a) for a in self.encoder_architectures)
        if not architectures:
            raise ValueError("encoder_architectures must contain at least one architecture")
        if len(architectures) != len(set(architectures)):
            raise ValueError(f"duplicate encoder architectures are not allowed: {architectures}")
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if self.export_max_tokens <= 0:
            raise ValueError("export_max_tokens must be positive")
        if self.export_max_tokens > self.num_slots:
            raise ValueError("export_max_tokens cannot exceed num_slots")
        if int(self.min_tokens_per_view) < 0:
            raise ValueError("min_tokens_per_view must be non-negative")
        if int(self.min_tokens_per_view) > int(self.export_max_tokens):
            raise ValueError("min_tokens_per_view cannot exceed export_max_tokens")
        if float(self.confidence_threshold) < 0.0 or float(self.confidence_threshold) > 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.particle_feature_dim <= 0:
            raise ValueError("particle_feature_dim must be positive")
        if int(self.embed_dim) <= 0:
            raise ValueError("embed_dim must be positive")
        if int(self.decoder_layers) <= 0:
            raise ValueError("decoder_layers must be positive")
        if int(self.decoder_heads) <= 0:
            raise ValueError("decoder_heads must be positive")
        if int(self.embed_dim) % int(self.decoder_heads) != 0:
            raise ValueError("embed_dim must be divisible by decoder_heads")
        object.__setattr__(self, "split_sizes", split_sizes)
        object.__setattr__(self, "encoder_architectures", architectures)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "split_sizes": {split: int(self.split_sizes[split]) for split in DETR_SLOT_SPLIT_ORDER},
            "encoder_architectures": list(self.encoder_architectures),
            "particle_feature_dim": int(self.particle_feature_dim),
            "num_slots": int(self.num_slots),
            "export_max_tokens": int(self.export_max_tokens),
            "min_tokens_per_view": int(self.min_tokens_per_view),
            "confidence_threshold": float(self.confidence_threshold),
            "embed_dim": int(self.embed_dim),
            "decoder_layers": int(self.decoder_layers),
            "decoder_heads": int(self.decoder_heads),
            "contract": DETR_SLOT_RECONSTRUCTOR_CONTRACT,
        }


def build_detr_slot_view_specs() -> tuple[DetrSlotViewSpec, ...]:
    specs: list[DetrSlotViewSpec] = [
        DetrSlotViewSpec(
            name=DETR_SLOT_HLT_VIEW_NAME,
            view_kind=DETR_SLOT_VIEW_KIND_ORIGINAL_HLT,
        )
    ]
    for architecture in DETR_SLOT_ENCODER_ARCHITECTURES:
        specs.append(
            DetrSlotViewSpec(
                name=detr_slot_view_name_for_encoder(architecture),
                view_kind=DETR_SLOT_VIEW_KIND_RECONSTRUCTED,
                encoder_architecture=architecture,
            )
        )
    return tuple(specs)


def build_detr_slot_five_view_group() -> DetrSlotFiveViewGroupSpec:
    return DetrSlotFiveViewGroupSpec(
        name=DETR_SLOT_FIVE_VIEW_GROUP_NAME,
        view_names=DETR_SLOT_VIEW_NAMES,
        description="Original HLT view plus four DETR/free-slot reconstructed views.",
    )


def default_detr_slot_experiment_config() -> DetrSlotExperimentConfig:
    return DetrSlotExperimentConfig()
