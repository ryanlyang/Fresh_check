"""Configuration boundary for reliability-gated subtoken Particle Transformers.

Step 1 is intentionally config-only.  It names the branch, records the HLT-only
and privileged-training versions, defines the raw-token modality groups, and
establishes validation-heavy dataclasses for later model/training code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM, SPLIT_ORDER
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES


SUBTOKEN_PART_EXPERIMENT_NAME = "reliability_gated_subtoken_part"
SUBTOKEN_PART_EXPERIMENT_STEP = "subtoken_part_step1_config"
SUBTOKEN_PART_CONTRACT = "hlt_to_reliability_gated_subtoken_part_v1"

SUBTOKEN_PART_VERSION_A = "hlt_only"
SUBTOKEN_PART_VERSION_B = "privileged_offline"
SUBTOKEN_PART_VERSIONS: tuple[str, ...] = (SUBTOKEN_PART_VERSION_A, SUBTOKEN_PART_VERSION_B)
SUBTOKEN_PART_VERSION_ALIASES: dict[str, str] = {
    "a": SUBTOKEN_PART_VERSION_A,
    "version_a": SUBTOKEN_PART_VERSION_A,
    "hlt": SUBTOKEN_PART_VERSION_A,
    "hlt_only": SUBTOKEN_PART_VERSION_A,
    "supervised": SUBTOKEN_PART_VERSION_A,
    "b": SUBTOKEN_PART_VERSION_B,
    "version_b": SUBTOKEN_PART_VERSION_B,
    "privileged": SUBTOKEN_PART_VERSION_B,
    "privileged_offline": SUBTOKEN_PART_VERSION_B,
    "offline_privileged": SUBTOKEN_PART_VERSION_B,
}

SUBTOKEN_PART_SPLIT_ORDER: tuple[str, ...] = tuple(SPLIT_ORDER)
SUBTOKEN_PART_SPLIT_SIZES: dict[str, int] = {
    "model_train": 500_000,
    "model_val": 150_000,
    "stack_train": 500_000,
    "stack_val": 150_000,
    "final_test": 500_000,
}

SUBTOKEN_PART_RAW_FEATURE_NAMES: tuple[str, ...] = (
    "pt",
    "eta",
    "phi",
    "energy",
    "charge",
    "isChargedHadron",
    "isNeutralHadron",
    "isPhoton",
    "isElectron",
    "isMuon",
    "d0",
    "d0err",
    "dz",
    "dzerr",
)
SUBTOKEN_PART_PARTICLE_FEATURE_DIM = len(PF_FEATURE_NAMES)

SUBTOKEN_MODALITY_KINEMATICS = "kinematics"
SUBTOKEN_MODALITY_IDENTITY = "identity"
SUBTOKEN_MODALITY_TRACK = "track"
SUBTOKEN_PART_MODALITIES: tuple[str, ...] = (
    SUBTOKEN_MODALITY_KINEMATICS,
    SUBTOKEN_MODALITY_IDENTITY,
    SUBTOKEN_MODALITY_TRACK,
)

SUBTOKEN_PART_GATE_NONE = "none"
SUBTOKEN_PART_GATE_LOCAL_SOFTMAX = "local_softmax"
SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX = "context_softmax"
SUBTOKEN_PART_GATE_CONTEXT_SIGMOID = "context_sigmoid"
SUBTOKEN_PART_GATE_MODES: tuple[str, ...] = (
    SUBTOKEN_PART_GATE_NONE,
    SUBTOKEN_PART_GATE_LOCAL_SOFTMAX,
    SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX,
    SUBTOKEN_PART_GATE_CONTEXT_SIGMOID,
)

SUBTOKEN_PART_POOL_MEAN = "mean"
SUBTOKEN_PART_POOL_LEARNED_QUERY = "learned_query"
SUBTOKEN_PART_POOL_CLS_TOKEN = "cls_token"
SUBTOKEN_PART_POOL_MODES: tuple[str, ...] = (
    SUBTOKEN_PART_POOL_MEAN,
    SUBTOKEN_PART_POOL_LEARNED_QUERY,
    SUBTOKEN_PART_POOL_CLS_TOKEN,
)

SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE = "hlt_part_baseline"
SUBTOKEN_PART_VARIANT_NO_GATE = "subtoken_no_gate"
SUBTOKEN_PART_VARIANT_LOCAL_GATE = "subtoken_gate_local_only"
SUBTOKEN_PART_VARIANT_CONTEXT_GATE = "subtoken_gate_context"
SUBTOKEN_PART_VARIANT_DUAL_CROSS_ATTENTION = "dual_part_subtoken_cross_attention"
SUBTOKEN_PART_VARIANT_SCALAR_LOCAL = "scalar_token_local_only"
SUBTOKEN_PART_VARIANTS: tuple[str, ...] = (
    SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE,
    SUBTOKEN_PART_VARIANT_NO_GATE,
    SUBTOKEN_PART_VARIANT_LOCAL_GATE,
    SUBTOKEN_PART_VARIANT_CONTEXT_GATE,
    SUBTOKEN_PART_VARIANT_DUAL_CROSS_ATTENTION,
    SUBTOKEN_PART_VARIANT_SCALAR_LOCAL,
)
SUBTOKEN_PART_VARIANT_ALIASES: dict[str, str] = {
    "part": SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE,
    "hlt_part": SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE,
    "baseline": SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE,
    "no_gate": SUBTOKEN_PART_VARIANT_NO_GATE,
    "subtoken": SUBTOKEN_PART_VARIANT_NO_GATE,
    "local_gate": SUBTOKEN_PART_VARIANT_LOCAL_GATE,
    "gate_local": SUBTOKEN_PART_VARIANT_LOCAL_GATE,
    "context_gate": SUBTOKEN_PART_VARIANT_CONTEXT_GATE,
    "gate_context": SUBTOKEN_PART_VARIANT_CONTEXT_GATE,
    "full": SUBTOKEN_PART_VARIANT_CONTEXT_GATE,
    "dual": SUBTOKEN_PART_VARIANT_DUAL_CROSS_ATTENTION,
    "crossvit": SUBTOKEN_PART_VARIANT_DUAL_CROSS_ATTENTION,
    "scalar": SUBTOKEN_PART_VARIANT_SCALAR_LOCAL,
    "scalar_local": SUBTOKEN_PART_VARIANT_SCALAR_LOCAL,
}

SUBTOKEN_PART_DEFAULT_EMBED_DIM = 128
SUBTOKEN_PART_DEFAULT_LOCAL_LAYERS = 1
SUBTOKEN_PART_DEFAULT_LOCAL_HEADS = 4
SUBTOKEN_PART_DEFAULT_CONTEXT_LAYERS = 2
SUBTOKEN_PART_DEFAULT_CONTEXT_HEADS = 4
SUBTOKEN_PART_DEFAULT_GLOBAL_LAYERS = 6
SUBTOKEN_PART_DEFAULT_GLOBAL_HEADS = 8
SUBTOKEN_PART_DEFAULT_DROPOUT = 0.05
SUBTOKEN_PART_DEFAULT_ATTENTION_DROPOUT = 0.05


def _alias_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def normalize_subtoken_part_version(value: str) -> str:
    key = _alias_key(value)
    normalized = SUBTOKEN_PART_VERSION_ALIASES.get(key, key)
    if normalized not in SUBTOKEN_PART_VERSIONS:
        raise ValueError(f"Unknown subtoken ParT version {value!r}; expected one of {SUBTOKEN_PART_VERSIONS}")
    return normalized


def normalize_subtoken_part_variant(value: str) -> str:
    key = _alias_key(value)
    normalized = SUBTOKEN_PART_VARIANT_ALIASES.get(key, key)
    if normalized not in SUBTOKEN_PART_VARIANTS:
        raise ValueError(f"Unknown subtoken ParT variant {value!r}; expected one of {SUBTOKEN_PART_VARIANTS}")
    return normalized


def normalize_subtoken_gate_mode(value: str) -> str:
    key = _alias_key(value)
    if key in {"off", "disabled", "disable"}:
        key = SUBTOKEN_PART_GATE_NONE
    if key not in SUBTOKEN_PART_GATE_MODES:
        raise ValueError(f"Unknown subtoken gate mode {value!r}; expected one of {SUBTOKEN_PART_GATE_MODES}")
    return key


def normalize_subtoken_pool_mode(value: str) -> str:
    key = _alias_key(value)
    if key in {"avg", "average", "mean_pool", "mean_pooling"}:
        key = SUBTOKEN_PART_POOL_MEAN
    if key in {"attention", "attn", "learned", "query", "query_attention", "learned_query_attention"}:
        key = SUBTOKEN_PART_POOL_LEARNED_QUERY
    if key in {"cls", "class", "class_token", "summary_token"}:
        key = SUBTOKEN_PART_POOL_CLS_TOKEN
    if key not in SUBTOKEN_PART_POOL_MODES:
        raise ValueError(f"Unknown subtoken pool mode {value!r}; expected one of {SUBTOKEN_PART_POOL_MODES}")
    return key


def normalize_subtoken_split_name(value: str) -> str:
    split = str(value).strip()
    if split not in SUBTOKEN_PART_SPLIT_ORDER:
        raise ValueError(f"Unknown subtoken split {value!r}; expected one of {SUBTOKEN_PART_SPLIT_ORDER}")
    return split


def _validate_probability(value: float, *, name: str) -> float:
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _validate_attention_heads(embed_dim: int, heads: int, *, name: str) -> None:
    if int(heads) <= 0:
        raise ValueError(f"{name} must be positive")
    if int(embed_dim) % int(heads) != 0:
        raise ValueError(f"embed_dim must be divisible by {name}")


@dataclass(frozen=True)
class SubtokenModalitySpec:
    """One physical feature group inside a particle."""

    name: str
    raw_indices: tuple[int, ...]
    description: str = ""

    def __post_init__(self) -> None:
        name = _alias_key(self.name)
        if not name:
            raise ValueError("modality name must be non-empty")
        raw_indices = tuple(int(index) for index in self.raw_indices)
        if not raw_indices:
            raise ValueError(f"modality {name!r} must contain at least one raw feature index")
        if len(raw_indices) != len(set(raw_indices)):
            raise ValueError(f"modality {name!r} contains duplicate raw feature indices")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "raw_indices", raw_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "raw_indices": list(self.raw_indices),
            "raw_feature_names": [SUBTOKEN_PART_RAW_FEATURE_NAMES[index] for index in self.raw_indices],
            "description": self.description,
        }


def default_subtoken_modality_specs() -> tuple[SubtokenModalitySpec, ...]:
    return (
        SubtokenModalitySpec(
            name=SUBTOKEN_MODALITY_KINEMATICS,
            raw_indices=(0, 1, 2, 3),
            description="pt, eta, phi, energy",
        ),
        SubtokenModalitySpec(
            name=SUBTOKEN_MODALITY_IDENTITY,
            raw_indices=(4, 5, 6, 7, 8, 9),
            description="charge and PID flags",
        ),
        SubtokenModalitySpec(
            name=SUBTOKEN_MODALITY_TRACK,
            raw_indices=(10, 11, 12, 13),
            description="impact parameter and track-quality-like features",
        ),
    )


@dataclass(frozen=True)
class SubtokenFeatureConfig:
    """Feature grouping contract for the subtoken branch."""

    raw_token_dim: int = RAW_TOKEN_DIM
    particle_feature_dim: int = SUBTOKEN_PART_PARTICLE_FEATURE_DIM
    modalities: tuple[SubtokenModalitySpec, ...] = field(default_factory=default_subtoken_modality_specs)
    include_derived_kinematics: bool | None = None
    include_part_style_derived_features: bool | None = None
    anchor_source: str = "raw"
    require_raw_index_partition: bool = True

    def __post_init__(self) -> None:
        if int(self.raw_token_dim) <= 0:
            raise ValueError("raw_token_dim must be positive")
        if int(self.particle_feature_dim) <= 0:
            raise ValueError("particle_feature_dim must be positive")
        modalities = tuple(
            modality if isinstance(modality, SubtokenModalitySpec) else SubtokenModalitySpec(**modality)
            for modality in self.modalities
        )
        if not modalities:
            raise ValueError("modalities must contain at least one modality")
        names = [modality.name for modality in modalities]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate modality names are not allowed: {names}")
        all_indices: list[int] = []
        for modality in modalities:
            for index in modality.raw_indices:
                if index < 0 or index >= int(self.raw_token_dim):
                    raise ValueError(
                        f"raw feature index {index} in modality {modality.name!r} is outside "
                        f"[0, {int(self.raw_token_dim)})"
                    )
                all_indices.append(index)
        if len(all_indices) != len(set(all_indices)):
            raise ValueError("raw feature indices cannot appear in multiple modalities")
        if self.require_raw_index_partition:
            expected = set(range(int(self.raw_token_dim)))
            actual = set(all_indices)
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise ValueError(f"modalities must partition raw features; missing={missing}, extra={extra}")
        include_part_style = self.include_part_style_derived_features
        include_derived_kinematics = self.include_derived_kinematics
        if include_part_style is None and include_derived_kinematics is None:
            include_part_style = True
        elif include_part_style is None:
            include_part_style = bool(include_derived_kinematics)
        elif include_derived_kinematics is not None and bool(include_part_style) != bool(include_derived_kinematics):
            raise ValueError(
                "include_part_style_derived_features and legacy include_derived_kinematics cannot disagree"
            )
        include_part_style = bool(include_part_style)
        anchor_source = _alias_key(self.anchor_source)
        if anchor_source not in {"raw", "part_features", "raw_and_part_features"}:
            raise ValueError("anchor_source must be one of raw, part_features, raw_and_part_features")
        if anchor_source in {"part_features", "raw_and_part_features"} and not include_part_style:
            raise ValueError(f"anchor_source={anchor_source!r} requires include_part_style_derived_features=True")
        object.__setattr__(self, "modalities", modalities)
        object.__setattr__(self, "include_part_style_derived_features", include_part_style)
        object.__setattr__(self, "include_derived_kinematics", include_part_style)
        object.__setattr__(self, "anchor_source", anchor_source)

    @property
    def num_modalities(self) -> int:
        return len(self.modalities)

    @property
    def modality_names(self) -> tuple[str, ...]:
        return tuple(modality.name for modality in self.modalities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_token_dim": int(self.raw_token_dim),
            "particle_feature_dim": int(self.particle_feature_dim),
            "modalities": [modality.to_dict() for modality in self.modalities],
            "num_modalities": self.num_modalities,
            "include_part_style_derived_features": bool(self.include_part_style_derived_features),
            "include_derived_kinematics": bool(self.include_derived_kinematics),
            "anchor_source": self.anchor_source,
            "require_raw_index_partition": bool(self.require_raw_index_partition),
        }


@dataclass(frozen=True)
class SubtokenPartConfig:
    """Model-shape configuration for a reliability-gated subtoken ParT."""

    num_classes: int
    feature_config: SubtokenFeatureConfig = field(default_factory=SubtokenFeatureConfig)
    variant: str = SUBTOKEN_PART_VARIANT_CONTEXT_GATE
    version: str = SUBTOKEN_PART_VERSION_A
    embed_dim: int = SUBTOKEN_PART_DEFAULT_EMBED_DIM
    local_layers: int = SUBTOKEN_PART_DEFAULT_LOCAL_LAYERS
    local_heads: int = SUBTOKEN_PART_DEFAULT_LOCAL_HEADS
    context_layers: int = SUBTOKEN_PART_DEFAULT_CONTEXT_LAYERS
    context_heads: int = SUBTOKEN_PART_DEFAULT_CONTEXT_HEADS
    global_layers: int = SUBTOKEN_PART_DEFAULT_GLOBAL_LAYERS
    global_heads: int = SUBTOKEN_PART_DEFAULT_GLOBAL_HEADS
    gate_mode: str = SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX
    local_pool_mode: str = SUBTOKEN_PART_POOL_LEARNED_QUERY
    use_pairwise_bias: bool = True
    use_particle_anchor: bool = True
    use_modality_type_embeddings: bool = True
    use_pt_rank_embedding: bool = False
    modality_dropout: float = 0.0
    dropout: float = SUBTOKEN_PART_DEFAULT_DROPOUT
    attention_dropout: float = SUBTOKEN_PART_DEFAULT_ATTENTION_DROPOUT

    def __post_init__(self) -> None:
        if int(self.num_classes) <= 1:
            raise ValueError("num_classes must be greater than one")
        feature_config = (
            self.feature_config
            if isinstance(self.feature_config, SubtokenFeatureConfig)
            else SubtokenFeatureConfig(**self.feature_config)
        )
        variant = normalize_subtoken_part_variant(self.variant)
        version = normalize_subtoken_part_version(self.version)
        gate_mode = normalize_subtoken_gate_mode(self.gate_mode)
        local_pool_mode = normalize_subtoken_pool_mode(self.local_pool_mode)
        if int(self.embed_dim) <= 0:
            raise ValueError("embed_dim must be positive")
        for name in ("local_layers", "context_layers", "global_layers"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        _validate_attention_heads(self.embed_dim, self.local_heads, name="local_heads")
        _validate_attention_heads(self.embed_dim, self.context_heads, name="context_heads")
        _validate_attention_heads(self.embed_dim, self.global_heads, name="global_heads")
        modality_dropout = _validate_probability(self.modality_dropout, name="modality_dropout")
        if modality_dropout >= 1.0:
            raise ValueError("modality_dropout must be less than 1.0")
        dropout = _validate_probability(self.dropout, name="dropout")
        attention_dropout = _validate_probability(self.attention_dropout, name="attention_dropout")
        if variant == SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE and gate_mode != SUBTOKEN_PART_GATE_NONE:
            raise ValueError("hlt_part_baseline must use gate_mode='none'")
        if gate_mode == SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX and int(self.context_layers) <= 0:
            raise ValueError("context_softmax gate mode requires context_layers > 0")
        object.__setattr__(self, "feature_config", feature_config)
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "gate_mode", gate_mode)
        object.__setattr__(self, "local_pool_mode", local_pool_mode)
        object.__setattr__(self, "modality_dropout", modality_dropout)
        object.__setattr__(self, "dropout", dropout)
        object.__setattr__(self, "attention_dropout", attention_dropout)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": SUBTOKEN_PART_CONTRACT,
            "num_classes": int(self.num_classes),
            "variant": self.variant,
            "version": self.version,
            "feature_config": self.feature_config.to_dict(),
            "embed_dim": int(self.embed_dim),
            "local_layers": int(self.local_layers),
            "local_heads": int(self.local_heads),
            "context_layers": int(self.context_layers),
            "context_heads": int(self.context_heads),
            "global_layers": int(self.global_layers),
            "global_heads": int(self.global_heads),
            "gate_mode": self.gate_mode,
            "local_pool_mode": self.local_pool_mode,
            "use_pairwise_bias": bool(self.use_pairwise_bias),
            "use_particle_anchor": bool(self.use_particle_anchor),
            "use_modality_type_embeddings": bool(self.use_modality_type_embeddings),
            "use_pt_rank_embedding": bool(self.use_pt_rank_embedding),
            "modality_dropout": float(self.modality_dropout),
            "dropout": float(self.dropout),
            "attention_dropout": float(self.attention_dropout),
        }


@dataclass(frozen=True)
class SubtokenTrainingConfig:
    """Training configuration shared by Version A and Version B."""

    version: str = SUBTOKEN_PART_VERSION_A
    train_split: str = "model_train"
    val_split: str = "model_val"
    final_test_split: str = "final_test"
    split_sizes: Mapping[str, int] = field(default_factory=lambda: dict(SUBTOKEN_PART_SPLIT_SIZES))
    seed: int = 2607
    epochs: int = 45
    batch_size: int = 64
    eval_batch_size: int = 128
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    early_stop_patience: int = 6
    grad_clip_norm: float = 1.0
    selection_metric: str = "fpr_at_signal_eff_0p50"
    teacher_distill_weight: float = 0.0
    residual_weight: float = 0.0
    masked_subtoken_weight: float = 0.0
    gate_entropy_weight: float = 0.0

    def __post_init__(self) -> None:
        version = normalize_subtoken_part_version(self.version)
        train_split = normalize_subtoken_split_name(self.train_split)
        val_split = normalize_subtoken_split_name(self.val_split)
        final_test_split = normalize_subtoken_split_name(self.final_test_split)
        split_sizes = {normalize_subtoken_split_name(k): int(v) for k, v in self.split_sizes.items()}
        missing = [split for split in SUBTOKEN_PART_SPLIT_ORDER if split not in split_sizes]
        extra = [split for split in split_sizes if split not in SUBTOKEN_PART_SPLIT_ORDER]
        if missing or extra:
            raise ValueError(f"split_sizes must contain exactly {SUBTOKEN_PART_SPLIT_ORDER}")
        for split, size in split_sizes.items():
            if size <= 0:
                raise ValueError(f"split size for {split} must be positive")
        for name in ("seed", "epochs", "batch_size", "eval_batch_size", "early_stop_patience"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if float(self.grad_clip_norm) <= 0.0:
            raise ValueError("grad_clip_norm must be positive")
        weights = {
            "teacher_distill_weight": float(self.teacher_distill_weight),
            "residual_weight": float(self.residual_weight),
            "masked_subtoken_weight": float(self.masked_subtoken_weight),
            "gate_entropy_weight": float(self.gate_entropy_weight),
        }
        for name, value in weights.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        privileged_weight = (
            weights["teacher_distill_weight"] + weights["residual_weight"] + weights["masked_subtoken_weight"]
        )
        if version == SUBTOKEN_PART_VERSION_A and privileged_weight > 0.0:
            raise ValueError("Version A is HLT-only and cannot enable offline privileged loss weights")
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "train_split", train_split)
        object.__setattr__(self, "val_split", val_split)
        object.__setattr__(self, "final_test_split", final_test_split)
        object.__setattr__(self, "split_sizes", split_sizes)
        for name, value in weights.items():
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "final_test_split": self.final_test_split,
            "split_sizes": {split: int(self.split_sizes[split]) for split in SUBTOKEN_PART_SPLIT_ORDER},
            "seed": int(self.seed),
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "eval_batch_size": int(self.eval_batch_size),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "early_stop_patience": int(self.early_stop_patience),
            "grad_clip_norm": float(self.grad_clip_norm),
            "selection_metric": self.selection_metric,
            "teacher_distill_weight": float(self.teacher_distill_weight),
            "residual_weight": float(self.residual_weight),
            "masked_subtoken_weight": float(self.masked_subtoken_weight),
            "gate_entropy_weight": float(self.gate_entropy_weight),
        }


@dataclass(frozen=True)
class SubtokenVariantConfig:
    """Named architecture variant used by ablations and submitters."""

    name: str
    version: str = SUBTOKEN_PART_VERSION_A
    gate_mode: str = SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX
    use_subtoken_encoder: bool = True
    use_context_stage: bool = True
    use_pairwise_bias: bool = True
    use_standard_part_branch: bool = False
    use_scalar_tokens: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        name = normalize_subtoken_part_variant(self.name)
        version = normalize_subtoken_part_version(self.version)
        gate_mode = normalize_subtoken_gate_mode(self.gate_mode)
        if name == SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE:
            if self.use_subtoken_encoder:
                raise ValueError("hlt_part_baseline cannot use the subtoken encoder")
            if gate_mode != SUBTOKEN_PART_GATE_NONE:
                raise ValueError("hlt_part_baseline must use gate_mode='none'")
        if name == SUBTOKEN_PART_VARIANT_SCALAR_LOCAL and not self.use_scalar_tokens:
            raise ValueError("scalar_token_local_only must set use_scalar_tokens=True")
        if gate_mode.startswith("context") and not self.use_context_stage:
            raise ValueError("context gate variants require use_context_stage=True")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "gate_mode", gate_mode)

    def to_model_config(self, *, num_classes: int, feature_config: SubtokenFeatureConfig | None = None) -> SubtokenPartConfig:
        return SubtokenPartConfig(
            num_classes=num_classes,
            feature_config=feature_config or SubtokenFeatureConfig(),
            variant=self.name,
            version=self.version,
            gate_mode=self.gate_mode,
            use_pairwise_bias=self.use_pairwise_bias,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "gate_mode": self.gate_mode,
            "use_subtoken_encoder": bool(self.use_subtoken_encoder),
            "use_context_stage": bool(self.use_context_stage),
            "use_pairwise_bias": bool(self.use_pairwise_bias),
            "use_standard_part_branch": bool(self.use_standard_part_branch),
            "use_scalar_tokens": bool(self.use_scalar_tokens),
            "description": self.description,
        }


@dataclass(frozen=True)
class SubtokenExperimentLayout:
    """Path helper for the reliability-gated subtoken branch."""

    output_root: str | Path = "checkpoints"
    experiment_name: str = SUBTOKEN_PART_EXPERIMENT_NAME

    @property
    def root(self) -> Path:
        return Path(self.output_root) / self.experiment_name

    @property
    def binary_inputs_dir(self) -> Path:
        return self.root / "binary_inputs"

    @property
    def manifest_path(self) -> Path:
        return self.binary_inputs_dir / "split_manifest.json.gz"

    @property
    def hlt_cache_dir(self) -> Path:
        return self.binary_inputs_dir / "hlt_cache"

    @property
    def taggers_dir(self) -> Path:
        return self.root / "taggers"

    def tagger_dir(self, variant: str) -> Path:
        return self.taggers_dir / normalize_subtoken_part_variant(variant)

    @property
    def final_report_dir(self) -> Path:
        return self.root / "final_report"

    def to_dict(self) -> dict[str, str]:
        return {
            "root": self.root.as_posix(),
            "binary_inputs_dir": self.binary_inputs_dir.as_posix(),
            "manifest_path": self.manifest_path.as_posix(),
            "hlt_cache_dir": self.hlt_cache_dir.as_posix(),
            "taggers_dir": self.taggers_dir.as_posix(),
            "final_report_dir": self.final_report_dir.as_posix(),
        }


def build_subtoken_variant_config(name: str, *, version: str = SUBTOKEN_PART_VERSION_A) -> SubtokenVariantConfig:
    variant = normalize_subtoken_part_variant(name)
    version = normalize_subtoken_part_version(version)
    if variant == SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE:
        return SubtokenVariantConfig(
            name=variant,
            version=version,
            gate_mode=SUBTOKEN_PART_GATE_NONE,
            use_subtoken_encoder=False,
            use_context_stage=False,
            use_pairwise_bias=True,
            use_standard_part_branch=True,
            description="standard HLT Particle Transformer baseline",
        )
    if variant == SUBTOKEN_PART_VARIANT_NO_GATE:
        return SubtokenVariantConfig(
            name=variant,
            version=version,
            gate_mode=SUBTOKEN_PART_GATE_NONE,
            use_subtoken_encoder=True,
            use_context_stage=False,
            use_pairwise_bias=True,
            description="subtoken local mixer pooled to particle tokens without reliability gates",
        )
    if variant == SUBTOKEN_PART_VARIANT_LOCAL_GATE:
        return SubtokenVariantConfig(
            name=variant,
            version=version,
            gate_mode=SUBTOKEN_PART_GATE_LOCAL_SOFTMAX,
            use_subtoken_encoder=True,
            use_context_stage=False,
            use_pairwise_bias=True,
            description="reliability gates use only within-particle modality evidence",
        )
    if variant == SUBTOKEN_PART_VARIANT_CONTEXT_GATE:
        return SubtokenVariantConfig(
            name=variant,
            version=version,
            gate_mode=SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX,
            use_subtoken_encoder=True,
            use_context_stage=True,
            use_pairwise_bias=True,
            description="full context-aware reliability-gated subtoken ParT",
        )
    if variant == SUBTOKEN_PART_VARIANT_DUAL_CROSS_ATTENTION:
        return SubtokenVariantConfig(
            name=variant,
            version=version,
            gate_mode=SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX,
            use_subtoken_encoder=True,
            use_context_stage=True,
            use_pairwise_bias=True,
            use_standard_part_branch=True,
            description="CrossViT-style fusion of standard ParT and subtoken branches",
        )
    if variant == SUBTOKEN_PART_VARIANT_SCALAR_LOCAL:
        return SubtokenVariantConfig(
            name=variant,
            version=version,
            gate_mode=SUBTOKEN_PART_GATE_NONE,
            use_subtoken_encoder=True,
            use_context_stage=False,
            use_pairwise_bias=True,
            use_scalar_tokens=True,
            description="research control using scalar tokens locally inside each particle",
        )
    raise AssertionError(f"Unhandled subtoken variant {variant!r}")


def build_subtoken_variant_configs(
    *,
    version: str = SUBTOKEN_PART_VERSION_A,
    variants: tuple[str, ...] = SUBTOKEN_PART_VARIANTS,
) -> tuple[SubtokenVariantConfig, ...]:
    return tuple(build_subtoken_variant_config(variant, version=version) for variant in variants)


def default_subtoken_part_config(*, num_classes: int = 2, version: str = SUBTOKEN_PART_VERSION_A) -> SubtokenPartConfig:
    return SubtokenPartConfig(num_classes=num_classes, version=version)


def default_subtoken_training_config(*, version: str = SUBTOKEN_PART_VERSION_A) -> SubtokenTrainingConfig:
    return SubtokenTrainingConfig(version=version)


def subtoken_part_model_name(variant: str, *, version: str = SUBTOKEN_PART_VERSION_A) -> str:
    normalized_variant = normalize_subtoken_part_variant(variant)
    normalized_version = normalize_subtoken_part_version(version)
    if normalized_version == SUBTOKEN_PART_VERSION_A:
        return normalized_variant
    return f"{normalized_variant}_{normalized_version}"
