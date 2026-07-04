"""Configuration and variant registry for Architecture-View Residual ParT.

Step 1 is intentionally limited to architecture-view generation.  Later steps
will inject these views into the real HLT ParT embedding space and train/report
the full QCD-vs-Hgg HLT0.6 experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from jetclass_fresh.jetclass_data import LABEL_NAMES, RAW_TOKEN_DIM


ARCHITECTURE_VIEW_PART_EXPERIMENT_NAME = "architecture_view_part"
ARCHITECTURE_VIEW_PART_STEP = "architecture_view_part_step1_view_branches"
ARCHITECTURE_VIEW_PART_CONTRACT = "architecture_view_residual_part_view_branches_v1"

ARCHITECTURE_VIEW_BACKGROUND_LABEL = "QCD"
ARCHITECTURE_VIEW_SIGNAL_LABEL = "Hgg"
ARCHITECTURE_VIEW_LABEL_NAMES = (ARCHITECTURE_VIEW_BACKGROUND_LABEL, ARCHITECTURE_VIEW_SIGNAL_LABEL)
ARCHITECTURE_VIEW_BINARY_LABEL_FILTER = (0, 1)
ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES = tuple(str(name) for name in LABEL_NAMES)
ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER = tuple(range(len(ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES)))
ARCHITECTURE_VIEW_HLT_DEGRADATION_STRENGTH = 0.6
ARCHITECTURE_VIEW_PRIMARY_METRIC = "fpr_at_signal_eff_0p50"
ARCHITECTURE_VIEW_PRIMARY_METRIC_DIRECTION = "minimize"
ARCHITECTURE_VIEW_10CLASS_PRIMARY_METRIC = "accuracy"
ARCHITECTURE_VIEW_10CLASS_SELECTION_METRICS = (
    "accuracy",
    "macro_per_class_accuracy",
    "loss",
)

ARCHITECTURE_VIEW_BRANCH_PN = "pn"
ARCHITECTURE_VIEW_BRANCH_PFN = "pfn"
ARCHITECTURE_VIEW_BRANCH_PCNN = "pcnn"
ARCHITECTURE_VIEW_BRANCHES = (
    ARCHITECTURE_VIEW_BRANCH_PN,
    ARCHITECTURE_VIEW_BRANCH_PFN,
    ARCHITECTURE_VIEW_BRANCH_PCNN,
)

ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK = "av_baseline_recheck"
ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS = "av_all_views"
ARCHITECTURE_VIEW_VARIANT_PN_ONLY = "av_pn_only"
ARCHITECTURE_VIEW_VARIANT_PFN_ONLY = "av_pfn_only"
ARCHITECTURE_VIEW_VARIANT_PCNN_ONLY = "av_pcnn_only"
ARCHITECTURE_VIEW_VARIANT_RANDOM_VIEW_CONTROL = "av_random_view_control"
ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL = "av_context_mlp_control"
ARCHITECTURE_VIEW_VARIANTS = (
    ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS,
    ARCHITECTURE_VIEW_VARIANT_PN_ONLY,
    ARCHITECTURE_VIEW_VARIANT_PFN_ONLY,
    ARCHITECTURE_VIEW_VARIANT_PCNN_ONLY,
    ARCHITECTURE_VIEW_VARIANT_RANDOM_VIEW_CONTROL,
    ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
)
ARCHITECTURE_VIEW_DEFAULT_PILOT_VARIANTS = (
    ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS,
    ARCHITECTURE_VIEW_VARIANT_PN_ONLY,
    ARCHITECTURE_VIEW_VARIANT_PFN_ONLY,
    ARCHITECTURE_VIEW_VARIANT_PCNN_ONLY,
    ARCHITECTURE_VIEW_VARIANT_RANDOM_VIEW_CONTROL,
    ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
)

ARCHITECTURE_VIEW_10CLASS_VARIANT_BASELINE_RECHECK = "av10_baseline_recheck"
ARCHITECTURE_VIEW_10CLASS_VARIANT_PART_CONTEXT_TO_PART = "av10_part_context_to_part"
ARCHITECTURE_VIEW_10CLASS_VARIANT_PN_CONTEXT_TO_PART = "av10_pn_context_to_part"
ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART = "av10_pfn_context_to_part"
ARCHITECTURE_VIEW_10CLASS_VARIANT_PCNN_CONTEXT_TO_PART = "av10_pcnn_context_to_part"
ARCHITECTURE_VIEW_10CLASS_VARIANT_ALL_VIEWS_TO_PART = "av10_all_views_to_part"
ARCHITECTURE_VIEW_10CLASS_VARIANT_RANDOM_VIEW_CONTROL = "av10_random_view_control"
ARCHITECTURE_VIEW_10CLASS_VARIANT_CONTEXT_MLP_CONTROL = "av10_context_mlp_control"
ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK = "av10_hlt_baseline_recheck"
ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART = "av10_larger_part"
ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK = "av10_extra_part_block"
ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER = "av10_part_only_adapter"
ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER = "av10_feature_mlp_adapter"
ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES = "av10_lc_mlp_delta_features"
ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER_WIDE = "av10_feature_mlp_adapter_wide"
ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER = "av10_frozen_part_feature_adapter"
ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER = "av10_shuffled_feature_adapter"
ARCHITECTURE_VIEW_10CLASS_ABLATION_PCNN_CONTEXT_REPEAT = "av10_pcnn_context_repeat"
ARCHITECTURE_VIEW_10CLASS_ABLATION_PFN_CONTEXT_REPEAT = "av10_pfn_context_repeat"
ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE = "av10_offline_part_baseline"
ARCHITECTURE_VIEW_10CLASS_OFFLINE_FEATURE_MLP_ADAPTER = "av10_offline_feature_mlp_adapter"
ARCHITECTURE_VIEW_10CLASS_OFFLINE_PCNN_CONTEXT = "av10_offline_pcnn_context"
ARCHITECTURE_VIEW_10CLASS_VARIANTS = (
    ARCHITECTURE_VIEW_10CLASS_VARIANT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PART_CONTEXT_TO_PART,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PN_CONTEXT_TO_PART,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PCNN_CONTEXT_TO_PART,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_ALL_VIEWS_TO_PART,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_RANDOM_VIEW_CONTROL,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_CONTEXT_MLP_CONTROL,
)
ARCHITECTURE_VIEW_10CLASS_ABLATION_VARIANTS = (
    ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER_WIDE,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_PCNN_CONTEXT_REPEAT,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_PFN_CONTEXT_REPEAT,
)
ARCHITECTURE_VIEW_10CLASS_ABLATION_DEFAULT_VARIANTS = (
    ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER,
)
ARCHITECTURE_VIEW_10CLASS_OFFLINE_TRANSFER_VARIANTS = (
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE,
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_FEATURE_MLP_ADAPTER,
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_PCNN_CONTEXT,
)
ARCHITECTURE_VIEW_10CLASS_ALL_VARIANTS = (
    ARCHITECTURE_VIEW_10CLASS_VARIANTS
    + ARCHITECTURE_VIEW_10CLASS_ABLATION_VARIANTS
    + ARCHITECTURE_VIEW_10CLASS_OFFLINE_TRANSFER_VARIANTS
)
ARCHITECTURE_VIEW_10CLASS_DEFAULT_VARIANTS = (
    ARCHITECTURE_VIEW_10CLASS_VARIANT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PART_CONTEXT_TO_PART,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PN_CONTEXT_TO_PART,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PCNN_CONTEXT_TO_PART,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_ALL_VIEWS_TO_PART,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_RANDOM_VIEW_CONTROL,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_CONTEXT_MLP_CONTROL,
)
ARCHITECTURE_VIEW_ALL_VARIANTS = (
    ARCHITECTURE_VIEW_VARIANTS
    + ARCHITECTURE_VIEW_10CLASS_VARIANTS
    + ARCHITECTURE_VIEW_10CLASS_ABLATION_VARIANTS
    + ARCHITECTURE_VIEW_10CLASS_OFFLINE_TRANSFER_VARIANTS
)

ARCHITECTURE_VIEW_10CLASS_VARIANT_TO_BINARY_BEHAVIOR: dict[str, str] = {
    ARCHITECTURE_VIEW_10CLASS_VARIANT_BASELINE_RECHECK: ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PART_CONTEXT_TO_PART: ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PN_CONTEXT_TO_PART: ARCHITECTURE_VIEW_VARIANT_PN_ONLY,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART: ARCHITECTURE_VIEW_VARIANT_PFN_ONLY,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_PCNN_CONTEXT_TO_PART: ARCHITECTURE_VIEW_VARIANT_PCNN_ONLY,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_ALL_VIEWS_TO_PART: ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_RANDOM_VIEW_CONTROL: ARCHITECTURE_VIEW_VARIANT_RANDOM_VIEW_CONTROL,
    ARCHITECTURE_VIEW_10CLASS_VARIANT_CONTEXT_MLP_CONTROL: ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK: ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART: ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK: ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER: ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER: ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES: ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER_WIDE: ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER: ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER: ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_PCNN_CONTEXT_REPEAT: ARCHITECTURE_VIEW_VARIANT_PCNN_ONLY,
    ARCHITECTURE_VIEW_10CLASS_ABLATION_PFN_CONTEXT_REPEAT: ARCHITECTURE_VIEW_VARIANT_PFN_ONLY,
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE: ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_FEATURE_MLP_ADAPTER: ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
    ARCHITECTURE_VIEW_10CLASS_OFFLINE_PCNN_CONTEXT: ARCHITECTURE_VIEW_VARIANT_PCNN_ONLY,
}

ARCHITECTURE_VIEW_VARIANT_ALIASES: dict[str, str] = {
    "baseline": ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    "hlt_part": ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    "hlt_part_baseline": ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    "av_baseline": ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
    "all": ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS,
    "all_views": ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS,
    "full": ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS,
    "pn": ARCHITECTURE_VIEW_VARIANT_PN_ONLY,
    "pn_only": ARCHITECTURE_VIEW_VARIANT_PN_ONLY,
    "particlenet": ARCHITECTURE_VIEW_VARIANT_PN_ONLY,
    "pfn": ARCHITECTURE_VIEW_VARIANT_PFN_ONLY,
    "pfn_only": ARCHITECTURE_VIEW_VARIANT_PFN_ONLY,
    "pcnn": ARCHITECTURE_VIEW_VARIANT_PCNN_ONLY,
    "pcnn_only": ARCHITECTURE_VIEW_VARIANT_PCNN_ONLY,
    "random": ARCHITECTURE_VIEW_VARIANT_RANDOM_VIEW_CONTROL,
    "random_view": ARCHITECTURE_VIEW_VARIANT_RANDOM_VIEW_CONTROL,
    "random_view_control": ARCHITECTURE_VIEW_VARIANT_RANDOM_VIEW_CONTROL,
    "context": ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
    "context_mlp": ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
    "context_mlp_control": ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
    "av10_baseline": ARCHITECTURE_VIEW_10CLASS_VARIANT_BASELINE_RECHECK,
    "av10_hlt_part": ARCHITECTURE_VIEW_10CLASS_VARIANT_BASELINE_RECHECK,
    "av10_hlt_part_baseline": ARCHITECTURE_VIEW_10CLASS_VARIANT_BASELINE_RECHECK,
    "av10_part": ARCHITECTURE_VIEW_10CLASS_VARIANT_PART_CONTEXT_TO_PART,
    "av10_part_context": ARCHITECTURE_VIEW_10CLASS_VARIANT_PART_CONTEXT_TO_PART,
    "av10_pn": ARCHITECTURE_VIEW_10CLASS_VARIANT_PN_CONTEXT_TO_PART,
    "av10_pn_only": ARCHITECTURE_VIEW_10CLASS_VARIANT_PN_CONTEXT_TO_PART,
    "av10_pfn": ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART,
    "av10_pfn_only": ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART,
    "av10_pcnn": ARCHITECTURE_VIEW_10CLASS_VARIANT_PCNN_CONTEXT_TO_PART,
    "av10_pcnn_only": ARCHITECTURE_VIEW_10CLASS_VARIANT_PCNN_CONTEXT_TO_PART,
    "av10_all": ARCHITECTURE_VIEW_10CLASS_VARIANT_ALL_VIEWS_TO_PART,
    "av10_full": ARCHITECTURE_VIEW_10CLASS_VARIANT_ALL_VIEWS_TO_PART,
    "av10_all_views": ARCHITECTURE_VIEW_10CLASS_VARIANT_ALL_VIEWS_TO_PART,
    "av10_random": ARCHITECTURE_VIEW_10CLASS_VARIANT_RANDOM_VIEW_CONTROL,
    "av10_random_view": ARCHITECTURE_VIEW_10CLASS_VARIANT_RANDOM_VIEW_CONTROL,
    "av10_context": ARCHITECTURE_VIEW_10CLASS_VARIANT_CONTEXT_MLP_CONTROL,
    "av10_context_mlp": ARCHITECTURE_VIEW_10CLASS_VARIANT_CONTEXT_MLP_CONTROL,
    "av10_ablation_baseline": ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
    "av10_hlt_baseline_recheck": ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
    "av10_larger": ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART,
    "av10_larger_part": ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART,
    "av10_big_part": ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART,
    "av10_extra_part": ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK,
    "av10_extra_part_block": ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK,
    "av10_part_adapter": ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER,
    "av10_part_only_adapter": ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER,
    "av10_feature_adapter": ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
    "av10_feature_mlp": ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
    "av10_feature_mlp_adapter": ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
    "av10_lc_mlp_delta": ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
    "av10_lc_mlp_delta_features": ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
    "av10_input_delta": ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
    "av10_input_feature_delta": ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
    "av10_feature_delta": ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
    "av10_wide_feature_mlp": ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER_WIDE,
    "av10_feature_mlp_wide": ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER_WIDE,
    "av10_feature_mlp_adapter_wide": ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER_WIDE,
    "av10_frozen_feature_adapter": ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER,
    "av10_frozen_part_feature_adapter": ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER,
    "av10_shuffled_feature_adapter": ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER,
    "av10_shuffled_features": ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER,
    "av10_pcnn_repeat": ARCHITECTURE_VIEW_10CLASS_ABLATION_PCNN_CONTEXT_REPEAT,
    "av10_pcnn_context_repeat": ARCHITECTURE_VIEW_10CLASS_ABLATION_PCNN_CONTEXT_REPEAT,
    "av10_pfn_repeat": ARCHITECTURE_VIEW_10CLASS_ABLATION_PFN_CONTEXT_REPEAT,
    "av10_pfn_context_repeat": ARCHITECTURE_VIEW_10CLASS_ABLATION_PFN_CONTEXT_REPEAT,
    "av10_offline_baseline": ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE,
    "av10_offline_part": ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE,
    "av10_offline_part_baseline": ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE,
    "av10_offline_feature_mlp": ARCHITECTURE_VIEW_10CLASS_OFFLINE_FEATURE_MLP_ADAPTER,
    "av10_offline_feature_adapter": ARCHITECTURE_VIEW_10CLASS_OFFLINE_FEATURE_MLP_ADAPTER,
    "av10_offline_feature_mlp_adapter": ARCHITECTURE_VIEW_10CLASS_OFFLINE_FEATURE_MLP_ADAPTER,
    "av10_offline_pcnn": ARCHITECTURE_VIEW_10CLASS_OFFLINE_PCNN_CONTEXT,
    "av10_offline_pcnn_context": ARCHITECTURE_VIEW_10CLASS_OFFLINE_PCNN_CONTEXT,
}

ARCHITECTURE_VIEW_RAW_FEATURE_NAMES = (
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


def _alias_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def normalize_architecture_view_variant(value: str) -> str:
    """Resolve a user-facing variant alias to the canonical registry value."""

    key = _alias_key(value)
    normalized = ARCHITECTURE_VIEW_VARIANT_ALIASES.get(key, key)
    if normalized not in ARCHITECTURE_VIEW_ALL_VARIANTS:
        raise ValueError(
            f"Unknown architecture-view variant {value!r}; expected one of {ARCHITECTURE_VIEW_ALL_VARIANTS}"
        )
    return normalized


def is_architecture_view_10class_variant(value: str) -> bool:
    """Return whether a variant belongs to the 10-class architecture-view suite."""

    return normalize_architecture_view_variant(value) in ARCHITECTURE_VIEW_10CLASS_ALL_VARIANTS


def architecture_view_effective_variant(value: str) -> str:
    """Map a reporting variant to the branch behavior implemented by the model."""

    normalized = normalize_architecture_view_variant(value)
    return ARCHITECTURE_VIEW_10CLASS_VARIANT_TO_BINARY_BEHAVIOR.get(normalized, normalized)


def architecture_view_variant_num_classes(value: str) -> int:
    return 10 if is_architecture_view_10class_variant(value) else 2


def normalize_architecture_view_branch(value: str) -> str:
    key = _alias_key(value)
    normalized = {
        "particlenet": ARCHITECTURE_VIEW_BRANCH_PN,
        "particle_net": ARCHITECTURE_VIEW_BRANCH_PN,
    }.get(key, key)
    if normalized not in ARCHITECTURE_VIEW_BRANCHES:
        raise ValueError(
            f"Unknown architecture-view branch {value!r}; expected one of {ARCHITECTURE_VIEW_BRANCHES}"
        )
    return normalized


@dataclass(frozen=True)
class ArchitectureViewVariantSpec:
    """Small registry entry for planned training/reporting variants."""

    name: str
    enabled_views: tuple[str, ...]
    description: str
    is_control: bool = False
    suite: str = "architecture_view_part"
    input_source: str = "hlt"
    adapter_type: str = "architecture_view"
    part_size: str = "baseline"
    freeze_policy: str = "joint_finetune"
    shuffle_policy: str = "none"
    parameter_target: str = "default"
    is_candidate: bool = True
    is_runnable: bool = True
    implementation_step: str = "implemented"

    def __post_init__(self) -> None:
        name = normalize_architecture_view_variant(self.name)
        views = tuple(normalize_architecture_view_branch(v) for v in _as_tuple(self.enabled_views))
        if len(set(views)) != len(views):
            raise ValueError(f"enabled_views contains duplicates: {views}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "enabled_views", views)
        object.__setattr__(self, "is_control", bool(self.is_control))
        object.__setattr__(self, "is_candidate", bool(self.is_candidate))
        object.__setattr__(self, "is_runnable", bool(self.is_runnable))
        for field in (
            "suite",
            "input_source",
            "adapter_type",
            "part_size",
            "freeze_policy",
            "shuffle_policy",
            "parameter_target",
            "implementation_step",
        ):
            object.__setattr__(self, field, str(getattr(self, field)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def architecture_view_variant_specs() -> dict[str, ArchitectureViewVariantSpec]:
    """Return the frozen Step 1 variant registry."""

    specs = (
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_VARIANT_BASELINE_RECHECK,
            enabled_views=(),
            description="Exact HLT ParT recheck with no architecture-view injection.",
            is_control=True,
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_VARIANT_ALL_VIEWS,
            enabled_views=ARCHITECTURE_VIEW_BRANCHES,
            description="PN, PFN, and PCNN per-particle views fused into the ParT embedding residual.",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_VARIANT_PN_ONLY,
            enabled_views=(ARCHITECTURE_VIEW_BRANCH_PN,),
            description="ParticleNet-style local eta-phi graph view only.",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_VARIANT_PFN_ONLY,
            enabled_views=(ARCHITECTURE_VIEW_BRANCH_PFN,),
            description="PFN-style particle-plus-global-summary view only.",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_VARIANT_PCNN_ONLY,
            enabled_views=(ARCHITECTURE_VIEW_BRANCH_PCNN,),
            description="PCNN-style ordered-particle convolutional view only.",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_VARIANT_RANDOM_VIEW_CONTROL,
            enabled_views=ARCHITECTURE_VIEW_BRANCHES,
            description="All view capacity with randomized view semantics for control runs.",
            is_control=True,
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_VARIANT_CONTEXT_MLP_CONTROL,
            enabled_views=(),
            description="Non-architecture context adapter control for later embedding-injection steps.",
            is_control=True,
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_VARIANT_BASELINE_RECHECK,
            enabled_views=(),
            description="10-class exact HLT ParT recheck with no architecture-view injection.",
            is_control=True,
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_VARIANT_PART_CONTEXT_TO_PART,
            enabled_views=(),
            description="10-class ParT-context MLP residual into the real ParT embedding.",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_VARIANT_PN_CONTEXT_TO_PART,
            enabled_views=(ARCHITECTURE_VIEW_BRANCH_PN,),
            description="10-class PN latent particle context injected into ParT.",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_VARIANT_PFN_CONTEXT_TO_PART,
            enabled_views=(ARCHITECTURE_VIEW_BRANCH_PFN,),
            description="10-class PFN latent particle context injected into ParT.",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_VARIANT_PCNN_CONTEXT_TO_PART,
            enabled_views=(ARCHITECTURE_VIEW_BRANCH_PCNN,),
            description="10-class PCNN latent particle context injected into ParT.",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_VARIANT_ALL_VIEWS_TO_PART,
            enabled_views=ARCHITECTURE_VIEW_BRANCHES,
            description="10-class PN, PFN, and PCNN latent particle views injected into ParT.",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_VARIANT_RANDOM_VIEW_CONTROL,
            enabled_views=ARCHITECTURE_VIEW_BRANCHES,
            description="10-class randomized view-semantics control.",
            is_control=True,
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_VARIANT_CONTEXT_MLP_CONTROL,
            enabled_views=(),
            description="10-class context-only MLP residual control.",
            is_control=True,
            suite="av10_original",
            adapter_type="feature_mlp_context",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_ABLATION_HLT_BASELINE_RECHECK,
            enabled_views=(),
            description="AV10 ablation A0: exact HLT ParT baseline recheck.",
            is_control=True,
            suite="av10_ablation",
            adapter_type="none",
            parameter_target="baseline_part",
            is_candidate=False,
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_ABLATION_LARGER_PART,
            enabled_views=(),
            description="AV10 ablation A1: larger vanilla HLT ParT with no adapter.",
            is_control=True,
            suite="av10_ablation",
            adapter_type="none",
            part_size="larger",
            parameter_target="larger_part_capacity_control",
            is_candidate=False,
            implementation_step="implemented_step2_larger_part_control",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_ABLATION_EXTRA_PART_BLOCK,
            enabled_views=(),
            description="AV10 ablation A2: parameter-matched extra ParT/Transformer refinement block.",
            is_control=True,
            suite="av10_ablation",
            adapter_type="extra_part_block",
            parameter_target="match_successful_av10_adapter",
            is_candidate=False,
            implementation_step="implemented_step2_extra_part_block_control",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_ABLATION_PART_ONLY_ADAPTER,
            enabled_views=(),
            description="AV10 ablation A3: h_base-only MLP adapter with no raw feature or architecture-view input.",
            is_control=True,
            suite="av10_ablation",
            adapter_type="part_embedding_mlp",
            parameter_target="match_feature_mlp_adapter",
            is_candidate=False,
            implementation_step="implemented_step2_part_only_adapter_control",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER,
            enabled_views=(),
            description="AV10 ablation A4: canonical feature MLP adapter, the current best single mechanism.",
            suite="av10_ablation",
            adapter_type="feature_mlp_context",
            parameter_target="current_context_mlp_adapter",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_ABLATION_LC_MLP_DELTA_FEATURES,
            enabled_views=(),
            description="AV10 ablation A5: LC-style MLP predicts bounded delta_F before ParT embedding.",
            suite="av10_ablation",
            adapter_type="feature_mlp_delta_F",
            parameter_target="input_feature_repair_adapter",
            implementation_step="implemented_step6_lc_mlp_delta_features",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_ABLATION_FEATURE_MLP_ADAPTER_WIDE,
            enabled_views=(),
            description="AV10 ablation A6: deeper/wider canonical feature MLP adapter.",
            suite="av10_ablation",
            adapter_type="feature_mlp_context_wide",
            parameter_target="scaled_feature_mlp_adapter",
            implementation_step="implemented_step3_wide_feature_mlp_adapter",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_ABLATION_FROZEN_PART_FEATURE_ADAPTER,
            enabled_views=(),
            description="AV10 ablation A7: train feature adapter while keeping the ParT backbone frozen.",
            suite="av10_ablation",
            adapter_type="feature_mlp_context",
            freeze_policy="frozen_part_adapter_only",
            parameter_target="current_context_mlp_adapter",
            implementation_step="implemented_step3_frozen_part_adapter",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_ABLATION_SHUFFLED_FEATURE_ADAPTER,
            enabled_views=(),
            description="AV10 ablation A8: same-capacity feature adapter with destroyed feature semantics.",
            is_control=True,
            suite="av10_ablation",
            adapter_type="feature_mlp_context",
            shuffle_policy="shuffled_features",
            parameter_target="current_context_mlp_adapter",
            is_candidate=False,
            implementation_step="implemented_step3_shuffled_feature_control",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_ABLATION_PCNN_CONTEXT_REPEAT,
            enabled_views=(ARCHITECTURE_VIEW_BRANCH_PCNN,),
            description="AV10 ablation A9: repeat of the strong PCNN-context architecture-view branch.",
            suite="av10_ablation",
            adapter_type="pcnn_context",
            parameter_target="current_pcnn_context_adapter",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_ABLATION_PFN_CONTEXT_REPEAT,
            enabled_views=(ARCHITECTURE_VIEW_BRANCH_PFN,),
            description="AV10 ablation A10: repeat of the strong PFN-context architecture-view branch.",
            suite="av10_ablation",
            adapter_type="pfn_context",
            parameter_target="current_pfn_context_adapter",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_OFFLINE_PART_BASELINE,
            enabled_views=(),
            description="AV10 offline transfer O0: normal offline ParT baseline with no HLT-degraded inputs.",
            is_control=True,
            suite="av10_offline_transfer",
            input_source="offline",
            adapter_type="none",
            parameter_target="offline_baseline_part",
            is_candidate=False,
            implementation_step="implemented_step4_offline_transfer",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_OFFLINE_FEATURE_MLP_ADAPTER,
            enabled_views=(),
            description="AV10 offline transfer O1: offline feature-MLP residual adapter into ParT.",
            suite="av10_offline_transfer",
            input_source="offline",
            adapter_type="feature_mlp_context",
            parameter_target="offline_feature_mlp_adapter",
            implementation_step="implemented_step4_offline_transfer",
        ),
        ArchitectureViewVariantSpec(
            name=ARCHITECTURE_VIEW_10CLASS_OFFLINE_PCNN_CONTEXT,
            enabled_views=(ARCHITECTURE_VIEW_BRANCH_PCNN,),
            description="AV10 offline transfer O2: optional offline PCNN-context residual adapter into ParT.",
            suite="av10_offline_transfer",
            input_source="offline",
            adapter_type="pcnn_context",
            parameter_target="offline_pcnn_context_adapter",
            implementation_step="implemented_step4_offline_transfer",
        ),
    )
    return {spec.name: spec for spec in specs}


def architecture_view_variant_spec(value: str) -> ArchitectureViewVariantSpec:
    return architecture_view_variant_specs()[normalize_architecture_view_variant(value)]


def enabled_views_for_variant(value: str) -> tuple[str, ...]:
    return architecture_view_variant_spec(value).enabled_views


def architecture_view_variant_is_runnable(value: str) -> bool:
    return bool(architecture_view_variant_spec(value).is_runnable)


def architecture_view_variant_is_baseline_recheck(value: str) -> bool:
    spec = architecture_view_variant_spec(value)
    return bool(
        spec.adapter_type == "none"
        and spec.part_size == "baseline"
        and spec.input_source == "hlt"
        and "baseline" in spec.name
    )


def architecture_view_runnable_variants(*, include_planned: bool = False) -> tuple[str, ...]:
    if include_planned:
        return ARCHITECTURE_VIEW_ALL_VARIANTS
    specs = architecture_view_variant_specs()
    return tuple(variant for variant in ARCHITECTURE_VIEW_ALL_VARIANTS if bool(specs[variant].is_runnable))


@dataclass(frozen=True)
class ArchitectureViewConfig:
    """Validated config for Step 1 view branches and fusion modules."""

    raw_token_dim: int = RAW_TOKEN_DIM
    view_dim: int = 32
    hidden_dim: int = 64
    pn_k: int = 16
    pn_layers: int = 2
    pfn_hidden_dim: int = 64
    pfn_use_global_context: bool = True
    pcnn_channels: int = 64
    pcnn_layers: int = 2
    pcnn_kernel_sizes: tuple[int, ...] = (3, 5)
    fusion_hidden_dim: int = 96
    part_embed_dim: int = 128
    num_classes: int = 2
    dropout: float = 0.05
    attention_dropout: float = 0.05
    gate_bias_init: float = -5.0
    input_delta_scale: float = 1.0
    use_feature_wise_input_delta_scales: bool = True
    freeze_input_delta_pid: bool = False
    freeze_input_delta_geometry: bool = False
    enabled_views: tuple[str, ...] = ARCHITECTURE_VIEW_BRANCHES
    random_control_seed: int = 2907
    eta_index: int = 1
    phi_index: int = 2
    pt_index: int = 0
    energy_index: int = 3
    eps: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "raw_token_dim",
            "view_dim",
            "hidden_dim",
            "pn_k",
            "pn_layers",
            "pfn_hidden_dim",
            "pcnn_channels",
            "pcnn_layers",
            "fusion_hidden_dim",
            "part_embed_dim",
            "num_classes",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            object.__setattr__(self, name, value)
        raw_dim = int(self.raw_token_dim)
        for name in ("eta_index", "phi_index", "pt_index", "energy_index"):
            index = int(getattr(self, name))
            if index < 0 or index >= raw_dim:
                raise ValueError(f"{name}={index} is outside raw_token_dim={raw_dim}")
            object.__setattr__(self, name, index)
        for name in ("dropout", "attention_dropout"):
            value = float(getattr(self, name))
            if value < 0.0 or value >= 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}")
            object.__setattr__(self, name, value)
        eps = float(self.eps)
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        object.__setattr__(self, "eps", eps)
        input_delta_scale = float(self.input_delta_scale)
        if input_delta_scale < 0.0:
            raise ValueError("input_delta_scale must be non-negative")
        object.__setattr__(self, "input_delta_scale", input_delta_scale)
        kernels = tuple(int(k) for k in _as_tuple(self.pcnn_kernel_sizes))
        if not kernels:
            raise ValueError("pcnn_kernel_sizes must not be empty")
        if any(k <= 0 or k % 2 == 0 for k in kernels):
            raise ValueError(f"pcnn_kernel_sizes must be positive odd integers, got {kernels}")
        object.__setattr__(self, "pcnn_kernel_sizes", kernels)
        views = tuple(normalize_architecture_view_branch(v) for v in _as_tuple(self.enabled_views))
        if len(set(views)) != len(views):
            raise ValueError(f"enabled_views contains duplicates: {views}")
        object.__setattr__(self, "enabled_views", views)
        object.__setattr__(self, "random_control_seed", int(self.random_control_seed))
        object.__setattr__(self, "pfn_use_global_context", bool(self.pfn_use_global_context))
        object.__setattr__(self, "gate_bias_init", float(self.gate_bias_init))
        object.__setattr__(
            self,
            "use_feature_wise_input_delta_scales",
            bool(self.use_feature_wise_input_delta_scales),
        )
        object.__setattr__(self, "freeze_input_delta_pid", bool(self.freeze_input_delta_pid))
        object.__setattr__(self, "freeze_input_delta_geometry", bool(self.freeze_input_delta_geometry))

    def with_variant(self, variant: str) -> "ArchitectureViewConfig":
        return ArchitectureViewConfig.from_dict(
            {
                **self.to_dict(),
                "enabled_views": enabled_views_for_variant(variant),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract"] = ARCHITECTURE_VIEW_PART_CONTRACT
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArchitectureViewConfig":
        data = dict(payload)
        data.pop("contract", None)
        return cls(**data)


def architecture_view_config_manifest(config: ArchitectureViewConfig | None = None) -> dict[str, Any]:
    cfg = config or ArchitectureViewConfig()
    return {
        "experiment": ARCHITECTURE_VIEW_PART_EXPERIMENT_NAME,
        "step": ARCHITECTURE_VIEW_PART_STEP,
        "contract": ARCHITECTURE_VIEW_PART_CONTRACT,
        "hlt_degradation_strength": ARCHITECTURE_VIEW_HLT_DEGRADATION_STRENGTH,
        "label_names": list(ARCHITECTURE_VIEW_LABEL_NAMES),
        "ten_class_label_names": list(ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES),
        "ten_class_label_filter": list(ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER),
        "primary_metric": ARCHITECTURE_VIEW_PRIMARY_METRIC,
        "primary_metric_direction": ARCHITECTURE_VIEW_PRIMARY_METRIC_DIRECTION,
        "ten_class_primary_metric": ARCHITECTURE_VIEW_10CLASS_PRIMARY_METRIC,
        "ten_class_selection_metrics": list(ARCHITECTURE_VIEW_10CLASS_SELECTION_METRICS),
        "raw_feature_names": list(ARCHITECTURE_VIEW_RAW_FEATURE_NAMES),
        "branches": list(ARCHITECTURE_VIEW_BRANCHES),
        "binary_variants": list(ARCHITECTURE_VIEW_VARIANTS),
        "ten_class_variants": list(ARCHITECTURE_VIEW_10CLASS_VARIANTS),
        "ten_class_ablation_variants": list(ARCHITECTURE_VIEW_10CLASS_ABLATION_VARIANTS),
        "ten_class_offline_transfer_variants": list(ARCHITECTURE_VIEW_10CLASS_OFFLINE_TRANSFER_VARIANTS),
        "ten_class_all_variants": list(ARCHITECTURE_VIEW_10CLASS_ALL_VARIANTS),
        "variants": {name: spec.to_dict() for name, spec in architecture_view_variant_specs().items()},
        "config": cfg.to_dict(),
    }
