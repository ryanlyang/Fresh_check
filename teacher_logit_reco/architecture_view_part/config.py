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
    if normalized not in ARCHITECTURE_VIEW_VARIANTS:
        raise ValueError(
            f"Unknown architecture-view variant {value!r}; expected one of {ARCHITECTURE_VIEW_VARIANTS}"
        )
    return normalized


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

    def __post_init__(self) -> None:
        name = normalize_architecture_view_variant(self.name)
        views = tuple(normalize_architecture_view_branch(v) for v in _as_tuple(self.enabled_views))
        if len(set(views)) != len(views):
            raise ValueError(f"enabled_views contains duplicates: {views}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "enabled_views", views)
        object.__setattr__(self, "is_control", bool(self.is_control))

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
    )
    return {spec.name: spec for spec in specs}


def enabled_views_for_variant(value: str) -> tuple[str, ...]:
    return architecture_view_variant_specs()[normalize_architecture_view_variant(value)].enabled_views


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
        "primary_metric": ARCHITECTURE_VIEW_PRIMARY_METRIC,
        "primary_metric_direction": ARCHITECTURE_VIEW_PRIMARY_METRIC_DIRECTION,
        "raw_feature_names": list(ARCHITECTURE_VIEW_RAW_FEATURE_NAMES),
        "branches": list(ARCHITECTURE_VIEW_BRANCHES),
        "variants": {name: spec.to_dict() for name, spec in architecture_view_variant_specs().items()},
        "config": cfg.to_dict(),
    }
