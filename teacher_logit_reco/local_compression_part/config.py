"""Configuration contract for local-compression residual feature-adapter ParT.

Step 1 is intentionally config/protocol only.  It names the branch, freezes the
QCD-vs-Hgg HLT0.6 comparison contract, and defines validation-heavy dataclasses
for the later feature, adapter, model, training, and report steps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from jetclass_fresh.jetclass_data import LABEL_NAMES, RAW_TOKEN_DIM, SPLIT_ORDER
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES, PF_POINT_NAMES, PF_VECTOR_NAMES


LOCAL_COMPRESSION_PART_EXPERIMENT_NAME = "local_compression_part"
LOCAL_COMPRESSION_PART_EXPERIMENT_STEP = "local_compression_part_step1_config"
LOCAL_COMPRESSION_PART_CONTRACT = "local_compression_part_qcd_hgg_hlt06_residual_feature_adapter_v1"

LOCAL_COMPRESSION_PART_TASK_NAME = "qcd_vs_hgg_hlt06_local_compression"
LOCAL_COMPRESSION_PART_INFERENCE_VIEW = "hlt"
LOCAL_COMPRESSION_PART_HLT_DEGRADATION_STRENGTH = 0.6

LOCAL_COMPRESSION_BACKGROUND_LABEL = "QCD"
LOCAL_COMPRESSION_SIGNAL_LABEL = "Hgg"
LOCAL_COMPRESSION_SOURCE_LABEL_NAMES = (LOCAL_COMPRESSION_BACKGROUND_LABEL, LOCAL_COMPRESSION_SIGNAL_LABEL)
LOCAL_COMPRESSION_SOURCE_LABEL_INDICES = (0, 3)
LOCAL_COMPRESSION_BINARY_LABEL_FILTER = (0, 1)

LOCAL_COMPRESSION_BACKBONE_EXACT_HLT_PART = "exact_hlt_part"
LOCAL_COMPRESSION_ADAPTER_RESIDUAL_DELTA_FEATURES = "residual_delta_features"
LOCAL_COMPRESSION_LOSS_CROSS_ENTROPY_2LOGIT = "cross_entropy_2logit"

LOCAL_COMPRESSION_PRIMARY_METRIC = "fpr_at_signal_eff_0p50"
LOCAL_COMPRESSION_PRIMARY_METRIC_DIRECTION = "minimize"
LOCAL_COMPRESSION_VALIDATION_THRESHOLD_METRIC = "validation_threshold_final_test_fpr"

LOCAL_COMPRESSION_VARIANT_BASELINE_RECHECK = "hlt_part_baseline_recheck"
LOCAL_COMPRESSION_VARIANT_MLP_DELTA = "lc_mlp_delta"
LOCAL_COMPRESSION_VARIANT_LOCAL_NO_CONTEXT = "lc_local_compression_no_context"
LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED = "lc_context_gated"
LOCAL_COMPRESSION_VARIANT_CONTEXT_DELTA_NO_MODALITIES = "lc_context_delta_no_modalities"
LOCAL_COMPRESSION_VARIANT_RANDOM_GROUPING = "lc_random_grouping"
LOCAL_COMPRESSION_VARIANT_LARGER_HLT_PART_CONTROL = "lc_larger_hlt_part_control"
LOCAL_COMPRESSION_VARIANTS = (
    LOCAL_COMPRESSION_VARIANT_BASELINE_RECHECK,
    LOCAL_COMPRESSION_VARIANT_MLP_DELTA,
    LOCAL_COMPRESSION_VARIANT_LOCAL_NO_CONTEXT,
    LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED,
    LOCAL_COMPRESSION_VARIANT_CONTEXT_DELTA_NO_MODALITIES,
    LOCAL_COMPRESSION_VARIANT_RANDOM_GROUPING,
    LOCAL_COMPRESSION_VARIANT_LARGER_HLT_PART_CONTROL,
)
LOCAL_COMPRESSION_DEFAULT_PILOT_VARIANTS = (
    LOCAL_COMPRESSION_VARIANT_BASELINE_RECHECK,
    LOCAL_COMPRESSION_VARIANT_MLP_DELTA,
    LOCAL_COMPRESSION_VARIANT_LOCAL_NO_CONTEXT,
    LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED,
    LOCAL_COMPRESSION_VARIANT_CONTEXT_DELTA_NO_MODALITIES,
    LOCAL_COMPRESSION_VARIANT_RANDOM_GROUPING,
)
LOCAL_COMPRESSION_VARIANT_ALIASES: dict[str, str] = {
    "baseline": LOCAL_COMPRESSION_VARIANT_BASELINE_RECHECK,
    "hlt_part": LOCAL_COMPRESSION_VARIANT_BASELINE_RECHECK,
    "hlt_part_baseline": LOCAL_COMPRESSION_VARIANT_BASELINE_RECHECK,
    "hlt_part_baseline_recheck": LOCAL_COMPRESSION_VARIANT_BASELINE_RECHECK,
    "mlp": LOCAL_COMPRESSION_VARIANT_MLP_DELTA,
    "mlp_delta": LOCAL_COMPRESSION_VARIANT_MLP_DELTA,
    "lc_mlp": LOCAL_COMPRESSION_VARIANT_MLP_DELTA,
    "local": LOCAL_COMPRESSION_VARIANT_LOCAL_NO_CONTEXT,
    "no_context": LOCAL_COMPRESSION_VARIANT_LOCAL_NO_CONTEXT,
    "local_no_context": LOCAL_COMPRESSION_VARIANT_LOCAL_NO_CONTEXT,
    "context": LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED,
    "context_gated": LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED,
    "gated": LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED,
    "full": LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED,
    "context_only": LOCAL_COMPRESSION_VARIANT_CONTEXT_DELTA_NO_MODALITIES,
    "context_delta": LOCAL_COMPRESSION_VARIANT_CONTEXT_DELTA_NO_MODALITIES,
    "context_delta_no_modalities": LOCAL_COMPRESSION_VARIANT_CONTEXT_DELTA_NO_MODALITIES,
    "random": LOCAL_COMPRESSION_VARIANT_RANDOM_GROUPING,
    "random_grouping": LOCAL_COMPRESSION_VARIANT_RANDOM_GROUPING,
    "larger_part": LOCAL_COMPRESSION_VARIANT_LARGER_HLT_PART_CONTROL,
    "larger_hlt_part": LOCAL_COMPRESSION_VARIANT_LARGER_HLT_PART_CONTROL,
}

LOCAL_COMPRESSION_MODALITY_GEOMETRY = "geometry"
LOCAL_COMPRESSION_MODALITY_ENERGY_MOMENTUM = "energy_momentum"
LOCAL_COMPRESSION_MODALITY_IDENTITY = "identity"
LOCAL_COMPRESSION_MODALITY_TRACKING_ERROR = "tracking_error"
LOCAL_COMPRESSION_MODALITY_QUALITY_CONSISTENCY = "quality_consistency"
LOCAL_COMPRESSION_MODALITIES = (
    LOCAL_COMPRESSION_MODALITY_GEOMETRY,
    LOCAL_COMPRESSION_MODALITY_ENERGY_MOMENTUM,
    LOCAL_COMPRESSION_MODALITY_IDENTITY,
    LOCAL_COMPRESSION_MODALITY_TRACKING_ERROR,
    LOCAL_COMPRESSION_MODALITY_QUALITY_CONSISTENCY,
)

LOCAL_COMPRESSION_RAW_FEATURE_NAMES = (
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
LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES = tuple(PF_FEATURE_NAMES)
LOCAL_COMPRESSION_CANONICAL_POINT_NAMES = tuple(PF_POINT_NAMES)
LOCAL_COMPRESSION_CANONICAL_VECTOR_NAMES = tuple(PF_VECTOR_NAMES)

LOCAL_COMPRESSION_DERIVED_FIELD_NAMES = (
    "sin_phi",
    "cos_phi",
    "pt_rank",
    "log_pt_rank",
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
    "valid_mask",
    "all_finite",
    "charged_pid_consistency",
    "neutral_track_applicability",
    "track_error_summary",
)

LOCAL_COMPRESSION_POOL_LEARNED_QUERY = "learned_query"
LOCAL_COMPRESSION_POOL_MEAN = "mean"
LOCAL_COMPRESSION_POOL_MODES = (LOCAL_COMPRESSION_POOL_LEARNED_QUERY, LOCAL_COMPRESSION_POOL_MEAN)

LOCAL_COMPRESSION_GATE_NONE = "none"
LOCAL_COMPRESSION_GATE_CONTEXT_SIGMOID = "context_sigmoid"
LOCAL_COMPRESSION_GATE_MODES = (LOCAL_COMPRESSION_GATE_NONE, LOCAL_COMPRESSION_GATE_CONTEXT_SIGMOID)

LOCAL_COMPRESSION_DEFAULT_SPLIT_SIZES = {
    "model_train": 500_000,
    "model_val": 150_000,
    "stack_train": 500_000,
    "stack_val": 150_000,
    "final_test": 500_000,
}


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


def normalize_local_compression_variant(value: str) -> str:
    """Resolve user-facing aliases into the frozen local-compression variant names."""

    key = _alias_key(value)
    normalized = LOCAL_COMPRESSION_VARIANT_ALIASES.get(key, key)
    if normalized not in LOCAL_COMPRESSION_VARIANTS:
        raise ValueError(
            f"Unknown local-compression variant {value!r}; expected one of {LOCAL_COMPRESSION_VARIANTS}"
        )
    return normalized


def normalize_local_compression_split_name(value: str) -> str:
    split = str(value).strip()
    if split not in SPLIT_ORDER:
        raise ValueError(f"Unknown local-compression split {value!r}; expected one of {tuple(SPLIT_ORDER)}")
    return split


def normalize_local_compression_pool_mode(value: str) -> str:
    key = _alias_key(value)
    if key in {"attention", "attn", "query", "learned", "learned_query_attention"}:
        key = LOCAL_COMPRESSION_POOL_LEARNED_QUERY
    if key in {"avg", "average", "mean_pool", "mean_pooling"}:
        key = LOCAL_COMPRESSION_POOL_MEAN
    if key not in LOCAL_COMPRESSION_POOL_MODES:
        raise ValueError(f"Unknown local-compression pool mode {value!r}; expected {LOCAL_COMPRESSION_POOL_MODES}")
    return key


def normalize_local_compression_gate_mode(value: str) -> str:
    key = _alias_key(value)
    if key in {"off", "disable", "disabled", "no_gate"}:
        key = LOCAL_COMPRESSION_GATE_NONE
    if key in {"sigmoid", "context", "context_gate"}:
        key = LOCAL_COMPRESSION_GATE_CONTEXT_SIGMOID
    if key not in LOCAL_COMPRESSION_GATE_MODES:
        raise ValueError(f"Unknown local-compression gate mode {value!r}; expected {LOCAL_COMPRESSION_GATE_MODES}")
    return key


def _validate_probability(value: float, *, name: str, allow_one: bool = True) -> float:
    value = float(value)
    upper_ok = value <= 1.0 if allow_one else value < 1.0
    if value < 0.0 or not upper_ok:
        upper = "1" if allow_one else "1, exclusive"
        raise ValueError(f"{name} must be in [0, {upper}]")
    return value


def _validate_positive_int(value: int, *, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_nonnegative_float(value: float, *, name: str) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_attention_heads(embed_dim: int, heads: int, *, name: str) -> None:
    if int(heads) <= 0:
        raise ValueError(f"{name} must be positive")
    if int(embed_dim) % int(heads) != 0:
        raise ValueError(f"embed_dim must be divisible by {name}")


def default_feature_delta_scales() -> tuple[float, ...]:
    """Conservative per-PF-feature scales for Step 1's residual-input contract."""

    scales_by_feature = {
        "part_pt_log": 0.05,
        "part_e_log": 0.05,
        "part_logptrel": 0.05,
        "part_logerel": 0.05,
        "part_deltaR": 0.025,
        "part_charge": 0.05,
        "part_isChargedHadron": 0.02,
        "part_isNeutralHadron": 0.02,
        "part_isPhoton": 0.02,
        "part_isElectron": 0.02,
        "part_isMuon": 0.02,
        "part_d0": 0.20,
        "part_d0err": 0.20,
        "part_dz": 0.20,
        "part_dzerr": 0.20,
        "part_deta": 0.025,
        "part_dphi": 0.025,
    }
    return tuple(float(scales_by_feature[name]) for name in LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES)


@dataclass(frozen=True)
class LocalCompressionSplitSpec:
    """One named split and its intended maximum jet count."""

    name: str
    max_jets: int
    role: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", normalize_local_compression_split_name(self.name))
        object.__setattr__(self, "max_jets", _validate_positive_int(self.max_jets, name="max_jets"))
        if not str(self.role).strip():
            raise ValueError("split role must be non-empty")


@dataclass(frozen=True)
class LocalCompressionMetricSpec:
    """Metric name plus comparison direction."""

    name: str
    direction: str
    description: str
    required_on_final_test: bool = True

    def __post_init__(self) -> None:
        direction = _alias_key(self.direction)
        if direction not in {"minimize", "maximize"}:
            raise ValueError("metric direction must be minimize or maximize")
        if not str(self.name).strip():
            raise ValueError("metric name must be non-empty")
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "direction", direction)


@dataclass(frozen=True)
class LocalCompressionVariantSpec:
    """A planned model/control variant for the local-compression branch."""

    name: str
    role: str
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", normalize_local_compression_variant(self.name))
        if not str(self.role).strip():
            raise ValueError("variant role must be non-empty")


@dataclass(frozen=True)
class LocalCompressionModalitySpec:
    """One semantic feature group inside a particle."""

    name: str
    raw_feature_names: tuple[str, ...] = ()
    pf_feature_names: tuple[str, ...] = ()
    derived_feature_names: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        name = _alias_key(self.name)
        if name not in LOCAL_COMPRESSION_MODALITIES:
            raise ValueError(f"unknown local-compression modality {self.name!r}")
        raw_feature_names = tuple(str(item) for item in _as_tuple(self.raw_feature_names))
        pf_feature_names = tuple(str(item) for item in _as_tuple(self.pf_feature_names))
        derived_feature_names = tuple(str(item) for item in _as_tuple(self.derived_feature_names))
        if not (raw_feature_names or pf_feature_names or derived_feature_names):
            raise ValueError(f"modality {name!r} must include at least one field")
        unknown_raw = [item for item in raw_feature_names if item not in LOCAL_COMPRESSION_RAW_FEATURE_NAMES]
        unknown_pf = [item for item in pf_feature_names if item not in LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES]
        unknown_derived = [item for item in derived_feature_names if item not in LOCAL_COMPRESSION_DERIVED_FIELD_NAMES]
        if unknown_raw:
            raise ValueError(f"unknown raw feature names in modality {name!r}: {unknown_raw}")
        if unknown_pf:
            raise ValueError(f"unknown PF feature names in modality {name!r}: {unknown_pf}")
        if unknown_derived:
            raise ValueError(f"unknown derived field names in modality {name!r}: {unknown_derived}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "raw_feature_names", raw_feature_names)
        object.__setattr__(self, "pf_feature_names", pf_feature_names)
        object.__setattr__(self, "derived_feature_names", derived_feature_names)

    @property
    def field_count(self) -> int:
        return len(self.raw_feature_names) + len(self.pf_feature_names) + len(self.derived_feature_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "raw_feature_names": list(self.raw_feature_names),
            "pf_feature_names": list(self.pf_feature_names),
            "derived_feature_names": list(self.derived_feature_names),
            "field_count": int(self.field_count),
            "description": self.description,
        }


def default_local_compression_modality_specs() -> tuple[LocalCompressionModalitySpec, ...]:
    """Return the plan's default modality grouping for Step 1."""

    return (
        LocalCompressionModalitySpec(
            name=LOCAL_COMPRESSION_MODALITY_GEOMETRY,
            raw_feature_names=("eta", "phi"),
            pf_feature_names=("part_deltaR", "part_deta", "part_dphi"),
            derived_feature_names=("sin_phi", "cos_phi", "pt_rank", "log_pt_rank"),
            description="Particle direction, jet-relative geometry, phi wrapping, and pT order.",
        ),
        LocalCompressionModalitySpec(
            name=LOCAL_COMPRESSION_MODALITY_ENERGY_MOMENTUM,
            raw_feature_names=("pt", "energy"),
            pf_feature_names=("part_pt_log", "part_e_log", "part_logptrel", "part_logerel"),
            derived_feature_names=("part_px", "part_py", "part_pz", "part_energy"),
            description="Momentum and energy evidence, using log-scaled canonical features as the core.",
        ),
        LocalCompressionModalitySpec(
            name=LOCAL_COMPRESSION_MODALITY_IDENTITY,
            raw_feature_names=(
                "charge",
                "isChargedHadron",
                "isNeutralHadron",
                "isPhoton",
                "isElectron",
                "isMuon",
            ),
            pf_feature_names=(
                "part_charge",
                "part_isChargedHadron",
                "part_isNeutralHadron",
                "part_isPhoton",
                "part_isElectron",
                "part_isMuon",
            ),
            description="Charge and HLT particle-ID evidence.",
        ),
        LocalCompressionModalitySpec(
            name=LOCAL_COMPRESSION_MODALITY_TRACKING_ERROR,
            raw_feature_names=("d0", "d0err", "dz", "dzerr"),
            pf_feature_names=("part_d0", "part_d0err", "part_dz", "part_dzerr"),
            description="Impact-parameter and track-error evidence.",
        ),
        LocalCompressionModalitySpec(
            name=LOCAL_COMPRESSION_MODALITY_QUALITY_CONSISTENCY,
            raw_feature_names=(
                "charge",
                "isChargedHadron",
                "isNeutralHadron",
                "isPhoton",
                "isElectron",
                "isMuon",
                "d0err",
                "dzerr",
            ),
            derived_feature_names=(
                "valid_mask",
                "all_finite",
                "charged_pid_consistency",
                "neutral_track_applicability",
                "track_error_summary",
                "pt_rank",
            ),
            description="Reliability and feature-consistency hints for context-aware gates.",
        ),
    )


@dataclass(frozen=True)
class LocalCompressionPartProtocol:
    """Single source of truth for the QCD/Hgg HLT0.6 local-compression experiment."""

    experiment_step: str = LOCAL_COMPRESSION_PART_EXPERIMENT_STEP
    contract: str = LOCAL_COMPRESSION_PART_CONTRACT
    task_name: str = LOCAL_COMPRESSION_PART_TASK_NAME
    inference_view: str = LOCAL_COMPRESSION_PART_INFERENCE_VIEW
    offline_view_allowed_at_inference: bool = False
    hlt_degradation_strength: float = LOCAL_COMPRESSION_PART_HLT_DEGRADATION_STRENGTH
    background_label: str = LOCAL_COMPRESSION_BACKGROUND_LABEL
    signal_label: str = LOCAL_COMPRESSION_SIGNAL_LABEL
    source_label_names: tuple[str, ...] = LOCAL_COMPRESSION_SOURCE_LABEL_NAMES
    source_label_indices: tuple[int, ...] = LOCAL_COMPRESSION_SOURCE_LABEL_INDICES
    binary_label_filter: tuple[int, ...] = LOCAL_COMPRESSION_BINARY_LABEL_FILTER
    binary_label_names: tuple[str, ...] = LOCAL_COMPRESSION_SOURCE_LABEL_NAMES
    num_classes: int = 2
    primary_metric: str = LOCAL_COMPRESSION_PRIMARY_METRIC
    selection_metric: str = LOCAL_COMPRESSION_PRIMARY_METRIC
    selection_metric_direction: str = LOCAL_COMPRESSION_PRIMARY_METRIC_DIRECTION
    validation_threshold_metric: str = LOCAL_COMPRESSION_VALIDATION_THRESHOLD_METRIC
    comparison_split: str = "final_test"
    confirm_final_test: bool = True
    backbone_type: str = LOCAL_COMPRESSION_BACKBONE_EXACT_HLT_PART
    adapter_mode: str = LOCAL_COMPRESSION_ADAPTER_RESIDUAL_DELTA_FEATURES
    baseline_recoverable_at_zero_delta: bool = True
    loss_name: str = LOCAL_COMPRESSION_LOSS_CROSS_ENTROPY_2LOGIT
    baseline_checkpoint_selection_metric: str = LOCAL_COMPRESSION_PRIMARY_METRIC
    split_specs: tuple[LocalCompressionSplitSpec, ...] = (
        LocalCompressionSplitSpec("model_train", 500_000, "train LC adapter and optional ParT fine-tune"),
        LocalCompressionSplitSpec("model_val", 150_000, "select checkpoints and validation threshold"),
        LocalCompressionSplitSpec("stack_train", 500_000, "reserved for later fusion/residual analysis"),
        LocalCompressionSplitSpec("stack_val", 150_000, "reserved for unbiased validation diagnostics"),
        LocalCompressionSplitSpec("final_test", 500_000, "final held-out comparison"),
    )
    metric_specs: tuple[LocalCompressionMetricSpec, ...] = (
        LocalCompressionMetricSpec(
            "fpr_at_signal_eff_0p50",
            "minimize",
            "Primary binary metric: QCD false-positive rate at 50% Hgg signal efficiency.",
        ),
        LocalCompressionMetricSpec(
            "background_rejection_at_signal_eff_0p50",
            "maximize",
            "Equivalent rejection view of the primary metric.",
        ),
        LocalCompressionMetricSpec("auc", "maximize", "Secondary ranking metric."),
        LocalCompressionMetricSpec("accuracy", "maximize", "Sanity metric only; not for checkpoint selection."),
        LocalCompressionMetricSpec(
            LOCAL_COMPRESSION_VALIDATION_THRESHOLD_METRIC,
            "minimize",
            "Final-test FPR after choosing the 50% signal-efficiency threshold on model_val.",
        ),
    )
    variant_specs: tuple[LocalCompressionVariantSpec, ...] = (
        LocalCompressionVariantSpec("hlt_part_baseline_recheck", "Exact HLT ParT baseline recovery check."),
        LocalCompressionVariantSpec("lc_mlp_delta", "MLP feature-delta control without modality subtokens."),
        LocalCompressionVariantSpec(
            "lc_local_compression_no_context",
            "Within-particle modality compression without particle-context gates.",
        ),
        LocalCompressionVariantSpec("lc_context_gated", "Primary context-aware local-compression feature adapter."),
        LocalCompressionVariantSpec(
            "lc_context_delta_no_modalities",
            "Context-only pre-ParT delta control without modality subtokens.",
        ),
        LocalCompressionVariantSpec(
            "lc_random_grouping",
            "Random feature grouping control preserving modality group sizes.",
        ),
        LocalCompressionVariantSpec("lc_larger_hlt_part_control", "Optional parameter-matched larger ParT.", False),
    )

    def validate(self) -> None:
        """Raise if the frozen local-compression protocol drifts."""

        if self.contract != LOCAL_COMPRESSION_PART_CONTRACT:
            raise ValueError(f"unexpected local-compression contract: {self.contract}")
        if self.inference_view != "hlt" or bool(self.offline_view_allowed_at_inference):
            raise ValueError("local-compression protocol must be HLT-only at inference")
        if abs(float(self.hlt_degradation_strength) - LOCAL_COMPRESSION_PART_HLT_DEGRADATION_STRENGTH) > 1.0e-12:
            raise ValueError("local-compression protocol is frozen to HLT degradation strength 0.6")
        if tuple(self.source_label_names) != LOCAL_COMPRESSION_SOURCE_LABEL_NAMES:
            raise ValueError("local-compression protocol is frozen to QCD vs Hgg")
        if tuple(self.source_label_indices) != LOCAL_COMPRESSION_SOURCE_LABEL_INDICES:
            raise ValueError("source label ids must be original JetClass QCD=0, Hgg=3")
        for name, index in zip(self.source_label_names, self.source_label_indices):
            if LABEL_NAMES[int(index)] != name:
                raise ValueError(f"JetClass label id mismatch for {name}: got {index}")
        if tuple(self.binary_label_filter) != LOCAL_COMPRESSION_BINARY_LABEL_FILTER:
            raise ValueError("binary label filter must be remapped QCD=0, Hgg=1")
        if int(self.num_classes) != 2:
            raise ValueError("local-compression protocol is binary only")
        if self.primary_metric != LOCAL_COMPRESSION_PRIMARY_METRIC:
            raise ValueError("primary metric must be FPR@50")
        if self.selection_metric != self.primary_metric:
            raise ValueError("checkpoint selection must use the primary metric")
        if self.selection_metric_direction != "minimize":
            raise ValueError("FPR@50 selection direction must be minimize")
        if self.validation_threshold_metric != LOCAL_COMPRESSION_VALIDATION_THRESHOLD_METRIC:
            raise ValueError("validation-threshold final-test metric must be present")
        if self.comparison_split != "final_test" or not bool(self.confirm_final_test):
            raise ValueError("final-test confirmation is required")
        if self.backbone_type != LOCAL_COMPRESSION_BACKBONE_EXACT_HLT_PART:
            raise ValueError("local-compression mainline must use the exact HLT ParT backbone")
        if self.adapter_mode != LOCAL_COMPRESSION_ADAPTER_RESIDUAL_DELTA_FEATURES:
            raise ValueError("local-compression mainline must use residual delta-F adaptation")
        if not bool(self.baseline_recoverable_at_zero_delta):
            raise ValueError("baseline recovery at zero delta is required")
        if self.loss_name != LOCAL_COMPRESSION_LOSS_CROSS_ENTROPY_2LOGIT:
            raise ValueError("local-compression binary training must default to two-logit CrossEntropyLoss")
        if self.baseline_checkpoint_selection_metric != LOCAL_COMPRESSION_PRIMARY_METRIC:
            raise ValueError("baseline checkpoint must be selected by FPR@50")
        expected_splits = dict(LOCAL_COMPRESSION_DEFAULT_SPLIT_SIZES)
        if self.split_size_by_name != expected_splits:
            raise ValueError(f"unexpected split protocol: {self.split_size_by_name}")
        metric_direction = self.metric_direction_by_name
        if metric_direction.get(self.primary_metric) != "minimize":
            raise ValueError("FPR@50 must be minimized")
        if metric_direction.get(self.validation_threshold_metric) != "minimize":
            raise ValueError("validation-threshold final-test FPR must be minimized")
        if self.required_variant_names != LOCAL_COMPRESSION_DEFAULT_PILOT_VARIANTS:
            raise ValueError(f"unexpected required variants: {self.required_variant_names}")

    @property
    def split_size_by_name(self) -> dict[str, int]:
        return {spec.name: int(spec.max_jets) for spec in self.split_specs}

    @property
    def metric_direction_by_name(self) -> dict[str, str]:
        return {spec.name: spec.direction for spec in self.metric_specs}

    @property
    def required_variant_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.variant_specs if spec.required)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return json.loads(json.dumps(asdict(self), sort_keys=True))


@dataclass(frozen=True)
class LocalCompressionFeatureConfig:
    """Feature/modality contract for the local-compression adapter."""

    raw_token_dim: int = RAW_TOKEN_DIM
    canonical_feature_names: tuple[str, ...] = LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES
    canonical_point_names: tuple[str, ...] = LOCAL_COMPRESSION_CANONICAL_POINT_NAMES
    canonical_vector_names: tuple[str, ...] = LOCAL_COMPRESSION_CANONICAL_VECTOR_NAMES
    modalities: tuple[LocalCompressionModalitySpec, ...] = field(
        default_factory=default_local_compression_modality_specs
    )
    require_exact_raw_token_dim: bool = True
    require_exact_canonical_features: bool = True
    use_feature_wise_delta_scales: bool = True
    feature_delta_scales: tuple[float, ...] = field(default_factory=default_feature_delta_scales)
    delta_l2_weight: float = 1.0e-4

    def __post_init__(self) -> None:
        raw_token_dim = int(self.raw_token_dim)
        if raw_token_dim <= 0:
            raise ValueError("raw_token_dim must be positive")
        if self.require_exact_raw_token_dim and raw_token_dim != RAW_TOKEN_DIM:
            raise ValueError(f"local-compression raw token contract requires RAW_TOKEN_DIM={RAW_TOKEN_DIM}")
        canonical_feature_names = tuple(str(name) for name in self.canonical_feature_names)
        canonical_point_names = tuple(str(name) for name in self.canonical_point_names)
        canonical_vector_names = tuple(str(name) for name in self.canonical_vector_names)
        if self.require_exact_canonical_features:
            if canonical_feature_names != LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES:
                raise ValueError("canonical feature order must match PF_FEATURE_NAMES")
            if canonical_point_names != LOCAL_COMPRESSION_CANONICAL_POINT_NAMES:
                raise ValueError("canonical point order must match PF_POINT_NAMES")
            if canonical_vector_names != LOCAL_COMPRESSION_CANONICAL_VECTOR_NAMES:
                raise ValueError("canonical vector order must match PF_VECTOR_NAMES")
        modalities = tuple(
            spec if isinstance(spec, LocalCompressionModalitySpec) else LocalCompressionModalitySpec(**spec)
            for spec in self.modalities
        )
        if not modalities:
            raise ValueError("modalities must be non-empty")
        names = tuple(spec.name for spec in modalities)
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate modality names are not allowed: {names}")
        if names != LOCAL_COMPRESSION_MODALITIES:
            raise ValueError(
                "modality order must match LOCAL_COMPRESSION_MODALITIES exactly: "
                f"expected {LOCAL_COMPRESSION_MODALITIES}, got {names}"
            )
        feature_delta_scales = tuple(float(value) for value in self.feature_delta_scales)
        if len(feature_delta_scales) != len(canonical_feature_names):
            raise ValueError("feature_delta_scales must match canonical feature count")
        if any(value < 0.0 for value in feature_delta_scales):
            raise ValueError("feature_delta_scales must be non-negative")
        object.__setattr__(self, "raw_token_dim", raw_token_dim)
        object.__setattr__(self, "canonical_feature_names", canonical_feature_names)
        object.__setattr__(self, "canonical_point_names", canonical_point_names)
        object.__setattr__(self, "canonical_vector_names", canonical_vector_names)
        object.__setattr__(self, "modalities", modalities)
        object.__setattr__(self, "feature_delta_scales", feature_delta_scales)
        object.__setattr__(self, "delta_l2_weight", _validate_nonnegative_float(self.delta_l2_weight, name="delta_l2_weight"))

    @property
    def num_modalities(self) -> int:
        return len(self.modalities)

    @property
    def modality_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.modalities)

    @property
    def feature_delta_scale_by_name(self) -> dict[str, float]:
        return dict(zip(self.canonical_feature_names, self.feature_delta_scales))

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_token_dim": int(self.raw_token_dim),
            "canonical_feature_names": list(self.canonical_feature_names),
            "canonical_point_names": list(self.canonical_point_names),
            "canonical_vector_names": list(self.canonical_vector_names),
            "modalities": [spec.to_dict() for spec in self.modalities],
            "num_modalities": int(self.num_modalities),
            "require_exact_raw_token_dim": bool(self.require_exact_raw_token_dim),
            "require_exact_canonical_features": bool(self.require_exact_canonical_features),
            "use_feature_wise_delta_scales": bool(self.use_feature_wise_delta_scales),
            "feature_delta_scales": list(self.feature_delta_scales),
            "feature_delta_scale_by_name": self.feature_delta_scale_by_name,
            "delta_l2_weight": float(self.delta_l2_weight),
        }


@dataclass(frozen=True)
class LocalCompressionPartConfig:
    """Model-shape contract for the exact-ParT residual feature adapter."""

    num_classes: int = 2
    variant: str = LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED
    backbone_type: str = LOCAL_COMPRESSION_BACKBONE_EXACT_HLT_PART
    adapter_mode: str = LOCAL_COMPRESSION_ADAPTER_RESIDUAL_DELTA_FEATURES
    loss_name: str = LOCAL_COMPRESSION_LOSS_CROSS_ENTROPY_2LOGIT
    feature_config: LocalCompressionFeatureConfig = field(default_factory=LocalCompressionFeatureConfig)
    embed_dim: int = 96
    local_layers: int = 2
    local_heads: int = 4
    context_layers: int = 1
    context_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.05
    attention_dropout: float = 0.05
    pool_mode: str = LOCAL_COMPRESSION_POOL_LEARNED_QUERY
    gate_mode: str = LOCAL_COMPRESSION_GATE_CONTEXT_SIGMOID
    use_particle_anchor: bool = True
    use_modality_type_embeddings: bool = True
    use_pt_rank_embedding: bool = True
    zero_init_delta_projection: bool = True
    baseline_recoverable_at_zero_delta: bool = True
    delta_scale: float = 1.0
    freeze_pid_deltas: bool = False
    freeze_geometry_deltas: bool = False

    def __post_init__(self) -> None:
        if int(self.num_classes) != 2:
            raise ValueError("local-compression QCD/Hgg config is binary and must use num_classes=2")
        variant = normalize_local_compression_variant(self.variant)
        if self.backbone_type != LOCAL_COMPRESSION_BACKBONE_EXACT_HLT_PART:
            raise ValueError("local-compression mainline must use exact_hlt_part")
        if self.adapter_mode != LOCAL_COMPRESSION_ADAPTER_RESIDUAL_DELTA_FEATURES:
            raise ValueError("local-compression adapter_mode must be residual_delta_features")
        if self.loss_name != LOCAL_COMPRESSION_LOSS_CROSS_ENTROPY_2LOGIT:
            raise ValueError("local-compression training must default to two-logit CrossEntropyLoss")
        feature_config = (
            self.feature_config
            if isinstance(self.feature_config, LocalCompressionFeatureConfig)
            else LocalCompressionFeatureConfig(**self.feature_config)
        )
        embed_dim = _validate_positive_int(self.embed_dim, name="embed_dim")
        local_layers = _validate_positive_int(self.local_layers, name="local_layers")
        context_layers = _validate_positive_int(self.context_layers, name="context_layers")
        _validate_attention_heads(embed_dim, self.local_heads, name="local_heads")
        _validate_attention_heads(embed_dim, self.context_heads, name="context_heads")
        if float(self.mlp_ratio) <= 0.0:
            raise ValueError("mlp_ratio must be positive")
        dropout = _validate_probability(self.dropout, name="dropout")
        attention_dropout = _validate_probability(self.attention_dropout, name="attention_dropout")
        pool_mode = normalize_local_compression_pool_mode(self.pool_mode)
        gate_mode = normalize_local_compression_gate_mode(self.gate_mode)
        if variant == LOCAL_COMPRESSION_VARIANT_BASELINE_RECHECK:
            if gate_mode != LOCAL_COMPRESSION_GATE_NONE:
                raise ValueError("baseline recheck must use gate_mode='none'")
        if variant == LOCAL_COMPRESSION_VARIANT_MLP_DELTA and gate_mode != LOCAL_COMPRESSION_GATE_NONE:
            raise ValueError("MLP delta control must use gate_mode='none'")
        if variant == LOCAL_COMPRESSION_VARIANT_CONTEXT_DELTA_NO_MODALITIES and gate_mode != LOCAL_COMPRESSION_GATE_NONE:
            raise ValueError("context-delta no-modalities control must use gate_mode='none'")
        if variant == LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED and gate_mode != LOCAL_COMPRESSION_GATE_CONTEXT_SIGMOID:
            raise ValueError("context-gated variant must use context_sigmoid gates")
        if variant == LOCAL_COMPRESSION_VARIANT_LOCAL_NO_CONTEXT and int(context_layers) != 1:
            # The config keeps a positive layer count for constructor simplicity;
            # later builders should ignore context_layers for this variant.
            pass
        if not bool(self.zero_init_delta_projection):
            raise ValueError("zero_init_delta_projection is required for baseline recovery")
        if not bool(self.baseline_recoverable_at_zero_delta):
            raise ValueError("baseline_recoverable_at_zero_delta must be true")
        if float(self.delta_scale) < 0.0:
            raise ValueError("delta_scale must be non-negative")
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "feature_config", feature_config)
        object.__setattr__(self, "embed_dim", embed_dim)
        object.__setattr__(self, "local_layers", local_layers)
        object.__setattr__(self, "context_layers", context_layers)
        object.__setattr__(self, "dropout", dropout)
        object.__setattr__(self, "attention_dropout", attention_dropout)
        object.__setattr__(self, "pool_mode", pool_mode)
        object.__setattr__(self, "gate_mode", gate_mode)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": LOCAL_COMPRESSION_PART_CONTRACT,
            "num_classes": int(self.num_classes),
            "variant": self.variant,
            "backbone_type": self.backbone_type,
            "adapter_mode": self.adapter_mode,
            "loss_name": self.loss_name,
            "feature_config": self.feature_config.to_dict(),
            "embed_dim": int(self.embed_dim),
            "local_layers": int(self.local_layers),
            "local_heads": int(self.local_heads),
            "context_layers": int(self.context_layers),
            "context_heads": int(self.context_heads),
            "mlp_ratio": float(self.mlp_ratio),
            "dropout": float(self.dropout),
            "attention_dropout": float(self.attention_dropout),
            "pool_mode": self.pool_mode,
            "gate_mode": self.gate_mode,
            "use_particle_anchor": bool(self.use_particle_anchor),
            "use_modality_type_embeddings": bool(self.use_modality_type_embeddings),
            "use_pt_rank_embedding": bool(self.use_pt_rank_embedding),
            "zero_init_delta_projection": bool(self.zero_init_delta_projection),
            "baseline_recoverable_at_zero_delta": bool(self.baseline_recoverable_at_zero_delta),
            "delta_scale": float(self.delta_scale),
            "freeze_pid_deltas": bool(self.freeze_pid_deltas),
            "freeze_geometry_deltas": bool(self.freeze_geometry_deltas),
        }


@dataclass(frozen=True)
class LocalCompressionTrainingConfig:
    """Training configuration shared by local-compression variants."""

    train_split: str = "model_train"
    val_split: str = "model_val"
    final_test_split: str = "final_test"
    split_sizes: Mapping[str, int] = field(default_factory=lambda: dict(LOCAL_COMPRESSION_DEFAULT_SPLIT_SIZES))
    seed: int = 2907
    epochs: int = 45
    batch_size: int = 64
    eval_batch_size: int = 128
    adapter_lr: float = 3.0e-4
    part_lr: float = 3.0e-5
    weight_decay: float = 1.0e-4
    early_stop_patience: int = 6
    grad_clip_norm: float = 1.0
    freeze_part_epochs: int = 0
    selection_metric: str = LOCAL_COMPRESSION_PRIMARY_METRIC
    selection_metric_direction: str = LOCAL_COMPRESSION_PRIMARY_METRIC_DIRECTION
    loss_name: str = LOCAL_COMPRESSION_LOSS_CROSS_ENTROPY_2LOGIT
    confirm_final_test: bool = True
    require_baseline_checkpoint: bool = True
    require_fpr50_selected_baseline: bool = True
    require_hlt06_baseline: bool = True

    def __post_init__(self) -> None:
        train_split = normalize_local_compression_split_name(self.train_split)
        val_split = normalize_local_compression_split_name(self.val_split)
        final_test_split = normalize_local_compression_split_name(self.final_test_split)
        if train_split != "model_train" or val_split != "model_val" or final_test_split != "final_test":
            raise ValueError("local-compression training is frozen to model_train/model_val/final_test")
        split_sizes = {normalize_local_compression_split_name(split): int(size) for split, size in self.split_sizes.items()}
        if set(split_sizes) != set(SPLIT_ORDER):
            raise ValueError(f"split_sizes must contain exactly {tuple(SPLIT_ORDER)}")
        for split, size in split_sizes.items():
            if size <= 0:
                raise ValueError(f"split size for {split} must be positive")
        for name in ("seed", "epochs", "batch_size", "eval_batch_size", "early_stop_patience"):
            _validate_positive_int(getattr(self, name), name=name)
        if float(self.adapter_lr) <= 0.0:
            raise ValueError("adapter_lr must be positive")
        if float(self.part_lr) <= 0.0:
            raise ValueError("part_lr must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if float(self.grad_clip_norm) <= 0.0:
            raise ValueError("grad_clip_norm must be positive")
        if int(self.freeze_part_epochs) < 0:
            raise ValueError("freeze_part_epochs must be non-negative")
        if self.selection_metric != LOCAL_COMPRESSION_PRIMARY_METRIC:
            raise ValueError("local-compression checkpoint selection must use FPR@50")
        if self.selection_metric_direction != "minimize":
            raise ValueError("FPR@50 selection direction must be minimize")
        if self.loss_name != LOCAL_COMPRESSION_LOSS_CROSS_ENTROPY_2LOGIT:
            raise ValueError("local-compression training must use two-logit CrossEntropyLoss by default")
        if not bool(self.confirm_final_test):
            raise ValueError("final-test confirmation is required")
        object.__setattr__(self, "train_split", train_split)
        object.__setattr__(self, "val_split", val_split)
        object.__setattr__(self, "final_test_split", final_test_split)
        object.__setattr__(self, "split_sizes", split_sizes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_split": self.train_split,
            "val_split": self.val_split,
            "final_test_split": self.final_test_split,
            "split_sizes": {split: int(self.split_sizes[split]) for split in SPLIT_ORDER},
            "seed": int(self.seed),
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "eval_batch_size": int(self.eval_batch_size),
            "adapter_lr": float(self.adapter_lr),
            "part_lr": float(self.part_lr),
            "weight_decay": float(self.weight_decay),
            "early_stop_patience": int(self.early_stop_patience),
            "grad_clip_norm": float(self.grad_clip_norm),
            "freeze_part_epochs": int(self.freeze_part_epochs),
            "selection_metric": self.selection_metric,
            "selection_metric_direction": self.selection_metric_direction,
            "loss_name": self.loss_name,
            "confirm_final_test": bool(self.confirm_final_test),
            "require_baseline_checkpoint": bool(self.require_baseline_checkpoint),
            "require_fpr50_selected_baseline": bool(self.require_fpr50_selected_baseline),
            "require_hlt06_baseline": bool(self.require_hlt06_baseline),
        }


@dataclass(frozen=True)
class LocalCompressionVariantConfig:
    """Named architecture/control settings for submitters and reports."""

    name: str = LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED
    use_modalities: bool = True
    use_local_compressor: bool = True
    use_particle_context: bool = True
    use_context_gates: bool = True
    random_grouping_seed: int | None = None
    required_for_pilot: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        name = normalize_local_compression_variant(self.name)
        if name == LOCAL_COMPRESSION_VARIANT_BASELINE_RECHECK:
            expected = (False, False, False, False)
        elif name == LOCAL_COMPRESSION_VARIANT_MLP_DELTA:
            expected = (False, False, False, False)
        elif name == LOCAL_COMPRESSION_VARIANT_LOCAL_NO_CONTEXT:
            expected = (True, True, False, False)
        elif name == LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED:
            expected = (True, True, True, True)
        elif name == LOCAL_COMPRESSION_VARIANT_CONTEXT_DELTA_NO_MODALITIES:
            expected = (False, False, True, False)
        elif name == LOCAL_COMPRESSION_VARIANT_RANDOM_GROUPING:
            expected = (True, True, True, True)
        else:
            expected = (False, False, False, False)
        actual = (
            bool(self.use_modalities),
            bool(self.use_local_compressor),
            bool(self.use_particle_context),
            bool(self.use_context_gates),
        )
        if actual != expected:
            raise ValueError(f"variant {name!r} expects modality/local/context/gate flags {expected}, got {actual}")
        if name == LOCAL_COMPRESSION_VARIANT_RANDOM_GROUPING and self.random_grouping_seed is None:
            raise ValueError("random grouping variant requires random_grouping_seed")
        object.__setattr__(self, "name", name)

    def to_model_config(self, *, num_classes: int = 2) -> LocalCompressionPartConfig:
        gate_mode = (
            LOCAL_COMPRESSION_GATE_CONTEXT_SIGMOID
            if self.use_context_gates
            else LOCAL_COMPRESSION_GATE_NONE
        )
        return LocalCompressionPartConfig(num_classes=num_classes, variant=self.name, gate_mode=gate_mode)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "use_modalities": bool(self.use_modalities),
            "use_local_compressor": bool(self.use_local_compressor),
            "use_particle_context": bool(self.use_particle_context),
            "use_context_gates": bool(self.use_context_gates),
            "random_grouping_seed": self.random_grouping_seed,
            "required_for_pilot": bool(self.required_for_pilot),
            "description": self.description,
        }


def default_local_compression_variant_configs() -> tuple[LocalCompressionVariantConfig, ...]:
    """Return the pilot variants with strict flag settings."""

    return (
        LocalCompressionVariantConfig(
            LOCAL_COMPRESSION_VARIANT_BASELINE_RECHECK,
            use_modalities=False,
            use_local_compressor=False,
            use_particle_context=False,
            use_context_gates=False,
            description="Exact baseline checkpoint re-evaluation and zero-delta recovery check.",
        ),
        LocalCompressionVariantConfig(
            LOCAL_COMPRESSION_VARIANT_MLP_DELTA,
            use_modalities=False,
            use_local_compressor=False,
            use_particle_context=False,
            use_context_gates=False,
            description="Feature-delta MLP control with no modality tokens.",
        ),
        LocalCompressionVariantConfig(
            LOCAL_COMPRESSION_VARIANT_LOCAL_NO_CONTEXT,
            use_modalities=True,
            use_local_compressor=True,
            use_particle_context=False,
            use_context_gates=False,
            description="Within-particle local compression without particle-context gates.",
        ),
        LocalCompressionVariantConfig(
            LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED,
            use_modalities=True,
            use_local_compressor=True,
            use_particle_context=True,
            use_context_gates=True,
            description="Primary context-aware reliability-gated local-compression adapter.",
        ),
        LocalCompressionVariantConfig(
            LOCAL_COMPRESSION_VARIANT_CONTEXT_DELTA_NO_MODALITIES,
            use_modalities=False,
            use_local_compressor=False,
            use_particle_context=True,
            use_context_gates=False,
            description="Control for context-only feature deltas without local modality compression.",
        ),
        LocalCompressionVariantConfig(
            LOCAL_COMPRESSION_VARIANT_RANDOM_GROUPING,
            use_modalities=True,
            use_local_compressor=True,
            use_particle_context=True,
            use_context_gates=True,
            random_grouping_seed=2907,
            description="Random feature-grouping control preserving modality group sizes.",
        ),
    )


@dataclass(frozen=True)
class LocalCompressionExperimentLayout:
    """Path helper for later scripts and reports."""

    output_root: str | Path = "checkpoints"
    experiment_name: str = LOCAL_COMPRESSION_PART_EXPERIMENT_NAME

    @property
    def root(self) -> Path:
        return Path(self.output_root) / self.experiment_name

    @property
    def taggers_dir(self) -> Path:
        return self.root / "taggers"

    @property
    def report_dir(self) -> Path:
        return self.root / "final_report"

    def variant_dir(self, variant: str) -> Path:
        return self.taggers_dir / normalize_local_compression_variant(variant)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "experiment_name": self.experiment_name,
            "root": str(self.root),
            "taggers_dir": str(self.taggers_dir),
            "report_dir": str(self.report_dir),
        }


def default_local_compression_part_protocol() -> LocalCompressionPartProtocol:
    """Return the frozen local-compression protocol after validation."""

    protocol = LocalCompressionPartProtocol()
    protocol.validate()
    return protocol


def local_compression_part_protocol_manifest() -> dict[str, Any]:
    """JSON-serializable protocol manifest for run reports."""

    return default_local_compression_part_protocol().to_dict()


def local_compression_part_config_manifest(
    *,
    model_config: LocalCompressionPartConfig | None = None,
    training_config: LocalCompressionTrainingConfig | None = None,
) -> dict[str, Any]:
    """Return a combined protocol/model/training config payload."""

    protocol = default_local_compression_part_protocol()
    model_config = model_config or LocalCompressionPartConfig()
    training_config = training_config or LocalCompressionTrainingConfig()
    return {
        "protocol": protocol.to_dict(),
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
    }
