"""Configuration contract for the reliability-gated dual-view ParT branch.

Step 1 is intentionally config-only.  The branch is deliberately narrow:
QCD/Hgg, HLT degradation strength 0.6, 500k/150k/500k split sizes, and a
two-view deployment surface consisting of the original HLT view and a PN
reconstructed view.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from jetclass_fresh.jetclass_data import LABEL_NAMES, RAW_TOKEN_DIM, SPLIT_ORDER


DUALVIEW_PART_EXPERIMENT_NAME = "reliability_gated_dualview_part"
DUALVIEW_PART_EXPERIMENT_STEP = "dualview_part_step1_config"
DUALVIEW_PART_CONTRACT = "qcd_hgg_hlt0p6_hlt_plus_pn_reco_part_v1"

DUALVIEW_PART_SOURCE_LABEL_NAMES: tuple[str, str] = ("QCD", "Hgg")
DUALVIEW_PART_BINARY_LABEL_FILTER_NAMES: tuple[str, str] = ("0", "1")
DUALVIEW_PART_NUM_CLASSES = 2
DUALVIEW_PART_POSITIVE_CLASS_NAME = "Hgg"
DUALVIEW_PART_POSITIVE_CLASS_INDEX = 1
DUALVIEW_PART_HLT_DEGRADATION_STRENGTH = 0.6

DUALVIEW_PART_VIEW_HLT = "hlt"
DUALVIEW_PART_VIEW_PN_RECO = "pn_reco"
DUALVIEW_PART_REQUIRED_VIEWS: tuple[str, str] = (DUALVIEW_PART_VIEW_HLT, DUALVIEW_PART_VIEW_PN_RECO)
DUALVIEW_PART_RECONSTRUCTOR_ARCHITECTURE = "pn"
DUALVIEW_PART_ANCHOR_ARCHITECTURE = "part"
DUALVIEW_PART_OFFLINE_REFERENCE_ARCHITECTURE = "part"

DUALVIEW_PART_PRIMARY_METRIC = "fpr_at_signal_eff_0p50"
DUALVIEW_PART_SECONDARY_METRICS: tuple[str, ...] = (
    "fpr_at_signal_eff_0p30",
    "auc",
    "accuracy",
)
DUALVIEW_PART_LOWER_IS_BETTER_METRICS: tuple[str, ...] = (
    "fpr_at_signal_eff_0p30",
    "fpr_at_signal_eff_0p50",
)

DUALVIEW_PART_SPLIT_ORDER: tuple[str, ...] = tuple(SPLIT_ORDER)
DUALVIEW_PART_SPLIT_SIZES: dict[str, int] = {
    "model_train": 500_000,
    "model_val": 150_000,
    "stack_train": 500_000,
    "stack_val": 150_000,
    "final_test": 500_000,
}

DUALVIEW_PART_VARIANT_HLT_PART_BASELINE = "hlt_part_baseline"
DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL = "frozen_anchor_pn_residual"
DUALVIEW_PART_VARIANT_FROZEN_CROSS_ATTENTION = "frozen_anchor_pn_cross_attention"
DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL_CROSS_ATTENTION = "frozen_anchor_pn_residual_cross_attention"
DUALVIEW_PART_VARIANT_WARM_RESIDUAL = "warm_anchor_pn_residual"
DUALVIEW_PART_VARIANT_SHUFFLED_PN_CONTROL = "frozen_anchor_shuffled_pn_control"
DUALVIEW_PART_VARIANTS: tuple[str, ...] = (
    DUALVIEW_PART_VARIANT_HLT_PART_BASELINE,
    DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL,
    DUALVIEW_PART_VARIANT_FROZEN_CROSS_ATTENTION,
    DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL_CROSS_ATTENTION,
    DUALVIEW_PART_VARIANT_WARM_RESIDUAL,
    DUALVIEW_PART_VARIANT_SHUFFLED_PN_CONTROL,
)
DUALVIEW_PART_VARIANT_ALIASES: dict[str, str] = {
    "baseline": DUALVIEW_PART_VARIANT_HLT_PART_BASELINE,
    "hlt": DUALVIEW_PART_VARIANT_HLT_PART_BASELINE,
    "hlt_part": DUALVIEW_PART_VARIANT_HLT_PART_BASELINE,
    "part": DUALVIEW_PART_VARIANT_HLT_PART_BASELINE,
    "residual": DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL,
    "frozen_residual": DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL,
    "pn_residual": DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL,
    "cross": DUALVIEW_PART_VARIANT_FROZEN_CROSS_ATTENTION,
    "cross_attention": DUALVIEW_PART_VARIANT_FROZEN_CROSS_ATTENTION,
    "pn_cross_attention": DUALVIEW_PART_VARIANT_FROZEN_CROSS_ATTENTION,
    "full": DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL_CROSS_ATTENTION,
    "residual_cross": DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL_CROSS_ATTENTION,
    "warm": DUALVIEW_PART_VARIANT_WARM_RESIDUAL,
    "warm_residual": DUALVIEW_PART_VARIANT_WARM_RESIDUAL,
    "shuffled": DUALVIEW_PART_VARIANT_SHUFFLED_PN_CONTROL,
    "shuffle_control": DUALVIEW_PART_VARIANT_SHUFFLED_PN_CONTROL,
}

_REPORT_ONLY_CONFIG_KEYS: tuple[str, ...] = (
    "contract",
    "experiment_step",
    "experiment_tag",
    "source_label_indices",
    "primary_metric_direction",
    "selection_metric_direction",
)


def _alias_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_tuple(values: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        values = values.split()
    try:
        normalized = tuple(str(value).strip() for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of strings") from exc
    if any(not value for value in normalized):
        raise ValueError(f"{name} cannot contain empty values")
    return normalized


def normalize_dualview_part_variant(value: str) -> str:
    key = _alias_key(value)
    normalized = DUALVIEW_PART_VARIANT_ALIASES.get(key, key)
    if normalized not in DUALVIEW_PART_VARIANTS:
        raise ValueError(f"Unknown dual-view ParT variant {value!r}; expected one of {DUALVIEW_PART_VARIANTS}")
    return normalized


def normalize_dualview_split_name(value: str) -> str:
    split = str(value).strip()
    if split not in DUALVIEW_PART_SPLIT_ORDER:
        raise ValueError(f"Unknown dual-view split {value!r}; expected one of {DUALVIEW_PART_SPLIT_ORDER}")
    return split


def dualview_metric_direction(metric: str) -> str:
    metric = str(metric).strip()
    if metric in DUALVIEW_PART_LOWER_IS_BETTER_METRICS:
        return "minimize"
    return "maximize"


def canonical_dualview_part_tag(*, hlt_degradation_strength: float = DUALVIEW_PART_HLT_DEGRADATION_STRENGTH) -> str:
    strength_tag = f"hlt{float(hlt_degradation_strength):0.1f}".replace(".", "p")
    return f"dualview_part_qcd_hgg_binary_{strength_tag}_true500k"


@dataclass(frozen=True)
class DualViewPartExperimentConfig:
    """Strict experiment contract for the QCD/Hgg HLT0.6 dual-view branch."""

    source_label_names: tuple[str, ...] = DUALVIEW_PART_SOURCE_LABEL_NAMES
    downstream_label_filter_names: tuple[str, ...] = DUALVIEW_PART_BINARY_LABEL_FILTER_NAMES
    label_names: tuple[str, ...] = DUALVIEW_PART_SOURCE_LABEL_NAMES
    num_classes: int = DUALVIEW_PART_NUM_CLASSES
    positive_class_name: str = DUALVIEW_PART_POSITIVE_CLASS_NAME
    positive_class_index: int = DUALVIEW_PART_POSITIVE_CLASS_INDEX
    hlt_degradation_strength: float = DUALVIEW_PART_HLT_DEGRADATION_STRENGTH
    split_sizes: Mapping[str, int] = field(default_factory=lambda: dict(DUALVIEW_PART_SPLIT_SIZES))
    required_views: tuple[str, ...] = DUALVIEW_PART_REQUIRED_VIEWS
    pn_reconstructor_architecture: str = DUALVIEW_PART_RECONSTRUCTOR_ARCHITECTURE
    anchor_architecture: str = DUALVIEW_PART_ANCHOR_ARCHITECTURE
    offline_reference_architecture: str = DUALVIEW_PART_OFFLINE_REFERENCE_ARCHITECTURE
    primary_metric: str = DUALVIEW_PART_PRIMARY_METRIC
    secondary_metrics: tuple[str, ...] = DUALVIEW_PART_SECONDARY_METRICS
    selection_metric: str = DUALVIEW_PART_PRIMARY_METRIC
    raw_token_dim: int = RAW_TOKEN_DIM
    max_constituents: int = 128
    confirm_final_test: bool = True
    allow_noncanonical: bool = False

    def __post_init__(self) -> None:
        source_label_names = _normalize_tuple(self.source_label_names, name="source_label_names")
        downstream_label_filter_names = _normalize_tuple(
            self.downstream_label_filter_names,
            name="downstream_label_filter_names",
        )
        label_names = _normalize_tuple(self.label_names, name="label_names")
        required_views = tuple(_alias_key(view) for view in _normalize_tuple(self.required_views, name="required_views"))
        secondary_metrics = _normalize_tuple(self.secondary_metrics, name="secondary_metrics")
        split_sizes = {normalize_dualview_split_name(key): int(value) for key, value in self.split_sizes.items()}
        missing = [split for split in DUALVIEW_PART_SPLIT_ORDER if split not in split_sizes]
        extra = [split for split in split_sizes if split not in DUALVIEW_PART_SPLIT_ORDER]
        if missing or extra:
            raise ValueError(f"split_sizes must contain exactly {DUALVIEW_PART_SPLIT_ORDER}")
        for split, size in split_sizes.items():
            if size <= 0:
                raise ValueError(f"split size for {split} must be positive")
        if int(self.num_classes) != len(label_names):
            raise ValueError("num_classes must match len(label_names)")
        if self.positive_class_name not in label_names:
            raise ValueError("positive_class_name must be present in label_names")
        expected_positive_index = label_names.index(self.positive_class_name)
        if int(self.positive_class_index) != expected_positive_index:
            raise ValueError(
                f"positive_class_index must be {expected_positive_index} for positive class "
                f"{self.positive_class_name!r}"
            )
        if int(self.raw_token_dim) <= 0:
            raise ValueError("raw_token_dim must be positive")
        if int(self.max_constituents) <= 0:
            raise ValueError("max_constituents must be positive")
        primary_metric = str(self.primary_metric).strip()
        selection_metric = str(self.selection_metric).strip()
        if not primary_metric:
            raise ValueError("primary_metric must be non-empty")
        if not selection_metric:
            raise ValueError("selection_metric must be non-empty")
        if primary_metric not in (DUALVIEW_PART_PRIMARY_METRIC, *DUALVIEW_PART_SECONDARY_METRICS):
            raise ValueError(f"Unsupported primary_metric {primary_metric!r}")
        if selection_metric not in (DUALVIEW_PART_PRIMARY_METRIC, *DUALVIEW_PART_SECONDARY_METRICS):
            raise ValueError(f"Unsupported selection_metric {selection_metric!r}")
        hlt_degradation_strength = float(self.hlt_degradation_strength)
        if hlt_degradation_strength <= 0.0:
            raise ValueError("hlt_degradation_strength must be positive")
        if not self.allow_noncanonical:
            self._validate_canonical_contract(
                source_label_names=source_label_names,
                downstream_label_filter_names=downstream_label_filter_names,
                label_names=label_names,
                required_views=required_views,
                split_sizes=split_sizes,
                hlt_degradation_strength=hlt_degradation_strength,
                primary_metric=primary_metric,
                selection_metric=selection_metric,
            )
        object.__setattr__(self, "source_label_names", source_label_names)
        object.__setattr__(self, "downstream_label_filter_names", downstream_label_filter_names)
        object.__setattr__(self, "label_names", label_names)
        object.__setattr__(self, "required_views", required_views)
        object.__setattr__(self, "secondary_metrics", secondary_metrics)
        object.__setattr__(self, "split_sizes", split_sizes)
        object.__setattr__(self, "num_classes", int(self.num_classes))
        object.__setattr__(self, "positive_class_index", int(self.positive_class_index))
        object.__setattr__(self, "hlt_degradation_strength", hlt_degradation_strength)
        object.__setattr__(self, "pn_reconstructor_architecture", _alias_key(self.pn_reconstructor_architecture))
        object.__setattr__(self, "anchor_architecture", _alias_key(self.anchor_architecture))
        object.__setattr__(self, "offline_reference_architecture", _alias_key(self.offline_reference_architecture))
        object.__setattr__(self, "primary_metric", primary_metric)
        object.__setattr__(self, "selection_metric", selection_metric)
        object.__setattr__(self, "raw_token_dim", int(self.raw_token_dim))
        object.__setattr__(self, "max_constituents", int(self.max_constituents))
        object.__setattr__(self, "confirm_final_test", bool(self.confirm_final_test))
        object.__setattr__(self, "allow_noncanonical", bool(self.allow_noncanonical))

    def _validate_canonical_contract(
        self,
        *,
        source_label_names: tuple[str, ...],
        downstream_label_filter_names: tuple[str, ...],
        label_names: tuple[str, ...],
        required_views: tuple[str, ...],
        split_sizes: Mapping[str, int],
        hlt_degradation_strength: float,
        primary_metric: str,
        selection_metric: str,
    ) -> None:
        if source_label_names != DUALVIEW_PART_SOURCE_LABEL_NAMES:
            raise ValueError("canonical dual-view branch is locked to source labels QCD/Hgg")
        if label_names != DUALVIEW_PART_SOURCE_LABEL_NAMES:
            raise ValueError("canonical dual-view branch is locked to label names QCD/Hgg")
        if downstream_label_filter_names != DUALVIEW_PART_BINARY_LABEL_FILTER_NAMES:
            raise ValueError("canonical dual-view branch expects compact downstream labels 0/1")
        if tuple(required_views) != DUALVIEW_PART_REQUIRED_VIEWS:
            raise ValueError("canonical dual-view branch requires exactly HLT and PN-reco views")
        if dict(split_sizes) != DUALVIEW_PART_SPLIT_SIZES:
            raise ValueError("canonical dual-view branch is locked to 500k/150k/500k split sizes")
        if abs(hlt_degradation_strength - DUALVIEW_PART_HLT_DEGRADATION_STRENGTH) > 1.0e-9:
            raise ValueError("canonical dual-view branch is locked to HLT degradation strength 0.6")
        if primary_metric != DUALVIEW_PART_PRIMARY_METRIC:
            raise ValueError("canonical dual-view branch uses FPR@50 as the primary metric")
        if selection_metric != DUALVIEW_PART_PRIMARY_METRIC:
            raise ValueError("canonical dual-view branch selects by FPR@50")

    @property
    def primary_metric_direction(self) -> str:
        return dualview_metric_direction(self.primary_metric)

    @property
    def selection_metric_direction(self) -> str:
        return dualview_metric_direction(self.selection_metric)

    @property
    def source_label_indices(self) -> tuple[int, ...]:
        return tuple(LABEL_NAMES.index(name) for name in self.source_label_names)

    @property
    def experiment_tag(self) -> str:
        return canonical_dualview_part_tag(hlt_degradation_strength=self.hlt_degradation_strength)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | "DualViewPartExperimentConfig" | None,
    ) -> "DualViewPartExperimentConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        field_names = {field.name for field in fields(cls)}
        clean_value = {key: field_value for key, field_value in value.items() if key not in _REPORT_ONLY_CONFIG_KEYS}
        unknown = sorted(set(clean_value) - field_names)
        if unknown:
            raise ValueError(f"Unknown DualViewPartExperimentConfig keys: {unknown}")
        return cls(**clean_value)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "contract": DUALVIEW_PART_CONTRACT,
                "experiment_step": DUALVIEW_PART_EXPERIMENT_STEP,
                "experiment_tag": self.experiment_tag,
                "source_label_indices": list(self.source_label_indices),
                "primary_metric_direction": self.primary_metric_direction,
                "selection_metric_direction": self.selection_metric_direction,
            }
        )
        payload["split_sizes"] = {split: int(self.split_sizes[split]) for split in DUALVIEW_PART_SPLIT_ORDER}
        return payload


@dataclass(frozen=True)
class DualViewPartExperimentLayout:
    """Path helper for the dual-view ParT branch."""

    output_root: str | Path = "checkpoints"
    experiment_name: str | None = None
    config: DualViewPartExperimentConfig = field(default_factory=DualViewPartExperimentConfig)

    def __post_init__(self) -> None:
        config = DualViewPartExperimentConfig.from_mapping(self.config)
        experiment_name = str(self.experiment_name or config.experiment_tag)
        if not experiment_name:
            raise ValueError("experiment_name must be non-empty")
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "experiment_name", experiment_name)

    @property
    def root(self) -> Path:
        return Path(self.output_root) / str(self.experiment_name)

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
    def pn_reconstructor_dir(self) -> Path:
        return self.root / "reconstructors" / self.config.pn_reconstructor_architecture

    @property
    def pn_reconstructed_view_dir(self) -> Path:
        return self.root / "reconstructed_views" / self.config.pn_reconstructor_architecture

    @property
    def anchor_dir(self) -> Path:
        return self.root / "anchor_hlt_part"

    @property
    def offline_reference_dir(self) -> Path:
        return self.root / "offline_teacher_reference"

    @property
    def taggers_dir(self) -> Path:
        return self.root / "taggers"

    def tagger_dir(self, variant: str) -> Path:
        return self.taggers_dir / normalize_dualview_part_variant(variant)

    @property
    def diagnostics_dir(self) -> Path:
        return self.root / "diagnostics"

    @property
    def final_report_dir(self) -> Path:
        return self.root / "final_report"

    def to_dict(self) -> dict[str, str]:
        return {
            "root": self.root.as_posix(),
            "binary_inputs_dir": self.binary_inputs_dir.as_posix(),
            "manifest_path": self.manifest_path.as_posix(),
            "hlt_cache_dir": self.hlt_cache_dir.as_posix(),
            "pn_reconstructor_dir": self.pn_reconstructor_dir.as_posix(),
            "pn_reconstructed_view_dir": self.pn_reconstructed_view_dir.as_posix(),
            "anchor_dir": self.anchor_dir.as_posix(),
            "offline_reference_dir": self.offline_reference_dir.as_posix(),
            "taggers_dir": self.taggers_dir.as_posix(),
            "diagnostics_dir": self.diagnostics_dir.as_posix(),
            "final_report_dir": self.final_report_dir.as_posix(),
        }


def default_dualview_part_config() -> DualViewPartExperimentConfig:
    return DualViewPartExperimentConfig()


def default_dualview_part_layout(
    *,
    output_root: str | Path = "checkpoints",
    experiment_name: str | None = None,
) -> DualViewPartExperimentLayout:
    return DualViewPartExperimentLayout(output_root=output_root, experiment_name=experiment_name)
