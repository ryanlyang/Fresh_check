"""Frozen protocol for multi-scale subjet HLT ParT experiments.

Every runner/report for this branch should import this contract instead of
retyping labels, split sizes, variants, or metric direction.  The protocol is
intentionally narrow: QCD vs Hgg, HLT-only inference, fixed-HLT degradation
strength 0.6, and model selection by FPR@50.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

from jetclass_fresh.jetclass_data import LABEL_NAMES


MULTISCALE_SUBJET_PROTOCOL_STEP = "multiscale_subjet_part_step1_protocol"
MULTISCALE_SUBJET_CONTRACT = "multiscale_subjet_part_qcd_hgg_hlt06_protocol_v1"

MULTISCALE_SUBJET_INFERENCE_VIEW = "hlt"
MULTISCALE_SUBJET_HLT_DEGRADATION_STRENGTH = 0.6

MULTISCALE_SUBJET_BACKGROUND_LABEL = "QCD"
MULTISCALE_SUBJET_SIGNAL_LABEL = "Hgg"
MULTISCALE_SUBJET_SOURCE_LABEL_NAMES = (
    MULTISCALE_SUBJET_BACKGROUND_LABEL,
    MULTISCALE_SUBJET_SIGNAL_LABEL,
)
MULTISCALE_SUBJET_SOURCE_LABEL_INDICES = (0, 3)

# Binary HLT caches/remapped manifests should map QCD -> 0 and Hgg -> 1.
MULTISCALE_SUBJET_BINARY_LABEL_FILTER = (0, 1)

MULTISCALE_SUBJET_PRIMARY_METRIC = "fpr_at_signal_eff_0p50"
MULTISCALE_SUBJET_BASELINE_VARIANT = "hlt_part_baseline"
MULTISCALE_SUBJET_PRIMARY_VARIANT = "multiscale_subjet_residual_part_adapter"
MULTISCALE_SUBJET_DEFAULT_VARIANTS = (
    MULTISCALE_SUBJET_BASELINE_VARIANT,
    MULTISCALE_SUBJET_PRIMARY_VARIANT,
    "pure_perceiver_latent_control",
    "part_plus_random_subjet_control",
)
MULTISCALE_SUBJET_EXTENDED_CONTROL_VARIANTS = (
    "subjet_branch_only",
    "part_plus_single_scale_subjets",
    "part_plus_multiscale_subjets_cls_fusion_control",
    "part_plus_multiscale_subjets_late_fusion_control",
    "residual_adapter_no_physics_bias",
    "larger_hlt_part_control",
    "two_part_ensemble_control",
)


@dataclass(frozen=True)
class MultiscaleSubjetSplitSpec:
    """One named split and its intended maximum jet count."""

    name: str
    max_jets: int
    role: str


@dataclass(frozen=True)
class MultiscaleSubjetMetricSpec:
    """Metric name plus comparison direction."""

    name: str
    direction: str
    description: str
    required_on_final_test: bool = True


@dataclass(frozen=True)
class MultiscaleSubjetVariantSpec:
    """A model/control variant planned for this branch."""

    name: str
    role: str
    required: bool = True


@dataclass(frozen=True)
class MultiscaleSubjetPartProtocol:
    """Single source of truth for the QCD/Hgg HLT0.6 subjet experiment."""

    experiment_step: str = MULTISCALE_SUBJET_PROTOCOL_STEP
    contract: str = MULTISCALE_SUBJET_CONTRACT
    task_name: str = "qcd_vs_hgg_hlt06_multiscale_subjet"
    inference_view: str = MULTISCALE_SUBJET_INFERENCE_VIEW
    offline_view_allowed_at_inference: bool = False
    hlt_degradation_strength: float = MULTISCALE_SUBJET_HLT_DEGRADATION_STRENGTH
    background_label: str = MULTISCALE_SUBJET_BACKGROUND_LABEL
    signal_label: str = MULTISCALE_SUBJET_SIGNAL_LABEL
    source_label_names: tuple[str, ...] = MULTISCALE_SUBJET_SOURCE_LABEL_NAMES
    source_label_indices: tuple[int, ...] = MULTISCALE_SUBJET_SOURCE_LABEL_INDICES
    binary_label_filter: tuple[int, ...] = MULTISCALE_SUBJET_BINARY_LABEL_FILTER
    binary_label_names: tuple[str, ...] = MULTISCALE_SUBJET_SOURCE_LABEL_NAMES
    num_classes: int = 2
    primary_metric: str = MULTISCALE_SUBJET_PRIMARY_METRIC
    selection_metric: str = MULTISCALE_SUBJET_PRIMARY_METRIC
    selection_metric_direction: str = "minimize"
    comparison_split: str = "final_test"
    confirm_final_test: bool = True
    train_epochs: int = 45
    baseline_variant: str = MULTISCALE_SUBJET_BASELINE_VARIANT
    primary_variant: str = MULTISCALE_SUBJET_PRIMARY_VARIANT
    main_architecture: str = "zero_init_residual_subjet_adapter_into_reference_hlt_part"
    integration_mode: str = "residual_feature_adapter"
    split_specs: tuple[MultiscaleSubjetSplitSpec, ...] = (
        MultiscaleSubjetSplitSpec("model_train", 500_000, "train model weights"),
        MultiscaleSubjetSplitSpec("model_val", 150_000, "select checkpoints"),
        MultiscaleSubjetSplitSpec("stack_train", 500_000, "reserved for fusion/control training"),
        MultiscaleSubjetSplitSpec("stack_val", 150_000, "report validation diagnostics"),
        MultiscaleSubjetSplitSpec("final_test", 500_000, "final held-out comparison"),
    )
    metric_specs: tuple[MultiscaleSubjetMetricSpec, ...] = (
        MultiscaleSubjetMetricSpec(
            "fpr_at_signal_eff_0p50",
            "minimize",
            "Primary binary metric: QCD false-positive rate at 50% Hgg efficiency.",
        ),
        MultiscaleSubjetMetricSpec(
            "background_rejection_at_signal_eff_0p50",
            "maximize",
            "Equivalent rejection view of the primary metric.",
        ),
        MultiscaleSubjetMetricSpec(
            "fpr_at_signal_eff_0p30",
            "minimize",
            "Secondary low-signal-efficiency false-positive rate.",
        ),
        MultiscaleSubjetMetricSpec("auc", "maximize", "Secondary ranking metric."),
        MultiscaleSubjetMetricSpec("accuracy", "maximize", "Sanity metric only; not for checkpoint selection."),
        MultiscaleSubjetMetricSpec(
            "validation_threshold_final_test_fpr",
            "minimize",
            "Final-test FPR after choosing threshold on model_val at 50% signal efficiency.",
        ),
    )
    variant_specs: tuple[MultiscaleSubjetVariantSpec, ...] = (
        MultiscaleSubjetVariantSpec("hlt_part_baseline", "Exact HLT ParT baseline on the same cache/splits."),
        MultiscaleSubjetVariantSpec(
            "multiscale_subjet_residual_part_adapter",
            "Primary zero-init residual subjet adapter feeding the real warm-started HLT ParT.",
        ),
        MultiscaleSubjetVariantSpec(
            "pure_perceiver_latent_control",
            "Same broad latent-token capacity without seeded/physics-subjet structure.",
        ),
        MultiscaleSubjetVariantSpec(
            "part_plus_random_subjet_control",
            "Same-parameter control with random/permuted subjet assignment structure.",
        ),
        MultiscaleSubjetVariantSpec("subjet_branch_only", "Subjet branch without ParT anchor.", required=False),
        MultiscaleSubjetVariantSpec("part_plus_single_scale_subjets", "Single-scale hierarchy ablation.", required=False),
        MultiscaleSubjetVariantSpec(
            "part_plus_multiscale_subjets_cls_fusion_control",
            "CLS/embedding fusion control; not the mainline architecture.",
            required=False,
        ),
        MultiscaleSubjetVariantSpec(
            "part_plus_multiscale_subjets_late_fusion_control",
            "Late logit fusion control; ensemble-like sanity path.",
            required=False,
        ),
        MultiscaleSubjetVariantSpec(
            "residual_adapter_no_physics_bias",
            "Residual adapter with pair/subjet physics biases removed.",
            required=False,
        ),
        MultiscaleSubjetVariantSpec(
            "larger_hlt_part_control",
            "Parameter/compute-matched larger HLT ParT control.",
            required=False,
        ),
        MultiscaleSubjetVariantSpec(
            "two_part_ensemble_control",
            "Two independently trained HLT ParT models fused as an ensemble control.",
            required=False,
        ),
    )

    def validate(self) -> None:
        """Raise if the frozen protocol drifts from the intended experiment."""

        if self.contract != MULTISCALE_SUBJET_CONTRACT:
            raise ValueError(f"unexpected multiscale subjet contract: {self.contract}")
        if self.inference_view != "hlt" or bool(self.offline_view_allowed_at_inference):
            raise ValueError("multi-scale subjet protocol must be HLT-only at inference")
        if abs(float(self.hlt_degradation_strength) - MULTISCALE_SUBJET_HLT_DEGRADATION_STRENGTH) > 1.0e-12:
            raise ValueError("multi-scale subjet protocol is frozen to HLT degradation strength 0.6")
        if tuple(self.source_label_names) != MULTISCALE_SUBJET_SOURCE_LABEL_NAMES:
            raise ValueError("multi-scale subjet protocol is frozen to QCD vs Hgg")
        if tuple(self.source_label_indices) != MULTISCALE_SUBJET_SOURCE_LABEL_INDICES:
            raise ValueError("source label ids must be original JetClass QCD=0, Hgg=3")
        for name, index in zip(self.source_label_names, self.source_label_indices):
            if LABEL_NAMES[int(index)] != name:
                raise ValueError(f"JetClass label id mismatch for {name}: got {index}")
        if tuple(self.binary_label_filter) != MULTISCALE_SUBJET_BINARY_LABEL_FILTER:
            raise ValueError("binary label filter must be remapped QCD=0, Hgg=1")
        if int(self.num_classes) != 2:
            raise ValueError("multi-scale subjet protocol is binary only")
        if self.primary_metric != MULTISCALE_SUBJET_PRIMARY_METRIC:
            raise ValueError("primary metric must be FPR@50")
        if self.selection_metric != self.primary_metric:
            raise ValueError("checkpoint selection must use the primary metric")
        if self.selection_metric_direction != "minimize":
            raise ValueError("FPR@50 selection direction must be minimize")
        if self.comparison_split != "final_test" or not bool(self.confirm_final_test):
            raise ValueError("final-test confirmation is required")
        if self.baseline_variant != MULTISCALE_SUBJET_BASELINE_VARIANT:
            raise ValueError("baseline variant must be exact HLT ParT")
        if self.primary_variant != MULTISCALE_SUBJET_PRIMARY_VARIANT:
            raise ValueError("primary variant must be the residual subjet ParT adapter")
        if self.integration_mode != "residual_feature_adapter":
            raise ValueError("primary integration mode must be residual_feature_adapter")
        expected_splits = {
            "model_train": 500_000,
            "model_val": 150_000,
            "stack_train": 500_000,
            "stack_val": 150_000,
            "final_test": 500_000,
        }
        if self.split_size_by_name != expected_splits:
            raise ValueError(f"unexpected split protocol: {self.split_size_by_name}")
        metric_direction = {metric.name: metric.direction for metric in self.metric_specs}
        if metric_direction.get(self.primary_metric) != "minimize":
            raise ValueError("FPR@50 must be minimized")
        if metric_direction.get("background_rejection_at_signal_eff_0p50") != "maximize":
            raise ValueError("background rejection must be maximized")
        variant_names = tuple(variant.name for variant in self.variant_specs if variant.required)
        if variant_names != MULTISCALE_SUBJET_DEFAULT_VARIANTS:
            raise ValueError(f"unexpected required variants: {variant_names}")
        primary_count = sum(1 for variant in self.variant_specs if variant.name == self.primary_variant)
        baseline_count = sum(1 for variant in self.variant_specs if variant.name == self.baseline_variant)
        if primary_count != 1 or baseline_count != 1:
            raise ValueError("protocol must define exactly one primary variant and one baseline variant")

    @property
    def split_size_by_name(self) -> dict[str, int]:
        return {spec.name: int(spec.max_jets) for spec in self.split_specs}

    @property
    def metric_direction_by_name(self) -> dict[str, str]:
        return {spec.name: spec.direction for spec in self.metric_specs}

    @property
    def required_variant_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.variant_specs if spec.required)

    @property
    def optional_variant_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.variant_specs if not spec.required)

    @property
    def control_variant_names(self) -> tuple[str, ...]:
        return self.optional_variant_names

    @property
    def variant_role_by_name(self) -> dict[str, str]:
        return {spec.name: spec.role for spec in self.variant_specs}

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return json.loads(json.dumps(asdict(self), sort_keys=True))


def default_multiscale_subjet_part_protocol() -> MultiscaleSubjetPartProtocol:
    """Return the frozen multi-scale subjet protocol after validation."""

    protocol = MultiscaleSubjetPartProtocol()
    protocol.validate()
    return protocol


def multiscale_subjet_part_protocol_manifest() -> dict[str, Any]:
    """JSON-serializable protocol manifest for run reports."""

    return default_multiscale_subjet_part_protocol().to_dict()


# Public aliases use the explicit ``*_PART_*`` prefix because later training
# code should make clear that this is the Particle Transformer comparison
# protocol, not a generic subjet utility package.
MULTISCALE_SUBJET_PART_PROTOCOL_STEP = MULTISCALE_SUBJET_PROTOCOL_STEP
MULTISCALE_SUBJET_PART_CONTRACT = MULTISCALE_SUBJET_CONTRACT
MULTISCALE_SUBJET_PART_INFERENCE_VIEW = MULTISCALE_SUBJET_INFERENCE_VIEW
MULTISCALE_SUBJET_PART_HLT_DEGRADATION_STRENGTH = MULTISCALE_SUBJET_HLT_DEGRADATION_STRENGTH
MULTISCALE_SUBJET_PART_BACKGROUND_LABEL = MULTISCALE_SUBJET_BACKGROUND_LABEL
MULTISCALE_SUBJET_PART_SIGNAL_LABEL = MULTISCALE_SUBJET_SIGNAL_LABEL
MULTISCALE_SUBJET_PART_SOURCE_LABEL_NAMES = MULTISCALE_SUBJET_SOURCE_LABEL_NAMES
MULTISCALE_SUBJET_PART_SOURCE_LABEL_INDICES = MULTISCALE_SUBJET_SOURCE_LABEL_INDICES
MULTISCALE_SUBJET_PART_BINARY_LABEL_FILTER = MULTISCALE_SUBJET_BINARY_LABEL_FILTER
MULTISCALE_SUBJET_PART_PRIMARY_METRIC = MULTISCALE_SUBJET_PRIMARY_METRIC
MULTISCALE_SUBJET_PART_BASELINE_VARIANT = MULTISCALE_SUBJET_BASELINE_VARIANT
MULTISCALE_SUBJET_PART_PRIMARY_VARIANT = MULTISCALE_SUBJET_PRIMARY_VARIANT
MULTISCALE_SUBJET_PART_DEFAULT_VARIANTS = MULTISCALE_SUBJET_DEFAULT_VARIANTS
MULTISCALE_SUBJET_PART_CONTROL_VARIANTS = MULTISCALE_SUBJET_EXTENDED_CONTROL_VARIANTS

MultiScaleSubjetSplitSpec = MultiscaleSubjetSplitSpec
MultiScaleSubjetMetricSpec = MultiscaleSubjetMetricSpec
MultiScaleSubjetVariantSpec = MultiscaleSubjetVariantSpec
MultiScaleSubjetPartProtocol = MultiscaleSubjetPartProtocol
