"""Frozen protocol for the QCD-vs-Hgg local graph Particle Transformer runs.

The point of this module is intentionally modest: every runner/report for the
local-graph work should import this contract instead of retyping labels, split
names, split sizes, and metric direction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

from jetclass_fresh.jetclass_data import LABEL_NAMES


LOCAL_GRAPH_PART_PROTOCOL_STEP = "local_graph_part_step1_protocol"
LOCAL_GRAPH_PART_CONTRACT = "local_graph_part_qcd_hgg_hlt06_protocol_v1"

LOCAL_GRAPH_PART_INFERENCE_VIEW = "hlt"
LOCAL_GRAPH_PART_HLT_DEGRADATION_STRENGTH = 0.6

LOCAL_GRAPH_PART_BACKGROUND_LABEL = "QCD"
LOCAL_GRAPH_PART_SIGNAL_LABEL = "Hgg"
LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES = (LOCAL_GRAPH_PART_BACKGROUND_LABEL, LOCAL_GRAPH_PART_SIGNAL_LABEL)
LOCAL_GRAPH_PART_SOURCE_LABEL_INDICES = (0, 3)

# After building the binary QCD/Hgg cache/manifest, labels are remapped to
# QCD=0 and Hgg=1. Keep this separate from the original JetClass ids above.
LOCAL_GRAPH_PART_BINARY_LABEL_FILTER = (0, 1)

LOCAL_GRAPH_PART_PRIMARY_METRIC = "fpr_at_signal_eff_0p50"
LOCAL_GRAPH_PART_DEFAULT_VARIANTS = (
    "hlt_part_baseline",
    "local_edgeconv_adapter",
    "local_point_attention_adapter",
    "local_point_attention_adapter_warmstart",
)


@dataclass(frozen=True)
class LocalGraphSplitSpec:
    """One named split and its intended maximum jet count."""

    name: str
    max_jets: int
    role: str


@dataclass(frozen=True)
class LocalGraphMetricSpec:
    """Metric name plus comparison direction."""

    name: str
    direction: str
    description: str
    required_on_final_test: bool = True


@dataclass(frozen=True)
class LocalGraphVariantSpec:
    """A model variant that the first serious runner should understand."""

    name: str
    role: str
    required: bool = True


@dataclass(frozen=True)
class LocalGraphPartProtocol:
    """Single source of truth for the local-graph QCD/Hgg experiment."""

    experiment_step: str = LOCAL_GRAPH_PART_PROTOCOL_STEP
    contract: str = LOCAL_GRAPH_PART_CONTRACT
    task_name: str = "qcd_vs_hgg_hlt06"
    inference_view: str = LOCAL_GRAPH_PART_INFERENCE_VIEW
    offline_view_allowed_at_inference: bool = False
    hlt_degradation_strength: float = LOCAL_GRAPH_PART_HLT_DEGRADATION_STRENGTH
    background_label: str = LOCAL_GRAPH_PART_BACKGROUND_LABEL
    signal_label: str = LOCAL_GRAPH_PART_SIGNAL_LABEL
    source_label_names: tuple[str, ...] = LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES
    source_label_indices: tuple[int, ...] = LOCAL_GRAPH_PART_SOURCE_LABEL_INDICES
    binary_label_filter: tuple[int, ...] = LOCAL_GRAPH_PART_BINARY_LABEL_FILTER
    binary_label_names: tuple[str, ...] = LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES
    num_classes: int = 2
    primary_metric: str = LOCAL_GRAPH_PART_PRIMARY_METRIC
    selection_metric: str = LOCAL_GRAPH_PART_PRIMARY_METRIC
    comparison_split: str = "final_test"
    confirm_final_test: bool = True
    train_epochs: int = 45
    split_specs: tuple[LocalGraphSplitSpec, ...] = (
        LocalGraphSplitSpec("model_train", 500_000, "train model weights"),
        LocalGraphSplitSpec("model_val", 150_000, "select checkpoints"),
        LocalGraphSplitSpec("stack_train", 500_000, "reserved for stacked/comparison training"),
        LocalGraphSplitSpec("stack_val", 150_000, "report unbiased validation diagnostics"),
        LocalGraphSplitSpec("final_test", 500_000, "final held-out comparison"),
    )
    metric_specs: tuple[LocalGraphMetricSpec, ...] = (
        LocalGraphMetricSpec(
            "fpr_at_signal_eff_0p50",
            "minimize",
            "Primary binary tagging metric: background false-positive rate at 50% Hgg efficiency.",
        ),
        LocalGraphMetricSpec(
            "background_rejection_at_signal_eff_0p50",
            "maximize",
            "Equivalent rejection view of the primary metric.",
        ),
        LocalGraphMetricSpec(
            "fpr_at_signal_eff_0p30",
            "minimize",
            "Secondary low-signal-efficiency false-positive rate.",
        ),
        LocalGraphMetricSpec("auc", "maximize", "Secondary ranking metric."),
        LocalGraphMetricSpec("accuracy", "maximize", "Sanity metric only; not the model-selection target."),
    )
    variant_specs: tuple[LocalGraphVariantSpec, ...] = (
        LocalGraphVariantSpec("hlt_part_baseline", "HLT ParT trained on the same HLT cache and splits."),
        LocalGraphVariantSpec("local_edgeconv_adapter", "Residual EdgeConv local graph adapter."),
        LocalGraphVariantSpec("local_point_attention_adapter", "Residual point-attention local graph adapter."),
        LocalGraphVariantSpec(
            "local_point_attention_adapter_warmstart",
            "Point-attention adapter initialized from, or compared against, the HLT ParT baseline.",
        ),
    )

    def validate(self) -> None:
        """Raise if the frozen protocol drifts from the intended experiment."""

        if self.contract != LOCAL_GRAPH_PART_CONTRACT:
            raise ValueError(f"unexpected local graph contract: {self.contract}")
        if self.inference_view != "hlt" or bool(self.offline_view_allowed_at_inference):
            raise ValueError("local graph protocol must be HLT-only at inference")
        if abs(float(self.hlt_degradation_strength) - LOCAL_GRAPH_PART_HLT_DEGRADATION_STRENGTH) > 1.0e-12:
            raise ValueError("local graph protocol is frozen to HLT degradation strength 0.6")
        if tuple(self.source_label_names) != LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES:
            raise ValueError("local graph protocol is frozen to QCD vs Hgg")
        if tuple(self.source_label_indices) != LOCAL_GRAPH_PART_SOURCE_LABEL_INDICES:
            raise ValueError("source label ids must be original JetClass QCD=0, Hgg=3")
        for name, index in zip(self.source_label_names, self.source_label_indices):
            if LABEL_NAMES[int(index)] != name:
                raise ValueError(f"JetClass label id mismatch for {name}: got {index}")
        if tuple(self.binary_label_filter) != LOCAL_GRAPH_PART_BINARY_LABEL_FILTER:
            raise ValueError("binary label filter must be remapped QCD=0, Hgg=1")
        if int(self.num_classes) != 2:
            raise ValueError("local graph protocol is binary only")
        if self.primary_metric != LOCAL_GRAPH_PART_PRIMARY_METRIC:
            raise ValueError("primary metric must be FPR@50")
        if self.selection_metric != self.primary_metric:
            raise ValueError("checkpoint selection must use the primary metric")
        if self.comparison_split != "final_test" or not bool(self.confirm_final_test):
            raise ValueError("final-test confirmation is required")
        required_splits = {
            "model_train": 500_000,
            "model_val": 150_000,
            "stack_train": 500_000,
            "stack_val": 150_000,
            "final_test": 500_000,
        }
        if self.split_size_by_name != required_splits:
            raise ValueError(f"unexpected split protocol: {self.split_size_by_name}")
        metric_direction = {metric.name: metric.direction for metric in self.metric_specs}
        if metric_direction.get(self.primary_metric) != "minimize":
            raise ValueError("FPR@50 must be minimized")
        variant_names = tuple(variant.name for variant in self.variant_specs if variant.required)
        if variant_names != LOCAL_GRAPH_PART_DEFAULT_VARIANTS:
            raise ValueError(f"unexpected required variants: {variant_names}")

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


def default_local_graph_part_protocol() -> LocalGraphPartProtocol:
    """Return the frozen local-graph protocol after validation."""

    protocol = LocalGraphPartProtocol()
    protocol.validate()
    return protocol


def local_graph_part_protocol_manifest() -> dict[str, Any]:
    """JSON-serializable protocol manifest for run reports."""

    return default_local_graph_part_protocol().to_dict()
