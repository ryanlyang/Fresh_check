"""Frozen protocol constants for the local-graph residual expert V2 path.

V2 is the strict residual-expert variant that must anchor on the exact trained
HLT ParT baseline and a true ParT penultimate embedding.  This module keeps the
task metadata and V2 contract names in one place so later cache/model/report
steps do not retype the QCD/Hgg HLT0.6 assumptions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

from .model import LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE
from .protocol import (
    LOCAL_GRAPH_PART_BINARY_LABEL_FILTER,
    LOCAL_GRAPH_PART_CONTRACT,
    LOCAL_GRAPH_PART_HLT_DEGRADATION_STRENGTH,
    LOCAL_GRAPH_PART_INFERENCE_VIEW,
    LOCAL_GRAPH_PART_PRIMARY_METRIC,
    LOCAL_GRAPH_PART_PROTOCOL_STEP,
    LOCAL_GRAPH_PART_SOURCE_LABEL_INDICES,
    LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES,
    LocalGraphPartProtocol,
    default_local_graph_part_protocol,
)


LOCAL_GRAPH_RESIDUAL_V2_PROTOCOL_STEP = "local_graph_residual_expert_v2_step1_protocol"
LOCAL_GRAPH_RESIDUAL_V2_PROTOCOL_CONTRACT = "local_graph_residual_expert_v2_protocol_v1"

LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_STEP = "local_graph_residual_expert_v2_step2_hlt_part_embedding_anchor"
LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_CONTRACT = "local_graph_residual_expert_v2_hlt_part_embedding_anchor_v1"
LOCAL_GRAPH_RESIDUAL_V2_CACHE_STEP = "local_graph_residual_expert_v2_step4_baseline_embedding_cache"
LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT = "local_graph_residual_expert_v2_baseline_embedding_cache_v1"
LOCAL_GRAPH_RESIDUAL_V2_MODEL_STEP = "local_graph_residual_expert_v2_step6_model"
LOCAL_GRAPH_RESIDUAL_V2_MODEL_CONTRACT = "local_graph_residual_expert_v2_model_v1"
LOCAL_GRAPH_RESIDUAL_V2_TRAIN_STEP = "local_graph_residual_expert_v2_step8_train"
LOCAL_GRAPH_RESIDUAL_V2_TRAIN_CONTRACT = "local_graph_residual_expert_v2_train_v1"
LOCAL_GRAPH_RESIDUAL_V2_REPORT_STEP = "local_graph_residual_expert_v2_step12_report"
LOCAL_GRAPH_RESIDUAL_V2_REPORT_CONTRACT = "local_graph_residual_expert_v2_report_v1"

LOCAL_GRAPH_RESIDUAL_V2_TASK_NAME = "qcd_vs_hgg_hlt06_residual_expert_v2"
LOCAL_GRAPH_RESIDUAL_V2_BASELINE_VARIANT = LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE
LOCAL_GRAPH_RESIDUAL_V2_POSITIVE_CLASS_NAME = "Hgg"
LOCAL_GRAPH_RESIDUAL_V2_POSITIVE_CLASS_INDEX = 1
LOCAL_GRAPH_RESIDUAL_V2_PRIMARY_METRIC = LOCAL_GRAPH_PART_PRIMARY_METRIC
LOCAL_GRAPH_RESIDUAL_V2_PRIMARY_METRIC_DIRECTION = "minimize"
LOCAL_GRAPH_RESIDUAL_V2_SELECTION_SPLIT = "model_val"
LOCAL_GRAPH_RESIDUAL_V2_TRAIN_SPLITS = ("model_train",)
LOCAL_GRAPH_RESIDUAL_V2_EVAL_SPLITS = ("model_val", "stack_train", "stack_val", "final_test")
LOCAL_GRAPH_RESIDUAL_V2_CACHE_SPLITS = (
    "model_train",
    "model_val",
    "stack_train",
    "stack_val",
    "final_test",
)

LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE = "residual_v2_weighted_bce"
LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE = "residual_v2_boundary_pairwise"
LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR = "residual_v2_boundary_pairwise_bce_anchor"
LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR = (
    "residual_v2_boundary_pairwise_soft_fpr_bce_anchor"
)
LOCAL_GRAPH_RESIDUAL_V2_TRAIN_LOSS_MODES = (
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE,
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR,
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR,
)
LOCAL_GRAPH_RESIDUAL_V2_ABLATION_LOSS_MODES = (LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE,)
LOCAL_GRAPH_RESIDUAL_V2_REPORT_POLICY_ALPHA_SHRINK = "validation_shrinkage_over_learned_correction"

LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_REQUIRED = True
LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE = "true_hlt_part_penultimate_embedding"
LOCAL_GRAPH_RESIDUAL_V2_DISALLOWED_EMBEDDING_FALLBACKS = (
    "widened_classifier_logits",
    "num_classes_embedding_proxy",
    "raw_hlt_summary_only",
    "local_graph_logits",
)


@dataclass(frozen=True)
class LocalGraphResidualV2ContractSpec:
    """One named output contract used by the V2 implementation ladder."""

    name: str
    step: str
    contract: str
    description: str


@dataclass(frozen=True)
class LocalGraphResidualV2Protocol:
    """Single source of truth for the V2 residual-expert experiment."""

    experiment_step: str = LOCAL_GRAPH_RESIDUAL_V2_PROTOCOL_STEP
    contract: str = LOCAL_GRAPH_RESIDUAL_V2_PROTOCOL_CONTRACT
    task_name: str = LOCAL_GRAPH_RESIDUAL_V2_TASK_NAME
    base_protocol_step: str = LOCAL_GRAPH_PART_PROTOCOL_STEP
    base_protocol_contract: str = LOCAL_GRAPH_PART_CONTRACT
    inference_view: str = LOCAL_GRAPH_PART_INFERENCE_VIEW
    hlt_degradation_strength: float = LOCAL_GRAPH_PART_HLT_DEGRADATION_STRENGTH
    label_names: tuple[str, ...] = LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES
    source_label_indices: tuple[int, ...] = LOCAL_GRAPH_PART_SOURCE_LABEL_INDICES
    binary_label_filter: tuple[int, ...] = LOCAL_GRAPH_PART_BINARY_LABEL_FILTER
    positive_class_name: str = LOCAL_GRAPH_RESIDUAL_V2_POSITIVE_CLASS_NAME
    positive_class_index: int = LOCAL_GRAPH_RESIDUAL_V2_POSITIVE_CLASS_INDEX
    num_classes: int = 2
    baseline_variant: str = LOCAL_GRAPH_RESIDUAL_V2_BASELINE_VARIANT
    baseline_must_be_exact_hlt_part: bool = True
    true_embedding_required: bool = LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_REQUIRED
    required_embedding_role: str = LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE
    disallowed_embedding_fallbacks: tuple[str, ...] = LOCAL_GRAPH_RESIDUAL_V2_DISALLOWED_EMBEDDING_FALLBACKS
    primary_metric: str = LOCAL_GRAPH_RESIDUAL_V2_PRIMARY_METRIC
    primary_metric_direction: str = LOCAL_GRAPH_RESIDUAL_V2_PRIMARY_METRIC_DIRECTION
    selection_metric: str = LOCAL_GRAPH_RESIDUAL_V2_PRIMARY_METRIC
    selection_split: str = LOCAL_GRAPH_RESIDUAL_V2_SELECTION_SPLIT
    train_splits: tuple[str, ...] = LOCAL_GRAPH_RESIDUAL_V2_TRAIN_SPLITS
    eval_splits: tuple[str, ...] = LOCAL_GRAPH_RESIDUAL_V2_EVAL_SPLITS
    cache_splits: tuple[str, ...] = LOCAL_GRAPH_RESIDUAL_V2_CACHE_SPLITS
    train_loss_modes: tuple[str, ...] = LOCAL_GRAPH_RESIDUAL_V2_TRAIN_LOSS_MODES
    ablation_loss_modes: tuple[str, ...] = LOCAL_GRAPH_RESIDUAL_V2_ABLATION_LOSS_MODES
    alpha_shrinkage_policy: str = LOCAL_GRAPH_RESIDUAL_V2_REPORT_POLICY_ALPHA_SHRINK
    contract_specs: tuple[LocalGraphResidualV2ContractSpec, ...] = (
        LocalGraphResidualV2ContractSpec(
            "anchor",
            LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_STEP,
            LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_CONTRACT,
            "Exact frozen HLT ParT baseline wrapper returning logits plus true penultimate embedding.",
        ),
        LocalGraphResidualV2ContractSpec(
            "cache",
            LOCAL_GRAPH_RESIDUAL_V2_CACHE_STEP,
            LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT,
            "Per-split frozen baseline logits, embeddings, condition features, and strict alignment metadata.",
        ),
        LocalGraphResidualV2ContractSpec(
            "model",
            LOCAL_GRAPH_RESIDUAL_V2_MODEL_STEP,
            LOCAL_GRAPH_RESIDUAL_V2_MODEL_CONTRACT,
            "Trainable local-graph residual expert consuming HLT tokens plus frozen ParT score/embedding.",
        ),
        LocalGraphResidualV2ContractSpec(
            "train",
            LOCAL_GRAPH_RESIDUAL_V2_TRAIN_STEP,
            LOCAL_GRAPH_RESIDUAL_V2_TRAIN_CONTRACT,
            "Model-train/model-val-only V2 training report selected by FPR@50.",
        ),
        LocalGraphResidualV2ContractSpec(
            "report",
            LOCAL_GRAPH_RESIDUAL_V2_REPORT_STEP,
            LOCAL_GRAPH_RESIDUAL_V2_REPORT_CONTRACT,
            "Final comparison report ranking rows by binary QCD/Hgg FPR@50.",
        ),
    )

    def validate(self) -> None:
        """Raise if V2 drifts from the strict QCD/Hgg HLT0.6 residual protocol."""

        base_protocol: LocalGraphPartProtocol = default_local_graph_part_protocol()
        if self.contract != LOCAL_GRAPH_RESIDUAL_V2_PROTOCOL_CONTRACT:
            raise ValueError(f"unexpected V2 protocol contract: {self.contract}")
        if self.base_protocol_step != base_protocol.experiment_step:
            raise ValueError("V2 must reference the frozen local-graph base protocol step")
        if self.base_protocol_contract != base_protocol.contract:
            raise ValueError("V2 must reference the frozen local-graph base protocol contract")
        if self.inference_view != base_protocol.inference_view:
            raise ValueError("V2 residual expert must be HLT-only at inference")
        if abs(float(self.hlt_degradation_strength) - float(base_protocol.hlt_degradation_strength)) > 1.0e-12:
            raise ValueError("V2 residual expert is frozen to HLT degradation strength 0.6")
        if tuple(self.label_names) != tuple(base_protocol.binary_label_names):
            raise ValueError("V2 residual expert is frozen to QCD/Hgg label names")
        if tuple(self.source_label_indices) != tuple(base_protocol.source_label_indices):
            raise ValueError("V2 source label ids must be original JetClass QCD=0, Hgg=3")
        if tuple(self.binary_label_filter) != tuple(base_protocol.binary_label_filter):
            raise ValueError("V2 binary cache labels must be remapped QCD=0, Hgg=1")
        if self.positive_class_name != LOCAL_GRAPH_RESIDUAL_V2_POSITIVE_CLASS_NAME:
            raise ValueError("V2 positive class must be Hgg")
        if int(self.positive_class_index) != LOCAL_GRAPH_RESIDUAL_V2_POSITIVE_CLASS_INDEX:
            raise ValueError("V2 positive class index must be 1")
        if int(self.num_classes) != 2:
            raise ValueError("V2 residual expert is binary only")
        if self.baseline_variant != LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE:
            raise ValueError("V2 baseline anchor must be the exact hlt_part_baseline variant")
        if not bool(self.baseline_must_be_exact_hlt_part):
            raise ValueError("V2 may not relax the exact HLT ParT baseline requirement")
        if not bool(self.true_embedding_required):
            raise ValueError("V2 requires a true HLT ParT embedding")
        if self.required_embedding_role != LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE:
            raise ValueError("V2 required embedding role changed unexpectedly")
        if "widened_classifier_logits" not in set(self.disallowed_embedding_fallbacks):
            raise ValueError("V2 must explicitly disallow widened-head embedding proxies")
        if self.primary_metric != LOCAL_GRAPH_PART_PRIMARY_METRIC:
            raise ValueError("V2 primary metric must be FPR@50")
        if self.primary_metric_direction != "minimize":
            raise ValueError("V2 FPR@50 must be minimized")
        if self.selection_metric != self.primary_metric:
            raise ValueError("V2 checkpoint selection must use the primary metric")
        if self.selection_split != "model_val":
            raise ValueError("V2 checkpoint selection must use model_val")
        if tuple(self.train_splits) != ("model_train",):
            raise ValueError("V2 residual training may only use model_train")
        if tuple(self.cache_splits) != LOCAL_GRAPH_RESIDUAL_V2_CACHE_SPLITS:
            raise ValueError("V2 cache splits changed unexpectedly")
        if set(self.train_loss_modes) != set(LOCAL_GRAPH_RESIDUAL_V2_TRAIN_LOSS_MODES):
            raise ValueError("V2 submitted train modes must be A/C/D only")
        if LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE in set(self.train_loss_modes):
            raise ValueError("V2 B mode is an ablation, not a default submitted training mode")
        if self.alpha_shrinkage_policy != LOCAL_GRAPH_RESIDUAL_V2_REPORT_POLICY_ALPHA_SHRINK:
            raise ValueError("V2 alpha shrinkage must be report-time validation shrinkage")
        names = {spec.name for spec in self.contract_specs}
        if names != {"anchor", "cache", "model", "train", "report"}:
            raise ValueError(f"unexpected V2 contract specs: {sorted(names)}")

    @property
    def contract_by_name(self) -> dict[str, str]:
        return {spec.name: spec.contract for spec in self.contract_specs}

    @property
    def step_by_name(self) -> dict[str, str]:
        return {spec.name: spec.step for spec in self.contract_specs}

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return json.loads(json.dumps(asdict(self), sort_keys=True))


def default_local_graph_residual_v2_protocol() -> LocalGraphResidualV2Protocol:
    """Return the strict V2 residual protocol after validation."""

    protocol = LocalGraphResidualV2Protocol()
    protocol.validate()
    return protocol


def local_graph_residual_v2_protocol_manifest() -> dict[str, Any]:
    """JSON-serializable V2 protocol manifest for future reports."""

    return default_local_graph_residual_v2_protocol().to_dict()


__all__ = [
    "LOCAL_GRAPH_RESIDUAL_V2_ABLATION_LOSS_MODES",
    "LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_CONTRACT",
    "LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_STEP",
    "LOCAL_GRAPH_RESIDUAL_V2_BASELINE_VARIANT",
    "LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT",
    "LOCAL_GRAPH_RESIDUAL_V2_CACHE_SPLITS",
    "LOCAL_GRAPH_RESIDUAL_V2_CACHE_STEP",
    "LOCAL_GRAPH_RESIDUAL_V2_DISALLOWED_EMBEDDING_FALLBACKS",
    "LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_REQUIRED",
    "LOCAL_GRAPH_RESIDUAL_V2_EVAL_SPLITS",
    "LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE",
    "LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR",
    "LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR",
    "LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE",
    "LOCAL_GRAPH_RESIDUAL_V2_MODEL_CONTRACT",
    "LOCAL_GRAPH_RESIDUAL_V2_MODEL_STEP",
    "LOCAL_GRAPH_RESIDUAL_V2_POSITIVE_CLASS_INDEX",
    "LOCAL_GRAPH_RESIDUAL_V2_POSITIVE_CLASS_NAME",
    "LOCAL_GRAPH_RESIDUAL_V2_PRIMARY_METRIC",
    "LOCAL_GRAPH_RESIDUAL_V2_PRIMARY_METRIC_DIRECTION",
    "LOCAL_GRAPH_RESIDUAL_V2_PROTOCOL_CONTRACT",
    "LOCAL_GRAPH_RESIDUAL_V2_PROTOCOL_STEP",
    "LOCAL_GRAPH_RESIDUAL_V2_REPORT_CONTRACT",
    "LOCAL_GRAPH_RESIDUAL_V2_REPORT_POLICY_ALPHA_SHRINK",
    "LOCAL_GRAPH_RESIDUAL_V2_REPORT_STEP",
    "LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE",
    "LOCAL_GRAPH_RESIDUAL_V2_SELECTION_SPLIT",
    "LOCAL_GRAPH_RESIDUAL_V2_TASK_NAME",
    "LOCAL_GRAPH_RESIDUAL_V2_TRAIN_CONTRACT",
    "LOCAL_GRAPH_RESIDUAL_V2_TRAIN_LOSS_MODES",
    "LOCAL_GRAPH_RESIDUAL_V2_TRAIN_SPLITS",
    "LOCAL_GRAPH_RESIDUAL_V2_TRAIN_STEP",
    "LocalGraphResidualV2ContractSpec",
    "LocalGraphResidualV2Protocol",
    "default_local_graph_residual_v2_protocol",
    "local_graph_residual_v2_protocol_manifest",
]
