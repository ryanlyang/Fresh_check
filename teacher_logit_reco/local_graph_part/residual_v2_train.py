"""Training loop for local-graph residual expert V2.

V2 trains only the residual expert around a frozen exact HLT ParT baseline.
The baseline logits, true ParT penultimate embeddings, and condition features
come from the strict Step 4/5 V2 cache.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json, set_training_seed
from jetclass_fresh.hlt_cache import load_cached_hlt_view
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.set_matching.five_view_train import classification_metrics_from_predictions
from teacher_logit_reco.set_matching.train import source_metadata
from teacher_logit_reco.subtoken_part.train import SubtokenHLTJetDataset, make_subtoken_hlt_loader

from .fusion import binary_logits_from_log_odds, binary_margin_from_logits
from .protocol import (
    LOCAL_GRAPH_PART_BINARY_LABEL_FILTER,
    LOCAL_GRAPH_PART_CONTRACT,
    LOCAL_GRAPH_PART_HLT_DEGRADATION_STRENGTH,
    LOCAL_GRAPH_PART_PRIMARY_METRIC,
    LOCAL_GRAPH_PART_PROTOCOL_STEP,
    LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES,
)
from .residual_v2_cache import (
    LocalGraphResidualV2BaselineEmbeddingBlock,
    load_residual_v2_embedding_block,
    verify_residual_v2_embedding_block_alignment,
    verify_residual_v2_embedding_cache_family,
)
from .residual_v2_losses import (
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_STEP,
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE,
    LocalGraphResidualV2LossConfig,
    compute_local_graph_residual_v2_loss_from_output,
    normalize_local_graph_residual_v2_loss_mode,
)
from .residual_losses import select_alpha_shrinkage
from .residual_v2_model import (
    LOCAL_GRAPH_RESIDUAL_V2_INPUT_FULL,
    LOCAL_GRAPH_RESIDUAL_V2_INPUT_MODES,
    LOCAL_GRAPH_RESIDUAL_V2_POOL_MEAN_MAX,
    LocalGraphResidualExpertV2,
    LocalGraphResidualExpertV2Config,
    build_local_graph_residual_expert_v2,
)
from .residual_v2_protocol import (
    LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_MODEL_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_MODEL_STEP,
    LOCAL_GRAPH_RESIDUAL_V2_PRIMARY_METRIC,
    LOCAL_GRAPH_RESIDUAL_V2_PRIMARY_METRIC_DIRECTION,
    LOCAL_GRAPH_RESIDUAL_V2_PROTOCOL_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_PROTOCOL_STEP,
    LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE,
    LOCAL_GRAPH_RESIDUAL_V2_SELECTION_SPLIT,
    LOCAL_GRAPH_RESIDUAL_V2_TRAIN_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_TRAIN_SPLITS,
    LOCAL_GRAPH_RESIDUAL_V2_TRAIN_STEP,
    default_local_graph_residual_v2_protocol,
    local_graph_residual_v2_protocol_manifest,
)
from .train import LOCAL_GRAPH_LOWER_IS_BETTER_SELECTION_METRICS, _verify_hlt_cache_protocol


LOCAL_GRAPH_RESIDUAL_V2_SELECTION_METRIC = LOCAL_GRAPH_PART_PRIMARY_METRIC
LOCAL_GRAPH_RESIDUAL_V2_LOWER_IS_BETTER_SELECTION_METRICS = LOCAL_GRAPH_LOWER_IS_BETTER_SELECTION_METRICS
LOCAL_GRAPH_RESIDUAL_V2_GAMMA_SHRINKAGE_GRID = (
    0.0,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.35,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
)
LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_NORMAL = "normal"
LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_SHUFFLED = "shuffled"
LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_MODES = (
    LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_NORMAL,
    LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_SHUFFLED,
)
LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_NORMAL = "normal"
LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_SHUFFLED = "shuffled"
LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_MODES = (
    LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_NORMAL,
    LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_SHUFFLED,
)


def _optional_nonnegative_int(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative when provided")
    return value


def _optional_positive_int(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive when provided")
    return value


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float(default)
    return output if np.isfinite(output) else float(default)


def _move_batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("tokens", "mask", "labels", "indices"):
        value = moved.get(key)
        if hasattr(value, "to"):
            moved[key] = value.to(device, non_blocking=True)
    return moved


def _flatten_scalar_diagnostics(diagnostics: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in diagnostics.items():
        if hasattr(value, "detach"):
            try:
                if int(value.numel()) == 1:
                    output[str(key)] = float(value.detach().cpu().item())
            except Exception:
                continue
        elif isinstance(value, (int, float)):
            numeric = float(value)
            if np.isfinite(numeric):
                output[str(key)] = numeric
    return output


def _selection_score(metrics: Mapping[str, Any], metric_name: str) -> tuple[float, float]:
    if metric_name in metrics:
        value = _finite_float(metrics.get(metric_name), default=float("nan"))
    else:
        binary = metrics.get("binary_metrics")
        if isinstance(binary, Mapping) and metric_name in binary:
            value = _finite_float(binary.get(metric_name), default=float("nan"))
        else:
            raise KeyError(f"validation metrics do not contain selection metric {metric_name!r}")
    if np.isnan(value):
        return float("-inf"), value
    if metric_name in LOCAL_GRAPH_RESIDUAL_V2_LOWER_IS_BETTER_SELECTION_METRICS:
        return -float(value), float(value)
    return float(value), float(value)


def _metrics_from_binary_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    loss_sum: float | None = None,
    label_names: tuple[str, ...] = LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES,
) -> dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.argmax(logits, axis=1).astype(np.int64) if logits.size else np.asarray([], dtype=np.int64)
    metrics = classification_metrics_from_predictions(
        preds=preds,
        labels=labels,
        loss_sum=loss_sum,
        logits=logits,
        label_names=label_names,
    )
    binary = metrics.get("binary_metrics")
    if isinstance(binary, Mapping):
        for key in (
            "auc",
            "fpr_at_signal_eff_0p30",
            "fpr_at_signal_eff_0p50",
            "background_rejection_at_signal_eff_0p30",
            "background_rejection_at_signal_eff_0p50",
        ):
            if key in binary and key not in metrics:
                metrics[key] = binary[key]
    return metrics


def _strip_prediction_arrays(metrics: Mapping[str, Any]) -> dict[str, Any]:
    clean = dict(metrics)
    clean.pop("_prediction_arrays", None)
    return clean


def select_local_graph_residual_v2_gamma_shrinkage(
    *,
    labels: Any,
    baseline_logits: Any,
    correction_logits: Any,
    gamma_grid: tuple[float, ...] = LOCAL_GRAPH_RESIDUAL_V2_GAMMA_SHRINKAGE_GRID,
) -> dict[str, Any]:
    """Select validation gamma over the learned correction delta.

    The correction logits are already ``learned_gamma * delta``.  The selected
    report-time gamma therefore scales the learned correction, not the raw
    residual head output.
    """

    labels_np = np.asarray(labels, dtype=np.int64).reshape(-1)
    baseline_margin = binary_margin_from_logits(np.asarray(baseline_logits, dtype=np.float32))
    learned_correction = binary_margin_from_logits(np.asarray(correction_logits, dtype=np.float32))
    report = select_alpha_shrinkage(
        labels=labels_np,
        baseline_logit=baseline_margin,
        residual_logit=learned_correction,
        alpha_grid=gamma_grid,
        target_signal_efficiency=0.50,
    )
    selected_gamma = float(report["selected_alpha"])
    report.update(
        {
            "v2_step": LOCAL_GRAPH_RESIDUAL_V2_TRAIN_STEP,
            "v2_contract": LOCAL_GRAPH_RESIDUAL_V2_TRAIN_CONTRACT,
            "selected_gamma": selected_gamma,
            "gamma_grid": [float(gamma) for gamma in gamma_grid],
            "shrinkage_applies_to": "learned_correction_delta",
            "score_formula": "z_base + gamma_val * (learned_gamma * delta)",
        }
    )
    return report


def _shrunk_logits_from_prediction_arrays(prediction_arrays: Mapping[str, Any], gamma: float) -> np.ndarray:
    baseline_logits = np.asarray(prediction_arrays.get("baseline_logits"), dtype=np.float32)
    correction_logits = np.asarray(prediction_arrays.get("correction_logits"), dtype=np.float32)
    baseline_margin = binary_margin_from_logits(baseline_logits)
    learned_correction = binary_margin_from_logits(correction_logits)
    return binary_logits_from_log_odds(baseline_margin + float(gamma) * learned_correction).astype(np.float32)


def _model_val_gamma_shrinkage_from_predictions(
    prediction_arrays: Mapping[str, Any],
    *,
    label_names: tuple[str, ...],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    labels_np = np.asarray(prediction_arrays.get("labels"), dtype=np.int64)
    baseline_logits = np.asarray(prediction_arrays.get("baseline_logits"), dtype=np.float32)
    correction_logits = np.asarray(prediction_arrays.get("correction_logits"), dtype=np.float32)
    if labels_np.ndim != 1 or baseline_logits.ndim != 2 or correction_logits.ndim != 2:
        return None, None
    gamma_report = select_local_graph_residual_v2_gamma_shrinkage(
        labels=labels_np,
        baseline_logits=baseline_logits,
        correction_logits=correction_logits,
    )
    shrunk_logits = _shrunk_logits_from_prediction_arrays(
        prediction_arrays,
        gamma=float(gamma_report["selected_gamma"]),
    )
    shrunk_metrics = _metrics_from_binary_logits(shrunk_logits, labels_np, label_names=label_names)
    return gamma_report, shrunk_metrics


@dataclass
class LocalGraphResidualExpertV2TrainConfig:
    """Configuration for Step 8 V2 residual training."""

    output_dir: str
    hlt_cache_dir: str
    baseline_embedding_cache_dir: str
    train_split: str = "model_train"
    val_split: str = "model_val"
    confirm_split_settings: bool = False
    seed: int = 5207
    batch_size: int = 64
    eval_batch_size: int = 128
    epochs: int = 30
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 6
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_train_jets: int | None = None
    max_val_jets: int | None = None
    selection_metric: str = LOCAL_GRAPH_RESIDUAL_V2_SELECTION_METRIC
    compile_model: bool = False
    verify_hlt_hash: bool = True
    verify_hlt_params: bool = True
    expected_hlt_degradation_strength: float = LOCAL_GRAPH_PART_HLT_DEGRADATION_STRENGTH

    baseline_embedding_dim: int | None = None
    max_constits: int = 128
    k: int = 16
    local_embed_dim: int = 128
    local_heads: int = 8
    local_hidden_dim: int | None = None
    local_adapter_gamma_init: float = 1.0
    local_pool_mode: str = LOCAL_GRAPH_RESIDUAL_V2_POOL_MEAN_MAX
    dropout: float = 0.05
    attention_dropout: float = 0.05
    weight_threshold: float = 0.0
    condition_embed_dim: int = 64
    local_context_dim: int = 128
    residual_hidden_dim: int = 256
    residual_dropout: float = 0.05
    residual_output_scale: float = 1.0
    gate_bias_init: float = -1.0
    delta_init_std: float = 1.0e-3
    gamma_initial: float = 0.1
    gamma_learnable: bool = True
    gamma_max: float | None = 2.0
    residual_input_mode: str = LOCAL_GRAPH_RESIDUAL_V2_INPUT_FULL

    condition_control_mode: str = LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_NORMAL
    condition_shuffle_seed: int = 520701
    label_control_mode: str = LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_NORMAL
    label_shuffle_seed: int = 520702

    loss_mode: str = LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE
    bce_anchor_weight: float = 0.05
    soft_fpr_weight: float = 0.25
    correction_l2_weight: float = 1.0e-4
    pairwise_weight: float = 1.0
    weighted_bce_weight: float = 1.0
    pairwise_temperature: float = 0.20
    soft_fpr_epsilon: float = 0.20
    cvar_top_fraction: float = 0.50
    hard_background_fraction: float = 0.20
    signal_boundary_quantile_low: float = 0.40
    signal_boundary_quantile_high: float = 0.60
    bce_boundary_scale: float | None = None

    label_names: tuple[str, ...] = LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES
    label_filter: tuple[int, ...] = LOCAL_GRAPH_PART_BINARY_LABEL_FILTER

    def __post_init__(self) -> None:
        protocol = default_local_graph_residual_v2_protocol()
        if self.train_split != "model_train" or self.val_split != "model_val":
            raise ValueError("Step 8 V2 trains only on model_train and selects only on model_val")
        if not bool(self.confirm_split_settings):
            raise ValueError("Set --confirm-split-settings to acknowledge V2 residual train/val-only selection")
        if tuple(protocol.train_splits) != LOCAL_GRAPH_RESIDUAL_V2_TRAIN_SPLITS:
            raise ValueError("V2 protocol train split constants changed unexpectedly")
        if str(self.selection_metric) != str(protocol.selection_metric):
            raise ValueError(f"Step 8 V2 selects checkpoints with {protocol.selection_metric}")
        self.expected_hlt_degradation_strength = float(self.expected_hlt_degradation_strength)
        if abs(float(self.expected_hlt_degradation_strength) - float(protocol.hlt_degradation_strength)) > 1.0e-12:
            raise ValueError("Step 8 V2 is fixed to HLT degradation strength 0.6")
        for field_name in (
            "batch_size",
            "eval_batch_size",
            "epochs",
            "max_constits",
            "k",
            "local_embed_dim",
            "local_heads",
            "condition_embed_dim",
            "local_context_dim",
            "residual_hidden_dim",
        ):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            setattr(self, field_name, value)
        if self.baseline_embedding_dim is not None:
            self.baseline_embedding_dim = _optional_positive_int(
                self.baseline_embedding_dim,
                field_name="baseline_embedding_dim",
            )
        if int(self.num_workers) < 0:
            raise ValueError("num_workers cannot be negative")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if float(self.grad_clip_norm) < 0.0:
            raise ValueError("grad_clip_norm cannot be negative")
        if int(self.early_stop_patience) < -1:
            raise ValueError("early_stop_patience must be -1 or greater")
        for field_name in ("dropout", "attention_dropout", "residual_dropout"):
            value = float(getattr(self, field_name))
            if value < 0.0 or value >= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1)")
            setattr(self, field_name, value)
        if self.local_hidden_dim is not None:
            self.local_hidden_dim = _optional_positive_int(self.local_hidden_dim, field_name="local_hidden_dim")
        self.residual_input_mode = str(self.residual_input_mode)
        if self.residual_input_mode not in LOCAL_GRAPH_RESIDUAL_V2_INPUT_MODES:
            raise ValueError(f"residual_input_mode must be one of {LOCAL_GRAPH_RESIDUAL_V2_INPUT_MODES}")
        self.condition_control_mode = str(self.condition_control_mode)
        if self.condition_control_mode not in LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_MODES:
            raise ValueError(
                f"condition_control_mode must be one of {LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_MODES}"
            )
        self.label_control_mode = str(self.label_control_mode)
        if self.label_control_mode not in LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_MODES:
            raise ValueError(f"label_control_mode must be one of {LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_MODES}")
        self.condition_shuffle_seed = int(self.condition_shuffle_seed)
        self.label_shuffle_seed = int(self.label_shuffle_seed)
        self.max_train_batches = _optional_nonnegative_int(self.max_train_batches, field_name="max_train_batches")
        self.max_val_batches = _optional_nonnegative_int(self.max_val_batches, field_name="max_val_batches")
        self.max_train_jets = _optional_positive_int(self.max_train_jets, field_name="max_train_jets")
        self.max_val_jets = _optional_positive_int(self.max_val_jets, field_name="max_val_jets")
        self.loss_mode = normalize_local_graph_residual_v2_loss_mode(self.loss_mode)
        self.label_names = tuple(str(name) for name in self.label_names)
        self.label_filter = tuple(int(label) for label in self.label_filter)
        if tuple(self.label_names) != tuple(protocol.label_names):
            raise ValueError("Step 8 V2 residual protocol is frozen to QCD/Hgg labels")
        if tuple(self.label_filter) != tuple(protocol.binary_label_filter):
            raise ValueError("Step 8 V2 expects binary-cache labels QCD=0, Hgg=1")

    def model_config(self, *, baseline_embedding_dim: int | None = None) -> LocalGraphResidualExpertV2Config:
        embedding_dim = baseline_embedding_dim if baseline_embedding_dim is not None else self.baseline_embedding_dim
        if embedding_dim is None:
            raise ValueError("baseline_embedding_dim must be known before building V2 model")
        return LocalGraphResidualExpertV2Config(
            baseline_embedding_dim=int(embedding_dim),
            max_constits=int(self.max_constits),
            k=int(self.k),
            local_embed_dim=int(self.local_embed_dim),
            local_heads=int(self.local_heads),
            local_hidden_dim=self.local_hidden_dim,
            local_adapter_gamma_init=float(self.local_adapter_gamma_init),
            local_pool_mode=str(self.local_pool_mode),
            dropout=float(self.dropout),
            attention_dropout=float(self.attention_dropout),
            weight_threshold=float(self.weight_threshold),
            condition_embed_dim=int(self.condition_embed_dim),
            local_context_dim=int(self.local_context_dim),
            residual_hidden_dim=int(self.residual_hidden_dim),
            residual_dropout=float(self.residual_dropout),
            residual_output_scale=float(self.residual_output_scale),
            gate_bias_init=float(self.gate_bias_init),
            delta_init_std=float(self.delta_init_std),
            gamma_initial=float(self.gamma_initial),
            gamma_learnable=bool(self.gamma_learnable),
            gamma_max=self.gamma_max,
            residual_input_mode=str(self.residual_input_mode),
        )

    def loss_config(self, train_block: LocalGraphResidualV2BaselineEmbeddingBlock | None = None) -> LocalGraphResidualV2LossConfig:
        tau50 = 0.0
        tau30 = 0.0
        if train_block is not None:
            reference = train_block.condition_reference(require=True)
            tau50 = float(reference["tau50"])
            tau30 = float(reference["tau30"])
        return LocalGraphResidualV2LossConfig(
            mode=str(self.loss_mode),
            bce_anchor_weight=float(self.bce_anchor_weight),
            soft_fpr_weight=float(self.soft_fpr_weight),
            correction_l2_weight=float(self.correction_l2_weight),
            pairwise_weight=float(self.pairwise_weight),
            weighted_bce_weight=float(self.weighted_bce_weight),
            pairwise_temperature=float(self.pairwise_temperature),
            soft_fpr_epsilon=float(self.soft_fpr_epsilon),
            cvar_top_fraction=float(self.cvar_top_fraction),
            hard_background_fraction=float(self.hard_background_fraction),
            signal_boundary_quantile_low=float(self.signal_boundary_quantile_low),
            signal_boundary_quantile_high=float(self.signal_boundary_quantile_high),
            bce_boundary_scale=self.bce_boundary_scale,
            default_tau50=tau50,
            default_tau30=tau30,
        )


def _load_residual_v2_dataset(
    config: LocalGraphResidualExpertV2TrainConfig,
    split: str,
    *,
    max_jets: int | None,
) -> SubtokenHLTJetDataset:
    view = load_cached_hlt_view(config.hlt_cache_dir, split, verify_hash=bool(config.verify_hlt_hash))
    view.metadata["hlt_protocol_audit"] = _verify_hlt_cache_protocol(
        view.metadata,
        split=split,
        expected_strength=float(config.expected_hlt_degradation_strength),
        required=bool(config.verify_hlt_params),
    )
    dataset = SubtokenHLTJetDataset(
        view,
        label_filter=tuple(config.label_filter),
        label_names=tuple(config.label_names),
        max_jets=max_jets,
    )
    dataset.metadata["hlt_protocol_audit"] = view.metadata["hlt_protocol_audit"]
    dataset.metadata["split_manifest_hash"] = (
        view.metadata.get("source_manifest_hash")
        or view.metadata.get("manifest_hash")
        or view.metadata.get("split_manifest_hash")
    )
    return dataset


def _stable_split_seed(split: str, seed: int) -> int:
    digest = hashlib.sha256(str(split).encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "little", signed=False)
    return (int(seed) + offset) % (2**32)


def _split_permutation(block: LocalGraphResidualV2BaselineEmbeddingBlock, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(_stable_split_seed(block.split, int(seed)))
    return rng.permutation(int(block.labels.shape[0])).astype(np.int64)


def _v2_cache_arrays_for_batch(
    block: LocalGraphResidualV2BaselineEmbeddingBlock,
    indices,
    *,
    labels,
    device,
    condition_control_mode: str = LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_NORMAL,
    condition_shuffle_seed: int = 0,
) -> tuple[Any, Any, Any]:
    torch = require_torch()
    indices_cpu = indices.detach().cpu().numpy().astype(np.int64)
    arrays = block.arrays_for_indices(indices_cpu)
    cached_labels = np.asarray(arrays["labels"], dtype=np.int64)
    batch_labels = labels.detach().cpu().numpy().astype(np.int64)
    if not np.array_equal(cached_labels, batch_labels):
        raise ValueError("V2 embedding cache labels do not align with the HLT dataset batch")
    condition = np.asarray(arrays["condition_features"], dtype=np.float32)
    if str(condition_control_mode) == LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_SHUFFLED:
        all_condition, _metadata = block.condition_features(require_reference=True)
        positions = np.asarray(arrays["positions"], dtype=np.int64)
        permutation = _split_permutation(block, seed=int(condition_shuffle_seed))
        condition = np.asarray(all_condition[permutation[positions]], dtype=np.float32)
    elif str(condition_control_mode) != LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_NORMAL:
        raise ValueError(f"unknown V2 condition_control_mode: {condition_control_mode!r}")
    z_base = torch.from_numpy(np.asarray(arrays["margin"], dtype=np.float32)).to(device=device, non_blocking=True)
    embedding = torch.from_numpy(np.asarray(arrays["embedding"], dtype=np.float32)).to(device=device, non_blocking=True)
    condition_tensor = torch.from_numpy(condition).to(
        device=device,
        non_blocking=True,
    )
    return z_base, embedding, condition_tensor


def _v2_labels_for_loss(
    block: LocalGraphResidualV2BaselineEmbeddingBlock,
    indices,
    labels,
    *,
    training: bool,
    label_control_mode: str,
    label_shuffle_seed: int,
    device,
):
    if not bool(training) or str(label_control_mode) == LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_NORMAL:
        return labels
    if str(label_control_mode) != LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_SHUFFLED:
        raise ValueError(f"unknown V2 label_control_mode: {label_control_mode!r}")
    torch = require_torch()
    indices_cpu = indices.detach().cpu().numpy().astype(np.int64)
    positions = block.positions_for_indices(indices_cpu)
    permutation = _split_permutation(block, seed=int(label_shuffle_seed))
    shuffled = np.asarray(block.labels[permutation[positions]], dtype=np.int64)
    return torch.from_numpy(shuffled).to(device=device, non_blocking=True)


def _v2_thresholds(block: LocalGraphResidualV2BaselineEmbeddingBlock) -> dict[str, float]:
    reference = block.condition_reference(require=True)
    return {"tau50": float(reference["tau50"]), "tau30": float(reference["tau30"])}


def run_local_graph_residual_expert_v2_epoch(
    model,
    loader,
    *,
    baseline_block: LocalGraphResidualV2BaselineEmbeddingBlock,
    device,
    loss_config: LocalGraphResidualV2LossConfig,
    optimizer=None,
    scaler=None,
    amp: bool = True,
    grad_clip_norm: float | None = 1.0,
    max_batches: int | None = None,
    collect_predictions: bool = False,
    collect_diagnostics: bool = False,
    label_names: tuple[str, ...] = LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES,
    condition_control_mode: str = LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_NORMAL,
    condition_shuffle_seed: int = 0,
    label_control_mode: str = LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_NORMAL,
    label_shuffle_seed: int = 0,
) -> dict[str, Any]:
    """Run one train/eval epoch for the V2 residual expert."""

    torch = require_torch()
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_seen = 0
    component_totals: dict[str, float] = {}
    diagnostic_totals: dict[str, float] = {}
    diagnostic_weight_sum = 0.0
    collected_fused: list[np.ndarray] = []
    collected_baseline: list[np.ndarray] = []
    collected_residual: list[np.ndarray] = []
    collected_correction: list[np.ndarray] = []
    collected_labels: list[np.ndarray] = []
    collected_indices: list[np.ndarray] = []
    thresholds = _v2_thresholds(baseline_block)
    autocast_enabled = bool(amp and device.type == "cuda")
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = _move_batch_to_device(batch, device)
            z_base, embedding, condition = _v2_cache_arrays_for_batch(
                baseline_block,
                batch["indices"],
                labels=batch["labels"],
                device=device,
                condition_control_mode=str(condition_control_mode),
                condition_shuffle_seed=int(condition_shuffle_seed),
            )
            labels_for_loss = _v2_labels_for_loss(
                baseline_block,
                batch["indices"],
                batch["labels"],
                training=training,
                label_control_mode=str(label_control_mode),
                label_shuffle_seed=int(label_shuffle_seed),
                device=device,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                output = model(
                    batch["tokens"],
                    batch["mask"],
                    baseline_logit=z_base,
                    baseline_embedding=embedding,
                    baseline_condition_features=condition,
                    return_outputs=True,
                )
                loss_output = compute_local_graph_residual_v2_loss_from_output(
                    output,
                    labels_for_loss,
                    tau50=thresholds["tau50"],
                    tau30=thresholds["tau30"],
                    config=loss_config,
                )
                loss = loss_output.total_loss
            if training:
                if not bool(getattr(loss, "requires_grad", False)):
                    raise RuntimeError("V2 residual expert training loss does not require grad")
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                    optimizer.step()

            batch_size = int(batch["labels"].numel())
            total_loss += float(loss.detach().item()) * batch_size
            total_seen += batch_size
            for key, value in loss_output.scalar_components().items():
                component_totals[key] = component_totals.get(key, 0.0) + float(value) * batch_size
            scalar_diagnostics = dict(loss_output.scalar_diagnostics())
            if bool(collect_diagnostics):
                scalar_diagnostics.update(_flatten_scalar_diagnostics(output.diagnostics()))
            for key, value in scalar_diagnostics.items():
                diagnostic_totals[key] = diagnostic_totals.get(key, 0.0) + float(value) * batch_size
            diagnostic_weight_sum += float(batch_size)
            if collect_predictions:
                collected_fused.append(output.fused_logits.detach().cpu().numpy().astype(np.float32))
                collected_baseline.append(
                    binary_logits_from_log_odds(output.baseline_logit.detach().cpu().numpy()).astype(np.float32)
                )
                collected_residual.append(output.residual_logits.detach().cpu().numpy().astype(np.float32))
                collected_correction.append(output.correction_logits.detach().cpu().numpy().astype(np.float32))
                collected_labels.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
                collected_indices.append(batch["indices"].detach().cpu().numpy().astype(np.int64))

    if total_seen == 0:
        return {"loss": float("nan"), "accuracy": 0.0, "n_jets": 0}

    metrics: dict[str, Any] = {
        "loss": total_loss / float(total_seen),
        "n_jets": int(total_seen),
        "loss_components": {key: value / float(total_seen) for key, value in sorted(component_totals.items())},
        "loss_weights": dict(loss_config.active_weights()),
        "baseline_thresholds": thresholds,
    }
    if diagnostic_weight_sum > 0.0:
        metrics["diagnostics"] = {
            key: value / diagnostic_weight_sum for key, value in sorted(diagnostic_totals.items())
        }
    if collect_predictions:
        fused_logits = np.concatenate(collected_fused, axis=0) if collected_fused else np.zeros((0, 2), dtype=np.float32)
        baseline_logits = (
            np.concatenate(collected_baseline, axis=0) if collected_baseline else np.zeros((0, 2), dtype=np.float32)
        )
        residual_logits = (
            np.concatenate(collected_residual, axis=0) if collected_residual else np.zeros((0, 2), dtype=np.float32)
        )
        correction_logits = (
            np.concatenate(collected_correction, axis=0) if collected_correction else np.zeros((0, 2), dtype=np.float32)
        )
        labels_np = np.concatenate(collected_labels, axis=0) if collected_labels else np.asarray([], dtype=np.int64)
        indices_np = np.concatenate(collected_indices, axis=0) if collected_indices else np.asarray([], dtype=np.int64)
        fused_metrics = _metrics_from_binary_logits(fused_logits, labels_np, loss_sum=total_loss, label_names=label_names)
        baseline_metrics = _metrics_from_binary_logits(baseline_logits, labels_np, label_names=label_names)
        residual_metrics = _metrics_from_binary_logits(residual_logits, labels_np, label_names=label_names)
        correction_metrics = _metrics_from_binary_logits(correction_logits, labels_np, label_names=label_names)
        metrics.update(fused_metrics)
        metrics["fused_metrics"] = fused_metrics
        metrics["baseline_metrics"] = baseline_metrics
        metrics["residual_metrics"] = residual_metrics
        metrics["correction_metrics"] = correction_metrics
        metrics["_prediction_arrays"] = {
            "indices": indices_np,
            "labels": labels_np,
            "fused_logits": fused_logits,
            "baseline_logits": baseline_logits,
            "residual_logits": residual_logits,
            "correction_logits": correction_logits,
        }
    return metrics


def local_graph_residual_expert_v2_checkpoint_payload(
    model,
    optimizer,
    *,
    epoch: int,
    config: LocalGraphResidualExpertV2TrainConfig,
    loss_config: LocalGraphResidualV2LossConfig,
    metrics: Mapping[str, Any],
    source: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "loss_config": loss_config.to_dict(),
        "model_config": model.to_config_dict(),
        "metrics": dict(metrics),
        "variant": model.config.variant,
        "label_names": list(config.label_names),
        "label_filter": list(config.label_filter),
        "source": dict(source),
        "cache_identity": dict(cache_identity),
        "experiment_step": LOCAL_GRAPH_RESIDUAL_V2_TRAIN_STEP,
        "protocol_step": LOCAL_GRAPH_RESIDUAL_V2_PROTOCOL_STEP,
        "protocol_contract": LOCAL_GRAPH_RESIDUAL_V2_PROTOCOL_CONTRACT,
        "base_protocol_step": LOCAL_GRAPH_PART_PROTOCOL_STEP,
        "base_protocol_contract": LOCAL_GRAPH_PART_CONTRACT,
        "cache_contract": LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT,
        "loss_step": LOCAL_GRAPH_RESIDUAL_V2_LOSS_STEP,
        "loss_contract": LOCAL_GRAPH_RESIDUAL_V2_LOSS_CONTRACT,
        "model_step": LOCAL_GRAPH_RESIDUAL_V2_MODEL_STEP,
        "output_contract": model.output_contract,
        "embedding_contract": LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE,
    }


def load_local_graph_residual_expert_v2_checkpoint(path: str | Path, *, device=None):
    torch = require_torch()
    payload = torch.load(path, map_location=device or "cpu")
    config_payload = payload.get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("checkpoint payload does not contain a V2 residual train config")
    config = LocalGraphResidualExpertV2TrainConfig(**dict(config_payload))
    model = build_local_graph_residual_expert_v2(config.model_config())
    model.load_state_dict(payload["model_state_dict"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, payload


def _save_model_val_prediction_arrays(path: Path, prediction_arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        indices=np.asarray(prediction_arrays.get("indices"), dtype=np.int64),
        labels=np.asarray(prediction_arrays.get("labels"), dtype=np.int64),
        fused_logits=np.asarray(prediction_arrays.get("fused_logits"), dtype=np.float32),
        baseline_logits=np.asarray(prediction_arrays.get("baseline_logits"), dtype=np.float32),
        residual_logits=np.asarray(prediction_arrays.get("residual_logits"), dtype=np.float32),
        correction_logits=np.asarray(prediction_arrays.get("correction_logits"), dtype=np.float32),
    )


def train_local_graph_residual_expert_v2(
    config: LocalGraphResidualExpertV2TrainConfig,
    *,
    model: LocalGraphResidualExpertV2 | None = None,
    train_dataset: SubtokenHLTJetDataset | None = None,
    val_dataset: SubtokenHLTJetDataset | None = None,
    baseline_blocks: Mapping[str, LocalGraphResidualV2BaselineEmbeddingBlock] | None = None,
) -> dict[str, Any]:
    """Train/select one V2 residual expert on model_train/model_val only."""

    run_start_time = time.perf_counter()
    torch = require_torch()
    set_training_seed(int(config.seed))
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = train_dataset or _load_residual_v2_dataset(config, config.train_split, max_jets=config.max_train_jets)
    val_dataset = val_dataset or _load_residual_v2_dataset(config, config.val_split, max_jets=config.max_val_jets)
    if train_dataset.tokens.shape[-1] != RAW_TOKEN_DIM or val_dataset.tokens.shape[-1] != RAW_TOKEN_DIM:
        raise ValueError(f"V2 residual expert expects raw token dim {RAW_TOKEN_DIM}")

    baseline_blocks = dict(baseline_blocks or {})
    train_block = baseline_blocks.get(config.train_split) or load_residual_v2_embedding_block(
        config.baseline_embedding_cache_dir,
        config.train_split,
        require_metadata=True,
    )
    val_block = baseline_blocks.get(config.val_split) or load_residual_v2_embedding_block(
        config.baseline_embedding_cache_dir,
        config.val_split,
        require_metadata=True,
    )
    train_alignment = verify_residual_v2_embedding_block_alignment(
        train_block,
        train_dataset.metadata,
        split=config.train_split,
        dataset_length=len(train_dataset),
        expected_indices=np.arange(len(train_dataset), dtype=np.int64),
        expected_labels=train_dataset.labels,
        expected_embedding_dim=config.baseline_embedding_dim,
    )
    val_alignment = verify_residual_v2_embedding_block_alignment(
        val_block,
        val_dataset.metadata,
        split=config.val_split,
        dataset_length=len(val_dataset),
        expected_indices=np.arange(len(val_dataset), dtype=np.int64),
        expected_labels=val_dataset.labels,
        expected_checkpoint_identity=train_alignment["checkpoint_identity"],
        expected_condition_reference=train_alignment["condition_reference"],
        expected_embedding_dim=train_block.embedding_dim,
    )
    cache_family = verify_residual_v2_embedding_cache_family(
        (train_block, val_block),
        expected_checkpoint_identity=train_alignment["checkpoint_identity"],
    )
    if config.baseline_embedding_dim is not None and int(config.baseline_embedding_dim) != int(train_block.embedding_dim):
        raise ValueError("configured baseline_embedding_dim does not match V2 cache")
    config.baseline_embedding_dim = int(train_block.embedding_dim)

    train_loader = make_subtoken_hlt_loader(
        train_dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        seed=int(config.seed),
    )
    val_loader = make_subtoken_hlt_loader(
        val_dataset,
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 1,
    )

    checkpoint_model = model or build_local_graph_residual_expert_v2(config.model_config())
    checkpoint_model = checkpoint_model.to(device)
    train_model = checkpoint_model
    if bool(config.compile_model) and hasattr(torch, "compile"):
        train_model = torch.compile(train_model)

    loss_config = config.loss_config(train_block)
    optimizer = torch.optim.AdamW(checkpoint_model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))
    source = source_metadata()
    run_metadata = {
        "experiment_step": LOCAL_GRAPH_RESIDUAL_V2_TRAIN_STEP,
        "contract": LOCAL_GRAPH_RESIDUAL_V2_TRAIN_CONTRACT,
        "protocol": local_graph_residual_v2_protocol_manifest(),
        "base_protocol_step": LOCAL_GRAPH_PART_PROTOCOL_STEP,
        "base_protocol_contract": LOCAL_GRAPH_PART_CONTRACT,
        "cache_contract": LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT,
        "output_contract": checkpoint_model.output_contract,
        "variant": checkpoint_model.config.variant,
        "config": asdict(config),
        "model_config": checkpoint_model.to_config_dict(),
        "loss_config": loss_config.to_dict(),
        "source": source,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "baseline_embedding_cache_dir": str(config.baseline_embedding_cache_dir),
        "train_baseline_embedding_cache": train_block.to_manifest_row(),
        "val_baseline_embedding_cache": val_block.to_manifest_row(),
        "cache_alignment": {"train": train_alignment, "val": val_alignment, "family": cache_family},
        "selection_metric": str(config.selection_metric),
        "leakage_rule": "Step 8 V2 uses only model_train and model_val. stack/final splits are not loaded.",
        "control_modes": {
            "residual_input_mode": str(config.residual_input_mode),
            "condition_control_mode": str(config.condition_control_mode),
            "condition_shuffle_seed": int(config.condition_shuffle_seed),
            "label_control_mode": str(config.label_control_mode),
            "label_shuffle_seed": int(config.label_shuffle_seed),
            "label_shuffle_applies_to": "model_train_loss_only",
            "condition_shuffle_applies_to": "residual_condition_features_only",
        },
        "stack_or_final_loaded": False,
        "inference_consumes_hlt_only": True,
        "embedding_contract": LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE,
    }
    save_json(output_dir / "config.json", run_metadata)

    curves: list[dict[str, Any]] = []
    best_val_score = float("-inf")
    best_selection_metric_value = float("nan")
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, int(config.epochs) + 1):
        train_metrics = run_local_graph_residual_expert_v2_epoch(
            train_model,
            train_loader,
            baseline_block=train_block,
            device=device,
            loss_config=loss_config,
            optimizer=optimizer,
            scaler=scaler,
            amp=bool(config.amp),
            grad_clip_norm=float(config.grad_clip_norm),
            max_batches=config.max_train_batches,
            collect_predictions=False,
            collect_diagnostics=False,
            label_names=tuple(config.label_names),
            condition_control_mode=str(config.condition_control_mode),
            condition_shuffle_seed=int(config.condition_shuffle_seed),
            label_control_mode=str(config.label_control_mode),
            label_shuffle_seed=int(config.label_shuffle_seed),
        )
        val_metrics = run_local_graph_residual_expert_v2_epoch(
            train_model,
            val_loader,
            baseline_block=val_block,
            device=device,
            loss_config=loss_config,
            amp=False,
            max_batches=config.max_val_batches,
            collect_predictions=True,
            collect_diagnostics=False,
            label_names=tuple(config.label_names),
            condition_control_mode=str(config.condition_control_mode),
            condition_shuffle_seed=int(config.condition_shuffle_seed),
            label_control_mode=str(config.label_control_mode),
            label_shuffle_seed=int(config.label_shuffle_seed),
        )
        val_metrics_for_json = _strip_prediction_arrays(val_metrics)
        row = {"epoch": int(epoch), "train": train_metrics, "model_val": val_metrics_for_json}
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves})

        val_score, selection_value = _selection_score(val_metrics, str(config.selection_metric))
        val_loss = _finite_float(val_metrics.get("loss"), default=float("inf"))
        improved = val_score > best_val_score or (np.isclose(val_score, best_val_score) and val_loss < best_val_loss)
        payload = local_graph_residual_expert_v2_checkpoint_payload(
            checkpoint_model,
            optimizer,
            epoch=epoch,
            config=config,
            loss_config=loss_config,
            metrics=row,
            source=source,
            cache_identity=cache_family,
        )
        torch.save(payload, output_dir / "last.pt")
        if improved:
            best_val_score = float(val_score)
            best_selection_metric_value = float(selection_value)
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(payload, output_dir / "best_model_val.pt")
        else:
            epochs_without_improvement += 1
        if int(config.early_stop_patience) >= 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    if best_epoch < 0 or not (output_dir / "best_model_val.pt").exists():
        raise FloatingPointError("V2 residual expert did not produce a valid model_val checkpoint")

    best_payload = torch.load(output_dir / "best_model_val.pt", map_location=device)
    if model is None:
        best_model, _ = load_local_graph_residual_expert_v2_checkpoint(output_dir / "best_model_val.pt", device=device)
    else:
        checkpoint_model.load_state_dict(best_payload["model_state_dict"])
        checkpoint_model.eval()
        best_model = checkpoint_model

    best_val_metrics = run_local_graph_residual_expert_v2_epoch(
        best_model,
        val_loader,
        baseline_block=val_block,
        device=device,
        loss_config=loss_config,
        amp=False,
        max_batches=config.max_val_batches,
        collect_predictions=True,
        collect_diagnostics=True,
        label_names=tuple(config.label_names),
        condition_control_mode=str(config.condition_control_mode),
        condition_shuffle_seed=int(config.condition_shuffle_seed),
        label_control_mode=str(config.label_control_mode),
        label_shuffle_seed=int(config.label_shuffle_seed),
    )
    prediction_arrays = best_val_metrics.get("_prediction_arrays", {})
    model_val_prediction_path = diagnostics_dir / "model_val_learned_gamma_predictions.npz"
    if isinstance(prediction_arrays, Mapping):
        _save_model_val_prediction_arrays(model_val_prediction_path, prediction_arrays)
    gamma_shrinkage_report = None
    gamma_shrunk_model_val_metrics = None
    if isinstance(prediction_arrays, Mapping):
        gamma_shrinkage_report, gamma_shrunk_model_val_metrics = _model_val_gamma_shrinkage_from_predictions(
            prediction_arrays,
            label_names=tuple(config.label_names),
        )
    best_val_metrics_json = _strip_prediction_arrays(best_val_metrics)
    elapsed_seconds = float(time.perf_counter() - run_start_time)

    report = {
        "experiment_step": LOCAL_GRAPH_RESIDUAL_V2_TRAIN_STEP,
        "contract": LOCAL_GRAPH_RESIDUAL_V2_TRAIN_CONTRACT,
        "protocol_step": LOCAL_GRAPH_RESIDUAL_V2_PROTOCOL_STEP,
        "protocol_contract": LOCAL_GRAPH_RESIDUAL_V2_PROTOCOL_CONTRACT,
        "protocol": local_graph_residual_v2_protocol_manifest(),
        "base_protocol_step": LOCAL_GRAPH_PART_PROTOCOL_STEP,
        "base_protocol_contract": LOCAL_GRAPH_PART_CONTRACT,
        "cache_contract": LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT,
        "loss_step": LOCAL_GRAPH_RESIDUAL_V2_LOSS_STEP,
        "loss_contract": LOCAL_GRAPH_RESIDUAL_V2_LOSS_CONTRACT,
        "model_step": LOCAL_GRAPH_RESIDUAL_V2_MODEL_STEP,
        "output_contract": LOCAL_GRAPH_RESIDUAL_V2_MODEL_CONTRACT,
        "embedding_contract": LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE,
        "variant": best_model.config.variant,
        "loss_mode": str(loss_config.mode),
        "best_epoch": int(best_epoch),
        "selection_metric": str(config.selection_metric),
        "selection_metric_direction": LOCAL_GRAPH_RESIDUAL_V2_PRIMARY_METRIC_DIRECTION,
        "control_modes": {
            "residual_input_mode": str(config.residual_input_mode),
            "condition_control_mode": str(config.condition_control_mode),
            "condition_shuffle_seed": int(config.condition_shuffle_seed),
            "label_control_mode": str(config.label_control_mode),
            "label_shuffle_seed": int(config.label_shuffle_seed),
            "label_shuffle_applies_to": "model_train_loss_only",
            "condition_shuffle_applies_to": "residual_condition_features_only",
        },
        "best_model_selection_metric_value": float(best_selection_metric_value),
        "best_model_selection_score": float(best_val_score),
        "best_model_val_loss": float(best_val_loss),
        "best_model_val_metrics": best_val_metrics_json,
        "baseline_model_val_metrics": best_val_metrics_json.get("baseline_metrics"),
        "fused_model_val_metrics": best_val_metrics_json.get("fused_metrics"),
        "fused_model_val_learned_gamma_metrics": best_val_metrics_json.get("fused_metrics"),
        "fused_model_val_val_shrunk_metrics": gamma_shrunk_model_val_metrics,
        "residual_model_val_metrics": best_val_metrics_json.get("residual_metrics"),
        "correction_model_val_metrics": best_val_metrics_json.get("correction_metrics"),
        "gamma_shrinkage_model_val": gamma_shrinkage_report,
        "alpha_shrinkage_model_val": gamma_shrinkage_report,
        "selected_gamma_model_val": (
            None if gamma_shrinkage_report is None else float(gamma_shrinkage_report["selected_gamma"])
        ),
        "selected_gamma_metrics_model_val": gamma_shrunk_model_val_metrics,
        "model_val_learned_gamma_predictions": str(model_val_prediction_path),
        "model_val_prediction_arrays_include": [
            "indices",
            "labels",
            "fused_logits",
            "baseline_logits",
            "residual_logits",
            "correction_logits",
        ],
        "epochs_completed": len(curves),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "training_curves": str(output_dir / "training_curves.json"),
        "config": asdict(config),
        "loss_config": loss_config.to_dict(),
        "model_config": best_model.to_config_dict(),
        "label_names": list(config.label_names),
        "label_filter": list(config.label_filter),
        "source": source,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "train_baseline_embedding_cache": train_block.to_manifest_row(),
        "val_baseline_embedding_cache": val_block.to_manifest_row(),
        "cache_alignment": {"train": train_alignment, "val": val_alignment, "family": cache_family},
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "elapsed_minutes": elapsed_seconds / 60.0,
            "epochs_completed": len(curves),
            "seconds_per_completed_epoch": elapsed_seconds / float(len(curves)) if curves else None,
        },
        "walltime_seconds": elapsed_seconds,
        "stack_or_final_loaded": False,
        "inference_consumes_hlt_only": True,
    }
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    return report


__all__ = [
    "LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_MODES",
    "LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_NORMAL",
    "LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_SHUFFLED",
    "LOCAL_GRAPH_RESIDUAL_V2_GAMMA_SHRINKAGE_GRID",
    "LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_MODES",
    "LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_NORMAL",
    "LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_SHUFFLED",
    "LOCAL_GRAPH_RESIDUAL_V2_LOWER_IS_BETTER_SELECTION_METRICS",
    "LOCAL_GRAPH_RESIDUAL_V2_SELECTION_METRIC",
    "LocalGraphResidualExpertV2TrainConfig",
    "load_local_graph_residual_expert_v2_checkpoint",
    "local_graph_residual_expert_v2_checkpoint_payload",
    "run_local_graph_residual_expert_v2_epoch",
    "select_local_graph_residual_v2_gamma_shrinkage",
    "train_local_graph_residual_expert_v2",
]
