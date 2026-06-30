"""Training loop for local-graph residual experts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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

from .fusion import binary_logits_from_log_odds
from .model import LOCAL_GRAPH_ADAPTER_POINT_ATTENTION, normalize_local_graph_adapter
from .protocol import (
    LOCAL_GRAPH_PART_BINARY_LABEL_FILTER,
    LOCAL_GRAPH_PART_CONTRACT,
    LOCAL_GRAPH_PART_HLT_DEGRADATION_STRENGTH,
    LOCAL_GRAPH_PART_PRIMARY_METRIC,
    LOCAL_GRAPH_PART_PROTOCOL_STEP,
    LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES,
    default_local_graph_part_protocol,
    local_graph_part_protocol_manifest,
)
from .residual_cache import (
    LOCAL_GRAPH_BASELINE_CONDITION_FEATURE_NAMES,
    LocalGraphBaselineLogitBlock,
    load_baseline_logit_block,
    verify_baseline_logit_block_alignment,
    verify_baseline_logit_cache_family,
)
from .residual_losses import (
    LOCAL_GRAPH_RESIDUAL_LOSS_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_LOSS_STEP,
    LOCAL_GRAPH_RESIDUAL_LOSS_WEIGHTED_BCE,
    LocalGraphResidualLossConfig,
    compute_local_graph_residual_loss_from_output,
    normalize_local_graph_residual_loss_mode,
    select_alpha_shrinkage,
)
from .residual_diagnostics import (
    LOCAL_GRAPH_RESIDUAL_DIAGNOSTICS_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_DIAGNOSTICS_STEP,
    residual_correction_diagnostics,
)
from .residual_model import (
    LOCAL_GRAPH_RESIDUAL_EXPERT_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_EXPERT_STEP,
    LocalGraphResidualExpert,
    LocalGraphResidualExpertConfig,
    build_local_graph_residual_expert,
)
from .train import (
    LOCAL_GRAPH_LOWER_IS_BETTER_SELECTION_METRICS,
    _set_part_model_trainable,
    _verify_hlt_cache_protocol,
    warm_start_local_graph_part_model,
)


LOCAL_GRAPH_RESIDUAL_EXPERT_TRAIN_STEP = "local_graph_residual_expert_step4_train"
LOCAL_GRAPH_RESIDUAL_EXPERT_TRAIN_CONTRACT = "local_graph_residual_expert_train_v1"
LOCAL_GRAPH_RESIDUAL_SELECTION_METRIC = LOCAL_GRAPH_PART_PRIMARY_METRIC
LOCAL_GRAPH_RESIDUAL_LOWER_IS_BETTER_SELECTION_METRICS = LOCAL_GRAPH_LOWER_IS_BETTER_SELECTION_METRICS


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
    if metric_name in LOCAL_GRAPH_RESIDUAL_LOWER_IS_BETTER_SELECTION_METRICS:
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
    return classification_metrics_from_predictions(
        preds=preds,
        labels=labels,
        loss_sum=loss_sum,
        logits=logits,
        label_names=label_names,
    )


def _strip_prediction_arrays(metrics: Mapping[str, Any]) -> dict[str, Any]:
    clean = dict(metrics)
    clean.pop("_prediction_arrays", None)
    return clean


@dataclass
class LocalGraphResidualExpertTrainConfig:
    """Configuration for Step 4 residual expert training."""

    output_dir: str
    hlt_cache_dir: str
    baseline_logit_cache_dir: str
    train_split: str = "model_train"
    val_split: str = "model_val"
    confirm_split_settings: bool = False
    seed: int = 4107
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
    selection_metric: str = LOCAL_GRAPH_RESIDUAL_SELECTION_METRIC
    compile_model: bool = False
    verify_hlt_hash: bool = True
    verify_hlt_params: bool = True
    expected_hlt_degradation_strength: float = LOCAL_GRAPH_PART_HLT_DEGRADATION_STRENGTH

    model_size: str = "base"
    max_constits: int = 128
    local_adapter: str = LOCAL_GRAPH_ADAPTER_POINT_ATTENTION
    k: int = 16
    local_embed_dim: int = 128
    local_heads: int = 8
    local_hidden_dim: int | None = None
    dropout: float = 0.05
    attention_dropout: float = 0.05
    residual_gamma_init: float = 0.0
    weight_threshold: float = 0.0
    backbone_output_dim: int = 128
    condition_embed_dim: int = 64
    residual_hidden_dim: int = 128
    residual_dropout: float = 0.05
    alpha_initial: float = 0.1
    alpha_learnable: bool = True
    alpha_max: float | None = 2.0

    loss_mode: str = LOCAL_GRAPH_RESIDUAL_LOSS_WEIGHTED_BCE
    bce_anchor_weight: float = 0.10
    soft_fpr_weight: float = 0.25
    residual_l2_weight: float = 1.0e-4
    pairwise_weight: float = 1.0
    weighted_bce_weight: float = 1.0
    pairwise_temperature: float = 0.20
    soft_fpr_epsilon: float = 0.20
    cvar_top_fraction: float = 0.50
    hard_background_fraction: float = 0.20
    signal_boundary_quantile_low: float = 0.40
    signal_boundary_quantile_high: float = 0.60
    bce_boundary_scale: float | None = None

    warm_start_checkpoint: str | None = None
    require_warm_start: bool = False
    freeze_part_epochs: int = 0

    label_names: tuple[str, ...] = LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES
    label_filter: tuple[int, ...] = LOCAL_GRAPH_PART_BINARY_LABEL_FILTER

    def __post_init__(self) -> None:
        protocol = default_local_graph_part_protocol()
        if self.train_split != "model_train" or self.val_split != "model_val":
            raise ValueError("Step 4 trains only on model_train and selects only on model_val")
        if not bool(self.confirm_split_settings):
            raise ValueError("Set --confirm-split-settings to acknowledge residual train/val-only selection")
        if str(self.selection_metric) != str(protocol.selection_metric):
            raise ValueError(f"Step 4 residual experts select checkpoints with {protocol.selection_metric}")
        self.expected_hlt_degradation_strength = float(self.expected_hlt_degradation_strength)
        if abs(float(self.expected_hlt_degradation_strength) - float(protocol.hlt_degradation_strength)) > 1.0e-12:
            raise ValueError(
                f"Step 4 protocol requires HLT degradation strength {protocol.hlt_degradation_strength}"
            )
        for field_name in (
            "batch_size",
            "eval_batch_size",
            "epochs",
            "max_constits",
            "k",
            "local_embed_dim",
            "local_heads",
            "backbone_output_dim",
            "condition_embed_dim",
            "residual_hidden_dim",
        ):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            setattr(self, field_name, value)
        if str(self.model_size) not in {"base", "tiny"}:
            raise ValueError("model_size must be 'base' or 'tiny'")
        self.local_adapter = normalize_local_graph_adapter(self.local_adapter)
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
        self.freeze_part_epochs = _optional_nonnegative_int(self.freeze_part_epochs, field_name="freeze_part_epochs") or 0
        self.max_train_batches = _optional_nonnegative_int(self.max_train_batches, field_name="max_train_batches")
        self.max_val_batches = _optional_nonnegative_int(self.max_val_batches, field_name="max_val_batches")
        self.max_train_jets = _optional_positive_int(self.max_train_jets, field_name="max_train_jets")
        self.max_val_jets = _optional_positive_int(self.max_val_jets, field_name="max_val_jets")
        self.loss_mode = normalize_local_graph_residual_loss_mode(self.loss_mode)
        self.label_names = tuple(str(name) for name in self.label_names)
        self.label_filter = tuple(int(label) for label in self.label_filter)
        if tuple(self.label_names) != tuple(protocol.binary_label_names):
            raise ValueError("Step 4 residual protocol is frozen to QCD/Hgg labels")
        if tuple(self.label_filter) != tuple(protocol.binary_label_filter):
            raise ValueError("Step 4 expects binary-cache labels QCD=0, Hgg=1")
        if self.warm_start_checkpoint is not None:
            self.warm_start_checkpoint = str(self.warm_start_checkpoint) or None
        self.require_warm_start = bool(self.require_warm_start)

    def model_config(self) -> LocalGraphResidualExpertConfig:
        return LocalGraphResidualExpertConfig(
            model_size=str(self.model_size),
            max_constits=int(self.max_constits),
            local_adapter=str(self.local_adapter),
            k=int(self.k),
            local_embed_dim=int(self.local_embed_dim),
            local_heads=int(self.local_heads),
            local_hidden_dim=self.local_hidden_dim,
            dropout=float(self.dropout),
            attention_dropout=float(self.attention_dropout),
            residual_gamma_init=float(self.residual_gamma_init),
            weight_threshold=float(self.weight_threshold),
            backbone_output_dim=int(self.backbone_output_dim),
            condition_embed_dim=int(self.condition_embed_dim),
            residual_hidden_dim=int(self.residual_hidden_dim),
            residual_dropout=float(self.residual_dropout),
            alpha_initial=float(self.alpha_initial),
            alpha_learnable=bool(self.alpha_learnable),
            alpha_max=self.alpha_max,
        )

    def loss_config(self, train_block: LocalGraphBaselineLogitBlock | None = None) -> LocalGraphResidualLossConfig:
        tau50 = 0.0
        tau30 = 0.0
        if train_block is not None:
            reference = train_block.condition_reference(require=False)
            if reference is not None:
                tau50 = float(reference["tau50"])
                tau30 = float(reference["tau30"])
            else:
                ops = train_block.operating_points().get("margin", {})
                tau50 = float(ops.get("signal_eff_0p50", {}).get("threshold", 0.0))
                tau30 = float(ops.get("signal_eff_0p30", {}).get("threshold", 0.0))
        return LocalGraphResidualLossConfig(
            mode=str(self.loss_mode),
            bce_anchor_weight=float(self.bce_anchor_weight),
            soft_fpr_weight=float(self.soft_fpr_weight),
            residual_l2_weight=float(self.residual_l2_weight),
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


def _load_residual_dataset(
    config: LocalGraphResidualExpertTrainConfig,
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
    return dataset


def _baseline_arrays_for_batch(
    block: LocalGraphBaselineLogitBlock,
    indices,
    *,
    labels,
    device,
) -> tuple[Any, Any]:
    torch = require_torch()
    indices_cpu = indices.detach().cpu().numpy().astype(np.int64)
    positions = block.positions_for_indices(indices_cpu)
    cached_labels = block.labels[positions]
    batch_labels = labels.detach().cpu().numpy().astype(np.int64)
    if not np.array_equal(cached_labels, batch_labels):
        raise ValueError("baseline logit cache labels do not align with the HLT dataset batch")
    condition_np, _ = block.condition_features_for_positions(positions, require_reference=True)
    z_base = torch.from_numpy(block.z_base[positions].astype(np.float32)).to(device=device, non_blocking=True)
    condition = torch.from_numpy(condition_np.astype(np.float32, copy=False)).to(device=device, non_blocking=True)
    return z_base, condition


def _baseline_thresholds(block: LocalGraphBaselineLogitBlock) -> dict[str, float]:
    reference = block.condition_reference(require=True)
    return {
        "tau50": float(reference["tau50"]),
        "tau30": float(reference["tau30"]),
    }


def run_local_graph_residual_expert_epoch(
    model,
    loader,
    *,
    baseline_block: LocalGraphBaselineLogitBlock,
    device,
    loss_config: LocalGraphResidualLossConfig,
    optimizer=None,
    scaler=None,
    amp: bool = True,
    grad_clip_norm: float | None = 1.0,
    max_batches: int | None = None,
    collect_predictions: bool = False,
    collect_diagnostics: bool = False,
    label_names: tuple[str, ...] = LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES,
) -> dict[str, Any]:
    """Run one train/eval epoch for the residual expert."""

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
    thresholds = _baseline_thresholds(baseline_block)
    autocast_enabled = bool(amp and device.type == "cuda")
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = _move_batch_to_device(batch, device)
            z_base, condition = _baseline_arrays_for_batch(
                baseline_block,
                batch["indices"],
                labels=batch["labels"],
                device=device,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                output = model(
                    batch["tokens"],
                    batch["mask"],
                    baseline_logit=z_base,
                    baseline_condition_features=condition,
                    return_outputs=True,
                )
                loss_output = compute_local_graph_residual_loss_from_output(
                    output,
                    batch["labels"],
                    tau50=thresholds["tau50"],
                    tau30=thresholds["tau30"],
                    config=loss_config,
                )
                loss = loss_output.total_loss
            if training:
                if not bool(getattr(loss, "requires_grad", False)):
                    raise RuntimeError("residual expert training loss does not require grad")
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
                correction_logit = output.alpha.reshape(()) * output.residual_logit
                collected_correction.append(binary_logits_from_log_odds(correction_logit.detach().cpu().numpy()).astype(np.float32))
                collected_labels.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
                collected_indices.append(batch["indices"].detach().cpu().numpy().astype(np.int64))

    if total_seen == 0:
        return {"loss": float("nan"), "accuracy": 0.0, "n_jets": 0}

    metrics: dict[str, Any] = {
        "loss": total_loss / float(total_seen),
        "n_jets": int(total_seen),
        "loss_components": {
            key: value / float(total_seen)
            for key, value in sorted(component_totals.items())
        },
        "loss_weights": dict(loss_config.active_weights()),
        "baseline_thresholds": thresholds,
    }
    if diagnostic_weight_sum > 0.0:
        metrics["diagnostics"] = {
            key: value / diagnostic_weight_sum
            for key, value in sorted(diagnostic_totals.items())
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
        metrics.update(fused_metrics)
        metrics["fused_metrics"] = fused_metrics
        metrics["baseline_metrics"] = baseline_metrics
        metrics["residual_metrics"] = residual_metrics
        metrics["_prediction_arrays"] = {
            "indices": indices_np,
            "labels": labels_np,
            "fused_logits": fused_logits,
            "baseline_logits": baseline_logits,
            "residual_logits": residual_logits,
            "correction_logits": correction_logits,
        }
    return metrics


def local_graph_residual_expert_checkpoint_payload(
    model,
    optimizer,
    *,
    epoch: int,
    config: LocalGraphResidualExpertTrainConfig,
    loss_config: LocalGraphResidualLossConfig,
    metrics: Mapping[str, Any],
    source: Mapping[str, Any],
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
        "experiment_step": LOCAL_GRAPH_RESIDUAL_EXPERT_TRAIN_STEP,
        "protocol_step": LOCAL_GRAPH_PART_PROTOCOL_STEP,
        "protocol_contract": LOCAL_GRAPH_PART_CONTRACT,
        "loss_step": LOCAL_GRAPH_RESIDUAL_LOSS_STEP,
        "loss_contract": LOCAL_GRAPH_RESIDUAL_LOSS_CONTRACT,
        "output_contract": model.output_contract,
    }


def load_local_graph_residual_expert_checkpoint(path: str | Path, *, device=None, part_model: Any | None = None):
    torch = require_torch()
    payload = torch.load(path, map_location=device or "cpu")
    config_payload = payload.get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("checkpoint payload does not contain a residual train config")
    config = LocalGraphResidualExpertTrainConfig(**dict(config_payload))
    model = build_local_graph_residual_expert(config.model_config(), part_model=part_model)
    model.load_state_dict(payload["model_state_dict"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, payload


def train_local_graph_residual_expert(
    config: LocalGraphResidualExpertTrainConfig,
    *,
    model: LocalGraphResidualExpert | None = None,
    train_dataset: SubtokenHLTJetDataset | None = None,
    val_dataset: SubtokenHLTJetDataset | None = None,
    baseline_blocks: Mapping[str, LocalGraphBaselineLogitBlock] | None = None,
) -> dict[str, Any]:
    """Train/select one residual expert on model_train/model_val."""

    run_start_time = time.perf_counter()
    torch = require_torch()
    set_training_seed(int(config.seed))
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = train_dataset or _load_residual_dataset(config, config.train_split, max_jets=config.max_train_jets)
    val_dataset = val_dataset or _load_residual_dataset(config, config.val_split, max_jets=config.max_val_jets)
    if train_dataset.tokens.shape[-1] != RAW_TOKEN_DIM or val_dataset.tokens.shape[-1] != RAW_TOKEN_DIM:
        raise ValueError(f"residual expert expects raw token dim {RAW_TOKEN_DIM}")

    baseline_blocks = dict(baseline_blocks or {})
    train_block = baseline_blocks.get(config.train_split) or load_baseline_logit_block(
        config.baseline_logit_cache_dir,
        config.train_split,
        require_metadata=True,
    )
    val_block = baseline_blocks.get(config.val_split) or load_baseline_logit_block(
        config.baseline_logit_cache_dir,
        config.val_split,
        require_metadata=True,
    )
    train_alignment = verify_baseline_logit_block_alignment(
        train_block,
        train_dataset.metadata,
        split=config.train_split,
        dataset_length=len(train_dataset),
    )
    val_alignment = verify_baseline_logit_block_alignment(
        val_block,
        val_dataset.metadata,
        split=config.val_split,
        dataset_length=len(val_dataset),
    )
    baseline_cache_family = verify_baseline_logit_cache_family(
        (train_block, val_block),
        require_condition_reference=True,
    )

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

    checkpoint_model = model or build_local_graph_residual_expert(config.model_config())
    checkpoint_model = checkpoint_model.to(device)
    warm_start_report = None
    if config.warm_start_checkpoint:
        warm_start_report = warm_start_local_graph_part_model(
            checkpoint_model,
            config.warm_start_checkpoint,
            device=device,
            require=bool(config.require_warm_start),
        )
        save_json(diagnostics_dir / "warm_start_report.json", warm_start_report)

    freeze_schedule = {
        "enabled": int(config.freeze_part_epochs) > 0,
        "freeze_part_epochs": int(config.freeze_part_epochs),
        "adapter_only_epochs": int(config.freeze_part_epochs),
        "full_finetune_starts_epoch": int(config.freeze_part_epochs) + 1 if int(config.freeze_part_epochs) > 0 else 1,
    }
    freeze_events: list[dict[str, Any]] = []
    if int(config.freeze_part_epochs) > 0:
        event = {"epoch": 1, "event": "freeze_part_model"}
        event.update(_set_part_model_trainable(checkpoint_model, False))
        freeze_events.append(event)

    train_model = checkpoint_model
    if bool(config.compile_model) and hasattr(torch, "compile"):
        train_model = torch.compile(train_model)

    loss_config = config.loss_config(train_block)
    optimizer = torch.optim.AdamW(checkpoint_model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))
    source = source_metadata()
    run_metadata = {
        "experiment_step": LOCAL_GRAPH_RESIDUAL_EXPERT_TRAIN_STEP,
        "contract": LOCAL_GRAPH_RESIDUAL_EXPERT_TRAIN_CONTRACT,
        "protocol": local_graph_part_protocol_manifest(),
        "output_contract": checkpoint_model.output_contract,
        "variant": checkpoint_model.config.variant,
        "config": asdict(config),
        "model_config": checkpoint_model.to_config_dict(),
        "loss_config": loss_config.to_dict(),
        "source": source,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "baseline_logit_cache_dir": str(config.baseline_logit_cache_dir),
        "train_baseline_cache": train_block.to_manifest_row(),
        "val_baseline_cache": val_block.to_manifest_row(),
        "baseline_cache_alignment": {
            "train": train_alignment,
            "val": val_alignment,
            "family": baseline_cache_family,
        },
        "selection_metric": str(config.selection_metric),
        "warm_start": warm_start_report,
        "freeze_schedule": freeze_schedule,
        "leakage_rule": (
            "Step 4 residual training uses model_train and model_val only. stack_train, stack_val, "
            "and final_test are intentionally not loaded in this step."
        ),
        "stack_or_final_loaded": False,
        "inference_consumes_hlt_only": True,
    }
    save_json(output_dir / "config.json", run_metadata)

    curves: list[dict[str, Any]] = []
    best_val_score = float("-inf")
    best_selection_metric_value = float("nan")
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, int(config.epochs) + 1):
        if int(config.freeze_part_epochs) > 0 and epoch == int(config.freeze_part_epochs) + 1:
            event = {"epoch": int(epoch), "event": "unfreeze_part_model"}
            event.update(_set_part_model_trainable(checkpoint_model, True))
            freeze_events.append(event)

        train_metrics = run_local_graph_residual_expert_epoch(
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
        )
        val_metrics = run_local_graph_residual_expert_epoch(
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
        )
        val_metrics_for_json = _strip_prediction_arrays(val_metrics)
        row = {
            "epoch": int(epoch),
            "phase": "adapter_only" if int(epoch) <= int(config.freeze_part_epochs) else "full_finetune",
            "train": train_metrics,
            "model_val": val_metrics_for_json,
        }
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves})

        val_score, selection_value = _selection_score(val_metrics, str(config.selection_metric))
        val_loss = _finite_float(val_metrics.get("loss"), default=float("inf"))
        improved = val_score > best_val_score or (np.isclose(val_score, best_val_score) and val_loss < best_val_loss)
        payload = local_graph_residual_expert_checkpoint_payload(
            checkpoint_model,
            optimizer,
            epoch=epoch,
            config=config,
            loss_config=loss_config,
            metrics=row,
            source=source,
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

    if int(config.freeze_part_epochs) > 0:
        event = {"epoch": len(curves), "event": "restore_part_model_trainable_after_training"}
        event.update(_set_part_model_trainable(checkpoint_model, True))
        freeze_events.append(event)

    if best_epoch < 0 or not (output_dir / "best_model_val.pt").exists():
        raise FloatingPointError("residual expert did not produce a valid model_val checkpoint")

    best_payload = torch.load(output_dir / "best_model_val.pt", map_location=device)
    if model is None:
        best_model, _ = load_local_graph_residual_expert_checkpoint(output_dir / "best_model_val.pt", device=device)
    else:
        checkpoint_model.load_state_dict(best_payload["model_state_dict"])
        checkpoint_model.eval()
        best_model = checkpoint_model

    best_val_metrics = run_local_graph_residual_expert_epoch(
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
    )
    prediction_arrays = best_val_metrics.get("_prediction_arrays", {})
    alpha_report = None
    if isinstance(prediction_arrays, Mapping):
        baseline_logits = np.asarray(prediction_arrays.get("baseline_logits"), dtype=np.float32)
        residual_logits = np.asarray(prediction_arrays.get("residual_logits"), dtype=np.float32)
        correction_logits = np.asarray(prediction_arrays.get("correction_logits", residual_logits), dtype=np.float32)
        fused_logits = np.asarray(prediction_arrays.get("fused_logits"), dtype=np.float32)
        labels_np = np.asarray(prediction_arrays.get("labels"), dtype=np.int64)
        if baseline_logits.ndim == 2 and correction_logits.ndim == 2 and labels_np.ndim == 1:
            alpha_report = select_alpha_shrinkage(
                labels=labels_np,
                baseline_logit=baseline_logits[:, 1] - baseline_logits[:, 0],
                residual_logit=correction_logits[:, 1] - correction_logits[:, 0],
            )
            alpha_report["shrinkage_applies_to"] = "learned_correction_delta"
    residual_diagnostics = None
    if isinstance(prediction_arrays, Mapping):
        baseline_logits = np.asarray(prediction_arrays.get("baseline_logits"), dtype=np.float32)
        fused_logits = np.asarray(prediction_arrays.get("fused_logits"), dtype=np.float32)
        residual_logits = np.asarray(prediction_arrays.get("residual_logits"), dtype=np.float32)
        correction_logits = np.asarray(prediction_arrays.get("correction_logits", residual_logits), dtype=np.float32)
        labels_np = np.asarray(prediction_arrays.get("labels"), dtype=np.int64)
        indices_np = np.asarray(prediction_arrays.get("indices"), dtype=np.int64)
        if (
            baseline_logits.ndim == 2
            and fused_logits.ndim == 2
            and residual_logits.ndim == 2
            and labels_np.ndim == 1
        ):
            residual_diagnostics = residual_correction_diagnostics(
                labels=labels_np,
                baseline_logits=baseline_logits,
                fused_logits=fused_logits,
                residual_logits=correction_logits,
                indices=indices_np if indices_np.ndim == 1 else None,
                alpha_report=alpha_report,
            )
            save_json(diagnostics_dir / "residual_diagnostics_model_val.json", residual_diagnostics)
    best_val_metrics_json = _strip_prediction_arrays(best_val_metrics)
    elapsed_seconds = float(time.perf_counter() - run_start_time)

    report = {
        "experiment_step": LOCAL_GRAPH_RESIDUAL_EXPERT_TRAIN_STEP,
        "contract": LOCAL_GRAPH_RESIDUAL_EXPERT_TRAIN_CONTRACT,
        "protocol_step": LOCAL_GRAPH_PART_PROTOCOL_STEP,
        "protocol_contract": LOCAL_GRAPH_PART_CONTRACT,
        "protocol": local_graph_part_protocol_manifest(),
        "loss_step": LOCAL_GRAPH_RESIDUAL_LOSS_STEP,
        "loss_contract": LOCAL_GRAPH_RESIDUAL_LOSS_CONTRACT,
        "model_step": LOCAL_GRAPH_RESIDUAL_EXPERT_STEP,
        "output_contract": LOCAL_GRAPH_RESIDUAL_EXPERT_CONTRACT,
        "variant": best_model.config.variant,
        "best_epoch": int(best_epoch),
        "selection_metric": str(config.selection_metric),
        "selection_metric_direction": "minimize",
        "best_model_selection_metric_value": float(best_selection_metric_value),
        "best_model_selection_score": float(best_val_score),
        "best_model_val_loss": float(best_val_loss),
        "best_model_val_metrics": best_val_metrics_json,
        "baseline_model_val_metrics": best_val_metrics_json.get("baseline_metrics"),
        "fused_model_val_metrics": best_val_metrics_json.get("fused_metrics"),
        "residual_model_val_metrics": best_val_metrics_json.get("residual_metrics"),
        "alpha_shrinkage_model_val": alpha_report,
        "diagnostics_step": LOCAL_GRAPH_RESIDUAL_DIAGNOSTICS_STEP,
        "diagnostics_contract": LOCAL_GRAPH_RESIDUAL_DIAGNOSTICS_CONTRACT,
        "residual_diagnostics_model_val": residual_diagnostics,
        "residual_diagnostics_model_val_path": (
            str(diagnostics_dir / "residual_diagnostics_model_val.json")
            if residual_diagnostics is not None
            else None
        ),
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
        "train_baseline_cache": train_block.to_manifest_row(),
        "val_baseline_cache": val_block.to_manifest_row(),
        "baseline_cache_alignment": {
            "train": train_alignment,
            "val": val_alignment,
            "family": baseline_cache_family,
        },
        "warm_start": warm_start_report,
        "freeze_schedule": freeze_schedule,
        "freeze_events": freeze_events,
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
    "LOCAL_GRAPH_RESIDUAL_EXPERT_TRAIN_CONTRACT",
    "LOCAL_GRAPH_RESIDUAL_EXPERT_TRAIN_STEP",
    "LOCAL_GRAPH_RESIDUAL_LOWER_IS_BETTER_SELECTION_METRICS",
    "LOCAL_GRAPH_RESIDUAL_SELECTION_METRIC",
    "LocalGraphResidualExpertTrainConfig",
    "load_local_graph_residual_expert_checkpoint",
    "local_graph_residual_expert_checkpoint_payload",
    "run_local_graph_residual_expert_epoch",
    "train_local_graph_residual_expert",
]
