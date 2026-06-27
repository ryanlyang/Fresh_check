"""Step 7 training loop for the reliability-gated dual-view ParT model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json, set_training_seed

from teacher_logit_reco.set_matching.five_view_train import (
    classification_metrics_from_predictions,
)

from .anchor import load_hlt_part_anchor
from .config import (
    DUALVIEW_PART_NUM_CLASSES,
    DUALVIEW_PART_PRIMARY_METRIC,
    DUALVIEW_PART_SOURCE_LABEL_NAMES,
    DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL,
    DualViewPartExperimentConfig,
    dualview_metric_direction,
)
from .data import (
    DualViewPartDatasetConfig,
    DualViewPartJetDataset,
    make_dualview_part_loader,
)
from .diagnostics import (
    DUALVIEW_PART_DIAGNOSTICS_CONTRACT,
    DUALVIEW_PART_STEP8,
    build_residual_case_rows,
    summarize_residual_behavior,
    write_residual_diagnostics,
)
from .pn_encoder import PNMemoryEncoderConfig, build_pn_memory_encoder
from .reliability import ReliabilityFeatureConfig, reliability_feature_dim
from .residual import (
    DUALVIEW_PART_RESIDUAL_CONTRACT,
    DUALVIEW_PART_STEP6,
    DualViewResidualParT,
    DualViewResidualParTConfig,
    build_dualview_residual_part,
)


DUALVIEW_PART_STEP7 = "reliability_gated_dualview_part_step7_train_residual"
DUALVIEW_PART_STEP9 = "reliability_gated_dualview_part_step9_shuffled_pn_control"
DUALVIEW_PART_STEP10 = "reliability_gated_dualview_part_step10_smoke_test"
DUALVIEW_PART_TRAINING_CONTRACT = "dualview_part_residual_training_stack_val_selected_v1"
DUALVIEW_PART_SHUFFLED_PN_CONTRACT = "dualview_part_pn_view_row_shuffle_negative_control_v1"
DUALVIEW_PART_SELECTION_METRICS: tuple[str, ...] = (
    "accuracy",
    "loss",
    "macro_per_class_accuracy",
    "auc",
    "fpr_at_signal_eff_0p30",
    "fpr_at_signal_eff_0p50",
    "background_rejection_at_signal_eff_0p30",
    "background_rejection_at_signal_eff_0p50",
)
LOWER_IS_BETTER_SELECTION_METRICS = {"loss", "fpr_at_signal_eff_0p30", "fpr_at_signal_eff_0p50"}


def _optional_positive_int(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive when provided")
    return value


def _optional_nonnegative_int(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative when provided")
    return value


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float(default)
    return output if np.isfinite(output) else float(default)


def _cuda_autocast(torch_module, *, enabled: bool):
    if hasattr(torch_module, "amp") and hasattr(torch_module.amp, "autocast"):
        return torch_module.amp.autocast("cuda", enabled=bool(enabled))
    return torch_module.cuda.amp.autocast(enabled=bool(enabled))


def _cuda_grad_scaler(torch_module, *, enabled: bool):
    if hasattr(torch_module, "amp") and hasattr(torch_module.amp, "GradScaler"):
        try:
            return torch_module.amp.GradScaler("cuda", enabled=bool(enabled))
        except TypeError:  # pragma: no cover - older torch signature
            return torch_module.amp.GradScaler(enabled=bool(enabled))
    return torch_module.cuda.amp.GradScaler(enabled=bool(enabled))


@dataclass
class DualViewResidualTrainConfig:
    """Configuration for Step 7 residual training."""

    output_dir: str
    hlt_anchor_checkpoint: str | None = None
    hlt_cache_dir: str | None = None
    pn_reconstructed_view_dir: str | None = None
    experiment_dir: str | None = None
    diagnostics_mirror_dir: str | None = None

    train_split: str = "stack_train"
    val_split: str = "stack_val"
    final_test_split: str = "final_test"
    confirm_split_settings: bool = False
    confirm_final_test: bool = False

    seed: int = 2205
    batch_size: int = 64
    eval_batch_size: int = 128
    epochs: int = 45
    lr: float = 3.0e-4
    anchor_lr: float | None = None
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 6
    selection_metric: str = DUALVIEW_PART_PRIMARY_METRIC

    max_train_jets: int | None = None
    max_val_jets: int | None = None
    max_final_test_jets: int | None = None
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_final_test_batches: int | None = None
    run_initialization_check: bool = True
    initialization_check_batches: int = 1
    max_case_rows_per_type: int | None = 1000

    freeze_anchor: bool = True
    enforce_anchor_contract: bool = True
    anchor_model_size: str = "base"
    anchor_context_dim: int = 128
    anchor_summary_hidden_dim: int = 128
    anchor_summary_dropout: float = 0.0
    anchor_strict: bool = True

    max_hlt_constits: int = 128
    hlt_weight_threshold: float = 0.0
    max_pn_tokens: int = 128
    min_pn_tokens: int = 8
    confidence_threshold: float = 0.05
    selection_mode: str = "topk_or_threshold"
    verify_hlt_hash: bool = True
    enforce_dataset_contract: bool = True
    enforce_split_size: bool = False

    pn_embed_dim: int = 128
    pn_layers: int = 2
    pn_heads: int = 4
    pn_mlp_ratio: float = 4.0
    pn_dropout: float = 0.05
    pn_attention_dropout: float = 0.05
    pn_use_confidence: bool = True

    residual_hidden_dim: int = 128
    residual_layers: int = 2
    residual_dropout: float = 0.05
    gate_bias_init: float = -5.0
    use_anchor_context: bool = True
    use_reliability_features: bool = True

    compile_model: bool = False
    shuffle_pn_view: bool = False
    pn_view_shuffle_seed: int = 2205
    label_names: tuple[str, ...] = DUALVIEW_PART_SOURCE_LABEL_NAMES
    num_classes: int = DUALVIEW_PART_NUM_CLASSES

    def __post_init__(self) -> None:
        if self.train_split != "stack_train" or self.val_split != "stack_val" or self.final_test_split != "final_test":
            raise ValueError("Step 7 trains on stack_train, selects on stack_val, and reserves final_test")
        if not bool(self.confirm_split_settings):
            raise ValueError("Set --confirm-split-settings to acknowledge stack_train/stack_val-only model selection")
        self.output_dir = str(self.output_dir)
        if self.hlt_anchor_checkpoint is not None:
            self.hlt_anchor_checkpoint = str(self.hlt_anchor_checkpoint)
        for field_name in ("batch_size", "eval_batch_size", "epochs", "anchor_context_dim", "anchor_summary_hidden_dim"):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            setattr(self, field_name, value)
        for field_name in ("max_hlt_constits", "max_pn_tokens", "pn_embed_dim", "pn_layers", "pn_heads", "residual_hidden_dim", "residual_layers"):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            setattr(self, field_name, value)
        if int(self.min_pn_tokens) < 0:
            raise ValueError("min_pn_tokens cannot be negative")
        if int(self.min_pn_tokens) > int(self.max_pn_tokens):
            raise ValueError("min_pn_tokens cannot exceed max_pn_tokens")
        if int(self.num_workers) < 0:
            raise ValueError("num_workers cannot be negative")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be positive")
        if self.anchor_lr is not None and float(self.anchor_lr) <= 0.0:
            raise ValueError("anchor_lr must be positive when provided")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if float(self.grad_clip_norm) < 0.0:
            raise ValueError("grad_clip_norm cannot be negative")
        if int(self.early_stop_patience) < -1:
            raise ValueError("early_stop_patience must be -1 or greater")
        if not 0.0 <= float(self.confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        for field_name in ("anchor_summary_dropout", "pn_dropout", "pn_attention_dropout", "residual_dropout"):
            value = float(getattr(self, field_name))
            if value < 0.0 or value >= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1)")
            setattr(self, field_name, value)
        self.selection_metric = str(self.selection_metric)
        if self.selection_metric not in DUALVIEW_PART_SELECTION_METRICS:
            raise ValueError(f"selection_metric must be one of {DUALVIEW_PART_SELECTION_METRICS}")
        self.max_train_jets = _optional_positive_int(self.max_train_jets, field_name="max_train_jets")
        self.max_val_jets = _optional_positive_int(self.max_val_jets, field_name="max_val_jets")
        self.max_final_test_jets = _optional_positive_int(self.max_final_test_jets, field_name="max_final_test_jets")
        self.max_train_batches = _optional_nonnegative_int(self.max_train_batches, field_name="max_train_batches")
        self.max_val_batches = _optional_nonnegative_int(self.max_val_batches, field_name="max_val_batches")
        self.max_final_test_batches = _optional_nonnegative_int(
            self.max_final_test_batches,
            field_name="max_final_test_batches",
        )
        self.max_case_rows_per_type = _optional_nonnegative_int(
            self.max_case_rows_per_type,
            field_name="max_case_rows_per_type",
        )
        self.initialization_check_batches = int(self.initialization_check_batches)
        if self.initialization_check_batches <= 0:
            raise ValueError("initialization_check_batches must be positive")
        self.run_initialization_check = bool(self.run_initialization_check)
        self.label_names = tuple(str(name) for name in self.label_names)
        if int(self.num_classes) != len(self.label_names):
            raise ValueError("num_classes must match len(label_names)")
        if int(self.pn_embed_dim) % int(self.pn_heads) != 0:
            raise ValueError("pn_embed_dim must be divisible by pn_heads")
        if not bool(self.use_reliability_features):
            raise ValueError("Step 7 residual training currently requires reliability features")
        self.freeze_anchor = bool(self.freeze_anchor)
        self.enforce_anchor_contract = bool(self.enforce_anchor_contract)
        self.verify_hlt_hash = bool(self.verify_hlt_hash)
        self.enforce_dataset_contract = bool(self.enforce_dataset_contract)
        self.enforce_split_size = bool(self.enforce_split_size)
        self.shuffle_pn_view = bool(self.shuffle_pn_view)
        self.pn_view_shuffle_seed = int(self.pn_view_shuffle_seed)
        if self.pn_view_shuffle_seed < 0:
            raise ValueError("pn_view_shuffle_seed cannot be negative")

    @property
    def selection_metric_direction(self) -> str:
        return dualview_metric_direction(self.selection_metric)

    def experiment_config(self) -> DualViewPartExperimentConfig:
        return DualViewPartExperimentConfig(
            selection_metric=str(self.selection_metric),
            allow_noncanonical=False,
        )

    def dataset_config(self, split: str) -> DualViewPartDatasetConfig:
        return DualViewPartDatasetConfig(
            output_dir=self.experiment_dir,
            hlt_cache_dir=self.hlt_cache_dir,
            pn_reconstructed_view_dir=self.pn_reconstructed_view_dir,
            split=split,
            max_hlt_constits=int(self.max_hlt_constits),
            max_pn_tokens=int(self.max_pn_tokens),
            min_pn_tokens=int(self.min_pn_tokens),
            confidence_threshold=float(self.confidence_threshold),
            selection_mode=str(self.selection_mode),
            verify_hlt_hash=bool(self.verify_hlt_hash),
            enforce_canonical_contract=bool(self.enforce_dataset_contract),
            enforce_split_size=bool(self.enforce_split_size),
        )

    def pn_encoder_config(self) -> PNMemoryEncoderConfig:
        return PNMemoryEncoderConfig(
            embed_dim=int(self.pn_embed_dim),
            num_layers=int(self.pn_layers),
            num_heads=int(self.pn_heads),
            mlp_ratio=float(self.pn_mlp_ratio),
            dropout=float(self.pn_dropout),
            attention_dropout=float(self.pn_attention_dropout),
            use_confidence=bool(self.pn_use_confidence),
        )

    def residual_model_config(self) -> DualViewResidualParTConfig:
        return DualViewResidualParTConfig(
            num_classes=int(self.num_classes),
            hlt_context_dim=int(self.anchor_context_dim) if bool(self.use_anchor_context) else 0,
            pn_context_dim=int(self.pn_embed_dim),
            reliability_dim=reliability_feature_dim(),
            hidden_dim=int(self.residual_hidden_dim),
            num_hidden_layers=int(self.residual_layers),
            dropout=float(self.residual_dropout),
            gate_bias_init=float(self.gate_bias_init),
            zero_initialize_delta=True,
            zero_initialize_gate_head=True,
        )


def _slice_dataset(dataset: DualViewPartJetDataset, max_jets: int | None) -> DualViewPartJetDataset:
    if max_jets is None or int(max_jets) >= len(dataset):
        return dataset
    limit = int(max_jets)
    metadata = dict(dataset.metadata)
    metadata["limited_from_n_jets"] = len(dataset)
    metadata["n_jets"] = limit
    metadata["max_jets_limit"] = limit
    return DualViewPartJetDataset(
        hlt_tokens=dataset.hlt_tokens[:limit],
        hlt_mask=dataset.hlt_mask[:limit],
        pn_reco_tokens=dataset.pn_reco_tokens[:limit],
        pn_reco_mask=dataset.pn_reco_mask[:limit],
        pn_reco_confidence=dataset.pn_reco_confidence[:limit],
        labels=dataset.labels[:limit],
        jet_ids=dataset.jet_ids[:limit],
        split=dataset.split,
        pn_source_indices=dataset.pn_source_indices[:limit],
        metadata=metadata,
    )


def _load_dataset(config: DualViewResidualTrainConfig, split: str, *, max_jets: int | None) -> DualViewPartJetDataset:
    return _slice_dataset(DualViewPartJetDataset.from_caches(config.dataset_config(split)), max_jets)


def _stable_split_seed_offset(split: str) -> int:
    text = str(split)
    return sum((index + 1) * ord(char) for index, char in enumerate(text))


def shuffle_dualview_part_pn_view(
    dataset: DualViewPartJetDataset,
    *,
    seed: int,
    split: str | None = None,
) -> DualViewPartJetDataset:
    """Return a negative-control dataset with PN rows shuffled within one split.

    HLT tokens, labels, jet identities, and sample ordering are kept fixed.  The
    PN reconstructed view arrays move together as rows, so the control preserves
    the marginal PN distribution while destroying event-level HLT/PN alignment.
    """

    n_jets = len(dataset)
    resolved_split = str(split or dataset.split)
    rng = np.random.default_rng(int(seed) + _stable_split_seed_offset(resolved_split))
    permutation = rng.permutation(n_jets).astype(np.int64, copy=False)
    if n_jets > 1 and np.array_equal(permutation, np.arange(n_jets, dtype=np.int64)):
        permutation = np.roll(permutation, 1)
    identity_fraction = float(np.mean(permutation == np.arange(n_jets, dtype=np.int64))) if n_jets else 0.0
    metadata = dict(dataset.metadata)
    metadata["pn_view_shuffle_control"] = {
        "enabled": True,
        "experiment_step": DUALVIEW_PART_STEP9,
        "output_contract": DUALVIEW_PART_SHUFFLED_PN_CONTRACT,
        "description": "PN reconstructed view rows are shuffled within split; HLT rows and labels stay fixed.",
        "seed": int(seed),
        "split": resolved_split,
        "n_jets": int(n_jets),
        "identity_fraction": identity_fraction,
        "permutation_preview": [int(value) for value in permutation[: min(16, n_jets)].tolist()],
    }
    metadata["content_hash_note"] = "content_hash, alignment_audit, and jet_identity_hash describe the unshuffled source cache"
    return DualViewPartJetDataset(
        hlt_tokens=dataset.hlt_tokens.copy(),
        hlt_mask=dataset.hlt_mask.copy(),
        pn_reco_tokens=dataset.pn_reco_tokens[permutation].copy(),
        pn_reco_mask=dataset.pn_reco_mask[permutation].copy(),
        pn_reco_confidence=dataset.pn_reco_confidence[permutation].copy(),
        labels=dataset.labels.copy(),
        jet_ids=list(dataset.jet_ids),
        split=dataset.split,
        pn_source_indices=dataset.pn_source_indices[permutation].copy(),
        metadata=metadata,
    )


def _maybe_shuffle_pn_view(
    config: DualViewResidualTrainConfig,
    dataset: DualViewPartJetDataset,
    *,
    split: str,
) -> DualViewPartJetDataset:
    if not bool(config.shuffle_pn_view):
        return dataset
    return shuffle_dualview_part_pn_view(
        dataset,
        seed=int(config.pn_view_shuffle_seed),
        split=split,
    )


def _move_batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    torch = require_torch()
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "hlt_inputs":
            moved[key] = {subkey: subvalue.to(device, non_blocking=True) for subkey, subvalue in value.items()}
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _lookup_selection_metric(metrics: Mapping[str, Any], metric_name: str) -> float:
    if metric_name in metrics:
        return _finite_float(metrics[metric_name], default=float("nan"))
    binary = metrics.get("binary_metrics")
    if isinstance(binary, Mapping) and metric_name in binary:
        return _finite_float(binary[metric_name], default=float("nan"))
    raise KeyError(f"metrics do not contain selection metric {metric_name!r}")


def _selection_metric_requires_predictions(metric_name: str) -> bool:
    return metric_name not in {"accuracy", "loss"}


def _selection_score(metrics: Mapping[str, Any], metric_name: str) -> tuple[float, float]:
    value = _lookup_selection_metric(metrics, metric_name)
    if np.isnan(value):
        return float("-inf"), value
    if metric_name in LOWER_IS_BETTER_SELECTION_METRICS:
        return -float(value), float(value)
    return float(value), float(value)


def _accumulate_diag(totals: dict[str, float], diagnostics: Mapping[str, Any], *, batch_size: int) -> None:
    for key in (
        "gate_mean",
        "gate_std",
        "gate_min",
        "gate_max",
        "delta_abs_mean",
        "residual_abs_mean",
        "gated_delta_abs_mean",
    ):
        value = diagnostics.get(key)
        if isinstance(value, (int, float)):
            totals[key] = totals.get(key, 0.0) + float(value) * int(batch_size)


def run_dualview_residual_epoch(
    model: DualViewResidualParT,
    loader,
    *,
    device,
    criterion,
    optimizer=None,
    scaler=None,
    amp: bool = True,
    grad_clip_norm: float | None = 1.0,
    max_batches: int | None = None,
    collect_predictions: bool = False,
    collect_case_rows: bool = False,
    max_case_rows_per_type: int | None = 1000,
    label_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run one training/evaluation epoch for the dual-view residual model."""

    torch = require_torch()
    training = optimizer is not None
    model.train(training)
    autocast_enabled = bool(amp and device.type == "cuda")
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    diag_totals: dict[str, float] = {}
    collected_preds: list[np.ndarray] = []
    collected_labels: list[np.ndarray] = []
    collected_logits: list[np.ndarray] = []
    collected_hlt_logits: list[np.ndarray] = []
    collected_gate: list[np.ndarray] = []
    collected_delta_logits: list[np.ndarray] = []
    collected_residual_logits: list[np.ndarray] = []
    collected_indices: list[np.ndarray] = []
    collected_jet_ids: list[Any] = []
    collected_splits: list[str] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = _move_batch_to_device(batch, device)
            labels = batch["labels"]
            if training:
                optimizer.zero_grad(set_to_none=True)

            with _cuda_autocast(torch, enabled=autocast_enabled):
                output = model(
                    hlt_inputs=batch["hlt_inputs"],
                    hlt_tokens=batch["hlt_tokens"],
                    hlt_mask=batch["hlt_mask"],
                    pn_reco_tokens=batch["pn_reco_tokens"],
                    pn_reco_mask=batch["pn_reco_mask"],
                    pn_reco_confidence=batch["pn_reco_confidence"],
                    return_diagnostics=True,
                )
                loss = criterion(output.logits, labels)

            if training:
                if scaler is not None and bool(scaler.is_enabled()):
                    scaler.scale(loss).backward()
                    if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [param for param in model.parameters() if param.requires_grad],
                            float(grad_clip_norm),
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            [param for param in model.parameters() if param.requires_grad],
                            float(grad_clip_norm),
                        )
                    optimizer.step()

            batch_size = int(labels.shape[0])
            preds = torch.argmax(output.logits, dim=1)
            total_loss += float(loss.detach().item()) * batch_size
            total_correct += int((preds == labels).sum().item())
            total_seen += batch_size
            _accumulate_diag(diag_totals, output.diagnostics, batch_size=batch_size)
            if collect_predictions:
                collected_preds.append(preds.detach().cpu().numpy().astype(np.int64))
                collected_labels.append(labels.detach().cpu().numpy().astype(np.int64))
                collected_logits.append(output.logits.detach().cpu().numpy().astype(np.float32))
                collected_hlt_logits.append(output.hlt_logits.detach().cpu().numpy().astype(np.float32))
                collected_gate.append(output.gate.detach().cpu().numpy().astype(np.float32))
                collected_delta_logits.append(output.delta_logits.detach().cpu().numpy().astype(np.float32))
                collected_residual_logits.append(output.residual_logits.detach().cpu().numpy().astype(np.float32))
                if collect_case_rows:
                    collected_indices.append(batch["indices"].detach().cpu().numpy().astype(np.int64))
                    collected_jet_ids.extend(list(batch.get("jet_ids", [])))
                    collected_splits.append(str(batch.get("split", "")))

    if total_seen == 0:
        return {"loss": float("nan"), "accuracy": 0.0, "n_jets": 0}
    metrics: dict[str, Any] = {
        "loss": total_loss / float(total_seen),
        "accuracy": total_correct / float(total_seen),
        "n_jets": int(total_seen),
    }
    if diag_totals:
        metrics["residual_diagnostics"] = {key: value / float(total_seen) for key, value in sorted(diag_totals.items())}
    if collect_predictions:
        preds_np = np.concatenate(collected_preds, axis=0) if collected_preds else np.asarray([], dtype=np.int64)
        labels_np = np.concatenate(collected_labels, axis=0) if collected_labels else np.asarray([], dtype=np.int64)
        logits_np = np.concatenate(collected_logits, axis=0) if collected_logits else None
        hlt_logits_np = np.concatenate(collected_hlt_logits, axis=0) if collected_hlt_logits else None
        gate_np = np.concatenate(collected_gate, axis=0) if collected_gate else None
        delta_logits_np = np.concatenate(collected_delta_logits, axis=0) if collected_delta_logits else None
        residual_logits_np = np.concatenate(collected_residual_logits, axis=0) if collected_residual_logits else None
        indices_np = np.concatenate(collected_indices, axis=0) if collected_indices else None
        metrics.update(
            classification_metrics_from_predictions(
                preds=preds_np,
                labels=labels_np,
                loss_sum=total_loss,
                logits=logits_np,
                label_names=label_names,
            )
        )
        hlt_preds = np.argmax(hlt_logits_np, axis=1).astype(np.int64) if hlt_logits_np is not None else preds_np
        hlt_metrics = classification_metrics_from_predictions(
            preds=hlt_preds,
            labels=labels_np,
            logits=hlt_logits_np,
            label_names=label_names,
        )
        metrics["anchor_hlt_metrics"] = hlt_metrics
        metrics["prediction_change_fraction"] = float(np.mean(preds_np != hlt_preds)) if labels_np.size else 0.0
        if gate_np is not None and delta_logits_np is not None and residual_logits_np is not None:
            metrics["residual_analysis"] = summarize_residual_behavior(
                logits=logits_np,
                hlt_logits=hlt_logits_np,
                labels=labels_np,
                gate=gate_np,
                delta_logits=delta_logits_np,
                residual_logits=residual_logits_np,
                label_names=label_names,
            )
            if collect_case_rows:
                split_name = next((split for split in collected_splits if split), "")
                metrics["residual_case_rows"] = build_residual_case_rows(
                    split=split_name,
                    logits=logits_np,
                    hlt_logits=hlt_logits_np,
                    labels=labels_np,
                    gate=gate_np,
                    delta_logits=delta_logits_np,
                    residual_logits=residual_logits_np,
                    sample_indices=indices_np,
                    jet_ids=collected_jet_ids,
                    label_names=label_names,
                    max_cases_per_type=max_case_rows_per_type,
                )
    return metrics


def _make_optimizer(model: DualViewResidualParT, config: DualViewResidualTrainConfig):
    torch = require_torch()
    new_params = []
    anchor_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(("hlt_anchor.model.", "anchor.model.")):
            anchor_params.append(param)
        else:
            new_params.append(param)
    groups = []
    if new_params:
        groups.append({"params": new_params, "lr": float(config.lr), "weight_decay": float(config.weight_decay)})
    if anchor_params:
        groups.append(
            {
                "params": anchor_params,
                "lr": float(config.anchor_lr if config.anchor_lr is not None else config.lr * 0.1),
                "weight_decay": float(config.weight_decay),
            }
        )
    if not groups:
        raise ValueError("No trainable parameters for dual-view residual model")
    return torch.optim.AdamW(groups)


def _apply_anchor_training_mode(model: DualViewResidualParT, *, freeze_anchor: bool) -> None:
    if bool(freeze_anchor):
        model.hlt_anchor.freeze_anchor_parameters()
    else:
        model.hlt_anchor.unfreeze_anchor_parameters()


def _initialization_check_passed(metrics: Mapping[str, Any]) -> bool:
    diagnostics = metrics.get("residual_diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        return False
    return bool(
        _finite_float(metrics.get("prediction_change_fraction"), default=1.0) <= 1.0e-8
        and _finite_float(diagnostics.get("residual_abs_mean"), default=1.0) <= 1.0e-8
        and _finite_float(diagnostics.get("gate_mean"), default=1.0) < 0.02
        and np.isfinite(_finite_float(metrics.get("loss"), default=float("nan")))
    )


def _pop_residual_case_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = metrics.pop("residual_case_rows", [])
    if not rows:
        return []
    return [dict(row) for row in rows]


def build_dualview_residual_model_from_config(config: DualViewResidualTrainConfig, *, device) -> DualViewResidualParT:
    if config.hlt_anchor_checkpoint is None:
        raise ValueError("hlt_anchor_checkpoint is required when a model is not injected")
    anchor = load_hlt_part_anchor(
        config.hlt_anchor_checkpoint,
        device=str(device),
        freeze_anchor=bool(config.freeze_anchor),
        strict=bool(config.anchor_strict),
        context_mode="summary" if bool(config.use_anchor_context) else "none",
        context_dim=int(config.anchor_context_dim),
        summary_hidden_dim=int(config.anchor_summary_hidden_dim),
        summary_dropout=float(config.anchor_summary_dropout),
        max_constits=int(config.max_hlt_constits),
        weight_threshold=float(config.hlt_weight_threshold),
        fallback_num_classes=int(config.num_classes),
        fallback_model_size=str(config.anchor_model_size),
        fallback_label_names=tuple(config.label_names),
        enforce_canonical_contract=bool(config.enforce_anchor_contract),
    )
    pn_encoder = build_pn_memory_encoder(config.pn_encoder_config())
    model = build_dualview_residual_part(
        anchor,
        pn_encoder,
        config=config.residual_model_config(),
        reliability_config=ReliabilityFeatureConfig(max_constituents=int(config.max_hlt_constits)),
        infer_dims_from_modules=bool(config.use_anchor_context),
    )
    _apply_anchor_training_mode(model, freeze_anchor=bool(config.freeze_anchor))
    return model.to(device)


def dualview_residual_training_checkpoint_payload(
    model: DualViewResidualParT,
    optimizer,
    *,
    epoch: int,
    config: DualViewResidualTrainConfig,
    metrics: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "metrics": dict(metrics),
        "source": dict(source),
        "label_names": list(config.label_names),
        "num_classes": int(config.num_classes),
        "model_config": model.to_config_dict(),
        "experiment_step": DUALVIEW_PART_STEP7,
        "model_step": DUALVIEW_PART_STEP6,
        "output_contract": DUALVIEW_PART_TRAINING_CONTRACT,
        "residual_output_contract": DUALVIEW_PART_RESIDUAL_CONTRACT,
    }


def _write_epoch_metrics_csv(path: Path, curves: list[dict[str, Any]]) -> None:
    keys = {"epoch"}
    rows = []
    for row in curves:
        flat = {"epoch": row["epoch"]}
        for split in ("train", "stack_val"):
            for key, value in row.get(split, {}).items():
                if isinstance(value, (int, float)):
                    flat[f"{split}_{key}"] = value
                    keys.add(f"{split}_{key}")
            diagnostics = row.get(split, {}).get("residual_diagnostics", {})
            if isinstance(diagnostics, Mapping):
                for key, value in diagnostics.items():
                    flat[f"{split}_{key}"] = value
                    keys.add(f"{split}_{key}")
        rows.append(flat)
    ordered = ["epoch"] + sorted(key for key in keys if key != "epoch")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def _write_per_class_csv(path: Path, metrics_by_split: Mapping[str, Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for split, metrics in metrics_by_split.items():
        for row in metrics.get("per_class_accuracy", []):
            payload = {"split": split}
            payload.update(dict(row))
            rows.append(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "class_index", "class_name", "support", "correct", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def _mirror_diagnostics(config: DualViewResidualTrainConfig, output_dir: Path, diagnostics_dir: Path) -> None:
    if not config.diagnostics_mirror_dir:
        return
    mirror = Path(config.diagnostics_mirror_dir)
    mirror.mkdir(parents=True, exist_ok=True)
    for source in (
        output_dir / "run_report.json",
        output_dir / "model_val_report.json",
        output_dir / "training_curves.json",
        output_dir / "config.json",
        diagnostics_dir / "epoch_metrics.csv",
        diagnostics_dir / "per_class_metrics.csv",
        diagnostics_dir / "residual_diagnostics.json",
        diagnostics_dir / "gate_by_class.csv",
        diagnostics_dir / "gate_by_hlt_confidence.csv",
        diagnostics_dir / "gate_by_hlt_correctness.csv",
        diagnostics_dir / "prediction_change_summary.csv",
    ):
        if source.exists() and source.is_file():
            shutil.copy2(source, mirror / source.name)


def train_dualview_residual_part(
    config: DualViewResidualTrainConfig,
    *,
    model: DualViewResidualParT | None = None,
    train_dataset: DualViewPartJetDataset | None = None,
    val_dataset: DualViewPartJetDataset | None = None,
    final_test_dataset: DualViewPartJetDataset | None = None,
) -> dict[str, Any]:
    """Train the Step 7 residual model and optionally evaluate final_test."""

    torch = require_torch()
    set_training_seed(int(config.seed))
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = train_dataset or _load_dataset(config, config.train_split, max_jets=config.max_train_jets)
    val_dataset = val_dataset or _load_dataset(config, config.val_split, max_jets=config.max_val_jets)
    train_dataset = _maybe_shuffle_pn_view(config, train_dataset, split=config.train_split)
    val_dataset = _maybe_shuffle_pn_view(config, val_dataset, split=config.val_split)
    train_loader = make_dualview_part_loader(
        train_dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        seed=int(config.seed),
        max_hlt_constits=int(config.max_hlt_constits),
        hlt_weight_threshold=float(config.hlt_weight_threshold),
    )
    val_loader = make_dualview_part_loader(
        val_dataset,
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 1,
        max_hlt_constits=int(config.max_hlt_constits),
        hlt_weight_threshold=float(config.hlt_weight_threshold),
    )
    model = model or build_dualview_residual_model_from_config(config, device=device)
    model = model.to(device)
    _apply_anchor_training_mode(model, freeze_anchor=bool(config.freeze_anchor))
    checkpoint_model = model
    train_model = model
    if config.compile_model and hasattr(torch, "compile"):
        train_model = torch.compile(model)

    criterion = torch.nn.CrossEntropyLoss()
    initialization_check = None
    if bool(config.run_initialization_check):
        initialization_check = run_dualview_residual_epoch(
            checkpoint_model,
            val_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=int(config.initialization_check_batches),
            collect_predictions=True,
            label_names=tuple(config.label_names),
        )
        initialization_check["passed"] = _initialization_check_passed(initialization_check)
        if not bool(initialization_check["passed"]):
            raise FloatingPointError(
                "dual-view residual initialization check failed: model does not start as closed HLT anchor"
            )

    optimizer = _make_optimizer(model, config)
    scaler = _cuda_grad_scaler(torch, enabled=bool(config.amp and device.type == "cuda"))
    source = {
        "experiment_step": DUALVIEW_PART_STEP7,
        "output_contract": DUALVIEW_PART_TRAINING_CONTRACT,
        "variant": DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL if bool(config.freeze_anchor) else "warm_anchor_pn_residual",
        "anchor_checkpoint": config.hlt_anchor_checkpoint,
        "selection_metric": str(config.selection_metric),
        "selection_metric_direction": config.selection_metric_direction,
        "final_test_guard": "final_test is evaluated only after best stack_val checkpoint selection",
        "shuffle_pn_view": bool(config.shuffle_pn_view),
        "pn_view_shuffle_seed": int(config.pn_view_shuffle_seed),
        "negative_control_step": DUALVIEW_PART_STEP9 if bool(config.shuffle_pn_view) else None,
        "negative_control_contract": DUALVIEW_PART_SHUFFLED_PN_CONTRACT if bool(config.shuffle_pn_view) else None,
    }
    metadata = {
        "experiment_step": DUALVIEW_PART_STEP7,
        "output_contract": DUALVIEW_PART_TRAINING_CONTRACT,
        "config": asdict(config),
        "model_config": checkpoint_model.to_config_dict(),
        "source": source,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "train_n_jets": len(train_dataset),
        "val_n_jets": len(val_dataset),
        "initialization_check": initialization_check,
    }
    save_json(output_dir / "config.json", metadata)

    curves: list[dict[str, Any]] = []
    best_val_score = float("-inf")
    best_selection_metric_value = float("nan")
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, int(config.epochs) + 1):
        train_metrics = run_dualview_residual_epoch(
            train_model,
            train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            amp=bool(config.amp),
            grad_clip_norm=float(config.grad_clip_norm),
            max_batches=config.max_train_batches,
            label_names=tuple(config.label_names),
        )
        val_metrics = run_dualview_residual_epoch(
            train_model,
            val_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=config.max_val_batches,
            collect_predictions=_selection_metric_requires_predictions(str(config.selection_metric)),
            label_names=tuple(config.label_names),
        )
        row = {"epoch": int(epoch), "train": train_metrics, "stack_val": val_metrics}
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves})
        _write_epoch_metrics_csv(diagnostics_dir / "epoch_metrics.csv", curves)

        val_loss = _finite_float(val_metrics.get("loss"), default=float("inf"))
        val_accuracy = _finite_float(val_metrics.get("accuracy"), default=-1.0)
        val_score, selection_value = _selection_score(val_metrics, str(config.selection_metric))
        improved = val_score > best_val_score or (np.isclose(val_score, best_val_score) and val_loss < best_val_loss)
        payload = dualview_residual_training_checkpoint_payload(
            checkpoint_model,
            optimizer,
            epoch=epoch,
            config=config,
            metrics=row,
            source=source,
        )
        torch.save(payload, output_dir / "last.pt")
        if improved:
            best_val_score = float(val_score)
            best_selection_metric_value = float(selection_value)
            best_val_accuracy = float(val_accuracy)
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(payload, output_dir / "best_model_val.pt")
        else:
            epochs_without_improvement += 1
        if int(config.early_stop_patience) >= 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    if best_epoch < 0 or not (output_dir / "best_model_val.pt").exists():
        raise FloatingPointError("dual-view residual model did not produce a valid stack_val checkpoint")
    best_payload = torch.load(output_dir / "best_model_val.pt", map_location=device)
    checkpoint_model.load_state_dict(best_payload["model_state_dict"])
    best_val_metrics = run_dualview_residual_epoch(
        checkpoint_model,
        val_loader,
        device=device,
        criterion=criterion,
        amp=False,
        max_batches=config.max_val_batches,
        collect_predictions=True,
        collect_case_rows=True,
        max_case_rows_per_type=config.max_case_rows_per_type,
        label_names=tuple(config.label_names),
    )
    residual_case_rows_by_split = {"stack_val": _pop_residual_case_rows(best_val_metrics)}
    metrics_by_split = {"stack_val": best_val_metrics}
    final_test_metrics = None
    final_test_metadata = None
    if bool(config.confirm_final_test):
        final_test_dataset = final_test_dataset or _load_dataset(
            config,
            config.final_test_split,
            max_jets=config.max_final_test_jets,
        )
        final_test_dataset = _maybe_shuffle_pn_view(config, final_test_dataset, split=config.final_test_split)
        final_loader = make_dualview_part_loader(
            final_test_dataset,
            batch_size=int(config.eval_batch_size),
            shuffle=False,
            num_workers=int(config.num_workers),
            seed=int(config.seed) + 2,
            max_hlt_constits=int(config.max_hlt_constits),
            hlt_weight_threshold=float(config.hlt_weight_threshold),
        )
        final_test_metrics = run_dualview_residual_epoch(
            checkpoint_model,
            final_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=config.max_final_test_batches,
            collect_predictions=True,
            collect_case_rows=True,
            max_case_rows_per_type=config.max_case_rows_per_type,
            label_names=tuple(config.label_names),
        )
        residual_case_rows_by_split["final_test"] = _pop_residual_case_rows(final_test_metrics)
        metrics_by_split["final_test"] = final_test_metrics
        final_test_metadata = dict(final_test_dataset.metadata)

    _write_per_class_csv(diagnostics_dir / "per_class_metrics.csv", metrics_by_split)
    residual_diagnostic_files = write_residual_diagnostics(
        diagnostics_dir,
        {
            split: metrics.get("residual_analysis", {})
            for split, metrics in metrics_by_split.items()
        },
        case_rows_by_split=residual_case_rows_by_split,
    )
    report = {
        "experiment_step": DUALVIEW_PART_STEP7,
        "model_step": DUALVIEW_PART_STEP6,
        "diagnostics_step": DUALVIEW_PART_STEP8,
        "negative_control_step": DUALVIEW_PART_STEP9 if bool(config.shuffle_pn_view) else None,
        "output_contract": DUALVIEW_PART_TRAINING_CONTRACT,
        "diagnostics_output_contract": DUALVIEW_PART_DIAGNOSTICS_CONTRACT,
        "negative_control_contract": DUALVIEW_PART_SHUFFLED_PN_CONTRACT if bool(config.shuffle_pn_view) else None,
        "residual_output_contract": DUALVIEW_PART_RESIDUAL_CONTRACT,
        "shuffle_pn_view": bool(config.shuffle_pn_view),
        "pn_view_shuffle_seed": int(config.pn_view_shuffle_seed),
        "best_epoch": int(best_epoch),
        "selection_metric": str(config.selection_metric),
        "selection_metric_direction": config.selection_metric_direction,
        "best_model_selection_metric_value": float(best_selection_metric_value),
        "best_model_selection_score": float(best_val_score),
        "best_model_val_accuracy": float(best_val_accuracy),
        "best_model_val_loss": float(best_val_loss),
        "best_stack_val_metrics": best_val_metrics,
        "initialization_check": initialization_check,
        "initialization_check_passed": None if initialization_check is None else bool(initialization_check.get("passed")),
        "final_test_metrics": final_test_metrics,
        "epochs_completed": len(curves),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "training_curves": str(output_dir / "training_curves.json"),
        "per_class_metrics_csv": str(diagnostics_dir / "per_class_metrics.csv"),
        "residual_diagnostics": residual_diagnostic_files,
        "config": asdict(config),
        "model_config": checkpoint_model.to_config_dict(),
        "label_names": list(config.label_names),
        "num_classes": int(config.num_classes),
        "source": source,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "final_test_dataset": final_test_metadata,
        "final_test_evaluated": bool(config.confirm_final_test),
        "no_final_test_evaluation": not bool(config.confirm_final_test),
        "inference_consumes_hlt_plus_hlt_derived_pn_reco_only": True,
    }
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    _mirror_diagnostics(config, output_dir, diagnostics_dir)
    return report


__all__ = [
    "DUALVIEW_PART_SELECTION_METRICS",
    "DUALVIEW_PART_SHUFFLED_PN_CONTRACT",
    "DUALVIEW_PART_STEP7",
    "DUALVIEW_PART_STEP9",
    "DUALVIEW_PART_STEP10",
    "DUALVIEW_PART_TRAINING_CONTRACT",
    "DualViewResidualTrainConfig",
    "build_dualview_residual_model_from_config",
    "dualview_residual_training_checkpoint_payload",
    "run_dualview_residual_epoch",
    "shuffle_dualview_part_pn_view",
    "train_dualview_residual_part",
]
