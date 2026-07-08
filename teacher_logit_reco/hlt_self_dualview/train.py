"""Training, prediction caching, and metrics for HLT self-dualview models."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.fusion import PredictionBlock, save_prediction_block, softmax_np
from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json, set_training_seed
from jetclass_fresh.hlt_cache import jet_identity_hash
from jetclass_fresh.jetclass_data import LABEL_NAMES
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES

from teacher_logit_reco.privileged_distill_10class.config import (
    PD10_NUM_CLASSES,
    PD10_REPRESENTATION_DIM,
    PD10_SPLIT_ORDER,
    PD10_SPLIT_SIZES,
)
from teacher_logit_reco.privileged_distill_10class.metrics import pd10_prediction_metrics_from_logits

from .config import (
    HLT_SDV_ALLOWED_INPUTS,
    HLT_SDV_DEPLOYMENT_INPUTS,
    HLT_SDV_EXPERIMENT_NAME,
    HLT_SDV_VARIANT_SAME_VIEW,
    default_hlt_sdv_experiment_layout,
    hlt_sdv_strength_from_variant,
    normalize_hlt_sdv_variant,
)
from .data import (
    HLT_SDV_BRANCH2_HLT2,
    HLT_SDV_BRANCH2_SAME_HLT,
    HLTSelfDualViewDataset,
    collate_hlt_sdv_batch,
    hlt_sdv_branch2_mode_from_variant,
    load_hlt_sdv_dataset,
    make_hlt_sdv_data_loader,
    move_hlt_sdv_batch_to_device,
    normalize_hlt_sdv_branch2_mode,
)
from .hlt2_cache import default_hlt2_cache_dir
from .model import (
    HLT_SDV_DEFAULT_DROPOUT,
    HLT_SDV_DEFAULT_FUSION_HIDDEN_DIM,
    HLT_SDV_DEFAULT_MODEL_SIZE,
    HLT_SDV_MODEL_CONTRACT,
    HLT_SDV_MODEL_ARCHITECTURE,
    HLTSelfDualViewFusionModel,
    build_hlt_sdv_fusion_model,
    initialize_hlt_sdv_branches_from_checkpoints,
    sha256_file,
)


HLT_SDV_STEP5_EXPERIMENT_STEP = "hlt_sdv_step5_train_predict_metrics"
HLT_SDV_TRAINING_CONTRACT = "hlt_self_dualview_training_v1"
HLT_SDV_PREDICTION_CONTRACT = "hlt_self_dualview_prediction_cache_v1"
HLT_SDV_DEFAULT_SEED = 8801
HLT_SDV_DEFAULT_BATCH_SIZE = 128
HLT_SDV_DEFAULT_EVAL_BATCH_SIZE = 128
HLT_SDV_DEFAULT_EPOCHS = 10
HLT_SDV_DEFAULT_HEAD_WARMUP_EPOCHS = 1
HLT_SDV_DEFAULT_HEAD_WARMUP_LR = 3.0e-4
HLT_SDV_DEFAULT_BRANCH_LR = 3.0e-5
HLT_SDV_DEFAULT_HEAD_LR = 3.0e-4
HLT_SDV_DEFAULT_WEIGHT_DECAY = 1.0e-4
HLT_SDV_DEFAULT_EARLY_STOP_PATIENCE = 3


@dataclass(frozen=True)
class HLTSDVTrainConfig:
    """High-data training config for one HLT self-dualview model variant."""

    output_dir: str
    hlt_cache_dir: str
    hlt_teacher_checkpoint: str
    variant_name: str
    hlt2_branch_checkpoint: str | None = None
    hlt2_cache_dir: str | None = None
    branch2_mode: str | None = None
    train_split: str = "model_train"
    val_split: str = "model_val"
    final_test_split: str = "final_test"
    seed: int = HLT_SDV_DEFAULT_SEED
    batch_size: int = HLT_SDV_DEFAULT_BATCH_SIZE
    eval_batch_size: int = HLT_SDV_DEFAULT_EVAL_BATCH_SIZE
    epochs: int = HLT_SDV_DEFAULT_EPOCHS
    head_warmup_epochs: int = HLT_SDV_DEFAULT_HEAD_WARMUP_EPOCHS
    head_warmup_lr: float = HLT_SDV_DEFAULT_HEAD_WARMUP_LR
    branch_lr: float = HLT_SDV_DEFAULT_BRANCH_LR
    head_lr: float = HLT_SDV_DEFAULT_HEAD_LR
    weight_decay: float = HLT_SDV_DEFAULT_WEIGHT_DECAY
    num_workers: int = 0
    device: str = "auto"
    amp: bool = False
    grad_clip_norm: float = 1.0
    early_stop_patience: int = HLT_SDV_DEFAULT_EARLY_STOP_PATIENCE
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_final_test_batches: int | None = None
    max_train_jets: int | None = PD10_SPLIT_SIZES["model_train"]
    max_val_jets: int | None = PD10_SPLIT_SIZES["model_val"]
    max_final_test_jets: int | None = PD10_SPLIT_SIZES["final_test"]
    model_size: str = HLT_SDV_DEFAULT_MODEL_SIZE
    compile_model: bool = False
    fusion_hidden_dim: int = HLT_SDV_DEFAULT_FUSION_HIDDEN_DIM
    representation_dim: int = PD10_REPRESENTATION_DIM
    dropout: float = HLT_SDV_DEFAULT_DROPOUT
    verify_hlt_hash: bool = True
    verify_hlt2_hash: bool = True
    initialize_branches: bool = True
    evaluate_model_val_predictions: bool = True
    evaluate_final_test: bool = True
    confirm_final_test: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        variant = normalize_hlt_sdv_variant(self.variant_name)
        if self.branch2_mode is None:
            branch2_mode = hlt_sdv_branch2_mode_from_variant(variant)
        else:
            branch2_mode = normalize_hlt_sdv_branch2_mode(self.branch2_mode)
        if (self.train_split, self.val_split, self.final_test_split) != PD10_SPLIT_ORDER:
            raise ValueError(f"HLT-SDV split order must be {PD10_SPLIT_ORDER}")
        if branch2_mode == HLT_SDV_BRANCH2_HLT2 and not self.hlt2_cache_dir:
            raise ValueError("hlt2_cache_dir is required for HLT+HLT2 variants")
        if branch2_mode == HLT_SDV_BRANCH2_SAME_HLT and self.hlt2_cache_dir:
            raise ValueError("same-view HLT-SDV variant must not provide hlt2_cache_dir")
        if self.hlt2_branch_checkpoint and not bool(self.initialize_branches):
            raise ValueError("hlt2_branch_checkpoint cannot be used when initialize_branches=False")
        if self.evaluate_final_test and not bool(self.confirm_final_test):
            raise ValueError("HLT-SDV final-test evaluation requires confirm_final_test=True")
        if int(self.batch_size) <= 0 or int(self.eval_batch_size) <= 0:
            raise ValueError("batch_size and eval_batch_size must be positive")
        if int(self.epochs) <= 0:
            raise ValueError("epochs must be positive")
        if int(self.head_warmup_epochs) < 0:
            raise ValueError("head_warmup_epochs cannot be negative")
        if float(self.head_warmup_lr) <= 0.0 or float(self.branch_lr) <= 0.0 or float(self.head_lr) <= 0.0:
            raise ValueError("learning rates must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if self.model_size not in {"tiny", "base", "large"}:
            raise ValueError("model_size must be 'tiny', 'base', or 'large'")
        if int(self.fusion_hidden_dim) <= 0 or int(self.representation_dim) <= 0:
            raise ValueError("fusion_hidden_dim and representation_dim must be positive")
        if float(self.dropout) < 0.0 or float(self.dropout) >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        for split, value in (
            ("model_train", self.max_train_jets),
            ("model_val", self.max_val_jets),
            ("final_test", self.max_final_test_jets),
        ):
            if value is not None and int(value) > int(PD10_SPLIT_SIZES[split]):
                raise ValueError(f"max jets for {split} cannot exceed {PD10_SPLIT_SIZES[split]}")
        object.__setattr__(self, "variant_name", variant)
        object.__setattr__(self, "branch2_mode", branch2_mode)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "head_warmup_epochs", int(self.head_warmup_epochs))
        object.__setattr__(self, "fusion_hidden_dim", int(self.fusion_hidden_dim))
        object.__setattr__(self, "representation_dim", int(self.representation_dim))

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.output_dir) / "best_model_val.pt"

    @property
    def last_checkpoint_path(self) -> Path:
        return Path(self.output_dir) / "last.pt"

    @property
    def prediction_dir(self) -> Path:
        return Path(self.output_dir) / "predictions"

    @property
    def model_name(self) -> str:
        return self.variant_name

    @property
    def hlt2_strength(self) -> float | None:
        return hlt_sdv_strength_from_variant(self.variant_name)


@dataclass(frozen=True)
class HLTSDVEvalConfig:
    """Evaluation/prediction-cache config for one selected HLT-SDV checkpoint."""

    checkpoint: str
    output_dir: str
    hlt_cache_dir: str
    variant_name: str
    hlt2_cache_dir: str | None = None
    branch2_mode: str | None = None
    split: str = "final_test"
    batch_size: int = HLT_SDV_DEFAULT_EVAL_BATCH_SIZE
    num_workers: int = 0
    device: str = "auto"
    max_jets: int | None = None
    max_batches: int | None = None
    verify_hlt_hash: bool = True
    verify_hlt2_hash: bool = True
    confirm_final_test: bool = False
    overwrite: bool = False
    validation_thresholds_by_class: Mapping[str, Any] | None = None
    validation_binary_thresholds: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        variant = normalize_hlt_sdv_variant(self.variant_name)
        if self.branch2_mode is None:
            branch2_mode = hlt_sdv_branch2_mode_from_variant(variant)
        else:
            branch2_mode = normalize_hlt_sdv_branch2_mode(self.branch2_mode)
        if self.split not in PD10_SPLIT_ORDER:
            raise ValueError(f"unknown HLT-SDV eval split {self.split!r}")
        if self.split == "final_test" and not bool(self.confirm_final_test):
            raise ValueError("HLT-SDV final-test evaluation requires confirm_final_test=True")
        if branch2_mode == HLT_SDV_BRANCH2_HLT2 and not self.hlt2_cache_dir:
            raise ValueError("hlt2_cache_dir is required for HLT+HLT2 evaluation")
        if branch2_mode == HLT_SDV_BRANCH2_SAME_HLT and self.hlt2_cache_dir:
            raise ValueError("same-view evaluation must not provide hlt2_cache_dir")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        object.__setattr__(self, "variant_name", variant)
        object.__setattr__(self, "branch2_mode", branch2_mode)

    @property
    def model_name(self) -> str:
        return self.variant_name

    @property
    def prediction_dir(self) -> Path:
        return Path(self.output_dir) / "predictions"


def _max_jets_for_split(config: HLTSDVTrainConfig, split: str) -> int | None:
    if split == config.train_split:
        return config.max_train_jets
    if split == config.val_split:
        return config.max_val_jets
    if split == config.final_test_split:
        return config.max_final_test_jets
    return None


def load_hlt_sdv_dataset_for_train_config(config: HLTSDVTrainConfig, split: str) -> HLTSelfDualViewDataset:
    return load_hlt_sdv_dataset(
        config.hlt_cache_dir,
        split,
        hlt2_cache_dir=config.hlt2_cache_dir,
        branch2_mode=str(config.branch2_mode),
        max_jets=_max_jets_for_split(config, split),
        verify_hlt_hash=bool(config.verify_hlt_hash),
        verify_hlt2_hash=bool(config.verify_hlt2_hash),
    )


def load_hlt_sdv_dataset_for_eval_config(config: HLTSDVEvalConfig) -> HLTSelfDualViewDataset:
    return load_hlt_sdv_dataset(
        config.hlt_cache_dir,
        config.split,
        hlt2_cache_dir=config.hlt2_cache_dir,
        branch2_mode=str(config.branch2_mode),
        max_jets=config.max_jets,
        verify_hlt_hash=bool(config.verify_hlt_hash),
        verify_hlt2_hash=bool(config.verify_hlt2_hash),
    )


def _optimizer_for_stage(model, config: HLTSDVTrainConfig, *, stage: str):
    torch = require_torch()
    if stage == "head_warmup":
        model.set_branches_trainable(False)
        return torch.optim.AdamW(
            [parameter for parameter in model.head_parameters() if parameter.requires_grad],
            lr=float(config.head_warmup_lr),
            weight_decay=float(config.weight_decay),
        )
    model.set_branches_trainable(True)
    return torch.optim.AdamW(
        [
            {"params": list(model.branch_parameters()), "lr": float(config.branch_lr)},
            {"params": list(model.head_parameters()), "lr": float(config.head_lr)},
        ],
        weight_decay=float(config.weight_decay),
    )


def _amp_autocast_context(torch, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _require_finite_metrics(metrics: Mapping[str, Any], *, context: str) -> None:
    for key in ("loss", "cross_entropy", "accuracy"):
        if key not in metrics:
            continue
        value = metrics.get(key)
        if value is None:
            continue
        try:
            finite = np.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not bool(finite):
            raise FloatingPointError(f"{context} produced non-finite metric {key}={value!r}")


def _require_finite_logits(logits: np.ndarray, *, context: str) -> None:
    finite = np.isfinite(logits)
    if bool(finite.all()):
        return
    bad_rows = np.where(~np.all(finite, axis=1))[0]
    first_bad = int(bad_rows[0]) if int(bad_rows.shape[0]) else -1
    finite_logits = logits[finite]
    if finite_logits.size:
        finite_min = float(np.min(finite_logits))
        finite_max = float(np.max(finite_logits))
    else:
        finite_min = None
        finite_max = None
    raise FloatingPointError(
        f"{context} produced non-finite logits: "
        f"bad_rows={int(bad_rows.shape[0])}/{int(logits.shape[0])}, "
        f"first_bad_row={first_bad}, finite_min={finite_min}, finite_max={finite_max}"
    )


def run_hlt_sdv_epoch(
    model,
    loader,
    *,
    device,
    optimizer=None,
    scaler=None,
    amp: bool = False,
    grad_clip_norm: float = 0.0,
    max_batches: int | None = None,
) -> dict[str, Any]:
    torch = require_torch()
    criterion = torch.nn.CrossEntropyLoss()
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    context = nullcontext() if is_train else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_hlt_sdv_batch_to_device(batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            autocast_enabled = bool(amp and device.type == "cuda")
            with _amp_autocast_context(torch, autocast_enabled):
                logits = model(batch["hlt_inputs"], batch["hlt2_inputs"])
                loss = criterion(logits, batch["labels"])
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError(
                    f"HLT-SDV {'train' if is_train else 'eval'} batch {batch_index} produced non-finite logits"
                )
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError(
                    f"HLT-SDV {'train' if is_train else 'eval'} batch {batch_index} produced non-finite loss"
                )
            if is_train:
                if scaler is not None and autocast_enabled:
                    scaler.scale(loss).backward()
                    if float(grad_clip_norm) > 0.0:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                        if not bool(torch.isfinite(grad_norm).all()):
                            raise FloatingPointError(
                                f"HLT-SDV train batch {batch_index} produced non-finite gradient norm"
                            )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if float(grad_clip_norm) > 0.0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                        if not bool(torch.isfinite(grad_norm).all()):
                            raise FloatingPointError(
                                f"HLT-SDV train batch {batch_index} produced non-finite gradient norm"
                            )
                    optimizer.step()
            labels = batch["labels"]
            batch_size = int(labels.numel())
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_correct += int((logits.detach().argmax(dim=1) == labels).sum().detach().cpu().item())
            total_seen += batch_size
    if total_seen == 0:
        return {"loss": float("nan"), "cross_entropy": float("nan"), "accuracy": 0.0, "n_jets": 0}
    ce = total_loss / float(total_seen)
    metrics = {
        "loss": float(ce),
        "cross_entropy": float(ce),
        "accuracy": total_correct / float(total_seen),
        "n_jets": int(total_seen),
    }
    _require_finite_metrics(metrics, context="HLT-SDV epoch")
    return metrics


def _checkpoint_model(model):
    return getattr(model, "_orig_mod", model)


def _checkpoint_payload(
    model,
    optimizer,
    *,
    epoch: int,
    config: HLTSDVTrainConfig,
    metrics: Mapping[str, Any],
    branch_initialization: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_model = _checkpoint_model(model)
    return {
        "contract": HLT_SDV_TRAINING_CONTRACT,
        "experiment_name": HLT_SDV_EXPERIMENT_NAME,
        "experiment_step": HLT_SDV_STEP5_EXPERIMENT_STEP,
        "variant_name": config.variant_name,
        "model_name": config.model_name,
        "branch2_mode": config.branch2_mode,
        "hlt2_strength": config.hlt2_strength,
        "epoch": int(epoch),
        "model_state_dict": checkpoint_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": asdict(config),
        "metrics": dict(metrics),
        "model_config": getattr(checkpoint_model, "config", {}),
        "branch_initialization": dict(branch_initialization),
        "label_names": list(LABEL_NAMES),
        "pf_feature_names": list(PF_FEATURE_NAMES),
        "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
        "student_deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "requires_offline_inputs": False,
        "requires_teacher_features": False,
        "requires_deterministic_hlt2_transform": True,
        "returns_offline_particles": False,
        "inference_export_requires_teacher_features": False,
        "selection_metric": "model_val_cross_entropy",
        "selection_tie_breaker": "model_val_accuracy",
        "no_final_test_used_for_selection": True,
    }


def load_hlt_sdv_model_from_checkpoint(checkpoint: str | Path, *, device):
    torch = require_torch()
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(checkpoint, map_location=device)
    model_config = dict(payload.get("model_config") or {})
    model = HLTSelfDualViewFusionModel(
        num_classes=int(model_config.get("num_classes", PD10_NUM_CLASSES)),
        model_size=str(model_config.get("model_size", HLT_SDV_DEFAULT_MODEL_SIZE)),
        branch_config=model_config.get("branch_config"),
        fusion_hidden_dim=int(model_config.get("fusion_hidden_dim", HLT_SDV_DEFAULT_FUSION_HIDDEN_DIM)),
        representation_dim=int(model_config.get("representation_dim", PD10_REPRESENTATION_DIM)),
        dropout=float(model_config.get("dropout", HLT_SDV_DEFAULT_DROPOUT)),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, payload


def collect_hlt_sdv_outputs(
    model,
    dataset: HLTSelfDualViewDataset,
    *,
    batch_size: int,
    num_workers: int,
    device,
    seed: int,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[Any], dict[str, Any]]:
    torch = require_torch()
    loader = make_hlt_sdv_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=int(seed),
    )
    logits_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    jet_ids: list[Any] = []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_hlt_sdv_batch_to_device(batch, device)
            logits = model(batch["hlt_inputs"], batch["hlt2_inputs"])
            logits_chunks.append(logits.detach().cpu().float().numpy())
            labels_chunks.append(batch["labels"].detach().cpu().long().numpy())
            jet_ids.extend(batch["jet_ids"])
    if not logits_chunks:
        raise ValueError("No HLT-SDV prediction batches were produced")
    logits_np = np.concatenate(logits_chunks, axis=0).astype(np.float32)
    _require_finite_logits(logits_np, context=f"HLT-SDV:{dataset.split}:prediction")
    labels_np = np.concatenate(labels_chunks, axis=0).astype(np.int64)
    metrics = pd10_prediction_metrics_from_logits(logits_np, labels_np)
    return logits_np, labels_np, jet_ids, metrics


def build_hlt_sdv_prediction_block(
    *,
    config: HLTSDVTrainConfig | HLTSDVEvalConfig,
    split: str,
    dataset: HLTSelfDualViewDataset,
    logits: np.ndarray,
    labels: np.ndarray,
    jet_ids: list[Any],
    checkpoint: str | Path,
    checkpoint_payload: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> PredictionBlock:
    metadata = {
        "contract": HLT_SDV_PREDICTION_CONTRACT,
        "experiment_name": HLT_SDV_EXPERIMENT_NAME,
        "experiment_step": HLT_SDV_STEP5_EXPERIMENT_STEP,
        "variant_name": config.variant_name,
        "model_name": config.model_name,
        "architecture": HLT_SDV_MODEL_ARCHITECTURE,
        "split": split,
        "branch2_mode": config.branch2_mode,
        "hlt2_strength": hlt_sdv_strength_from_variant(config.variant_name),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": None if checkpoint_payload is None else checkpoint_payload.get("epoch"),
        "checkpoint_experiment_step": None
        if checkpoint_payload is None
        else checkpoint_payload.get("experiment_step"),
        "dataset": dataset.to_metadata(),
        "hlt_cache_dir": config.hlt_cache_dir,
        "hlt2_cache_dir": config.hlt2_cache_dir,
        "hlt_jet_identity_hash": jet_identity_hash(dataset.jet_ids),
        "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
        "student_deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "requires_offline_inputs": False,
        "requires_teacher_features": False,
        "requires_deterministic_hlt2_transform": True,
        "returns_offline_particles": False,
        "inference_export_requires_teacher_features": False,
        "no_final_test_used_for_selection": True,
    }
    if metrics is not None:
        metadata["rich_metrics"] = dict(metrics)
    return PredictionBlock(
        model_name=config.model_name,
        split=split,
        logits=np.asarray(logits, dtype=np.float32),
        probs=softmax_np(logits),
        labels=np.asarray(labels, dtype=np.int64),
        jet_ids=list(jet_ids),
        metadata=metadata,
    )


def collect_and_save_hlt_sdv_predictions(
    model,
    dataset: HLTSelfDualViewDataset,
    *,
    config: HLTSDVTrainConfig | HLTSDVEvalConfig,
    split: str,
    checkpoint: str | Path,
    checkpoint_payload: Mapping[str, Any] | None,
    batch_size: int,
    num_workers: int,
    device,
    seed: int,
    max_batches: int | None = None,
    overwrite: bool = False,
    validation_thresholds_by_class: Mapping[str, Any] | None = None,
    validation_binary_thresholds: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    logits, labels, jet_ids, metrics = collect_hlt_sdv_outputs(
        model,
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        seed=seed,
        max_batches=max_batches,
    )
    if validation_thresholds_by_class is not None or validation_binary_thresholds is not None:
        metrics = pd10_prediction_metrics_from_logits(
            logits,
            labels,
            validation_thresholds_by_class=validation_thresholds_by_class,
            validation_binary_thresholds=validation_binary_thresholds,
        )
    block = build_hlt_sdv_prediction_block(
        config=config,
        split=split,
        dataset=dataset,
        logits=logits,
        labels=labels,
        jet_ids=jet_ids,
        checkpoint=checkpoint,
        checkpoint_payload=checkpoint_payload,
        metrics=metrics,
    )
    metadata = save_prediction_block(block, config.prediction_dir, overwrite=overwrite)
    return metrics, metadata


def train_hlt_sdv_model(
    config: HLTSDVTrainConfig,
    *,
    model=None,
    train_dataset: HLTSelfDualViewDataset | None = None,
    val_dataset: HLTSelfDualViewDataset | None = None,
) -> dict[str, Any]:
    torch = require_torch()
    set_training_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    if config.checkpoint_path.exists() and not config.overwrite:
        raise FileExistsError(f"HLT-SDV checkpoint already exists: {config.checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = train_dataset or load_hlt_sdv_dataset_for_train_config(config, config.train_split)
    val_dataset = val_dataset or load_hlt_sdv_dataset_for_train_config(config, config.val_split)
    train_loader = make_hlt_sdv_data_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=int(config.seed) + 101,
    )
    val_loader = make_hlt_sdv_data_loader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=int(config.seed) + 202,
    )

    provided_model = model is not None
    if model is None:
        model = build_hlt_sdv_fusion_model(
            num_classes=PD10_NUM_CLASSES,
            model_size=config.model_size,
            fusion_hidden_dim=config.fusion_hidden_dim,
            representation_dim=config.representation_dim,
            dropout=config.dropout,
        )
    model = model.to(device)
    branch_initialization: dict[str, Any] = {"enabled": bool(config.initialize_branches)}
    if config.initialize_branches:
        branch_initialization.update(
            initialize_hlt_sdv_branches_from_checkpoints(
                model,
                hlt_checkpoint=config.hlt_teacher_checkpoint,
                hlt2_checkpoint=config.hlt2_branch_checkpoint,
                device=device,
            )
        )
    checkpoint_model = model
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
        checkpoint_model = getattr(model, "_orig_mod", checkpoint_model)

    scaler = None
    if bool(config.amp and device.type == "cuda"):
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    run_metadata = {
        "ok": True,
        "contract": HLT_SDV_TRAINING_CONTRACT,
        "model_contract": HLT_SDV_MODEL_CONTRACT,
        "experiment_name": HLT_SDV_EXPERIMENT_NAME,
        "experiment_step": HLT_SDV_STEP5_EXPERIMENT_STEP,
        "variant_name": config.variant_name,
        "model_name": config.model_name,
        "architecture": HLT_SDV_MODEL_ARCHITECTURE,
        "branch2_mode": config.branch2_mode,
        "hlt2_strength": config.hlt2_strength,
        "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
        "student_deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "requires_offline_inputs": False,
        "requires_teacher_features": False,
        "requires_deterministic_hlt2_transform": True,
        "returns_offline_particles": False,
        "inference_export_requires_teacher_features": False,
        "train_dataset": train_dataset.to_metadata(),
        "model_val_dataset": val_dataset.to_metadata(),
        "branch_initialization": branch_initialization,
        "selection_metric": "model_val_cross_entropy",
        "selection_tie_breaker": "model_val_accuracy",
        "no_final_test_used_for_selection": True,
        "config": asdict(config),
    }
    save_json(output_dir / "config.json", run_metadata)

    curves: list[dict[str, Any]] = []
    best_val_ce = float("inf")
    best_val_accuracy = -1.0
    best_epoch = -1
    best_row: dict[str, Any] | None = None
    epochs_without_improvement = 0
    current_stage: str | None = None
    optimizer = None
    for epoch in range(1, int(config.epochs) + 1):
        stage = "head_warmup" if epoch <= int(config.head_warmup_epochs) else "full_finetune"
        if optimizer is None or stage != current_stage:
            optimizer = _optimizer_for_stage(model, config, stage=stage)
            current_stage = stage
        train_metrics = run_hlt_sdv_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp=config.amp,
            grad_clip_norm=config.grad_clip_norm,
            max_batches=config.max_train_batches,
        )
        _require_finite_metrics(train_metrics, context=f"{config.variant_name}:epoch{epoch}:train")
        val_metrics = run_hlt_sdv_epoch(
            model,
            val_loader,
            device=device,
            amp=False,
            max_batches=config.max_val_batches,
        )
        _require_finite_metrics(val_metrics, context=f"{config.variant_name}:epoch{epoch}:model_val")
        row = {"epoch": int(epoch), "stage": stage, "train": train_metrics, "model_val": val_metrics}
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves})
        torch.save(
            _checkpoint_payload(
                checkpoint_model,
                optimizer,
                epoch=epoch,
                config=config,
                metrics=row,
                branch_initialization=branch_initialization,
            ),
            config.last_checkpoint_path,
        )
        val_ce = float(val_metrics["cross_entropy"])
        val_accuracy = float(val_metrics["accuracy"])
        improved = val_ce < best_val_ce or (
            np.isclose(val_ce, best_val_ce) and val_accuracy > best_val_accuracy
        )
        if improved:
            best_val_ce = val_ce
            best_val_accuracy = val_accuracy
            best_epoch = int(epoch)
            best_row = row
            epochs_without_improvement = 0
            torch.save(
                _checkpoint_payload(
                    checkpoint_model,
                    optimizer,
                    epoch=epoch,
                    config=config,
                    metrics=row,
                    branch_initialization=branch_initialization,
                ),
                config.checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if int(config.early_stop_patience) >= 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    selected_model = model if provided_model else None
    selected_payload: Mapping[str, Any] | None = None
    if selected_model is None:
        selected_model, selected_payload = load_hlt_sdv_model_from_checkpoint(config.checkpoint_path, device=device)
    else:
        selected_payload = _checkpoint_payload(
            checkpoint_model,
            None,
            epoch=best_epoch,
            config=config,
            metrics=best_row or {},
            branch_initialization=branch_initialization,
        )
    model_val_prediction_metrics = None
    model_val_prediction_metadata = None
    final_test_metrics = None
    final_test_prediction_metadata = None
    validation_thresholds = None
    validation_binary_thresholds = None
    if config.evaluate_model_val_predictions:
        model_val_prediction_metrics, model_val_prediction_metadata = collect_and_save_hlt_sdv_predictions(
            selected_model,
            val_dataset,
            config=config,
            split=config.val_split,
            checkpoint=config.checkpoint_path,
            checkpoint_payload=selected_payload,
            batch_size=config.eval_batch_size,
            num_workers=config.num_workers,
            device=device,
            seed=int(config.seed) + 303,
            max_batches=config.max_val_batches,
            overwrite=config.overwrite,
        )
        validation_thresholds = model_val_prediction_metrics.get("score_thresholds_by_class")
        validation_binary_thresholds = model_val_prediction_metrics.get("binary_score_thresholds")
    if config.evaluate_final_test:
        final_test_dataset = load_hlt_sdv_dataset_for_train_config(config, config.final_test_split)
        final_test_metrics, final_test_prediction_metadata = collect_and_save_hlt_sdv_predictions(
            selected_model,
            final_test_dataset,
            config=config,
            split=config.final_test_split,
            checkpoint=config.checkpoint_path,
            checkpoint_payload=selected_payload,
            batch_size=config.eval_batch_size,
            num_workers=config.num_workers,
            device=device,
            seed=int(config.seed) + 404,
            max_batches=config.max_final_test_batches,
            overwrite=config.overwrite,
            validation_thresholds_by_class=validation_thresholds,
            validation_binary_thresholds=validation_binary_thresholds,
        )

    report = {
        **run_metadata,
        "best_epoch": int(best_epoch),
        "best_model_val_cross_entropy": float(best_val_ce),
        "best_model_val_accuracy": float(best_val_accuracy),
        "best_model_val_training_metrics": None if best_row is None else dict(best_row["model_val"]),
        "epochs_completed": int(len(curves)),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(config.checkpoint_path),
        "checkpoint_sha256": sha256_file(config.checkpoint_path),
        "last_checkpoint": str(config.last_checkpoint_path),
        "model_val_prediction_metrics": model_val_prediction_metrics,
        "model_val_prediction_metadata": model_val_prediction_metadata,
        "final_test_metrics": final_test_metrics,
        "final_test_prediction_metadata": final_test_prediction_metadata,
    }
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    if final_test_metrics is not None:
        save_json(output_dir / "final_test_report.json", {**report, "metrics": final_test_metrics})
    return report


def evaluate_hlt_sdv_model(
    config: HLTSDVEvalConfig,
    *,
    model=None,
) -> dict[str, Any]:
    torch = require_torch()
    device = resolve_device(config.device)
    if model is None:
        model, payload = load_hlt_sdv_model_from_checkpoint(config.checkpoint, device=device)
    else:
        model = model.to(device)
        payload = None
    dataset = load_hlt_sdv_dataset_for_eval_config(config)
    metrics, prediction_metadata = collect_and_save_hlt_sdv_predictions(
        model,
        dataset,
        config=config,
        split=config.split,
        checkpoint=config.checkpoint,
        checkpoint_payload=payload,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        seed=HLT_SDV_DEFAULT_SEED + 505,
        max_batches=config.max_batches,
        overwrite=config.overwrite,
        validation_thresholds_by_class=config.validation_thresholds_by_class,
        validation_binary_thresholds=config.validation_binary_thresholds,
    )
    report = {
        "ok": True,
        "contract": HLT_SDV_PREDICTION_CONTRACT,
        "experiment_name": HLT_SDV_EXPERIMENT_NAME,
        "experiment_step": HLT_SDV_STEP5_EXPERIMENT_STEP,
        "variant_name": config.variant_name,
        "model_name": config.model_name,
        "architecture": HLT_SDV_MODEL_ARCHITECTURE,
        "split": config.split,
        "branch2_mode": config.branch2_mode,
        "hlt2_strength": hlt_sdv_strength_from_variant(config.variant_name),
        "checkpoint": str(config.checkpoint),
        "checkpoint_sha256": sha256_file(config.checkpoint),
        "prediction_metadata": prediction_metadata,
        "metrics": metrics,
        "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
        "student_deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "requires_offline_inputs": False,
        "requires_teacher_features": False,
        "requires_deterministic_hlt2_transform": True,
        "returns_offline_particles": False,
        "no_final_test_used_for_selection": True,
    }
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    suffix = "final_test" if config.split == "final_test" else config.split
    save_json(output / f"{suffix}_report.json", report)
    return report


def default_train_config_for_pd10_root(
    pd10_root: str | Path,
    *,
    variant_name: str,
    output_dir: str | Path | None = None,
    hlt2_cache_dir: str | Path | None = None,
    confirm_final_test: bool = False,
    overwrite: bool = False,
) -> HLTSDVTrainConfig:
    pd10_root = Path(pd10_root)
    layout = default_hlt_sdv_experiment_layout(
        output_root=pd10_root.parent,
        pd10_experiment_name=pd10_root.name,
    )
    variant = normalize_hlt_sdv_variant(variant_name)
    branch2_mode = hlt_sdv_branch2_mode_from_variant(variant)
    strength = hlt_sdv_strength_from_variant(variant)
    resolved_hlt2_cache_dir = None
    if branch2_mode == HLT_SDV_BRANCH2_HLT2:
        resolved_hlt2_cache_dir = Path(hlt2_cache_dir) if hlt2_cache_dir else default_hlt2_cache_dir(
            pd10_root,
            float(strength),
        )
    return HLTSDVTrainConfig(
        output_dir=str(Path(output_dir) if output_dir else layout.variant_dir(variant)),
        hlt_cache_dir=str(pd10_root / "hlt_cache"),
        hlt2_cache_dir=None if resolved_hlt2_cache_dir is None else str(resolved_hlt2_cache_dir),
        hlt_teacher_checkpoint=str(layout.hlt_teacher_checkpoint),
        variant_name=variant,
        branch2_mode=branch2_mode,
        confirm_final_test=confirm_final_test,
        overwrite=overwrite,
    )


__all__ = [
    "HLTSDVEvalConfig",
    "HLTSDVTrainConfig",
    "HLT_SDV_DEFAULT_BATCH_SIZE",
    "HLT_SDV_DEFAULT_BRANCH_LR",
    "HLT_SDV_DEFAULT_EARLY_STOP_PATIENCE",
    "HLT_SDV_DEFAULT_EPOCHS",
    "HLT_SDV_DEFAULT_EVAL_BATCH_SIZE",
    "HLT_SDV_DEFAULT_HEAD_LR",
    "HLT_SDV_DEFAULT_HEAD_WARMUP_EPOCHS",
    "HLT_SDV_DEFAULT_HEAD_WARMUP_LR",
    "HLT_SDV_DEFAULT_SEED",
    "HLT_SDV_DEFAULT_WEIGHT_DECAY",
    "HLT_SDV_PREDICTION_CONTRACT",
    "HLT_SDV_STEP5_EXPERIMENT_STEP",
    "HLT_SDV_TRAINING_CONTRACT",
    "build_hlt_sdv_prediction_block",
    "collect_and_save_hlt_sdv_predictions",
    "collect_hlt_sdv_outputs",
    "default_train_config_for_pd10_root",
    "evaluate_hlt_sdv_model",
    "load_hlt_sdv_dataset_for_eval_config",
    "load_hlt_sdv_dataset_for_train_config",
    "load_hlt_sdv_model_from_checkpoint",
    "run_hlt_sdv_epoch",
    "train_hlt_sdv_model",
]
