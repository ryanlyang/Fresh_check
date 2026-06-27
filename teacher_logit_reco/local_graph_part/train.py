"""Step 6 training loop for local-graph HLT ParT comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
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

from .model import (
    LOCAL_GRAPH_COMPARISON_VARIANTS,
    LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE,
    LOCAL_GRAPH_MODEL_VARIANT_NO_LOCAL,
    LocalGraphAugmentedParticleTransformerClassifier,
    HLTPartBaselineRawTokenClassifier,
    build_local_graph_comparison_classifier,
    normalize_local_graph_comparison_variant,
)
from .protocol import (
    LOCAL_GRAPH_PART_BINARY_LABEL_FILTER,
    LOCAL_GRAPH_PART_CONTRACT,
    LOCAL_GRAPH_PART_PRIMARY_METRIC,
    LOCAL_GRAPH_PART_PROTOCOL_STEP,
    LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES,
    default_local_graph_part_protocol,
    local_graph_part_protocol_manifest,
)


LOCAL_GRAPH_PART_TRAIN_STEP = "local_graph_part_step6_train_baseline_and_adapters"
LOCAL_GRAPH_WARM_START_STEP = "local_graph_part_step7_warm_start"
LOCAL_GRAPH_SELECTION_METRICS = (
    "accuracy",
    "loss",
    "macro_per_class_accuracy",
    "auc",
    "fpr_at_signal_eff_0p30",
    "fpr_at_signal_eff_0p50",
    "background_rejection_at_signal_eff_0p30",
    "background_rejection_at_signal_eff_0p50",
)
LOCAL_GRAPH_LOWER_IS_BETTER_SELECTION_METRICS = {"loss", "fpr_at_signal_eff_0p30", "fpr_at_signal_eff_0p50"}
LOCAL_GRAPH_ALLOWED_STEP6_VARIANTS = LOCAL_GRAPH_COMPARISON_VARIANTS + (LOCAL_GRAPH_MODEL_VARIANT_NO_LOCAL,)


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


def _lookup_selection_metric(metrics: Mapping[str, Any], metric_name: str) -> float:
    if metric_name in metrics:
        try:
            return float(metrics[metric_name])
        except (TypeError, ValueError):
            return float("nan")
    binary = metrics.get("binary_metrics")
    if isinstance(binary, Mapping) and metric_name in binary:
        try:
            return float(binary[metric_name])
        except (TypeError, ValueError):
            return float("nan")
    raise KeyError(f"validation metrics do not contain selection metric {metric_name!r}")


def _selection_metric_requires_predictions(metric_name: str) -> bool:
    return metric_name not in {"accuracy", "loss"}


def _selection_score(metrics: Mapping[str, Any], metric_name: str) -> tuple[float, float]:
    value = _lookup_selection_metric(metrics, metric_name)
    if np.isnan(value):
        return float("-inf"), value
    if metric_name in LOCAL_GRAPH_LOWER_IS_BETTER_SELECTION_METRICS:
        return -float(value), float(value)
    return float(value), float(value)


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
                    output[key] = float(value.detach().cpu().item())
            except Exception:
                continue
        elif isinstance(value, (int, float)):
            numeric = float(value)
            if np.isfinite(numeric):
                output[key] = numeric
    return output


def _write_summary_csv(path: Path, metrics_by_split: Mapping[str, Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for split, metrics in metrics_by_split.items():
        binary = metrics.get("binary_metrics", {})
        rows.append(
            {
                "split": split,
                "loss": metrics.get("loss"),
                "accuracy": metrics.get("accuracy"),
                "macro_per_class_accuracy": metrics.get("macro_per_class_accuracy"),
                "auc": binary.get("auc") if isinstance(binary, Mapping) else None,
                "fpr_at_signal_eff_0p30": binary.get("fpr_at_signal_eff_0p30") if isinstance(binary, Mapping) else None,
                "fpr_at_signal_eff_0p50": binary.get("fpr_at_signal_eff_0p50") if isinstance(binary, Mapping) else None,
                "background_rejection_at_signal_eff_0p30": (
                    binary.get("background_rejection_at_signal_eff_0p30") if isinstance(binary, Mapping) else None
                ),
                "background_rejection_at_signal_eff_0p50": (
                    binary.get("background_rejection_at_signal_eff_0p50") if isinstance(binary, Mapping) else None
                ),
                "n_jets": metrics.get("n_jets"),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "loss",
                "accuracy",
                "macro_per_class_accuracy",
                "auc",
                "fpr_at_signal_eff_0p30",
                "fpr_at_signal_eff_0p50",
                "background_rejection_at_signal_eff_0p30",
                "background_rejection_at_signal_eff_0p50",
                "n_jets",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_best_metrics_csv(path: Path, report: Mapping[str, Any]) -> None:
    row = {
        "variant": report.get("variant"),
        "best_epoch": report.get("best_epoch"),
        "selection_metric": report.get("selection_metric"),
        "best_model_selection_metric_value": report.get("best_model_selection_metric_value"),
        "best_model_selection_score": report.get("best_model_selection_score"),
        "best_model_val_accuracy": report.get("best_model_val_accuracy"),
        "best_model_val_loss": report.get("best_model_val_loss"),
    }
    for split_key, metrics_key in (
        ("model_val", "best_model_val_metrics"),
        ("stack_val", "stack_val_metrics"),
        ("final_test", "final_test_metrics"),
    ):
        metrics = report.get(metrics_key)
        if not isinstance(metrics, Mapping):
            continue
        row[f"{split_key}_accuracy"] = metrics.get("accuracy")
        row[f"{split_key}_loss"] = metrics.get("loss")
        binary = metrics.get("binary_metrics")
        if isinstance(binary, Mapping):
            row[f"{split_key}_auc"] = binary.get("auc")
            row[f"{split_key}_fpr_at_signal_eff_0p30"] = binary.get("fpr_at_signal_eff_0p30")
            row[f"{split_key}_fpr_at_signal_eff_0p50"] = binary.get("fpr_at_signal_eff_0p50")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _strip_checkpoint_key_prefixes(key: str) -> str:
    clean = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "_orig_mod."):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
                changed = True
    return clean


def _extract_checkpoint_state_dict(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping) and isinstance(payload.get("model_state_dict"), Mapping):
        return payload["model_state_dict"]
    if isinstance(payload, Mapping) and all(isinstance(key, str) for key in payload.keys()):
        return payload
    raise ValueError("checkpoint does not contain a recognizable model state dict")


def _warm_start_candidate_keys(source_key: str) -> tuple[str, ...]:
    clean = _strip_checkpoint_key_prefixes(source_key)
    candidates: list[str] = []
    if clean.startswith("part_model."):
        candidates.append(clean)
    else:
        candidates.append(f"part_model.{clean}")
    if clean.startswith("model."):
        candidates.append(f"part_model.{clean[len('model.') :]}")
    if clean.startswith("mod."):
        candidates.append(f"part_model.{clean}")
    output: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = _strip_checkpoint_key_prefixes(candidate)
        if candidate not in seen:
            seen.add(candidate)
            output.append(candidate)
    return tuple(output)


def warm_start_local_graph_part_model(
    model,
    checkpoint_path: str | Path,
    *,
    device=None,
    require: bool = False,
) -> dict[str, Any]:
    """Load HLT ParT baseline weights into the local-adapter model backbone.

    Supports both Step 6 raw-token baseline checkpoints with keys like
    ``part_model.mod.*`` and older direct HLT baseline checkpoints with keys
    like ``mod.*``.
    """

    torch = require_torch()
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location=device or "cpu")
    source_state = _extract_checkpoint_state_dict(payload)
    target_state = model.state_dict()
    updated_state = dict(target_state)
    loaded: list[dict[str, str]] = []
    skipped_shape: list[dict[str, Any]] = []
    unmatched: list[str] = []
    non_tensor: list[str] = []

    for source_key, source_value in source_state.items():
        if not hasattr(source_value, "shape"):
            non_tensor.append(str(source_key))
            continue
        matched_key = None
        for candidate in _warm_start_candidate_keys(str(source_key)):
            if not candidate.startswith("part_model."):
                continue
            if candidate not in target_state:
                continue
            target_value = target_state[candidate]
            if tuple(target_value.shape) != tuple(source_value.shape):
                skipped_shape.append(
                    {
                        "source_key": str(source_key),
                        "target_key": candidate,
                        "source_shape": list(source_value.shape),
                        "target_shape": list(target_value.shape),
                    }
                )
                continue
            updated_state[candidate] = source_value.detach().to(device=target_value.device, dtype=target_value.dtype)
            matched_key = candidate
            loaded.append({"source_key": str(source_key), "target_key": candidate})
            break
        if matched_key is None:
            unmatched.append(str(source_key))

    model.load_state_dict(updated_state, strict=True)
    report = {
        "step": LOCAL_GRAPH_WARM_START_STEP,
        "checkpoint": str(path),
        "checkpoint_exists": path.exists(),
        "loaded_key_count": len(loaded),
        "loaded_keys": loaded,
        "unmatched_key_count": len(unmatched),
        "unmatched_source_keys_sample": unmatched[:50],
        "shape_mismatch_count": len(skipped_shape),
        "shape_mismatches": skipped_shape[:50],
        "non_tensor_key_count": len(non_tensor),
        "non_tensor_keys_sample": non_tensor[:50],
        "required": bool(require),
    }
    if len(loaded) == 0 and bool(require):
        raise ValueError(f"warm start loaded zero keys from {path}")
    return report


def _set_part_model_trainable(model, trainable: bool) -> dict[str, Any]:
    part_model = getattr(model, "part_model", None)
    changed = 0
    total = 0
    if part_model is not None and hasattr(part_model, "parameters"):
        for param in part_model.parameters():
            total += 1
            if bool(param.requires_grad) != bool(trainable):
                changed += 1
            param.requires_grad_(bool(trainable))
    return {
        "part_model_parameter_tensors": int(total),
        "changed_parameter_tensors": int(changed),
        "part_model_trainable": bool(trainable),
    }


@dataclass
class LocalGraphTaggerTrainConfig:
    """Configuration for one Step 6 baseline/local-adapter training run."""

    output_dir: str
    hlt_cache_dir: str
    variant: str = "local_point_attention_adapter"
    train_split: str = "model_train"
    val_split: str = "model_val"
    stack_val_split: str = "stack_val"
    final_test_split: str = "final_test"
    confirm_split_settings: bool = False
    confirm_final_test: bool = False
    seed: int = 3107
    batch_size: int = 64
    eval_batch_size: int = 128
    epochs: int = 45
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 6
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_stack_val_batches: int | None = None
    max_final_test_batches: int | None = None
    max_train_jets: int | None = None
    max_val_jets: int | None = None
    max_stack_val_jets: int | None = None
    max_final_test_jets: int | None = None
    selection_metric: str = LOCAL_GRAPH_PART_PRIMARY_METRIC
    compile_model: bool = False
    verify_hlt_hash: bool = True

    num_classes: int = 2
    label_names: tuple[str, ...] = LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES
    label_filter: tuple[int, ...] = LOCAL_GRAPH_PART_BINARY_LABEL_FILTER

    model_size: str = "base"
    max_constits: int = 128
    k: int = 16
    local_embed_dim: int = 128
    local_heads: int = 8
    local_hidden_dim: int | None = None
    dropout: float = 0.05
    attention_dropout: float = 0.05
    residual_gamma_init: float = 0.0
    weight_threshold: float = 0.0
    warm_start_checkpoint: str | None = None
    require_warm_start: bool = False
    freeze_part_epochs: int = 0

    def __post_init__(self) -> None:
        self.variant = normalize_local_graph_comparison_variant(self.variant)
        if self.variant not in LOCAL_GRAPH_ALLOWED_STEP6_VARIANTS:
            raise ValueError(f"variant must be one of {LOCAL_GRAPH_ALLOWED_STEP6_VARIANTS}")
        if self.train_split != "model_train" or self.val_split != "model_val":
            raise ValueError("Step 6 trains only on model_train and selects only on model_val")
        if self.stack_val_split != "stack_val" or self.final_test_split != "final_test":
            raise ValueError("Step 6 evaluates only stack_val/final_test after model_val selection")
        if not bool(self.confirm_split_settings):
            raise ValueError("Set --confirm-split-settings to acknowledge model_train/model_val-only selection")
        if str(self.selection_metric) not in LOCAL_GRAPH_SELECTION_METRICS:
            raise ValueError(f"selection_metric must be one of {LOCAL_GRAPH_SELECTION_METRICS}")
        protocol = default_local_graph_part_protocol()
        if str(self.selection_metric) != str(protocol.selection_metric):
            raise ValueError(f"Step 6 protocol selects checkpoints with {protocol.selection_metric}")
        for field_name in (
            "batch_size",
            "eval_batch_size",
            "epochs",
            "num_classes",
            "max_constits",
            "k",
            "local_embed_dim",
            "local_heads",
        ):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            setattr(self, field_name, value)
        if str(self.model_size) not in {"base", "tiny"}:
            raise ValueError("model_size must be 'base' or 'tiny'")
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
        for field_name in ("dropout", "attention_dropout"):
            value = float(getattr(self, field_name))
            if value < 0.0 or value >= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1)")
            setattr(self, field_name, value)
        if self.local_hidden_dim is not None:
            self.local_hidden_dim = _optional_positive_int(self.local_hidden_dim, field_name="local_hidden_dim")
        self.freeze_part_epochs = _optional_nonnegative_int(self.freeze_part_epochs, field_name="freeze_part_epochs") or 0
        if self.warm_start_checkpoint is not None:
            self.warm_start_checkpoint = str(self.warm_start_checkpoint)
            if not self.warm_start_checkpoint:
                self.warm_start_checkpoint = None
        self.require_warm_start = bool(self.require_warm_start)
        if self.warm_start_checkpoint and self.variant == LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE:
            raise ValueError("warm_start_checkpoint is for local adapter variants, not hlt_part_baseline")
        if int(self.freeze_part_epochs) > 0 and self.variant == LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE:
            raise ValueError("freeze_part_epochs is for local adapter warm-start variants, not hlt_part_baseline")
        self.max_train_batches = _optional_nonnegative_int(self.max_train_batches, field_name="max_train_batches")
        self.max_val_batches = _optional_nonnegative_int(self.max_val_batches, field_name="max_val_batches")
        self.max_stack_val_batches = _optional_nonnegative_int(
            self.max_stack_val_batches,
            field_name="max_stack_val_batches",
        )
        self.max_final_test_batches = _optional_nonnegative_int(
            self.max_final_test_batches,
            field_name="max_final_test_batches",
        )
        self.max_train_jets = _optional_positive_int(self.max_train_jets, field_name="max_train_jets")
        self.max_val_jets = _optional_positive_int(self.max_val_jets, field_name="max_val_jets")
        self.max_stack_val_jets = _optional_positive_int(self.max_stack_val_jets, field_name="max_stack_val_jets")
        self.max_final_test_jets = _optional_positive_int(self.max_final_test_jets, field_name="max_final_test_jets")
        self.label_names = tuple(str(name) for name in self.label_names)
        self.label_filter = tuple(int(label) for label in self.label_filter)
        if int(self.num_classes) != len(self.label_names) or int(self.num_classes) != len(self.label_filter):
            raise ValueError("num_classes, label_names, and label_filter must have matching lengths")
        if int(self.num_classes) != 2:
            raise ValueError("Step 6 local graph protocol is binary QCD/Hgg only")
        if tuple(self.label_names) != tuple(protocol.binary_label_names):
            raise ValueError("Step 6 local graph protocol is frozen to QCD/Hgg labels")
        if tuple(self.label_filter) != tuple(protocol.binary_label_filter):
            raise ValueError("Step 6 expects binary-cache labels QCD=0, Hgg=1")


def _load_local_graph_dataset(
    config: LocalGraphTaggerTrainConfig,
    split: str,
    *,
    max_jets: int | None,
) -> SubtokenHLTJetDataset:
    view = load_cached_hlt_view(config.hlt_cache_dir, split, verify_hash=bool(config.verify_hlt_hash))
    return SubtokenHLTJetDataset(
        view,
        label_filter=tuple(config.label_filter),
        label_names=tuple(config.label_names),
        max_jets=max_jets,
    )


def build_local_graph_tagger_for_config(
    config: LocalGraphTaggerTrainConfig,
    *,
    part_model: Any | None = None,
):
    """Build the true baseline or one local-adapter comparison model."""

    return build_local_graph_comparison_classifier(
        config.variant,
        num_classes=int(config.num_classes),
        model_size=str(config.model_size),
        max_constits=int(config.max_constits),
        k=int(config.k),
        local_embed_dim=int(config.local_embed_dim),
        local_heads=int(config.local_heads),
        local_hidden_dim=config.local_hidden_dim,
        dropout=float(config.dropout),
        attention_dropout=float(config.attention_dropout),
        residual_gamma_init=float(config.residual_gamma_init),
        weight_threshold=float(config.weight_threshold),
        part_model=part_model,
    )


def run_local_graph_tagger_epoch(
    model,
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
    collect_diagnostics: bool = False,
    label_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run one train/eval epoch for a Step 6 local-graph comparison model."""

    torch = require_torch()
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    collected_preds: list[np.ndarray] = []
    collected_labels: list[np.ndarray] = []
    collected_logits: list[np.ndarray] = []
    diagnostic_totals: dict[str, float] = {}
    diagnostic_weight_sum = 0.0
    autocast_enabled = bool(amp and device.type == "cuda")
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = _move_batch_to_device(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                if collect_diagnostics:
                    output = model(batch["tokens"], batch["mask"], return_outputs=True)
                    logits = output.logits
                else:
                    output = None
                    logits = model(batch["tokens"], batch["mask"])
                loss = criterion(logits, batch["labels"])

            if training:
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

            labels = batch["labels"]
            preds = logits.detach().argmax(dim=1)
            batch_size = int(labels.numel())
            total_loss += float(loss.detach().item()) * batch_size
            total_correct += int((preds == labels).sum().item())
            total_seen += batch_size
            if collect_predictions:
                collected_preds.append(preds.detach().cpu().numpy().astype(np.int64))
                collected_labels.append(labels.detach().cpu().numpy().astype(np.int64))
                collected_logits.append(logits.detach().cpu().numpy().astype(np.float32))
            if collect_diagnostics and output is not None:
                diagnostics = _flatten_scalar_diagnostics(output.diagnostics())
                for key, value in diagnostics.items():
                    diagnostic_totals[key] = diagnostic_totals.get(key, 0.0) + float(value) * batch_size
                diagnostic_weight_sum += float(batch_size)

    if total_seen == 0:
        return {"loss": float("nan"), "accuracy": 0.0, "n_jets": 0}
    metrics: dict[str, Any] = {
        "loss": total_loss / float(total_seen),
        "accuracy": total_correct / float(total_seen),
        "n_jets": int(total_seen),
    }
    if collect_predictions:
        preds_np = np.concatenate(collected_preds, axis=0) if collected_preds else np.asarray([], dtype=np.int64)
        labels_np = np.concatenate(collected_labels, axis=0) if collected_labels else np.asarray([], dtype=np.int64)
        logits_np = np.concatenate(collected_logits, axis=0) if collected_logits else None
        metrics.update(
            classification_metrics_from_predictions(
                preds=preds_np,
                labels=labels_np,
                loss_sum=total_loss,
                logits=logits_np,
                label_names=label_names,
            )
        )
    if diagnostic_weight_sum > 0.0:
        metrics["diagnostics"] = {
            key: value / diagnostic_weight_sum
            for key, value in sorted(diagnostic_totals.items())
        }
    return metrics


def local_graph_tagger_checkpoint_payload(
    model,
    optimizer,
    *,
    epoch: int,
    config: LocalGraphTaggerTrainConfig,
    metrics: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "model_config": model.to_config_dict(),
        "metrics": dict(metrics),
        "variant": str(config.variant),
        "label_names": list(config.label_names),
        "label_filter": list(config.label_filter),
        "num_classes": int(config.num_classes),
        "source": dict(source),
        "experiment_step": LOCAL_GRAPH_PART_TRAIN_STEP,
        "protocol_step": LOCAL_GRAPH_PART_PROTOCOL_STEP,
        "protocol_contract": LOCAL_GRAPH_PART_CONTRACT,
        "output_contract": model.output_contract,
    }


def load_local_graph_tagger_checkpoint(path: str | Path, *, device=None):
    torch = require_torch()
    payload = torch.load(path, map_location=device or "cpu")
    config_payload = payload.get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("checkpoint payload does not contain a train config")
    config = LocalGraphTaggerTrainConfig(**dict(config_payload))
    model = build_local_graph_tagger_for_config(config)
    model.load_state_dict(payload["model_state_dict"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, payload


def train_local_graph_tagger(
    config: LocalGraphTaggerTrainConfig,
    *,
    model: HLTPartBaselineRawTokenClassifier | LocalGraphAugmentedParticleTransformerClassifier | None = None,
    train_dataset: SubtokenHLTJetDataset | None = None,
    val_dataset: SubtokenHLTJetDataset | None = None,
    stack_val_dataset: SubtokenHLTJetDataset | None = None,
    final_test_dataset: SubtokenHLTJetDataset | None = None,
) -> dict[str, Any]:
    """Train/evaluate one Step 6 HLT ParT baseline or local-adapter variant."""

    run_start_time = time.perf_counter()
    torch = require_torch()
    set_training_seed(int(config.seed))
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = train_dataset or _load_local_graph_dataset(
        config,
        config.train_split,
        max_jets=config.max_train_jets,
    )
    val_dataset = val_dataset or _load_local_graph_dataset(
        config,
        config.val_split,
        max_jets=config.max_val_jets,
    )
    if train_dataset.tokens.shape[-1] != RAW_TOKEN_DIM or val_dataset.tokens.shape[-1] != RAW_TOKEN_DIM:
        raise ValueError(f"local graph tagger expects raw token dim {RAW_TOKEN_DIM}")
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

    checkpoint_model = model or build_local_graph_tagger_for_config(config)
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

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(checkpoint_model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))
    source = source_metadata()
    run_metadata = {
        "experiment_step": LOCAL_GRAPH_PART_TRAIN_STEP,
        "protocol": local_graph_part_protocol_manifest(),
        "output_contract": checkpoint_model.output_contract,
        "variant": str(config.variant),
        "config": asdict(config),
        "model_config": checkpoint_model.to_config_dict(),
        "source": source,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "label_names": list(config.label_names),
        "label_filter": list(config.label_filter),
        "num_classes": int(config.num_classes),
        "selection_metric": str(config.selection_metric),
        "warm_start": warm_start_report,
        "freeze_schedule": freeze_schedule,
        "leakage_rule": (
            "Step 6 consumes cached fixed-HLT tokens and labels only. Training uses model_train, "
            "checkpoint selection uses model_val, and stack_val/final_test are loaded only after "
            "model_val checkpoint selection."
        ),
        "final_test_loaded_during_training": False,
        "inference_consumes_hlt_only": True,
        "baseline_and_adapters_share_runner": True,
    }
    save_json(output_dir / "config.json", run_metadata)

    curves: list[dict[str, Any]] = []
    best_val_score = float("-inf")
    best_selection_metric_value = float("nan")
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, int(config.epochs) + 1):
        if int(config.freeze_part_epochs) > 0 and epoch == int(config.freeze_part_epochs) + 1:
            event = {"epoch": int(epoch), "event": "unfreeze_part_model"}
            event.update(_set_part_model_trainable(checkpoint_model, True))
            freeze_events.append(event)
        train_metrics = run_local_graph_tagger_epoch(
            train_model,
            train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            amp=bool(config.amp),
            grad_clip_norm=float(config.grad_clip_norm),
            max_batches=config.max_train_batches,
            collect_predictions=False,
            collect_diagnostics=False,
            label_names=tuple(config.label_names),
        )
        val_metrics = run_local_graph_tagger_epoch(
            train_model,
            val_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=config.max_val_batches,
            collect_predictions=_selection_metric_requires_predictions(str(config.selection_metric)),
            collect_diagnostics=False,
            label_names=tuple(config.label_names),
        )
        phase = "adapter_only" if int(epoch) <= int(config.freeze_part_epochs) else "full_finetune"
        row = {"epoch": int(epoch), "phase": phase, "train": train_metrics, "model_val": val_metrics}
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves})

        val_accuracy = _finite_float(val_metrics.get("accuracy"), default=-1.0)
        val_loss = _finite_float(val_metrics.get("loss"), default=float("inf"))
        val_score, selection_value = _selection_score(val_metrics, str(config.selection_metric))
        improved = val_score > best_val_score or (np.isclose(val_score, best_val_score) and val_loss < best_val_loss)
        payload = local_graph_tagger_checkpoint_payload(
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

    if int(config.freeze_part_epochs) > 0:
        event = {"epoch": len(curves), "event": "restore_part_model_trainable_after_training"}
        event.update(_set_part_model_trainable(checkpoint_model, True))
        freeze_events.append(event)

    if best_epoch < 0 or not (output_dir / "best_model_val.pt").exists():
        raise FloatingPointError("local graph tagger did not produce a valid model_val checkpoint")

    best_payload = torch.load(output_dir / "best_model_val.pt", map_location=device)
    if model is None:
        best_model, _ = load_local_graph_tagger_checkpoint(output_dir / "best_model_val.pt", device=device)
    else:
        checkpoint_model.load_state_dict(best_payload["model_state_dict"])
        checkpoint_model.eval()
        best_model = checkpoint_model

    best_val_metrics = run_local_graph_tagger_epoch(
        best_model,
        val_loader,
        device=device,
        criterion=criterion,
        amp=False,
        max_batches=config.max_val_batches,
        collect_predictions=True,
        collect_diagnostics=True,
        label_names=tuple(config.label_names),
    )
    metrics_by_split: dict[str, Mapping[str, Any]] = {"model_val": best_val_metrics}

    stack_val_dataset = stack_val_dataset or _load_local_graph_dataset(
        config,
        config.stack_val_split,
        max_jets=config.max_stack_val_jets,
    )
    stack_val_loader = make_subtoken_hlt_loader(
        stack_val_dataset,
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 2,
    )
    stack_val_metrics = run_local_graph_tagger_epoch(
        best_model,
        stack_val_loader,
        device=device,
        criterion=criterion,
        amp=False,
        max_batches=config.max_stack_val_batches,
        collect_predictions=True,
        collect_diagnostics=True,
        label_names=tuple(config.label_names),
    )
    metrics_by_split["stack_val"] = stack_val_metrics

    final_test_metrics = None
    final_test_metadata = None
    if bool(config.confirm_final_test):
        final_test_dataset = final_test_dataset or _load_local_graph_dataset(
            config,
            config.final_test_split,
            max_jets=config.max_final_test_jets,
        )
        final_test_loader = make_subtoken_hlt_loader(
            final_test_dataset,
            batch_size=int(config.eval_batch_size),
            shuffle=False,
            num_workers=int(config.num_workers),
            seed=int(config.seed) + 3,
        )
        final_test_metrics = run_local_graph_tagger_epoch(
            best_model,
            final_test_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=config.max_final_test_batches,
            collect_predictions=True,
            collect_diagnostics=True,
            label_names=tuple(config.label_names),
        )
        metrics_by_split["final_test"] = final_test_metrics
        final_test_metadata = dict(final_test_dataset.metadata)

    elapsed_seconds = float(time.perf_counter() - run_start_time)
    _write_summary_csv(diagnostics_dir / "summary_metrics.csv", metrics_by_split)

    report = {
        "experiment_step": LOCAL_GRAPH_PART_TRAIN_STEP,
        "protocol_step": LOCAL_GRAPH_PART_PROTOCOL_STEP,
        "protocol_contract": LOCAL_GRAPH_PART_CONTRACT,
        "protocol": local_graph_part_protocol_manifest(),
        "output_contract": best_model.output_contract,
        "variant": str(config.variant),
        "best_epoch": int(best_epoch),
        "selection_metric": str(config.selection_metric),
        "selection_metric_direction": (
            "minimize" if str(config.selection_metric) in LOCAL_GRAPH_LOWER_IS_BETTER_SELECTION_METRICS else "maximize"
        ),
        "warm_start": warm_start_report,
        "freeze_schedule": freeze_schedule,
        "freeze_events": freeze_events,
        "best_model_selection_metric_value": float(best_selection_metric_value),
        "best_model_selection_score": float(best_val_score),
        "best_model_val_accuracy": float(best_val_accuracy),
        "best_model_val_loss": float(best_val_loss),
        "best_model_val_metrics": best_val_metrics,
        "stack_val_metrics": stack_val_metrics,
        "final_test_metrics": final_test_metrics,
        "epochs_completed": len(curves),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "training_curves": str(output_dir / "training_curves.json"),
        "summary_metrics_csv": str(diagnostics_dir / "summary_metrics.csv"),
        "config": asdict(config),
        "model_config": best_model.to_config_dict(),
        "label_names": list(config.label_names),
        "label_filter": list(config.label_filter),
        "num_classes": int(config.num_classes),
        "source": source,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "stack_val_dataset": dict(stack_val_dataset.metadata),
        "final_test_dataset": final_test_metadata,
        "final_test_evaluated": bool(config.confirm_final_test),
        "no_final_test_evaluation": not bool(config.confirm_final_test),
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "elapsed_minutes": elapsed_seconds / 60.0,
            "epochs_completed": len(curves),
            "seconds_per_completed_epoch": elapsed_seconds / float(len(curves)) if curves else None,
        },
        "walltime_seconds": elapsed_seconds,
        "inference_consumes_hlt_only": True,
        "baseline_and_adapters_share_runner": True,
    }
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    _write_best_metrics_csv(diagnostics_dir / "best_metrics.csv", report)
    return report


__all__ = [
    "LOCAL_GRAPH_ALLOWED_STEP6_VARIANTS",
    "LOCAL_GRAPH_LOWER_IS_BETTER_SELECTION_METRICS",
    "LOCAL_GRAPH_PART_TRAIN_STEP",
    "LOCAL_GRAPH_SELECTION_METRICS",
    "LocalGraphTaggerTrainConfig",
    "build_local_graph_tagger_for_config",
    "load_local_graph_tagger_checkpoint",
    "local_graph_tagger_checkpoint_payload",
    "run_local_graph_tagger_epoch",
    "train_local_graph_tagger",
    "warm_start_local_graph_part_model",
]
