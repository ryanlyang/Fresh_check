"""Training loop for the set-matching five-view tagger."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json, set_training_seed
from jetclass_fresh.jetclass_data import LABEL_NAMES

from .experiment import SPLIT_SIZES, normalize_split_name
from .five_view_data import FiveViewDatasetConfig, FiveViewJetDataset, make_five_view_loader
from .five_view_model import (
    SET_MATCHING_FIVE_VIEW_TAGGER_STEP,
    FiveViewParticleTransformerConfig,
    FiveViewParticleTransformerTagger,
    build_five_view_tagger,
    five_view_tagger_checkpoint_payload as five_view_tagger_model_checkpoint_payload,
    load_five_view_tagger_checkpoint,
)
from .train import source_metadata


SET_MATCHING_FIVE_VIEW_TRAIN_STEP = "set_matching_multiview_step10_train_five_view_tagger"
FIVE_VIEW_SELECTION_METRICS = (
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


@dataclass
class FiveViewTaggerTrainConfig:
    """Configuration for Step 10 five-view tagger training."""

    output_dir: str
    hlt_cache_dir: str
    experiment_dir: str | None = None
    reconstructed_view_dir: str | None = None
    train_split: str = "stack_train"
    val_split: str = "stack_val"
    final_test_split: str = "final_test"
    confirm_split_settings: bool = False
    confirm_final_test: bool = False
    seed: int = 1205
    batch_size: int = 64
    epochs: int = 30
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 5
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_final_test_batches: int | None = None
    max_train_jets: int | None = None
    max_val_jets: int | None = None
    max_final_test_jets: int | None = None
    selection_metric: str = "accuracy"
    compile_model: bool = False
    num_classes: int | None = None
    label_names: tuple[str, ...] = ()
    label_filter: tuple[int, ...] = ()

    max_tokens_per_view: int = 128
    min_tokens_per_view: int = 8
    confidence_threshold: float = 0.05
    selection_mode: str = "topk_or_threshold"
    drop_views: tuple[str, ...] = ()
    shuffle_view_labels: bool = False
    view_label_shuffle_seed: int = 1205
    verify_hlt_hash: bool = True

    embed_dim: int = 128
    stage1_layers: int = 2
    stage1_heads: int = 4
    stage2_layers: int = 4
    stage2_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.05
    attention_dropout: float = 0.05
    classifier_hidden_dim: int | None = None
    use_confidence: bool = True
    use_view_embedding: bool = True
    use_source_embedding: bool = True
    use_view_summaries: bool = True
    use_geometry_attention: bool = False
    geometry_hidden_dim: int = 64
    geometry_dropout: float = 0.0

    def __post_init__(self) -> None:
        self.train_split = normalize_split_name(self.train_split)
        self.val_split = normalize_split_name(self.val_split)
        self.final_test_split = normalize_split_name(self.final_test_split)
        if self.train_split != "stack_train" or self.val_split != "stack_val" or self.final_test_split != "final_test":
            raise ValueError("Step 10 trains on stack_train, selects on stack_val, and reserves final_test")
        if not bool(self.confirm_split_settings):
            raise ValueError("Set --confirm-split-settings to acknowledge stack_train/stack_val-only model selection")
        for field_name in ("batch_size", "epochs", "max_tokens_per_view", "embed_dim", "stage1_layers", "stage1_heads", "stage2_layers", "stage2_heads"):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            setattr(self, field_name, value)
        if int(self.min_tokens_per_view) < 0:
            raise ValueError("min_tokens_per_view cannot be negative")
        if int(self.min_tokens_per_view) > int(self.max_tokens_per_view):
            raise ValueError("min_tokens_per_view cannot exceed max_tokens_per_view")
        if int(self.num_workers) < 0:
            raise ValueError("num_workers cannot be negative")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if float(self.mlp_ratio) <= 0.0:
            raise ValueError("mlp_ratio must be positive")
        if float(self.grad_clip_norm) < 0.0:
            raise ValueError("grad_clip_norm cannot be negative")
        if int(self.early_stop_patience) < -1:
            raise ValueError("early_stop_patience must be -1 or greater")
        if not 0.0 <= float(self.confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= float(self.attention_dropout) < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if not 0.0 <= float(self.geometry_dropout) < 1.0:
            raise ValueError("geometry_dropout must be in [0, 1)")
        self.selection_metric = str(self.selection_metric)
        if self.selection_metric not in FIVE_VIEW_SELECTION_METRICS:
            raise ValueError(f"selection_metric must be one of {FIVE_VIEW_SELECTION_METRICS}")
        if self.classifier_hidden_dim is not None:
            self.classifier_hidden_dim = _optional_positive_int(
                self.classifier_hidden_dim,
                field_name="classifier_hidden_dim",
            )
        if int(self.geometry_hidden_dim) <= 0:
            raise ValueError("geometry_hidden_dim must be positive")
        if int(self.embed_dim) % int(self.stage1_heads) != 0:
            raise ValueError("embed_dim must be divisible by stage1_heads")
        if int(self.embed_dim) % int(self.stage2_heads) != 0:
            raise ValueError("embed_dim must be divisible by stage2_heads")
        self.max_train_batches = _optional_nonnegative_int(self.max_train_batches, field_name="max_train_batches")
        self.max_val_batches = _optional_nonnegative_int(self.max_val_batches, field_name="max_val_batches")
        self.max_final_test_batches = _optional_nonnegative_int(self.max_final_test_batches, field_name="max_final_test_batches")
        self.max_train_jets = _optional_positive_int(self.max_train_jets, field_name="max_train_jets")
        self.max_val_jets = _optional_positive_int(self.max_val_jets, field_name="max_val_jets")
        self.max_final_test_jets = _optional_positive_int(self.max_final_test_jets, field_name="max_final_test_jets")
        if self.num_classes is None:
            self.num_classes = len(LABEL_NAMES)
        else:
            self.num_classes = _optional_positive_int(self.num_classes, field_name="num_classes")
        if not self.label_names:
            self.label_names = tuple(LABEL_NAMES[: int(self.num_classes)])
        else:
            self.label_names = tuple(str(name) for name in self.label_names)
        if len(self.label_names) != int(self.num_classes):
            raise ValueError("label_names length must match num_classes")
        self.label_filter = tuple(int(label) for label in self.label_filter)
        if len(set(self.label_filter)) != len(self.label_filter):
            raise ValueError(f"label_filter contains duplicates: {self.label_filter}")
        self.drop_views = tuple(self.drop_views)

    def dataset_config(self, split: str) -> FiveViewDatasetConfig:
        return FiveViewDatasetConfig(
            output_dir=str(self.experiment_root),
            hlt_cache_dir=self.hlt_cache_dir,
            reconstructed_view_dir=self.reconstructed_view_dir,
            split=split,
            max_tokens_per_view=int(self.max_tokens_per_view),
            min_tokens_per_view=int(self.min_tokens_per_view),
            confidence_threshold=float(self.confidence_threshold),
            selection_mode=self.selection_mode,
            drop_views=tuple(self.drop_views),
            label_filter=tuple(self.label_filter),
            shuffle_view_labels=bool(self.shuffle_view_labels),
            view_label_shuffle_seed=int(self.view_label_shuffle_seed),
            verify_hlt_hash=bool(self.verify_hlt_hash),
        )

    @property
    def experiment_root(self) -> Path:
        if self.experiment_dir:
            return Path(self.experiment_dir)
        return infer_experiment_dir_from_tagger_output(self.output_dir)

    def model_config(self, *, particle_feature_dim: int) -> FiveViewParticleTransformerConfig:
        return FiveViewParticleTransformerConfig(
            particle_feature_dim=int(particle_feature_dim),
            num_classes=int(self.num_classes or len(LABEL_NAMES)),
            embed_dim=int(self.embed_dim),
            stage1_layers=int(self.stage1_layers),
            stage1_heads=int(self.stage1_heads),
            stage2_layers=int(self.stage2_layers),
            stage2_heads=int(self.stage2_heads),
            mlp_ratio=float(self.mlp_ratio),
            dropout=float(self.dropout),
            attention_dropout=float(self.attention_dropout),
            classifier_hidden_dim=self.classifier_hidden_dim,
            use_confidence=bool(self.use_confidence),
            use_view_embedding=bool(self.use_view_embedding),
            use_source_embedding=bool(self.use_source_embedding),
            use_view_summaries=bool(self.use_view_summaries),
            use_geometry_attention=bool(self.use_geometry_attention),
            geometry_hidden_dim=int(self.geometry_hidden_dim),
            geometry_dropout=float(self.geometry_dropout),
        )


def _slice_five_view_dataset(dataset: FiveViewJetDataset, max_jets: int | None) -> FiveViewJetDataset:
    if max_jets is None or int(max_jets) >= len(dataset):
        return dataset
    limit = int(max_jets)
    metadata = dict(dataset.metadata)
    metadata["limited_from_n_jets"] = len(dataset)
    metadata["n_jets"] = limit
    metadata["max_jets_limit"] = limit
    return FiveViewJetDataset(
        view_features=dataset.view_features[:limit],
        view_masks=dataset.view_masks[:limit],
        view_confidence=dataset.view_confidence[:limit],
        labels=dataset.labels[:limit],
        jet_ids=dataset.jet_ids[:limit],
        split=dataset.split,
        view_names=dataset.view_names,
        source_types=dataset.source_types,
        view_ids=dataset.view_ids,
        source_type_ids=dataset.source_type_ids,
        source_indices=dataset.source_indices[:limit],
        metadata=metadata,
    )


def _load_five_view_dataset(config: FiveViewTaggerTrainConfig, split: str, *, max_jets: int | None) -> FiveViewJetDataset:
    dataset = FiveViewJetDataset.from_caches(config.dataset_config(split))
    return _slice_five_view_dataset(dataset, max_jets)


def _move_five_view_batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("view_features", "view_masks", "view_confidence", "view_ids", "source_type_ids", "labels", "indices"):
        value = moved.get(key)
        if hasattr(value, "to"):
            moved[key] = value.to(device, non_blocking=True)
    return moved


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum_pos = float(np.sum(ranks[positives]))
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg))


def _fpr_at_signal_efficiency(labels: np.ndarray, scores: np.ndarray, target_efficiency: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))
    if n_pos == 0 or n_neg == 0:
        return {
            "target_signal_efficiency": float(target_efficiency),
            "threshold": float("nan"),
            "signal_efficiency": float("nan"),
            "false_positive_rate": float("nan"),
            "background_rejection": float("nan"),
        }
    positive_scores = np.sort(scores[positives])[::-1]
    threshold_index = min(max(int(np.ceil(float(target_efficiency) * n_pos)) - 1, 0), n_pos - 1)
    threshold = float(positive_scores[threshold_index])
    signal_efficiency = float(np.mean(scores[positives] >= threshold))
    false_positive_rate = float(np.mean(scores[negatives] >= threshold))
    return {
        "target_signal_efficiency": float(target_efficiency),
        "threshold": threshold,
        "signal_efficiency": signal_efficiency,
        "false_positive_rate": false_positive_rate,
        "background_rejection": float("inf") if false_positive_rate == 0.0 else float(1.0 / false_positive_rate),
    }


def binary_classification_metrics_from_logits(
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    signal_class_index: int = 1,
    label_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("binary metrics require logits with shape [N, 2]")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("binary metrics require labels encoded as 0/1")
    names = tuple(label_names or ("background", "signal"))
    probs = _softmax_numpy(logits)
    signal_scores = probs[:, int(signal_class_index)]
    by_eff = {
        "signal_eff_0p30": _fpr_at_signal_efficiency(labels, signal_scores, 0.30),
        "signal_eff_0p50": _fpr_at_signal_efficiency(labels, signal_scores, 0.50),
    }
    return {
        "positive_class_index": int(signal_class_index),
        "positive_class_name": names[int(signal_class_index)] if int(signal_class_index) < len(names) else str(signal_class_index),
        "auc": _binary_auc(labels, signal_scores),
        "fpr_at_signal_efficiency": by_eff,
        "fpr_at_signal_eff_0p30": by_eff["signal_eff_0p30"]["false_positive_rate"],
        "fpr_at_signal_eff_0p50": by_eff["signal_eff_0p50"]["false_positive_rate"],
        "background_rejection_at_signal_eff_0p30": by_eff["signal_eff_0p30"]["background_rejection"],
        "background_rejection_at_signal_eff_0p50": by_eff["signal_eff_0p50"]["background_rejection"],
    }


def classification_metrics_from_predictions(
    *,
    preds: np.ndarray,
    labels: np.ndarray,
    loss_sum: float | None = None,
    logits: np.ndarray | None = None,
    label_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    if labels.shape != preds.shape:
        raise ValueError("preds and labels must have matching shapes")
    names = tuple(label_names or LABEL_NAMES)
    n_classes = len(names)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, preds):
        if 0 <= int(true_label) < n_classes and 0 <= int(pred_label) < n_classes:
            confusion[int(true_label), int(pred_label)] += 1
    per_class = []
    for index, name in enumerate(names):
        support = int(confusion[index].sum())
        correct = int(confusion[index, index])
        accuracy = correct / float(support) if support else 0.0
        per_class.append({"class_index": index, "class_name": name, "support": support, "correct": correct, "accuracy": accuracy})
    n_jets = int(labels.size)
    metrics = {
        "accuracy": float(np.mean(preds == labels)) if n_jets else 0.0,
        "n_jets": n_jets,
        "confusion_matrix": confusion.astype(int).tolist(),
        "per_class_accuracy": per_class,
        "macro_per_class_accuracy": float(np.mean([row["accuracy"] for row in per_class])) if per_class else 0.0,
    }
    if loss_sum is not None:
        metrics["loss"] = float(loss_sum) / float(n_jets) if n_jets else float("nan")
    if logits is not None and n_classes == 2:
        metrics["binary_metrics"] = binary_classification_metrics_from_logits(logits=logits, labels=labels, label_names=names)
    return metrics


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
    if metric_name in LOWER_IS_BETTER_SELECTION_METRICS:
        return -float(value), float(value)
    return float(value), float(value)


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


def run_five_view_tagger_epoch(
    model: FiveViewParticleTransformerTagger,
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
    label_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run one train/eval epoch for the five-view tagger."""

    torch = require_torch()
    training = optimizer is not None
    model.train(training)
    autocast_enabled = bool(amp and device.type == "cuda")
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    collected_preds: list[np.ndarray] = []
    collected_labels: list[np.ndarray] = []
    collected_logits: list[np.ndarray] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = _move_five_view_batch_to_device(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                logits = model(
                    batch["view_features"],
                    batch["view_masks"],
                    batch["view_confidence"],
                    batch["view_ids"],
                    batch["source_type_ids"],
                )
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
    return metrics


def infer_experiment_dir_from_tagger_output(output_dir: str | Path) -> Path:
    """Infer the set-matching experiment root from a tagger output path.

    The canonical layout is:

    ``<experiment_root>/taggers/<tagger_name>``

    Step 10 usually receives the leaf tagger output directory from a runner.
    This helper keeps the CLI ergonomic while still letting callers override
    the root explicitly with ``experiment_dir``.
    """

    path = Path(output_dir)
    if "taggers" in path.parts:
        taggers_index = path.parts.index("taggers")
        if taggers_index <= 0:
            return path.parent
        return Path(*path.parts[:taggers_index])
    return path.parent


def five_view_tagger_training_checkpoint_payload(
    model: FiveViewParticleTransformerTagger,
    optimizer,
    *,
    epoch: int,
    config: FiveViewTaggerTrainConfig,
    metrics: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return five_view_tagger_model_checkpoint_payload(
        model,
        extra_payload={
            "epoch": int(epoch),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "metrics": dict(metrics),
            "source": dict(source),
            "experiment_step": SET_MATCHING_FIVE_VIEW_TRAIN_STEP,
            "tagger_model_step": SET_MATCHING_FIVE_VIEW_TAGGER_STEP,
        },
    )


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
        rows.append(flat)
    ordered = ["epoch"] + sorted(key for key in keys if key != "epoch")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def _view_ablation_metrics(
    *,
    model: FiveViewParticleTransformerTagger,
    dataset: FiveViewJetDataset,
    config: FiveViewTaggerTrainConfig,
    device,
    criterion,
    split: str,
    max_batches: int | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for view_index, view_name in enumerate(dataset.view_names):
        ablated = _copy_dataset_with_view_dropped(dataset, view_index=view_index)
        loader = make_five_view_loader(
            ablated,
            batch_size=int(config.batch_size),
            shuffle=False,
            num_workers=int(config.num_workers),
            seed=int(config.seed) + 500 + int(view_index),
        )
        metrics[f"drop_{view_name}"] = run_five_view_tagger_epoch(
            model,
            loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=max_batches,
            collect_predictions=False,
        )
    return {"split": split, "metrics": metrics}


def _copy_dataset_with_view_dropped(dataset: FiveViewJetDataset, *, view_index: int) -> FiveViewJetDataset:
    view_features = dataset.view_features.copy()
    view_masks = dataset.view_masks.copy()
    view_confidence = dataset.view_confidence.copy()
    source_indices = dataset.source_indices.copy()
    view_features[:, int(view_index)] = 0.0
    view_masks[:, int(view_index)] = False
    view_confidence[:, int(view_index)] = 0.0
    source_indices[:, int(view_index)] = -1
    metadata = dict(dataset.metadata)
    metadata["ablation_dropped_view"] = dataset.view_names[int(view_index)]
    return FiveViewJetDataset(
        view_features=view_features,
        view_masks=view_masks,
        view_confidence=view_confidence,
        labels=dataset.labels,
        jet_ids=dataset.jet_ids,
        split=dataset.split,
        view_names=dataset.view_names,
        source_types=dataset.source_types,
        view_ids=dataset.view_ids,
        source_type_ids=dataset.source_type_ids,
        source_indices=source_indices,
        metadata=metadata,
    )


def train_five_view_tagger(
    config: FiveViewTaggerTrainConfig,
    *,
    model: FiveViewParticleTransformerTagger | None = None,
    train_dataset: FiveViewJetDataset | None = None,
    val_dataset: FiveViewJetDataset | None = None,
    final_test_dataset: FiveViewJetDataset | None = None,
) -> dict[str, Any]:
    """Train the Step 10 five-view tagger and optionally evaluate final_test."""

    torch = require_torch()
    set_training_seed(int(config.seed))
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = train_dataset or _load_five_view_dataset(config, config.train_split, max_jets=config.max_train_jets)
    val_dataset = val_dataset or _load_five_view_dataset(config, config.val_split, max_jets=config.max_val_jets)
    if train_dataset.feature_dim != val_dataset.feature_dim:
        raise ValueError("train and validation five-view feature dimensions do not match")
    if train_dataset.view_names != val_dataset.view_names:
        raise ValueError("train and validation view names do not match")

    train_loader = make_five_view_loader(
        train_dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        seed=int(config.seed),
    )
    val_loader = make_five_view_loader(
        val_dataset,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 1,
    )

    checkpoint_model = model or build_five_view_tagger(config.model_config(particle_feature_dim=train_dataset.feature_dim))
    checkpoint_model = checkpoint_model.to(device)
    train_model = checkpoint_model
    if config.compile_model and hasattr(torch, "compile"):
        train_model = torch.compile(train_model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(checkpoint_model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))
    source = source_metadata()
    run_metadata = {
        "experiment_step": SET_MATCHING_FIVE_VIEW_TRAIN_STEP,
        "tagger_model_step": SET_MATCHING_FIVE_VIEW_TAGGER_STEP,
        "output_contract": checkpoint_model.output_contract,
        "config": asdict(config),
        "model_config": checkpoint_model.to_config_dict(),
        "source": source,
        "expected_split_sizes": dict(SPLIT_SIZES),
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "view_names": list(train_dataset.view_names),
        "label_names": list(config.label_names),
        "num_classes": int(config.num_classes or len(config.label_names)),
        "label_filter": list(config.label_filter),
        "leakage_rule": (
            "Step 10 trains the five-view tagger only on stack_train and selects only on stack_val. "
            "The five input views are fixed-HLT-derived cached views. final_test is loaded only with "
            "--confirm-final-test after checkpoint selection."
        ),
        "final_test_loaded_during_training": False,
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
        train_metrics = run_five_view_tagger_epoch(
            train_model,
            train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            amp=bool(config.amp),
            grad_clip_norm=float(config.grad_clip_norm),
            max_batches=config.max_train_batches,
        )
        val_metrics = run_five_view_tagger_epoch(
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

        val_accuracy = _finite_float(val_metrics.get("accuracy"), default=-1.0)
        val_loss = _finite_float(val_metrics.get("loss"), default=float("inf"))
        val_score, selection_value = _selection_score(val_metrics, str(config.selection_metric))
        improved = val_score > best_val_score or (np.isclose(val_score, best_val_score) and val_loss < best_val_loss)
        payload = five_view_tagger_training_checkpoint_payload(
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
        raise FloatingPointError("five-view tagger did not produce a valid stack_val checkpoint")

    best_model, _ = load_five_view_tagger_checkpoint(output_dir / "best_model_val.pt", device=device)
    best_val_metrics = run_five_view_tagger_epoch(
        best_model,
        val_loader,
        device=device,
        criterion=criterion,
        amp=False,
        max_batches=config.max_val_batches,
        collect_predictions=True,
        label_names=tuple(config.label_names),
    )
    metrics_by_split = {"stack_val": best_val_metrics}
    view_ablation = {
        "stack_val": _view_ablation_metrics(
            model=best_model,
            dataset=val_dataset,
            config=config,
            device=device,
            criterion=criterion,
            split="stack_val",
            max_batches=config.max_val_batches,
        )
    }
    final_test_metrics = None
    final_test_metadata = None
    if bool(config.confirm_final_test):
        final_test_dataset = final_test_dataset or _load_five_view_dataset(
            config,
            config.final_test_split,
            max_jets=config.max_final_test_jets,
        )
        final_test_loader = make_five_view_loader(
            final_test_dataset,
            batch_size=int(config.batch_size),
            shuffle=False,
            num_workers=int(config.num_workers),
            seed=int(config.seed) + 2,
        )
        final_test_metrics = run_five_view_tagger_epoch(
            best_model,
            final_test_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=config.max_final_test_batches,
            collect_predictions=True,
            label_names=tuple(config.label_names),
        )
        metrics_by_split["final_test"] = final_test_metrics
        final_test_metadata = dict(final_test_dataset.metadata)
        view_ablation["final_test"] = _view_ablation_metrics(
            model=best_model,
            dataset=final_test_dataset,
            config=config,
            device=device,
            criterion=criterion,
            split="final_test",
            max_batches=config.max_final_test_batches,
        )

    _write_per_class_csv(diagnostics_dir / "per_class_metrics.csv", metrics_by_split)
    save_json(diagnostics_dir / "view_ablation_metrics.json", view_ablation)
    report = {
        "experiment_step": SET_MATCHING_FIVE_VIEW_TRAIN_STEP,
        "tagger_model_step": SET_MATCHING_FIVE_VIEW_TAGGER_STEP,
        "output_contract": best_model.output_contract,
        "best_epoch": int(best_epoch),
        "selection_metric": str(config.selection_metric),
        "selection_metric_direction": "minimize" if str(config.selection_metric) in LOWER_IS_BETTER_SELECTION_METRICS else "maximize",
        "best_model_selection_metric_value": float(best_selection_metric_value),
        "best_model_selection_score": float(best_val_score),
        "best_model_val_accuracy": float(best_val_accuracy),
        "best_model_val_loss": float(best_val_loss),
        "best_stack_val_metrics": best_val_metrics,
        "final_test_metrics": final_test_metrics,
        "epochs_completed": len(curves),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "training_curves": str(output_dir / "training_curves.json"),
        "per_class_metrics_csv": str(diagnostics_dir / "per_class_metrics.csv"),
        "view_ablation_metrics": str(diagnostics_dir / "view_ablation_metrics.json"),
        "config": asdict(config),
        "model_config": best_model.to_config_dict(),
        "label_names": list(config.label_names),
        "num_classes": int(config.num_classes or len(config.label_names)),
        "label_filter": list(config.label_filter),
        "source": source,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "final_test_dataset": final_test_metadata,
        "final_test_evaluated": bool(config.confirm_final_test),
        "no_final_test_evaluation": not bool(config.confirm_final_test),
        "inference_consumes_hlt_only_plus_cached_hlt_derived_reco_views": True,
    }
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    return report


__all__ = [
    "SET_MATCHING_FIVE_VIEW_TRAIN_STEP",
    "FiveViewTaggerTrainConfig",
    "classification_metrics_from_predictions",
    "five_view_tagger_training_checkpoint_payload",
    "infer_experiment_dir_from_tagger_output",
    "run_five_view_tagger_epoch",
    "train_five_view_tagger",
]
