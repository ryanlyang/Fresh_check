"""Version A supervised training for reliability-gated subtoken taggers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json, set_training_seed
from jetclass_fresh.hlt_cache import load_cached_hlt_view
from jetclass_fresh.jetclass_data import LABEL_NAMES, JetView, RAW_TOKEN_DIM

from teacher_logit_reco.set_matching.five_view_train import classification_metrics_from_predictions
from teacher_logit_reco.set_matching.train import source_metadata

from .classifier import (
    SUBTOKEN_PART_PAIRWISE_CLASSIFIER_STEP,
    SubtokenParticleTransformerClassifier,
    build_subtoken_particle_transformer_classifier,
)
from .config import (
    SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX,
    SUBTOKEN_PART_VARIANT_CONTEXT_GATE,
    SUBTOKEN_PART_VERSION_A,
    SubtokenFeatureConfig,
    SubtokenModalitySpec,
    SubtokenPartConfig,
    normalize_subtoken_part_variant,
    normalize_subtoken_gate_mode,
    normalize_subtoken_dual_fusion_mode,
    normalize_subtoken_pool_mode,
    normalize_subtoken_split_name,
)


SUBTOKEN_PART_TRAIN_STEP = "subtoken_part_step12_train_version_a"
SUBTOKEN_PART_SELECTION_METRICS = (
    "accuracy",
    "loss",
    "macro_per_class_accuracy",
    "auc",
    "fpr_at_signal_eff_0p30",
    "fpr_at_signal_eff_0p50",
    "background_rejection_at_signal_eff_0p30",
    "background_rejection_at_signal_eff_0p50",
)
SUBTOKEN_PART_LOWER_IS_BETTER_SELECTION_METRICS = {"loss", "fpr_at_signal_eff_0p30", "fpr_at_signal_eff_0p50"}
SUBTOKEN_PART_DEFAULT_BINARY_SELECTION_METRIC = "fpr_at_signal_eff_0p50"
SUBTOKEN_PART_DEFAULT_MULTICLASS_SELECTION_METRIC = "accuracy"


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


def default_subtoken_selection_metric(num_classes: int) -> str:
    """Default checkpoint-selection metric for a subtoken tagging task."""

    return (
        SUBTOKEN_PART_DEFAULT_BINARY_SELECTION_METRIC
        if int(num_classes) == 2
        else SUBTOKEN_PART_DEFAULT_MULTICLASS_SELECTION_METRIC
    )


def normalize_subtoken_selection_metric(value: str | None, *, num_classes: int) -> str:
    """Resolve optional/auto selection metrics after class metadata is known."""

    if value is None or str(value).strip() == "":
        metric = default_subtoken_selection_metric(int(num_classes))
    else:
        metric = str(value).strip()
    if metric not in SUBTOKEN_PART_SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {SUBTOKEN_PART_SELECTION_METRICS}")
    return metric


def _feature_config_from_payload(payload: Mapping[str, Any]) -> SubtokenFeatureConfig:
    clean = dict(payload)
    clean.pop("num_modalities", None)
    modalities = []
    for row in clean.get("modalities", ()):
        if isinstance(row, Mapping):
            row = dict(row)
            row.pop("raw_feature_names", None)
            modalities.append(SubtokenModalitySpec(**row))
        else:
            modalities.append(row)
    if modalities:
        clean["modalities"] = tuple(modalities)
    return SubtokenFeatureConfig(**clean)


@dataclass
class SubtokenTaggerTrainConfig:
    """Configuration for Step 12 supervised HLT-only subtoken tagger training."""

    output_dir: str
    hlt_cache_dir: str
    train_split: str = "model_train"
    val_split: str = "model_val"
    stack_val_split: str = "stack_val"
    final_test_split: str = "final_test"
    confirm_split_settings: bool = False
    confirm_final_test: bool = False
    seed: int = 2607
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
    selection_metric: str | None = None
    compile_model: bool = False
    verify_hlt_hash: bool = True
    num_classes: int | None = None
    label_names: tuple[str, ...] = ()
    label_filter: tuple[int, ...] = ()

    variant: str = SUBTOKEN_PART_VARIANT_CONTEXT_GATE
    embed_dim: int = 128
    local_layers: int = 1
    local_heads: int = 4
    context_layers: int = 2
    context_heads: int = 4
    global_layers: int = 6
    global_heads: int = 8
    gate_mode: str = SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX
    local_pool_mode: str = "learned_query"
    use_pairwise_bias: bool = True
    use_particle_anchor: bool = True
    use_modality_type_embeddings: bool = True
    use_pt_rank_embedding: bool = False
    modality_dropout: float = 0.0
    dropout: float = 0.05
    attention_dropout: float = 0.05
    dual_fusion_mode: str = "cross_attention_class_token_fusion"
    standard_branch_layers: int = 6
    standard_branch_use_pairwise_bias: bool = True
    anchor_source: str = "raw"
    include_part_style_derived_features: bool = True

    def __post_init__(self) -> None:
        self.train_split = normalize_subtoken_split_name(self.train_split)
        self.val_split = normalize_subtoken_split_name(self.val_split)
        self.stack_val_split = normalize_subtoken_split_name(self.stack_val_split)
        self.final_test_split = normalize_subtoken_split_name(self.final_test_split)
        if self.train_split != "model_train" or self.val_split != "model_val":
            raise ValueError("Step 12 trains only on model_train and selects only on model_val")
        if self.stack_val_split != "stack_val" or self.final_test_split != "final_test":
            raise ValueError("Step 12 evaluates only stack_val/final_test after model_val selection")
        if not bool(self.confirm_split_settings):
            raise ValueError("Set --confirm-split-settings to acknowledge model_train/model_val-only selection")
        for field_name in (
            "batch_size",
            "eval_batch_size",
            "epochs",
            "embed_dim",
            "local_layers",
            "local_heads",
            "context_layers",
            "context_heads",
            "global_layers",
            "global_heads",
            "standard_branch_layers",
        ):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            setattr(self, field_name, value)
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
        for field_name in ("dropout", "attention_dropout", "modality_dropout"):
            value = float(getattr(self, field_name))
            if value < 0.0 or value >= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1)")
            setattr(self, field_name, value)
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
        self.gate_mode = normalize_subtoken_gate_mode(self.gate_mode)
        self.local_pool_mode = normalize_subtoken_pool_mode(self.local_pool_mode)
        self.dual_fusion_mode = normalize_subtoken_dual_fusion_mode(self.dual_fusion_mode)
        self.label_filter = tuple(int(label) for label in self.label_filter)
        if len(set(self.label_filter)) != len(self.label_filter):
            raise ValueError(f"label_filter contains duplicates: {self.label_filter}")
        if self.label_names:
            self.label_names = tuple(str(name) for name in self.label_names)
            if not self.label_filter:
                by_name = {name: index for index, name in enumerate(LABEL_NAMES)}
                if all(name in by_name for name in self.label_names):
                    self.label_filter = tuple(by_name[name] for name in self.label_names)
        if self.num_classes is not None:
            self.num_classes = _optional_positive_int(self.num_classes, field_name="num_classes")
        self.selection_metric = normalize_subtoken_selection_metric(
            self.selection_metric,
            num_classes=int(self.resolved_num_classes),
        )
        self.variant = normalize_subtoken_part_variant(self.variant)
        self.validate_label_metadata()

    @property
    def resolved_label_filter(self) -> tuple[int, ...]:
        if self.label_filter:
            return tuple(self.label_filter)
        if self.num_classes is None:
            return tuple(range(len(LABEL_NAMES)))
        return tuple(range(int(self.num_classes)))

    @property
    def resolved_label_names(self) -> tuple[str, ...]:
        if self.label_names:
            return tuple(self.label_names)
        return tuple(LABEL_NAMES[index] for index in self.resolved_label_filter)

    @property
    def resolved_num_classes(self) -> int:
        if self.num_classes is not None:
            return int(self.num_classes)
        return len(self.resolved_label_filter)

    def validate_label_metadata(self) -> None:
        if len(self.resolved_label_filter) != int(self.resolved_num_classes):
            raise ValueError("label_filter length must match num_classes")
        if len(self.resolved_label_names) != int(self.resolved_num_classes):
            raise ValueError("label_names length must match num_classes")
        if int(self.resolved_num_classes) <= 1:
            raise ValueError("num_classes must be greater than one")
        if max(self.resolved_label_filter, default=-1) >= len(LABEL_NAMES):
            raise ValueError(f"label_filter contains invalid JetClass label ids: {self.resolved_label_filter}")
        if int(self.resolved_num_classes) != 2 and self.selection_metric in {
            "auc",
            "fpr_at_signal_eff_0p30",
            "fpr_at_signal_eff_0p50",
            "background_rejection_at_signal_eff_0p30",
            "background_rejection_at_signal_eff_0p50",
        }:
            raise ValueError(f"selection_metric={self.selection_metric!r} requires a binary two-class setup")

    def model_config(self) -> SubtokenPartConfig:
        self.validate_label_metadata()
        feature_config = SubtokenFeatureConfig(
            raw_token_dim=RAW_TOKEN_DIM,
            anchor_source=str(self.anchor_source),
            include_part_style_derived_features=bool(self.include_part_style_derived_features),
        )
        return SubtokenPartConfig(
            num_classes=int(self.resolved_num_classes),
            feature_config=feature_config,
            variant=str(self.variant),
            version=SUBTOKEN_PART_VERSION_A,
            embed_dim=int(self.embed_dim),
            local_layers=int(self.local_layers),
            local_heads=int(self.local_heads),
            context_layers=int(self.context_layers),
            context_heads=int(self.context_heads),
            global_layers=int(self.global_layers),
            global_heads=int(self.global_heads),
            gate_mode=str(self.gate_mode),
            local_pool_mode=str(self.local_pool_mode),
            use_pairwise_bias=bool(self.use_pairwise_bias),
            use_particle_anchor=bool(self.use_particle_anchor),
            use_modality_type_embeddings=bool(self.use_modality_type_embeddings),
            use_pt_rank_embedding=bool(self.use_pt_rank_embedding),
            modality_dropout=float(self.modality_dropout),
            dropout=float(self.dropout),
            attention_dropout=float(self.attention_dropout),
            dual_fusion_mode=str(self.dual_fusion_mode),
            standard_branch_layers=int(self.standard_branch_layers),
            standard_branch_use_pairwise_bias=bool(self.standard_branch_use_pairwise_bias),
        )


class SubtokenHLTJetDataset:
    """Dataset over cached raw HLT tokens for the subtoken classifier."""

    def __init__(
        self,
        view: JetView,
        *,
        label_filter: Sequence[int],
        label_names: Sequence[str],
        max_jets: int | None = None,
    ) -> None:
        if view.metadata.get("view") not in (None, "fixed_hlt"):
            raise ValueError(f"Expected fixed_hlt cached view, got {view.metadata.get('view')!r}")
        labels = np.asarray(view.labels, dtype=np.int64)
        label_filter = tuple(int(label) for label in label_filter)
        label_names = tuple(str(name) for name in label_names)
        if len(label_filter) != len(label_names):
            raise ValueError("label_filter and label_names must have the same length")
        keep = np.isin(labels, np.asarray(label_filter, dtype=np.int64))
        if not np.all(keep):
            labels = labels[keep]
            tokens = np.asarray(view.tokens, dtype=np.float32)[keep]
            mask = np.asarray(view.mask, dtype=bool)[keep]
            jet_ids = [jet_id for jet_id, should_keep in zip(view.jet_ids, keep) if bool(should_keep)]
        else:
            tokens = np.asarray(view.tokens, dtype=np.float32)
            mask = np.asarray(view.mask, dtype=bool)
            jet_ids = list(view.jet_ids)
        remap = {source_label: index for index, source_label in enumerate(label_filter)}
        remapped = np.asarray([remap[int(label)] for label in labels], dtype=np.int64)
        if max_jets is not None:
            limit = min(int(max_jets), int(remapped.shape[0]))
            tokens = tokens[:limit]
            mask = mask[:limit]
            remapped = remapped[:limit]
            jet_ids = jet_ids[:limit]
        self.tokens = np.asarray(tokens, dtype=np.float32)
        self.mask = np.asarray(mask, dtype=bool)
        self.labels = remapped
        self.jet_ids = jet_ids
        self.split = view.split
        self.label_names = label_names
        self.label_filter = label_filter
        self.metadata = {
            "split": view.split,
            "source_view": "fixed_hlt",
            "n_jets": int(remapped.shape[0]),
            "raw_token_dim": int(self.tokens.shape[-1]),
            "max_constits": int(self.tokens.shape[1]),
            "label_filter": list(label_filter),
            "label_names": list(label_names),
            "label_counts": self.label_counts(),
            "max_jets_limit": None if max_jets is None else int(max_jets),
            "hlt_content_hash": view.metadata.get("hlt_content_hash"),
            "jet_identity_hash": view.metadata.get("jet_identity_hash"),
            "hlt_params": view.metadata.get("hlt_params"),
            "hlt_seed": view.metadata.get("seed"),
        }

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "tokens": self.tokens[index],
            "mask": self.mask[index],
            "labels": self.labels[index],
            "indices": np.int64(index),
        }

    def label_counts(self) -> dict[str, int]:
        return {
            self.label_names[index]: int(np.sum(self.labels == index))
            for index in range(len(self.label_names))
        }


def make_subtoken_hlt_loader(
    dataset: SubtokenHLTJetDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
):
    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_subtoken_hlt_batch,
        generator=generator,
    )


def collate_subtoken_hlt_batch(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    torch = require_torch()
    tokens = np.stack([sample["tokens"] for sample in samples], axis=0).astype(np.float32, copy=False)
    mask = np.stack([sample["mask"] for sample in samples], axis=0).astype(bool, copy=False)
    labels = np.asarray([sample["labels"] for sample in samples], dtype=np.int64)
    indices = np.asarray([sample["indices"] for sample in samples], dtype=np.int64)
    return {
        "tokens": torch.from_numpy(tokens).float(),
        "mask": torch.from_numpy(mask).bool(),
        "labels": torch.from_numpy(labels).long(),
        "indices": torch.from_numpy(indices).long(),
    }


def _move_batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("tokens", "mask", "labels", "indices"):
        value = moved.get(key)
        if hasattr(value, "to"):
            moved[key] = value.to(device, non_blocking=True)
    return moved


def _load_subtoken_hlt_dataset(
    config: SubtokenTaggerTrainConfig,
    split: str,
    *,
    max_jets: int | None,
) -> SubtokenHLTJetDataset:
    view = load_cached_hlt_view(config.hlt_cache_dir, split, verify_hash=bool(config.verify_hlt_hash))
    return SubtokenHLTJetDataset(
        view,
        label_filter=config.resolved_label_filter,
        label_names=config.resolved_label_names,
        max_jets=max_jets,
    )


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
    if metric_name in SUBTOKEN_PART_LOWER_IS_BETTER_SELECTION_METRICS:
        return -float(value), float(value)
    return float(value), float(value)


def _write_epoch_metrics_csv(path: Path, curves: list[dict[str, Any]]) -> None:
    keys: set[str] = {"epoch"}
    rows: list[dict[str, Any]] = []
    for row in curves:
        flat = {"epoch": row["epoch"]}
        for split in ("train", "model_val"):
            for key, value in row.get(split, {}).items():
                if isinstance(value, (int, float)):
                    flat[f"{split}_{key}"] = value
                    keys.add(f"{split}_{key}")
                binary = value if key == "binary_metrics" and isinstance(value, Mapping) else None
                if binary:
                    for b_key, b_value in binary.items():
                        if isinstance(b_value, (int, float)):
                            flat[f"{split}_{b_key}"] = b_value
                            keys.add(f"{split}_{b_key}")
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


def run_subtoken_tagger_epoch(
    model: SubtokenParticleTransformerClassifier,
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
    """Run one train/eval epoch for a subtoken tagger."""

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


def subtoken_tagger_checkpoint_payload(
    model: SubtokenParticleTransformerClassifier,
    optimizer,
    *,
    epoch: int,
    config: SubtokenTaggerTrainConfig,
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
        "label_names": list(config.resolved_label_names),
        "label_filter": list(config.resolved_label_filter),
        "num_classes": int(config.resolved_num_classes),
        "source": dict(source),
        "experiment_step": SUBTOKEN_PART_TRAIN_STEP,
        "tagger_model_step": SUBTOKEN_PART_PAIRWISE_CLASSIFIER_STEP,
        "output_contract": model.output_contract,
    }


def _model_config_from_checkpoint_payload(payload: Mapping[str, Any]) -> SubtokenPartConfig:
    model_config = payload.get("model_config")
    if isinstance(model_config, Mapping):
        model_config = dict(model_config)
        model_config.pop("contract", None)
        feature_payload = model_config.get("feature_config")
        if isinstance(feature_payload, Mapping):
            model_config["feature_config"] = _feature_config_from_payload(feature_payload)
        return SubtokenPartConfig(**model_config)
    config_payload = payload.get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("checkpoint payload does not contain model_config or config")
    train_config = SubtokenTaggerTrainConfig(**dict(config_payload))
    return train_config.model_config()


def load_subtoken_tagger_checkpoint(path: str | Path, *, device=None) -> tuple[SubtokenParticleTransformerClassifier, dict[str, Any]]:
    torch = require_torch()
    payload = torch.load(path, map_location=device or "cpu")
    model_config = _model_config_from_checkpoint_payload(payload)
    model = build_subtoken_particle_transformer_classifier(model_config)
    model.load_state_dict(payload["model_state_dict"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, payload


def train_subtoken_tagger(
    config: SubtokenTaggerTrainConfig,
    *,
    model: SubtokenParticleTransformerClassifier | None = None,
    train_dataset: SubtokenHLTJetDataset | None = None,
    val_dataset: SubtokenHLTJetDataset | None = None,
    stack_val_dataset: SubtokenHLTJetDataset | None = None,
    final_test_dataset: SubtokenHLTJetDataset | None = None,
) -> dict[str, Any]:
    """Train the Step 12 HLT-only subtoken classifier and evaluate guarded splits."""

    run_start_time = time.perf_counter()
    config.validate_label_metadata()
    torch = require_torch()
    set_training_seed(int(config.seed))
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = train_dataset or _load_subtoken_hlt_dataset(
        config,
        config.train_split,
        max_jets=config.max_train_jets,
    )
    val_dataset = val_dataset or _load_subtoken_hlt_dataset(
        config,
        config.val_split,
        max_jets=config.max_val_jets,
    )
    if train_dataset.tokens.shape[-1] != RAW_TOKEN_DIM or val_dataset.tokens.shape[-1] != RAW_TOKEN_DIM:
        raise ValueError(f"Subtoken tagger expects raw token dim {RAW_TOKEN_DIM}")
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

    checkpoint_model = model or build_subtoken_particle_transformer_classifier(config.model_config())
    checkpoint_model = checkpoint_model.to(device)
    train_model = checkpoint_model
    if config.compile_model and hasattr(torch, "compile"):
        train_model = torch.compile(train_model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(checkpoint_model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))
    source = source_metadata()
    run_metadata = {
        "experiment_step": SUBTOKEN_PART_TRAIN_STEP,
        "tagger_model_step": SUBTOKEN_PART_PAIRWISE_CLASSIFIER_STEP,
        "output_contract": checkpoint_model.output_contract,
        "config": asdict(config),
        "model_config": checkpoint_model.to_config_dict(),
        "source": source,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "label_names": list(config.resolved_label_names),
        "label_filter": list(config.resolved_label_filter),
        "num_classes": int(config.resolved_num_classes),
        "selection_metric": str(config.selection_metric),
        "leakage_rule": (
            "Step 12 Version A consumes only cached fixed-HLT tokens and labels. "
            "Training uses model_train, checkpoint selection uses model_val, and stack_val/final_test "
            "are loaded only after model_val checkpoint selection."
        ),
        "final_test_loaded_during_training": False,
        "inference_consumes_hlt_only": True,
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
        train_metrics = run_subtoken_tagger_epoch(
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
            label_names=tuple(config.resolved_label_names),
        )
        val_metrics = run_subtoken_tagger_epoch(
            train_model,
            val_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=config.max_val_batches,
            collect_predictions=_selection_metric_requires_predictions(str(config.selection_metric)),
            collect_diagnostics=False,
            label_names=tuple(config.resolved_label_names),
        )
        row = {"epoch": int(epoch), "train": train_metrics, "model_val": val_metrics}
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves})
        _write_epoch_metrics_csv(diagnostics_dir / "epoch_metrics.csv", curves)

        val_accuracy = _finite_float(val_metrics.get("accuracy"), default=-1.0)
        val_loss = _finite_float(val_metrics.get("loss"), default=float("inf"))
        val_score, selection_value = _selection_score(val_metrics, str(config.selection_metric))
        improved = val_score > best_val_score or (np.isclose(val_score, best_val_score) and val_loss < best_val_loss)
        payload = subtoken_tagger_checkpoint_payload(
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
        raise FloatingPointError("subtoken tagger did not produce a valid model_val checkpoint")

    best_model, _ = load_subtoken_tagger_checkpoint(output_dir / "best_model_val.pt", device=device)
    best_val_metrics = run_subtoken_tagger_epoch(
        best_model,
        val_loader,
        device=device,
        criterion=criterion,
        amp=False,
        max_batches=config.max_val_batches,
        collect_predictions=True,
        collect_diagnostics=True,
        label_names=tuple(config.resolved_label_names),
    )
    metrics_by_split: dict[str, Mapping[str, Any]] = {"model_val": best_val_metrics}

    stack_val_dataset = stack_val_dataset or _load_subtoken_hlt_dataset(
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
    stack_val_metrics = run_subtoken_tagger_epoch(
        best_model,
        stack_val_loader,
        device=device,
        criterion=criterion,
        amp=False,
        max_batches=config.max_stack_val_batches,
        collect_predictions=True,
        collect_diagnostics=True,
        label_names=tuple(config.resolved_label_names),
    )
    metrics_by_split["stack_val"] = stack_val_metrics

    final_test_metrics = None
    final_test_metadata = None
    if bool(config.confirm_final_test):
        final_test_dataset = final_test_dataset or _load_subtoken_hlt_dataset(
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
        final_test_metrics = run_subtoken_tagger_epoch(
            best_model,
            final_test_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=config.max_final_test_batches,
            collect_predictions=True,
            collect_diagnostics=True,
            label_names=tuple(config.resolved_label_names),
        )
        metrics_by_split["final_test"] = final_test_metrics
        final_test_metadata = dict(final_test_dataset.metadata)

    elapsed_seconds = float(time.perf_counter() - run_start_time)
    _write_per_class_csv(diagnostics_dir / "per_class_metrics.csv", metrics_by_split)
    _write_summary_csv(diagnostics_dir / "summary_metrics.csv", metrics_by_split)

    report = {
        "experiment_step": SUBTOKEN_PART_TRAIN_STEP,
        "tagger_model_step": SUBTOKEN_PART_PAIRWISE_CLASSIFIER_STEP,
        "output_contract": best_model.output_contract,
        "best_epoch": int(best_epoch),
        "selection_metric": str(config.selection_metric),
        "selection_metric_direction": (
            "minimize" if str(config.selection_metric) in SUBTOKEN_PART_LOWER_IS_BETTER_SELECTION_METRICS else "maximize"
        ),
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
        "diagnostic_csv": str(diagnostics_dir / "epoch_metrics.csv"),
        "summary_metrics_csv": str(diagnostics_dir / "summary_metrics.csv"),
        "per_class_metrics_csv": str(diagnostics_dir / "per_class_metrics.csv"),
        "config": asdict(config),
        "model_config": best_model.to_config_dict(),
        "label_names": list(config.resolved_label_names),
        "label_filter": list(config.resolved_label_filter),
        "num_classes": int(config.resolved_num_classes),
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
    }
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    _write_best_metrics_csv(diagnostics_dir / "best_metrics.csv", report)
    return report


__all__ = [
    "SUBTOKEN_PART_DEFAULT_BINARY_SELECTION_METRIC",
    "SUBTOKEN_PART_DEFAULT_MULTICLASS_SELECTION_METRIC",
    "SUBTOKEN_PART_LOWER_IS_BETTER_SELECTION_METRICS",
    "SUBTOKEN_PART_SELECTION_METRICS",
    "SUBTOKEN_PART_TRAIN_STEP",
    "SubtokenHLTJetDataset",
    "SubtokenTaggerTrainConfig",
    "collate_subtoken_hlt_batch",
    "default_subtoken_selection_metric",
    "load_subtoken_tagger_checkpoint",
    "make_subtoken_hlt_loader",
    "normalize_subtoken_selection_metric",
    "run_subtoken_tagger_epoch",
    "subtoken_tagger_checkpoint_payload",
    "train_subtoken_tagger",
]
