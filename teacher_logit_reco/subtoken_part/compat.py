"""Step 13 compatibility runner for HLT ParT and subtoken variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import (
    build_particle_transformer_classifier,
    require_torch,
    resolve_device,
    save_json,
    set_training_seed,
)
from jetclass_fresh.hlt_cache import load_cached_hlt_view
from jetclass_fresh.jetclass_data import LABEL_NAMES, RAW_TOKEN_DIM
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES, build_particle_transformer_inputs_from_tokens

from teacher_logit_reco.set_matching.five_view_train import classification_metrics_from_predictions
from teacher_logit_reco.set_matching.train import source_metadata

from .config import (
    SUBTOKEN_PART_VARIANT_CONTEXT_GATE,
    SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE,
    SUBTOKEN_PART_VARIANT_LOCAL_GATE,
    SUBTOKEN_PART_VARIANT_NO_GATE,
    build_subtoken_variant_config,
    normalize_subtoken_part_variant,
    normalize_subtoken_split_name,
)
from .train import (
    SUBTOKEN_PART_LOWER_IS_BETTER_SELECTION_METRICS,
    SUBTOKEN_PART_SELECTION_METRICS,
    SubtokenHLTJetDataset,
    SubtokenTaggerTrainConfig,
    _finite_float,
    _lookup_selection_metric,
    _optional_nonnegative_int,
    _optional_positive_int,
    _selection_metric_requires_predictions,
    _selection_score,
    _write_best_metrics_csv,
    _write_epoch_metrics_csv,
    _write_per_class_csv,
    _write_summary_csv,
    train_subtoken_tagger,
)


SUBTOKEN_PART_COMPAT_STEP = "subtoken_part_step13_hlt_part_compat"
HLT_PART_BASELINE_COMPAT_CONTRACT = "standard_hlt_particle_transformer_baseline_for_subtoken_comparison_v1"
SUBTOKEN_PART_STEP13_VARIANTS = (
    SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE,
    SUBTOKEN_PART_VARIANT_NO_GATE,
    SUBTOKEN_PART_VARIANT_LOCAL_GATE,
    SUBTOKEN_PART_VARIANT_CONTEXT_GATE,
)


def _normalize_step13_variants(values: Sequence[str]) -> tuple[str, ...]:
    variants = tuple(normalize_subtoken_part_variant(value) for value in values)
    if not variants:
        raise ValueError("variants must contain at least one Step 13 variant")
    invalid = [variant for variant in variants if variant not in SUBTOKEN_PART_STEP13_VARIANTS]
    if invalid:
        raise ValueError(f"Step 13 supports only {SUBTOKEN_PART_STEP13_VARIANTS}; got invalid variants {invalid}")
    if len(set(variants)) != len(variants):
        raise ValueError(f"variants contains duplicates: {variants}")
    return variants


@dataclass
class SubtokenPartCompatibilityConfig:
    """Run standard HLT ParT and subtoken variants on identical HLT splits."""

    output_dir: str
    hlt_cache_dir: str
    variants: tuple[str, ...] = field(default_factory=lambda: SUBTOKEN_PART_STEP13_VARIANTS)
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
    selection_metric: str = "accuracy"
    compile_model: bool = False
    verify_hlt_hash: bool = True
    num_classes: int | None = None
    label_names: tuple[str, ...] = ()
    label_filter: tuple[int, ...] = ()

    baseline_model_size: str = "base"
    embed_dim: int = 128
    local_layers: int = 1
    local_heads: int = 4
    context_layers: int = 2
    context_heads: int = 4
    global_layers: int = 6
    global_heads: int = 8
    local_pool_mode: str = "learned_query"
    use_particle_anchor: bool = True
    use_modality_type_embeddings: bool = True
    use_pt_rank_embedding: bool = False
    modality_dropout: float = 0.0
    dropout: float = 0.05
    attention_dropout: float = 0.05
    anchor_source: str = "raw"
    include_part_style_derived_features: bool = True

    def __post_init__(self) -> None:
        self.variants = _normalize_step13_variants(self.variants)
        self.train_split = normalize_subtoken_split_name(self.train_split)
        self.val_split = normalize_subtoken_split_name(self.val_split)
        self.stack_val_split = normalize_subtoken_split_name(self.stack_val_split)
        self.final_test_split = normalize_subtoken_split_name(self.final_test_split)
        if self.train_split != "model_train" or self.val_split != "model_val":
            raise ValueError("Step 13 trains only on model_train and selects only on model_val")
        if self.stack_val_split != "stack_val" or self.final_test_split != "final_test":
            raise ValueError("Step 13 evaluates only stack_val/final_test after model_val selection")
        if not bool(self.confirm_split_settings):
            raise ValueError("Set confirm_split_settings=True to acknowledge identical split usage")
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
        if self.selection_metric not in SUBTOKEN_PART_SELECTION_METRICS:
            raise ValueError(f"selection_metric must be one of {SUBTOKEN_PART_SELECTION_METRICS}")
        if self.baseline_model_size not in {"tiny", "base"}:
            raise ValueError("baseline_model_size must be 'tiny' or 'base'")
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

    def child_output_dir(self, variant: str) -> Path:
        return Path(self.output_dir) / normalize_subtoken_part_variant(variant)

    def to_subtoken_train_config(self, variant: str) -> SubtokenTaggerTrainConfig:
        variant = normalize_subtoken_part_variant(variant)
        if variant == SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE:
            raise ValueError("Use train_hlt_part_baseline_for_subtoken_comparison for hlt_part_baseline")
        variant_config = build_subtoken_variant_config(variant)
        return SubtokenTaggerTrainConfig(
            output_dir=str(self.child_output_dir(variant)),
            hlt_cache_dir=self.hlt_cache_dir,
            train_split=self.train_split,
            val_split=self.val_split,
            stack_val_split=self.stack_val_split,
            final_test_split=self.final_test_split,
            confirm_split_settings=True,
            confirm_final_test=bool(self.confirm_final_test),
            seed=int(self.seed),
            batch_size=int(self.batch_size),
            eval_batch_size=int(self.eval_batch_size),
            epochs=int(self.epochs),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
            num_workers=int(self.num_workers),
            device=str(self.device),
            amp=bool(self.amp),
            grad_clip_norm=float(self.grad_clip_norm),
            early_stop_patience=int(self.early_stop_patience),
            max_train_batches=self.max_train_batches,
            max_val_batches=self.max_val_batches,
            max_stack_val_batches=self.max_stack_val_batches,
            max_final_test_batches=self.max_final_test_batches,
            max_train_jets=self.max_train_jets,
            max_val_jets=self.max_val_jets,
            max_stack_val_jets=self.max_stack_val_jets,
            max_final_test_jets=self.max_final_test_jets,
            selection_metric=str(self.selection_metric),
            compile_model=bool(self.compile_model),
            verify_hlt_hash=bool(self.verify_hlt_hash),
            num_classes=int(self.resolved_num_classes),
            label_names=tuple(self.resolved_label_names),
            label_filter=tuple(self.resolved_label_filter),
            variant=variant,
            embed_dim=int(self.embed_dim),
            local_layers=int(self.local_layers),
            local_heads=int(self.local_heads),
            context_layers=int(self.context_layers),
            context_heads=int(self.context_heads),
            global_layers=int(self.global_layers),
            global_heads=int(self.global_heads),
            gate_mode=str(variant_config.gate_mode),
            local_pool_mode=str(self.local_pool_mode),
            use_pairwise_bias=bool(variant_config.use_pairwise_bias),
            use_particle_anchor=bool(self.use_particle_anchor),
            use_modality_type_embeddings=bool(self.use_modality_type_embeddings),
            use_pt_rank_embedding=bool(self.use_pt_rank_embedding),
            modality_dropout=float(self.modality_dropout),
            dropout=float(self.dropout),
            attention_dropout=float(self.attention_dropout),
            anchor_source=str(self.anchor_source),
            include_part_style_derived_features=bool(self.include_part_style_derived_features),
        )


def _load_compat_dataset(
    config: SubtokenPartCompatibilityConfig,
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


def load_subtoken_compat_datasets(config: SubtokenPartCompatibilityConfig) -> dict[str, SubtokenHLTJetDataset]:
    """Load each split once so every Step 13 variant consumes identical jets."""

    datasets = {
        "model_train": _load_compat_dataset(config, config.train_split, max_jets=config.max_train_jets),
        "model_val": _load_compat_dataset(config, config.val_split, max_jets=config.max_val_jets),
        "stack_val": _load_compat_dataset(config, config.stack_val_split, max_jets=config.max_stack_val_jets),
    }
    if bool(config.confirm_final_test):
        datasets["final_test"] = _load_compat_dataset(
            config,
            config.final_test_split,
            max_jets=config.max_final_test_jets,
        )
    return datasets


def collate_hlt_part_baseline_batch(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate the shared raw-HLT dataset into Weaver ParticleTransformer inputs."""

    torch = require_torch()
    tokens = np.stack([sample["tokens"] for sample in samples], axis=0).astype(np.float32, copy=False)
    mask = np.stack([sample["mask"] for sample in samples], axis=0).astype(bool, copy=False)
    labels = np.asarray([sample["labels"] for sample in samples], dtype=np.int64)
    indices = np.asarray([sample["indices"] for sample in samples], dtype=np.int64)
    part_inputs = build_particle_transformer_inputs_from_tokens(
        tokens,
        mask,
        labels=labels,
        source_view="fixed_hlt",
    )
    return {
        "points": torch.from_numpy(part_inputs.pf_points).float(),
        "features": torch.from_numpy(part_inputs.pf_features).float(),
        "lorentz_vectors": torch.from_numpy(part_inputs.pf_vectors).float(),
        "mask": torch.from_numpy(part_inputs.pf_mask).bool(),
        "labels": torch.from_numpy(labels).long(),
        "indices": torch.from_numpy(indices).long(),
    }


def make_hlt_part_baseline_loader(
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
        collate_fn=collate_hlt_part_baseline_batch,
        generator=generator,
    )


def _move_hlt_part_batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("points", "features", "lorentz_vectors", "mask", "labels", "indices"):
        value = moved.get(key)
        if hasattr(value, "to"):
            moved[key] = value.to(device, non_blocking=True)
    return moved


def run_hlt_part_baseline_epoch(
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
    label_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run one train/eval epoch for the standard HLT Particle Transformer."""

    torch = require_torch()
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    collected_preds: list[np.ndarray] = []
    collected_labels: list[np.ndarray] = []
    collected_logits: list[np.ndarray] = []
    autocast_enabled = bool(amp and device.type == "cuda")
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = _move_hlt_part_batch_to_device(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                logits = model(batch["points"], batch["features"], batch["lorentz_vectors"], batch["mask"])
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


def hlt_part_baseline_checkpoint_payload(
    model,
    optimizer,
    *,
    epoch: int,
    config: SubtokenPartCompatibilityConfig,
    metrics: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "metrics": dict(metrics),
        "label_names": list(config.resolved_label_names),
        "label_filter": list(config.resolved_label_filter),
        "num_classes": int(config.resolved_num_classes),
        "pf_feature_names": list(PF_FEATURE_NAMES),
        "model_config": getattr(model, "config", {}),
        "source": dict(source),
        "experiment_step": SUBTOKEN_PART_COMPAT_STEP,
        "output_contract": HLT_PART_BASELINE_COMPAT_CONTRACT,
    }


def load_hlt_part_baseline_compat_checkpoint(path: str | Path, *, device=None):
    torch = require_torch()
    payload = torch.load(path, map_location=device or "cpu")
    num_classes = int(payload.get("num_classes", len(payload.get("label_names", LABEL_NAMES))))
    config_payload = payload.get("config", {})
    model_size = str(config_payload.get("baseline_model_size", "base")) if isinstance(config_payload, Mapping) else "base"
    model = build_particle_transformer_classifier(num_classes=num_classes, model_size=model_size)
    model.load_state_dict(payload["model_state_dict"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, payload


def train_hlt_part_baseline_for_subtoken_comparison(
    config: SubtokenPartCompatibilityConfig,
    *,
    train_dataset: SubtokenHLTJetDataset | None = None,
    val_dataset: SubtokenHLTJetDataset | None = None,
    stack_val_dataset: SubtokenHLTJetDataset | None = None,
    final_test_dataset: SubtokenHLTJetDataset | None = None,
) -> dict[str, Any]:
    """Train the standard HLT ParT baseline on the Step 13 shared splits."""

    config.validate_label_metadata()
    torch = require_torch()
    set_training_seed(int(config.seed))
    device = resolve_device(config.device)
    output_dir = config.child_output_dir(SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = train_dataset or _load_compat_dataset(config, config.train_split, max_jets=config.max_train_jets)
    val_dataset = val_dataset or _load_compat_dataset(config, config.val_split, max_jets=config.max_val_jets)
    if train_dataset.tokens.shape[-1] != RAW_TOKEN_DIM or val_dataset.tokens.shape[-1] != RAW_TOKEN_DIM:
        raise ValueError(f"HLT ParT baseline expects raw token dim {RAW_TOKEN_DIM}")
    train_loader = make_hlt_part_baseline_loader(
        train_dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        seed=int(config.seed),
    )
    val_loader = make_hlt_part_baseline_loader(
        val_dataset,
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 1,
    )

    checkpoint_model = build_particle_transformer_classifier(
        num_classes=int(config.resolved_num_classes),
        model_size=str(config.baseline_model_size),
    )
    checkpoint_model = checkpoint_model.to(device)
    train_model = checkpoint_model
    if config.compile_model and hasattr(torch, "compile"):
        train_model = torch.compile(train_model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        checkpoint_model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))
    source = source_metadata()
    run_metadata = {
        "experiment_step": SUBTOKEN_PART_COMPAT_STEP,
        "variant": SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE,
        "output_contract": HLT_PART_BASELINE_COMPAT_CONTRACT,
        "config": asdict(config),
        "model_config": getattr(checkpoint_model, "config", {}),
        "source": source,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "label_names": list(config.resolved_label_names),
        "label_filter": list(config.resolved_label_filter),
        "num_classes": int(config.resolved_num_classes),
        "selection_metric": str(config.selection_metric),
        "input_feature_convention": "raw HLT tokens collated through jetclass_fresh.part_inputs",
        "leakage_rule": (
            "Step 13 HLT ParT baseline consumes the same cached fixed-HLT dataset objects as "
            "the subtoken variants; model_val selects checkpoints, stack_val/final_test are "
            "loaded only after checkpoint selection."
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
        train_metrics = run_hlt_part_baseline_epoch(
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
            label_names=tuple(config.resolved_label_names),
        )
        val_metrics = run_hlt_part_baseline_epoch(
            train_model,
            val_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=config.max_val_batches,
            collect_predictions=_selection_metric_requires_predictions(str(config.selection_metric)),
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
        payload = hlt_part_baseline_checkpoint_payload(
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
        raise FloatingPointError("HLT ParT baseline did not produce a valid model_val checkpoint")

    best_model, _ = load_hlt_part_baseline_compat_checkpoint(output_dir / "best_model_val.pt", device=device)
    best_val_metrics = run_hlt_part_baseline_epoch(
        best_model,
        val_loader,
        device=device,
        criterion=criterion,
        amp=False,
        max_batches=config.max_val_batches,
        collect_predictions=True,
        label_names=tuple(config.resolved_label_names),
    )
    metrics_by_split: dict[str, Mapping[str, Any]] = {"model_val": best_val_metrics}

    stack_val_dataset = stack_val_dataset or _load_compat_dataset(
        config,
        config.stack_val_split,
        max_jets=config.max_stack_val_jets,
    )
    stack_val_loader = make_hlt_part_baseline_loader(
        stack_val_dataset,
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 2,
    )
    stack_val_metrics = run_hlt_part_baseline_epoch(
        best_model,
        stack_val_loader,
        device=device,
        criterion=criterion,
        amp=False,
        max_batches=config.max_stack_val_batches,
        collect_predictions=True,
        label_names=tuple(config.resolved_label_names),
    )
    metrics_by_split["stack_val"] = stack_val_metrics

    final_test_metrics = None
    final_test_metadata = None
    if bool(config.confirm_final_test):
        final_test_dataset = final_test_dataset or _load_compat_dataset(
            config,
            config.final_test_split,
            max_jets=config.max_final_test_jets,
        )
        final_test_loader = make_hlt_part_baseline_loader(
            final_test_dataset,
            batch_size=int(config.eval_batch_size),
            shuffle=False,
            num_workers=int(config.num_workers),
            seed=int(config.seed) + 3,
        )
        final_test_metrics = run_hlt_part_baseline_epoch(
            best_model,
            final_test_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=config.max_final_test_batches,
            collect_predictions=True,
            label_names=tuple(config.resolved_label_names),
        )
        metrics_by_split["final_test"] = final_test_metrics
        final_test_metadata = dict(final_test_dataset.metadata)

    _write_per_class_csv(diagnostics_dir / "per_class_metrics.csv", metrics_by_split)
    _write_summary_csv(diagnostics_dir / "summary_metrics.csv", metrics_by_split)

    report = {
        "experiment_step": SUBTOKEN_PART_COMPAT_STEP,
        "variant": SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE,
        "output_contract": HLT_PART_BASELINE_COMPAT_CONTRACT,
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
        "model_config": getattr(best_model, "config", {}),
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
        "inference_consumes_hlt_only": True,
    }
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    _write_best_metrics_csv(diagnostics_dir / "best_metrics.csv", report)
    return report


def _metrics_for_split(report: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    key = {
        "model_val": "best_model_val_metrics",
        "stack_val": "stack_val_metrics",
        "final_test": "final_test_metrics",
    }[split]
    metrics = report.get(key)
    return metrics if isinstance(metrics, Mapping) else None


def _metric_value(metrics: Mapping[str, Any] | None, metric_name: str) -> float | None:
    if metrics is None:
        return None
    try:
        value = _lookup_selection_metric(metrics, metric_name)
    except KeyError:
        return None
    return None if np.isnan(value) else float(value)


def _flatten_report_row(variant: str, report: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": variant,
        "experiment_step": report.get("experiment_step"),
        "output_contract": report.get("output_contract"),
        "best_epoch": report.get("best_epoch"),
        "epochs_completed": report.get("epochs_completed"),
        "selection_metric": report.get("selection_metric"),
        "selection_metric_direction": report.get("selection_metric_direction"),
        "best_model_selection_metric_value": report.get("best_model_selection_metric_value"),
        "best_model_val_accuracy": report.get("best_model_val_accuracy"),
        "checkpoint": report.get("checkpoint"),
        "run_report": str(Path(str(report.get("checkpoint", ""))).parent / "run_report.json") if report.get("checkpoint") else None,
    }
    for split in ("model_val", "stack_val", "final_test"):
        metrics = _metrics_for_split(report, split)
        row[f"{split}_n_jets"] = metrics.get("n_jets") if isinstance(metrics, Mapping) else None
        row[f"{split}_loss"] = metrics.get("loss") if isinstance(metrics, Mapping) else None
        row[f"{split}_accuracy"] = metrics.get("accuracy") if isinstance(metrics, Mapping) else None
        row[f"{split}_macro_per_class_accuracy"] = (
            metrics.get("macro_per_class_accuracy") if isinstance(metrics, Mapping) else None
        )
        binary = metrics.get("binary_metrics") if isinstance(metrics, Mapping) else None
        if isinstance(binary, Mapping):
            row[f"{split}_auc"] = binary.get("auc")
            row[f"{split}_fpr_at_signal_eff_0p30"] = binary.get("fpr_at_signal_eff_0p30")
            row[f"{split}_fpr_at_signal_eff_0p50"] = binary.get("fpr_at_signal_eff_0p50")
            row[f"{split}_background_rejection_at_signal_eff_0p30"] = binary.get(
                "background_rejection_at_signal_eff_0p30"
            )
            row[f"{split}_background_rejection_at_signal_eff_0p50"] = binary.get(
                "background_rejection_at_signal_eff_0p50"
            )
    return row


def _dataset_signature(dataset: SubtokenHLTJetDataset) -> dict[str, Any]:
    keys = (
        "split",
        "source_view",
        "n_jets",
        "raw_token_dim",
        "max_constits",
        "label_filter",
        "label_names",
        "label_counts",
        "max_jets_limit",
        "hlt_content_hash",
        "jet_identity_hash",
        "hlt_seed",
    )
    return {key: dataset.metadata.get(key) for key in keys}


def _write_comparison_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_subtoken_part_compat_report(
    *,
    config: SubtokenPartCompatibilityConfig,
    child_reports: Mapping[str, Mapping[str, Any]],
    datasets: Mapping[str, SubtokenHLTJetDataset],
) -> dict[str, Any]:
    comparison_rows = [_flatten_report_row(variant, child_reports[variant]) for variant in config.variants]
    comparison_split = "final_test" if bool(config.confirm_final_test) else "stack_val"
    primary_metric = str(config.selection_metric)
    lower_is_better = primary_metric in SUBTOKEN_PART_LOWER_IS_BETTER_SELECTION_METRICS
    scored_rows = []
    for row in comparison_rows:
        variant = str(row["variant"])
        metrics = _metrics_for_split(child_reports[variant], comparison_split)
        value = _metric_value(metrics, primary_metric)
        if value is not None:
            scored_rows.append((float(value), variant))
    best_variant = None
    best_metric_value = None
    if scored_rows:
        best_metric_value, best_variant = min(scored_rows) if lower_is_better else max(scored_rows)
    return {
        "experiment_step": SUBTOKEN_PART_COMPAT_STEP,
        "ok": True,
        "variants": list(config.variants),
        "comparison_split": comparison_split,
        "primary_metric": primary_metric,
        "primary_metric_direction": "minimize" if lower_is_better else "maximize",
        "best_variant": best_variant,
        "best_metric_value": best_metric_value,
        "comparison_rows": comparison_rows,
        "child_reports": {
            variant: str(config.child_output_dir(variant) / "run_report.json")
            for variant in config.variants
        },
        "shared_datasets": {split: _dataset_signature(dataset) for split, dataset in datasets.items()},
        "label_names": list(config.resolved_label_names),
        "label_filter": list(config.resolved_label_filter),
        "num_classes": int(config.resolved_num_classes),
        "config": asdict(config),
        "source": source_metadata(),
        "identical_split_rule": (
            "All Step 13 child runs receive the same in-memory SubtokenHLTJetDataset objects for "
            "model_train/model_val/stack_val and, when enabled, final_test. The HLT ParT baseline "
            "only changes the collate/model path."
        ),
        "inference_consumes_hlt_only": True,
    }


def run_subtoken_part_compat_experiment(config: SubtokenPartCompatibilityConfig) -> dict[str, Any]:
    """Run Step 13 comparison over HLT ParT plus planned subtoken variants."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = load_subtoken_compat_datasets(config)
    save_json(
        output_dir / "config.json",
        {
            "experiment_step": SUBTOKEN_PART_COMPAT_STEP,
            "config": asdict(config),
            "shared_datasets": {split: _dataset_signature(dataset) for split, dataset in datasets.items()},
            "source": source_metadata(),
        },
    )

    child_reports: dict[str, Mapping[str, Any]] = {}
    for variant in config.variants:
        if variant == SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE:
            report = train_hlt_part_baseline_for_subtoken_comparison(
                config,
                train_dataset=datasets["model_train"],
                val_dataset=datasets["model_val"],
                stack_val_dataset=datasets["stack_val"],
                final_test_dataset=datasets.get("final_test"),
            )
        else:
            train_config = config.to_subtoken_train_config(variant)
            report = train_subtoken_tagger(
                train_config,
                train_dataset=datasets["model_train"],
                val_dataset=datasets["model_val"],
                stack_val_dataset=datasets["stack_val"],
                final_test_dataset=datasets.get("final_test"),
            )
        child_reports[variant] = report

    report = build_subtoken_part_compat_report(config=config, child_reports=child_reports, datasets=datasets)
    save_json(output_dir / "run_report.json", report)
    save_json(output_dir / "model_val_report.json", report)
    _write_comparison_csv(output_dir / "diagnostics" / "comparison_metrics.csv", report["comparison_rows"])
    return report


__all__ = [
    "HLT_PART_BASELINE_COMPAT_CONTRACT",
    "SUBTOKEN_PART_COMPAT_STEP",
    "SUBTOKEN_PART_STEP13_VARIANTS",
    "SubtokenPartCompatibilityConfig",
    "build_subtoken_part_compat_report",
    "collate_hlt_part_baseline_batch",
    "load_hlt_part_baseline_compat_checkpoint",
    "load_subtoken_compat_datasets",
    "make_hlt_part_baseline_loader",
    "run_hlt_part_baseline_epoch",
    "run_subtoken_part_compat_experiment",
    "train_hlt_part_baseline_for_subtoken_comparison",
]
