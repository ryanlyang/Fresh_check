"""Reco-domain taggers trained behind frozen cross-architecture reconstructors.

This module implements the follow-up strategy:

    fixed HLT -> frozen teacher-logit reconstructor -> trainable tagger

The reconstructor is never updated here.  The new tagger is trained on
``model_train`` and selected on ``model_val`` using the synthetic reconstructed
view produced from cached fixed-HLT inputs.  Prediction collection writes the
same fusion-compatible prediction blocks as the rest of the crossarch workflow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.dual_view import build_part_inputs_torch
from jetclass_fresh.fusion import PredictionBlock, STACK_SPLITS, prediction_paths, save_prediction_block, softmax_np
from jetclass_fresh.heterogeneous_hlt import (
    balanced_limit_jet_view,
    build_heterogeneous_hlt_classifier,
    load_heterogeneous_hlt_model_from_checkpoint,
)
from jetclass_fresh.hlt_baseline import checkpoint_payload, require_torch, resolve_device, save_json, set_training_seed
from jetclass_fresh.hlt_cache import load_cached_hlt_view
from jetclass_fresh.jetclass_data import LABEL_NAMES, JetIdentity, JetView

from .crossarch_experiment import (
    EXPERIMENT_NAME,
    RECONSTRUCTOR_IMPLEMENTATIONS,
    SPLIT_SIZES,
    TEACHER_ARCHITECTURES,
    normalize_reconstructor_architecture,
    normalize_teacher_architecture,
    reco_model_name,
)
from .crossarch_offline_teachers import sha256_file
from .reconstructor_builders import load_teacher_logit_reconstructor_checkpoint


EXPERIMENT_STEP = "crossarch_reco_domain_taggers"
TRAIN_EXPERIMENT_STEP = f"{EXPERIMENT_STEP}:train"
PREDICT_EXPERIMENT_STEP = f"{EXPERIMENT_STEP}:predict"
PREDICTION_SPLITS = tuple(STACK_SPLITS)


def reco_domain_tagger_model_name(reco_architecture: str, teacher_architecture: str) -> str:
    """Distinct fusion source name for an adapted tagger behind a frozen reco."""

    reco = normalize_reconstructor_architecture(reco_architecture)
    teacher = normalize_teacher_architecture(teacher_architecture)
    return f"{reco}_reco_to_{teacher}_adapted_tagger"


@dataclass
class CrossArchRecoDomainTaggerTrainConfig:
    """Train one tagger architecture on synthetic views from one frozen reco."""

    reco_architecture: str
    teacher_architecture: str
    reconstructor_checkpoint: str
    output_dir: str
    cache_dir: str
    train_split: str = "model_train"
    val_split: str = "model_val"
    seed: int = 2205
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 5
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_train_jets: int | None = SPLIT_SIZES["model_train"]
    max_val_jets: int | None = SPLIT_SIZES["model_val"]
    model_size: str = "base"
    compile_model: bool = False
    max_constits: int = 128
    teacher_weight_threshold: float = 0.0
    strict_reconstructor_checkpoint: bool = True

    def __post_init__(self) -> None:
        self.reco_architecture = normalize_reconstructor_architecture(self.reco_architecture)
        self.teacher_architecture = normalize_teacher_architecture(self.teacher_architecture)
        if self.train_split != "model_train" or self.val_split != "model_val":
            raise ValueError("Reco-domain taggers may train only on model_train and select only on model_val")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.epochs) <= 0:
            raise ValueError("epochs must be positive")
        if self.teacher_architecture not in TEACHER_ARCHITECTURES:
            raise ValueError(f"Unknown tagger/teacher architecture: {self.teacher_architecture}")
        if self.model_size not in {"base", "tiny"}:
            raise ValueError("model_size must be 'base' or 'tiny'")
        if int(self.max_constits) <= 0:
            raise ValueError("max_constits must be positive")

    @property
    def model_name(self) -> str:
        return reco_domain_tagger_model_name(self.reco_architecture, self.teacher_architecture)


@dataclass
class CrossArchRecoDomainTaggerPredictionConfig:
    """Collect prediction blocks for one trained reco-domain tagger."""

    reco_architecture: str
    teacher_architecture: str
    reconstructor_checkpoint: str
    tagger_checkpoint: str
    cache_dir: str
    prediction_dir: str
    output_dir: str
    model_name: str | None = None
    splits: list[str] = field(default_factory=lambda: list(PREDICTION_SPLITS))
    batch_size: int = 128
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    stack_train_size: int | None = SPLIT_SIZES["stack_train"]
    stack_val_size: int | None = SPLIT_SIZES["stack_val"]
    final_test_size: int | None = SPLIT_SIZES["final_test"]
    max_jets_per_split: int | None = None
    overwrite_predictions: bool = False
    skip_existing_predictions: bool = True
    confirm_final_test: bool = False
    control_seed: int = 12345
    max_constits: int = 128
    teacher_weight_threshold: float = 0.0
    strict_reconstructor_checkpoint: bool = True

    def __post_init__(self) -> None:
        self.reco_architecture = normalize_reconstructor_architecture(self.reco_architecture)
        self.teacher_architecture = normalize_teacher_architecture(self.teacher_architecture)
        unknown = sorted(set(self.splits) - set(PREDICTION_SPLITS))
        if unknown:
            raise ValueError(f"Unknown prediction splits: {unknown}")
        if "final_test" in self.splits and not self.confirm_final_test:
            raise ValueError("Refusing to evaluate final_test without confirm_final_test=True")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.max_constits) <= 0:
            raise ValueError("max_constits must be positive")

    @property
    def resolved_model_name(self) -> str:
        return self.model_name or reco_domain_tagger_model_name(self.reco_architecture, self.teacher_architecture)


class HLTRecoDomainDataset:
    """Dataset over cached fixed-HLT tokens, preserving labels and jet IDs."""

    def __init__(self, view: JetView) -> None:
        require_torch()
        self.tokens = np.asarray(view.tokens, dtype=np.float32)
        self.mask = np.asarray(view.mask, dtype=bool)
        self.labels = np.asarray(view.labels, dtype=np.int64)
        self.jet_ids = list(view.jet_ids)
        self.split = view.split
        self.metadata = dict(view.metadata)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        return self.tokens[index], self.mask[index], self.labels[index], self.jet_ids[index]


def collate_reco_domain_batch(samples):
    torch = require_torch()
    return {
        "hlt_tokens": torch.from_numpy(np.stack([row[0] for row in samples], axis=0)).float(),
        "hlt_mask": torch.from_numpy(np.stack([row[1] for row in samples], axis=0)).bool(),
        "labels": torch.as_tensor([row[2] for row in samples], dtype=torch.long),
        "jet_ids": [row[3] for row in samples],
    }


def make_reco_domain_loader(
    dataset: HLTRecoDomainDataset,
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
        collate_fn=collate_reco_domain_batch,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _freeze_module(module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _build_reco_domain_inputs(
    reconstructor,
    batch: Mapping[str, Any],
    *,
    device,
    split: str,
    amp: bool,
    max_constits: int,
    teacher_weight_threshold: float,
):
    torch = require_torch()
    hlt_tokens = batch["hlt_tokens"].to(device=device, non_blocking=True)
    hlt_mask = batch["hlt_mask"].to(device=device, non_blocking=True)
    labels = batch["labels"].to(device=device, non_blocking=True)
    autocast_enabled = bool(amp and getattr(device, "type", None) == "cuda")
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=autocast_enabled):
        reco_view = reconstructor(
            hlt_tokens,
            hlt_mask,
            labels=labels,
            jet_ids=batch["jet_ids"],
            split=split,
        )
    return build_part_inputs_torch(
        reco_view.tokens,
        reco_view.mask,
        weights=reco_view.weights,
        max_constits=int(max_constits),
        weight_threshold=float(teacher_weight_threshold),
    ), labels


def run_reco_domain_tagger_epoch(
    model,
    reconstructor,
    loader,
    *,
    device,
    criterion,
    optimizer=None,
    scaler=None,
    amp: bool = False,
    grad_clip_norm: float = 0.0,
    max_batches: int | None = None,
    max_constits: int = 128,
    teacher_weight_threshold: float = 0.0,
) -> dict[str, float]:
    """Train/evaluate the tagger while the reconstructor remains frozen."""

    torch = require_torch()
    is_train = optimizer is not None
    model.train(is_train)
    reconstructor.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    autocast_enabled = bool(amp and getattr(device, "type", None) == "cuda")
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            inputs, labels = _build_reco_domain_inputs(
                reconstructor,
                batch,
                device=device,
                split=loader.dataset.split,
                amp=amp,
                max_constits=max_constits,
                teacher_weight_threshold=teacher_weight_threshold,
            )
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                logits = model(inputs["points"], inputs["features"], inputs["lorentz_vectors"], inputs["mask"])
                loss = criterion(logits, labels)

            if is_train:
                if scaler is not None and autocast_enabled:
                    scaler.scale(loss).backward()
                    if grad_clip_norm and grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm and grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                    optimizer.step()

            batch_size = int(labels.numel())
            total_loss += float(loss.detach().item()) * batch_size
            total_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
            total_seen += batch_size

    if total_seen == 0:
        return {"loss": float("nan"), "accuracy": 0.0, "n_jets": 0}
    return {
        "loss": total_loss / float(total_seen),
        "accuracy": total_correct / float(total_seen),
        "n_jets": int(total_seen),
    }


def _source_metadata(
    config: CrossArchRecoDomainTaggerTrainConfig,
    *,
    train_view: JetView,
    val_view: JetView,
    subset_selection: Mapping[str, Any],
    reconstructor_payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_step": TRAIN_EXPERIMENT_STEP,
        "model_name": config.model_name,
        "source_kind": "reco_domain_tagger",
        "tagger_architecture": config.teacher_architecture,
        "reco_architecture": config.reco_architecture,
        "source_reco_model_name": reco_model_name(config.reco_architecture, config.teacher_architecture),
        "source_reconstructor_checkpoint": config.reconstructor_checkpoint,
        "source_reconstructor_checkpoint_sha256": sha256_file(config.reconstructor_checkpoint),
        "source_reconstructor_epoch": reconstructor_payload.get("epoch"),
        "source_reconstructor_experiment_step": reconstructor_payload.get("experiment_step"),
        "cache_dir": config.cache_dir,
        "train_split": config.train_split,
        "val_split": config.val_split,
        "train_hlt_content_hash": train_view.metadata.get("hlt_content_hash"),
        "val_hlt_content_hash": val_view.metadata.get("hlt_content_hash"),
        "train_n_jets": int(len(train_view.labels)),
        "val_n_jets": int(len(val_view.labels)),
        "subset_selection": dict(subset_selection),
        "allowed_inputs": "cached_fixed_hlt_to_frozen_reconstructor_to_trainable_tagger",
        "no_offline_inputs_loaded": True,
        "no_stack_or_final_test_partitions_loaded": True,
    }


def train_crossarch_reco_domain_tagger(
    config: CrossArchRecoDomainTaggerTrainConfig,
    *,
    reconstructor=None,
    model=None,
    train_view: JetView | None = None,
    val_view: JetView | None = None,
) -> dict[str, Any]:
    """Train one reco-domain tagger on model_train/model_val only."""

    torch = require_torch()
    set_training_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_view = train_view or load_cached_hlt_view(config.cache_dir, config.train_split)
    val_view = val_view or load_cached_hlt_view(config.cache_dir, config.val_split)
    train_view, train_selection = balanced_limit_jet_view(train_view, config.max_train_jets, seed=int(config.seed))
    val_view, val_selection = balanced_limit_jet_view(val_view, config.max_val_jets, seed=int(config.seed) + 1)
    train_dataset = HLTRecoDomainDataset(train_view)
    val_dataset = HLTRecoDomainDataset(val_view)
    train_loader = make_reco_domain_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    val_loader = make_reco_domain_loader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed + 1,
    )

    if reconstructor is None:
        reconstructor, reconstructor_payload = load_teacher_logit_reconstructor_checkpoint(
            config.reconstructor_checkpoint,
            device=device,
            strict=bool(config.strict_reconstructor_checkpoint),
            expected_architecture=RECONSTRUCTOR_IMPLEMENTATIONS[config.reco_architecture],
        )
    else:
        reconstructor_payload = {}
    _freeze_module(reconstructor)

    model = model or build_heterogeneous_hlt_classifier(
        config.teacher_architecture,
        num_classes=len(LABEL_NAMES),
        model_size=config.model_size,
    )
    model = model.to(device)
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))
    subset_selection = {"model_train": train_selection, "model_val": val_selection}
    source_metadata = _source_metadata(
        config,
        train_view=train_view,
        val_view=val_view,
        subset_selection=subset_selection,
        reconstructor_payload=reconstructor_payload,
    )
    save_json(
        output_dir / "config.json",
        {
            "config": asdict(config),
            "source_metadata": source_metadata,
            "model_config": getattr(model, "config", {}),
        },
    )
    save_json(output_dir / "source_metadata.json", source_metadata)

    curves: list[dict[str, Any]] = []
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    for epoch in range(1, int(config.epochs) + 1):
        train_metrics = run_reco_domain_tagger_epoch(
            model,
            reconstructor,
            train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            amp=config.amp,
            grad_clip_norm=config.grad_clip_norm,
            max_batches=config.max_train_batches,
            max_constits=config.max_constits,
            teacher_weight_threshold=config.teacher_weight_threshold,
        )
        val_metrics = run_reco_domain_tagger_epoch(
            model,
            reconstructor,
            val_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=config.max_val_batches,
            max_constits=config.max_constits,
            teacher_weight_threshold=config.teacher_weight_threshold,
        )
        row = {"epoch": int(epoch), "train": train_metrics, "model_val": val_metrics}
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves})

        improved = (
            val_metrics["accuracy"] > best_val_accuracy
            or (np.isclose(val_metrics["accuracy"], best_val_accuracy) and val_metrics["loss"] < best_val_loss)
        )
        torch.save(
            checkpoint_payload(
                model,
                optimizer,
                epoch=epoch,
                config=config,
                metrics=row,
                experiment_step=TRAIN_EXPERIMENT_STEP,
            ),
            output_dir / "last.pt",
        )
        if improved:
            best_val_accuracy = float(val_metrics["accuracy"])
            best_val_loss = float(val_metrics["loss"])
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(
                checkpoint_payload(
                    model,
                    optimizer,
                    epoch=epoch,
                    config=config,
                    metrics=row,
                    experiment_step=TRAIN_EXPERIMENT_STEP,
                ),
                output_dir / "best_model_val.pt",
            )
        else:
            epochs_without_improvement += 1
        if config.early_stop_patience >= 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    report = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_step": TRAIN_EXPERIMENT_STEP,
        "model_name": config.model_name,
        "reco_architecture": config.reco_architecture,
        "tagger_architecture": config.teacher_architecture,
        "source_reco_model_name": reco_model_name(config.reco_architecture, config.teacher_architecture),
        "best_epoch": int(best_epoch),
        "best_model_val_accuracy": float(best_val_accuracy),
        "best_model_val_loss": float(best_val_loss),
        "epochs_completed": len(curves),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "source_metadata_path": str(output_dir / "source_metadata.json"),
        "no_final_test_evaluation": True,
        "allowed_inputs": "cached_fixed_hlt_to_frozen_reconstructor_to_trainable_tagger",
    }
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    return report


def split_size_for_reco_domain_prediction(
    config: CrossArchRecoDomainTaggerPredictionConfig,
    split: str,
) -> int | None:
    if config.max_jets_per_split is not None:
        return int(config.max_jets_per_split)
    if split == "stack_train":
        return config.stack_train_size
    if split == "stack_val":
        return config.stack_val_size
    if split == "final_test":
        return config.final_test_size
    return None


def evaluate_reco_domain_tagger_model(
    model_name: str,
    model,
    reconstructor,
    view: JetView,
    *,
    reco_architecture: str,
    tagger_architecture: str,
    batch_size: int,
    num_workers: int,
    device,
    amp: bool,
    max_jets: int | None,
    selection_seed: int,
    max_constits: int,
    teacher_weight_threshold: float,
    checkpoint_metadata: Mapping[str, Any] | None = None,
) -> PredictionBlock:
    torch = require_torch()
    view, selection_report = balanced_limit_jet_view(view, max_jets, seed=int(selection_seed))
    dataset = HLTRecoDomainDataset(view)
    loader = make_reco_domain_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=selection_seed,
    )
    model.eval()
    reconstructor.eval()
    logits_rows: list[np.ndarray] = []
    labels_rows: list[np.ndarray] = []
    jet_ids: list[JetIdentity] = []
    autocast_enabled = bool(amp and getattr(device, "type", None) == "cuda")
    with torch.no_grad():
        for batch in loader:
            inputs, labels = _build_reco_domain_inputs(
                reconstructor,
                batch,
                device=device,
                split=view.split,
                amp=amp,
                max_constits=max_constits,
                teacher_weight_threshold=teacher_weight_threshold,
            )
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                logits = model(inputs["points"], inputs["features"], inputs["lorentz_vectors"], inputs["mask"])
            logits_rows.append(logits.detach().cpu().numpy().astype(np.float32))
            labels_rows.append(labels.detach().cpu().numpy().astype(np.int64))
            jet_ids.extend(batch["jet_ids"])
    if not logits_rows:
        raise ValueError(f"No predictions were produced for {model_name}/{view.split}")
    logits_np = np.concatenate(logits_rows, axis=0)
    labels_np = np.concatenate(labels_rows, axis=0)
    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_step": PREDICT_EXPERIMENT_STEP,
        "training_step": TRAIN_EXPERIMENT_STEP,
        "source_kind": "reco_domain_tagger",
        "model_kind": "frozen_reco_then_adapted_tagger",
        "model_name": model_name,
        "reco_architecture": normalize_reconstructor_architecture(reco_architecture),
        "tagger_architecture": normalize_teacher_architecture(tagger_architecture),
        "source_reco_model_name": reco_model_name(reco_architecture, tagger_architecture),
        "hlt_content_hash": view.metadata.get("hlt_content_hash"),
        "allowed_inputs": "cached_fixed_hlt_to_frozen_reconstructor_to_reco_domain_tagger",
        "subset_selection": selection_report,
    }
    metadata.update(dict(checkpoint_metadata or {}))
    return PredictionBlock(
        model_name=model_name,
        split=view.split,
        logits=logits_np,
        probs=softmax_np(logits_np),
        labels=labels_np,
        jet_ids=jet_ids,
        metadata=metadata,
    )


def collect_crossarch_reco_domain_tagger_predictions(
    config: CrossArchRecoDomainTaggerPredictionConfig,
) -> dict[str, Any]:
    """Write fusion-compatible prediction blocks for one adapted tagger source."""

    torch = require_torch()
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    prediction_dir = Path(config.prediction_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "prediction_config.json", {"config": asdict(config)})

    model_name = config.resolved_model_name
    reconstructor, reconstructor_payload = load_teacher_logit_reconstructor_checkpoint(
        config.reconstructor_checkpoint,
        device=device,
        strict=bool(config.strict_reconstructor_checkpoint),
        expected_architecture=RECONSTRUCTOR_IMPLEMENTATIONS[config.reco_architecture],
    )
    _freeze_module(reconstructor)
    model, tagger_payload = load_heterogeneous_hlt_model_from_checkpoint(config.tagger_checkpoint, device=device)
    reports: dict[str, Any] = {}
    for split in config.splits:
        npz_path, _ = prediction_paths(prediction_dir, model_name, split)
        if npz_path.exists() and config.skip_existing_predictions and not config.overwrite_predictions:
            from jetclass_fresh.fusion import load_prediction_block

            reports[split] = load_prediction_block(prediction_dir, model_name, split).metadata
            continue
        view = load_cached_hlt_view(config.cache_dir, split)
        split_index = PREDICTION_SPLITS.index(split)
        block = evaluate_reco_domain_tagger_model(
            model_name,
            model,
            reconstructor,
            view,
            reco_architecture=config.reco_architecture,
            tagger_architecture=config.teacher_architecture,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            device=device,
            amp=config.amp,
            max_jets=split_size_for_reco_domain_prediction(config, split),
            selection_seed=int(config.control_seed) + 2003 * (split_index + 1),
            max_constits=config.max_constits,
            teacher_weight_threshold=config.teacher_weight_threshold,
            checkpoint_metadata={
                "reconstructor_checkpoint": config.reconstructor_checkpoint,
                "reconstructor_checkpoint_sha256": sha256_file(config.reconstructor_checkpoint),
                "reconstructor_checkpoint_epoch": reconstructor_payload.get("epoch"),
                "tagger_checkpoint": config.tagger_checkpoint,
                "tagger_checkpoint_sha256": sha256_file(config.tagger_checkpoint),
                "tagger_checkpoint_epoch": tagger_payload.get("epoch"),
                "tagger_checkpoint_experiment_step": tagger_payload.get("experiment_step"),
                "split_expected_size": split_size_for_reco_domain_prediction(config, split),
            },
        )
        reports[split] = save_prediction_block(block, prediction_dir, overwrite=config.overwrite_predictions)

    del model, reconstructor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    report = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_step": PREDICT_EXPERIMENT_STEP,
        "model_name": model_name,
        "reco_architecture": config.reco_architecture,
        "tagger_architecture": config.teacher_architecture,
        "source_reco_model_name": reco_model_name(config.reco_architecture, config.teacher_architecture),
        "reconstructor_checkpoint": config.reconstructor_checkpoint,
        "tagger_checkpoint": config.tagger_checkpoint,
        "prediction_dir": str(prediction_dir / model_name),
        "splits": list(config.splits),
        "split_reports": reports,
    }
    save_json(output_dir / "prediction_collection_report.json", report)
    return report


__all__ = [
    "CrossArchRecoDomainTaggerPredictionConfig",
    "CrossArchRecoDomainTaggerTrainConfig",
    "PREDICT_EXPERIMENT_STEP",
    "PREDICTION_SPLITS",
    "TRAIN_EXPERIMENT_STEP",
    "collect_crossarch_reco_domain_tagger_predictions",
    "evaluate_reco_domain_tagger_model",
    "reco_domain_tagger_model_name",
    "run_reco_domain_tagger_epoch",
    "split_size_for_reco_domain_prediction",
    "train_crossarch_reco_domain_tagger",
]
