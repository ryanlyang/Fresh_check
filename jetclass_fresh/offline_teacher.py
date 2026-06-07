"""Offline-only Particle Transformer teacher reference for Step 6."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from .hlt_baseline import (
    ParticleViewTorchDataset,
    build_particle_transformer_classifier,
    checkpoint_payload,
    make_data_loader,
    require_torch,
    resolve_device,
    run_epoch,
    save_json,
    set_training_seed,
)
from .jetclass_data import (
    LABEL_NAMES,
    JetView,
    load_offline_view,
    load_split_manifest,
    manifest_hash,
)


@dataclass
class OfflineTeacherTrainConfig:
    """Training configuration for the offline-only teacher reference."""

    output_dir: str
    manifest_path: str
    data_dir: str | None = None
    train_split: str = "model_train"
    val_split: str = "model_val"
    seed: int = 707
    batch_size: int = 128
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
    model_size: str = "base"
    compile_model: bool = False
    verify_label_branches: bool = False
    read_chunk_size: int = 50_000


def train_offline_teacher(
    config: OfflineTeacherTrainConfig,
    *,
    model=None,
    train_view: JetView | None = None,
    val_view: JetView | None = None,
    max_train_jets: int | None = None,
    max_val_jets: int | None = None,
) -> Dict[str, Any]:
    """Train the offline-only Particle Transformer upper-reference model."""

    if config.train_split != "model_train" or config.val_split != "model_val":
        raise ValueError("Step 6 may train only on model_train and select only on model_val")

    torch = require_torch()
    set_training_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = None
    manifest_sha = None
    if train_view is None or val_view is None:
        manifest = load_split_manifest(config.manifest_path)
        manifest_sha = manifest_hash(manifest)
    if train_view is None:
        train_view = load_offline_view(
            manifest,
            config.train_split,
            data_dir=config.data_dir,
            verify_label_branches=config.verify_label_branches,
            read_chunk_size=config.read_chunk_size,
        )
    if val_view is None:
        val_view = load_offline_view(
            manifest,
            config.val_split,
            data_dir=config.data_dir,
            verify_label_branches=config.verify_label_branches,
            read_chunk_size=config.read_chunk_size,
        )
    if manifest_sha is None:
        manifest_sha = train_view.metadata.get("source_manifest_hash")

    train_dataset = ParticleViewTorchDataset(train_view, max_jets=max_train_jets, expected_view="offline")
    val_dataset = ParticleViewTorchDataset(val_view, max_jets=max_val_jets, expected_view="offline")
    train_loader = make_data_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
        source_view="offline",
    )
    val_loader = make_data_loader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed + 1,
        source_view="offline",
    )

    model = model or build_particle_transformer_classifier(num_classes=len(LABEL_NAMES), model_size=config.model_size)
    model = model.to(device)
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))

    run_metadata = {
        "config": asdict(config),
        "manifest_hash": manifest_sha,
        "train_source_view": train_view.metadata.get("view"),
        "val_source_view": val_view.metadata.get("view"),
        "train_n_jets": len(train_dataset),
        "val_n_jets": len(val_dataset),
        "reference_role": "offline_upper_reference_only",
        "leakage_rule": (
            "Offline constituents are intentionally used as inputs for this teacher reference. "
            "Teacher logits/probabilities must not be used as HLT-side fusion features."
        ),
        "no_stack_or_final_test_partitions_loaded": True,
    }
    save_json(output_dir / "config.json", run_metadata)

    curves: List[Dict[str, Any]] = []
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, int(config.epochs) + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            amp=config.amp,
            grad_clip_norm=config.grad_clip_norm,
            max_batches=config.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=config.max_val_batches,
        )
        row = {
            "epoch": int(epoch),
            "train": train_metrics,
            "model_val": val_metrics,
        }
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves})

        improved = (
            val_metrics["accuracy"] > best_val_accuracy
            or (
                np.isclose(val_metrics["accuracy"], best_val_accuracy)
                and val_metrics["loss"] < best_val_loss
            )
        )
        torch.save(
            checkpoint_payload(
                model,
                optimizer,
                epoch=epoch,
                config=config,
                metrics=row,
                experiment_step="step6_offline_teacher_reference",
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
                    experiment_step="step6_offline_teacher_reference",
                ),
                output_dir / "best_model_val.pt",
            )
        else:
            epochs_without_improvement += 1

        if config.early_stop_patience >= 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    report = {
        "experiment_step": "step6_offline_teacher_reference",
        "reference_role": "offline_upper_reference_only",
        "best_epoch": int(best_epoch),
        "best_model_val_accuracy": float(best_val_accuracy),
        "best_model_val_loss": float(best_val_loss),
        "epochs_completed": len(curves),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "no_final_test_evaluation": True,
        "not_allowed_for_fusion_features": True,
    }
    save_json(output_dir / "model_val_report.json", report)
    return report
