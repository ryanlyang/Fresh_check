"""Offline teacher training/registration for the cross-architecture experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.heterogeneous_hlt import (
    balanced_limit_jet_view,
    build_heterogeneous_hlt_classifier,
    normalize_architecture_name,
)
from jetclass_fresh.hlt_baseline import (
    ParticleViewTorchDataset,
    checkpoint_payload,
    make_data_loader,
    require_torch,
    resolve_device,
    run_epoch,
    save_json,
    set_training_seed,
)
from jetclass_fresh.jetclass_data import (
    LABEL_NAMES,
    JetView,
    load_offline_view,
    load_split_manifest,
    manifest_hash,
)

from .crossarch_experiment import EXPERIMENT_NAME, TEACHER_ARCHITECTURES, CrossArchExperimentLayout


EXPERIMENT_STEP = "crossarch_step3_offline_teacher"
TRAIN_EXPERIMENT_STEP = f"{EXPERIMENT_STEP}:train"
REGISTER_EXPERIMENT_STEP = f"{EXPERIMENT_STEP}:register"


@dataclass
class CrossArchOfflineTeacherTrainConfig:
    """Training configuration for one cross-arch frozen offline teacher."""

    architecture: str
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
    max_train_jets: int | None = None
    max_val_jets: int | None = None
    model_size: str = "base"
    compile_model: bool = False
    verify_label_branches: bool = False
    read_chunk_size: int = 50_000

    def __post_init__(self) -> None:
        self.architecture = normalize_crossarch_teacher_architecture(self.architecture)
        if self.train_split != "model_train" or self.val_split != "model_val":
            raise ValueError("CrossArch offline teachers may train only on model_train and select only on model_val")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.epochs) <= 0:
            raise ValueError("epochs must be positive")
        if self.model_size not in {"base", "tiny"}:
            raise ValueError("model_size must be 'base' or 'tiny'")


def normalize_crossarch_teacher_architecture(architecture: str) -> str:
    arch = normalize_architecture_name(architecture)
    if arch not in TEACHER_ARCHITECTURES:
        raise ValueError(f"Unknown crossarch offline teacher architecture {architecture!r}")
    return arch


def crossarch_offline_teacher_dir(
    architecture: str,
    *,
    output_root: str | Path = "checkpoints",
) -> Path:
    layout = CrossArchExperimentLayout(output_root=output_root)
    return layout.offline_teacher_dir(normalize_crossarch_teacher_architecture(architecture))


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def maybe_load_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    try:
        torch = require_torch()
        payload = torch.load(Path(path), map_location="cpu")
    except Exception as exc:  # pragma: no cover - depends on file/env
        return {"checkpoint_metadata_loaded": False, "checkpoint_metadata_error": str(exc)}
    if not isinstance(payload, Mapping):
        return {"checkpoint_metadata_loaded": False, "checkpoint_metadata_error": "checkpoint payload is not a mapping"}
    metrics = dict(payload.get("metrics") or {})
    model_val = dict(metrics.get("model_val") or {})
    return {
        "checkpoint_metadata_loaded": True,
        "epoch": payload.get("epoch"),
        "experiment_step": payload.get("experiment_step"),
        "model_config": dict(payload.get("model_config") or {}),
        "config": dict(payload.get("config") or {}),
        "label_names": list(payload.get("label_names") or []),
        "model_val_accuracy": model_val.get("accuracy"),
        "model_val_loss": model_val.get("loss"),
    }


def _load_offline_train_val_views(config: CrossArchOfflineTeacherTrainConfig) -> tuple[JetView, JetView, str]:
    manifest = load_split_manifest(config.manifest_path)
    manifest_sha = manifest_hash(manifest)
    train_view = load_offline_view(
        manifest,
        config.train_split,
        data_dir=config.data_dir,
        verify_label_branches=config.verify_label_branches,
        read_chunk_size=config.read_chunk_size,
    )
    val_view = load_offline_view(
        manifest,
        config.val_split,
        data_dir=config.data_dir,
        verify_label_branches=config.verify_label_branches,
        read_chunk_size=config.read_chunk_size,
    )
    return train_view, val_view, manifest_sha


def _source_metadata(
    config: CrossArchOfflineTeacherTrainConfig,
    *,
    manifest_sha: str | None,
    train_view: JetView | None = None,
    val_view: JetView | None = None,
    subset_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_step": TRAIN_EXPERIMENT_STEP,
        "architecture": config.architecture,
        "source_type": "trained_offline_view",
        "manifest_path": config.manifest_path,
        "manifest_hash": manifest_sha,
        "train_split": config.train_split,
        "val_split": config.val_split,
        "train_source_view": None if train_view is None else train_view.metadata.get("view"),
        "val_source_view": None if val_view is None else val_view.metadata.get("view"),
        "train_n_jets": None if train_view is None else int(len(train_view.labels)),
        "val_n_jets": None if val_view is None else int(len(val_view.labels)),
        "subset_selection": dict(subset_selection or {}),
        "allowed_downstream_use": "frozen_offline_teacher_in_teacher_logit_reconstructor_path",
        "not_direct_hlt_baseline": True,
        "no_stack_or_final_test_partitions_loaded": True,
    }


def train_crossarch_offline_teacher(
    config: CrossArchOfflineTeacherTrainConfig,
    *,
    model=None,
    train_view: JetView | None = None,
    val_view: JetView | None = None,
) -> dict[str, Any]:
    """Train one offline teacher architecture on offline model_train/model_val."""

    torch = require_torch()
    set_training_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_sha = None
    if train_view is None or val_view is None:
        train_view, val_view, manifest_sha = _load_offline_train_val_views(config)
    if manifest_sha is None:
        manifest_sha = train_view.metadata.get("source_manifest_hash")

    train_view, train_selection = balanced_limit_jet_view(
        train_view,
        config.max_train_jets,
        seed=int(config.seed),
    )
    val_view, val_selection = balanced_limit_jet_view(
        val_view,
        config.max_val_jets,
        seed=int(config.seed) + 1,
    )
    train_dataset = ParticleViewTorchDataset(train_view, expected_view="offline")
    val_dataset = ParticleViewTorchDataset(val_view, expected_view="offline")
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

    model = model or build_heterogeneous_hlt_classifier(
        config.architecture,
        num_classes=len(LABEL_NAMES),
        model_size=config.model_size,
    )
    model = model.to(device)
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))

    subset_selection = {
        "model_train": train_selection,
        "model_val": val_selection,
    }
    source_metadata = _source_metadata(
        config,
        manifest_sha=manifest_sha,
        train_view=train_view,
        val_view=val_view,
        subset_selection=subset_selection,
    )
    run_metadata = {
        "config": asdict(config),
        "source_metadata": source_metadata,
        "model_config": getattr(model, "config", {}),
    }
    save_json(output_dir / "config.json", run_metadata)
    save_json(output_dir / "source_metadata.json", source_metadata)

    curves: list[dict[str, Any]] = []
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
        "architecture": config.architecture,
        "best_epoch": int(best_epoch),
        "best_model_val_accuracy": float(best_val_accuracy),
        "best_model_val_loss": float(best_val_loss),
        "epochs_completed": len(curves),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "source_metadata_path": str(output_dir / "source_metadata.json"),
        "no_final_test_evaluation": True,
        "allowed_downstream_use": "frozen_offline_teacher_in_teacher_logit_reconstructor_path",
        "not_direct_hlt_baseline": True,
    }
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    return report


def register_crossarch_offline_teacher_checkpoint(
    *,
    architecture: str,
    source_checkpoint: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    source_report: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Register an already-trained offline teacher checkpoint into the fresh namespace."""

    arch = normalize_crossarch_teacher_architecture(architecture)
    source_checkpoint = Path(source_checkpoint)
    output_dir = Path(output_dir)
    target_checkpoint = output_dir / "best_model_val.pt"
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source_checkpoint}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if target_checkpoint.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite registered checkpoint: {target_checkpoint}")

    shutil.copy2(source_checkpoint, target_checkpoint)
    source_sha = sha256_file(source_checkpoint)
    target_sha = sha256_file(target_checkpoint)
    checkpoint_metadata = maybe_load_checkpoint_metadata(target_checkpoint)

    manifest_sha = None
    if manifest_path is not None and Path(manifest_path).exists():
        manifest_sha = manifest_hash(load_split_manifest(manifest_path))

    registered_source_report = None
    source_report_sha = None
    if source_report is not None and Path(source_report).exists():
        registered_source_report = output_dir / "registered_source_report.json"
        shutil.copy2(source_report, registered_source_report)
        source_report_sha = sha256_file(source_report)

    source_metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_step": REGISTER_EXPERIMENT_STEP,
        "architecture": arch,
        "source_type": "registered_existing_checkpoint",
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha,
        "registered_checkpoint": str(target_checkpoint),
        "registered_checkpoint_sha256": target_sha,
        "source_report": None if source_report is None else str(source_report),
        "source_report_sha256": source_report_sha,
        "registered_source_report": None if registered_source_report is None else str(registered_source_report),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "manifest_hash": manifest_sha,
        "allowed_downstream_use": "frozen_offline_teacher_in_teacher_logit_reconstructor_path",
        "not_direct_hlt_baseline": True,
        "requires_user_trust_that_source_used_compatible_offline_model_train_model_val": True,
        "checkpoint_metadata": checkpoint_metadata,
    }
    save_json(output_dir / "source_metadata.json", source_metadata)
    save_json(
        output_dir / "config.json",
        {
            "registration": True,
            "architecture": arch,
            "source_metadata": source_metadata,
        },
    )

    report = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_step": REGISTER_EXPERIMENT_STEP,
        "architecture": arch,
        "registration": True,
        "checkpoint": str(target_checkpoint),
        "checkpoint_sha256": target_sha,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha,
        "source_metadata_path": str(output_dir / "source_metadata.json"),
        "best_epoch": checkpoint_metadata.get("epoch"),
        "best_model_val_accuracy": checkpoint_metadata.get("model_val_accuracy"),
        "best_model_val_loss": checkpoint_metadata.get("model_val_loss"),
        "no_final_test_evaluation": True,
        "allowed_downstream_use": "frozen_offline_teacher_in_teacher_logit_reconstructor_path",
        "not_direct_hlt_baseline": True,
    }
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    save_json(output_dir / "registration_report.json", report)
    return report


__all__ = [
    "EXPERIMENT_STEP",
    "REGISTER_EXPERIMENT_STEP",
    "TRAIN_EXPERIMENT_STEP",
    "CrossArchOfflineTeacherTrainConfig",
    "crossarch_offline_teacher_dir",
    "maybe_load_checkpoint_metadata",
    "normalize_crossarch_teacher_architecture",
    "register_crossarch_offline_teacher_checkpoint",
    "sha256_file",
    "train_crossarch_offline_teacher",
]
