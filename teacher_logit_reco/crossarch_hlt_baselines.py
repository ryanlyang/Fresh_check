"""Direct fixed-HLT baselines for the cross-architecture experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.fusion import PredictionBlock, STACK_SPLITS, prediction_paths, save_prediction_block
from jetclass_fresh.heterogeneous_hlt import (
    balanced_limit_jet_view,
    build_heterogeneous_hlt_classifier,
    evaluate_heterogeneous_hlt_model,
    load_heterogeneous_hlt_model_from_checkpoint,
    normalize_architecture_name,
)
from jetclass_fresh.hlt_baseline import (
    JetViewTorchDataset,
    checkpoint_payload,
    make_data_loader,
    require_torch,
    resolve_device,
    run_epoch,
    save_json,
    set_training_seed,
)
from jetclass_fresh.hlt_cache import load_cached_hlt_view
from jetclass_fresh.jetclass_data import LABEL_NAMES, JetView

from .crossarch_experiment import (
    DIRECT_HLT_ARCHITECTURES,
    EXPERIMENT_NAME,
    SPLIT_SIZES,
    CrossArchExperimentLayout,
    hlt_model_name,
)
from .crossarch_offline_teachers import sha256_file


EXPERIMENT_STEP = "crossarch_step4_direct_hlt_baseline"
TRAIN_EXPERIMENT_STEP = f"{EXPERIMENT_STEP}:train"
PREDICT_EXPERIMENT_STEP = f"{EXPERIMENT_STEP}:predict"
PREDICTION_SPLITS = tuple(STACK_SPLITS)


@dataclass
class CrossArchHLTBaselineTrainConfig:
    """Training configuration for one direct fixed-HLT baseline."""

    architecture: str
    output_dir: str
    cache_dir: str
    train_split: str = "model_train"
    val_split: str = "model_val"
    seed: int = 101
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

    def __post_init__(self) -> None:
        self.architecture = normalize_crossarch_hlt_architecture(self.architecture)
        if self.train_split != "model_train" or self.val_split != "model_val":
            raise ValueError("CrossArch HLT baselines may train only on model_train and select only on model_val")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.epochs) <= 0:
            raise ValueError("epochs must be positive")
        if self.model_size not in {"base", "tiny"}:
            raise ValueError("model_size must be 'base' or 'tiny'")


@dataclass
class CrossArchHLTBaselinePredictionConfig:
    """Prediction-block collection config for one direct HLT baseline."""

    architecture: str
    checkpoint: str
    cache_dir: str
    prediction_dir: str
    output_dir: str
    splits: list[str] = field(default_factory=lambda: list(PREDICTION_SPLITS))
    batch_size: int = 128
    num_workers: int = 0
    device: str = "auto"
    stack_train_size: int | None = SPLIT_SIZES["stack_train"]
    stack_val_size: int | None = SPLIT_SIZES["stack_val"]
    final_test_size: int | None = SPLIT_SIZES["final_test"]
    overwrite_predictions: bool = False
    skip_existing_predictions: bool = True
    confirm_final_test: bool = False
    control_seed: int = 12345

    def __post_init__(self) -> None:
        self.architecture = normalize_crossarch_hlt_architecture(self.architecture)
        unknown = sorted(set(self.splits) - set(PREDICTION_SPLITS))
        if unknown:
            raise ValueError(f"Unknown prediction splits: {unknown}")
        if "final_test" in self.splits and not self.confirm_final_test:
            raise ValueError("Refusing to evaluate final_test without confirm_final_test=True")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")


def normalize_crossarch_hlt_architecture(architecture: str) -> str:
    arch = normalize_architecture_name(architecture)
    if arch not in DIRECT_HLT_ARCHITECTURES:
        raise ValueError(f"Unknown crossarch HLT baseline architecture {architecture!r}")
    return arch


def crossarch_hlt_baseline_dir(
    architecture: str,
    *,
    output_root: str | Path = "checkpoints",
) -> Path:
    layout = CrossArchExperimentLayout(output_root=output_root)
    return layout.hlt_baseline_dir(normalize_crossarch_hlt_architecture(architecture))


def crossarch_hlt_checkpoint_path(
    architecture: str,
    *,
    output_root: str | Path = "checkpoints",
) -> Path:
    return crossarch_hlt_baseline_dir(architecture, output_root=output_root) / "best_model_val.pt"


def split_size_for_crossarch_hlt_prediction(
    config: CrossArchHLTBaselinePredictionConfig,
    split: str,
) -> int | None:
    if split == "stack_train":
        return config.stack_train_size
    if split == "stack_val":
        return config.stack_val_size
    if split == "final_test":
        return config.final_test_size
    return None


def _source_metadata(
    config: CrossArchHLTBaselineTrainConfig,
    *,
    train_view: JetView,
    val_view: JetView,
    subset_selection: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_step": TRAIN_EXPERIMENT_STEP,
        "architecture": config.architecture,
        "model_name": hlt_model_name(config.architecture),
        "source_kind": "direct_hlt",
        "cache_dir": config.cache_dir,
        "train_split": config.train_split,
        "val_split": config.val_split,
        "train_source_view": train_view.metadata.get("view"),
        "val_source_view": val_view.metadata.get("view"),
        "train_hlt_content_hash": train_view.metadata.get("hlt_content_hash"),
        "val_hlt_content_hash": val_view.metadata.get("hlt_content_hash"),
        "train_seed": train_view.metadata.get("seed"),
        "val_seed": val_view.metadata.get("seed"),
        "train_n_jets": int(len(train_view.labels)),
        "val_n_jets": int(len(val_view.labels)),
        "subset_selection": dict(subset_selection),
        "allowed_inputs": "cached_fixed_hlt_only",
        "no_offline_inputs_loaded": True,
        "no_stack_or_final_test_partitions_loaded": True,
    }


def train_crossarch_hlt_baseline(
    config: CrossArchHLTBaselineTrainConfig,
    *,
    model=None,
    train_view: JetView | None = None,
    val_view: JetView | None = None,
) -> dict[str, Any]:
    """Train one direct HLT baseline on cached fixed-HLT model_train/model_val."""

    torch = require_torch()
    set_training_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_view = train_view or load_cached_hlt_view(config.cache_dir, config.train_split)
    val_view = val_view or load_cached_hlt_view(config.cache_dir, config.val_split)
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
    train_dataset = JetViewTorchDataset(train_view)
    val_dataset = JetViewTorchDataset(val_view)
    train_loader = make_data_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    val_loader = make_data_loader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed + 1,
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
        train_view=train_view,
        val_view=val_view,
        subset_selection=subset_selection,
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
        "model_name": hlt_model_name(config.architecture),
        "best_epoch": int(best_epoch),
        "best_model_val_accuracy": float(best_val_accuracy),
        "best_model_val_loss": float(best_val_loss),
        "epochs_completed": len(curves),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "source_metadata_path": str(output_dir / "source_metadata.json"),
        "no_final_test_evaluation": True,
        "allowed_inputs": "cached_fixed_hlt_only",
    }
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    return report


def _prediction_metadata(
    *,
    block: PredictionBlock,
    config: CrossArchHLTBaselinePredictionConfig,
    checkpoint_payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_step": PREDICT_EXPERIMENT_STEP,
        "training_step": TRAIN_EXPERIMENT_STEP,
        "prediction_step": PREDICT_EXPERIMENT_STEP,
        "source_kind": "direct_hlt",
        "model_kind": "crossarch_direct_hlt",
        "model_name": block.model_name,
        "hlt_architecture": config.architecture,
        "checkpoint": config.checkpoint,
        "checkpoint_sha256": sha256_file(config.checkpoint),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "checkpoint_experiment_step": checkpoint_payload.get("experiment_step"),
        "checkpoint_best_model_val_accuracy": (
            (checkpoint_payload.get("metrics") or {}).get("model_val", {}) or {}
        ).get("accuracy"),
        "allowed_inputs": "cached_fixed_hlt_only",
        "split_expected_size": split_size_for_crossarch_hlt_prediction(config, block.split),
    }


def collect_crossarch_hlt_baseline_predictions(config: CrossArchHLTBaselinePredictionConfig) -> dict[str, Any]:
    """Write fusion-compatible prediction blocks for one direct HLT source."""

    torch = require_torch()
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    prediction_dir = Path(config.prediction_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "prediction_config.json", {"config": asdict(config)})

    model_name = hlt_model_name(config.architecture)
    model, payload = load_heterogeneous_hlt_model_from_checkpoint(config.checkpoint, device=device)
    reports: dict[str, Any] = {}
    for split in config.splits:
        npz_path, _ = prediction_paths(prediction_dir, model_name, split)
        if npz_path.exists() and config.skip_existing_predictions and not config.overwrite_predictions:
            from jetclass_fresh.fusion import load_prediction_block

            reports[split] = load_prediction_block(prediction_dir, model_name, split).metadata
            continue
        view = load_cached_hlt_view(config.cache_dir, split)
        block = evaluate_heterogeneous_hlt_model(
            model,
            view,
            model_name=model_name,
            architecture=config.architecture,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            device=device,
            max_jets=split_size_for_crossarch_hlt_prediction(config, split),
            selection_seed=int(config.control_seed) + 1009 * (PREDICTION_SPLITS.index(split) + 1),
        )
        block.metadata.update(_prediction_metadata(block=block, config=config, checkpoint_payload=payload))
        reports[split] = save_prediction_block(
            block,
            prediction_dir,
            overwrite=config.overwrite_predictions,
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    report = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_step": PREDICT_EXPERIMENT_STEP,
        "architecture": config.architecture,
        "model_name": model_name,
        "checkpoint": config.checkpoint,
        "prediction_dir": str(prediction_dir / model_name),
        "splits": list(config.splits),
        "split_reports": reports,
    }
    save_json(output_dir / "prediction_collection_report.json", report)
    return report


__all__ = [
    "EXPERIMENT_STEP",
    "PREDICT_EXPERIMENT_STEP",
    "PREDICTION_SPLITS",
    "TRAIN_EXPERIMENT_STEP",
    "CrossArchHLTBaselinePredictionConfig",
    "CrossArchHLTBaselineTrainConfig",
    "collect_crossarch_hlt_baseline_predictions",
    "crossarch_hlt_baseline_dir",
    "crossarch_hlt_checkpoint_path",
    "normalize_crossarch_hlt_architecture",
    "split_size_for_crossarch_hlt_prediction",
    "train_crossarch_hlt_baseline",
]
