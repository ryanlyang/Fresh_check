"""Single fixed-HLT Particle Transformer baseline training for Step 5."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from pathlib import Path
from functools import partial
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from .hlt_cache import load_cached_hlt_view
from .jetclass_data import LABEL_NAMES, JetView
from .part_inputs import PF_FEATURE_NAMES, build_particle_transformer_inputs_from_tokens

try:  # Keep the module importable on machines without the training stack.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass

    class _DatasetBase:
        pass
else:
    _ModuleBase = _torch.nn.Module
    _DatasetBase = _torch.utils.data.Dataset


@dataclass
class HLTBaselineTrainConfig:
    """Training configuration for the single HLT-only baseline."""

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
    log_every: int = 50
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    model_size: str = "base"
    compile_model: bool = False


def require_torch():
    """Import torch lazily so non-training utilities remain lightweight."""

    if _torch is None:  # pragma: no cover - depends on environment
        raise ImportError("Particle Transformer training requires PyTorch")
    return _torch


def set_training_seed(seed: int) -> None:
    """Seed Python, numpy, and torch RNGs for reproducible training setup."""

    torch = require_torch()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(device: str):
    torch = require_torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def default_part_config(*, num_classes: int = 10, model_size: str = "base") -> Dict[str, Any]:
    """Particle Transformer config matching the reference wrapper by default."""

    cfg = {
        "input_dim": len(PF_FEATURE_NAMES),
        "num_classes": int(num_classes),
        "pair_input_dim": 4,
        "use_pre_activation_pair": False,
        "embed_dims": [128, 512, 128],
        "pair_embed_dims": [64, 64, 64],
        "num_heads": 8,
        "num_layers": 8,
        "num_cls_layers": 2,
        "block_params": None,
        "cls_block_params": {"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
        "fc_params": [],
        "activation": "gelu",
        "trim": True,
        "for_inference": False,
    }
    if model_size == "base":
        return cfg
    if model_size == "tiny":
        cfg.update(
            {
                "embed_dims": [32, 64, 32],
                "pair_embed_dims": [16, 16, 16],
                "num_heads": 4,
                "num_layers": 2,
                "num_cls_layers": 1,
            }
        )
        return cfg
    raise ValueError(f"Unknown model_size {model_size!r}; expected 'base' or 'tiny'")


class ParticleTransformerHLTClassifier(_ModuleBase):
    """Thin local wrapper around Weaver's ParticleTransformer reference model."""

    def __init__(self, **kwargs) -> None:
        require_torch()
        super().__init__()
        try:
            from weaver.nn.model.ParticleTransformer import ParticleTransformer
        except ImportError as exc:  # pragma: no cover - depends on research env
            raise ImportError(
                "Particle Transformer training requires weaver-core. "
                "Install it on the research compute, e.g. pip install 'weaver-core>=0.4'."
            ) from exc

        self.config = dict(kwargs)
        self.mod = ParticleTransformer(**kwargs)

    def no_weight_decay(self) -> set[str]:
        return {"mod.cls_token"}

    def forward(self, points, features, lorentz_vectors, mask):
        del points
        return self.mod(features, v=lorentz_vectors, mask=mask)


def build_particle_transformer_classifier(
    *,
    num_classes: int = 10,
    model_size: str = "base",
    overrides: Mapping[str, Any] | None = None,
):
    """Build a Particle Transformer classifier using the reference config."""

    cfg = default_part_config(num_classes=num_classes, model_size=model_size)
    if overrides:
        cfg.update(dict(overrides))
    return ParticleTransformerHLTClassifier(**cfg)


def build_hlt_classifier(*, num_classes: int = 10, model_size: str = "base", overrides: Mapping[str, Any] | None = None):
    """Build the Step 5 HLT-only Particle Transformer classifier."""

    return build_particle_transformer_classifier(num_classes=num_classes, model_size=model_size, overrides=overrides)


class ParticleViewTorchDataset(_DatasetBase):
    """PyTorch dataset over one constituent JetView."""

    def __init__(
        self,
        view: JetView,
        *,
        max_jets: int | None = None,
        expected_view: str | None = None,
    ) -> None:
        require_torch()
        source_view = view.metadata.get("view")
        if expected_view is not None and source_view not in (None, expected_view):
            raise ValueError(f"Expected a {expected_view} view, got {source_view!r}")
        limit = len(view.labels) if max_jets is None else min(int(max_jets), len(view.labels))
        self.tokens = np.asarray(view.tokens[:limit], dtype=np.float32)
        self.mask = np.asarray(view.mask[:limit], dtype=bool)
        self.labels = np.asarray(view.labels[:limit], dtype=np.int64)
        self.split = view.split
        self.source_view = source_view or expected_view

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.int64]:
        return self.tokens[index], self.mask[index], self.labels[index]


class JetViewTorchDataset(ParticleViewTorchDataset):
    """PyTorch dataset over a cached HLT JetView."""

    def __init__(self, view: JetView, *, max_jets: int | None = None) -> None:
        super().__init__(view, max_jets=max_jets, expected_view="fixed_hlt")


def collate_particle_transformer_batch(samples, *, source_view: str = "fixed_hlt"):
    """Build one torch batch of Particle Transformer inputs from raw tokens."""

    torch = require_torch()
    tokens = np.stack([sample[0] for sample in samples], axis=0)
    mask = np.stack([sample[1] for sample in samples], axis=0)
    labels = np.asarray([sample[2] for sample in samples], dtype=np.int64)
    part_inputs = build_particle_transformer_inputs_from_tokens(
        tokens,
        mask,
        labels=labels,
        source_view=source_view,
    )
    return {
        "points": torch.from_numpy(part_inputs.pf_points).float(),
        "features": torch.from_numpy(part_inputs.pf_features).float(),
        "lorentz_vectors": torch.from_numpy(part_inputs.pf_vectors).float(),
        "mask": torch.from_numpy(part_inputs.pf_mask).bool(),
        "labels": torch.from_numpy(labels).long(),
    }


def make_particle_transformer_collate(*, source_view: str):
    return partial(collate_particle_transformer_batch, source_view=source_view)


def move_batch_to_device(batch: Mapping[str, Any], device) -> Dict[str, Any]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def make_data_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    source_view: str = "fixed_hlt",
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
        collate_fn=make_particle_transformer_collate(source_view=source_view),
        generator=generator,
    )


def accuracy_from_logits(logits, labels) -> tuple[int, int]:
    preds = logits.argmax(dim=1)
    correct = int((preds == labels).sum().item())
    total = int(labels.numel())
    return correct, total


def run_epoch(
    model,
    loader,
    *,
    device,
    criterion,
    optimizer=None,
    scaler=None,
    amp: bool = False,
    grad_clip_norm: float = 0.0,
    max_batches: int | None = None,
) -> Dict[str, float]:
    """Train or evaluate one epoch depending on whether optimizer is supplied."""

    torch = require_torch()
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_batch_to_device(batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            autocast_enabled = bool(amp and device.type == "cuda")
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                logits = model(
                    batch["points"],
                    batch["features"],
                    batch["lorentz_vectors"],
                    batch["mask"],
                )
                loss = criterion(logits, batch["labels"])

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

            batch_size = int(batch["labels"].numel())
            total_loss += float(loss.detach().item()) * batch_size
            correct, seen = accuracy_from_logits(logits.detach(), batch["labels"])
            total_correct += correct
            total_seen += seen

    if total_seen == 0:
        return {"loss": float("nan"), "accuracy": 0.0, "n_jets": 0}
    return {
        "loss": total_loss / float(total_seen),
        "accuracy": total_correct / float(total_seen),
        "n_jets": int(total_seen),
    }


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def checkpoint_payload(
    model,
    optimizer,
    *,
    epoch: int,
    config: Any,
    metrics: Mapping[str, Any],
    experiment_step: str = "step5_single_hlt_baseline",
):
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "metrics": dict(metrics),
        "label_names": list(LABEL_NAMES),
        "pf_feature_names": list(PF_FEATURE_NAMES),
        "model_config": getattr(model, "config", {}),
        "experiment_step": experiment_step,
    }


def train_hlt_baseline(
    config: HLTBaselineTrainConfig,
    *,
    model=None,
    train_view: JetView | None = None,
    val_view: JetView | None = None,
    max_train_jets: int | None = None,
    max_val_jets: int | None = None,
) -> Dict[str, Any]:
    """Train the single HLT-only Particle Transformer baseline."""

    if config.train_split != "model_train" or config.val_split != "model_val":
        raise ValueError("Step 5 may train only on model_train and select only on model_val")

    torch = require_torch()
    set_training_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_view = train_view or load_cached_hlt_view(config.cache_dir, config.train_split)
    val_view = val_view or load_cached_hlt_view(config.cache_dir, config.val_split)
    train_dataset = JetViewTorchDataset(train_view, max_jets=max_train_jets)
    val_dataset = JetViewTorchDataset(val_view, max_jets=max_val_jets)
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

    model = model or build_hlt_classifier(num_classes=len(LABEL_NAMES), model_size=config.model_size)
    model = model.to(device)
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))

    run_metadata = {
        "config": asdict(config),
        "train_hlt_hash": train_view.metadata.get("hlt_content_hash"),
        "val_hlt_hash": val_view.metadata.get("hlt_content_hash"),
        "train_seed": train_view.metadata.get("seed"),
        "val_seed": val_view.metadata.get("seed"),
        "train_n_jets": len(train_dataset),
        "val_n_jets": len(val_dataset),
        "leakage_rule": "HLT baseline consumes cached fixed_hlt tokens only; no stack/final_test partitions are loaded.",
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
            checkpoint_payload(model, optimizer, epoch=epoch, config=config, metrics=row),
            output_dir / "last.pt",
        )
        if improved:
            best_val_accuracy = float(val_metrics["accuracy"])
            best_val_loss = float(val_metrics["loss"])
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(
                checkpoint_payload(model, optimizer, epoch=epoch, config=config, metrics=row),
                output_dir / "best_model_val.pt",
            )
        else:
            epochs_without_improvement += 1

        if config.early_stop_patience >= 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    report = {
        "experiment_step": "step5_single_hlt_baseline",
        "best_epoch": int(best_epoch),
        "best_model_val_accuracy": float(best_val_accuracy),
        "best_model_val_loss": float(best_val_loss),
        "epochs_completed": len(curves),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "no_final_test_evaluation": True,
    }
    save_json(output_dir / "model_val_report.json", report)
    return report
