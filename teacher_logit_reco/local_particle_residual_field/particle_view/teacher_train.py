"""Locked training loop for particle-view Step-2 ParT teachers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch, resolve_device

from .offline_teacher import (
    ParticleViewTeacherRecipe,
    build_particle_view_teacher_model,
    build_teacher_checkpoint_payload,
    build_teacher_registration,
    select_teacher_checkpoint,
    teacher_learning_rate,
    write_teacher_registration,
)


@dataclass(frozen=True)
class ParticleViewTeacherTrainConfig:
    output_dir: str
    device: str = "auto"
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    amp: bool = True

    def __post_init__(self) -> None:
        for name in ("max_train_batches", "max_val_batches"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive")


def expected_calibration_error(logits, labels, *, bins: int = 15) -> float:
    torch = require_torch()
    probabilities = torch.softmax(logits.float(), dim=1)
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(labels).float()
    result = torch.zeros((), dtype=torch.float64, device=logits.device)
    edges = torch.linspace(0.0, 1.0, bins + 1, device=logits.device)
    for index in range(bins):
        selected = confidence.ge(edges[index]) & (
            confidence.le(edges[index + 1])
            if index == bins - 1
            else confidence.lt(edges[index + 1])
        )
        if selected.any():
            result += selected.float().mean().double() * (
                correct[selected].mean().double()
                - confidence[selected].mean().double()
            ).abs()
    return float(result.item())


def _move(batch: Mapping[str, Any], device) -> dict[str, Any]:
    required = {"points", "features", "lorentz_vectors", "mask", "labels"}
    if set(batch) != required:
        raise ValueError("teacher batch field inventory mismatch")
    return {
        name: value.to(device=device, non_blocking=True)
        for name, value in batch.items()
    }


def evaluate_particle_view_teacher(
    model, loader, *, device, max_batches: int | None = None
) -> dict[str, float]:
    torch = require_torch()
    model.eval()
    logits_rows, labels_rows = [], []
    loss_sum = 0.0
    total = correct = 0
    with torch.no_grad():
        for index, raw in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            batch = _move(raw, device)
            logits = model(
                batch["points"],
                batch["features"],
                batch["lorentz_vectors"],
                batch["mask"],
            )
            if not torch.isfinite(logits).all():
                raise FloatingPointError("teacher validation logits are non-finite")
            loss_sum += float(
                torch.nn.functional.cross_entropy(
                    logits.float(), batch["labels"], reduction="sum"
                ).item()
            )
            total += int(batch["labels"].numel())
            correct += int(
                logits.argmax(dim=1).eq(batch["labels"]).sum().item()
            )
            logits_rows.append(logits.detach().float().cpu())
            labels_rows.append(batch["labels"].detach().cpu())
    if total == 0:
        raise ValueError("validation loader produced no examples")
    logits = torch.cat(logits_rows)
    labels = torch.cat(labels_rows)
    return {
        "accuracy": correct / total,
        "cross_entropy": loss_sum / total,
        "ece": expected_calibration_error(logits, labels),
        "examples": float(total),
    }


def _autocast(torch, enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _scaler(torch, enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:  # pragma: no cover
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_particle_view_teacher(
    *,
    recipe: ParticleViewTeacherRecipe,
    train_loader,
    model_val_stop_loader,
    config: ParticleViewTeacherTrainConfig,
    model=None,
) -> dict[str, Any]:
    """Train/select only on train/model_val_stop and publish an immutable registration."""

    torch = require_torch()
    recipe_payload = recipe.to_payload()
    device = resolve_device(config.device)
    model = (model or build_particle_view_teacher_model(recipe)).to(device)
    batch_size = getattr(train_loader, "batch_size", None)
    if batch_size is not None and batch_size != recipe_payload["physical_batch_size"]:
        raise ValueError("loader batch size differs from locked recipe")
    accumulation = recipe_payload["gradient_accumulation_steps"]
    batches = len(train_loader)
    if config.max_train_batches is not None:
        batches = min(batches, config.max_train_batches)
    updates_per_epoch = math.ceil(batches / accumulation)
    epochs = recipe_payload["schedule"]["maximum_epochs"]
    total_planned_updates = updates_per_epoch * epochs
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=recipe_payload["optimizer"]["learning_rate"],
        weight_decay=recipe_payload["optimizer"]["weight_decay"],
        betas=tuple(recipe_payload["optimizer"]["betas"]),
    )
    amp = bool(config.amp and device.type == "cuda")
    scaler = _scaler(torch, amp)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best_model_val_stop.pt"
    curves = []
    updates = 0
    current_selection = None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        train_loss = 0.0
        examples = 0
        for index, raw in enumerate(train_loader):
            if index >= batches:
                break
            batch = _move(raw, device)
            with _autocast(torch, amp):
                logits = model(
                    batch["points"],
                    batch["features"],
                    batch["lorentz_vectors"],
                    batch["mask"],
                )
                loss = torch.nn.functional.cross_entropy(
                    logits, batch["labels"]
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("teacher training loss is non-finite")
            scaler.scale(loss / accumulation).backward()
            pending += 1
            count = int(batch["labels"].numel())
            train_loss += float(loss.detach().item()) * count
            examples += count
            if pending == accumulation or index + 1 == batches:
                lr = teacher_learning_rate(
                    update_index=updates,
                    total_updates=total_planned_updates,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    recipe_payload["optimizer"]["gradient_norm_clip"],
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                pending = 0
        metrics = evaluate_particle_view_teacher(
            model,
            model_val_stop_loader,
            device=device,
            max_batches=config.max_val_batches,
        )
        row = {
            "epoch": epoch,
            "optimizer_updates": updates,
            "train_cross_entropy": train_loss / max(examples, 1),
            **metrics,
        }
        curves.append(row)
        selected = select_teacher_checkpoint(curves)
        selected_epoch = int(selected["epoch"])
        if selected_epoch != current_selection:
            current_selection = selected_epoch
            stale_epochs = 0
            if selected_epoch == epoch:
                torch.save(
                    build_teacher_checkpoint_payload(
                        recipe=recipe,
                        model=model,
                        epoch=epoch,
                        optimizer_updates=updates,
                        model_val_stop_metrics=metrics,
                    ),
                    best_path,
                )
        else:
            stale_epochs += 1
        (output / "training_curves.json").write_text(
            json.dumps(
                {
                    "recipe": recipe_payload,
                    "recipe_sha256": recipe.content_hash,
                    "epochs": curves,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if stale_epochs >= recipe_payload["schedule"]["early_stop_patience"]:
            break
    registration = build_teacher_registration(
        recipe=recipe, checkpoint_path=best_path
    )
    registration_path = output / "teacher_registration.json"
    write_teacher_registration(registration_path, registration)
    report = {
        "status": "COMPLETE",
        "role": recipe.role,
        "architecture": recipe.architecture,
        "seed": recipe.seed,
        "recipe_sha256": recipe.content_hash,
        "checkpoint": str(best_path.resolve()),
        "checkpoint_sha256": registration["checkpoint_sha256"],
        "registration": str(registration_path.resolve()),
        "registration_sha256": registration["content_hash"],
        "epochs_completed": len(curves),
        "optimizer_updates": updates,
        "selected_epoch": registration["selected_epoch"],
        "model_val_stop": registration["model_val_stop"],
        "model_val_select_evaluated": False,
        "stack_val_loaded": False,
        "final_test_loaded": False,
    }
    (output / "teacher_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "ParticleViewTeacherTrainConfig",
    "evaluate_particle_view_teacher",
    "expected_calibration_error",
    "train_particle_view_teacher",
]
