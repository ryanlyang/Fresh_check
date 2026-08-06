"""Fixed-budget resumable training for supplemental two-teacher KD."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

from .determinism import optimizer_update_counts, scheduled_learning_rate
from .expert_training import (
    _atomic_torch_save,
    _basic_validation,
    _configure_attention_backend,
    _cpu_state,
    _forward,
    _move_batch,
    _resolve_precision,
    _restore_rng_state,
    _rng_state,
    _set_seed,
    _state_sha256,
    preferred_expert_epoch,
)
from .supplemental_specialist_kd import specialist_kd_objective


def train_specialist_kd_student(
    *,
    model: Any,
    train_loader: Any,
    val_stop_loader: Any,
    output_dir: str | Path,
    condition: str,
    run_id: str,
    plan_sha256: str,
    device: Any,
    maximum_epochs: int = 40,
    checkpoint_contract: str = "retb_specialist_kd_checkpoint_v1",
    resume_contract: str = "retb_specialist_kd_resume_v1",
) -> dict[str, Any]:
    """Run every planned update and select only with authenticated val-stop."""

    import torch

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    last_path = root / "last.pt"
    best_path = root / "best_model_val.pt"
    frontier = root / ".checkpoint_frontier"
    frontier.mkdir(exist_ok=True)
    resolved = torch.device(device)
    model.to(resolved)
    precision = _resolve_precision(resolved)
    attention = _configure_attention_backend(model, device=resolved)
    rows: list[dict[str, Any]] = []
    update_ordinal = 0
    start_epoch = 1
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0e-3, betas=(0.9, 0.999), weight_decay=1.0e-4
    )
    if last_path.is_file():
        resume = torch.load(last_path, map_location="cpu", weights_only=False)
        expected = {
            "contract": resume_contract,
            "run_id": run_id,
            "condition": condition,
            "plan_sha256": plan_sha256,
        }
        if any(resume.get(k) != v for k, v in expected.items()):
            raise ValueError("specialist KD resume lineage differs")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        rows = list(resume["rows"])
        update_ordinal = int(resume["optimizer_update_ordinal"])
        start_epoch = int(resume["epoch_completed"]) + 1
        _restore_rng_state(resume["rng_state"])
    else:
        if best_path.exists():
            raise FileExistsError("nonresumable specialist KD checkpoint exists")
        _set_seed(101)
    training_events = len(train_loader.dataset)
    counts = optimizer_update_counts(
        training_event_count=training_events,
        maximum_epochs=maximum_epochs,
        microbatch_size=64,
        gradient_accumulation_steps=2,
    )
    if len(train_loader) != counts["microbatches_per_epoch"]:
        raise ValueError("specialist KD loader length differs from schedule")
    for epoch in range(start_epoch, maximum_epochs + 1):
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        event_count = 0
        accumulated = 0
        sums = {"cross_entropy": 0.0, "common_kd": 0.0, "specialist_kd": 0.0, "total": 0.0}
        for batch_index, raw in enumerate(train_loader, start=1):
            batch = _move_batch(raw, resolved)
            labels = batch["labels"]
            teacher = batch.get("teacher_logits", {})
            if set(teacher) != {"COMMON", "SPECIALIST"}:
                raise ValueError("specialist KD teacher batch differs")
            current = int(labels.numel())
            with torch.autocast(
                device_type=resolved.type,
                dtype=precision["dtype"],
                enabled=precision["autocast"],
            ):
                logits = _forward(model, batch, details=False)
                loss, components = specialist_kd_objective(
                    logits,
                    labels,
                    condition=condition,
                    common_teacher_logits=teacher["COMMON"],
                    specialist_teacher_logits=teacher["SPECIALIST"],
                )
                loss_sum = loss * current
            loss_sum.backward()
            for name in sums:
                sums[name] += float(components[name].float().cpu()) * current
            event_count += current
            accumulated += current
            step_now = batch_index % 2 == 0 or batch_index == len(train_loader)
            if not step_now:
                continue
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated)
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise FloatingPointError("specialist KD gradient is nonfinite")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("specialist KD gradient norm is nonfinite")
            update_ordinal += 1
            rate = scheduled_learning_rate(
                update_ordinal=update_ordinal,
                total_optimizer_updates=counts["total_optimizer_updates"],
                warmup_updates=counts["warmup_updates"],
                base_learning_rate=1.0e-3,
                minimum_learning_rate=1.0e-5,
            )
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
        if event_count != training_events:
            raise RuntimeError("specialist KD epoch coverage differs")
        if update_ordinal != epoch * counts["optimizer_updates_per_epoch"]:
            raise RuntimeError("specialist KD optimizer update count differs")
        metrics = _basic_validation(model, val_stop_loader, device=resolved)
        row = {
            "epoch": epoch,
            "optimizer_update_ordinal": update_ordinal,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_objective": {name: value / event_count for name, value in sums.items()},
            "val_stop": metrics,
        }
        rows.append(row)
        state = _cpu_state(model)
        candidate = frontier / f"epoch_{epoch:03d}.pt"
        _atomic_torch_save(
            {"epoch": epoch, "model_state_dict": state, "model_state_sha256": _state_sha256(state)},
            candidate,
        )
        maximum = max(float(value["val_stop"]["accuracy"]) for value in rows)
        retained = {
            int(value["epoch"])
            for value in rows
            if maximum - float(value["val_stop"]["accuracy"]) <= 0.0001
        }
        for stale in frontier.glob("epoch_*.pt"):
            if int(stale.stem.rsplit("_", 1)[1]) not in retained:
                stale.unlink()
        selected = preferred_expert_epoch(rows, accuracy_window=0.0001)
        selected_payload = torch.load(
            frontier / f"epoch_{int(selected['epoch']):03d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        _atomic_torch_save(
            {
                "contract": checkpoint_contract,
                "run_id": run_id,
                "condition": condition,
                "plan_sha256": plan_sha256,
                "epoch": int(selected["epoch"]),
                "selection_metrics": dict(selected["val_stop"]),
                "model_state_dict": selected_payload["model_state_dict"],
                "model_state_sha256": selected_payload["model_state_sha256"],
            },
            best_path,
        )
        current = _cpu_state(model)
        _atomic_torch_save(
            {
                "contract": resume_contract,
                "run_id": run_id,
                "condition": condition,
                "plan_sha256": plan_sha256,
                "epoch_completed": epoch,
                "model_state_dict": current,
                "optimizer_state_dict": optimizer.state_dict(),
                "optimizer_update_ordinal": update_ordinal,
                "rows": rows,
                "rng_state": _rng_state(),
            },
            last_path,
        )
    selected = preferred_expert_epoch(rows, accuracy_window=0.0001)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    model.to(resolved)
    last_path.unlink(missing_ok=True)
    shutil.rmtree(frontier)
    return {
        "rows": rows,
        "selected_epoch": int(selected["epoch"]),
        "selected_val_stop": dict(selected["val_stop"]),
        "optimizer_updates_completed": update_ordinal,
        "model_state_sha256": best["model_state_sha256"],
        "attention_backend": attention,
        "precision_mode": precision["mode"],
        "planned_update_counts": counts,
    }


__all__ = ["train_specialist_kd_student"]
