"""Fixed-budget, resumable numerical training for HOSD Stage-C baselines."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

import numpy as np

from teacher_logit_reco.relation_expert_token_bridge.determinism import (
    optimizer_update_counts,
    scheduled_learning_rate,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (
    evaluate_classification,
)

from .baselines import HOSDTrainingProtocol, classification_kd_loss
from .contracts import (
    BASELINE_CHECKPOINT_CONTRACT,
    BASELINE_COMPLETION_CONTRACT,
    require_sha256,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _move(value: Any, device: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("resume checkpoint requires CUDA RNG state")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _forward(model: Any, batch: Mapping[str, Any]) -> Any:
    vectors = batch.get("lorentz_vectors", batch.get("vectors"))
    if vectors is None:
        raise ValueError("Stage-C batch lacks Lorentz vectors")
    points = batch.get("points")
    if points is None:
        # Canonical feature channels 15/16 are exactly the repository
        # part_deta/part_dphi point coordinates.
        points = batch["features"][:, 15:17]
    return model(
        points,
        batch["features"],
        vectors,
        batch["mask"],
    )


def _balanced_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    if logits.ndim != 2 or logits.shape[1] != 10 or labels.shape != (len(logits),):
        raise ValueError("classification metric arrays differ")
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    log_probability = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    prediction = logits.argmax(axis=1)
    per_class = []
    for index in range(10):
        selected = labels == index
        per_class.append(
            None if not selected.any() else float((prediction[selected] == index).mean())
        )
    present = [value for value in per_class if value is not None]
    if not present:
        raise ValueError("classification validation population is empty")
    return {
        "accuracy": float((prediction == labels).mean()),
        "balanced_accuracy": float(np.mean(present)),
        "per_class_accuracy": per_class,
        "cross_entropy": float(-log_probability[np.arange(len(labels)), labels].mean()),
        "event_count": int(len(labels)),
        "all_ten_classes_present": len(present) == 10,
    }


def evaluate_classifier(
    model: Any,
    loader: Any,
    *,
    device: Any,
    split: str = "val_stop",
) -> dict[str, Any]:
    if split not in {"val_stop", "design_select", "design_confirm"}:
        raise ValueError("Stage-C classification evaluation split differs")
    was_training = bool(model.training)
    model.eval()
    logits, labels = [], []
    try:
        with torch.no_grad():
            for raw in loader:
                batch = _move(raw, device)
                value = _forward(model, batch)
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("Stage-C validation logits are nonfinite")
                logits.append(value.float().cpu().numpy())
                labels.append(batch["labels"].cpu().numpy())
    finally:
        if was_training:
            model.train()
    values = np.concatenate(logits)
    truth = np.concatenate(labels)
    return (
        _balanced_metrics(values, truth)
        if split == "val_stop"
        else evaluate_classification(values, truth, split=split)
    )


def _preferred(rows: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    return min(
        rows,
        key=lambda row: (
            -float(row["val_stop"]["balanced_accuracy"]),
            float(row["val_stop"]["cross_entropy"]),
            int(row["epoch"]),
        ),
    )


def train_stage_c_baseline(
    *,
    model: Any,
    train_loader: Any,
    val_stop_loader: Any,
    output_dir: str | Path,
    baseline_id: str,
    seed: int,
    component_seed: int | None = None,
    baseline_registry_sha256: str,
    campaign_spec_sha256: str,
    lineage_hashes: Mapping[str, str],
    protocol: HOSDTrainingProtocol,
    teacher_id: str | None = None,
    teacher_logit_key: str | None = None,
    device: str | Any = "cpu",
    source: Mapping[str, Any],
    resume: bool = True,
    checkpoint_contract: str = BASELINE_CHECKPOINT_CONTRACT,
    completion_contract: str = BASELINE_COMPLETION_CONTRACT,
    completion_filename: str = "baseline_completion.json",
    curves_contract: str = "hosd_baseline_training_curves_v1",
    stage_label: str = "Stage-C",
) -> dict[str, Any]:
    """Train all epochs; metrics never stop or cancel this or future rows."""

    if torch is None:
        raise RuntimeError("PyTorch is required for Stage-C training")
    protocol.validate()
    resolved_component_seed = int(
        seed if component_seed is None else component_seed
    )
    checked_lineage = {
        str(key): require_sha256(value, name=f"lineage.{key}")
        for key, value in sorted(lineage_hashes.items())
    }
    if (teacher_id is None) != (teacher_logit_key is None):
        raise ValueError("KD teacher ID and batch field must be paired")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    best_path, last_path = root / "best_model_val.pt", root / "last.pt"
    completion_path = root / completion_filename
    if completion_path.exists():
        completion = load_hashed_json(
            completion_path, expected_contract=completion_contract
        )
        if (
            completion.get("source") != dict(source)
            or completion.get("baseline_id") != baseline_id
            or completion.get("seed") != int(seed)
            or completion.get("component_seed") != resolved_component_seed
            or completion.get("baseline_registry_sha256")
            != baseline_registry_sha256
            or completion.get("campaign_spec_sha256")
            != campaign_spec_sha256
            or completion.get("lineage_hashes") != checked_lineage
        ):
            raise ValueError("reusable baseline completion lineage differs")
        checkpoint = root / completion["checkpoint_file"]
        if (
            not checkpoint.is_file()
            or hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            != completion["checkpoint_sha256"]
        ):
            raise ValueError("reusable baseline checkpoint bytes differ")
        return completion
    resolved = torch.device(device)
    model.to(resolved)
    random.seed(resolved_component_seed)
    np.random.seed(resolved_component_seed % (2**32))
    torch.manual_seed(resolved_component_seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=protocol.base_learning_rate,
        betas=(protocol.beta1, protocol.beta2),
        weight_decay=protocol.weight_decay,
    )
    counts = optimizer_update_counts(
        training_event_count=len(train_loader.dataset),
        maximum_epochs=protocol.maximum_epochs,
        microbatch_size=protocol.microbatch_size,
        gradient_accumulation_steps=protocol.gradient_accumulation_steps,
    )
    rows: list[dict[str, Any]] = []
    best_states: dict[int, dict[str, Any]] = {}
    start_epoch, update_ordinal = 1, 0
    if resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        expected = {
            "contract": checkpoint_contract,
            "baseline_id": baseline_id,
            "seed": int(seed),
            "component_seed": resolved_component_seed,
            "baseline_registry_sha256": baseline_registry_sha256,
            "campaign_spec_sha256": campaign_spec_sha256,
            "source": dict(source),
            "lineage_hashes": checked_lineage,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{stage_label} resume lineage differs")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        rows = list(state["rows"])
        best_states = {
            int(key): value for key, value in state["candidate_states"].items()
        }
        start_epoch = int(state["epoch_completed"]) + 1
        update_ordinal = int(state["optimizer_update_ordinal"])
        _restore_rng_state(state["rng_state"])

    for epoch in range(start_epoch, protocol.maximum_epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_events = 0
        loss_sum, event_sum = 0.0, 0
        for batch_index, raw in enumerate(train_loader, start=1):
            batch = _move(raw, resolved)
            with torch.autocast(
                device_type=resolved.type,
                dtype=torch.bfloat16,
                enabled=resolved.type == "cuda",
            ):
                if baseline_id == "H_NATIVE_REL_AUX":
                    vectors = batch.get("lorentz_vectors", batch.get("vectors"))
                    points = batch.get("points")
                    if points is None:
                        points = batch["features"][:, 15:17]
                    logits, auxiliary_output = model.forward_with_aux(
                        points,
                        batch["features"],
                        vectors,
                        batch["mask"],
                    )
                    auxiliary = auxiliary_output["value"]
                else:
                    logits = _forward(model, batch)
                teacher = (
                    None
                    if teacher_logit_key is None
                    else batch.get(teacher_logit_key)
                )
                if teacher_logit_key is not None and teacher is None:
                    raise ValueError(f"KD batch lacks {teacher_logit_key}")
                loss, _ = classification_kd_loss(
                    logits, batch["labels"], teacher
                )
                if baseline_id == "H_NATIVE_REL_AUX":
                    target = batch.get("offline_target_tokens")
                    if target is None:
                        raise ValueError(
                            "H_NATIVE_REL_AUX batch lacks seven-family targets"
                        )
                    packed = target.squeeze(1)
                    if packed.shape[-1] != 2 * auxiliary.shape[-1] + 7:
                        raise ValueError("native relation target shape differs")
                    target = packed[:, : auxiliary.shape[-1]]
                    target_mask = packed[
                        :, auxiliary.shape[-1] : 2 * auxiliary.shape[-1]
                    ]
                    availability = packed[:, 2 * auxiliary.shape[-1] :]
                    target_mask = target_mask.bool()
                    per_component = torch.nn.functional.huber_loss(
                        auxiliary, target, delta=1.0, reduction="none"
                    )
                    per_event_count = target_mask.sum(dim=-1)
                    applicable = per_event_count > 0
                    if bool(applicable.any()):
                        per_event = (
                            per_component.masked_fill(~target_mask, 0).sum(dim=-1)
                            / per_event_count.clamp_min(1)
                        )
                        value_loss = per_event[applicable].mean()
                    else:
                        value_loss = auxiliary.sum() * 0
                    availability_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        auxiliary_output["availability_logits"], availability
                    )
                    native_loss = 0.5 * (value_loss + availability_loss)
                    loss = loss + 0.30 * native_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Stage-C training loss is nonfinite")
            events = int(batch["labels"].numel())
            (loss * events).backward()
            loss_sum += float(loss.detach().cpu()) * events
            event_sum += events
            accumulated_events += events
            step_now = (
                batch_index % protocol.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if not step_now:
                continue
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_events)
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), protocol.gradient_clip_norm
            )
            if not bool(torch.isfinite(gradient)):
                raise FloatingPointError("Stage-C gradient is nonfinite")
            update_ordinal += 1
            lr = scheduled_learning_rate(
                update_ordinal=update_ordinal,
                total_optimizer_updates=counts["total_optimizer_updates"],
                warmup_updates=counts["warmup_updates"],
                base_learning_rate=protocol.base_learning_rate,
                minimum_learning_rate=protocol.minimum_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated_events = 0
        metrics = evaluate_classifier(model, val_stop_loader, device=resolved)
        rows.append(
            {
                "epoch": epoch,
                "optimizer_update_ordinal": update_ordinal,
                "train_objective": loss_sum / event_sum,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "val_stop": metrics,
            }
        )
        state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_states[epoch] = state
        selected = _preferred(rows)
        selected_epoch = int(selected["epoch"])
        # Only frontier states can still win an exact deterministic selector.
        best_states = {
            key: value
            for key, value in best_states.items()
            if key == selected_epoch or key == epoch
        }
        selected_state = best_states[selected_epoch]
        _atomic_torch_save(
            {
                "contract": checkpoint_contract,
                "schema_version": 1,
                "kind": "selected_inference",
                "baseline_id": baseline_id,
                "seed": int(seed),
                "component_seed": resolved_component_seed,
                "epoch": selected_epoch,
                "baseline_registry_sha256": baseline_registry_sha256,
                "campaign_spec_sha256": campaign_spec_sha256,
                "source": dict(source),
                "lineage_hashes": checked_lineage,
                "model_state_dict": selected_state,
                "selection_metrics": selected["val_stop"],
            },
            best_path,
        )
        _atomic_torch_save(
            {
                "contract": checkpoint_contract,
                "schema_version": 1,
                "kind": "resumable_last",
                "baseline_id": baseline_id,
                "seed": int(seed),
                "component_seed": resolved_component_seed,
                "epoch_completed": epoch,
                "baseline_registry_sha256": baseline_registry_sha256,
                "campaign_spec_sha256": campaign_spec_sha256,
                "source": dict(source),
                "lineage_hashes": checked_lineage,
                "model_state_dict": state,
                "optimizer_state_dict": optimizer.state_dict(),
                "optimizer_update_ordinal": update_ordinal,
                "rows": rows,
                "candidate_states": best_states,
                "rng_state": _rng_state(),
            },
            last_path,
        )
    selected = _preferred(rows)
    checkpoint_sha = hashlib.sha256(best_path.read_bytes()).hexdigest()
    completion = with_content_hash(
        {
            "contract": completion_contract,
            "schema_version": 1,
            "source": dict(source),
            "baseline_id": baseline_id,
            "seed": int(seed),
            "component_seed": resolved_component_seed,
            "teacher_id": teacher_id,
            "baseline_registry_sha256": require_sha256(
                baseline_registry_sha256, name="baseline_registry_sha256"
            ),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "lineage_hashes": checked_lineage,
            "checkpoint_file": best_path.name,
            "checkpoint_sha256": checkpoint_sha,
            "selected_epoch": int(selected["epoch"]),
            "selected_val_stop": dict(selected["val_stop"]),
            "epochs_completed": len(rows),
            "optimizer_updates_completed": update_ordinal,
            "label_presentations": len(train_loader.dataset) * protocol.maximum_epochs,
            "early_stopping": False,
            "performance_based_termination": False,
            "hlt_only_inference": True,
        }
    )
    write_immutable_json(root / "training_curves.json", with_content_hash({
        "contract": curves_contract,
        "schema_version": 1,
        "source": dict(source),
        "baseline_id": baseline_id,
        "rows": rows,
        "selected_epoch": int(selected["epoch"]),
        "fixed_epoch_budget": True,
    }))
    write_immutable_json(completion_path, completion)
    if last_path.exists():
        last_path.unlink()
    return completion


__all__ = ["evaluate_classifier", "train_stage_c_baseline"]
