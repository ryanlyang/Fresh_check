"""Fixed-budget trainer and profiler for executable offline capacity controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    bind_source,
    load_hashed_json,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .determinism import optimizer_update_counts, scheduled_learning_rate
from .evaluation import CLASSIFICATION_METRICS_CONTRACT, evaluate_classification
from .expert_training import (
    _atomic_torch_save,
    _basic_validation,
    _cpu_state,
    _file_sha256,
    _forward,
    _move_batch,
    _state_sha256,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


OFFLINE_CAPACITY_TRAINING_CONTRACT = (
    "retb_offline_capacity_fixed_budget_training_v1"
)
OFFLINE_CAPACITY_CURVES_CONTRACT = "retb_offline_capacity_curves_v1"
OFFLINE_CAPACITY_PROFILE_CONTRACT = "retb_offline_capacity_profile_v1"
HLT_CAPACITY_TRAINING_CONTRACT = (
    "retb_hlt_capacity_fixed_budget_training_v1"
)
HLT_CAPACITY_CURVES_CONTRACT = "retb_hlt_capacity_curves_v1"
HLT_CAPACITY_PROFILE_CONTRACT = "retb_hlt_capacity_profile_v1"
HLT_CAPACITY_REGISTRATION_CONTRACT = (
    "retb_hlt_capacity_training_registration_v1"
)
OFFLINE_CAPACITY_REGISTRATION_CONTRACT = (
    "retb_offline_capacity_training_registration_v1"
)
CAPACITY_VAL_DESIGN_METRICS_CONTRACT = (
    "retb_capacity_val_design_metrics_v1"
)


def _capacity_contracts(control_id: str) -> dict[str, str]:
    if str(control_id).startswith("H_"):
        return {
            "training": HLT_CAPACITY_TRAINING_CONTRACT,
            "curves": HLT_CAPACITY_CURVES_CONTRACT,
            "profile": HLT_CAPACITY_PROFILE_CONTRACT,
            "registration": HLT_CAPACITY_REGISTRATION_CONTRACT,
            "checkpoint": "retb_hlt_capacity_checkpoint_v1",
        }
    return {
        "training": OFFLINE_CAPACITY_TRAINING_CONTRACT,
        "curves": OFFLINE_CAPACITY_CURVES_CONTRACT,
        "profile": OFFLINE_CAPACITY_PROFILE_CONTRACT,
        "registration": OFFLINE_CAPACITY_REGISTRATION_CONTRACT,
        "checkpoint": "retb_offline_capacity_checkpoint_v1",
    }


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for offline capacity training")
    return torch


def build_capacity_val_design_metrics(
    *,
    classification_metrics: Mapping[str, Any],
    control_id: str,
    checkpoint_sha256: str | None,
) -> dict[str, Any]:
    classification_sha = validate_content_hash(
        classification_metrics,
        expected_contract=CLASSIFICATION_METRICS_CONTRACT,
    )
    payload = dict(classification_metrics)
    payload.pop("content_hash")
    payload.pop("source", None)
    payload.update(
        {
            "contract": CAPACITY_VAL_DESIGN_METRICS_CONTRACT,
            "schema_version": 1,
            "classification_metrics_contract": (
                CLASSIFICATION_METRICS_CONTRACT
            ),
            "classification_metrics_sha256": classification_sha,
            "control_id": str(control_id),
            "checkpoint_sha256": (
                None
                if checkpoint_sha256 is None
                else require_sha256(
                    checkpoint_sha256, name="checkpoint_sha256"
                )
            ),
        }
    )
    return with_content_hash(payload)


@dataclass(frozen=True)
class OfflineCapacityTrainingConfig:
    control_id: str
    seed: int
    maximum_epochs: int = 40
    microbatch_size: int = 64
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1.0e-3
    minimum_learning_rate: float = 1.0e-5
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    campaign_profile: str = "production"
    exact_optimizer_update_budget: int | None = None

    def validate(self) -> None:
        if (
            not self.control_id.startswith(("O_", "H_"))
            or self.seed <= 0
            or self.maximum_epochs <= 0
            or self.microbatch_size <= 0
            or self.gradient_accumulation_steps <= 0
            or self.learning_rate <= 0
            or self.minimum_learning_rate < 0
            or self.campaign_profile not in {"production", "miniature_test"}
            or (
                self.exact_optimizer_update_budget is not None
                and self.exact_optimizer_update_budget <= 0
            )
        ):
            raise ValueError("offline capacity training config differs")
        if self.campaign_profile == "production" and (
            self.maximum_epochs != 40
            or self.microbatch_size != 64
            or self.gradient_accumulation_steps != 2
            or self.learning_rate != 1.0e-3
            or self.minimum_learning_rate != 1.0e-5
        ):
            raise ValueError("production capacity protocol drifted")

    def artifact(
        self,
        *,
        global_determinism_sha256: str,
        execution_registry_sha256: str,
    ) -> dict[str, Any]:
        self.validate()
        return with_content_hash(
            {
                "contract": _capacity_contracts(self.control_id)["training"],
                "schema_version": 1,
                "config": asdict(self),
                "global_determinism_sha256": require_sha256(
                    global_determinism_sha256,
                    name="global_determinism_sha256",
                ),
                "execution_registry_sha256": require_sha256(
                    execution_registry_sha256,
                    name="execution_registry_sha256",
                ),
                "objective": (
                    "unweighted_HLT_cross_entropy"
                    if self.control_id.startswith("H_")
                    else "unweighted_offline_cross_entropy"
                ),
                "fixed_budget": True,
                "early_stopping": False,
                "performance_based_termination": False,
            }
        )


def build_capacity_profile(
    *,
    control_id: str,
    model: Any,
    analytical_flops_batch1: int,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = sum(
        int(value.numel()) for value in model.parameters()
    )
    flops = int(analytical_flops_batch1)
    if parameters <= 0 or flops <= 0:
        raise ValueError("capacity profile totals must be positive")
    return bind_source(
        with_content_hash(
            {
                "contract": _capacity_contracts(control_id)["profile"],
                "schema_version": 1,
                "control_id": str(control_id),
                "parameter_count": parameters,
                "inference_flops_batch1": flops,
                "inference_flops_batch128": 128 * flops,
                "maximum_particles": 128,
                "multiply_add_counts_as_two_flops": True,
                "analytical_not_measured": True,
                "measured_latency_used_for_selection": False,
            }
        ),
        source_snapshot=source_snapshot,
    )


def _collect_predictions(
    model: Any, loader: Any, *, device: Any
) -> dict[str, Any]:
    module = _require_torch()
    was_training = bool(model.training)
    model.eval()
    logits, labels, identities = [], [], []
    try:
        with module.no_grad():
            for raw in loader:
                batch = _move_batch(raw, device)
                values = _forward(model, batch, details=False)
                if not bool(module.isfinite(values).all()):
                    raise FloatingPointError("capacity logits are nonfinite")
                logits.append(values.float().cpu())
                labels.append(batch["labels"].long().cpu())
                identities.extend(
                    str(value) for value in batch["event_identities"]
                )
    finally:
        model.train(was_training)
    if not logits:
        raise ValueError("capacity prediction loader is empty")
    scores = module.cat(logits).numpy()
    truth = module.cat(labels).numpy()
    return {
        "logits": scores,
        "labels": truth,
        "identities": identities,
        "metrics": evaluate_classification(
            scores,
            truth,
            split="val_design",
        ),
    }


def train_offline_capacity_model(
    *,
    model: Any,
    train_loader: Any,
    val_stop_loader: Any,
    val_design_loader: Any,
    output_dir: str | Path,
    config: OfflineCapacityTrainingConfig,
    global_determinism_sha256: str,
    execution_registry_sha256: str,
    lineage_hashes: Mapping[str, str],
    profile: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    device: Any,
) -> dict[str, Any]:
    """Train every planned update and retain the deterministic val-stop best."""

    module = _require_torch()
    config.validate()
    root = Path(output_dir)
    contracts = _capacity_contracts(config.control_id)
    root.mkdir(parents=True, exist_ok=True)
    registration_path = root / "training_registration.json"
    if registration_path.exists():
        registration = load_hashed_json(
            registration_path,
            expected_contract=contracts["registration"],
        )
        if (
            registration.get("control_id") != config.control_id
            or registration.get("profile_sha256")
            != profile.get("content_hash")
            or registration.get("fixed_budget_completed") is not True
            or registration.get("lineage_hashes")
            != {
                name: require_sha256(value, name=name)
                for name, value in sorted(lineage_hashes.items())
            }
        ):
            raise ValueError("reusable capacity training differs")
        return registration
    training = bind_source(
        config.artifact(
            global_determinism_sha256=global_determinism_sha256,
            execution_registry_sha256=execution_registry_sha256,
        ),
        source_snapshot=source_snapshot,
    )
    write_immutable_json(root / "training_contract.json", training)
    resolved = module.device(device)
    if config.campaign_profile == "production":
        if (
            resolved.type != "cuda"
            or not module.cuda.is_bf16_supported()
            or "GH200" not in module.cuda.get_device_name(resolved).upper()
        ):
            raise RuntimeError("production capacity training requires GH200 BF16")
    model.to(resolved)
    optimizer = module.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )
    event_count = len(train_loader.dataset)
    counts = optimizer_update_counts(
        training_event_count=event_count,
        maximum_epochs=config.maximum_epochs,
        microbatch_size=config.microbatch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    planned_updates = (
        counts["total_optimizer_updates"]
        if config.exact_optimizer_update_budget is None
        else int(config.exact_optimizer_update_budget)
    )
    epochs = (
        config.maximum_epochs
        if config.exact_optimizer_update_budget is None
        else math.ceil(
            planned_updates / counts["optimizer_updates_per_epoch"]
        )
    )
    warmup = min(planned_updates, max(1, int(math.ceil(0.05 * planned_updates))))
    rows = []
    frontier: tuple[tuple[float, float, int], dict[str, Any]] | None = None
    update = 0
    start_epoch = 1
    resume_path = root / "resume_state.pt"
    if resume_path.is_file():
        resume = module.load(
            resume_path, map_location=resolved, weights_only=False
        )
        if (
            not isinstance(resume, Mapping)
            or resume.get("training_contract_sha256")
            != training["content_hash"]
            or resume.get("profile_sha256") != profile["content_hash"]
            or int(resume.get("planned_updates", -1)) != planned_updates
        ):
            raise ValueError("capacity resume state differs")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        rows = list(resume["rows"])
        update = int(resume["optimizer_updates_completed"])
        start_epoch = int(resume["next_epoch"])
        frontier_payload = resume.get("frontier")
        if frontier_payload is not None:
            frontier = (
                tuple(frontier_payload["key"]),
                frontier_payload["model_state_dict"],
            )
        if (
            start_epoch != len(rows) + 1
            or update < 0
            or update > planned_updates
        ):
            raise ValueError("capacity resume progress differs")
    for epoch in range(start_epoch, epochs + 1):
        dataset = getattr(train_loader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        train_loss = 0.0
        observed = 0
        for batch_index, raw in enumerate(train_loader, start=1):
            if update >= planned_updates:
                break
            batch = _move_batch(raw, resolved)
            labels = batch["labels"].long()
            logits = _forward(model, batch, details=False)
            loss = module.nn.functional.cross_entropy(
                logits, labels, reduction="sum"
            )
            loss.backward()
            current = int(labels.numel())
            accumulated += current
            observed += current
            train_loss += float(loss.detach().float().cpu())
            step = (
                batch_index % config.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if not step:
                continue
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated)
            norm = module.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            if not bool(module.isfinite(norm)):
                raise FloatingPointError("capacity gradient norm is nonfinite")
            update += 1
            lr = scheduled_learning_rate(
                update_ordinal=update,
                total_optimizer_updates=planned_updates,
                warmup_updates=warmup,
                base_learning_rate=config.learning_rate,
                minimum_learning_rate=config.minimum_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
        metrics = _basic_validation(model, val_stop_loader, device=resolved)
        row = {
            "epoch": epoch,
            "optimizer_updates_completed": update,
            "training_examples_presented": observed,
            "train_cross_entropy": train_loss / max(observed, 1),
            "val_stop": metrics,
        }
        rows.append(row)
        state = _cpu_state(model)
        key = (
            -float(metrics["accuracy"]),
            float(metrics["cross_entropy"]),
            epoch,
        )
        if frontier is None or key < frontier[0]:
            frontier = (key, state)
        _atomic_torch_save(
            {
                "contract": "retb_capacity_resume_state_v1",
                "schema_version": 1,
                "training_contract_sha256": training["content_hash"],
                "profile_sha256": profile["content_hash"],
                "planned_updates": planned_updates,
                "next_epoch": epoch + 1,
                "optimizer_updates_completed": update,
                "rows": rows,
                "model_state_dict": _cpu_state(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "frontier": {
                    "key": list(frontier[0]),
                    "model_state_dict": frontier[1],
                },
            },
            resume_path,
        )
        if update >= planned_updates:
            break
    if update != planned_updates or frontier is None:
        raise RuntimeError("capacity fixed optimizer-update budget was not completed")
    model.load_state_dict(frontier[1], strict=True)
    checkpoint = root / "best_model_val.pt"
    _atomic_torch_save(
        {
            "contract": contracts["checkpoint"],
            "schema_version": 1,
            "control_id": config.control_id,
            "training_contract_sha256": training["content_hash"],
            "model_state_dict": frontier[1],
            "model_state_sha256": _state_sha256(frontier[1]),
        },
        checkpoint,
    )
    prediction = _collect_predictions(
        model, val_design_loader, device=resolved
    )
    import numpy as np

    prediction_path = root / "val_design_predictions.npz"
    with prediction_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            logits=prediction["logits"],
            labels=prediction["labels"],
            identities=np.asarray(prediction["identities"], dtype="U"),
        )
    metrics = bind_source(
        build_capacity_val_design_metrics(
            classification_metrics=prediction["metrics"],
            control_id=config.control_id,
            checkpoint_sha256=_file_sha256(checkpoint),
        ),
        source_snapshot=source_snapshot,
    )
    write_immutable_json(root / "val_design_metrics.json", metrics)
    curves = bind_source(
        with_content_hash(
            {
                "contract": contracts["curves"],
                "schema_version": 1,
                "control_id": config.control_id,
                "rows": rows,
                "fixed_budget_completed": True,
                "optimizer_updates_completed": update,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    write_immutable_json(root / "training_curves.json", curves)
    registration = bind_source(
        with_content_hash(
            {
                "contract": contracts["registration"],
                "schema_version": 1,
                "control_id": config.control_id,
                "checkpoint_sha256": _file_sha256(checkpoint),
                "training_contract_sha256": training["content_hash"],
                "training_curves_sha256": curves["content_hash"],
                "profile_sha256": profile["content_hash"],
                "val_design_prediction_sha256": _file_sha256(prediction_path),
                "val_design_metrics_sha256": metrics["content_hash"],
                "lineage_hashes": {
                    name: require_sha256(value, name=name)
                    for name, value in sorted(lineage_hashes.items())
                },
                "labeled_example_presentations": sum(
                    int(row["training_examples_presented"]) for row in rows
                ),
                "optimizer_updates_completed": update,
                "fixed_budget_completed": True,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    write_immutable_json(registration_path, registration)
    return registration


__all__ = [
    "CAPACITY_VAL_DESIGN_METRICS_CONTRACT",
    "OFFLINE_CAPACITY_CURVES_CONTRACT",
    "OFFLINE_CAPACITY_PROFILE_CONTRACT",
    "OFFLINE_CAPACITY_TRAINING_CONTRACT",
    "HLT_CAPACITY_CURVES_CONTRACT",
    "HLT_CAPACITY_PROFILE_CONTRACT",
    "HLT_CAPACITY_REGISTRATION_CONTRACT",
    "HLT_CAPACITY_TRAINING_CONTRACT",
    "OfflineCapacityTrainingConfig",
    "build_capacity_val_design_metrics",
    "build_capacity_profile",
    "train_offline_capacity_model",
]
