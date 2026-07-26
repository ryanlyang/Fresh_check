"""Locked Step-4 training and counterfactual evaluation for view consumers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from .consumer import ParticleViewConsumer
from .contracts import (
    canonical_sha256,
    require_sha256,
    sha256_file,
    with_content_hash,
    write_immutable_json,
)
from .oracle_discovery import OracleObjectiveConfig, oracle_discovery_loss


PARTICLE_VIEW_CONSUMER_TRAINING_CONTRACT = (
    "particle_view_consumer_training_recipe_v1"
)
PARTICLE_VIEW_CONSUMER_REGISTRATION_CONTRACT = (
    "particle_view_consumer_registration_v1"
)
PARTICLE_VIEW_COUNTERFACTUAL_METRICS_CONTRACT = (
    "particle_view_counterfactual_metrics_v1"
)
PARTICLE_VIEW_CONSUMER_ROLES = (
    "Cview_discovery",
    "Cview_probe",
    "Cview_clean",
)


@dataclass(frozen=True)
class ParticleViewConsumerTrainConfig:
    role: str
    maximum_epochs: int
    early_stop_patience: int | None
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    batch_size: int = 128
    gradient_clip: float = 1.0
    seed: int = 101
    contract: str = PARTICLE_VIEW_CONSUMER_TRAINING_CONTRACT

    def __post_init__(self) -> None:
        expected = {
            "Cview_discovery": (30, 6),
            "Cview_probe": (12, None),
            "Cview_clean": (40, 8),
        }
        if self.role not in expected:
            raise ValueError("consumer training role is invalid")
        if (self.maximum_epochs, self.early_stop_patience) != expected[
            self.role
        ]:
            raise ValueError("consumer epoch/patience contract changed")
        if (
            self.learning_rate != 3.0e-4
            or self.weight_decay != 1.0e-4
            or self.batch_size != 128
            or self.gradient_clip != 1.0
        ):
            raise ValueError("consumer optimizer contract changed")
        if self.seed not in {101, 202, 303}:
            raise ValueError("consumer seed must be registered")

    @classmethod
    def for_role(
        cls, role: str, *, seed: int = 101
    ) -> "ParticleViewConsumerTrainConfig":
        budgets = {
            "Cview_discovery": (30, 6),
            "Cview_probe": (12, None),
            "Cview_clean": (40, 8),
        }
        if role not in budgets:
            raise ValueError("consumer training role is invalid")
        maximum, patience = budgets[role]
        return cls(
            role=role,
            maximum_epochs=maximum,
            early_stop_patience=patience,
            seed=seed,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "optimizer": "AdamW",
            "checkpoint_selection_split": "model_val_stop",
            "checkpoint_order": [
                "highest_accuracy",
                "lowest_cross_entropy",
                "earliest_epoch",
            ],
            "budget_shortening_allowed": self.role != "Cview_probe",
            "clean_augmentation_enabled": self.role == "Cview_clean",
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_payload())


def _move_batch(
    raw: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    required = {
        "points",
        "features",
        "lorentz_vectors",
        "mask",
        "labels",
    }
    if not required.issubset(raw):
        raise ValueError("consumer batch is missing HLT/label fields")
    result = {}
    for name, value in raw.items():
        result[name] = (
            value.to(device=device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
    return result


def _resolve_view(
    batch: Mapping[str, Any],
    view_provider: Callable[[Mapping[str, Any]], Any] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    provided = view_provider(batch) if view_provider is not None else batch
    if isinstance(provided, torch.Tensor):
        view = provided
        raw_centered = provided
    elif isinstance(provided, Mapping):
        key = "view" if "view" in provided else "true_view"
        if key not in provided:
            raise ValueError("view provider did not return a view")
        view = provided[key]
        raw_centered = provided.get("raw_centered_view", view)
    else:
        raise TypeError("view provider must return a tensor or mapping")
    if not isinstance(view, torch.Tensor) or not isinstance(
        raw_centered, torch.Tensor
    ):
        raise TypeError("consumer views must be tensors")
    return view, raw_centered


def _forward_consumer(
    model: ParticleViewConsumer,
    batch: Mapping[str, Any],
    view: torch.Tensor,
    *,
    augment_clean_view: bool,
):
    return model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        view,
        augment_clean_view=augment_clean_view,
    )


def evaluate_particle_view_consumer(
    model: ParticleViewConsumer,
    loader,
    *,
    device: str | torch.device,
    view_provider: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, float]:
    device = torch.device(device)
    model.eval()
    total = correct = 0
    ce_sum = 0.0
    with torch.no_grad():
        for raw in loader:
            batch = _move_batch(raw, device)
            view, _ = _resolve_view(batch, view_provider)
            logits = _forward_consumer(
                model, batch, view, augment_clean_view=False
            ).logits
            if not torch.isfinite(logits).all():
                raise FloatingPointError("consumer validation logits are non-finite")
            labels = batch["labels"]
            ce_sum += float(
                torch.nn.functional.cross_entropy(
                    logits.float(), labels, reduction="sum"
                ).item()
            )
            total += int(labels.numel())
            correct += int(logits.argmax(dim=1).eq(labels).sum().item())
    if total == 0:
        raise ValueError("consumer validation loader is empty")
    return {
        "accuracy": correct / total,
        "cross_entropy": ce_sum / total,
        "examples": float(total),
    }


def _selection_key(row: Mapping[str, Any]) -> tuple[float, float, int]:
    return (
        -float(row["model_val_stop"]["accuracy"]),
        float(row["model_val_stop"]["cross_entropy"]),
        int(row["epoch"]),
    )


def train_particle_view_consumer(
    *,
    model: ParticleViewConsumer,
    train_loader,
    model_val_stop_loader,
    config: ParticleViewConsumerTrainConfig,
    output_dir: str | Path,
    lineage: Mapping[str, str],
    view_provider: Callable[[Mapping[str, Any]], Any] | None = None,
    validation_view_provider: Callable[[Mapping[str, Any]], Any] | None = None,
    oracle_config: OracleObjectiveConfig | None = None,
    joint_trainable_modules: Mapping[str, torch.nn.Module] | None = None,
    augment_clean_view_override: bool | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Train discovery/probe/clean consumers under their exact Step-4 budget."""

    required_lineage = {
        "a0_registration_sha256",
        "train_identity_sha256",
        "model_val_stop_split_sha256",
    }
    if not required_lineage.issubset(lineage):
        raise ValueError("consumer lineage is incomplete")
    if config.role == "Cview_discovery":
        if not (
            "target_recipe_sha256" in lineage
            or "target_registration_sha256" in lineage
        ):
            raise ValueError(
                "discovery lineage requires its acyclic target recipe"
            )
    elif "target_registration_sha256" not in lineage:
        raise ValueError("normalized consumer requires target registration")
    for name, value in lineage.items():
        require_sha256(name, value)
    if config.role != "Cview_discovery" and "normalizer_sha256" not in lineage:
        raise ValueError("normalized consumer requires normalizer lineage")
    if config.role == "Cview_clean" and (
        "selected_view_publication_sha256" not in lineage
        or view_provider is not None
    ):
        raise ValueError(
            "clean consumer requires canonical cache lineage and no live provider"
        )
    if config.role == "Cview_discovery" and oracle_config is None:
        raise ValueError("discovery consumer requires the oracle objective")
    joint_modules = dict(joint_trainable_modules or {})
    if (
        augment_clean_view_override is not None
        and (
            not isinstance(augment_clean_view_override, bool)
            or config.role not in {"Cview_probe", "Cview_clean"}
        )
    ):
        raise ValueError(
            "clean-view augmentation override is screen/clean-only"
        )
    if config.role == "Cview_discovery" and "Gview" not in joint_modules:
        raise ValueError(
            "discovery training requires a checkpointed joint Gview module"
        )
    if any(not name or not isinstance(module, torch.nn.Module) for name, module in joint_modules.items()):
        raise ValueError("joint trainable module inventory is invalid")
    loader_batch_size = getattr(train_loader, "batch_size", None)
    if loader_batch_size is not None and int(loader_batch_size) != config.batch_size:
        raise ValueError("consumer loader batch size differs from recipe")

    torch.manual_seed(config.seed)
    device = torch.device(device)
    model = model.to(device)
    for module in joint_modules.values():
        module.to(device)
    parameters = list(model.parameters())
    for module in joint_modules.values():
        parameters.extend(module.parameters())
    unique = []
    seen = set()
    for parameter in parameters:
        if parameter.requires_grad and id(parameter) not in seen:
            unique.append(parameter)
            seen.add(id(parameter))
    optimizer = torch.optim.AdamW(
        unique, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    best_path = root / "best_model_val_stop.pt"
    rows: list[dict[str, Any]] = []
    best_key = None
    stale = 0
    for epoch in range(1, config.maximum_epochs + 1):
        model.train()
        for module in joint_modules.values():
            module.train()
        train_total = 0.0
        train_examples = 0
        for raw in train_loader:
            batch = _move_batch(raw, device)
            view, raw_centered = _resolve_view(batch, view_provider)
            optimizer.zero_grad(set_to_none=True)
            output = _forward_consumer(
                model,
                batch,
                view,
                augment_clean_view=(
                    config.role == "Cview_clean"
                    if augment_clean_view_override is None
                    else augment_clean_view_override
                ),
            )
            if config.role == "Cview_discovery":
                if "offline_logits" not in batch:
                    raise ValueError(
                        "discovery batch requires frozen offline logits"
                    )
                losses = oracle_discovery_loss(
                    consumer_logits=output.logits,
                    labels=batch["labels"],
                    offline_logits=batch["offline_logits"],
                    raw_centered_view=raw_centered,
                    mask=batch["mask"][:, 0],
                    trust_loss=output.trust_loss,
                    config=oracle_config,
                )
                loss = losses["total"]
            else:
                loss = torch.nn.functional.cross_entropy(
                    output.logits, batch["labels"]
                ) + 0.01 * output.trust_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("consumer training loss is non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unique, config.gradient_clip)
            optimizer.step()
            count = int(batch["labels"].numel())
            train_total += float(loss.detach().item()) * count
            train_examples += count
        for module in joint_modules.values():
            module.eval()
        metrics = evaluate_particle_view_consumer(
            model,
            model_val_stop_loader,
            device=device,
            view_provider=validation_view_provider or view_provider,
        )
        row = {
            "epoch": epoch,
            "train_objective": train_total / max(train_examples, 1),
            "model_val_stop": metrics,
        }
        rows.append(row)
        key = _selection_key(row)
        if best_key is None or key < best_key:
            best_key = key
            stale = 0
            torch.save(
                {
                    "contract": PARTICLE_VIEW_CONSUMER_REGISTRATION_CONTRACT,
                    "role": config.role,
                    "training_config": config.to_payload(),
                    "training_config_sha256": config.content_hash,
                    "consumer_config": model.config.to_payload(),
                    "consumer_config_sha256": model.config.content_hash,
                    "model_state_dict": model.state_dict(),
                    "joint_model_state_dicts": {
                        name: module.state_dict()
                        for name, module in sorted(joint_modules.items())
                    },
                    "epoch": epoch,
                    "model_val_stop": metrics,
                    "lineage": dict(lineage),
                },
                best_path,
            )
        else:
            stale += 1
        if (
            config.early_stop_patience is not None
            and stale >= config.early_stop_patience
        ):
            break
    selected = min(rows, key=_selection_key)
    if config.role == "Cview_probe" and len(rows) != 12:
        raise RuntimeError("Cview_probe did not complete its fixed 12 epochs")
    registration = with_content_hash(
        {
            "contract": PARTICLE_VIEW_CONSUMER_REGISTRATION_CONTRACT,
            "role": config.role,
            "training_config": config.to_payload(),
            "training_config_sha256": config.content_hash,
            "consumer_config": model.config.to_payload(),
            "consumer_config_sha256": model.config.content_hash,
            "checkpoint_sha256": sha256_file(best_path),
            "selected_epoch": int(selected["epoch"]),
            "epochs_completed": len(rows),
            "model_val_stop": dict(selected["model_val_stop"]),
            "lineage": dict(lineage),
            "initialized_from_exact_a0": True,
            "zero_scaled_epoch_zero": True,
            "coordinate_mode": {
                "Cview_discovery": "raw_bounded_centered",
                "Cview_probe": "candidate_train_fit_normalized",
                "Cview_clean": "canonical_float32_selected_view_cache",
            }[config.role],
            "deployable_registration": config.role == "Cview_clean",
            "provisional_discovery_consumer": (
                config.role == "Cview_discovery"
            ),
            "joint_checkpointed_module_names": sorted(joint_modules),
            "model_val_select_loaded": False,
            "clean_view_augmentation_override": (
                augment_clean_view_override
            ),
            "stack_val_loaded": False,
            "final_test_loaded": False,
        }
    )
    write_immutable_json(
        root / "training_curves.json",
        with_content_hash({
            "contract": "particle_view_consumer_training_curves_v1",
            "training_config_sha256": config.content_hash,
            "epochs": rows,
            "lineage": dict(lineage),
        }),
    )
    write_immutable_json(
        root / "consumer_registration.json", registration
    )
    return registration


def evaluate_view_counterfactuals(
    model: ParticleViewConsumer,
    loader,
    *,
    device: str | torch.device = "cpu",
    split: str = "model_val_select",
) -> dict[str, Any]:
    """Evaluate true/predicted/zero views through one exact frozen consumer."""

    if split != "model_val_select":
        raise ValueError("candidate counterfactual ranking is model_val_select-only")
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    totals = {
        name: {"correct": 0, "ce": 0.0}
        for name in ("zero", "true", "predicted")
    }
    examples = 0
    with torch.no_grad():
        for raw in loader:
            batch = _move_batch(raw, device)
            if not {"true_view", "predicted_view"}.issubset(batch):
                raise ValueError(
                    "counterfactual batch needs true_view and predicted_view"
                )
            labels = batch["labels"]
            views = {
                "zero": torch.zeros_like(batch["true_view"]),
                "true": batch["true_view"],
                "predicted": batch["predicted_view"],
            }
            for name, view in views.items():
                logits = _forward_consumer(
                    model, batch, view, augment_clean_view=False
                ).logits
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(
                        f"{name} counterfactual logits are non-finite"
                    )
                totals[name]["correct"] += int(
                    logits.argmax(dim=1).eq(labels).sum().item()
                )
                totals[name]["ce"] += float(
                    torch.nn.functional.cross_entropy(
                        logits.float(), labels, reduction="sum"
                    ).item()
                )
            examples += int(labels.numel())
    if examples == 0:
        raise ValueError("counterfactual loader is empty")
    rows = {
        name: {
            "accuracy": values["correct"] / examples,
            "cross_entropy": values["ce"] / examples,
        }
        for name, values in totals.items()
    }
    oracle_gain = rows["true"]["accuracy"] - rows["zero"]["accuracy"]
    predicted_gain = (
        rows["predicted"]["accuracy"] - rows["zero"]["accuracy"]
    )
    recovery_status = "finite" if oracle_gain > 0 else "undefined"
    recovered = predicted_gain / oracle_gain if oracle_gain > 0 else None
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_COUNTERFACTUAL_METRICS_CONTRACT,
            "split": split,
            "examples": examples,
            "same_consumer_checkpoint": True,
            "zero_view": rows["zero"],
            "true_view": rows["true"],
            "predicted_view": rows["predicted"],
            "oracle_gain": oracle_gain,
            "predicted_view_gain": predicted_gain,
            "recovery_status": recovery_status,
            "recovered_fraction": recovered,
        }
    )


__all__ = [
    "PARTICLE_VIEW_CONSUMER_REGISTRATION_CONTRACT",
    "PARTICLE_VIEW_CONSUMER_ROLES",
    "PARTICLE_VIEW_CONSUMER_TRAINING_CONTRACT",
    "PARTICLE_VIEW_COUNTERFACTUAL_METRICS_CONTRACT",
    "ParticleViewConsumerTrainConfig",
    "evaluate_particle_view_consumer",
    "evaluate_view_counterfactuals",
    "train_particle_view_consumer",
]
