"""Fixed-budget Pview_0 representation warm-up and registration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from .contracts import (
    canonical_sha256,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .losses import (
    particle_view_representation_losses,
    uncertainty_calibration_metrics,
)
from .predictor import HierarchicalParticleViewPredictor


PARTICLE_VIEW_WARMUP_CONFIG_CONTRACT = "particle_view_warmup_config_v1"
PARTICLE_VIEW_PVIEW0_CHECKPOINT_CONTRACT = "particle_view_pview0_checkpoint_v1"
PARTICLE_VIEW_PVIEW0_REGISTRATION_CONTRACT = (
    "particle_view_pview0_registration_v1"
)

PVIEW0_LINEAGE_FIELDS = (
    "source_manifest_sha256",
    "train_identity_sha256",
    "model_val_stop_split_sha256",
    "hlt_preprocessing_sha256",
    "target_selection_sha256",
    "coordinate_binding_sha256",
    "selected_view_publication_sha256",
    "train_view_cache_manifest_sha256",
    "model_val_stop_view_cache_manifest_sha256",
    "clean_consumer_registration_sha256",
    "clean_consumer_checkpoint_sha256",
)


@dataclass(frozen=True)
class ParticleViewWarmupConfig:
    epochs: int = 4
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    batch_size: int = 128
    gradient_clip: float = 1.0
    seed: int = 101
    amp: bool = True
    contract: str = PARTICLE_VIEW_WARMUP_CONFIG_CONTRACT

    def __post_init__(self) -> None:
        if (
            self.epochs != 4
            or self.learning_rate != 3.0e-4
            or self.weight_decay != 1.0e-4
            or (self.adam_beta1, self.adam_beta2) != (0.9, 0.999)
            or self.batch_size != 128
            or self.gradient_clip != 1.0
        ):
            raise ValueError("Pview_0 warm-up recipe changed")
        if self.seed not in {101, 202, 303}:
            raise ValueError("Pview_0 seed must be registered")
        if not isinstance(self.amp, bool):
            raise ValueError("Pview_0 AMP flag must be boolean")

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "optimizer": "AdamW",
            "learning_rate_schedule": "constant_fixed_four_epoch_warmup",
            "objective": {
                "huber": 1.0,
                "cosine": 0.25,
                "relational": 0.15,
                "uncertainty": 0.05,
                "architecture_balance_loss": 1.0,
            },
            "performance_early_termination": False,
            "checkpoint_epoch": 4,
            "retained_snapshot_epochs": [2, 3, 4],
            "labels_exposed": False,
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_payload())


def _validate_lineage(lineage: Mapping[str, str]) -> dict[str, str]:
    if set(lineage) != set(PVIEW0_LINEAGE_FIELDS):
        raise ValueError("Pview_0 lineage field inventory mismatch")
    result = dict(lineage)
    for name, value in result.items():
        require_sha256(name, value)
    return result


def _move_batch(raw: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    required = {"features", "lorentz_vectors", "mask", "true_view"}
    if not required.issubset(raw):
        raise ValueError("Pview_0 batch is missing HLT/view fields")
    forbidden = {
        "labels",
        "offline_tokens",
        "offline_logits",
        "teacher_logits",
        "oracle_logits",
    }
    exposed = forbidden.intersection(raw)
    if exposed:
        raise ValueError(
            "Pview_0 representation warm-up exposed forbidden fields: "
            + ",".join(sorted(exposed))
        )
    batch = {
        name: raw[name].to(device=device, non_blocking=True)
        for name in required
    }
    if batch["mask"].ndim == 3:
        valid = batch["mask"][:, 0]
    else:
        valid = batch["mask"]
    if valid.dtype != torch.bool or valid.shape != batch["true_view"].shape[:2]:
        raise ValueError("Pview_0 mask/true-view shape mismatch")
    if (~valid.any(dim=1)).any():
        raise ValueError("Pview_0 does not accept all-padding events")
    if not torch.isfinite(batch["true_view"][valid]).all():
        raise ValueError("Pview_0 true view contains non-finite values")
    if (~valid).any() and (
        batch["true_view"][~valid].abs().max().item() != 0
    ):
        raise ValueError("canonical true view is nonzero on padding")
    return batch


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:  # pragma: no cover - older torch
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def evaluate_pview_representation(
    model: HierarchicalParticleViewPredictor,
    loader,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    device = torch.device(device)
    model.eval()
    names = ("total", "huber", "cosine", "relational", "uncertainty", "balance")
    sums = {name: 0.0 for name in names}
    batches = 0
    predictions = []
    targets = []
    variances = []
    masks = []
    with torch.no_grad():
        for raw in loader:
            batch = _move_batch(raw, device)
            valid = batch["mask"][:, 0] if batch["mask"].ndim == 3 else batch["mask"]
            output = model(
                batch["features"], batch["lorentz_vectors"], batch["mask"]
            )
            losses = particle_view_representation_losses(
                output, batch["true_view"], valid
            )
            for name in names:
                sums[name] += float(losses[name].item())
            predictions.append(output.mean.detach().cpu())
            targets.append(batch["true_view"].detach().cpu())
            variances.append(output.log_variance.detach().cpu())
            masks.append(valid.detach().cpu())
            batches += 1
    if not batches:
        raise ValueError("Pview_0 validation loader is empty")
    calibration = uncertainty_calibration_metrics(
        torch.cat(predictions),
        torch.cat(targets),
        torch.cat(variances),
        torch.cat(masks),
    )
    return {
        **{name: value / batches for name, value in sums.items()},
        "batches": batches,
        "uncertainty_calibration": calibration,
    }


def collect_pview_predictions(
    model: HierarchicalParticleViewPredictor,
    loader,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    """Collect RAM-resident ordered predictions for sampler fitting."""

    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Pview_0 must be frozen before residual collection")
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    predictions = []
    targets = []
    masks = []
    with torch.no_grad():
        for raw in loader:
            batch = _move_batch(raw, device)
            valid = (
                batch["mask"][:, 0]
                if batch["mask"].ndim == 3
                else batch["mask"]
            )
            output = model(
                batch["features"], batch["lorentz_vectors"], batch["mask"]
            )
            predictions.append(output.mean.detach().float().cpu().numpy())
            targets.append(batch["true_view"].detach().float().cpu().numpy())
            masks.append(valid.detach().cpu().numpy())
    if not predictions:
        raise ValueError("Pview_0 prediction collection loader is empty")
    return {
        "prediction": np.ascontiguousarray(
            np.concatenate(predictions), dtype="<f4"
        ),
        "true_view": np.ascontiguousarray(
            np.concatenate(targets), dtype="<f4"
        ),
        "mask": np.ascontiguousarray(
            np.concatenate(masks), dtype=np.bool_
        ),
    }


def _checkpoint_payload(
    *,
    model: HierarchicalParticleViewPredictor,
    config: ParticleViewWarmupConfig,
    lineage: Mapping[str, str],
    epoch: int,
    optimizer_updates: int,
    model_val_stop: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "contract": PARTICLE_VIEW_PVIEW0_CHECKPOINT_CONTRACT,
        "role": "Pview_0",
        "architecture_config": model.config.to_payload(),
        "architecture_config_sha256": model.config.content_hash,
        "warmup_config": config.to_payload(),
        "warmup_config_sha256": config.content_hash,
        "lineage": dict(lineage),
        "epoch": epoch,
        "optimizer_updates": optimizer_updates,
        "model_val_stop": dict(model_val_stop),
        "model_state_dict": model.state_dict(),
    }


def train_pview0(
    *,
    model: HierarchicalParticleViewPredictor,
    train_loader,
    model_val_stop_loader,
    output_dir: str | Path,
    lineage: Mapping[str, str],
    config: ParticleViewWarmupConfig | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run exactly four epochs and retain the final three frozen snapshots."""

    config = config or ParticleViewWarmupConfig()
    lineage = _validate_lineage(lineage)
    loader_batch_size = getattr(train_loader, "batch_size", None)
    if loader_batch_size is not None and int(loader_batch_size) != config.batch_size:
        raise ValueError("Pview_0 train loader batch size differs from recipe")
    _set_seed(config.seed)
    device = torch.device(device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
    )
    amp_enabled = bool(config.amp and device.type == "cuda")
    scaler = _grad_scaler(amp_enabled)
    root = Path(output_dir)
    snapshots = root / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    rows = []
    updates = 0
    snapshot_paths: list[Path] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        sums = {
            name: 0.0
            for name in ("total", "huber", "cosine", "relational", "uncertainty", "balance")
        }
        batches = 0
        for raw in train_loader:
            batch = _move_batch(raw, device)
            valid = batch["mask"][:, 0] if batch["mask"].ndim == 3 else batch["mask"]
            optimizer.zero_grad(set_to_none=True)
            with _autocast(amp_enabled):
                output = model(
                    batch["features"], batch["lorentz_vectors"], batch["mask"]
                )
                losses = particle_view_representation_losses(
                    output, batch["true_view"], valid
                )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            scaler.step(optimizer)
            scaler.update()
            updates += 1
            batches += 1
            for name in sums:
                sums[name] += float(losses[name].detach().item())
        if not batches:
            raise ValueError("Pview_0 train loader is empty")
        validation = evaluate_pview_representation(
            model, model_val_stop_loader, device=device
        )
        rows.append(
            {
                "epoch": epoch,
                "optimizer_updates": updates,
                "train": {
                    name: value / batches for name, value in sums.items()
                },
                "model_val_stop": validation,
            }
        )
        if epoch >= 2:
            path = snapshots / f"pview0_epoch_{epoch:03d}.pt"
            torch.save(
                _checkpoint_payload(
                    model=model,
                    config=config,
                    lineage=lineage,
                    epoch=epoch,
                    optimizer_updates=updates,
                    model_val_stop=validation,
                ),
                path,
            )
            snapshot_paths.append(path)
    if len(snapshot_paths) != 3:
        raise RuntimeError("Pview_0 final-three snapshot contract failed")
    final_path = snapshot_paths[-1]
    registration = with_content_hash(
        {
            "contract": PARTICLE_VIEW_PVIEW0_REGISTRATION_CONTRACT,
            "role": "Pview_0",
            "architecture_config": model.config.to_payload(),
            "architecture_config_sha256": model.config.content_hash,
            "warmup_config": config.to_payload(),
            "warmup_config_sha256": config.content_hash,
            "lineage": lineage,
            "checkpoint_file": str(final_path.relative_to(root)),
            "checkpoint_sha256": sha256_file(final_path),
            "snapshot_files": [
                str(path.relative_to(root)) for path in snapshot_paths
            ],
            "snapshot_epochs": [2, 3, 4],
            "snapshot_sha256": [
                sha256_file(path) for path in snapshot_paths
            ],
            "selected_epoch": 4,
            "epochs_completed": 4,
            "optimizer_updates": updates,
            "model_val_stop": rows[-1]["model_val_stop"],
            "fixed_budget_completed": True,
            "performance_early_termination": False,
            "labels_exposed": False,
            "model_val_select_loaded": False,
            "stack_val_loaded": False,
            "final_test_loaded": False,
            "frozen_after_registration": True,
        }
    )
    write_immutable_json(
        root / "pview0_training_curves.json",
        with_content_hash(
            {
                "contract": "particle_view_pview0_training_curves_v1",
                "warmup_config_sha256": config.content_hash,
                "architecture_config_sha256": model.config.content_hash,
                "lineage": lineage,
                "epochs": rows,
            }
        ),
    )
    write_immutable_json(root / "pview0_registration.json", registration)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return registration


def validate_pview0_registration(
    registration: Mapping[str, Any],
    *,
    root: str | Path,
    expected_lineage: Mapping[str, str] | None = None,
    expected_architecture_config_sha256: str | None = None,
) -> dict[str, Any]:
    validate_content_hash(
        registration,
        expected_contract=PARTICLE_VIEW_PVIEW0_REGISTRATION_CONTRACT,
    )
    warmup_payload = registration.get("warmup_config")
    if not isinstance(warmup_payload, Mapping):
        raise ValueError("Pview_0 warm-up config is missing")
    expected_warmup = ParticleViewWarmupConfig(
        seed=int(warmup_payload.get("seed")),
        amp=warmup_payload.get("amp"),
    ).to_payload()
    if (
        warmup_payload != expected_warmup
        or registration.get("warmup_config_sha256")
        != canonical_sha256(expected_warmup)
    ):
        raise ValueError("Pview_0 warm-up optimizer/objective contract mismatch")
    if registration.get("architecture_config_sha256") != canonical_sha256(
        registration.get("architecture_config")
    ):
        raise ValueError("Pview_0 architecture config hash mismatch")
    if expected_lineage is not None and registration["lineage"] != _validate_lineage(
        expected_lineage
    ):
        raise ValueError("Pview_0 lineage mismatch")
    if (
        expected_architecture_config_sha256 is not None
        and registration["architecture_config_sha256"]
        != expected_architecture_config_sha256
    ):
        raise ValueError("Pview_0 architecture config mismatch")
    if (
        registration["snapshot_epochs"] != [2, 3, 4]
        or registration["selected_epoch"] != 4
        or registration["epochs_completed"] != 4
        or not registration["fixed_budget_completed"]
    ):
        raise ValueError("Pview_0 fixed-budget contract failed")
    root = Path(root)
    paths = [root / name for name in registration["snapshot_files"]]
    hashes = [sha256_file(path) for path in paths]
    if hashes != registration["snapshot_sha256"]:
        raise ValueError("Pview_0 snapshot hash mismatch")
    if (
        sha256_file(root / registration["checkpoint_file"])
        != registration["checkpoint_sha256"]
        or registration["checkpoint_sha256"] != hashes[-1]
    ):
        raise ValueError("Pview_0 final checkpoint hash mismatch")
    return {
        "ok": True,
        "checkpoint": str(root / registration["checkpoint_file"]),
        "snapshot_paths": [str(path) for path in paths],
        "content_hash": registration["content_hash"],
    }


def load_registered_pview0(
    model: HierarchicalParticleViewPredictor,
    *,
    registration_path: str | Path,
    expected_lineage: Mapping[str, str] | None = None,
) -> HierarchicalParticleViewPredictor:
    path = Path(registration_path)
    registration = json.loads(path.read_text(encoding="utf-8"))
    validated = validate_pview0_registration(
        registration,
        root=path.parent,
        expected_lineage=expected_lineage,
        expected_architecture_config_sha256=model.config.content_hash,
    )
    checkpoint = torch.load(
        validated["checkpoint"], map_location="cpu", weights_only=False
    )
    if (
        checkpoint.get("contract") != PARTICLE_VIEW_PVIEW0_CHECKPOINT_CONTRACT
        or checkpoint.get("architecture_config_sha256")
        != model.config.content_hash
        or checkpoint.get("lineage") != registration["lineage"]
        or checkpoint.get("epoch") != 4
        or checkpoint.get("warmup_config_sha256")
        != registration["warmup_config_sha256"]
    ):
        raise ValueError("Pview_0 checkpoint payload mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


__all__ = [
    "PARTICLE_VIEW_PVIEW0_CHECKPOINT_CONTRACT",
    "PARTICLE_VIEW_PVIEW0_REGISTRATION_CONTRACT",
    "PARTICLE_VIEW_WARMUP_CONFIG_CONTRACT",
    "PVIEW0_LINEAGE_FIELDS",
    "ParticleViewWarmupConfig",
    "collect_pview_predictions",
    "evaluate_pview_representation",
    "load_registered_pview0",
    "train_pview0",
    "validate_pview0_registration",
]
