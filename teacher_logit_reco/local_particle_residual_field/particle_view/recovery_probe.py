"""Fixed-capacity label-free recovery probe used to rank view targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .contracts import (
    canonical_sha256,
    require_sha256,
    sha256_file,
    with_content_hash,
    write_immutable_json,
)


PARTICLE_VIEW_RECOVERY_PROBE_CONTRACT = "particle_view_recovery_probe_v1"
PARTICLE_VIEW_RECOVERY_PROBE_REGISTRATION_CONTRACT = (
    "particle_view_recovery_probe_registration_v1"
)


@dataclass(frozen=True)
class RecoveryProbeConfig:
    view_dim: int
    input_dim: int = 17
    width: int = 96
    num_blocks: int = 3
    num_heads: int = 8
    feed_forward_dim: int = 384
    dropout: float = 0.0
    epochs: int = 8
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    batch_size: int = 128
    gradient_clip: float = 1.0
    seed: int = 101
    contract: str = PARTICLE_VIEW_RECOVERY_PROBE_CONTRACT

    def __post_init__(self) -> None:
        if self.view_dim not in {1, 2, 4, 8}:
            raise ValueError("probe view_dim must be 1, 2, 4, or 8")
        if (
            self.input_dim != 17
            or self.width != 96
            or self.num_blocks != 3
            or self.num_heads != 8
            or self.feed_forward_dim != 384
            or self.dropout != 0.0
            or self.epochs != 8
            or self.learning_rate != 3.0e-4
            or self.weight_decay != 1.0e-4
            or self.batch_size != 128
            or self.gradient_clip != 1.0
        ):
            raise ValueError("fixed-capacity recovery-probe recipe changed")
        if self.seed not in {101, 202, 303}:
            raise ValueError("probe seed must be registered")

    def to_payload(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "view_dim": self.view_dim,
            "input_dim": self.input_dim,
            "width": self.width,
            "num_blocks": self.num_blocks,
            "num_heads": self.num_heads,
            "feed_forward_dim": self.feed_forward_dim,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "gradient_clip": self.gradient_clip,
            "seed": self.seed,
            "uses_labels": False,
            "uses_consumer_gradients": False,
            "uses_ce": False,
            "uses_kd": False,
            "objective": {"huber": 1.0, "cosine": 0.25, "relational": 0.15},
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_payload())


class FixedCapacityRecoveryProbe(nn.Module):
    def __init__(self, config: RecoveryProbeConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Sequential(
            nn.Linear(config.input_dim, config.width),
            nn.LayerNorm(config.width),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=config.width,
                nhead=config.num_heads,
                dim_feedforward=config.feed_forward_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(config.num_blocks)
        )
        self.output = nn.Linear(config.width, config.view_dim)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != self.config.input_dim:
            raise ValueError("probe features must be [B,17,P]")
        if mask.shape != (features.shape[0], features.shape[2]) or mask.dtype != torch.bool:
            raise ValueError("probe mask must be boolean [B,P]")
        if (~mask.any(dim=1)).any():
            raise ValueError(
                "recovery probe does not accept all-padding events"
            )
        values = self.embedding(features.transpose(1, 2))
        for block in self.blocks:
            values = block(values, src_key_padding_mask=~mask)
            values = torch.where(mask[:, :, None], values, torch.zeros_like(values))
        result = self.output(values)
        return torch.where(mask[:, :, None], result, torch.zeros_like(result))


def _event_average(values: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
    if eligible.any():
        return values[eligible].mean()
    return values.new_zeros(())


def view_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 0.1,
) -> torch.Tensor:
    error = (prediction - target).abs()
    penalty = torch.where(
        error <= beta,
        error.square() / (2.0 * beta),
        error - beta / 2.0,
    )
    valid = mask[:, :, None].expand_as(penalty)
    counts = valid.sum(dim=(1, 2))
    event = (penalty * valid).sum(dim=(1, 2)) / counts.clamp_min(1)
    return _event_average(event, counts > 0)


def view_cosine_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred_norm = prediction.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    eligible = mask & target_norm.ge(1.0e-6)
    cosine = (
        (prediction * target).sum(dim=-1)
        / pred_norm.clamp_min(1.0e-6)
        / target_norm.clamp_min(1.0e-6)
    )
    counts = eligible.sum(dim=1)
    event = ((1.0 - cosine) * eligible).sum(dim=1) / counts.clamp_min(1)
    return _event_average(event, counts > 0)


def view_relational_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred = prediction / prediction.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    truth = target / target.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    difference = (
        torch.matmul(pred, pred.transpose(1, 2))
        - torch.matmul(truth, truth.transpose(1, 2))
    ).abs()
    pair = mask[:, :, None] & mask[:, None, :]
    diagonal = torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
    pair &= ~diagonal
    counts = pair.sum(dim=(1, 2))
    event = (difference * pair).sum(dim=(1, 2)) / counts.clamp_min(1)
    return _event_average(event, counts > 0)


def recovery_probe_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape or mask.shape != target.shape[:2]:
        raise ValueError("probe prediction/target/mask shape mismatch")
    huber = view_huber_loss(prediction, target, mask)
    cosine = view_cosine_loss(prediction, target, mask)
    relational = view_relational_loss(prediction, target, mask)
    return {
        "huber": huber,
        "cosine": cosine,
        "relational": relational,
        "total": huber + 0.25 * cosine + 0.15 * relational,
    }


def _probe_batch(raw: Mapping[str, Any], device) -> dict[str, torch.Tensor]:
    if set(raw) != {"features", "mask", "true_view"}:
        if "labels" in raw:
            raise ValueError("recovery probe batches must not expose labels")
        raise ValueError("recovery probe batch inventory mismatch")
    return {name: value.to(device) for name, value in raw.items()}


def evaluate_recovery_probe(model, loader, *, device) -> dict[str, float]:
    model.eval()
    sums = {name: 0.0 for name in ("huber", "cosine", "relational", "total")}
    batches = 0
    with torch.no_grad():
        for raw in loader:
            batch = _probe_batch(raw, device)
            losses = recovery_probe_losses(
                model(batch["features"], batch["mask"]),
                batch["true_view"],
                batch["mask"],
            )
            for name in sums:
                sums[name] += float(losses[name].item())
            batches += 1
    if not batches:
        raise ValueError("recovery probe validation loader is empty")
    return {name: value / batches for name, value in sums.items()}


def train_recovery_probe(
    *,
    config: RecoveryProbeConfig,
    train_loader,
    model_val_stop_loader,
    output_dir: str | Path,
    target_registration_sha256: str,
    normalizer_sha256: str,
    train_identity_sha256: str,
    model_val_stop_split_sha256: str,
    hlt_preprocessing_sha256: str,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run all eight epochs; validation selects but never shortens the budget."""

    torch.manual_seed(config.seed)
    lineage = {
        "target_registration_sha256": target_registration_sha256,
        "normalizer_sha256": normalizer_sha256,
        "train_identity_sha256": train_identity_sha256,
        "model_val_stop_split_sha256": model_val_stop_split_sha256,
        "hlt_preprocessing_sha256": hlt_preprocessing_sha256,
    }
    for name, value in lineage.items():
        require_sha256(name, value)
    device = torch.device(device)
    model = FixedCapacityRecoveryProbe(config).to(device)
    loader_batch_size = getattr(train_loader, "batch_size", None)
    if (
        loader_batch_size is not None
        and int(loader_batch_size) != config.batch_size
    ):
        raise ValueError("recovery-probe loader batch size differs from recipe")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    best_key = None
    best_path = root / "best_model_val_stop.pt"
    for epoch in range(1, config.epochs + 1):
        model.train()
        for raw in train_loader:
            batch = _probe_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            losses = recovery_probe_losses(
                model(batch["features"], batch["mask"]),
                batch["true_view"],
                batch["mask"],
            )
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("recovery probe loss is non-finite")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        validation = evaluate_recovery_probe(
            model, model_val_stop_loader, device=device
        )
        row = {"epoch": epoch, "model_val_stop": validation}
        rows.append(row)
        key = (validation["total"], validation["huber"], epoch)
        if best_key is None or key < best_key:
            best_key = key
            torch.save(
                {
                    "contract": PARTICLE_VIEW_RECOVERY_PROBE_CONTRACT,
                    "config": config.to_payload(),
                    "config_sha256": config.content_hash,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "model_val_stop": validation,
                    "lineage": lineage,
                },
                best_path,
            )
    registration = with_content_hash(
        {
            "contract": PARTICLE_VIEW_RECOVERY_PROBE_REGISTRATION_CONTRACT,
            "config": config.to_payload(),
            "config_sha256": config.content_hash,
            "checkpoint_sha256": sha256_file(best_path),
            "selected_epoch": min(
                rows,
                key=lambda row: (
                    row["model_val_stop"]["total"],
                    row["model_val_stop"]["huber"],
                    row["epoch"],
                ),
            )["epoch"],
            "epochs_completed": config.epochs,
            "labels_exposed": False,
            "consumer_gradients_exposed": False,
            "fixed_budget_completed": True,
            **lineage,
        }
    )
    write_immutable_json(
        root / "training_curves.json",
        with_content_hash({
            "contract": "particle_view_recovery_probe_curves_v1",
            "config_sha256": config.content_hash,
            "epochs": rows,
            **lineage,
        }),
    )
    write_immutable_json(
        root / "recovery_probe_registration.json", registration
    )
    return registration


__all__ = [
    "PARTICLE_VIEW_RECOVERY_PROBE_CONTRACT",
    "PARTICLE_VIEW_RECOVERY_PROBE_REGISTRATION_CONTRACT",
    "FixedCapacityRecoveryProbe",
    "RecoveryProbeConfig",
    "evaluate_recovery_probe",
    "recovery_probe_losses",
    "train_recovery_probe",
    "view_cosine_loss",
    "view_huber_loss",
    "view_relational_loss",
]
