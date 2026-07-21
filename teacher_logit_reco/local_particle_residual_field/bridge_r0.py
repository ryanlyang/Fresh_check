"""Streamed ordinary R0 training and frozen publication for the bridge pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from jetclass_fresh.hlt_baseline import require_torch, resolve_device, set_training_seed

from .bridge_contracts import with_content_hash, write_immutable_json
from .bridge_ram import build_frozen_r0_registration
from .model import LocalResidualFieldReconstructorConfig, build_local_residual_field_reconstructor
from .targets import DEFAULT_LOCAL_RESIDUAL_RADII, local_particle_residual_field_layout
from .train import compute_local_residual_reconstruction_loss


PREDICTION_ANCHORED_STREAMED_R0_TRAIN_CONTRACT = "prediction_anchored_streamed_r0_train_v1"
StreamBatchFactory = Callable[[int], Iterable[Mapping[str, np.ndarray]]]


@dataclass(frozen=True)
class StreamedR0TrainConfig:
    output_dir: str
    epochs: int = 60
    seed: int = 10421
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 8
    device: str = "auto"
    variant: str = "C0"
    d_model: int = 160
    num_heads: int = 5
    num_layers: int = 4
    context_layers: int = 1
    dropout: float = 0.05
    attention_dropout: float = 0.05
    huber_beta: float = 0.1

    def __post_init__(self) -> None:
        for name in ("epochs", "d_model", "num_heads", "num_layers", "context_layers"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.d_model) % int(self.num_heads) != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if float(self.lr) <= 0 or float(self.huber_beta) <= 0:
            raise ValueError("lr and huber_beta must be positive")
        if int(self.early_stop_patience) < -1:
            raise ValueError("early_stop_patience must be at least -1")


def _tensor_batch(batch: Mapping[str, np.ndarray], device: Any) -> dict[str, Any]:
    torch = require_torch()
    required = {"hlt_tokens", "hlt_mask", "target_fields", "target_mask"}
    missing = sorted(required - set(batch))
    if missing:
        raise ValueError(f"streamed R0 batch is missing {missing}")
    tokens = np.asarray(batch["hlt_tokens"], dtype=np.float32)
    mask = np.asarray(batch["hlt_mask"], dtype=bool)
    targets = np.asarray(batch["target_fields"], dtype=np.float32)
    target_mask = np.asarray(batch["target_mask"], dtype=bool)
    if mask.shape != tokens.shape[:2] or target_mask.shape != mask.shape:
        raise ValueError("streamed R0 masks do not align")
    if targets.shape != (*mask.shape, 50):
        raise ValueError("streamed R0 targets must have shape [B,P,50]")
    if not np.array_equal(mask, target_mask):
        raise ValueError("R0 HLT and target masks differ")
    if not np.isfinite(tokens).all() or not np.isfinite(targets).all():
        raise ValueError("streamed R0 batch contains non-finite values")
    return {
        "hlt_tokens": torch.as_tensor(tokens, dtype=torch.float32, device=device),
        "hlt_mask": torch.as_tensor(mask, dtype=torch.bool, device=device),
        "target_fields": torch.as_tensor(targets, dtype=torch.float32, device=device),
        "target_mask": torch.as_tensor(target_mask, dtype=torch.bool, device=device),
    }


def _epoch(
    model: Any,
    batches: Iterable[Mapping[str, np.ndarray]],
    *,
    device: Any,
    optimizer: Any | None,
    field_names: tuple[str, ...],
    field_groups: Mapping[str, tuple[int, ...]],
    huber_beta: float,
    grad_clip_norm: float,
) -> dict[str, float]:
    torch = require_torch()
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "mae": 0.0, "mse": 0.0, "weighted_score": 0.0}
    weight = 0
    for raw_batch in batches:
        batch = _tensor_batch(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(batch["hlt_tokens"], batch["hlt_mask"])
            loss, diagnostics = compute_local_residual_reconstruction_loss(
                output,
                batch,
                field_names=field_names,
                field_groups=field_groups,
                selected_indices=tuple(range(50)),
                huber_beta=float(huber_beta),
            )
            if training:
                loss.backward()
                if float(grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                optimizer.step()
        batch_weight = int(diagnostics["n_valid_particles"])
        weight += batch_weight
        for name in totals:
            totals[name] += float(diagnostics[name]) * batch_weight
    if weight == 0:
        raise ValueError("streamed R0 epoch contained no valid particles")
    return {name: value / weight for name, value in totals.items()} | {"valid_particles": weight}


def train_streamed_r0(
    config: StreamedR0TrainConfig,
    *,
    train_batches: StreamBatchFactory,
    model_val_stop_batches: StreamBatchFactory,
    provenance_hashes: Mapping[str, str],
    matching_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Train R0 from live streamed truth and publish one weights-only checkpoint.

    Batch factories receive the epoch number.  Optimizer/checkpoint rotation
    state never enters the persistent output directory.
    """

    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"streamed R0 output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch = require_torch()
    set_training_seed(int(config.seed))
    device = resolve_device(str(config.device))
    names, groups, _ = local_particle_residual_field_layout(DEFAULT_LOCAL_RESIDUAL_RADII)
    field_names = tuple(names)
    field_groups = {name: tuple(indices) for name, indices in groups.items()}
    model_config = LocalResidualFieldReconstructorConfig(
        variant=config.variant,
        particle_dim=RAW_TOKEN_DIM,
        field_dim=50,
        d_model=int(config.d_model),
        num_heads=int(config.num_heads),
        num_layers=int(config.num_layers),
        context_layers=int(config.context_layers),
        dropout=float(config.dropout),
        attention_dropout=float(config.attention_dropout),
        field_groups=field_groups,
        field_names=field_names,
    )
    model = build_local_residual_field_reconstructor(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay)
    )
    best_state: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None
    best_epoch = -1
    stale = 0
    curves: list[dict[str, Any]] = []
    for epoch_index in range(int(config.epochs)):
        train_metrics = _epoch(
            model,
            train_batches(epoch_index),
            device=device,
            optimizer=optimizer,
            field_names=field_names,
            field_groups=field_groups,
            huber_beta=float(config.huber_beta),
            grad_clip_norm=float(config.grad_clip_norm),
        )
        with torch.no_grad():
            val_metrics = _epoch(
                model,
                model_val_stop_batches(epoch_index),
                device=device,
                optimizer=None,
                field_names=field_names,
                field_groups=field_groups,
                huber_beta=float(config.huber_beta),
                grad_clip_norm=0.0,
            )
        curves.append({"epoch": epoch_index, "train": train_metrics, "model_val_stop": val_metrics})
        if np.isfinite(val_metrics["mae"]) and (
            best_metrics is None or val_metrics["mae"] < best_metrics["mae"]
        ):
            best_epoch = epoch_index
            best_metrics = dict(val_metrics)
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if int(config.early_stop_patience) >= 0 and stale > int(config.early_stop_patience):
            break
    if best_state is None or best_metrics is None:
        raise RuntimeError("streamed R0 training produced no finite checkpoint")
    checkpoint = output / "r0_weights.pt"
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite R0 checkpoint: {checkpoint}")
    checkpoint_payload = {
        "checkpoint_contract": PREDICTION_ANCHORED_STREAMED_R0_TRAIN_CONTRACT,
        "model_config": model_config.to_dict(),
        "model_state_dict": best_state,
        "epoch": best_epoch,
        "metrics": {"model_val_stop": best_metrics},
        "selected_field_indices": list(range(50)),
        "streamed_targets": True,
        "optimizer_state_persisted": False,
    }
    torch.save(checkpoint_payload, checkpoint)
    required_hashes = {
        "preprocessing_sha256": provenance_hashes.get("preprocessing_sha256", ""),
        "target_schema_sha256": provenance_hashes.get("target_schema_sha256", ""),
        "split_manifest_sha256": provenance_hashes.get("split_manifest_sha256", ""),
    }
    registration = build_frozen_r0_registration(
        checkpoint,
        preprocessing_sha256=required_hashes["preprocessing_sha256"],
        target_schema_sha256=required_hashes["target_schema_sha256"],
        split_manifest_sha256=required_hashes["split_manifest_sha256"],
        matching_policy=matching_policy,
        validation_metrics=best_metrics,
    )
    write_immutable_json(output / "r0_registration.json", registration)
    metrics_artifact = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_STREAMED_R0_TRAIN_CONTRACT,
            "best_epoch": best_epoch,
            "best_model_val_stop": best_metrics,
            "curves": curves,
            "config": asdict(config),
            "registration_sha256": registration["content_hash"],
            "persistent_dense_fields_written": False,
            "optimizer_state_persisted": False,
        }
    )
    write_immutable_json(output / "r0_metrics.json", metrics_artifact)
    persistent_names = sorted(path.name for path in output.iterdir())
    allowed = {"r0_weights.pt", "r0_registration.json", "r0_metrics.json"}
    if set(persistent_names) != allowed:
        raise RuntimeError(f"unexpected persistent R0 artifacts: {persistent_names}")
    return {
        "ok": True,
        "contract": PREDICTION_ANCHORED_STREAMED_R0_TRAIN_CONTRACT,
        "checkpoint": str(checkpoint),
        "registration": str(output / "r0_registration.json"),
        "metrics": str(output / "r0_metrics.json"),
        "best_epoch": best_epoch,
        "best_model_val_stop": best_metrics,
        "persistent_artifacts": persistent_names,
        "persistent_dense_fields_written": False,
    }


__all__ = [
    "PREDICTION_ANCHORED_STREAMED_R0_TRAIN_CONTRACT",
    "StreamedR0TrainConfig",
    "train_streamed_r0",
]
