"""Training loop for local particle residual-field reconstructors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import fnmatch
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import (
    amp_autocast_context,
    amp_grad_scaler,
    require_torch,
    resolve_device,
    save_json,
    set_training_seed,
)
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from .data import (
    LocalParticleResidualFieldDataset,
    LocalParticleResidualFieldDatasetConfig,
    load_local_particle_residual_field_dataset,
    make_local_particle_residual_field_loader,
    move_local_particle_residual_field_batch_to_device,
)
from .model import (
    LOCAL_RESIDUAL_RECONSTRUCTOR_CONTRACT,
    LOCAL_RESIDUAL_RECONSTRUCTOR_VARIANTS,
    RECONSTRUCTOR_VARIANT_C0,
    RECONSTRUCTOR_VARIANT_C5_UNCERTAINTY,
    RECONSTRUCTOR_VARIANT_C6_CONSISTENCY,
    LocalResidualFieldReconstructorConfig,
    LocalResidualFieldReconstructorOutput,
    build_local_residual_field_reconstructor,
    normalize_local_residual_reconstructor_variant,
)


LOCAL_RESIDUAL_RECONSTRUCTOR_TRAIN_CONTRACT = "local_particle_residual_field_reconstructor_train_v1"
LOCAL_RESIDUAL_RECONSTRUCTOR_SELECTION_METRICS = (
    "mae",
    "weighted_score",
    "huber_loss",
    "mse",
    "loss",
)


@dataclass
class LocalResidualReconstructorTrainConfig:
    """Configuration for one Tier C residual-field reconstructor training run."""

    output_dir: str
    hlt_cache_dir: str
    target_cache_dir: str
    manifest_path: str | None = None
    train_split: str = "model_train"
    val_split: str = "model_val"
    stack_val_split: str = "stack_val"
    seed: int = 10421
    batch_size: int = 128
    eval_batch_size: int = 256
    epochs: int = 60
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 8
    max_train_jets: int | None = None
    max_val_jets: int | None = None
    max_stack_val_jets: int | None = None
    variant: str = RECONSTRUCTOR_VARIANT_C0
    d_model: int = 160
    num_heads: int = 5
    num_layers: int = 4
    context_layers: int = 1
    mlp_ratio: float = 2.0
    dropout: float = 0.05
    attention_dropout: float = 0.05
    local_radius: float = 0.12
    hard_local_radius: float = 0.08
    use_zero_init_output: bool = True
    field_subset: tuple[str, ...] = ()
    field_group_weights: Mapping[str, float] = field(default_factory=dict)
    huber_beta: float = 0.1
    uncertainty_loss_weight: float = 1.0
    consistency_loss_weight: float = 0.0
    selection_metric: str = "mae"
    verify_hash: bool = True
    require_manifest_match: bool = True
    save_last_checkpoint: bool = True

    def __post_init__(self) -> None:
        self.output_dir = str(self.output_dir)
        self.hlt_cache_dir = str(self.hlt_cache_dir)
        self.target_cache_dir = str(self.target_cache_dir)
        self.manifest_path = None if not self.manifest_path else str(self.manifest_path)
        self.train_split = str(self.train_split)
        self.val_split = str(self.val_split)
        self.stack_val_split = str(self.stack_val_split)
        self.variant = normalize_local_residual_reconstructor_variant(self.variant)
        if self.variant not in LOCAL_RESIDUAL_RECONSTRUCTOR_VARIANTS:
            raise ValueError(f"unknown reconstructor variant {self.variant!r}")
        for name in (
            "batch_size",
            "eval_batch_size",
            "epochs",
            "d_model",
            "num_heads",
            "num_layers",
            "context_layers",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        if int(self.d_model) % int(self.num_heads) != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if int(self.num_workers) < 0:
            raise ValueError("num_workers cannot be negative")
        if int(self.early_stop_patience) < -1:
            raise ValueError("early_stop_patience must be -1 or greater")
        for name in ("lr", "mlp_ratio", "huber_beta", "local_radius", "hard_local_radius"):
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        for name in ("weight_decay", "grad_clip_norm", "uncertainty_loss_weight", "consistency_loss_weight"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            setattr(self, name, value)
        for name in ("dropout", "attention_dropout"):
            value = float(getattr(self, name))
            if value < 0.0 or value >= 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
            setattr(self, name, value)
        self.field_subset = tuple(str(value).strip() for value in self.field_subset if str(value).strip())
        self.field_group_weights = {
            str(key): float(value)
            for key, value in dict(self.field_group_weights or {}).items()
        }
        self.selection_metric = str(self.selection_metric)
        if self.selection_metric not in LOCAL_RESIDUAL_RECONSTRUCTOR_SELECTION_METRICS:
            raise ValueError(
                f"selection_metric must be one of {LOCAL_RESIDUAL_RECONSTRUCTOR_SELECTION_METRICS}"
            )
        self.max_train_jets = _optional_positive_int(self.max_train_jets, field_name="max_train_jets")
        self.max_val_jets = _optional_positive_int(self.max_val_jets, field_name="max_val_jets")
        self.max_stack_val_jets = _optional_positive_int(self.max_stack_val_jets, field_name="max_stack_val_jets")
        self.verify_hash = bool(self.verify_hash)
        self.require_manifest_match = bool(self.require_manifest_match)
        self.use_zero_init_output = bool(self.use_zero_init_output)
        self.save_last_checkpoint = bool(self.save_last_checkpoint)

    def max_jets_for_split(self, split: str) -> int | None:
        if str(split) == self.train_split:
            return self.max_train_jets
        if str(split) == self.val_split:
            return self.max_val_jets
        if str(split) == self.stack_val_split:
            return self.max_stack_val_jets
        return None


def _optional_positive_int(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    output = int(value)
    if output <= 0:
        raise ValueError(f"{field_name} must be positive when provided")
    return output


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        try:
            if int(value.numel()) == 1:
                return float(value.detach().cpu().item())
        except Exception:
            return str(value)
    return value


def _flatten_scalar_diagnostics(diagnostics: Mapping[str, Any], *, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in diagnostics.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            output.update(_flatten_scalar_diagnostics(value, prefix=f"{name}."))
            continue
        if hasattr(value, "detach"):
            try:
                if int(value.numel()) != 1:
                    continue
                numeric = float(value.detach().cpu().item())
            except Exception:
                continue
        else:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
        if np.isfinite(numeric):
            output[name] = float(numeric)
    return output


def _write_epoch_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened: list[dict[str, Any]] = []
    for row in rows:
        payload: dict[str, Any] = {"epoch": row.get("epoch")}
        for split in ("train", "model_val", "stack_val"):
            metrics = row.get(split)
            if not isinstance(metrics, Mapping):
                continue
            for key, value in metrics.items():
                if isinstance(value, Mapping):
                    for sub_key, sub_value in _flatten_scalar_diagnostics(value, prefix=f"{key}.").items():
                        payload[f"{split}_{sub_key.replace('.', '_')}"] = sub_value
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(numeric):
                    payload[f"{split}_{key}"] = numeric
        flattened.append(payload)
    fieldnames: list[str] = []
    for row in flattened:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["epoch"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flattened)


def _torch_load_checkpoint(path: str | Path, *, map_location: Any) -> Any:
    torch = require_torch()
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_dataset(
    config: LocalResidualReconstructorTrainConfig,
    split: str,
    *,
    max_jets: int | None,
) -> LocalParticleResidualFieldDataset:
    return load_local_particle_residual_field_dataset(
        LocalParticleResidualFieldDatasetConfig(
            hlt_cache_dir=config.hlt_cache_dir,
            target_cache_dir=config.target_cache_dir,
            split=str(split),
            manifest_path=config.manifest_path,
            max_jets=max_jets,
            verify_hash=bool(config.verify_hash),
            require_manifest_match=bool(config.require_manifest_match),
        )
    )


def resolve_local_residual_field_indices(
    *,
    field_names: Sequence[str],
    field_groups: Mapping[str, Sequence[int]],
    subset: Sequence[str] = (),
) -> tuple[int, ...]:
    """Resolve field subset selectors to unique field indices.

    Selectors may be field-group names, exact field names, radius prefixes such
    as ``r0p02``, glob-style patterns such as ``r0p02.*``, or a
    comma-separated mixture of these.  Empty ``subset`` means all fields.
    """

    names = tuple(str(name) for name in field_names)
    groups = {
        str(key): tuple(int(index) for index in value)
        for key, value in dict(field_groups).items()
    }
    raw_selectors: list[str] = []
    for value in subset:
        raw_selectors.extend(part.strip() for part in str(value).split(",") if part.strip())
    if not raw_selectors:
        return tuple(range(len(names)))
    selected: list[int] = []
    for selector in raw_selectors:
        if selector in groups:
            selected.extend(groups[selector])
        elif selector in names:
            selected.append(names.index(selector))
        elif any(char in selector for char in "*?["):
            matches = [index for index, name in enumerate(names) if fnmatch.fnmatchcase(name, selector)]
            if not matches:
                raise ValueError(f"unknown field subset selector {selector!r}; glob matched no field names")
            selected.extend(matches)
        elif any(name.startswith(f"{selector}.") for name in names):
            selected.extend(index for index, name in enumerate(names) if name.startswith(f"{selector}."))
        else:
            raise ValueError(
                f"unknown field subset selector {selector!r}; expected one of groups={sorted(groups)} "
                f"or field names/patterns"
            )
    deduped: list[int] = []
    seen: set[int] = set()
    for index in selected:
        if index < 0 or index >= len(names):
            raise ValueError(f"selected field index {index} outside field_dim={len(names)}")
        if index not in seen:
            seen.add(index)
            deduped.append(int(index))
    return tuple(deduped)


def local_residual_field_weights(
    *,
    field_dim: int,
    field_groups: Mapping[str, Sequence[int]],
    field_group_weights: Mapping[str, float] | None = None,
    device: Any | None = None,
):
    torch = require_torch()
    weights = torch.ones(int(field_dim), dtype=torch.float32, device=device)
    for group, weight in dict(field_group_weights or {}).items():
        if str(group) not in field_groups:
            raise ValueError(f"field group weight references unknown group {group!r}")
        indices = [int(index) for index in field_groups[str(group)]]
        if indices:
            weights[indices] = float(weight)
    return weights


def _masked_field_mean(values: Any, mask: Any) -> Any:
    denom = mask.to(dtype=values.dtype).sum().clamp_min(1.0)
    return (values * mask.to(dtype=values.dtype)).sum() / denom


def _per_group_metrics(
    residual_full: Any,
    mask: Any,
    *,
    field_groups: Mapping[str, Sequence[int]],
    selected_indices: Sequence[int],
) -> dict[str, dict[str, float]]:
    selected = set(int(index) for index in selected_indices)
    output: dict[str, dict[str, float]] = {}
    for group, raw_indices in field_groups.items():
        indices = [int(index) for index in raw_indices if int(index) in selected]
        if not indices:
            continue
        group_residual = residual_full[..., indices]
        group_mask = mask.expand_as(group_residual)
        output[str(group)] = {
            "mae": float(_masked_field_mean(group_residual.abs(), group_mask).detach().cpu().item()),
            "mse": float(_masked_field_mean(group_residual.square(), group_mask).detach().cpu().item()),
        }
    return output


def _per_field_mae(
    residual_full: Any,
    mask: Any,
    *,
    field_names: Sequence[str],
    selected_indices: Sequence[int],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for index in selected_indices:
        field_residual = residual_full[..., int(index)]
        output[str(field_names[int(index)])] = float(
            _masked_field_mean(field_residual.abs(), mask).detach().cpu().item()
        )
    return output


def _global_consistency_target(target: Any, mask: Any, field_names: Sequence[str]) -> Any:
    torch = require_torch()
    names = tuple(str(name) for name in field_names)
    mask_f = mask.to(dtype=target.dtype)

    def masked_mean_for(pattern: str) -> Any:
        indices = [idx for idx, name in enumerate(names) if pattern in name]
        if not indices:
            return torch.zeros(target.shape[0], dtype=target.dtype, device=target.device)
        values = target[..., indices].mean(dim=-1)
        return (values * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)

    return torch.stack(
        [
            masked_mean_for("delta_log_pt_sum"),
            masked_mean_for("delta_pt_frac"),
            masked_mean_for("missing_pt_frac"),
            masked_mean_for("delta_log_n"),
        ],
        dim=1,
    )


def compute_local_residual_reconstruction_loss(
    output: LocalResidualFieldReconstructorOutput,
    batch: Mapping[str, Any],
    *,
    field_names: Sequence[str],
    field_groups: Mapping[str, Sequence[int]],
    selected_indices: Sequence[int],
    field_group_weights: Mapping[str, float] | None = None,
    huber_beta: float = 0.1,
    uncertainty_loss_weight: float = 0.0,
    consistency_loss_weight: float = 0.0,
) -> tuple[Any, dict[str, Any]]:
    """Compute masked multi-task reconstruction loss and diagnostics."""

    torch = require_torch()
    target = batch["target_fields"].to(device=output.predicted_fields.device, dtype=output.predicted_fields.dtype)
    mask = batch["target_mask"].to(device=output.predicted_fields.device, dtype=torch.bool) & output.field_mask.bool()
    if output.predicted_fields.shape != target.shape:
        raise ValueError(f"prediction shape {tuple(output.predicted_fields.shape)} != target shape {tuple(target.shape)}")
    indices = tuple(int(index) for index in selected_indices)
    if not indices:
        raise ValueError("selected_indices must not be empty")
    pred_sel = output.predicted_fields[..., list(indices)]
    target_sel = target[..., list(indices)]
    residual = pred_sel - target_sel
    residual_full = output.predicted_fields - target
    mask_sel = mask.unsqueeze(-1).expand_as(residual)
    weights_full = local_residual_field_weights(
        field_dim=len(field_names),
        field_groups=field_groups,
        field_group_weights=field_group_weights,
        device=output.predicted_fields.device,
    )
    weights = weights_full[list(indices)].to(dtype=residual.dtype)
    beta = float(huber_beta)
    abs_residual = residual.abs()
    huber = torch.where(
        abs_residual < beta,
        0.5 * residual.square() / beta,
        abs_residual - 0.5 * beta,
    )
    weighted_huber = huber * weights.view(1, 1, -1)
    huber_loss = _masked_field_mean(weighted_huber, mask_sel)
    loss = huber_loss
    uncertainty_loss = output.predicted_fields.new_zeros(())
    if output.log_sigma is not None and float(uncertainty_loss_weight) > 0.0:
        log_sigma = output.log_sigma[..., list(indices)].to(dtype=residual.dtype).clamp(min=-8.0, max=8.0)
        inv_var = torch.exp(-2.0 * log_sigma)
        nll = 0.5 * inv_var * residual.square() + log_sigma
        uncertainty_loss = _masked_field_mean(nll * weights.view(1, 1, -1), mask_sel)
        loss = loss + float(uncertainty_loss_weight) * uncertainty_loss
    consistency_loss = output.predicted_fields.new_zeros(())
    consistency_prediction = output.diagnostics.get("global_consistency_prediction")
    if consistency_prediction is not None and float(consistency_loss_weight) > 0.0:
        consistency_target = _global_consistency_target(target, mask, field_names).to(
            device=output.predicted_fields.device,
            dtype=output.predicted_fields.dtype,
        )
        consistency_loss = torch.nn.functional.mse_loss(consistency_prediction, consistency_target)
        loss = loss + float(consistency_loss_weight) * consistency_loss
    mae = _masked_field_mean(abs_residual, mask_sel)
    mse = _masked_field_mean(residual.square(), mask_sel)
    zero_residual = target_sel
    zero_mae = _masked_field_mean(zero_residual.abs(), mask_sel)
    zero_mse = _masked_field_mean(zero_residual.square(), mask_sel)
    weighted_score = mae / zero_mae.clamp_min(1.0e-8)
    diagnostics = {
        "loss": float(loss.detach().cpu().item()),
        "huber_loss": float(huber_loss.detach().cpu().item()),
        "mae": float(mae.detach().cpu().item()),
        "mse": float(mse.detach().cpu().item()),
        "zero_baseline_mae": float(zero_mae.detach().cpu().item()),
        "zero_baseline_mse": float(zero_mse.detach().cpu().item()),
        "relative_mae_vs_zero": float(weighted_score.detach().cpu().item()),
        "weighted_score": float(weighted_score.detach().cpu().item()),
        "uncertainty_loss": float(uncertainty_loss.detach().cpu().item()),
        "consistency_loss": float(consistency_loss.detach().cpu().item()),
        "n_valid_particles": int(mask.sum().detach().cpu().item()),
        "n_selected_fields": int(len(indices)),
        "per_group": _per_group_metrics(
            residual_full,
            mask.unsqueeze(-1),
            field_groups=field_groups,
            selected_indices=indices,
        ),
        "per_field_mae": _per_field_mae(
            residual_full,
            mask,
            field_names=field_names,
            selected_indices=indices,
        ),
    }
    return loss, diagnostics


def _merge_average_metrics(items: Sequence[Mapping[str, Any]], weights: Sequence[int]) -> dict[str, Any]:
    total = int(sum(int(weight) for weight in weights))
    if total <= 0:
        return {"n_jets": 0}
    output: dict[str, Any] = {"n_jets": total}
    scalar_keys: set[str] = set()
    for item in items:
        for key, value in item.items():
            if isinstance(value, Mapping):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                scalar_keys.add(str(key))
    for key in sorted(scalar_keys):
        numerator = 0.0
        denom = 0
        for item, weight in zip(items, weights):
            if key not in item:
                continue
            try:
                numeric = float(item[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                numerator += numeric * int(weight)
                denom += int(weight)
        if denom > 0:
            output[key] = numerator / float(denom)
    group_payload: dict[str, dict[str, float]] = {}
    group_names = sorted(
        {
            str(group)
            for item in items
            if isinstance(item.get("per_group"), Mapping)
            for group in item["per_group"].keys()
        }
    )
    for group in group_names:
        group_payload[group] = {}
        for metric in ("mae", "mse"):
            numerator = 0.0
            denom = 0
            for item, weight in zip(items, weights):
                group_metrics = item.get("per_group")
                if not isinstance(group_metrics, Mapping) or group not in group_metrics:
                    continue
                try:
                    numeric = float(group_metrics[group][metric])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(numeric):
                    numerator += numeric * int(weight)
                    denom += int(weight)
            if denom > 0:
                group_payload[group][metric] = numerator / float(denom)
    if group_payload:
        output["per_group"] = group_payload
    return output


def _run_epoch(
    model: Any,
    loader: Any,
    *,
    device: Any,
    optimizer: Any | None,
    scaler: Any | None,
    amp_enabled: bool,
    grad_clip_norm: float,
    field_names: Sequence[str],
    field_groups: Mapping[str, Sequence[int]],
    selected_indices: Sequence[int],
    field_group_weights: Mapping[str, float],
    huber_beta: float,
    uncertainty_loss_weight: float,
    consistency_loss_weight: float,
) -> dict[str, Any]:
    torch = require_torch()
    training = optimizer is not None
    model.train(training)
    metrics: list[Mapping[str, Any]] = []
    weights: list[int] = []
    nonfinite_batches = 0
    for batch in loader:
        batch = move_local_particle_residual_field_batch_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with amp_autocast_context(bool(amp_enabled)):
            output = model(batch["tokens"], batch["raw_mask"])
            loss, batch_metrics = compute_local_residual_reconstruction_loss(
                output,
                batch,
                field_names=field_names,
                field_groups=field_groups,
                selected_indices=selected_indices,
                field_group_weights=field_group_weights,
                huber_beta=float(huber_beta),
                uncertainty_loss_weight=float(uncertainty_loss_weight),
                consistency_loss_weight=float(consistency_loss_weight),
            )
        if not bool(torch.isfinite(loss).detach().cpu().item()):
            nonfinite_batches += 1
            continue
        if training:
            assert optimizer is not None
            if scaler is not None and bool(amp_enabled):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if float(grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if float(grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                optimizer.step()
        metrics.append(batch_metrics)
        weights.append(int(batch["labels"].numel()))
    merged = _merge_average_metrics(metrics, weights)
    merged["nonfinite_batches"] = int(nonfinite_batches)
    return merged


def _evaluate(
    model: Any,
    loader: Any,
    *,
    device: Any,
    amp_enabled: bool,
    field_names: Sequence[str],
    field_groups: Mapping[str, Sequence[int]],
    selected_indices: Sequence[int],
    field_group_weights: Mapping[str, float],
    huber_beta: float,
    uncertainty_loss_weight: float,
    consistency_loss_weight: float,
) -> dict[str, Any]:
    torch = require_torch()
    with torch.no_grad():
        return _run_epoch(
            model,
            loader,
            device=device,
            optimizer=None,
            scaler=None,
            amp_enabled=bool(amp_enabled),
            grad_clip_norm=0.0,
            field_names=field_names,
            field_groups=field_groups,
            selected_indices=selected_indices,
            field_group_weights=field_group_weights,
            huber_beta=float(huber_beta),
            uncertainty_loss_weight=float(uncertainty_loss_weight),
            consistency_loss_weight=float(consistency_loss_weight),
        )


def _checkpoint_payload(
    *,
    model: Any,
    optimizer: Any | None,
    epoch: int,
    config: LocalResidualReconstructorTrainConfig,
    model_config: LocalResidualFieldReconstructorConfig,
    metrics: Mapping[str, Any],
    dataset_metadata: Mapping[str, Any],
    selected_indices: Sequence[int],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checkpoint_contract": LOCAL_RESIDUAL_RECONSTRUCTOR_TRAIN_CONTRACT,
        "model_contract": LOCAL_RESIDUAL_RECONSTRUCTOR_CONTRACT,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "config": _jsonable(asdict(config)),
        "model_config": model_config.to_dict(),
        "metrics": _jsonable(metrics),
        "dataset_metadata": _jsonable(dataset_metadata),
        "selected_field_indices": [int(index) for index in selected_indices],
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def train_local_residual_reconstructor(config: LocalResidualReconstructorTrainConfig) -> dict[str, Any]:
    """Train a C-tier local residual-field reconstructor and write reports."""

    torch = require_torch()
    set_training_seed(int(config.seed))
    output_dir = Path(config.output_dir)
    diagnostics_dir = output_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(config.device))
    amp_enabled = bool(config.amp and getattr(device, "type", str(device)) == "cuda")

    train_dataset = _load_dataset(config, config.train_split, max_jets=config.max_train_jets)
    val_dataset = _load_dataset(config, config.val_split, max_jets=config.max_val_jets)
    stack_val_dataset = (
        _load_dataset(config, config.stack_val_split, max_jets=config.max_stack_val_jets)
        if config.stack_val_split
        else None
    )
    field_names = tuple(train_dataset.field_names)
    field_groups = {
        str(group): tuple(int(index) for index in indices)
        for group, indices in dict(train_dataset.field_groups).items()
    }
    selected_indices = resolve_local_residual_field_indices(
        field_names=field_names,
        field_groups=field_groups,
        subset=tuple(config.field_subset),
    )
    if tuple(val_dataset.field_names) != field_names:
        raise ValueError("model_val field_names differ from model_train")
    if stack_val_dataset is not None and tuple(stack_val_dataset.field_names) != field_names:
        raise ValueError("stack_val field_names differ from model_train")

    model_config = LocalResidualFieldReconstructorConfig(
        variant=config.variant,
        particle_dim=RAW_TOKEN_DIM,
        field_dim=len(field_names),
        d_model=int(config.d_model),
        num_heads=int(config.num_heads),
        num_layers=int(config.num_layers),
        context_layers=int(config.context_layers),
        mlp_ratio=float(config.mlp_ratio),
        dropout=float(config.dropout),
        attention_dropout=float(config.attention_dropout),
        local_radius=float(config.local_radius),
        hard_local_radius=float(config.hard_local_radius),
        use_zero_init_output=bool(config.use_zero_init_output),
        field_groups=field_groups,
        field_names=field_names,
    )
    model = build_local_residual_field_reconstructor(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    scaler = amp_grad_scaler(bool(amp_enabled))

    train_loader = make_local_particle_residual_field_loader(
        train_dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        seed=int(config.seed),
    )
    val_loader = make_local_particle_residual_field_loader(
        val_dataset,
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 1,
    )
    stack_val_loader = (
        make_local_particle_residual_field_loader(
            stack_val_dataset,
            batch_size=int(config.eval_batch_size),
            shuffle=False,
            num_workers=int(config.num_workers),
            seed=int(config.seed) + 2,
        )
        if stack_val_dataset is not None
        else None
    )

    dataset_metadata = {
        "model_train": train_dataset.metadata,
        "model_val": val_dataset.metadata,
        "stack_val": stack_val_dataset.metadata if stack_val_dataset is not None else None,
    }
    source_metadata = {
        "contract": LOCAL_RESIDUAL_RECONSTRUCTOR_TRAIN_CONTRACT,
        "config": _jsonable(asdict(config)),
        "model_config": model_config.to_dict(),
        "dataset_metadata": _jsonable(dataset_metadata),
        "field_names": list(field_names),
        "field_groups": {group: list(indices) for group, indices in field_groups.items()},
        "selected_field_indices": [int(index) for index in selected_indices],
        "selected_field_names": [field_names[int(index)] for index in selected_indices],
    }
    save_json(output_dir / "source_metadata.json", source_metadata)

    best_epoch = -1
    best_score = float("inf")
    best_metrics: Mapping[str, Any] = {}
    epochs_without_improvement = 0
    curves: list[dict[str, Any]] = []
    for epoch in range(int(config.epochs)):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=bool(amp_enabled),
            grad_clip_norm=float(config.grad_clip_norm),
            field_names=field_names,
            field_groups=field_groups,
            selected_indices=selected_indices,
            field_group_weights=dict(config.field_group_weights),
            huber_beta=float(config.huber_beta),
            uncertainty_loss_weight=float(config.uncertainty_loss_weight),
            consistency_loss_weight=float(config.consistency_loss_weight),
        )
        val_metrics = _evaluate(
            model,
            val_loader,
            device=device,
            amp_enabled=bool(amp_enabled),
            field_names=field_names,
            field_groups=field_groups,
            selected_indices=selected_indices,
            field_group_weights=dict(config.field_group_weights),
            huber_beta=float(config.huber_beta),
            uncertainty_loss_weight=float(config.uncertainty_loss_weight),
            consistency_loss_weight=float(config.consistency_loss_weight),
        )
        row = {
            "epoch": int(epoch),
            "train": train_metrics,
            "model_val": val_metrics,
        }
        curves.append(row)
        score = float(val_metrics.get(config.selection_metric, float("inf")))
        improved = np.isfinite(score) and score < best_score
        if improved:
            best_epoch = int(epoch)
            best_score = score
            best_metrics = val_metrics
            epochs_without_improvement = 0
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    model_config=model_config,
                    metrics=val_metrics,
                    dataset_metadata=dataset_metadata,
                    selected_indices=selected_indices,
                ),
                output_dir / "best_model_val.pt",
            )
        else:
            epochs_without_improvement += 1
        if bool(config.save_last_checkpoint):
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    model_config=model_config,
                    metrics=val_metrics,
                    dataset_metadata=dataset_metadata,
                    selected_indices=selected_indices,
                ),
                output_dir / "last.pt",
            )
        save_json(output_dir / "training_curves.json", {"epochs": curves, "selection_metric": config.selection_metric})
        _write_epoch_metrics_csv(diagnostics_dir / "epoch_metrics.csv", curves)
        if int(config.early_stop_patience) >= 0 and epochs_without_improvement > int(config.early_stop_patience):
            break

    if best_epoch < 0 or not (output_dir / "best_model_val.pt").exists():
        raise RuntimeError("training did not produce a valid best_model_val.pt")

    best_payload = _torch_load_checkpoint(output_dir / "best_model_val.pt", map_location=device)
    model.load_state_dict(best_payload["model_state_dict"])
    stack_val_metrics = (
        _evaluate(
            model,
            stack_val_loader,
            device=device,
            amp_enabled=bool(amp_enabled),
            field_names=field_names,
            field_groups=field_groups,
            selected_indices=selected_indices,
            field_group_weights=dict(config.field_group_weights),
            huber_beta=float(config.huber_beta),
            uncertainty_loss_weight=float(config.uncertainty_loss_weight),
            consistency_loss_weight=float(config.consistency_loss_weight),
        )
        if stack_val_loader is not None
        else None
    )
    if curves:
        curves[-1]["best_model_val_reloaded_stack_val"] = stack_val_metrics
        save_json(output_dir / "training_curves.json", {"epochs": curves, "selection_metric": config.selection_metric})
        _write_epoch_metrics_csv(diagnostics_dir / "epoch_metrics.csv", curves)

    report = {
        "ok": True,
        "contract": LOCAL_RESIDUAL_RECONSTRUCTOR_TRAIN_CONTRACT,
        "model_contract": LOCAL_RESIDUAL_RECONSTRUCTOR_CONTRACT,
        "variant": str(config.variant),
        "output_dir": str(output_dir),
        "best_epoch": int(best_epoch),
        "selection_metric": str(config.selection_metric),
        "best_model_selection_metric_value": float(best_score),
        "best_model_val": _jsonable(best_metrics),
        "stack_val": _jsonable(stack_val_metrics),
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt") if bool(config.save_last_checkpoint) else None,
        "training_curves": str(output_dir / "training_curves.json"),
        "diagnostic_csv": str(diagnostics_dir / "epoch_metrics.csv"),
        "source_metadata": str(output_dir / "source_metadata.json"),
        "field_subset": list(config.field_subset),
        "selected_field_indices": [int(index) for index in selected_indices],
        "selected_field_names": [field_names[int(index)] for index in selected_indices],
        "field_group_weights": dict(config.field_group_weights),
        "model_config": model_config.to_dict(),
        "dataset_metadata": _jsonable(dataset_metadata),
    }
    save_json(output_dir / "run_report.json", report)
    return report


__all__ = [
    "LOCAL_RESIDUAL_RECONSTRUCTOR_TRAIN_CONTRACT",
    "LOCAL_RESIDUAL_RECONSTRUCTOR_SELECTION_METRICS",
    "LocalResidualReconstructorTrainConfig",
    "resolve_local_residual_field_indices",
    "local_residual_field_weights",
    "compute_local_residual_reconstruction_loss",
    "train_local_residual_reconstructor",
    "RECONSTRUCTOR_VARIANT_C0",
    "RECONSTRUCTOR_VARIANT_C5_UNCERTAINTY",
    "RECONSTRUCTOR_VARIANT_C6_CONSISTENCY",
]
