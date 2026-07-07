"""Pretraining loop for target-conditioned particle denoisers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, set_training_seed
from jetclass_fresh.hlt_cache import HLT_PROFILE_V2_REALISTIC, HLT_PROFILE_V2_REALISTIC_VERSION

from .data import (
    DENOISING_TARGET_NAMES,
    TargetDenoisingDatasetConfig,
    TargetDenoisingPairedDataset,
    load_target_denoising_dataset,
    make_target_denoising_loader,
    target_denoising_batch_to_device,
)
from .model import (
    TARGET_DENOISING_MODEL_CONTRACT,
    TARGET_DENOISING_STEP2,
    TargetConditionedDenoiserConfig,
    TargetConditionedPairwiseDenoiser,
    TargetDenoisingOutput,
)


TARGET_DENOISING_STEP3 = "target_conditioned_denoising_part_step3_pretraining"
TARGET_DENOISING_TRAINING_CONTRACT = "target_conditioned_pairwise_denoising_pretraining_v1"
TARGET_DENOISING_SELECTION_METRICS = (
    "normalized_rmse",
    "loss",
    "nll_loss",
    "smooth_l1_loss",
)
TARGET_DENOISING_LOWER_IS_BETTER = set(TARGET_DENOISING_SELECTION_METRICS)


@dataclass(frozen=True)
class TargetDenoisingPretrainConfig:
    """Configuration for Step 3 denoiser pretraining."""

    output_dir: str
    manifest_path: str
    hlt_cache_dir: str
    data_dir: str | None = None
    train_split: str = "model_train"
    val_split: str = "model_val"
    seed: int = 7207
    batch_size: int = 128
    eval_batch_size: int = 256
    epochs: int = 30
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 6
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_train_jets: int | None = None
    max_val_jets: int | None = None
    selection_metric: str = "normalized_rmse"
    compile_model: bool = False
    verify_hlt_hash: bool = True
    verify_label_branches: bool = False
    read_chunk_size: int = 50_000
    alignment_mode: str = "aligned_direct"
    shuffle_target_residuals: bool = False
    target_shuffle_seed: int = 93217
    expected_hlt_profile: str | None = HLT_PROFILE_V2_REALISTIC
    expected_hlt_profile_version: str | None = HLT_PROFILE_V2_REALISTIC_VERSION
    expected_hlt_degradation_strength: float | None = 1.0
    embed_dim: int = 64
    num_heads: int = 4
    pair_hidden_dim: int = 64
    head_hidden_dim: int = 128
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    use_pair_bias: bool = True
    use_local_kernel: bool = True
    local_kernel_radius: float = 0.12
    local_kernel_init: float = 0.0
    pair_bias_max_abs: float = 4.0
    max_delta_log_pt: float = 0.30
    max_delta_eta: float = 0.08
    max_delta_phi: float = 0.08
    max_delta_log_energy: float = 0.30
    smooth_l1_weight: float = 0.5
    nll_weight: float = 0.5
    reliability_weight: float = 0.05
    delta_l2_weight: float = 1.0e-4
    pair_bias_l2_weight: float = 1.0e-5

    def __post_init__(self) -> None:
        if self.train_split != "model_train" or self.val_split != "model_val":
            raise ValueError("Step 3 trains only on model_train and selects only on model_val")
        if self.selection_metric not in TARGET_DENOISING_SELECTION_METRICS:
            raise ValueError(f"selection_metric must be one of {TARGET_DENOISING_SELECTION_METRICS}")
        for name in ("batch_size", "eval_batch_size", "epochs", "embed_dim", "num_heads"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("lr", "weight_decay", "grad_clip_norm"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        for name in (
            "smooth_l1_weight",
            "nll_weight",
            "reliability_weight",
            "delta_l2_weight",
            "pair_bias_l2_weight",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")

    def dataset_config(self, split: str, *, max_jets: int | None = None) -> TargetDenoisingDatasetConfig:
        return TargetDenoisingDatasetConfig(
            manifest_path=self.manifest_path,
            hlt_cache_dir=self.hlt_cache_dir,
            data_dir=self.data_dir,
            split=split,
            max_jets=max_jets,
            alignment_mode=self.alignment_mode,
            expected_hlt_profile=self.expected_hlt_profile,
            expected_hlt_profile_version=self.expected_hlt_profile_version,
            expected_hlt_degradation_strength=self.expected_hlt_degradation_strength,
            allow_final_test_targets=False,
            verify_hlt_hash=bool(self.verify_hlt_hash),
            verify_label_branches=bool(self.verify_label_branches),
            read_chunk_size=int(self.read_chunk_size),
        )

    def model_config(self) -> TargetConditionedDenoiserConfig:
        return TargetConditionedDenoiserConfig(
            embed_dim=int(self.embed_dim),
            num_heads=int(self.num_heads),
            pair_hidden_dim=int(self.pair_hidden_dim),
            head_hidden_dim=int(self.head_hidden_dim),
            mlp_ratio=float(self.mlp_ratio),
            dropout=float(self.dropout),
            attention_dropout=float(self.attention_dropout),
            use_pair_bias=bool(self.use_pair_bias),
            use_local_kernel=bool(self.use_local_kernel),
            local_kernel_radius=float(self.local_kernel_radius),
            local_kernel_init=float(self.local_kernel_init),
            pair_bias_max_abs=float(self.pair_bias_max_abs),
            delta_bounds=(
                float(self.max_delta_log_pt),
                float(self.max_delta_eta),
                float(self.max_delta_phi),
                float(self.max_delta_log_energy),
            ),
        )


def _shuffle_target_batch_for_control(batch: Mapping[str, Any], *, seed: int, batch_index: int) -> dict[str, Any]:
    torch = require_torch()
    size = int(batch["target_residuals"].shape[0])
    if size <= 1:
        return dict(batch)
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(batch_index))
    permutation = torch.randperm(size, generator=generator).to(device=batch["target_residuals"].device)
    output = dict(batch)
    for key in ("target_residuals", "target_mask", "target_weights", "target_status"):
        output[key] = batch[key].index_select(0, permutation)
    output["target_shuffle_permutation"] = permutation
    return output


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        tensor = value.detach()
        if tensor.numel() == 1:
            return float(tensor.cpu().item())
        return tensor.cpu().tolist()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_epoch_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in keys})


def _weighted_mean(value: Any, weights: Any, eps: float = 1.0e-8) -> Any:
    return (value * weights).sum() / weights.sum().clamp_min(float(eps))


def target_denoising_loss(
    output: TargetDenoisingOutput,
    batch: Mapping[str, Any],
    config: TargetDenoisingPretrainConfig,
) -> tuple[Any, dict[str, Any]]:
    """Compute Step 3 denoising loss and batch diagnostics."""

    torch = require_torch()
    target = batch["target_residuals"].to(device=output.deltas.device, dtype=output.deltas.dtype)
    mask = batch["target_mask"].to(device=output.deltas.device, dtype=torch.bool)
    weights = batch["target_weights"].to(device=output.deltas.device, dtype=output.deltas.dtype) * mask.to(output.deltas.dtype)
    if target.shape != output.deltas.shape:
        raise ValueError(f"target residual shape {tuple(target.shape)} does not match output {tuple(output.deltas.shape)}")
    err = output.deltas - target
    weights3 = weights[:, :, None]
    target_count = weights.sum()

    abs_err = err.abs()
    sq_err = err.square()
    smooth = torch.nn.functional.smooth_l1_loss(output.deltas, target, reduction="none", beta=1.0)
    smooth_loss = _weighted_mean(smooth, weights3)
    nll = 0.5 * torch.exp(-output.log_variances) * sq_err + 0.5 * output.log_variances
    nll_loss = _weighted_mean(nll, weights3)
    delta_l2_loss = _weighted_mean(output.deltas.square(), weights3)
    pair_bias_l2_loss = output.attention_bias.square().mean()

    reliability_target = mask.to(output.reliability.dtype)
    reliability_loss = torch.nn.functional.binary_cross_entropy(
        torch.clamp(output.reliability, min=1.0e-6, max=1.0 - 1.0e-6),
        reliability_target,
    )
    total = (
        float(config.smooth_l1_weight) * smooth_loss
        + float(config.nll_weight) * nll_loss
        + float(config.reliability_weight) * reliability_loss
        + float(config.delta_l2_weight) * delta_l2_loss
        + float(config.pair_bias_l2_weight) * pair_bias_l2_loss
    )
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite target denoising loss")

    per_target_rmse: dict[str, Any] = {}
    per_target_abs: dict[str, Any] = {}
    for index, name in enumerate(DENOISING_TARGET_NAMES):
        component_weight = weights
        component_sq = sq_err[:, :, index]
        component_abs = abs_err[:, :, index]
        per_target_rmse[f"rmse_{name}"] = torch.sqrt(_weighted_mean(component_sq, component_weight))
        per_target_abs[f"abs_mean_{name}"] = _weighted_mean(component_abs, component_weight)
    normalized_terms = []
    bounds = output.deltas.new_tensor(
        [
            float(config.max_delta_log_pt),
            float(config.max_delta_eta),
            float(config.max_delta_phi),
            float(config.max_delta_log_energy),
        ]
    )
    for index in range(len(DENOISING_TARGET_NAMES)):
        normalized_terms.append(
            _weighted_mean(sq_err[:, :, index] / bounds[index].clamp_min(1.0e-8).square(), weights)
        )
    normalized_rmse = torch.sqrt(torch.stack(normalized_terms).mean())
    diagnostics = {
        "loss": total.detach(),
        "smooth_l1_loss": smooth_loss.detach(),
        "nll_loss": nll_loss.detach(),
        "reliability_loss": reliability_loss.detach(),
        "delta_l2_loss": delta_l2_loss.detach(),
        "pair_bias_l2_loss": pair_bias_l2_loss.detach(),
        "normalized_rmse": normalized_rmse.detach(),
        "target_count": target_count.detach(),
        "target_fraction": weights.mean().detach(),
        **{key: value.detach() for key, value in per_target_rmse.items()},
        **{key: value.detach() for key, value in per_target_abs.items()},
    }
    for key, value in output.diagnostics.items():
        if hasattr(value, "detach"):
            diagnostics[f"model_{key}"] = value.detach()
    return total, diagnostics


def _summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"n_jets": 0.0}
    weights = np.asarray([max(float(row.get("n_jets", 0.0)), 1.0) for row in rows], dtype=np.float64)
    keys = sorted({key for row in rows for key in row if key != "n_jets"})
    summary: dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(row.get(key, np.nan)) for row in rows], dtype=np.float64)
        valid = np.isfinite(values)
        summary[key] = float(np.average(values[valid], weights=weights[valid])) if np.any(valid) else float("nan")
    summary["n_jets"] = float(sum(float(row.get("n_jets", 0.0)) for row in rows))
    return summary


def _tensor_diag_to_float(row: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in row.items():
        if hasattr(value, "detach"):
            tensor = value.detach()
            if tensor.numel() == 1:
                result[str(key)] = float(tensor.cpu().item())
        elif isinstance(value, (int, float, np.number)):
            result[str(key)] = float(value)
    return result


def run_target_denoising_epoch(
    model: Any,
    loader: Any,
    config: TargetDenoisingPretrainConfig,
    *,
    device: Any,
    optimizer: Any | None = None,
    scaler: Any | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run one train or eval epoch for the target-conditioned denoiser."""

    torch = require_torch()
    is_train = optimizer is not None
    model.train(is_train)
    rows: list[dict[str, float]] = []
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = target_denoising_batch_to_device(batch, device)
            if bool(config.shuffle_target_residuals):
                batch = _shuffle_target_batch_for_control(
                    batch,
                    seed=int(config.target_shuffle_seed),
                    batch_index=int(batch_index),
                )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            autocast_enabled = bool(config.amp and getattr(device, "type", str(device)) == "cuda")
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                output = model(batch["hlt_tokens"], batch["hlt_constituent_mask"], need_weights=False)
                loss, diagnostics = target_denoising_loss(output, batch, config)
            if is_train:
                if scaler is not None and autocast_enabled:
                    scaler.scale(loss).backward()
                    if float(config.grad_clip_norm) > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            float(config.grad_clip_norm),
                            error_if_nonfinite=True,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if float(config.grad_clip_norm) > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            float(config.grad_clip_norm),
                            error_if_nonfinite=True,
                        )
                    optimizer.step()
            row = _tensor_diag_to_float(diagnostics)
            row["n_jets"] = float(batch["hlt_tokens"].shape[0])
            row["target_shuffle_control"] = float(bool(config.shuffle_target_residuals))
            rows.append(row)
    return _summarize_rows(rows)


def _checkpoint_payload(
    *,
    model: Any,
    optimizer: Any,
    epoch: int,
    config: TargetDenoisingPretrainConfig,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_step": TARGET_DENOISING_STEP3,
        "output_contract": TARGET_DENOISING_TRAINING_CONTRACT,
        "model_contract": TARGET_DENOISING_MODEL_CONTRACT,
        "model_step": TARGET_DENOISING_STEP2,
        "epoch": int(epoch),
        "config": asdict(config),
        "model_config": asdict(config.model_config()),
        "metrics": dict(metrics),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }


def _is_better(value: float, best_value: float | None, metric: str) -> bool:
    if best_value is None:
        return True
    if metric in TARGET_DENOISING_LOWER_IS_BETTER:
        return float(value) < float(best_value)
    return float(value) > float(best_value)


def _load_default_datasets(
    config: TargetDenoisingPretrainConfig,
) -> tuple[TargetDenoisingPairedDataset, TargetDenoisingPairedDataset]:
    train_dataset = load_target_denoising_dataset(
        config.dataset_config(config.train_split, max_jets=config.max_train_jets)
    )
    val_dataset = load_target_denoising_dataset(
        config.dataset_config(config.val_split, max_jets=config.max_val_jets)
    )
    return train_dataset, val_dataset


def train_target_conditioned_denoiser(
    config: TargetDenoisingPretrainConfig,
    *,
    model: Any | None = None,
    train_dataset: TargetDenoisingPairedDataset | None = None,
    val_dataset: TargetDenoisingPairedDataset | None = None,
) -> dict[str, Any]:
    """Train Step 3 target-conditioned denoiser and write standard artifacts."""

    torch = require_torch()
    set_training_seed(int(config.seed))
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    if train_dataset is None or val_dataset is None:
        loaded_train, loaded_val = _load_default_datasets(config)
        train_dataset = train_dataset or loaded_train
        val_dataset = val_dataset or loaded_val

    train_loader = make_target_denoising_loader(
        train_dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        seed=int(config.seed),
    )
    val_loader = make_target_denoising_loader(
        val_dataset,
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 1,
    )
    model = model or TargetConditionedPairwiseDenoiser(config.model_config())
    model.to(device)
    if bool(config.compile_model) and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))

    _write_json(output_dir / "config.json", {"config": asdict(config), "model_config": asdict(config.model_config())})
    _write_json(diagnostics_dir / "train_dataset_metadata.json", train_dataset.to_metadata())
    _write_json(diagnostics_dir / "model_val_dataset_metadata.json", val_dataset.to_metadata())

    curves: list[dict[str, Any]] = []
    best_value: float | None = None
    best_epoch = -1
    best_metrics: dict[str, Any] | None = None
    epochs_without_improvement = 0
    best_path = output_dir / "best_denoiser_model_val.pt"
    last_path = output_dir / "last.pt"

    for epoch in range(int(config.epochs)):
        train_metrics = run_target_denoising_epoch(
            model,
            train_loader,
            config,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            max_batches=config.max_train_batches,
        )
        val_metrics = run_target_denoising_epoch(
            model,
            val_loader,
            config,
            device=device,
            max_batches=config.max_val_batches,
        )
        metric_value = float(val_metrics.get(config.selection_metric, float("inf")))
        improved = _is_better(metric_value, best_value, config.selection_metric)
        if improved:
            best_value = metric_value
            best_epoch = int(epoch)
            best_metrics = dict(val_metrics)
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    metrics=val_metrics,
                ),
                best_path,
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        torch.save(
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                metrics=val_metrics,
            ),
            last_path,
        )
        row = {
            "epoch": int(epoch),
            "improved": bool(improved),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"model_val_{key}": value for key, value in val_metrics.items()},
        }
        curves.append(row)
        _write_json(output_dir / "training_curves.json", {"epochs": curves})
        _write_epoch_csv(diagnostics_dir / "epoch_metrics.csv", curves)
        if int(config.early_stop_patience) > 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    if best_metrics is None:
        raise RuntimeError("training did not produce validation metrics")

    model_val_report = {
        "split": config.val_split,
        "metrics": best_metrics,
        "selected_epoch": int(best_epoch),
        "selection_metric": config.selection_metric,
        "selection_metric_value": float(best_value),
        "dataset_metadata": val_dataset.to_metadata(),
    }
    _write_json(output_dir / "model_val_diagnostics.json", model_val_report)

    report = {
        "ok": True,
        "experiment_step": TARGET_DENOISING_STEP3,
        "output_contract": TARGET_DENOISING_TRAINING_CONTRACT,
        "model_contract": TARGET_DENOISING_MODEL_CONTRACT,
        "output_dir": str(output_dir),
        "checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "run_report": str(output_dir / "run_report.json"),
        "training_curves": str(output_dir / "training_curves.json"),
        "model_val_diagnostics": str(output_dir / "model_val_diagnostics.json"),
        "best_epoch": int(best_epoch),
        "selection_metric": config.selection_metric,
        "best_model_selection_metric_value": float(best_value),
        "best_model_val_metrics": best_metrics,
        "target_names": list(DENOISING_TARGET_NAMES),
        "train_dataset": train_dataset.to_metadata(),
        "model_val_dataset": val_dataset.to_metadata(),
        "config": asdict(config),
        "model_config": asdict(config.model_config()),
        "final_test_evaluated": False,
        "final_test_note": "Step 3 pretraining trains on model_train and selects on model_val only.",
    }
    _write_json(output_dir / "run_report.json", report)
    return report


__all__ = [
    "TARGET_DENOISING_LOWER_IS_BETTER",
    "TARGET_DENOISING_SELECTION_METRICS",
    "TARGET_DENOISING_STEP3",
    "TARGET_DENOISING_TRAINING_CONTRACT",
    "TargetDenoisingPretrainConfig",
    "run_target_denoising_epoch",
    "target_denoising_loss",
    "train_target_conditioned_denoiser",
]
