"""Training loop for set-matching reconstructors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json, set_training_seed
from jetclass_fresh.jetclass_data import LABEL_NAMES, manifest_hash

from .data import (
    SetMatchingJetDataset,
    audit_set_matching_pair,
    load_manifest_for_set_matching,
    load_set_matching_dataset,
    make_set_matching_loader,
)
from .experiment import SPLIT_SIZES, normalize_split_name
from .losses import SetMatchingLossConfig, compute_set_matching_loss
from .reconstructors import (
    SET_MATCHING_RECONSTRUCTOR_CONTRACT,
    SET_MATCHING_RECONSTRUCTOR_STEP,
    SetMatchingReconstructorAdapter,
    SetMatchingReconstructorConfig,
    build_set_matching_reconstructor,
    normalize_set_matching_reconstructor_architecture,
    set_matching_reconstructor_checkpoint_payload,
)


SET_MATCHING_TRAIN_STEP = "set_matching_multiview_step5_train_reconstructor"


def source_metadata(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Record source commit/status if git is available."""

    repo = Path(repo_root or Path(__file__).resolve().parents[2])
    metadata = {"source_commit": "unknown", "source_status_hash": "unknown"}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        metadata["source_commit"] = commit
    except Exception:
        pass
    try:
        status = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--short"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        metadata["source_status_hash"] = hashlib.sha256(status.encode("utf-8")).hexdigest()
    except Exception:
        pass
    return metadata


def _positive_int_tuple(value: Any, *, field_name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        output = (int(value),)
    else:
        output = tuple(int(item) for item in value)
    if not output:
        raise ValueError(f"{field_name} must contain at least one value")
    if any(item <= 0 for item in output):
        raise ValueError(f"{field_name} must contain only positive values")
    return output


def _optional_nonnegative_int(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative when provided")
    return value


@dataclass
class SetMatchingReconstructorTrainConfig:
    """Configuration for Step 5 set-matching reconstructor training."""

    output_dir: str
    manifest_path: str
    hlt_cache_dir: str
    data_dir: str | None = None
    architecture: str = "gt"
    train_split: str = "model_train"
    val_split: str = "model_val"
    confirm_split_settings: bool = False
    seed: int = 1205
    batch_size: int = 64
    epochs: int = 20
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 5
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_train_jets: int | None = None
    max_val_jets: int | None = None
    label_filter: tuple[int, ...] = ()
    trim_to_valid: bool = True
    verify_hlt_hash: bool = True
    verify_label_branches: bool = False
    read_chunk_size: int = 50_000
    compile_model: bool = False

    # Global Transformer encoder.
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4

    # ParticleNet encoder.
    edgeconv_dims: tuple[int, ...] = (64, 128, 128)
    k: int = 16

    # PFN encoder.
    phi_dims: tuple[int, ...] = (128, 128, 128)

    # PFN/P-CNN shared context settings.
    context_dim: int = 256
    context_mlp_dims: tuple[int, ...] = (256, 256)

    # P-CNN encoder.
    hidden_channels: int = 128
    num_blocks: int = 6
    kernel_sizes: tuple[int, ...] = (5, 5, 3, 3, 3, 3)
    dilations: tuple[int, ...] = (1, 2, 4, 1, 2, 4)

    # Shared aggressive generation head reused by set matching.
    embedding_dim: int = 128
    dropout: float = 0.05
    num_extra_candidates: int = 64
    max_delta_logpt: float = 1.0
    max_delta_eta: float = 0.20
    max_delta_phi: float = 0.20
    max_delta_loge: float = 1.0
    parent_weight_bias: float = 2.0
    extra_weight_bias: float = -2.0
    max_total_extra_pt_fraction: float = 0.50
    max_extra_delta_eta: float = 1.50
    max_extra_delta_phi: float = 1.50
    max_global_logpt_scale: float = 0.35
    max_global_loge_scale: float = 0.35
    max_global_eta_shift: float = 0.05
    max_global_phi_shift: float = 0.05
    extra_usage_weight_threshold: float = 0.05
    eta_limit: float = 5.0
    min_pt: float = 1.0e-4
    weight_logit_epsilon: float = 1.0e-4
    output_weight_threshold: float = 0.0

    # Set-matching loss weights.
    matched_core_weight: float = 1.0
    matched_aux_weight: float = 0.10
    existence_weight: float = 1.0
    existence_positive_weight: float = 1.0
    count_weight: float = 0.10
    jet_summary_weight: float = 0.05
    correction_budget_weight: float = 0.02
    chamfer_weight: float = 0.0
    missing_target_weight: float = 0.25
    huber_beta: float = 1.0
    max_slots: int | None = None
    max_abs_eta: float = 5.0
    hlt_support_budget_weight: float = 0.0
    max_nearest_hlt_delta_r: float = 0.8
    use_core_normalization: bool = True
    brute_force_fallback_limit: int = 8

    def __post_init__(self) -> None:
        self.architecture = normalize_set_matching_reconstructor_architecture(self.architecture)
        self.train_split = normalize_split_name(self.train_split)
        self.val_split = normalize_split_name(self.val_split)
        if self.train_split != "model_train" or self.val_split != "model_val":
            raise ValueError("Step 5 trains only on model_train and selects only on model_val")
        if not bool(self.confirm_split_settings):
            raise ValueError("Set --confirm-split-settings to acknowledge model_train/model_val-only training")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(self.epochs) <= 0:
            raise ValueError("epochs must be positive")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be positive")
        if int(self.num_workers) < 0:
            raise ValueError("num_workers cannot be negative")
        if int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(self.num_layers) <= 0:
            raise ValueError("num_layers must be positive")
        if int(self.num_heads) <= 0:
            raise ValueError("num_heads must be positive")
        if int(self.hidden_dim) % int(self.num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if int(self.k) <= 0:
            raise ValueError("k must be positive")
        if int(self.context_dim) <= 0:
            raise ValueError("context_dim must be positive")
        if int(self.hidden_channels) <= 0:
            raise ValueError("hidden_channels must be positive")
        if int(self.num_blocks) <= 0:
            raise ValueError("num_blocks must be positive")
        if len(self.kernel_sizes) != int(self.num_blocks):
            raise ValueError("kernel_sizes length must match num_blocks")
        if len(self.dilations) != int(self.num_blocks):
            raise ValueError("dilations length must match num_blocks")
        if any(int(kernel) % 2 == 0 for kernel in self.kernel_sizes):
            raise ValueError("kernel_sizes must contain odd values")
        if int(self.embedding_dim) <= 0:
            raise ValueError("embedding_dim must be positive")
        if int(self.num_extra_candidates) < 0:
            raise ValueError("num_extra_candidates must be non-negative")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        for field_name in ("max_train_batches", "max_val_batches", "max_train_jets", "max_val_jets", "max_slots"):
            setattr(self, field_name, _optional_nonnegative_int(getattr(self, field_name), field_name=field_name))
        if self.max_slots == 0:
            raise ValueError("max_slots must be positive when provided")
        self.label_filter = tuple(int(label) for label in self.label_filter)
        if len(set(self.label_filter)) != len(self.label_filter):
            raise ValueError(f"label_filter contains duplicates: {self.label_filter}")
        if int(self.brute_force_fallback_limit) < 1:
            raise ValueError("brute_force_fallback_limit must be at least 1")
        self.edgeconv_dims = _positive_int_tuple(self.edgeconv_dims, field_name="edgeconv_dims")
        self.phi_dims = _positive_int_tuple(self.phi_dims, field_name="phi_dims")
        self.context_mlp_dims = _positive_int_tuple(self.context_mlp_dims, field_name="context_mlp_dims")
        self.kernel_sizes = _positive_int_tuple(self.kernel_sizes, field_name="kernel_sizes")
        self.dilations = _positive_int_tuple(self.dilations, field_name="dilations")

    def aggressive_head_config(self) -> dict[str, Any]:
        return {
            "num_extra_candidates": int(self.num_extra_candidates),
            "dropout": float(self.dropout),
            "max_delta_logpt": float(self.max_delta_logpt),
            "max_delta_eta": float(self.max_delta_eta),
            "max_delta_phi": float(self.max_delta_phi),
            "max_delta_loge": float(self.max_delta_loge),
            "parent_weight_bias": float(self.parent_weight_bias),
            "extra_weight_bias": float(self.extra_weight_bias),
            "max_total_extra_pt_fraction": float(self.max_total_extra_pt_fraction),
            "max_extra_delta_eta": float(self.max_extra_delta_eta),
            "max_extra_delta_phi": float(self.max_extra_delta_phi),
            "max_global_logpt_scale": float(self.max_global_logpt_scale),
            "max_global_loge_scale": float(self.max_global_loge_scale),
            "max_global_eta_shift": float(self.max_global_eta_shift),
            "max_global_phi_shift": float(self.max_global_phi_shift),
            "extra_usage_weight_threshold": float(self.extra_usage_weight_threshold),
            "eta_limit": float(self.eta_limit),
            "min_pt": float(self.min_pt),
        }

    def model_config(self) -> dict[str, Any]:
        head_config = self.aggressive_head_config()
        if self.architecture == "gt":
            return {
                "hidden_dim": int(self.hidden_dim),
                "num_layers": int(self.num_layers),
                "num_heads": int(self.num_heads),
                "dropout": float(self.dropout),
                "aggressive_head_config": head_config,
            }
        if self.architecture == "pn":
            return {
                "edgeconv_dims": tuple(self.edgeconv_dims),
                "k": int(self.k),
                "dropout": float(self.dropout),
                "embedding_dim": int(self.embedding_dim),
                "aggressive_head_config": head_config,
            }
        if self.architecture == "pfn":
            return {
                "phi_dims": tuple(self.phi_dims),
                "context_dim": int(self.context_dim),
                "context_mlp_dims": tuple(self.context_mlp_dims),
                "dropout": float(self.dropout),
                "embedding_dim": int(self.embedding_dim),
                "aggressive_head_config": head_config,
            }
        if self.architecture == "pcnn":
            return {
                "hidden_channels": int(self.hidden_channels),
                "num_blocks": int(self.num_blocks),
                "kernel_sizes": tuple(self.kernel_sizes),
                "dilations": tuple(self.dilations),
                "context_dim": int(self.context_dim),
                "context_mlp_dims": tuple(self.context_mlp_dims),
                "dropout": float(self.dropout),
                "embedding_dim": int(self.embedding_dim),
                "aggressive_head_config": head_config,
            }
        raise ValueError(f"Unknown architecture {self.architecture!r}")

    def wrapper_config(self) -> SetMatchingReconstructorConfig:
        return SetMatchingReconstructorConfig(
            architecture=self.architecture,
            model_config=self.model_config(),
            weight_logit_epsilon=float(self.weight_logit_epsilon),
            output_weight_threshold=float(self.output_weight_threshold),
        )

    def loss_config(self, *, core_mean: tuple[float, float, float, float] | None = None, core_std=None) -> SetMatchingLossConfig:
        return SetMatchingLossConfig(
            core_mean=core_mean,
            core_std=core_std,
            matched_core_weight=float(self.matched_core_weight),
            matched_aux_weight=float(self.matched_aux_weight),
            existence_weight=float(self.existence_weight),
            existence_positive_weight=float(self.existence_positive_weight),
            count_weight=float(self.count_weight),
            jet_summary_weight=float(self.jet_summary_weight),
            correction_budget_weight=float(self.correction_budget_weight),
            chamfer_weight=float(self.chamfer_weight),
            missing_target_weight=float(self.missing_target_weight),
            huber_beta=float(self.huber_beta),
            max_active_slots=self.max_slots,
            max_abs_eta=float(self.max_abs_eta),
            hlt_support_budget_weight=float(self.hlt_support_budget_weight),
            max_nearest_hlt_delta_r=float(self.max_nearest_hlt_delta_r),
            brute_force_fallback_limit=int(self.brute_force_fallback_limit),
        )


def compute_core_normalization_from_dataset(dataset: SetMatchingJetDataset) -> dict[str, Any]:
    """Compute log-pt/eta/phi/log-energy stats from model_train offline particles."""

    tokens = dataset.offline_tokens
    mask = dataset.offline_mask
    valid = tokens[mask]
    if int(valid.shape[0]) == 0:
        raise ValueError("cannot compute core normalization from an empty offline target set")
    values = np.stack(
        [
            np.log(np.clip(valid[:, 0], 1.0e-6, None)),
            valid[:, 1],
            valid[:, 2],
            np.log(np.clip(valid[:, 3], 1.0e-6, None)),
        ],
        axis=1,
    ).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std = np.where(std <= 1.0e-6, 1.0, std).astype(np.float32)
    return {
        "source": "model_train_offline_core_features",
        "count": int(values.shape[0]),
        "core_feature_names": ["log_pt", "eta", "phi", "log_energy"],
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }


class _MetricAccumulator:
    def __init__(self) -> None:
        self.weight_sum = 0.0
        self.totals: dict[str, float] = {}

    def update(self, metrics: Mapping[str, Any], *, weight: int) -> None:
        weight = int(weight)
        if weight <= 0:
            return
        self.weight_sum += float(weight)
        for key, value in metrics.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric):
                continue
            self.totals[key] = self.totals.get(key, 0.0) + numeric * float(weight)

    def summary(self) -> dict[str, float]:
        if self.weight_sum <= 0.0:
            return {}
        return {key: value / self.weight_sum for key, value in sorted(self.totals.items())}


def _move_batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("hlt_tokens", "hlt_mask", "offline_tokens", "offline_mask", "labels", "indices"):
        value = moved.get(key)
        if hasattr(value, "to"):
            moved[key] = value.to(device)
    return moved


def _write_epoch_metrics_csv(path: Path, curves: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: set[str] = {"epoch"}
    rows = []
    for row in curves:
        flat = {"epoch": row["epoch"]}
        for split in ("train", "model_val"):
            for key, value in row.get(split, {}).items():
                flat[f"{split}_{key}"] = value
                keys.add(f"{split}_{key}")
        rows.append(flat)
    ordered = ["epoch"] + sorted(key for key in keys if key != "epoch")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def run_set_matching_reco_epoch(
    model: SetMatchingReconstructorAdapter,
    loader,
    *,
    device,
    loss_config: SetMatchingLossConfig,
    optimizer=None,
    scaler=None,
    amp: bool = True,
    grad_clip_norm: float | None = 1.0,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run one train or validation epoch."""

    torch = require_torch()
    training = optimizer is not None
    model.train(training)
    accumulator = _MetricAccumulator()
    num_batches = 0
    num_jets = 0
    autocast_enabled = bool(amp and device.type == "cuda")

    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > int(max_batches):
            break
        batch = _move_batch_to_device(batch, device)
        labels = batch["labels"]
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            output = model(
                batch["hlt_tokens"],
                batch["hlt_mask"],
                labels=labels,
                jet_ids=batch.get("jet_ids"),
                split=str(batch.get("split", "in_memory")),
            )
            loss_output = compute_set_matching_loss(
                **output.to_loss_kwargs(
                    offline_features=batch["offline_tokens"],
                    offline_mask=batch["offline_mask"],
                    hlt_features=batch["hlt_tokens"],
                    hlt_mask=batch["hlt_mask"],
                ),
                config=loss_config,
            )
            loss = loss_output.total_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                optimizer.step()

        batch_size = int(labels.shape[0])
        metrics = loss_output.detached_float_dict(prefix="")
        metrics.update({f"reco_{key}": value for key, value in output.diagnostics.items() if isinstance(value, (int, float))})
        accumulator.update(metrics, weight=batch_size)
        num_batches += 1
        num_jets += batch_size

    summary = accumulator.summary()
    summary["num_batches"] = float(num_batches)
    summary["num_jets"] = float(num_jets)
    return summary


def set_matching_checkpoint_payload(
    model: SetMatchingReconstructorAdapter,
    optimizer,
    *,
    epoch: int,
    config: SetMatchingReconstructorTrainConfig,
    loss_config: SetMatchingLossConfig,
    metrics: Mapping[str, Any],
    source: Mapping[str, Any],
    core_normalization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return set_matching_reconstructor_checkpoint_payload(
        model,
        extra_payload={
            "epoch": int(epoch),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "loss_config": loss_config.to_dict(),
            "metrics": dict(metrics),
            "label_names": list(LABEL_NAMES),
            "source": dict(source),
            "core_normalization": None if core_normalization is None else dict(core_normalization),
            "experiment_step": SET_MATCHING_TRAIN_STEP,
            "reconstructor_step": SET_MATCHING_RECONSTRUCTOR_STEP,
        },
    )


def _load_train_val_datasets(config: SetMatchingReconstructorTrainConfig) -> tuple[SetMatchingJetDataset, SetMatchingJetDataset]:
    train_dataset = load_set_matching_dataset(
        manifest_path=config.manifest_path,
        hlt_cache_dir=config.hlt_cache_dir,
        split=config.train_split,
        data_dir=config.data_dir,
        max_jets=config.max_train_jets,
        label_filter=config.label_filter,
        trim_to_valid=bool(config.trim_to_valid),
        verify_hlt_hash=bool(config.verify_hlt_hash),
        verify_label_branches=bool(config.verify_label_branches),
        read_chunk_size=int(config.read_chunk_size),
    )
    val_dataset = load_set_matching_dataset(
        manifest_path=config.manifest_path,
        hlt_cache_dir=config.hlt_cache_dir,
        split=config.val_split,
        data_dir=config.data_dir,
        max_jets=config.max_val_jets,
        label_filter=config.label_filter,
        trim_to_valid=bool(config.trim_to_valid),
        verify_hlt_hash=bool(config.verify_hlt_hash),
        verify_label_branches=bool(config.verify_label_branches),
        read_chunk_size=int(config.read_chunk_size),
    )
    return train_dataset, val_dataset


def train_set_matching_reconstructor(
    config: SetMatchingReconstructorTrainConfig,
    *,
    model: SetMatchingReconstructorAdapter | None = None,
    train_dataset: SetMatchingJetDataset | None = None,
    val_dataset: SetMatchingJetDataset | None = None,
) -> dict[str, Any]:
    """Train one gt/pn/pfn/pcnn set-matching reconstructor."""

    torch = require_torch()
    set_training_seed(int(config.seed))
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    if train_dataset is None or val_dataset is None:
        train_dataset, val_dataset = _load_train_val_datasets(config)

    manifest = None
    manifest_sha = None
    try:
        manifest = load_manifest_for_set_matching(config.manifest_path)
        manifest_sha = manifest_hash(manifest)
    except Exception:
        manifest = None
        manifest_sha = None

    train_audit = audit_set_matching_pair(train_dataset, manifest=manifest, expected_split=config.train_split)
    val_audit = audit_set_matching_pair(val_dataset, manifest=manifest, expected_split=config.val_split)
    save_json(diagnostics_dir / "data_audit.json", {"train": train_audit, "model_val": val_audit})
    if not train_audit.get("ok", False) or not val_audit.get("ok", False):
        raise ValueError(f"Set-matching data audit failed: train={train_audit['problems']} val={val_audit['problems']}")

    core_normalization = compute_core_normalization_from_dataset(train_dataset) if config.use_core_normalization else None
    loss_config = config.loss_config(
        core_mean=None if core_normalization is None else tuple(core_normalization["mean"]),
        core_std=None if core_normalization is None else tuple(core_normalization["std"]),
    )

    train_loader = make_set_matching_loader(
        train_dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        seed=int(config.seed),
    )
    val_loader = make_set_matching_loader(
        val_dataset,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 1,
    )

    checkpoint_model = model or build_set_matching_reconstructor(config=config.wrapper_config())
    checkpoint_model = checkpoint_model.to(device)
    reconstructor = checkpoint_model
    if config.compile_model and hasattr(torch, "compile"):
        reconstructor = torch.compile(reconstructor)

    optimizer = torch.optim.AdamW(
        checkpoint_model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))
    source = source_metadata()
    run_metadata = {
        "experiment_step": SET_MATCHING_TRAIN_STEP,
        "reconstructor_step": SET_MATCHING_RECONSTRUCTOR_STEP,
        "output_contract": SET_MATCHING_RECONSTRUCTOR_CONTRACT,
        "architecture": config.architecture,
        "config": asdict(config),
        "model_config": checkpoint_model.to_config_dict(),
        "loss_config": loss_config.to_dict(),
        "core_normalization": core_normalization,
        "source": source,
        "manifest_hash": manifest_sha,
        "expected_split_sizes": dict(SPLIT_SIZES),
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "train_audit": train_audit,
        "model_val_audit": val_audit,
        "leakage_rule": (
            "Step 5 trains set-matching reconstructors only on model_train and selects only on model_val. "
            "Offline constituents are used only as train/validation set targets. "
            "No stack_train, stack_val, or final_test rows are loaded."
        ),
        "no_stack_or_final_test_partitions_loaded": True,
    }
    save_json(output_dir / "config.json", run_metadata)

    curves: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, int(config.epochs) + 1):
        train_metrics = run_set_matching_reco_epoch(
            reconstructor,
            train_loader,
            device=device,
            loss_config=loss_config,
            optimizer=optimizer,
            scaler=scaler,
            amp=bool(config.amp),
            grad_clip_norm=float(config.grad_clip_norm),
            max_batches=config.max_train_batches,
        )
        val_metrics = run_set_matching_reco_epoch(
            reconstructor,
            val_loader,
            device=device,
            loss_config=loss_config,
            amp=False,
            max_batches=config.max_val_batches,
        )
        row = {"epoch": int(epoch), "train": train_metrics, "model_val": val_metrics}
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves})
        _write_epoch_metrics_csv(diagnostics_dir / "epoch_metrics.csv", curves)

        val_loss = float(val_metrics.get("total", float("nan")))
        improved = np.isfinite(val_loss) and val_loss < best_val_loss
        payload = set_matching_checkpoint_payload(
            checkpoint_model,
            optimizer,
            epoch=epoch,
            config=config,
            loss_config=loss_config,
            metrics=row,
            source=source,
            core_normalization=core_normalization,
        )
        torch.save(payload, output_dir / "last.pt")
        if improved:
            best_val_loss = val_loss
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(payload, output_dir / "best_model_val.pt")
        else:
            epochs_without_improvement += 1

        if int(config.early_stop_patience) >= 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    if best_epoch < 0 or not (output_dir / "best_model_val.pt").exists():
        raise FloatingPointError(
            f"Set-matching reconstructor {config.architecture} did not produce a finite model_val total loss, "
            "so no best_model_val.pt was written"
        )

    best_val_metrics = curves[best_epoch - 1]["model_val"]
    report = {
        "experiment_step": SET_MATCHING_TRAIN_STEP,
        "reconstructor_step": SET_MATCHING_RECONSTRUCTOR_STEP,
        "output_contract": SET_MATCHING_RECONSTRUCTOR_CONTRACT,
        "architecture": config.architecture,
        "best_epoch": int(best_epoch),
        "best_model_val_total_loss": float(best_val_loss),
        "best_model_val_metrics": best_val_metrics,
        "epochs_completed": len(curves),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "training_curves": str(output_dir / "training_curves.json"),
        "diagnostic_csv": str(diagnostics_dir / "epoch_metrics.csv"),
        "data_audit": str(diagnostics_dir / "data_audit.json"),
        "core_normalization": core_normalization,
        "source": source,
        "not_a_classifier": True,
        "inference_consumes_hlt_only": True,
        "no_final_test_evaluation": True,
    }
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    return report


__all__ = [
    "SET_MATCHING_TRAIN_STEP",
    "SetMatchingReconstructorTrainConfig",
    "compute_core_normalization_from_dataset",
    "run_set_matching_reco_epoch",
    "set_matching_checkpoint_payload",
    "train_set_matching_reconstructor",
]
