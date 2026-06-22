"""Training loop for DETR/free-slot reconstructors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import importlib.util
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json, set_training_seed
from jetclass_fresh.jetclass_data import LABEL_NAMES, RAW_TOKEN_DIM, manifest_hash

from teacher_logit_reco.set_matching.data import (
    SetMatchingJetDataset,
    audit_set_matching_pair,
    load_manifest_for_set_matching,
    load_set_matching_dataset,
    make_set_matching_loader,
)
from teacher_logit_reco.set_matching.detr_slots.decoder import (
    DetrPredictionHeads,
    DetrPredictionHeadsConfig,
    DetrSlotDecoder,
    DetrSlotDecoderConfig,
)
from teacher_logit_reco.set_matching.detr_slots.encoders import (
    GlobalTransformerHLTEncoderAdapter,
    GlobalTransformerHLTEncoderConfig,
    ParticleCnnHLTEncoderAdapter,
    ParticleCnnHLTEncoderConfig,
    ParticleFlowHLTEncoderAdapter,
    ParticleFlowHLTEncoderConfig,
    ParticleNetHLTEncoderAdapter,
    ParticleNetHLTEncoderConfig,
)
from teacher_logit_reco.set_matching.detr_slots.experiment import (
    DETR_SLOT_RECONSTRUCTOR_CONTRACT,
    normalize_detr_slot_encoder_architecture,
    normalize_detr_slot_split_name,
)
from teacher_logit_reco.set_matching.detr_slots.features import DetrSlotFeatureConfig
from teacher_logit_reco.set_matching.detr_slots.losses import (
    DetrSlotHungarianLossConfig,
    compute_detr_slot_hungarian_loss,
)
from teacher_logit_reco.set_matching.train import source_metadata


DETR_SLOT_TRAIN_STEP = "detr_free_slot_step11_train_reconstructor"

if importlib.util.find_spec("torch") is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    import torch as _torch

    _ModuleBase = _torch.nn.Module


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
class DetrSlotReconstructorTrainConfig:
    """Configuration for Step 11 DETR/free-slot reconstructor training."""

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

    # Shared DETR dimensions.
    feature_dim: int = RAW_TOKEN_DIM
    num_slots: int = 160
    export_max_tokens: int = 128
    memory_dim: int = 128
    context_dim: int | None = None
    embed_dim: int = 128
    dropout: float = 0.05
    max_abs_eta: float = 5.0

    # Decoder and heads.
    decoder_layers: int = 4
    decoder_heads: int = 4
    decoder_mlp_ratio: float = 4.0
    head_hidden_dim: int = 256
    existence_bias: float = -2.0
    core_output_scale: float = 1.0

    # Global Transformer encoder.
    gt_layers: int = 4
    gt_heads: int = 4
    gt_mlp_ratio: float = 4.0

    # ParticleNet encoder.
    edgeconv_dims: tuple[int, ...] = (64, 128, 128)
    k: int = 16

    # PFN encoder.
    phi_dims: tuple[int, ...] = (128, 128, 128)

    # PFN/P-CNN shared context settings.
    context_mlp_dims: tuple[int, ...] = (256, 256)

    # P-CNN encoder.
    hidden_channels: int = 128
    kernel_sizes: tuple[int, ...] = (5, 5, 3, 3, 3, 3)
    dilations: tuple[int, ...] = (1, 2, 4, 1, 2, 4)

    # Hungarian loss.
    assignment_aux_weight: float = 0.05
    matched_core_weight: float = 1.0
    matched_aux_weight: float = 0.10
    existence_weight: float = 1.0
    existence_positive_weight: float = 1.0
    existence_negative_weight: float = 0.20
    count_weight: float = 0.10
    jet_summary_weight: float = 0.05
    duplicate_weight: float = 0.0
    hlt_support_weight: float = 0.0
    max_nearest_hlt_delta_r: float = 0.8
    duplicate_delta_r_scale: float = 0.10
    duplicate_probability_threshold: float = 0.25
    max_count_for_summary: float = 128.0
    huber_beta: float = 1.0
    brute_force_fallback_limit: int = 8
    allow_bruteforce_fallback: bool = False

    def __post_init__(self) -> None:
        self.architecture = normalize_detr_slot_encoder_architecture(self.architecture)
        self.train_split = normalize_detr_slot_split_name(self.train_split)
        self.val_split = normalize_detr_slot_split_name(self.val_split)
        if self.train_split != "model_train" or self.val_split != "model_val":
            raise ValueError("Step 11 trains only on model_train and selects only on model_val")
        if not bool(self.confirm_split_settings):
            raise ValueError("Set --confirm-split-settings to acknowledge model_train/model_val-only training")
        for field_name in (
            "batch_size",
            "epochs",
            "feature_dim",
            "num_slots",
            "export_max_tokens",
            "memory_dim",
            "embed_dim",
            "decoder_layers",
            "decoder_heads",
            "head_hidden_dim",
            "gt_layers",
            "gt_heads",
            "k",
            "hidden_channels",
            "brute_force_fallback_limit",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if int(self.feature_dim) != RAW_TOKEN_DIM:
            raise ValueError(f"feature_dim must be {RAW_TOKEN_DIM}, got {self.feature_dim}")
        if int(self.export_max_tokens) > int(self.num_slots):
            raise ValueError("export_max_tokens cannot exceed num_slots")
        if self.context_dim is not None and int(self.context_dim) <= 0:
            raise ValueError("context_dim must be positive when provided")
        if int(self.embed_dim) % int(self.decoder_heads) != 0:
            raise ValueError("embed_dim must be divisible by decoder_heads")
        if int(self.memory_dim) % int(self.gt_heads) != 0:
            raise ValueError("memory_dim must be divisible by gt_heads")
        if int(self.num_workers) < 0:
            raise ValueError("num_workers cannot be negative")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if float(self.max_abs_eta) <= 0.0:
            raise ValueError("max_abs_eta must be positive")
        if float(self.decoder_mlp_ratio) <= 0.0 or float(self.gt_mlp_ratio) <= 0.0:
            raise ValueError("mlp ratios must be positive")
        if float(self.core_output_scale) <= 0.0:
            raise ValueError("core_output_scale must be positive")
        for field_name in ("max_train_batches", "max_val_batches", "max_train_jets", "max_val_jets"):
            setattr(self, field_name, _optional_nonnegative_int(getattr(self, field_name), field_name=field_name))
        self.label_filter = tuple(int(label) for label in self.label_filter)
        if len(set(self.label_filter)) != len(self.label_filter):
            raise ValueError(f"label_filter contains duplicates: {self.label_filter}")
        self.edgeconv_dims = _positive_int_tuple(self.edgeconv_dims, field_name="edgeconv_dims")
        self.phi_dims = _positive_int_tuple(self.phi_dims, field_name="phi_dims")
        self.context_mlp_dims = _positive_int_tuple(self.context_mlp_dims, field_name="context_mlp_dims")
        self.kernel_sizes = _positive_int_tuple(self.kernel_sizes, field_name="kernel_sizes")
        self.dilations = _positive_int_tuple(self.dilations, field_name="dilations")
        if len(self.kernel_sizes) != len(self.dilations):
            raise ValueError("kernel_sizes and dilations must have the same length")
        if any(int(kernel) % 2 == 0 for kernel in self.kernel_sizes):
            raise ValueError("kernel_sizes must contain odd values")

    @property
    def resolved_context_dim(self) -> int:
        return int(self.memory_dim if self.context_dim is None else self.context_dim)

    def feature_config(self) -> DetrSlotFeatureConfig:
        return DetrSlotFeatureConfig(feature_dim=int(self.feature_dim), max_abs_eta=float(self.max_abs_eta))

    def encoder_config(self) -> dict[str, Any]:
        base = {
            "input_dim": int(self.feature_dim),
            "memory_dim": int(self.memory_dim),
            "context_dim": int(self.resolved_context_dim),
            "dropout": float(self.dropout),
            "max_abs_eta": float(self.max_abs_eta),
        }
        if self.architecture == "gt":
            return {
                **base,
                "num_layers": int(self.gt_layers),
                "num_heads": int(self.gt_heads),
                "mlp_ratio": float(self.gt_mlp_ratio),
            }
        if self.architecture == "pn":
            return {**base, "edgeconv_dims": tuple(self.edgeconv_dims), "k": int(self.k)}
        if self.architecture == "pfn":
            return {
                **base,
                "phi_dims": tuple(self.phi_dims),
                "context_mlp_dims": tuple(self.context_mlp_dims),
            }
        if self.architecture == "pcnn":
            return {
                **base,
                "hidden_channels": int(self.hidden_channels),
                "kernel_sizes": tuple(self.kernel_sizes),
                "dilations": tuple(self.dilations),
                "context_mlp_dims": tuple(self.context_mlp_dims),
            }
        raise ValueError(f"Unknown DETR-slot architecture {self.architecture!r}")

    def decoder_config(self) -> dict[str, Any]:
        return {
            "num_slots": int(self.num_slots),
            "embed_dim": int(self.embed_dim),
            "memory_dim": int(self.memory_dim),
            "context_dim": int(self.resolved_context_dim),
            "num_layers": int(self.decoder_layers),
            "num_heads": int(self.decoder_heads),
            "mlp_ratio": float(self.decoder_mlp_ratio),
            "dropout": float(self.dropout),
        }

    def heads_config(self) -> dict[str, Any]:
        return {
            "embed_dim": int(self.embed_dim),
            "hidden_dim": int(self.head_hidden_dim),
            "context_dim": int(self.resolved_context_dim),
            "feature_config": self.feature_config(),
            "dropout": float(self.dropout),
            "existence_bias": float(self.existence_bias),
            "core_output_scale": float(self.core_output_scale),
        }

    def loss_config(self) -> DetrSlotHungarianLossConfig:
        return DetrSlotHungarianLossConfig(
            feature_config=self.feature_config(),
            assignment_aux_weight=float(self.assignment_aux_weight),
            matched_core_weight=float(self.matched_core_weight),
            matched_aux_weight=float(self.matched_aux_weight),
            existence_weight=float(self.existence_weight),
            existence_positive_weight=float(self.existence_positive_weight),
            existence_negative_weight=float(self.existence_negative_weight),
            count_weight=float(self.count_weight),
            jet_summary_weight=float(self.jet_summary_weight),
            duplicate_weight=float(self.duplicate_weight),
            hlt_support_weight=float(self.hlt_support_weight),
            max_nearest_hlt_delta_r=float(self.max_nearest_hlt_delta_r),
            duplicate_delta_r_scale=float(self.duplicate_delta_r_scale),
            duplicate_probability_threshold=float(self.duplicate_probability_threshold),
            max_count_for_summary=float(self.max_count_for_summary),
            huber_beta=float(self.huber_beta),
            brute_force_fallback_limit=int(self.brute_force_fallback_limit),
            allow_bruteforce_fallback=bool(self.allow_bruteforce_fallback),
        )


class DetrSlotReconstructor(_ModuleBase):
    """HLT -> encoder memory -> DETR decoder slots -> raw reconstructed tokens."""

    def __init__(
        self,
        *,
        architecture: str,
        encoder,
        decoder: DetrSlotDecoder,
        heads: DetrPredictionHeads,
        config: DetrSlotReconstructorTrainConfig,
    ) -> None:
        super().__init__()
        self.architecture = normalize_detr_slot_encoder_architecture(architecture)
        self.encoder = encoder
        self.decoder = decoder
        self.heads = heads
        self.config = config

    def forward(self, hlt_tokens, hlt_mask=None, **_: Any):
        encoded = self.encoder(hlt_tokens, hlt_mask)
        slots = self.decoder(
            encoded.memory_tokens,
            encoded.memory_mask,
            global_context=encoded.global_context,
        )
        aux = {
            "train_step": DETR_SLOT_TRAIN_STEP,
            "encoder_architecture": self.architecture,
            **dict(encoded.aux),
        }
        return self.heads(
            slots,
            global_context=encoded.global_context,
            aux=aux,
        )

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "encoder_config": _module_config_dict(self.encoder),
            "decoder_config": self.decoder.config.to_dict(),
            "heads_config": self.heads.config.to_dict(),
            "contract": DETR_SLOT_RECONSTRUCTOR_CONTRACT,
            "train_step": DETR_SLOT_TRAIN_STEP,
        }


def _module_config_dict(module) -> dict[str, Any]:
    config = getattr(module, "config", None)
    if config is None:
        return {}
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return asdict(config)


def build_detr_slot_encoder(config: DetrSlotReconstructorTrainConfig):
    payload = config.encoder_config()
    if config.architecture == "gt":
        return GlobalTransformerHLTEncoderAdapter(GlobalTransformerHLTEncoderConfig(**payload))
    if config.architecture == "pn":
        return ParticleNetHLTEncoderAdapter(ParticleNetHLTEncoderConfig(**payload))
    if config.architecture == "pfn":
        return ParticleFlowHLTEncoderAdapter(ParticleFlowHLTEncoderConfig(**payload))
    if config.architecture == "pcnn":
        return ParticleCnnHLTEncoderAdapter(ParticleCnnHLTEncoderConfig(**payload))
    raise ValueError(f"Unknown DETR-slot architecture {config.architecture!r}")


def build_detr_slot_reconstructor(config: DetrSlotReconstructorTrainConfig) -> DetrSlotReconstructor:
    return DetrSlotReconstructor(
        architecture=config.architecture,
        encoder=build_detr_slot_encoder(config),
        decoder=DetrSlotDecoder(DetrSlotDecoderConfig(**config.decoder_config())),
        heads=DetrPredictionHeads(DetrPredictionHeadsConfig(**config.heads_config())),
        config=config,
    )


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


def _write_best_metrics_csv(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "architecture": report.get("architecture"),
        "best_epoch": report.get("best_epoch"),
        "best_model_val_total_loss": report.get("best_model_val_total_loss"),
        "epochs_completed": report.get("epochs_completed"),
    }
    for key, value in dict(report.get("best_model_val_metrics") or {}).items():
        row[f"model_val_{key}"] = value
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _load_train_val_datasets(config: DetrSlotReconstructorTrainConfig) -> tuple[SetMatchingJetDataset, SetMatchingJetDataset]:
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


def run_detr_slot_reco_epoch(
    model,
    loader,
    *,
    device,
    loss_config: DetrSlotHungarianLossConfig,
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
    grad_context = torch.enable_grad if training else torch.no_grad

    for batch_index, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_index > int(max_batches):
            break
        batch = _move_batch_to_device(batch, device)
        labels = batch["labels"]
        with grad_context():
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                output = model(
                    batch["hlt_tokens"],
                    batch["hlt_mask"],
                    labels=labels,
                    jet_ids=batch.get("jet_ids"),
                    split=str(batch.get("split", "in_memory")),
                )
                loss_output = compute_detr_slot_hungarian_loss(
                    **output.to_loss_kwargs(
                        offline_features=batch["offline_tokens"],
                        offline_mask=batch["offline_mask"],
                        hlt_features=batch["hlt_tokens"],
                        hlt_mask=batch["hlt_mask"],
                        include_aux_logits=True,
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


def detr_slot_checkpoint_payload(
    model: DetrSlotReconstructor,
    optimizer,
    *,
    epoch: int,
    config: DetrSlotReconstructorTrainConfig,
    loss_config: DetrSlotHungarianLossConfig,
    metrics: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "model_config": model.to_config_dict(),
        "loss_config": loss_config.to_dict(),
        "metrics": dict(metrics),
        "label_names": list(LABEL_NAMES),
        "source": dict(source),
        "experiment_step": DETR_SLOT_TRAIN_STEP,
        "reconstructor_contract": DETR_SLOT_RECONSTRUCTOR_CONTRACT,
        "architecture": config.architecture,
    }


def train_detr_slot_reconstructor(
    config: DetrSlotReconstructorTrainConfig,
    *,
    model: DetrSlotReconstructor | None = None,
    train_dataset: SetMatchingJetDataset | None = None,
    val_dataset: SetMatchingJetDataset | None = None,
) -> dict[str, Any]:
    """Train one DETR/free-slot reconstructor and select by model_val total loss."""

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
        raise ValueError(f"DETR-slot data audit failed: train={train_audit['problems']} val={val_audit['problems']}")

    loss_config = config.loss_config()
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

    checkpoint_model = model or build_detr_slot_reconstructor(config)
    checkpoint_model = checkpoint_model.to(device)
    train_model = checkpoint_model
    if config.compile_model and hasattr(torch, "compile"):
        train_model = torch.compile(train_model)

    optimizer = torch.optim.AdamW(
        checkpoint_model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))
    source = source_metadata()
    run_metadata = {
        "experiment_step": DETR_SLOT_TRAIN_STEP,
        "reconstructor_contract": DETR_SLOT_RECONSTRUCTOR_CONTRACT,
        "architecture": config.architecture,
        "config": asdict(config),
        "model_config": checkpoint_model.to_config_dict(),
        "loss_config": loss_config.to_dict(),
        "source": source,
        "manifest_hash": manifest_sha,
        "train_dataset": dict(train_dataset.metadata),
        "val_dataset": dict(val_dataset.metadata),
        "train_audit": train_audit,
        "model_val_audit": val_audit,
        "selection_metric": "model_val_total_loss",
        "uses_aux_logits_for_bce": True,
        "export_max_tokens": int(config.export_max_tokens),
        "leakage_rule": (
            "Step 11 trains DETR/free-slot reconstructors only on model_train and selects only on model_val. "
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
        train_metrics = run_detr_slot_reco_epoch(
            train_model,
            train_loader,
            device=device,
            loss_config=loss_config,
            optimizer=optimizer,
            scaler=scaler,
            amp=bool(config.amp),
            grad_clip_norm=float(config.grad_clip_norm),
            max_batches=config.max_train_batches,
        )
        val_metrics = run_detr_slot_reco_epoch(
            train_model,
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
        payload = detr_slot_checkpoint_payload(
            checkpoint_model,
            optimizer,
            epoch=epoch,
            config=config,
            loss_config=loss_config,
            metrics=row,
            source=source,
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
            f"DETR-slot reconstructor {config.architecture} did not produce a finite model_val total loss, "
            "so no best_model_val.pt was written"
        )

    best_val_metrics = curves[best_epoch - 1]["model_val"]
    report = {
        "experiment_step": DETR_SLOT_TRAIN_STEP,
        "reconstructor_contract": DETR_SLOT_RECONSTRUCTOR_CONTRACT,
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
        "best_metrics_csv": str(diagnostics_dir / "best_metrics.csv"),
        "data_audit": str(diagnostics_dir / "data_audit.json"),
        "source": source,
        "uses_aux_logits_for_bce": True,
        "not_a_classifier": True,
        "inference_consumes_hlt_only": True,
        "no_final_test_evaluation": True,
    }
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    _write_best_metrics_csv(diagnostics_dir / "best_metrics.csv", report)
    return report


__all__ = [
    "DETR_SLOT_TRAIN_STEP",
    "DetrSlotReconstructor",
    "DetrSlotReconstructorTrainConfig",
    "build_detr_slot_encoder",
    "build_detr_slot_reconstructor",
    "detr_slot_checkpoint_payload",
    "run_detr_slot_reco_epoch",
    "train_detr_slot_reconstructor",
]
