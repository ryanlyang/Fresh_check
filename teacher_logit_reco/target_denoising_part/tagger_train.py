"""Training loop for target-conditioned denoising ParT taggers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, set_training_seed
from jetclass_fresh.hlt_cache import (
    HLT_PROFILE_V2_REALISTIC,
    HLT_PROFILE_V2_REALISTIC_VERSION,
    jet_identity_hash,
    load_cached_hlt_view,
    normalize_hlt_profile,
)
from jetclass_fresh.jetclass_data import JetIdentity, load_split_manifest, manifest_hash

from .model import TargetConditionedDenoiserConfig
from .data import (
    TargetDenoisingDatasetConfig,
    TargetDenoisingPairedDataset,
    collate_target_denoising_batch,
    load_target_denoising_dataset,
)
from .tagger import (
    TARGET_DENOISING_DENOISER_VARIANTS,
    TARGET_DENOISING_TAGGER_CONTRACT,
    TARGET_DENOISING_TAGGER_VARIANTS,
    TARGET_DENOISING_VARIANT_DENOISER_LOCAL_KERNEL_ONLY,
    TARGET_DENOISING_VARIANT_DENOISER_NO_PAIR_BIAS,
    TARGET_DENOISING_VARIANT_DENOISER_SHUFFLED_TARGETS,
    TARGET_DENOISING_VARIANT_DENOISER_TAG_ONLY_SAME_ARCH,
    TargetDenoisingAugmentedParT,
    TargetDenoisingAugmentedParTConfig,
    TargetDenoisingAugmentedParTOutput,
    load_target_denoising_pretrained_checkpoint,
)


TARGET_DENOISING_STEP6_TAGGER = "target_conditioned_denoising_part_step6_tagger_training"
TARGET_DENOISING_TAGGER_TRAINING_CONTRACT = "target_conditioned_denoising_part_tagger_training_v1"
TARGET_DENOISING_TAGGER_SELECTION_METRICS = ("accuracy", "loss", "cross_entropy")
TARGET_DENOISING_TAGGER_LOWER_IS_BETTER = {"loss", "cross_entropy"}


@dataclass(frozen=True)
class TargetDenoisingTaggerTrainConfig:
    """Configuration for Step 6 target-denoising-augmented tagger training."""

    output_dir: str
    manifest_path: str
    hlt_cache_dir: str
    offline_cache_dir: str | None = None
    data_dir: str | None = None
    denoiser_checkpoint: str | None = None
    variant: str = "denoiser_features_frozen"
    train_split: str = "model_train"
    val_split: str = "model_val"
    final_test_split: str = "final_test"
    seed: int = 7307
    batch_size: int = 64
    eval_batch_size: int = 128
    epochs: int = 35
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 6
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_final_test_batches: int | None = None
    max_train_jets: int | None = None
    max_val_jets: int | None = None
    max_final_test_jets: int | None = None
    evaluate_final_test: bool = False
    confirm_final_test: bool = False
    selection_metric: str = "accuracy"
    compile_model: bool = False
    verify_hlt_hash: bool = True
    strict_hlt_metadata: bool = True
    require_same_manifest_hash: bool = True
    require_same_jet_identity: bool = True
    expected_hlt_profile: str | None = HLT_PROFILE_V2_REALISTIC
    expected_hlt_profile_version: str | None = HLT_PROFILE_V2_REALISTIC_VERSION
    expected_hlt_degradation_strength: float | None = 1.0
    num_classes: int = 10
    model_size: str = "base"
    part_embed_dim: int = 128
    max_constits: int = 128
    weight_threshold: float = 0.0
    adapter_hidden_dim: int = 128
    adapter_dropout: float = 0.0
    adapter_gate_bias_init: float = -2.0
    freeze_denoiser: bool | None = None
    require_denoiser_checkpoint: bool = True
    strict_denoiser_checkpoint: bool = True
    require_compatible_denoiser_checkpoint: bool = True
    reconstruction_anchor_weight: float = 0.0
    reconstruction_anchor_smooth_l1_beta: float = 1.0
    alignment_mode: str = "aligned_direct"
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

    def __post_init__(self) -> None:
        variant = str(self.variant)
        if variant not in TARGET_DENOISING_TAGGER_VARIANTS:
            raise ValueError(f"variant must be one of {TARGET_DENOISING_TAGGER_VARIANTS}")
        if self.train_split != "model_train" or self.val_split != "model_val":
            raise ValueError("Step 6 taggers train only on model_train and select only on model_val")
        if self.selection_metric not in TARGET_DENOISING_TAGGER_SELECTION_METRICS:
            raise ValueError(f"selection_metric must be one of {TARGET_DENOISING_TAGGER_SELECTION_METRICS}")
        if bool(self.evaluate_final_test) and not bool(self.confirm_final_test):
            raise ValueError("confirm_final_test must be true before evaluating final_test")
        for name in ("batch_size", "eval_batch_size", "epochs", "num_classes", "part_embed_dim", "max_constits"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("lr", "weight_decay", "grad_clip_norm", "adapter_dropout", "dropout", "attention_dropout"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("reconstruction_anchor_weight", "reconstruction_anchor_smooth_l1_beta"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        object.__setattr__(self, "variant", variant)

    @property
    def uses_pretrained_denoiser_checkpoint(self) -> bool:
        return self.variant in TARGET_DENOISING_DENOISER_VARIANTS and self.variant != TARGET_DENOISING_VARIANT_DENOISER_TAG_ONLY_SAME_ARCH

    def denoiser_config(self) -> TargetConditionedDenoiserConfig:
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

    def tagger_config(self) -> TargetDenoisingAugmentedParTConfig:
        return TargetDenoisingAugmentedParTConfig(
            variant=str(self.variant),
            num_classes=int(self.num_classes),
            model_size=str(self.model_size),
            part_embed_dim=int(self.part_embed_dim),
            max_constits=int(self.max_constits),
            weight_threshold=float(self.weight_threshold),
            adapter_hidden_dim=int(self.adapter_hidden_dim),
            adapter_dropout=float(self.adapter_dropout),
            adapter_gate_bias_init=float(self.adapter_gate_bias_init),
            freeze_denoiser=self.freeze_denoiser,
            denoiser_config=self.denoiser_config(),
        )

    def paired_dataset_config(self, split: str, *, max_jets: int | None = None) -> TargetDenoisingDatasetConfig:
        return TargetDenoisingDatasetConfig(
            manifest_path=self.manifest_path,
            hlt_cache_dir=self.hlt_cache_dir,
            offline_cache_dir=self.offline_cache_dir,
            data_dir=self.data_dir,
            split=split,
            max_jets=max_jets,
            alignment_mode=self.alignment_mode,
            expected_hlt_profile=self.expected_hlt_profile,
            expected_hlt_profile_version=self.expected_hlt_profile_version,
            expected_hlt_degradation_strength=self.expected_hlt_degradation_strength,
            require_hlt_contract=bool(self.strict_hlt_metadata),
            require_same_manifest_hash=bool(self.require_same_manifest_hash),
            require_same_jet_identity=bool(self.require_same_jet_identity),
            allow_final_test_targets=False,
            verify_hlt_hash=bool(self.verify_hlt_hash),
        )


class TargetDenoisingTaggerDataset:
    """Thin raw-token dataset over a cached fixed-HLT JetView."""

    def __init__(self, view: Any, *, max_jets: int | None = None) -> None:
        limit = None if max_jets is None else max(0, min(int(max_jets), int(view.tokens.shape[0])))
        if limit is None:
            self.tokens = view.tokens.astype(np.float32, copy=False)
            self.mask = view.mask.astype(bool, copy=False)
            self.labels = view.labels.astype(np.int64, copy=False)
            self.jet_ids = list(view.jet_ids)
        else:
            self.tokens = view.tokens[:limit].astype(np.float32, copy=False)
            self.mask = view.mask[:limit].astype(bool, copy=False)
            self.labels = view.labels[:limit].astype(np.int64, copy=False)
            self.jet_ids = list(view.jet_ids[:limit])
        self.split = str(view.split)
        self.metadata = dict(view.metadata or {})
        self.source_n_jets = int(view.tokens.shape[0])

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "hlt_tokens": self.tokens[index],
            "hlt_constituent_mask": self.mask[index],
            "labels": int(self.labels[index]),
        }

    def to_metadata(self) -> dict[str, Any]:
        label_counts = {
            str(int(label)): int(count)
            for label, count in zip(*np.unique(self.labels.astype(np.int64), return_counts=True))
        }
        keys = (
            "hlt_profile",
            "hlt_profile_version",
            "hlt_degradation_strength",
            "hlt_content_hash",
            "source_manifest_hash",
            "jet_identity_hash",
            "seed",
            "n_jets",
        )
        return {
            "split": self.split,
            "n_jets": int(len(self)),
            "source_n_jets": int(self.source_n_jets),
            "label_counts": label_counts,
            "metadata": {key: self.metadata.get(key) for key in keys},
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "detach"):
        tensor = value.detach()
        if int(tensor.numel()) == 1:
            return float(tensor.cpu().item())
        return tensor.cpu().tolist()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_epoch_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in keys})


def collate_target_denoising_tagger_batch(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    torch = require_torch()
    return {
        "hlt_tokens": torch.as_tensor(np.stack([sample["hlt_tokens"] for sample in samples]), dtype=torch.float32),
        "hlt_constituent_mask": torch.as_tensor(np.stack([sample["hlt_constituent_mask"] for sample in samples]), dtype=torch.bool),
        "labels": torch.as_tensor([int(sample["labels"]) for sample in samples], dtype=torch.long),
    }


def target_denoising_tagger_batch_to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    torch = require_torch()
    return {key: value.to(device=device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _make_loader(dataset: Any, *, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> Any:
    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    collate_fn = collate_target_denoising_batch if isinstance(dataset, TargetDenoisingPairedDataset) else collate_target_denoising_tagger_batch
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        generator=generator,
    )


def _identity_key(identity: JetIdentity) -> tuple[str, int, int]:
    return str(identity.file), int(identity.entry), int(identity.label)


def _load_manifest_info(config: TargetDenoisingTaggerTrainConfig) -> tuple[Any | None, str | None]:
    if not (bool(config.require_same_manifest_hash) or bool(config.require_same_jet_identity)):
        return None, None
    manifest = load_split_manifest(config.manifest_path)
    return manifest, manifest_hash(manifest)


def _validate_hlt_metadata(
    metadata: Mapping[str, Any],
    config: TargetDenoisingTaggerTrainConfig,
    *,
    split: str,
    manifest_sha: str | None,
) -> None:
    problems: list[str] = []
    if bool(config.strict_hlt_metadata):
        if config.expected_hlt_profile is not None:
            actual_profile = normalize_hlt_profile(metadata.get("hlt_profile"))
            expected_profile = normalize_hlt_profile(config.expected_hlt_profile)
            if actual_profile != expected_profile:
                problems.append(f"hlt_profile={actual_profile!r}, expected {expected_profile!r}")
        if config.expected_hlt_profile_version is not None:
            actual_version = str(metadata.get("hlt_profile_version") or "")
            expected_version = str(config.expected_hlt_profile_version)
            if actual_version != expected_version:
                problems.append(f"hlt_profile_version={actual_version!r}, expected {expected_version!r}")
        if config.expected_hlt_degradation_strength is not None:
            actual_strength = metadata.get("hlt_degradation_strength")
            if actual_strength is None or abs(float(actual_strength) - float(config.expected_hlt_degradation_strength)) > 1.0e-12:
                problems.append(
                    f"hlt_degradation_strength={actual_strength!r}, expected {float(config.expected_hlt_degradation_strength)!r}"
                )
    if bool(config.require_same_manifest_hash) and manifest_sha is not None:
        actual_manifest_hash = metadata.get("source_manifest_hash")
        if actual_manifest_hash != manifest_sha:
            problems.append(f"source_manifest_hash={actual_manifest_hash!r}, expected {manifest_sha!r}")
    if problems:
        joined = "; ".join(problems)
        raise ValueError(f"HLT cache metadata mismatch for split {split}: {joined}")


def load_target_denoising_tagger_dataset(
    config: TargetDenoisingTaggerTrainConfig,
    split: str,
    *,
    max_jets: int | None = None,
) -> TargetDenoisingTaggerDataset:
    manifest, manifest_sha = _load_manifest_info(config)
    view = load_cached_hlt_view(config.hlt_cache_dir, split, verify_hash=bool(config.verify_hlt_hash))
    _validate_hlt_metadata(view.metadata, config, split=split, manifest_sha=manifest_sha)
    if bool(config.require_same_jet_identity) and manifest is not None:
        expected_ids = list(manifest.splits[split])
        actual_keys = [_identity_key(identity) for identity in view.jet_ids]
        expected_keys = [_identity_key(identity) for identity in expected_ids]
        if actual_keys != expected_keys:
            raise ValueError(f"HLT cache jet identities for split {split} do not match manifest")
        metadata_identity_hash = view.metadata.get("jet_identity_hash")
        actual_identity_hash = jet_identity_hash(view.jet_ids)
        if metadata_identity_hash != actual_identity_hash:
            raise ValueError(
                f"HLT metadata jet_identity_hash mismatch for split {split}: "
                f"{metadata_identity_hash!r} != {actual_identity_hash!r}"
            )
    return TargetDenoisingTaggerDataset(view, max_jets=max_jets)


def load_target_denoising_tagger_train_dataset(
    config: TargetDenoisingTaggerTrainConfig,
    split: str,
    *,
    max_jets: int | None = None,
) -> TargetDenoisingTaggerDataset | TargetDenoisingPairedDataset:
    if float(config.reconstruction_anchor_weight) > 0.0 and split in {config.train_split, config.val_split}:
        return load_target_denoising_dataset(config.paired_dataset_config(split, max_jets=max_jets))
    return load_target_denoising_tagger_dataset(config, split, max_jets=max_jets)


def _flatten_numeric(prefix: str, value: Any, output: dict[str, float]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten_numeric(f"{prefix}.{key}" if prefix else str(key), item, output)
        return
    try:
        number = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(number):
        output[prefix] = number


def _diagnostics_from_output(output: TargetDenoisingAugmentedParTOutput) -> dict[str, float]:
    flattened: dict[str, float] = {}
    _flatten_numeric("", output.diagnostics(), flattened)
    return flattened


def _clip_gradients_or_skip(model: Any, max_norm: float) -> tuple[bool, float]:
    """Clip trainable gradients and report whether they are finite."""

    torch = require_torch()
    params = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
    if not params:
        return True, 0.0
    if float(max_norm) > 0.0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            params,
            float(max_norm),
            error_if_nonfinite=False,
        )
    else:
        grad_norm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(parameter.grad.detach(), ord=2) for parameter in params]),
            ord=2,
        )
    finite = bool(torch.isfinite(grad_norm).detach().cpu().item())
    return finite, float(grad_norm.detach().cpu().item()) if finite else float("nan")


def _merge_diagnostics(total: dict[str, float], batch_diag: Mapping[str, float], weight: int) -> None:
    for key, value in batch_diag.items():
        total[key] = total.get(key, 0.0) + float(value) * int(weight)


def _classification_metrics(
    *,
    loss_sum: float,
    ce_loss_sum: float,
    reconstruction_loss_sum: float,
    n_jets: int,
    correct: int,
    confusion: np.ndarray,
    diagnostics_sum: Mapping[str, float],
    diagnostic_weight: int,
) -> dict[str, Any]:
    n = max(int(n_jets), 1)
    per_class = {}
    for idx in range(int(confusion.shape[0])):
        denom = int(confusion[idx].sum())
        per_class[str(idx)] = None if denom <= 0 else float(confusion[idx, idx] / denom)
    metrics: dict[str, Any] = {
        "loss": float(loss_sum / n),
        "cross_entropy": float(ce_loss_sum / n),
        "reconstruction_anchor_loss": float(reconstruction_loss_sum / n),
        "accuracy": float(correct / n),
        "n_jets": int(n_jets),
        "correct": int(correct),
        "confusion_matrix": confusion.astype(int).tolist(),
        "per_class_accuracy": per_class,
    }
    if diagnostic_weight > 0:
        metrics["diagnostics"] = {key: float(value / diagnostic_weight) for key, value in diagnostics_sum.items()}
    else:
        metrics["diagnostics"] = {}
    return metrics


def _reconstruction_anchor_loss(
    output: Any,
    batch: Mapping[str, Any],
    config: TargetDenoisingTaggerTrainConfig,
) -> Any:
    torch = require_torch()
    logits = output.logits if hasattr(output, "logits") else output
    if float(config.reconstruction_anchor_weight) <= 0.0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    if not isinstance(output, TargetDenoisingAugmentedParTOutput) or output.denoiser_output is None:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    required = ("target_residuals", "target_mask", "target_weights")
    if any(key not in batch for key in required):
        return torch.zeros((), device=output.logits.device, dtype=output.logits.dtype)
    target = batch["target_residuals"].to(device=output.denoiser_output.deltas.device, dtype=output.denoiser_output.deltas.dtype)
    mask = batch["target_mask"].to(device=output.denoiser_output.deltas.device, dtype=torch.bool)
    weights = batch["target_weights"].to(device=output.denoiser_output.deltas.device, dtype=output.denoiser_output.deltas.dtype)
    weights = weights[:, :, None] * mask[:, :, None].to(dtype=weights.dtype)
    if target.shape != output.denoiser_output.deltas.shape:
        raise ValueError(f"target residual shape {tuple(target.shape)} does not match denoiser output {tuple(output.denoiser_output.deltas.shape)}")
    loss_rows = torch.nn.functional.smooth_l1_loss(
        output.denoiser_output.deltas,
        target,
        reduction="none",
        beta=float(config.reconstruction_anchor_smooth_l1_beta),
    )
    denom = weights.sum().clamp_min(1.0)
    return (loss_rows * weights).sum() / denom


def run_target_denoising_tagger_epoch(
    model: Any,
    loader: Any,
    config: TargetDenoisingTaggerTrainConfig,
    *,
    device: Any,
    optimizer: Any | None = None,
    scaler: Any | None = None,
    max_batches: int | None = None,
    collect_diagnostics: bool = False,
) -> dict[str, Any]:
    torch = require_torch()
    is_train = optimizer is not None
    model.train(is_train)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    ce_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    correct = 0
    n_jets = 0
    confusion = np.zeros((int(config.num_classes), int(config.num_classes)), dtype=np.int64)
    skipped_nonfinite_batches = 0
    diagnostics_sum: dict[str, float] = {}
    diagnostic_weight = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and int(batch_index) >= int(max_batches):
            break
        batch = target_denoising_tagger_batch_to_device(batch, device)
        labels = batch["labels"]
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        autocast_enabled = bool(config.amp and getattr(device, "type", str(device)) == "cuda")
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            output = model(
                batch["hlt_tokens"],
                batch["hlt_constituent_mask"],
                return_outputs=True,
                need_denoiser_weights=False,
            )
            logits = output.logits if isinstance(output, TargetDenoisingAugmentedParTOutput) else output
            ce_loss = criterion(logits, labels)
            reconstruction_loss = _reconstruction_anchor_loss(output, batch, config)
            loss = ce_loss + float(config.reconstruction_anchor_weight) * reconstruction_loss * max(int(labels.numel()), 1)
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            skipped_nonfinite_batches += 1
            continue
        if is_train:
            mean_loss = loss / max(int(labels.numel()), 1)
            if scaler is not None and autocast_enabled:
                scaler.scale(mean_loss).backward()
                if float(config.grad_clip_norm) > 0.0:
                    scaler.unscale_(optimizer)
                gradients_ok, _ = _clip_gradients_or_skip(model, float(config.grad_clip_norm))
                if not gradients_ok:
                    skipped_nonfinite_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
                scaler.step(optimizer)
                scaler.update()
            else:
                mean_loss.backward()
                gradients_ok, _ = _clip_gradients_or_skip(model, float(config.grad_clip_norm))
                if not gradients_ok:
                    skipped_nonfinite_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()
        batch_size = int(labels.numel())
        loss_sum += float(loss.detach().cpu().item())
        ce_loss_sum += float(ce_loss.detach().cpu().item())
        reconstruction_loss_sum += float(reconstruction_loss.detach().cpu().item()) * batch_size
        predictions = logits.detach().argmax(dim=-1)
        correct += int((predictions == labels).sum().detach().cpu().item())
        n_jets += batch_size
        encoded = (labels.detach().cpu().numpy().astype(np.int64) * int(config.num_classes)) + predictions.detach().cpu().numpy().astype(np.int64)
        counts = np.bincount(encoded, minlength=int(config.num_classes) * int(config.num_classes))
        confusion += counts.reshape(int(config.num_classes), int(config.num_classes))
        if collect_diagnostics and isinstance(output, TargetDenoisingAugmentedParTOutput):
            batch_diag = _diagnostics_from_output(output)
            _merge_diagnostics(diagnostics_sum, batch_diag, batch_size)
            diagnostic_weight += batch_size

    metrics = _classification_metrics(
        loss_sum=loss_sum,
        ce_loss_sum=ce_loss_sum,
        reconstruction_loss_sum=reconstruction_loss_sum,
        n_jets=n_jets,
        correct=correct,
        confusion=confusion,
        diagnostics_sum=diagnostics_sum,
        diagnostic_weight=diagnostic_weight,
    )
    metrics["skipped_nonfinite_batches"] = int(skipped_nonfinite_batches)
    return metrics


def _metric_is_better(name: str, value: float, best: float | None) -> bool:
    if best is None:
        return True
    if name in TARGET_DENOISING_TAGGER_LOWER_IS_BETTER:
        return float(value) < float(best)
    return float(value) > float(best)


def _model_config_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "to_config_dict"):
        return dict(model.to_config_dict())
    return {"model_class": type(model).__name__}


def _checkpoint_payload(
    *,
    model: Any,
    config: TargetDenoisingTaggerTrainConfig,
    epoch: int,
    metrics: Mapping[str, Any],
    train_dataset: TargetDenoisingTaggerDataset,
    val_dataset: TargetDenoisingTaggerDataset,
) -> dict[str, Any]:
    return {
        "output_contract": TARGET_DENOISING_TAGGER_TRAINING_CONTRACT,
        "tagger_contract": TARGET_DENOISING_TAGGER_CONTRACT,
        "step": TARGET_DENOISING_STEP6_TAGGER,
        "epoch": int(epoch),
        "selection_metric": str(config.selection_metric),
        "metrics": dict(metrics),
        "config": asdict(config),
        "model_config": _model_config_dict(model),
        "model_state_dict": model.state_dict(),
        "train_dataset": train_dataset.to_metadata(),
        "model_val_dataset": val_dataset.to_metadata(),
    }


def _load_denoiser_if_needed(
    model: Any,
    config: TargetDenoisingTaggerTrainConfig,
    *,
    device: Any,
) -> dict[str, Any] | None:
    if not config.uses_pretrained_denoiser_checkpoint:
        return None
    if not config.denoiser_checkpoint:
        if bool(config.require_denoiser_checkpoint):
            raise ValueError(f"variant {config.variant!r} requires --denoiser-checkpoint")
        return None
    report = load_target_denoising_pretrained_checkpoint(
        config.denoiser_checkpoint,
        model,
        map_location=device,
        strict=bool(config.strict_denoiser_checkpoint),
    )
    return report


def _metadata_block(dataset: Any) -> Mapping[str, Any]:
    metadata = dataset.to_metadata() if hasattr(dataset, "to_metadata") else {}
    return metadata if isinstance(metadata, Mapping) else {}


def _nested_metadata_value(block: Mapping[str, Any], key: str) -> Any:
    if key in block:
        return block.get(key)
    metadata = block.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata.get(key)
    return None


def _first_nested_metadata_value(block: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = _nested_metadata_value(block, key)
        if value is not None:
            return value
    return None


def _compare_required_metadata(
    problems: list[str],
    *,
    split_name: str,
    label: str,
    checkpoint_block: Mapping[str, Any],
    current_block: Mapping[str, Any],
    checkpoint_keys: Sequence[str],
    current_keys: Sequence[str],
) -> None:
    expected = _first_nested_metadata_value(current_block, current_keys)
    if expected is None:
        return
    actual = _first_nested_metadata_value(checkpoint_block, checkpoint_keys)
    if actual is None:
        problems.append(f"{split_name}.{label} is missing from denoiser checkpoint")
    elif actual != expected:
        problems.append(f"{split_name}.{label} checkpoint={actual!r}, current={expected!r}")


def _validate_denoiser_checkpoint_compatibility(
    checkpoint_report: Mapping[str, Any] | None,
    config: TargetDenoisingTaggerTrainConfig,
    *,
    train_dataset: Any,
    val_dataset: Any,
) -> None:
    if not checkpoint_report or not bool(config.require_compatible_denoiser_checkpoint):
        return
    problems: list[str] = []
    ckpt_config = checkpoint_report.get("config") if isinstance(checkpoint_report.get("config"), Mapping) else {}
    ckpt_model_config = checkpoint_report.get("model_config") if isinstance(checkpoint_report.get("model_config"), Mapping) else {}
    ckpt_train = checkpoint_report.get("train_dataset") if isinstance(checkpoint_report.get("train_dataset"), Mapping) else {}
    ckpt_val = checkpoint_report.get("model_val_dataset") if isinstance(checkpoint_report.get("model_val_dataset"), Mapping) else {}
    current_train = _metadata_block(train_dataset)
    current_val = _metadata_block(val_dataset)
    for split_name, ckpt_block, current_block in (
        ("model_train", ckpt_train, current_train),
        ("model_val", ckpt_val, current_val),
    ):
        if not ckpt_block:
            problems.append(f"{split_name} dataset metadata is missing from denoiser checkpoint")
        for key in ("source_manifest_hash", "expected_manifest_hash", "hlt_profile", "hlt_profile_version", "hlt_degradation_strength"):
            _compare_required_metadata(
                problems,
                split_name=split_name,
                label=key,
                checkpoint_block=ckpt_block,
                current_block=current_block,
                checkpoint_keys=(key,),
                current_keys=(key,),
            )
        _compare_required_metadata(
            problems,
            split_name=split_name,
            label="hlt_content_hash",
            checkpoint_block=ckpt_block,
            current_block=current_block,
            checkpoint_keys=("hlt_content_hash",),
            current_keys=("hlt_content_hash",),
        )
        _compare_required_metadata(
            problems,
            split_name=split_name,
            label="hlt_jet_identity_hash",
            checkpoint_block=ckpt_block,
            current_block=current_block,
            checkpoint_keys=("hlt_jet_identity_hash", "jet_identity_hash"),
            current_keys=("jet_identity_hash", "hlt_jet_identity_hash"),
        )
        _compare_required_metadata(
            problems,
            split_name=split_name,
            label="offline_jet_identity_hash",
            checkpoint_block=ckpt_block,
            current_block=current_block,
            checkpoint_keys=("offline_jet_identity_hash",),
            current_keys=("offline_jet_identity_hash",),
        )
        expected_alignment = str(config.alignment_mode)
        actual_alignment = ckpt_block.get("alignment_mode") or ckpt_config.get("alignment_mode")
        if actual_alignment is None:
            problems.append(f"{split_name}.alignment_mode is missing from denoiser checkpoint")
        elif str(actual_alignment) != str(expected_alignment):
            problems.append(f"{split_name}.alignment_mode checkpoint={actual_alignment!r}, current={expected_alignment!r}")

    shuffle = bool(ckpt_config.get("shuffle_target_residuals", False))
    use_pair_bias = bool(ckpt_model_config.get("use_pair_bias", True))
    use_local_kernel = bool(ckpt_model_config.get("use_local_kernel", True))
    if config.variant == TARGET_DENOISING_VARIANT_DENOISER_SHUFFLED_TARGETS and not shuffle:
        problems.append("denoiser_shuffled_targets requires a checkpoint trained with shuffle_target_residuals=true")
    if config.variant != TARGET_DENOISING_VARIANT_DENOISER_SHUFFLED_TARGETS and shuffle:
        problems.append(f"{config.variant} must not consume a shuffled-target denoiser checkpoint")
    if config.variant == TARGET_DENOISING_VARIANT_DENOISER_NO_PAIR_BIAS and (use_pair_bias or use_local_kernel):
        problems.append("denoiser_no_pair_bias requires a checkpoint with use_pair_bias=false and use_local_kernel=false")
    if config.variant == TARGET_DENOISING_VARIANT_DENOISER_LOCAL_KERNEL_ONLY and (use_pair_bias or not use_local_kernel):
        problems.append("denoiser_local_kernel_only requires a checkpoint with use_pair_bias=false and use_local_kernel=true")
    if config.variant not in {
        TARGET_DENOISING_VARIANT_DENOISER_NO_PAIR_BIAS,
        TARGET_DENOISING_VARIANT_DENOISER_LOCAL_KERNEL_ONLY,
    } and not shuffle:
        if not use_pair_bias:
            problems.append(f"{config.variant} expects a real denoiser checkpoint with use_pair_bias=true")
    if problems:
        raise ValueError("denoiser checkpoint is not compatible with tagger run: " + "; ".join(problems))


def train_target_denoising_tagger(
    config: TargetDenoisingTaggerTrainConfig,
    *,
    model: Any | None = None,
    train_dataset: TargetDenoisingTaggerDataset | None = None,
    val_dataset: TargetDenoisingTaggerDataset | None = None,
    final_test_dataset: TargetDenoisingTaggerDataset | None = None,
) -> dict[str, Any]:
    """Train one Step 6 denoising-augmented ParT tagger and write artifacts."""

    torch = require_torch()
    set_training_seed(int(config.seed))
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    diagnostics_dir = output_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = train_dataset or load_target_denoising_tagger_train_dataset(
        config,
        config.train_split,
        max_jets=config.max_train_jets,
    )
    val_dataset = val_dataset or load_target_denoising_tagger_train_dataset(
        config,
        config.val_split,
        max_jets=config.max_val_jets,
    )
    if bool(config.evaluate_final_test):
        final_test_dataset = final_test_dataset or load_target_denoising_tagger_dataset(
            config,
            config.final_test_split,
            max_jets=config.max_final_test_jets,
        )

    model = model or TargetDenoisingAugmentedParT(config.tagger_config())
    checkpoint_report = _load_denoiser_if_needed(model, config, device=device)
    _validate_denoiser_checkpoint_compatibility(
        checkpoint_report,
        config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    model.to(device)
    if bool(config.compile_model) and hasattr(torch, "compile"):
        model = torch.compile(model)

    train_loader = _make_loader(
        train_dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        seed=int(config.seed),
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 1,
    )
    final_test_loader = None
    if bool(config.evaluate_final_test) and final_test_dataset is not None:
        final_test_loader = _make_loader(
            final_test_dataset,
            batch_size=int(config.eval_batch_size),
            shuffle=False,
            num_workers=int(config.num_workers),
            seed=int(config.seed) + 2,
        )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))

    _write_json(
        output_dir / "config.json",
        {
            "config": asdict(config),
            "model_config": _model_config_dict(model),
            "denoiser_checkpoint_report": checkpoint_report,
        },
    )
    _write_json(diagnostics_dir / "train_dataset_metadata.json", train_dataset.to_metadata())
    _write_json(diagnostics_dir / "model_val_dataset_metadata.json", val_dataset.to_metadata())
    if final_test_dataset is not None:
        _write_json(diagnostics_dir / "final_test_dataset_metadata.json", final_test_dataset.to_metadata())

    curves: list[dict[str, Any]] = []
    best_value: float | None = None
    best_epoch = -1
    best_metrics: dict[str, Any] | None = None
    patience = 0

    for epoch in range(int(config.epochs)):
        train_metrics = run_target_denoising_tagger_epoch(
            model,
            train_loader,
            config,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            max_batches=config.max_train_batches,
            collect_diagnostics=False,
        )
        val_metrics = run_target_denoising_tagger_epoch(
            model,
            val_loader,
            config,
            device=device,
            optimizer=None,
            scaler=None,
            max_batches=config.max_val_batches,
            collect_diagnostics=True,
        )
        metric_value = float(val_metrics[str(config.selection_metric)])
        improved = _metric_is_better(str(config.selection_metric), metric_value, best_value)
        row = {
            "epoch": int(epoch),
            "train.loss": train_metrics["loss"],
            "train.accuracy": train_metrics["accuracy"],
            "model_val.loss": val_metrics["loss"],
            "model_val.accuracy": val_metrics["accuracy"],
            "model_val.selection_metric": metric_value,
            "improved": bool(improved),
            "train.skipped_nonfinite_batches": train_metrics.get("skipped_nonfinite_batches", 0),
            "model_val.skipped_nonfinite_batches": val_metrics.get("skipped_nonfinite_batches", 0),
        }
        for key, value in (val_metrics.get("diagnostics") or {}).items():
            row[f"model_val.diagnostics.{key}"] = value
        curves.append(row)
        if improved:
            best_value = metric_value
            best_epoch = int(epoch)
            best_metrics = dict(val_metrics)
            torch.save(
                _checkpoint_payload(
                    model=model,
                    config=config,
                    epoch=epoch,
                    metrics=val_metrics,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                ),
                output_dir / "best_model_val.pt",
            )
            _write_json(
                output_dir / "model_val_report.json",
                {
                    "ok": True,
                    "split": "model_val",
                    "epoch": int(epoch),
                    "variant": str(config.variant),
                    "metrics": val_metrics,
                    "selection_metric": str(config.selection_metric),
                    "selection_metric_value": metric_value,
                },
            )
            patience = 0
        else:
            patience += 1
        if int(config.early_stop_patience) > 0 and patience >= int(config.early_stop_patience):
            break

    final_epoch = curves[-1]["epoch"] if curves else -1
    torch.save(
        _checkpoint_payload(
            model=model,
            config=config,
            epoch=int(final_epoch),
            metrics=best_metrics or {},
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        ),
        output_dir / "last.pt",
    )

    final_test_metrics: dict[str, Any] | None = None
    if final_test_loader is not None:
        best_payload = torch.load(str(output_dir / "best_model_val.pt"), map_location=device)
        target_model = model.module if hasattr(model, "module") else model
        target_model.load_state_dict(best_payload["model_state_dict"])
        final_test_metrics = run_target_denoising_tagger_epoch(
            model,
            final_test_loader,
            config,
            device=device,
            optimizer=None,
            scaler=None,
            max_batches=config.max_final_test_batches,
            collect_diagnostics=True,
        )
        _write_json(
            output_dir / "final_test_report.json",
            {
                "ok": True,
                "split": "final_test",
                "variant": str(config.variant),
                "checkpoint": str(output_dir / "best_model_val.pt"),
                "checkpoint_epoch": int(best_epoch),
                "confirm_final_test": bool(config.confirm_final_test),
                "metrics": final_test_metrics,
            },
        )

    _write_json(output_dir / "training_curves.json", {"epochs": curves})
    _write_epoch_csv(diagnostics_dir / "epoch_metrics.csv", curves)

    report = {
        "ok": True,
        "output_contract": TARGET_DENOISING_TAGGER_TRAINING_CONTRACT,
        "tagger_contract": TARGET_DENOISING_TAGGER_CONTRACT,
        "step": TARGET_DENOISING_STEP6_TAGGER,
        "output_dir": str(output_dir),
        "variant": str(config.variant),
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "run_report": str(output_dir / "run_report.json"),
        "best_epoch": int(best_epoch),
        "selection_metric": str(config.selection_metric),
        "best_model_selection_metric_value": None if best_value is None else float(best_value),
        "best_model_val_metrics": best_metrics or {},
        "train_dataset": train_dataset.to_metadata(),
        "model_val_dataset": val_dataset.to_metadata(),
        "final_test_evaluated": bool(final_test_metrics is not None),
        "final_test_metrics": final_test_metrics,
        "config": asdict(config),
        "model_config": _model_config_dict(model),
        "denoiser_checkpoint_report": checkpoint_report,
    }
    _write_json(output_dir / "run_report.json", report)
    return report


__all__ = [
    "TARGET_DENOISING_STEP6_TAGGER",
    "TARGET_DENOISING_TAGGER_LOWER_IS_BETTER",
    "TARGET_DENOISING_TAGGER_SELECTION_METRICS",
    "TARGET_DENOISING_TAGGER_TRAINING_CONTRACT",
    "TargetDenoisingTaggerDataset",
    "TargetDenoisingTaggerTrainConfig",
    "collate_target_denoising_tagger_batch",
    "load_target_denoising_tagger_dataset",
    "load_target_denoising_tagger_train_dataset",
    "run_target_denoising_tagger_epoch",
    "target_denoising_tagger_batch_to_device",
    "train_target_denoising_tagger",
]
