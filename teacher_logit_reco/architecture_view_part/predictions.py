"""Prediction caching for 10-class architecture-view ParT models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json
from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash, load_cached_hlt_view
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM

from teacher_logit_reco.set_matching.five_view_train import classification_metrics_from_predictions
from teacher_logit_reco.set_matching.train import source_metadata
from teacher_logit_reco.subtoken_part.train import SubtokenHLTJetDataset, make_subtoken_hlt_loader

from .checkpoint import sha256_file
from .config import (
    ARCHITECTURE_VIEW_10CLASS_DEFAULT_VARIANTS,
    ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER,
    ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
    ARCHITECTURE_VIEW_10CLASS_VARIANTS,
    normalize_architecture_view_variant,
)
from .train import (
    architecture_view_binary_projection_metrics,
    load_architecture_view_tagger_checkpoint,
)


ARCHITECTURE_VIEW_10CLASS_PREDICTION_STEP = "architecture_view_10class_step2_prediction_cache"
ARCHITECTURE_VIEW_10CLASS_PREDICTION_CONTRACT = "architecture_view_10class_prediction_cache_v1"
ARCHITECTURE_VIEW_10CLASS_PREDICTION_SPLITS = (
    "model_val",
    "stack_train",
    "stack_val",
    "final_test",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _identity_arrays(jet_ids: Sequence[JetIdentity]) -> tuple[list[str], np.ndarray, np.ndarray]:
    files: list[str] = []
    file_to_index: dict[str, int] = {}
    file_indices = np.zeros((len(jet_ids),), dtype=np.int32)
    entries = np.zeros((len(jet_ids),), dtype=np.int64)
    for index, identity in enumerate(jet_ids):
        if identity.file not in file_to_index:
            file_to_index[identity.file] = len(files)
            files.append(identity.file)
        file_indices[index] = file_to_index[identity.file]
        entries[index] = int(identity.entry)
    return files, file_indices, entries


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.clip(exp.sum(axis=1, keepdims=True), 1.0e-300, None)).astype(np.float32)


def architecture_view_10class_prediction_paths(
    prediction_dir: str | Path,
    variant: str,
    split: str,
) -> tuple[Path, Path]:
    """Return the Step 2 logits and metadata paths."""

    canonical = normalize_architecture_view_variant(variant)
    root = Path(prediction_dir) / canonical
    return root / f"{split}_logits.npz", root / f"{split}_metadata.json"


def _checkpoint_path(checkpoint_root: str | Path, variant: str) -> Path:
    return Path(checkpoint_root) / normalize_architecture_view_variant(variant) / "best_model_val.pt"


def _max_jets_for_split(config: "ArchitectureView10ClassPredictionCacheConfig", split: str) -> int | None:
    if split == "model_val":
        return config.max_model_val_jets
    if split == "stack_train":
        return config.max_stack_train_jets
    if split == "stack_val":
        return config.max_stack_val_jets
    if split == "final_test":
        return config.max_final_test_jets
    return None


def _require_av10_payload(payload: Mapping[str, Any], *, variant: str, checkpoint: Path | None = None) -> None:
    if int(payload.get("num_classes", -1)) != 10:
        raise ValueError(f"AV10 prediction cache requires num_classes=10 for {variant}")
    label_names = tuple(str(name) for name in payload.get("label_names", ()))
    label_filter = tuple(int(index) for index in payload.get("label_filter", ()))
    if label_names != ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES:
        raise ValueError(f"AV10 checkpoint label_names mismatch for {variant}: {label_names}")
    if label_filter != ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER:
        raise ValueError(f"AV10 checkpoint label_filter mismatch for {variant}: {label_filter}")
    payload_variant = payload.get("variant")
    if payload_variant is not None and normalize_architecture_view_variant(str(payload_variant)) != variant:
        raise ValueError(
            f"AV10 checkpoint variant mismatch for {checkpoint or variant}: "
            f"{payload_variant!r} != {variant!r}"
        )


def _load_or_build_dataset(
    *,
    hlt_cache_dir: str | Path,
    split: str,
    max_jets: int | None,
    views_by_split: Mapping[str, JetView] | None,
    verify_hlt_hash: bool,
) -> SubtokenHLTJetDataset:
    view = views_by_split[split] if views_by_split is not None and split in views_by_split else load_cached_hlt_view(
        hlt_cache_dir,
        split,
        verify_hash=bool(verify_hlt_hash),
    )
    if view.metadata.get("view") not in (None, "fixed_hlt"):
        raise ValueError(f"Expected fixed_hlt cached view for {split}, got {view.metadata.get('view')!r}")
    if int(view.tokens.shape[-1]) != RAW_TOKEN_DIM:
        raise ValueError(f"Expected raw token dim {RAW_TOKEN_DIM}, got {view.tokens.shape[-1]}")
    return SubtokenHLTJetDataset(
        view,
        label_filter=ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER,
        label_names=ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
        max_jets=max_jets,
    )


def _collect_logits_for_dataset(
    model: Any,
    dataset: SubtokenHLTJetDataset,
    *,
    batch_size: int,
    num_workers: int,
    device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    torch = require_torch()
    loader = make_subtoken_hlt_loader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        seed=int(seed),
    )
    logits_rows: list[np.ndarray] = []
    labels_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device, non_blocking=True).float()
            mask = batch["mask"].to(device, non_blocking=True).bool()
            logits = model(tokens, mask, max_constits=int(tokens.shape[1]))
            logits_rows.append(logits.detach().cpu().numpy().astype(np.float32))
            labels_rows.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
    logits_np = np.concatenate(logits_rows, axis=0) if logits_rows else np.zeros((0, 10), dtype=np.float32)
    labels_np = np.concatenate(labels_rows, axis=0) if labels_rows else np.zeros((0,), dtype=np.int64)
    if logits_np.ndim != 2 or int(logits_np.shape[1]) != 10:
        raise ValueError(f"AV10 model must emit logits with shape [N, 10], got {logits_np.shape}")
    if int(labels_np.shape[0]) != int(logits_np.shape[0]):
        raise ValueError("labels/logits row count mismatch")
    if not np.isfinite(logits_np).all():
        raise FloatingPointError("AV10 prediction logits contain non-finite values")
    return logits_np, labels_np


def save_architecture_view_10class_prediction_cache(
    *,
    prediction_dir: str | Path,
    variant: str,
    split: str,
    logits: np.ndarray,
    labels: np.ndarray,
    jet_ids: Sequence[JetIdentity],
    metadata: Mapping[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Save one variant/split prediction cache block."""

    variant = normalize_architecture_view_variant(variant)
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if logits.ndim != 2 or int(logits.shape[1]) != 10:
        raise ValueError(f"AV10 logits must have shape [N, 10], got {logits.shape}")
    if int(labels.shape[0]) != int(logits.shape[0]):
        raise ValueError("labels/logits row count mismatch")
    if len(jet_ids) != int(labels.shape[0]):
        raise ValueError("jet_ids length does not match labels/logits")
    if not np.array_equal(labels, np.asarray([int(identity.label) for identity in jet_ids], dtype=np.int64)):
        raise ValueError("jet identity labels do not match cached labels/order")
    preds = np.argmax(logits, axis=1).astype(np.int64)
    probs = _softmax_np(logits)
    jet_files, file_indices, entries = _identity_arrays(jet_ids)
    arrays = {
        "logits": logits.astype(np.float32, copy=False),
        "labels": labels.astype(np.int64, copy=False),
        "preds": preds.astype(np.int64, copy=False),
        "probs": probs.astype(np.float32, copy=False),
        "jet_file_indices": file_indices,
        "jet_entries": entries,
    }
    array_path, metadata_path = architecture_view_10class_prediction_paths(prediction_dir, variant, split)
    if not bool(overwrite) and (array_path.exists() or metadata_path.exists()):
        raise FileExistsError(f"AV10 prediction cache already exists: {array_path}")
    array_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(array_path, **arrays)
    prediction_hash = hash_arrays(arrays)
    metrics = classification_metrics_from_predictions(
        preds=preds,
        labels=labels,
        logits=logits,
        label_names=ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
    )
    metrics["binary_projection_metrics"] = architecture_view_binary_projection_metrics(
        logits,
        labels,
        label_names=ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
        pairs=(("QCD", "Hgg"), ("QCD", "Hbb"), ("QCD", "Tbqq"), ("QCD", "Wqq"), ("QCD", "Zqq")),
    )
    payload = {
        **dict(metadata),
        "contract": ARCHITECTURE_VIEW_10CLASS_PREDICTION_CONTRACT,
        "step": ARCHITECTURE_VIEW_10CLASS_PREDICTION_STEP,
        "variant": variant,
        "split": str(split),
        "array_path": str(array_path),
        "metadata_path": str(metadata_path),
        "array_keys": sorted(arrays),
        "prediction_content_hash": prediction_hash,
        "n_jets": int(labels.shape[0]),
        "num_classes": 10,
        "label_names": list(ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES),
        "label_filter": list(ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER),
        "jet_files": jet_files,
        "jet_identity_hash": jet_identity_hash(jet_ids),
        "metrics": metrics,
    }
    save_json(metadata_path, _jsonable(payload))
    return _jsonable(payload)


def load_architecture_view_10class_prediction_metadata(
    prediction_dir: str | Path,
    variant: str,
    split: str,
    *,
    expected_label_names: Sequence[str] = ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES,
    expected_label_filter: Sequence[int] = ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER,
    expected_hlt_content_hash: str | None = None,
    expected_jet_identity_hash: str | None = None,
) -> dict[str, Any]:
    """Load and validate one saved AV10 prediction metadata block."""

    variant = normalize_architecture_view_variant(variant)
    array_path, metadata_path = architecture_view_10class_prediction_paths(prediction_dir, variant, split)
    if not array_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing AV10 prediction cache for {variant}/{split}: {array_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("contract") != ARCHITECTURE_VIEW_10CLASS_PREDICTION_CONTRACT:
        raise ValueError(f"prediction cache contract mismatch for {variant}/{split}")
    if normalize_architecture_view_variant(str(metadata.get("variant"))) != variant:
        raise ValueError(f"prediction cache variant mismatch for {variant}/{split}")
    if str(metadata.get("split")) != str(split):
        raise ValueError(f"prediction cache split mismatch for {variant}/{split}")
    if tuple(str(name) for name in metadata.get("label_names", ())) != tuple(str(name) for name in expected_label_names):
        raise ValueError(f"prediction cache label_names mismatch for {variant}/{split}")
    if tuple(int(index) for index in metadata.get("label_filter", ())) != tuple(int(index) for index in expected_label_filter):
        raise ValueError(f"prediction cache label_filter mismatch for {variant}/{split}")
    if int(metadata.get("num_classes", -1)) != 10:
        raise ValueError(f"prediction cache num_classes mismatch for {variant}/{split}")
    if expected_hlt_content_hash and metadata.get("hlt_content_hash") != expected_hlt_content_hash:
        raise ValueError(f"prediction cache HLT content hash mismatch for {variant}/{split}")
    if expected_jet_identity_hash and metadata.get("jet_identity_hash") != expected_jet_identity_hash:
        raise ValueError(f"prediction cache jet identity hash mismatch for {variant}/{split}")
    with np.load(array_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
        actual_hash = hash_arrays(arrays)
        labels = np.asarray(data["labels"], dtype=np.int64)
        logits = np.asarray(data["logits"], dtype=np.float32)
    if actual_hash != metadata.get("prediction_content_hash"):
        raise ValueError(f"prediction cache content hash mismatch for {variant}/{split}: {actual_hash}")
    if logits.ndim != 2 or int(logits.shape[1]) != 10:
        raise ValueError(f"prediction cache logits shape mismatch for {variant}/{split}: {logits.shape}")
    if int(labels.shape[0]) != int(logits.shape[0]):
        raise ValueError(f"prediction cache labels/logits row mismatch for {variant}/{split}")
    return metadata


def validate_architecture_view_10class_prediction_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate that all cached variants share label order and split row identity."""

    rows = manifest.get("prediction_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("prediction_manifest must contain non-empty prediction_rows")
    expected_labels = tuple(ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES)
    expected_filter = tuple(ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER)
    split_identities: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("prediction_manifest row is not an object")
        labels = tuple(str(name) for name in row.get("label_names", ()))
        label_filter = tuple(int(index) for index in row.get("label_filter", ()))
        if labels != expected_labels or label_filter != expected_filter:
            raise ValueError("prediction_manifest label order mismatch")
        split = str(row.get("split"))
        identity = str(row.get("jet_identity_hash"))
        previous = split_identities.setdefault(split, identity)
        if identity != previous:
            raise ValueError(f"prediction_manifest jet order mismatch for split {split}")


@dataclass(frozen=True)
class ArchitectureView10ClassPredictionCacheConfig:
    """Configuration for Step 2 AV10 prediction caching."""

    output_dir: str
    hlt_cache_dir: str
    checkpoint_root: str
    variants: tuple[str, ...] = ARCHITECTURE_VIEW_10CLASS_DEFAULT_VARIANTS
    splits: tuple[str, ...] = ARCHITECTURE_VIEW_10CLASS_PREDICTION_SPLITS
    batch_size: int = 128
    num_workers: int = 0
    device: str = "auto"
    max_model_val_jets: int | None = None
    max_stack_train_jets: int | None = None
    max_stack_val_jets: int | None = None
    max_final_test_jets: int | None = None
    overwrite: bool = False
    skip_existing: bool = True
    verify_hlt_hash: bool = True
    confirm_final_test: bool = False
    seed: int = 7207

    def __post_init__(self) -> None:
        if not str(self.output_dir):
            raise ValueError("output_dir is required")
        if not str(self.hlt_cache_dir):
            raise ValueError("hlt_cache_dir is required")
        if not str(self.checkpoint_root):
            raise ValueError("checkpoint_root is required")
        variants = tuple(normalize_architecture_view_variant(variant) for variant in self.variants)
        if not variants:
            raise ValueError("at least one AV10 variant is required")
        bad = [variant for variant in variants if variant not in ARCHITECTURE_VIEW_10CLASS_VARIANTS]
        if bad:
            raise ValueError(f"prediction cache runner only accepts AV10 variants, got {bad}")
        splits = tuple(str(split) for split in self.splits)
        if not splits:
            raise ValueError("at least one split is required")
        unknown = [split for split in splits if split not in ARCHITECTURE_VIEW_10CLASS_PREDICTION_SPLITS]
        if unknown:
            raise ValueError(f"unknown prediction splits: {unknown}")
        if "final_test" in splits and not bool(self.confirm_final_test):
            raise ValueError("Refusing to cache final_test predictions without confirm_final_test=True")
        object.__setattr__(self, "variants", variants)
        object.__setattr__(self, "splits", splits)
        for name in ("batch_size",):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if int(self.num_workers) < 0:
            raise ValueError("num_workers cannot be negative")
        object.__setattr__(self, "num_workers", int(self.num_workers))
        for name in ("max_model_val_jets", "max_stack_train_jets", "max_stack_val_jets", "max_final_test_jets"):
            value = getattr(self, name)
            if value is not None:
                value = int(value)
                if value <= 0:
                    raise ValueError(f"{name} must be positive when provided")
                object.__setattr__(self, name, value)
        object.__setattr__(self, "seed", int(self.seed))


def cache_architecture_view_10class_predictions(
    config: ArchitectureView10ClassPredictionCacheConfig,
    *,
    models_by_variant: Mapping[str, Any] | None = None,
    payloads_by_variant: Mapping[str, Mapping[str, Any]] | None = None,
    views_by_split: Mapping[str, JetView] | None = None,
) -> dict[str, Any]:
    """Write AV10 prediction caches for requested variants and splits."""

    torch = require_torch()
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    prediction_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    by_variant: dict[str, dict[str, Any]] = {}
    for variant_index, variant in enumerate(config.variants):
        checkpoint = _checkpoint_path(config.checkpoint_root, variant)
        if models_by_variant is not None and variant in models_by_variant:
            model = models_by_variant[variant]
            payload = dict(payloads_by_variant.get(variant, {})) if payloads_by_variant is not None else {}
            checkpoint_hash = None
        else:
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing AV10 checkpoint for {variant}: {checkpoint}")
            model, payload = load_architecture_view_tagger_checkpoint(checkpoint, device=device)
            checkpoint_hash = sha256_file(checkpoint)
        _require_av10_payload(payload, variant=variant, checkpoint=checkpoint)
        model = model.to(device) if hasattr(model, "to") else model
        by_variant[variant] = {}
        for split_index, split in enumerate(config.splits):
            array_path, _metadata_path = architecture_view_10class_prediction_paths(prediction_dir, variant, split)
            if array_path.exists() and bool(config.skip_existing) and not bool(config.overwrite):
                dataset_for_hash = _load_or_build_dataset(
                    hlt_cache_dir=config.hlt_cache_dir,
                    split=split,
                    max_jets=_max_jets_for_split(config, split),
                    views_by_split=views_by_split,
                    verify_hlt_hash=bool(config.verify_hlt_hash),
                )
                metadata = load_architecture_view_10class_prediction_metadata(
                    prediction_dir,
                    variant,
                    split,
                    expected_hlt_content_hash=dataset_for_hash.metadata.get("hlt_content_hash"),
                    expected_jet_identity_hash=jet_identity_hash(dataset_for_hash.jet_ids),
                )
                by_variant[variant][split] = metadata
                rows.append(metadata)
                continue
            dataset = _load_or_build_dataset(
                hlt_cache_dir=config.hlt_cache_dir,
                split=split,
                max_jets=_max_jets_for_split(config, split),
                views_by_split=views_by_split,
                verify_hlt_hash=bool(config.verify_hlt_hash),
            )
            logits, labels = _collect_logits_for_dataset(
                model,
                dataset,
                batch_size=int(config.batch_size),
                num_workers=int(config.num_workers),
                device=device,
                seed=int(config.seed) + 1009 * (variant_index + 1) + 37 * (split_index + 1),
            )
            metadata = save_architecture_view_10class_prediction_cache(
                prediction_dir=prediction_dir,
                variant=variant,
                split=split,
                logits=logits,
                labels=labels,
                jet_ids=dataset.jet_ids,
                metadata={
                    "checkpoint": str(checkpoint),
                    "checkpoint_hash": checkpoint_hash,
                    "checkpoint_epoch": payload.get("epoch"),
                    "checkpoint_selection_metric": payload.get("selection_metric")
                    or (payload.get("config") or {}).get("selection_metric"),
                    "checkpoint_best_model_val_metric": (payload.get("metrics") or {}).get("selection_metric_value"),
                    "model_output_contract": payload.get("output_contract"),
                    "model_variant_behavior": payload.get("variant_behavior"),
                    "hlt_content_hash": dataset.metadata.get("hlt_content_hash"),
                    "source_hlt_jet_identity_hash": dataset.metadata.get("jet_identity_hash"),
                    "hlt_params": dataset.metadata.get("hlt_params"),
                    "hlt_seed": dataset.metadata.get("hlt_seed"),
                    "max_jets": _max_jets_for_split(config, split),
                    "dataset_metadata": dict(dataset.metadata),
                    "allowed_inputs": "cached_fixed_hlt_only",
                },
                overwrite=bool(config.overwrite),
            )
            by_variant[variant][split] = metadata
            rows.append(metadata)
        if models_by_variant is None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    manifest = {
        "contract": ARCHITECTURE_VIEW_10CLASS_PREDICTION_CONTRACT,
        "step": ARCHITECTURE_VIEW_10CLASS_PREDICTION_STEP,
        "output_dir": str(output_dir),
        "prediction_dir": str(prediction_dir),
        "config": asdict(config),
        "source": source_metadata(),
        "variants": list(config.variants),
        "splits": list(config.splits),
        "label_names": list(ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES),
        "label_filter": list(ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER),
        "prediction_rows": rows,
        "predictions": by_variant,
    }
    validate_architecture_view_10class_prediction_manifest(manifest)
    save_json(output_dir / "prediction_manifest.json", _jsonable(manifest))
    return _jsonable(manifest)


__all__ = [
    "ARCHITECTURE_VIEW_10CLASS_PREDICTION_CONTRACT",
    "ARCHITECTURE_VIEW_10CLASS_PREDICTION_SPLITS",
    "ARCHITECTURE_VIEW_10CLASS_PREDICTION_STEP",
    "ArchitectureView10ClassPredictionCacheConfig",
    "architecture_view_10class_prediction_paths",
    "cache_architecture_view_10class_predictions",
    "load_architecture_view_10class_prediction_metadata",
    "save_architecture_view_10class_prediction_cache",
    "validate_architecture_view_10class_prediction_manifest",
]
