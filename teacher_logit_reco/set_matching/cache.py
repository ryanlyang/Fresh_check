"""Cache reconstructed views from trained set-matching reconstructors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json
from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash
from jetclass_fresh.jetclass_data import LABEL_NAMES, manifest_hash

from .data import (
    SetMatchingJetDataset,
    audit_set_matching_pair,
    load_manifest_for_set_matching,
    load_set_matching_dataset,
    make_set_matching_loader,
)
from .experiment import SetMatchingMultiViewLayout, normalize_split_name
from .losses import SetMatchingLossConfig, compute_set_matching_loss
from .reconstructors import (
    SET_MATCHING_RECONSTRUCTOR_CONTRACT,
    SetMatchingReconstructorAdapter,
    load_set_matching_reconstructor_checkpoint,
    normalize_set_matching_reconstructor_architecture,
)
from .train import source_metadata


SET_MATCHING_CACHE_STEP = "set_matching_multiview_step6_cache_reconstructed_views"
DEFAULT_CACHE_SPLITS: tuple[str, ...] = ("stack_train", "stack_val", "final_test")
CACHE_ARRAY_VERSION = 1


def _optional_nonnegative_int(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative when provided")
    return value


def _positive_int(value: int, *, field_name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _metadata_path_for_npz(path: Path) -> Path:
    return path.with_name(f"{path.stem}_metadata.json")


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return value


def _identity_arrays(jet_ids) -> tuple[list[str], np.ndarray, np.ndarray]:
    unique_files: list[str] = []
    file_to_index: dict[str, int] = {}
    file_indices = np.zeros((len(jet_ids),), dtype=np.int32)
    entries = np.zeros((len(jet_ids),), dtype=np.int64)
    for index, identity in enumerate(jet_ids):
        if identity.file not in file_to_index:
            file_to_index[identity.file] = len(unique_files)
            unique_files.append(identity.file)
        file_indices[index] = file_to_index[identity.file]
        entries[index] = int(identity.entry)
    return unique_files, file_indices, entries


def _loss_config_from_payload(payload: Mapping[str, Any]) -> SetMatchingLossConfig:
    raw = dict(payload.get("loss_config") or {})
    allowed = set(SetMatchingLossConfig.__dataclass_fields__)
    kwargs = {key: value for key, value in raw.items() if key in allowed}
    return SetMatchingLossConfig(**kwargs)


class _MetricAccumulator:
    def __init__(self) -> None:
        self.weight_sum = 0.0
        self.totals: dict[str, float] = {}

    def update(self, values: Mapping[str, Any], *, weight: int) -> None:
        weight = int(weight)
        if weight <= 0:
            return
        self.weight_sum += float(weight)
        for key, value in values.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric):
                continue
            self.totals[str(key)] = self.totals.get(str(key), 0.0) + numeric * float(weight)

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


def _as_numpy(value, *, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _pad_concat_3d(chunks: Sequence[np.ndarray], *, dtype: np.dtype, pad_value: float = 0.0) -> np.ndarray:
    if not chunks:
        raise ValueError("cannot concatenate an empty 3D chunk list")
    feature_dims = {int(chunk.shape[2]) for chunk in chunks}
    if len(feature_dims) != 1:
        raise ValueError(f"3D chunks have inconsistent feature dimensions: {sorted(feature_dims)}")
    total = sum(int(chunk.shape[0]) for chunk in chunks)
    max_slots = max(int(chunk.shape[1]) for chunk in chunks)
    feature_dim = next(iter(feature_dims))
    output = np.full((total, max_slots, feature_dim), pad_value, dtype=dtype)
    cursor = 0
    for chunk in chunks:
        rows = int(chunk.shape[0])
        slots = int(chunk.shape[1])
        output[cursor : cursor + rows, :slots] = chunk.astype(dtype, copy=False)
        cursor += rows
    return output


def _pad_concat_2d(chunks: Sequence[np.ndarray], *, dtype: np.dtype, pad_value: float | bool = 0.0) -> np.ndarray:
    if not chunks:
        raise ValueError("cannot concatenate an empty 2D chunk list")
    total = sum(int(chunk.shape[0]) for chunk in chunks)
    max_slots = max(int(chunk.shape[1]) for chunk in chunks)
    output = np.full((total, max_slots), pad_value, dtype=dtype)
    cursor = 0
    for chunk in chunks:
        rows = int(chunk.shape[0])
        slots = int(chunk.shape[1])
        output[cursor : cursor + rows, :slots] = chunk.astype(dtype, copy=False)
        cursor += rows
    return output


def _confidence_from_output(output):
    torch = require_torch()
    if output.candidate_weights is not None:
        return output.candidate_weights
    return torch.sigmoid(output.existence_logits)


@dataclass
class SetMatchingRecoViewCacheConfig:
    """Configuration for Step 6 reconstructed-view cache generation."""

    output_dir: str
    manifest_path: str
    hlt_cache_dir: str
    reconstructor_checkpoint: str
    architecture: str | None = None
    data_dir: str | None = None
    splits: tuple[str, ...] = DEFAULT_CACHE_SPLITS
    batch_size: int = 128
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    max_jets_per_split: int | None = None
    label_filter: tuple[int, ...] = ()
    overwrite: bool = False
    skip_existing: bool = True
    confirm_final_test: bool = False
    strict_checkpoint: bool = True
    compute_set_matching_metrics: bool = True
    trim_to_valid: bool = False
    verify_hlt_hash: bool = True
    verify_label_branches: bool = False
    read_chunk_size: int = 50_000
    seed: int = 1205

    def __post_init__(self) -> None:
        if not str(self.output_dir):
            raise ValueError("output_dir is required")
        if not str(self.manifest_path):
            raise ValueError("manifest_path is required")
        if not str(self.hlt_cache_dir):
            raise ValueError("hlt_cache_dir is required")
        if not str(self.reconstructor_checkpoint):
            raise ValueError("reconstructor_checkpoint is required")
        if self.architecture is not None:
            self.architecture = normalize_set_matching_reconstructor_architecture(self.architecture)
        self.splits = tuple(normalize_split_name(split) for split in self.splits)
        if not self.splits:
            raise ValueError("at least one split is required")
        if len(set(self.splits)) != len(self.splits):
            raise ValueError(f"splits contain duplicates: {self.splits}")
        if "final_test" in self.splits and not bool(self.confirm_final_test):
            raise ValueError("Set --confirm-final-test to cache final_test reconstructed views")
        self.batch_size = _positive_int(self.batch_size, field_name="batch_size")
        if int(self.num_workers) < 0:
            raise ValueError("num_workers cannot be negative")
        self.max_jets_per_split = _optional_nonnegative_int(self.max_jets_per_split, field_name="max_jets_per_split")
        if self.max_jets_per_split == 0:
            raise ValueError("max_jets_per_split must be positive when provided")
        self.label_filter = tuple(int(label) for label in self.label_filter)
        if len(set(self.label_filter)) != len(self.label_filter):
            raise ValueError(f"label_filter contains duplicates: {self.label_filter}")
        self.read_chunk_size = _positive_int(self.read_chunk_size, field_name="read_chunk_size")

    @property
    def layout(self) -> SetMatchingMultiViewLayout:
        output_path = Path(self.output_dir)
        return SetMatchingMultiViewLayout(output_root=output_path.parent, experiment_name=output_path.name)

    def cache_path(self, architecture: str, split: str) -> Path:
        return self.layout.reconstructed_view_cache_path(architecture, split)

    def metadata_path(self, architecture: str, split: str) -> Path:
        return _metadata_path_for_npz(self.cache_path(architecture, split))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_dataset_for_split(config: SetMatchingRecoViewCacheConfig, split: str) -> SetMatchingJetDataset:
    return load_set_matching_dataset(
        manifest_path=config.manifest_path,
        hlt_cache_dir=config.hlt_cache_dir,
        split=split,
        data_dir=config.data_dir,
        max_jets=config.max_jets_per_split,
        label_filter=config.label_filter,
        trim_to_valid=bool(config.trim_to_valid),
        verify_hlt_hash=bool(config.verify_hlt_hash),
        verify_label_branches=bool(config.verify_label_branches),
        read_chunk_size=int(config.read_chunk_size),
    )


def cache_set_matching_reco_split(
    *,
    config: SetMatchingRecoViewCacheConfig,
    model: SetMatchingReconstructorAdapter,
    checkpoint_payload: Mapping[str, Any],
    architecture: str,
    split: str,
    device,
    dataset: SetMatchingJetDataset | None = None,
    manifest=None,
    loss_config: SetMatchingLossConfig | None = None,
) -> dict[str, Any]:
    """Generate and save one reconstructed-view cache split."""

    torch = require_torch()
    split = normalize_split_name(split)
    architecture = normalize_set_matching_reconstructor_architecture(architecture)
    cache_path = config.cache_path(architecture, split)
    metadata_path = config.metadata_path(architecture, split)
    if cache_path.exists() or metadata_path.exists():
        if config.overwrite:
            cache_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
        elif config.skip_existing and cache_path.exists() and metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            return {"split": split, "architecture": architecture, "skipped_existing": True, **metadata}
        else:
            raise FileExistsError(f"Reconstructed-view cache already exists: {cache_path}")

    dataset = dataset or _load_dataset_for_split(config, split)
    alignment_audit = audit_set_matching_pair(dataset, manifest=manifest, expected_split=split)
    if not alignment_audit.get("ok", False):
        raise ValueError(f"Set-matching cache alignment audit failed for {split}: {alignment_audit['problems']}")

    loader = make_set_matching_loader(
        dataset,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        seed=int(config.seed),
    )
    loss_config = loss_config or _loss_config_from_payload(checkpoint_payload)
    autocast_enabled = bool(config.amp and getattr(device, "type", None) == "cuda")
    model.eval()

    feature_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []
    confidence_chunks: list[np.ndarray] = []
    logit_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    accumulator = _MetricAccumulator()
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                output = model(
                    batch["hlt_tokens"],
                    batch["hlt_mask"],
                    labels=batch["labels"],
                    jet_ids=batch.get("jet_ids"),
                    split=split,
                )
                if config.compute_set_matching_metrics:
                    loss_output = compute_set_matching_loss(
                        **output.to_loss_kwargs(
                            offline_features=batch["offline_tokens"],
                            offline_mask=batch["offline_mask"],
                            hlt_features=batch["hlt_tokens"],
                            hlt_mask=batch["hlt_mask"],
                        ),
                        config=loss_config,
                    )
                    metrics = loss_output.detached_float_dict(prefix="")
                    metrics.update(
                        {
                            f"reco_{key}": value
                            for key, value in output.diagnostics.items()
                            if isinstance(value, (int, float))
                        }
                    )
                    accumulator.update(metrics, weight=int(batch["labels"].shape[0]))

            feature_chunks.append(_as_numpy(output.predicted_features, dtype=np.float32))
            mask_chunks.append(_as_numpy(output.candidate_mask, dtype=bool))
            confidence_chunks.append(_as_numpy(_confidence_from_output(output), dtype=np.float32))
            logit_chunks.append(_as_numpy(output.existence_logits, dtype=np.float32))
            label_chunks.append(_as_numpy(batch["labels"], dtype=np.int64))
            index_chunks.append(_as_numpy(batch["indices"], dtype=np.int64))
            num_batches += 1

    features = _pad_concat_3d(feature_chunks, dtype=np.float32)
    masks = _pad_concat_2d(mask_chunks, dtype=bool, pad_value=False)
    confidences = _pad_concat_2d(confidence_chunks, dtype=np.float32)
    existence_logits = _pad_concat_2d(logit_chunks, dtype=np.float32)
    labels = np.concatenate(label_chunks, axis=0).astype(np.int64, copy=False)
    indices = np.concatenate(index_chunks, axis=0).astype(np.int64, copy=False)
    if features.shape[0] != len(dataset):
        raise ValueError(f"Cached row count {features.shape[0]} != dataset length {len(dataset)} for {split}")
    if not np.array_equal(labels, dataset.labels.astype(np.int64, copy=False)):
        raise ValueError(f"Cached labels do not match dataset labels for {split}")
    if not np.array_equal(indices, np.arange(len(dataset), dtype=np.int64)):
        raise ValueError(f"Cached dataset indices are not sequential for {split}")

    jet_files, jet_file_indices, jet_entries = _identity_arrays(dataset.jet_ids)
    arrays = {
        "tokens": features,
        "features": features,
        "mask": masks,
        "candidate_mask": masks,
        "confidence": confidences,
        "candidate_weights": confidences,
        "existence_logits": existence_logits,
        "labels": labels,
        "indices": indices,
        "jet_file_indices": jet_file_indices,
        "jet_entries": jet_entries,
    }
    content_hash = hash_arrays(
        {
            "tokens": features,
            "mask": masks,
            "confidence": confidences,
            "existence_logits": existence_logits,
            "labels": labels,
            "indices": indices,
            "jet_file_indices": jet_file_indices,
            "jet_entries": jet_entries,
        }
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)

    source_checkpoint = str(Path(config.reconstructor_checkpoint))
    checkpoint_sha256 = _file_sha256(source_checkpoint) if Path(source_checkpoint).exists() else None
    metrics = accumulator.summary() if config.compute_set_matching_metrics else {}
    candidate_counts = masks.sum(axis=1).astype(np.float64)
    active_counts = (confidences * masks.astype(np.float32)).sum(axis=1).astype(np.float64)
    metadata = {
        "version": CACHE_ARRAY_VERSION,
        "experiment_step": SET_MATCHING_CACHE_STEP,
        "output_contract": SET_MATCHING_RECONSTRUCTOR_CONTRACT,
        "architecture": architecture,
        "split": split,
        "n_jets": int(features.shape[0]),
        "n_candidates": int(features.shape[1]),
        "feature_dim": int(features.shape[2]),
        "array_path": str(cache_path),
        "metadata_path": str(metadata_path),
        "array_keys": sorted(arrays.keys()),
        "source_checkpoint": source_checkpoint,
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_checkpoint_epoch": checkpoint_payload.get("epoch"),
        "source_checkpoint_experiment_step": checkpoint_payload.get("experiment_step"),
        "source_checkpoint_metrics": checkpoint_payload.get("metrics"),
        "source_checkpoint_model_config": checkpoint_payload.get("model_config"),
        "source_checkpoint_loss_config": checkpoint_payload.get("loss_config"),
        "loss_config_used_for_metrics": loss_config.to_dict(),
        "label_names": list(LABEL_NAMES),
        "jet_files": jet_files,
        "jet_identity_hash": jet_identity_hash(dataset.jet_ids),
        "cache_content_hash": content_hash,
        "alignment_audit": alignment_audit,
        "dataset_metadata": dict(dataset.metadata),
        "heldout_set_matching_metrics": metrics,
        "num_batches": int(num_batches),
        "candidate_count_summary": {
            "min": float(candidate_counts.min()) if candidate_counts.size else 0.0,
            "mean": float(candidate_counts.mean()) if candidate_counts.size else 0.0,
            "max": float(candidate_counts.max()) if candidate_counts.size else 0.0,
            "std": float(candidate_counts.std()) if candidate_counts.size else 0.0,
        },
        "active_confidence_count_summary": {
            "min": float(active_counts.min()) if active_counts.size else 0.0,
            "mean": float(active_counts.mean()) if active_counts.size else 0.0,
            "max": float(active_counts.max()) if active_counts.size else 0.0,
            "std": float(active_counts.std()) if active_counts.size else 0.0,
        },
        "source": source_metadata(),
        "config": config.to_dict(),
        "leakage_rule": (
            "Step 6 freezes a model_val-selected reconstructor and applies it to stack_train, "
            "stack_val, and final_test using fixed HLT inputs. Offline constituents are loaded "
            "only to audit/evaluate set-matching cache quality; cached inference arrays are "
            "reconstructor outputs produced from HLT tokens."
        ),
        "offline_targets_not_saved": True,
        "inference_consumes_hlt_only": True,
    }
    save_json(metadata_path, _jsonable(metadata))
    return dict(metadata)


def cache_set_matching_reco_views(
    config: SetMatchingRecoViewCacheConfig,
    *,
    model: SetMatchingReconstructorAdapter | None = None,
    checkpoint_payload: Mapping[str, Any] | None = None,
    datasets: Mapping[str, SetMatchingJetDataset] | None = None,
) -> dict[str, Any]:
    """Cache reconstructed views for all configured heldout splits."""

    torch = require_torch()
    device = resolve_device(config.device)
    if model is None:
        model, loaded_payload = load_set_matching_reconstructor_checkpoint(
            config.reconstructor_checkpoint,
            device=device,
            strict=bool(config.strict_checkpoint),
            expected_architecture=config.architecture,
        )
        checkpoint_payload = loaded_payload
    else:
        model = model.to(device)
        model.eval()
        checkpoint_payload = dict(checkpoint_payload or {})

    architecture = normalize_set_matching_reconstructor_architecture(
        config.architecture or getattr(model, "architecture", checkpoint_payload.get("set_matching_architecture", "gt"))
    )
    loss_config = _loss_config_from_payload(checkpoint_payload or {})
    manifest = None
    manifest_sha = None
    try:
        manifest = load_manifest_for_set_matching(config.manifest_path)
        manifest_sha = manifest_hash(manifest)
    except Exception:
        manifest = None
        manifest_sha = None

    split_reports: dict[str, Any] = {}
    for split in config.splits:
        split_reports[split] = cache_set_matching_reco_split(
            config=config,
            model=model,
            checkpoint_payload=checkpoint_payload or {},
            architecture=architecture,
            split=split,
            device=device,
            dataset=None if datasets is None else datasets.get(split),
            manifest=manifest,
            loss_config=loss_config,
        )

    report = {
        "experiment_step": SET_MATCHING_CACHE_STEP,
        "architecture": architecture,
        "splits": list(config.splits),
        "output_dir": str(Path(config.output_dir)),
        "reconstructor_checkpoint": str(Path(config.reconstructor_checkpoint)),
        "manifest_hash": manifest_sha,
        "cache_paths": {
            split: str(config.cache_path(architecture, split))
            for split in config.splits
        },
        "metadata_paths": {
            split: str(config.metadata_path(architecture, split))
            for split in config.splits
        },
        "split_reports": split_reports,
        "source": source_metadata(),
    }
    report_path = Path(config.output_dir) / "reconstructed_views" / architecture / "cache_report.json"
    save_json(report_path, _jsonable(report))
    report["report_path"] = str(report_path)
    torch.cuda.empty_cache() if getattr(device, "type", None) == "cuda" else None
    return _jsonable(report)


__all__ = [
    "CACHE_ARRAY_VERSION",
    "DEFAULT_CACHE_SPLITS",
    "SET_MATCHING_CACHE_STEP",
    "SetMatchingRecoViewCacheConfig",
    "cache_set_matching_reco_split",
    "cache_set_matching_reco_views",
]
