"""Cache DETR/free-slot reconstructed views for downstream five-view taggers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, save_json
from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash
from jetclass_fresh.jetclass_data import LABEL_NAMES, manifest_hash

from teacher_logit_reco.set_matching.data import (
    SetMatchingJetDataset,
    audit_set_matching_pair,
    load_manifest_for_set_matching,
    load_set_matching_dataset,
    make_set_matching_loader,
)
from teacher_logit_reco.set_matching.train import source_metadata

from .experiment import (
    DETR_SLOT_DEFAULT_CONFIDENCE_THRESHOLD,
    DETR_SLOT_DEFAULT_MIN_TOKENS_PER_VIEW,
    DETR_SLOT_RECONSTRUCTOR_CONTRACT,
    normalize_detr_slot_encoder_architecture,
    normalize_detr_slot_split_name,
)
from .losses import DetrSlotHungarianLossConfig, compute_detr_slot_hungarian_loss
from .outputs import DetrSlotOutput
from .training import DetrSlotReconstructor, DetrSlotReconstructorTrainConfig, build_detr_slot_reconstructor


DETR_SLOT_CACHE_STEP = "detr_free_slot_step12_cache_reconstructed_views"
DEFAULT_DETR_SLOT_CACHE_SPLITS: tuple[str, ...] = ("stack_train", "stack_val", "final_test")
DETR_SLOT_CACHE_ARRAY_VERSION = 1


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


def _as_numpy(value, *, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _move_batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("hlt_tokens", "hlt_mask", "offline_tokens", "offline_mask", "labels", "indices"):
        value = moved.get(key)
        if hasattr(value, "to"):
            moved[key] = value.to(device)
    return moved


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


def _loss_config_from_payload(payload: Mapping[str, Any]) -> DetrSlotHungarianLossConfig:
    raw = dict(payload.get("loss_config") or {})
    allowed = set(DetrSlotHungarianLossConfig.__dataclass_fields__)
    kwargs = {key: value for key, value in raw.items() if key in allowed}
    return DetrSlotHungarianLossConfig(**kwargs)


def _training_config_from_payload(payload: Mapping[str, Any], *, checkpoint_path: str | Path) -> DetrSlotReconstructorTrainConfig:
    raw = dict(payload.get("config") or {})
    raw.setdefault("output_dir", str(Path(checkpoint_path).parent))
    raw.setdefault("manifest_path", "unused_manifest.json.gz")
    raw.setdefault("hlt_cache_dir", "unused_hlt_cache")
    raw["confirm_split_settings"] = True
    return DetrSlotReconstructorTrainConfig(**raw)


def load_detr_slot_reconstructor_checkpoint(
    checkpoint_path: str | Path,
    *,
    device=None,
    strict: bool = True,
    expected_architecture: str | None = None,
) -> tuple[DetrSlotReconstructor, dict[str, Any]]:
    """Load a Step 11 DETR/free-slot reconstructor checkpoint."""

    torch = require_torch()
    resolved_device = device or resolve_device("auto")
    payload = torch.load(checkpoint_path, map_location=resolved_device)
    if not isinstance(payload, Mapping):
        raise TypeError(f"DETR-slot checkpoint must contain a mapping, got {type(payload).__name__}")
    state_dict = payload.get("model_state_dict")
    if state_dict is None:
        raise KeyError(f"DETR-slot checkpoint {checkpoint_path} is missing model_state_dict")
    config = _training_config_from_payload(payload, checkpoint_path=checkpoint_path)
    architecture = normalize_detr_slot_encoder_architecture(str(payload.get("architecture") or config.architecture))
    if expected_architecture is not None:
        expected = normalize_detr_slot_encoder_architecture(expected_architecture)
        if architecture != expected:
            raise ValueError(f"Checkpoint architecture {architecture!r} does not match expected {expected!r}")
    if config.architecture != architecture:
        config_payload = asdict(config)
        config_payload["architecture"] = architecture
        config = DetrSlotReconstructorTrainConfig(**config_payload)

    model = build_detr_slot_reconstructor(config)
    load_info = model.load_state_dict(state_dict, strict=bool(strict))
    model.to(resolved_device)
    model.eval()
    payload = dict(payload)
    payload["checkpoint_load_info"] = {
        "missing_keys": list(getattr(load_info, "missing_keys", [])),
        "unexpected_keys": list(getattr(load_info, "unexpected_keys", [])),
        "strict": bool(strict),
    }
    return model, payload


@dataclass
class DetrSlotRecoViewCacheConfig:
    """Configuration for Step 12 DETR reconstructed-view cache generation."""

    output_dir: str
    manifest_path: str
    hlt_cache_dir: str
    reconstructor_checkpoint: str
    architecture: str | None = None
    data_dir: str | None = None
    splits: tuple[str, ...] = DEFAULT_DETR_SLOT_CACHE_SPLITS
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
    compute_detr_metrics: bool = True
    export_max_tokens: int | None = None
    confidence_threshold: float = DETR_SLOT_DEFAULT_CONFIDENCE_THRESHOLD
    min_tokens_per_view: int = DETR_SLOT_DEFAULT_MIN_TOKENS_PER_VIEW
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
            self.architecture = normalize_detr_slot_encoder_architecture(self.architecture)
        self.splits = tuple(normalize_detr_slot_split_name(split) for split in self.splits)
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
        if self.export_max_tokens is not None:
            self.export_max_tokens = _positive_int(self.export_max_tokens, field_name="export_max_tokens")
        if int(self.min_tokens_per_view) < 0:
            raise ValueError("min_tokens_per_view cannot be negative")
        if self.export_max_tokens is not None and int(self.min_tokens_per_view) > int(self.export_max_tokens):
            raise ValueError("min_tokens_per_view cannot exceed export_max_tokens")
        if not 0.0 <= float(self.confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.label_filter = tuple(int(label) for label in self.label_filter)
        if len(set(self.label_filter)) != len(self.label_filter):
            raise ValueError(f"label_filter contains duplicates: {self.label_filter}")
        self.read_chunk_size = _positive_int(self.read_chunk_size, field_name="read_chunk_size")

    def cache_path(self, architecture: str, split: str) -> Path:
        arch = normalize_detr_slot_encoder_architecture(architecture)
        normalized_split = normalize_detr_slot_split_name(split)
        return Path(self.output_dir) / arch / f"{normalized_split}_reconstructed_view.npz"

    def metadata_path(self, architecture: str, split: str) -> Path:
        return _metadata_path_for_npz(self.cache_path(architecture, split))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_dataset_for_split(config: DetrSlotRecoViewCacheConfig, split: str) -> SetMatchingJetDataset:
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


def _safe_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "std": 0.0}
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
        "std": float(values.std()),
    }


def _select_top_detr_slots(
    output: DetrSlotOutput,
    *,
    export_max_tokens: int,
    confidence_threshold: float,
    min_tokens_per_view: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return top DETR slots ordered by existence probability.

    Returned arrays are ``tokens, mask, confidence, existence_logits,
    source_indices``.  Tokens are hard-sanitized export features, not the
    smooth loss-facing representation.
    """

    tokens = _as_numpy(output.export_features, dtype=np.float32)
    candidate_mask = _as_numpy(output.candidate_mask, dtype=bool)
    confidence = _as_numpy(output.candidate_weights, dtype=np.float32)
    existence_logits = _as_numpy(output.existence_logits, dtype=np.float32)
    if tokens.ndim != 3:
        raise ValueError(f"tokens must be [batch, slots, features], got {tokens.shape}")
    if candidate_mask.shape != tokens.shape[:2]:
        raise ValueError("candidate_mask shape does not match token leading dimensions")
    if confidence.shape != tokens.shape[:2]:
        raise ValueError("confidence shape does not match token leading dimensions")
    if existence_logits.shape != tokens.shape[:2]:
        raise ValueError("existence_logits shape does not match token leading dimensions")

    batch_size, _, feature_dim = tokens.shape
    k = int(export_max_tokens)
    threshold = float(confidence_threshold)
    min_tokens = int(min_tokens_per_view)
    selected_tokens = np.zeros((batch_size, k, feature_dim), dtype=np.float32)
    selected_mask = np.zeros((batch_size, k), dtype=bool)
    selected_confidence = np.zeros((batch_size, k), dtype=np.float32)
    selected_logits = np.zeros((batch_size, k), dtype=np.float32)
    source_indices = np.full((batch_size, k), -1, dtype=np.int32)

    for row in range(batch_size):
        valid = np.flatnonzero(candidate_mask[row])
        if valid.size == 0:
            continue
        order = np.lexsort((valid, -confidence[row, valid]))
        ordered = valid[order][:k]
        length = int(ordered.size)
        selected_tokens[row, :length] = tokens[row, ordered]
        selected_confidence[row, :length] = confidence[row, ordered]
        selected_logits[row, :length] = existence_logits[row, ordered]
        source_indices[row, :length] = ordered.astype(np.int32, copy=False)

        active = selected_confidence[row, :length] >= threshold
        if min_tokens > 0 and int(active.sum()) < min(min_tokens, length):
            active[: min(min_tokens, length)] = True
        selected_mask[row, :length] = active

    return selected_tokens, selected_mask, selected_confidence, selected_logits, source_indices


def _write_cache_summary_csv(path: Path, split_reports: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, report in split_reports.items():
        rows.append(
            {
                "split": split,
                "architecture": report.get("architecture"),
                "n_jets": report.get("n_jets"),
                "n_candidates": report.get("n_candidates"),
                "candidate_count_mean": (report.get("candidate_count_summary") or {}).get("mean"),
                "active_confidence_count_mean": (report.get("active_confidence_count_summary") or {}).get("mean"),
                "exported_tokens_mean": (report.get("exported_tokens_summary") or {}).get("mean"),
                "top_existence_score_mean": report.get("top_existence_score_mean"),
                "nonfinite_count": report.get("nonfinite_count"),
                "skipped_existing": bool(report.get("skipped_existing", False)),
                "cache_path": report.get("array_path"),
            }
        )
    fieldnames = list(rows[0].keys()) if rows else ["split"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cache_detr_slot_reco_split(
    *,
    config: DetrSlotRecoViewCacheConfig,
    model: DetrSlotReconstructor,
    checkpoint_payload: Mapping[str, Any],
    architecture: str,
    split: str,
    device,
    dataset: SetMatchingJetDataset | None = None,
    manifest=None,
    loss_config: DetrSlotHungarianLossConfig | None = None,
) -> dict[str, Any]:
    """Generate and save one DETR reconstructed-view cache split."""

    torch = require_torch()
    split = normalize_detr_slot_split_name(split)
    architecture = normalize_detr_slot_encoder_architecture(architecture)
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
            raise FileExistsError(f"DETR reconstructed-view cache already exists: {cache_path}")

    dataset = dataset or _load_dataset_for_split(config, split)
    alignment_audit = audit_set_matching_pair(dataset, manifest=manifest, expected_split=split)
    if not alignment_audit.get("ok", False):
        raise ValueError(f"DETR-slot cache alignment audit failed for {split}: {alignment_audit['problems']}")

    train_config = _training_config_from_payload(checkpoint_payload, checkpoint_path=config.reconstructor_checkpoint)
    export_max_tokens = int(config.export_max_tokens or train_config.export_max_tokens)
    if export_max_tokens > int(train_config.num_slots):
        raise ValueError("export_max_tokens cannot exceed the checkpoint model num_slots")
    min_tokens = int(config.min_tokens_per_view)
    if min_tokens > export_max_tokens:
        raise ValueError("min_tokens_per_view cannot exceed export_max_tokens")

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
    source_index_chunks: list[np.ndarray] = []
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
                if config.compute_detr_metrics:
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
                    metrics = loss_output.detached_float_dict(prefix="")
                    metrics.update(
                        {
                            f"reco_{key}": value
                            for key, value in output.diagnostics.items()
                            if isinstance(value, (int, float))
                        }
                    )
                    accumulator.update(metrics, weight=int(batch["labels"].shape[0]))

            selected = _select_top_detr_slots(
                output,
                export_max_tokens=export_max_tokens,
                confidence_threshold=float(config.confidence_threshold),
                min_tokens_per_view=min_tokens,
            )
            selected_features, selected_mask, selected_confidence, selected_logits, source_indices = selected
            feature_chunks.append(selected_features)
            mask_chunks.append(selected_mask)
            confidence_chunks.append(selected_confidence)
            logit_chunks.append(selected_logits)
            source_index_chunks.append(source_indices)
            label_chunks.append(_as_numpy(batch["labels"], dtype=np.int64))
            index_chunks.append(_as_numpy(batch["indices"], dtype=np.int64))
            num_batches += 1

    features = np.concatenate(feature_chunks, axis=0).astype(np.float32, copy=False)
    masks = np.concatenate(mask_chunks, axis=0).astype(bool, copy=False)
    confidences = np.concatenate(confidence_chunks, axis=0).astype(np.float32, copy=False)
    existence_logits = np.concatenate(logit_chunks, axis=0).astype(np.float32, copy=False)
    source_indices = np.concatenate(source_index_chunks, axis=0).astype(np.int32, copy=False)
    labels = np.concatenate(label_chunks, axis=0).astype(np.int64, copy=False)
    indices = np.concatenate(index_chunks, axis=0).astype(np.int64, copy=False)
    if features.shape[0] != len(dataset):
        raise ValueError(f"Cached row count {features.shape[0]} != dataset length {len(dataset)} for {split}")
    if features.shape[-1] != dataset.feature_dim:
        raise ValueError(f"Cached feature dim {features.shape[-1]} != dataset feature dim {dataset.feature_dim}")
    if not np.array_equal(labels, dataset.labels.astype(np.int64, copy=False)):
        raise ValueError(f"Cached labels do not match dataset labels for {split}")
    if not np.array_equal(indices, np.arange(len(dataset), dtype=np.int64)):
        raise ValueError(f"Cached dataset indices are not sequential for {split}")
    nonfinite_count = int(np.size(features) - np.isfinite(features).sum())
    if nonfinite_count:
        raise FloatingPointError(f"DETR cache features contain {nonfinite_count} non-finite values")

    jet_files, jet_file_indices, jet_entries = _identity_arrays(dataset.jet_ids)
    arrays = {
        "tokens": features,
        "features": features,
        "mask": masks,
        "candidate_mask": masks,
        "confidence": confidences,
        "candidate_weights": confidences,
        "existence_scores": confidences,
        "existence_logits": existence_logits,
        "source_slot_indices": source_indices,
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
            "source_slot_indices": source_indices,
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
    metrics = accumulator.summary() if config.compute_detr_metrics else {}
    candidate_counts = masks.sum(axis=1).astype(np.float64)
    active_counts = (confidences * masks.astype(np.float32)).sum(axis=1).astype(np.float64)
    exported_counts = (source_indices >= 0).sum(axis=1).astype(np.float64)
    top_scores = confidences[:, 0] if confidences.shape[1] else np.zeros((confidences.shape[0],), dtype=np.float32)
    metadata = {
        "version": DETR_SLOT_CACHE_ARRAY_VERSION,
        "experiment_step": DETR_SLOT_CACHE_STEP,
        "output_contract": DETR_SLOT_RECONSTRUCTOR_CONTRACT,
        "architecture": architecture,
        "split": split,
        "n_jets": int(features.shape[0]),
        "n_candidates": int(features.shape[1]),
        "feature_dim": int(features.shape[2]),
        "array_path": str(cache_path),
        "metadata_path": str(metadata_path),
        "array_keys": sorted(arrays.keys()),
        "selection_mode": "topk_or_threshold",
        "export_max_tokens": int(export_max_tokens),
        "confidence_threshold": float(config.confidence_threshold),
        "min_tokens_per_view": int(config.min_tokens_per_view),
        "source_checkpoint": source_checkpoint,
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_checkpoint_epoch": checkpoint_payload.get("epoch"),
        "source_checkpoint_experiment_step": checkpoint_payload.get("experiment_step"),
        "source_checkpoint_metrics": checkpoint_payload.get("metrics"),
        "source_checkpoint_model_config": checkpoint_payload.get("model_config"),
        "source_checkpoint_loss_config": checkpoint_payload.get("loss_config"),
        "checkpoint_load_info": checkpoint_payload.get("checkpoint_load_info"),
        "loss_config_used_for_metrics": loss_config.to_dict(),
        "label_names": list(LABEL_NAMES),
        "jet_files": jet_files,
        "jet_identity_hash": jet_identity_hash(dataset.jet_ids),
        "cache_content_hash": content_hash,
        "alignment_audit": alignment_audit,
        "dataset_metadata": dict(dataset.metadata),
        "heldout_detr_slot_metrics": metrics,
        "num_batches": int(num_batches),
        "candidate_count_summary": _safe_summary(candidate_counts),
        "active_confidence_count_summary": _safe_summary(active_counts),
        "exported_tokens_summary": _safe_summary(exported_counts),
        "top_existence_score_mean": float(top_scores.mean()) if top_scores.size else 0.0,
        "nonfinite_count": int(nonfinite_count),
        "source": source_metadata(),
        "config": config.to_dict(),
        "leakage_rule": (
            "Step 12 freezes a model_val-selected DETR/free-slot reconstructor and applies it to "
            "stack_train, stack_val, and final_test using fixed HLT inputs. Offline constituents are "
            "loaded only to audit/evaluate set-matching cache quality; cached inference arrays are "
            "DETR slot outputs produced from HLT tokens."
        ),
        "offline_targets_not_saved": True,
        "inference_consumes_hlt_only": True,
    }
    save_json(metadata_path, _jsonable(metadata))
    return dict(metadata)


def cache_detr_slot_reco_views(
    config: DetrSlotRecoViewCacheConfig,
    *,
    model: DetrSlotReconstructor | None = None,
    checkpoint_payload: Mapping[str, Any] | None = None,
    datasets: Mapping[str, SetMatchingJetDataset] | None = None,
) -> dict[str, Any]:
    """Cache DETR/free-slot reconstructed views for all configured splits."""

    torch = require_torch()
    device = resolve_device(config.device)
    if model is None:
        model, loaded_payload = load_detr_slot_reconstructor_checkpoint(
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

    architecture = normalize_detr_slot_encoder_architecture(
        config.architecture or getattr(model, "architecture", checkpoint_payload.get("architecture", "gt"))
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
        split_reports[split] = cache_detr_slot_reco_split(
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

    report_path = Path(config.output_dir) / architecture / "cache_report.json"
    diagnostics_path = Path(config.output_dir) / architecture / "cache_summary.csv"
    report = {
        "experiment_step": DETR_SLOT_CACHE_STEP,
        "architecture": architecture,
        "splits": list(config.splits),
        "output_dir": str(Path(config.output_dir)),
        "reconstructor_checkpoint": str(Path(config.reconstructor_checkpoint)),
        "manifest_hash": manifest_sha,
        "cache_paths": {split: str(config.cache_path(architecture, split)) for split in config.splits},
        "metadata_paths": {split: str(config.metadata_path(architecture, split)) for split in config.splits},
        "summary_csv": str(diagnostics_path),
        "split_reports": split_reports,
        "source": source_metadata(),
    }
    save_json(report_path, _jsonable(report))
    _write_cache_summary_csv(diagnostics_path, split_reports)
    report["report_path"] = str(report_path)
    torch.cuda.empty_cache() if getattr(device, "type", None) == "cuda" else None
    return _jsonable(report)


__all__ = [
    "DEFAULT_DETR_SLOT_CACHE_SPLITS",
    "DETR_SLOT_CACHE_ARRAY_VERSION",
    "DETR_SLOT_CACHE_STEP",
    "DetrSlotRecoViewCacheConfig",
    "cache_detr_slot_reco_split",
    "cache_detr_slot_reco_views",
    "load_detr_slot_reconstructor_checkpoint",
    "_select_top_detr_slots",
]
