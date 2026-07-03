"""PD10-V2 teacher representation cache utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import save_json
from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash
from jetclass_fresh.jetclass_data import JetIdentity, LABEL_NAMES

from .config import (
    PD10_EXPERIMENT_NAME,
    PD10_EXTENDED_REAL_TEACHERS,
    PD10_EXTENDED_TEACHER_ALLOWED_INPUTS,
    PD10_NUM_CLASSES,
    PD10_REPRESENTATION_DIM,
    PD10_SPLIT_ORDER,
    PD10_SPLIT_SIZES,
    PD10_TEACHER_NONE,
    default_pd10_experiment_layout,
    normalize_pd10_extended_teacher_target,
    pd10_extended_teacher_model_name,
)


PD10_V2_STEP1_EXPERIMENT_STEP = "pd10_v2_step1_representation_cache_contract"
PD10_TEACHER_REPRESENTATION_CACHE_CONTRACT = "pd10_teacher_representation_cache_v1"
PD10_TEACHER_REPRESENTATION_CACHE_MANIFEST = "teacher_representation_manifest.json"
PD10_TEACHER_REPRESENTATION_CACHE_REPORT = "teacher_representation_cache_report.json"
PD10_TEACHER_REPRESENTATION_SPLITS: tuple[str, ...] = PD10_SPLIT_ORDER


@dataclass(frozen=True)
class PD10TeacherRepresentationCacheConfig:
    """Configuration for one train-time teacher representation cache."""

    teacher_target: str
    output_dir: str
    splits: tuple[str, ...] = field(default_factory=lambda: PD10_TEACHER_REPRESENTATION_SPLITS)
    representation_dim: int = PD10_REPRESENTATION_DIM
    overwrite: bool = False
    skip_existing: bool = True
    confirm_final_test: bool = False

    def __post_init__(self) -> None:
        target = normalize_pd10_representation_teacher_target(self.teacher_target)
        splits = tuple(str(split) for split in self.splits)
        if not splits:
            raise ValueError("at least one representation split is required")
        unknown = [split for split in splits if split not in PD10_TEACHER_REPRESENTATION_SPLITS]
        if unknown:
            raise ValueError(f"unknown PD10 representation splits: {unknown}")
        if "final_test" in splits and not bool(self.confirm_final_test):
            raise ValueError("Refusing to cache final_test teacher representations without confirm_final_test=True")
        if int(self.representation_dim) <= 0:
            raise ValueError("representation_dim must be positive")
        object.__setattr__(self, "teacher_target", target)
        object.__setattr__(self, "splits", splits)
        object.__setattr__(self, "representation_dim", int(self.representation_dim))

    @property
    def model_name(self) -> str:
        return pd10_extended_teacher_model_name(self.teacher_target)

    @property
    def teacher_output_dir(self) -> Path:
        return Path(self.output_dir) / self.model_name


@dataclass(frozen=True)
class PD10TeacherRepresentationBlock:
    """One teacher's hidden representation rows for one split."""

    teacher_target: str
    model_name: str
    split: str
    representations: np.ndarray
    labels: np.ndarray
    jet_ids: list[JetIdentity]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        target = normalize_pd10_representation_teacher_target(self.teacher_target)
        model_name = pd10_extended_teacher_model_name(target)
        if self.model_name != model_name:
            raise ValueError(f"model_name mismatch: {self.model_name!r} != {model_name!r}")
        if self.split not in PD10_TEACHER_REPRESENTATION_SPLITS:
            raise ValueError(f"unknown PD10 representation split {self.split!r}")
        representations = np.asarray(self.representations, dtype=np.float32)
        labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        if representations.ndim != 2:
            raise ValueError(f"teacher representations must be 2D [N, D], got {representations.shape}")
        if int(representations.shape[0]) != int(labels.shape[0]):
            raise ValueError("labels/representations row count mismatch")
        if len(self.jet_ids) != int(labels.shape[0]):
            raise ValueError("jet_ids length does not match labels/representations")
        identity_labels = np.asarray([int(identity.label) for identity in self.jet_ids], dtype=np.int64)
        if not np.array_equal(labels, identity_labels):
            raise ValueError("labels and jet ids are not aligned")
        if not np.isfinite(representations).all():
            raise FloatingPointError("teacher representations contain non-finite values")
        object.__setattr__(self, "teacher_target", target)
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "representations", representations)
        object.__setattr__(self, "labels", labels)


def normalize_pd10_representation_teacher_target(value: str) -> str:
    target = normalize_pd10_extended_teacher_target(value)
    if target == PD10_TEACHER_NONE:
        raise ValueError("representation caches require a real teacher target, not 'none'")
    if target not in PD10_EXTENDED_REAL_TEACHERS:
        raise ValueError(f"teacher target {target!r} is not a representation-capable PD10 teacher")
    return target


def pd10_teacher_representation_cache_dir(
    teacher_target: str,
    *,
    output_root: str | Path = "checkpoints",
) -> Path:
    layout = default_pd10_experiment_layout(output_root=output_root)
    return layout.root / "teacher_representations" / pd10_extended_teacher_model_name(teacher_target)


def pd10_teacher_representation_paths(
    output_dir: str | Path,
    teacher_target: str,
    split: str,
) -> tuple[Path, Path]:
    model_name = pd10_extended_teacher_model_name(teacher_target)
    root = Path(output_dir) / model_name
    return root / f"{split}_representations.npz", root / f"{split}_representations_metadata.json"


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


def _representation_arrays(block: PD10TeacherRepresentationBlock) -> tuple[dict[str, np.ndarray], list[str]]:
    jet_files, file_indices, entries = _identity_arrays(block.jet_ids)
    arrays = {
        "representations": block.representations.astype(np.float32, copy=False),
        "labels": block.labels.astype(np.int64, copy=False),
        "jet_file_indices": file_indices.astype(np.int32, copy=False),
        "jet_entries": entries.astype(np.int64, copy=False),
    }
    return arrays, jet_files


def build_pd10_teacher_representation_block(
    config: PD10TeacherRepresentationCacheConfig,
    split: str,
    *,
    representations: np.ndarray,
    labels: np.ndarray,
    jet_ids: Sequence[JetIdentity],
    source_metadata: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> PD10TeacherRepresentationBlock:
    target = normalize_pd10_representation_teacher_target(config.teacher_target)
    split = str(split)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    representations = np.asarray(representations, dtype=np.float32)
    if representations.ndim != 2 or int(representations.shape[1]) != int(config.representation_dim):
        raise ValueError(
            f"teacher representations must have shape [N, {int(config.representation_dim)}], "
            f"got {representations.shape}"
        )
    metadata = {
        "contract": PD10_TEACHER_REPRESENTATION_CACHE_CONTRACT,
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_V2_STEP1_EXPERIMENT_STEP,
        "teacher_target": target,
        "model_name": config.model_name,
        "model_kind": "pd10_teacher_representations",
        "split": split,
        "split_expected_size": int(PD10_SPLIT_SIZES[split]),
        "representation_dim": int(config.representation_dim),
        "label_names": list(LABEL_NAMES),
        "num_classes": PD10_NUM_CLASSES,
        "allowed_inputs": PD10_EXTENDED_TEACHER_ALLOWED_INPUTS[target],
        "student_deployment_inputs": "HLT_only",
        "teacher_representations_train_time_only": True,
        "returns_offline_particles": False,
        "inference_export_requires_teacher_features": False,
        **dict(source_metadata or {}),
        **dict(extra_metadata or {}),
    }
    return PD10TeacherRepresentationBlock(
        teacher_target=target,
        model_name=config.model_name,
        split=split,
        representations=representations,
        labels=labels,
        jet_ids=list(jet_ids),
        metadata=metadata,
    )


def save_pd10_teacher_representation_block(
    block: PD10TeacherRepresentationBlock,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    npz_path, meta_path = pd10_teacher_representation_paths(output_dir, block.teacher_target, block.split)
    if not overwrite and (npz_path.exists() or meta_path.exists()):
        raise FileExistsError(f"Teacher representation block already exists: {npz_path}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays, jet_files = _representation_arrays(block)
    np.savez_compressed(npz_path, **arrays)
    metadata = {
        **dict(block.metadata),
        "model_name": block.model_name,
        "teacher_target": block.teacher_target,
        "split": block.split,
        "npz_path": str(npz_path),
        "metadata_path": str(meta_path),
        "jet_files": jet_files,
        "jet_identity_hash": jet_identity_hash(block.jet_ids),
        "representation_content_hash": hash_arrays(arrays),
        "n_jets": int(block.labels.shape[0]),
        "representation_dim": int(block.representations.shape[1]),
        "num_classes": PD10_NUM_CLASSES,
    }
    validate_pd10_teacher_representation_metadata(
        metadata,
        teacher_target=block.teacher_target,
        split=block.split,
        representation_dim=int(block.representations.shape[1]),
    )
    save_json(meta_path, metadata)
    return metadata


def validate_pd10_teacher_representation_metadata(
    metadata: Mapping[str, Any],
    *,
    teacher_target: str,
    split: str | None = None,
    representation_dim: int | None = None,
) -> None:
    target = normalize_pd10_representation_teacher_target(teacher_target)
    model_name = pd10_extended_teacher_model_name(target)
    if metadata.get("contract") != PD10_TEACHER_REPRESENTATION_CACHE_CONTRACT:
        raise ValueError("PD10 teacher representation cache contract mismatch")
    if metadata.get("experiment_step") != PD10_V2_STEP1_EXPERIMENT_STEP:
        raise ValueError("PD10 teacher representation cache step mismatch")
    if metadata.get("teacher_target") != target:
        raise ValueError(f"teacher_target mismatch: {metadata.get('teacher_target')} != {target}")
    if metadata.get("model_name") != model_name:
        raise ValueError(f"model_name mismatch: {metadata.get('model_name')} != {model_name}")
    if split is not None and metadata.get("split") != split:
        raise ValueError(f"split mismatch: {metadata.get('split')} != {split}")
    if int(metadata.get("n_jets", -1)) <= 0:
        raise ValueError("PD10 teacher representation cache contains no rows")
    expected_dim = PD10_REPRESENTATION_DIM if representation_dim is None else int(representation_dim)
    if int(metadata.get("representation_dim", -1)) != expected_dim:
        raise ValueError(f"representation_dim mismatch: {metadata.get('representation_dim')} != {expected_dim}")
    if int(metadata.get("num_classes", -1)) != PD10_NUM_CLASSES:
        raise ValueError("PD10 teacher representation cache must declare 10 classes")
    if metadata.get("allowed_inputs") != PD10_EXTENDED_TEACHER_ALLOWED_INPUTS[target]:
        raise ValueError(
            f"allowed_inputs mismatch: {metadata.get('allowed_inputs')} "
            f"!= {PD10_EXTENDED_TEACHER_ALLOWED_INPUTS[target]}"
        )
    if metadata.get("student_deployment_inputs") != "HLT_only":
        raise ValueError("teacher representation cache must declare HLT-only student deployment")
    if not bool(metadata.get("teacher_representations_train_time_only")):
        raise ValueError("teacher representation cache must be train-time only")
    if bool(metadata.get("inference_export_requires_teacher_features")):
        raise ValueError("student inference/export must not require teacher representations")


def load_pd10_teacher_representation_block(
    output_dir: str | Path,
    teacher_target: str,
    split: str,
    *,
    verify_hash: bool = True,
) -> PD10TeacherRepresentationBlock:
    target = normalize_pd10_representation_teacher_target(teacher_target)
    npz_path, meta_path = pd10_teacher_representation_paths(output_dir, target, split)
    with meta_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with np.load(npz_path, allow_pickle=False) as data:
        representations = data["representations"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        file_indices = data["jet_file_indices"].astype(np.int64)
        entries = data["jet_entries"].astype(np.int64)
    if verify_hash:
        actual = hash_arrays(
            {
                "representations": representations,
                "labels": labels,
                "jet_file_indices": file_indices.astype(np.int32),
                "jet_entries": entries,
            }
        )
        if actual != metadata.get("representation_content_hash"):
            raise ValueError(f"Teacher representation hash mismatch for {target}/{split}: {actual}")
    jet_files = [str(path) for path in metadata["jet_files"]]
    jet_ids = [
        JetIdentity(file=jet_files[int(file_index)], entry=int(entry), label=int(label))
        for file_index, entry, label in zip(file_indices, entries, labels)
    ]
    block = PD10TeacherRepresentationBlock(
        teacher_target=target,
        model_name=pd10_extended_teacher_model_name(target),
        split=split,
        representations=representations,
        labels=labels,
        jet_ids=jet_ids,
        metadata=dict(metadata),
    )
    validate_pd10_teacher_representation_metadata(
        block.metadata,
        teacher_target=target,
        split=split,
        representation_dim=int(block.representations.shape[1]),
    )
    return block


def write_pd10_teacher_representation_manifest(
    config: PD10TeacherRepresentationCacheConfig,
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    manifest = {
        "ok": True,
        "contract": PD10_TEACHER_REPRESENTATION_CACHE_CONTRACT,
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_V2_STEP1_EXPERIMENT_STEP,
        "teacher_target": config.teacher_target,
        "model_name": config.model_name,
        "allowed_inputs": PD10_EXTENDED_TEACHER_ALLOWED_INPUTS[config.teacher_target],
        "output_dir": str(config.output_dir),
        "teacher_representation_dir": str(config.teacher_output_dir),
        "splits": list(config.splits),
        "representation_dim": int(config.representation_dim),
        "split_sizes": {split: int(PD10_SPLIT_SIZES[split]) for split in PD10_SPLIT_ORDER},
        "prediction_rows": [dict(row) for row in prediction_rows],
        "config": asdict(config),
    }
    save_json(config.teacher_output_dir / PD10_TEACHER_REPRESENTATION_CACHE_MANIFEST, manifest)
    save_json(config.teacher_output_dir / PD10_TEACHER_REPRESENTATION_CACHE_REPORT, manifest)
    return manifest


__all__ = [
    "PD10_TEACHER_REPRESENTATION_CACHE_CONTRACT",
    "PD10_TEACHER_REPRESENTATION_CACHE_MANIFEST",
    "PD10_TEACHER_REPRESENTATION_CACHE_REPORT",
    "PD10_TEACHER_REPRESENTATION_SPLITS",
    "PD10_V2_STEP1_EXPERIMENT_STEP",
    "PD10TeacherRepresentationBlock",
    "PD10TeacherRepresentationCacheConfig",
    "build_pd10_teacher_representation_block",
    "load_pd10_teacher_representation_block",
    "normalize_pd10_representation_teacher_target",
    "pd10_teacher_representation_cache_dir",
    "pd10_teacher_representation_paths",
    "save_pd10_teacher_representation_block",
    "validate_pd10_teacher_representation_metadata",
    "write_pd10_teacher_representation_manifest",
]

