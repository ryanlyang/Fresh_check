"""Datasets and collate helpers for local particle residual-field experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.hlt_cache import jet_identity_hash, load_cached_hlt_view
from jetclass_fresh.jetclass_data import JetIdentity, load_split_manifest, manifest_hash
from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens

from .targets import (
    LocalParticleResidualFieldCache,
    load_local_particle_residual_field_cache,
)


LOCAL_PARTICLE_RESIDUAL_FIELD_DATASET_CONTRACT = "local_particle_residual_field_dataset_v1"
LOCAL_PARTICLE_RESIDUAL_FIELD_HLT_ONLY_DATASET_CONTRACT = "local_particle_residual_field_hlt_only_dataset_v1"
LOCAL_PARTICLE_RESIDUAL_FIELD_ALLOWED_INPUTS = (
    "HLT_particles_at_inference__offline_local_residual_targets_train_time_only"
)


@dataclass(frozen=True)
class LocalParticleResidualFieldDatasetConfig:
    """Filesystem and split contract for one local residual-field dataset."""

    hlt_cache_dir: str
    target_cache_dir: str
    split: str = "model_train"
    manifest_path: str | None = None
    max_jets: int | None = None
    teacher_logits_path: str | None = None
    teacher_logits_metadata_path: str | None = None
    include_oracle_fields: bool = False
    allow_final_test_targets: bool = False
    verify_hash: bool = True
    require_manifest_match: bool = True
    require_teacher_logits_metadata: bool = False
    require_same_jet_identity: bool = True
    require_same_labels: bool = True
    require_same_mask: bool = True

    def __post_init__(self) -> None:
        if self.max_jets is not None and int(self.max_jets) <= 0:
            raise ValueError("max_jets must be positive when provided")
        object.__setattr__(self, "split", str(self.split))


@dataclass(frozen=True)
class TeacherLogitBlock:
    """Optional train-time teacher logits aligned to one split."""

    logits: np.ndarray
    labels: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _load_teacher_logits(
    path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> TeacherLogitBlock:
    npz_path = Path(path)
    with np.load(npz_path, allow_pickle=False) as data:
        if "logits" in data.files:
            logits = data["logits"].astype(np.float32, copy=False)
        elif "teacher_logits" in data.files:
            logits = data["teacher_logits"].astype(np.float32, copy=False)
        else:
            raise ValueError(f"teacher logits npz must contain logits or teacher_logits: {npz_path}")
        labels = data["labels"].astype(np.int64, copy=False) if "labels" in data.files else None
    meta_path = Path(metadata_path) if metadata_path else npz_path.with_name(npz_path.stem + "_metadata.json")
    metadata: Mapping[str, Any] = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return TeacherLogitBlock(logits=logits, labels=labels, metadata=metadata)


def _identity_key(identity: JetIdentity) -> tuple[str, int, int]:
    return str(identity.file), int(identity.entry), int(identity.label)


def _truncate_sequence(values: Sequence[Any], n: int) -> tuple[Any, ...]:
    return tuple(values[: int(n)])


def _validate_dataset_alignment(
    *,
    config: LocalParticleResidualFieldDatasetConfig,
    hlt_view: Any,
    target_cache: LocalParticleResidualFieldCache,
    teacher_logits: TeacherLogitBlock | None,
) -> dict[str, Any]:
    problems: list[str] = []
    split = str(config.split)
    hlt_ids = tuple(hlt_view.jet_ids)
    target_ids = tuple(target_cache.jet_ids)
    if len(hlt_ids) != len(target_ids):
        problems.append(f"HLT rows {len(hlt_ids)} != target rows {len(target_ids)}")
    elif bool(config.require_same_jet_identity) and hlt_ids != target_ids:
        first_bad = next(
            (
                index
                for index, (hlt_id, target_id) in enumerate(zip(hlt_ids, target_ids))
                if _identity_key(hlt_id) != _identity_key(target_id)
            ),
            None,
        )
        problems.append(f"HLT/target jet identities differ at row {first_bad}")

    if bool(config.require_same_labels) and not np.array_equal(hlt_view.labels, target_cache.labels):
        problems.append("HLT labels do not match target labels")
    if bool(config.require_same_mask) and not np.array_equal(hlt_view.mask, target_cache.target_mask):
        problems.append("HLT particle mask does not match target mask")

    hlt_hash = hlt_view.metadata.get("hlt_content_hash")
    target_hlt_hash = target_cache.metadata.get("hlt_content_hash")
    if hlt_hash and target_hlt_hash and hlt_hash != target_hlt_hash:
        problems.append(f"target cache hlt_content_hash {target_hlt_hash} != loaded HLT hash {hlt_hash}")
    hlt_identity_hash = hlt_view.metadata.get("jet_identity_hash") or jet_identity_hash(hlt_ids)
    target_identity_hash = target_cache.metadata.get("jet_identity_hash")
    if hlt_identity_hash and target_identity_hash and hlt_identity_hash != target_identity_hash:
        problems.append("target cache jet identity hash does not match loaded HLT cache")

    expected_manifest_hash = None
    if config.manifest_path:
        manifest = load_split_manifest(config.manifest_path)
        expected_manifest_hash = manifest_hash(manifest)
        if bool(config.require_manifest_match):
            expected_ids = tuple(manifest.splits[split])
            if hlt_ids != expected_ids:
                problems.append("HLT jet identities do not match split manifest")
            target_manifest_hash = target_cache.metadata.get("source_manifest_hash")
            if target_manifest_hash != expected_manifest_hash:
                problems.append(
                    f"target source_manifest_hash {target_manifest_hash} != manifest hash {expected_manifest_hash}"
                )
            hlt_manifest_hash = hlt_view.metadata.get("source_manifest_hash")
            if hlt_manifest_hash not in (None, expected_manifest_hash):
                problems.append(f"HLT source_manifest_hash {hlt_manifest_hash} != manifest hash {expected_manifest_hash}")

    if teacher_logits is not None:
        if teacher_logits.logits.ndim != 2:
            problems.append(f"teacher logits must have shape [N, C], got {teacher_logits.logits.shape}")
        elif int(teacher_logits.logits.shape[0]) != int(len(hlt_ids)):
            problems.append(
                f"teacher logits rows {teacher_logits.logits.shape[0]} != dataset rows {len(hlt_ids)}"
            )
        if teacher_logits.labels is None:
            if bool(config.require_teacher_logits_metadata):
                problems.append("teacher logits labels are required for supervised KD alignment")
        elif not np.array_equal(teacher_logits.labels, hlt_view.labels):
            problems.append("teacher logit labels do not match HLT labels")
        teacher_identity_hash = teacher_logits.metadata.get("jet_identity_hash")
        if bool(config.require_teacher_logits_metadata) and not teacher_identity_hash:
            problems.append("teacher logits jet_identity_hash metadata is required")
        if teacher_identity_hash and hlt_identity_hash and teacher_identity_hash != hlt_identity_hash:
            problems.append("teacher logits jet_identity_hash does not match HLT cache")
        teacher_split = teacher_logits.metadata.get("split")
        if bool(config.require_teacher_logits_metadata) and not teacher_split:
            problems.append("teacher logits split metadata is required")
        if teacher_split and str(teacher_split) != split:
            problems.append(f"teacher logits split {teacher_split} != {split}")

    if problems:
        raise ValueError("; ".join(problems))

    return {
        "ok": True,
        "problems": [],
        "split": split,
        "n_jets": int(len(hlt_ids)),
        "hlt_content_hash": hlt_hash,
        "target_content_hash": target_cache.metadata.get("target_content_hash"),
        "offline_content_hash": target_cache.metadata.get("offline_content_hash"),
        "jet_identity_hash": hlt_identity_hash,
        "source_manifest_hash": expected_manifest_hash or target_cache.metadata.get("source_manifest_hash"),
        "teacher_logits_present": bool(teacher_logits is not None),
        "target_field_dim": int(target_cache.target_fields.shape[-1]),
        "target_field_names": list(target_cache.field_names),
        "target_field_groups": {
            str(key): [int(index) for index in value]
            for key, value in target_cache.field_groups.items()
        },
    }


def _validate_hlt_only_alignment(
    *,
    config: LocalParticleResidualFieldDatasetConfig,
    hlt_view: Any,
    teacher_logits: TeacherLogitBlock | None,
) -> dict[str, Any]:
    problems: list[str] = []
    split = str(config.split)
    hlt_ids = tuple(hlt_view.jet_ids)
    hlt_identity_hash = hlt_view.metadata.get("jet_identity_hash") or jet_identity_hash(hlt_ids)
    expected_manifest_hash = None
    if config.manifest_path:
        manifest = load_split_manifest(config.manifest_path)
        expected_manifest_hash = manifest_hash(manifest)
        if bool(config.require_manifest_match):
            expected_ids = tuple(manifest.splits[split])
            if hlt_ids != expected_ids:
                problems.append("HLT jet identities do not match split manifest")
            hlt_manifest_hash = hlt_view.metadata.get("source_manifest_hash")
            if hlt_manifest_hash not in (None, expected_manifest_hash):
                problems.append(f"HLT source_manifest_hash {hlt_manifest_hash} != manifest hash {expected_manifest_hash}")
    if teacher_logits is not None:
        if teacher_logits.logits.ndim != 2:
            problems.append(f"teacher logits must have shape [N, C], got {teacher_logits.logits.shape}")
        elif int(teacher_logits.logits.shape[0]) != int(len(hlt_ids)):
            problems.append(f"teacher logits rows {teacher_logits.logits.shape[0]} != dataset rows {len(hlt_ids)}")
        if teacher_logits.labels is None:
            if bool(config.require_teacher_logits_metadata):
                problems.append("teacher logits labels are required for supervised KD alignment")
        elif not np.array_equal(teacher_logits.labels, hlt_view.labels):
            problems.append("teacher logit labels do not match HLT labels")
        teacher_identity_hash = teacher_logits.metadata.get("jet_identity_hash")
        if bool(config.require_teacher_logits_metadata) and not teacher_identity_hash:
            problems.append("teacher logits jet_identity_hash metadata is required")
        if teacher_identity_hash and hlt_identity_hash and teacher_identity_hash != hlt_identity_hash:
            problems.append("teacher logits jet_identity_hash does not match HLT cache")
        teacher_split = teacher_logits.metadata.get("split")
        if bool(config.require_teacher_logits_metadata) and not teacher_split:
            problems.append("teacher logits split metadata is required")
        if teacher_split and str(teacher_split) != split:
            problems.append(f"teacher logits split {teacher_split} != {split}")
    if problems:
        raise ValueError("; ".join(problems))
    return {
        "ok": True,
        "problems": [],
        "split": split,
        "n_jets": int(len(hlt_ids)),
        "hlt_content_hash": hlt_view.metadata.get("hlt_content_hash"),
        "jet_identity_hash": hlt_identity_hash,
        "source_manifest_hash": expected_manifest_hash or hlt_view.metadata.get("source_manifest_hash"),
        "teacher_logits_present": bool(teacher_logits is not None),
        "target_field_dim": 0,
        "target_field_names": [],
        "target_field_groups": {},
    }


class LocalParticleResidualFieldDataset:
    """Dataset over HLT particles and aligned local residual-field targets."""

    def __init__(
        self,
        hlt_view: Any,
        target_cache: LocalParticleResidualFieldCache,
        *,
        config: LocalParticleResidualFieldDatasetConfig | None = None,
        teacher_logits: TeacherLogitBlock | None = None,
    ) -> None:
        self.config = config or LocalParticleResidualFieldDatasetConfig(
            hlt_cache_dir="",
            target_cache_dir="",
            split=str(hlt_view.split),
        )
        self.alignment_report = _validate_dataset_alignment(
            config=self.config,
            hlt_view=hlt_view,
            target_cache=target_cache,
            teacher_logits=teacher_logits,
        )
        n_rows = int(len(hlt_view.labels))
        if self.config.max_jets is not None:
            n_rows = min(n_rows, int(self.config.max_jets))
        self.tokens = np.asarray(hlt_view.tokens[:n_rows], dtype=np.float32)
        self.mask = np.asarray(hlt_view.mask[:n_rows], dtype=bool)
        self.labels = np.asarray(hlt_view.labels[:n_rows], dtype=np.int64)
        self.jet_ids = _truncate_sequence(tuple(hlt_view.jet_ids), n_rows)
        self.target_fields = np.asarray(target_cache.target_fields[:n_rows], dtype=np.float32)
        self.target_mask = np.asarray(target_cache.target_mask[:n_rows], dtype=bool)
        self.target_fields = np.where(self.target_mask[:, :, None], self.target_fields, 0.0).astype(np.float32)
        self.teacher_logits = (
            np.asarray(teacher_logits.logits[:n_rows], dtype=np.float32)
            if teacher_logits is not None
            else None
        )
        self.teacher_logits_metadata = dict(teacher_logits.metadata) if teacher_logits is not None else None
        self.field_names = tuple(target_cache.field_names)
        self.field_groups = {
            str(key): [int(index) for index in value]
            for key, value in target_cache.field_groups.items()
        }
        self.radii = tuple(float(radius) for radius in target_cache.radii)
        self.metadata = {
            "contract": LOCAL_PARTICLE_RESIDUAL_FIELD_DATASET_CONTRACT,
            "allowed_inputs": LOCAL_PARTICLE_RESIDUAL_FIELD_ALLOWED_INPUTS,
            "split": str(self.config.split),
            "n_jets": int(n_rows),
            "max_particles": int(self.tokens.shape[1]),
            "raw_token_dim": int(self.tokens.shape[2]),
            "target_field_dim": int(self.target_fields.shape[2]),
            "field_names": list(self.field_names),
            "field_groups": self.field_groups,
            "radii": list(self.radii),
            "include_oracle_fields": bool(self.config.include_oracle_fields),
            "target_mask_matches_hlt_mask": bool(np.array_equal(self.mask, self.target_mask)),
            "teacher_logits_present": bool(self.teacher_logits is not None),
            "alignment_report": dict(self.alignment_report),
            "hlt_metadata": {
                "view": hlt_view.metadata.get("view"),
                "hlt_content_hash": hlt_view.metadata.get("hlt_content_hash"),
                "jet_identity_hash": hlt_view.metadata.get("jet_identity_hash"),
                "source_manifest_hash": hlt_view.metadata.get("source_manifest_hash"),
                "hlt_profile": hlt_view.metadata.get("hlt_profile"),
                "hlt_profile_version": hlt_view.metadata.get("hlt_profile_version"),
                "hlt_degradation_strength": hlt_view.metadata.get("hlt_degradation_strength"),
            },
            "target_metadata": {
                "target_content_hash": target_cache.metadata.get("target_content_hash"),
                "offline_content_hash": target_cache.metadata.get("offline_content_hash"),
                "zero_baseline_metrics": target_cache.metadata.get("zero_baseline_metrics"),
            },
        }

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {
            "tokens": self.tokens[index],
            "mask": self.mask[index],
            "labels": np.int64(self.labels[index]),
            "indices": np.int64(index),
            "target_fields": self.target_fields[index],
            "target_mask": self.target_mask[index],
        }
        if bool(self.config.include_oracle_fields):
            item["oracle_fields"] = self.target_fields[index]
            item["oracle_mask"] = self.target_mask[index]
        if self.teacher_logits is not None:
            item["teacher_logits"] = self.teacher_logits[index]
        return item

    def label_counts(self) -> dict[int, int]:
        values, counts = np.unique(self.labels, return_counts=True)
        return {int(value): int(count) for value, count in zip(values, counts)}


class LocalParticleResidualFieldHLTOnlyDataset:
    """Dataset over HLT particles only, used for clean ParT baselines."""

    field_names: tuple[str, ...] = ()
    field_groups: dict[str, list[int]] = {}
    radii: tuple[float, ...] = ()

    def __init__(
        self,
        hlt_view: Any,
        *,
        config: LocalParticleResidualFieldDatasetConfig | None = None,
        teacher_logits: TeacherLogitBlock | None = None,
    ) -> None:
        self.config = config or LocalParticleResidualFieldDatasetConfig(
            hlt_cache_dir="",
            target_cache_dir="",
            split=str(hlt_view.split),
        )
        self.alignment_report = _validate_hlt_only_alignment(
            config=self.config,
            hlt_view=hlt_view,
            teacher_logits=teacher_logits,
        )
        n_rows = int(len(hlt_view.labels))
        if self.config.max_jets is not None:
            n_rows = min(n_rows, int(self.config.max_jets))
        self.tokens = np.asarray(hlt_view.tokens[:n_rows], dtype=np.float32)
        self.mask = np.asarray(hlt_view.mask[:n_rows], dtype=bool)
        self.labels = np.asarray(hlt_view.labels[:n_rows], dtype=np.int64)
        self.jet_ids = _truncate_sequence(tuple(hlt_view.jet_ids), n_rows)
        self.teacher_logits = (
            np.asarray(teacher_logits.logits[:n_rows], dtype=np.float32)
            if teacher_logits is not None
            else None
        )
        self.teacher_logits_metadata = dict(teacher_logits.metadata) if teacher_logits is not None else None
        self.metadata = {
            "contract": LOCAL_PARTICLE_RESIDUAL_FIELD_HLT_ONLY_DATASET_CONTRACT,
            "allowed_inputs": "HLT_particles_only_clean_baseline",
            "split": str(self.config.split),
            "n_jets": int(n_rows),
            "max_particles": int(self.tokens.shape[1]),
            "raw_token_dim": int(self.tokens.shape[2]),
            "target_field_dim": 0,
            "field_names": [],
            "field_groups": {},
            "radii": [],
            "include_oracle_fields": False,
            "target_fields_present": False,
            "teacher_logits_present": bool(self.teacher_logits is not None),
            "alignment_report": dict(self.alignment_report),
            "hlt_metadata": {
                "view": hlt_view.metadata.get("view"),
                "hlt_content_hash": hlt_view.metadata.get("hlt_content_hash"),
                "jet_identity_hash": hlt_view.metadata.get("jet_identity_hash"),
                "source_manifest_hash": hlt_view.metadata.get("source_manifest_hash"),
                "hlt_profile": hlt_view.metadata.get("hlt_profile"),
                "hlt_profile_version": hlt_view.metadata.get("hlt_profile_version"),
                "hlt_degradation_strength": hlt_view.metadata.get("hlt_degradation_strength"),
            },
            "target_metadata": {},
        }

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {
            "tokens": self.tokens[index],
            "mask": self.mask[index],
            "labels": np.int64(self.labels[index]),
            "indices": np.int64(index),
        }
        if self.teacher_logits is not None:
            item["teacher_logits"] = self.teacher_logits[index]
        return item

    def label_counts(self) -> dict[int, int]:
        values, counts = np.unique(self.labels, return_counts=True)
        return {int(value): int(count) for value, count in zip(values, counts)}


def load_local_particle_residual_field_hlt_only_dataset(
    config: LocalParticleResidualFieldDatasetConfig,
) -> LocalParticleResidualFieldHLTOnlyDataset:
    hlt_view = load_cached_hlt_view(config.hlt_cache_dir, config.split, verify_hash=bool(config.verify_hash))
    teacher_logits = (
        _load_teacher_logits(
            config.teacher_logits_path,
            metadata_path=config.teacher_logits_metadata_path,
        )
        if config.teacher_logits_path
        else None
    )
    return LocalParticleResidualFieldHLTOnlyDataset(
        hlt_view,
        config=config,
        teacher_logits=teacher_logits,
    )


def load_local_particle_residual_field_dataset(
    config: LocalParticleResidualFieldDatasetConfig,
) -> LocalParticleResidualFieldDataset:
    hlt_view = load_cached_hlt_view(config.hlt_cache_dir, config.split, verify_hash=bool(config.verify_hash))
    target_cache = load_local_particle_residual_field_cache(
        config.target_cache_dir,
        config.split,
        verify_hash=bool(config.verify_hash),
        allow_final_test_targets=bool(config.allow_final_test_targets),
    )
    teacher_logits = (
        _load_teacher_logits(
            config.teacher_logits_path,
            metadata_path=config.teacher_logits_metadata_path,
        )
        if config.teacher_logits_path
        else None
    )
    return LocalParticleResidualFieldDataset(
        hlt_view,
        target_cache,
        config=config,
        teacher_logits=teacher_logits,
    )


def collate_local_particle_residual_field_batch(
    samples: Sequence[Mapping[str, Any]],
    *,
    source_view: str = "fixed_hlt",
) -> dict[str, Any]:
    """Collate local residual-field samples into ParT-ready tensors."""

    torch = require_torch()
    tokens = np.stack([np.asarray(sample["tokens"], dtype=np.float32) for sample in samples], axis=0)
    raw_mask = np.stack([np.asarray(sample["mask"], dtype=bool) for sample in samples], axis=0)
    labels = np.asarray([sample["labels"] for sample in samples], dtype=np.int64)
    part_inputs = build_particle_transformer_inputs_from_tokens(
        tokens,
        raw_mask,
        labels=labels,
        source_view=source_view,
    )
    batch = {
        "tokens": torch.from_numpy(tokens).float(),
        "raw_mask": torch.from_numpy(raw_mask).bool(),
        "points": torch.from_numpy(part_inputs.pf_points).float(),
        "features": torch.from_numpy(part_inputs.pf_features).float(),
        "lorentz_vectors": torch.from_numpy(part_inputs.pf_vectors).float(),
        "mask": torch.from_numpy(part_inputs.pf_mask).bool(),
        "labels": torch.from_numpy(labels).long(),
        "indices": torch.from_numpy(np.asarray([sample["indices"] for sample in samples], dtype=np.int64)).long(),
    }
    if all("target_fields" in sample for sample in samples):
        target_fields = np.stack([np.asarray(sample["target_fields"], dtype=np.float32) for sample in samples], axis=0)
        target_mask = np.stack([np.asarray(sample["target_mask"], dtype=bool) for sample in samples], axis=0)
        if tokens.shape[:2] != target_fields.shape[:2] or tokens.shape[:2] != target_mask.shape:
            raise ValueError("target fields/mask must match HLT token leading shape")
        strict_target_mask = raw_mask & target_mask
        target_fields = np.where(strict_target_mask[:, :, None], target_fields, 0.0).astype(np.float32)
        batch["target_fields"] = torch.from_numpy(target_fields).float()
        batch["target_features"] = torch.from_numpy(np.transpose(target_fields, (0, 2, 1))).float()
        batch["target_mask"] = torch.from_numpy(strict_target_mask).bool()
        batch["target_weights"] = torch.from_numpy(strict_target_mask.astype(np.float32)).float()
    if all("oracle_fields" in sample for sample in samples):
        oracle_fields = np.stack([np.asarray(sample["oracle_fields"], dtype=np.float32) for sample in samples], axis=0)
        oracle_mask = np.stack([np.asarray(sample.get("oracle_mask", sample["target_mask"]), dtype=bool) for sample in samples], axis=0)
        oracle_mask = oracle_mask & raw_mask
        oracle_fields = np.where(oracle_mask[:, :, None], oracle_fields, 0.0).astype(np.float32)
        batch["oracle_fields"] = torch.from_numpy(oracle_fields).float()
        batch["oracle_features"] = torch.from_numpy(np.transpose(oracle_fields, (0, 2, 1))).float()
        batch["oracle_mask"] = torch.from_numpy(oracle_mask).bool()
    if all("teacher_logits" in sample for sample in samples):
        teacher_logits = np.stack([np.asarray(sample["teacher_logits"], dtype=np.float32) for sample in samples], axis=0)
        batch["teacher_logits"] = torch.from_numpy(teacher_logits).float()
    return batch


def make_local_particle_residual_field_collate(*, source_view: str = "fixed_hlt"):
    from functools import partial

    return partial(collate_local_particle_residual_field_batch, source_view=source_view)


def make_local_particle_residual_field_loader(
    dataset: LocalParticleResidualFieldDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    source_view: str = "fixed_hlt",
):
    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=make_local_particle_residual_field_collate(source_view=source_view),
        generator=generator,
    )


def move_local_particle_residual_field_batch_to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {
        str(key): value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


__all__ = [
    "LOCAL_PARTICLE_RESIDUAL_FIELD_DATASET_CONTRACT",
    "LOCAL_PARTICLE_RESIDUAL_FIELD_HLT_ONLY_DATASET_CONTRACT",
    "LOCAL_PARTICLE_RESIDUAL_FIELD_ALLOWED_INPUTS",
    "LocalParticleResidualFieldDatasetConfig",
    "TeacherLogitBlock",
    "LocalParticleResidualFieldDataset",
    "LocalParticleResidualFieldHLTOnlyDataset",
    "load_local_particle_residual_field_dataset",
    "load_local_particle_residual_field_hlt_only_dataset",
    "collate_local_particle_residual_field_batch",
    "make_local_particle_residual_field_collate",
    "make_local_particle_residual_field_loader",
    "move_local_particle_residual_field_batch_to_device",
]
