"""PD10 Step 6 HLT-only student distillation datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.fusion import PredictionBlock
from jetclass_fresh.hlt_baseline import build_particle_transformer_inputs_from_tokens, require_torch
from jetclass_fresh.hlt_cache import jet_identity_hash, load_cached_hlt_view
from jetclass_fresh.jetclass_data import JetIdentity, JetView

from .config import (
    PD10_EXPERIMENT_NAME,
    PD10_NUM_CLASSES,
    PD10_TEACHER_DUAL_VIEW,
    PD10_TEACHER_HLT,
    PD10_TEACHER_NONE,
    PD10_TEACHER_OFFLINE,
    normalize_pd10_teacher_target,
)
from .dual_view_teacher import load_pd10_dual_view_logit_block
from .logits import load_pd10_teacher_logit_block


PD10_STEP6_EXPERIMENT_STEP = "pd10_step6_student_distillation_dataset"
PD10_STUDENT_DATASET_CONTRACT = "pd10_hlt_only_student_distillation_dataset_v1"
PD10_STUDENT_ALLOWED_INPUTS = "HLT_only"


@dataclass(frozen=True)
class PD10StudentBatchMetadata:
    """Small serializable summary for auditing student dataset construction."""

    contract: str = PD10_STUDENT_DATASET_CONTRACT
    experiment_name: str = PD10_EXPERIMENT_NAME
    experiment_step: str = PD10_STEP6_EXPERIMENT_STEP
    split: str = ""
    teacher_target: str = PD10_TEACHER_NONE
    n_jets: int = 0
    hlt_content_hash: str | None = None
    hlt_jet_identity_hash: str | None = None
    teacher_prediction_content_hash: str | None = None
    teacher_jet_identity_hash: str | None = None
    student_allowed_inputs: str = PD10_STUDENT_ALLOWED_INPUTS
    returns_teacher_logits: bool = False
    returns_offline_particles: bool = False
    inference_export_requires_teacher_logits: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "experiment_name": self.experiment_name,
            "experiment_step": self.experiment_step,
            "split": self.split,
            "teacher_target": self.teacher_target,
            "n_jets": int(self.n_jets),
            "hlt_content_hash": self.hlt_content_hash,
            "hlt_jet_identity_hash": self.hlt_jet_identity_hash,
            "teacher_prediction_content_hash": self.teacher_prediction_content_hash,
            "teacher_jet_identity_hash": self.teacher_jet_identity_hash,
            "student_allowed_inputs": self.student_allowed_inputs,
            "returns_teacher_logits": bool(self.returns_teacher_logits),
            "returns_offline_particles": bool(self.returns_offline_particles),
            "inference_export_requires_teacher_logits": bool(self.inference_export_requires_teacher_logits),
            "notes": list(self.notes),
        }


def _identity_key(identity: JetIdentity) -> tuple[str, int, int]:
    return str(identity.file), int(identity.entry), int(identity.label)


def _check_fixed_hlt_view(view: JetView) -> None:
    source_view = view.metadata.get("view")
    if source_view not in (None, "fixed_hlt"):
        raise ValueError(f"PD10 student dataset requires fixed_hlt view, got {source_view!r}")


def _check_teacher_logits(block: PredictionBlock, teacher_target: str, split: str) -> None:
    target = normalize_pd10_teacher_target(teacher_target)
    if target == PD10_TEACHER_NONE:
        raise ValueError("CE-only student datasets must not provide teacher logits")
    if block.split != split:
        raise ValueError(f"teacher split mismatch: {block.split} != {split}")
    if block.metadata.get("teacher_target") != target:
        raise ValueError(f"teacher target mismatch: {block.metadata.get('teacher_target')} != {target}")
    if block.logits.ndim != 2 or int(block.logits.shape[1]) != PD10_NUM_CLASSES:
        raise ValueError(f"teacher logits must have shape [N, {PD10_NUM_CLASSES}], got {block.logits.shape}")
    if int(block.labels.shape[0]) != int(block.logits.shape[0]):
        raise ValueError("teacher labels/logits row count mismatch")
    if len(block.jet_ids) != int(block.logits.shape[0]):
        raise ValueError("teacher jet_ids/logits row count mismatch")
    if not np.isfinite(block.logits).all():
        raise FloatingPointError("teacher logits contain non-finite values")
    if not bool(block.metadata.get("teacher_logits_train_time_only")):
        raise ValueError("teacher block must declare teacher_logits_train_time_only=True")
    if block.metadata.get("student_deployment_inputs") not in (None, PD10_STUDENT_ALLOWED_INPUTS):
        raise ValueError("teacher block is not compatible with HLT-only student deployment")


def _view_index_by_identity(view: JetView) -> dict[tuple[str, int, int], int]:
    result: dict[tuple[str, int, int], int] = {}
    for index, identity in enumerate(view.jet_ids):
        key = _identity_key(identity)
        if key in result:
            raise ValueError(f"duplicate HLT jet identity in {view.split}: {identity}")
        result[key] = int(index)
    return result


def align_hlt_view_to_teacher_block(
    view: JetView,
    teacher_block: PredictionBlock,
    *,
    teacher_target: str,
) -> JetView:
    """Return an HLT-only view ordered exactly like the teacher-logit block."""

    _check_fixed_hlt_view(view)
    _check_teacher_logits(teacher_block, teacher_target, view.split)
    index_by_id = _view_index_by_identity(view)
    indices: list[int] = []
    for row, identity in enumerate(teacher_block.jet_ids):
        key = _identity_key(identity)
        if key not in index_by_id:
            raise ValueError(f"teacher jet identity row {row} is missing from HLT view: {identity}")
        view_index = index_by_id[key]
        if int(view.labels[view_index]) != int(teacher_block.labels[row]):
            raise ValueError(f"teacher/HLT label mismatch at teacher row {row}")
        indices.append(view_index)
    aligned = JetView(
        tokens=view.tokens[indices],
        mask=view.mask[indices],
        labels=view.labels[indices],
        jet_ids=[view.jet_ids[index] for index in indices],
        split=view.split,
        metadata={
            **dict(view.metadata),
            "aligned_to_teacher_target": normalize_pd10_teacher_target(teacher_target),
            "aligned_teacher_rows": int(len(indices)),
            "student_allowed_inputs": PD10_STUDENT_ALLOWED_INPUTS,
            "returns_offline_particles": False,
        },
    )
    if not np.array_equal(aligned.labels, teacher_block.labels):
        raise ValueError("aligned HLT labels do not match teacher labels")
    if aligned.jet_ids != teacher_block.jet_ids:
        raise ValueError("aligned HLT jet identities do not match teacher jet identities")
    return aligned


def limit_hlt_view_without_teacher(view: JetView, max_jets: int | None) -> JetView:
    _check_fixed_hlt_view(view)
    if max_jets is None or int(max_jets) >= int(len(view.labels)):
        return view
    limit = int(max_jets)
    return JetView(
        tokens=view.tokens[:limit],
        mask=view.mask[:limit],
        labels=view.labels[:limit],
        jet_ids=view.jet_ids[:limit],
        split=view.split,
        metadata={
            **dict(view.metadata),
            "student_allowed_inputs": PD10_STUDENT_ALLOWED_INPUTS,
            "returns_offline_particles": False,
            "subset_selection": {"strategy": "first_n_for_teacher_free_inference", "selected_n_jets": limit},
        },
    )


class PD10StudentDistillationDataset:
    """HLT-only student dataset with optional train-time teacher logits."""

    def __init__(
        self,
        hlt_view: JetView,
        *,
        teacher_target: str = PD10_TEACHER_NONE,
        teacher_block: PredictionBlock | None = None,
        max_jets: int | None = None,
    ) -> None:
        require_torch()
        target = normalize_pd10_teacher_target(teacher_target)
        _check_fixed_hlt_view(hlt_view)
        if target == PD10_TEACHER_NONE:
            if teacher_block is not None:
                raise ValueError("teacher_target='none' cannot be paired with teacher_block")
            aligned_view = limit_hlt_view_without_teacher(hlt_view, max_jets)
            teacher_logits = None
            teacher_metadata: dict[str, Any] = {}
        else:
            if teacher_block is None:
                raise FileNotFoundError(f"teacher logits are required for teacher_target={target!r}")
            aligned_view = align_hlt_view_to_teacher_block(hlt_view, teacher_block, teacher_target=target)
            if max_jets is not None and int(max_jets) < int(len(aligned_view.labels)):
                limit = int(max_jets)
                aligned_view = JetView(
                    tokens=aligned_view.tokens[:limit],
                    mask=aligned_view.mask[:limit],
                    labels=aligned_view.labels[:limit],
                    jet_ids=aligned_view.jet_ids[:limit],
                    split=aligned_view.split,
                    metadata={**dict(aligned_view.metadata), "student_max_jets": limit},
                )
                teacher_logits = np.asarray(teacher_block.logits[:limit], dtype=np.float32)
            else:
                teacher_logits = np.asarray(teacher_block.logits, dtype=np.float32)
            teacher_metadata = dict(teacher_block.metadata)
            if int(teacher_logits.shape[0]) != int(len(aligned_view.labels)):
                raise ValueError("teacher logits row count does not match aligned HLT rows")

        self.tokens = np.asarray(aligned_view.tokens, dtype=np.float32)
        self.hlt_mask = np.asarray(aligned_view.mask, dtype=bool)
        self.labels = np.asarray(aligned_view.labels, dtype=np.int64)
        self.jet_ids = list(aligned_view.jet_ids)
        self.split = aligned_view.split
        self.teacher_target = target
        self.teacher_logits = teacher_logits
        self.hlt_metadata = dict(aligned_view.metadata)
        self.teacher_metadata = teacher_metadata
        self.metadata = PD10StudentBatchMetadata(
            split=self.split,
            teacher_target=self.teacher_target,
            n_jets=int(len(self.labels)),
            hlt_content_hash=self.hlt_metadata.get("hlt_content_hash"),
            hlt_jet_identity_hash=jet_identity_hash(self.jet_ids),
            teacher_prediction_content_hash=teacher_metadata.get("prediction_content_hash"),
            teacher_jet_identity_hash=teacher_metadata.get("jet_identity_hash"),
            returns_teacher_logits=self.has_teacher_logits,
            notes=(
                ("Teacher logits are train-time supervision only.",)
                if self.has_teacher_logits
                else ("Teacher-free HLT-only dataset for CE, validation, or export.",)
            ),
        )

    @property
    def has_teacher_logits(self) -> bool:
        return self.teacher_logits is not None

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "hlt_tokens": self.tokens[index],
            "hlt_mask": self.hlt_mask[index],
            "label": np.int64(self.labels[index]),
            "jet_id": self.jet_ids[index],
        }
        if self.teacher_logits is not None:
            sample["teacher_logits"] = self.teacher_logits[index]
        return sample

    def to_metadata(self) -> dict[str, Any]:
        return self.metadata.to_dict()


def collate_pd10_student_batch(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Build one HLT-only student batch, optionally with teacher logits."""

    if not samples:
        raise ValueError("cannot collate an empty PD10 student batch")
    torch = require_torch()
    has_teacher = ["teacher_logits" in sample for sample in samples]
    if any(has_teacher) and not all(has_teacher):
        raise ValueError("PD10 student batch mixes samples with and without teacher logits")
    tokens = np.stack([np.asarray(sample["hlt_tokens"], dtype=np.float32) for sample in samples], axis=0)
    hlt_mask = np.stack([np.asarray(sample["hlt_mask"], dtype=bool) for sample in samples], axis=0)
    labels = np.asarray([sample["label"] for sample in samples], dtype=np.int64)
    part_inputs = build_particle_transformer_inputs_from_tokens(
        tokens,
        hlt_mask,
        labels=labels,
        source_view="fixed_hlt",
    )
    jet_ids = [sample["jet_id"] for sample in samples]
    batch: dict[str, Any] = {
        "points": torch.from_numpy(part_inputs.pf_points).float(),
        "features": torch.from_numpy(part_inputs.pf_features).float(),
        "lorentz_vectors": torch.from_numpy(part_inputs.pf_vectors).float(),
        "mask": torch.from_numpy(part_inputs.pf_mask).bool(),
        "labels": torch.from_numpy(labels).long(),
        "hlt_tokens": torch.from_numpy(tokens).float(),
        "hlt_constituent_mask": torch.from_numpy(hlt_mask).bool(),
        "jet_ids": jet_ids,
        "jet_files": [identity.file for identity in jet_ids],
        "jet_entries": torch.tensor([int(identity.entry) for identity in jet_ids], dtype=torch.long),
        "jet_identity_labels": torch.tensor([int(identity.label) for identity in jet_ids], dtype=torch.long),
        "jet_keys": [identity.key() for identity in jet_ids],
        "student_allowed_inputs": PD10_STUDENT_ALLOWED_INPUTS,
        "returns_offline_particles": False,
        "has_teacher_logits": bool(all(has_teacher)),
    }
    if all(has_teacher):
        teacher_logits = np.stack([np.asarray(sample["teacher_logits"], dtype=np.float32) for sample in samples], axis=0)
        if teacher_logits.ndim != 2 or int(teacher_logits.shape[1]) != PD10_NUM_CLASSES:
            raise ValueError(f"teacher_logits must have shape [N, {PD10_NUM_CLASSES}], got {teacher_logits.shape}")
        batch["teacher_logits"] = torch.from_numpy(teacher_logits).float()
    return batch


def make_pd10_student_data_loader(
    dataset: PD10StudentDistillationDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 12345,
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
        collate_fn=collate_pd10_student_batch,
        generator=generator,
    )


def move_pd10_student_batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    """Move tensor values to device while preserving jet-id lists/metadata."""

    torch = require_torch()
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def load_pd10_student_teacher_block(
    teacher_logit_dir: str | Path,
    teacher_target: str,
    split: str,
    *,
    verify_hash: bool = True,
) -> PredictionBlock | None:
    target = normalize_pd10_teacher_target(teacher_target)
    if target == PD10_TEACHER_NONE:
        return None
    if teacher_logit_dir is None:
        raise FileNotFoundError(f"teacher_logit_dir is required for teacher_target={target!r}")
    if target in (PD10_TEACHER_HLT, PD10_TEACHER_OFFLINE):
        return load_pd10_teacher_logit_block(teacher_logit_dir, target, split, verify_hash=verify_hash)
    if target == PD10_TEACHER_DUAL_VIEW:
        return load_pd10_dual_view_logit_block(teacher_logit_dir, split, verify_hash=verify_hash)
    raise AssertionError(f"Unhandled teacher target {target!r}")


def build_pd10_student_dataset_from_view(
    hlt_view: JetView,
    *,
    teacher_target: str = PD10_TEACHER_NONE,
    teacher_logit_dir: str | Path | None = None,
    teacher_block: PredictionBlock | None = None,
    max_jets: int | None = None,
    verify_hash: bool = True,
) -> PD10StudentDistillationDataset:
    target = normalize_pd10_teacher_target(teacher_target)
    block = teacher_block
    if target != PD10_TEACHER_NONE and block is None:
        if teacher_logit_dir is None:
            raise FileNotFoundError(f"teacher_logit_dir is required for teacher_target={target!r}")
        block = load_pd10_student_teacher_block(teacher_logit_dir, target, hlt_view.split, verify_hash=verify_hash)
    return PD10StudentDistillationDataset(
        hlt_view,
        teacher_target=target,
        teacher_block=block,
        max_jets=max_jets,
    )


def load_pd10_student_dataset(
    hlt_cache_dir: str | Path,
    split: str,
    *,
    teacher_target: str = PD10_TEACHER_NONE,
    teacher_logit_dir: str | Path | None = None,
    max_jets: int | None = None,
    verify_hlt_hash: bool = True,
    verify_teacher_hash: bool = True,
) -> PD10StudentDistillationDataset:
    hlt_view = load_cached_hlt_view(hlt_cache_dir, split, verify_hash=bool(verify_hlt_hash))
    return build_pd10_student_dataset_from_view(
        hlt_view,
        teacher_target=teacher_target,
        teacher_logit_dir=teacher_logit_dir,
        max_jets=max_jets,
        verify_hash=bool(verify_teacher_hash),
    )


def assert_pd10_student_batch_hlt_only(batch: Mapping[str, Any]) -> None:
    allowed_audit_keys = {"returns_offline_particles"}
    forbidden = [
        key
        for key in batch
        if "offline" in str(key).lower() and str(key) not in allowed_audit_keys
    ]
    if forbidden:
        raise ValueError(f"PD10 student batch exposes offline fields: {forbidden}")
    if batch.get("student_allowed_inputs") != PD10_STUDENT_ALLOWED_INPUTS:
        raise ValueError("PD10 student batch must declare HLT_only inputs")
    if bool(batch.get("returns_offline_particles")):
        raise ValueError("PD10 student batch must not return offline particles")


__all__ = [
    "PD10_STEP6_EXPERIMENT_STEP",
    "PD10_STUDENT_ALLOWED_INPUTS",
    "PD10_STUDENT_DATASET_CONTRACT",
    "PD10StudentBatchMetadata",
    "PD10StudentDistillationDataset",
    "align_hlt_view_to_teacher_block",
    "assert_pd10_student_batch_hlt_only",
    "build_pd10_student_dataset_from_view",
    "collate_pd10_student_batch",
    "limit_hlt_view_without_teacher",
    "load_pd10_student_dataset",
    "load_pd10_student_teacher_block",
    "make_pd10_student_data_loader",
    "move_pd10_student_batch_to_device",
]
