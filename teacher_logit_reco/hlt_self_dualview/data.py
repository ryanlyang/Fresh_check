"""Datasets and collate helpers for deployable HLT self-dualview models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.hlt_baseline import build_particle_transformer_inputs_from_tokens, require_torch
from jetclass_fresh.hlt_cache import jet_identity_hash, load_cached_hlt_view
from jetclass_fresh.jetclass_data import JetIdentity, JetView

from .config import (
    HLT_SDV_ALLOWED_INPUTS,
    HLT_SDV_DEPLOYMENT_INPUTS,
    HLT_SDV_VARIANT_HLT2_PREFIX,
    HLT_SDV_VARIANT_SAME_VIEW,
    hlt_sdv_dual_hlt2_variant_name,
)

HLT_SDV_STEP3_EXPERIMENT_STEP = "hlt_sdv_step3_dual_hlt_dataset"
HLT_SDV_DATASET_CONTRACT = "hlt_self_dualview_dual_hlt_dataset_v1"
HLT_SDV_BRANCH2_HLT2 = "hlt2"
HLT_SDV_BRANCH2_SAME_HLT = "same_hlt"


@dataclass(frozen=True)
class HLTSelfDualViewDatasetMetadata:
    """Serializable summary for HLT self-dualview dataset construction."""

    contract: str = HLT_SDV_DATASET_CONTRACT
    experiment_step: str = HLT_SDV_STEP3_EXPERIMENT_STEP
    split: str = ""
    n_jets: int = 0
    branch2_mode: str = HLT_SDV_BRANCH2_HLT2
    hlt_content_hash: str | None = None
    hlt2_content_hash: str | None = None
    hlt_jet_identity_hash: str | None = None
    hlt2_jet_identity_hash: str | None = None
    allowed_inputs: str = HLT_SDV_ALLOWED_INPUTS
    deployment_inputs: str = HLT_SDV_DEPLOYMENT_INPUTS
    returns_offline_particles: bool = False
    requires_offline_inputs: bool = False
    requires_teacher_features: bool = False
    branch2_uses_parent_hlt_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "experiment_step": self.experiment_step,
            "split": self.split,
            "n_jets": int(self.n_jets),
            "branch2_mode": self.branch2_mode,
            "hlt_content_hash": self.hlt_content_hash,
            "hlt2_content_hash": self.hlt2_content_hash,
            "hlt_jet_identity_hash": self.hlt_jet_identity_hash,
            "hlt2_jet_identity_hash": self.hlt2_jet_identity_hash,
            "allowed_inputs": self.allowed_inputs,
            "deployment_inputs": self.deployment_inputs,
            "returns_offline_particles": bool(self.returns_offline_particles),
            "requires_offline_inputs": bool(self.requires_offline_inputs),
            "requires_teacher_features": bool(self.requires_teacher_features),
            "branch2_uses_parent_hlt_cache": bool(self.branch2_uses_parent_hlt_cache),
        }


def _identity_key(identity: JetIdentity) -> tuple[str, int, int]:
    return str(identity.file), int(identity.entry), int(identity.label)


def _view_index_by_identity(view: JetView) -> dict[tuple[str, int, int], int]:
    result: dict[tuple[str, int, int], int] = {}
    for index, identity in enumerate(view.jet_ids):
        key = _identity_key(identity)
        if key in result:
            raise ValueError(f"duplicate jet identity in {view.split}: {identity}")
        result[key] = int(index)
    return result


def _limit_view(view: JetView, max_jets: int | None) -> JetView:
    if max_jets is None or int(max_jets) >= int(len(view.labels)):
        return view
    limit = max(int(max_jets), 0)
    return JetView(
        tokens=np.asarray(view.tokens[:limit], dtype=np.float32),
        mask=np.asarray(view.mask[:limit], dtype=bool),
        labels=np.asarray(view.labels[:limit], dtype=np.int64),
        jet_ids=list(view.jet_ids[:limit]),
        split=view.split,
        metadata={**dict(view.metadata), "hlt_sdv_max_jets": int(limit)},
    )


def _check_parent_hlt_view(view: JetView) -> None:
    source_view = view.metadata.get("view")
    if source_view not in (None, "fixed_hlt"):
        raise ValueError(f"HLT-SDV branch 1 requires fixed_hlt view, got {source_view!r}")
    if not np.isfinite(view.tokens).all():
        raise FloatingPointError("parent HLT tokens contain non-finite values")
    if view.mask.shape != view.tokens.shape[:2]:
        raise ValueError("parent HLT mask shape does not match tokens")


def _check_branch2_view(view: JetView, branch2_mode: str) -> None:
    if branch2_mode == HLT_SDV_BRANCH2_SAME_HLT:
        _check_parent_hlt_view(view)
        return
    if branch2_mode != HLT_SDV_BRANCH2_HLT2:
        raise ValueError(f"unknown HLT-SDV branch2_mode {branch2_mode!r}")
    source_view = view.metadata.get("view")
    if source_view not in (None, "hlt2"):
        raise ValueError(f"HLT-SDV branch 2 requires hlt2 view, got {source_view!r}")
    if bool(view.metadata.get("uses_offline_particles")):
        raise ValueError("HLT2 cache metadata says it used offline particles")
    if view.metadata.get("allowed_inputs") not in (None, HLT_SDV_ALLOWED_INPUTS):
        raise ValueError("HLT2 cache is not compatible with HLT-only inputs")
    if not np.isfinite(view.tokens).all():
        raise FloatingPointError("HLT2 tokens contain non-finite values")
    if view.mask.shape != view.tokens.shape[:2]:
        raise ValueError("HLT2 mask shape does not match tokens")


def normalize_hlt_sdv_branch2_mode(mode: str) -> str:
    value = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "hlt2": HLT_SDV_BRANCH2_HLT2,
        "h2": HLT_SDV_BRANCH2_HLT2,
        "second_hlt": HLT_SDV_BRANCH2_HLT2,
        "same": HLT_SDV_BRANCH2_SAME_HLT,
        "same_hlt": HLT_SDV_BRANCH2_SAME_HLT,
        "hlt_same": HLT_SDV_BRANCH2_SAME_HLT,
        "same_view": HLT_SDV_BRANCH2_SAME_HLT,
    }
    if value not in aliases:
        raise ValueError(f"unknown HLT-SDV branch2 mode {mode!r}")
    return aliases[value]


def hlt_sdv_branch2_mode_from_variant(variant: str) -> str:
    value = str(variant).strip()
    if value == HLT_SDV_VARIANT_SAME_VIEW:
        return HLT_SDV_BRANCH2_SAME_HLT
    if value.startswith(f"{HLT_SDV_VARIANT_HLT2_PREFIX}_"):
        return HLT_SDV_BRANCH2_HLT2
    raise ValueError(f"variant {variant!r} does not map to a Step-3 dual-HLT dataset mode")


def align_hlt_sdv_views(
    hlt_view: JetView,
    branch2_view: JetView,
    *,
    branch2_mode: str = HLT_SDV_BRANCH2_HLT2,
) -> tuple[JetView, JetView]:
    """Return HLT and branch-2 views ordered by the parent HLT rows."""

    mode = normalize_hlt_sdv_branch2_mode(branch2_mode)
    _check_parent_hlt_view(hlt_view)
    _check_branch2_view(branch2_view, mode)
    if hlt_view.split != branch2_view.split:
        raise ValueError(f"split mismatch: {hlt_view.split} != {branch2_view.split}")
    branch2_index = _view_index_by_identity(branch2_view)
    indices: list[int] = []
    for row, identity in enumerate(hlt_view.jet_ids):
        key = _identity_key(identity)
        if key not in branch2_index:
            raise ValueError(f"branch-2 view is missing parent HLT identity at row {row}: {identity}")
        branch2_row = branch2_index[key]
        if int(branch2_view.labels[branch2_row]) != int(hlt_view.labels[row]):
            raise ValueError(f"branch-2 label mismatch at parent row {row}")
        indices.append(branch2_row)

    aligned_branch2 = JetView(
        tokens=np.asarray(branch2_view.tokens[indices], dtype=np.float32),
        mask=np.asarray(branch2_view.mask[indices], dtype=bool),
        labels=np.asarray(branch2_view.labels[indices], dtype=np.int64),
        jet_ids=[branch2_view.jet_ids[index] for index in indices],
        split=branch2_view.split,
        metadata={
            **dict(branch2_view.metadata),
            "aligned_to_parent_hlt": True,
            "aligned_rows": int(len(indices)),
            "hlt_sdv_branch2_mode": mode,
        },
    )
    if aligned_branch2.jet_ids != hlt_view.jet_ids:
        raise ValueError("aligned branch-2 jet identities do not match parent HLT")
    if not np.array_equal(aligned_branch2.labels, hlt_view.labels):
        raise ValueError("aligned branch-2 labels do not match parent HLT labels")
    aligned_hlt = JetView(
        tokens=np.asarray(hlt_view.tokens, dtype=np.float32),
        mask=np.asarray(hlt_view.mask, dtype=bool),
        labels=np.asarray(hlt_view.labels, dtype=np.int64),
        jet_ids=list(hlt_view.jet_ids),
        split=hlt_view.split,
        metadata={
            **dict(hlt_view.metadata),
            "hlt_sdv_branch1": "HLT",
            "hlt_sdv_branch2_mode": mode,
        },
    )
    return aligned_hlt, aligned_branch2


class HLTSelfDualViewDataset:
    """Paired deployable HLT/HLT2 dataset for self-dualview fusion."""

    def __init__(
        self,
        hlt_view: JetView,
        branch2_view: JetView | None = None,
        *,
        branch2_mode: str = HLT_SDV_BRANCH2_HLT2,
        max_jets: int | None = None,
    ) -> None:
        require_torch()
        mode = normalize_hlt_sdv_branch2_mode(branch2_mode)
        parent_view = _limit_view(hlt_view, max_jets)
        secondary_view = parent_view if mode == HLT_SDV_BRANCH2_SAME_HLT else branch2_view
        if secondary_view is None:
            raise ValueError("HLT2 branch requires branch2_view unless branch2_mode='same_hlt'")
        if mode != HLT_SDV_BRANCH2_SAME_HLT:
            secondary_view = _limit_view(secondary_view, max_jets)
        aligned_hlt, aligned_branch2 = align_hlt_sdv_views(
            parent_view,
            secondary_view,
            branch2_mode=mode,
        )
        self.hlt_tokens = np.asarray(aligned_hlt.tokens, dtype=np.float32)
        self.hlt_mask = np.asarray(aligned_hlt.mask, dtype=bool)
        self.hlt2_tokens = np.asarray(aligned_branch2.tokens, dtype=np.float32)
        self.hlt2_mask = np.asarray(aligned_branch2.mask, dtype=bool)
        self.labels = np.asarray(aligned_hlt.labels, dtype=np.int64)
        self.jet_ids = list(aligned_hlt.jet_ids)
        self.split = aligned_hlt.split
        self.branch2_mode = mode
        self.hlt_metadata = dict(aligned_hlt.metadata)
        self.hlt2_metadata = dict(aligned_branch2.metadata)
        self.metadata = HLTSelfDualViewDatasetMetadata(
            split=self.split,
            n_jets=int(len(self.labels)),
            branch2_mode=self.branch2_mode,
            hlt_content_hash=self.hlt_metadata.get("hlt_content_hash"),
            hlt2_content_hash=self.hlt2_metadata.get("hlt2_content_hash")
            or self.hlt2_metadata.get("hlt_content_hash"),
            hlt_jet_identity_hash=jet_identity_hash(self.jet_ids),
            hlt2_jet_identity_hash=jet_identity_hash(aligned_branch2.jet_ids),
            branch2_uses_parent_hlt_cache=self.branch2_mode == HLT_SDV_BRANCH2_SAME_HLT,
        )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "hlt_tokens": self.hlt_tokens[index],
            "hlt_mask": self.hlt_mask[index],
            "hlt2_tokens": self.hlt2_tokens[index],
            "hlt2_mask": self.hlt2_mask[index],
            "label": np.int64(self.labels[index]),
            "jet_id": self.jet_ids[index],
            "branch2_mode": self.branch2_mode,
        }

    def to_metadata(self) -> dict[str, Any]:
        return self.metadata.to_dict()


def _particle_input_dict(tokens: np.ndarray, mask: np.ndarray, labels: np.ndarray, *, source_view: str) -> dict[str, Any]:
    torch = require_torch()
    inputs = build_particle_transformer_inputs_from_tokens(
        tokens,
        mask,
        labels=labels,
        source_view=source_view,
    )
    return {
        "points": torch.from_numpy(inputs.pf_points).float(),
        "features": torch.from_numpy(inputs.pf_features).float(),
        "lorentz_vectors": torch.from_numpy(inputs.pf_vectors).float(),
        "mask": torch.from_numpy(inputs.pf_mask).bool(),
    }


def collate_hlt_sdv_batch(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate HLT and HLT2 particles into Particle Transformer inputs."""

    if not samples:
        raise ValueError("cannot collate an empty HLT-SDV batch")
    torch = require_torch()
    branch_modes = {str(sample.get("branch2_mode")) for sample in samples}
    if len(branch_modes) != 1:
        raise ValueError(f"HLT-SDV batch mixes branch2 modes: {sorted(branch_modes)}")
    branch2_mode = branch_modes.pop()
    hlt_tokens = np.stack([np.asarray(sample["hlt_tokens"], dtype=np.float32) for sample in samples], axis=0)
    hlt_mask = np.stack([np.asarray(sample["hlt_mask"], dtype=bool) for sample in samples], axis=0)
    hlt2_tokens = np.stack([np.asarray(sample["hlt2_tokens"], dtype=np.float32) for sample in samples], axis=0)
    hlt2_mask = np.stack([np.asarray(sample["hlt2_mask"], dtype=bool) for sample in samples], axis=0)
    labels = np.asarray([sample["label"] for sample in samples], dtype=np.int64)
    hlt_inputs = _particle_input_dict(hlt_tokens, hlt_mask, labels, source_view="fixed_hlt")
    hlt2_inputs = _particle_input_dict(
        hlt2_tokens,
        hlt2_mask,
        labels,
        source_view="fixed_hlt" if branch2_mode == HLT_SDV_BRANCH2_SAME_HLT else "hlt2",
    )
    jet_ids = [sample["jet_id"] for sample in samples]
    batch = {
        "hlt_inputs": hlt_inputs,
        "hlt2_inputs": hlt2_inputs,
        "hlt": hlt_inputs,
        "hlt2": hlt2_inputs,
        "labels": torch.from_numpy(labels).long(),
        "hlt_tokens": torch.from_numpy(hlt_tokens).float(),
        "hlt2_tokens": torch.from_numpy(hlt2_tokens).float(),
        "hlt_constituent_mask": torch.from_numpy(hlt_mask).bool(),
        "hlt2_constituent_mask": torch.from_numpy(hlt2_mask).bool(),
        "jet_ids": jet_ids,
        "jet_files": [identity.file for identity in jet_ids],
        "jet_entries": torch.tensor([int(identity.entry) for identity in jet_ids], dtype=torch.long),
        "jet_identity_labels": torch.tensor([int(identity.label) for identity in jet_ids], dtype=torch.long),
        "jet_keys": [identity.key() for identity in jet_ids],
        "branch2_mode": branch2_mode,
        "student_allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
        "deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "returns_offline_particles": False,
        "requires_offline_inputs": False,
        "requires_teacher_features": False,
    }
    assert_hlt_sdv_batch_deployable(batch)
    return batch


def make_hlt_sdv_data_loader(
    dataset: HLTSelfDualViewDataset,
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
        collate_fn=collate_hlt_sdv_batch,
        generator=generator,
    )


def move_hlt_sdv_batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    torch = require_torch()
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Mapping):
            moved[key] = {
                sub_key: sub_value.to(device, non_blocking=True) if torch.is_tensor(sub_value) else sub_value
                for sub_key, sub_value in value.items()
            }
        else:
            moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def load_hlt_sdv_dataset(
    hlt_cache_dir: str | Path,
    split: str,
    *,
    hlt2_cache_dir: str | Path | None = None,
    branch2_mode: str = HLT_SDV_BRANCH2_HLT2,
    max_jets: int | None = None,
    verify_hlt_hash: bool = True,
    verify_hlt2_hash: bool = True,
) -> HLTSelfDualViewDataset:
    mode = normalize_hlt_sdv_branch2_mode(branch2_mode)
    hlt_view = load_cached_hlt_view(hlt_cache_dir, split, verify_hash=bool(verify_hlt_hash))
    hlt2_view: JetView | None
    if mode == HLT_SDV_BRANCH2_SAME_HLT:
        hlt2_view = None
    else:
        if hlt2_cache_dir is None:
            raise ValueError("hlt2_cache_dir is required when branch2_mode='hlt2'")
        hlt2_view = load_cached_hlt_view(hlt2_cache_dir, split, verify_hash=bool(verify_hlt2_hash))
    return HLTSelfDualViewDataset(
        hlt_view,
        hlt2_view,
        branch2_mode=mode,
        max_jets=max_jets,
    )


def assert_hlt_sdv_batch_deployable(batch: Mapping[str, Any]) -> None:
    allowed_audit_keys = {"returns_offline_particles", "requires_offline_inputs"}
    forbidden = [
        key
        for key in batch
        if "offline" in str(key).lower() and str(key) not in allowed_audit_keys
    ]
    if forbidden:
        raise ValueError(f"HLT-SDV batch exposes offline fields: {forbidden}")
    if batch.get("student_allowed_inputs") != HLT_SDV_ALLOWED_INPUTS:
        raise ValueError("HLT-SDV batch must declare HLT_only inputs")
    if batch.get("deployment_inputs") != HLT_SDV_DEPLOYMENT_INPUTS:
        raise ValueError("HLT-SDV batch must declare deterministic HLT2 deployment inputs")
    if bool(batch.get("returns_offline_particles")):
        raise ValueError("HLT-SDV batch must not return offline particles")
    if bool(batch.get("requires_offline_inputs")):
        raise ValueError("HLT-SDV batch must not require offline inputs")
    if bool(batch.get("requires_teacher_features")):
        raise ValueError("HLT-SDV batch must not require teacher features")


__all__ = [
    "HLT_SDV_BRANCH2_HLT2",
    "HLT_SDV_BRANCH2_SAME_HLT",
    "HLT_SDV_DATASET_CONTRACT",
    "HLT_SDV_STEP3_EXPERIMENT_STEP",
    "HLTSelfDualViewDataset",
    "HLTSelfDualViewDatasetMetadata",
    "align_hlt_sdv_views",
    "assert_hlt_sdv_batch_deployable",
    "collate_hlt_sdv_batch",
    "hlt_sdv_branch2_mode_from_variant",
    "load_hlt_sdv_dataset",
    "make_hlt_sdv_data_loader",
    "move_hlt_sdv_batch_to_device",
    "normalize_hlt_sdv_branch2_mode",
]
