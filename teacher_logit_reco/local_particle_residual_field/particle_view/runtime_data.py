"""Authenticated cache access for the particle-view production campaign.

The scientific stages consume two representations of the *same* ordered jets:
fixed-HLT particles and offline particles.  This module makes that alignment a
runtime invariant, applies the logical stop/select slicing from the unified
manifest, and provides deliberately label-free recovery-probe loaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import (
    collate_particle_transformer_batch,
    require_torch,
)
from jetclass_fresh.hlt_cache import (
    HLT_ARRAY_FILENAME,
    HLT_METADATA_FILENAME,
    load_cached_hlt_view,
)
from jetclass_fresh.jetclass_data import (
    JetIdentity,
    JetView,
    SplitManifest,
    load_split_manifest,
    manifest_hash,
)
from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
from teacher_logit_reco.architecture_view_part.train import (
    ARCHITECTURE_VIEW_OFFLINE_ARRAY_FILENAME,
    ARCHITECTURE_VIEW_OFFLINE_METADATA_FILENAME,
    load_cached_offline_view,
)

from .contracts import (
    load_hashed_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)
from .registry import validate_particle_view_registry
from .runtime import validate_runtime_task_result
from .splits import (
    PARTICLE_VIEW_LOGICAL_SPLITS,
    PARTICLE_VIEW_UNIFIED_SPLIT_CONTRACT,
    ParticleViewSplitConfig,
    audit_unified_split_manifest,
    logical_split_identities,
)


PARTICLE_VIEW_RUNTIME_DATA_CONFIG_CONTRACT = (
    "particle_view_runtime_data_config_v1"
)
PARTICLE_VIEW_SOURCE_CACHE_AUDIT_CONTRACT = (
    "particle_view_source_cache_audit_v1"
)


def _config_from_unified(payload: Mapping[str, Any]) -> ParticleViewSplitConfig:
    raw = payload["split_config"]
    mapping = raw["source_mapping"]
    counts = raw["source_counts"]
    unused = tuple(raw["unused_parent_splits"])
    return ParticleViewSplitConfig(
        contract=str(raw["contract"]),
        class_names=tuple(raw["class_names"]),
        max_particles=int(raw["max_particles"]),
        train_parent_split=str(mapping["train"]),
        train_count=int(counts[mapping["train"]]),
        model_val_parent_split=str(mapping["model_val_stop"]),
        model_val_count=int(counts[mapping["model_val_stop"]]),
        stack_val_parent_split=str(mapping["stack_val"]),
        stack_val_count=int(counts[mapping["stack_val"]]),
        final_test_parent_split=str(mapping["final_test"]),
        final_test_count=int(counts[mapping["final_test"]]),
        unused_parent_splits=unused,
        unused_parent_split_counts=tuple(
            (name, int(counts[name])) for name in unused
        ),
        model_val_partition_seed=int(raw["model_val_partition_seed"]),
    )


def _source_parent_splits(unified: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(unified["logical_splits"][name]["parent_split"])
                for name in PARTICLE_VIEW_LOGICAL_SPLITS
            }
        )
    )


def _file_binding(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(f"runtime source file is absent or unsafe: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def build_runtime_data_config(
    *,
    parent_manifest_path: str | Path,
    unified_manifest_path: str | Path,
    hlt_cache_dir: str | Path,
    offline_cache_dir: str | Path,
) -> dict[str, Any]:
    """Bind the exact manifests and cache files used by production factories."""

    parent_path = Path(parent_manifest_path).resolve()
    unified_path = Path(unified_manifest_path).resolve()
    parent = load_split_manifest(parent_path)
    unified = load_hashed_json(unified_path)
    validate_content_hash(
        unified, expected_contract=PARTICLE_VIEW_UNIFIED_SPLIT_CONTRACT
    )
    config = _config_from_unified(unified)
    audit_unified_split_manifest(unified, parent=parent, config=config)

    hlt_root = Path(hlt_cache_dir).resolve()
    offline_root = Path(offline_cache_dir).resolve()
    records = []
    for split in _source_parent_splits(unified):
        records.append(
            {
                "parent_split": split,
                "hlt_array": _file_binding(
                    hlt_root / HLT_ARRAY_FILENAME.format(split=split)
                ),
                "hlt_metadata": _file_binding(
                    hlt_root / HLT_METADATA_FILENAME.format(split=split)
                ),
                "offline_array": _file_binding(
                    offline_root
                    / ARCHITECTURE_VIEW_OFFLINE_ARRAY_FILENAME.format(split=split)
                ),
                "offline_metadata": _file_binding(
                    offline_root
                    / ARCHITECTURE_VIEW_OFFLINE_METADATA_FILENAME.format(split=split)
                ),
            }
        )
    payload = with_content_hash(
        {
            "contract": PARTICLE_VIEW_RUNTIME_DATA_CONFIG_CONTRACT,
            "parent_manifest": {
                **_file_binding(parent_path),
                "manifest_sha256": manifest_hash(parent),
            },
            "unified_manifest": {
                **_file_binding(unified_path),
                "manifest_sha256": unified["content_hash"],
            },
            "hlt_cache_dir": str(hlt_root),
            "offline_cache_dir": str(offline_root),
            "parent_cache_records": records,
            "logical_split_order": list(PARTICLE_VIEW_LOGICAL_SPLITS),
            "training_topology": "single_pool_no_crossfit_v1",
        }
    )
    validate_runtime_data_config(payload, verify_cache_files=True)
    return payload


def _load_bound_manifests(
    payload: Mapping[str, Any],
) -> tuple[SplitManifest, dict[str, Any], ParticleViewSplitConfig]:
    parent_binding = payload["parent_manifest"]
    unified_binding = payload["unified_manifest"]
    parent = load_split_manifest(parent_binding["path"])
    unified = load_hashed_json(unified_binding["path"])
    if manifest_hash(parent) != parent_binding["manifest_sha256"]:
        raise ValueError("runtime parent manifest content changed")
    if unified["content_hash"] != unified_binding["manifest_sha256"]:
        raise ValueError("runtime unified manifest content changed")
    config = _config_from_unified(unified)
    audit_unified_split_manifest(unified, parent=parent, config=config)
    return parent, unified, config


def validate_runtime_data_config(
    payload: Mapping[str, Any],
    *,
    verify_cache_files: bool = True,
) -> dict[str, Any]:
    """Validate source lineage and, by default, every bound cache file hash."""

    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_RUNTIME_DATA_CONFIG_CONTRACT
    )
    expected = {
        "contract",
        "parent_manifest",
        "unified_manifest",
        "hlt_cache_dir",
        "offline_cache_dir",
        "parent_cache_records",
        "logical_split_order",
        "training_topology",
        "content_hash",
    }
    if set(payload) != expected:
        raise ValueError("runtime data config field inventory mismatch")
    if (
        payload["logical_split_order"] != list(PARTICLE_VIEW_LOGICAL_SPLITS)
        or payload["training_topology"] != "single_pool_no_crossfit_v1"
    ):
        raise ValueError("runtime data split/topology contract changed")
    for name in ("parent_manifest", "unified_manifest"):
        binding = payload[name]
        if set(binding) != {"path", "sha256", "manifest_sha256"}:
            raise ValueError(f"{name} binding field inventory mismatch")
        path = Path(binding["path"])
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"{name} is absent or unsafe")
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"{name} file hash changed")

    parent, unified, _ = _load_bound_manifests(payload)
    expected_splits = _source_parent_splits(unified)
    records = payload["parent_cache_records"]
    if (
        not isinstance(records, list)
        or [row.get("parent_split") for row in records] != list(expected_splits)
    ):
        raise ValueError("runtime parent-cache split inventory mismatch")
    expected_record_fields = {
        "parent_split",
        "hlt_array",
        "hlt_metadata",
        "offline_array",
        "offline_metadata",
    }
    for row in records:
        if set(row) != expected_record_fields:
            raise ValueError("runtime cache record field inventory mismatch")
        split = row["parent_split"]
        if split not in parent.splits:
            raise ValueError("runtime cache record references an unknown split")
        expected_paths = {
            "hlt_array": Path(payload["hlt_cache_dir"])
            / HLT_ARRAY_FILENAME.format(split=split),
            "hlt_metadata": Path(payload["hlt_cache_dir"])
            / HLT_METADATA_FILENAME.format(split=split),
            "offline_array": Path(payload["offline_cache_dir"])
            / ARCHITECTURE_VIEW_OFFLINE_ARRAY_FILENAME.format(split=split),
            "offline_metadata": Path(payload["offline_cache_dir"])
            / ARCHITECTURE_VIEW_OFFLINE_METADATA_FILENAME.format(split=split),
        }
        for kind, expected_path in expected_paths.items():
            binding = row[kind]
            if set(binding) != {"path", "sha256"}:
                raise ValueError("runtime cache-file binding inventory mismatch")
            path = Path(binding["path"])
            if path.resolve() != expected_path.resolve():
                raise ValueError("runtime cache file escaped its bound cache root")
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError("runtime cache file is absent or unsafe")
            if verify_cache_files and sha256_file(path) != binding["sha256"]:
                raise ValueError(f"runtime {split} {kind} hash changed")
    return {
        "ok": True,
        "content_hash": payload["content_hash"],
        "parent_manifest_sha256": manifest_hash(parent),
        "unified_manifest_sha256": unified["content_hash"],
        "parent_splits": list(expected_splits),
    }


def _identity_keys(identities: Sequence[JetIdentity]) -> list[str]:
    return [identity.key() for identity in identities]


def _require_exact_parent_alignment(
    *,
    parent: SplitManifest,
    split: str,
    hlt: JetView,
    offline: JetView,
) -> None:
    expected = parent.splits[split]
    expected_keys = _identity_keys(expected)
    if _identity_keys(hlt.jet_ids) != expected_keys:
        raise ValueError(f"HLT cache identity order differs from parent split {split}")
    if _identity_keys(offline.jet_ids) != expected_keys:
        raise ValueError(
            f"offline cache identity order differs from parent split {split}"
        )
    expected_labels = np.asarray([row.label for row in expected], dtype=np.int64)
    if not np.array_equal(hlt.labels, expected_labels):
        raise ValueError(f"HLT cache labels differ from parent split {split}")
    if not np.array_equal(offline.labels, expected_labels):
        raise ValueError(f"offline cache labels differ from parent split {split}")


@dataclass(frozen=True)
class AlignedLogicalJetView:
    """Two authenticated parent views plus one immutable logical row index."""

    logical_split: str
    parent_split: str
    hlt: JetView
    offline: JetView
    parent_row_indices: np.ndarray
    logical_split_sha256: str
    ordered_identity_sha256: str

    def __len__(self) -> int:
        return int(self.parent_row_indices.shape[0])

    @property
    def identities(self) -> list[JetIdentity]:
        return [
            self.hlt.jet_ids[int(index)] for index in self.parent_row_indices
        ]


def load_aligned_logical_jet_view(
    payload: Mapping[str, Any],
    logical_split: str,
) -> AlignedLogicalJetView:
    """Load and align one HLT/offline logical split, failing closed on drift."""

    validate_runtime_data_config(payload, verify_cache_files=False)
    parent, unified, config = _load_bound_manifests(payload)
    if logical_split not in PARTICLE_VIEW_LOGICAL_SPLITS:
        raise KeyError(f"unknown particle-view logical split {logical_split!r}")
    split_payload = unified["logical_splits"][logical_split]
    parent_split = str(split_payload["parent_split"])
    record = next(
        row
        for row in payload["parent_cache_records"]
        if row["parent_split"] == parent_split
    )
    for kind in (
        "hlt_array",
        "hlt_metadata",
        "offline_array",
        "offline_metadata",
    ):
        path = Path(record[kind]["path"])
        if sha256_file(path) != record[kind]["sha256"]:
            raise ValueError(f"runtime {parent_split} {kind} hash changed")
    hlt = load_cached_hlt_view(
        payload["hlt_cache_dir"], parent_split, verify_hash=True
    )
    offline = load_cached_offline_view(
        payload["offline_cache_dir"], parent_split, verify_hash=True
    )
    _require_exact_parent_alignment(
        parent=parent,
        split=parent_split,
        hlt=hlt,
        offline=offline,
    )
    expected = logical_split_identities(
        unified,
        parent=parent,
        split_name=logical_split,
        config=config,
    )
    if split_payload["membership_kind"] == "complete_parent_alias":
        indices = np.arange(len(expected), dtype=np.int64)
    else:
        indices = np.asarray(split_payload["parent_row_indices"], dtype=np.int64)
    observed = [hlt.jet_ids[int(index)] for index in indices]
    if _identity_keys(observed) != _identity_keys(expected):
        raise ValueError("logical split slicing changed ordered identities")
    return AlignedLogicalJetView(
        logical_split=logical_split,
        parent_split=parent_split,
        hlt=hlt,
        offline=offline,
        parent_row_indices=indices,
        logical_split_sha256=str(split_payload["content_hash"]),
        ordered_identity_sha256=str(split_payload["ordered_identity_sha256"]),
    )


class AlignedLogicalJetDataset:
    """Row-level aligned HLT/offline samples without particle matching."""

    def __init__(self, aligned: AlignedLogicalJetView) -> None:
        self.aligned = aligned

    def __len__(self) -> int:
        return len(self.aligned)

    def __getitem__(self, index: int) -> dict[str, Any]:
        parent_index = int(self.aligned.parent_row_indices[index])
        return {
            "hlt_tokens": self.aligned.hlt.tokens[parent_index],
            "hlt_mask": self.aligned.hlt.mask[parent_index],
            "offline_tokens": self.aligned.offline.tokens[parent_index],
            "offline_mask": self.aligned.offline.mask[parent_index],
            "label": np.int64(self.aligned.hlt.labels[parent_index]),
            "parent_index": np.int64(parent_index),
        }


class LogicalParticleDataset:
    """Projection of an aligned split into one standard ParT input view."""

    def __init__(
        self, aligned: AlignedLogicalJetView, *, source_view: str
    ) -> None:
        if source_view not in {"fixed_hlt", "offline"}:
            raise ValueError("source_view must be fixed_hlt or offline")
        self.aligned = aligned
        self.source_view = source_view

    def __len__(self) -> int:
        return len(self.aligned)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.int64]:
        parent_index = int(self.aligned.parent_row_indices[index])
        view = (
            self.aligned.hlt
            if self.source_view == "fixed_hlt"
            else self.aligned.offline
        )
        return (
            view.tokens[parent_index],
            view.mask[parent_index],
            np.int64(view.labels[parent_index]),
        )


def collate_aligned_particle_view_batch(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build standard HLT inputs and prefixed offline-memory inputs."""

    torch = require_torch()
    hlt_samples = [
        (row["hlt_tokens"], row["hlt_mask"], row["label"]) for row in samples
    ]
    offline_samples = [
        (row["offline_tokens"], row["offline_mask"], row["label"])
        for row in samples
    ]
    hlt = collate_particle_transformer_batch(
        hlt_samples, source_view="fixed_hlt"
    )
    offline = collate_particle_transformer_batch(
        offline_samples, source_view="offline"
    )
    batch = dict(hlt)
    batch.update(
        {
            "hlt_tokens": torch.from_numpy(
                np.stack([row["hlt_tokens"] for row in samples])
            ).float(),
            "hlt_particle_mask": torch.from_numpy(
                np.stack([row["hlt_mask"] for row in samples])
            ).bool(),
            "offline_tokens": torch.from_numpy(
                np.stack([row["offline_tokens"] for row in samples])
            ).float(),
            "offline_particle_mask": torch.from_numpy(
                np.stack([row["offline_mask"] for row in samples])
            ).bool(),
            "offline_points": offline["points"],
            "offline_features": offline["features"],
            "offline_lorentz_vectors": offline["lorentz_vectors"],
            "offline_mask": offline["mask"],
            "parent_indices": torch.from_numpy(
                np.asarray(
                    [row["parent_index"] for row in samples], dtype=np.int64
                )
            ).long(),
        }
    )
    return batch


class RecoveryProbeLogicalDataset:
    """HLT inputs paired with canonical views; labels never enter the sample."""

    def __init__(
        self,
        aligned: AlignedLogicalJetView,
        true_views: np.ndarray,
    ) -> None:
        values = np.asarray(true_views, dtype=np.float32)
        if (
            values.ndim != 3
            or values.shape[0] != len(aligned)
            or values.shape[1] != aligned.hlt.tokens.shape[1]
        ):
            raise ValueError("recovery-probe true views have incompatible shape")
        masks = aligned.hlt.mask[aligned.parent_row_indices]
        if not np.isfinite(values[masks]).all():
            raise ValueError("recovery-probe true views contain nonfinite values")
        if np.any(~masks) and np.max(np.abs(values[~masks])) != 0.0:
            raise ValueError("recovery-probe true views must be zero on padding")
        self.aligned = aligned
        self.true_views = values

    def __len__(self) -> int:
        return len(self.aligned)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parent_index = int(self.aligned.parent_row_indices[index])
        return (
            self.aligned.hlt.tokens[parent_index],
            self.aligned.hlt.mask[parent_index],
            self.true_views[index],
        )


def collate_recovery_probe_batch(
    samples: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Return the exact label-free inventory required by recovery_probe.py."""

    torch = require_torch()
    tokens = np.stack([row[0] for row in samples])
    mask = np.stack([row[1] for row in samples])
    views = np.stack([row[2] for row in samples])
    inputs = build_particle_transformer_inputs_from_tokens(
        tokens,
        mask,
        labels=None,
        source_view="fixed_hlt",
    )
    return {
        "features": torch.from_numpy(inputs.pf_features).float(),
        "mask": torch.from_numpy(mask).bool(),
        "true_view": torch.from_numpy(views).float(),
    }


def make_logical_data_loader(
    aligned: AlignedLogicalJetView,
    *,
    mode: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    true_views: np.ndarray | None = None,
):
    """Construct deterministic HLT, offline, aligned, or label-free loaders."""

    torch = require_torch()
    if mode == "aligned":
        dataset = AlignedLogicalJetDataset(aligned)
        collate = collate_aligned_particle_view_batch
    elif mode in {"fixed_hlt", "offline"}:
        dataset = LogicalParticleDataset(aligned, source_view=mode)
        collate = partial(
            collate_particle_transformer_batch, source_view=mode
        )
    elif mode == "recovery_probe":
        if true_views is None:
            raise ValueError("recovery_probe mode requires true_views")
        dataset = RecoveryProbeLogicalDataset(aligned, true_views)
        collate = collate_recovery_probe_batch
    else:
        raise ValueError("unknown logical loader mode")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    worker_count = int(num_workers)
    kwargs: dict[str, Any] = {}
    if worker_count > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=worker_count,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
        generator=generator,
        **kwargs,
    )


def audit_runtime_data_sources(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Read each parent pair once and publish a compact alignment audit."""

    config_audit = validate_runtime_data_config(
        payload, verify_cache_files=True
    )
    parent, unified, _ = _load_bound_manifests(payload)
    parent_reports: dict[str, Any] = {}
    for split in config_audit["parent_splits"]:
        hlt = load_cached_hlt_view(
            payload["hlt_cache_dir"], split, verify_hash=True
        )
        offline = load_cached_offline_view(
            payload["offline_cache_dir"], split, verify_hash=True
        )
        _require_exact_parent_alignment(
            parent=parent, split=split, hlt=hlt, offline=offline
        )
        parent_reports[split] = {
            "count": len(hlt.jet_ids),
            "hlt_content_sha256": hlt.metadata.get("hlt_content_hash"),
            "offline_content_sha256": offline.metadata.get(
                "offline_content_hash"
            ),
            "identity_order_matches_parent": True,
        }
    logical_reports = {
        name: {
            "parent_split": unified["logical_splits"][name]["parent_split"],
            "count": unified["logical_splits"][name]["count"],
            "logical_split_sha256": unified["logical_splits"][name][
                "content_hash"
            ],
            "ordered_identity_sha256": unified["logical_splits"][name][
                "ordered_identity_sha256"
            ],
        }
        for name in PARTICLE_VIEW_LOGICAL_SPLITS
    }
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_SOURCE_CACHE_AUDIT_CONTRACT,
            "runtime_data_config_sha256": payload["content_hash"],
            "parent_manifest_sha256": config_audit[
                "parent_manifest_sha256"
            ],
            "unified_manifest_sha256": config_audit[
                "unified_manifest_sha256"
            ],
            "parent_splits": parent_reports,
            "logical_splits": logical_reports,
            "hlt_offline_identity_alignment": True,
            "single_training_pool": True,
            "cross_fit": False,
            "labels_used": False,
        }
    )


def resolve_parent_task_artifacts(
    *,
    registry: Mapping[str, Any],
    artifact_root: str | Path,
    run_id: str,
    seed: int,
) -> dict[str, Any]:
    """Resolve and authenticate the exact seed-compatible parent task results."""

    validate_particle_view_registry(registry)
    runs = {row["run_id"]: row for row in registry["runs"]}
    if run_id not in runs:
        raise KeyError(f"unknown particle-view run {run_id!r}")
    if int(seed) not in runs[run_id]["seed_ids"]:
        raise ValueError("requested seed is not registered for this run")
    root = Path(artifact_root).resolve()
    resolved: dict[str, Any] = {}
    for parent_id in runs[run_id]["parent_run_ids"]:
        parent_seeds = [int(value) for value in runs[parent_id]["seed_ids"]]
        parent_seed = int(seed) if int(seed) in parent_seeds else 101
        if parent_seed not in parent_seeds:
            raise ValueError("no seed-compatible parent replica exists")
        task_id = f"{parent_id}__seed_{parent_seed}"
        result_path = root / "runtime_tasks" / task_id / "task_result.json"
        result = load_hashed_json(result_path)
        validate_runtime_task_result(result, expected_task_id=task_id)
        by_name: dict[str, dict[str, str]] = {}
        for artifact in result["artifacts"]:
            name = Path(artifact["path"]).name
            if name in by_name:
                raise ValueError(
                    f"parent task {task_id} has duplicate artifact name {name}"
                )
            by_name[name] = dict(artifact)
        resolved[parent_id] = {
            "task_id": task_id,
            "seed": parent_seed,
            "result_path": str(result_path),
            "result_sha256": result["content_hash"],
            "artifacts": by_name,
        }
    return resolved


__all__ = [
    "AlignedLogicalJetDataset",
    "AlignedLogicalJetView",
    "LogicalParticleDataset",
    "PARTICLE_VIEW_RUNTIME_DATA_CONFIG_CONTRACT",
    "PARTICLE_VIEW_SOURCE_CACHE_AUDIT_CONTRACT",
    "RecoveryProbeLogicalDataset",
    "audit_runtime_data_sources",
    "build_runtime_data_config",
    "collate_aligned_particle_view_batch",
    "collate_recovery_probe_batch",
    "load_aligned_logical_jet_view",
    "make_logical_data_loader",
    "resolve_parent_task_artifacts",
    "validate_runtime_data_config",
]
