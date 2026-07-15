"""Fail-closed input validation and a clean HLT-only dataset for ABPH."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_cache import jet_identity_hash, load_cached_hlt_view, normalize_hlt_profile
from jetclass_fresh.jetclass_data import (
    JetIdentity,
    JetView,
    LABEL_NAMES,
    RAW_TOKEN_DIM,
    SplitManifest,
    audit_split_manifest,
    load_split_manifest,
    manifest_hash,
)
from teacher_logit_reco.architecture_view_part import load_cached_offline_view

from .config import (
    ABPH_HLT_DEGRADATION_STRENGTH,
    ABPH_HLT_PROFILE,
    ABPH_HLT_PROFILE_VERSION,
    ABPH_INPUT_CONTRACT,
    ABPH_LABEL_NAMES,
    ABPH_OFFLINE_SUPERVISION_SPLITS,
    ABPH_SPLIT_ORDER,
    abph_hlt_params_dict,
    abph_split_sizes,
)
from .schemas import ABPH_MAX_PARTICLES


ABPH_HLT_ONLY_DATASET_CONTRACT = "adaptive_binary_pseudooffline_hlt_only_dataset_v1"
ABPH_INPUT_AUDIT_CONTRACT = "adaptive_binary_pseudooffline_input_audit_v1"


def _hash_labels(labels: np.ndarray) -> str:
    values = np.ascontiguousarray(labels, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _identity_tuple(identity: JetIdentity) -> tuple[str, int, int]:
    return str(identity.file), int(identity.entry), int(identity.label)


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def validate_manifest_contract(
    manifest: SplitManifest,
    *,
    expected_split_sizes: Mapping[str, int],
) -> dict[str, Any]:
    """Validate split shape, labels, constituent contract, and non-overlap."""

    _require_equal(tuple(manifest.class_names), ABPH_LABEL_NAMES, "manifest class mapping")
    _require_equal(int(manifest.max_constits), ABPH_MAX_PARTICLES, "manifest max_constits")
    if set(manifest.splits) != set(ABPH_SPLIT_ORDER):
        raise ValueError(
            f"manifest split keys must be exactly {ABPH_SPLIT_ORDER}, got {tuple(manifest.splits.keys())}"
        )
    if set(expected_split_sizes) != set(ABPH_SPLIT_ORDER):
        raise ValueError(f"expected_split_sizes keys must be exactly {ABPH_SPLIT_ORDER}")
    for split in ABPH_SPLIT_ORDER:
        expected = int(expected_split_sizes[split])
        declared = int(manifest.split_sizes.get(split, -1))
        actual = len(manifest.splits[split])
        if declared != expected or actual != expected:
            raise ValueError(
                f"split {split} size mismatch: declared={declared}, actual={actual}, expected={expected}"
            )
        labels = np.asarray([row.label for row in manifest.splits[split]], dtype=np.int64)
        if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= len(LABEL_NAMES)):
            raise ValueError(f"split {split} contains out-of-range labels")
        if expected % len(LABEL_NAMES) == 0:
            expected_per_class = expected // len(LABEL_NAMES)
            counts = np.bincount(labels, minlength=len(LABEL_NAMES))
            if not np.all(counts == expected_per_class):
                raise ValueError(
                    f"split {split} is not class-balanced: {counts.tolist()} != {expected_per_class} each"
                )
    base_audit = audit_split_manifest(manifest)
    if not bool(base_audit.get("ok")):
        raise ValueError(f"base split-manifest audit failed: {base_audit}")
    return {
        "ok": True,
        "manifest_hash": manifest_hash(manifest),
        "max_constits": int(manifest.max_constits),
        "class_names": list(manifest.class_names),
        "split_sizes": {split: len(manifest.splits[split]) for split in ABPH_SPLIT_ORDER},
        "base_audit": base_audit,
    }


def validate_hlt_view_contract(
    view: JetView,
    manifest: SplitManifest,
    split: str,
    *,
    expected_n_jets: int,
) -> dict[str, Any]:
    """Bind one HLT cache view to the active manifest and HLT-v2 contract."""

    if split not in ABPH_SPLIT_ORDER:
        raise ValueError(f"unknown split {split!r}")
    _require_equal(str(view.split), split, "HLT view split")
    expected_ids = tuple(manifest.splits[split])
    actual_ids = tuple(view.jet_ids)
    if actual_ids != expected_ids:
        first_bad = next(
            (
                index
                for index, (actual, expected) in enumerate(zip(actual_ids, expected_ids))
                if _identity_tuple(actual) != _identity_tuple(expected)
            ),
            min(len(actual_ids), len(expected_ids)),
        )
        raise ValueError(f"HLT jet identities do not match manifest for {split}; first mismatch={first_bad}")
    if int(len(actual_ids)) != int(expected_n_jets):
        raise ValueError(f"HLT {split} rows {len(actual_ids)} != expected {expected_n_jets}")
    tokens = np.asarray(view.tokens)
    mask = np.asarray(view.mask)
    labels = np.asarray(view.labels)
    expected_shape = (int(expected_n_jets), ABPH_MAX_PARTICLES, RAW_TOKEN_DIM)
    if tuple(tokens.shape) != expected_shape:
        raise ValueError(f"HLT {split} tokens shape {tokens.shape} != {expected_shape}")
    if tuple(mask.shape) != expected_shape[:2] or mask.dtype != np.bool_:
        raise ValueError(f"HLT {split} mask must be bool with shape {expected_shape[:2]}")
    if tuple(labels.shape) != (int(expected_n_jets),):
        raise ValueError(f"HLT {split} labels shape {labels.shape} is invalid")
    identity_labels = np.asarray([row.label for row in actual_ids], dtype=np.int64)
    if not np.array_equal(labels.astype(np.int64, copy=False), identity_labels):
        raise ValueError(f"HLT {split} labels do not match jet identities")
    if not np.isfinite(tokens[mask]).all():
        raise ValueError(f"HLT {split} contains nonfinite valid-particle features")

    metadata = dict(view.metadata)
    manifest_sha = manifest_hash(manifest)
    _require_equal(metadata.get("source_manifest_hash"), manifest_sha, f"HLT {split} source_manifest_hash")
    _require_equal(normalize_hlt_profile(metadata.get("hlt_profile")), ABPH_HLT_PROFILE, f"HLT {split} profile")
    _require_equal(
        str(metadata.get("hlt_profile_version") or ""),
        ABPH_HLT_PROFILE_VERSION,
        f"HLT {split} profile version",
    )
    strength = metadata.get("hlt_degradation_strength")
    if strength is None or abs(float(strength) - ABPH_HLT_DEGRADATION_STRENGTH) > 1.0e-12:
        raise ValueError(
            f"HLT {split} degradation strength mismatch: {strength} != {ABPH_HLT_DEGRADATION_STRENGTH}"
        )
    _require_equal(metadata.get("hlt_params"), abph_hlt_params_dict(), f"HLT {split} parameters")
    content_hash = metadata.get("hlt_content_hash")
    if not content_hash:
        raise ValueError(f"HLT {split} metadata lacks hlt_content_hash")
    actual_identity_hash = jet_identity_hash(actual_ids)
    _require_equal(metadata.get("jet_identity_hash"), actual_identity_hash, f"HLT {split} identity hash")
    return {
        "ok": True,
        "split": split,
        "n_jets": int(expected_n_jets),
        "source_manifest_hash": manifest_sha,
        "hlt_content_hash": str(content_hash),
        "jet_identity_hash": actual_identity_hash,
        "label_hash": _hash_labels(labels),
        "hlt_profile": ABPH_HLT_PROFILE,
        "hlt_profile_version": ABPH_HLT_PROFILE_VERSION,
        "hlt_degradation_strength": ABPH_HLT_DEGRADATION_STRENGTH,
    }


def validate_offline_view_contract(
    view: JetView,
    manifest: SplitManifest,
    split: str,
    *,
    expected_n_jets: int,
    hlt_view: JetView | None = None,
) -> dict[str, Any]:
    """Bind one offline target source to the same jets and labels as HLT."""

    _require_equal(str(view.split), split, "offline view split")
    expected_ids = tuple(manifest.splits[split])
    actual_ids = tuple(view.jet_ids)
    if actual_ids != expected_ids:
        raise ValueError(f"offline jet identities do not match manifest for {split}")
    if len(actual_ids) != int(expected_n_jets):
        raise ValueError(f"offline {split} rows {len(actual_ids)} != expected {expected_n_jets}")
    tokens = np.asarray(view.tokens)
    mask = np.asarray(view.mask)
    labels = np.asarray(view.labels, dtype=np.int64)
    expected_shape = (int(expected_n_jets), ABPH_MAX_PARTICLES, RAW_TOKEN_DIM)
    if tuple(tokens.shape) != expected_shape or tuple(mask.shape) != expected_shape[:2]:
        raise ValueError(f"offline {split} arrays violate fixed {expected_shape} token contract")
    if mask.dtype != np.bool_ or not np.isfinite(tokens[mask]).all():
        raise ValueError(f"offline {split} mask/features are invalid")
    identity_labels = np.asarray([row.label for row in actual_ids], dtype=np.int64)
    if not np.array_equal(labels, identity_labels):
        raise ValueError(f"offline {split} labels do not match identities")
    metadata = dict(view.metadata)
    manifest_sha = manifest_hash(manifest)
    _require_equal(metadata.get("source_manifest_hash"), manifest_sha, f"offline {split} source_manifest_hash")
    content_hash = metadata.get("offline_content_hash") or metadata.get("content_hash")
    if not content_hash:
        raise ValueError(f"offline {split} metadata lacks offline_content_hash")
    actual_identity_hash = jet_identity_hash(actual_ids)
    _require_equal(metadata.get("jet_identity_hash"), actual_identity_hash, f"offline {split} identity hash")
    if hlt_view is not None:
        if tuple(hlt_view.jet_ids) != actual_ids:
            raise ValueError(f"HLT/offline identities differ for {split}")
        if not np.array_equal(np.asarray(hlt_view.labels, dtype=np.int64), labels):
            raise ValueError(f"HLT/offline labels differ for {split}")
    return {
        "ok": True,
        "split": split,
        "n_jets": int(expected_n_jets),
        "source_manifest_hash": manifest_sha,
        "offline_content_hash": str(content_hash),
        "jet_identity_hash": actual_identity_hash,
        "label_hash": _hash_labels(labels),
    }


@dataclass(frozen=True)
class AdaptiveBinaryInputAuditConfig:
    manifest_path: str
    hlt_cache_dir: str
    offline_cache_dir: str | None = None
    campaign_mode: str = "highdata"
    hlt_splits: tuple[str, ...] = ABPH_SPLIT_ORDER
    offline_splits: tuple[str, ...] = ABPH_OFFLINE_SUPERVISION_SPLITS
    require_offline: bool = True
    verify_hash: bool = True
    expected_split_sizes: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        source_expected = abph_split_sizes(self.campaign_mode) if self.expected_split_sizes is None else self.expected_split_sizes
        if set(source_expected) != set(ABPH_SPLIT_ORDER):
            raise ValueError(f"expected_split_sizes keys must be exactly {ABPH_SPLIT_ORDER}")
        expected = {split: int(source_expected[split]) for split in ABPH_SPLIT_ORDER}
        hlt_splits = tuple(str(row) for row in self.hlt_splits)
        offline_splits = tuple(str(row) for row in self.offline_splits)
        if any(split not in ABPH_SPLIT_ORDER for split in hlt_splits + offline_splits):
            raise ValueError("audit contains an unknown split")
        if bool(self.require_offline) and not self.offline_cache_dir:
            raise ValueError("require_offline=True requires offline_cache_dir")
        if "final_test" in offline_splits:
            raise ValueError("deployable Step 1 audit never requires an offline final_test cache")
        object.__setattr__(self, "hlt_splits", hlt_splits)
        object.__setattr__(self, "offline_splits", offline_splits)
        object.__setattr__(self, "expected_split_sizes", expected)


def audit_input_contract(config: AdaptiveBinaryInputAuditConfig) -> dict[str, Any]:
    """Load, hash-check, and cross-bind every requested input split."""

    manifest = load_split_manifest(config.manifest_path)
    manifest_report = validate_manifest_contract(
        manifest,
        expected_split_sizes=config.expected_split_sizes,
    )
    hlt_reports: dict[str, Any] = {}
    offline_reports: dict[str, Any] = {}
    hlt_views: dict[str, JetView] = {}
    for split in config.hlt_splits:
        view = load_cached_hlt_view(config.hlt_cache_dir, split, verify_hash=bool(config.verify_hash))
        hlt_views[split] = view
        hlt_reports[split] = validate_hlt_view_contract(
            view,
            manifest,
            split,
            expected_n_jets=int(config.expected_split_sizes[split]),
        )
    if config.require_offline:
        for split in config.offline_splits:
            view = load_cached_offline_view(config.offline_cache_dir, split, verify_hash=bool(config.verify_hash))
            hlt_view = hlt_views.get(split)
            if hlt_view is None:
                hlt_view = load_cached_hlt_view(config.hlt_cache_dir, split, verify_hash=bool(config.verify_hash))
                validate_hlt_view_contract(
                    hlt_view,
                    manifest,
                    split,
                    expected_n_jets=int(config.expected_split_sizes[split]),
                )
            offline_reports[split] = validate_offline_view_contract(
                view,
                manifest,
                split,
                expected_n_jets=int(config.expected_split_sizes[split]),
                hlt_view=hlt_view,
            )
    return {
        "ok": True,
        "contract": ABPH_INPUT_AUDIT_CONTRACT,
        "input_contract": ABPH_INPUT_CONTRACT,
        "campaign_mode": config.campaign_mode,
        "manifest": manifest_report,
        "hlt_splits": hlt_reports,
        "offline_splits": offline_reports,
        "offline_final_test_loaded": False,
    }


class AdaptiveBinaryHLTOnlyDataset:
    """Target-free HLT particle dataset used by clean baselines and deployment."""

    def __init__(
        self,
        view: JetView,
        manifest: SplitManifest,
        *,
        expected_n_jets: int,
        max_jets: int | None = None,
    ) -> None:
        report = validate_hlt_view_contract(
            view,
            manifest,
            str(view.split),
            expected_n_jets=int(expected_n_jets),
        )
        n_rows = int(expected_n_jets)
        if max_jets is not None:
            if int(max_jets) <= 0:
                raise ValueError("max_jets must be positive")
            n_rows = min(n_rows, int(max_jets))
        self.tokens = np.asarray(view.tokens[:n_rows], dtype=np.float32)
        self.mask = np.asarray(view.mask[:n_rows], dtype=bool)
        self.labels = np.asarray(view.labels[:n_rows], dtype=np.int64)
        self.jet_ids = tuple(view.jet_ids[:n_rows])
        self.metadata = {
            "contract": ABPH_HLT_ONLY_DATASET_CONTRACT,
            "allowed_inputs": ["hlt_particles", "hlt_mask", "labels", "jet_identity"],
            "split": str(view.split),
            "n_jets": n_rows,
            "target_cache_loaded": False,
            "offline_inputs_loaded": False,
            "teacher_logits_loaded": False,
            "input_validation": report,
        }

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "tokens": self.tokens[index],
            "mask": self.mask[index],
            "labels": np.int64(self.labels[index]),
            "indices": np.int64(index),
        }

    def label_counts(self) -> dict[int, int]:
        values, counts = np.unique(self.labels, return_counts=True)
        return {int(value): int(count) for value, count in zip(values, counts)}


def load_hlt_only_dataset(
    manifest_path: str | Path,
    hlt_cache_dir: str | Path,
    split: str,
    *,
    campaign_mode: str = "highdata",
    expected_split_sizes: Mapping[str, int] | None = None,
    expected_n_jets: int | None = None,
    max_jets: int | None = None,
    verify_hash: bool = True,
) -> AdaptiveBinaryHLTOnlyDataset:
    """Load a clean baseline dataset without accepting any privileged path."""

    manifest = load_split_manifest(manifest_path)
    source_sizes = abph_split_sizes(campaign_mode) if expected_split_sizes is None else expected_split_sizes
    if set(source_sizes) != set(ABPH_SPLIT_ORDER):
        raise ValueError(f"expected_split_sizes keys must be exactly {ABPH_SPLIT_ORDER}")
    expected_sizes = {name: int(source_sizes[name]) for name in ABPH_SPLIT_ORDER}
    if expected_n_jets is not None:
        expected_sizes[split] = int(expected_n_jets)
    validate_manifest_contract(manifest, expected_split_sizes=expected_sizes)
    view = load_cached_hlt_view(hlt_cache_dir, split, verify_hash=bool(verify_hash))
    return AdaptiveBinaryHLTOnlyDataset(
        view,
        manifest,
        expected_n_jets=int(expected_sizes[split]),
        max_jets=max_jets,
    )


def write_input_audit_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = [
    "ABPH_HLT_ONLY_DATASET_CONTRACT",
    "ABPH_INPUT_AUDIT_CONTRACT",
    "AdaptiveBinaryHLTOnlyDataset",
    "AdaptiveBinaryInputAuditConfig",
    "audit_input_contract",
    "load_hlt_only_dataset",
    "validate_hlt_view_contract",
    "validate_manifest_contract",
    "validate_offline_view_contract",
    "write_input_audit_report",
]
