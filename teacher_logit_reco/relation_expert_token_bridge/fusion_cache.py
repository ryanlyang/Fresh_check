"""Identity-bound immutable frozen expert-token caches for offline fusion."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    bind_source,
    load_hashed_json,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .registry import EXPERT_ORDER


FROZEN_TOKEN_CACHE_CONTRACT = "retb_frozen_offline_token_cache_v1"
ALLOWED_SPLITS = ("model_train", "val_stop", "val_design")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_frozen_token_cache_manifest(
    *,
    split: str,
    pipeline_seed: int,
    shape_id: str,
    allocation: Mapping[str, Sequence[int] | tuple[int, int]],
    expert_checkpoint_hashes: Mapping[str, str],
    expert_registration_hashes: Mapping[str, str],
    identity_manifest_sha256: str,
    label_manifest_sha256: str,
    event_count: int,
    npz_filename: str,
    npz_sha256: str,
) -> dict[str, Any]:
    if split not in ALLOWED_SPLITS:
        raise ValueError("frozen token cache split is unauthorized")
    if int(pipeline_seed) not in {101, 202, 303}:
        raise ValueError("frozen token cache seed is not registered")
    if int(event_count) <= 0:
        raise ValueError("frozen token cache is empty")
    if Path(npz_filename).name != npz_filename or not npz_filename.endswith(".npz"):
        raise ValueError("frozen token cache filename is unsafe")
    if set(allocation) != set(EXPERT_ORDER):
        raise ValueError("frozen token allocation differs from expert order")
    normalized_allocation = {}
    for expert in EXPERT_ORDER:
        shape = list(allocation[expert])
        if len(shape) != 2 or int(shape[0]) not in {1, 2, 4, 8, 16}:
            raise ValueError("frozen token allocation has invalid K")
        if int(shape[1]) not in {64, 128}:
            raise ValueError("frozen token allocation has invalid D")
        normalized_allocation[expert] = [int(shape[0]), int(shape[1])]
    checkpoints = {
        expert: require_sha256(
            expert_checkpoint_hashes.get(expert),
            name=f"expert_checkpoint_hashes.{expert}",
        )
        for expert in EXPERT_ORDER
    }
    registrations = {
        expert: require_sha256(
            expert_registration_hashes.get(expert),
            name=f"expert_registration_hashes.{expert}",
        )
        for expert in EXPERT_ORDER
    }
    return with_content_hash(
        {
            "contract": FROZEN_TOKEN_CACHE_CONTRACT,
            "schema_version": 1,
            "split": split,
            "pipeline_seed": int(pipeline_seed),
            "shape_id": str(shape_id),
            "expert_order": list(EXPERT_ORDER),
            "allocation": normalized_allocation,
            "expert_checkpoint_hashes": checkpoints,
            "expert_registration_hashes": registrations,
            "identity_manifest_sha256": require_sha256(
                identity_manifest_sha256, name="identity_manifest_sha256"
            ),
            "label_manifest_sha256": require_sha256(
                label_manifest_sha256, name="label_manifest_sha256"
            ),
            "event_count": int(event_count),
            "npz_filename": npz_filename,
            "npz_sha256": require_sha256(npz_sha256, name="npz_sha256"),
            "contains": {
                "identity": True,
                "label": True,
                "tokens_for_all_seven_experts": True,
                "logits_for_all_seven_experts": True,
                "raw_particles": False,
                "attention_maps": False,
            },
            "expert_weights_frozen_during_generation": True,
        }
    )


def validate_frozen_token_cache_manifest(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=FROZEN_TOKEN_CACHE_CONTRACT
    )
    expected = build_frozen_token_cache_manifest(
        split=payload.get("split"),
        pipeline_seed=int(payload.get("pipeline_seed", -1)),
        shape_id=payload.get("shape_id"),
        allocation=payload.get("allocation", {}),
        expert_checkpoint_hashes=payload.get("expert_checkpoint_hashes", {}),
        expert_registration_hashes=payload.get(
            "expert_registration_hashes", {}
        ),
        identity_manifest_sha256=payload.get("identity_manifest_sha256"),
        label_manifest_sha256=payload.get("label_manifest_sha256"),
        event_count=int(payload.get("event_count", 0)),
        npz_filename=payload.get("npz_filename", ""),
        npz_sha256=payload.get("npz_sha256"),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("frozen token cache manifest differs")
    return digest


def _validate_arrays(
    *,
    identities: np.ndarray,
    labels: np.ndarray,
    token_banks: Mapping[str, np.ndarray],
    expert_logits: Mapping[str, np.ndarray],
) -> tuple[int, dict[str, list[int]]]:
    identity = np.asarray(identities)
    truth = np.asarray(labels)
    if identity.ndim != 1 or truth.shape != identity.shape or len(identity) == 0:
        raise ValueError("frozen cache identities/labels have incompatible shapes")
    strings = [str(value) for value in identity.tolist()]
    if any(not value for value in strings) or len(strings) != len(set(strings)):
        raise ValueError("frozen cache identities are empty or duplicated")
    if bool(((truth < 0) | (truth >= 10)).any()):
        raise ValueError("frozen cache labels lie outside 0..9")
    if set(token_banks) != set(EXPERT_ORDER) or set(expert_logits) != set(
        EXPERT_ORDER
    ):
        raise ValueError("frozen cache expert fields differ")
    allocation = {}
    for expert in EXPERT_ORDER:
        tokens = np.asarray(token_banks[expert])
        logits = np.asarray(expert_logits[expert])
        if (
            tokens.ndim != 3
            or tokens.shape[0] != len(identity)
            or tokens.shape[1] not in {1, 2, 4, 8, 16}
            or tokens.shape[2] not in {64, 128}
        ):
            raise ValueError(f"frozen cache token bank {expert} has wrong shape")
        if logits.shape != (len(identity), 10):
            raise ValueError(f"frozen cache logits {expert} have wrong shape")
        if not (np.isfinite(tokens).all() and np.isfinite(logits).all()):
            raise FloatingPointError("frozen cache contains nonfinite values")
        allocation[expert] = [int(tokens.shape[1]), int(tokens.shape[2])]
    return len(identity), allocation


def publish_frozen_token_cache(
    *,
    output_dir: str | Path,
    split: str,
    pipeline_seed: int,
    shape_id: str,
    identities: np.ndarray,
    labels: np.ndarray,
    token_banks: Mapping[str, np.ndarray],
    expert_logits: Mapping[str, np.ndarray],
    expert_checkpoint_hashes: Mapping[str, str],
    expert_registration_hashes: Mapping[str, str],
    identity_manifest_sha256: str,
    label_manifest_sha256: str,
    source_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    count, allocation = _validate_arrays(
        identities=identities,
        labels=labels,
        token_banks=token_banks,
        expert_logits=expert_logits,
    )
    root = Path(output_dir)
    if root.exists() and root.is_symlink():
        raise ValueError("frozen cache output cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    npz_path = root / f"{split}_frozen_tokens.npz"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{npz_path.name}.", suffix=".tmp.npz", dir=root
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        arrays: dict[str, np.ndarray] = {
            "identities": np.asarray(identities),
            "labels": np.asarray(labels, dtype=np.int64),
        }
        for expert in EXPERT_ORDER:
            arrays[f"tokens_{expert}"] = np.asarray(
                token_banks[expert], dtype=np.float32
            )
            arrays[f"logits_{expert}"] = np.asarray(
                expert_logits[expert], dtype=np.float32
            )
        np.savez_compressed(temporary, **arrays)
        encoded = temporary.read_bytes()
        if npz_path.exists():
            if npz_path.is_symlink() or npz_path.read_bytes() != encoded:
                raise FileExistsError("refusing to overwrite frozen token cache")
        else:
            os.link(temporary, npz_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = build_frozen_token_cache_manifest(
        split=split,
        pipeline_seed=pipeline_seed,
        shape_id=shape_id,
        allocation=allocation,
        expert_checkpoint_hashes=expert_checkpoint_hashes,
        expert_registration_hashes=expert_registration_hashes,
        identity_manifest_sha256=identity_manifest_sha256,
        label_manifest_sha256=label_manifest_sha256,
        event_count=count,
        npz_filename=npz_path.name,
        npz_sha256=_sha256(npz_path),
    )
    if source_snapshot is not None:
        manifest = bind_source(manifest, source_snapshot=source_snapshot)
    write_immutable_json(root / f"{split}_frozen_tokens.json", manifest)
    return manifest


def load_frozen_token_cache(
    manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(manifest_path)
    manifest = load_hashed_json(
        path, expected_contract=FROZEN_TOKEN_CACHE_CONTRACT
    )
    validate_frozen_token_cache_manifest(manifest)
    npz_path = path.parent / manifest["npz_filename"]
    if (
        not npz_path.is_file()
        or npz_path.is_symlink()
        or _sha256(npz_path) != manifest["npz_sha256"]
    ):
        raise ValueError("frozen token cache bytes differ")
    with np.load(npz_path, allow_pickle=False) as payload:
        expected_fields = {"identities", "labels"} | {
            f"{kind}_{expert}"
            for expert in EXPERT_ORDER
            for kind in ("tokens", "logits")
        }
        if set(payload.files) != expected_fields:
            raise ValueError("frozen token cache fields differ")
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    token_banks = {
        expert: arrays[f"tokens_{expert}"] for expert in EXPERT_ORDER
    }
    expert_logits = {
        expert: arrays[f"logits_{expert}"] for expert in EXPERT_ORDER
    }
    count, allocation = _validate_arrays(
        identities=arrays["identities"],
        labels=arrays["labels"],
        token_banks=token_banks,
        expert_logits=expert_logits,
    )
    if count != manifest["event_count"] or allocation != manifest["allocation"]:
        raise ValueError("frozen token cache arrays differ from manifest")
    return manifest, {
        "identities": arrays["identities"],
        "labels": arrays["labels"],
        "token_banks": token_banks,
        "expert_logits": expert_logits,
    }


def assert_cache_identity_alignment(
    caches: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> None:
    if not caches:
        raise ValueError("cache alignment requires at least one cache")
    reference_ids = np.asarray(caches[0][1]["identities"])
    reference_labels = np.asarray(caches[0][1]["labels"])
    for manifest, arrays in caches[1:]:
        if not np.array_equal(reference_ids, arrays["identities"]):
            raise ValueError(
                f"cache identities differ for {manifest.get('shape_id')}"
            )
        if not np.array_equal(reference_labels, arrays["labels"]):
            raise ValueError("cache labels differ across aligned shapes")


__all__ = [
    "FROZEN_TOKEN_CACHE_CONTRACT",
    "assert_cache_identity_alignment",
    "build_frozen_token_cache_manifest",
    "load_frozen_token_cache",
    "publish_frozen_token_cache",
    "validate_frozen_token_cache_manifest",
]
