"""Exact seven-family HLT-native relation auxiliary targets."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .contracts import (
    TARGET_CACHE_SPEC_CONTRACT,
    load_hashed_json,
    require_sha256,
    with_content_hash,
    write_immutable_json,
)
from .target_cache import load_target_cache
from .target_schemas import target_declarations


NATIVE_RELATION_TARGET_CONTRACT = "hosd_native_relation_target_v1"
NATIVE_RELATION_WAVE_CONTRACT = "hosd_native_relation_target_wave_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_native_npz(
    output: Path,
    *,
    identities: tuple[str, ...],
    targets: np.ndarray,
    target_mask: np.ndarray,
    availability: np.ndarray,
) -> str:
    arrays = {
        "identities": np.asarray(identities),
        "targets": targets.astype(np.float32, copy=False),
        "target_mask": target_mask.astype(bool, copy=False),
        "availability": availability,
    }

    def validate_existing() -> str:
        if output.is_symlink() or not output.is_file():
            raise FileExistsError("native relation destination is unsafe")
        with np.load(output, allow_pickle=False) as payload:
            if set(payload.files) != set(arrays) or any(
                not np.array_equal(payload[name], value)
                for name, value in arrays.items()
            ):
                raise FileExistsError(
                    "reusable native relation target differs"
                )
        return _sha256_file(output)

    if output.exists():
        return validate_existing()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".npz", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        np.savez_compressed(temporary, **arrays)
        try:
            os.link(temporary, output)
        except FileExistsError:
            return validate_existing()
        return _sha256_file(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def native_relation_target_ids() -> tuple[str, ...]:
    rows = tuple(
        row.target_id
        for row in target_declarations()
        if row.target_id.startswith("T_OFFLINE_RELATION_")
    )
    if len(rows) != 7:
        raise AssertionError("native relation family coverage differs")
    return rows


def materialize_native_relation_target(
    *,
    target_cache_root: str | Path,
    output_path: str | Path,
    campaign_spec_sha256: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(target_cache_root)
    spec = load_hashed_json(
        root / "cache_spec.json", expected_contract=TARGET_CACHE_SPEC_CONTRACT
    )
    cache = load_target_cache(root, cache_spec=spec)
    ids = native_relation_target_ids()
    if not set(ids).issubset(cache.values):
        raise ValueError("HLT analogue cache lacks native relation families")
    values = np.concatenate([cache.values[key] for key in ids], axis=1)
    masks = np.concatenate([cache.masks[key] for key in ids], axis=1)
    availability = np.stack(
        [cache.masks[key].any(axis=1) for key in ids], axis=1
    ).astype(np.float32)
    if (
        values.shape != (len(cache.identities), 545)
        or masks.shape != values.shape
        or availability.shape != (len(cache.identities), 7)
        or not np.isfinite(values).all()
    ):
        raise ValueError("native relation concatenation shape/finiteness differs")
    output = Path(output_path)
    digest = _publish_native_npz(
        output,
        identities=cache.identities,
        targets=values,
        target_mask=masks,
        availability=availability,
    )
    artifact = with_content_hash(
        {
            "contract": NATIVE_RELATION_TARGET_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "target_cache_manifest_sha256": cache.manifest["content_hash"],
            "split": cache.manifest["split"],
            "hlt_replica_id": cache.manifest["hlt_replica_id"],
            "target_ids": list(ids),
            "component_dimensions": {
                key: int(cache.values[key].shape[1]) for key in ids
            },
            "concatenated_dimension": 545,
            "availability_dimension": 7,
            "event_count": len(cache.identities),
            "npz_filename": output.name,
            "npz_sha256": digest,
            "same_hlt_view_as_student": True,
            "offline_information_consumed": False,
        }
    )
    write_immutable_json(output.with_suffix(".manifest.json"), artifact)
    return artifact


def materialize_native_relation_target_from_family_caches(
    *,
    target_cache_roots: Mapping[str, str | Path],
    output_path: str | Path,
    campaign_spec_sha256: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Join the seven independently refit scale caches into one native target."""
    ids = native_relation_target_ids()
    if set(target_cache_roots) != set(ids):
        raise ValueError("native relation family-cache coverage differs")
    output = Path(output_path)
    parent_manifests = {
        target_id: load_hashed_json(
            Path(target_cache_roots[target_id]) / "target_manifest.json"
        )
        for target_id in ids
    }
    parent_hashes = {
        key: parent_manifests[key]["content_hash"] for key in ids
    }
    existing_manifest_path = output.with_suffix(".manifest.json")
    if output.is_file() and existing_manifest_path.is_file():
        existing = load_hashed_json(
            existing_manifest_path,
            expected_contract=NATIVE_RELATION_TARGET_CONTRACT,
        )
        if (
            existing.get("source") == dict(source)
            and existing.get("campaign_spec_sha256")
            == require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            )
            and existing.get("target_cache_manifest_hashes")
            == parent_hashes
            and existing.get("target_ids") == list(ids)
            and _sha256_file(output) == existing.get("npz_sha256")
        ):
            return existing
        raise FileExistsError(
            "reusable native relation target lineage differs"
        )
    loaded = {}
    for target_id in ids:
        root = Path(target_cache_roots[target_id])
        spec = load_hashed_json(
            root / "cache_spec.json",
            expected_contract=TARGET_CACHE_SPEC_CONTRACT,
        )
        cache = load_target_cache(root, cache_spec=spec)
        if len(cache.values) != 1:
            raise ValueError("scale native relation cache is not family-pure")
        coordinate = next(iter(cache.values))
        loaded[target_id] = (cache, coordinate)
    first = loaded[ids[0]][0]
    if any(
        cache.identities != first.identities
        or cache.manifest["split"] != first.manifest["split"]
        or cache.manifest["hlt_replica_id"]
        != first.manifest["hlt_replica_id"]
        for cache, _ in loaded.values()
    ):
        raise ValueError("scale native relation family identities/views differ")
    values = np.concatenate(
        [loaded[key][0].values[loaded[key][1]] for key in ids], axis=1
    )
    masks = np.concatenate(
        [loaded[key][0].masks[loaded[key][1]] for key in ids], axis=1
    )
    availability = np.stack(
        [
            loaded[key][0].masks[loaded[key][1]].any(axis=1)
            for key in ids
        ],
        axis=1,
    ).astype(np.float32)
    if (
        values.shape != (len(first.identities), 545)
        or masks.shape != values.shape
        or availability.shape != (len(first.identities), 7)
        or not np.isfinite(values).all()
    ):
        raise ValueError("scale native relation concatenation differs")
    digest = _publish_native_npz(
        output,
        identities=first.identities,
        targets=values,
        target_mask=masks,
        availability=availability,
    )
    artifact = with_content_hash(
        {
            "contract": NATIVE_RELATION_TARGET_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "target_cache_manifest_sha256": hashlib.sha256(
                "".join(
                    loaded[key][0].manifest["content_hash"] for key in ids
                ).encode("ascii")
            ).hexdigest(),
            "target_cache_manifest_hashes": {
                key: loaded[key][0].manifest["content_hash"] for key in ids
            },
            "split": first.manifest["split"],
            "hlt_replica_id": first.manifest["hlt_replica_id"],
            "target_ids": list(ids),
            "component_dimensions": {
                key: int(
                    loaded[key][0].values[loaded[key][1]].shape[1]
                )
                for key in ids
            },
            "concatenated_dimension": 545,
            "availability_dimension": 7,
            "event_count": len(first.identities),
            "npz_filename": output.name,
            "npz_sha256": digest,
            "same_hlt_view_as_student": True,
            "offline_information_consumed": False,
        }
    )
    write_immutable_json(output.with_suffix(".manifest.json"), artifact)
    return artifact


__all__ = [
    "NATIVE_RELATION_TARGET_CONTRACT",
    "NATIVE_RELATION_WAVE_CONTRACT",
    "materialize_native_relation_target",
    "materialize_native_relation_target_from_family_caches",
    "native_relation_target_ids",
]
