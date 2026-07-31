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
from .target_cache import load_target_cache, load_target_cache_sharded
from .target_schemas import target_declarations


NATIVE_RELATION_TARGET_CONTRACT = "hosd_native_relation_target_v2"
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
) -> tuple[str, dict[str, Any]]:
    arrays = {
        "identities": np.asarray(identities),
        "targets": targets.astype(np.float32, copy=False),
        "target_mask": target_mask.astype(bool, copy=False),
        "availability": availability,
    }

    def publish_store() -> dict[str, Any]:
        store = output.with_name(output.name + ".arrays")
        if not store.exists():
            temporary_store = Path(
                tempfile.mkdtemp(prefix=f".{store.name}.", dir=output.parent)
            )
            try:
                for name, value in sorted(arrays.items()):
                    np.save(temporary_store / f"{name}.npy", value, allow_pickle=False)
                try:
                    temporary_store.rename(store)
                except FileExistsError:
                    pass
            finally:
                if temporary_store.exists():
                    for child in temporary_store.iterdir():
                        child.unlink()
                    temporary_store.rmdir()
        if store.is_symlink() or not store.is_dir():
            raise FileExistsError("native relation memory-map store is unsafe")
        members = {}
        for name in sorted(arrays):
            path = store / f"{name}.npy"
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if value.shape != arrays[name].shape or value.dtype != arrays[name].dtype:
                raise FileExistsError("native relation memory-map metadata differs")
            members[name] = {
                "filename": path.name,
                "sha256": _sha256_file(path),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        return {
            "contract": "hosd_native_relation_npy_store_v1",
            "directory": store.name,
            "members": members,
        }

    def validate_existing() -> tuple[str, dict[str, Any]]:
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
        return _sha256_file(output), publish_store()

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
        return _sha256_file(output), publish_store()
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


def _publish_joined_sharded_native(
    *,
    loaded: Mapping[str, tuple[Any, str]],
    output: Path,
) -> tuple[str, dict[str, Any]]:
    ids = native_relation_target_ids()
    first = loaded[ids[0]][0]
    count = len(first.identities)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.join.", dir=output.parent
    ) as directory:
        root = Path(directory)
        values = np.lib.format.open_memmap(
            root / "targets.npy", mode="w+", dtype=np.float32, shape=(count, 545)
        )
        masks = np.lib.format.open_memmap(
            root / "target_mask.npy", mode="w+", dtype=bool, shape=(count, 545)
        )
        availability = np.lib.format.open_memmap(
            root / "availability.npy", mode="w+", dtype=np.float32, shape=(count, 7)
        )
        chunk_size = 4096
        for start in range(0, count, chunk_size):
            stop = min(count, start + chunk_size)
            value_parts = [
                np.asarray(loaded[key][0].values[loaded[key][1]][start:stop])
                for key in ids
            ]
            mask_parts = [
                np.asarray(loaded[key][0].masks[loaded[key][1]][start:stop])
                for key in ids
            ]
            values[start:stop] = np.concatenate(value_parts, axis=1)
            masks[start:stop] = np.concatenate(mask_parts, axis=1)
            availability[start:stop] = np.stack(
                [part.any(axis=1) for part in mask_parts], axis=1
            ).astype(np.float32)
        values.flush()
        masks.flush()
        availability.flush()
        if not np.isfinite(values).all():
            raise ValueError("native relation concatenation is nonfinite")
        published = _publish_native_npz(
            output,
            identities=first.identities,
            targets=values,
            target_mask=masks,
            availability=availability,
        )
        del values, masks, availability
        return published


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
    cache = load_target_cache_sharded(root, cache_spec=spec)
    ids = native_relation_target_ids()
    if not set(ids).issubset(cache.values):
        raise ValueError("HLT analogue cache lacks native relation families")
    output = Path(output_path)
    digest, mmap_store = _publish_joined_sharded_native(
        loaded={key: (cache, key) for key in ids}, output=output
    )
    artifact = with_content_hash(
        {
            "contract": NATIVE_RELATION_TARGET_CONTRACT,
            "schema_version": 2,
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
            "storage_layout": "compressed_npz_plus_authenticated_npy_mmap_v1",
            "mmap_store": mmap_store,
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
        cache = load_target_cache_sharded(root, cache_spec=spec)
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
    digest, mmap_store = _publish_joined_sharded_native(
        loaded=loaded, output=output
    )
    artifact = with_content_hash(
        {
            "contract": NATIVE_RELATION_TARGET_CONTRACT,
            "schema_version": 2,
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
            "storage_layout": "compressed_npz_plus_authenticated_npy_mmap_v1",
            "mmap_store": mmap_store,
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
