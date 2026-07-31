"""Deterministic, label-blind HOSD target-cache publication.

The cache stores only canonical identities, target values, and applicability
masks.  Labels are intentionally absent from every API in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import zipfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .contracts import (
    TARGET_CACHE_MANIFEST_CONTRACT,
    TARGET_CACHE_SPEC_CONTRACT,
    TARGET_SHARD_CONTRACT,
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_bytes,
    write_immutable_json,
)
from .extractors import TargetBatch


PRELOCK_SPLITS = frozenset({"model_train", "val_stop", "val_design"})
SCALE_SPLITS = frozenset({"scale_train"})
PRIVILEGED_POSTLOCK_SPLITS = frozenset({"stack_val", "final_test"})
PERSISTABLE_KINDS = frozenset(
    {"canonical_offline", "hlt_analogue", "teacher_output", "residual", "control"}
)
STREAM_STORAGE_MODE = "stream_same_view_node_or_pair"
PERSIST_STORAGE_MODE = "persist_compact_jet_target"


def _require_hashes(values: Mapping[str, Any], *, name: str) -> dict[str, str]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    return {
        str(key): require_sha256(value, name=f"{name}.{key}")
        for key, value in sorted(values.items())
    }


def _canonical_identities(identities: Sequence[str]) -> tuple[tuple[str, ...], np.ndarray]:
    raw = tuple(str(value) for value in identities)
    if any(not value for value in raw):
        raise ValueError("identities must be non-empty strings")
    if len(set(raw)) != len(raw):
        raise ValueError("target-cache identities contain duplicates")
    order = np.asarray(sorted(range(len(raw)), key=lambda index: raw[index]), dtype=np.int64)
    return tuple(raw[int(index)] for index in order), order


def identity_order_sha256(identities: Sequence[str]) -> str:
    digest = hashlib.sha256(b"hosd_target_identity_order_v1\0")
    for value in identities:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _encode_string_table(values: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    payload = bytearray()
    for value in values:
        payload.extend(value.encode("utf-8"))
        offsets.append(len(payload))
    return np.asarray(offsets, dtype=np.int64), np.frombuffer(bytes(payload), dtype=np.uint8)


def _decode_string_table(offsets: np.ndarray, payload: np.ndarray) -> tuple[str, ...]:
    if offsets.ndim != 1 or payload.ndim != 1 or offsets.dtype != np.int64:
        raise ValueError("invalid identity string table")
    if offsets.size == 0 or int(offsets[0]) != 0 or int(offsets[-1]) != payload.size:
        raise ValueError("invalid identity string-table boundaries")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("identity string-table offsets are not monotonic")
    raw = payload.astype(np.uint8, copy=False).tobytes()
    return tuple(
        raw[int(offsets[index]) : int(offsets[index + 1])].decode("utf-8")
        for index in range(offsets.size - 1)
    )


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Serialize NumPy arrays with fixed ordering, metadata, and timestamps."""

    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            if "/" in name or "\\" in name or not name:
                raise ValueError(f"unsafe NPZ array name {name!r}")
            array = np.ascontiguousarray(np.asarray(arrays[name]))
            encoded = io.BytesIO()
            np.lib.format.write_array(encoded, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            archive.writestr(info, encoded.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    return output.getvalue()


def _load_npz_bytes(encoded: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def build_target_cache_spec(
    *,
    cache_id: str,
    split: str,
    artifact_kind: str,
    identities: Sequence[str],
    target_components: Mapping[str, Sequence[str]],
    parent_hashes: Mapping[str, str],
    source: Mapping[str, Any],
    shard_size: int = 2048,
    dtype: str = "float32",
    storage_modes: Mapping[str, str] | None = None,
    hlt_replica_id: str | None = None,
    access_authorization_hash: str | None = None,
) -> dict[str, Any]:
    """Freeze shard boundaries and complete lineage before values are inspected."""

    if artifact_kind not in PERSISTABLE_KINDS:
        raise ValueError(f"unknown cache artifact kind {artifact_kind!r}")
    if split not in PRELOCK_SPLITS | SCALE_SPLITS | PRIVILEGED_POSTLOCK_SPLITS:
        raise ValueError(f"unsupported target-cache split {split!r}")
    if split in PRIVILEGED_POSTLOCK_SPLITS:
        require_sha256(
            access_authorization_hash, name="access_authorization_hash"
        )
    elif access_authorization_hash is not None:
        require_sha256(
            access_authorization_hash, name="access_authorization_hash"
        )
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if dtype != "float32":
        raise ValueError("canonical target caches must use float32")
    canonical, _ = _canonical_identities(identities)
    if not canonical:
        raise ValueError("target-cache identity population must not be empty")
    components = {
        str(target_id): [str(component) for component in names]
        for target_id, names in sorted(target_components.items())
    }
    if not components or any(not names for names in components.values()):
        raise ValueError("every cached target requires a non-empty component schema")
    modes = {
        target_id: (
            dict(storage_modes or {}).get(target_id, PERSIST_STORAGE_MODE)
        )
        for target_id in components
    }
    if any(mode not in {PERSIST_STORAGE_MODE, STREAM_STORAGE_MODE} for mode in modes.values()):
        raise ValueError("unknown target storage mode")
    if artifact_kind in {"hlt_analogue", "residual"} and not hlt_replica_id:
        raise ValueError(f"{artifact_kind} cache must bind an HLT replica")
    if artifact_kind == "canonical_offline" and hlt_replica_id is not None:
        raise ValueError("canonical offline targets are replica independent")
    authorization_parent = (
        "locked_hosd_finalists" if split == "stack_val"
        else "final_test_execution_lock" if split == "final_test"
        else None
    )
    resolved_parents = _require_hashes(parent_hashes, name="parent_hashes")
    if authorization_parent is not None:
        if resolved_parents.get(authorization_parent) != access_authorization_hash:
            raise ValueError(
                f"{split} target cache must bind {authorization_parent} as its "
                "exact access authorization"
            )
    return with_content_hash(
        {
            "contract": TARGET_CACHE_SPEC_CONTRACT,
            "schema_version": 1,
            "cache_id": str(cache_id),
            "split": split,
            "artifact_kind": artifact_kind,
            "event_count": len(canonical),
            "canonical_identity_order_sha256": identity_order_sha256(canonical),
            "target_order": sorted(components),
            "target_components": components,
            "storage_modes": modes,
            "persisted_target_ids": sorted(
                target_id
                for target_id, mode in modes.items()
                if mode == PERSIST_STORAGE_MODE
            ),
            "streamed_target_ids": sorted(
                target_id
                for target_id, mode in modes.items()
                if mode == STREAM_STORAGE_MODE
            ),
            "shard_size": int(shard_size),
            "shard_count": (
                (len(canonical) + shard_size - 1) // shard_size
                if canonical
                else 0
            ),
            "dtype": dtype,
            "hlt_replica_id": hlt_replica_id,
            "parent_hashes": resolved_parents,
            "access_authorization_sha256": access_authorization_hash,
            "label_access_for_extraction": False,
            "labels_stored": False,
            "constituent_matching_used": False,
            "runtime_deployable_input_permitted": False,
            "source": dict(source),
        }
    )


@dataclass(frozen=True)
class LoadedTargetCache:
    identities: tuple[str, ...]
    values: Mapping[str, np.ndarray]
    masks: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]


def _coerce_batch(
    result: Mapping[str, TargetBatch],
    *,
    expected_targets: Sequence[str],
    expected_components: Mapping[str, Sequence[str]],
    count: int,
) -> dict[str, TargetBatch]:
    if set(result) != set(expected_targets):
        raise ValueError("target generator returned incorrect target coverage")
    converted: dict[str, TargetBatch] = {}
    for target_id in expected_targets:
        batch = result[target_id]
        if batch.target_id != target_id:
            raise ValueError("target generator returned a mismatched target ID")
        if tuple(batch.component_names) != tuple(expected_components[target_id]):
            raise ValueError(f"{target_id} component schema differs from cache spec")
        values = np.asarray(batch.values.detach().cpu().numpy(), dtype=np.float32)
        masks = np.asarray(batch.loss_mask.detach().cpu().numpy(), dtype=bool)
        if values.shape != masks.shape or values.ndim != 2 or values.shape[0] != count:
            raise ValueError(f"{target_id} must be a [batch,component] target")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{target_id} contains non-finite values")
        if np.any(values[~masks] != 0):
            raise ValueError(f"{target_id} stores nonzero unavailable values")
        converted[target_id] = batch
    return converted


def publish_target_cache_shard(
    output_dir: str | Path,
    *,
    cache_spec: Mapping[str, Any],
    canonical_identities: Sequence[str],
    canonical_to_source: np.ndarray,
    shard_index: int,
    generator: Callable[[np.ndarray], Mapping[str, TargetBatch]],
) -> dict[str, Any]:
    validate_content_hash(cache_spec, expected_contract=TARGET_CACHE_SPEC_CONTRACT)
    if not 0 <= shard_index < int(cache_spec["shard_count"]):
        raise IndexError("target-cache shard index is out of range")
    if identity_order_sha256(canonical_identities) != cache_spec[
        "canonical_identity_order_sha256"
    ]:
        raise ValueError("identity population differs from cache specification")
    start = shard_index * int(cache_spec["shard_size"])
    stop = min(start + int(cache_spec["shard_size"]), len(canonical_identities))
    root = Path(output_dir)
    npz_path = root / "shards" / f"shard_{shard_index:06d}.npz"
    json_path = root / "shards" / f"shard_{shard_index:06d}.json"
    if json_path.exists():
        from .contracts import load_hashed_json

        existing = load_hashed_json(json_path, expected_contract=TARGET_SHARD_CONTRACT)
        if (
            existing.get("cache_spec_sha256") != cache_spec["content_hash"]
            or existing.get("shard_index") != shard_index
            or not npz_path.is_file()
            or hashlib.sha256(npz_path.read_bytes()).hexdigest()
            != existing.get("npz_sha256")
        ):
            raise ValueError(f"stale or corrupt reusable shard {shard_index}")
        return existing
    if npz_path.exists():
        raise ValueError(f"unattested partial shard exists at {npz_path}")
    target_ids = list(cache_spec["persisted_target_ids"])
    source_indices = canonical_to_source[start:stop]
    batches = _coerce_batch(
        generator(source_indices),
        expected_targets=target_ids,
        expected_components=cache_spec["target_components"],
        count=stop - start,
    )
    offsets, identity_bytes = _encode_string_table(canonical_identities[start:stop])
    arrays: dict[str, np.ndarray] = {
        "identity_indices": np.arange(start, stop, dtype=np.int64),
        "identity_offsets": offsets,
        "identity_bytes": identity_bytes,
    }
    target_records = []
    for position, target_id in enumerate(target_ids):
        batch = batches[target_id]
        values = np.asarray(batch.values.detach().cpu().numpy(), dtype=np.float32)
        masks = np.asarray(batch.loss_mask.detach().cpu().numpy(), dtype=bool)
        value_key = f"target_{position:04d}_values"
        mask_key = f"target_{position:04d}_mask"
        arrays[value_key] = values
        arrays[mask_key] = masks
        target_records.append(
            {
                "target_id": target_id,
                "value_array": value_key,
                "mask_array": mask_key,
                "shape": list(values.shape),
                "valid_component_count": int(masks.sum()),
                "value_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
                "mask_sha256": hashlib.sha256(masks.tobytes(order="C")).hexdigest(),
            }
        )
    encoded = deterministic_npz_bytes(arrays)
    npz_result = write_immutable_bytes(npz_path, encoded)
    manifest = with_content_hash(
        {
            "contract": TARGET_SHARD_CONTRACT,
            "schema_version": 1,
            "cache_spec_sha256": cache_spec["content_hash"],
            "shard_index": shard_index,
            "identity_start": start,
            "identity_stop": stop,
            "identity_count": stop - start,
            "identity_order_sha256": identity_order_sha256(
                canonical_identities[start:stop]
            ),
            "npz_filename": npz_path.name,
            "npz_sha256": npz_result["file_sha256"],
            "npz_bytes": len(encoded),
            "targets": target_records,
            "label_access_for_extraction": False,
            "labels_stored": False,
            "runtime_deployable_input_permitted": False,
            "source": dict(cache_spec["source"]),
        }
    )
    write_immutable_json(json_path, manifest)
    return manifest


def validate_target_cache(
    output_dir: str | Path,
    *,
    cache_spec: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(cache_spec, expected_contract=TARGET_CACHE_SPEC_CONTRACT)
    root = Path(output_dir)
    shard_records = []
    cursor = 0
    all_identities: list[str] = []
    for shard_index in range(int(cache_spec["shard_count"])):
        from .contracts import load_hashed_json

        record = load_hashed_json(
            root / "shards" / f"shard_{shard_index:06d}.json",
            expected_contract=TARGET_SHARD_CONTRACT,
        )
        if record["cache_spec_sha256"] != cache_spec["content_hash"]:
            raise ValueError("target shard is bound to a different cache specification")
        if record["identity_start"] != cursor or record["shard_index"] != shard_index:
            raise ValueError("target shard coverage is noncanonical")
        npz_path = root / "shards" / record["npz_filename"]
        encoded = npz_path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != record["npz_sha256"]:
            raise ValueError("target shard bytes do not match the attestation")
        arrays = _load_npz_bytes(encoded)
        identities = _decode_string_table(
            arrays["identity_offsets"], arrays["identity_bytes"]
        )
        if identity_order_sha256(identities) != record["identity_order_sha256"]:
            raise ValueError("target shard identities differ from their attestation")
        indices = arrays["identity_indices"]
        if not np.array_equal(
            indices, np.arange(record["identity_start"], record["identity_stop"])
        ):
            raise ValueError("target shard identity indices are not exact")
        for target in record["targets"]:
            values = arrays[target["value_array"]]
            masks = arrays[target["mask_array"]]
            if values.dtype != np.float32 or masks.dtype != np.bool_:
                raise ValueError("target shard dtype contract was violated")
            if values.shape != masks.shape or list(values.shape) != target["shape"]:
                raise ValueError("target shard shape contract was violated")
            if hashlib.sha256(values.tobytes(order="C")).hexdigest() != target[
                "value_sha256"
            ]:
                raise ValueError("target value bytes differ from attestation")
            if hashlib.sha256(masks.tobytes(order="C")).hexdigest() != target[
                "mask_sha256"
            ]:
                raise ValueError("target mask bytes differ from attestation")
            if np.any(values[~masks] != 0) or not np.all(np.isfinite(values)):
                raise ValueError("target values violate missingness/finiteness contract")
        cursor = int(record["identity_stop"])
        all_identities.extend(identities)
        shard_records.append(record)
    if cursor != int(cache_spec["event_count"]):
        raise ValueError("target-cache shard coverage is incomplete")
    if all_identities != sorted(all_identities) or len(set(all_identities)) != len(
        all_identities
    ):
        raise ValueError("target-cache identity coverage is not unique and sorted")
    if identity_order_sha256(all_identities) != cache_spec[
        "canonical_identity_order_sha256"
    ]:
        raise ValueError("target-cache identity population differs from specification")
    return with_content_hash(
        {
            "contract": TARGET_CACHE_MANIFEST_CONTRACT,
            "schema_version": 1,
            "cache_spec_sha256": cache_spec["content_hash"],
            "cache_id": cache_spec["cache_id"],
            "split": cache_spec["split"],
            "artifact_kind": cache_spec["artifact_kind"],
            "hlt_replica_id": cache_spec["hlt_replica_id"],
            "parent_hashes": dict(cache_spec["parent_hashes"]),
            "access_authorization_sha256": cache_spec[
                "access_authorization_sha256"
            ],
            "event_count": cursor,
            "canonical_identity_order_sha256": cache_spec[
                "canonical_identity_order_sha256"
            ],
            "target_order": list(cache_spec["target_order"]),
            "target_components": dict(cache_spec["target_components"]),
            "persisted_target_ids": list(cache_spec["persisted_target_ids"]),
            "streamed_target_ids": list(cache_spec["streamed_target_ids"]),
            "shards": [
                {
                    "shard_index": row["shard_index"],
                    "manifest_sha256": row["content_hash"],
                    "npz_sha256": row["npz_sha256"],
                    "identity_start": row["identity_start"],
                    "identity_stop": row["identity_stop"],
                }
                for row in shard_records
            ],
            "complete_exact_identity_coverage": True,
            "duplicate_identity_count": 0,
            "label_access_for_extraction": False,
            "labels_stored": False,
            "source": dict(cache_spec["source"]),
        }
    )


def publish_target_cache(
    output_dir: str | Path,
    *,
    cache_spec: Mapping[str, Any],
    identities: Sequence[str],
    generator: Callable[[np.ndarray], Mapping[str, TargetBatch]],
    shard_order: Iterable[int] | None = None,
) -> dict[str, Any]:
    canonical, order = _canonical_identities(identities)
    if identity_order_sha256(canonical) != cache_spec[
        "canonical_identity_order_sha256"
    ]:
        raise ValueError("identities differ from the frozen cache specification")
    expected = tuple(range(int(cache_spec["shard_count"])))
    execution_order = tuple(expected if shard_order is None else shard_order)
    if sorted(execution_order) != list(expected):
        raise ValueError("shard_order must contain every shard exactly once")
    for shard_index in execution_order:
        publish_target_cache_shard(
            output_dir,
            cache_spec=cache_spec,
            canonical_identities=canonical,
            canonical_to_source=order,
            shard_index=int(shard_index),
            generator=generator,
        )
    manifest = validate_target_cache(output_dir, cache_spec=cache_spec)
    write_immutable_json(Path(output_dir) / "target_manifest.json", manifest)
    write_immutable_json(Path(output_dir) / "cache_spec.json", dict(cache_spec))
    return manifest


def load_target_cache(
    output_dir: str | Path,
    *,
    cache_spec: Mapping[str, Any],
) -> LoadedTargetCache:
    manifest = validate_target_cache(output_dir, cache_spec=cache_spec)
    identities: list[str] = []
    values: dict[str, list[np.ndarray]] = {
        target_id: [] for target_id in cache_spec["persisted_target_ids"]
    }
    masks: dict[str, list[np.ndarray]] = {
        target_id: [] for target_id in cache_spec["persisted_target_ids"]
    }
    root = Path(output_dir)
    for shard in manifest["shards"]:
        record_path = root / "shards" / f"shard_{shard['shard_index']:06d}.json"
        from .contracts import load_hashed_json

        record = load_hashed_json(record_path, expected_contract=TARGET_SHARD_CONTRACT)
        arrays = _load_npz_bytes(
            (root / "shards" / record["npz_filename"]).read_bytes()
        )
        identities.extend(
            _decode_string_table(arrays["identity_offsets"], arrays["identity_bytes"])
        )
        for target in record["targets"]:
            target_id = target["target_id"]
            values[target_id].append(arrays[target["value_array"]])
            masks[target_id].append(arrays[target["mask_array"]])
    return LoadedTargetCache(
        identities=tuple(identities),
        values={
            target_id: np.concatenate(parts, axis=0)
            for target_id, parts in values.items()
        },
        masks={
            target_id: np.concatenate(parts, axis=0)
            for target_id, parts in masks.items()
        },
        manifest=manifest,
    )


__all__ = [
    "LoadedTargetCache",
    "PERSIST_STORAGE_MODE",
    "PRELOCK_SPLITS",
    "SCALE_SPLITS",
    "STREAM_STORAGE_MODE",
    "build_target_cache_spec",
    "deterministic_npz_bytes",
    "identity_order_sha256",
    "load_target_cache",
    "publish_target_cache",
    "publish_target_cache_shard",
    "validate_target_cache",
]
