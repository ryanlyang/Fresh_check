"""Lossless compact codec for adaptive-binary hierarchy target shards."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_cache import hash_arrays


ABPH_COMPACT_TARGET_CODEC_CONTRACT = (
    "adaptive_binary_pseudooffline_compact_target_codec_v1"
)
ABPH_COMPACT_TARGET_CODEC_NAME = "compact_lossless_v1"
ABPH_LEGACY_TARGET_CODEC_NAME = "legacy_npz_v1"
ABPH_TARGET_STORAGE_CODECS = (
    ABPH_LEGACY_TARGET_CODEC_NAME,
    ABPH_COMPACT_TARGET_CODEC_NAME,
)
ABPH_COMPACT_CODEC_MANIFEST_KEY = "__compact_codec_manifest__"


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _shape(array: np.ndarray) -> list[int]:
    return [int(value) for value in array.shape]


def _stored_key(name: str) -> str:
    return f"payload__{name}"


def byte_shuffle(array: np.ndarray) -> np.ndarray:
    """Group equal-significance bytes while preserving every source bit."""

    source = np.ascontiguousarray(array)
    if source.dtype.kind != "f":
        raise TypeError("byte shuffle is restricted to floating arrays")
    byte_width = int(source.dtype.itemsize)
    matrix = source.view(np.uint8).reshape(-1, byte_width)
    return np.ascontiguousarray(matrix.T.reshape(-1), dtype=np.uint8)


def byte_unshuffle(
    shuffled: np.ndarray,
    *,
    dtype: str | np.dtype[Any],
    shape: Sequence[int],
) -> np.ndarray:
    resolved_dtype = np.dtype(dtype)
    if resolved_dtype.kind != "f":
        raise TypeError("byte unshuffle is restricted to floating arrays")
    resolved_shape = tuple(int(value) for value in shape)
    expected_items = int(np.prod(resolved_shape, dtype=np.int64))
    byte_width = int(resolved_dtype.itemsize)
    source = np.ascontiguousarray(shuffled, dtype=np.uint8).reshape(-1)
    if source.size != expected_items * byte_width:
        raise ValueError("byte-shuffled payload length does not match declared shape")
    matrix = source.reshape(byte_width, expected_items).T.copy()
    return matrix.reshape(-1).view(resolved_dtype).reshape(resolved_shape)


def pack_boolean_array(array: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.ascontiguousarray(array, dtype=bool)
    if source.ndim == 0:
        source = source.reshape(1)
    axis = source.ndim - 1
    packed = np.packbits(source, axis=axis, bitorder="little")
    return np.ascontiguousarray(packed), {
        "codec": "packbits_little",
        "dtype": "bool",
        "shape": _shape(source),
        "packed_axis": axis,
        "unpacked_axis_length": int(source.shape[axis]),
    }


def unpack_boolean_array(array: np.ndarray, spec: Mapping[str, Any]) -> np.ndarray:
    shape = tuple(int(value) for value in spec["shape"])
    axis = int(spec["packed_axis"])
    if not shape or not 0 <= axis < len(shape):
        raise ValueError("packed boolean axis is invalid")
    expected_packed_shape = list(shape)
    expected_packed_shape[axis] = (shape[axis] + 7) // 8
    packed = np.asarray(array, dtype=np.uint8)
    if tuple(packed.shape) != tuple(expected_packed_shape):
        raise ValueError("packed boolean shape does not match codec manifest")
    unpacked = np.unpackbits(
        packed,
        axis=axis,
        count=int(spec["unpacked_axis_length"]),
        bitorder="little",
    )
    return np.ascontiguousarray(unpacked.reshape(shape), dtype=bool)


def smallest_signed_dtype(minimum: int, maximum: int) -> np.dtype[Any]:
    if minimum > maximum:
        raise ValueError("integer range is inverted")
    for dtype in (np.int8, np.int16, np.int32, np.int64):
        info = np.iinfo(dtype)
        if minimum >= info.min and maximum <= info.max:
            return np.dtype(dtype)
    raise OverflowError(f"integer range [{minimum}, {maximum}] exceeds int64")


def narrow_signed_integer_array(
    array: np.ndarray,
    *,
    dtype: str | np.dtype[Any] | None = None,
) -> np.ndarray:
    source = np.asarray(array)
    if source.dtype.kind not in "iub":
        raise TypeError("integer narrowing requires an integer or boolean array")
    minimum = int(source.min(initial=0))
    maximum = int(source.max(initial=0))
    resolved = smallest_signed_dtype(minimum, maximum) if dtype is None else np.dtype(dtype)
    if resolved.kind != "i":
        raise TypeError("compact integer storage must use a signed dtype")
    bounds = np.iinfo(resolved)
    if minimum < bounds.min or maximum > bounds.max:
        raise OverflowError(
            f"integer range [{minimum}, {maximum}] does not fit {resolved}"
        )
    return np.ascontiguousarray(source, dtype=resolved)


def _logical_array_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "dtype": np.asarray(array).dtype.str,
            "shape": _shape(np.asarray(array)),
        }
        for name, array in sorted(arrays.items())
    }


def encode_compact_target_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    omitted_identity_keys: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Encode one complete logical shard without lossy dtype conversions."""

    omitted = tuple(str(value) for value in omitted_identity_keys)
    missing = sorted(set(omitted).difference(arrays))
    if missing:
        raise KeyError(f"omitted compact identity keys are absent: {missing}")
    logical = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
    encoded: dict[str, np.ndarray] = {}
    specs: dict[str, Any] = {}
    for name in sorted(logical):
        if name in omitted:
            continue
        source = logical[name]
        key = _stored_key(name)
        if source.dtype.kind == "b":
            stored, spec = pack_boolean_array(source)
        elif source.dtype.kind == "f":
            if source.dtype != np.dtype("float32"):
                raise ValueError(
                    f"compact target float {name} must remain float32, got {source.dtype}"
                )
            stored = byte_shuffle(source)
            spec = {
                "codec": "byte_shuffle",
                "dtype": source.dtype.str,
                "shape": _shape(source),
                "byte_width": int(source.dtype.itemsize),
            }
        elif source.dtype.kind in "iu":
            stored = narrow_signed_integer_array(source)
            spec = {
                "codec": "narrow_signed",
                "dtype": source.dtype.str,
                "stored_dtype": stored.dtype.str,
                "shape": _shape(source),
                "minimum": int(source.min(initial=0)),
                "maximum": int(source.max(initial=0)),
            }
        else:
            raise TypeError(
                f"compact target array {name} has unsupported dtype {source.dtype}"
            )
        encoded[key] = np.ascontiguousarray(stored)
        specs[name] = {**spec, "stored_key": key}

    manifest: dict[str, Any] = {
        "contract": ABPH_COMPACT_TARGET_CODEC_CONTRACT,
        "codec": ABPH_COMPACT_TARGET_CODEC_NAME,
        "logical_array_manifest": _logical_array_manifest(logical),
        "logical_array_keys": sorted(logical),
        "omitted_identity_keys": list(omitted),
        "array_codecs": specs,
        "logical_content_hash": hash_arrays(logical),
    }
    manifest["manifest_hash"] = _json_hash(manifest)
    encoded[ABPH_COMPACT_CODEC_MANIFEST_KEY] = np.frombuffer(
        _canonical_json(manifest), dtype=np.uint8
    ).copy()
    manifest["encoded_content_hash"] = hash_arrays(encoded)
    return encoded, manifest


def compact_manifest_from_arrays(
    encoded_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    if ABPH_COMPACT_CODEC_MANIFEST_KEY not in encoded_arrays:
        raise ValueError("compact shard lacks its codec manifest")
    raw = np.ascontiguousarray(
        encoded_arrays[ABPH_COMPACT_CODEC_MANIFEST_KEY], dtype=np.uint8
    ).tobytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("compact codec manifest must be a mapping")
    if payload.get("contract") != ABPH_COMPACT_TARGET_CODEC_CONTRACT:
        raise ValueError("compact target codec contract mismatch")
    if payload.get("codec") != ABPH_COMPACT_TARGET_CODEC_NAME:
        raise ValueError("compact target codec name mismatch")
    saved_hash = payload.get("manifest_hash")
    hash_payload = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if saved_hash != _json_hash(hash_payload):
        raise ValueError("compact target codec manifest hash mismatch")
    return payload


def decode_compact_target_arrays(
    encoded_arrays: Mapping[str, np.ndarray],
    *,
    expected_encoded_content_hash: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    encoded = {
        name: np.ascontiguousarray(value) for name, value in encoded_arrays.items()
    }
    actual_encoded_hash = hash_arrays(encoded)
    if (
        expected_encoded_content_hash is not None
        and actual_encoded_hash != expected_encoded_content_hash
    ):
        raise ValueError("compact target encoded content hash mismatch")
    manifest = compact_manifest_from_arrays(encoded)
    specs = manifest.get("array_codecs")
    logical_manifest = manifest.get("logical_array_manifest")
    if not isinstance(specs, dict) or not isinstance(logical_manifest, dict):
        raise ValueError("compact target codec manifest is incomplete")
    arrays: dict[str, np.ndarray] = {}
    for name, spec_value in sorted(specs.items()):
        if not isinstance(spec_value, dict):
            raise ValueError(f"compact target codec spec for {name} is malformed")
        spec = spec_value
        key = str(spec["stored_key"])
        if key not in encoded:
            raise KeyError(f"compact target payload {key} is missing")
        stored = encoded[key]
        codec = str(spec["codec"])
        if codec == "packbits_little":
            restored = unpack_boolean_array(stored, spec)
        elif codec == "byte_shuffle":
            restored = byte_unshuffle(
                stored,
                dtype=str(spec["dtype"]),
                shape=spec["shape"],
            )
        elif codec == "narrow_signed":
            minimum = int(stored.min(initial=0))
            maximum = int(stored.max(initial=0))
            if minimum < int(spec["minimum"]) or maximum > int(spec["maximum"]):
                raise ValueError(f"compact integer payload {name} exceeds declared range")
            restored = np.ascontiguousarray(stored, dtype=np.dtype(str(spec["dtype"])))
            if _shape(restored) != [int(value) for value in spec["shape"]]:
                raise ValueError(f"compact integer payload {name} shape mismatch")
        else:
            raise ValueError(f"unknown compact target codec {codec!r} for {name}")
        expected = logical_manifest.get(name)
        if not isinstance(expected, dict):
            raise ValueError(f"logical manifest for {name} is missing")
        if restored.dtype.str != str(expected["dtype"]):
            raise ValueError(f"decoded target dtype mismatch for {name}")
        if _shape(restored) != [int(value) for value in expected["shape"]]:
            raise ValueError(f"decoded target shape mismatch for {name}")
        arrays[name] = np.ascontiguousarray(restored)
    expected_keys = set(manifest["logical_array_keys"])
    omitted = set(manifest["omitted_identity_keys"])
    if set(arrays) != expected_keys.difference(omitted):
        raise ValueError("decoded target keys differ from compact logical manifest")
    return arrays, manifest


__all__ = [
    "ABPH_COMPACT_CODEC_MANIFEST_KEY",
    "ABPH_COMPACT_TARGET_CODEC_CONTRACT",
    "ABPH_COMPACT_TARGET_CODEC_NAME",
    "ABPH_LEGACY_TARGET_CODEC_NAME",
    "ABPH_TARGET_STORAGE_CODECS",
    "byte_shuffle",
    "byte_unshuffle",
    "compact_manifest_from_arrays",
    "decode_compact_target_arrays",
    "encode_compact_target_arrays",
    "narrow_signed_integer_array",
    "pack_boolean_array",
    "smallest_signed_dtype",
    "unpack_boolean_array",
]
