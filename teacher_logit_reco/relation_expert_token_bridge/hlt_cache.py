"""Authenticated, non-interchangeable RETB HLT-v3 cache artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_cache import hash_arrays

from .contracts import (
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .hlt_v3 import (
    DEGRADATION_PROFILES,
    HLT_V3_PROFILE_NAME,
    HLT_V3_PROFILE_VERSION,
    build_hlt_v3_view,
    measurement_validity_states,
    validate_hlt_v3_profile_contract,
)
from .replicas import DOMAIN_SEEDS, REALIZATION_POLICIES


HLT_V3_CACHE_CONTRACT = "retb_hlt_v3_cache_v1"
HLT_V3_ARRAY_FILENAME = "hlt_v3_arrays.npz"
HLT_V3_METADATA_FILENAME = "hlt_v3_metadata.json"


def identity_order_hash(identities: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"retb_hlt_v3_identity_order_v1\0")
    for identity in identities:
        digest.update(str(identity).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def cache_array_content_hash(
    *,
    tokens: np.ndarray,
    mask: np.ndarray,
    measurement_states: np.ndarray,
    identities: Sequence[str],
) -> str:
    return canonical_sha256(
        {
            "numeric_arrays_sha256": hash_arrays(
                {
                    "tokens": np.asarray(tokens, dtype=np.float32),
                    "mask": np.asarray(mask, dtype=bool),
                    "measurement_states": np.asarray(
                        measurement_states, dtype=np.int8
                    ),
                }
            ),
            "identity_order_sha256": identity_order_hash(identities),
        }
    )


def build_hlt_v3_cache(
    tokens: np.ndarray,
    mask: np.ndarray,
    *,
    canonical_identities: Sequence[str],
    logical_role: str,
    replica_id: int,
    realization_policy: str,
    profile_id: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    if np.asarray(tokens).dtype != np.float32:
        raise ValueError("HLT-v3 cache source tokens must be canonical float32")
    if np.asarray(mask).dtype != np.bool_:
        raise ValueError("HLT-v3 cache source mask must be canonical bool")
    output, output_mask, states, diagnostics = build_hlt_v3_view(
        tokens,
        mask,
        canonical_identities=canonical_identities,
        logical_role=logical_role,
        replica_id=replica_id,
        realization_policy=realization_policy,
        profile_id=profile_id,
    )
    max_identity_length = max((len(str(value)) for value in canonical_identities), default=1)
    arrays = {
        "tokens": np.asarray(output, dtype=np.float32),
        "mask": np.asarray(output_mask, dtype=bool),
        "measurement_states": np.asarray(states, dtype=np.int8),
        "identities": np.asarray(
            [str(value) for value in canonical_identities],
            dtype=f"<U{max_identity_length}",
        ),
    }
    return arrays, diagnostics


def build_hlt_v3_cache_metadata(
    *,
    arrays: Mapping[str, np.ndarray],
    diagnostics: Sequence[Mapping[str, Any]],
    logical_role: str,
    replica_id: int,
    realization_policy: str,
    degradation_profile_id: str,
    profile_contract: Mapping[str, Any],
    split_manifest_sha256: str,
    identity_manifest_sha256: str,
    raw_input_sha256: str,
) -> dict[str, Any]:
    profile_sha = validate_hlt_v3_profile_contract(profile_contract)
    identities = [str(value) for value in np.asarray(arrays["identities"]).tolist()]
    array_hash = cache_array_content_hash(
        tokens=arrays["tokens"],
        mask=arrays["mask"],
        measurement_states=arrays["measurement_states"],
        identities=identities,
    )
    aggregate_counts: dict[str, int] = {}
    for row in diagnostics:
        for name, value in row.get("mechanism_counts", {}).items():
            aggregate_counts[name] = aggregate_counts.get(name, 0) + int(value)
    return with_content_hash(
        {
            "contract": HLT_V3_CACHE_CONTRACT,
            "schema_version": 1,
            "profile_name": HLT_V3_PROFILE_NAME,
            "profile_version": HLT_V3_PROFILE_VERSION,
            "profile_contract_sha256": profile_sha,
            "logical_role": logical_role,
            "replica_id": int(replica_id),
            "realization_policy": realization_policy,
            "degradation_profile_id": degradation_profile_id,
            "split_manifest_sha256": require_sha256(
                split_manifest_sha256, name="split_manifest_sha256"
            ),
            "identity_manifest_sha256": require_sha256(
                identity_manifest_sha256, name="identity_manifest_sha256"
            ),
            "raw_input_sha256": require_sha256(
                raw_input_sha256, name="raw_input_sha256"
            ),
            "array_content_sha256": array_hash,
            "identity_order_sha256": identity_order_hash(identities),
            "diagnostics_sha256": canonical_sha256(list(diagnostics)),
            "aggregate_mechanism_counts": aggregate_counts,
            "shape": list(np.asarray(arrays["tokens"]).shape),
            "dtype": "float32",
            "mask_dtype": "bool",
            "measurement_state_dtype": "int8",
            "identity_aligned": True,
            "constituent_matching_fields_present": False,
            "offline_arrays_persisted": False,
        }
    )


def validate_hlt_v3_cache(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    expected_profile_contract_sha256: str | None = None,
    expected_logical_role: str | None = None,
    expected_replica_id: int | None = None,
) -> str:
    digest = validate_content_hash(
        metadata, expected_contract=HLT_V3_CACHE_CONTRACT
    )
    if metadata.get("profile_name") != HLT_V3_PROFILE_NAME:
        raise ValueError("cache profile is not HLT-v3 track-dominant")
    if metadata.get("profile_version") != HLT_V3_PROFILE_VERSION:
        raise ValueError("cache profile version is not HLT-v3 v1")
    if int(metadata.get("schema_version", -1)) != 1:
        raise ValueError("cache metadata schema version differs")
    if metadata.get("logical_role") not in DOMAIN_SEEDS:
        raise ValueError("cache logical role lies outside the locked roles")
    if metadata.get("realization_policy") not in REALIZATION_POLICIES:
        raise ValueError("cache realization policy differs")
    profile_id = metadata.get("degradation_profile_id")
    if (
        profile_id not in DEGRADATION_PROFILES
        or DEGRADATION_PROFILES[str(profile_id)].legacy_profile is not None
    ):
        raise ValueError("cache degradation profile is not an HLT-v3 mode")
    replica_id = int(metadata.get("replica_id", -1))
    if replica_id not in range(4):
        raise ValueError("cache replica lies outside 0..3")
    for name in (
        "profile_contract_sha256",
        "split_manifest_sha256",
        "identity_manifest_sha256",
        "raw_input_sha256",
        "array_content_sha256",
        "identity_order_sha256",
        "diagnostics_sha256",
    ):
        require_sha256(metadata.get(name), name=f"cache.{name}")
    if expected_profile_contract_sha256 is not None and metadata.get(
        "profile_contract_sha256"
    ) != require_sha256(
        expected_profile_contract_sha256,
        name="expected_profile_contract_sha256",
    ):
        raise ValueError("cache HLT-v3 profile parent differs")
    if expected_logical_role is not None and metadata.get(
        "logical_role"
    ) != expected_logical_role:
        raise ValueError("cache logical role differs")
    if expected_replica_id is not None and int(
        metadata.get("replica_id", -1)
    ) != int(expected_replica_id):
        raise ValueError("cache replica differs")
    identities = [str(value) for value in np.asarray(arrays["identities"]).tolist()]
    if len(identities) != len(set(identities)):
        raise ValueError("cache identities are not unique")
    if len(identities) != len(np.asarray(arrays["tokens"])):
        raise ValueError("cache identity count differs from token count")
    if np.asarray(arrays["tokens"]).dtype != np.float32:
        raise ValueError("cache token dtype is not float32")
    if np.asarray(arrays["mask"]).dtype != np.bool_:
        raise ValueError("cache mask dtype is not bool")
    if np.asarray(arrays["measurement_states"]).dtype != np.int8:
        raise ValueError("cache measurement-state dtype is not int8")
    actual = cache_array_content_hash(
        tokens=arrays["tokens"],
        mask=arrays["mask"],
        measurement_states=arrays["measurement_states"],
        identities=identities,
    )
    if actual != metadata.get("array_content_sha256"):
        raise ValueError("cache array content hash differs")
    if tuple(np.asarray(arrays["mask"]).shape) != tuple(
        np.asarray(arrays["tokens"]).shape[:2]
    ):
        raise ValueError("cache mask shape differs from token shape")
    if not bool(np.isfinite(np.asarray(arrays["tokens"])).all()):
        raise ValueError("cache tokens contain nonfinite values")
    expected_states = measurement_validity_states(
        np.asarray(arrays["tokens"]), np.asarray(arrays["mask"])
    )
    if not np.array_equal(expected_states, arrays["measurement_states"]):
        raise ValueError("cache measurement states differ from raw fields")
    return digest


def load_hlt_v3_cache(
    cache_dir: str | Path,
    *,
    expected_profile_contract_sha256: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = Path(cache_dir)
    with np.load(root / HLT_V3_ARRAY_FILENAME, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    metadata = json.loads(
        (root / HLT_V3_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    validate_hlt_v3_cache(
        arrays,
        metadata,
        expected_profile_contract_sha256=expected_profile_contract_sha256,
    )
    return arrays, metadata


def _publish_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    if path.exists():
        with np.load(path, allow_pickle=False) as payload:
            existing = {name: np.asarray(payload[name]) for name in payload.files}
        existing_hash = cache_array_content_hash(
            tokens=existing["tokens"],
            mask=existing["mask"],
            measurement_states=existing["measurement_states"],
            identities=[str(value) for value in existing["identities"].tolist()],
        )
        new_hash = cache_array_content_hash(
            tokens=arrays["tokens"],
            mask=arrays["mask"],
            measurement_states=arrays["measurement_states"],
            identities=[str(value) for value in arrays["identities"].tolist()],
        )
        if existing_hash != new_hash:
            raise FileExistsError("refusing to overwrite a different HLT-v3 cache")
        return "already_present"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.npz", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError("HLT-v3 cache appeared during publication") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return "published"


def publish_hlt_v3_cache(
    cache_dir: str | Path,
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    validate_hlt_v3_cache(arrays, metadata)
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    array_status = _publish_npz(root / HLT_V3_ARRAY_FILENAME, arrays)
    metadata_status = write_immutable_json(
        root / HLT_V3_METADATA_FILENAME, metadata
    )
    return {
        "cache_dir": str(root.resolve()),
        "array_status": array_status,
        "metadata_status": metadata_status["status"],
        "metadata_sha256": metadata["content_hash"],
        "array_content_sha256": metadata["array_content_sha256"],
    }


__all__ = [
    "HLT_V3_ARRAY_FILENAME",
    "HLT_V3_CACHE_CONTRACT",
    "HLT_V3_METADATA_FILENAME",
    "build_hlt_v3_cache",
    "build_hlt_v3_cache_metadata",
    "cache_array_content_hash",
    "identity_order_hash",
    "load_hlt_v3_cache",
    "publish_hlt_v3_cache",
    "validate_hlt_v3_cache",
]
