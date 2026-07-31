"""Materialize authenticated label-blind offline or HLT target-builder views."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from jetclass_fresh.jetclass_data import (
    JetIdentity,
    load_offline_view,
    load_split_manifest,
    manifest_hash,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (
    HLT_V3_ARRAY_FILENAME,
    load_hlt_v3_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    load_hashed_json as load_retb_hashed_json,
)

from .contracts import (
    INPUT_VIEW_MANIFEST_CONTRACT,
    require_sha256,
    with_content_hash,
    write_immutable_json,
)
from .target_cache import identity_order_sha256


def _vectors(tokens: np.ndarray) -> np.ndarray:
    pt = tokens[:, :, 0]
    eta = tokens[:, :, 1]
    phi = tokens[:, :, 2]
    energy = tokens[:, :, 3]
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return np.stack((px, py, pz, energy), axis=1).astype(np.float32)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_deterministic_npz(
    output: Path, arrays: Mapping[str, np.ndarray]
) -> str:
    """Stream one deterministic NPZ to an immutable destination.

    Scale views can exceed tens of gigabytes uncompressed.  Building a
    ``bytes`` object for each NPY member and then another for the whole ZIP
    doubles that footprint.  ZipFile's writable member streams preserve the
    exact fixed metadata contract without retaining encoded members in RAM.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            for name in sorted(arrays):
                if "/" in name or "\\" in name or not name:
                    raise ValueError(f"unsafe NPZ array name {name!r}")
                array = np.ascontiguousarray(np.asarray(arrays[name]))
                info = zipfile.ZipInfo(
                    f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                info.create_system = 3
                with archive.open(info, "w", force_zip64=True) as member:
                    np.lib.format.write_array(
                        member, array, allow_pickle=False
                    )
        digest = _sha256_file(temporary)
        if output.exists():
            if output.is_symlink() or not output.is_file():
                raise FileExistsError(
                    f"immutable destination is unsafe: {output}"
                )
            if _sha256_file(output) != digest:
                raise FileExistsError(
                    f"refusing to overwrite different input view: {output}"
                )
            return digest
        try:
            os.link(temporary, output)
        except FileExistsError:
            if (
                output.is_symlink()
                or not output.is_file()
                or _sha256_file(output) != digest
            ):
                raise FileExistsError(
                    f"immutable input-view destination appeared: {output}"
                )
        return digest
    finally:
        if temporary.exists():
            temporary.unlink()


def _publish(
    *,
    output: Path,
    identities: Sequence[str],
    raw_tokens: np.ndarray,
    mask: np.ndarray,
    view_kind: str,
    split: str,
    replica_id: int | None,
    parent_hashes: Mapping[str, str],
    source: Mapping[str, Any],
    measurement_states: np.ndarray | None = None,
) -> dict[str, Any]:
    ids = tuple(str(value) for value in identities)
    normalized_parents = {
        key: require_sha256(value, name=f"parent.{key}")
        for key, value in sorted(parent_hashes.items())
    }
    manifest_path = output.with_suffix(output.suffix + ".json")
    if output.is_file() and manifest_path.is_file():
        existing = load_retb_hashed_json(
            manifest_path, expected_contract=INPUT_VIEW_MANIFEST_CONTRACT
        )
        expected_existing = {
            "source": dict(source),
            "view_kind": view_kind,
            "split": split,
            "replica_id": replica_id,
            "identity_count": len(ids),
            "identity_order_sha256": identity_order_sha256(ids),
            "parent_hashes": normalized_parents,
            "contains_labels": False,
            "contains_degradation_construction_indices": False,
            "contains_measurement_states": measurement_states is not None,
        }
        if (
            all(existing.get(key) == value for key, value in expected_existing.items())
            and _sha256_file(output) == existing.get("npz_sha256")
        ):
            return existing
        raise FileExistsError(
            f"reusable input-view lineage differs: {output}"
        )
    tokens = np.asarray(raw_tokens, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    if (
        not ids
        or len(ids) != len(set(ids))
        or tokens.shape != (len(ids), 128, 14)
        or valid.shape != (len(ids), 128)
        or not np.isfinite(tokens).all()
    ):
        raise ValueError("label-blind input-view population differs")
    tokens = np.where(valid[:, :, None], tokens, 0).astype(np.float32)
    arrays: dict[str, np.ndarray] = {
        "identity": np.asarray(ids, dtype=f"<U{max(map(len, ids))}"),
        "mask": valid,
        "raw_tokens": tokens,
        "vectors": _vectors(tokens),
    }
    if measurement_states is not None:
        states = np.asarray(measurement_states)
        if states.dtype != np.int8 or states.shape[:2] != valid.shape:
            raise ValueError("HLT measurement-state population differs")
        arrays["measurement_states"] = states
    digest = _publish_deterministic_npz(output, arrays)
    manifest = with_content_hash(
        {
            "contract": INPUT_VIEW_MANIFEST_CONTRACT,
            "schema_version": 2,
            "source": dict(source),
            "view_kind": view_kind,
            "split": split,
            "replica_id": replica_id,
            "identity_count": len(ids),
            "identity_order_sha256": identity_order_sha256(ids),
            "npz_path": str(output.resolve()),
            "npz_sha256": digest,
            "parent_hashes": normalized_parents,
            "contains_labels": False,
            "contains_degradation_construction_indices": False,
            "contains_measurement_states": measurement_states is not None,
        }
    )
    write_immutable_json(output.with_suffix(output.suffix + ".json"), manifest)
    return manifest


def materialize_offline_input_view(
    *,
    split_manifest_path: str | Path,
    split: str,
    data_dirs: Sequence[str | Path] | None,
    output: str | Path,
    parent_hashes: Mapping[str, str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = load_split_manifest(split_manifest_path)
    view = load_offline_view(
        manifest,
        split,
        data_dir=None if data_dirs is None else list(data_dirs),
        verify_label_branches=False,
    )
    identities = [identity.key() for identity in view.jet_ids]
    return _publish(
        output=Path(output),
        identities=identities,
        raw_tokens=view.tokens,
        mask=view.mask,
        view_kind="canonical_offline",
        split=split,
        replica_id=None,
        parent_hashes={
            **parent_hashes,
            "split_manifest": manifest_hash(manifest),
        },
        source=source,
    )


def materialize_hlt_input_view(
    *,
    hlt_cache_path: str | Path,
    split: str,
    replica_id: int,
    output: str | Path,
    parent_hashes: Mapping[str, str],
    source: Mapping[str, Any],
    expected_identity_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    cache_root = Path(hlt_cache_path)
    arrays, metadata = load_hlt_v3_cache(cache_root)
    expected_policy = (
        "R_MULTI" if split in {"model_train", "scale_train"} else "R_FIXED"
    )
    if (
        metadata.get("logical_role") != split
        or int(metadata.get("replica_id", -1)) != int(replica_id)
        or metadata.get("realization_policy") != expected_policy
        or metadata.get("degradation_profile_id") != "D_NOMINAL"
        or (
            expected_identity_manifest_sha256 is not None
            and metadata.get("identity_manifest_sha256")
            != require_sha256(
                expected_identity_manifest_sha256,
                name="expected_identity_manifest_sha256",
            )
        )
        or (
            metadata.get("source") is not None
            and metadata.get("source") != dict(source)
        )
    ):
        raise ValueError("HLT input-view source coordinate differs")
    identities = tuple(str(value) for value in arrays["identities"].tolist())
    array_path = (
        cache_root / HLT_V3_ARRAY_FILENAME
        if cache_root.is_dir()
        else cache_root
    )
    if not array_path.is_file() or array_path.is_symlink():
        raise FileNotFoundError(
            f"HLT cache array is absent or unsafe: {array_path}"
        )
    cache_sha256 = _sha256_file(array_path)
    return _publish(
        output=Path(output),
        identities=identities,
        raw_tokens=np.asarray(arrays["tokens"], dtype=np.float32),
        mask=np.asarray(arrays["mask"], dtype=bool),
        view_kind="hlt_analogue",
        split=split,
        replica_id=int(replica_id),
        parent_hashes={
            **parent_hashes,
            "hlt_cache": cache_sha256,
            "hlt_cache_metadata": require_sha256(
                metadata["content_hash"], name="hlt_cache_metadata"
            ),
        },
        source=source,
        measurement_states=np.asarray(
            arrays["measurement_states"], dtype=np.int8
        ),
    )


def materialize_retb_offline_input_view(
    *,
    offline_cache_dir: str | Path,
    split: str,
    output: str | Path,
    parent_hashes: Mapping[str, str],
    source: Mapping[str, Any],
    expected_identity_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Convert one authenticated RETB offline cache to the HOSD view schema."""

    cache = Path(offline_cache_dir)
    metadata_path = cache / "offline_input_manifest.json"
    metadata = load_retb_hashed_json(
        metadata_path, expected_contract="retb_offline_input_cache_v1"
    )
    npz_path = cache / str(metadata["npz_filename"])
    if (
        metadata.get("logical_role") != split
        or not npz_path.is_file()
        or npz_path.is_symlink()
        or (
            metadata.get("source") is not None
            and metadata.get("source") != dict(source)
        )
        or (
            expected_identity_manifest_sha256 is not None
            and metadata.get("identity_manifest_sha256")
            != require_sha256(
                expected_identity_manifest_sha256,
                name="expected_identity_manifest_sha256",
            )
        )
    ):
        raise ValueError("RETB offline input-view source coordinate differs")
    npz_sha256 = _sha256_file(npz_path)
    if npz_sha256 != metadata.get("npz_sha256"):
        raise ValueError("RETB offline input-view bytes differ")
    with np.load(npz_path, allow_pickle=False) as payload:
        if not {"identities", "tokens", "mask"}.issubset(payload.files):
            raise ValueError("RETB offline input cache fields differ")
        identities = tuple(str(value) for value in payload["identities"].tolist())
        raw_tokens = np.asarray(payload["tokens"], dtype=np.float32)
        mask = np.asarray(payload["mask"], dtype=bool)
    return _publish(
        output=Path(output),
        identities=identities,
        raw_tokens=raw_tokens,
        mask=mask,
        view_kind="canonical_offline",
        split=split,
        replica_id=None,
        parent_hashes={
            **parent_hashes,
            "retb_offline_cache": npz_sha256,
            "retb_offline_cache_metadata": require_sha256(
                metadata["content_hash"], name="retb_offline_cache_metadata"
            ),
            "identity_manifest": require_sha256(
                metadata["identity_manifest_sha256"],
                name="retb_offline_identity_manifest",
            ),
        },
        source=source,
    )


def load_materialized_hlt_input_view(
    path: str | Path,
    *,
    expected_source: Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a HOSD HLT view with its adjacent byte/coordinate attestation."""

    input_path = Path(path)
    manifest = load_retb_hashed_json(
        input_path.with_suffix(input_path.suffix + ".json"),
        expected_contract=INPUT_VIEW_MANIFEST_CONTRACT,
    )
    if (
        manifest.get("view_kind") != "hlt_analogue"
        or manifest.get("contains_measurement_states") is not True
        or _sha256_file(input_path) != manifest.get("npz_sha256")
        or (
            expected_source is not None
            and manifest.get("source") != dict(expected_source)
        )
    ):
        raise ValueError("materialized HLT input-view lineage differs")
    with np.load(input_path, allow_pickle=False) as payload:
        required = {
            "identity",
            "raw_tokens",
            "mask",
            "measurement_states",
        }
        if not required.issubset(payload.files):
            raise ValueError("materialized HLT input-view fields differ")
        arrays = {
            "identities": np.asarray(payload["identity"]),
            "tokens": np.asarray(payload["raw_tokens"], dtype=np.float32),
            "mask": np.asarray(payload["mask"], dtype=bool),
            "measurement_states": np.asarray(
                payload["measurement_states"], dtype=np.int8
            ),
        }
    split = str(manifest["split"])
    metadata = {
        "content_hash": manifest["content_hash"],
        "array_content_sha256": manifest["npz_sha256"],
        "logical_role": split,
        "replica_id": int(manifest["replica_id"]),
        "realization_policy": (
            "R_MULTI"
            if split in {"model_train", "scale_train"}
            else "R_FIXED"
        ),
        "degradation_profile_id": "D_NOMINAL",
        "source": manifest["source"],
    }
    return arrays, metadata


__all__ = [
    "materialize_hlt_input_view",
    "materialize_offline_input_view",
    "materialize_retb_offline_input_view",
    "load_materialized_hlt_input_view",
]
