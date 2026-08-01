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


def _publish_mmap_store(
    output: Path, arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Publish immutable NPY members for bounded-resident scale loading."""

    store = output.with_name(output.name + ".arrays")
    required = {
        "identities": "identity",
        "tokens": "raw_tokens",
        "mask": "mask",
        "vectors": "vectors",
    }
    if "measurement_states" in arrays:
        required["measurement_states"] = "measurement_states"

    def descriptor(root: Path) -> dict[str, Any]:
        members = {}
        for public_name, source_name in sorted(required.items()):
            path = root / f"{public_name}.npy"
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(f"memory-map member is absent: {path}")
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            members[public_name] = {
                "filename": path.name,
                "sha256": _sha256_file(path),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "bytes": int(path.stat().st_size),
                "source_npz_member": source_name,
            }
        return {
            "contract": "hosd_npy_mmap_store_v2",
            "directory": store.name,
            "members": members,
            "total_bytes": sum(row["bytes"] for row in members.values()),
        }

    if store.exists():
        if store.is_symlink() or not store.is_dir():
            raise FileExistsError(f"unsafe memory-map store: {store}")
        return descriptor(store)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{store.name}.", dir=output.parent)
    )
    try:
        for public_name, source_name in sorted(required.items()):
            np.save(
                temporary / f"{public_name}.npy",
                np.asarray(arrays[source_name]),
                allow_pickle=False,
            )
        try:
            temporary.rename(store)
        except FileExistsError:
            pass
        return descriptor(store)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()


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
            "storage_layout": "deterministic_npz_plus_authenticated_npy_mmap_v2",
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
    mmap_store = _publish_mmap_store(output, arrays)
    manifest = with_content_hash(
        {
            "contract": INPUT_VIEW_MANIFEST_CONTRACT,
            "schema_version": 4,
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
            "storage_layout": "deterministic_npz_plus_authenticated_npy_mmap_v2",
            "mmap_store": mmap_store,
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
    validation_partition_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_split_manifest(split_manifest_path)
    source_manifest_sha256 = manifest_hash(manifest)
    validation_roles = {"val_stop", "val_design"}
    if split in validation_roles:
        if validation_partition_path is None:
            raise ValueError(
                f"{split} materialization requires the validation partition"
            )
        validation = load_retb_hashed_json(
            validation_partition_path,
            expected_contract="retb_validation_partition_manifest_v1",
        )
        roles = validation.get("roles")
        if (
            validation.get("source_manifest_sha256")
            != source_manifest_sha256
            or not isinstance(roles, Mapping)
            or set(roles) != validation_roles
        ):
            raise ValueError("validation partition lineage or roles differ")
        requested_by_role = {
            role: [JetIdentity.from_dict(row) for row in roles[role]]
            for role in sorted(validation_roles)
        }
        role_keys = {
            role: [identity.key() for identity in identities]
            for role, identities in requested_by_role.items()
        }
        if (
            any(
                len(keys) != len(set(keys))
                for keys in role_keys.values()
            )
            or set(role_keys["val_stop"]) & set(role_keys["val_design"])
            or any(
                int(validation.get("counts", {}).get(role, -1))
                != len(role_keys[role])
                for role in validation_roles
            )
        ):
            raise ValueError("validation partition populations differ")
        physical_split = "model_val"
    else:
        if validation_partition_path is not None:
            raise ValueError(
                "validation partition may accompany only val_stop/val_design"
            )
        validation = None
        requested_by_role = {}
        role_keys = {}
        physical_split = split
    view = load_offline_view(
        manifest,
        physical_split,
        data_dir=None if data_dirs is None else list(data_dirs),
        verify_label_branches=False,
    )
    if validation is not None:
        positions = {
            identity.key(): index for index, identity in enumerate(view.jet_ids)
        }
        if len(positions) != len(view.jet_ids):
            raise ValueError("model_val offline view contains duplicate identities")
        if (
            set(role_keys["val_stop"]) | set(role_keys["val_design"])
            != set(positions)
        ):
            raise ValueError(
                "validation roles do not exactly partition model_val"
            )
        requested = requested_by_role[split]
        missing_or_relabeled = [
            identity.key()
            for identity in requested
            if identity.key() not in positions
            or int(view.jet_ids[positions[identity.key()]].label)
            != int(identity.label)
        ]
        if missing_or_relabeled:
            raise ValueError(
                f"{split} identities are absent or relabeled in model_val"
            )
        indices = [positions[identity.key()] for identity in requested]
        identities = [identity.key() for identity in requested]
        raw_tokens = view.tokens[indices]
        mask = view.mask[indices]
    else:
        identities = [identity.key() for identity in view.jet_ids]
        raw_tokens = view.tokens
        mask = view.mask
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
            "split_manifest": source_manifest_sha256,
            **(
                {"validation_partition": validation["content_hash"]}
                if validation is not None
                else {}
            ),
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
            "hlt_array_content": require_sha256(
                metadata["array_content_sha256"],
                name="hlt_array_content_sha256",
            ),
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


def load_materialized_input_view(
    path: str | Path,
    *,
    expected_view_kind: str | None = None,
    expected_source: Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a HOSD view through authenticated, bounded-resident NPY members."""

    input_path = Path(path)
    manifest = load_retb_hashed_json(
        input_path.with_suffix(input_path.suffix + ".json"),
        expected_contract=INPUT_VIEW_MANIFEST_CONTRACT,
    )
    if (
        (
            expected_view_kind is not None
            and manifest.get("view_kind") != expected_view_kind
        )
        or _sha256_file(input_path) != manifest.get("npz_sha256")
        or manifest.get("storage_layout")
        != "deterministic_npz_plus_authenticated_npy_mmap_v2"
        or (
            expected_source is not None
            and manifest.get("source") != dict(expected_source)
        )
    ):
        raise ValueError("materialized HLT input-view lineage differs")
    mmap_store = manifest.get("mmap_store")
    if not isinstance(mmap_store, Mapping) or mmap_store.get("contract") != (
        "hosd_npy_mmap_store_v2"
    ):
        raise ValueError("materialized HLT memory-map contract differs")
    store = input_path.parent / str(mmap_store.get("directory", ""))
    if store.is_symlink() or not store.is_dir():
        raise ValueError("materialized HLT memory-map store is unsafe")
    arrays = {}
    expected_dtypes = {
        "identities": None,
        "tokens": np.dtype(np.float32),
        "mask": np.dtype(bool),
        "vectors": np.dtype(np.float32),
    }
    if manifest.get("contains_measurement_states") is True:
        expected_dtypes["measurement_states"] = np.dtype(np.int8)
    for name, expected_dtype in expected_dtypes.items():
        row = mmap_store.get("members", {}).get(name)
        path = store / str(row.get("filename", "")) if isinstance(row, Mapping) else store
        if (
            not isinstance(row, Mapping)
            or path.parent != store
            or path.is_symlink()
            or not path.is_file()
            or _sha256_file(path) != row.get("sha256")
        ):
            raise ValueError("materialized HLT memory-map member differs")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            list(value.shape) != row.get("shape")
            or str(value.dtype) != row.get("dtype")
            or (expected_dtype is not None and value.dtype != expected_dtype)
        ):
            raise ValueError("materialized HLT memory-map metadata differs")
        arrays[name] = value
    identity_count = int(arrays["identities"].shape[0])
    if (
        identity_count != int(manifest["identity_count"])
        or identity_order_sha256(arrays["identities"])
        != manifest["identity_order_sha256"]
        or arrays["tokens"].shape != (identity_count, 128, 14)
        or arrays["mask"].shape != (identity_count, 128)
        or arrays["vectors"].shape != (identity_count, 4, 128)
        or (
            "measurement_states" in arrays
            and arrays["measurement_states"].shape[:2]
            != arrays["mask"].shape
        )
    ):
        raise ValueError("materialized HLT memory-map population differs")
    split = str(manifest["split"])
    metadata = {
        "content_hash": manifest["content_hash"],
        "array_content_sha256": manifest["npz_sha256"],
        "logical_role": split,
        "replica_id": (
            None
            if manifest.get("replica_id") is None
            else int(manifest["replica_id"])
        ),
        "realization_policy": (
            "R_MULTI"
            if split in {"model_train", "scale_train"}
            else "R_FIXED"
        ),
        "degradation_profile_id": "D_NOMINAL",
        "source": manifest["source"],
    }
    return arrays, metadata


def load_materialized_hlt_input_view(
    path: str | Path,
    *,
    expected_source: Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load an authenticated HLT view without decompressing its full NPZ."""

    arrays, metadata = load_materialized_input_view(
        path,
        expected_view_kind="hlt_analogue",
        expected_source=expected_source,
    )
    if "measurement_states" not in arrays:
        raise ValueError("materialized HLT view lacks measurement states")
    return arrays, metadata


__all__ = [
    "materialize_hlt_input_view",
    "materialize_offline_input_view",
    "materialize_retb_offline_input_view",
    "load_materialized_input_view",
    "load_materialized_hlt_input_view",
]
