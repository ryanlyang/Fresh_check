"""Single-node RAM streaming for prediction-anchored bridge fields.

Persistent compressed NPZ inputs are opened once by allocation rank zero,
verified, and unpacked into non-evictable RAM-backed ``.npy`` shards.  Only
derived data enter the bounded LRU.  The mutable allocation ledger is protected
by a cross-process lock so packed GPU ranks cannot oversubscribe node RAM.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence
import uuid

import numpy as np

from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash
from jetclass_fresh.jetclass_data import JetIdentity
from teacher_logit_reco.adaptive_binary_pseudooffline.ram_workspace import probe_ram_capacity

from .bridge import (
    BRIDGE_CHANNEL_PHYSICAL45,
    apply_bridge_control,
    build_matched_wrong_event_map,
    virtual_bridge,
)
from .bridge_contracts import canonical_sha256, sha256_file, with_content_hash
from .targets import DEFAULT_LOCAL_RESIDUAL_RADII, compute_local_particle_residual_fields


PREDICTION_ANCHORED_RAM_LEDGER_CONTRACT = "prediction_anchored_allocation_ram_ledger_v1"
PREDICTION_ANCHORED_RAW_STAGE_CONTRACT = "prediction_anchored_raw_npz_stage_v1"
PREDICTION_ANCHORED_RAM_TELEMETRY_CONTRACT = "prediction_anchored_ram_telemetry_v1"
PREDICTION_ANCHORED_R0_REGISTRATION_CONTRACT = "prediction_anchored_frozen_r0_registration_v1"
DEFAULT_RAW_SHARD_SIZE = 8192
MANDATORY_HEADROOM_FRACTION = 0.20
_RAW_KEYS = ("tokens", "mask", "labels", "jet_file_indices", "jet_entries")
_DTYPES = {
    "tokens": np.dtype(np.float32),
    "mask": np.dtype(bool),
    "labels": np.dtype(np.int64),
    "jet_file_indices": np.dtype(np.int32),
    "jet_entries": np.dtype(np.int64),
}


def require_single_node(env: Mapping[str, str] | None = None) -> int:
    values = os.environ if env is None else env
    raw = values.get("SLURM_NNODES", values.get("SLURM_JOB_NUM_NODES", "1"))
    try:
        nodes = int(str(raw))
    except ValueError as exc:
        raise ValueError(f"invalid Slurm node count {raw!r}") from exc
    if nodes != 1:
        raise RuntimeError(
            f"prediction-anchored packed jobs require exactly one node; allocation declares {nodes}"
        )
    return nodes


def deterministic_rank_range(total: int, rank: int, world_size: int) -> tuple[int, int]:
    size = int(total)
    worker = int(rank)
    workers = int(world_size)
    if size < 0 or workers <= 0 or worker < 0 or worker >= workers:
        raise ValueError("invalid total/rank/world_size")
    quotient, remainder = divmod(size, workers)
    start = worker * quotient + min(worker, remainder)
    stop = start + quotient + (1 if worker < remainder else 0)
    return start, stop


_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _locked_file(path: Path) -> Iterator[None]:
    """Cross-platform advisory lock, combined with an in-process thread lock."""

    path.parent.mkdir(parents=True, exist_ok=True)
    local = _thread_lock(path)
    with local:
        with path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                        break
                    except OSError:
                        time.sleep(0.01)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class AllocationRamLedger:
    """Allocation-wide byte reservations with mandatory 20% headroom."""

    def __init__(
        self,
        ram_root: str | Path,
        *,
        allocation_id: str,
        env: Mapping[str, str] | None = None,
        capacity_bytes: int | None = None,
        allow_unverified_test_root: bool = False,
    ) -> None:
        require_single_node(env)
        if not str(allocation_id).strip():
            raise ValueError("allocation_id is required")
        self.allocation_id = str(allocation_id)
        base = Path(ram_root).resolve()
        base.mkdir(parents=True, exist_ok=True)
        if capacity_bytes is None:
            probe = probe_ram_capacity(base, env=env)
            capacity = int(probe.governing_limit_bytes)
            filesystem = str(probe.filesystem_type)
        else:
            if not allow_unverified_test_root:
                raise ValueError("explicit capacity is allowed only for an unverified test root")
            capacity = int(capacity_bytes)
            filesystem = "test_injected_ram"
        if capacity <= 0:
            raise ValueError("RAM capacity must be positive")
        self.root = base / f"prediction_anchored_bridge_{self.allocation_id}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.root / "ledger.lock"
        self.manifest_path = self.root / "allocation_ledger.json"
        self.owner_path = self.root / "owner.json"
        with _locked_file(self.lock_path):
            if self.manifest_path.exists():
                state = self._read_unlocked()
                if state.get("allocation_id") != self.allocation_id:
                    raise PermissionError("RAM ledger ownership mismatch")
                if int(state.get("capacity_bytes", -1)) != capacity:
                    raise ValueError("RAM capacity changed within one allocation")
            else:
                state = {
                    "contract": PREDICTION_ANCHORED_RAM_LEDGER_CONTRACT,
                    "allocation_id": self.allocation_id,
                    "capacity_bytes": capacity,
                    "reservation_limit_bytes": int(capacity * (1.0 - MANDATORY_HEADROOM_FRACTION)),
                    "mandatory_headroom_fraction": MANDATORY_HEADROOM_FRACTION,
                    "filesystem_type": filesystem,
                    "reservations": {},
                    "peak_reserved_bytes": 0,
                    "raw_stage_finalized": False,
                }
                _atomic_json(self.manifest_path, state)
                _atomic_json(
                    self.owner_path,
                    {"contract": PREDICTION_ANCHORED_RAM_LEDGER_CONTRACT, "allocation_id": self.allocation_id},
                )

    def _read_unlocked(self) -> dict[str, Any]:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("contract") != PREDICTION_ANCHORED_RAM_LEDGER_CONTRACT:
            raise ValueError("RAM ledger contract mismatch")
        return payload

    @staticmethod
    def _used(state: Mapping[str, Any]) -> int:
        return sum(
            int(item.get("measured_bytes", item.get("expected_bytes", 0)))
            for item in dict(state.get("reservations", {})).values()
            if item.get("state") in {"reserved", "committed"}
        )

    @property
    def reservation_limit_bytes(self) -> int:
        with _locked_file(self.lock_path):
            return int(self._read_unlocked()["reservation_limit_bytes"])

    @property
    def reserved_bytes(self) -> int:
        with _locked_file(self.lock_path):
            return self._used(self._read_unlocked())

    def reserve(self, *, owner: str, role: str, expected_bytes: int, category: str) -> str:
        requested = int(expected_bytes)
        if requested < 0 or category not in {"raw", "derived", "other"}:
            raise ValueError("invalid RAM reservation")
        reservation_id = uuid.uuid4().hex
        with _locked_file(self.lock_path):
            state = self._read_unlocked()
            if category == "raw" and bool(state.get("raw_stage_finalized")):
                raise RuntimeError("raw RAM stage is already finalized and immutable")
            after = self._used(state) + requested
            limit = int(state["reservation_limit_bytes"])
            if after > limit:
                raise MemoryError(
                    f"allocation RAM reservation would use {after} bytes; 20%-headroom limit is {limit}"
                )
            state["reservations"][reservation_id] = {
                "owner": str(owner),
                "role": str(role),
                "category": str(category),
                "expected_bytes": requested,
                "state": "reserved",
            }
            state["peak_reserved_bytes"] = max(int(state.get("peak_reserved_bytes", 0)), after)
            _atomic_json(self.manifest_path, state)
        return reservation_id

    def commit(self, reservation_id: str, *, measured_bytes: int) -> None:
        measured = int(measured_bytes)
        if measured < 0:
            raise ValueError("measured_bytes cannot be negative")
        with _locked_file(self.lock_path):
            state = self._read_unlocked()
            item = state["reservations"].get(str(reservation_id))
            if item is None or item.get("state") not in {"reserved", "committed"}:
                raise KeyError(f"unknown active RAM reservation {reservation_id}")
            previous = int(item.get("measured_bytes", item.get("expected_bytes", 0)))
            after = self._used(state) - previous + measured
            if after > int(state["reservation_limit_bytes"]):
                raise MemoryError("measured RAM allocation violates mandatory headroom")
            item["measured_bytes"] = measured
            item["state"] = "committed"
            state["peak_reserved_bytes"] = max(int(state.get("peak_reserved_bytes", 0)), after)
            _atomic_json(self.manifest_path, state)

    def finalize_raw_stage(self) -> None:
        with _locked_file(self.lock_path):
            state = self._read_unlocked()
            if not any(
                item.get("category") == "raw" and item.get("state") == "committed"
                for item in state["reservations"].values()
            ):
                raise RuntimeError("cannot finalize an empty raw stage")
            state["raw_stage_finalized"] = True
            _atomic_json(self.manifest_path, state)

    def release(self, reservation_id: str, *, allocation_cleanup: bool = False) -> None:
        with _locked_file(self.lock_path):
            state = self._read_unlocked()
            item = state["reservations"].get(str(reservation_id))
            if item is None:
                return
            if item.get("category") == "raw" and bool(state.get("raw_stage_finalized")) and not allocation_cleanup:
                raise RuntimeError("raw staged shards are non-evictable for the allocation lifetime")
            item["state"] = "released"
            _atomic_json(self.manifest_path, state)

    def snapshot(self) -> dict[str, Any]:
        with _locked_file(self.lock_path):
            state = self._read_unlocked()
        state["reserved_bytes"] = self._used(state)
        return state

    def cleanup(self) -> None:
        """Remove only this ledger's explicitly owned job-scoped tree."""

        owner = json.loads(self.owner_path.read_text(encoding="utf-8"))
        if owner.get("allocation_id") != self.allocation_id:
            raise PermissionError("refusing to clean a RAM tree owned by another allocation")
        expected_name = f"prediction_anchored_bridge_{self.allocation_id}"
        if self.root.name != expected_name or self.root.parent == self.root:
            raise RuntimeError("unsafe allocation cleanup target")
        shutil.rmtree(self.root)


def _event_identities(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> tuple[JetIdentity, ...]:
    files = [str(value) for value in metadata.get("jet_files", [])]
    indices = np.asarray(arrays["jet_file_indices"], dtype=np.int64)
    entries = np.asarray(arrays["jet_entries"], dtype=np.int64)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    if not files or np.any(indices < 0) or np.any(indices >= len(files)):
        raise ValueError("source metadata has invalid jet_files/file indices")
    identities = tuple(
        JetIdentity(file=files[int(file_index)], entry=int(entry), label=int(label))
        for file_index, entry, label in zip(indices, entries, labels)
    )
    identity_keys = [(item.file, int(item.entry), int(item.label)) for item in identities]
    if len(set(identity_keys)) != len(identity_keys):
        raise ValueError("duplicate event IDs are forbidden in staged sources")
    return identities


@dataclass(frozen=True)
class StagedSource:
    name: str
    root: Path
    manifest: Mapping[str, Any]

    @property
    def n_events(self) -> int:
        return int(self.manifest["n_events"])

    @property
    def shard_size(self) -> int:
        return int(self.manifest["shard_size"])

    def _shards(self, name: str) -> Sequence[Mapping[str, Any]]:
        return self.manifest["arrays"][str(name)]["shards"]

    def read_indices(self, indices: Sequence[int] | np.ndarray, names: Sequence[str] = _RAW_KEYS) -> dict[str, np.ndarray]:
        requested = np.asarray(indices, dtype=np.int64)
        if requested.ndim != 1 or np.any(requested < 0) or np.any(requested >= self.n_events):
            raise IndexError("staged source indices are out of range")
        output: dict[str, np.ndarray] = {}
        for name in names:
            spec = self.manifest["arrays"].get(str(name))
            if spec is None:
                raise KeyError(f"staged source has no array {name!r}")
            shape = (requested.size, *tuple(int(v) for v in spec["shape"][1:]))
            destination = np.empty(shape, dtype=np.dtype(spec["dtype"]))
            shard_ids = requested // self.shard_size
            for shard_id in np.unique(shard_ids):
                positions = np.flatnonzero(shard_ids == shard_id)
                descriptor = spec["shards"][int(shard_id)]
                shard = np.load(self.root / descriptor["relative_path"], mmap_mode="r", allow_pickle=False)
                local = requested[positions] - int(descriptor["start"])
                destination[positions] = shard[local]
            output[str(name)] = destination
        return output

    def read_range(self, start: int, stop: int, names: Sequence[str] = _RAW_KEYS) -> dict[str, np.ndarray]:
        return self.read_indices(np.arange(int(start), int(stop), dtype=np.int64), names=names)

    def verify_shards(self) -> None:
        for spec in self.manifest["arrays"].values():
            for shard in spec["shards"]:
                path = self.root / shard["relative_path"]
                if sha256_file(path) != shard["sha256"]:
                    raise ValueError(f"staged RAM shard hash mismatch: {path}")


class AllocationNpzStager:
    """Rank-zero one-open staging for an aligned HLT/offline source pair."""

    def __init__(self, ledger: AllocationRamLedger, *, rank: int = 0, world_size: int = 1) -> None:
        self.ledger = ledger
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("invalid stager rank/world_size")
        self.persistent_npz_open_counts: dict[str, int] = {}
        self.raw_reservation_id: str | None = None
        self.raw_reservation_ids: list[str] = []

    @staticmethod
    def _validate_loaded(name: str, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> tuple[str, tuple[JetIdentity, ...]]:
        n_events = int(arrays["tokens"].shape[0])
        if arrays["tokens"].ndim != 3 or arrays["mask"].shape != arrays["tokens"].shape[:2]:
            raise ValueError(f"{name} tokens/mask have incompatible shapes")
        if any(np.asarray(arrays[key]).shape[0] != n_events for key in _RAW_KEYS):
            raise ValueError(f"{name} arrays have inconsistent event counts")
        if not np.isfinite(arrays["tokens"]).all():
            raise ValueError(f"{name} tokens contain non-finite values")
        identities = _event_identities(arrays, metadata)
        identity_hash = jet_identity_hash(identities)
        expected_identity = metadata.get("jet_identity_hash")
        if expected_identity and identity_hash != expected_identity:
            raise ValueError(f"{name} event-order hash mismatch")
        content_hash = hash_arrays({key: arrays[key] for key in _RAW_KEYS})
        expected_content = (
            metadata.get("hlt_content_hash")
            if name == "hlt"
            else metadata.get("offline_content_hash") or metadata.get("content_hash")
        )
        if expected_content and content_hash != expected_content:
            raise ValueError(f"{name} content hash mismatch")
        return content_hash, identities

    def stage_pair(
        self,
        *,
        hlt_npz: str | Path,
        hlt_metadata: str | Path,
        offline_npz: str | Path,
        offline_metadata: str | Path,
        shard_size: int = DEFAULT_RAW_SHARD_SIZE,
    ) -> tuple[StagedSource, StagedSource, dict[str, Any]]:
        # The allocation-wide stage lock is held across both persistent opens
        # and publication.  A second stager therefore fails before touching a
        # compressed source rather than discovering the existing stage later.
        with _locked_file(self.ledger.root / "raw_stage.lock"):
            if bool(self.ledger.snapshot().get("raw_stage_finalized")) or (self.ledger.root / "raw").exists():
                raise RuntimeError("allocation raw sources were already staged; refusing a persistent reopen")
            return self._stage_pair_locked(
                hlt_npz=hlt_npz,
                hlt_metadata=hlt_metadata,
                offline_npz=offline_npz,
                offline_metadata=offline_metadata,
                shard_size=shard_size,
            )

    def stage_named_pairs(
        self,
        pairs: Mapping[str, Mapping[str, str | Path]],
        *,
        shard_size: int = DEFAULT_RAW_SHARD_SIZE,
    ) -> tuple[dict[str, tuple[StagedSource, StagedSource, dict[str, Any]]], dict[str, Any]]:
        """Stage several disjoint parent splits in one allocation-wide raw stage.

        Each namespace must provide ``hlt_npz``, ``hlt_metadata``,
        ``offline_npz``, and ``offline_metadata``.  Every compressed source is
        opened exactly once, all resulting raw shards remain non-evictable,
        and the ledger is finalized only after every namespace is committed.
        """

        if not pairs:
            raise ValueError("at least one named HLT/offline source pair is required")
        required = {"hlt_npz", "hlt_metadata", "offline_npz", "offline_metadata"}
        normalized: list[tuple[str, Mapping[str, str | Path]]] = []
        for raw_name, source in pairs.items():
            name = str(raw_name)
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
                raise ValueError(f"unsafe raw-stage namespace {name!r}")
            missing = sorted(required - set(source))
            extra = sorted(set(source) - required)
            if missing or extra:
                raise ValueError(
                    f"raw-stage namespace {name!r} has missing={missing}, extra={extra}"
                )
            normalized.append((name, source))
        normalized.sort(key=lambda item: item[0])
        with _locked_file(self.ledger.root / "raw_stage.lock"):
            raw_root = self.ledger.root / "raw"
            if bool(self.ledger.snapshot().get("raw_stage_finalized")) or raw_root.exists():
                raise RuntimeError("allocation raw sources were already staged; refusing a persistent reopen")
            staged: dict[str, tuple[StagedSource, StagedSource, dict[str, Any]]] = {}
            for name, source in normalized:
                staged[name] = self._stage_pair_locked(
                    hlt_npz=source["hlt_npz"],
                    hlt_metadata=source["hlt_metadata"],
                    offline_npz=source["offline_npz"],
                    offline_metadata=source["offline_metadata"],
                    shard_size=shard_size,
                    stage_root=raw_root / name,
                    source_namespace=name,
                    finalize_raw=False,
                )
            self.ledger.finalize_raw_stage()
            report = with_content_hash(
                {
                    "contract": PREDICTION_ANCHORED_RAW_STAGE_CONTRACT,
                    "allocation_id": self.ledger.allocation_id,
                    "source_namespaces": [name for name, _ in normalized],
                    "namespace_reports": {
                        name: result[2] for name, result in staged.items()
                    },
                    "persistent_npz_open_counts": dict(self.persistent_npz_open_counts),
                    "all_persistent_npz_open_counts_equal_one": all(
                        count == 1 for count in self.persistent_npz_open_counts.values()
                    ),
                    "raw_non_evictable": True,
                    "persistent_field_tensors_written": False,
                }
            )
            return staged, report

    def _stage_pair_locked(
        self,
        *,
        hlt_npz: str | Path,
        hlt_metadata: str | Path,
        offline_npz: str | Path,
        offline_metadata: str | Path,
        shard_size: int = DEFAULT_RAW_SHARD_SIZE,
        stage_root: Path | None = None,
        source_namespace: str | None = None,
        finalize_raw: bool = True,
    ) -> tuple[StagedSource, StagedSource, dict[str, Any]]:
        if self.rank != 0:
            raise RuntimeError("only allocation rank 0 may open persistent compressed sources")
        logical_size = int(shard_size)
        if logical_size <= 0:
            raise ValueError("shard_size must be positive")
        inputs: dict[str, tuple[Path, Path]] = {
            "hlt": (Path(hlt_npz), Path(hlt_metadata)),
            "offline": (Path(offline_npz), Path(offline_metadata)),
        }
        loaded: dict[str, dict[str, np.ndarray]] = {}
        metadata_by_name: dict[str, dict[str, Any]] = {}
        source_file_hashes: dict[str, str] = {}
        content_hashes: dict[str, str] = {}
        identities_by_name: dict[str, tuple[JetIdentity, ...]] = {}
        # NPZ members are lazy.  Keep the sole file handle for each source open,
        # inspect central-directory uncompressed sizes, reserve the full staging
        # peak, and only then decompress any numeric array.
        with ExitStack() as stack:
            lazy_sources: dict[str, Any] = {}
            projected_uncompressed_bytes = 0
            for name, (npz_path, metadata_path) in inputs.items():
                if not npz_path.is_file() or not metadata_path.is_file():
                    raise FileNotFoundError(f"missing {name} source or metadata")
                key = str(npz_path.resolve())
                if self.persistent_npz_open_counts.get(key, 0) != 0:
                    raise RuntimeError(f"persistent compressed source was already opened: {npz_path}")
                self.persistent_npz_open_counts[key] = 1
                raw = stack.enter_context(npz_path.open("rb"))
                digest = hashlib.sha256()
                for chunk in iter(lambda: raw.read(1024 * 1024), b""):
                    digest.update(chunk)
                raw.seek(0)
                data = stack.enter_context(np.load(raw, allow_pickle=False))
                missing = sorted(set(_RAW_KEYS) - set(data.files))
                if missing:
                    raise ValueError(f"source NPZ {npz_path} is missing arrays {missing}")
                wanted_members = {f"{array_name}.npy" for array_name in _RAW_KEYS}
                projected_uncompressed_bytes += sum(
                    int(info.file_size)
                    for info in data.zip.infolist()
                    if info.filename in wanted_members
                )
                metadata_by_name[name] = json.loads(metadata_path.read_text(encoding="utf-8"))
                source_file_hashes[name] = digest.hexdigest()
                lazy_sources[name] = data
            raw_preflight_bytes = 2 * projected_uncompressed_bytes + 2 * 1024 * 1024
            reservation_id = self.ledger.reserve(
                owner="rank0",
                role=(
                    "aligned_hlt_offline_raw_stage_and_decompression_peak"
                    if source_namespace is None
                    else f"aligned_hlt_offline_raw_stage_and_decompression_peak:{source_namespace}"
                ),
                expected_bytes=raw_preflight_bytes,
                category="raw",
            )
            self.raw_reservation_id = reservation_id
            self.raw_reservation_ids.append(reservation_id)
            for name, data in lazy_sources.items():
                arrays = {
                    array_name: np.asarray(data[array_name], dtype=_DTYPES[array_name]).copy()
                    for array_name in _RAW_KEYS
                }
                content_hash, identities = self._validate_loaded(
                    name, arrays, metadata_by_name[name]
                )
                loaded[name] = arrays
                content_hashes[name] = content_hash
                identities_by_name[name] = identities
        if identities_by_name["hlt"] != identities_by_name["offline"]:
            raise ValueError("HLT/offline event order or labels do not match")
        if not np.array_equal(loaded["hlt"]["labels"], loaded["offline"]["labels"]):
            raise ValueError("HLT/offline labels do not match")
        for unit_key in ("units", "token_units", "feature_units"):
            hlt_units = metadata_by_name["hlt"].get(unit_key)
            offline_units = metadata_by_name["offline"].get(unit_key)
            if hlt_units is not None and offline_units is not None and hlt_units != offline_units:
                raise ValueError(f"HLT/offline {unit_key} metadata are incompatible")
        stage_root = self.ledger.root / "raw" if stage_root is None else Path(stage_root)
        stage_root.mkdir(parents=True, exist_ok=False)
        manifests: dict[str, dict[str, Any]] = {}
        measured_bytes = 0
        for name, arrays in loaded.items():
            source_root = stage_root / name
            source_root.mkdir(parents=True, exist_ok=False)
            array_specs: dict[str, Any] = {}
            n_events = int(arrays["tokens"].shape[0])
            for array_name, array in arrays.items():
                descriptors: list[dict[str, Any]] = []
                array_dir = source_root / array_name
                array_dir.mkdir()
                for shard_id, start in enumerate(range(0, n_events, logical_size)):
                    stop = min(start + logical_size, n_events)
                    path = array_dir / f"shard_{shard_id:06d}.npy"
                    np.save(path, np.asarray(array[start:stop]), allow_pickle=False)
                    written = np.load(path, mmap_mode="r", allow_pickle=False)
                    if not np.array_equal(written, np.asarray(array[start:stop])):
                        raise ValueError(f"staged RAM shard differs from decompressed source: {path}")
                    del written
                    size = int(path.stat().st_size)
                    measured_bytes += size
                    descriptors.append(
                        {
                            "shard_id": shard_id,
                            "start": start,
                            "stop": stop,
                            "relative_path": str(path.relative_to(source_root)),
                            "bytes": size,
                            "sha256": sha256_file(path),
                        }
                    )
                array_specs[array_name] = {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "shards": descriptors,
                }
            ownership = [
                {"rank": rank, "start": deterministic_rank_range(n_events, rank, self.world_size)[0], "stop": deterministic_rank_range(n_events, rank, self.world_size)[1]}
                for rank in range(self.world_size)
            ]
            manifests[name] = {
                "contract": PREDICTION_ANCHORED_RAW_STAGE_CONTRACT,
                "source_name": name,
                "source_file_sha256": source_file_hashes[name],
                "source_metadata_sha256": sha256_file(inputs[name][1]),
                "source_content_hash": content_hashes[name],
                "event_order_sha256": jet_identity_hash(identities_by_name[name]),
                "jet_files": [str(value) for value in metadata_by_name[name]["jet_files"]],
                "n_events": n_events,
                "shard_size": logical_size,
                "arrays": array_specs,
                "rank_ownership": ownership,
                "raw_non_evictable": True,
                "staged_array_values_verified": True,
                "persistent_npz_open_count": 1,
            }
            _atomic_json(source_root / "stage_manifest.json", manifests[name])
            measured_bytes += int((source_root / "stage_manifest.json").stat().st_size)
        # Release decompressed process copies; the verified RAM shards are now
        # the sole allocation source.
        loaded.clear()
        self.ledger.commit(reservation_id, measured_bytes=measured_bytes)
        if finalize_raw:
            self.ledger.finalize_raw_stage()
        hlt_stage = StagedSource("hlt", stage_root / "hlt", manifests["hlt"])
        offline_stage = StagedSource("offline", stage_root / "offline", manifests["offline"])
        hlt_stage.verify_shards()
        offline_stage.verify_shards()
        report = with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_RAW_STAGE_CONTRACT,
                "allocation_id": self.ledger.allocation_id,
                "source_namespace": source_namespace,
                "n_events": hlt_stage.n_events,
                "shard_size": logical_size,
                "world_size": self.world_size,
                "event_order_sha256": manifests["hlt"]["event_order_sha256"],
                "source_file_sha256": source_file_hashes,
                "source_content_hash": content_hashes,
                "persistent_npz_open_counts": dict(self.persistent_npz_open_counts),
                "measured_raw_stage_bytes": measured_bytes,
                "raw_stage_preflight_peak_bytes": raw_preflight_bytes,
                "raw_non_evictable": True,
                "persistent_field_tensors_written": False,
            }
        )
        return hlt_stage, offline_stage, report


class DerivedShardLRU:
    """Bounded host-RAM LRU; raw staged shards never enter this cache."""

    def __init__(
        self,
        *,
        ledger: AllocationRamLedger,
        owner: str,
        capacity_bytes: int,
        generator: Callable[[Any], Mapping[str, np.ndarray]],
    ) -> None:
        self.ledger = ledger
        self.owner = str(owner)
        self.capacity_bytes = int(capacity_bytes)
        if self.capacity_bytes <= 0:
            raise ValueError("derived LRU capacity must be positive")
        self.generator = generator
        self.reservation_id = ledger.reserve(
            owner=self.owner,
            role="derived_lru_capacity",
            expected_bytes=self.capacity_bytes,
            category="derived",
        )
        ledger.commit(self.reservation_id, measured_bytes=self.capacity_bytes)
        self._cache: OrderedDict[Any, tuple[dict[str, np.ndarray], int]] = OrderedDict()
        self._seen: set[Any] = set()
        self._resident_bytes = 0
        self._telemetry = {
            "cache_hits": 0,
            "cache_misses": 0,
            "regenerations": 0,
            "evictions": 0,
            "peak_derived_resident_bytes": 0,
        }

    @staticmethod
    def _bytes(arrays: Mapping[str, np.ndarray]) -> int:
        return sum(int(np.asarray(value).nbytes) for value in arrays.values())

    def get(self, key: Any) -> Mapping[str, np.ndarray]:
        if key in self._cache:
            arrays, size = self._cache.pop(key)
            self._cache[key] = (arrays, size)
            self._telemetry["cache_hits"] += 1
            return arrays
        self._telemetry["cache_misses"] += 1
        if key in self._seen:
            self._telemetry["regenerations"] += 1
        arrays = {str(name): np.asarray(value) for name, value in self.generator(key).items()}
        size = self._bytes(arrays)
        if size > self.capacity_bytes:
            raise MemoryError("one derived shard exceeds the bounded LRU capacity")
        while self._cache and self._resident_bytes + size > self.capacity_bytes:
            _, (_, removed_size) = self._cache.popitem(last=False)
            self._resident_bytes -= removed_size
            self._telemetry["evictions"] += 1
        self._cache[key] = (arrays, size)
        self._seen.add(key)
        self._resident_bytes += size
        self._telemetry["peak_derived_resident_bytes"] = max(
            self._telemetry["peak_derived_resident_bytes"], self._resident_bytes
        )
        return arrays

    def telemetry(self) -> dict[str, Any]:
        requests = self._telemetry["cache_hits"] + self._telemetry["cache_misses"]
        return {
            "contract": PREDICTION_ANCHORED_RAM_TELEMETRY_CONTRACT,
            **self._telemetry,
            "current_derived_resident_bytes": self._resident_bytes,
            "derived_capacity_bytes": self.capacity_bytes,
            "cache_hit_rate": None if requests == 0 else self._telemetry["cache_hits"] / requests,
            "raw_shards_evictable": False,
            "persistent_field_tensors_written": False,
        }

    def close(self) -> None:
        self._cache.clear()
        self._resident_bytes = 0
        self.ledger.release(self.reservation_id)


class R0Generator(Protocol):
    checkpoint_sha256: str

    def predict_numpy(self, tokens: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class FrozenR0Runner:
    """Frozen deterministic adapter for existing local residual checkpoints."""

    def __init__(self, checkpoint_path: str | Path, *, device: Any = "cpu") -> None:
        from .tagger import load_local_residual_reconstructor_from_checkpoint

        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self.model, self.payload = load_local_residual_reconstructor_from_checkpoint(
            self.checkpoint_path, map_location=device
        )
        self.device = device
        self.model.to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def predict_numpy(self, tokens: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch

        particles = np.asarray(tokens, dtype=np.float32)
        valid = np.asarray(mask, dtype=bool)
        if particles.ndim != 3 or valid.shape != particles.shape[:2]:
            raise ValueError("R0 tokens/mask shapes do not align")
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                torch.as_tensor(particles, dtype=torch.float32, device=self.device),
                torch.as_tensor(valid, dtype=torch.bool, device=self.device),
            )
        f0 = output.predicted_fields.detach().cpu().numpy().astype(np.float32, copy=False)
        h0 = output.hidden.detach().cpu().numpy().astype(np.float32, copy=False)
        f0[~valid] = 0.0
        h0[~valid] = 0.0
        if not np.isfinite(f0).all() or not np.isfinite(h0).all():
            raise ValueError("frozen R0 produced non-finite values")
        return f0, h0


def build_frozen_r0_registration(
    checkpoint_path: str | Path,
    *,
    preprocessing_sha256: str,
    target_schema_sha256: str,
    split_manifest_sha256: str,
    matching_policy: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    target_normalization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a trained ordinary predictor to its data and target contracts."""

    import torch

    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_config = payload.get("model_config") if isinstance(payload, Mapping) else None
    if not isinstance(model_config, Mapping):
        raise ValueError("R0 checkpoint is missing model_config")
    if int(model_config.get("field_dim", -1)) != 50 or int(model_config.get("particle_dim", -1)) != 14:
        raise ValueError("R0 checkpoint does not implement the 14-input/50-field target schema")
    state_dict = payload.get("model_state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("R0 checkpoint is incomplete or missing model_state_dict")
    for name, value in state_dict.items():
        if hasattr(value, "is_floating_point") and value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"R0 checkpoint tensor {name!r} contains non-finite values")
    if not validation_metrics or not all(np.isfinite(float(v)) for v in validation_metrics.values() if isinstance(v, (int, float))):
        raise ValueError("R0 registration requires finite validation metrics")
    hashes = {
        "preprocessing_sha256": str(preprocessing_sha256),
        "target_schema_sha256": str(target_schema_sha256),
        "split_manifest_sha256": str(split_manifest_sha256),
    }
    if any(len(value) != 64 for value in hashes.values()):
        raise ValueError("R0 parent hashes must be SHA-256 values")
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_R0_REGISTRATION_CONTRACT,
            "checkpoint_path": str(path.resolve()),
            "checkpoint_sha256": sha256_file(path),
            "checkpoint_contract": payload.get("checkpoint_contract"),
            "model_config": dict(model_config),
            "preprocessing": hashes["preprocessing_sha256"],
            "target_schema": hashes["target_schema_sha256"],
            "target_normalization": dict(
                target_normalization or {"mode": "none", "space": "physical_repository_native_units"}
            ),
            "split_manifest": hashes["split_manifest_sha256"],
            "matching_policy": dict(matching_policy),
            "validation_metrics": dict(validation_metrics),
            "input_availability": "hlt_only",
            "target": "f_true",
            "frozen": True,
            "exposes": ["f0", "h0"],
            "masked_outputs_exact_zero": True,
        }
    )


class PredictionAnchoredBridgeProvider:
    """Aligned streamed truth/R0 provider backed only by resident raw shards."""

    def __init__(
        self,
        *,
        hlt: StagedSource,
        offline: StagedSource,
        r0: R0Generator,
        ledger: AllocationRamLedger,
        rank: int,
        world_size: int,
        derived_capacity_bytes: int,
        radii: Sequence[float] = DEFAULT_LOCAL_RESIDUAL_RADII,
    ) -> None:
        if hlt.n_events != offline.n_events:
            raise ValueError("staged HLT/offline event counts differ")
        if hlt.manifest["event_order_sha256"] != offline.manifest["event_order_sha256"]:
            raise ValueError("staged HLT/offline event order differs")
        self.hlt = hlt
        self.offline = offline
        self.r0 = r0
        self.ledger = ledger
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.rank_start, self.rank_stop = deterministic_rank_range(hlt.n_events, rank, world_size)
        self.radii = tuple(float(value) for value in radii)
        self._control_maps: dict[str, Mapping[str, Any]] = {}
        self._lru = DerivedShardLRU(
            ledger=ledger,
            owner=f"rank{rank}",
            capacity_bytes=int(derived_capacity_bytes),
            generator=self._generate,
        )

    def _range_for_shard(self, shard_id: int) -> tuple[int, int]:
        raw_start = int(shard_id) * self.hlt.shard_size
        raw_stop = min(raw_start + self.hlt.shard_size, self.hlt.n_events)
        start = max(raw_start, self.rank_start)
        stop = min(raw_stop, self.rank_stop)
        if start < 0 or start >= stop:
            raise IndexError("derived shard does not intersect this rank's deterministic ownership")
        return start, stop

    def _owned_indices(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        requested = np.asarray(indices, dtype=np.int64)
        if requested.ndim != 1 or np.any(requested < self.rank_start) or np.any(requested >= self.rank_stop):
            raise ValueError("requested indices escape this rank's deterministic ownership")
        return requested

    def _raw(self, start: int, stop: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        hlt = self.hlt.read_range(start, stop)
        offline = self.offline.read_range(start, stop)
        for key in ("labels", "jet_file_indices", "jet_entries"):
            if not np.array_equal(hlt[key], offline[key]):
                raise ValueError(f"staged pair changed event order for {key}")
        return hlt, offline

    def _generate(self, key: Any) -> Mapping[str, np.ndarray]:
        if not isinstance(key, tuple) or len(key) < 2:
            raise ValueError("derived key must be (role, shard_id, ...)")
        role, shard_id, *options = key
        start, stop = self._range_for_shard(int(shard_id))
        hlt, offline = self._raw(start, stop)
        if role == "f_true":
            fields, target_mask, _, _, _ = compute_local_particle_residual_fields(
                hlt["tokens"], hlt["mask"], offline["tokens"], offline["mask"], radii=self.radii
            )
            return {"fields": fields, "mask": target_mask}
        f0, _ = self.r0.predict_numpy(hlt["tokens"], hlt["mask"])
        if role == "f0":
            return {"fields": f0, "mask": hlt["mask"]}
        truth = self._lru.get(("f_true", int(shard_id)))
        if role == "bridge":
            rho = options[0] if options else "0.100"
            policy = options[1] if len(options) > 1 else BRIDGE_CHANNEL_PHYSICAL45
            fields = virtual_bridge(f0, truth["fields"], truth["mask"], rho=rho, channel_policy=policy)
            return {"fields": fields, "mask": truth["mask"]}
        if role == "control":
            if len(options) < 2:
                raise ValueError("control key requires control type and seed")
            files = [str(value) for value in self.hlt.manifest["jet_files"]]
            event_ids = [
                f"{files[int(file_index)]}\0{int(entry)}\0{int(label)}"
                for file_index, entry, label in zip(
                    hlt["jet_file_indices"], hlt["jet_entries"], hlt["labels"]
                )
            ]
            wrong_map: Any = None
            if len(options) > 2:
                wrong_map = self._control_maps.get(str(options[2]), options[2])
            fields = apply_bridge_control(
                f0,
                truth["fields"],
                truth["mask"],
                control_type=str(options[0]),
                seed=int(options[1]),
                event_ids=event_ids,
                wrong_event_map=wrong_map,
            )
            return {"fields": fields, "mask": truth["mask"]}
        raise ValueError(f"unknown derived role {role!r}")

    def derived_shard(self, key: Any) -> Mapping[str, np.ndarray]:
        return self._lru.get(key)

    def control_shard(
        self,
        shard_id: int,
        *,
        control_type: str,
        seed: int,
        wrong_event_map: Mapping[str, Any] | None = None,
    ) -> Mapping[str, np.ndarray]:
        map_hash: str | None = None
        if wrong_event_map is not None:
            map_hash = str(wrong_event_map.get("content_hash", ""))
            unhashed = dict(wrong_event_map)
            claimed = unhashed.pop("content_hash", None)
            if not map_hash or claimed != canonical_sha256(unhashed):
                raise ValueError("control shard received an invalid wrong-event map")
            self._control_maps[map_hash] = dict(wrong_event_map)
        key = ("control", int(shard_id), str(control_type), int(seed), map_hash)
        return self._lru.get(key)

    def truth_for_indices(self, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        requested = self._owned_indices(indices)
        hlt = self.hlt.read_indices(requested, names=("tokens", "mask"))
        offline = self.offline.read_indices(requested, names=("tokens", "mask"))
        fields, target_mask, _, _, _ = compute_local_particle_residual_fields(
            hlt["tokens"], hlt["mask"], offline["tokens"], offline["mask"], radii=self.radii
        )
        return fields, target_mask

    def r0_for_indices(self, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        requested = self._owned_indices(indices)
        hlt = self.hlt.read_indices(requested, names=("tokens", "mask"))
        return self.r0.predict_numpy(hlt["tokens"], hlt["mask"])

    def matched_wrong_event_map(
        self,
        indices: Sequence[int],
        *,
        seed: int,
        bin_edges: Mapping[str, Sequence[float]] | None = None,
    ) -> dict[str, Any]:
        requested = self._owned_indices(indices)
        hlt = self.hlt.read_indices(
            requested,
            names=("tokens", "mask", "labels", "jet_file_indices", "jet_entries"),
        )
        files = [str(value) for value in self.hlt.manifest["jet_files"]]
        event_ids = [
            f"{files[int(file_index)]}\0{int(entry)}\0{int(label)}"
            for file_index, entry, label in zip(
                hlt["jet_file_indices"], hlt["jet_entries"], hlt["labels"]
            )
        ]
        return build_matched_wrong_event_map(
            tokens=hlt["tokens"],
            mask=hlt["mask"],
            labels=hlt["labels"],
            event_ids=event_ids,
            seed=int(seed),
            bin_edges=bin_edges,
            logical_block_size=self.hlt.shard_size,
            source_block_ids=requested // int(self.hlt.shard_size),
        )

    def ordered_training_indices(self, indices: Sequence[int], *, seed: int) -> np.ndarray:
        """Permute resident shard order, then event order within each shard."""

        values = np.asarray(indices, dtype=np.int64)
        if values.ndim != 1 or np.any(values < self.rank_start) or np.any(values >= self.rank_stop):
            raise ValueError("training indices escape this rank's deterministic ownership")
        rng = np.random.default_rng(int(seed))
        by_shard: dict[int, np.ndarray] = {}
        for shard in np.unique(values // self.hlt.shard_size):
            block = values[values // self.hlt.shard_size == shard].copy()
            rng.shuffle(block)
            by_shard[int(shard)] = block
        shard_order = np.asarray(sorted(by_shard), dtype=np.int64)
        rng.shuffle(shard_order)
        return np.concatenate([by_shard[int(shard)] for shard in shard_order]) if shard_order.size else values

    def iter_batches(
        self,
        indices: Sequence[int],
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> Iterator[dict[str, np.ndarray]]:
        order = np.asarray(indices, dtype=np.int64)
        if shuffle:
            order = self.ordered_training_indices(order, seed=seed)
        for start in range(0, order.size, int(batch_size)):
            selected = order[start : start + int(batch_size)]
            hlt = self.hlt.read_indices(selected, names=("tokens", "mask", "labels"))
            truth, target_mask = self.truth_for_indices(selected)
            yield {
                "indices": selected,
                "hlt_tokens": hlt["tokens"],
                "hlt_mask": hlt["mask"],
                "labels": hlt["labels"],
                "target_fields": truth,
                "target_mask": target_mask,
            }

    def telemetry(self) -> dict[str, Any]:
        return {
            **self._lru.telemetry(),
            "rank": self.rank,
            "world_size": self.world_size,
            "rank_range": [self.rank_start, self.rank_stop],
            "raw_stage_event_order_sha256": self.hlt.manifest["event_order_sha256"],
            "r0_checkpoint_sha256": self.r0.checkpoint_sha256,
            "h0_cached": False,
        }

    def close(self) -> None:
        self._lru.close()


__all__ = [
    "PREDICTION_ANCHORED_RAM_LEDGER_CONTRACT",
    "PREDICTION_ANCHORED_RAW_STAGE_CONTRACT",
    "PREDICTION_ANCHORED_RAM_TELEMETRY_CONTRACT",
    "PREDICTION_ANCHORED_R0_REGISTRATION_CONTRACT",
    "DEFAULT_RAW_SHARD_SIZE",
    "MANDATORY_HEADROOM_FRACTION",
    "AllocationRamLedger",
    "AllocationNpzStager",
    "DerivedShardLRU",
    "FrozenR0Runner",
    "PredictionAnchoredBridgeProvider",
    "StagedSource",
    "build_frozen_r0_registration",
    "deterministic_rank_range",
    "require_single_node",
]
