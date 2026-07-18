"""Verified, bounded rank-local RAM workspaces for ABPH workers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
from typing import Any, Iterator, Mapping
import uuid


ABPH_RAM_WORKSPACE_CONTRACT = "adaptive_binary_pseudooffline_ram_workspace_v1"
ABPH_RAM_RESERVATION_CONTRACT = "adaptive_binary_pseudooffline_ram_reservation_v1"
ABPH_RAM_MINIMUM_HEADROOM_FRACTION = 0.20
ABPH_RAM_WORKSPACE_DIRS = (
    "inputs",
    "targets",
    "pseudo",
    "checkpoints",
    "codec_scratch",
)
_MEMORY_FILESYSTEMS = frozenset({"tmpfs", "ramfs"})


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _parse_limit(path: Path) -> int | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    if value in {"", "max"}:
        return None
    parsed = int(value)
    if parsed <= 0 or parsed >= 2**60:
        return None
    return parsed


def _parse_mountinfo(path: Path = Path("/proc/self/mountinfo")) -> tuple[tuple[Path, str], ...]:
    if not path.is_file():
        return ()
    rows: list[tuple[Path, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if "-" not in fields or len(fields) < 7:
            continue
        separator = fields.index("-")
        mount_point = fields[4].replace("\\040", " ")
        filesystem = fields[separator + 1]
        rows.append((Path(mount_point).resolve(), filesystem))
    return tuple(rows)


def filesystem_type(path: str | Path) -> str | None:
    """Return the longest-prefix mount filesystem type for ``path``."""

    resolved = Path(path).resolve()
    matches: list[tuple[int, str]] = []
    for mount, kind in _parse_mountinfo():
        try:
            resolved.relative_to(mount)
        except ValueError:
            continue
        matches.append((len(mount.parts), kind))
    return max(matches, default=(0, None))[1]


@dataclass(frozen=True)
class RamCapacityProbe:
    root: Path
    filesystem_type: str
    filesystem_free_bytes: int
    cgroup_limit_bytes: int | None
    cgroup_current_bytes: int
    slurm_allocation_bytes: int | None
    headroom_fraction: float = ABPH_RAM_MINIMUM_HEADROOM_FRACTION

    def __post_init__(self) -> None:
        if self.filesystem_type not in _MEMORY_FILESYSTEMS:
            raise ValueError(
                f"RAM workspace requires tmpfs/ramfs, got {self.filesystem_type!r}"
            )
        if self.filesystem_free_bytes <= 0 or self.cgroup_current_bytes < 0:
            raise ValueError("RAM capacity probe contains invalid byte counts")
        if not 0.0 < self.headroom_fraction < 1.0:
            raise ValueError("RAM workspace headroom fraction must lie in (0, 1)")

    @property
    def governing_limit_bytes(self) -> int:
        limits = [int(self.filesystem_free_bytes)]
        if self.cgroup_limit_bytes is not None:
            limits.append(max(0, int(self.cgroup_limit_bytes) - int(self.cgroup_current_bytes)))
        if self.slurm_allocation_bytes is not None:
            limits.append(max(0, int(self.slurm_allocation_bytes) - int(self.cgroup_current_bytes)))
        return min(limits)

    @property
    def reservation_limit_bytes(self) -> int:
        usable = [
            int(self.filesystem_free_bytes * (1.0 - self.headroom_fraction))
        ]
        if self.cgroup_limit_bytes is not None:
            usable.append(
                max(
                    0,
                    int(self.cgroup_limit_bytes * (1.0 - self.headroom_fraction))
                    - int(self.cgroup_current_bytes),
                )
            )
        if self.slurm_allocation_bytes is not None:
            usable.append(
                max(
                    0,
                    int(self.slurm_allocation_bytes * (1.0 - self.headroom_fraction))
                    - int(self.cgroup_current_bytes),
                )
            )
        return min(usable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root.resolve()),
            "filesystem_type": self.filesystem_type,
            "filesystem_free_bytes": int(self.filesystem_free_bytes),
            "cgroup_limit_bytes": self.cgroup_limit_bytes,
            "cgroup_current_bytes": int(self.cgroup_current_bytes),
            "slurm_allocation_bytes": self.slurm_allocation_bytes,
            "headroom_fraction": float(self.headroom_fraction),
            "governing_limit_bytes": int(self.governing_limit_bytes),
            "reservation_limit_bytes": int(self.reservation_limit_bytes),
        }


def _slurm_allocation_bytes(env: Mapping[str, str]) -> int | None:
    raw = env.get("SLURM_MEM_PER_NODE")
    if raw in {None, "", "0", "N/A"}:
        return None
    # Slurm exports SLURM_MEM_PER_NODE in MiB.
    return int(raw) * 1024 * 1024


def probe_ram_capacity(
    root: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    expected_filesystem_type: str | None = None,
) -> RamCapacityProbe:
    environment = os.environ if env is None else env
    resolved = Path(root).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    kind = expected_filesystem_type or filesystem_type(resolved)
    if kind not in _MEMORY_FILESYSTEMS:
        raise RuntimeError(
            f"refusing disk-backed ABPH workspace {resolved}; detected filesystem {kind!r}"
        )
    usage = shutil.disk_usage(resolved)
    cgroup_limit = _parse_limit(Path("/sys/fs/cgroup/memory.max"))
    cgroup_current = _parse_limit(Path("/sys/fs/cgroup/memory.current")) or 0
    if cgroup_limit is None:
        cgroup_limit = _parse_limit(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
        cgroup_current = (
            _parse_limit(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")) or 0
        )
    probe = RamCapacityProbe(
        root=resolved,
        filesystem_type=kind,
        filesystem_free_bytes=int(usage.free),
        cgroup_limit_bytes=cgroup_limit,
        cgroup_current_bytes=int(cgroup_current),
        slurm_allocation_bytes=_slurm_allocation_bytes(environment),
    )
    if probe.reservation_limit_bytes <= 0:
        raise MemoryError("RAM workspace has no capacity after mandatory 20% headroom")
    return probe


def select_ram_root(*, env: Mapping[str, str] | None = None) -> Path:
    environment = os.environ if env is None else env
    candidates: list[Path] = []
    if environment.get("ABPH_RAM_WORKSPACE_BASE"):
        candidates.append(Path(environment["ABPH_RAM_WORKSPACE_BASE"]))
    candidates.append(Path("/dev/shm"))
    if environment.get("SLURM_TMPDIR"):
        candidates.append(Path(environment["SLURM_TMPDIR"]))
    failures: list[str] = []
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe_ram_capacity(candidate, env=environment)
            return candidate.resolve()
        except (OSError, RuntimeError, ValueError, MemoryError) as exc:
            failures.append(f"{candidate}: {exc}")
    raise RuntimeError("no verified memory-backed workspace is available: " + "; ".join(failures))


class RankLocalWorkspace:
    """Own one rank's tmpfs workspace and account every resident byte."""

    def __init__(
        self,
        root: str | Path,
        *,
        job_id: str,
        rank: int,
        probe: RamCapacityProbe | None = None,
        create: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self.job_id = str(job_id)
        self.rank = int(rank)
        if not self.job_id or self.rank < 0:
            raise ValueError("workspace job ID and rank are required")
        self.probe = probe or probe_ram_capacity(self.root.parent)
        self._reservation_limit_override = int(self.probe.reservation_limit_bytes)
        self._lock = threading.RLock()
        self.manifest_path = self.root / "workspace_manifest.json"
        if create:
            self._initialize()
        else:
            self._require_owner()

    @classmethod
    def from_environment(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        rank: int | None = None,
    ) -> "RankLocalWorkspace":
        environment = os.environ if env is None else env
        job_id = str(environment.get("SLURM_JOB_ID") or environment.get("ABPH_JOB_ID") or "local")
        resolved_rank = int(
            rank
            if rank is not None
            else environment.get("SLURM_PROCID", environment.get("RANK", "0"))
        )
        base = select_ram_root(env=environment)
        root = base / f"abph-{job_id}-{resolved_rank}"
        return cls(root, job_id=job_id, rank=resolved_rank)

    def _initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        for name in ABPH_RAM_WORKSPACE_DIRS:
            path = self.root / name
            path.mkdir(exist_ok=True, mode=0o700)
            os.chmod(path, 0o700)
        if self.manifest_path.exists():
            existing = self._require_owner()
            prior_limit = int(existing["capacity"]["reservation_limit_bytes"])
            self._reservation_limit_override = min(
                prior_limit, int(self.probe.reservation_limit_bytes)
            )
            if self._reservation_limit_override < prior_limit:
                existing["capacity"] = {
                    **self.probe.to_dict(),
                    "reservation_limit_bytes": self._reservation_limit_override,
                }
                self._write(existing)
            return
        payload: dict[str, Any] = {
            "contract": ABPH_RAM_WORKSPACE_CONTRACT,
            "job_id": self.job_id,
            "rank": self.rank,
            "root": str(self.root),
            "capacity": self.probe.to_dict(),
            "reservations": {},
        }
        payload["content_hash"] = _canonical_hash(payload)
        _atomic_json(self.manifest_path, payload)

    def _read(self) -> dict[str, Any]:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        expected = payload.pop("content_hash", None)
        if expected != _canonical_hash(payload):
            raise ValueError("RAM workspace manifest hash mismatch")
        payload["content_hash"] = expected
        return payload

    def _write(self, payload: Mapping[str, Any]) -> None:
        values = {key: value for key, value in payload.items() if key != "content_hash"}
        values["content_hash"] = _canonical_hash(values)
        _atomic_json(self.manifest_path, values)

    def _require_owner(self) -> dict[str, Any]:
        payload = self._read()
        if (
            payload.get("contract") != ABPH_RAM_WORKSPACE_CONTRACT
            or payload.get("job_id") != self.job_id
            or int(payload.get("rank", -1)) != self.rank
            or Path(str(payload.get("root", ""))).resolve() != self.root
        ):
            raise PermissionError("RAM workspace ownership contract mismatch")
        return payload

    @property
    def reservation_limit_bytes(self) -> int:
        return int(self._reservation_limit_override)

    @property
    def reserved_bytes(self) -> int:
        with self._lock:
            payload = self._require_owner()
            return sum(int(row["bytes"]) for row in payload["reservations"].values())

    def reserve(self, *, owner: str, role: str, expected_bytes: int) -> str:
        requested = int(expected_bytes)
        if requested <= 0 or not owner or not role:
            raise ValueError("RAM reservation requires owner, role, and positive bytes")
        with self._lock:
            payload = self._require_owner()
            reservations = dict(payload["reservations"])
            active = sum(int(row["bytes"]) for row in reservations.values())
            if active + requested > self.reservation_limit_bytes:
                raise MemoryError(
                    f"RAM workspace reservation would use {active + requested} bytes; "
                    f"limit is {self.reservation_limit_bytes} with 20% headroom"
                )
            reservation_id = uuid.uuid4().hex
            reservations[reservation_id] = {
                "contract": ABPH_RAM_RESERVATION_CONTRACT,
                "owner": str(owner),
                "role": str(role),
                "bytes": requested,
                "state": "reserved",
            }
            payload["reservations"] = reservations
            self._write(payload)
            return reservation_id

    def commit(self, reservation_id: str, *, measured_bytes: int) -> None:
        measured = int(measured_bytes)
        if measured < 0:
            raise ValueError("measured RAM bytes cannot be negative")
        with self._lock:
            payload = self._require_owner()
            reservations = dict(payload["reservations"])
            if reservation_id not in reservations:
                raise KeyError(f"unknown RAM reservation {reservation_id}")
            active_without = sum(
                int(row["bytes"])
                for key, row in reservations.items()
                if key != reservation_id
            )
            if active_without + measured > self.reservation_limit_bytes:
                raise MemoryError("measured RAM allocation exceeds the bounded workspace")
            row = dict(reservations[reservation_id])
            row.update({"bytes": measured, "state": "committed"})
            reservations[reservation_id] = row
            payload["reservations"] = reservations
            self._write(payload)

    def commit_tree(self, reservation_id: str, measured_path: str | Path) -> int:
        """Measure and commit a tree that is owned by this workspace."""

        measured_root = Path(measured_path).resolve()
        try:
            relative = measured_root.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(
                f"RAM reservation measured path escapes owned workspace: {measured_root}"
            ) from exc
        if not relative.parts:
            raise ValueError("RAM reservation must measure a stage beneath workspace root")
        if not measured_root.exists():
            raise FileNotFoundError(
                f"RAM reservation measured path is missing: {measured_root}"
            )
        if measured_root.is_symlink():
            raise ValueError("RAM reservation measured path cannot be a symbolic link")
        if measured_root.is_file():
            measured_bytes = measured_root.stat().st_size
        else:
            measured_bytes = 0
            for path in measured_root.rglob("*"):
                if path.is_symlink():
                    raise ValueError(
                        f"RAM reservation measured tree contains a symbolic link: {path}"
                    )
                if path.is_file():
                    resolved = path.resolve()
                    try:
                        resolved.relative_to(measured_root)
                    except ValueError as exc:
                        raise ValueError(
                            f"RAM reservation measured file escapes stage tree: {path}"
                        ) from exc
                    measured_bytes += path.stat().st_size
        self.commit(reservation_id, measured_bytes=measured_bytes)
        return int(measured_bytes)

    def release(self, reservation_id: str) -> None:
        with self._lock:
            payload = self._require_owner()
            reservations = dict(payload["reservations"])
            if reservation_id not in reservations:
                return
            reservations.pop(reservation_id)
            payload["reservations"] = reservations
            self._write(payload)

    @contextmanager
    def reservation(
        self, *, owner: str, role: str, expected_bytes: int
    ) -> Iterator[str]:
        reservation_id = self.reserve(
            owner=owner, role=role, expected_bytes=expected_bytes
        )
        try:
            yield reservation_id
        except Exception:
            self.release(reservation_id)
            raise

    def cleanup(self, *, require_empty: bool = False) -> None:
        with self._lock:
            payload = self._require_owner()
            if require_empty and payload["reservations"]:
                raise RuntimeError("refusing to clean a workspace with active reservations")
            shutil.rmtree(self.root)


__all__ = [
    "ABPH_RAM_MINIMUM_HEADROOM_FRACTION",
    "ABPH_RAM_RESERVATION_CONTRACT",
    "ABPH_RAM_WORKSPACE_CONTRACT",
    "ABPH_RAM_WORKSPACE_DIRS",
    "RamCapacityProbe",
    "RankLocalWorkspace",
    "filesystem_type",
    "probe_ram_capacity",
    "select_ram_root",
]
