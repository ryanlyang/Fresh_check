"""Persistent-storage contracts and quota accounting for ABPH campaigns."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence
import uuid


ABPH_STREAMING_STORAGE_CONTRACT = "adaptive_binary_pseudooffline_streaming_30gb_v1"
ABPH_STORAGE_CONTRACT = "adaptive_binary_pseudooffline_storage_contract_v1"
ABPH_STORAGE_LEDGER_CONTRACT = "adaptive_binary_pseudooffline_quota_ledger_v1"
ABPH_STORAGE_PROJECTION_CONTRACT = "adaptive_binary_pseudooffline_storage_projection_v1"
ABPH_STORAGE_AUDIT_CONTRACT = "adaptive_binary_pseudooffline_storage_audit_v1"
ABPH_STORAGE_RESERVATION_CONTRACT = "adaptive_binary_pseudooffline_storage_reservation_v1"

ABPH_CACHE_HEAVY_STORAGE_PROFILE = "cache_heavy_v1"
ABPH_STREAMING_STORAGE_PROFILE = "streaming_30gb_v1"
ABPH_STORAGE_PROFILE_NAMES = (
    ABPH_CACHE_HEAVY_STORAGE_PROFILE,
    ABPH_STREAMING_STORAGE_PROFILE,
)

ABPH_MAX_PERSISTENT_BYTES = 30_000_000_000
ABPH_MAX_PROJECTED_PEAK_BYTES = 24_000_000_000
ABPH_TARGET_STEADY_STATE_BYTES = 20_000_000_000
ABPH_MINIMUM_SAFETY_HEADROOM_BYTES = 10_000_000_000
ABPH_RESERVATION_METADATA_ALLOWANCE_BYTES = 64 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON mapping")
    return payload


def _content_hash_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "content_hash"}


def _require_content_hash(payload: Mapping[str, Any], *, label: str) -> None:
    expected = payload.get("content_hash")
    actual = _canonical_hash(_content_hash_payload(payload))
    if not isinstance(expected, str) or expected != actual:
        raise ValueError(f"{label} content hash mismatch")


class StorageArtifactClass(str, Enum):
    PERSISTENT_ESSENTIAL = "persistent_essential"
    SHARED_TRANSIENT = "shared_transient"
    RANK_LOCAL_EPHEMERAL = "rank_local_ephemeral"
    FORBIDDEN_PERSISTENT = "forbidden_persistent"


PERSISTENT_ARTIFACT_CLASSES = frozenset(
    {
        StorageArtifactClass.PERSISTENT_ESSENTIAL,
        StorageArtifactClass.SHARED_TRANSIENT,
    }
)


@dataclass(frozen=True)
class StorageProfile:
    name: str
    implementation_contract: str
    enforce_quota: bool
    max_persistent_bytes: int
    max_projected_peak_bytes: int
    target_steady_state_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.max_persistent_bytes,
            self.max_projected_peak_bytes,
            self.target_steady_state_bytes,
        )
        if not self.name or not self.implementation_contract:
            raise ValueError("storage profile name and contract are required")
        if any(int(value) <= 0 for value in values):
            raise ValueError("storage profile limits must be positive")
        if self.max_projected_peak_bytes > self.max_persistent_bytes:
            raise ValueError("projected peak gate exceeds the hard storage cap")
        if self.target_steady_state_bytes > self.max_projected_peak_bytes:
            raise ValueError("retained-byte gate exceeds the projected peak gate")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "implementation_contract": self.implementation_contract,
            "enforce_quota": self.enforce_quota,
            "max_persistent_bytes": int(self.max_persistent_bytes),
            "max_projected_peak_bytes": int(self.max_projected_peak_bytes),
            "target_steady_state_bytes": int(self.target_steady_state_bytes),
            "minimum_safety_headroom_bytes": int(
                self.max_persistent_bytes - self.target_steady_state_bytes
            ),
        }


_PROFILES: Mapping[str, StorageProfile] = {
    ABPH_CACHE_HEAVY_STORAGE_PROFILE: StorageProfile(
        name=ABPH_CACHE_HEAVY_STORAGE_PROFILE,
        implementation_contract="adaptive_binary_pseudooffline_cache_heavy_v1",
        enforce_quota=False,
        max_persistent_bytes=2**63 - 1,
        max_projected_peak_bytes=2**63 - 1,
        target_steady_state_bytes=2**63 - 1,
    ),
    ABPH_STREAMING_STORAGE_PROFILE: StorageProfile(
        name=ABPH_STREAMING_STORAGE_PROFILE,
        implementation_contract=ABPH_STREAMING_STORAGE_CONTRACT,
        enforce_quota=True,
        max_persistent_bytes=ABPH_MAX_PERSISTENT_BYTES,
        max_projected_peak_bytes=ABPH_MAX_PROJECTED_PEAK_BYTES,
        target_steady_state_bytes=ABPH_TARGET_STEADY_STATE_BYTES,
    ),
}


def resolve_storage_profile(profile: str | StorageProfile) -> StorageProfile:
    if isinstance(profile, StorageProfile):
        return profile
    try:
        return _PROFILES[str(profile)]
    except KeyError as exc:
        raise ValueError(f"unknown ABPH storage profile {profile!r}") from exc


@dataclass(frozen=True)
class StorageProjectionRow:
    artifact_family: str
    artifact_class: StorageArtifactClass
    expected_bytes: int
    active_from_wave: int
    active_through_wave: int
    retained: bool = False
    atomic_write_overhead_bytes: int = 0
    measurement_source: str = ""

    def __post_init__(self) -> None:
        if not self.artifact_family or not self.measurement_source:
            raise ValueError("projection rows require family and measurement source")
        if self.artifact_class not in PERSISTENT_ARTIFACT_CLASSES:
            raise ValueError("persistent projection cannot contain ephemeral/forbidden artifacts")
        if int(self.expected_bytes) < 0 or int(self.atomic_write_overhead_bytes) < 0:
            raise ValueError("projection byte counts cannot be negative")
        if int(self.active_from_wave) < 0 or self.active_through_wave < self.active_from_wave:
            raise ValueError("projection wave interval is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_family": self.artifact_family,
            "artifact_class": self.artifact_class.value,
            "expected_bytes": int(self.expected_bytes),
            "active_from_wave": int(self.active_from_wave),
            "active_through_wave": int(self.active_through_wave),
            "retained": bool(self.retained),
            "atomic_write_overhead_bytes": int(self.atomic_write_overhead_bytes),
            "measurement_source": self.measurement_source,
        }


def build_storage_projection(
    *,
    campaign_root: str | Path,
    campaign_mode: str,
    profile: str | StorageProfile,
    rows: Sequence[StorageProjectionRow],
    measurement_contract: str,
    sample_provenance_hash: str,
) -> dict[str, Any]:
    """Build and validate an immutable, wave-aware storage projection."""

    resolved = resolve_storage_profile(profile)
    if campaign_mode not in {"pilot", "highdata"}:
        raise ValueError("storage projection campaign mode must be pilot or highdata")
    if not rows:
        raise ValueError("storage projection requires measured artifact rows")
    if not measurement_contract or not sample_provenance_hash:
        raise ValueError("storage projection requires measurement and sample provenance")
    wave_totals, projected_peak, retained = _derive_projection_totals(rows)
    payload: dict[str, Any] = {
        "contract": ABPH_STORAGE_PROJECTION_CONTRACT,
        "implementation_contract": resolved.implementation_contract,
        "storage_profile": resolved.name,
        "campaign_root": str(Path(campaign_root).resolve()),
        "campaign_mode": campaign_mode,
        "measurement_contract": measurement_contract,
        "sample_provenance_hash": sample_provenance_hash,
        "projection_complete": True,
        "rows": [row.to_dict() for row in rows],
        "wave_totals_bytes": wave_totals,
        "projected_peak_persistent_bytes": int(projected_peak),
        "projected_final_retained_bytes": int(retained),
        "limits": resolved.to_dict(),
    }
    problems = _projection_problems(payload, resolved)
    payload["problems"] = problems
    payload["ok"] = not problems
    payload["content_hash"] = _canonical_hash(payload)
    return payload


def _derive_projection_totals(
    rows: Sequence[StorageProjectionRow],
) -> tuple[dict[str, int], int, int]:
    final_wave = max(row.active_through_wave for row in rows)
    wave_totals: dict[str, int] = {}
    for wave in range(final_wave + 1):
        wave_totals[str(wave)] = sum(
            row.expected_bytes
            + (row.atomic_write_overhead_bytes if wave == row.active_from_wave else 0)
            for row in rows
            if row.active_from_wave <= wave <= row.active_through_wave
        )
    projected_peak = max(wave_totals.values(), default=0)
    retained = sum(row.expected_bytes for row in rows if row.retained)
    return wave_totals, projected_peak, retained


def _projection_problems(
    payload: Mapping[str, Any], profile: StorageProfile
) -> list[str]:
    problems: list[str] = []
    peak = int(payload.get("projected_peak_persistent_bytes", -1))
    retained = int(payload.get("projected_final_retained_bytes", -1))
    if payload.get("projection_complete") is not True:
        problems.append("projection is incomplete")
    if peak < 0 or peak > profile.max_projected_peak_bytes:
        problems.append(
            f"projected peak {peak} exceeds gate {profile.max_projected_peak_bytes}"
        )
    if retained < 0 or retained > profile.target_steady_state_bytes:
        problems.append(
            f"projected retained bytes {retained} exceed gate "
            f"{profile.target_steady_state_bytes}"
        )
    if peak > profile.max_persistent_bytes:
        problems.append(
            f"projected peak {peak} exceeds hard cap {profile.max_persistent_bytes}"
        )
    return problems


def write_storage_projection(path: str | Path, payload: Mapping[str, Any]) -> None:
    _require_content_hash(payload, label="storage projection")
    _atomic_json(Path(path), dict(payload))


def require_storage_projection(
    path: str | Path,
    *,
    campaign_root: str | Path,
    campaign_mode: str,
    profile: str | StorageProfile,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"storage projection is missing: {source}")
    payload = _read_mapping(source)
    if payload.get("contract") != ABPH_STORAGE_PROJECTION_CONTRACT:
        raise ValueError("storage projection contract mismatch")
    _require_content_hash(payload, label="storage projection")
    resolved = resolve_storage_profile(profile)
    if payload.get("storage_profile") != resolved.name:
        raise ValueError("storage projection profile mismatch")
    if payload.get("implementation_contract") != resolved.implementation_contract:
        raise ValueError("storage projection implementation contract mismatch")
    if Path(str(payload.get("campaign_root", ""))).resolve() != Path(campaign_root).resolve():
        raise ValueError("storage projection belongs to a different campaign root")
    if payload.get("campaign_mode") != campaign_mode:
        raise ValueError("storage projection campaign mode mismatch")
    if payload.get("limits") != resolved.to_dict():
        raise ValueError("storage projection limits differ from the active profile")
    serialized_rows = payload.get("rows")
    if not isinstance(serialized_rows, list) or not serialized_rows:
        raise ValueError("storage projection contains no measured rows")
    rows = tuple(
        StorageProjectionRow(
            artifact_family=str(row["artifact_family"]),
            artifact_class=StorageArtifactClass(str(row["artifact_class"])),
            expected_bytes=int(row["expected_bytes"]),
            active_from_wave=int(row["active_from_wave"]),
            active_through_wave=int(row["active_through_wave"]),
            retained=bool(row["retained"]),
            atomic_write_overhead_bytes=int(row["atomic_write_overhead_bytes"]),
            measurement_source=str(row["measurement_source"]),
        )
        for row in serialized_rows
    )
    wave_totals, projected_peak, retained = _derive_projection_totals(rows)
    if payload.get("wave_totals_bytes") != wave_totals:
        raise ValueError("storage projection wave totals do not match artifact rows")
    if int(payload.get("projected_peak_persistent_bytes", -1)) != projected_peak:
        raise ValueError("storage projection peak does not match artifact rows")
    if int(payload.get("projected_final_retained_bytes", -1)) != retained:
        raise ValueError("storage projection retained total does not match artifact rows")
    problems = _projection_problems(payload, resolved)
    saved_problems = payload.get("problems")
    if problems or saved_problems or payload.get("ok") is not True:
        raise ValueError("storage projection failed: " + "; ".join(problems or saved_problems or []))
    return payload


def storage_paths(campaign_root: str | Path) -> dict[str, Path]:
    storage = Path(campaign_root).resolve() / "storage"
    return {
        "root": storage,
        "contract": storage / "storage_contract.json",
        "ledger": storage / "quota_ledger.json",
        "lock": storage / "quota_ledger.lock",
        "reservations": storage / "reservations",
        "cleanup_receipts": storage / "cleanup_receipts",
        "audits": storage / "storage_audits",
    }


def _resolved_inside(path: str | Path, root: str | Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{label} escapes campaign root: {resolved}") from exc
    return resolved


@contextmanager
def _exclusive_lock(path: Path, *, timeout_seconds: float = 30.0) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(b"\0")
    with path.open("r+b") as handle:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out acquiring storage quota lock {path}")
                time.sleep(0.05)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def campaign_storage_audit(
    campaign_root: str | Path,
    *,
    additional_roots: Iterable[str | Path] = (),
    profile: str | StorageProfile = ABPH_STREAMING_STORAGE_PROFILE,
) -> dict[str, Any]:
    root = Path(campaign_root).resolve()
    resolved_profile = resolve_storage_profile(profile)
    roots = (root, *(Path(path).resolve() for path in additional_roots))
    seen: set[Path] = set()
    bytes_by_root: dict[str, int] = {}
    bytes_by_top_level: dict[str, int] = {}
    file_count = 0
    total_bytes = 0
    for audit_root in roots:
        root_bytes = 0
        if audit_root.exists():
            for path in audit_root.rglob("*"):
                if not path.is_file():
                    continue
                resolved_path = path.resolve()
                if resolved_path in seen:
                    continue
                seen.add(resolved_path)
                size = path.stat().st_size
                root_bytes += size
                total_bytes += size
                file_count += 1
                if audit_root == root:
                    relative = path.relative_to(root)
                    top = relative.parts[0] if relative.parts else "."
                    bytes_by_top_level[top] = bytes_by_top_level.get(top, 0) + size
        bytes_by_root[str(audit_root)] = root_bytes
    payload: dict[str, Any] = {
        "contract": ABPH_STORAGE_AUDIT_CONTRACT,
        "campaign_root": str(root),
        "storage_profile": resolved_profile.name,
        "audited_roots": [str(path) for path in roots],
        "bytes_by_root": bytes_by_root,
        "bytes_by_campaign_top_level": dict(sorted(bytes_by_top_level.items())),
        "file_count": file_count,
        "measured_persistent_bytes": total_bytes,
        "max_persistent_bytes": resolved_profile.max_persistent_bytes,
        "remaining_bytes": resolved_profile.max_persistent_bytes - total_bytes,
        "ok": total_bytes <= resolved_profile.max_persistent_bytes,
    }
    payload["content_hash"] = _canonical_hash(payload)
    return payload


def _new_ledger(root: Path, profile: StorageProfile) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contract": ABPH_STORAGE_LEDGER_CONTRACT,
        "campaign_root": str(root),
        "storage_profile": profile.name,
        "implementation_contract": profile.implementation_contract,
        "max_persistent_bytes": profile.max_persistent_bytes,
        "revision": 0,
        "active_reservations": {},
        "committed_artifacts": {},
        "last_measured_campaign_bytes": 0,
        "updated_at": _utc_now(),
    }
    payload["content_hash"] = _canonical_hash(payload)
    return payload


def _load_ledger(path: Path, *, root: Path, profile: StorageProfile) -> dict[str, Any]:
    payload = _read_mapping(path)
    if payload.get("contract") != ABPH_STORAGE_LEDGER_CONTRACT:
        raise ValueError("storage quota ledger contract mismatch")
    _require_content_hash(payload, label="storage quota ledger")
    if Path(str(payload.get("campaign_root", ""))).resolve() != root:
        raise ValueError("storage quota ledger campaign root mismatch")
    if payload.get("storage_profile") != profile.name:
        raise ValueError("storage quota ledger profile mismatch")
    if int(payload.get("max_persistent_bytes", -1)) != profile.max_persistent_bytes:
        raise ValueError("storage quota ledger hard cap mismatch")
    if not isinstance(payload.get("active_reservations"), dict):
        raise ValueError("storage quota ledger reservations are malformed")
    if not isinstance(payload.get("committed_artifacts"), dict):
        raise ValueError("storage quota ledger committed artifacts are malformed")
    return payload


def _write_ledger(path: Path, payload: dict[str, Any]) -> None:
    payload["revision"] = int(payload.get("revision", 0)) + 1
    payload["updated_at"] = _utc_now()
    payload["content_hash"] = _canonical_hash(_content_hash_payload(payload))
    _atomic_json(path, payload)


def initialize_storage_accounting(
    campaign_root: str | Path,
    *,
    profile: str | StorageProfile = ABPH_STREAMING_STORAGE_PROFILE,
    projection: Mapping[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Initialize or validate the immutable campaign storage contract."""

    root = Path(campaign_root).resolve()
    resolved_profile = resolve_storage_profile(profile)
    projection_hash = None if projection is None else projection.get("content_hash")
    contract: dict[str, Any] = {
        "contract": ABPH_STORAGE_CONTRACT,
        "implementation_contract": resolved_profile.implementation_contract,
        "campaign_root": str(root),
        "profile": resolved_profile.to_dict(),
        "storage_projection_content_hash": projection_hash,
        "artifact_classes": [item.value for item in StorageArtifactClass],
    }
    contract["content_hash"] = _canonical_hash(contract)
    if dry_run:
        return {"dry_run": True, "contract": contract, "paths_created": []}
    paths = storage_paths(root)
    for key in ("root", "reservations", "cleanup_receipts", "audits"):
        paths[key].mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(paths["lock"]):
        if paths["contract"].exists():
            existing = _read_mapping(paths["contract"])
            _require_content_hash(existing, label="storage contract")
            if existing != contract:
                raise ValueError("existing storage contract differs from active contract")
        else:
            _atomic_json(paths["contract"], contract)
        if paths["ledger"].exists():
            ledger = _load_ledger(paths["ledger"], root=root, profile=resolved_profile)
        else:
            ledger = _new_ledger(root, resolved_profile)
            _write_ledger(paths["ledger"], ledger)
        audit = campaign_storage_audit(root, profile=resolved_profile)
        if not audit["ok"]:
            raise RuntimeError(
                f"campaign already uses {audit['measured_persistent_bytes']} bytes, "
                f"above hard cap {resolved_profile.max_persistent_bytes}"
            )
    return {
        "dry_run": False,
        "contract": contract,
        "ledger_revision": int(ledger["revision"]),
        "measured_persistent_bytes": int(audit["measured_persistent_bytes"]),
        "paths_created": [str(paths[key]) for key in paths if key != "lock"],
    }


def _reservation_remaining_bytes(reservation: Mapping[str, Any]) -> int:
    expected = int(reservation["expected_bytes"])
    temporary = Path(str(reservation["temporary_path"]))
    present = temporary.stat().st_size if temporary.is_file() else 0
    return max(expected - present, 0)


def reserve_persistent_artifact(
    campaign_root: str | Path,
    *,
    destination: str | Path,
    expected_bytes: int,
    artifact_class: StorageArtifactClass | str,
    artifact_role: str,
    source_provenance_hash: str,
    run_id: str,
    job_id: str | None = None,
    expires_after: timedelta = timedelta(days=7),
    profile: str | StorageProfile = ABPH_STREAMING_STORAGE_PROFILE,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Atomically reserve worst-case persistent bytes before opening an output."""

    root = Path(campaign_root).resolve()
    resolved_profile = resolve_storage_profile(profile)
    try:
        resolved_class = StorageArtifactClass(artifact_class)
    except ValueError as exc:
        raise ValueError(f"unknown storage artifact class {artifact_class!r}") from exc
    if resolved_class not in PERSISTENT_ARTIFACT_CLASSES:
        raise PermissionError(f"artifact class {resolved_class.value} cannot be persisted")
    if int(expected_bytes) < 0:
        raise ValueError("expected reservation bytes cannot be negative")
    if not artifact_role or not source_provenance_hash or not run_id:
        raise ValueError("reservation role, provenance hash, and run ID are required")
    resolved_destination = _resolved_inside(destination, root, label="artifact destination")
    reservation_id = uuid.uuid4().hex
    temporary = resolved_destination.with_name(
        f".{resolved_destination.name}.abph-{reservation_id}.tmp"
    )
    now = datetime.now(timezone.utc)
    reservation: dict[str, Any] = {
        "contract": ABPH_STORAGE_RESERVATION_CONTRACT,
        "reservation_id": reservation_id,
        "campaign_root": str(root),
        "job_id": job_id or os.environ.get("SLURM_JOB_ID") or f"pid-{os.getpid()}",
        "run_id": run_id,
        "artifact_role": artifact_role,
        "artifact_class": resolved_class.value,
        "destination": str(resolved_destination),
        "temporary_path": str(temporary),
        "expected_bytes": int(expected_bytes),
        "source_provenance_hash": source_provenance_hash,
        "created_at": now.isoformat(),
        "expires_at": (now + expires_after).isoformat(),
        "status": "dry_run" if dry_run else "reserved",
    }
    reservation["content_hash"] = _canonical_hash(reservation)
    if dry_run:
        return reservation
    paths = storage_paths(root)
    if not paths["ledger"].is_file():
        raise FileNotFoundError("storage accounting must be initialized before reserving")
    with _exclusive_lock(paths["lock"]):
        ledger = _load_ledger(paths["ledger"], root=root, profile=resolved_profile)
        audit = campaign_storage_audit(root, profile=resolved_profile)
        active_remaining = sum(
            _reservation_remaining_bytes(row)
            for row in ledger["active_reservations"].values()
        )
        projected = (
            int(audit["measured_persistent_bytes"])
            + active_remaining
            + int(expected_bytes)
            + ABPH_RESERVATION_METADATA_ALLOWANCE_BYTES
        )
        if projected > resolved_profile.max_persistent_bytes:
            raise RuntimeError(
                "persistent storage reservation rejected: "
                f"measured={audit['measured_persistent_bytes']} "
                f"active_remaining={active_remaining} requested={expected_bytes} "
                f"projected={projected} cap={resolved_profile.max_persistent_bytes}"
            )
        ledger["active_reservations"][reservation_id] = reservation
        ledger["last_measured_campaign_bytes"] = int(audit["measured_persistent_bytes"])
        _atomic_json(paths["reservations"] / f"{reservation_id}.json", reservation)
        _write_ledger(paths["ledger"], ledger)
    return reservation


def commit_persistent_artifact(
    campaign_root: str | Path,
    reservation_id: str,
    *,
    profile: str | StorageProfile = ABPH_STREAMING_STORAGE_PROFILE,
) -> dict[str, Any]:
    """Atomically publish a reserved temporary file and commit measured bytes."""

    root = Path(campaign_root).resolve()
    resolved_profile = resolve_storage_profile(profile)
    paths = storage_paths(root)
    with _exclusive_lock(paths["lock"]):
        ledger = _load_ledger(paths["ledger"], root=root, profile=resolved_profile)
        reservation = ledger["active_reservations"].get(reservation_id)
        if not isinstance(reservation, dict):
            raise KeyError(f"unknown active storage reservation {reservation_id}")
        temporary = _resolved_inside(
            reservation["temporary_path"], root, label="reservation temporary path"
        )
        destination = _resolved_inside(
            reservation["destination"], root, label="reservation destination"
        )
        if not temporary.is_file():
            raise FileNotFoundError(f"reserved temporary artifact is missing: {temporary}")
        actual_bytes = temporary.stat().st_size
        expected_bytes = int(reservation["expected_bytes"])
        if actual_bytes > expected_bytes:
            raise RuntimeError(
                f"artifact uses {actual_bytes} bytes, above reservation {expected_bytes}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
        committed = {
            **reservation,
            "status": "committed",
            "committed_at": _utc_now(),
            "actual_bytes": actual_bytes,
            "sha256": _sha256_file(destination),
        }
        committed["content_hash"] = _canonical_hash(
            {key: value for key, value in committed.items() if key != "content_hash"}
        )
        del ledger["active_reservations"][reservation_id]
        ledger["committed_artifacts"][str(destination)] = committed
        ledger["last_measured_campaign_bytes"] = int(
            campaign_storage_audit(root, profile=resolved_profile)[
                "measured_persistent_bytes"
            ]
        )
        _atomic_json(paths["reservations"] / f"{reservation_id}.json", committed)
        _write_ledger(paths["ledger"], ledger)
    return committed


def release_persistent_reservation(
    campaign_root: str | Path,
    reservation_id: str,
    *,
    profile: str | StorageProfile = ABPH_STREAMING_STORAGE_PROFILE,
    stale_job_state_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Release one reservation; expired foreign jobs require an explicit audit."""

    root = Path(campaign_root).resolve()
    resolved_profile = resolve_storage_profile(profile)
    paths = storage_paths(root)
    with _exclusive_lock(paths["lock"]):
        ledger = _load_ledger(paths["ledger"], root=root, profile=resolved_profile)
        reservation = ledger["active_reservations"].get(reservation_id)
        if not isinstance(reservation, dict):
            raise KeyError(f"unknown active storage reservation {reservation_id}")
        owner = str(reservation["job_id"])
        caller = os.environ.get("SLURM_JOB_ID") or f"pid-{os.getpid()}"
        expired = datetime.fromisoformat(str(reservation["expires_at"])) <= datetime.now(timezone.utc)
        if owner != caller:
            if not expired:
                raise PermissionError("only the owning job may release an active reservation")
            if stale_job_state_audit is None:
                raise PermissionError("stale reservation release requires a Slurm job-state audit")
        temporary = _resolved_inside(
            reservation["temporary_path"], root, label="reservation temporary path"
        )
        temporary.unlink(missing_ok=True)
        released = {
            **reservation,
            "status": "released",
            "released_at": _utc_now(),
            "released_by": caller,
            "stale_job_state_audit": dict(stale_job_state_audit or {}),
        }
        released["content_hash"] = _canonical_hash(
            {key: value for key, value in released.items() if key != "content_hash"}
        )
        del ledger["active_reservations"][reservation_id]
        _atomic_json(paths["reservations"] / f"{reservation_id}.json", released)
        _write_ledger(paths["ledger"], ledger)
    return released


def write_quota_managed_bytes(
    campaign_root: str | Path,
    destination: str | Path,
    data: bytes,
    *,
    artifact_class: StorageArtifactClass | str,
    artifact_role: str,
    source_provenance_hash: str,
    run_id: str,
    profile: str | StorageProfile = ABPH_STREAMING_STORAGE_PROFILE,
) -> dict[str, Any]:
    reservation = reserve_persistent_artifact(
        campaign_root,
        destination=destination,
        expected_bytes=len(data),
        artifact_class=artifact_class,
        artifact_role=artifact_role,
        source_provenance_hash=source_provenance_hash,
        run_id=run_id,
        profile=profile,
    )
    temporary = Path(str(reservation["temporary_path"]))
    try:
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(data)
        return commit_persistent_artifact(
            campaign_root,
            str(reservation["reservation_id"]),
            profile=profile,
        )
    except BaseException:
        try:
            release_persistent_reservation(
                campaign_root,
                str(reservation["reservation_id"]),
                profile=profile,
            )
        except (KeyError, FileNotFoundError):
            pass
        raise


def publish_quota_managed_file(
    campaign_root: str | Path,
    source: str | Path,
    destination: str | Path,
    *,
    artifact_class: StorageArtifactClass | str,
    artifact_role: str,
    source_provenance_hash: str,
    run_id: str,
    profile: str | StorageProfile = ABPH_STREAMING_STORAGE_PROFILE,
    copy_buffer_bytes: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    """Stream one completed ephemeral file through the quota reservation boundary."""

    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"quota publication source is missing: {source_path}")
    if source_path.is_symlink():
        raise ValueError(f"quota publication refuses symbolic links: {source_path}")
    if int(copy_buffer_bytes) <= 0:
        raise ValueError("quota publication copy buffer must be positive")
    expected_bytes = source_path.stat().st_size
    reservation = reserve_persistent_artifact(
        campaign_root,
        destination=destination,
        expected_bytes=expected_bytes,
        artifact_class=artifact_class,
        artifact_role=artifact_role,
        source_provenance_hash=source_provenance_hash,
        run_id=run_id,
        profile=profile,
    )
    temporary = Path(str(reservation["temporary_path"]))
    try:
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with source_path.open("rb") as source_handle, temporary.open("xb") as output:
            shutil.copyfileobj(
                source_handle,
                output,
                length=int(copy_buffer_bytes),
            )
            output.flush()
            os.fsync(output.fileno())
        if source_path.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"quota publication source changed while copying: {source_path}"
            )
        return commit_persistent_artifact(
            campaign_root,
            str(reservation["reservation_id"]),
            profile=profile,
        )
    except BaseException:
        try:
            release_persistent_reservation(
                campaign_root,
                str(reservation["reservation_id"]),
                profile=profile,
            )
        except (KeyError, FileNotFoundError):
            pass
        raise


def write_quota_managed_json(
    campaign_root: str | Path,
    destination: str | Path,
    payload: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    return write_quota_managed_bytes(campaign_root, destination, data, **kwargs)


def write_campaign_storage_audit(
    campaign_root: str | Path,
    *,
    profile: str | StorageProfile = ABPH_STREAMING_STORAGE_PROFILE,
    additional_roots: Iterable[str | Path] = (),
    label: str = "campaign",
) -> tuple[Path, dict[str, Any]]:
    root = Path(campaign_root).resolve()
    audit = campaign_storage_audit(
        root,
        additional_roots=additional_roots,
        profile=profile,
    )
    path = storage_paths(root)["audits"] / f"{label}.json"
    write_quota_managed_json(
        root,
        path,
        audit,
        artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
        artifact_role="storage_audit",
        source_provenance_hash=str(audit["content_hash"]),
        run_id=label,
        profile=profile,
    )
    return path, audit


def reconcile_storage_accounting(
    campaign_root: str | Path,
    *,
    allowed_missing_paths: Iterable[str | Path] = (),
    profile: str | StorageProfile = ABPH_STREAMING_STORAGE_PROFILE,
) -> dict[str, Any]:
    """Reconcile the quota ledger after audited exact-path cleanup."""

    root = Path(campaign_root).resolve()
    resolved_profile = resolve_storage_profile(profile)
    paths = storage_paths(root)
    allowed = {
        _resolved_inside(path, root, label="allowed missing artifact")
        for path in allowed_missing_paths
    }
    with _exclusive_lock(paths["lock"]):
        ledger = _load_ledger(paths["ledger"], root=root, profile=resolved_profile)
        removed: list[str] = []
        verified: list[str] = []
        committed = dict(ledger.get("committed_artifacts", {}))
        for raw_path, row in tuple(committed.items()):
            path = _resolved_inside(raw_path, root, label="committed artifact")
            if not path.is_file():
                if path not in allowed:
                    raise FileNotFoundError(
                        "committed artifact disappeared outside cleanup contract: "
                        f"{path}"
                    )
                committed.pop(raw_path)
                removed.append(str(path))
                continue
            expected_hash = str(row.get("sha256", ""))
            if not expected_hash or _sha256_file(path) != expected_hash:
                raise ValueError(f"committed artifact hash changed: {path}")
            verified.append(str(path))
        ledger["committed_artifacts"] = committed
        ledger["last_measured_campaign_bytes"] = int(
            campaign_storage_audit(root, profile=resolved_profile)[
                "measured_persistent_bytes"
            ]
        )
        ledger["revision"] = int(ledger.get("revision", 0)) + 1
        ledger["updated_at"] = _utc_now()
        _write_ledger(paths["ledger"], ledger)
    return {
        "removed_committed_paths": sorted(removed),
        "verified_committed_paths": sorted(verified),
        "measured_persistent_bytes": int(ledger["last_measured_campaign_bytes"]),
        "ledger_revision": int(ledger["revision"]),
    }


def cleanup_quota_managed_artifact(
    campaign_root: str | Path,
    artifact_path: str | Path,
    *,
    expected_sha256: str,
    expected_artifact_role: str,
    profile: str | StorageProfile = ABPH_STREAMING_STORAGE_PROFILE,
) -> dict[str, Any]:
    """Hash-check, remove, and reconcile one quota-managed artifact."""

    root = Path(campaign_root).resolve()
    resolved_profile = resolve_storage_profile(profile)
    artifact = _resolved_inside(
        artifact_path, root, label="quota-managed cleanup artifact"
    )
    if not expected_sha256 or not expected_artifact_role:
        raise ValueError("cleanup requires an expected hash and artifact role")
    paths = storage_paths(root)
    with _exclusive_lock(paths["lock"]):
        ledger = _load_ledger(paths["ledger"], root=root, profile=resolved_profile)
        committed = dict(ledger.get("committed_artifacts", {}))
        row = committed.get(str(artifact))
        if not isinstance(row, Mapping):
            raise KeyError(f"artifact is not committed in the quota ledger: {artifact}")
        if row.get("artifact_role") != expected_artifact_role:
            raise ValueError(f"quota-managed cleanup role mismatch: {artifact}")
        if row.get("sha256") != expected_sha256:
            raise ValueError(f"quota-managed cleanup ledger hash mismatch: {artifact}")
        if not artifact.is_file() or artifact.is_symlink():
            raise FileNotFoundError(
                f"quota-managed cleanup artifact is absent or unsafe: {artifact}"
            )
        actual_sha256 = _sha256_file(artifact)
        actual_bytes = int(artifact.stat().st_size)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"quota-managed cleanup artifact hash changed: {artifact}")
        if actual_bytes != int(row.get("actual_bytes", -1)):
            raise ValueError(f"quota-managed cleanup artifact size changed: {artifact}")
        artifact.unlink()
        committed.pop(str(artifact))
        ledger["committed_artifacts"] = committed
        ledger["last_measured_campaign_bytes"] = int(
            campaign_storage_audit(root, profile=resolved_profile)[
                "measured_persistent_bytes"
            ]
        )
        _write_ledger(paths["ledger"], ledger)
    return {
        "contract": "adaptive_binary_quota_managed_cleanup_v1",
        "ok": True,
        "campaign_root": str(root),
        "artifact_path": str(artifact),
        "artifact_role": expected_artifact_role,
        "sha256": actual_sha256,
        "bytes": actual_bytes,
        "destination_removed": not artifact.exists(),
        "ledger_reconciled": str(artifact) not in ledger["committed_artifacts"],
        "ledger_revision": int(ledger["revision"]),
        "measured_persistent_bytes": int(ledger["last_measured_campaign_bytes"]),
    }


__all__ = [
    "ABPH_CACHE_HEAVY_STORAGE_PROFILE",
    "ABPH_MAX_PERSISTENT_BYTES",
    "ABPH_MAX_PROJECTED_PEAK_BYTES",
    "ABPH_MINIMUM_SAFETY_HEADROOM_BYTES",
    "ABPH_STORAGE_AUDIT_CONTRACT",
    "ABPH_STORAGE_CONTRACT",
    "ABPH_STORAGE_LEDGER_CONTRACT",
    "ABPH_STORAGE_PROFILE_NAMES",
    "ABPH_STORAGE_PROJECTION_CONTRACT",
    "ABPH_STREAMING_STORAGE_CONTRACT",
    "ABPH_STREAMING_STORAGE_PROFILE",
    "ABPH_TARGET_STEADY_STATE_BYTES",
    "StorageArtifactClass",
    "StorageProfile",
    "StorageProjectionRow",
    "build_storage_projection",
    "campaign_storage_audit",
    "cleanup_quota_managed_artifact",
    "commit_persistent_artifact",
    "initialize_storage_accounting",
    "publish_quota_managed_file",
    "reconcile_storage_accounting",
    "release_persistent_reservation",
    "require_storage_projection",
    "reserve_persistent_artifact",
    "resolve_storage_profile",
    "storage_paths",
    "write_campaign_storage_audit",
    "write_quota_managed_bytes",
    "write_quota_managed_json",
    "write_storage_projection",
]
