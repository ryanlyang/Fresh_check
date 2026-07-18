"""Fail-closed Step-10 acceptance for the 30 GB streaming campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .accounting_preflight import ABPH_STEP4_PREFLIGHT_CONTRACT
from .runtime_acceptance import require_runtime_acceptance
from .storage_lifecycle import require_artifact_manifest
from .storage_quota import (
    ABPH_MAX_PERSISTENT_BYTES,
    ABPH_STREAMING_STORAGE_PROFILE,
    StorageArtifactClass,
    require_storage_projection,
    storage_paths,
    write_quota_managed_json,
)
from .target_mode import load_target_mode_selection


ABPH_STORAGE_TEST_EVIDENCE_CONTRACT = (
    "adaptive_binary_pseudooffline_storage_test_evidence_v1"
)
ABPH_STORAGE_ACCEPTANCE_CONTRACT = (
    "adaptive_binary_pseudooffline_storage_acceptance_v1"
)
ABPH_RAM_LIFECYCLE_SMOKE_CONTRACT = (
    "adaptive_binary_pseudooffline_ram_lifecycle_smoke_v1"
)
ABPH_STORAGE_ACCEPTANCE_TEST_FILES: tuple[str, ...] = (
    "tests/test_adaptive_binary_pseudooffline_compact_target_codec.py",
    "tests/test_adaptive_binary_pseudooffline_compact_checkpoints.py",
    "tests/test_adaptive_binary_pseudooffline_pseudo_ram.py",
    "tests/test_adaptive_binary_pseudooffline_production.py",
    "tests/test_adaptive_binary_pseudooffline_step10_tagger.py",
    "tests/test_adaptive_binary_pseudooffline_distributed_stream.py",
    "tests/test_adaptive_binary_pseudooffline_distributed_runtime.py",
    "tests/test_adaptive_binary_pseudooffline_tagger_distributed.py",
    "tests/test_adaptive_binary_pseudooffline_ram_workspace.py",
    "tests/test_adaptive_binary_pseudooffline_storage_quota.py",
    "tests/test_adaptive_binary_pseudooffline_storage_lifecycle.py",
    "tests/test_adaptive_binary_pseudooffline_storage_acceptance.py",
    "tests/test_adaptive_binary_pseudooffline_bundled_scoring.py",
)


def canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a JSON mapping")
    return payload


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_content_hash(payload: Mapping[str, Any], *, label: str) -> None:
    expected = payload.get("content_hash")
    actual = canonical_hash(
        {key: value for key, value in payload.items() if key != "content_hash"}
    )
    if expected != actual:
        raise ValueError(f"{label} content hash mismatch")


def _ram_lifecycle_smoke_problems(
    payload: Mapping[str, Any], *, campaign_root: Path
) -> list[str]:
    problems: list[str] = []
    expected_destination = (
        campaign_root / "storage" / "lifecycle_smoke_payload" / "payload.bin"
    ).resolve()
    destination = Path(str(payload.get("persistent_destination", ""))).resolve()
    if Path(str(payload.get("campaign_root", ""))).resolve() != campaign_root:
        problems.append("RAM lifecycle smoke belongs to another campaign")
    if destination != expected_destination:
        problems.append("RAM lifecycle smoke did not use the locked persistent path")
    if payload.get("persistent_publication_verified") is not True:
        problems.append("RAM lifecycle smoke did not verify persistent publication")
    if payload.get("source_sha256") != payload.get("published_sha256"):
        problems.append("RAM lifecycle smoke publication hash differs from source")
    cleanup = payload.get("persistent_cleanup")
    if not isinstance(cleanup, Mapping):
        problems.append("RAM lifecycle smoke lacks cleanup evidence")
    elif (
        cleanup.get("contract") != "adaptive_binary_quota_managed_cleanup_v1"
        or cleanup.get("ok") is not True
        or cleanup.get("artifact_path") != str(expected_destination)
        or cleanup.get("artifact_role") != "ram_lifecycle_smoke_payload"
        or cleanup.get("sha256") != payload.get("source_sha256")
        or cleanup.get("destination_removed") is not True
        or cleanup.get("ledger_reconciled") is not True
    ):
        problems.append("RAM lifecycle smoke cleanup evidence is invalid")
    if expected_destination.exists():
        problems.append("RAM lifecycle smoke persistent payload was not cleaned")
    ledger_path = storage_paths(campaign_root)["ledger"]
    if not ledger_path.is_file():
        problems.append("RAM lifecycle smoke campaign quota ledger is missing")
    else:
        ledger = _read_json(ledger_path)
        _require_content_hash(ledger, label="storage quota ledger")
        if str(expected_destination) in ledger.get("committed_artifacts", {}):
            problems.append("RAM lifecycle smoke payload remains committed")
        for reservation in ledger.get("active_reservations", {}).values():
            if isinstance(reservation, Mapping) and Path(
                str(reservation.get("destination", ""))
            ).resolve() == expected_destination:
                problems.append("RAM lifecycle smoke reservation remains active")
                break
    return problems


def require_storage_test_evidence(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("contract") != ABPH_STORAGE_TEST_EVIDENCE_CONTRACT:
        raise ValueError("storage acceptance test evidence contract mismatch")
    _require_content_hash(payload, label="storage acceptance test evidence")
    if payload.get("ok") is not True or int(payload.get("return_code", -1)) != 0:
        raise PermissionError("storage acceptance component/parity tests did not pass")
    if tuple(payload.get("test_files", ())) != ABPH_STORAGE_ACCEPTANCE_TEST_FILES:
        raise ValueError("storage acceptance test membership changed")
    if payload.get("source_git_commit") in (None, ""):
        raise ValueError("storage acceptance tests lack source commit identity")
    return payload


def build_storage_acceptance(
    *,
    campaign_root: str | Path,
    campaign_mode: str,
    storage_projection: str | Path,
    target_mode_selection: str | Path,
    target_feasibility: str | Path,
    wave_two_audit: str | Path,
    artifact_manifest: str | Path,
    runtime_acceptance: str | Path,
    test_evidence: str | Path,
    ram_lifecycle_smoke: str | Path,
    source_git_commit: str,
    source_status_hash: str,
) -> dict[str, Any]:
    """Compile the immutable gate that every expensive pilot model depends on."""

    root = Path(campaign_root).resolve()
    problems: list[str] = []
    projection = require_storage_projection(
        storage_projection,
        campaign_root=root,
        campaign_mode=campaign_mode,
        profile=ABPH_STREAMING_STORAGE_PROFILE,
    )
    target_mode = load_target_mode_selection(
        target_mode_selection, campaign_root=root
    )
    feasibility = _read_json(target_feasibility)
    if feasibility.get("contract") != ABPH_STEP4_PREFLIGHT_CONTRACT:
        problems.append("actual-target feasibility contract mismatch")
    if feasibility.get("ok") is not True:
        problems.append("actual-target feasibility failed")
    if feasibility.get("target_mode_selection_hash") != target_mode.get(
        "content_hash"
    ):
        problems.append("target feasibility uses another target-mode selection")

    audit = _read_json(wave_two_audit)
    _require_content_hash(audit, label="Wave-2 storage audit")
    if Path(str(audit.get("campaign_root", ""))).resolve() != root:
        problems.append("Wave-2 storage audit belongs to another campaign")
    if audit.get("storage_profile") != ABPH_STREAMING_STORAGE_PROFILE:
        problems.append("Wave-2 storage audit uses another profile")
    measured_bytes = int(audit.get("measured_persistent_bytes", -1))
    projected_wave_two = int(projection["wave_totals_bytes"].get("2", -1))
    if audit.get("ok") is not True or measured_bytes < 0:
        problems.append("Wave-2 measured storage audit failed")
    if measured_bytes > projected_wave_two:
        problems.append(
            f"Wave-2 measured bytes {measured_bytes} exceed projection "
            f"{projected_wave_two}"
        )
    if measured_bytes > ABPH_MAX_PERSISTENT_BYTES:
        problems.append("Wave-2 storage exceeds the hard 30 GB cap")

    manifest = require_artifact_manifest(root)
    if Path(artifact_manifest).resolve() != (
        root / "storage" / "artifact_manifest.json"
    ).resolve():
        problems.append("artifact manifest is not the canonical campaign artifact")
    forbidden_roles = [
        row.get("role")
        for row in manifest["artifacts"]
        if "pseudo" in str(row.get("role", "")).lower()
    ]
    persistent_pseudo_root = root / "pseudo_predictions"
    persistent_pseudo_files = (
        [str(path) for path in persistent_pseudo_root.rglob("*") if path.is_file()]
        if persistent_pseudo_root.exists()
        else []
    )
    if forbidden_roles or persistent_pseudo_files:
        problems.append("persistent pseudo-view artifacts exist inside the campaign")

    submission_path = root / "submission_logs" / "abph_full_submission.json"
    submission = _read_json(submission_path)
    submitted_commands = submission.get("submission_commands")
    scheduler_io_ok = isinstance(submitted_commands, list) and bool(
        submitted_commands
    )
    if scheduler_io_ok:
        for row in submitted_commands:
            command = row.get("command") if isinstance(row, Mapping) else None
            environment = row.get("environment") if isinstance(row, Mapping) else None
            if (
                not isinstance(command, list)
                or "--output=/dev/null" not in command
                or "--error=/dev/null" not in command
                or not isinstance(environment, Mapping)
                or environment.get("MIRROR_DIAGNOSTICS") != "0"
                or not str(environment.get("LOG_DIR", "")).startswith("/dev/shm/")
                or not str(environment.get("DIAGNOSTICS_ROOT", "")).startswith(
                    "/dev/shm/"
                )
            ):
                scheduler_io_ok = False
                break
    if not scheduler_io_ok:
        problems.append(
            "Slurm logs or mirrored diagnostics are not confined to ephemeral storage"
        )

    workspace = target_mode.get("workspace_capacity")
    if not isinstance(workspace, Mapping):
        problems.append("target-mode selection lacks RAM workspace evidence")
    else:
        reservation_limit = int(workspace.get("reservation_limit_bytes", -1))
        working_set = int(target_mode.get("rank_local_target_working_set_bytes", -1))
        if reservation_limit < 0 or working_set < 0 or working_set > reservation_limit:
            problems.append("rank-local target workspace does not fit its tested limit")
    if float(target_mode.get("minimum_ram_headroom_fraction", -1.0)) < 0.20:
        problems.append("RAM workspace has less than the required 20% headroom")

    runtime_scope = "highdata" if campaign_mode == "highdata" else "ddp4_runtime"
    runtime = require_runtime_acceptance(runtime_acceptance, scope=runtime_scope)
    tests = require_storage_test_evidence(test_evidence)
    ram_smoke = _read_json(ram_lifecycle_smoke)
    _require_content_hash(ram_smoke, label="RAM lifecycle smoke")
    if (
        ram_smoke.get("contract") != ABPH_RAM_LIFECYCLE_SMOKE_CONTRACT
        or ram_smoke.get("ok") is not True
        or ram_smoke.get("problems")
    ):
        problems.append("real tmpfs reserve/build/publish/release smoke failed")
    problems.extend(_ram_lifecycle_smoke_problems(ram_smoke, campaign_root=root))
    if tests.get("source_git_commit") != source_git_commit:
        problems.append("component/parity tests ran against another source commit")
    if tests.get("source_status_hash") != source_status_hash:
        problems.append("component/parity tests ran against another source status")
    if ram_smoke.get("source_git_commit") != source_git_commit:
        problems.append("RAM lifecycle smoke ran against another source commit")
    if ram_smoke.get("source_status_hash") != source_status_hash:
        problems.append("RAM lifecycle smoke ran against another source status")

    payload: dict[str, Any] = {
        "contract": ABPH_STORAGE_ACCEPTANCE_CONTRACT,
        "ok": not problems,
        "campaign_root": str(root),
        "campaign_mode": str(campaign_mode),
        "storage_profile": ABPH_STREAMING_STORAGE_PROFILE,
        "source_git_commit": str(source_git_commit),
        "source_status_hash": str(source_status_hash),
        "projection": {
            "path": str(Path(storage_projection).resolve()),
            "sha256": _sha256(storage_projection),
            "content_hash": projection["content_hash"],
            "projected_peak_persistent_bytes": projection[
                "projected_peak_persistent_bytes"
            ],
            "projected_final_retained_bytes": projection[
                "projected_final_retained_bytes"
            ],
            "projected_wave_two_bytes": projected_wave_two,
        },
        "real_data_storage_smoke": {
            "target_mode_selection": {
                "path": str(Path(target_mode_selection).resolve()),
                "sha256": _sha256(target_mode_selection),
                "content_hash": target_mode["content_hash"],
            },
            "target_mode_selection_hash": target_mode["content_hash"],
            "target_measurement_hash": target_mode["measurement_content_hash"],
            "actual_target_feasibility": {
                "path": str(Path(target_feasibility).resolve()),
                "sha256": _sha256(target_feasibility),
                "contract": feasibility.get("contract"),
                "target_mode_selection_hash": feasibility.get(
                    "target_mode_selection_hash"
                ),
            },
            "actual_target_feasibility_path": str(Path(target_feasibility).resolve()),
            "wave_two_audit": {
                "path": str(Path(wave_two_audit).resolve()),
                "sha256": _sha256(wave_two_audit),
                "content_hash": audit["content_hash"],
            },
            "wave_two_audit_path": str(Path(wave_two_audit).resolve()),
            "wave_two_audit_hash": audit["content_hash"],
            "measured_wave_two_bytes": measured_bytes,
            "ram_workspace": dict(workspace or {}),
            "minimum_ram_headroom_fraction": target_mode.get(
                "minimum_ram_headroom_fraction"
            ),
            "persistent_pseudo_files": persistent_pseudo_files,
        },
        "artifact_manifest": {
            "path": str(Path(artifact_manifest).resolve()),
            "sha256": _sha256(artifact_manifest),
            "content_hash": manifest["content_hash"],
        },
        "artifact_manifest_hash": manifest["content_hash"],
        "scheduler_io_contract": {
            "path": str(submission_path.resolve()),
            "sha256": _sha256(submission_path),
            "job_count": len(submitted_commands or ()),
            "slurm_stdout_stderr_persistent": False,
            "diagnostics_mirrored_persistently": False,
        },
        "runtime_acceptance": {
            "path": str(Path(runtime_acceptance).resolve()),
            "sha256": _sha256(runtime_acceptance),
            "required_scope": runtime_scope,
            "content_hash": runtime["acceptance_content_hash"],
        },
        "component_parity_tests": {
            "path": str(Path(test_evidence).resolve()),
            "sha256": _sha256(test_evidence),
            "content_hash": tests["content_hash"],
            "passed": tests.get("passed"),
            "skipped": tests.get("skipped"),
        },
        "ram_lifecycle_smoke": {
            "path": str(Path(ram_lifecycle_smoke).resolve()),
            "sha256": _sha256(ram_lifecycle_smoke),
            "content_hash": ram_smoke["content_hash"],
            "source_bytes": ram_smoke.get("source_bytes"),
        },
        "gates": {
            "projection_within_24gb": True,
            "retained_projection_within_20gb": True,
            "measured_within_projection": measured_bytes <= projected_wave_two,
            "measured_within_30gb": measured_bytes <= ABPH_MAX_PERSISTENT_BYTES,
            "ram_headroom_at_least_20_percent": float(
                target_mode.get("minimum_ram_headroom_fraction", -1.0)
            )
            >= 0.20,
            "no_persistent_pseudo_cache": not forbidden_roles
            and not persistent_pseudo_files,
            "actual_targets_feasible": feasibility.get("ok") is True,
            "scheduler_io_ephemeral": scheduler_io_ok,
            "runtime_accepted": True,
            "component_parity_tests_passed": True,
            "real_tmpfs_lifecycle_smoke_passed": ram_smoke.get("ok") is True,
        },
        "problems": problems,
    }
    payload["content_hash"] = canonical_hash(payload)
    return payload


def write_storage_acceptance(
    campaign_root: str | Path,
    destination: str | Path,
    payload: Mapping[str, Any],
) -> None:
    _require_content_hash(payload, label="storage acceptance")
    write_quota_managed_json(
        campaign_root,
        destination,
        payload,
        artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
        artifact_role="storage_campaign_acceptance",
        source_provenance_hash=str(payload["content_hash"]),
        run_id="step10_storage_acceptance",
        profile=ABPH_STREAMING_STORAGE_PROFILE,
    )


def require_storage_acceptance(
    path: str | Path,
    *,
    campaign_root: str | Path,
    campaign_mode: str = "pilot",
) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("contract") != ABPH_STORAGE_ACCEPTANCE_CONTRACT:
        raise ValueError("storage acceptance contract mismatch")
    _require_content_hash(payload, label="storage acceptance")
    if Path(str(payload.get("campaign_root", ""))).resolve() != Path(
        campaign_root
    ).resolve():
        raise ValueError("storage acceptance belongs to another campaign")
    if payload.get("campaign_mode") != campaign_mode:
        raise ValueError("storage acceptance campaign mode mismatch")
    if payload.get("ok") is not True or payload.get("problems"):
        raise PermissionError("storage-constrained campaign acceptance failed")
    root = Path(campaign_root).resolve()
    real_data = payload.get("real_data_storage_smoke")
    if not isinstance(real_data, Mapping):
        raise ValueError("storage acceptance lacks real-data evidence")
    evidence_rows = (
        payload.get("projection"),
        payload.get("runtime_acceptance"),
        payload.get("component_parity_tests"),
        payload.get("ram_lifecycle_smoke"),
        real_data.get("target_mode_selection"),
        real_data.get("actual_target_feasibility"),
        real_data.get("wave_two_audit"),
        payload.get("artifact_manifest"),
        payload.get("scheduler_io_contract"),
    )
    for row in evidence_rows:
        if not isinstance(row, Mapping):
            raise ValueError("storage acceptance contains incomplete evidence rows")
        evidence = Path(str(row.get("path", "")))
        expected = str(row.get("sha256", ""))
        if not evidence.is_file() or not expected or _sha256(evidence) != expected:
            raise ValueError(f"storage acceptance evidence changed: {evidence}")

    projection = require_storage_projection(
        payload["projection"]["path"],
        campaign_root=root,
        campaign_mode=campaign_mode,
        profile=ABPH_STREAMING_STORAGE_PROFILE,
    )
    if projection.get("content_hash") != payload["projection"].get("content_hash"):
        raise ValueError("storage acceptance projection identity changed")
    target_mode = load_target_mode_selection(
        real_data["target_mode_selection"]["path"], campaign_root=root
    )
    if target_mode.get("content_hash") != real_data["target_mode_selection"].get(
        "content_hash"
    ):
        raise ValueError("storage acceptance target-mode identity changed")
    feasibility = _read_json(real_data["actual_target_feasibility"]["path"])
    if (
        feasibility.get("contract") != ABPH_STEP4_PREFLIGHT_CONTRACT
        or feasibility.get("ok") is not True
        or feasibility.get("target_mode_selection_hash")
        != target_mode.get("content_hash")
    ):
        raise ValueError("storage acceptance target feasibility is stale or invalid")
    audit = _read_json(real_data["wave_two_audit"]["path"])
    _require_content_hash(audit, label="Wave-2 storage audit")
    if (
        audit.get("content_hash") != real_data["wave_two_audit"].get("content_hash")
        or Path(str(audit.get("campaign_root", ""))).resolve() != root
        or audit.get("storage_profile") != ABPH_STREAMING_STORAGE_PROFILE
        or audit.get("ok") is not True
    ):
        raise ValueError("storage acceptance Wave-2 audit is stale or invalid")
    manifest = require_artifact_manifest(root)
    if manifest.get("content_hash") != payload["artifact_manifest"].get(
        "content_hash"
    ):
        raise ValueError("storage acceptance artifact manifest identity changed")
    ram_smoke = _read_json(payload["ram_lifecycle_smoke"]["path"])
    _require_content_hash(ram_smoke, label="RAM lifecycle smoke")
    if (
        ram_smoke.get("contract") != ABPH_RAM_LIFECYCLE_SMOKE_CONTRACT
        or ram_smoke.get("ok") is not True
        or ram_smoke.get("content_hash")
        != payload["ram_lifecycle_smoke"].get("content_hash")
    ):
        raise ValueError("storage acceptance RAM lifecycle smoke is stale or invalid")
    smoke_problems = _ram_lifecycle_smoke_problems(ram_smoke, campaign_root=root)
    if smoke_problems:
        raise ValueError(
            "storage acceptance RAM lifecycle smoke cleanup is stale or invalid: "
            + "; ".join(smoke_problems)
        )
    return payload


__all__ = [
    "ABPH_STORAGE_ACCEPTANCE_CONTRACT",
    "ABPH_RAM_LIFECYCLE_SMOKE_CONTRACT",
    "ABPH_STORAGE_ACCEPTANCE_TEST_FILES",
    "ABPH_STORAGE_TEST_EVIDENCE_CONTRACT",
    "build_storage_acceptance",
    "require_storage_acceptance",
    "require_storage_test_evidence",
    "write_storage_acceptance",
]
