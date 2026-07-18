#!/usr/bin/env python3
"""Build projections and operate the ABPH persistent-storage quota ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline.storage_quota import (  # noqa: E402
    ABPH_STORAGE_PROFILE_NAMES,
    ABPH_STREAMING_STORAGE_PROFILE,
    StorageArtifactClass,
    StorageProjectionRow,
    build_storage_projection,
    campaign_storage_audit,
    commit_persistent_artifact,
    initialize_storage_accounting,
    release_persistent_reservation,
    require_storage_projection,
    reserve_persistent_artifact,
    write_campaign_storage_audit,
    write_storage_projection,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    projection = subparsers.add_parser("projection")
    projection.add_argument("--campaign-root", required=True)
    projection.add_argument("--campaign-mode", choices=("pilot", "highdata"), required=True)
    projection.add_argument("--storage-profile", choices=ABPH_STORAGE_PROFILE_NAMES, default=ABPH_STREAMING_STORAGE_PROFILE)
    projection.add_argument("--rows-json", required=True)
    projection.add_argument("--measurement-contract", required=True)
    projection.add_argument("--sample-provenance-hash", required=True)
    projection.add_argument("--output", required=True)

    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--campaign-root", required=True)
    initialize.add_argument("--campaign-mode", choices=("pilot", "highdata"), required=True)
    initialize.add_argument("--storage-profile", choices=ABPH_STORAGE_PROFILE_NAMES, default=ABPH_STREAMING_STORAGE_PROFILE)
    initialize.add_argument("--projection", required=True)
    initialize.add_argument("--dry-run", action="store_true")

    audit = subparsers.add_parser("audit")
    audit.add_argument("--campaign-root", required=True)
    audit.add_argument("--storage-profile", choices=ABPH_STORAGE_PROFILE_NAMES, default=ABPH_STREAMING_STORAGE_PROFILE)
    audit.add_argument("--additional-root", action="append", default=[])
    audit.add_argument("--write-label")

    reserve = subparsers.add_parser("reserve")
    reserve.add_argument("--campaign-root", required=True)
    reserve.add_argument("--storage-profile", choices=ABPH_STORAGE_PROFILE_NAMES, default=ABPH_STREAMING_STORAGE_PROFILE)
    reserve.add_argument("--destination", required=True)
    reserve.add_argument("--expected-bytes", required=True, type=int)
    reserve.add_argument("--artifact-class", choices=tuple(item.value for item in StorageArtifactClass), required=True)
    reserve.add_argument("--artifact-role", required=True)
    reserve.add_argument("--source-provenance-hash", required=True)
    reserve.add_argument("--run-id", required=True)
    reserve.add_argument("--job-id")
    reserve.add_argument("--dry-run", action="store_true")

    commit = subparsers.add_parser("commit")
    commit.add_argument("--campaign-root", required=True)
    commit.add_argument("--storage-profile", choices=ABPH_STORAGE_PROFILE_NAMES, default=ABPH_STREAMING_STORAGE_PROFILE)
    commit.add_argument("--reservation-id", required=True)

    release = subparsers.add_parser("release")
    release.add_argument("--campaign-root", required=True)
    release.add_argument("--storage-profile", choices=ABPH_STORAGE_PROFILE_NAMES, default=ABPH_STREAMING_STORAGE_PROFILE)
    release.add_argument("--reservation-id", required=True)
    release.add_argument("--stale-job-state-audit")
    return parser


def _load_rows(path: str | Path) -> tuple[StorageProjectionRow, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("projection row input must be a list or a mapping with rows")
    return tuple(
        StorageProjectionRow(
            artifact_family=str(row["artifact_family"]),
            artifact_class=StorageArtifactClass(str(row["artifact_class"])),
            expected_bytes=int(row["expected_bytes"]),
            active_from_wave=int(row["active_from_wave"]),
            active_through_wave=int(row["active_through_wave"]),
            retained=bool(row.get("retained", False)),
            atomic_write_overhead_bytes=int(row.get("atomic_write_overhead_bytes", 0)),
            measurement_source=str(row["measurement_source"]),
        )
        for row in rows
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "projection":
        result = build_storage_projection(
            campaign_root=args.campaign_root,
            campaign_mode=args.campaign_mode,
            profile=args.storage_profile,
            rows=_load_rows(args.rows_json),
            measurement_contract=args.measurement_contract,
            sample_provenance_hash=args.sample_provenance_hash,
        )
        write_storage_projection(args.output, result)
        if not result["ok"]:
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2
    elif args.command == "initialize":
        projection = require_storage_projection(
            args.projection,
            campaign_root=args.campaign_root,
            campaign_mode=args.campaign_mode,
            profile=args.storage_profile,
        )
        result = initialize_storage_accounting(
            args.campaign_root,
            profile=args.storage_profile,
            projection=projection,
            dry_run=args.dry_run,
        )
    elif args.command == "audit":
        if args.write_label:
            path, result = write_campaign_storage_audit(
                args.campaign_root,
                profile=args.storage_profile,
                additional_roots=args.additional_root,
                label=args.write_label,
            )
            result = {**result, "output": str(path)}
        else:
            result = campaign_storage_audit(
                args.campaign_root,
                profile=args.storage_profile,
                additional_roots=args.additional_root,
            )
    elif args.command == "reserve":
        result = reserve_persistent_artifact(
            args.campaign_root,
            destination=args.destination,
            expected_bytes=args.expected_bytes,
            artifact_class=args.artifact_class,
            artifact_role=args.artifact_role,
            source_provenance_hash=args.source_provenance_hash,
            run_id=args.run_id,
            job_id=args.job_id,
            profile=args.storage_profile,
            dry_run=args.dry_run,
        )
    elif args.command == "commit":
        result = commit_persistent_artifact(
            args.campaign_root,
            args.reservation_id,
            profile=args.storage_profile,
        )
    else:
        stale_audit = None
        if args.stale_job_state_audit:
            stale_audit = json.loads(
                Path(args.stale_job_state_audit).read_text(encoding="utf-8")
            )
        result = release_persistent_reservation(
            args.campaign_root,
            args.reservation_id,
            profile=args.storage_profile,
            stale_job_state_audit=stale_audit,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
