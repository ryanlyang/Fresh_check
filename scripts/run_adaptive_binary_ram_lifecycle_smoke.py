#!/usr/bin/env python3
"""Prepare or verify the real tmpfs reserve/build/publish/release smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline.storage_quota import (  # noqa: E402
    ABPH_STREAMING_STORAGE_PROFILE,
    StorageArtifactClass,
    cleanup_quota_managed_artifact,
    storage_paths,
    write_quota_managed_json,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.storage_acceptance import (  # noqa: E402
    ABPH_RAM_LIFECYCLE_SMOKE_CONTRACT,
)


SMOKE_BYTES = 256 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


SMOKE_ARTIFACT_ROLE = "ram_lifecycle_smoke_payload"


def _paths(workspace: Path, campaign_root: Path) -> tuple[Path, Path, Path]:
    stage = workspace / "codec_scratch" / "ram_lifecycle_stage"
    source = stage / "payload.bin"
    destination = campaign_root / "storage" / "lifecycle_smoke_payload" / "payload.bin"
    return stage, source, destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--workspace", required=True)
    prepare.add_argument("--campaign-root", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--workspace", required=True)
    verify.add_argument("--campaign-root", required=True)
    verify.add_argument("--output", required=True)
    verify.add_argument("--source-git-commit", required=True)
    verify.add_argument("--source-status-hash", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace = Path(args.workspace).resolve()
    campaign_root = Path(args.campaign_root).resolve()
    stage, source, destination = _paths(workspace, campaign_root)
    if args.command == "prepare":
        stage.mkdir(parents=True, exist_ok=False)
        source.write_bytes(bytes(range(256)) * (SMOKE_BYTES // 256))
        if not storage_paths(campaign_root)["ledger"].is_file():
            raise FileNotFoundError(
                "real campaign storage accounting must exist before lifecycle smoke"
            )
        print(
            json.dumps(
                {
                    "stage": str(stage),
                    "source": str(source),
                    "destination": str(destination),
                },
                sort_keys=True,
            )
        )
        return 0

    workspace_manifest = json.loads(
        (workspace / "workspace_manifest.json").read_text(encoding="utf-8")
    )
    ledger = json.loads(
        storage_paths(campaign_root)["ledger"].read_text(encoding="utf-8")
    )
    problems: list[str] = []
    if workspace_manifest.get("reservations"):
        problems.append("RAM reservation remained active after release")
    if ledger.get("active_reservations"):
        problems.append("persistent publication reservation remained active")
    if not source.is_file() or not destination.is_file():
        problems.append("smoke source or published artifact is missing")
    elif _sha256(source) != _sha256(destination):
        problems.append("published smoke artifact differs from its tmpfs source")
    committed = ledger.get("committed_artifacts", {})
    committed_row = committed.get(str(destination))
    if not isinstance(committed_row, dict):
        problems.append("published smoke artifact is absent from the quota ledger")
    elif committed_row.get("artifact_role") != SMOKE_ARTIFACT_ROLE:
        problems.append("published smoke artifact has the wrong quota role")
    source_sha256 = _sha256(source) if source.is_file() else None
    published_sha256 = _sha256(destination) if destination.is_file() else None
    cleanup: dict | None = None
    if not problems and source_sha256 is not None:
        cleanup = cleanup_quota_managed_artifact(
            campaign_root,
            destination,
            expected_sha256=source_sha256,
            expected_artifact_role=SMOKE_ARTIFACT_ROLE,
            profile=ABPH_STREAMING_STORAGE_PROFILE,
        )
        if (
            cleanup.get("ok") is not True
            or cleanup.get("destination_removed") is not True
            or cleanup.get("ledger_reconciled") is not True
        ):
            problems.append("persistent smoke cleanup contract failed")
    payload = {
        "contract": ABPH_RAM_LIFECYCLE_SMOKE_CONTRACT,
        "ok": not problems,
        "campaign_root": str(campaign_root),
        "workspace_filesystem_required": "tmpfs_or_ramfs",
        "reservation_released_before_verification": True,
        "persistent_destination": str(destination),
        "persistent_publication_verified": isinstance(committed_row, dict)
        and source_sha256 == published_sha256,
        "persistent_cleanup": cleanup,
        "source_bytes": source.stat().st_size if source.is_file() else -1,
        "source_sha256": source_sha256,
        "published_sha256": published_sha256,
        "source_git_commit": args.source_git_commit,
        "source_status_hash": args.source_status_hash,
        "problems": problems,
    }
    payload["content_hash"] = _canonical_hash(payload)
    write_quota_managed_json(
        campaign_root,
        args.output,
        payload,
        artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
        artifact_role="ram_lifecycle_smoke_evidence",
        source_provenance_hash=payload["content_hash"],
        run_id="ram_lifecycle_smoke",
        profile=ABPH_STREAMING_STORAGE_PROFILE,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
