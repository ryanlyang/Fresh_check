#!/usr/bin/env python3
"""Publish a completed tmpfs tree through the ABPH persistent quota ledger."""

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
    publish_quota_managed_file,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--destination-dir", required=True)
    parser.add_argument("--artifact-role", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    campaign_root = Path(args.campaign_root).resolve()
    source_dir = Path(args.source_dir).resolve()
    destination_dir = Path(args.destination_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"quota publication source tree is missing: {source_dir}")
    try:
        destination_dir.relative_to(campaign_root)
    except ValueError as exc:
        raise ValueError("quota publication destination escapes campaign root") from exc

    files = tuple(
        path
        for path in sorted(source_dir.rglob("*"))
        if path.is_file() and not path.is_symlink()
    )
    if not files:
        raise ValueError(f"quota publication source tree is empty: {source_dir}")
    published: list[dict[str, object]] = []
    skipped: list[str] = []
    for source in files:
        relative = source.relative_to(source_dir)
        destination = destination_dir / relative
        source_hash = _sha256(source)
        if destination.is_file():
            if destination.stat().st_size == source.stat().st_size and _sha256(destination) == source_hash:
                skipped.append(str(destination))
                continue
            if not args.overwrite:
                raise FileExistsError(
                    f"quota publication destination already differs: {destination}"
                )
        receipt = publish_quota_managed_file(
            campaign_root,
            source,
            destination,
            artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
            artifact_role=args.artifact_role,
            source_provenance_hash=source_hash,
            run_id=args.run_id,
            profile=ABPH_STREAMING_STORAGE_PROFILE,
        )
        published.append(
            {
                "source": str(source),
                "destination": str(destination),
                "bytes": int(receipt["actual_bytes"]),
                "sha256": str(receipt["sha256"]),
                "reservation_id": str(receipt["reservation_id"]),
            }
        )
    print(
        json.dumps(
            {
                "ok": True,
                "campaign_root": str(campaign_root),
                "source_dir": str(source_dir),
                "destination_dir": str(destination_dir),
                "artifact_role": args.artifact_role,
                "published": published,
                "skipped": skipped,
                "published_bytes": sum(int(row["bytes"]) for row in published),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
