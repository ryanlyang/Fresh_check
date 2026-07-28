#!/usr/bin/env python3
"""Persist one bounded streaming-job failure report through the quota ledger."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline.config import canonical_hash
from teacher_logit_reco.adaptive_binary_pseudooffline.storage_quota import (
    ABPH_STREAMING_STORAGE_PROFILE,
    StorageArtifactClass,
    write_quota_managed_json,
)


ABPH_JOB_FAILURE_REPORT_CONTRACT = (
    "adaptive_binary_pseudooffline_job_failure_report_v1"
)
ABPH_JOB_FAILURE_LOG_MAX_BYTES = 256 * 1024


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not cleaned:
        raise ValueError("failure-report path component is empty")
    return cleaned


def _stderr_tail(path: Path, maximum_bytes: int) -> bytes:
    if maximum_bytes <= 0:
        raise ValueError("failure-report byte limit must be positive")
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > maximum_bytes:
            handle.seek(size - maximum_bytes)
        return handle.read(maximum_bytes)


def write_job_failure_report(
    campaign_root: str | Path,
    *,
    variant: str,
    job_id: str,
    exit_status: int,
    stderr_log: str | Path,
    maximum_bytes: int = ABPH_JOB_FAILURE_LOG_MAX_BYTES,
) -> tuple[Path, dict]:
    root = Path(campaign_root).resolve()
    source = Path(stderr_log)
    tail = _stderr_tail(source, int(maximum_bytes))
    source_size = source.stat().st_size
    payload = {
        "contract": ABPH_JOB_FAILURE_REPORT_CONTRACT,
        "campaign_root": str(root),
        "variant": str(variant),
        "slurm_job_id": str(job_id),
        "exit_status": int(exit_status),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stderr_source_bytes": int(source_size),
        "stderr_retained_bytes": len(tail),
        "stderr_truncated": source_size > len(tail),
        "stderr_tail_sha256": hashlib.sha256(tail).hexdigest(),
        "stderr_tail": tail.decode("utf-8", errors="replace"),
    }
    payload["content_hash"] = canonical_hash(payload)
    destination = (
        root
        / "audits"
        / "job_failures"
        / f"{_safe_component(job_id)}_{_safe_component(variant)}.json"
    )
    write_quota_managed_json(
        root,
        destination,
        payload,
        artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
        artifact_role="bounded_job_failure_report",
        source_provenance_hash=str(payload["stderr_tail_sha256"]),
        run_id=f"failure_{_safe_component(job_id)}",
        profile=ABPH_STREAMING_STORAGE_PROFILE,
    )
    return destination, payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--exit-status", required=True, type=int)
    parser.add_argument("--stderr-log", required=True)
    parser.add_argument(
        "--maximum-bytes",
        type=int,
        default=ABPH_JOB_FAILURE_LOG_MAX_BYTES,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    destination, _payload = write_job_failure_report(
        args.campaign_root,
        variant=args.variant,
        job_id=args.job_id,
        exit_status=args.exit_status,
        stderr_log=args.stderr_log,
        maximum_bytes=args.maximum_bytes,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
