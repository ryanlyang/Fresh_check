#!/usr/bin/env python3
"""Query or atomically update the authenticated dynamic-job ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import (  # noqa: E402
    load_hashed_json,
    with_content_hash,
)


CONTRACT = "relational_part_dynamic_job_ledger_v1"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--logical-name", required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--dependency")
    parser.add_argument("--query", action="store_true")
    args = parser.parse_args()
    campaign = load_hashed_json(args.campaign_spec)
    if args.ledger.is_file():
        ledger = load_hashed_json(args.ledger, expected_contract=CONTRACT)
        if ledger["campaign_spec_sha256"] != campaign["content_hash"]:
            raise ValueError("dynamic ledger belongs to another campaign")
        jobs = dict(ledger["jobs"])
    else:
        jobs = {}
    if args.query:
        row = jobs.get(args.logical_name)
        if row is not None:
            print(row["job_id"])
        return 0
    if (
        args.job_id is None
        or not args.job_id.isdigit()
        or args.dependency is None
    ):
        raise ValueError("recording requires numeric --job-id and --dependency")
    row = {
        "job_id": args.job_id,
        "dependency": args.dependency,
    }
    existing = jobs.get(args.logical_name)
    if existing is not None and existing != row:
        raise ValueError(
            f"dynamic job {args.logical_name} already has another binding"
        )
    jobs[args.logical_name] = row
    artifact = with_content_hash(
        {
            "contract": CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign["content_hash"],
            "jobs": dict(sorted(jobs.items())),
            "job_count": len(jobs),
            "unique_logical_names": True,
        }
    )
    _write(args.ledger, artifact)
    print(args.job_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
