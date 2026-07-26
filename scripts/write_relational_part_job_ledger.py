#!/usr/bin/env python3
"""Publish authenticated initial or completed Slurm job ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import (  # noqa: E402
    build_job_ledger,
    load_hashed_json,
    sha256_file,
    with_content_hash,
    write_immutable_json,
)


FINAL_LEDGER_CONTRACT = "relational_part_completed_job_ledger_v1"


def _pairs(values: Sequence[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("job values must use NAME=ID")
        name, job_id = value.split("=", 1)
        if not name or name in result or not job_id.isdigit():
            raise ValueError(f"invalid job binding {value!r}")
        result[name] = job_id
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-graph", type=Path, required=True)
    parser.add_argument("--job", action="append", default=[])
    parser.add_argument("--dynamic-ledger", type=Path)
    parser.add_argument("--initial-ledger", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args(argv)
    graph = load_hashed_json(args.production_graph)
    jobs = _pairs(args.job)
    if not args.final:
        artifact = build_job_ledger(
            production_graph=graph,
            jobs=jobs,
            submission_mode="submitted",
        )
    else:
        dynamic_rows = []
        if args.dynamic_ledger is not None and args.dynamic_ledger.is_file():
            for line in args.dynamic_ledger.read_text(
                encoding="utf-8"
            ).splitlines():
                if not line.strip():
                    continue
                fields = line.split("\t")
                if len(fields) != 3 or not fields[1].isdigit():
                    raise ValueError("dynamic job ledger row is malformed")
                dynamic_rows.append(
                    {
                        "logical_name": fields[0],
                        "job_id": fields[1],
                        "dependency": fields[2],
                    }
                )
        if args.report_json is None or not args.report_json.is_file():
            raise FileNotFoundError("completed ledger requires the final report")
        artifact = with_content_hash(
            {
                "contract": FINAL_LEDGER_CONTRACT,
                "schema_version": 1,
                "production_graph_sha256": graph["content_hash"],
                "campaign_id": graph["campaign_id"],
                "campaign_root": graph["campaign_root"],
                "initial_jobs": jobs,
                "initial_submission_ledger_sha256": (
                    None
                    if args.initial_ledger is None
                    else load_hashed_json(args.initial_ledger)[
                        "content_hash"
                    ]
                ),
                "dynamic_jobs": dynamic_rows,
                "final_report_file_sha256": sha256_file(args.report_json),
                "workflow_reached_final_report": True,
            }
        )
    publication = write_immutable_json(args.output, artifact)
    print(
        json.dumps(
            {"artifact": artifact, "publication": publication},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
