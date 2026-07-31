#!/usr/bin/env python3
"""Compile authenticated production resources from completed miniature evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_resource_measurements,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    PRODUCTION_EXECUTION_PLAN_CONTRACT,
    RESOURCE_MEASUREMENT_EVIDENCE_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--miniature-execution-plan", required=True, type=Path)
    parser.add_argument("--measurement-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(
        args.campaign_root, repo_root=REPO_ROOT
    )
    miniature = load_hashed_json(
        args.miniature_execution_plan,
        expected_contract=PRODUCTION_EXECUTION_PLAN_CONTRACT,
    )
    if (
        miniature.get("profile") != "miniature_test"
        or miniature.get("source") != campaign["source"]
    ):
        raise ValueError("resource measurements require current miniature plan")
    raw = load_hashed_json(
        args.measurement_evidence,
        expected_contract=RESOURCE_MEASUREMENT_EVIDENCE_CONTRACT,
    )
    if (
        raw.get("source") != campaign["source"]
        or raw.get("miniature_execution_plan_sha256")
        != miniature["content_hash"]
        or raw.get("hand_authored_measurements") is not False
    ):
        raise ValueError("miniature resource measurement evidence differs")
    artifact = build_resource_measurements(
        miniature_execution_plan_sha256=miniature["content_hash"],
        scheduler_evidence_sha256=raw["scheduler_evidence_sha256"],
        requests_by_class=raw["requests_by_class"],
        projected_target_extraction_seconds=raw[
            "projected_target_extraction_seconds"
        ],
        projected_gpu_hours_by_stage=raw["projected_gpu_hours_by_stage"],
        maximum_concurrent_jobs=raw["maximum_concurrent_jobs"],
        checkpoint_bytes=raw["checkpoint_bytes"],
        export_bytes=raw["export_bytes"],
        scale_resident_layout_ledger=raw["scale_resident_layout_ledger"],
        scale_resident_memory_projections=raw[
            "scale_resident_memory_projections"
        ],
        measurement_evidence_sha256=raw["content_hash"],
        source=campaign["source"],
    )
    publication = write_immutable_json(args.output, artifact)
    print(
        json.dumps(
            {
                "content_hash": artifact["content_hash"],
                "publication": publication["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
