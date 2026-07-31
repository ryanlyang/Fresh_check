#!/usr/bin/env python3
"""Freeze Stage-D phase-two coordinates or the final single-family choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    authorize_access,
    build_single_family_phase_lock,
    build_single_family_selection,
    load_and_validate_campaign,
    validate_stage_d_plan,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    AUXILIARY_PREDICTION_CONTRACT,
    SINGLE_FAMILY_PHASE_LOCK_CONTRACT,
    STAGE_D_PLAN_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--mode", required=True, choices=("phase-lock", "final-selection")
    )
    parser.add_argument("--result", action="append", default=[], type=Path)
    parser.add_argument("--phase-lock", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    authorize_access(
        worker_role="design_selector",
        requested_resource="design_select_predictions",
    )
    plan = load_hashed_json(
        args.campaign_root / "job_ledgers" / "stage_d_execution_plan.json",
        expected_contract=STAGE_D_PLAN_CONTRACT,
    )
    validate_stage_d_plan(plan)
    if plan.get("source") != campaign["source"]:
        raise ValueError("Stage-D plan source differs")
    results = [
        load_hashed_json(path, expected_contract=AUXILIARY_PREDICTION_CONTRACT)
        for path in args.result
    ]
    if any(result.get("source") != campaign["source"] for result in results):
        raise ValueError("Stage-D result source differs")
    if args.mode == "phase-lock":
        artifact = build_single_family_phase_lock(
            stage_d_plan=plan,
            primary_results=results,
            source=campaign["source"],
        )
        output = args.output or (
            args.campaign_root / "auxiliary" / "single_family_phase_lock.json"
        )
    else:
        if args.phase_lock is None:
            raise ValueError("final selection requires --phase-lock")
        lock = load_hashed_json(
            args.phase_lock, expected_contract=SINGLE_FAMILY_PHASE_LOCK_CONTRACT
        )
        artifact = build_single_family_selection(
            stage_d_plan=plan,
            phase_lock=lock,
            results=results,
            source=campaign["source"],
        )
        output = args.output or (
            args.campaign_root / "auxiliary" / "locked_single_family_choices.json"
        )
    publication = write_immutable_json(output, artifact)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "output": str(output.resolve()),
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
