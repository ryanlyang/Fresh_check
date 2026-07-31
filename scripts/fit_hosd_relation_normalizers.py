#!/usr/bin/env python3
"""Dry-run or submit shared HLT/offline relation-normalizer rebuilds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation.parent_submission import (  # noqa: E402
    build_parent_submission_plan,
    finalize_parent_group,
    plan_json,
    prepare_shared_parent_runtime,
    submit_parent_plan,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    if args.submit:
        prepare_shared_parent_runtime(
            campaign_root=args.campaign_root, repo_root=REPO_ROOT
        )
    plan = build_parent_submission_plan(
        campaign_root=args.campaign_root,
        repo_root=REPO_ROOT,
        group="normalization",
    )
    jobs = submit_parent_plan(plan) if args.submit else None
    if jobs is not None:
        finalize_parent_group(
            plan, campaign_root=args.campaign_root, submitted_job_ids=jobs
        )
    print(plan_json(plan, submitted_job_ids=jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
