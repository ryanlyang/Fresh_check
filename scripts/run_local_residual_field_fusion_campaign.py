#!/usr/bin/env python3
"""Fit one locked group/candidate pair on stack_train/stack_val only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    FUSION_CANDIDATE_IDS, FUSION_GROUP_METHOD, FUSION_GROUP_SEED,
    FusionCandidateRunConfig, run_fusion_candidate,
)
from teacher_logit_reco.local_particle_residual_field.fusion_stability import require_stability_candidate  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--group-id", required=True, choices=(FUSION_GROUP_METHOD, FUSION_GROUP_SEED))
    parser.add_argument("--candidate-id", required=True, choices=FUSION_CANDIDATE_IDS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-sources", required=True)
    parser.add_argument("--source-artifact-audit", required=True)
    parser.add_argument("--feature-root")
    parser.add_argument("--phase", choices=("screening", "stability"), default="screening")
    parser.add_argument("--stability-plan", help="In stability phase, skip candidates outside this frozen union.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--stacker-max-steps", type=int, default=80)
    parser.add_argument("--classwise-steps", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true", help="Debug only; production artifacts are immutable.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stability_plan = args.stability_plan
    delattr(args, "stability_plan")
    if args.phase == "stability":
        if not stability_plan:
            raise ValueError("stability phase requires --stability-plan")
        if not require_stability_candidate(stability_plan, campaign_id=args.campaign_id, candidate_id=args.candidate_id):
            print(json.dumps({"ok": True, "skipped": True, "reason": "not_in_frozen_stability_union",
                              "group_id": args.group_id, "candidate_id": args.candidate_id}, indent=2, sort_keys=True))
            return 0
    report = run_fusion_candidate(FusionCandidateRunConfig(**vars(args)))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
