#!/usr/bin/env python3
"""Select the paired PT/TRACK Stage-B optimizer without performance gating."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step4 import (  # noqa: E402
    build_locked_optimization_selection,
    validate_optimization_candidate_metrics,
    validate_locked_optimization_selection,
    validate_stage_b_run_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--candidate-metrics", required=True, type=Path)
    parser.add_argument("--baseline-accuracy", type=float)
    parser.add_argument(
        "--capacity-control-reproduces-gain",
        choices=("true", "false", "unknown"),
        default="unknown",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root,
        repo_root=REPO_ROOT,
    )
    authorize_dataset_access(
        worker_role="design_worker",
        requested_resource="val_design",
    )
    registry = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_b_runs.json"
    )
    validate_stage_b_run_registry(registry)
    metrics = load_hashed_json(args.candidate_metrics)
    validate_optimization_candidate_metrics(metrics, run_registry=registry)
    if metrics.get("source") != campaign.get("source"):
        raise ValueError(
            "optimization candidate metrics belong to another source snapshot"
        )
    capacity = (
        None
        if args.capacity_control_reproduces_gain == "unknown"
        else args.capacity_control_reproduces_gain == "true"
    )
    selection = build_locked_optimization_selection(
        candidate_metrics=metrics,
        run_registry=registry,
        source_snapshot=source_snapshot(REPO_ROOT),
        baseline_accuracy=args.baseline_accuracy,
        capacity_control_reproduces_gain=capacity,
    )
    validate_locked_optimization_selection(
        selection,
        candidate_metrics=metrics,
        run_registry=registry,
    )
    output = args.output or (
        args.campaign_root
        / "selection"
        / "retb_stage_b_optimization_selection.json"
    )
    result = {
        "dry_run": bool(args.dry_run),
        "candidate_metrics_sha256": metrics["content_hash"],
        "selected_run_id": selection["selected_run_id"],
        "selected_configuration": selection["selected_configuration"],
        "winner_followup_run_ids": [
            row["run_id"] for row in selection["winner_followup_rows"]
        ],
        "output": str(output.resolve()),
        "selection": selection,
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(output, selection)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
