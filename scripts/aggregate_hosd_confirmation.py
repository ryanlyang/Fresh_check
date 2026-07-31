#!/usr/bin/env python3
"""Compile or aggregate Stage-I three-seed confirmation and controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    aggregate_confirmation,
    build_confirmation_plan,
    build_confirmation_plan_from_registry,
    build_confirmation_result,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    CONFIRMATION_PLAN_CONTRACT,
    CAPACITY_CONTROL_EXECUTION_PLAN_CONTRACT,
    GRAPH_REGISTRY_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("compile", "record-training", "aggregate"),
        required=True,
    )
    parser.add_argument("--graph-registry", type=Path)
    parser.add_argument("--roles-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--training-result", action="append", default=[], type=Path)
    parser.add_argument("--capacity-result", action="append", default=[], type=Path)
    parser.add_argument("--capacity-execution-plan", type=Path)
    parser.add_argument("--row-id")
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--prediction-sha256")
    parser.add_argument("--training-completion-sha256")
    parser.add_argument("--deployable-export-sha256")
    parser.add_argument("--deployable-export-file")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    if args.mode == "compile":
        if args.graph_registry is None:
            raise ValueError("confirmation compilation requires --graph-registry")
        registry = load_hashed_json(
            args.graph_registry, expected_contract=GRAPH_REGISTRY_CONTRACT
        )
        artifact = build_confirmation_plan_from_registry(
            graph_registry=registry,
            parent_lock_hashes=registry["parent_locks"],
            source=campaign["source"],
        )
        output = args.output or (
            args.campaign_root / "confirmation_500k" / "execution_plan.json"
        )
    elif args.mode == "record-training":
        required = (
            args.row_id,
            args.metrics_json,
            args.checkpoint_sha256,
            args.prediction_sha256,
            args.training_completion_sha256,
            args.deployable_export_sha256,
            args.deployable_export_file,
        )
        if any(value is None for value in required):
            raise ValueError("confirmation training record inputs are incomplete")
        plan = load_hashed_json(
            args.campaign_root / "confirmation_500k" / "execution_plan.json",
            expected_contract=CONFIRMATION_PLAN_CONTRACT,
        )
        artifact = build_confirmation_result(
            plan=plan,
            row_id=args.row_id,
            classification_metrics=json.loads(
                args.metrics_json.read_text(encoding="utf-8")
            ),
            checkpoint_sha256=args.checkpoint_sha256,
            prediction_sha256=args.prediction_sha256,
            training_completion_sha256=args.training_completion_sha256,
            deployable_export_sha256=args.deployable_export_sha256,
            deployable_export_file=args.deployable_export_file,
            source=campaign["source"],
        )
        output = args.output or (
            args.campaign_root
            / "confirmation_500k"
            / "results"
            / f"{args.row_id}.json"
        )
    else:
        if args.capacity_execution_plan is None:
            raise ValueError("confirmation aggregation requires capacity execution plan")
        plan = load_hashed_json(
            args.campaign_root / "confirmation_500k" / "execution_plan.json",
            expected_contract=CONFIRMATION_PLAN_CONTRACT,
        )
        artifact = aggregate_confirmation(
            plan=plan,
            training_results=[load_hashed_json(path) for path in args.training_result],
            capacity_results=[load_hashed_json(path) for path in args.capacity_result],
            capacity_execution_plan=load_hashed_json(
                args.capacity_execution_plan,
                expected_contract=CAPACITY_CONTROL_EXECUTION_PLAN_CONTRACT,
            ),
            source=campaign["source"],
        )
        output = args.output or (
            args.campaign_root / "confirmation_500k" / "summary.json"
        )
    publication = write_immutable_json(output, artifact)
    print(json.dumps({"content_hash": artifact["content_hash"], "publication": publication["status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
