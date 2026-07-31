#!/usr/bin/env python3
"""Compile the immutable Stage-J teacher/target/student scale execution plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_scale_completion,
    build_scale_execution_plan,
    build_scale_row_result,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    SCALE_SHORTLIST_CONTRACT,
    SCALE_EXECUTION_PLAN_CONTRACT,
    SCALE_ROW_RESULT_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("compile", "record-graph", "finalize"),
        default="compile",
    )
    parser.add_argument("--scale-train-manifest-sha256")
    parser.add_argument("--graph-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--deployable-export-sha256")
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--analytical-forward-flops-json", type=Path)
    parser.add_argument("--teacher-completion", action="append", default=[])
    parser.add_argument("--target-completion", action="append", default=[])
    parser.add_argument(
        "--pre-student-artifact", action="append", default=[]
    )
    parser.add_argument("--graph-result", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    if args.mode == "compile":
        if args.scale_train_manifest_sha256 is None:
            raise ValueError("scale compilation requires the scale-train hash")
        shortlist = load_hashed_json(
            args.campaign_root / "selection" / "locked_scale_shortlist.json",
            expected_contract=SCALE_SHORTLIST_CONTRACT,
        )
        artifact = build_scale_execution_plan(
            shortlist=shortlist,
            scale_train_manifest_sha256=args.scale_train_manifest_sha256,
            source=campaign["source"],
        )
        output = args.output or (
            args.campaign_root / "scale_up" / "execution_plan.json"
        )
    else:
        plan = load_hashed_json(
            args.campaign_root / "scale_up" / "execution_plan.json",
            expected_contract=SCALE_EXECUTION_PLAN_CONTRACT,
        )
        if args.mode == "record-graph":
            required = (
                args.graph_id,
                args.seed,
                args.checkpoint_sha256,
                args.deployable_export_sha256,
                args.metrics_json,
                args.analytical_forward_flops_json,
            )
            if any(value is None for value in required):
                raise ValueError("scale graph record inputs are incomplete")
            artifact = build_scale_row_result(
                scale_plan=plan,
                graph_id=args.graph_id,
                seed=args.seed,
                checkpoint_sha256=args.checkpoint_sha256,
                deployable_export_sha256=args.deployable_export_sha256,
                classification_metrics=json.loads(
                    args.metrics_json.read_text(encoding="utf-8")
                ),
                analytical_forward_flops_by_role=json.loads(
                    args.analytical_forward_flops_json.read_text(
                        encoding="utf-8"
                    )
                ),
                source=campaign["source"],
            )
            output = args.output or (
                args.campaign_root
                / "scale_up"
                / "results"
                / f"{args.graph_id}__seed_{args.seed}.json"
            )
        else:
            def pairs(values: list[str], *, name: str) -> dict[str, str]:
                result = {}
                for value in values:
                    key, separator, digest = value.partition("=")
                    if not separator or not key or key in result:
                        raise ValueError(f"{name} must be unique ID=SHA256")
                    result[key] = digest
                return result

            artifact = build_scale_completion(
                scale_plan=plan,
                teacher_completion_hashes=pairs(
                    args.teacher_completion, name="teacher completion"
                ),
                target_completion_hashes=pairs(
                    args.target_completion, name="target completion"
                ),
                pre_student_artifact_hashes=pairs(
                    args.pre_student_artifact,
                    name="pre-student artifact",
                ),
                graph_results=[
                    load_hashed_json(
                        path, expected_contract=SCALE_ROW_RESULT_CONTRACT
                    )
                    for path in args.graph_result
                ],
                source=campaign["source"],
            )
            output = args.output or (
                args.campaign_root / "scale_up" / "completion.json"
            )
    publication = write_immutable_json(output, artifact)
    print(json.dumps({"content_hash": artifact["content_hash"], "publication": publication["status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
