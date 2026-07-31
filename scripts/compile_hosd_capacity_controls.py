#!/usr/bin/env python3
"""Build the exact HOSD capacity grid or graph-specific control selections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_capacity_grid_artifact,
    build_capacity_control_execution_plan,
    build_graph_capacity_profile,
    compile_graph_capacity_controls,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    CAPACITY_GRID_CONTRACT,
    CAPACITY_PROFILE_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--mode", required=True, choices=("grid", "profile", "select", "execution")
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grid", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--graph-id")
    parser.add_argument("--deployed-parameters", type=int)
    parser.add_argument("--deployed-flops", type=int)
    parser.add_argument("--export-sha256")
    parser.add_argument("--export-manifest", type=Path)
    parser.add_argument("--confirmation-plan", type=Path)
    parser.add_argument("--compilation", action="append", default=[], type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    source = campaign["source"]
    if args.mode == "grid":
        artifact = build_capacity_grid_artifact(source=source)
    elif args.mode == "profile":
        if args.export_manifest is not None:
            export = load_hashed_json(args.export_manifest)
            args.graph_id = export["descriptor"]["graph_id"]
            args.deployed_parameters = export["deployed_trainable_parameters"]
            args.deployed_flops = export[
                "analytical_inference_flops_batch1_n128"
            ]
            args.export_sha256 = export["export_sha256"]
        if (
            args.graph_id is None
            or args.deployed_parameters is None
            or args.deployed_flops is None
            or args.export_sha256 is None
        ):
            parser.error(
                "profile requires --graph-id, --deployed-parameters, "
                "--deployed-flops, and --export-sha256"
            )
        artifact = build_graph_capacity_profile(
            graph_id=args.graph_id,
            deployed_parameter_count=args.deployed_parameters,
            deployed_analytical_flops=args.deployed_flops,
            export_sha256=args.export_sha256,
            source=source,
        )
    elif args.mode == "select":
        if args.grid is None or args.profile is None:
            parser.error("select requires --grid and --profile")
        artifact = compile_graph_capacity_controls(
            graph_profile=load_hashed_json(
                args.profile, expected_contract=CAPACITY_PROFILE_CONTRACT
            ),
            grid=load_hashed_json(
                args.grid, expected_contract=CAPACITY_GRID_CONTRACT
            ),
            source=source,
        )
    else:
        if args.confirmation_plan is None or not args.compilation:
            parser.error("execution requires --confirmation-plan and --compilation")
        artifact = build_capacity_control_execution_plan(
            confirmation_plan=load_hashed_json(
                args.confirmation_plan,
                expected_contract="hosd_confirmation_plan_v2",
            ),
            compilations=[
                load_hashed_json(
                    path,
                    expected_contract="hosd_capacity_control_compilation_v1",
                )
                for path in args.compilation
            ],
            source=source,
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
