#!/usr/bin/env python3
"""Compile every graph-specific Stage-I capacity control without performance reads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_capacity_control_execution_plan,
    build_capacity_grid_artifact,
    build_graph_capacity_profile,
    compile_graph_capacity_controls,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    CONFIRMATION_PLAN_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    plan = load_hashed_json(
        root / "confirmation_500k" / "execution_plan.json",
        expected_contract=CONFIRMATION_PLAN_CONTRACT,
    )
    if plan.get("source") != campaign["source"]:
        raise ValueError("capacity-wave confirmation source differs")
    export_root = root / "confirmation_500k" / "discovery_exports"
    expected_graphs = sorted(
        {str(row["parent_graph_id"]) for row in plan["capacity_control_rows"]}
    )
    export_by_graph = {}
    for path in sorted(export_root.glob("*.pt.json")):
        manifest = load_hashed_json(path)
        descriptor = manifest.get("descriptor", {})
        graph_id = str(descriptor.get("graph_id", ""))
        if graph_id in expected_graphs:
            if graph_id in export_by_graph:
                raise ValueError("discovery export graph is duplicated")
            export_by_graph[graph_id] = manifest
    if set(export_by_graph) != set(expected_graphs):
        missing = sorted(set(expected_graphs) - set(export_by_graph))
        raise FileNotFoundError(
            f"capacity-wave discovery export coverage differs: {missing}"
        )
    grid = build_capacity_grid_artifact(source=campaign["source"])
    compilations = []
    if not args.dry_run:
        write_immutable_json(
            root / "confirmation_500k" / "capacity_grid.json", grid
        )
    for graph_id in expected_graphs:
        export = export_by_graph[graph_id]
        if (
            export.get("source") != campaign["source"]
            or not bool(export.get("hlt_only"))
            or export.get("forbidden_runtime_dependencies")
        ):
            raise ValueError("capacity-wave discovery export is not deployable")
        profile = build_graph_capacity_profile(
            graph_id=graph_id,
            deployed_parameter_count=int(export["deployed_trainable_parameters"]),
            deployed_analytical_flops=int(
                export["analytical_inference_flops_batch1_n128"]
            ),
            export_sha256=str(export["export_sha256"]),
            source=campaign["source"],
        )
        compilation = compile_graph_capacity_controls(
            graph_profile=profile,
            grid=grid,
            source=campaign["source"],
        )
        compilations.append(compilation)
        if not args.dry_run:
            graph_root = root / "confirmation_500k" / "capacity" / graph_id
            write_immutable_json(graph_root / "profile.json", profile)
            write_immutable_json(graph_root / "compilation.json", compilation)
    execution = build_capacity_control_execution_plan(
        confirmation_plan=plan,
        compilations=compilations,
        source=campaign["source"],
    )
    output = args.output or (
        root / "confirmation_500k" / "capacity_execution_plan.json"
    )
    publication = (
        {"status": "dry_run"}
        if args.dry_run
        else write_immutable_json(output, execution)
    )
    print(
        json.dumps(
            {
                "content_hash": execution["content_hash"],
                "graph_count": len(expected_graphs),
                "row_count": execution["row_count"],
                "publication": publication["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
