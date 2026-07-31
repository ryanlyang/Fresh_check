#!/usr/bin/env python3
"""Compile deployable capacity evidence for every scaled stack candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    SCALE_EXECUTION_PLAN_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    plan = load_hashed_json(
        root / "scale_up" / "execution_plan.json",
        expected_contract=SCALE_EXECUTION_PLAN_CONTRACT,
    )
    audit = load_hashed_json(root / "scale_up" / "export_audit.json")
    by_graph = {}
    for row in plan["graph_rows"]:
        graph_id = str(row["graph_id"])
        export = load_hashed_json(
            root
            / "scale_up"
            / "exports"
            / f"{graph_id}__seed_{row['seed']}.pt.json"
        )
        efficiency = load_hashed_json(
            root
            / "scale_up"
            / "efficiency"
            / f"{graph_id}__seed_{row['seed']}.json"
        )
        if (
            export.get("source") != campaign["source"]
            or export.get("hlt_only") is not True
            or export.get("research_export_logits_parity") is not True
            or efficiency.get("source") != campaign["source"]
            or efficiency.get("graph_id") != graph_id
            or efficiency.get("export_sha256")
            != export["export_sha256"]
        ):
            raise ValueError("stack capacity export differs")
        candidate = {
            "deployable": True,
            "export_parity_validated": True,
            "inference_flops": int(
                export["analytical_inference_flops_batch1_n128"]
            ),
            "deployed_parameters": int(
                export["deployed_trainable_parameters"]
            ),
            "lineage_hashes": {
                "scale_execution_plan": plan["content_hash"],
                "scale_export_audit": audit["content_hash"],
                f"efficiency_seed_{row['seed']}": efficiency[
                    "content_hash"
                ],
            },
            "latency_diagnostic_only": True,
        }
        if graph_id in by_graph:
            prior = by_graph[graph_id]
            if {
                key: value
                for key, value in prior.items()
                if key != "lineage_hashes"
            } != {
                key: value
                for key, value in candidate.items()
                if key != "lineage_hashes"
            }:
                raise ValueError("stack capacity differs between seeds")
            prior["lineage_hashes"].update(candidate["lineage_hashes"])
        else:
            by_graph[graph_id] = candidate
    output = args.output or (
        root / "selection_predictions" / "stack_val" / "capacity.json"
    )
    artifact = with_content_hash(
        {
            "contract": "hosd_stack_capacity_v1",
            "schema_version": 1,
            "source": dict(campaign["source"]),
            "scale_execution_plan_sha256": plan["content_hash"],
            "scale_export_audit_sha256": audit["content_hash"],
            "capacity_by_graph": by_graph,
            "graph_count": len(by_graph),
            "coverage_exact": True,
        }
    )
    publication = write_immutable_json(output, artifact)
    print(
        json.dumps(
            {"graph_count": len(by_graph), "publication": publication["status"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
