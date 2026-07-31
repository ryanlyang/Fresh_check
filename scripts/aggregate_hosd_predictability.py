#!/usr/bin/env python3
"""Fail-closed aggregation of the complete HOSD Stage-C result matrix."""

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
    BASELINE_COMPLETION_CONTRACT,
    PREDICTABILITY_MATRIX_CONTRACT,
    PROBE_RESULT_CONTRACT,
    STAGE_C_PLAN_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    plan = load_hashed_json(
        args.campaign_root / "job_ledgers" / "stage_c_execution_plan.json",
        expected_contract=STAGE_C_PLAN_CONTRACT,
    )
    baselines = []
    for row in plan["baseline_rows"]:
        path = (
            args.campaign_root / "baselines" / row["baseline_id"]
            / "seed_101" / "baseline_completion.json"
        )
        artifact = load_hashed_json(
            path, expected_contract=BASELINE_COMPLETION_CONTRACT
        )
        if (
            artifact.get("source") != campaign["source"]
            or artifact["baseline_id"] != row["baseline_id"]
        ):
            raise ValueError("baseline completion lineage differs")
        baselines.append(artifact)
    probes = []
    for row in plan["probe_rows"]:
        artifact = load_hashed_json(
            args.campaign_root / "probes" / row["row_id"] / "probe_result.json",
            expected_contract=PROBE_RESULT_CONTRACT,
        )
        if (
            artifact.get("source") != campaign["source"]
            or artifact["row_id"] != row["row_id"]
        ):
            raise ValueError("probe result lineage differs")
        probes.append(artifact)
    matrix = with_content_hash({
        "contract": PREDICTABILITY_MATRIX_CONTRACT,
        "schema_version": 1,
        "source": campaign["source"],
        "campaign_spec_sha256": campaign["content_hash"],
        "stage_c_plan_sha256": plan["content_hash"],
        "baseline_results": [
            {
                "baseline_id": item["baseline_id"],
                "completion_sha256": item["content_hash"],
                "selected_val_stop": item["selected_val_stop"],
            }
            for item in baselines
        ],
        "probe_results": [
            {
                "row_id": item["row_id"],
                "target_id": item["target_id"],
                "probe_kind": item["probe_kind"],
                "tap": item["tap"],
                "metrics": item["metrics"],
                "result_sha256": item["content_hash"],
            }
            for item in probes
        ],
        "baseline_coverage_exact": len(baselines) == len(plan["baseline_rows"]),
        "probe_coverage_exact": len(probes) == len(plan["probe_rows"]),
        "all_targets_continue_to_stage_d": True,
        "performance_used_to_prune": False,
    })
    output = args.output or (
        args.campaign_root / "probes" / "predictability_matrix.json"
    )
    write_immutable_json(output, matrix)
    print(json.dumps({
        "predictability_matrix_sha256": matrix["content_hash"],
        "baseline_count": len(baselines),
        "probe_count": len(probes),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
