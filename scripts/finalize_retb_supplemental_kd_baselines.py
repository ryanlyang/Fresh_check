#!/usr/bin/env python3
"""Publish the six-run conventional KD baseline comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import source_snapshot
from teacher_logit_reco.relation_expert_token_bridge.supplemental_kd_baselines import (
    KD_BASELINE_ARCHITECTURES,
    KD_BASELINE_COORDINATES,
    KD_BASELINE_REPORT_CONTRACT,
    KD_BASELINE_RESULT_CONTRACT,
    KD_BASELINE_RESULT_CONTRACT_V1,
    validate_kd_baseline_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--supplemental-root", required=True, type=Path)
    args = parser.parse_args()
    plan = load_hashed_json(args.plan)
    validate_kd_baseline_plan(plan)
    current_source = source_snapshot(REPO_ROOT)
    rows = []
    for architecture, seed in KD_BASELINE_COORDINATES:
        result_root = (
            args.supplemental_root / "runs_recovery"
            if architecture == "O_FULLREL"
            and (
                args.supplemental_root
                / "runs_recovery"
                / architecture
                / f"seed_{seed}/result.json"
            ).is_file()
            else args.supplemental_root / "runs"
        )
        result_path = result_root / architecture / f"seed_{seed}/result.json"
        result = load_hashed_json(
            result_path
        )
        if result.get("contract") not in {
            KD_BASELINE_RESULT_CONTRACT_V1,
            KD_BASELINE_RESULT_CONTRACT,
        }:
            raise ValueError("supplemental KD result contract differs")
        if (
            result.get("plan_sha256") != plan["content_hash"]
            or result.get("architecture") != architecture
            or int(result.get("seed", -1)) != seed
            or result.get("fixed_budget_completed") is not True
            or result.get("final_test_accessed") is not False
        ):
            raise ValueError("supplemental KD result lineage differs")
        if architecture == "O_FULLREL" and (
            result.get("contract") != KD_BASELINE_RESULT_CONTRACT
            or set(result.get("region_tree_manifest_sha256s", {}))
            != {"model_train", "val_stop", "val_design"}
        ):
            raise ValueError(
                "corrected O_FULLREL KD result lacks complete tree evidence"
            )
        expected_source = (
            current_source if architecture == "O_FULLREL" else plan.get("source")
        )
        if result.get("source") != expected_source:
            raise ValueError(
                f"supplemental KD {architecture} result source differs"
            )
        rows.append(
            {
                "architecture": architecture,
                "seed": seed,
                "result_contract": result["contract"],
                "result_path": str(result_path.resolve()),
                "result_sha256": result["content_hash"],
                "result_source": result.get("source"),
                "region_tree_manifest_sha256s": result.get(
                    "region_tree_manifest_sha256s", {}
                ),
                "val_stop_accuracy": float(result["selected_val_stop"]["accuracy"]),
                "val_stop_cross_entropy": float(
                    result["selected_val_stop"]["cross_entropy"]
                ),
                "val_design_accuracy": float(
                    result["val_design_metrics"]["accuracy"]
                ),
                "val_design_cross_entropy": float(
                    result["val_design_metrics"]["cross_entropy"]
                ),
            }
        )
    summaries = {}
    for architecture in KD_BASELINE_ARCHITECTURES:
        current = [row for row in rows if row["architecture"] == architecture]
        accuracies = [row["val_design_accuracy"] for row in current]
        ces = [row["val_design_cross_entropy"] for row in current]
        baseline = plan["baseline_registrations"][architecture]["selected_val_stop"]
        seed101 = next(row for row in current if row["seed"] == 101)
        summaries[architecture] = {
            "seed_count": len(current),
            "mean_val_design_accuracy": statistics.fmean(accuracies),
            "sample_std_val_design_accuracy": statistics.stdev(accuracies),
            "mean_val_design_cross_entropy": statistics.fmean(ces),
            "sample_std_val_design_cross_entropy": statistics.stdev(ces),
            "parent_ce_seed101_val_stop": baseline,
            "kd_seed101_minus_parent_ce_val_stop_accuracy": (
                seed101["val_stop_accuracy"] - float(baseline["accuracy"])
            ),
            "kd_seed101_minus_parent_ce_val_stop_cross_entropy": (
                seed101["val_stop_cross_entropy"]
                - float(baseline["cross_entropy"])
            ),
        }
    report = bind_source(
        with_content_hash(
            {
                "contract": KD_BASELINE_REPORT_CONTRACT,
                "schema_version": 2,
                "plan_sha256": plan["content_hash"],
                "rows": rows,
                "summaries": summaries,
                "primary_comparison_split": "val_design",
                "checkpoint_selection_split": "val_stop",
                "final_test_accessed": False,
                "underperformance_blocks_completion": False,
            }
        ),
        source_snapshot=current_source,
    )
    output = args.supplemental_root / "reports/kd_baseline_report.json"
    write_immutable_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
