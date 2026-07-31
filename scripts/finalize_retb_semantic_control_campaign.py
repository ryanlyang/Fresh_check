#!/usr/bin/env python3
"""Aggregate bypass, substitution, and reconstruction semantic controls."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.execute_retb_semantic_control_campaign import CONTRACT, _run_id  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.contracts import bind_source, load_hashed_json, with_content_hash, write_immutable_json  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.provenance import source_snapshot  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.step7 import STAGE_E_SHAPES  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.workflow import load_and_validate_campaign_source  # noqa: E402


def _metric_path(root: Path, run_id: str) -> Path:
    run_root = root / "runs" / "final_consumers" / run_id
    return (
        run_root / "val_design" / "metrics.json"
        if (run_root / "val_design" / "metrics.json").is_file()
        else run_root / "reference_metrics.json"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    rows = []
    for role in STAGE_E_SHAPES:
        refiner_lock = load_hashed_json(
            root / "selection" / "final_consumers" / role / "token_refiner_lock.json"
        )
        for seed in (101, 202, 303):
            bypass = load_hashed_json(
                root / "controls" / "semantics" / "bypass" / role / f"seed_{seed}" / "bypass_controls.json"
            )
            if bypass.get("source") != campaign.get("source"):
                raise ValueError("semantic bypass source differs")
            rows.append(
                {
                    "kind": "BYPASS",
                    "carried_shape_role": role,
                    "pipeline_seed": seed,
                    "artifact_sha256": bypass["content_hash"],
                }
            )
            pf_id = f"RETB_{role}_PF_FROZEN_PF_FROZEN_ND0_NONE_TOKEN_PREDICTED_S{seed}"
            tr_id = (
                f"RETB_{role}_TR_REFINE_{refiner_lock['selected_variant']}_"
                f"ND0_NONE_TOKEN_PREDICTED_S{seed}"
            )
            unrestricted_id = _run_id(role, seed)
            metrics = {
                name: load_hashed_json(_metric_path(root, run_id))["content_hash"]
                for name, run_id in (
                    ("frozen_reconstruction", pf_id),
                    ("token_refiner", tr_id),
                    ("unrestricted_fusion", unrestricted_id),
                )
            }
            rows.append(
                {
                    "kind": "RECONSTRUCTION",
                    "carried_shape_role": role,
                    "pipeline_seed": seed,
                    "artifact_sha256": with_content_hash(metrics)["content_hash"],
                    "metric_hashes": metrics,
                }
            )
    stage_i = load_hashed_json(root / "selection" / "stage_i" / "stage_i_index.json")
    rows.append(
        {
            "kind": "SUBSTITUTION",
            "carried_shape_role": "PRIMARY_LOCKED_BUNDLE",
            "pipeline_seed": "ALL",
            "artifact_sha256": stage_i["content_hash"],
        }
    )
    artifact = bind_source(
        with_content_hash(
            {
                "contract": CONTRACT,
                "schema_version": 1,
                "control_kinds": ["BYPASS", "SUBSTITUTION", "RECONSTRUCTION"],
                "rows": rows,
                "complete_coverage": True,
                "scientific_underperformance_blocks_continuation": False,
                "stack_val_consumed": False,
                "final_test_consumed": False,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(args.output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
