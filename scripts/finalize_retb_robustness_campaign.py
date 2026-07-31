#!/usr/bin/env python3
"""Authenticate complete Stage-K robustness coverage."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.execute_retb_robustness_campaign import PROFILES, REPLICAS, CONTRACT  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import source_snapshot  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.step7 import STAGE_E_SHAPES  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.workflow import load_and_validate_campaign_source  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    rows = []
    for role in STAGE_E_SHAPES:
        for seed in (101, 202, 303):
            for profile in PROFILES:
                for replica in REPLICAS:
                    row = load_hashed_json(
                        root
                        / "controls"
                        / "robustness"
                        / "evaluations"
                        / role
                        / f"seed_{seed}"
                        / profile
                        / f"replica_{replica}"
                        / "evaluation.json"
                    )
                    if (
                        row.get("source") != campaign.get("source")
                        or row["profile"] != profile
                        or int(row["replica"]) != replica
                        or int(row["pipeline_seed"]) != seed
                    ):
                        raise ValueError("robustness result coverage differs")
                    rows.append(
                        {
                            "carried_shape_role": role,
                            "pipeline_seed": seed,
                            "profile": profile,
                            "replica": replica,
                            "evaluation_sha256": row["content_hash"],
                            "metrics_sha256": row["metrics_sha256"],
                        }
                    )
    expected = len(STAGE_E_SHAPES) * 3 * len(PROFILES) * len(REPLICAS)
    if len(rows) != expected:
        raise ValueError("robustness campaign cardinality differs")
    artifact = bind_source(
        with_content_hash(
            {
                "contract": CONTRACT,
                "schema_version": 1,
                "profiles": list(PROFILES),
                "replicas": list(REPLICAS),
                "carried_shape_roles": list(STAGE_E_SHAPES),
                "pipeline_seeds": [101, 202, 303],
                "row_count": len(rows),
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
