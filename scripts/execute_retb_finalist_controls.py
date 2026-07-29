#!/usr/bin/env python3
"""Train, attest, and lock all graph-specific finalist controls."""

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
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.final_seal import (  # noqa: E402
    build_finalist_controls,
    validate_finalist_controls,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_execution import (  # noqa: E402
    execute_plan_steps,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_execution import (  # noqa: E402
    FINALIST_CONTROLS_EXECUTION_PLAN_CONTRACT,
    validate_control_evidence,
    validate_finalist_controls_execution_plan,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_selection import (  # noqa: E402
    LOCKED_SCALE_FINALISTS_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--locked-scale-finalists", required=True, type=Path)
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    finalists = load_hashed_json(
        args.locked_scale_finalists,
        expected_contract=LOCKED_SCALE_FINALISTS_CONTRACT,
    )
    plan = load_hashed_json(
        args.execution_plan,
        expected_contract=FINALIST_CONTROLS_EXECUTION_PLAN_CONTRACT,
    )
    validate_finalist_controls_execution_plan(
        plan,
        campaign_source=campaign["source"],
        campaign_root=args.campaign_root,
        repo_root=REPO_ROOT,
    )
    if (
        finalists.get("source") != campaign["source"]
        or plan["locked_scale_finalists_sha256"]
        != finalists["content_hash"]
    ):
        raise ValueError("finalist-control execution lineage differs")
    receipts = execute_plan_steps(
        plan["steps"],
        campaign_root=args.campaign_root,
        repo_root=REPO_ROOT,
        forbidden_terms=(),
        forbidden_entrypoints=frozenset(
            {
                "attest_retb_finalist_controls.py",
                "execute_retb_finalist_controls.py",
            }
        ),
    )
    evidence = load_hashed_json(plan["control_evidence"])
    validate_control_evidence(
        evidence, locked_scale_finalists=finalists
    )
    artifact = bind_source(
        build_finalist_controls(
            locked_scale_finalists=finalists, rows=evidence["rows"]
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_finalist_controls(
        artifact, locked_scale_finalists=finalists
    )
    publication = write_immutable_json(args.output, artifact)
    print(
        json.dumps(
            {
                "finalist_controls_sha256": artifact["content_hash"],
                "execution_receipts": receipts,
                "performance_result_used_as_gate": False,
                "publication": publication,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
