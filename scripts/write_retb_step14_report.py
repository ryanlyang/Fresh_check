#!/usr/bin/env python3
"""Write complete source-bound Stage-M/N JSON and Markdown reports."""

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
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.final_seal import (  # noqa: E402
    FINAL_TEST_EVALUATION_CONTRACT,
    FINAL_TEST_EXECUTION_LOCK_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_up import (  # noqa: E402
    SCALE_COMPLETION_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_reporting import (  # noqa: E402
    build_stage_mn_report,
    publish_stage_mn_report,
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
    parser.add_argument("--scale-completion", required=True, type=Path)
    parser.add_argument("--locked-scale-finalists", required=True, type=Path)
    parser.add_argument("--execution-lock", required=True, type=Path)
    parser.add_argument("--final-evaluation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    scale = load_hashed_json(
        args.scale_completion, expected_contract=SCALE_COMPLETION_CONTRACT
    )
    finalists = load_hashed_json(
        args.locked_scale_finalists,
        expected_contract=LOCKED_SCALE_FINALISTS_CONTRACT,
    )
    execution = load_hashed_json(
        args.execution_lock,
        expected_contract=FINAL_TEST_EXECUTION_LOCK_CONTRACT,
    )
    final = load_hashed_json(
        args.final_evaluation,
        expected_contract=FINAL_TEST_EVALUATION_CONTRACT,
    )
    if any(
        row.get("source") != campaign.get("source")
        for row in (scale, finalists, execution, final)
    ):
        raise ValueError("Stage-M/N report source differs")
    artifact, markdown = build_stage_mn_report(
        scale_completion=scale,
        locked_scale_finalists=finalists,
        execution_lock=execution,
        final_evaluation=final,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    result = {
        "dry_run": args.dry_run,
        "stage_mn_report_sha256": artifact["content_hash"],
        "markdown_sha256": artifact["markdown_sha256"],
    }
    if not args.dry_run:
        result["publication"] = publish_stage_mn_report(
            output_dir=args.output_dir,
            artifact=artifact,
            markdown=markdown,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
