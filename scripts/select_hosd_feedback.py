#!/usr/bin/env python3
"""Lock the best deployable Stage-E feedback graph after complete coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    authorize_access,
    build_feedback_selection,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    FEEDBACK_RESULT_CONTRACT,
    STAGE_E_PLAN_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--result", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    authorize_access(
        worker_role="design_selector",
        requested_resource="design_select_predictions",
    )
    plan = load_hashed_json(
        args.campaign_root / "job_ledgers" / "stage_e_execution_plan.json",
        expected_contract=STAGE_E_PLAN_CONTRACT,
    )
    results = [
        load_hashed_json(path, expected_contract=FEEDBACK_RESULT_CONTRACT)
        for path in args.result
    ]
    artifact = build_feedback_selection(
        stage_e_plan=plan, results=results, source=campaign["source"]
    )
    output = args.output or (
        args.campaign_root / "feedback" / "locked_feedback_choices.json"
    )
    publication = write_immutable_json(output, artifact)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
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
