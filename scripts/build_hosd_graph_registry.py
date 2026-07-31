#!/usr/bin/env python3
"""Resolve HOSD discovery locks into the executable confirmation registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_locked_graph_registry,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    COMBINATION_SELECTION_CONTRACT,
    FEEDBACK_SELECTION_CONTRACT,
    SINGLE_FAMILY_SELECTION_CONTRACT,
    STAGE_F_PLAN_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--retb-comparators-json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    retb = (
        {}
        if args.retb_comparators_json is None
        else json.loads(args.retb_comparators_json.read_text(encoding="utf-8"))
    )
    artifact = build_locked_graph_registry(
        single_family_selection=load_hashed_json(
            root / "auxiliary" / "locked_single_family_choices.json",
            expected_contract=SINGLE_FAMILY_SELECTION_CONTRACT,
        ),
        feedback_selection=load_hashed_json(
            root / "feedback" / "locked_feedback_choices.json",
            expected_contract=FEEDBACK_SELECTION_CONTRACT,
        ),
        combination_selection=load_hashed_json(
            root / "combinations" / "locked_combination_choices.json",
            expected_contract=COMBINATION_SELECTION_CONTRACT,
        ),
        stage_f_plan=load_hashed_json(
            root / "job_ledgers" / "stage_f_execution_plan.json",
            expected_contract=STAGE_F_PLAN_CONTRACT,
        ),
        retb_comparators=retb,
        source=campaign["source"],
    )
    output = args.output or root / "registry" / "locked_graph_registry.json"
    publication = write_immutable_json(output, artifact)
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
