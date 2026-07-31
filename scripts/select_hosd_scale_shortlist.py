#!/usr/bin/env python3
"""Lock the bounded duplicate-free HOSD scale shortlist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_scale_shortlist,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    CONFIRMATION_SUMMARY_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--roles-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    summary = load_hashed_json(
        args.campaign_root / "confirmation_500k" / "summary.json",
        expected_contract=CONFIRMATION_SUMMARY_CONTRACT,
    )
    artifact = build_scale_shortlist(
        confirmation_summary=summary,
        role_graph_ids=None,
        source=campaign["source"],
    )
    output = args.output or (
        args.campaign_root / "selection" / "locked_scale_shortlist.json"
    )
    publication = write_immutable_json(output, artifact)
    print(json.dumps({"content_hash": artifact["content_hash"], "publication": publication["status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
