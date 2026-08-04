#!/usr/bin/env python3
"""Seal completed Stage-B parents for the supplemental offline fusion wave."""

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
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.supplemental_offline_fusion import (  # noqa: E402
    build_supplemental_plan,
    validate_supplemental_plan,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-campaign-root", required=True, type=Path)
    parser.add_argument("--supplemental-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = build_supplemental_plan(
        parent_root=args.parent_campaign_root,
        supplemental_id=args.supplemental_id,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_supplemental_plan(plan)
    publication = write_immutable_json(args.output, plan)
    print(json.dumps({"plan": plan, "publication": publication}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
