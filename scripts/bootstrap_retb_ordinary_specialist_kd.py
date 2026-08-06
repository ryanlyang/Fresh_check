#!/usr/bin/env python3
"""Seal completed compact specialist KD as ordinary-head control parents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.supplemental_ordinary_specialist_kd import (  # noqa: E402,E501
    build_ordinary_specialist_kd_plan,
    validate_ordinary_specialist_kd_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-specialist-root", required=True, type=Path)
    parser.add_argument("--supplemental-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = build_ordinary_specialist_kd_plan(
        compact_specialist_root=args.compact_specialist_root,
        supplemental_id=args.supplemental_id,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_ordinary_specialist_kd_plan(plan)
    publication = write_immutable_json(args.output, plan)
    print(json.dumps({"plan": plan, "publication": publication}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
