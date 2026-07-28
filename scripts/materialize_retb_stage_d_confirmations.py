#!/usr/bin/env python3
"""Freeze selected Stage-D semantics across pipeline seeds 101/202/303."""

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
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step6 import (  # noqa: E402
    materialize_stage_d_confirmation_rows,
    validate_stage_d_confirmation_registry,
    validate_stage_d_run_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--selected-run-id", action="append", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    registry = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_d_runs.json"
    )
    validate_stage_d_run_registry(registry)
    confirmation = bind_source(
        materialize_stage_d_confirmation_rows(
            registry, selected_run_ids=args.selected_run_id
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    if confirmation["source"] != campaign["source"]:
        raise ValueError("Stage-D confirmation source differs from campaign")
    validate_stage_d_confirmation_registry(confirmation)
    output = args.output or (
        args.campaign_root / "selection" / "retb_stage_d_confirmations.json"
    )
    result = {
        "dry_run": bool(args.dry_run),
        "confirmation_registry_sha256": confirmation["content_hash"],
        "row_count": len(confirmation["rows"]),
        "output": str(output.resolve()),
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(output, confirmation)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
