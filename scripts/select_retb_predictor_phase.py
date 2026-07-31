#!/usr/bin/env python3
"""Select one complete predictor phase without performance-based stopping."""

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
from teacher_logit_reco.relation_expert_token_bridge.predictor_campaign import (  # noqa: E402
    build_stage_f_optimizer_followup_registry,
    select_stage_f_architecture_families,
    select_stage_f_optimizer_configurations,
    select_stage_g_configurations,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("F_ARCHITECTURE", "F_OPTIMIZER", "G_CONFIGURATION"),
    )
    parser.add_argument("--metric", action="append", required=True, type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--followup-registry-output", type=Path)
    args = parser.parse_args(argv)
    load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    rows = [load_hashed_json(path) for path in args.metric]
    normalized = []
    for row in rows:
        normalized_row = {
            **row,
            "result_sha256": row["content_hash"],
        }
        if args.phase == "G_CONFIGURATION":
            normalized_row["template_id"] = row["run_id"]
        normalized.append(normalized_row)
    if args.phase == "F_ARCHITECTURE":
        selected = bind_source(
            select_stage_f_architecture_families(normalized),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        if args.followup_registry_output is None:
            raise ValueError("architecture selection requires followup registry")
        followup = bind_source(
            build_stage_f_optimizer_followup_registry(selected),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        write_immutable_json(args.followup_registry_output, followup)
    elif args.phase == "F_OPTIMIZER":
        if args.registry is None:
            raise ValueError("optimizer selection requires its registry")
        selected = bind_source(
            select_stage_f_optimizer_configurations(
                registry=load_hashed_json(args.registry),
                results=normalized,
            ),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
    else:
        selected = bind_source(
            select_stage_g_configurations(normalized),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
    publication = write_immutable_json(args.output, selected)
    print(json.dumps(publication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
