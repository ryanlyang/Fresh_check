#!/usr/bin/env python3
"""Materialize source-bound Step-10 predictor candidates and coordinates."""

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
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_bundle import (  # noqa: E402
    build_locked_target_coordinate,
    build_predictor_candidate,
    shared_predictor_configuration_id,
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
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    configuration = json.loads(args.configuration.read_text("utf-8"))
    if set(configuration) != {"candidates", "coordinates"}:
        raise ValueError("bundle-input configuration fields differ")
    snapshot = source_snapshot(REPO_ROOT)
    candidates = []
    for raw_row in configuration["candidates"]:
        row = dict(raw_row)
        row.setdefault(
            "shared_configuration_id",
            shared_predictor_configuration_id(
                **{
                    name: row[name]
                    for name in (
                        "architecture",
                        "context",
                        "objective_id",
                        "uncertainty_head",
                        "normalization_mode",
                        "learning_rate",
                        "dropout",
                        "hlt_evidence_mode",
                    )
                }
            ),
        )
        candidates.append(
            bind_source(
                build_predictor_candidate(**row), source_snapshot=snapshot
            )
        )
    coordinates = [
        bind_source(
            build_locked_target_coordinate(**row), source_snapshot=snapshot
        )
        for row in configuration["coordinates"]
    ]
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise ValueError("bundle-input candidate IDs are duplicated")
    if len({row["coordinate_id"] for row in coordinates}) != len(coordinates):
        raise ValueError("bundle-input coordinate IDs are duplicated")
    result = {
        "dry_run": args.dry_run,
        "candidate_count": len(candidates),
        "coordinate_count": len(coordinates),
        "candidate_hashes": {
            row["candidate_id"]: row["content_hash"] for row in candidates
        },
        "coordinate_hashes": {
            row["coordinate_id"]: row["content_hash"] for row in coordinates
        },
    }
    if not args.dry_run:
        result["publications"] = {
            "candidates": {
                row["candidate_id"]: write_immutable_json(
                    args.output_dir
                    / "candidates"
                    / f"{row['candidate_id']}.json",
                    row,
                )
                for row in candidates
            },
            "coordinates": {
                row["coordinate_id"]: write_immutable_json(
                    args.output_dir
                    / "coordinates"
                    / f"{row['coordinate_id']}.json",
                    row,
                )
                for row in coordinates
            },
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
