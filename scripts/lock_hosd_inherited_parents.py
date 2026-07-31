#!/usr/bin/env python3
"""Validate and lock every rebuilt/reused HOSD parent after Stage-A rebuilds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    PARENT_REQUIREMENTS,
    build_parent_status,
    load_and_validate_campaign,
    require_parents_ready,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge import (  # noqa: E402
    source_snapshot,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--inherited-parent",
        action="append",
        default=[],
        metavar="PARENT_ID=PATH",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _paths(root: Path, values: Sequence[str]) -> dict[str, Path]:
    shared = root / "inputs" / "shared_retb_parent_campaign"
    output = {}
    for requirement in PARENT_REQUIREMENTS:
        canonical = root / requirement.canonical_path
        inherited = shared / requirement.canonical_path
        output[requirement.parent_id] = (
            canonical if canonical.is_file() else inherited
        )
    for value in values:
        parent_id, separator, raw_path = value.partition("=")
        if not separator or parent_id not in output or not raw_path:
            raise ValueError(
                "--inherited-parent must be a registered PARENT_ID=PATH"
            )
        output[parent_id] = Path(raw_path)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    status = build_parent_status(
        source_snapshot=source_snapshot(REPO_ROOT),
        artifact_paths=_paths(root, args.inherited_parent),
    )
    require_parents_ready(status, before_stage="B")
    output = args.output or (
        root / "inputs" / "resolved_inherited_parent_lock.json"
    )
    result: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "campaign_spec_sha256": campaign["content_hash"],
        "resolved_parent_lock_sha256": status["content_hash"],
        "parent_count": len(status["requirements"]),
        "all_stage_b_parents_reusable": status[
            "all_stage_b_parents_reusable"
        ],
        "output": str(output.resolve()),
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(output, status)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
