#!/usr/bin/env python3
"""Publish a bounded, resumable target-cache shard execution plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge import (  # noqa: E402
    build_target_shard_plan,
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--target-cache-specification", required=True, type=Path)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--maximum-concurrent-tasks", type=int, default=12)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    specification = load_hashed_json(args.target_cache_specification)
    artifact = build_target_shard_plan(
        campaign_spec_sha256=campaign["content_hash"],
        target_cache_specification_sha256=specification["content_hash"],
        identity_order_sha256=specification["identity_order_sha256"],
        event_count=int(specification["event_count"]),
        shard_size=args.shard_size,
        maximum_concurrent_tasks=args.maximum_concurrent_tasks,
    )
    result: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "target_shard_plan_sha256": artifact["content_hash"],
        "shard_count": artifact["shard_count"],
        "slurm_array": artifact["slurm_array"],
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
