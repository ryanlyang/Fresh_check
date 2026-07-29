#!/usr/bin/env python3
"""Revalidate every RETB task row and publish its completeness attestation."""

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
    TASK_MANIFEST_CONTRACT,
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.task_completion import (  # noqa: E402
    publish_task_manifest_completion,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--task-manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    manifest = load_hashed_json(
        args.task_manifest, expected_contract=TASK_MANIFEST_CONTRACT
    )
    result = publish_task_manifest_completion(
        campaign_root=args.campaign_root,
        campaign=campaign,
        task_manifest=manifest,
    )
    print(
        json.dumps(
            {
                "node_id": manifest["node_id"],
                "task_count": manifest["task_count"],
                "manifest_completion_sha256": result["artifact"][
                    "content_hash"
                ],
                "manifest_completion_path": result["path"],
                "publication": result["publication"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
