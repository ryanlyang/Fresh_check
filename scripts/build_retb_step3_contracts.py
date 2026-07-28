#!/usr/bin/env python3
"""Build or dry-run the immutable RETB Step-3 architecture contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step3 import (  # noqa: E402
    build_step3_bundle,
    publish_step3_bundle,
    validate_step3_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    bundle = build_step3_bundle(
        campaign_spec_sha256=campaign["content_hash"],
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    bundle_sha = validate_step3_bundle(bundle)
    result: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "campaign_root": str(args.campaign_root.resolve()),
        "step3_bundle_sha256": bundle_sha,
        "artifact_hashes": bundle["step3_bundle"]["artifact_hashes"],
    }
    if not args.dry_run:
        result["publication"] = publish_step3_bundle(
            campaign_root=args.campaign_root,
            bundle=bundle,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
