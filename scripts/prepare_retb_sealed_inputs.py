#!/usr/bin/env python3
"""Publish checkpoint-free stack_val/final_test input-preparation lineage."""

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
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.target_cache import (  # noqa: E402
    build_sealed_input_preparation,
    validate_sealed_input_preparation,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("stack_val", "final_test"))
    parser.add_argument("--identity-manifest-sha256", required=True)
    parser.add_argument("--raw-input-manifest-sha256", required=True)
    parser.add_argument("--degraded-hlt-input-manifest-sha256", required=True)
    parser.add_argument("--relation-sidecar-manifest-sha256", required=True)
    parser.add_argument("--region-sidecar-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    artifact = bind_source(
        build_sealed_input_preparation(
            split=args.split,
            identity_manifest_sha256=args.identity_manifest_sha256,
            raw_input_manifest_sha256=args.raw_input_manifest_sha256,
            degraded_hlt_input_manifest_sha256=(
                args.degraded_hlt_input_manifest_sha256
            ),
            relation_sidecar_manifest_sha256=(
                args.relation_sidecar_manifest_sha256
            ),
            region_sidecar_manifest_sha256=(
                args.region_sidecar_manifest_sha256
            ),
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_sealed_input_preparation(artifact)
    publication = write_immutable_json(args.output, artifact)
    print(
        json.dumps(
            {
                "sealed_input_preparation_sha256": artifact["content_hash"],
                "publication": publication,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
