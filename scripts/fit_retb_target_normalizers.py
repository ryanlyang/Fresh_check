#!/usr/bin/env python3
"""Fit and publish RETB target-token normalizers from model_train only."""

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
from teacher_logit_reco.relation_expert_token_bridge.target_cache import (  # noqa: E402
    fit_target_normalizers,
    publish_target_normalizers,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--target-cache-manifest", required=True, type=Path)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--specification-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    artifacts = fit_target_normalizers(
        model_train_manifest_path=args.target_cache_manifest,
        expected_pipeline_seed=args.pipeline_seed,
        expected_specification_sha256=args.specification_sha256,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    result = publish_target_normalizers(
        output_dir=args.output_dir, artifacts=artifacts
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
