#!/usr/bin/env python3
"""Register exact Stage-M scale-train normalizer/calibrator refits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.confirmation import (  # noqa: E402
    SCALE_SHORTLIST_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_up import (  # noqa: E402
    build_scale_refit_bundle,
    validate_scale_refit_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--locked-scale-shortlist", required=True, type=Path)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    authorize_dataset_access(
        worker_role="scale_training_worker",
        requested_resource="scale_train",
    )
    authorize_dataset_access(
        worker_role="design_worker", requested_resource="val_design"
    )
    shortlist = load_hashed_json(
        args.locked_scale_shortlist,
        expected_contract=SCALE_SHORTLIST_CONTRACT,
    )
    configuration = json.loads(args.configuration.read_text("utf-8"))
    required = {
        "graph_id",
        "pipeline_seed",
        "scale_train_manifest_sha256",
        "val_design_identity_manifest_sha256",
        "refits",
        "five_hundred_k_artifact_hashes",
    }
    if (
        set(configuration) != required
        or shortlist.get("source") != campaign.get("source")
    ):
        raise ValueError("scale-refit configuration/source differs")
    artifact = bind_source(
        build_scale_refit_bundle(
            locked_scale_shortlist_sha256=shortlist["content_hash"],
            **configuration,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_scale_refit_bundle(artifact)
    result = {
        "dry_run": args.dry_run,
        "scale_refit_bundle_sha256": artifact["content_hash"],
        "graph_id": artifact["graph_id"],
        "pipeline_seed": artifact["pipeline_seed"],
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
