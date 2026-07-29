#!/usr/bin/env python3
"""Select immutable SHAPE_BRIDGE from compact/high val_design evidence."""

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
    select_bridge_shape,
    validate_bridge_shape_selection,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step12 import (  # noqa: E402
    STEP12_BUNDLE_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    authorize_dataset_access(
        worker_role="design_worker", requested_resource="val_design"
    )
    step12 = load_hashed_json(
        args.campaign_root
        / "registry"
        / "retb_step12_final_consumers_bundle.json",
        expected_contract=STEP12_BUNDLE_CONTRACT,
    )
    configuration = json.loads(args.configuration.read_text("utf-8"))
    if set(configuration) != {
        "compact_shape_id",
        "high_shape_id",
        "val_design_label_manifest_sha256",
        "rows",
    }:
        raise ValueError("bridge-shape configuration fields differ")
    artifact = bind_source(
        select_bridge_shape(
            rows=configuration["rows"],
            compact_shape_id=configuration["compact_shape_id"],
            high_shape_id=configuration["high_shape_id"],
            step12_bundle_sha256=step12["content_hash"],
            val_design_label_manifest_sha256=configuration[
                "val_design_label_manifest_sha256"
            ],
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_bridge_shape_selection(artifact)
    result = {
        "dry_run": args.dry_run,
        "bridge_shape_selection_sha256": artifact["content_hash"],
        "SHAPE_BRIDGE": artifact["SHAPE_BRIDGE"],
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
