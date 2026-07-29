#!/usr/bin/env python3
"""Join sealed labels in the selector and lock dual 3M finalists."""

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
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_up import (  # noqa: E402
    SCALE_COMPLETION_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.splits import (  # noqa: E402
    FINAL_SELECT_LABEL_MANIFEST_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_n_selection import (  # noqa: E402
    FINALIST_LINEAGE_KEYS,
    build_scale_finalist_bundle,
    load_stack_selection_prediction,
    publish_scale_finalist_bundle,
    validate_scale_finalist_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--locked-scale-shortlist", required=True, type=Path)
    parser.add_argument("--scale-completion", required=True, type=Path)
    parser.add_argument("--prediction-manifest", action="append", required=True, type=Path)
    parser.add_argument("--final-select-label-manifest", required=True, type=Path)
    parser.add_argument("--lineage-hashes", required=True, type=Path)
    parser.add_argument("--shape-assignments", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    authorize_dataset_access(
        worker_role="stage_n_selector",
        requested_resource="final_select_label_manifest",
    )
    shortlist = load_hashed_json(
        args.locked_scale_shortlist,
        expected_contract=SCALE_SHORTLIST_CONTRACT,
    )
    completion = load_hashed_json(
        args.scale_completion,
        expected_contract=SCALE_COMPLETION_CONTRACT,
    )
    labels = load_hashed_json(
        args.final_select_label_manifest,
        expected_contract=FINAL_SELECT_LABEL_MANIFEST_CONTRACT,
    )
    records = []
    for path in args.prediction_manifest:
        manifest, arrays = load_stack_selection_prediction(path)
        records.append({"manifest": manifest, **arrays})
    lineage = json.loads(args.lineage_hashes.read_text("utf-8"))
    shape_assignments = json.loads(
        args.shape_assignments.read_text("utf-8")
    )
    if (
        set(lineage) != set(FINALIST_LINEAGE_KEYS)
        or any(
            row.get("source") != campaign.get("source")
            for row in (
                shortlist,
                completion,
                labels,
                *(record["manifest"] for record in records),
            )
        )
    ):
        raise ValueError("scale-finalist source/lineage differs")
    bundle = build_scale_finalist_bundle(
        locked_scale_shortlist=shortlist,
        scale_completion=completion,
        prediction_records=records,
        final_select_label_manifest=labels,
        lineage_hashes=lineage,
        shape_assignments=shape_assignments,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_scale_finalist_bundle(
        bundle,
        locked_scale_shortlist=shortlist,
        scale_completion=completion,
        prediction_records=records,
        final_select_label_manifest=labels,
        shape_assignments=shape_assignments,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    result = {
        "dry_run": args.dry_run,
        "locked_scale_finalists_sha256": bundle[
            "locked_scale_finalists"
        ]["content_hash"],
        "ACCURACY_FINALIST": bundle["locked_scale_finalists"][
            "ACCURACY_FINALIST"
        ],
        "REJECTION_FINALIST": bundle["locked_scale_finalists"][
            "REJECTION_FINALIST"
        ],
    }
    if not args.dry_run:
        result["publication"] = publish_scale_finalist_bundle(
            campaign_root=args.campaign_root, bundle=bundle
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
