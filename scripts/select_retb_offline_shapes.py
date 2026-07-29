#!/usr/bin/env python3
"""Select RETB SHAPE_HIGH and SHAPE_COMPACT from complete three-seed metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    bind_source,
    load_hashed_json,
    validate_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import source_snapshot
from teacher_logit_reco.relation_expert_token_bridge.dynamic_continuation import (
    add_dynamic_continuation_arguments,
    resolve_selector_continuation,
)
from teacher_logit_reco.relation_expert_token_bridge.selection import (
    UNIFORM_SHAPE_METRICS_CONTRACT,
    select_offline_shapes,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--uniform-metrics", required=True, type=Path)
    parser.add_argument("--baseline-mean-accuracy", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    add_dynamic_continuation_arguments(parser)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    authorize_dataset_access(
        worker_role="design_worker", requested_resource="val_design"
    )
    metrics = load_hashed_json(
        args.uniform_metrics,
        expected_contract=UNIFORM_SHAPE_METRICS_CONTRACT,
    )
    validate_content_hash(metrics)
    if metrics.get("source") != campaign.get("source"):
        raise ValueError("uniform shape metrics belong to another source")
    selection = bind_source(
        select_offline_shapes(
            metrics, baseline_mean_accuracy=args.baseline_mean_accuracy
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    output = args.output or (
        args.campaign_root / "selection" / "retb_offline_shapes.json"
    )
    result = {
        "dry_run": bool(args.dry_run),
        "SHAPE_HIGH": selection["SHAPE_HIGH"],
        "SHAPE_COMPACT": selection["SHAPE_COMPACT"],
        "all_multi_expert_models_worse_than_baseline": selection[
            "all_multi_expert_models_worse_than_baseline"
        ],
        "selection": selection,
        "output": str(output.resolve()),
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(output, selection)
    continuation = resolve_selector_continuation(
        args=args,
        campaign=campaign,
        campaign_root=args.campaign_root,
        selector_output=selection,
        selector_output_path=output,
        load_hashed_json=load_hashed_json,
        dry_run=bool(args.dry_run),
    )
    if continuation is not None:
        result["continuation"] = continuation
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
