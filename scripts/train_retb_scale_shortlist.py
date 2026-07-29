#!/usr/bin/env python3
"""Register one completed locked Stage-M graph/seed scale retraining."""

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
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (  # noqa: E402
    CLASSIFICATION_METRICS_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_up import (  # noqa: E402
    SCALE_REFIT_BUNDLE_CONTRACT,
    build_scale_graph_run,
    validate_scale_graph_run,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--locked-scale-shortlist", required=True, type=Path)
    parser.add_argument("--scale-refit-bundle", required=True, type=Path)
    parser.add_argument("--pre-stack-metrics", required=True, type=Path)
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
        worker_role="scale_training_worker",
        requested_resource="val_stop",
    )
    shortlist = load_hashed_json(
        args.locked_scale_shortlist,
        expected_contract=SCALE_SHORTLIST_CONTRACT,
    )
    refit = load_hashed_json(
        args.scale_refit_bundle,
        expected_contract=SCALE_REFIT_BUNDLE_CONTRACT,
    )
    metrics = load_hashed_json(
        args.pre_stack_metrics,
        expected_contract=CLASSIFICATION_METRICS_CONTRACT,
    )
    configuration = json.loads(args.configuration.read_text("utf-8"))
    required = {
        "graph_id",
        "pipeline_seed",
        "component_hashes",
        "selected_epoch",
        "val_stop_accuracy",
        "val_stop_cross_entropy",
        "analytical_flops_batch1",
        "analytical_flops_batch128",
        "parameter_count",
    }
    if (
        set(configuration) != required
        or any(
            row.get("source") != campaign.get("source")
            for row in (shortlist, refit, metrics)
        )
    ):
        raise ValueError("scale graph registration source/config differs")
    artifact = bind_source(
        build_scale_graph_run(
            locked_scale_shortlist=shortlist,
            scale_refit_bundle=refit,
            pre_stack_confirmation_metrics=metrics,
            **configuration,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_scale_graph_run(
        artifact, locked_scale_shortlist=shortlist
    )
    result = {
        "dry_run": args.dry_run,
        "scale_graph_run_sha256": artifact["content_hash"],
        "graph_id": artifact["graph_id"],
        "pipeline_seed": artifact["pipeline_seed"],
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
