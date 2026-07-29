#!/usr/bin/env python3
"""Execute and attest one genuine locked Stage-M graph/seed training plan."""

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
from teacher_logit_reco.relation_expert_token_bridge.scale_execution import (  # noqa: E402
    SCALE_GRAPH_EXECUTION_PLAN_CONTRACT,
    execute_plan_steps,
    source_bound_artifact_digest,
    validate_scale_graph_execution_plan,
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
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    shortlist = load_hashed_json(
        args.locked_scale_shortlist,
        expected_contract=SCALE_SHORTLIST_CONTRACT,
    )
    refit = load_hashed_json(
        args.scale_refit_bundle,
        expected_contract=SCALE_REFIT_BUNDLE_CONTRACT,
    )
    plan = load_hashed_json(
        args.execution_plan,
        expected_contract=SCALE_GRAPH_EXECUTION_PLAN_CONTRACT,
    )
    validate_scale_graph_execution_plan(
        plan,
        campaign_source=campaign["source"],
        campaign_root=args.campaign_root,
        repo_root=REPO_ROOT,
    )
    if (
        shortlist.get("source") != campaign["source"]
        or refit.get("source") != campaign["source"]
        or plan["locked_scale_shortlist_sha256"]
        != shortlist["content_hash"]
        or plan["scale_refit_bundle_sha256"] != refit["content_hash"]
    ):
        raise ValueError("scale-graph execution lineage differs")
    authorize_dataset_access(
        worker_role="scale_training_worker",
        requested_resource="scale_train",
    )
    authorize_dataset_access(
        worker_role="scale_training_worker",
        requested_resource="val_stop",
    )
    receipts = execute_plan_steps(
        plan["steps"],
        campaign_root=args.campaign_root,
        repo_root=REPO_ROOT,
    )
    components = {
        name: source_bound_artifact_digest(
            path, campaign_source=campaign["source"]
        )
        for name, path in plan["component_artifacts"].items()
    }
    metrics = load_hashed_json(
        Path(plan["pre_stack_metrics"]),
        expected_contract=CLASSIFICATION_METRICS_CONTRACT,
    )
    summary = load_hashed_json(Path(plan["training_summary"]))
    required_summary = {
        "graph_id",
        "pipeline_seed",
        "training_population",
        "optimizer_update_count",
        "selected_epoch",
        "val_stop_accuracy",
        "val_stop_cross_entropy",
        "analytical_flops_batch1",
        "analytical_flops_batch128",
        "parameter_count",
        "actual_training_executed",
    }
    if (
        set(summary) - {"contract", "schema_version", "source", "content_hash"}
        != required_summary
        or summary.get("source") != campaign["source"]
        or summary["graph_id"] != plan["graph_id"]
        or int(summary["pipeline_seed"]) != int(plan["pipeline_seed"])
        or summary["training_population"] != "scale_train"
        or int(summary["optimizer_update_count"]) <= 0
        or summary["actual_training_executed"] is not True
        or metrics.get("source") != campaign["source"]
    ):
        raise ValueError("scale training completion evidence differs")
    artifact = bind_source(
        build_scale_graph_run(
            locked_scale_shortlist=shortlist,
            graph_id=plan["graph_id"],
            pipeline_seed=int(plan["pipeline_seed"]),
            scale_refit_bundle=refit,
            component_hashes=components,
            selected_epoch=int(summary["selected_epoch"]),
            val_stop_accuracy=float(summary["val_stop_accuracy"]),
            val_stop_cross_entropy=float(
                summary["val_stop_cross_entropy"]
            ),
            analytical_flops_batch1=int(
                summary["analytical_flops_batch1"]
            ),
            analytical_flops_batch128=int(
                summary["analytical_flops_batch128"]
            ),
            parameter_count=int(summary["parameter_count"]),
            pre_stack_confirmation_metrics=metrics,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_scale_graph_run(
        artifact, locked_scale_shortlist=shortlist
    )
    publication = write_immutable_json(args.output, artifact)
    print(
        json.dumps(
            {
                "graph_id": artifact["graph_id"],
                "pipeline_seed": artifact["pipeline_seed"],
                "scale_graph_run_sha256": artifact["content_hash"],
                "execution_receipts": receipts,
                "publication": publication,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
