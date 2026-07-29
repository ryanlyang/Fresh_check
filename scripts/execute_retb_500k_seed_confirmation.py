#!/usr/bin/env python3
"""Execute and attest one genuine matched-seed 500k confirmation graph."""

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
    STAGE_L_GRAPH_REGISTRY_CONTRACT,
    build_seed_confirmation,
    validate_seed_confirmation,
    validate_stage_l_graph_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.confirmation_execution import (  # noqa: E402
    CONFIRMATION_EXECUTION_PLAN_CONTRACT,
    validate_confirmation_execution_plan,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (  # noqa: E402
    CLASSIFICATION_METRICS_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.paired_statistics import (  # noqa: E402
    PAIRED_STATISTICS_CONTRACT,
    validate_paired_confirmation_statistics,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_execution import (  # noqa: E402
    execute_plan_steps,
    source_bound_artifact_digest,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--graph-registry", required=True, type=Path)
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    authorize_dataset_access(
        worker_role="training_worker", requested_resource="model_train"
    )
    authorize_dataset_access(
        worker_role="training_worker", requested_resource="val_stop"
    )
    authorize_dataset_access(
        worker_role="design_worker", requested_resource="val_design"
    )
    registry = load_hashed_json(
        args.graph_registry,
        expected_contract=STAGE_L_GRAPH_REGISTRY_CONTRACT,
    )
    validate_stage_l_graph_registry(registry)
    plan = load_hashed_json(
        args.execution_plan,
        expected_contract=CONFIRMATION_EXECUTION_PLAN_CONTRACT,
    )
    validate_confirmation_execution_plan(
        plan,
        campaign_source=campaign["source"],
        campaign_root=args.campaign_root,
        repo_root=REPO_ROOT,
    )
    if (
        registry.get("source") != campaign["source"]
        or plan["stage_l_graph_registry_sha256"]
        != registry["content_hash"]
    ):
        raise ValueError("500k confirmation execution lineage differs")
    definitions = {
        row["graph_id"]: row for row in registry["definitions"]
    }
    if plan["graph_id"] not in definitions:
        raise ValueError("500k confirmation graph is not registered")

    receipts = execute_plan_steps(
        plan["steps"],
        campaign_root=args.campaign_root,
        repo_root=REPO_ROOT,
    )
    component_hashes = {
        name: source_bound_artifact_digest(
            path, campaign_source=campaign["source"]
        )
        for name, path in plan["component_artifacts"].items()
    }
    metrics = load_hashed_json(
        Path(plan["component_artifacts"]["metrics_artifact"]),
        expected_contract=CLASSIFICATION_METRICS_CONTRACT,
    )
    paired = load_hashed_json(
        Path(plan["component_artifacts"]["paired_statistics"]),
        expected_contract=PAIRED_STATISTICS_CONTRACT,
    )
    validate_paired_confirmation_statistics(paired)
    summary = load_hashed_json(Path(plan["training_summary"]))
    required_summary = {
        "graph_id",
        "pipeline_seed",
        "training_population",
        "training_epochs",
        "optimizer_update_count",
        "normalized_token_error",
        "analytical_flops_batch1",
        "parameter_count",
        "actual_training_executed",
    }
    if (
        set(summary)
        - {"contract", "schema_version", "source", "content_hash"}
        != required_summary
        or summary.get("source") != campaign["source"]
        or metrics.get("source") != campaign["source"]
        or paired.get("source") != campaign["source"]
        or summary["graph_id"] != plan["graph_id"]
        or int(summary["pipeline_seed"]) != int(plan["pipeline_seed"])
        or summary["training_population"] != "model_train_500k"
        or int(summary["training_epochs"]) != 40
        or int(summary["optimizer_update_count"]) <= 0
        or summary["actual_training_executed"] is not True
    ):
        raise ValueError("500k confirmation completion evidence differs")
    artifact = bind_source(
        build_seed_confirmation(
            graph_definition=definitions[plan["graph_id"]],
            pipeline_seed=int(plan["pipeline_seed"]),
            classification_metrics=metrics,
            normalized_token_error=summary["normalized_token_error"],
            analytical_flops_batch1=int(
                summary["analytical_flops_batch1"]
            ),
            parameter_count=int(summary["parameter_count"]),
            component_hashes=component_hashes,
            paired_statistics=paired,
            val_design_label_manifest_sha256=plan[
                "val_design_label_manifest_sha256"
            ],
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_seed_confirmation(
        artifact, graph_definition=definitions[plan["graph_id"]]
    )
    publication = write_immutable_json(args.output, artifact)
    print(
        json.dumps(
            {
                "graph_id": artifact["graph_id"],
                "pipeline_seed": artifact["pipeline_seed"],
                "seed_confirmation_sha256": artifact["content_hash"],
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
