#!/usr/bin/env python3
"""Register one matched-pipeline-seed 500k val_design confirmation row."""

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
from teacher_logit_reco.relation_expert_token_bridge.paired_statistics import (  # noqa: E402
    PAIRED_STATISTICS_CONTRACT,
    validate_paired_confirmation_statistics,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--graph-registry", required=True, type=Path)
    parser.add_argument("--classification-metrics", required=True, type=Path)
    parser.add_argument("--paired-statistics", required=True, type=Path)
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
    registry = load_hashed_json(
        args.graph_registry,
        expected_contract=STAGE_L_GRAPH_REGISTRY_CONTRACT,
    )
    validate_stage_l_graph_registry(registry)
    metrics_artifact = load_hashed_json(args.classification_metrics)
    paired = load_hashed_json(
        args.paired_statistics,
        expected_contract=PAIRED_STATISTICS_CONTRACT,
    )
    validate_paired_confirmation_statistics(paired)
    metrics = (
        metrics_artifact
        if metrics_artifact.get("contract")
        == CLASSIFICATION_METRICS_CONTRACT
        else metrics_artifact.get("metrics")
    )
    if (
        not isinstance(metrics, dict)
        or metrics.get("contract") != CLASSIFICATION_METRICS_CONTRACT
    ):
        raise ValueError(
            "seed-confirmation artifact lacks classification metrics"
        )
    configuration = json.loads(args.configuration.read_text("utf-8"))
    required = {
        "graph_id",
        "pipeline_seed",
        "normalized_token_error",
        "analytical_flops_batch1",
        "parameter_count",
        "component_hashes",
        "val_design_label_manifest_sha256",
    }
    definitions = {
        row["graph_id"]: row for row in registry["definitions"]
    }
    if (
        set(configuration) != required
        or configuration["graph_id"] not in definitions
        or registry.get("source") != campaign.get("source")
        or metrics_artifact.get("source") != campaign.get("source")
        or paired.get("source") != campaign.get("source")
        or configuration["component_hashes"].get("metrics_artifact")
        != metrics_artifact["content_hash"]
        or configuration["component_hashes"].get("paired_statistics")
        != paired["content_hash"]
        or paired.get("candidate_graph_id")
        != configuration["graph_id"]
        or int(paired.get("pipeline_seed", -1))
        != int(configuration["pipeline_seed"])
        or paired.get("parents", {}).get("candidate_prediction")
        != configuration["component_hashes"].get("prediction_manifest")
    ):
        raise ValueError("seed-confirmation input lineage differs")
    definition = definitions[configuration["graph_id"]]
    if paired.get("baseline_graph_id") != definition[
        "named_baseline_graph_id"
    ]:
        raise ValueError("paired-statistics named baseline differs")
    artifact = bind_source(
        build_seed_confirmation(
            graph_definition=definition,
            pipeline_seed=configuration["pipeline_seed"],
            classification_metrics=metrics,
            normalized_token_error=configuration[
                "normalized_token_error"
            ],
            analytical_flops_batch1=configuration[
                "analytical_flops_batch1"
            ],
            parameter_count=configuration["parameter_count"],
            component_hashes=configuration["component_hashes"],
            paired_statistics=paired,
            val_design_label_manifest_sha256=configuration[
                "val_design_label_manifest_sha256"
            ],
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_seed_confirmation(
        artifact, graph_definition=definition
    )
    result = {
        "dry_run": args.dry_run,
        "seed_confirmation_sha256": artifact["content_hash"],
        "graph_id": artifact["graph_id"],
        "pipeline_seed": artifact["pipeline_seed"],
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
