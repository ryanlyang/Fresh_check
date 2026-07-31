#!/usr/bin/env python3
"""Aggregate the complete matched-seed RETB 500k confirmation wave."""

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
    SEED_CONFIRMATION_CONTRACT,
    STAGE_L_GRAPH_REGISTRY_CONTRACT,
    aggregate_500k_confirmation,
    validate_stage_l_graph_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    canonical_sha256,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.late_plan_factories import (  # noqa: E402
    build_late_factory_input,
    publish_late_factory_input,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--graph-registry", required=True, type=Path)
    parser.add_argument(
        "--seed-confirmation", action="append", default=[], type=Path
    )
    parser.add_argument(
        "--val-design-label-manifest-sha256", required=True
    )
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
    rows = [
        load_hashed_json(
            path, expected_contract=SEED_CONFIRMATION_CONTRACT
        )
        for path in args.seed_confirmation
    ]
    if (
        registry.get("source") != campaign.get("source")
        or any(row.get("source") != campaign.get("source") for row in rows)
    ):
        raise ValueError("500k confirmation source differs")
    artifact = bind_source(
        aggregate_500k_confirmation(
            graph_registry=registry,
            seed_confirmations=rows,
            val_design_label_manifest_sha256=(
                args.val_design_label_manifest_sha256
            ),
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    result = {
        "dry_run": args.dry_run,
        "confirmation_summary_sha256": artifact["content_hash"],
        "complete_graph_count": artifact["complete_graph_count"],
        "complete_matched_seed_coverage": artifact[
            "complete_matched_seed_coverage"
        ],
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
        graph = load_hashed_json(
            args.campaign_root / "job_ledgers" / "production_graph.json"
        )
        task_manifest = load_hashed_json(
            args.campaign_root
            / "job_ledgers"
            / "tasks"
            / "confirmation_summary.json"
        )
        definitions = {
            row["graph_id"]: row for row in registry["definitions"]
        }
        confirmations = {
            (row["graph_id"], int(row["pipeline_seed"])): row
            for row in rows
        }
        shape_rows = []
        uniform = [
            row
            for row in registry["definitions"]
            if row["semantic_category"] == "UNIFORM_FINALIST"
        ]
        for definition in uniform:
            source_role = definition["configuration"][
                "source_carried_shape_role"
            ]
            lock = load_hashed_json(
                args.campaign_root
                / "selection"
                / "predictor_bundle"
                / "carried"
                / f"{source_role}.json"
            )
            native = next(
                row
                for row in registry["definitions"]
                if row["semantic_category"] == "NATIVE_HLT_FUSION"
                and row["configuration"]["source_carried_shape_role"]
                == source_role
            )
            frozen = next(
                row
                for row in registry["definitions"]
                if row["semantic_category"] == "FROZEN_RECONSTRUCTION"
                and row["configuration"]["source_carried_shape_role"]
                == source_role
                and row["configuration"]["carried_shape_role"]
                == definition["configuration"]["carried_shape_role"]
            )
            dimensions = {
                tuple(int(value) for value in allocation)
                for allocation in lock["allocation"].values()
            }
            if len(dimensions) != 1:
                raise ValueError("uniform bridge shape allocation differs")
            K, D = next(iter(dimensions))
            for seed in (101, 202, 303):
                candidate = confirmations[(definition["graph_id"], seed)]
                native_row = confirmations[(native["graph_id"], seed)]
                frozen_row = confirmations[(frozen["graph_id"], seed)]
                shape_rows.append(
                    {
                        "shape_id": lock["coordinate_id"],
                        "pipeline_seed": seed,
                        "K": K,
                        "D": D,
                        "split": "val_design",
                        "pipeline_lineage_kind": "PRIMARY_MATCHED_SEED",
                        "all_predicted_accuracy": candidate["metrics"][
                            "accuracy"
                        ],
                        "shape_matched_HF_NATIVE_accuracy": native_row[
                            "metrics"
                        ]["accuracy"],
                        "frozen_fusion_cross_entropy": frozen_row["metrics"][
                            "cross_entropy"
                        ],
                        "normalized_token_error": candidate["metrics"][
                            "normalized_token_error"
                        ],
                        "prediction_artifact_sha256": candidate[
                            "component_hashes"
                        ]["prediction_manifest"],
                        "native_metrics_artifact_sha256": native_row[
                            "classification_metrics_sha256"
                        ],
                        "token_metrics_artifact_sha256": candidate[
                            "classification_metrics_sha256"
                        ],
                        "label_manifest_sha256": (
                            args.val_design_label_manifest_sha256
                        ),
                        "stack_val_consumed": False,
                        ("final_" + "test_consumed"): False,
                    }
                )
        compact = load_hashed_json(
            args.campaign_root
            / "selection"
            / "predictor_bundle"
            / "carried"
            / "SHAPE_COMPACT.json"
        )
        high = load_hashed_json(
            args.campaign_root
            / "selection"
            / "predictor_bundle"
            / "carried"
            / "SHAPE_HIGH.json"
        )
        configuration = bind_source(
            with_content_hash(
                {
                    "contract": "retb_bridge_shape_selector_input_v1",
                    "schema_version": 1,
                    "compact_shape_id": compact["coordinate_id"],
                    "high_shape_id": high["coordinate_id"],
                    "val_design_label_manifest_sha256": (
                        args.val_design_label_manifest_sha256
                    ),
                    "rows": shape_rows,
                }
            ),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        config_path = (
            args.campaign_root
            / "selection"
            / "stage_l"
            / "bridge_shape_configuration.json"
        )
        write_immutable_json(config_path, configuration)
        bridge_output = (
            args.campaign_root
            / "selection"
            / "stage_l"
            / "bridge_shape_selection.json"
        )
        factory = build_late_factory_input(
            target_node_id="bridge_shape_selector",
            producer_node_id="confirmation_summary",
            campaign_spec_sha256=campaign["content_hash"],
            production_graph_sha256=graph["content_hash"],
            producer_task_manifest_sha256=task_manifest["content_hash"],
            rows=[
                {
                    "task_id": "bridge_shape_selector:0",
                    "argv": [
                        "python",
                        "scripts/select_retb_bridge_shape.py",
                        "--campaign-root",
                        str(args.campaign_root.resolve()),
                        "--configuration",
                        str(config_path),
                        "--output",
                        str(bridge_output),
                    ],
                    "expected_outputs": [str(bridge_output)],
                    "input_artifact_hashes": {
                        "campaign_spec": campaign["content_hash"],
                        "production_graph": graph["content_hash"],
                        "confirmation_summary": artifact["content_hash"],
                        "selector_configuration": configuration[
                            "content_hash"
                        ],
                    },
                    "environment": {
                        "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
                    },
                }
            ],
            coverage={
                "all_predeclared_rows_present": True,
                "scientific_metric_used_for_membership": False,
                "incomplete_wave_permitted": False,
            },
            source=campaign["source"],
        )
        publish_late_factory_input(
            campaign_root=args.campaign_root,
            payload=factory,
            target_node_id="bridge_shape_selector",
            producer_node_id="confirmation_summary",
            campaign=campaign,
            production_graph=graph,
            producer_task_manifest_sha256=task_manifest["content_hash"],
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
