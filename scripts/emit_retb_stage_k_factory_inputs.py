#!/usr/bin/env python3
"""Emit authenticated Stage-K factory inputs from completed exports."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import load_hashed_json  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.late_plan_factories import (  # noqa: E402
    ROBUSTNESS_PROFILES,
    ROBUSTNESS_REPLICAS,
    SEMANTIC_CONTROL_KINDS,
    build_late_factory_input,
    publish_late_factory_input,
)
from teacher_logit_reco.relation_expert_token_bridge.production import task_manifest_path_for_graph  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.workflow import load_and_validate_campaign_source  # noqa: E402


def _row(
    *,
    target: str,
    script: str,
    controller_output: Path,
    scientific_output: Path,
    campaign_root: Path,
    campaign_sha: str,
    graph_sha: str,
    export_sha: str,
) -> dict:
    return {
        "task_id": f"{target}:0",
        "argv": [
            "python",
            script,
            "--campaign-root",
            str(campaign_root),
            "--output",
            str(controller_output),
        ],
        "expected_outputs": [
            str(controller_output),
            str(scientific_output),
        ],
        "input_artifact_hashes": {
            "campaign_spec": campaign_sha,
            "production_graph": graph_sha,
            "deployable_export_index": export_sha,
        },
        "environment": {
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    graph = load_hashed_json(root / "job_ledgers" / "production_graph.json")
    manifest_path = task_manifest_path_for_graph(
        graph, node_id="deployable_export", campaign_root=root
    )
    producer_manifest = load_hashed_json(manifest_path)
    export = load_hashed_json(
        root / "selection" / "deployable_export_index.json"
    )
    common = {
        "producer_node_id": "deployable_export",
        "campaign_spec_sha256": campaign["content_hash"],
        "production_graph_sha256": graph["content_hash"],
        "producer_task_manifest_sha256": producer_manifest["content_hash"],
        "source": campaign["source"],
    }
    specifications = {
        "robustness_controls": (
            _row(
                target="robustness_controls",
                script="scripts/execute_retb_robustness_campaign.py",
                controller_output=(
                    root / "job_ledgers" / "controllers" / "robustness_campaign.json"
                ),
                scientific_output=(
                    root / "controls" / "robustness" / "robustness_bundle.json"
                ),
                campaign_root=root,
                campaign_sha=campaign["content_hash"],
                graph_sha=graph["content_hash"],
                export_sha=export["content_hash"],
            ),
            {
                "required_profile_replica_coordinates": [
                    [profile, replica]
                    for profile in ROBUSTNESS_PROFILES
                    for replica in ROBUSTNESS_REPLICAS
                ]
            },
        ),
        "semantic_controls": (
            _row(
                target="semantic_controls",
                script="scripts/execute_retb_semantic_control_campaign.py",
                controller_output=(
                    root / "job_ledgers" / "controllers" / "semantic_control_campaign.json"
                ),
                scientific_output=(
                    root / "controls" / "semantics" / "semantic_controls_bundle.json"
                ),
                campaign_root=root,
                campaign_sha=campaign["content_hash"],
                graph_sha=graph["content_hash"],
                export_sha=export["content_hash"],
            ),
            {
                "required_semantic_control_kinds": list(
                    SEMANTIC_CONTROL_KINDS
                )
            },
        ),
    }
    for target, (row, target_coverage) in specifications.items():
        coverage = {
            "all_predeclared_rows_present": True,
            "scientific_metric_used_for_membership": False,
            "incomplete_wave_permitted": False,
            **target_coverage,
        }
        payload = build_late_factory_input(
            target_node_id=target,
            rows=[row],
            coverage=coverage,
            **common,
        )
        publish_late_factory_input(
            campaign_root=root,
            payload=payload,
            target_node_id=target,
            producer_node_id="deployable_export",
            campaign=campaign,
            production_graph=graph,
            producer_task_manifest_sha256=producer_manifest["content_hash"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
