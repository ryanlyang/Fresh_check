#!/usr/bin/env python3
"""Emit the authenticated Stage-L graph-registration controller row."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import load_hashed_json  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.late_plan_factories import build_late_factory_input, producer_plan_identity_sha256, publish_late_factory_input  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.workflow import load_and_validate_campaign_source  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    graph = load_hashed_json(root / "job_ledgers" / "production_graph.json")
    # Direct-worker identity is derived from the immutable graph, avoiding a
    # circular dependency on a completion that authenticates this file.
    plan_identity = producer_plan_identity_sha256(
        producer_node_id="step13_confirmation_contracts",
        production_graph=graph,
        completion={"contract": "retb_direct_node_completion_v1"},
    )
    step13 = load_hashed_json(
        root / "registry" / "retb_step13_confirmation_shortlist_bundle.json"
    )
    output = root / "selection" / "stage_l" / "registration_controller.json"
    registry = root / "selection" / "stage_l" / "graph_registry.json"
    confirmation_input = (
        root / "job_ledgers" / "factory_inputs" / "confirmation_500k.json"
    )
    row = {
        "task_id": "stage_l_graph_registration:0",
        "argv": [
            "python",
            "scripts/execute_retb_stage_l_registration.py",
            "--campaign-root",
            str(root),
            "--output",
            str(output),
        ],
        "expected_outputs": [
            str(output),
            str(registry),
            str(confirmation_input),
        ],
        "input_artifact_hashes": {
            "campaign_spec": campaign["content_hash"],
            "production_graph": graph["content_hash"],
            "step13_bundle": step13["content_hash"],
        },
        "environment": {
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
        },
    }
    payload = build_late_factory_input(
        target_node_id="stage_l_graph_registration",
        producer_node_id="step13_confirmation_contracts",
        campaign_spec_sha256=campaign["content_hash"],
        production_graph_sha256=graph["content_hash"],
        producer_task_manifest_sha256=plan_identity,
        rows=[row],
        coverage={
            "all_predeclared_rows_present": True,
            "scientific_metric_used_for_membership": False,
            "incomplete_wave_permitted": False,
        },
        source=campaign["source"],
    )
    publish_late_factory_input(
        campaign_root=root,
        payload=payload,
        target_node_id="stage_l_graph_registration",
        producer_node_id="step13_confirmation_contracts",
        campaign=campaign,
        production_graph=graph,
        producer_task_manifest_sha256=plan_identity,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
