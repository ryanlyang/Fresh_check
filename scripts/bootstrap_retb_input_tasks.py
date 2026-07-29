#!/usr/bin/env python3
"""Publish the checkpoint-free Stage-A offline and HLT-v3 task manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge import (  # noqa: E402
    PRODUCTION_GRAPH_CONTRACT,
    build_task_manifest,
    load_hashed_json,
    validate_production_campaign_binding,
    validate_task_manifest_for_graph,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


ROLES = (
    "model_train",
    "val_stop",
    "val_design",
    "stack_val",
    "final_test",
    "scale_train",
)


def _identity_parent(campaign: dict, role: str) -> str:
    parents = campaign["parent_artifact_hashes"]
    if role in {"model_train", "stack_val", "final_test"}:
        return parents["split_manifest"]
    if role in {"val_stop", "val_design"}:
        return parents["validation_partition_manifest"]
    return parents["scale_train_manifest"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--production-graph", required=True, type=Path)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    graph = load_hashed_json(
        args.production_graph, expected_contract=PRODUCTION_GRAPH_CONTRACT
    )
    validate_production_campaign_binding(graph, campaign)
    if (
        Path(graph["campaign_root"]).resolve() != args.campaign_root.resolve()
        or graph["degradation_profile"] != "D_NOMINAL"
    ):
        raise ValueError("production graph differs from the campaign contract")
    nodes = {row["node_id"]: row for row in graph["nodes"]}
    offline_rows = []
    for index, role in enumerate(ROLES):
        output_dir = args.campaign_root / "inputs" / "offline" / role
        offline_rows.append(
            {
                "task_id": f"offline_input_cache:{index}",
                "argv": [
                    sys.executable,
                    "scripts/build_retb_offline_input_cache.py",
                    "--campaign-root",
                    str(args.campaign_root),
                    "--logical-role",
                    role,
                    "--data-dir",
                    args.data_dir,
                    "--output-dir",
                    str(output_dir),
                ],
                "environment": {},
                "expected_outputs": [
                    str(output_dir / "offline_input_manifest.json")
                ],
                "input_artifact_hashes": {
                    "campaign_spec": campaign["content_hash"],
                    "identity_manifest": _identity_parent(campaign, role),
                },
            }
        )
    offline = build_task_manifest(
        campaign_spec_sha256=campaign["content_hash"],
        production_graph_sha256=graph["content_hash"],
        node_id="offline_input_cache",
        rows=offline_rows,
        maximum_concurrent_tasks=int(
            nodes["offline_input_cache"]["array"][
                "maximum_concurrent_tasks"
            ]
        ),
    )
    hlt_rows = []
    combinations = [
        *(
            (role, replica, "R_MULTI")
            for role in ("model_train", "scale_train")
            for replica in range(4)
        ),
        *(
            (role, 0, "R_FIXED")
            for role in ("val_stop", "val_design", "stack_val", "final_test")
        ),
    ]
    for index, (role, replica, policy) in enumerate(combinations):
        offline_dir = args.campaign_root / "inputs" / "offline" / role
        output_dir = (
            args.campaign_root
            / "inputs"
            / "hlt_v3"
            / role
            / f"replica_{replica}"
            / policy
            / "D_NOMINAL"
        )
        hlt_rows.append(
            {
                "task_id": f"hlt_v3_cache:{index}",
                "argv": [
                    sys.executable,
                    "scripts/build_retb_hlt_v3_from_offline_cache.py",
                    "--campaign-root",
                    str(args.campaign_root),
                    "--offline-input-manifest",
                    str(offline_dir / "offline_input_manifest.json"),
                    "--logical-role",
                    role,
                    "--replica-id",
                    str(replica),
                    "--realization-policy",
                    policy,
                    "--profile-id",
                    "D_NOMINAL",
                    "--output-dir",
                    str(output_dir),
                ],
                "environment": {},
                "expected_outputs": [
                    str(output_dir / "hlt_v3_metadata.json")
                ],
                "input_artifact_hashes": {
                    "campaign_spec": campaign["content_hash"],
                    "offline_input_task_manifest": offline["content_hash"],
                    "hlt_replica_manifest": campaign[
                        "parent_artifact_hashes"
                    ]["hlt_replica_manifest"],
                },
            }
        )
    hlt = build_task_manifest(
        campaign_spec_sha256=campaign["content_hash"],
        production_graph_sha256=graph["content_hash"],
        node_id="hlt_v3_cache",
        rows=hlt_rows,
        maximum_concurrent_tasks=int(
            nodes["hlt_v3_cache"]["array"]["maximum_concurrent_tasks"]
        ),
    )
    for manifest in (offline, hlt):
        validate_task_manifest_for_graph(
            manifest,
            production_graph=graph,
            campaign_root=args.campaign_root,
            repo_root=REPO_ROOT,
        )
    result: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "offline_input_tasks": offline["task_count"],
        "hlt_v3_tasks": hlt["task_count"],
        "offline_input_task_manifest_sha256": offline["content_hash"],
        "hlt_v3_task_manifest_sha256": hlt["content_hash"],
    }
    if not args.dry_run:
        task_root = args.campaign_root / "job_ledgers" / "tasks"
        result["publication"] = {
            "offline_input_cache": write_immutable_json(
                task_root / "offline_input_cache.json", offline
            ),
            "hlt_v3_cache": write_immutable_json(
                task_root / "hlt_v3_cache.json", hlt
            ),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
