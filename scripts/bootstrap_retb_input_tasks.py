#!/usr/bin/env python3
"""Publish all checkpoint-free Stage-A RETB contracts and task manifests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge import (  # noqa: E402
    PRODUCTION_GRAPH_CONTRACT,
    build_static_experiment_bundle,
    build_task_manifest,
    load_hashed_json,
    publish_static_experiment_bundle,
    validate_production_campaign_binding,
    validate_static_experiment_bundle,
    validate_task_manifest_for_graph,
    task_manifest_path_for_graph,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.stage_a import (  # noqa: E402
    STAGE_A_HLT_TREE_VIEWS,
    STAGE_A_OFFLINE_TREE_ROLES,
    build_stage_a_contract_bundle,
    publish_stage_a_contract_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)
from teacher_logit_reco.relation_expert_token_bridge.streamed_abc import (  # noqa: E402
    STREAMED_ABC_PROFILE,
    STREAMED_HLT_NORMALIZER_JETS_PER_REPLICA,
    STREAMED_HLT_VIEWS,
    STREAMED_OFFLINE_ROLES,
    build_streamed_abc_execution_profile,
)
from teacher_logit_reco.relation_expert_token_bridge.streamed_execution import (  # noqa: E402
    FULL_STREAMED_PROFILE,
    STREAMED_SMOKE_PROFILE,
)


ROLES = (
    "model_train",
    "val_stop",
    "val_design",
    "stack_val",
    "final_test",
    "scale_train",
)
HLT_VIEWS = (
    *(
        (role, replica, "R_MULTI")
        for role in ("model_train", "scale_train")
        for replica in range(4)
    ),
    *(
        (role, 0, "R_FIXED")
        for role in ("val_stop", "val_design", "stack_val", "final_test")
    ),
)
REGION_TREE_SHARD_SIZE = 10_000


def _identity_parent(campaign: Mapping[str, Any], role: str) -> str:
    parents = campaign["parent_artifact_hashes"]
    if role in {"model_train", "stack_val", "final_test"}:
        return str(parents["split_manifest"])
    if role in {"val_stop", "val_design"}:
        return str(parents["validation_partition_manifest"])
    return str(parents["scale_train_manifest"])


def _role_counts(graph: Mapping[str, Any]) -> dict[str, int]:
    sizes = {name: int(value) for name, value in graph["split_sizes"].items()}
    if sizes["model_val"] <= 0 or sizes["model_val"] % 2:
        raise ValueError("model_val cannot be partitioned exactly into validation roles")
    return {
        "model_train": sizes["model_train"],
        "val_stop": sizes["model_val"] // 2,
        "val_design": sizes["model_val"] // 2,
        "stack_val": sizes["stack_val"],
        "final_test": sizes["final_test"],
        "scale_train": sizes["scale_train"],
    }


def _maximum_concurrent_tasks(
    nodes: Mapping[str, Mapping[str, Any]], node_id: str
) -> int:
    array = nodes[node_id]["array"]
    return 1 if array is None else int(array["maximum_concurrent_tasks"])


def _manifest(
    *,
    campaign: Mapping[str, Any],
    graph: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
    node_id: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return build_task_manifest(
        campaign_spec_sha256=str(campaign["content_hash"]),
        production_graph_sha256=str(graph["content_hash"]),
        node_id=node_id,
        rows=rows,
        maximum_concurrent_tasks=_maximum_concurrent_tasks(nodes, node_id),
    )


def build_stage_a_task_manifests(
    *,
    campaign: Mapping[str, Any],
    graph: Mapping[str, Any],
    campaign_root: Path,
    data_dir: str,
    stage_a_contracts: Mapping[str, Mapping[str, Any]],
    execution_profile: str = "standard",
) -> dict[str, dict[str, Any]]:
    """Construct the complete deterministic Stage-A execution surface."""

    nodes = {str(row["node_id"]): row for row in graph["nodes"]}
    counts = _role_counts(graph)
    streamed = execution_profile == STREAMED_ABC_PROFILE
    if execution_profile not in {
        "standard", STREAMED_ABC_PROFILE, FULL_STREAMED_PROFILE,
        STREAMED_SMOKE_PROFILE,
    }:
        raise ValueError("Stage-A execution profile differs")
    offline_roles = STREAMED_OFFLINE_ROLES if streamed else ROLES
    hlt_views = STREAMED_HLT_VIEWS if streamed else HLT_VIEWS
    offline_rows: list[dict[str, Any]] = []
    for index, role in enumerate(offline_roles):
        output_dir = campaign_root / "inputs" / "offline" / role
        offline_rows.append(
            {
                "task_id": f"offline_input_cache:{index}",
                "argv": [
                    sys.executable,
                    "scripts/build_retb_offline_input_cache.py",
                    "--campaign-root",
                    str(campaign_root),
                    "--logical-role",
                    role,
                    "--data-dir",
                    data_dir,
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
    offline = _manifest(
        campaign=campaign,
        graph=graph,
        nodes=nodes,
        node_id="offline_input_cache",
        rows=offline_rows,
    )

    hlt_rows: list[dict[str, Any]] = []
    for index, (role, replica, policy) in enumerate(hlt_views):
        offline_dir = campaign_root / "inputs" / "offline" / role
        hlt_namespace = (
            "hlt_v3_streamed_normalizer_sample" if streamed else "hlt_v3"
        )
        output_dir = (
            campaign_root
            / "inputs"
            / hlt_namespace
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
                    str(campaign_root),
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
                    *(
                        [
                            "--selected-jet-limit",
                            str(STREAMED_HLT_NORMALIZER_JETS_PER_REPLICA),
                        ]
                        if streamed
                        else []
                    ),
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
                    "hlt_v3_profile": stage_a_contracts[
                        "hlt_v3_profile"
                    ]["content_hash"],
                },
            }
        )
    hlt = _manifest(
        campaign=campaign,
        graph=graph,
        nodes=nodes,
        node_id="hlt_v3_cache",
        rows=hlt_rows,
    )

    tree_resource = (
        campaign_root / "inputs" / "inherited_angular_tree_resource.json"
    )
    backend_manifest = campaign_root / "backend" / "backend_manifest.json"
    tree_rows: list[dict[str, Any]] = []
    offline_tree_roles = (
        STREAMED_OFFLINE_ROLES if streamed else STAGE_A_OFFLINE_TREE_ROLES
    )
    hlt_tree_views = (
        STREAMED_HLT_VIEWS if streamed else STAGE_A_HLT_TREE_VIEWS
    )
    tree_views = [
        *(
            ("offline", role, None, None)
            for role in offline_tree_roles
        ),
        *(
            ("hlt", role, replica, policy)
            for role, replica, policy in hlt_tree_views
        ),
    ]
    for view_kind, role, replica, policy in tree_views:
        if view_kind == "offline":
            cache_dir = campaign_root / "inputs" / "offline" / role
            output_dir = (
                campaign_root
                / "inputs"
                / "region_tree"
                / "offline"
                / f"{role}_exclusive_ca_v1"
            )
        else:
            hlt_namespace = (
                "hlt_v3_streamed_normalizer_sample"
                if streamed
                else "hlt_v3"
            )
            cache_dir = (
                campaign_root
                / "inputs"
                / hlt_namespace
                / role
                / f"replica_{replica}"
                / str(policy)
                / "D_NOMINAL"
            )
            output_dir = (
                campaign_root
                / "inputs"
                / "region_tree"
                / (
                    "hlt_streamed_normalizer_sample"
                    if streamed
                    else "hlt"
                )
                / f"{role}_r{replica}_exclusive_ca_v1"
            )
        view_count = (
            STREAMED_HLT_NORMALIZER_JETS_PER_REPLICA
            if streamed and view_kind == "hlt"
            else counts[role]
        )
        shard_count = math.ceil(view_count / REGION_TREE_SHARD_SIZE)
        task_index = len(tree_rows)
        argv = [
            sys.executable,
            "scripts/build_retb_region_tree_shard.py",
            "--campaign-root",
            str(campaign_root),
            "--view-kind",
            view_kind,
            "--cache-dir",
            str(cache_dir),
            "--logical-role",
            role,
            "--start",
            "0",
            "--stop",
            str(view_count),
            "--shard-index",
            "0",
            "--shard-size",
            str(REGION_TREE_SHARD_SIZE),
            "--output-dir",
            str(output_dir),
            "--tree-resource",
            str(tree_resource),
            "--backend-manifest",
            str(backend_manifest),
        ]
        if replica is not None:
            argv.extend(
                [
                    "--replica-id",
                    str(replica),
                    "--realization-policy",
                    str(policy),
                ]
            )
        expected_outputs = []
        for shard_index in range(shard_count):
            expected_outputs.extend(
                [
                    str(
                        output_dir
                        / "shards"
                        / f"shard_{shard_index:05d}.npz"
                    ),
                    str(
                        output_dir
                        / "shards"
                        / f"shard_{shard_index:05d}.metadata.json"
                    ),
                ]
            )
        tree_rows.append(
            {
                "task_id": f"region_tree_cache:{task_index}",
                "argv": argv,
                "environment": {},
                "expected_outputs": expected_outputs,
                "input_artifact_hashes": {
                    "campaign_spec": campaign["content_hash"],
                    "source_cache_task_manifest": (
                        offline["content_hash"]
                        if view_kind == "offline"
                        else hlt["content_hash"]
                    ),
                    "angular_tree_resource": stage_a_contracts[
                        "angular_tree_resource"
                    ]["content_hash"],
                },
            }
        )
    trees = _manifest(
        campaign=campaign,
        graph=graph,
        nodes=nodes,
        node_id="region_tree_cache",
        rows=tree_rows,
    )

    tree_index = campaign_root / "inputs" / "region_tree" / (
        "tree_cache_index_streamed_abc.json"
        if streamed
        else "tree_cache_index.json"
    )
    finalize = _manifest(
        campaign=campaign,
        graph=graph,
        nodes=nodes,
        node_id="region_tree_finalize",
        rows=[
            {
                "task_id": "region_tree_finalize:0",
                "argv": [
                    sys.executable,
                    "scripts/finalize_retb_region_tree_cache.py",
                    "--campaign-root",
                    str(campaign_root),
                    "--output",
                    str(tree_index),
                    *(["--streamed-abc"] if streamed else []),
                ],
                "environment": {},
                "expected_outputs": [str(tree_index)],
                "input_artifact_hashes": {
                    "campaign_spec": campaign["content_hash"],
                    "region_tree_task_manifest": trees["content_hash"],
                    "angular_tree_resource": stage_a_contracts[
                        "angular_tree_resource"
                    ]["content_hash"],
                },
            }
        ],
    )

    normalizer_bundle = (
        campaign_root
        / "inputs"
        / "normalization"
        / "stage_a_normalizer_bundle.json"
    )
    normalizers = _manifest(
        campaign=campaign,
        graph=graph,
        nodes=nodes,
        node_id="normalizers_500k",
        rows=[
            {
                "task_id": "normalizers_500k:0",
                "argv": [
                    sys.executable,
                    "scripts/fit_retb_normalizers.py",
                    "--campaign-root",
                    str(campaign_root),
                    "--output",
                    str(normalizer_bundle),
                    *(["--streamed-abc"] if streamed else []),
                ],
                "environment": {},
                "expected_outputs": [
                    str(
                        campaign_root
                        / "inputs"
                        / "normalization"
                        / "offline_500k"
                        / "relation.json"
                    ),
                    str(
                        campaign_root
                        / "inputs"
                        / "normalization"
                        / "offline_500k"
                        / "region.json"
                    ),
                    str(
                        campaign_root
                        / "inputs"
                        / "normalization"
                        / "hlt_shared_500k"
                        / "relation.json"
                    ),
                    str(
                        campaign_root
                        / "inputs"
                        / "normalization"
                        / "hlt_shared_500k"
                        / "region.json"
                    ),
                    str(normalizer_bundle),
                ],
                "input_artifact_hashes": {
                    "campaign_spec": campaign["content_hash"],
                    "stage_a_contract_bundle": stage_a_contracts[
                        "stage_a_contract_bundle"
                    ]["content_hash"],
                    "region_tree_finalize_task_manifest": finalize[
                        "content_hash"
                    ],
                },
            }
        ],
    )

    input_audit = campaign_root / "inputs" / (
        "input_audit_streamed_abc.json" if streamed else "input_audit.json"
    )
    degradation_audit = (
        campaign_root / "inputs" / "hlt_v3_degradation_audit.json"
    )
    audit = _manifest(
        campaign=campaign,
        graph=graph,
        nodes=nodes,
        node_id="input_audit",
        rows=[
            {
                "task_id": "input_audit:0",
                "argv": [
                    sys.executable,
                    "scripts/audit_retb_stage_a_inputs.py",
                    "--campaign-root",
                    str(campaign_root),
                    "--output",
                    str(input_audit),
                    *(["--streamed-abc"] if streamed else []),
                ],
                "environment": {},
                "expected_outputs": [
                    str(degradation_audit),
                    str(input_audit),
                ],
                "input_artifact_hashes": {
                    "campaign_spec": campaign["content_hash"],
                    "hlt_v3_task_manifest": hlt["content_hash"],
                    "region_tree_finalize_task_manifest": finalize[
                        "content_hash"
                    ],
                    "normalizer_task_manifest": normalizers["content_hash"],
                },
            }
        ],
    )
    return {
        "offline_input_cache": offline,
        "hlt_v3_cache": hlt,
        "region_tree_cache": trees,
        "region_tree_finalize": finalize,
        "normalizers_500k": normalizers,
        "input_audit": audit,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--production-graph", required=True, type=Path)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
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
    counts = _role_counts(graph)
    stage_a_contracts = build_stage_a_contract_bundle(
        campaign_spec=campaign,
        model_train_identity_count=counts["model_train"],
        scale_train_identity_count=counts["scale_train"],
        source_snapshot={
            "source_commit": campaign["source"]["commit"],
            "source_status_sha256": campaign["source"]["status_sha256"],
            "source_dirty": campaign["source"]["dirty"],
        },
    )
    manifests = build_stage_a_task_manifests(
        campaign=campaign,
        graph=graph,
        campaign_root=args.campaign_root,
        data_dir=args.data_dir,
        stage_a_contracts=stage_a_contracts,
        execution_profile=str(graph.get("execution_profile", "standard")),
    )
    static_experiments = build_static_experiment_bundle(
        campaign=campaign,
        production_graph=graph,
        campaign_root=args.campaign_root,
    )
    validate_static_experiment_bundle(
        static_experiments,
        campaign=campaign,
        production_graph=graph,
        campaign_root=args.campaign_root,
    )
    for manifest in manifests.values():
        validate_task_manifest_for_graph(
            manifest,
            production_graph=graph,
            campaign_root=args.campaign_root,
            repo_root=REPO_ROOT,
        )
    result: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "stage_a_contract_bundle_sha256": stage_a_contracts[
            "stage_a_contract_bundle"
        ]["content_hash"],
        "task_counts": {
            name: manifest["task_count"]
            for name, manifest in manifests.items()
        },
        "task_manifest_hashes": {
            name: manifest["content_hash"]
            for name, manifest in manifests.items()
        },
        "static_experiment_bundle_sha256": static_experiments[
            "static_experiment_bundle"
        ]["content_hash"],
        "static_experiment_counts": static_experiments[
            "static_experiment_plan"
        ]["execution_counts"],
    }
    if not args.dry_run:
        if graph.get("execution_profile") == STREAMED_ABC_PROFILE:
            profile = build_streamed_abc_execution_profile(
                campaign_id=campaign["campaign_id"],
                campaign_root=args.campaign_root,
                source=campaign["source"],
            )
            result["streamed_execution_profile_publication"] = (
                write_immutable_json(
                    args.campaign_root
                    / "registry"
                    / "retb_streamed_abc_execution_profile.json",
                    profile,
                )
            )
        result["contract_publication"] = publish_stage_a_contract_bundle(
            campaign_root=args.campaign_root,
            bundle=stage_a_contracts,
            campaign_spec=campaign,
        )
        result["task_manifest_publication"] = {
            name: write_immutable_json(
                task_manifest_path_for_graph(
                    graph,
                    node_id=name,
                    campaign_root=args.campaign_root,
                ),
                manifest,
            )
            for name, manifest in manifests.items()
        }
        result["static_experiment_publication"] = (
            publish_static_experiment_bundle(
                campaign_root=args.campaign_root,
                bundle=static_experiments,
                campaign=campaign,
                production_graph=graph,
            )
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
