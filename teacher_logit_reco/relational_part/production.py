"""Immutable Step-8 production DAG, resources, and job-ledger contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    require_git_object_id,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)


PRODUCTION_GRAPH_CONTRACT = "relational_part_production_graph_v2"
JOB_LEDGER_CONTRACT = "relational_part_job_ledger_v1"
TIGRIS_DEFAULTS = {
    "project_dir": "/home/ryreu/atlas/Fresh_check",
    "data_dir": "/home/ryreu/atlas/PracticeTagging/data",
    "output_root": "/home/ryreu/atlas/Fresh_check/checkpoints",
    "conda_base": "/home/ryreu/miniforge3-aarch64",
    "conda_env": "atlas_kd_tigris",
    "python_no_user_site": "1",
    "account": "reu-aisocial",
    "partition": "tigris",
    "gpu_gres": "gpu:gh200:1",
    "gpu_cpus_per_task": 16,
    "gpu_memory": "220G",
    "cpu_cpus_per_task": 16,
    "cpu_memory": "192G",
}


def _node(
    node_id: str,
    *,
    worker: str,
    dependencies: Sequence[str] = (),
    resource: str = "cpu",
    array: str | None = None,
    dynamic: bool = False,
    final_test_access: str = "none",
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "worker": worker,
        "dependencies": list(dependencies),
        "dependency_mode": "afterok",
        "resource": resource,
        "array": array,
        "dynamic_continuation": bool(dynamic),
        "final_test_access": final_test_access,
        "performance_warning_blocks_dependency": False,
        "provenance_failure_blocks_dependency": True,
    }


def build_production_graph(
    *,
    campaign_root: str | Path,
    campaign_id: str,
    source_commit: str,
    source_status_sha256: str,
    miniature: bool = False,
    screening_array_concurrency: int = 4,
    tree_array_concurrency: int = 16,
    region_array_concurrency: int = 16,
) -> dict[str, Any]:
    if int(screening_array_concurrency) <= 0:
        raise ValueError("screening array concurrency must be positive")
    if int(tree_array_concurrency) <= 0:
        raise ValueError("tree array concurrency must be positive")
    if int(region_array_concurrency) <= 0:
        raise ValueError("REGION array concurrency must be positive")
    split_sizes = (
        {
            "model_train": 20,
            "model_val": 10,
            "stack_train": 0,
            "stack_val": 10,
            "final_test": 20,
        }
        if miniature
        else {
            "model_train": 1_000_000,
            "model_val": 125_000,
            "stack_train": 0,
            "stack_val": 125_000,
            "final_test": 500_000,
        }
    )
    nonempty = ("model_train", "model_val", "stack_val", "final_test")
    nodes = [
        _node(
            "split_build",
            worker="run_build_relational_part_splits.sh",
        ),
        _node(
            "campaign_bootstrap",
            worker="run_build_relational_part_campaign.sh",
            dependencies=("split_build",),
        ),
        _node(
            "preconstruction_raw_audit",
            worker="run_audit_relational_part_raw_inputs.sh",
            dependencies=("campaign_bootstrap",),
        ),
        _node(
            "hlt_cache",
            worker="run_build_relational_part_hlt_cache.sh",
            dependencies=("preconstruction_raw_audit",),
            final_test_access="sealed_preparation_only",
        ),
        _node(
            "tree_backend",
            worker="run_build_relational_part_tree_backend.sh",
            dependencies=("preconstruction_raw_audit",),
        ),
        _node(
            "weaver_parity",
            worker="run_validate_relational_part_weaver_parity.sh",
            dependencies=("preconstruction_raw_audit",),
        ),
        _node(
            "relation_normalization",
            worker="run_fit_relational_part_normalization.sh",
            dependencies=("preconstruction_raw_audit", "hlt_cache"),
        ),
        _node(
            "tree_probe",
            worker="run_probe_relational_part_tree_backend.sh",
            dependencies=("hlt_cache", "tree_backend"),
        ),
    ]
    for split in nonempty:
        array_id = f"tree_shards_{split}"
        finalizer_id = f"tree_finalize_{split}"
        shard_count = (split_sizes[split] + 9_999) // 10_000
        nodes.append(
            _node(
                array_id,
                worker="run_build_relational_part_angular_tree_shard.sh",
                dependencies=("tree_probe",),
                array=f"0-{shard_count - 1}%{int(tree_array_concurrency)}",
                final_test_access=(
                    "sealed_preparation_only"
                    if split == "final_test"
                    else "none"
                ),
            )
        )
        nodes.append(
            _node(
                finalizer_id,
                worker="run_finalize_relational_part_angular_tree_cache.sh",
                dependencies=(array_id,),
                final_test_access=(
                    "sealed_preparation_only"
                    if split == "final_test"
                    else "none"
                ),
            )
        )
    nodes.extend(
        [
            _node(
                "region_normalization_plan",
                worker=(
                    "run_prepare_relational_part_"
                    "region_normalization_map.sh"
                ),
                dependencies=(
                    "relation_normalization",
                    "tree_finalize_model_train",
                ),
            ),
            _node(
                "region_normalization_shards",
                worker=(
                    "run_fit_relational_part_"
                    "region_normalization_shard.sh"
                ),
                dependencies=("region_normalization_plan",),
                array=(
                    f"0-{(split_sizes['model_train'] + 9_999) // 10_000 - 1}"
                    f"%{int(region_array_concurrency)}"
                ),
            ),
            _node(
                "region_normalization",
                worker=(
                    "run_finalize_relational_part_"
                    "region_normalization.sh"
                ),
                dependencies=("region_normalization_shards",),
            ),
            _node(
                "postconstruction_input_audit",
                worker="run_audit_relational_part_inputs.sh",
                dependencies=(
                    "hlt_cache",
                    "relation_normalization",
                    "region_normalization",
                    "tree_backend",
                    *tuple(f"tree_finalize_{split}" for split in nonempty),
                ),
                final_test_access="provenance_only",
            ),
            _node(
                "screening_model_contracts",
                worker="run_build_relational_part_model_contracts.sh",
                dependencies=(
                    "postconstruction_input_audit",
                    "weaver_parity",
                ),
            ),
            _node(
                "screening",
                worker="run_train_relational_part.sh",
                dependencies=("screening_model_contracts",),
                resource="gpu",
                array=f"0-20%{int(screening_array_concurrency)}",
            ),
            _node(
                "screening_selection",
                worker="run_select_relational_part_screening.sh",
                dependencies=("screening",),
            ),
            _node(
                "confirmation_submit",
                worker="run_submit_relational_part_confirmation.sh",
                dependencies=("screening_selection",),
                dynamic=True,
            ),
            _node(
                "confirmation_training",
                worker="run_train_relational_part.sh",
                dependencies=("confirmation_submit",),
                resource="gpu",
                dynamic=True,
            ),
            _node(
                "confirmation_summary",
                worker="run_aggregate_relational_part_confirmation.sh",
                dependencies=("confirmation_training",),
                dynamic=True,
            ),
            _node(
                "semantic_controls",
                worker="run_evaluate_relational_part_semantic_controls.sh",
                dependencies=("confirmation_summary",),
                resource="gpu",
                dynamic=True,
            ),
            _node(
                "unary_training",
                worker="run_train_relational_part.sh",
                dependencies=("confirmation_summary",),
                resource="gpu",
                dynamic=True,
            ),
            _node(
                "finalist_lock",
                worker="run_aggregate_relational_part_confirmation.sh",
                dependencies=("semantic_controls", "unary_training"),
                dynamic=True,
            ),
            _node(
                "final_test_submit",
                worker="run_submit_relational_part_final_test.sh",
                dependencies=("finalist_lock",),
                dynamic=True,
            ),
            _node(
                "final_test_evaluation",
                worker="run_evaluate_relational_part_final_test.sh",
                dependencies=("final_test_submit",),
                resource="gpu",
                dynamic=True,
                final_test_access="sealed_scientific_evaluation",
            ),
            _node(
                "final_report",
                worker="run_write_relational_part_report.sh",
                dependencies=("final_test_evaluation",),
                dynamic=True,
                final_test_access="authenticated_prediction_artifacts_only",
            ),
        ]
    )
    artifact = with_content_hash(
        {
            "contract": PRODUCTION_GRAPH_CONTRACT,
            "schema_version": 2,
            "campaign_id": str(campaign_id),
            "campaign_root": str(Path(campaign_root)),
            "campaign_profile": (
                "nonproduction_miniature_test"
                if miniature
                else "production_1m_125k_0_125k_500k"
            ),
            "scientific_results_allowed": not miniature,
            "source_commit": require_git_object_id(
                source_commit, name="source_commit"
            ),
            "source_status_sha256": require_sha256(
                source_status_sha256, name="source_status_sha256"
            ),
            "split_sizes": split_sizes,
            "nonempty_splits": list(nonempty),
            "tigris_defaults": dict(TIGRIS_DEFAULTS),
            "nodes": nodes,
            "screening_row_count": 21,
            "screening_array_concurrency": int(screening_array_concurrency),
            "tree_shard_size": 10_000,
            "tree_array_concurrency": int(tree_array_concurrency),
            "region_array_concurrency": int(region_array_concurrency),
            "final_test_metrics_before_lock_allowed": False,
            "production_submission_performed": False,
        }
    )
    validate_production_graph(artifact)
    return artifact


def _ancestors(nodes: Mapping[str, Mapping[str, Any]], node_id: str) -> set[str]:
    output: set[str] = set()
    pending = list(nodes[node_id]["dependencies"])
    while pending:
        current = pending.pop()
        if current in output:
            continue
        output.add(current)
        pending.extend(nodes[current]["dependencies"])
    return output


def validate_production_graph(graph: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        graph, expected_contract=PRODUCTION_GRAPH_CONTRACT
    )
    if int(graph.get("schema_version", -1)) != 2:
        raise ValueError("production graph schema version differs")
    nodes = list(graph.get("nodes", ()))
    by_id = {str(node["node_id"]): node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("production graph contains duplicate node IDs")
    for node in nodes:
        if node.get("dependency_mode") != "afterok":
            raise ValueError("every scientific dependency must use afterok")
        if any(parent not in by_id for parent in node["dependencies"]):
            raise ValueError(f"{node['node_id']} has an unknown dependency")
        if node.get("performance_warning_blocks_dependency") is not False:
            raise ValueError("performance warnings may not block the DAG")
        if node.get("provenance_failure_blocks_dependency") is not True:
            raise ValueError("provenance failures must block the DAG")
    required_ancestors = {
        "hlt_cache": {"preconstruction_raw_audit"},
        "tree_backend": {"preconstruction_raw_audit"},
        "tree_probe": {"hlt_cache", "tree_backend"},
        "relation_normalization": {
            "preconstruction_raw_audit",
            "hlt_cache",
        },
        "postconstruction_input_audit": {
            "hlt_cache",
            "relation_normalization",
            "region_normalization",
            "tree_backend",
            "tree_finalize_model_train",
            "tree_finalize_model_val",
            "tree_finalize_stack_val",
            "tree_finalize_final_test",
            "region_normalization_plan",
            "region_normalization_shards",
        },
        "screening": {
            "postconstruction_input_audit",
            "screening_model_contracts",
            "weaver_parity",
        },
        "finalist_lock": {
            "confirmation_summary",
            "semantic_controls",
            "unary_training",
        },
        "final_test_evaluation": {"finalist_lock"},
        "final_report": {"final_test_evaluation"},
    }
    for node_id, required in required_ancestors.items():
        if node_id not in by_id:
            raise ValueError(f"production graph lacks {node_id}")
        missing = required - _ancestors(by_id, node_id)
        if missing:
            raise ValueError(f"{node_id} lacks ancestors {sorted(missing)}")
    for node in nodes:
        access = str(node["final_test_access"])
        if access == "sealed_scientific_evaluation" and node[
            "node_id"
        ] != "final_test_evaluation":
            raise ValueError("only the locked evaluator may inspect final metrics")
        if node["node_id"] in {
            "screening",
            "screening_selection",
            "confirmation_training",
            "confirmation_summary",
            "semantic_controls",
            "unary_training",
            "finalist_lock",
        } and access != "none":
            raise ValueError("pre-lock scientific workers may not access final_test")
    defaults = graph.get("tigris_defaults", {})
    for key, expected in TIGRIS_DEFAULTS.items():
        if defaults.get(key) != expected:
            raise ValueError(f"Tigris default {key} drifted")
    return digest


def build_job_ledger(
    *,
    production_graph: Mapping[str, Any],
    jobs: Mapping[str, str | None],
    submission_mode: str,
) -> dict[str, Any]:
    graph_sha = validate_production_graph(production_graph)
    allowed_modes = {"dry_run", "submitted", "smoke_simulation"}
    if submission_mode not in allowed_modes:
        raise ValueError("invalid job-ledger submission mode")
    node_ids = {str(node["node_id"]) for node in production_graph["nodes"]}
    if not set(jobs).issubset(node_ids):
        raise ValueError("job ledger contains an unknown graph node")
    normalized = {}
    for node_id in sorted(jobs):
        value = jobs[node_id]
        if value is not None and (not str(value).isdigit() or int(value) <= 0):
            raise ValueError(f"invalid Slurm job ID for {node_id}")
        normalized[node_id] = None if value is None else str(value)
    return with_content_hash(
        {
            "contract": JOB_LEDGER_CONTRACT,
            "schema_version": 1,
            "production_graph_sha256": graph_sha,
            "campaign_id": production_graph["campaign_id"],
            "campaign_root": production_graph["campaign_root"],
            "submission_mode": submission_mode,
            "jobs": normalized,
            "submitted_node_count": sum(
                value is not None for value in normalized.values()
            ),
            "dynamic_continuations_pending": [
                node["node_id"]
                for node in production_graph["nodes"]
                if node["dynamic_continuation"]
                and normalized.get(node["node_id"]) is None
            ],
        }
    )


__all__ = [
    "JOB_LEDGER_CONTRACT",
    "PRODUCTION_GRAPH_CONTRACT",
    "TIGRIS_DEFAULTS",
    "build_job_ledger",
    "build_production_graph",
    "validate_production_graph",
]
