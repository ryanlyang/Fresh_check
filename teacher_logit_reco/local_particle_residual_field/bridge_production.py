"""Step 10 production graph, Slurm commands, ledgers, and CPU rehearsal."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bridge_campaign import (
    MEASUREMENT_MEASURED,
    PAIRED_SEED_IDS,
    POST_TEACHER_CONFIGURATION_COUNT,
    RECONSTRUCTION_BREADTH_COUNT,
    REGISTRY_CONFIGURATION_COUNT,
    validate_campaign_registry,
)
from .bridge_campaign_policy import (
    PREDICTION_ANCHORED_CAMPAIGN_RESERVATION_CONTRACT,
)
from .bridge_contracts import validate_content_hash, with_content_hash
from .bridge_execution import validate_prediction_anchored_execution_spec


PREDICTION_ANCHORED_TIGRIS_RESOURCES_CONTRACT = (
    "prediction_anchored_tigris_resources_v1"
)
PREDICTION_ANCHORED_PRODUCTION_NODE_CONTRACT = (
    "prediction_anchored_production_node_v1"
)
PREDICTION_ANCHORED_PRODUCTION_GRAPH_CONTRACT = (
    "prediction_anchored_production_graph_v1"
)
PREDICTION_ANCHORED_JOB_LEDGER_CONTRACT = "prediction_anchored_job_id_ledger_v1"
PREDICTION_ANCHORED_SCHEDULER_SIMULATION_CONTRACT = (
    "prediction_anchored_scheduler_simulation_v1"
)
PREDICTION_ANCHORED_CPU_REHEARSAL_CONTRACT = (
    "prediction_anchored_step10_cpu_rehearsal_v1"
)
PREDICTION_ANCHORED_ALLOCATION_LAUNCH_CONTRACT = (
    "prediction_anchored_allocation_launch_v1"
)

TIGRIS_ACCOUNT = "reu-aisocial"
TIGRIS_PARTITION = "tigris"
PAIRED_PROFILE = "paired3"
MAX_CONFIGS_PER_PACK = 4


@dataclass(frozen=True)
class TigrisResources:
    account: str = TIGRIS_ACCOUNT
    partition: str = TIGRIS_PARTITION
    nodes: int = 1
    # Tigris GH200 nodes expose one accelerator per node.  Paired3 denotes
    # three scientific seeds, not three/four simultaneously allocated GPUs;
    # the executors retain all seeds and run them sequentially when one device
    # is visible.
    gpus_per_node: int = 1
    cpus_per_task: int = 12
    host_memory_gib: int = 512
    walltime: str = "3-00:00:00"

    def __post_init__(self) -> None:
        if self.account != TIGRIS_ACCOUNT:
            raise ValueError("Tigris account must be the full reu-aisocial string")
        if self.partition != TIGRIS_PARTITION:
            raise ValueError("prediction-anchored production is locked to the tigris partition")
        if int(self.nodes) != 1:
            raise ValueError("packed prediction-anchored allocations must request exactly one node")
        if int(self.gpus_per_node) not in {0, 1} or int(self.cpus_per_task) <= 0:
            raise ValueError("invalid Tigris accelerator/CPU resource request")
        if int(self.host_memory_gib) < 64:
            raise ValueError("prediction-anchored jobs must request explicit host memory")
        if not str(self.walltime).strip():
            raise ValueError("Tigris walltime must be explicit")

    def to_artifact(self) -> dict[str, Any]:
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_TIGRIS_RESOURCES_CONTRACT,
                **asdict(self),
                "python_no_user_site": True,
                "single_node": True,
            }
        )


def _node(
    *,
    node_id: str,
    stage: str,
    runner: str,
    arguments: Sequence[str],
    dependencies: Sequence[str],
    afterany_dependencies: Sequence[str] = (),
    configuration_run_ids: Sequence[str] = (),
    resources: TigrisResources,
    shared_source_group: str,
    teacher_namespace: str | None = None,
    requires_selected_consumer: bool = False,
    protected_final_test: bool = False,
    persistent_reservation_bytes: int = 0,
) -> dict[str, Any]:
    configs = [str(value) for value in configuration_run_ids]
    if len(configs) != len(set(configs)):
        raise ValueError(f"production node {node_id} repeats a configuration")
    if persistent_reservation_bytes < 0:
        raise ValueError("node persistent reservation cannot be negative")
    afterany = [str(value) for value in afterany_dependencies]
    if not set(afterany).issubset(str(value) for value in dependencies):
        raise ValueError("afterany dependencies must be ordinary graph dependencies")
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_PRODUCTION_NODE_CONTRACT,
            "node_id": str(node_id),
            "stage": str(stage),
            "runner": str(runner),
            "arguments": [str(value) for value in arguments],
            "dependencies": [str(value) for value in dependencies],
            "afterany_dependencies": afterany,
            "configuration_run_ids": configs,
            "paired_seed_ids": list(PAIRED_SEED_IDS) if configs else [],
            "resources": resources.to_artifact(),
            "shared_source_group": str(shared_source_group),
            "teacher_namespace": teacher_namespace,
            "requires_selected_consumer": bool(requires_selected_consumer),
            "protected_final_test": bool(protected_final_test),
            "persistent_reservation_bytes": int(persistent_reservation_bytes),
            "allocation_packing": {
                "single_node": True,
                "allocation_leader_rank": 0,
                "one_persistent_source_open_by_leader": True,
                "shared_allocation_ram_ledger": True,
                "raw_shards_non_evictable": True,
                "derived_only_lru": True,
                "cross_allocation_resume": False,
                "preemption_policy": "restart_whole_configuration_pack",
            },
            "persistent_dense_field_output_paths": [],
            "publication_policy": "metrics_all_seeds__weights_ordered_median_only",
        }
    )


def _pack(values: Sequence[str], size: int = MAX_CONFIGS_PER_PACK) -> list[list[str]]:
    return [list(values[start : start + size]) for start in range(0, len(values), size)]


def _teacher_group(row: Mapping[str, Any]) -> str:
    run_id = str(row["canonical_run_id"])
    family = str(row["family"])
    if run_id.startswith("A0_CAP500_"):
        return "direct_hlt"
    if family == "all50":
        return "all50_selected_bridge_teacher"
    if family == "alternate_teacher":
        return "physical45_alternate_bridge_teacher"
    if run_id == "D10_N3_nonprivileged_teacher_kd":
        return "physical45_selected_teacher_on_f0_control"
    return "physical45_selected_bridge_teacher"


def _topological_node_ids(nodes: Sequence[Mapping[str, Any]]) -> list[str]:
    by_id = {str(row["node_id"]): row for row in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("production graph contains duplicate node IDs")
    unknown = {
        dependency
        for row in nodes
        for dependency in row["dependencies"]
        if dependency not in by_id
    }
    if unknown:
        raise ValueError(f"production graph contains unknown dependencies: {sorted(unknown)}")
    remaining = set(by_id)
    ordered: list[str] = []
    while remaining:
        ready = sorted(
            node_id
            for node_id in remaining
            if set(by_id[node_id]["dependencies"]).issubset(ordered)
        )
        if not ready:
            raise ValueError("production dependency graph contains a cycle")
        ordered.extend(ready)
        remaining.difference_update(ready)
    return ordered


def build_prediction_anchored_tigris_graph(
    registry: Mapping[str, Any],
    *,
    reservations: Mapping[str, Any],
    execution_spec: Mapping[str, Any],
    artifact_root: str,
    pack_size: int = MAX_CONFIGS_PER_PACK,
) -> dict[str, Any]:
    """Render B0--B6 directly from the immutable measured registry."""

    validate_campaign_registry(registry)
    validate_content_hash(
        reservations, expected_contract=PREDICTION_ANCHORED_CAMPAIGN_RESERVATION_CONTRACT
    )
    validate_content_hash(
        execution_spec, expected_contract="prediction_anchored_execution_spec_v1"
    )
    if reservations.get("registry_sha256") != registry["content_hash"]:
        raise ValueError("Step 10 reservations are bound to another registry")
    child_manifest = execution_spec.get("child_manifest", {})
    parent_manifest = execution_spec.get("parent_manifest", {})
    representative_sha256 = str(
        reservations.get("representative_reference_sha256", "")
    )
    if (
        len(representative_sha256) != 64
        or any(char not in "0123456789abcdef" for char in representative_sha256)
    ):
        raise ValueError(
            "Step 10 reservations have no valid representative-reference binding"
        )
    expected_bindings = {
        "execution_spec_sha256": execution_spec["content_hash"],
        "child_manifest_sha256": child_manifest.get("content_hash"),
        "parent_manifest_file_sha256": parent_manifest.get("sha256"),
        "representative_reference_sha256": representative_sha256,
    }
    for name, expected in expected_bindings.items():
        if reservations.get(name) != expected:
            raise ValueError(f"Step 10 reservations have a different {name}")
    if not str(artifact_root).strip():
        raise ValueError("production graph requires an explicit artifact root")
    if int(pack_size) <= 0 or int(pack_size) > 4:
        raise ValueError("Tigris pack size must be between one and four GPU tasks")
    unmeasured = [
        row["canonical_run_id"]
        for row in registry["runs"]
        if row["execution_status"] == "RUNNABLE"
        and row["measurement_status"] != MEASUREMENT_MEASURED
    ]
    if unmeasured:
        raise PermissionError(
            "production graph refuses runnable UNMEASURED rows: " + ", ".join(unmeasured)
        )
    if int(reservations["projected_persistent_bytes"]) > int(
        reservations["selected_budget_bytes"]
    ):
        raise PermissionError("production graph exceeds its measured persistent budget")

    gpu = TigrisResources()
    one_gpu = TigrisResources(gpus_per_node=1, cpus_per_task=12, host_memory_gib=512)
    cpu = TigrisResources(gpus_per_node=0, cpus_per_task=16, host_memory_gib=192, walltime="12:00:00")
    highmem_cpu = TigrisResources(gpus_per_node=0, cpus_per_task=24, host_memory_gib=512, walltime="1-00:00:00")
    run_bytes = {str(key): int(value) for key, value in reservations["run_reservations_bytes"].items()}
    nodes: list[dict[str, Any]] = []

    nodes.append(
        _node(
            node_id="b0_validate_preflight",
            stage="B0",
            runner="run_prepare_prediction_anchored_bridge_ram.sh",
            arguments=("B0",),
            dependencies=(),
            resources=highmem_cpu,
            shared_source_group="source_preflight",
        )
    )
    nodes.append(
        _node(
            node_id="b1_train_register_r0",
            stage="B1",
            runner="run_prepare_prediction_anchored_bridge_ram.sh",
            arguments=("B1",),
            dependencies=("b0_validate_preflight",),
            resources=one_gpu,
            shared_source_group="r0_streamed_truth",
        )
    )
    nodes.append(
        _node(
            node_id="b2_stage_recipes_scalers",
            stage="B2",
            runner="run_prepare_prediction_anchored_bridge_ram.sh",
            arguments=("B2",),
            dependencies=("b1_train_register_r0",),
            resources=highmem_cpu,
            shared_source_group="physical45_all50_sources",
        )
    )

    b3_rows = [row for row in registry["runs"] if row["stage"] == "B3"]
    l0 = [row["canonical_run_id"] for row in b3_rows if row["family"] == "loss"]
    upstream = [row["canonical_run_id"] for row in b3_rows if row["family"] == "upstream"]
    if l0 != ["D10_L0_bridge_only"] or len(upstream) != 8:
        raise ValueError("registry B3 inventory no longer matches L0 plus eight paired upstream rows")
    nodes.append(
        _node(
            node_id="b3_l0_paired3",
            stage="B3",
            runner="run_train_prediction_anchored_bridge_reconstructor.sh",
            arguments=("b3_l0_paired3",),
            dependencies=("b2_stage_recipes_scalers",),
            configuration_run_ids=l0,
            resources=gpu,
            shared_source_group="early_l0_physical45",
            persistent_reservation_bytes=sum(run_bytes[value] for value in l0),
        )
    )
    # All eight upstream rows share the Tpred/A0 RAM branch lineage and remain
    # in one allocation.  Their paired seeds execute sequentially on the
    # single Tigris accelerator without changing the scientific inventory.
    nodes.append(
        _node(
            node_id="b3_consumers_paired3",
            stage="B3",
            runner="run_train_prediction_anchored_bridge_consumer.sh",
            arguments=("b3_consumers_paired3",),
            dependencies=("b2_stage_recipes_scalers",),
            configuration_run_ids=upstream,
            resources=gpu,
            shared_source_group="consumer_tpred_branch_lineage",
            persistent_reservation_bytes=sum(run_bytes[value] for value in upstream),
        )
    )
    nodes.append(
        _node(
            node_id="b4_select_consumer",
            stage="B4",
            runner="run_prepare_prediction_anchored_bridge_ram.sh",
            arguments=("B4_SELECT",),
            dependencies=("b3_consumers_paired3",),
            resources=cpu,
            shared_source_group="consumer_metrics",
        )
    )
    nodes.append(
        _node(
            node_id="b4_confirm_consumer",
            stage="B4",
            runner="run_prepare_prediction_anchored_bridge_ram.sh",
            arguments=("B4_CONFIRM",),
            dependencies=("b4_select_consumer",),
            resources=cpu,
            shared_source_group="stack_val_consumer_one_shot",
            # This node creates the confirmed selection.  Requiring the final
            # file at allocation launch would make the gate impossible.
            requires_selected_consumer=False,
        )
    )
    nodes.append(
        _node(
            node_id="b4_publish_runtime_resources",
            stage="B4",
            runner="run_prepare_prediction_anchored_bridge_ram.sh",
            arguments=("B4_RUNTIME_RESOURCES",),
            dependencies=("b4_confirm_consumer",),
            resources=cpu,
            shared_source_group="confirmed_runtime_resource_profile",
            requires_selected_consumer=True,
        )
    )
    nodes.append(
        _node(
            node_id="b5_bind_teachers",
            stage="B5",
            runner="run_prepare_prediction_anchored_bridge_ram.sh",
            arguments=("B5_BIND",),
            dependencies=("b4_publish_runtime_resources",),
            resources=cpu,
            shared_source_group="immutable_teacher_bindings",
            requires_selected_consumer=True,
        )
    )
    cache_nodes = []
    for short, namespace in (
        ("primary", "physical45_selected_bridge_teacher"),
        ("all50", "all50_selected_bridge_teacher"),
        ("n3", "physical45_selected_teacher_on_f0_control"),
    ):
        node_id = f"b5_cache_{short}"
        cache_nodes.append(node_id)
        nodes.append(
            _node(
                node_id=node_id,
                stage="B5",
                runner="run_cache_prediction_anchored_bridge_logits.sh",
                arguments=(namespace,),
                dependencies=("b5_bind_teachers",),
                resources=one_gpu,
                shared_source_group=f"cache_{namespace}",
                teacher_namespace=namespace,
                requires_selected_consumer=namespace != "all50_selected_bridge_teacher",
            )
        )
    if bool(registry["alternate_teacher_valid"]):
        namespace = "physical45_alternate_bridge_teacher"
        cache_nodes.append("b5_cache_alternate")
        nodes.append(
            _node(
                node_id="b5_cache_alternate",
                stage="B5",
                runner="run_cache_prediction_anchored_bridge_logits.sh",
                arguments=(namespace,),
                dependencies=("b5_bind_teachers",),
                resources=one_gpu,
                shared_source_group=f"cache_{namespace}",
                teacher_namespace=namespace,
            )
        )
    nodes.append(
        _node(
            node_id="b5_release_postteacher",
            stage="B5",
            runner="run_prepare_prediction_anchored_bridge_ram.sh",
            arguments=("B5_RELEASE",),
            dependencies=tuple(cache_nodes),
            resources=cpu,
            shared_source_group="teacher_release_gate",
            requires_selected_consumer=True,
        )
    )
    nodes.append(
        _node(
            node_id="b6_l0_postteacher_eval_paired3",
            stage="B6",
            runner="run_train_prediction_anchored_bridge_reconstructor.sh",
            arguments=("b6_l0_postteacher_eval_paired3",),
            dependencies=("b3_l0_paired3", "b5_release_postteacher"),
            configuration_run_ids=(),
            resources=gpu,
            shared_source_group="l0_postteacher_common_model_val_select",
            teacher_namespace="physical45_selected_bridge_teacher",
            requires_selected_consumer=True,
            persistent_reservation_bytes=0,
        )
    )

    post_rows = [row for row in registry["runs"] if bool(row["post_teacher_configuration"])]
    if len(post_rows) != POST_TEACHER_CONFIGURATION_COUNT:
        raise ValueError("registry post-teacher inventory no longer contains 45 rows")
    grouped: dict[str, list[str]] = {}
    skipped = []
    for row in post_rows:
        run_id = str(row["canonical_run_id"])
        if row["execution_status"] != "RUNNABLE":
            skipped.append(
                {
                    "run_id": run_id,
                    "status": row["execution_status"],
                    "reason": "conditional parent is invalid",
                }
            )
            continue
        grouped.setdefault(_teacher_group(row), []).append(run_id)

    b6_nodes = []
    for group in sorted(grouped):
        for index, run_ids in enumerate(_pack(sorted(grouped[group]), int(pack_size)), start=1):
            safe_group = group.replace("_selected_bridge_teacher", "").replace(
                "physical45_", "p45_"
            )
            node_id = f"b6_{safe_group}_pack{index:02d}"
            b6_nodes.append(node_id)
            nodes.append(
                _node(
                    node_id=node_id,
                    stage="B6",
                    runner="run_train_prediction_anchored_bridge_reconstructor.sh",
                    arguments=(node_id,),
                    dependencies=("b5_release_postteacher",),
                    configuration_run_ids=run_ids,
                    resources=gpu,
                    shared_source_group=f"b6_{group}",
                    teacher_namespace=(None if group == "direct_hlt" else group),
                    requires_selected_consumer=group
                    in {
                        "physical45_selected_bridge_teacher",
                        "physical45_selected_teacher_on_f0_control",
                    },
                    persistent_reservation_bytes=sum(run_bytes[value] for value in run_ids),
                )
            )

    selection_dependencies = sorted(
        b6_nodes + ["b6_l0_postteacher_eval_paired3"]
    )
    nodes.append(
        _node(
            node_id="b6_aggregate_select_deployable",
            stage="B6",
            runner="run_prepare_prediction_anchored_bridge_ram.sh",
            arguments=("B6_SELECT",),
            dependencies=selection_dependencies,
            afterany_dependencies=selection_dependencies,
            resources=cpu,
            shared_source_group="paired3_model_val_select_metrics",
            requires_selected_consumer=True,
        )
    )
    nodes.append(
        _node(
            node_id="b6_confirm_deployable",
            stage="B6",
            runner="run_prepare_prediction_anchored_bridge_ram.sh",
            arguments=("DEPLOY_CONFIRM",),
            dependencies=("b6_aggregate_select_deployable",),
            resources=cpu,
            shared_source_group="stack_val_deploy_one_shot",
            requires_selected_consumer=True,
        )
    )
    nodes.append(
        _node(
            node_id="b6_report_export_reload",
            stage="B6",
            runner="run_prepare_prediction_anchored_bridge_ram.sh",
            arguments=("REPORT_EXPORT",),
            dependencies=("b6_confirm_deployable",),
            resources=cpu,
            shared_source_group="locked_bundle_and_reports",
            requires_selected_consumer=True,
            persistent_reservation_bytes=int(reservations["fixed_storage_reserved_bytes"]),
        )
    )
    nodes.append(
        _node(
            node_id="final_test_hlt_only",
            stage="FINAL_TEST",
            runner="run_prepare_prediction_anchored_bridge_ram.sh",
            arguments=("FINAL_TEST",),
            dependencies=("b6_report_export_reload",),
            resources=one_gpu,
            shared_source_group="locked_hlt_only_final_test",
            requires_selected_consumer=True,
            protected_final_test=True,
        )
    )

    ordered = _topological_node_ids(nodes)
    by_id = {row["node_id"]: row for row in nodes}
    nodes = [by_id[node_id] for node_id in ordered]
    covered = [run_id for row in nodes for run_id in row["configuration_run_ids"]]
    expected_runnable = {
        row["canonical_run_id"]
        for row in registry["runs"]
        if row["execution_status"] == "RUNNABLE"
    }
    if set(covered) != expected_runnable or len(covered) != len(set(covered)):
        raise AssertionError("generated production packs do not cover each runnable row exactly once")
    reserved_by_nodes = sum(int(row["persistent_reservation_bytes"]) for row in nodes)
    if reserved_by_nodes != int(reservations["projected_persistent_bytes"]):
        raise AssertionError(
            "allocation reservations do not reconcile with the measured campaign projection"
        )
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_PRODUCTION_GRAPH_CONTRACT,
            "profile": PAIRED_PROFILE,
            "registry_sha256": registry["content_hash"],
            "reservations_sha256": reservations["content_hash"],
            **expected_bindings,
            "artifact_root": str(Path(artifact_root)),
            "account": TIGRIS_ACCOUNT,
            "partition": TIGRIS_PARTITION,
            "python_no_user_site": True,
            "configuration_count": REGISTRY_CONFIGURATION_COUNT,
            "reconstruction_breadth_count": RECONSTRUCTION_BREADTH_COUNT,
            "post_teacher_configuration_count": POST_TEACHER_CONFIGURATION_COUNT,
            "runnable_configuration_count": len(expected_runnable),
            "covered_runnable_configuration_count": len(covered),
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "pack_size": int(pack_size),
            "nodes": nodes,
            "node_count": len(nodes),
            "topological_node_ids": ordered,
            "conditional_skips": skipped,
            "selected_consumer_runtime_path": str(
                Path(artifact_root) / "selection" / "selected_bridge_consumer.json"
            ),
            "stage1b_guessed_consumer_allowed": False,
            "final_test_automatic_submission": False,
            "final_test_hlt_only": True,
            "final_test_privileged_environment_scrub_required": True,
            "cross_allocation_resume": False,
            "allocation_preemption_policy": "restart_whole_configuration_pack",
            "persistent_dense_field_output_paths": [],
            "selected_budget_bytes": int(reservations["selected_budget_bytes"]),
            "projected_persistent_bytes": int(reservations["projected_persistent_bytes"]),
            "node_reservations_total_bytes": reserved_by_nodes,
            "node_reservations_reconciled": True,
            "production_submission_ready": True,
            "actual_submission_requires_configured_scientific_executors": False,
            "repository_owned_deployable_export": True,
            "repository_owned_hlt_only_final_test": True,
        }
    )


def validate_prediction_anchored_tigris_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    validate_content_hash(graph, expected_contract=PREDICTION_ANCHORED_PRODUCTION_GRAPH_CONTRACT)
    for name in (
        "registry_sha256",
        "reservations_sha256",
        "execution_spec_sha256",
        "child_manifest_sha256",
        "parent_manifest_file_sha256",
        "representative_reference_sha256",
    ):
        value = str(graph.get(name, ""))
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"production graph has an invalid {name}")
    if graph.get("account") != TIGRIS_ACCOUNT or graph.get("partition") != TIGRIS_PARTITION:
        raise ValueError("production graph changed its Tigris account/partition")
    if not bool(graph.get("python_no_user_site")):
        raise ValueError("production graph must disable user-site Python packages")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("production graph nodes must be a list")
    allowed_runners = {
        "run_prepare_prediction_anchored_bridge_ram.sh",
        "run_train_prediction_anchored_bridge_consumer.sh",
        "run_cache_prediction_anchored_bridge_logits.sh",
        "run_train_prediction_anchored_bridge_reconstructor.sh",
    }
    for row in nodes:
        validate_content_hash(row, expected_contract=PREDICTION_ANCHORED_PRODUCTION_NODE_CONTRACT)
        resources = row["resources"]
        validate_content_hash(resources, expected_contract=PREDICTION_ANCHORED_TIGRIS_RESOURCES_CONTRACT)
        TigrisResources(
            account=resources["account"],
            partition=resources["partition"],
            nodes=resources["nodes"],
            gpus_per_node=resources["gpus_per_node"],
            cpus_per_task=resources["cpus_per_task"],
            host_memory_gib=resources["host_memory_gib"],
            walltime=resources["walltime"],
        )
        packing = row["allocation_packing"]
        if (
            packing.get("allocation_leader_rank") != 0
            or not packing.get("one_persistent_source_open_by_leader")
            or not packing.get("shared_allocation_ram_ledger")
            or packing.get("cross_allocation_resume") is not False
        ):
            raise ValueError("production node changed the one-leader/RAM/restart contract")
        if row.get("persistent_dense_field_output_paths") != []:
            raise ValueError("production graph contains a dense field output path")
        if row.get("runner") not in allowed_runners:
            raise ValueError("production graph contains an unknown Slurm runner")
        if row.get("configuration_run_ids") and row.get("paired_seed_ids") != list(PAIRED_SEED_IDS):
            raise ValueError("a production training node changed the paired3 seed inventory")
        if row.get("configuration_run_ids") and row.get("publication_policy") != (
            "metrics_all_seeds__weights_ordered_median_only"
        ):
            raise ValueError("a production node changed median-only publication")
        if not set(row.get("afterany_dependencies", [])).issubset(row["dependencies"]):
            raise ValueError("production node has an invalid afterany dependency")
    order = _topological_node_ids(nodes)
    if order != graph.get("topological_node_ids"):
        raise ValueError("production graph topological order changed")
    covered = [value for row in nodes for value in row["configuration_run_ids"]]
    if len(covered) != int(graph["covered_runnable_configuration_count"]):
        raise ValueError("production graph configuration coverage changed")
    if len(covered) != len(set(covered)):
        raise ValueError("production graph schedules a configuration more than once")
    if int(graph.get("node_reservations_total_bytes", -1)) != sum(
        int(row["persistent_reservation_bytes"]) for row in nodes
    ):
        raise ValueError("production graph node reservations changed")
    if int(graph.get("node_reservations_total_bytes", -1)) != int(
        graph.get("projected_persistent_bytes", -2)
    ) or graph.get("node_reservations_reconciled") is not True:
        raise ValueError("production graph reservations no longer reconcile")
    selected_required = {
        "b5_bind_teachers",
        "b5_cache_primary",
        "b5_cache_n3",
        "b5_release_postteacher",
        "b6_aggregate_select_deployable",
        "b6_confirm_deployable",
        "b6_report_export_reload",
        "final_test_hlt_only",
    }
    selected_required.update(
        row["node_id"]
        for row in nodes
        if row.get("teacher_namespace")
        in {
            "physical45_selected_bridge_teacher",
            "physical45_selected_teacher_on_f0_control",
        }
    )
    by_id = {row["node_id"]: row for row in nodes}
    if any(not bool(by_id[node_id]["requires_selected_consumer"]) for node_id in selected_required):
        raise ValueError("a primary-teacher stage permits a guessed consumer")
    final_nodes = [row for row in nodes if row.get("protected_final_test")]
    if len(final_nodes) != 1 or final_nodes[0]["node_id"] != "final_test_hlt_only":
        raise ValueError("production graph changed the protected final-test node")
    if graph.get("final_test_automatic_submission") is not False:
        raise ValueError("final test may not be automatically submitted")
    if graph.get("final_test_hlt_only") is not True:
        raise ValueError("production final-test must be HLT-only")
    if graph.get("final_test_privileged_environment_scrub_required") is not True:
        raise ValueError("production final-test must scrub privileged environment variables")
    if graph.get("stage1b_guessed_consumer_allowed") is not False:
        raise ValueError("production graph permits a guessed Stage B5/B6 consumer")
    return {
        "ok": True,
        "node_count": len(nodes),
        "covered_runnable_configuration_count": len(covered),
        "conditional_skip_count": len(graph["conditional_skips"]),
        "configuration_count": int(graph["configuration_count"]),
        "reconstruction_breadth_count": int(graph["reconstruction_breadth_count"]),
        "post_teacher_configuration_count": int(graph["post_teacher_configuration_count"]),
    }


def render_tigris_sbatch_commands(
    graph: Mapping[str, Any],
    *,
    include_final_test: bool = False,
) -> dict[str, Any]:
    validate_prediction_anchored_tigris_graph(graph)
    commands = []
    for row in graph["nodes"]:
        if row["protected_final_test"] and not include_final_test:
            continue
        resources = row["resources"]
        argv = [
            "sbatch",
            "--parsable",
            f"--account={TIGRIS_ACCOUNT}",
            f"--partition={TIGRIS_PARTITION}",
            "--nodes=1",
            f"--cpus-per-task={resources['cpus_per_task']}",
            f"--mem={resources['host_memory_gib']}G",
            f"--time={resources['walltime']}",
            "--kill-on-invalid-dep=yes",
            (
                "--export=ALL,PYTHONNOUSERSITE=1,"
                "PREDICTION_ANCHORED_GRAPH=<IMMUTABLE_GRAPH_PATH>,"
                f"PREDICTION_ANCHORED_NODE_ID={row['node_id']},"
                f"PREDICTION_ANCHORED_ARTIFACT_ROOT={graph['artifact_root']}"
            ),
        ]
        if int(resources["gpus_per_node"]) > 0:
            argv.append(f"--gres=gpu:{resources['gpus_per_node']}")
        if row["dependencies"]:
            afterany = set(row.get("afterany_dependencies", []))
            afterok_ids = [value for value in row["dependencies"] if value not in afterany]
            clauses = []
            if afterok_ids:
                clauses.append("afterok:" + ":".join(f"${{JOB_{value}}}" for value in afterok_ids))
            if afterany:
                clauses.append("afterany:" + ":".join(f"${{JOB_{value}}}" for value in row["dependencies"] if value in afterany))
            argv.append("--dependency=" + ",".join(clauses))
        argv.extend(
            [
                f"sbatch/{row['runner']}",
                *row["arguments"],
            ]
        )
        commands.append(
            {
                "node_id": row["node_id"],
                "dependencies": list(row["dependencies"]),
                "argv": argv,
                "shell_preview": " ".join(argv),
                "submission_executed": False,
            }
        )
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_PRODUCTION_GRAPH_CONTRACT,
            "graph_sha256": graph["content_hash"],
            "include_final_test": bool(include_final_test),
            "commands": commands,
            "command_count": len(commands),
            "submission_executed": False,
            "explicit_execution_command": (
                "PREDICTION_ANCHORED_EXECUTE=1 "
                "bash sbatch/submit_prediction_anchored_bridge_pilot.sh"
            ),
        }
    )


def build_allocation_launch_manifest(
    graph: Mapping[str, Any],
    *,
    node_id: str,
    environment: Mapping[str, Any],
    ram_root: str,
    selected_consumer: Mapping[str, Any] | None = None,
    execution_spec: Mapping[str, Any] | None = None,
    reservations: Mapping[str, Any] | None = None,
    representative_reference: Mapping[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate one allocated node before any accelerator/model construction."""

    validate_prediction_anchored_tigris_graph(graph)
    if (
        execution_spec is None
        or reservations is None
        or representative_reference is None
    ):
        if not dry_run:
            raise PermissionError(
                "allocation launch requires execution-spec, reservation, and "
                "representative-reference artifacts"
            )
    else:
        validate_prediction_anchored_execution_spec(execution_spec, verify_file_hashes=True)
        validate_content_hash(
            reservations, expected_contract=PREDICTION_ANCHORED_CAMPAIGN_RESERVATION_CONTRACT
        )
        validate_content_hash(
            representative_reference,
            expected_contract=(
                "prediction_anchored_representative_architecture_resource_reference_v1"
            ),
        )
        expected = {
            "execution_spec_sha256": execution_spec["content_hash"],
            "reservations_sha256": reservations["content_hash"],
            "child_manifest_sha256": execution_spec["child_manifest"]["content_hash"],
            "parent_manifest_file_sha256": execution_spec["parent_manifest"]["sha256"],
            "representative_reference_sha256": representative_reference[
                "content_hash"
            ],
        }
        for field, value in expected.items():
            if graph.get(field) != value:
                raise ValueError(f"allocation {field} disagrees with immutable graph")
        if reservations.get("execution_spec_sha256") != execution_spec["content_hash"]:
            raise ValueError("allocation reservations belong to another execution spec")
        if (
            reservations.get("representative_reference_sha256")
            != representative_reference["content_hash"]
        ):
            raise ValueError(
                "allocation reservations bind another representative reference"
            )
        if (
            representative_reference.get("source_manifest_sha256")
            != execution_spec["parent_manifest"]["sha256"]
        ):
            raise ValueError(
                "representative reference belongs to another execution source"
            )
    by_id = {row["node_id"]: row for row in graph["nodes"]}
    if node_id not in by_id:
        raise KeyError(f"production graph has no node {node_id!r}")
    node = by_id[node_id]
    nnodes = int(environment.get("SLURM_NNODES", 1 if dry_run else 0))
    if nnodes != 1:
        raise PermissionError("prediction-anchored packed jobs require SLURM_NNODES=1")
    process_rank = int(environment.get("SLURM_PROCID", 0))
    if process_rank != 0:
        raise PermissionError("allocation launch manifest may be created only by leader rank 0")
    job_id = str(environment.get("SLURM_JOB_ID", "DRYRUN_LOCAL" if dry_run else ""))
    if not job_id or not (job_id.isdigit() or (dry_run and job_id.startswith("DRYRUN_"))):
        raise ValueError("allocation requires a Slurm job ID")
    memory_mb = int(
        environment.get(
            "SLURM_MEM_PER_NODE",
            int(node["resources"]["host_memory_gib"]) * 1024 if dry_run else 0,
        )
    )
    required_mb = int(node["resources"]["host_memory_gib"]) * 1024
    if memory_mb < required_mb:
        raise PermissionError(
            f"allocation host memory {memory_mb} MiB is below requested {required_mb} MiB"
        )
    if str(environment.get("PYTHONNOUSERSITE", "1" if dry_run else "")) != "1":
        raise PermissionError("allocation must export PYTHONNOUSERSITE=1")
    if node["requires_selected_consumer"]:
        if selected_consumer is None:
            raise PermissionError(
                f"{node_id} requires selected_bridge_consumer.json; guessing is forbidden"
            )
        validate_content_hash(selected_consumer, expected_contract="selected_bridge_consumer_v2")
        if selected_consumer.get("status") != "CONFIRMED_LOCKED":
            raise PermissionError("Stage B5/B6 requires a confirmed locked bridge consumer")
    root = Path(ram_root)
    if not dry_run:
        resolved = root.resolve(strict=False)
        if Path("/dev/shm") not in (resolved, *resolved.parents):
            raise ValueError("allocation RAM root must reside under /dev/shm")
    gpu_workers = int(node["resources"]["gpus_per_node"])
    worker_count = max(gpu_workers, 1)
    commands = {
        "B0": "scripts/run_prediction_anchored_bridge_campaign.py",
        "B1": "scripts/train_prediction_anchored_r0.py",
        "B2": "scripts/prepare_prediction_anchored_bridge_inputs.py",
        "B2_recipe": "scripts/write_prediction_anchored_bridge_recipe.py",
        "B3_consumer": "scripts/execute_prediction_anchored_bridge_consumers.py",
        "B3_reconstructor": "scripts/train_prediction_anchored_bridge_reconstructor.py",
        "B4": "scripts/select_prediction_anchored_bridge_consumer.py",
        "B5_cache": "scripts/cache_prediction_anchored_bridge_logits.py",
        "B5_cache_validate": "scripts/validate_prediction_anchored_teacher_logits.py",
        "B6": "scripts/train_prediction_anchored_bridge_reconstructor.py",
        "REPORT": "scripts/evaluate_prediction_anchored_bridge_campaign.py",
    }
    if node["runner"] == "run_train_prediction_anchored_bridge_consumer.sh":
        surface = commands["B3_consumer"]
    elif node["runner"] == "run_train_prediction_anchored_bridge_reconstructor.sh":
        surface = commands["B3_reconstructor"] if node["stage"] == "B3" else commands["B6"]
    elif node["runner"] == "run_cache_prediction_anchored_bridge_logits.sh":
        surface = commands["B5_cache"]
    elif node["arguments"] and node["arguments"][0] in {
        "B4_SELECT", "B4_CONFIRM", "B4_RUNTIME_RESOURCES", "B5_BIND"
    }:
        surface = commands["B4"]
    elif node["arguments"] and node["arguments"][0] in {"B6_SELECT", "DEPLOY_CONFIRM", "FINAL_TEST"}:
        surface = commands["REPORT"]
    elif node["arguments"] and node["arguments"][0] == "REPORT_EXPORT":
        surface = commands["REPORT"]
    else:
        surface = commands.get(str(node["stage"]), commands["B0"])
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_ALLOCATION_LAUNCH_CONTRACT,
            "graph_sha256": graph["content_hash"],
            "execution_spec_sha256": None if execution_spec is None else execution_spec["content_hash"],
            "reservations_sha256": None if reservations is None else reservations["content_hash"],
            "allocation_contracts_revalidated": (
                execution_spec is not None
                and reservations is not None
                and representative_reference is not None
            ),
            "node_sha256": node["content_hash"],
            "node_id": node_id,
            "stage": node["stage"],
            "slurm_job_id": job_id,
            "slurm_nnodes": nnodes,
            "allocation_leader_rank": process_rank,
            "worker_ranks": list(range(worker_count)),
            "gpu_worker_count": gpu_workers,
            "ram_root": str(root),
            "host_memory_requested_mib": required_mb,
            "host_memory_allocated_mib": memory_mb,
            "one_source_open_by_rank0": True,
            "all_workers_join_same_ram_ledger": True,
            "configuration_run_ids": list(node["configuration_run_ids"]),
            "paired_seed_ids": list(node["paired_seed_ids"]),
            "persistent_reservation_bytes": int(node["persistent_reservation_bytes"]),
            "selected_consumer_sha256": (
                None if selected_consumer is None else selected_consumer["content_hash"]
            ),
            "scientific_command_surface": surface,
            "connected_command_surfaces": sorted(set(commands.values())),
            "cross_allocation_resume": False,
            "preemption_policy": "restart_whole_configuration_pack",
            "persistent_dense_field_output_paths": [],
            "dry_run": bool(dry_run),
            "accelerator_allocation_validated_before_model_construction": True,
        }
    )


def build_prediction_anchored_job_ledger(
    graph: Mapping[str, Any],
    *,
    job_ids: Mapping[str, str | int],
    include_final_test: bool,
    reused_job_node_ids: Sequence[str] = (),
) -> dict[str, Any]:
    validate_prediction_anchored_tigris_graph(graph)
    expected = {
        row["node_id"]
        for row in graph["nodes"]
        if include_final_test or not row["protected_final_test"]
    }
    if set(job_ids) != expected:
        raise ValueError("job ledger IDs do not exactly match the submitted graph nodes")
    reused = {str(value) for value in reused_job_node_ids}
    if not reused.issubset(expected):
        raise ValueError("job ledger reuses an unknown or unsubmitted graph node")
    rows = []
    for node_id in graph["topological_node_ids"]:
        if node_id not in expected:
            continue
        value = str(job_ids[node_id])
        if not (value.isdigit() or value.startswith("DRYRUN_")):
            raise ValueError(f"invalid Slurm job ID for {node_id}: {value}")
        node = next(row for row in graph["nodes"] if row["node_id"] == node_id)
        rows.append(
            {
                "node_id": node_id,
                "job_id": value,
                "dependency_node_ids": list(node["dependencies"]),
                "dependency_job_ids": [str(job_ids[value]) for value in node["dependencies"]],
                "afterany_dependency_node_ids": list(node.get("afterany_dependencies", [])),
                "runner": node["runner"],
                "arguments": list(node["arguments"]),
                "submission_origin": (
                    "existing_slurm_job" if node_id in reused else "submitted_now"
                ),
            }
        )
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_JOB_LEDGER_CONTRACT,
            "graph_sha256": graph["content_hash"],
            "account": TIGRIS_ACCOUNT,
            "partition": TIGRIS_PARTITION,
            "include_final_test": bool(include_final_test),
            "jobs": rows,
            "job_count": len(rows),
            "reused_job_node_ids": sorted(reused),
            "reused_job_count": len(reused),
            "dependencies_recorded": True,
            "immutable_after_submission": True,
        }
    )


def simulate_prediction_anchored_scheduler(
    graph: Mapping[str, Any],
    *,
    requested_outcomes: Mapping[str, str] | None = None,
    include_final_test: bool = False,
) -> dict[str, Any]:
    """CPU-only afterok simulation for success, failure, and preemption."""

    validate_prediction_anchored_tigris_graph(graph)
    requested = {str(key): str(value) for key, value in (requested_outcomes or {}).items()}
    known = set(graph["topological_node_ids"])
    if set(requested) - known:
        raise KeyError("scheduler simulation contains an unknown node")
    allowed = {"COMPLETED", "FAILED", "PREEMPTED"}
    if set(requested.values()) - allowed:
        raise ValueError("scheduler outcomes must be COMPLETED, FAILED, or PREEMPTED")
    statuses = {}
    rows = []
    by_id = {row["node_id"]: row for row in graph["nodes"]}
    for node_id in graph["topological_node_ids"]:
        node = by_id[node_id]
        if node["protected_final_test"] and not include_final_test:
            status = "NOT_SUBMITTED_PROTECTED"
        elif any(
            statuses[parent] != "COMPLETED"
            for parent in node["dependencies"]
            if parent not in set(node.get("afterany_dependencies", []))
        ):
            status = "DEPENDENCY_NEVER_SATISFIED"
        else:
            outcome = requested.get(node_id, "COMPLETED")
            status = (
                "PREEMPTED_RESTART_WHOLE_CONFIGURATION_PACK"
                if outcome == "PREEMPTED"
                else outcome
            )
        statuses[node_id] = status
        rows.append(
            {
                "node_id": node_id,
                "status": status,
                "configuration_run_ids": list(node["configuration_run_ids"]),
                "partial_replica_resume_allowed": False,
                "restart_scope": (
                    "whole_configuration_pack"
                    if status == "PREEMPTED_RESTART_WHOLE_CONFIGURATION_PACK"
                    else None
                ),
            }
        )
    cache_started = any(
        row["node_id"].startswith("b5_cache_") and row["status"] == "COMPLETED"
        for row in rows
    )
    b6_started = any(
        row["node_id"].startswith("b6_") and "pack" in row["node_id"]
        and row["status"] == "COMPLETED"
        for row in rows
    )
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_SCHEDULER_SIMULATION_CONTRACT,
            "graph_sha256": graph["content_hash"],
            "include_final_test": bool(include_final_test),
            "rows": rows,
            "statuses": statuses,
            "consumer_selection_afterok_enforced": True,
            "exploratory_b6_completion_uses_afterany": True,
            "cache_started": cache_started,
            "b6_training_started": b6_started,
            "cross_allocation_resume": False,
            "preempted_pack_requires_whole_restart": any(
                value == "PREEMPTED_RESTART_WHOLE_CONFIGURATION_PACK"
                for value in statuses.values()
            ),
        }
    )


def rehearse_prediction_anchored_campaign_cpu(graph: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_prediction_anchored_tigris_graph(graph)
    commands = render_tigris_sbatch_commands(graph, include_final_test=False)
    success = simulate_prediction_anchored_scheduler(graph)
    failed_selection = simulate_prediction_anchored_scheduler(
        graph, requested_outcomes={"b4_confirm_consumer": "FAILED"}
    )
    first_pack = next(
        row["node_id"]
        for row in graph["nodes"]
        if row["node_id"].startswith("b6_") and "pack" in row["node_id"]
    )
    preempted = simulate_prediction_anchored_scheduler(
        graph, requested_outcomes={first_pack: "PREEMPTED"}
    )
    if failed_selection["cache_started"] or failed_selection["b6_training_started"]:
        raise AssertionError("failed consumer confirmation released Stage B5/B6")
    if not preempted["preempted_pack_requires_whole_restart"]:
        raise AssertionError("preemption rehearsal did not require whole-pack restart")
    if commands["submission_executed"]:
        raise AssertionError("local CPU rehearsal must never submit Slurm jobs")
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_CPU_REHEARSAL_CONTRACT,
            "graph_sha256": graph["content_hash"],
            "graph_validation": validation,
            "rendered_commands_sha256": commands["content_hash"],
            "success_simulation_sha256": success["content_hash"],
            "failed_selection_simulation_sha256": failed_selection["content_hash"],
            "preemption_simulation_sha256": preempted["content_hash"],
            "configuration_inventory": {
                "maximum": int(graph["configuration_count"]),
                "reconstruction_breadth": int(graph["reconstruction_breadth_count"]),
                "post_teacher": int(graph["post_teacher_configuration_count"]),
            },
            "submission_executed": False,
            "explicit_execution_command": commands["explicit_execution_command"],
            "final_test_submitted": False,
            "dense_field_output_paths_present": False,
        }
    )


__all__ = [
    "PREDICTION_ANCHORED_TIGRIS_RESOURCES_CONTRACT",
    "PREDICTION_ANCHORED_PRODUCTION_NODE_CONTRACT",
    "PREDICTION_ANCHORED_PRODUCTION_GRAPH_CONTRACT",
    "PREDICTION_ANCHORED_JOB_LEDGER_CONTRACT",
    "PREDICTION_ANCHORED_SCHEDULER_SIMULATION_CONTRACT",
    "PREDICTION_ANCHORED_CPU_REHEARSAL_CONTRACT",
    "PREDICTION_ANCHORED_ALLOCATION_LAUNCH_CONTRACT",
    "TIGRIS_ACCOUNT",
    "TIGRIS_PARTITION",
    "PAIRED_PROFILE",
    "MAX_CONFIGS_PER_PACK",
    "TigrisResources",
    "build_prediction_anchored_tigris_graph",
    "validate_prediction_anchored_tigris_graph",
    "render_tigris_sbatch_commands",
    "build_allocation_launch_manifest",
    "build_prediction_anchored_job_ledger",
    "simulate_prediction_anchored_scheduler",
    "rehearse_prediction_anchored_campaign_cpu",
]
