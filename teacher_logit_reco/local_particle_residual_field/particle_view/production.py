"""Production graph, recovery, warnings, and rehearsal for Step 10."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence

from .contracts import (
    canonical_sha256,
    canonical_json_bytes,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .registry import (
    PARTICLE_VIEW_QUALITY_POLICY,
    PARTICLE_VIEW_REGISTRY_CONTRACT,
    validate_particle_view_registry,
)


PARTICLE_VIEW_PRODUCTION_GRAPH_CONTRACT = "particle_view_production_graph_v1"
PARTICLE_VIEW_GRAPH_NODE_CONTRACT = "particle_view_graph_node_v1"
PARTICLE_VIEW_GRAPH_RECONCILIATION_CONTRACT = (
    "particle_view_graph_reconciliation_v1"
)
PARTICLE_VIEW_SUBMISSION_LEDGER_CONTRACT = "particle_view_submission_ledger_v1"
PARTICLE_VIEW_NODE_COMPLETION_CONTRACT = "particle_view_node_completion_v1"
PARTICLE_VIEW_QUALITY_WARNING_CONTRACT = "particle_view_quality_warning_v1"
PARTICLE_VIEW_QUALITY_WARNING_SUMMARY_CONTRACT = (
    "particle_view_quality_warning_summary_v1"
)
PARTICLE_VIEW_REHEARSAL_REPORT_CONTRACT = "particle_view_rehearsal_report_v1"

TIGRIS_ACCOUNT = "reu-aisocial"
TIGRIS_CONDA_ENV = "atlas_kd_tigris"
TIGRIS_CONDA_BASE = "/home/ryreu/miniforge3-aarch64"

LOGICAL_NODE_LAYOUT = (
    ("pv00_source", ("source",), ()),
    ("pv01_baselines", ("baseline",), ("pv00_source",)),
    (
        "pv02_oracle_target_screens",
        ("target_screen",),
        ("pv01_baselines",),
    ),
    (
        "pv03_selected_view_publication",
        ("view_publication",),
        ("pv02_oracle_target_screens",),
    ),
    (
        "pv04_representation_robust",
        ("representation",),
        ("pv03_selected_view_publication",),
    ),
    (
        "pv05_predictor_loss_packs",
        ("predictor",),
        ("pv04_representation_robust",),
    ),
    (
        "pv06_confirmation_selection",
        ("confirmation",),
        ("pv05_predictor_loss_packs",),
    ),
    (
        "pv07_selected_path_fairness",
        ("fairness",),
        ("pv06_confirmation_selection",),
    ),
    (
        "pv08_sealed_stack_fusion",
        ("stack",),
        ("pv07_selected_path_fairness",),
    ),
    (
        "pv09_report_export_reload",
        ("report_export",),
        ("pv08_sealed_stack_fusion",),
    ),
    (
        "pv10_hlt_only_final_test",
        ("final_test",),
        ("pv09_report_export_reload",),
    ),
)

NODE_WRAPPERS = {
    "pv00_source": "sbatch/run_prepare_particle_view_campaign.sh",
    "pv01_baselines": "sbatch/run_train_particle_view_baselines.sh",
    "pv02_oracle_target_screens": "sbatch/run_train_particle_view_oracle.sh",
    "pv03_selected_view_publication": "sbatch/run_train_particle_view_oracle.sh",
    "pv04_representation_robust": "sbatch/run_particle_view_predictor_pack.sh",
    "pv05_predictor_loss_packs": "sbatch/run_particle_view_predictor_pack.sh",
    "pv06_confirmation_selection": "sbatch/run_particle_view_confirmation.sh",
    "pv07_selected_path_fairness": "sbatch/run_particle_view_confirmation.sh",
    "pv08_sealed_stack_fusion": "sbatch/run_particle_view_fusion.sh",
    "pv09_report_export_reload": "sbatch/run_particle_view_export_report.sh",
    "pv10_hlt_only_final_test": "sbatch/run_particle_view_export_report.sh",
}

ACTIVE_SLURM_STATES = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
}
TERMINAL_SUCCESS_STATES = {"COMPLETED"}
REQUIRED_QUALITY_WARNING_CODES = (
    "WARN_WEAK_OR_NONPOSITIVE_ORACLE_GAIN",
    "WARN_WEAK_OR_NONPOSITIVE_RECOVERY",
    "WARN_LARGE_TRAIN_VALIDATION_GAP",
    "WARN_ROBUST_TAIL_SATURATION",
    "WARN_NEGATIVE_STACK_GAIN",
    "WARN_FAILED_MATERIAL_EFFECT",
    "WARN_CALIBRATION_DEGRADATION",
    "WARN_TRUST_OR_SLOT_SATURATION",
    "WARN_SUSPICIOUS_SHUFFLED_CONTROL",
    "WARN_SUSPICIOUS_HLT_MEMORY_CONTROL",
    "WARN_CONTROL_MATCH_TOLERANCE",
    "WARN_POSTSELECTION_CONTROL_WINS",
    "WARN_NEGATIVE_FINAL_TEST_GAIN",
)


def _source_commit(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("source_commit must be a 40- or 64-character hex digest")
    return value


def capture_clean_source_checkout(
    project_root: str | Path,
    *,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    """Bind an execution to the clean checked-out commit and Git tree."""

    root = Path(project_root).resolve()

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "git command failed")
        return completed.stdout.strip()

    commit = git("rev-parse", "HEAD").lower()
    _source_commit(commit)
    if expected_commit is not None and commit != _source_commit(expected_commit):
        raise ValueError("requested source commit is not the executed checkout")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValueError(
            "production execution requires a clean checkout; commit or "
            "remove every tracked/untracked source change first"
        )
    tree = git("rev-parse", "HEAD^{tree}").lower()
    _source_commit(tree)
    return {
        "source_commit": commit,
        "source_tree_git_oid": tree,
        "source_status_sha256": canonical_sha256(
            {"git_status_porcelain_v1": status}
        ),
        "source_checkout_clean": True,
    }


def _node_ids() -> tuple[str, ...]:
    return tuple(row[0] for row in LOGICAL_NODE_LAYOUT)


def _seed_expanded_task_waves(
    *,
    registry: Mapping[str, Any],
    run_ids: Sequence[str],
) -> list[list[str]]:
    """Partition one logical node into dependency-safe Slurm-array waves."""

    runs = {row["run_id"]: row for row in registry["runs"]}
    selected = set(run_ids)
    task_ids = {
        (run_id, int(seed)): f"{run_id}__seed_{int(seed)}"
        for run_id in selected
        for seed in runs[run_id]["seed_ids"]
    }
    levels: dict[tuple[str, int], int] = {}
    unresolved = set(task_ids)
    while unresolved:
        progressed = False
        for identity in sorted(unresolved):
            run_id, seed = identity
            same_node_parents: list[tuple[str, int]] = []
            ready = True
            for parent_run_id in runs[run_id]["parent_run_ids"]:
                if parent_run_id not in selected:
                    continue
                parent_seeds = [int(value) for value in runs[parent_run_id]["seed_ids"]]
                parent_seed = seed if seed in parent_seeds else 101
                parent_identity = (parent_run_id, parent_seed)
                if parent_identity not in levels:
                    ready = False
                    break
                same_node_parents.append(parent_identity)
            if not ready:
                continue
            levels[identity] = (
                1 + max(levels[parent] for parent in same_node_parents)
                if same_node_parents
                else 0
            )
            unresolved.remove(identity)
            progressed = True
        if not progressed:
            raise ValueError("cannot construct dependency-safe task waves")
    waves: list[list[str]] = []
    for level in range(max(levels.values(), default=-1) + 1):
        waves.append(
            sorted(
                task_ids[identity]
                for identity, task_level in levels.items()
                if task_level == level
            )
        )
    if any(not wave for wave in waves):
        raise ValueError("task-wave construction emitted an empty wave")
    return waves


def _topological_nodes(nodes: Mapping[str, Mapping[str, Any]]) -> list[str]:
    order: list[str] = []
    state: dict[str, int] = {}

    def visit(node_id: str) -> None:
        marker = state.get(node_id, 0)
        if marker == 1:
            raise ValueError(f"production graph cycle at {node_id}")
        if marker == 2:
            return
        if node_id not in nodes:
            raise ValueError(f"production graph references unknown node {node_id}")
        state[node_id] = 1
        for parent in nodes[node_id]["parent_node_ids"]:
            visit(parent)
        state[node_id] = 2
        order.append(node_id)

    for node_id in sorted(nodes):
        visit(node_id)
    return order


def build_particle_view_production_graph(
    *,
    registry: Mapping[str, Any],
    artifact_root: str,
    source_commit: str,
    source_checkout: Mapping[str, Any] | None = None,
    command_catalog: Mapping[str, Sequence[str]],
    graph_id: str = "particle_view_full_pilot_v1",
) -> dict[str, Any]:
    """Build the complete logical graph and reconcile every registry row."""

    registry_audit = validate_particle_view_registry(registry)
    _source_commit(source_commit)
    if source_checkout is None:
        source_checkout = {
            "source_commit": source_commit,
            "source_tree_git_oid": source_commit,
            "source_status_sha256": canonical_sha256(
                {"source_checkout": "not_execution_verified"}
            ),
            "source_checkout_clean": False,
        }
    if source_checkout.get("source_commit") != source_commit:
        raise ValueError("source checkout/graph commit mismatch")
    _source_commit(str(source_checkout.get("source_tree_git_oid")))
    require_sha256(
        "source_status_sha256", source_checkout.get("source_status_sha256")
    )
    if not isinstance(source_checkout.get("source_checkout_clean"), bool):
        raise ValueError("source checkout clean flag must be boolean")
    if not graph_id or not artifact_root:
        raise ValueError("graph_id and artifact_root must be nonempty")
    if set(command_catalog) != set(_node_ids()):
        missing = set(_node_ids()) - set(command_catalog)
        extra = set(command_catalog) - set(_node_ids())
        raise ValueError(
            f"command catalog/node mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    runs_by_stage: dict[str, list[str]] = {}
    for run in registry["runs"]:
        runs_by_stage.setdefault(str(run["stage"]), []).append(str(run["run_id"]))
    nodes = []
    for node_id, stages, parents in LOGICAL_NODE_LAYOUT:
        command = list(command_catalog[node_id])
        if not command or any(not isinstance(value, str) or not value for value in command):
            raise ValueError(f"node {node_id} has an invalid command")
        run_ids = sorted(
            run_id for stage in stages for run_id in runs_by_stage.get(stage, [])
        )
        task_waves = _seed_expanded_task_waves(
            registry=registry,
            run_ids=run_ids,
        )
        if node_id == "pv10_hlt_only_final_test":
            task_waves = [
                [task_id]
                for wave in task_waves
                for task_id in wave
            ]
        nodes.append(
            {
                "contract": PARTICLE_VIEW_GRAPH_NODE_CONTRACT,
                "node_id": node_id,
                "stages": list(stages),
                "parent_node_ids": list(parents),
                "run_ids": run_ids,
                "task_waves": task_waves,
                "seed_expanded_task_count": sum(map(len, task_waves)),
                "array_max_concurrency": 16,
                "command": command,
                "sbatch_script": NODE_WRAPPERS[node_id],
                "dependency_policy": "afterok_integrity_only",
                "scientific_warning_dependency": False,
                "quality_policy": PARTICLE_VIEW_QUALITY_POLICY,
                "tigris": {
                    "account": TIGRIS_ACCOUNT,
                    "conda_env": TIGRIS_CONDA_ENV,
                    "conda_base": TIGRIS_CONDA_BASE,
                    "python_no_user_site": "1",
                },
            }
        )
    candidate = {
        "contract": PARTICLE_VIEW_PRODUCTION_GRAPH_CONTRACT,
        "graph_id": graph_id,
        "registry_sha256": registry["content_hash"],
        "artifact_root": artifact_root,
        "source_commit": source_commit,
        "source_tree_git_oid": source_checkout["source_tree_git_oid"],
        "source_status_sha256": source_checkout["source_status_sha256"],
        "source_checkout_clean": source_checkout["source_checkout_clean"],
        "nodes": nodes,
        "quality_policy": PARTICLE_VIEW_QUALITY_POLICY,
        "scientific_warnings_are_non_gating": True,
        "performance_gates": False,
        "submission_modes": ["dry_run", "print_only", "execute"],
    }
    graph = with_content_hash(candidate)
    reconciliation = reconcile_particle_view_production_graph(
        graph=graph, registry=registry
    )
    if not reconciliation["reconciled"]:
        raise ValueError("production graph did not reconcile its campaign registry")
    return graph


def validate_particle_view_production_graph(
    graph: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        graph, expected_contract=PARTICLE_VIEW_PRODUCTION_GRAPH_CONTRACT
    )
    expected = {
        "contract",
        "graph_id",
        "registry_sha256",
        "artifact_root",
        "source_commit",
        "source_tree_git_oid",
        "source_status_sha256",
        "source_checkout_clean",
        "nodes",
        "quality_policy",
        "scientific_warnings_are_non_gating",
        "performance_gates",
        "submission_modes",
        "content_hash",
    }
    if set(graph) != expected:
        raise ValueError("production graph field inventory mismatch")
    require_sha256("registry_sha256", graph["registry_sha256"])
    _source_commit(graph["source_commit"])
    _source_commit(graph["source_tree_git_oid"])
    require_sha256("source_status_sha256", graph["source_status_sha256"])
    if not isinstance(graph["source_checkout_clean"], bool):
        raise ValueError("production source checkout flag is invalid")
    if graph["quality_policy"] != PARTICLE_VIEW_QUALITY_POLICY:
        raise ValueError("production graph quality policy mismatch")
    if (
        graph["scientific_warnings_are_non_gating"] is not True
        or graph["performance_gates"] is not False
    ):
        raise ValueError("scientific metrics may not gate production")
    if graph["submission_modes"] != ["dry_run", "print_only", "execute"]:
        raise ValueError("production graph submission modes changed")
    raw_nodes = graph["nodes"]
    if not isinstance(raw_nodes, list) or len(raw_nodes) != len(LOGICAL_NODE_LAYOUT):
        raise ValueError("production graph logical-node count mismatch")
    nodes: dict[str, Mapping[str, Any]] = {}
    for raw in raw_nodes:
        if set(raw) != {
            "contract",
            "node_id",
            "stages",
            "parent_node_ids",
            "run_ids",
            "task_waves",
            "seed_expanded_task_count",
            "array_max_concurrency",
            "command",
            "sbatch_script",
            "dependency_policy",
            "scientific_warning_dependency",
            "quality_policy",
            "tigris",
        }:
            raise ValueError("production graph node field inventory mismatch")
        if raw.get("contract") != PARTICLE_VIEW_GRAPH_NODE_CONTRACT:
            raise ValueError("production graph node contract mismatch")
        node_id = raw.get("node_id")
        if node_id in nodes or node_id not in _node_ids():
            raise ValueError("unknown or duplicate production node")
        if raw.get("sbatch_script") != NODE_WRAPPERS[node_id]:
            raise ValueError("production node uses the wrong Slurm wrapper")
        if raw.get("dependency_policy") != "afterok_integrity_only":
            raise ValueError("production dependency policy mismatch")
        if raw.get("scientific_warning_dependency") is not False:
            raise ValueError("scientific warning became a dependency")
        if raw.get("quality_policy") != PARTICLE_VIEW_QUALITY_POLICY:
            raise ValueError("node quality policy mismatch")
        tigris = raw.get("tigris")
        if tigris != {
            "account": TIGRIS_ACCOUNT,
            "conda_env": TIGRIS_CONDA_ENV,
            "conda_base": TIGRIS_CONDA_BASE,
            "python_no_user_site": "1",
        }:
            raise ValueError("Tigris execution contract mismatch")
        if not isinstance(raw.get("command"), list) or not raw["command"]:
            raise ValueError("production node command is empty")
        if "scripts/run_particle_view_campaign_node.py" in raw["command"]:
            try:
                binding_index = raw["command"].index(
                    "--expected-manifest-sha256"
                )
                require_sha256(
                    "expected runtime manifest sha256",
                    raw["command"][binding_index + 1],
                )
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    "runtime node command is not hash-bound to its manifest"
                ) from exc
        run_ids = raw.get("run_ids")
        if (
            not isinstance(run_ids, list)
            or run_ids != sorted(set(run_ids))
            or any(not isinstance(run_id, str) or not run_id for run_id in run_ids)
        ):
            raise ValueError(
                "production node run_ids must be sorted, unique, nonempty strings"
            )
        waves = raw.get("task_waves")
        if (
            not isinstance(waves, list)
            or any(
                not isinstance(wave, list)
                or not wave
                or wave != sorted(set(wave))
                for wave in waves
            )
        ):
            raise ValueError("production task waves are invalid")
        expanded = [task_id for wave in waves for task_id in wave]
        if (
            len(expanded) != len(set(expanded))
            or int(raw.get("seed_expanded_task_count", -1)) != len(expanded)
            or not 1 <= int(raw.get("array_max_concurrency", 0)) <= 64
        ):
            raise ValueError("production task-wave inventory is inconsistent")
        nodes[node_id] = raw
    expected_layout = {
        node_id: (list(stages), list(parents))
        for node_id, stages, parents in LOGICAL_NODE_LAYOUT
    }
    for node_id, (stages, parents) in expected_layout.items():
        if nodes[node_id]["stages"] != stages or nodes[node_id][
            "parent_node_ids"
        ] != parents:
            raise ValueError("production node layout changed")
    topological = _topological_nodes(nodes)
    return {
        "ok": True,
        "content_hash": graph["content_hash"],
        "nodes": nodes,
        "topological_order": topological,
    }


def reconcile_particle_view_production_graph(
    *, graph: Mapping[str, Any], registry: Mapping[str, Any]
) -> dict[str, Any]:
    graph_audit = validate_particle_view_production_graph(graph)
    registry_audit = validate_particle_view_registry(registry)
    if graph["registry_sha256"] != registry["content_hash"]:
        raise ValueError("production graph is bound to a different registry")
    registered = {row["run_id"]: row for row in registry["runs"]}
    run_to_node: dict[str, str] = {}
    assigned: list[str] = []
    for node in graph["nodes"]:
        allowed_stages = set(node["stages"])
        for run_id in node["run_ids"]:
            if run_id not in registered:
                raise ValueError(f"graph assigns unknown registry run {run_id}")
            if registered[run_id]["stage"] not in allowed_stages:
                raise ValueError(f"run {run_id} is assigned to the wrong node")
            assigned.append(run_id)
            run_to_node[run_id] = node["node_id"]
    duplicates = sorted(
        run_id for run_id in set(assigned) if assigned.count(run_id) > 1
    )
    missing = sorted(set(registered) - set(assigned))
    extra = sorted(set(assigned) - set(registered))
    ancestors: dict[str, set[str]] = {}
    for node_id in graph_audit["topological_order"]:
        direct = set(graph_audit["nodes"][node_id]["parent_node_ids"])
        ancestors[node_id] = direct | {
            ancestor
            for parent in direct
            for ancestor in ancestors[parent]
        }
    invalid_parent_edges = []
    if not missing and not extra and not duplicates:
        for run_id, run in registered.items():
            child_node = run_to_node[run_id]
            allowed_parent_nodes = ancestors[child_node] | {child_node}
            for parent_run_id in run["parent_run_ids"]:
                parent_node = run_to_node[parent_run_id]
                if parent_node not in allowed_parent_nodes:
                    invalid_parent_edges.append(
                        {
                            "run_id": run_id,
                            "run_node_id": child_node,
                            "parent_run_id": parent_run_id,
                            "parent_node_id": parent_node,
                        }
                    )
    declared_seed_replicas = sum(
        len(row["seed_ids"]) for row in registered.values()
    )
    generated_seed_replicas = sum(
        len(registered[run_id]["seed_ids"]) for run_id in assigned
    )
    waved_task_ids = [
        task_id
        for node in graph["nodes"]
        for wave in node["task_waves"]
        for task_id in wave
    ]
    expected_task_ids = {
        f"{run_id}__seed_{int(seed)}"
        for run_id, run in registered.items()
        for seed in run["seed_ids"]
    }
    task_wave_inventory_valid = (
        len(waved_task_ids) == len(set(waved_task_ids))
        and set(waved_task_ids) == expected_task_ids
    )
    counts = {
        "declared_runs": len(registered),
        "generated_run_assignments": len(assigned),
        "selectable_runs": sum(bool(row["selectable"]) for row in registered.values()),
        "diagnostic_runs": sum(bool(row["diagnostic"]) for row in registered.values()),
        "single_seed_screen_runs": sum(
            bool(row["single_seed_screen"]) for row in registered.values()
        ),
        "three_seed_confirmation_runs": sum(
            bool(row["three_seed_confirmation"]) for row in registered.values()
        ),
        "declared_seed_replicas": declared_seed_replicas,
        "generated_seed_replicas": generated_seed_replicas,
        "seed_expanded_replicas": generated_seed_replicas,
        "logical_nodes": len(graph_audit["nodes"]),
    }
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_GRAPH_RECONCILIATION_CONTRACT,
            "graph_sha256": graph["content_hash"],
            "source_commit": graph["source_commit"],
            "source_tree_git_oid": graph["source_tree_git_oid"],
            "source_status_sha256": graph["source_status_sha256"],
            "registry_sha256": registry["content_hash"],
            "counts": counts,
            "missing_run_ids": missing,
            "extra_run_ids": extra,
            "duplicate_run_ids": duplicates,
            "invalid_parent_edges": invalid_parent_edges,
            "reconciled": (
                not missing
                and not extra
                and not duplicates
                and not invalid_parent_edges
                and declared_seed_replicas == generated_seed_replicas
                and task_wave_inventory_valid
            ),
            "single_training_pool": registry_audit["single_training_pool"],
        }
    )


def build_quality_warning(
    *,
    warning_code: str,
    severity: str,
    graph_node: str,
    configuration_id: str,
    seed: int,
    split: str,
    observed_value: Any,
    reference_value: Any,
    warning_threshold: Any,
    interpretation: str,
    suggested_diagnostic: str,
    supporting_artifacts: Sequence[Mapping[str, str]],
    source_commit: str,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"WARN_[A-Z0-9_]+", warning_code):
        raise ValueError("warning_code must use WARN_* uppercase syntax")
    if severity not in {"info", "warning", "high"}:
        raise ValueError("quality-warning severity is invalid")
    if seed not in {101, 202, 303}:
        raise ValueError("quality-warning seed is invalid")
    _source_commit(source_commit)
    support = []
    for row in supporting_artifacts:
        support.append(
            {
                "path": str(row["path"]),
                "sha256": require_sha256("supporting sha256", row["sha256"]),
            }
        )
    if not support:
        raise ValueError("quality warning requires a supporting artifact")
    if timestamp_utc is None:
        timestamp_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        parsed = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("quality-warning timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("quality-warning timestamp must include UTC")
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("quality-warning timestamp must be UTC")
    for name, value in (
        ("graph_node", graph_node),
        ("configuration_id", configuration_id),
        ("split", split),
        ("interpretation", interpretation),
        ("suggested_diagnostic", suggested_diagnostic),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be nonempty")
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_QUALITY_WARNING_CONTRACT,
            "warning_code": warning_code,
            "severity": severity,
            "graph_node": graph_node,
            "configuration_id": configuration_id,
            "seed": seed,
            "split": split,
            "observed_value": observed_value,
            "reference_value": reference_value,
            "warning_threshold": warning_threshold,
            "interpretation": interpretation,
            "suggested_diagnostic": suggested_diagnostic,
            "supporting_artifacts": support,
            "timestamp_utc": timestamp_utc,
            "source_commit": source_commit,
            "non_gating": True,
            "exit_code": 0,
        }
    )


def aggregate_quality_warnings(
    warnings: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    normalized = []
    for warning in warnings:
        validate_content_hash(
            warning, expected_contract=PARTICLE_VIEW_QUALITY_WARNING_CONTRACT
        )
        if warning.get("non_gating") is not True or warning.get("exit_code") != 0:
            raise ValueError("scientific warning is not non-gating")
        normalized.append(dict(warning))
    normalized.sort(
        key=lambda row: (
            row["warning_code"],
            row["configuration_id"],
            row["seed"],
            row["content_hash"],
        )
    )
    counts: dict[str, int] = {}
    for warning in normalized:
        counts[warning["warning_code"]] = counts.get(warning["warning_code"], 0) + 1
    summary = with_content_hash(
        {
            "contract": PARTICLE_VIEW_QUALITY_WARNING_SUMMARY_CONTRACT,
            "warning_count": len(normalized),
            "counts_by_code": {key: counts[key] for key in sorted(counts)},
            "warnings": normalized,
            "warnings_are_non_gating": True,
            "aggregate_exit_code": 0,
        }
    )
    lines = [
        "# Particle-view quality warning summary",
        "",
        f"Total warnings: **{len(normalized)}**",
        "",
        "Scientific warnings never gate submission or change job exit codes.",
        "",
        "## Counts",
        "",
    ]
    if counts:
        lines.extend(
            f"- `{code}`: {counts[code]}" for code in sorted(counts)
        )
    else:
        lines.append("- No warnings.")
    lines.extend(["", "## Details", ""])
    for warning in normalized:
        support = warning["supporting_artifacts"][0]
        lines.extend(
            [
                f"- **{warning['warning_code']}** — "
                f"`{warning['configuration_id']}` seed {warning['seed']} "
                f"on `{warning['split']}`: {warning['interpretation']} "
                f"([artifact]({support['path']}))",
            ]
        )
    return summary, "\n".join(lines) + "\n"


def write_quality_warning_jsonl(
    path: str | Path, warnings: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Publish one node's immutable warning stream, including an empty stream."""

    normalized = []
    for warning in warnings:
        validate_content_hash(
            warning, expected_contract=PARTICLE_VIEW_QUALITY_WARNING_CONTRACT
        )
        if warning.get("non_gating") is not True or warning.get("exit_code") != 0:
            raise ValueError("quality warning JSONL contains a gating warning")
        normalized.append(dict(warning))
    encoded = b"".join(canonical_json_bytes(row) + b"\n" for row in normalized)
    destination = Path(path)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise FileExistsError("quality warning JSONL destination is unsafe")
        if destination.read_bytes() != encoded:
            raise FileExistsError("refusing to overwrite quality warning JSONL")
        status = "already_present"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
        status = "published"
    return {
        "path": str(destination.resolve()),
        "warning_count": len(normalized),
        "status": status,
    }


def load_quality_warning_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"quality warning JSONL is absent or unsafe: {source}")
    warnings = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            warning = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid quality warning JSONL line {line_number}"
            ) from exc
        if not isinstance(warning, dict):
            raise ValueError("quality warning JSONL rows must be objects")
        validate_content_hash(
            warning, expected_contract=PARTICLE_VIEW_QUALITY_WARNING_CONTRACT
        )
        if warning.get("non_gating") is not True or warning.get("exit_code") != 0:
            raise ValueError("quality warning JSONL contains a gating warning")
        warnings.append(warning)
    return warnings


def write_quality_warning_summary(
    *,
    output_dir: str | Path,
    warnings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary, markdown = aggregate_quality_warnings(warnings)
    json_result = write_immutable_json(
        destination / "quality_warning_summary.json", summary
    )
    markdown_path = destination / "quality_warning_summary.md"
    if markdown_path.exists():
        if markdown_path.read_text(encoding="utf-8") != markdown:
            raise FileExistsError("refusing to overwrite quality warning Markdown")
    else:
        markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
    return {
        "summary_sha256": summary["content_hash"],
        "json": json_result,
        "markdown_path": str(markdown_path.resolve()),
        "warning_count": summary["warning_count"],
    }


def _normalize_existing_jobs(
    existing_jobs: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for node_id, raw in (existing_jobs or {}).items():
        if node_id not in _node_ids():
            raise ValueError(f"existing job has unknown node {node_id}")
        job_id = str(raw.get("job_id", ""))
        state = str(raw.get("state", "")).upper().split("+", 1)[0]
        if not job_id.isdigit() or int(job_id) <= 0 or not state:
            raise ValueError("existing-job records require positive job_id/state")
        result[node_id] = {"job_id": job_id, "state": state}
    return result


def _validate_reusable_node_completion(
    *, graph: Mapping[str, Any], node_id: str
) -> Mapping[str, Any]:
    path = (
        Path(graph["artifact_root"])
        / "node_completions"
        / f"{node_id}.json"
    )
    if not path.is_file() or path.is_symlink():
        raise ValueError(
            f"completed Slurm job for {node_id} has no authenticated "
            "node completion"
        )
    completion = json.loads(path.read_text(encoding="utf-8"))
    validate_content_hash(
        completion,
        expected_contract=PARTICLE_VIEW_NODE_COMPLETION_CONTRACT,
    )
    expected = {
        "graph_sha256": graph["content_hash"],
        "source_commit": graph["source_commit"],
        "source_tree_git_oid": graph["source_tree_git_oid"],
        "source_status_sha256": graph["source_status_sha256"],
        "node_id": node_id,
        "run_ids": next(
            row["run_ids"] for row in graph["nodes"] if row["node_id"] == node_id
        ),
        "integrity_status": "complete",
        "exit_code": 0,
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            raise ValueError(f"stale node completion for {node_id}: {key}")
    if not completion.get("output_artifacts"):
        raise ValueError(
            f"node completion for {node_id} authenticates no outputs"
        )
    node = next(
        row for row in graph["nodes"] if row["node_id"] == node_id
    )
    if "scripts/run_particle_view_campaign_node.py" in node["command"]:
        expected_artifact_count = (
            2 * int(node["seed_expanded_task_count"]) + 1
        )
        if len(completion["output_artifacts"]) != expected_artifact_count:
            raise ValueError(
                f"node completion for {node_id} omits runtime task outputs"
            )
    for artifact in completion["output_artifacts"]:
        artifact_path = Path(str(artifact["path"]))
        if (
            artifact_path.is_symlink()
            or not artifact_path.is_file()
            or sha256_file(artifact_path) != artifact["sha256"]
        ):
            raise ValueError(
                f"node completion output is absent or stale: {artifact_path}"
            )
    return completion


def plan_particle_view_submissions(
    *,
    graph: Mapping[str, Any],
    graph_path: str,
    existing_jobs: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a recovery-safe Slurm plan without submitting anything."""

    audit = validate_particle_view_production_graph(graph)
    existing = _normalize_existing_jobs(existing_jobs)
    terminal_records: dict[str, dict[str, Any]] = {}
    output = []
    for node_id in audit["topological_order"]:
        node = audit["nodes"][node_id]
        prior = existing.get(node_id)
        if prior and prior["state"] in TERMINAL_SUCCESS_STATES:
            completion = _validate_reusable_node_completion(
                graph=graph, node_id=node_id
            )
            record = {
                "submission_id": node_id,
                "node_id": node_id,
                "submission_kind": "logical_node_barrier",
                "action": "reuse_completed",
                "job_id": prior["job_id"],
                "dependency_job_ids": [],
                "command": None,
                "node_completion_sha256": completion["content_hash"],
            }
            terminal_records[node_id] = record
            output.append(record)
            continue
        if prior and prior["state"] in ACTIVE_SLURM_STATES:
            raise ValueError(
                "active Slurm jobs cannot be reused without an authenticated "
                f"campaign progress ledger: {node_id}={prior['job_id']}. "
                "Wait for completion or cancel the active job before recovery."
            )
        parent_dependencies = []
        for parent in node["parent_node_ids"]:
            parent_record = terminal_records[parent]
            if parent_record["action"] != "reuse_completed":
                parent_dependencies.append(parent_record["job_id"])
        dependencies = parent_dependencies
        last_record = None
        for wave_index, wave in enumerate(node["task_waves"]):
            submission_id = f"{node_id}__wave_{wave_index:02d}"
            export = (
                "ALL,"
                f"PARTICLE_VIEW_GRAPH={graph_path},"
                f"PARTICLE_VIEW_NODE_ID={node_id},"
                f"PARTICLE_VIEW_TASK_WAVE_INDEX={wave_index}"
            )
            command = ["sbatch", "--parsable"]
            if dependencies:
                command.append(
                    f"--dependency=afterok:{':'.join(dependencies)}"
                )
            command.extend(
                [
                    (
                        f"--array=0-{len(wave) - 1}%"
                        f"{node['array_max_concurrency']}"
                    ),
                    f"--export={export}",
                    str(node["sbatch_script"]),
                    node_id,
                ]
            )
            record = {
                "submission_id": submission_id,
                "node_id": node_id,
                "submission_kind": "task_wave_array",
                "task_wave_index": wave_index,
                "array_task_count": len(wave),
                "action": "submit",
                "job_id": f"<{submission_id}:job_id>",
                "dependency_job_ids": list(dependencies),
                "command": command,
                "node_completion_sha256": None,
            }
            output.append(record)
            last_record = record
            dependencies = [record["job_id"]]
        if last_record is None:
            raise ValueError(f"logical node {node_id} has no task waves")
        submission_id = f"{node_id}__barrier"
        export = (
            "ALL,"
            f"PARTICLE_VIEW_GRAPH={graph_path},"
            f"PARTICLE_VIEW_NODE_ID={node_id},"
            "PARTICLE_VIEW_NODE_BARRIER=1"
        )
        command = [
            "sbatch",
            "--parsable",
            f"--dependency=afterok:{last_record['job_id']}",
            f"--export={export}",
            str(node["sbatch_script"]),
            node_id,
        ]
        record = {
            "submission_id": submission_id,
            "node_id": node_id,
            "submission_kind": "logical_node_barrier",
            "action": "submit",
            "job_id": f"<{submission_id}:job_id>",
            "dependency_job_ids": [last_record["job_id"]],
            "command": command,
            "node_completion_sha256": None,
        }
        terminal_records[node_id] = record
        output.append(record)
    return output


def submit_particle_view_graph(
    *,
    graph: Mapping[str, Any],
    graph_path: str,
    existing_jobs: Mapping[str, Mapping[str, Any]] | None = None,
    mode: str = "execute",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    progress_callback: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    if mode not in {"execute", "dry_run", "print_only"}:
        raise ValueError("submission mode must be execute/dry_run/print_only")
    plan = plan_particle_view_submissions(
        graph=graph, graph_path=graph_path, existing_jobs=existing_jobs
    )
    resolved_job_ids: dict[str, str] = {}
    records = []
    for planned in plan:
        record = dict(planned)
        dependencies = []
        for value in record["dependency_job_ids"]:
            if value.startswith("<"):
                submission_id = value[1:].split(":", 1)[0]
                dependencies.append(resolved_job_ids[submission_id])
            else:
                dependencies.append(value)
        if record["action"].startswith("reuse_"):
            resolved_job_ids[record["submission_id"]] = record["job_id"]
        elif mode == "execute":
            command = list(record["command"])
            for index, value in enumerate(command):
                if value.startswith("--dependency=afterok:"):
                    command[index] = (
                        f"--dependency=afterok:{':'.join(dependencies)}"
                    )
            completed = runner(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Slurm submission failed for {record['node_id']}: "
                    f"{completed.stderr.strip()}"
                )
            job_id = completed.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError("sbatch did not return a numeric job ID")
            record["command"] = command
            record["dependency_job_ids"] = dependencies
            record["job_id"] = job_id
            resolved_job_ids[record["submission_id"]] = job_id
        else:
            record["dependency_job_ids"] = dependencies
            resolved_job_ids[record["submission_id"]] = record["job_id"]
        records.append(record)
        if progress_callback is not None:
            progress_callback(tuple(dict(row) for row in records))
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_SUBMISSION_LEDGER_CONTRACT,
            "graph_sha256": graph["content_hash"],
            "graph_path": graph_path,
            "mode": mode,
            "records": records,
            "submitted_count": sum(
                row["action"] == "submit" for row in records
            )
            if mode == "execute"
            else 0,
            "planned_submit_count": sum(
                row["action"] == "submit" for row in records
            ),
            "scientific_warning_dependency_count": 0,
        }
    )


def build_node_completion(
    *,
    graph: Mapping[str, Any],
    node_id: str,
    output_artifacts: Sequence[Mapping[str, str]],
    warning_sha256: Sequence[str] = (),
    rehearsal: bool = False,
) -> dict[str, Any]:
    audit = validate_particle_view_production_graph(graph)
    if node_id not in audit["nodes"]:
        raise ValueError("unknown completion node")
    artifacts = []
    for row in output_artifacts:
        artifacts.append(
            {
                "path": str(row["path"]),
                "sha256": require_sha256("output artifact sha256", row["sha256"]),
            }
        )
    warnings = [
        require_sha256("warning_sha256", value) for value in warning_sha256
    ]
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_NODE_COMPLETION_CONTRACT,
            "graph_sha256": graph["content_hash"],
            "source_commit": graph["source_commit"],
            "source_tree_git_oid": graph["source_tree_git_oid"],
            "source_status_sha256": graph["source_status_sha256"],
            "node_id": node_id,
            "run_ids": list(audit["nodes"][node_id]["run_ids"]),
            "output_artifacts": artifacts,
            "quality_warning_sha256": warnings,
            "quality_warning_count": len(warnings),
            "scientific_warnings_non_gating": True,
            "integrity_status": "complete",
            "exit_code": 0,
            "rehearsal": bool(rehearsal),
        }
    )


__all__ = [
    "ACTIVE_SLURM_STATES",
    "LOGICAL_NODE_LAYOUT",
    "NODE_WRAPPERS",
    "PARTICLE_VIEW_GRAPH_NODE_CONTRACT",
    "PARTICLE_VIEW_GRAPH_RECONCILIATION_CONTRACT",
    "PARTICLE_VIEW_NODE_COMPLETION_CONTRACT",
    "PARTICLE_VIEW_PRODUCTION_GRAPH_CONTRACT",
    "PARTICLE_VIEW_QUALITY_WARNING_CONTRACT",
    "PARTICLE_VIEW_QUALITY_WARNING_SUMMARY_CONTRACT",
    "PARTICLE_VIEW_REHEARSAL_REPORT_CONTRACT",
    "PARTICLE_VIEW_SUBMISSION_LEDGER_CONTRACT",
    "REQUIRED_QUALITY_WARNING_CODES",
    "TIGRIS_ACCOUNT",
    "TIGRIS_CONDA_BASE",
    "TIGRIS_CONDA_ENV",
    "aggregate_quality_warnings",
    "build_node_completion",
    "build_particle_view_production_graph",
    "build_quality_warning",
    "capture_clean_source_checkout",
    "plan_particle_view_submissions",
    "load_quality_warning_jsonl",
    "reconcile_particle_view_production_graph",
    "submit_particle_view_graph",
    "validate_particle_view_production_graph",
    "write_quality_warning_summary",
    "write_quality_warning_jsonl",
]
