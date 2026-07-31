"""End-to-end control-plane validation and production submission gates."""

from __future__ import annotations

import ast
import contextlib
import io
import json
from pathlib import Path
import re
import runpy
import sys
from typing import Any, Mapping

from .contracts import (
    bind_source,
    load_hashed_json,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .manifest_orchestration import (
    MANIFEST_MATERIALIZATION_PLAN_CONTRACT,
    manifest_plan_path,
    validate_manifest_materialization_plan,
)
from .plan_factory_registry import (
    MANIFEST_PLAN_PRODUCER_AUDIT_CONTRACT,
)
from .dynamic_continuation import (
    build_dynamic_continuation,
    publish_dynamic_continuation,
    validate_published_dynamic_continuation,
)
from .final_continuation import (
    build_final_continuation,
    publish_final_continuation,
)
from .late_continuation import (
    build_late_continuation,
    publish_late_continuation,
)
from .middle_continuation import (
    build_middle_continuation,
    publish_middle_continuation,
)
from .production import (
    BOOTSTRAP_INPUT_MANIFEST_NODES,
    DIRECT_WORKER_NODES,
    FINAL_CONTINUATION_MANIFEST_NODES,
    FINAL_NODE_ENTRYPOINTS,
    JOB_LEDGER_CONTRACT,
    LATE_CONTINUATION_MANIFEST_NODES,
    LATE_NODE_ENTRYPOINTS,
    MIDDLE_CONTINUATION_MANIFEST_NODES,
    MIDDLE_NODE_ENTRYPOINTS,
    PRODUCTION_GRAPH_CONTRACT,
    STATIC_EXPERIMENT_MANIFEST_NODES,
    TASK_MANIFEST_PRODUCER_NODES,
    build_job_ledger,
    build_production_graph,
    build_resume_plan,
    build_task_manifest,
    task_manifest_path_for_graph,
    validate_job_ledger,
    validate_production_campaign_binding,
    validate_production_graph,
    validate_task_manifest_for_graph,
)
from .provenance import source_snapshot
from .storage import (
    STORAGE_MEASUREMENTS_CONTRACT,
    validate_storage_measurements,
)
from .task_completion import (
    publish_task_manifest_completion,
    publish_task_row_completion,
    reusable_task_row_completion,
    task_manifest_completion_path,
    validate_task_manifest_completion,
)
from .contracts import build_campaign_spec
from .workflow import validate_campaign_source


LOCAL_OPERATIONAL_REPORT_CONTRACT = "retb_local_operational_report_v3"
TIGRIS_SMOKE_EVIDENCE_CONTRACT = "retb_tigris_smoke_evidence_v1"
PRODUCTION_DRY_RUN_EVIDENCE_CONTRACT = (
    "retb_production_dry_run_evidence_v1"
)
FULL_SUBMISSION_AUTHORIZATION_CONTRACT = (
    "retb_full_submission_authorization_v2"
)

_PARENT_NAMES = (
    "artifact_layout",
    "final_select_label_manifest",
    "global_determinism",
    "hlt_replica_manifest",
    "raw_input_schema",
    "scale_train_manifest",
    "split_audit",
    "split_manifest",
    "storage_measurements",
    "validation_partition_manifest",
)
_PYTHON_REFERENCE = re.compile(r"(?:python|python3)\s+(scripts/[A-Za-z0-9_./-]+\.py)")


def _source_record(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "commit": snapshot["source_commit"],
        "status_sha256": snapshot["source_status_sha256"],
        "dirty": bool(snapshot["source_dirty"]),
        "status_hash_policy": (
            "git_diff_binary_HEAD_plus_sorted_untracked_file_bytes_v2"
        ),
    }


def _campaign_and_graph(
    root: Path, *, snapshot: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign = build_campaign_spec(
        campaign_id=root.name,
        campaign_profile="miniature_test",
        source_snapshot=snapshot,
        parent_artifact_hashes={
            name: f"{index + 1:064x}"
            for index, name in enumerate(_PARENT_NAMES)
        },
        run_registry_hashes={"operational-validation": "f" * 64},
    )
    graph = build_production_graph(
        campaign_root=root,
        campaign_id=root.name,
        source_commit=str(snapshot["source_commit"]),
        source_status_sha256=str(snapshot["source_status_sha256"]),
        storage_measurements_sha256=campaign[
            "parent_artifact_hashes"
        ]["storage_measurements"],
        miniature=True,
    )
    return campaign, graph


def audit_worker_interfaces(
    *,
    production_graph: Mapping[str, Any],
    repo_root: str | Path,
    probe_python_clis: bool = False,
) -> dict[str, Any]:
    """Parse every graph wrapper and referenced Python worker."""

    validate_production_graph(production_graph)
    root = Path(repo_root).resolve()
    wrappers: dict[str, str] = {}
    python_paths: set[str] = {
        "scripts/write_retb_synthetic_output.py",
        *MIDDLE_NODE_ENTRYPOINTS.values(),
        *(
            path
            for paths in LATE_NODE_ENTRYPOINTS.values()
            for path in paths
        ),
        *(
            path
            for paths in FINAL_NODE_ENTRYPOINTS.values()
            for path in paths
        ),
    }
    for node in production_graph["nodes"]:
        wrapper = root / "sbatch" / str(node["worker"])
        if not wrapper.is_file():
            raise FileNotFoundError(
                f"production worker wrapper is absent: {wrapper}"
            )
        source = wrapper.read_text(encoding="utf-8")
        if "set -euo pipefail" not in source:
            raise ValueError(f"{node['node_id']} wrapper is not fail-closed")
        wrappers[str(node["node_id"])] = str(wrapper)
        python_paths.update(_PYTHON_REFERENCE.findall(source.replace("\\", "/")))
    for relative in sorted(python_paths):
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("worker entry point escapes repository") from error
        if not path.is_file():
            raise FileNotFoundError(f"Python worker is absent: {path}")
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cli_probe_count = 0
    if probe_python_clis:
        original_argv = list(sys.argv)
        original_bytecode = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            for relative in sorted(python_paths):
                sys.argv = [relative, "--help"]
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        with contextlib.redirect_stderr(io.StringIO()):
                            runpy.run_path(
                                str((root / relative).resolve()),
                                run_name="__main__",
                            )
                except SystemExit as error:
                    if error.code not in {None, 0}:
                        raise RuntimeError(
                            f"worker CLI help failed: {relative}"
                        ) from error
                cli_probe_count += 1
        finally:
            sys.argv = original_argv
            sys.dont_write_bytecode = original_bytecode
    return {
        "wrapper_count": len(wrappers),
        "python_entrypoint_count": len(python_paths),
        "all_graph_wrappers_present": True,
        "all_wrappers_fail_closed": True,
        "all_referenced_python_entrypoints_parse": True,
        "python_cli_help_probed": bool(probe_python_clis),
        "python_cli_help_probe_count": cli_probe_count,
        "stages_covered": list("ABCDEFGHIJKLMN"),
    }


def audit_manifest_producer_invocations(
    *,
    production_graph: Mapping[str, Any],
    repo_root: str | Path,
    campaign_root: str | Path | None = None,
    plan_factory_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit source wiring and observed execution for every manifest target."""

    validate_production_graph(production_graph)
    root = Path(repo_root).resolve()
    nodes = {
        str(node["node_id"]): node for node in production_graph["nodes"]
    }
    execution_registry = {
        str(entry["node_id"]): entry
        for entry in production_graph["node_execution_registry"]["entries"]
    }
    bootstrap_wrapper = (
        root / "sbatch" / str(nodes["campaign_bootstrap"]["worker"])
    ).read_text(encoding="utf-8").replace("\\", "/")
    bootstrap_source = (
        root / "scripts" / "bootstrap_retb_input_tasks.py"
    ).read_text(encoding="utf-8")
    bootstrap_is_wired = (
        "scripts/bootstrap_retb_input_tasks.py" in bootstrap_wrapper
        and "publish_stage_a_contract_bundle" in bootstrap_source
        and "publish_static_experiment_bundle" in bootstrap_source
    )
    scalar_launcher = (
        root / "sbatch" / "run_retb_production_task.sh"
    ).read_text(encoding="utf-8").replace("\\", "/")
    array_launcher = (
        root / "sbatch" / "run_retb_array_launcher.sh"
    ).read_text(encoding="utf-8").replace("\\", "/")
    common = (root / "sbatch" / "retb_common.sh").read_text(
        encoding="utf-8"
    ).replace("\\", "/")
    materializer = (
        root / "scripts" / "materialize_retb_downstream_manifests.py"
    ).read_text(encoding="utf-8")
    shared_hook_is_complete = (
        "retb_materialize_downstream" in scalar_launcher
        and "retb_materialize_downstream" in array_launcher
        and "materialize_retb_downstream_manifests.py" in common
        and "materialize_downstream_manifests" in materializer
    )
    bootstrap_targets = set(BOOTSTRAP_INPUT_MANIFEST_NODES) | set(
        STATIC_EXPERIMENT_MANIFEST_NODES
    )
    declared_registry = (
        production_graph["manifest_plan_factory_registry"]
        if plan_factory_registry is None
        else plan_factory_registry
    )
    declared_entries = list(declared_registry.get("entries", ()))
    by_factory_target = {
        str(entry.get("target_node_id", "")): entry
        for entry in declared_entries
    }
    expected = sorted(TASK_MANIFEST_PRODUCER_NODES)
    downstream_targets = sorted(set(expected) - bootstrap_targets)
    missing_registrations = sorted(
        set(downstream_targets) - set(by_factory_target)
    )
    unexpected_registrations = sorted(
        set(by_factory_target) - set(downstream_targets)
    )

    observed_root = (
        None if campaign_root is None else Path(campaign_root).resolve()
    )
    campaign = None
    if observed_root is not None and (
        observed_root / "campaign_spec.json"
    ).is_file():
        campaign = load_hashed_json(observed_root / "campaign_spec.json")

    entries: list[dict[str, Any]] = []
    implementation_covered: list[str] = []
    execution_complete: list[str] = []
    for target in expected:
        producer_node = TASK_MANIFEST_PRODUCER_NODES[target]
        is_bootstrap = target in bootstrap_targets
        factory = by_factory_target.get(target)
        factory_path = (
            None
            if factory is None
            else root / str(factory["plan_factory_entrypoint"])
        )
        factory_source = (
            ""
            if factory_path is None or not factory_path.is_file()
            else factory_path.read_text(encoding="utf-8")
        )
        factory_implemented = bool(
            is_bootstrap
            and bootstrap_is_wired
            or factory is not None
            and factory_path is not None
            and factory_path.is_file()
            and str(factory["plan_factory_symbol"]) in factory_source
            and "publish_manifest_materialization_plan" in factory_source
            and target in factory_source
        )
        if factory_implemented:
            implementation_covered.append(target)

        plan_valid = is_bootstrap
        manifest_valid = False
        completion_valid = False
        outputs_genuine = False
        observation_error = None
        if observed_root is not None and campaign is not None:
            try:
                if not is_bootstrap:
                    plan_path = manifest_plan_path(
                        observed_root, target_node_id=target
                    )
                    if plan_path.is_file():
                        plan = load_hashed_json(
                            plan_path,
                            expected_contract=(
                                MANIFEST_MATERIALIZATION_PLAN_CONTRACT
                            ),
                        )
                        validate_manifest_materialization_plan(
                            plan,
                            campaign=campaign,
                            production_graph=production_graph,
                        )
                        plan_valid = (
                            plan["producer_node_id"] == producer_node
                            and plan["target_node_id"] == target
                            and plan["genuine_scientific_worker_rows"] is True
                            and plan["synthetic_control_plane_rows"] is False
                        )
                manifest_path = task_manifest_path_for_graph(
                    production_graph,
                    node_id=target,
                    campaign_root=observed_root,
                )
                if manifest_path.is_file():
                    manifest = load_hashed_json(manifest_path)
                    validate_task_manifest_for_graph(
                        manifest,
                        production_graph=production_graph,
                        campaign_root=observed_root,
                        repo_root=root,
                    )
                    manifest_valid = True
                    completion_path = task_manifest_completion_path(
                        observed_root, node_id=target
                    )
                    if completion_path.is_file():
                        completion = load_hashed_json(completion_path)
                        validate_task_manifest_completion(
                            completion,
                            campaign_root=observed_root,
                            campaign=campaign,
                            task_manifest=manifest,
                        )
                        completion_valid = True
                        manifest_rows_genuine = all(
                            "scripts/write_retb_synthetic_output.py"
                            not in {
                                str(value).replace("\\", "/")
                                for value in row["argv"]
                            }
                            and str(
                                row["environment"].get(
                                    "RETB_SYNTHETIC_CONTROL_PLANE", ""
                                )
                            )
                            != "1"
                            for row in manifest["rows"]
                        )
                        output_contracts = [
                            output["json_contract"]
                            for row_index in range(
                                int(manifest["task_count"])
                            )
                            for output in load_hashed_json(
                                observed_root
                                / "job_ledgers"
                                / "completions"
                                / target
                                / f"row_{row_index:06d}.json"
                            )["outputs"]
                        ]
                        outputs_genuine = (
                            manifest_rows_genuine
                            and bool(output_contracts)
                            and all(
                                contract
                                != "retb_synthetic_node_output_v1"
                                for contract in output_contracts
                            )
                        )
            except (
                FileNotFoundError,
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                observation_error = str(error)
        complete = bool(
            factory_implemented
            and plan_valid
            and manifest_valid
            and completion_valid
            and outputs_genuine
        )
        if complete:
            execution_complete.append(target)
        entries.append(
            {
                "target_node_id": target,
                "producer_node_id": producer_node,
                "manifest_origin": (
                    "authenticated_campaign_bootstrap"
                    if is_bootstrap
                    else "registered_downstream_plan_factory"
                ),
                "factory_registration_present": (
                    True if is_bootstrap else factory is not None
                ),
                "factory_implementation_present": factory_implemented,
                "shared_hook_present": shared_hook_is_complete,
                "valid_plan_observed": plan_valid,
                "valid_task_manifest_observed": manifest_valid,
                "complete_manifest_attestation_observed": completion_valid,
                "genuine_worker_outputs_observed": outputs_genuine,
                "execution_complete_producer_coverage": complete,
                "observation_error": observation_error,
            }
        )
    missing_execution = sorted(set(expected) - set(execution_complete))
    artifact = with_content_hash(
        {
            "contract": MANIFEST_PLAN_PRODUCER_AUDIT_CONTRACT,
            "schema_version": 1,
            "manifest_target_count": len(expected),
            "bootstrap_prepublished_target_count": len(bootstrap_targets),
            "bootstrap_prepublished_targets": sorted(bootstrap_targets),
            "downstream_plan_factory_target_count": len(downstream_targets),
            "registered_plan_factory_count": len(
                set(by_factory_target) & set(downstream_targets)
            ),
            "missing_plan_factory_registration_count": len(
                missing_registrations
            ),
            "missing_plan_factory_registrations": missing_registrations,
            "unexpected_plan_factory_registrations": unexpected_registrations,
            "implemented_producer_count": len(implementation_covered),
            "implemented_producer_targets": sorted(
                implementation_covered
            ),
            "execution_complete_producer_count": len(execution_complete),
            "execution_complete_producer_targets": sorted(
                execution_complete
            ),
            "missing_execution_complete_producer_count": len(
                missing_execution
            ),
            "missing_execution_complete_producers": missing_execution,
            "entries": entries,
            "shared_materialization_hook_present": shared_hook_is_complete,
            "shared_hook_counted_as_factory_evidence": False,
            "registration_alone_counted_as_execution_evidence": False,
            "synthetic_dag_counted_as_execution_evidence": False,
            "all_manifest_targets_execution_complete": not missing_execution,
            "full_submission_producer_gate_passed": not missing_execution,
        }
    )
    validate_manifest_plan_producer_audit(artifact)
    return artifact


def validate_manifest_plan_producer_audit(
    payload: Mapping[str, Any],
) -> str:
    """Recompute every readiness summary from the per-target evidence."""

    digest = validate_content_hash(
        payload, expected_contract=MANIFEST_PLAN_PRODUCER_AUDIT_CONTRACT
    )
    expected_targets = sorted(TASK_MANIFEST_PRODUCER_NODES)
    bootstrap_targets = set(BOOTSTRAP_INPUT_MANIFEST_NODES) | set(
        STATIC_EXPERIMENT_MANIFEST_NODES
    )
    entries = list(payload.get("entries", ()))
    targets = [str(entry.get("target_node_id", "")) for entry in entries]
    if targets != expected_targets or len(targets) != len(set(targets)):
        raise ValueError("plan-producer audit target coverage differs")
    for entry in entries:
        target = str(entry["target_node_id"])
        is_bootstrap = target in bootstrap_targets
        expected_origin = (
            "authenticated_campaign_bootstrap"
            if is_bootstrap
            else "registered_downstream_plan_factory"
        )
        expected_complete = all(
            entry.get(key) is True
            for key in (
                "factory_implementation_present",
                "valid_plan_observed",
                "valid_task_manifest_observed",
                "complete_manifest_attestation_observed",
                "genuine_worker_outputs_observed",
            )
        )
        if (
            entry.get("producer_node_id")
            != TASK_MANIFEST_PRODUCER_NODES[target]
            or entry.get("manifest_origin") != expected_origin
            or bool(entry.get("factory_registration_present"))
            != (
                True
                if is_bootstrap
                else target
                not in payload["missing_plan_factory_registrations"]
            )
            or bool(entry.get("execution_complete_producer_coverage"))
            != expected_complete
        ):
            raise ValueError(
                f"{target} plan-producer audit evidence differs"
            )
    downstream = sorted(set(expected_targets) - bootstrap_targets)
    implemented = sorted(
        entry["target_node_id"]
        for entry in entries
        if entry["factory_implementation_present"] is True
    )
    complete = sorted(
        entry["target_node_id"]
        for entry in entries
        if entry["execution_complete_producer_coverage"] is True
    )
    missing_complete = sorted(set(expected_targets) - set(complete))
    missing_registration = sorted(
        entry["target_node_id"]
        for entry in entries
        if entry["target_node_id"] in downstream
        and entry["factory_registration_present"] is not True
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("manifest_target_count") != len(expected_targets)
        or payload.get("bootstrap_prepublished_target_count")
        != len(bootstrap_targets)
        or payload.get("bootstrap_prepublished_targets")
        != sorted(bootstrap_targets)
        or payload.get("downstream_plan_factory_target_count")
        != len(downstream)
        or payload.get("registered_plan_factory_count")
        != len(downstream) - len(missing_registration)
        or payload.get("missing_plan_factory_registration_count")
        != len(missing_registration)
        or payload.get("missing_plan_factory_registrations")
        != missing_registration
        or payload.get("implemented_producer_count") != len(implemented)
        or payload.get("implemented_producer_targets") != implemented
        or payload.get("execution_complete_producer_count")
        != len(complete)
        or payload.get("execution_complete_producer_targets") != complete
        or payload.get("missing_execution_complete_producer_count")
        != len(missing_complete)
        or payload.get("missing_execution_complete_producers")
        != missing_complete
        or payload.get("shared_hook_counted_as_factory_evidence") is not False
        or payload.get(
            "registration_alone_counted_as_execution_evidence"
        )
        is not False
        or payload.get("synthetic_dag_counted_as_execution_evidence")
        is not False
        or bool(payload.get("all_manifest_targets_execution_complete"))
        != (not missing_complete)
        or bool(payload.get("full_submission_producer_gate_passed"))
        != (not missing_complete)
    ):
        raise ValueError("plan-producer audit summary differs")
    return digest


def _entrypoint(node_id: str) -> str:
    if node_id in MIDDLE_NODE_ENTRYPOINTS:
        return MIDDLE_NODE_ENTRYPOINTS[node_id]
    if node_id in LATE_NODE_ENTRYPOINTS:
        return LATE_NODE_ENTRYPOINTS[node_id][0]
    if node_id in FINAL_NODE_ENTRYPOINTS:
        return FINAL_NODE_ENTRYPOINTS[node_id][0]
    return "scripts/write_retb_synthetic_output.py"


def _rows(
    root: Path, *, node_id: str, count: int
) -> list[dict[str, Any]]:
    rows = []
    entrypoint = _entrypoint(node_id)
    for index in range(count):
        output = root / "operational_validation" / "outputs" / node_id / (
            f"row_{index}.json"
        )
        argv = [
            sys.executable,
            entrypoint,
            "--campaign-root",
            str(root),
        ]
        if entrypoint == "scripts/write_retb_synthetic_output.py":
            argv.extend(
                [
                    "--node-id",
                    node_id,
                    "--task-index",
                    str(index),
                    "--output",
                    str(output),
                ]
            )
        if node_id == "hlt_v3_cache":
            argv.extend(["--profile-id", "D_NOMINAL"])
        rows.append(
            {
                "task_id": f"{node_id}:{index}",
                "argv": argv,
                "environment": {"RETB_SYNTHETIC_CONTROL_PLANE": "1"},
                "expected_outputs": [str(output)],
                "input_artifact_hashes": {
                    "synthetic_configuration": f"{index + 1:064x}"
                },
            }
        )
    return rows


def _synthetic_output(
    *,
    root: Path,
    campaign: Mapping[str, Any],
    node_id: str,
    task_index: int,
    output: str | Path,
) -> dict[str, Any]:
    artifact = with_content_hash(
        {
            "contract": "retb_synthetic_node_output_v1",
            "schema_version": 1,
            "campaign_spec_sha256": campaign["content_hash"],
            "node_id": node_id,
            "task_index": int(task_index),
            "scientific_result": False,
            "performance_threshold_evaluated": False,
            "source": campaign["source"],
        }
    )
    write_immutable_json(output, artifact)
    return artifact


def _trigger(
    campaign: Mapping[str, Any], *, node_id: str
) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": "retb_synthetic_continuation_trigger_v1",
            "schema_version": 1,
            "node_id": node_id,
            "performance_result": "deliberately_negative",
            "source": campaign["source"],
        }
    )


def _publish_manifest(
    *,
    root: Path,
    repo_root: Path,
    campaign: Mapping[str, Any],
    graph: Mapping[str, Any],
    node: Mapping[str, Any],
    rows: list[dict[str, Any]],
    interrupted_probe: bool,
) -> tuple[dict[str, Any], bool]:
    node_id = str(node["node_id"])
    trigger = _trigger(campaign, node_id=node_id)
    if node_id in MIDDLE_CONTINUATION_MANIFEST_NODES:
        payload = build_middle_continuation(
            campaign=campaign,
            production_graph=graph,
            campaign_root=root,
            node_id=node_id,
            trigger_artifact=trigger,
            rows=rows,
        )
        publish_middle_continuation(campaign_root=root, payload=payload)
        return payload["dynamic_continuation"]["task_manifest"], interrupted_probe
    if node_id in LATE_CONTINUATION_MANIFEST_NODES:
        payload = build_late_continuation(
            campaign=campaign,
            production_graph=graph,
            campaign_root=root,
            node_id=node_id,
            trigger_artifact=trigger,
            rows=rows,
        )
        publish_late_continuation(campaign_root=root, payload=payload)
        return payload["dynamic_continuation"]["task_manifest"], interrupted_probe
    if node_id in FINAL_CONTINUATION_MANIFEST_NODES:
        payload = build_final_continuation(
            campaign=campaign,
            production_graph=graph,
            campaign_root=root,
            node_id=node_id,
            trigger_artifact=trigger,
            rows=rows,
        )
        publish_final_continuation(campaign_root=root, payload=payload)
        return payload["dynamic_continuation"]["task_manifest"], interrupted_probe
    if bool(node["dynamic_continuation"]):
        selector_path = (
            root
            / "operational_validation"
            / "selectors"
            / f"{node_id}.json"
        )
        write_immutable_json(selector_path, trigger)
        payload = build_dynamic_continuation(
            campaign=campaign,
            production_graph=graph,
            selector_output=trigger,
            selector_output_path=selector_path,
            downstream_node_id=node_id,
            rows=rows,
            campaign_root=root,
        )
        manifest_path = task_manifest_path_for_graph(
            graph, node_id=node_id, campaign_root=root
        )
        binding_path = Path(
            payload["continuation_binding"]["continuation_binding_path"]
        )
        if not interrupted_probe:
            write_immutable_json(manifest_path, payload["task_manifest"])
            try:
                validate_published_dynamic_continuation(
                    campaign=campaign,
                    production_graph=graph,
                    task_manifest=payload["task_manifest"],
                    campaign_root=root,
                )
            except ValueError as error:
                if "unique continuation binding" not in str(error):
                    raise
            else:
                raise AssertionError(
                    "interrupted continuation unexpectedly validated"
                )
            interrupted_probe = True
        publish_dynamic_continuation(
            bundle=payload,
            downstream_manifest_path=manifest_path,
            binding_path=binding_path,
        )
        validate_published_dynamic_continuation(
            campaign=campaign,
            production_graph=graph,
            task_manifest=payload["task_manifest"],
            campaign_root=root,
        )
        return payload["task_manifest"], interrupted_probe
    maximum = (
        1
        if node["array"] is None
        else int(node["array"]["maximum_concurrent_tasks"])
    )
    manifest = build_task_manifest(
        campaign_spec_sha256=campaign["content_hash"],
        production_graph_sha256=graph["content_hash"],
        node_id=node_id,
        rows=rows,
        maximum_concurrent_tasks=maximum,
    )
    validate_task_manifest_for_graph(
        manifest,
        production_graph=graph,
        campaign_root=root,
        repo_root=repo_root,
    )
    write_immutable_json(
        task_manifest_path_for_graph(
            graph, node_id=node_id, campaign_root=root
        ),
        manifest,
    )
    return manifest, interrupted_probe


def run_local_synthetic_dag(
    *,
    campaign_root: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Execute every production control-plane node locally without Slurm."""

    root = Path(campaign_root).resolve()
    source_root = Path(repo_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("synthetic campaign root must be empty")
    root.mkdir(parents=True, exist_ok=True)
    snapshot = source_snapshot(source_root)
    campaign, graph = _campaign_and_graph(root, snapshot=snapshot)
    write_immutable_json(root / "campaign_spec.json", campaign)
    write_immutable_json(
        root / "job_ledgers" / "production_graph.json", graph
    )
    interfaces = audit_worker_interfaces(
        production_graph=graph,
        repo_root=source_root,
        probe_python_clis=True,
    )
    manifests: dict[str, dict[str, Any]] = {}
    outputs: dict[str, dict[str, Any]] = {}
    aliases = 0
    interrupted_probe = False
    incomplete_array_rejected = False
    reuse_verified = False
    nodes = {str(row["node_id"]): row for row in graph["nodes"]}
    for node in graph["nodes"]:
        node_id = str(node["node_id"])
        alias = node["virtual_alias_of"]
        if alias is not None:
            outputs[node_id] = outputs[str(alias)]
            aliases += 1
            continue
        if node_id in DIRECT_WORKER_NODES:
            path = (
                root
                / "operational_validation"
                / "outputs"
                / node_id
                / "direct.json"
            )
            outputs[node_id] = _synthetic_output(
                root=root,
                campaign=campaign,
                node_id=node_id,
                task_index=0,
                output=path,
            )
            continue
        count = 2 if node_id == "offline_input_cache" else 1
        rows = _rows(root, node_id=node_id, count=count)
        manifest, interrupted_probe = _publish_manifest(
            root=root,
            repo_root=source_root,
            campaign=campaign,
            graph=graph,
            node=node,
            rows=rows,
            interrupted_probe=interrupted_probe,
        )
        manifests[node_id] = manifest
        for row in manifest["rows"]:
            index = int(row["task_index"])
            if node_id == "offline_input_cache" and index == 1:
                try:
                    publish_task_manifest_completion(
                        campaign_root=root,
                        campaign=campaign,
                        task_manifest=manifest,
                    )
                except FileNotFoundError:
                    incomplete_array_rejected = True
                else:
                    raise AssertionError(
                        "incomplete array unexpectedly aggregated"
                    )
            output = _synthetic_output(
                root=root,
                campaign=campaign,
                node_id=node_id,
                task_index=index,
                output=row["expected_outputs"][0],
            )
            publish_task_row_completion(
                campaign_root=root,
                campaign=campaign,
                task_manifest=manifest,
                task_index=index,
            )
            outputs[node_id] = output
        completion = publish_task_manifest_completion(
            campaign_root=root,
            campaign=campaign,
            task_manifest=manifest,
        )
        if node_id == "offline_input_cache":
            reusable = reusable_task_row_completion(
                campaign_root=root,
                campaign=campaign,
                task_manifest=manifest,
                task_index=0,
            )
            reuse_verified = (
                reusable is not None
                and reusable["content_hash"]
                == completion["artifact"]["rows"][0][
                    "row_completion_sha256"
                ]
            )
    if set(outputs) != set(nodes):
        raise AssertionError("synthetic DAG node coverage differs")
    producer_invocations = audit_manifest_producer_invocations(
        production_graph=graph,
        repo_root=source_root,
        campaign_root=root,
    )
    drifted = dict(campaign)
    drifted["source"] = dict(campaign["source"])
    drifted["source"]["status_sha256"] = "0" * 64
    drifted.pop("content_hash")
    drifted = with_content_hash(drifted)
    try:
        validate_campaign_source(drifted, repo_root=source_root)
    except ValueError:
        source_drift_rejected = True
    else:
        raise AssertionError("source drift unexpectedly validated")
    jobs = {
        node["node_id"]: str(70_000 + index)
        for index, node in enumerate(graph["nodes"])
    }
    selector_job = jobs["accuracy_finalist_selector"]
    jobs["rejection_finalist_selector"] = selector_job
    jobs["locked_scale_finalists"] = selector_job
    ledger = build_job_ledger(
        production_graph=graph,
        jobs=jobs,
        submission_mode="completed",
        completion_artifact_hashes={
            "locked_scale_finalists": outputs[
                "locked_scale_finalists"
            ]["content_hash"],
            "final_test_execution_lock": outputs[
                "final_test_execution_lock"
            ]["content_hash"],
            "sealed_final_test_evaluation": outputs[
                "sealed_final_test"
            ]["content_hash"],
            "final_report": outputs["final_report"]["content_hash"],
        },
    )
    validate_job_ledger(ledger, production_graph=graph)
    stale_candidate = jobs["campaign_bootstrap"]
    stale_resolved = validate_stale_cancellation_request(
        job_ledger=ledger,
        stale_job_ids=[stale_candidate, stale_candidate],
        source_validated=False,
    )
    if stale_resolved != [stale_candidate]:
        raise AssertionError("stale cancellation deduplication differs")
    try:
        validate_stale_cancellation_request(
            job_ledger=ledger,
            stale_job_ids=[stale_candidate],
            source_validated=True,
        )
    except ValueError:
        matching_source_cancellation_rejected = True
    else:
        raise AssertionError(
            "matching source unexpectedly authorized cancellation"
        )
    smoke_jobs = {
        node["node_id"]: str(80_000 + index)
        for index, node in enumerate(graph["nodes"])
    }
    smoke_jobs["rejection_finalist_selector"] = smoke_jobs[
        "accuracy_finalist_selector"
    ]
    smoke_jobs["locked_scale_finalists"] = smoke_jobs[
        "accuracy_finalist_selector"
    ]
    smoke_ledger = build_job_ledger(
        production_graph=graph,
        jobs=smoke_jobs,
        submission_mode="smoke_simulation",
    )
    validate_job_ledger(smoke_ledger, production_graph=graph)
    resume = build_resume_plan(
        production_graph=graph,
        previous_ledger=ledger,
        completed_nodes={
            "split_build": outputs["split_build"]["content_hash"]
        },
        failed_nodes=["campaign_bootstrap"],
    )
    if "campaign_bootstrap" not in {
        row["node_id"] for row in resume["ready_to_resubmit"]
    }:
        raise AssertionError("failed-node resume frontier differs")
    checks = {
        "all_graph_nodes_traversed": True,
        "all_stages_A_through_N_traversed": True,
        "synthetic_outputs_are_not_worker_execution_evidence": True,
        "incomplete_array_completion_rejected": incomplete_array_rejected,
        "authenticated_row_reuse_verified": reuse_verified,
        "source_drift_rejected": source_drift_rejected,
        "interrupted_selector_continuation_rejected_then_recovered": (
            interrupted_probe
        ),
        "failed_node_restart_frontier_verified": True,
        "completed_ledger_bound": True,
        "smoke_simulation_resolved_all_nodes": (
            smoke_ledger["all_nodes_bound"] is True
        ),
        "stale_cancellation_requires_explicit_drifted_lineage": (
            matching_source_cancellation_rejected
        ),
        "performance_threshold_abort_observed": False,
    }
    if not all(
        value is True
        for key, value in checks.items()
        if key != "performance_threshold_abort_observed"
    ) or checks["performance_threshold_abort_observed"] is not False:
        raise AssertionError("local operational checks differ")
    report = bind_source(
        with_content_hash(
            {
                "contract": LOCAL_OPERATIONAL_REPORT_CONTRACT,
                "schema_version": 3,
                "campaign_spec_sha256": campaign["content_hash"],
                "production_graph_sha256": graph["content_hash"],
                "node_count": len(graph["nodes"]),
                "manifest_node_count": len(manifests),
                "virtual_alias_count": aliases,
                "worker_interfaces": interfaces,
                "manifest_producer_invocations": producer_invocations,
                "checks": checks,
                "full_submission_eligible": producer_invocations[
                    "full_submission_producer_gate_passed"
                ],
                "real_miniature_campaign_execution_required": True,
                "synthetic_control_plane_execution_is_sufficient": False,
                "scientific_results_allowed": False,
                "slurm_used": False,
            }
        ),
        source_snapshot=snapshot,
    )
    write_immutable_json(
        root / "operational_validation" / "local_report.json", report
    )
    return report


def build_tigris_smoke_evidence(
    *,
    campaign_spec: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    completed_ledger: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(campaign_spec)
    graph_sha = validate_production_graph(production_graph)
    validate_production_campaign_binding(production_graph, campaign_spec)
    ledger_sha = validate_job_ledger(
        completed_ledger, production_graph=production_graph
    )
    if (
        campaign_spec["campaign_profile"] != "miniature_test"
        or production_graph["campaign_profile"]
        != "nonproduction_miniature_test"
        or completed_ledger["submission_mode"] != "completed"
        or not completed_ledger["all_nodes_bound"]
        or not completed_ledger["completed_after_final_report"]
        or production_graph["source_commit"]
        != source_snapshot["source_commit"]
        or production_graph["source_status_sha256"]
        != source_snapshot["source_status_sha256"]
        or campaign_spec.get("source") != _source_record(source_snapshot)
    ):
        raise ValueError("Tigris smoke evidence is incomplete")
    return bind_source(
        with_content_hash(
            {
                "contract": TIGRIS_SMOKE_EVIDENCE_CONTRACT,
                "schema_version": 1,
                "campaign_spec_sha256": campaign_spec["content_hash"],
                "production_graph_sha256": graph_sha,
                "completed_job_ledger_sha256": ledger_sha,
                "all_stages_A_through_N_completed_on_tigris": True,
                "final_test_executed_exactly_once": True,
                "scientific_results_allowed": False,
            }
        ),
        source_snapshot=source_snapshot,
    )


def build_production_dry_run_evidence(
    *,
    production_graph: Mapping[str, Any],
    dry_run_ledger: Mapping[str, Any],
    storage_measurements: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    graph_sha = validate_production_graph(production_graph)
    ledger_sha = validate_job_ledger(
        dry_run_ledger, production_graph=production_graph
    )
    storage_sha = validate_storage_measurements(storage_measurements)
    if (
        storage_measurements["contract"] != STORAGE_MEASUREMENTS_CONTRACT
        or storage_measurements["measurement_profile"]
        != "production_source_evidence"
        or production_graph["campaign_profile"]
        != "production_500k_100k_50k_300k_scale3m"
        or dry_run_ledger["submission_mode"] != "dry_run"
        or production_graph["storage_measurements_sha256"]
        not in {storage_sha, bind_source(
            storage_measurements, source_snapshot=source_snapshot
        )["content_hash"]}
    ):
        raise ValueError("production dry-run evidence differs")
    return bind_source(
        with_content_hash(
            {
                "contract": PRODUCTION_DRY_RUN_EVIDENCE_CONTRACT,
                "schema_version": 1,
                "production_graph_sha256": graph_sha,
                "dry_run_job_ledger_sha256": ledger_sha,
                "storage_measurements_sha256": storage_sha,
                "authenticated_storage_admitted": True,
                "production_submission_performed": False,
            }
        ),
        source_snapshot=source_snapshot,
    )


def build_full_submission_authorization(
    *,
    local_report: Mapping[str, Any],
    tigris_smoke_evidence: Mapping[str, Any],
    production_dry_run_evidence: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    local_sha = validate_content_hash(
        local_report, expected_contract=LOCAL_OPERATIONAL_REPORT_CONTRACT
    )
    smoke_sha = validate_content_hash(
        tigris_smoke_evidence,
        expected_contract=TIGRIS_SMOKE_EVIDENCE_CONTRACT,
    )
    dry_sha = validate_content_hash(
        production_dry_run_evidence,
        expected_contract=PRODUCTION_DRY_RUN_EVIDENCE_CONTRACT,
    )
    required_local_checks = {
        "all_graph_nodes_traversed": True,
        "all_stages_A_through_N_traversed": True,
        "synthetic_outputs_are_not_worker_execution_evidence": True,
        "incomplete_array_completion_rejected": True,
        "authenticated_row_reuse_verified": True,
        "source_drift_rejected": True,
        "interrupted_selector_continuation_rejected_then_recovered": True,
        "failed_node_restart_frontier_verified": True,
        "completed_ledger_bound": True,
        "smoke_simulation_resolved_all_nodes": True,
        "stale_cancellation_requires_explicit_drifted_lineage": True,
        "performance_threshold_abort_observed": False,
    }
    invocation_audit = local_report.get(
        "manifest_producer_invocations", {}
    )
    validate_manifest_plan_producer_audit(invocation_audit)
    if (
        local_report.get("schema_version") != 3
        or local_report.get("checks") != required_local_checks
        or local_report.get("scientific_results_allowed") is not False
        or local_report.get("slurm_used") is not False
        or local_report.get("full_submission_eligible") is not True
        or local_report.get(
            "real_miniature_campaign_execution_required"
        )
        is not True
        or local_report.get(
            "synthetic_control_plane_execution_is_sufficient"
        )
        is not False
        or invocation_audit.get("manifest_target_count") != 48
        or invocation_audit.get(
            "bootstrap_prepublished_target_count"
        )
        != 13
        or invocation_audit.get(
            "downstream_plan_factory_target_count"
        )
        != 35
        or invocation_audit.get(
            "missing_plan_factory_registration_count"
        )
        != 0
        or invocation_audit.get(
            "execution_complete_producer_count"
        )
        != 48
        or invocation_audit.get(
            "missing_execution_complete_producer_count"
        )
        != 0
        or invocation_audit.get(
            "all_manifest_targets_execution_complete"
        )
        is not True
        or invocation_audit.get(
            "full_submission_producer_gate_passed"
        )
        is not True
        or invocation_audit.get(
            "shared_hook_counted_as_factory_evidence"
        )
        is not False
        or invocation_audit.get(
            "registration_alone_counted_as_execution_evidence"
        )
        is not False
        or invocation_audit.get(
            "synthetic_dag_counted_as_execution_evidence"
        )
        is not False
    ):
        raise ValueError(
            "local operational validation is incomplete; all manifest "
            "targets require execution-complete genuine producer evidence"
        )
    if (
        tigris_smoke_evidence.get(
            "all_stages_A_through_N_completed_on_tigris"
        )
        is not True
        or tigris_smoke_evidence.get("final_test_executed_exactly_once")
        is not True
        or tigris_smoke_evidence.get("scientific_results_allowed")
        is not False
        or production_dry_run_evidence.get(
            "authenticated_storage_admitted"
        )
        is not True
        or production_dry_run_evidence.get(
            "production_submission_performed"
        )
        is not False
    ):
        raise ValueError("external operational validation is incomplete")
    expected_source = _source_record(source_snapshot)
    if any(
        artifact.get("source") != expected_source
        for artifact in (
            local_report,
            tigris_smoke_evidence,
            production_dry_run_evidence,
        )
    ):
        raise ValueError("operational authorization source differs")
    return bind_source(
        with_content_hash(
            {
                "contract": FULL_SUBMISSION_AUTHORIZATION_CONTRACT,
                "schema_version": 2,
                "local_operational_report_sha256": local_sha,
                "tigris_smoke_evidence_sha256": smoke_sha,
                "production_dry_run_evidence_sha256": dry_sha,
                "required_validation_order": [
                    "local_synthetic_DAG",
                    "local_miniature_worker_interfaces",
                    "smoke_simulate",
                    "execution_complete_manifest_plan_audit",
                    "real_miniature_Tigris_smoke",
                    "production_dry_run_authenticated_storage",
                ],
                "all_required_preproduction_gates_passed": True,
                "full_submission_authorized": True,
                "performance_threshold_required": False,
            }
        ),
        source_snapshot=source_snapshot,
    )


def validate_stale_cancellation_request(
    *,
    job_ledger: Mapping[str, Any],
    stale_job_ids: list[str],
    source_validated: bool,
) -> list[str]:
    """Resolve only explicitly named, ledger-bound stale jobs."""

    validate_content_hash(job_ledger, expected_contract=JOB_LEDGER_CONTRACT)
    bound = {
        str(value)
        for value in job_ledger["jobs"].values()
        if value is not None
    }
    stale = [str(value) for value in stale_job_ids]
    if (
        not stale
        or any(not value.isdigit() or value not in bound for value in stale)
    ):
        raise ValueError(
            "stale cancellation requires explicit ledger-bound job IDs"
        )
    if source_validated:
        raise ValueError(
            "matching source lineage cannot authorize stale cancellation"
        )
    return sorted(set(stale), key=int)


def validate_full_submission_authorization(
    payload: Mapping[str, Any],
    *,
    current_source_snapshot: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=FULL_SUBMISSION_AUTHORIZATION_CONTRACT
    )
    if (
        payload.get("source") != _source_record(current_source_snapshot)
        or payload.get("all_required_preproduction_gates_passed") is not True
        or payload.get("full_submission_authorized") is not True
        or payload.get("performance_threshold_required") is not False
    ):
        raise ValueError("full-submission authorization is stale or incomplete")
    if payload.get("required_validation_order") != [
        "local_synthetic_DAG",
        "local_miniature_worker_interfaces",
        "smoke_simulate",
        "execution_complete_manifest_plan_audit",
        "real_miniature_Tigris_smoke",
        "production_dry_run_authenticated_storage",
    ]:
        raise ValueError("full-submission authorization order differs")
    return digest


__all__ = [
    "FULL_SUBMISSION_AUTHORIZATION_CONTRACT",
    "LOCAL_OPERATIONAL_REPORT_CONTRACT",
    "PRODUCTION_DRY_RUN_EVIDENCE_CONTRACT",
    "TIGRIS_SMOKE_EVIDENCE_CONTRACT",
    "audit_manifest_producer_invocations",
    "audit_worker_interfaces",
    "build_full_submission_authorization",
    "build_production_dry_run_evidence",
    "build_tigris_smoke_evidence",
    "run_local_synthetic_dag",
    "validate_full_submission_authorization",
    "validate_manifest_plan_producer_audit",
    "validate_stale_cancellation_request",
]
