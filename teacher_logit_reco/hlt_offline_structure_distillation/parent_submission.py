"""Dry-run or submit shared RETB/RPT parent rebuild wrappers for HOSD."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from .contracts import (
    PARENT_GROUP_COMPLETION_CONTRACT,
    PARENT_REBUILD_PLAN_CONTRACT,
    load_hashed_json,
    validate_content_hash,
    with_content_hash,
    write_immutable_bytes,
    write_immutable_json,
)
from .parents import HLT_CACHE_SET_CONTRACT, PARENT_REQUIREMENTS
from .workflow import load_and_validate_campaign
from teacher_logit_reco.relation_expert_token_bridge import (
    TASK_MANIFEST_COMPLETION_CONTRACT,
    TASK_MANIFEST_CONTRACT,
    load_hashed_json as load_retb_hashed_json,
    validate_task_manifest_completion,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    bind_source,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (
    HLT_V3_ARRAY_FILENAME,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (
    source_snapshot,
)
from teacher_logit_reco.relational_part.ca_tree import validate_backend_manifest


GROUP_WRAPPERS = {
    "hlt": (
        "sbatch/run_retb_build_offline_inputs.sh",
        "sbatch/run_retb_build_hlt_v3.sh",
    ),
    "tree": (
        "sbatch/run_retb_compiled_region_backend.sh",
        "sbatch/run_retb_build_region_trees.sh",
        "sbatch/run_retb_finalize_region_trees.sh",
    ),
    "normalization": (
        "sbatch/run_retb_fit_normalizers.sh",
        "sbatch/run_retb_audit_inputs.sh",
    ),
}

WRAPPER_TASK_NODES = {
    "sbatch/run_retb_build_offline_inputs.sh": "offline_input_cache",
    "sbatch/run_retb_build_hlt_v3.sh": "hlt_v3_cache",
    "sbatch/run_retb_build_region_trees.sh": "region_tree_cache",
    "sbatch/run_retb_finalize_region_trees.sh": "region_tree_finalize",
    "sbatch/run_retb_fit_normalizers.sh": "normalizers_500k",
    "sbatch/run_retb_audit_inputs.sh": "input_audit",
}


def _run_controller_command(argv: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(argv, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            "shared RETB runtime preparation failed: "
            + " ".join(argv)
            + f"\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def shared_parent_runtime_commands(
    *,
    campaign_root: str | Path,
    repo_root: str | Path,
    data_dir: str | Path,
) -> list[list[str]]:
    """Return the deterministic commands that make the RETB child executable."""

    root = Path(campaign_root).resolve()
    repo = Path(repo_root).resolve()
    campaign = load_and_validate_campaign(root, repo_root=repo)
    shared = root / "inputs" / "shared_retb_parent_campaign"
    shared_campaign = load_hashed_json(shared / "campaign_spec.json")
    if shared_campaign.get("source") != campaign.get("source"):
        raise ValueError("shared RETB parent source differs from HOSD campaign")
    if shared_campaign.get("campaign_id") != shared.name:
        raise ValueError(
            "shared RETB campaign ID differs from its campaign-root basename"
        )
    profile = str(shared_campaign.get("campaign_profile"))
    if profile not in {"miniature_test", "production_500k_scale3m"}:
        raise ValueError("shared RETB parent campaign profile differs")
    graph = [
        sys.executable,
        "-s",
        str((repo / "scripts" / "submit_retb_graph.py").resolve()),
        "--campaign-id",
        str(shared_campaign["campaign_id"]),
        "--campaign-root",
        str(shared),
        "--storage-measurements",
        str(shared / "storage_measurements.json"),
        "--write-artifacts",
    ]
    if profile == "miniature_test":
        graph.append("--miniature")
    bootstrap = [
        sys.executable,
        "-s",
        str((repo / "scripts" / "bootstrap_retb_input_tasks.py").resolve()),
        "--campaign-root",
        str(shared),
        "--production-graph",
        str(shared / "job_ledgers" / "production_graph.json"),
        "--data-dir",
        str(Path(data_dir).resolve()),
    ]
    return [graph, bootstrap]


def prepare_shared_parent_runtime(
    *,
    campaign_root: str | Path,
    repo_root: str | Path,
    data_dir: str | Path | None = None,
) -> list[list[str]]:
    """Publish the shared RETB graph and genuine Stage-A task manifests."""

    resolved_data = data_dir or os.environ.get(
        "DATA_DIR", "/home/ryreu/atlas/PracticeTagging/data"
    )
    ensure_shared_bootstrap_split(campaign_root=campaign_root)
    commands = shared_parent_runtime_commands(
        campaign_root=campaign_root,
        repo_root=repo_root,
        data_dir=resolved_data,
    )
    for command in commands:
        _run_controller_command(command)
    shared = (
        Path(campaign_root).resolve()
        / "inputs"
        / "shared_retb_parent_campaign"
    )
    required = [
        shared / "bootstrap" / "split_manifest.json.gz",
        shared / "job_ledgers" / "production_graph.json",
        *(
            shared / "job_ledgers" / "tasks" / f"{node}.json"
            for node in sorted(set(WRAPPER_TASK_NODES.values()))
        ),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "shared RETB runtime preparation omitted required artifacts: "
            + repr(missing)
        )
    return commands


def ensure_shared_bootstrap_split(
    *, campaign_root: str | Path
) -> dict[str, Any]:
    """Materialize RETB's byte-identical pre-campaign split input."""

    shared = (
        Path(campaign_root).resolve()
        / "inputs"
        / "shared_retb_parent_campaign"
    )
    source = shared / "inputs" / "split_manifest.json.gz"
    destination = shared / "bootstrap" / "split_manifest.json.gz"
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(
            f"shared RETB input split manifest is absent or unsafe: {source}"
        )
    encoded = source.read_bytes()
    publication = write_immutable_bytes(destination, encoded)
    if destination.read_bytes() != encoded:
        raise ValueError("shared RETB bootstrap split differs from input split")
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "publication": publication["status"],
        "byte_identical": True,
    }


def build_parent_submission_plan(
    *,
    campaign_root: str | Path,
    repo_root: str | Path,
    group: str,
) -> dict[str, Any]:
    if group not in GROUP_WRAPPERS:
        raise ValueError(f"unknown HOSD parent rebuild group {group!r}")
    root = Path(campaign_root).resolve()
    campaign = load_and_validate_campaign(root, repo_root=repo_root)
    rebuild = load_hashed_json(
        root / "inputs" / "inherited_parent_rebuild_plan.json",
        expected_contract=PARENT_REBUILD_PLAN_CONTRACT,
    )
    if rebuild["source"] != campaign["source"]:
        raise ValueError("parent rebuild plan source differs from HOSD campaign")
    group_row = next(
        (row for row in rebuild["groups"] if row["group"] == group), None
    )
    resolved_path = root / "inputs" / "resolved_inherited_parent_lock.json"
    if resolved_path.is_file():
        resolved = load_hashed_json(
            resolved_path, expected_contract="hosd_parent_status_v1"
        )
        if resolved.get("source") != campaign["source"]:
            raise ValueError("resolved parent lock source differs")
        required_ids = {
            row.parent_id
            for row in PARENT_REQUIREMENTS
            if row.rebuild_group == group
        }
        resolved_rows = {
            row["parent_id"]: row for row in resolved["requirements"]
        }
        if required_ids and all(
            resolved_rows.get(parent_id, {}).get("reusable") is True
            for parent_id in required_ids
        ):
            group_row = None
    shared_root = root / "inputs" / "shared_retb_parent_campaign"
    log_directory = root / "job_ledgers" / "slurm" / "parent_controllers"
    commands = []
    runtime_ready = True
    if group_row is not None:
        for wrapper in GROUP_WRAPPERS[group]:
            task_node = WRAPPER_TASK_NODES.get(wrapper)
            array_argument: list[str] = []
            task_manifest_sha256 = None
            if task_node is not None:
                manifest_path = (
                    shared_root / "job_ledgers" / "tasks" / f"{task_node}.json"
                )
                if manifest_path.is_file():
                    manifest = load_retb_hashed_json(
                        manifest_path, expected_contract=TASK_MANIFEST_CONTRACT
                    )
                    task_manifest_sha256 = manifest["content_hash"]
                    array_argument = [f"--array={manifest['slurm_array']}"]
                else:
                    runtime_ready = False
            commands.append(
                {
                    "argv": [
                        "sbatch",
                        "--parsable",
                        "--wait",
                        f"--output={log_directory}/%x_%A_%a.out",
                        f"--error={log_directory}/%x_%A_%a.err",
                        *array_argument,
                        (
                            "--export=ALL,"
                            f"CAMPAIGN_ROOT={shared_root},"
                            f"PROJECT_DIR={Path(repo_root).resolve()}"
                        ),
                        str((Path(repo_root) / wrapper).resolve()),
                    ],
                    "wrapper": wrapper,
                    "task_node": task_node,
                    "task_manifest_sha256": task_manifest_sha256,
                }
            )
    return {
        "campaign_id": campaign["campaign_id"],
        "campaign_spec_sha256": campaign["content_hash"],
        "rebuild_plan_sha256": rebuild["content_hash"],
        "group": group,
        "parent_ids": [] if group_row is None else list(group_row["parent_ids"]),
        "commands": commands,
        "runtime_ready": runtime_ready,
        "log_directory": str(log_directory.resolve()),
        "already_satisfied": group_row is None,
        "scientific_results_consulted": False,
        "performance_based_submission_pruning": False,
    }


def submit_parent_plan(plan: Mapping[str, Any]) -> list[str]:
    """Submit an explicitly requested plan with afterok ordering."""

    if not bool(plan.get("runtime_ready", True)):
        raise RuntimeError(
            "shared RETB runtime is not prepared; publish its production graph "
            "and Stage-A task manifests before parent submission"
        )
    raw_log_directory = plan.get("log_directory")
    log_directory = (
        None
        if raw_log_directory is None
        else Path(str(raw_log_directory)).resolve()
    )
    if log_directory is not None:
        log_directory.mkdir(parents=True, exist_ok=True)
    job_ids: list[str] = []
    previous: str | None = None
    for row in plan["commands"]:
        argv = list(row["argv"])
        if previous is not None:
            argv.insert(2, f"--dependency=afterok:{previous}")
        completed = subprocess.run(argv, capture_output=True, text=True)
        if completed.returncode:
            diagnostics = []
            if log_directory is not None and log_directory.is_dir():
                logs = sorted(
                    (
                        path
                        for path in log_directory.iterdir()
                        if path.is_file() and path.suffix in {".out", ".err"}
                    ),
                    key=lambda path: path.stat().st_mtime_ns,
                    reverse=True,
                )[:6]
                for path in logs:
                    diagnostics.append(
                        f"--- {path} ---\n"
                        + path.read_text(encoding="utf-8", errors="replace")[-12000:]
                    )
            raise RuntimeError(
                f"parent Slurm worker failed for {row['wrapper']}: "
                f"exit={completed.returncode}\n"
                f"sbatch stdout:\n{completed.stdout}\n"
                f"sbatch stderr:\n{completed.stderr}\n"
                + "\n".join(diagnostics)
            )
        job_id = completed.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"sbatch returned an invalid job ID: {job_id!r}")
        job_ids.append(job_id)
        previous = job_id
    return job_ids


def _publish_hlt_cache_set(
    *, campaign_root: Path, shared_root: Path
) -> dict[str, Any]:
    campaign = load_and_validate_campaign(
        campaign_root, repo_root=Path(__file__).resolve().parents[2]
    )
    shared_campaign = load_hashed_json(shared_root / "campaign_spec.json")
    manifest = load_retb_hashed_json(
        shared_root / "job_ledgers" / "tasks" / "hlt_v3_cache.json",
        expected_contract=TASK_MANIFEST_CONTRACT,
    )
    completion = load_retb_hashed_json(
        shared_root
        / "job_ledgers"
        / "completions"
        / "hlt_v3_cache"
        / "manifest_completion.json",
        expected_contract=TASK_MANIFEST_COMPLETION_CONTRACT,
    )
    validate_task_manifest_completion(
        completion,
        campaign_root=shared_root,
        campaign=shared_campaign,
        task_manifest=manifest,
    )
    caches = []
    for row in manifest["rows"]:
        outputs = [
            Path(value)
            for value in row["expected_outputs"]
            if Path(value).name == "hlt_v3_metadata.json"
        ]
        if len(outputs) != 1:
            raise ValueError("HLT-v3 task row must publish one cache metadata")
        cache_root = outputs[0].parent
        array_path = cache_root / HLT_V3_ARRAY_FILENAME
        if not array_path.is_file() or array_path.is_symlink():
            raise FileNotFoundError(f"HLT-v3 cache array is absent: {array_path}")
        metadata = load_retb_hashed_json(
            outputs[0], expected_contract="retb_hlt_v3_cache_v1"
        )
        if metadata.get("source") != campaign.get("source"):
            raise ValueError("HLT-v3 cache source differs from HOSD campaign")
        caches.append(
            {
                "task_index": int(row["task_index"]),
                "logical_role": str(metadata["logical_role"]),
                "replica_id": int(metadata["replica_id"]),
                "realization_policy": str(metadata["realization_policy"]),
                "path": str(cache_root.resolve()),
                "metadata_sha256": str(metadata["content_hash"]),
            }
        )
    artifact = with_content_hash(
        {
            "contract": HLT_CACHE_SET_CONTRACT,
            "schema_version": 1,
            "source": campaign["source"],
            "campaign_spec_sha256": campaign["content_hash"],
            "shared_retb_campaign_spec_sha256": shared_campaign["content_hash"],
            "task_manifest_sha256": manifest["content_hash"],
            "task_manifest_completion_sha256": completion["content_hash"],
            "cache_count": len(caches),
            "caches": caches,
            "all_registered_coordinates_complete": True,
        }
    )
    write_immutable_json(
        campaign_root / "inputs" / "hlt_replicas" / "hlt_v3_cache_manifest.json",
        artifact,
    )
    return artifact


def _publish_source_bound_backend_alias(
    *, campaign_root: Path, shared_root: Path
) -> dict[str, Any]:
    raw = load_hashed_json(
        shared_root / "backend" / "backend_manifest.json",
        expected_contract="relational_ca_tree_backend_manifest_v3",
    )
    validate_backend_manifest(
        raw,
        binary_path=shared_root / "backend" / str(raw["binary_filename"]),
    )
    bound = bind_source(
        raw,
        source_snapshot=source_snapshot(Path(__file__).resolve().parents[2]),
    )
    write_immutable_json(
        campaign_root / "inputs" / "region_tree" / "backend_manifest.json",
        bound,
    )
    return bound


def finalize_parent_group(
    plan: Mapping[str, Any],
    *,
    campaign_root: str | Path,
    submitted_job_ids: list[str],
) -> dict[str, Any]:
    """Validate completed child outputs before the controller can succeed."""

    root = Path(campaign_root).resolve()
    names = {
        "hlt": "shared_hlt_parent_completion.json",
        "tree": "tree_parent_completion.json",
        "normalization": "relation_normalizer_parent_completion.json",
    }
    output_path = root / "inputs" / names[plan["group"]]
    if output_path.is_file():
        existing = load_hashed_json(
            output_path, expected_contract=PARENT_GROUP_COMPLETION_CONTRACT
        )
        if (
            existing.get("campaign_spec_sha256")
            != plan["campaign_spec_sha256"]
            or existing.get("rebuild_plan_sha256")
            != plan["rebuild_plan_sha256"]
            or existing.get("group") != plan["group"]
        ):
            raise ValueError("reusable parent-group completion lineage differs")
        return existing
    shared = root / "inputs" / "shared_retb_parent_campaign"
    if plan["group"] == "hlt" and plan["parent_ids"]:
        _publish_hlt_cache_set(campaign_root=root, shared_root=shared)
    if plan["group"] == "tree" and plan["parent_ids"]:
        _publish_source_bound_backend_alias(
            campaign_root=root, shared_root=shared
        )
    requirements = {
        row.parent_id: row
        for row in PARENT_REQUIREMENTS
        if row.rebuild_group == plan["group"]
    }
    if set(plan["parent_ids"]) - set(requirements):
        raise ValueError("parent group plan contains an unknown parent")
    rows = {}
    for parent_id in plan["parent_ids"]:
        requirement = requirements[parent_id]
        candidates = [
            root / requirement.canonical_path,
            shared / requirement.canonical_path,
        ]
        path = next((value for value in candidates if value.is_file()), None)
        if path is None:
            raise FileNotFoundError(
                f"completed parent rebuild did not publish {parent_id}"
            )
        artifact = load_hashed_json(
            path, expected_contract=requirement.expected_contract
        )
        rows[parent_id] = {
            "path": str(path.resolve()),
            "content_hash": artifact["content_hash"],
            "expected_contract": requirement.expected_contract,
        }
    artifact = with_content_hash(
        {
            "contract": PARENT_GROUP_COMPLETION_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": plan["campaign_spec_sha256"],
            "rebuild_plan_sha256": plan["rebuild_plan_sha256"],
            "group": plan["group"],
            "parents": rows,
            "submitted_job_ids": [str(value) for value in submitted_job_ids],
            "child_jobs_waited_for_success": True,
            "scientific_performance_read": False,
        }
    )
    write_immutable_json(output_path, artifact)
    return artifact


def plan_json(plan: Mapping[str, Any], *, submitted_job_ids: list[str] | None) -> str:
    payload = dict(plan)
    payload["submitted"] = submitted_job_ids is not None
    payload["submitted_job_ids"] = submitted_job_ids
    return json.dumps(payload, indent=2, sort_keys=True)


__all__ = [
    "GROUP_WRAPPERS",
    "WRAPPER_TASK_NODES",
    "build_parent_submission_plan",
    "ensure_shared_bootstrap_split",
    "prepare_shared_parent_runtime",
    "shared_parent_runtime_commands",
    "plan_json",
    "finalize_parent_group",
    "submit_parent_plan",
]
