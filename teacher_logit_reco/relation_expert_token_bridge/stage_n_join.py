"""Authenticated join for the independent Stage-N evidence branches."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import load_hashed_json, with_content_hash, write_immutable_json
from .dynamic_continuation import validate_published_dynamic_continuation
from .production import (
    task_manifest_path_for_graph,
    validate_production_campaign_binding,
    validate_task_manifest_for_graph,
)
from .task_completion import (
    task_manifest_completion_path,
    validate_task_manifest_completion,
)


STAGE_N_EVIDENCE_JOIN_CONTRACT = "retb_stage_n_evidence_join_v1"
STAGE_N_EVIDENCE_JOIN_PARENTS = (
    "postlock_oracle_targets",
    "finalist_controls",
    "prelock_final_inputs",
)


def build_stage_n_evidence_join(
    *,
    campaign_root: str | Path,
    repo_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate all branches and bind their complete output populations."""

    root = Path(campaign_root).resolve()
    source_root = Path(repo_root).resolve()
    graph_sha = validate_production_campaign_binding(
        production_graph, campaign
    )
    completion_hashes: dict[str, str] = {}
    manifest_hashes: dict[str, str] = {}
    output_hashes: dict[str, dict[str, str]] = {}
    for parent in STAGE_N_EVIDENCE_JOIN_PARENTS:
        manifest_path = task_manifest_path_for_graph(
            production_graph,
            node_id=parent,
            campaign_root=root,
        )
        completion_path = task_manifest_completion_path(
            root, node_id=parent
        )
        manifest = load_hashed_json(manifest_path)
        validate_task_manifest_for_graph(
            manifest,
            production_graph=production_graph,
            campaign_root=root,
            repo_root=source_root,
        )
        validate_published_dynamic_continuation(
            campaign=campaign,
            production_graph=production_graph,
            task_manifest=manifest,
            campaign_root=root,
        )
        completion = load_hashed_json(completion_path)
        completion_hashes[parent] = validate_task_manifest_completion(
            completion,
            campaign_root=root,
            campaign=campaign,
            task_manifest=manifest,
        )
        manifest_hashes[parent] = manifest["content_hash"]
        output_hashes[parent] = {
            path: digest
            for row in completion["rows"]
            for path, digest in row["output_hashes"].items()
        }
        if not output_hashes[parent]:
            raise ValueError(f"Stage-N join parent has no outputs: {parent}")
    return with_content_hash(
        {
            "contract": STAGE_N_EVIDENCE_JOIN_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign["content_hash"],
            "production_graph_sha256": graph_sha,
            "parent_node_order": list(STAGE_N_EVIDENCE_JOIN_PARENTS),
            "parent_task_manifest_hashes": manifest_hashes,
            "parent_completion_hashes": completion_hashes,
            "parent_output_hashes": output_hashes,
            "all_parent_rows_and_outputs_revalidated": True,
            "scientific_performance_used_as_join_gate": False,
            "only_integrity_or_execution_failures_block": True,
            "source": campaign["source"],
        }
    )


def publish_stage_n_evidence_join(
    *,
    output: str | Path,
    campaign_root: str | Path,
    repo_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = build_stage_n_evidence_join(
        campaign_root=campaign_root,
        repo_root=repo_root,
        campaign=campaign,
        production_graph=production_graph,
    )
    publication = write_immutable_json(output, artifact)
    return {"artifact": artifact, "publication": publication}


__all__ = [
    "STAGE_N_EVIDENCE_JOIN_CONTRACT",
    "STAGE_N_EVIDENCE_JOIN_PARENTS",
    "build_stage_n_evidence_join",
    "publish_stage_n_evidence_join",
]
