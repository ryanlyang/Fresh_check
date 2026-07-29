"""Completeness-gated dynamic continuation for sealed Stage N."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import validate_content_hash, with_content_hash, write_immutable_json
from .dynamic_continuation import (
    build_dynamic_continuation,
    publish_dynamic_continuation,
    validate_dynamic_continuation,
    validate_published_dynamic_continuation,
)
from .production import (
    FINAL_CONTINUATION_GATE_CONTRACT,
    FINAL_CONTINUATION_MANIFEST_NODES,
    FINAL_NODE_ENTRYPOINTS,
    task_manifest_path_for_graph,
    validate_production_campaign_binding,
    validate_production_graph,
    validate_task_manifest_for_graph,
)
from .task_completion import (
    task_manifest_completion_path,
    validate_task_manifest_completion,
)


FINAL_CONTINUATION_BUNDLE_CONTRACT = (
    "retb_stage_n_continuation_bundle_v1"
)
FINAL_WAVE_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "prelock_final_inputs": ("input_audit",),
    "stack_val_inference": ("scale_completion",),
    "accuracy_finalist_selector": ("stack_val_inference",),
    "postlock_oracle_targets": ("accuracy_finalist_selector",),
    "finalist_controls": ("accuracy_finalist_selector",),
    "final_test_execution_lock": (
        "postlock_oracle_targets",
        "finalist_controls",
        "prelock_final_inputs",
    ),
    "sealed_final_test": (
        "final_test_execution_lock",
    ),
    "final_report": ("sealed_final_test",),
}
if set(FINAL_WAVE_PREREQUISITES) != set(
    FINAL_CONTINUATION_MANIFEST_NODES
):
    raise RuntimeError("Stage-N continuation coverage differs")


def _validate_rows(
    *, node_id: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    if not rows:
        raise ValueError("Stage-N continuation rows are empty")
    if node_id in {
        "prelock_final_inputs",
        "accuracy_finalist_selector",
        "finalist_controls",
        "final_test_execution_lock",
        "sealed_final_test",
        "final_report",
    } and len(rows) != 1:
        raise ValueError(f"{node_id} requires exactly one aggregate row")
    allowed = FINAL_NODE_ENTRYPOINTS[node_id]
    for row in rows:
        argv = [str(value) for value in row.get("argv", ())]
        if (
            len(argv) < 2
            or argv[1].replace("\\", "/") not in allowed
            or "--dry-run" in argv
        ):
            raise ValueError(f"{node_id} Stage-N entry point differs")


def _load_prerequisites(
    *,
    node_id: str,
    campaign_root: Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    result = {}
    for parent in FINAL_WAVE_PREREQUISITES[node_id]:
        manifest_path = task_manifest_path_for_graph(
            production_graph,
            node_id=parent,
            campaign_root=campaign_root,
        )
        completion_path = task_manifest_completion_path(
            campaign_root, node_id=parent
        )
        if not manifest_path.is_file() or not completion_path.is_file():
            raise FileNotFoundError(
                f"Stage-N prerequisite is incomplete: {parent}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completion = json.loads(
            completion_path.read_text(encoding="utf-8")
        )
        validate_task_manifest_for_graph(
            manifest,
            production_graph=production_graph,
            campaign_root=campaign_root,
            repo_root=Path(__file__).resolve().parents[2],
        )
        parent_node = next(
            row
            for row in production_graph["nodes"]
            if row["node_id"] == parent
        )
        if parent_node["dynamic_continuation"]:
            validate_published_dynamic_continuation(
                campaign=campaign,
                production_graph=production_graph,
                task_manifest=manifest,
                campaign_root=campaign_root,
            )
        validate_task_manifest_completion(
            completion,
            campaign_root=campaign_root,
            campaign=campaign,
            task_manifest=manifest,
        )
        result[parent] = {
            "task_manifest": manifest,
            "completion": completion,
        }
    return result


def build_final_continuation_gate(
    *,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    campaign_root: str | Path,
    node_id: str,
    trigger_artifact: Mapping[str, Any],
    prerequisite_completions: Mapping[
        str, Mapping[str, Mapping[str, Any]]
    ],
) -> dict[str, Any]:
    campaign_sha = validate_content_hash(campaign)
    graph_sha = validate_production_graph(production_graph)
    validate_production_campaign_binding(production_graph, campaign)
    if node_id not in FINAL_WAVE_PREREQUISITES:
        raise ValueError("Stage-N continuation node is unregistered")
    trigger_sha = validate_content_hash(trigger_artifact)
    if trigger_artifact.get("source") != campaign["source"]:
        raise ValueError("Stage-N trigger source differs")
    required = FINAL_WAVE_PREREQUISITES[node_id]
    if set(prerequisite_completions) != set(required):
        raise ValueError("Stage-N prerequisite coverage differs")
    completion_hashes, output_hashes = {}, {}
    for parent in required:
        pair = prerequisite_completions[parent]
        if set(pair) != {"task_manifest", "completion"}:
            raise ValueError("Stage-N prerequisite pair differs")
        manifest, completion = pair["task_manifest"], pair["completion"]
        completion_hashes[parent] = validate_task_manifest_completion(
            completion,
            campaign_root=campaign_root,
            campaign=campaign,
            task_manifest=manifest,
        )
        output_hashes[parent] = {
            path: digest
            for row in completion["rows"]
            for path, digest in row["output_hashes"].items()
        }
    return with_content_hash(
        {
            "contract": FINAL_CONTINUATION_GATE_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign_sha,
            "production_graph_sha256": graph_sha,
            "node_id": node_id,
            "trigger_contract": trigger_artifact["contract"],
            "trigger_artifact_sha256": trigger_sha,
            "prerequisite_node_order": list(required),
            "prerequisite_completion_hashes": completion_hashes,
            "prerequisite_output_hashes": output_hashes,
            "all_prerequisite_rows_and_outputs_revalidated": True,
            "scientific_performance_used_as_gate": False,
            "only_integrity_or_execution_failures_block": True,
            "source": campaign["source"],
        }
    )


def build_final_continuation(
    *,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    campaign_root: str | Path,
    node_id: str,
    trigger_artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root = Path(campaign_root).resolve()
    _validate_rows(node_id=node_id, rows=rows)
    prerequisites = _load_prerequisites(
        node_id=node_id,
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
    )
    gate = build_final_continuation_gate(
        campaign=campaign,
        production_graph=production_graph,
        campaign_root=root,
        node_id=node_id,
        trigger_artifact=trigger_artifact,
        prerequisite_completions=prerequisites,
    )
    gate_path = (
        root
        / "job_ledgers"
        / "continuations"
        / "stage_n"
        / node_id
        / "gate.json"
    )
    dynamic = build_dynamic_continuation(
        campaign=campaign,
        production_graph=production_graph,
        selector_output=gate,
        selector_output_path=gate_path,
        downstream_node_id=node_id,
        rows=rows,
        campaign_root=root,
    )
    bundle = with_content_hash(
        {
            "contract": FINAL_CONTINUATION_BUNDLE_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign["content_hash"],
            "production_graph_sha256": production_graph["content_hash"],
            "node_id": node_id,
            "gate_sha256": gate["content_hash"],
            "continuation_intent_sha256": dynamic[
                "continuation_intent"
            ]["content_hash"],
            "task_manifest_sha256": dynamic["task_manifest"][
                "content_hash"
            ],
            "continuation_binding_sha256": dynamic[
                "continuation_binding"
            ]["content_hash"],
            "task_count": dynamic["task_manifest"]["task_count"],
            "performance_threshold_abort_allowed": False,
            "source": campaign["source"],
        }
    )
    return {
        "gate": gate,
        "dynamic_continuation": dynamic,
        "final_continuation_bundle": bundle,
    }


def validate_final_continuation(
    payload: Mapping[str, Any],
    *,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    campaign_root: str | Path,
    trigger_artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    if set(payload) != {
        "gate",
        "dynamic_continuation",
        "final_continuation_bundle",
    }:
        raise ValueError("Stage-N continuation bundle fields differ")
    bundle = payload["final_continuation_bundle"]
    digest = validate_content_hash(
        bundle, expected_contract=FINAL_CONTINUATION_BUNDLE_CONTRACT
    )
    expected = build_final_continuation(
        campaign=campaign,
        production_graph=production_graph,
        campaign_root=campaign_root,
        node_id=bundle["node_id"],
        trigger_artifact=trigger_artifact,
        rows=rows,
    )
    if dict(payload) != expected:
        raise ValueError("Stage-N continuation semantics differ")
    validate_dynamic_continuation(
        payload["dynamic_continuation"],
        campaign=campaign,
        production_graph=production_graph,
        selector_output=payload["gate"],
        selector_output_path=(
            Path(campaign_root)
            / "job_ledgers"
            / "continuations"
            / "stage_n"
            / bundle["node_id"]
            / "gate.json"
        ),
        rows=rows,
        campaign_root=campaign_root,
    )
    return digest


def publish_final_continuation(
    *, campaign_root: str | Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(campaign_root).resolve()
    node_id = payload["final_continuation_bundle"]["node_id"]
    gate_path = (
        root
        / "job_ledgers"
        / "continuations"
        / "stage_n"
        / node_id
        / "gate.json"
    )
    dynamic = payload["dynamic_continuation"]
    return {
        "gate": write_immutable_json(gate_path, payload["gate"]),
        "dynamic": publish_dynamic_continuation(
            bundle=dynamic,
            downstream_manifest_path=dynamic["continuation_binding"][
                "downstream_task_manifest_path"
            ],
            binding_path=dynamic["continuation_binding"][
                "continuation_binding_path"
            ],
        ),
        "bundle": write_immutable_json(
            gate_path.with_name("bundle.json"),
            payload["final_continuation_bundle"],
        ),
    }


__all__ = [
    "FINAL_CONTINUATION_BUNDLE_CONTRACT",
    "FINAL_WAVE_PREREQUISITES",
    "build_final_continuation",
    "build_final_continuation_gate",
    "publish_final_continuation",
    "validate_final_continuation",
]
