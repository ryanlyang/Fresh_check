"""Completeness-gated dynamic continuation for RETB Stages F--J."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .dynamic_continuation import (
    build_dynamic_continuation,
    publish_dynamic_continuation,
    validate_dynamic_continuation,
    validate_published_dynamic_continuation,
)
from .production import (
    MIDDLE_CONTINUATION_GATE_CONTRACT,
    MIDDLE_CONTINUATION_MANIFEST_NODES,
    MIDDLE_NODE_ENTRYPOINTS,
    task_manifest_path_for_graph,
    validate_production_campaign_binding,
    validate_production_graph,
    validate_task_manifest_for_graph,
)
from .task_completion import (
    task_manifest_completion_path,
    validate_task_manifest_completion,
)


MIDDLE_CONTINUATION_BUNDLE_CONTRACT = (
    "retb_stage_f_j_continuation_bundle_v1"
)

MIDDLE_WAVE_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "target_cache_build": (),
    "target_normalizers": ("target_cache_build",),
    "predictor_training": ("target_normalizers",),
    "uncertainty_calibration": ("predictor_training",),
    "predictor_bundle_selector": ("uncertainty_calibration",),
    "oracle_substitutions": ("predictor_bundle_selector",),
    "joint_predictor_training": ("predictor_bundle_selector",),
    "joint_predictor_selector": (
        "oracle_substitutions",
        "joint_predictor_training",
    ),
    "final_consumer_training": ("joint_predictor_selector",),
    "deployable_export": ("final_consumer_training",),
}

if set(MIDDLE_WAVE_PREREQUISITES) != set(
    MIDDLE_CONTINUATION_MANIFEST_NODES
):
    raise RuntimeError("Stage F--J continuation coverage differs")
if set(MIDDLE_NODE_ENTRYPOINTS) != set(MIDDLE_WAVE_PREREQUISITES):
    raise RuntimeError("Stage F--J entry-point coverage differs")


def _validate_middle_rows(
    *, node_id: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    expected_entrypoint = MIDDLE_NODE_ENTRYPOINTS[node_id]
    if not rows:
        raise ValueError("Stage F--J continuation rows are empty")
    for row in rows:
        argv = [str(value) for value in row.get("argv", ())]
        if len(argv) < 2 or argv[1].replace("\\", "/") != expected_entrypoint:
            raise ValueError(
                f"{node_id} row entry point differs from "
                f"{expected_entrypoint}"
            )
        if "--dry-run" in argv:
            raise ValueError(
                "production Stage F--J rows may not request dry-run"
            )


def _load_prerequisite_completions(
    *,
    node_id: str,
    campaign_root: Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    completions = {}
    for parent in MIDDLE_WAVE_PREREQUISITES[node_id]:
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
                f"Stage F--J prerequisite is incomplete: {parent}"
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
            row for row in production_graph["nodes"]
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
        completions[parent] = {
            "task_manifest": manifest,
            "completion": completion,
        }
    return completions


def build_middle_continuation_gate(
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
    if node_id not in MIDDLE_WAVE_PREREQUISITES:
        raise ValueError("Stage F--J continuation node is unregistered")
    trigger_sha = validate_content_hash(trigger_artifact)
    if trigger_artifact.get("source") != campaign["source"]:
        raise ValueError("Stage F--J trigger source differs")
    required = MIDDLE_WAVE_PREREQUISITES[node_id]
    if set(prerequisite_completions) != set(required):
        raise ValueError("Stage F--J prerequisite coverage differs")
    completion_hashes: dict[str, str] = {}
    output_hashes: dict[str, dict[str, str]] = {}
    for parent in required:
        pair = prerequisite_completions[parent]
        if set(pair) != {"task_manifest", "completion"}:
            raise ValueError("Stage F--J prerequisite pair differs")
        manifest = pair["task_manifest"]
        completion = pair["completion"]
        completion_sha = validate_task_manifest_completion(
            completion,
            campaign_root=campaign_root,
            campaign=campaign,
            task_manifest=manifest,
        )
        if manifest["node_id"] != parent:
            raise ValueError("Stage F--J prerequisite node differs")
        completion_hashes[parent] = completion_sha
        output_hashes[parent] = {
            path: digest
            for row in completion["rows"]
            for path, digest in row["output_hashes"].items()
        }
    return with_content_hash(
        {
            "contract": MIDDLE_CONTINUATION_GATE_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign_sha,
            "production_graph_sha256": graph_sha,
            "node_id": str(node_id),
            "stage": next(
                row["stage"]
                for row in production_graph["nodes"]
                if row["node_id"] == node_id
            ),
            "trigger_contract": str(trigger_artifact["contract"]),
            "trigger_artifact_sha256": trigger_sha,
            "prerequisite_node_order": list(required),
            "prerequisite_completion_hashes": completion_hashes,
            "prerequisite_output_hashes": output_hashes,
            "all_prerequisite_rows_and_outputs_revalidated": True,
            "scientific_performance_used_as_gate": False,
            "source": campaign["source"],
        }
    )


def build_middle_continuation(
    *,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    campaign_root: str | Path,
    node_id: str,
    trigger_artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root = Path(campaign_root).resolve()
    if node_id not in MIDDLE_WAVE_PREREQUISITES:
        raise ValueError("Stage F--J continuation node is unregistered")
    _validate_middle_rows(node_id=node_id, rows=rows)
    prerequisites = _load_prerequisite_completions(
        node_id=node_id,
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
    )
    gate = build_middle_continuation_gate(
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
        / "stage_f_j"
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
            "contract": MIDDLE_CONTINUATION_BUNDLE_CONTRACT,
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
            "prerequisite_completion_hashes": gate[
                "prerequisite_completion_hashes"
            ],
            "all_rows_run_even_if_metrics_are_negative": True,
            "source": campaign["source"],
        }
    )
    return {
        "gate": gate,
        "dynamic_continuation": dynamic,
        "middle_continuation_bundle": bundle,
    }


def validate_middle_continuation(
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
        "middle_continuation_bundle",
    }:
        raise ValueError("Stage F--J continuation bundle fields differ")
    bundle = payload["middle_continuation_bundle"]
    digest = validate_content_hash(
        bundle, expected_contract=MIDDLE_CONTINUATION_BUNDLE_CONTRACT
    )
    node_id = str(bundle["node_id"])
    expected = build_middle_continuation(
        campaign=campaign,
        production_graph=production_graph,
        campaign_root=campaign_root,
        node_id=node_id,
        trigger_artifact=trigger_artifact,
        rows=rows,
    )
    if dict(payload) != expected:
        raise ValueError("Stage F--J continuation semantics differ")
    validate_dynamic_continuation(
        payload["dynamic_continuation"],
        campaign=campaign,
        production_graph=production_graph,
        selector_output=payload["gate"],
        selector_output_path=(
            Path(campaign_root)
            / "job_ledgers"
            / "continuations"
            / "stage_f_j"
            / node_id
            / "gate.json"
        ),
        rows=rows,
        campaign_root=campaign_root,
    )
    return digest


def publish_middle_continuation(
    *,
    campaign_root: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(campaign_root).resolve()
    node_id = str(payload["middle_continuation_bundle"]["node_id"])
    gate_path = (
        root
        / "job_ledgers"
        / "continuations"
        / "stage_f_j"
        / node_id
        / "gate.json"
    )
    bundle_path = gate_path.with_name("bundle.json")
    dynamic = payload["dynamic_continuation"]
    binding_path = Path(
        dynamic["continuation_binding"]["continuation_binding_path"]
    )
    manifest_path = Path(
        dynamic["continuation_binding"][
            "downstream_task_manifest_path"
        ]
    )
    publications = {
        "gate": write_immutable_json(gate_path, payload["gate"]),
        "dynamic": publish_dynamic_continuation(
            bundle=dynamic,
            downstream_manifest_path=manifest_path,
            binding_path=binding_path,
        ),
        "bundle": write_immutable_json(
            bundle_path, payload["middle_continuation_bundle"]
        ),
    }
    return publications


__all__ = [
    "MIDDLE_CONTINUATION_BUNDLE_CONTRACT",
    "MIDDLE_CONTINUATION_GATE_CONTRACT",
    "MIDDLE_NODE_ENTRYPOINTS",
    "MIDDLE_WAVE_PREREQUISITES",
    "build_middle_continuation",
    "build_middle_continuation_gate",
    "publish_middle_continuation",
    "validate_middle_continuation",
]
