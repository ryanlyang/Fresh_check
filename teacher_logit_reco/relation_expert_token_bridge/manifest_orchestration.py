"""Automatic, completeness-gated production of downstream task manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    load_hashed_json,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
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
    FINAL_CONTINUATION_MANIFEST_NODES,
    LATE_CONTINUATION_MANIFEST_NODES,
    MIDDLE_CONTINUATION_MANIFEST_NODES,
    TASK_MANIFEST_PRODUCER_NODES,
    build_task_manifest,
    task_manifest_path_for_graph,
    validate_production_campaign_binding,
    validate_production_graph,
    validate_task_manifest_for_graph,
)


MANIFEST_MATERIALIZATION_PLAN_CONTRACT = (
    "retb_manifest_materialization_plan_v2"
)
MANIFEST_PRODUCER_RECEIPT_CONTRACT = "retb_manifest_producer_receipt_v2"


def manifest_plan_path(
    campaign_root: str | Path, *, target_node_id: str
) -> Path:
    return (
        Path(campaign_root).resolve()
        / "job_ledgers"
        / "manifest_plans"
        / f"{target_node_id}.json"
    )


def manifest_producer_receipt_path(
    campaign_root: str | Path, *, producer_node_id: str
) -> Path:
    return (
        Path(campaign_root).resolve()
        / "job_ledgers"
        / "manifest_producer_receipts"
        / f"{producer_node_id}.json"
    )


def producer_targets(producer_node_id: str) -> tuple[str, ...]:
    return tuple(
        target
        for target, producer in TASK_MANIFEST_PRODUCER_NODES.items()
        if producer == str(producer_node_id)
    )


def build_manifest_materialization_plan(
    *,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str,
    target_node_id: str,
    rows: Sequence[Mapping[str, Any]],
    trigger_artifact_path: str | Path | None = None,
    trigger_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    campaign_sha = validate_content_hash(campaign)
    graph_sha = validate_production_graph(production_graph)
    validate_production_campaign_binding(production_graph, campaign)
    target = str(target_node_id)
    producer = str(producer_node_id)
    if TASK_MANIFEST_PRODUCER_NODES.get(target) != producer:
        raise ValueError("manifest materialization producer assignment differs")
    nodes = {
        str(node["node_id"]): node for node in production_graph["nodes"]
    }
    node = nodes[target]
    factory_registry = production_graph["manifest_plan_factory_registry"]
    factory_entries = {
        str(entry["target_node_id"]): entry
        for entry in factory_registry["entries"]
    }
    if target not in factory_entries:
        raise ValueError(
            "downstream manifest plan lacks a registered plan factory"
        )
    factory = factory_entries[target]
    normalized_rows = [dict(row) for row in rows]
    if not normalized_rows:
        raise ValueError("manifest materialization plan rows are empty")
    dynamic = bool(node["dynamic_continuation"])
    if dynamic != (trigger_artifact_path is not None):
        raise ValueError(
            "dynamic manifest plan requires exactly one trigger artifact path"
        )
    if dynamic:
        trigger_path = Path(trigger_artifact_path).resolve()
        root = Path(production_graph["campaign_root"]).resolve()
        try:
            relative_trigger = str(trigger_path.relative_to(root))
        except ValueError as error:
            raise ValueError(
                "manifest trigger artifact escapes campaign root"
            ) from error
        if (
            not isinstance(trigger_artifact_sha256, str)
            or len(trigger_artifact_sha256) != 64
        ):
            raise ValueError("dynamic manifest trigger hash differs")
    else:
        relative_trigger = None
        if trigger_artifact_sha256 is not None:
            raise ValueError("static manifest plan declares a trigger hash")
    maximum = (
        1
        if node["array"] is None
        else int(node["array"]["maximum_concurrent_tasks"])
    )
    synthetic_rows = any(
        "scripts/write_retb_synthetic_output.py"
        in [str(value).replace("\\", "/") for value in row.get("argv", ())]
        or str(row.get("environment", {}).get(
            "RETB_SYNTHETIC_CONTROL_PLANE", ""
        ))
        == "1"
        for row in normalized_rows
    )
    if not synthetic_rows:
        allowed = set(factory["allowed_worker_entrypoints"])
        for row in normalized_rows:
            scripts = {
                str(value).replace("\\", "/")
                for value in row.get("argv", ())
                if str(value).replace("\\", "/").startswith("scripts/")
            }
            if len(scripts & allowed) != 1:
                raise ValueError(
                    f"{target} plan row worker is not registry-allowed"
                )
    return with_content_hash(
        {
            "contract": MANIFEST_MATERIALIZATION_PLAN_CONTRACT,
            "schema_version": 2,
            "campaign_spec_sha256": campaign_sha,
            "production_graph_sha256": graph_sha,
            "plan_factory_registry_sha256": factory_registry[
                "content_hash"
            ],
            "plan_factory_id": factory["factory_id"],
            "producer_node_id": producer,
            "target_node_id": target,
            "dynamic_continuation": dynamic,
            "trigger_artifact_path": relative_trigger,
            "trigger_artifact_sha256": trigger_artifact_sha256,
            "maximum_concurrent_tasks": maximum,
            "rows": normalized_rows,
            "row_count": len(normalized_rows),
            "synthetic_control_plane_rows": synthetic_rows,
            "genuine_scientific_worker_rows": not synthetic_rows,
            "scientific_performance_used_to_omit_rows": False,
            "source": campaign["source"],
        }
    )


def validate_manifest_materialization_plan(
    payload: Mapping[str, Any],
    *,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=MANIFEST_MATERIALIZATION_PLAN_CONTRACT
    )
    expected = build_manifest_materialization_plan(
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=payload["producer_node_id"],
        target_node_id=payload["target_node_id"],
        rows=payload["rows"],
        trigger_artifact_path=(
            None
            if payload["trigger_artifact_path"] is None
            else Path(production_graph["campaign_root"])
            / payload["trigger_artifact_path"]
        ),
        trigger_artifact_sha256=payload["trigger_artifact_sha256"],
    )
    if dict(payload) != expected:
        raise ValueError("manifest materialization plan semantics differ")
    return digest


def publish_manifest_materialization_plan(
    *,
    campaign_root: str | Path,
    plan: Mapping[str, Any],
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
) -> dict[str, str]:
    validate_manifest_materialization_plan(
        plan, campaign=campaign, production_graph=production_graph
    )
    return write_immutable_json(
        manifest_plan_path(
            campaign_root, target_node_id=plan["target_node_id"]
        ),
        plan,
    )


def _validate_existing_manifest(
    *,
    campaign_root: Path,
    repo_root: Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    target_node_id: str,
) -> dict[str, Any] | None:
    path = task_manifest_path_for_graph(
        production_graph,
        node_id=target_node_id,
        campaign_root=campaign_root,
    )
    if not path.is_file():
        return None
    manifest = load_hashed_json(path)
    validate_task_manifest_for_graph(
        manifest,
        production_graph=production_graph,
        campaign_root=campaign_root,
        repo_root=repo_root,
    )
    node = next(
        row
        for row in production_graph["nodes"]
        if row["node_id"] == target_node_id
    )
    if node["dynamic_continuation"]:
        validate_published_dynamic_continuation(
            campaign=campaign,
            production_graph=production_graph,
            task_manifest=manifest,
            campaign_root=campaign_root,
        )
    return manifest


def materialize_downstream_manifests(
    *,
    campaign_root: str | Path,
    repo_root: str | Path,
    producer_node_id: str,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish every manifest owned by one completed producer, exactly once."""

    root = Path(campaign_root).resolve()
    source_root = Path(repo_root).resolve()
    validate_production_campaign_binding(production_graph, campaign)
    targets = producer_targets(producer_node_id)
    publications: dict[str, Any] = {}
    manifest_hashes: dict[str, str] = {}
    plan_hashes: dict[str, str | None] = {}
    for target in targets:
        existing = _validate_existing_manifest(
            campaign_root=root,
            repo_root=source_root,
            campaign=campaign,
            production_graph=production_graph,
            target_node_id=target,
        )
        if existing is not None:
            publications[target] = {"status": "already_present"}
            manifest_hashes[target] = existing["content_hash"]
            existing_plan_path = manifest_plan_path(
                root, target_node_id=target
            )
            if existing_plan_path.is_file():
                existing_plan = load_hashed_json(
                    existing_plan_path,
                    expected_contract=MANIFEST_MATERIALIZATION_PLAN_CONTRACT,
                )
                validate_manifest_materialization_plan(
                    existing_plan,
                    campaign=campaign,
                    production_graph=production_graph,
                )
                plan_hashes[target] = existing_plan["content_hash"]
            else:
                plan_hashes[target] = None
            continue
        plan_path = manifest_plan_path(root, target_node_id=target)
        if not plan_path.is_file():
            raise FileNotFoundError(
                f"completed producer {producer_node_id} did not publish "
                f"the required manifest plan for {target}: {plan_path}"
            )
        plan = load_hashed_json(
            plan_path,
            expected_contract=MANIFEST_MATERIALIZATION_PLAN_CONTRACT,
        )
        validate_manifest_materialization_plan(
            plan, campaign=campaign, production_graph=production_graph
        )
        if plan["producer_node_id"] != producer_node_id:
            raise ValueError("manifest plan producer differs")
        rows = plan["rows"]
        trigger = None
        if plan["dynamic_continuation"]:
            trigger_path = root / plan["trigger_artifact_path"]
            trigger = load_hashed_json(trigger_path)
            if trigger["content_hash"] != plan["trigger_artifact_sha256"]:
                raise ValueError("manifest plan trigger artifact drifted")
        if target in MIDDLE_CONTINUATION_MANIFEST_NODES:
            payload = build_middle_continuation(
                campaign=campaign,
                production_graph=production_graph,
                campaign_root=root,
                node_id=target,
                trigger_artifact=trigger,
                rows=rows,
            )
            publications[target] = publish_middle_continuation(
                campaign_root=root, payload=payload
            )
            manifest = payload["dynamic_continuation"]["task_manifest"]
        elif target in LATE_CONTINUATION_MANIFEST_NODES:
            payload = build_late_continuation(
                campaign=campaign,
                production_graph=production_graph,
                campaign_root=root,
                node_id=target,
                trigger_artifact=trigger,
                rows=rows,
            )
            publications[target] = publish_late_continuation(
                campaign_root=root, payload=payload
            )
            manifest = payload["dynamic_continuation"]["task_manifest"]
        elif target in FINAL_CONTINUATION_MANIFEST_NODES:
            payload = build_final_continuation(
                campaign=campaign,
                production_graph=production_graph,
                campaign_root=root,
                node_id=target,
                trigger_artifact=trigger,
                rows=rows,
            )
            publications[target] = publish_final_continuation(
                campaign_root=root, payload=payload
            )
            manifest = payload["dynamic_continuation"]["task_manifest"]
        elif plan["dynamic_continuation"]:
            trigger_path = root / plan["trigger_artifact_path"]
            payload = build_dynamic_continuation(
                campaign=campaign,
                production_graph=production_graph,
                selector_output=trigger,
                selector_output_path=trigger_path,
                downstream_node_id=target,
                rows=rows,
                campaign_root=root,
            )
            publications[target] = publish_dynamic_continuation(
                bundle=payload,
                downstream_manifest_path=payload["continuation_binding"][
                    "downstream_task_manifest_path"
                ],
                binding_path=payload["continuation_binding"][
                    "continuation_binding_path"
                ],
            )
            manifest = payload["task_manifest"]
        else:
            manifest = build_task_manifest(
                campaign_spec_sha256=campaign["content_hash"],
                production_graph_sha256=production_graph["content_hash"],
                node_id=target,
                rows=rows,
                maximum_concurrent_tasks=plan[
                    "maximum_concurrent_tasks"
                ],
            )
            validate_task_manifest_for_graph(
                manifest,
                production_graph=production_graph,
                campaign_root=root,
                repo_root=source_root,
            )
            publications[target] = write_immutable_json(
                task_manifest_path_for_graph(
                    production_graph,
                    node_id=target,
                    campaign_root=root,
                ),
                manifest,
            )
        manifest_hashes[target] = manifest["content_hash"]
        plan_hashes[target] = plan["content_hash"]
    receipt = with_content_hash(
        {
            "contract": MANIFEST_PRODUCER_RECEIPT_CONTRACT,
            "schema_version": 2,
            "campaign_spec_sha256": campaign["content_hash"],
            "production_graph_sha256": production_graph["content_hash"],
            "producer_node_id": str(producer_node_id),
            "target_node_order": list(targets),
            "manifest_hashes": manifest_hashes,
            "materialization_plan_hashes": plan_hashes,
            "all_owned_manifests_present_and_valid": True,
            "row_execution_attestation_pending": True,
            "receipt_is_not_execution_completion_evidence": True,
            "scientific_performance_used_as_gate": False,
            "source": campaign["source"],
        }
    )
    receipt_publication = write_immutable_json(
        manifest_producer_receipt_path(
            root, producer_node_id=producer_node_id
        ),
        receipt,
    )
    return {
        "producer_node_id": str(producer_node_id),
        "target_count": len(targets),
        "manifest_hashes": manifest_hashes,
        "publications": publications,
        "receipt": receipt,
        "receipt_publication": receipt_publication,
    }


__all__ = [
    "MANIFEST_MATERIALIZATION_PLAN_CONTRACT",
    "MANIFEST_PRODUCER_RECEIPT_CONTRACT",
    "build_manifest_materialization_plan",
    "manifest_plan_path",
    "manifest_producer_receipt_path",
    "materialize_downstream_manifests",
    "producer_targets",
    "publish_manifest_materialization_plan",
    "validate_manifest_materialization_plan",
]
