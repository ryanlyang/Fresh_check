"""Authenticated completion records for singleton direct-worker nodes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .production import validate_production_campaign_binding


DIRECT_NODE_COMPLETION_CONTRACT = "retb_direct_node_completion_v1"


def direct_node_completion_path(
    campaign_root: str | Path, *, node_id: str
) -> Path:
    return (
        Path(campaign_root).resolve()
        / "job_ledgers"
        / "direct_completions"
        / f"{node_id}.json"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_direct_node_completion(
    *,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    campaign_root: str | Path,
    node_id: str,
    output_paths: Sequence[str | Path],
) -> dict[str, Any]:
    root = Path(campaign_root).resolve()
    campaign_sha = validate_content_hash(campaign)
    graph_sha = validate_production_campaign_binding(
        production_graph, campaign
    )
    nodes = {
        str(row["node_id"]): row for row in production_graph["nodes"]
    }
    node = nodes.get(str(node_id))
    execution = {
        str(row["node_id"]): row
        for row in production_graph["node_execution_registry"]["entries"]
    }.get(str(node_id))
    if (
        node is None
        or execution is None
        or execution["dispatch_mode"] != "direct_worker"
    ):
        raise ValueError("direct completion node is not a direct worker")
    hashes: dict[str, str] = {}
    for value in output_paths:
        path = Path(value).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                "direct completion output escapes campaign root"
            ) from error
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"direct completion output is absent or unsafe: {path}"
            )
        hashes[str(path)] = _file_sha256(path)
    if not hashes:
        raise ValueError("direct completion has no authenticated outputs")
    return with_content_hash(
        {
            "contract": DIRECT_NODE_COMPLETION_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign_sha,
            "production_graph_sha256": graph_sha,
            "node_id": str(node_id),
            "output_hashes": hashes,
            "output_count": len(hashes),
            "all_outputs_revalidated_after_worker_exit": True,
            "scientific_performance_used_as_completion_gate": False,
            "source": campaign["source"],
        }
    )


def validate_direct_node_completion(
    payload: Mapping[str, Any],
    *,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    campaign_root: str | Path,
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=DIRECT_NODE_COMPLETION_CONTRACT
    )
    expected = build_direct_node_completion(
        campaign=campaign,
        production_graph=production_graph,
        campaign_root=campaign_root,
        node_id=str(payload.get("node_id", "")),
        output_paths=list(payload.get("output_hashes", {})),
    )
    if (
        dict(payload) != expected
        or int(payload.get("output_count", -1))
        != len(payload.get("output_hashes", {}))
        or payload.get("campaign_spec_sha256")
        != require_sha256(
            campaign["content_hash"], name="campaign_spec_sha256"
        )
    ):
        raise ValueError("direct node completion semantics differ")
    return digest


def publish_direct_node_completion(
    *,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    campaign_root: str | Path,
    node_id: str,
    output_paths: Sequence[str | Path],
) -> dict[str, Any]:
    payload = build_direct_node_completion(
        campaign=campaign,
        production_graph=production_graph,
        campaign_root=campaign_root,
        node_id=node_id,
        output_paths=output_paths,
    )
    validate_direct_node_completion(
        payload,
        campaign=campaign,
        production_graph=production_graph,
        campaign_root=campaign_root,
    )
    return write_immutable_json(
        direct_node_completion_path(
            campaign_root, node_id=node_id
        ),
        payload,
    )


__all__ = [
    "DIRECT_NODE_COMPLETION_CONTRACT",
    "build_direct_node_completion",
    "direct_node_completion_path",
    "publish_direct_node_completion",
    "validate_direct_node_completion",
]
