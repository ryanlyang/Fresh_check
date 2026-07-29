"""Authenticated selector-to-task-manifest continuation.

A selector result and its downstream task manifest cannot directly contain
each other's final content hash without creating a hash cycle.  RETB seals the
pair with a third immutable binding artifact:

* every task row binds the selector result and a deterministic continuation
  intent;
* the binding artifact binds the selector result, intent, and completed task
  manifest.

The three artifacts are therefore inseparable without requiring a cyclic
serialization.  Publication is restart-safe: an interrupted writer may leave
the selection or task manifest behind, but a continuation is executable only
after the binding artifact validates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .production import (
    PRODUCTION_GRAPH_CONTRACT,
    build_task_manifest,
    validate_production_graph,
    validate_production_campaign_binding,
    validate_task_manifest_for_graph,
)


DYNAMIC_CONTINUATION_INTENT_CONTRACT = (
    "retb_dynamic_continuation_intent_v1"
)
DYNAMIC_CONTINUATION_BINDING_CONTRACT = (
    "retb_dynamic_continuation_binding_v1"
)

_REQUIRED_ROW_FIELDS = {
    "task_id",
    "argv",
    "environment",
    "expected_outputs",
    "input_artifact_hashes",
}


def add_dynamic_continuation_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add the common optional selector-continuation CLI surface."""

    parser.add_argument(
        "--downstream-node-id",
        help=(
            "Graph node whose authenticated task manifest is published by "
            "this selector"
        ),
    )
    parser.add_argument(
        "--downstream-rows-json",
        type=Path,
        help=(
            "JSON array of resolved task rows; required with "
            "--downstream-node-id"
        ),
    )
    parser.add_argument(
        "--production-graph",
        type=Path,
        help=(
            "Defaults to CAMPAIGN_ROOT/job_ledgers/production_graph.json "
            "when continuation is requested"
        ),
    )
    parser.add_argument(
        "--downstream-manifest",
        type=Path,
        help=(
            "Defaults to the graph-declared task-manifest path below the "
            "campaign root"
        ),
    )
    parser.add_argument(
        "--continuation-binding",
        type=Path,
        help=(
            "Defaults to selection/continuations/"
            "SELECTOR_to_DOWNSTREAM.json"
        ),
    )


def continuation_requested(args: argparse.Namespace) -> bool:
    values = (
        getattr(args, "downstream_node_id", None),
        getattr(args, "downstream_rows_json", None),
        getattr(args, "production_graph", None),
        getattr(args, "downstream_manifest", None),
        getattr(args, "continuation_binding", None),
    )
    return any(value is not None for value in values)


def _normalize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    downstream_node_id: str,
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("dynamic continuation rows are empty")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if set(raw) != _REQUIRED_ROW_FIELDS:
            raise ValueError("dynamic continuation row fields differ")
        row = {
            "task_id": f"{downstream_node_id}:{index}",
            "argv": [str(value) for value in raw["argv"]],
            "environment": {
                str(key): str(value)
                for key, value in sorted(raw["environment"].items())
            },
            "expected_outputs": [
                str(value) for value in raw["expected_outputs"]
            ],
            "input_artifact_hashes": {
                str(key): require_sha256(
                    value, name=f"input_artifact_hashes.{key}"
                )
                for key, value in sorted(
                    raw["input_artifact_hashes"].items()
                )
            },
        }
        normalized.append(row)
    return normalized


def _graph_node(
    production_graph: Mapping[str, Any],
    node_id: str,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in production_graph["nodes"]
        if row["node_id"] == node_id
    ]
    if len(matches) != 1:
        raise ValueError("dynamic continuation downstream node differs")
    node = matches[0]
    execution = {
        row["node_id"]: row
        for row in production_graph["node_execution_registry"]["entries"]
    }.get(node_id)
    if (
        execution is None
        or execution["manifest_required"] is not True
        or execution["row_resolution"] != "dynamic"
    ):
        raise ValueError(
            "dynamic continuation requires a dynamic manifest-driven node"
        )
    return node


def _manifest_path(
    *,
    campaign_root: Path,
    node: Mapping[str, Any],
) -> Path:
    if node["array"] is not None:
        relative = Path(node["array"]["task_manifest"])
    else:
        relative = Path("job_ledgers") / "tasks" / f"{node['node_id']}.json"
    return campaign_root / relative


def build_dynamic_continuation(
    *,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    selector_output: Mapping[str, Any],
    selector_output_path: str | Path,
    downstream_node_id: str,
    rows: Sequence[Mapping[str, Any]],
    campaign_root: str | Path,
    downstream_manifest_path: str | Path | None = None,
    binding_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build and validate a selector, task-manifest, and binding triple."""

    campaign_sha = validate_content_hash(campaign)
    graph_sha = validate_production_graph(production_graph)
    validate_production_campaign_binding(production_graph, campaign)
    selector_sha = validate_content_hash(selector_output)
    if (
        production_graph["campaign_id"] != campaign["campaign_id"]
        or production_graph["campaign_root"]
        != str(Path(campaign_root).resolve())
    ):
        raise ValueError("dynamic continuation campaign binding differs")
    root = Path(campaign_root).resolve()
    selection_path = Path(selector_output_path).resolve()
    try:
        selection_path.relative_to(root)
    except ValueError as error:
        raise ValueError("selector output escapes the campaign root") from error
    node = _graph_node(production_graph, downstream_node_id)
    normalized = _normalize_rows(
        rows, downstream_node_id=downstream_node_id
    )
    maximum = (
        1
        if node["array"] is None
        else int(node["array"]["maximum_tasks"])
    )
    concurrency = (
        1
        if node["array"] is None
        else int(node["array"]["maximum_concurrent_tasks"])
    )
    if len(normalized) > maximum:
        raise ValueError("dynamic continuation exceeds graph task maximum")
    row_payload_sha = canonical_sha256(normalized)
    intent = with_content_hash(
        {
            "contract": DYNAMIC_CONTINUATION_INTENT_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign_sha,
            "production_graph_sha256": graph_sha,
            "selector_contract": str(selector_output["contract"]),
            "selector_output_sha256": selector_sha,
            "selector_output_path": str(selection_path),
            "downstream_node_id": str(downstream_node_id),
            "resolved_row_payload_sha256": row_payload_sha,
            "task_count": len(normalized),
            "maximum_tasks": maximum,
            "scientific_underperformance_skips_rows": False,
        }
    )
    task_rows = []
    for row in normalized:
        hashes = dict(row["input_artifact_hashes"])
        reserved = {"selector_output", "continuation_intent"}
        if reserved.intersection(hashes):
            raise ValueError(
                "dynamic continuation row uses a reserved lineage name"
            )
        task_rows.append(
            {
                **row,
                "input_artifact_hashes": {
                    **hashes,
                    "selector_output": selector_sha,
                    "continuation_intent": intent["content_hash"],
                },
            }
        )
    manifest = build_task_manifest(
        campaign_spec_sha256=campaign_sha,
        production_graph_sha256=graph_sha,
        node_id=downstream_node_id,
        rows=task_rows,
        maximum_concurrent_tasks=concurrency,
    )
    validate_task_manifest_for_graph(
        manifest,
        production_graph=production_graph,
        campaign_root=root,
        repo_root=Path(__file__).resolve().parents[2],
    )
    manifest_path = (
        _manifest_path(campaign_root=root, node=node)
        if downstream_manifest_path is None
        else Path(downstream_manifest_path).resolve()
    )
    expected_manifest_path = _manifest_path(
        campaign_root=root, node=node
    ).resolve()
    if manifest_path != expected_manifest_path:
        raise ValueError(
            "dynamic continuation manifest path differs from graph"
        )
    default_binding = (
        root
        / "selection"
        / "continuations"
        / (
            f"{selection_path.stem}_to_{downstream_node_id}.json"
        )
    )
    resolved_binding = (
        default_binding
        if binding_path is None
        else Path(binding_path).resolve()
    )
    try:
        resolved_binding.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "dynamic continuation binding escapes campaign root"
        ) from error
    binding = with_content_hash(
        {
            "contract": DYNAMIC_CONTINUATION_BINDING_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign_sha,
            "production_graph_sha256": graph_sha,
            "selector_contract": str(selector_output["contract"]),
            "selector_output_sha256": selector_sha,
            "selector_output_path": str(selection_path),
            "continuation_intent": intent,
            "continuation_intent_sha256": intent["content_hash"],
            "downstream_node_id": str(downstream_node_id),
            "downstream_task_manifest_sha256": manifest["content_hash"],
            "downstream_task_manifest_path": str(manifest_path),
            "continuation_binding_path": str(resolved_binding),
            "task_count": int(manifest["task_count"]),
            "resolved_row_payload_sha256": row_payload_sha,
            "source": campaign["source"],
            "publication_complete": True,
            "executable_only_when_binding_validates": True,
            "scientific_underperformance_skips_rows": False,
        }
    )
    return {
        "continuation_intent": intent,
        "task_manifest": manifest,
        "continuation_binding": binding,
    }


def validate_dynamic_continuation(
    bundle: Mapping[str, Mapping[str, Any]],
    *,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    selector_output: Mapping[str, Any],
    selector_output_path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    campaign_root: str | Path,
) -> str:
    """Rebuild a continuation and reject any selection/row/path drift."""

    required = {
        "continuation_intent",
        "task_manifest",
        "continuation_binding",
    }
    if set(bundle) != required:
        raise ValueError("dynamic continuation bundle fields differ")
    binding = bundle["continuation_binding"]
    validate_content_hash(
        binding, expected_contract=DYNAMIC_CONTINUATION_BINDING_CONTRACT
    )
    expected = build_dynamic_continuation(
        campaign=campaign,
        production_graph=production_graph,
        selector_output=selector_output,
        selector_output_path=selector_output_path,
        downstream_node_id=str(binding["downstream_node_id"]),
        rows=rows,
        campaign_root=campaign_root,
        downstream_manifest_path=binding[
            "downstream_task_manifest_path"
        ],
        binding_path=(
            Path(campaign_root)
            / "selection"
            / "continuations"
            / (
                f"{Path(selector_output_path).stem}_to_"
                f"{binding['downstream_node_id']}.json"
            )
        ),
    )
    if dict(bundle) != expected:
        raise ValueError("dynamic continuation semantics differ")
    return str(binding["content_hash"])


def validate_published_dynamic_continuation(
    *,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
    campaign_root: str | Path,
) -> str:
    """Validate the unique binding receipt required to execute a manifest."""

    campaign_sha = validate_content_hash(campaign)
    graph_sha = validate_production_graph(production_graph)
    validate_production_campaign_binding(production_graph, campaign)
    manifest_sha = validate_content_hash(task_manifest)
    root = Path(campaign_root).resolve()
    validate_task_manifest_for_graph(
        task_manifest,
        production_graph=production_graph,
        campaign_root=root,
        repo_root=Path(__file__).resolve().parents[2],
    )
    binding_root = root / "selection" / "continuations"
    candidates: list[tuple[Path, dict[str, Any]]] = []
    if binding_root.is_dir():
        for path in sorted(binding_root.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("contract")
                == DYNAMIC_CONTINUATION_BINDING_CONTRACT
                and payload.get("downstream_task_manifest_sha256")
                == manifest_sha
            ):
                validate_content_hash(
                    payload,
                    expected_contract=DYNAMIC_CONTINUATION_BINDING_CONTRACT,
                )
                candidates.append((path.resolve(), payload))
    if len(candidates) != 1:
        raise ValueError(
            "dynamic task manifest lacks one unique continuation binding"
        )
    binding_file, binding = candidates[0]
    intent = binding.get("continuation_intent")
    if not isinstance(intent, Mapping):
        raise ValueError("dynamic continuation intent is absent")
    intent_sha = validate_content_hash(
        intent, expected_contract=DYNAMIC_CONTINUATION_INTENT_CONTRACT
    )
    selection_path = Path(binding["selector_output_path"]).resolve()
    try:
        selection_path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "published selector output escapes campaign root"
        ) from error
    selector = json.loads(selection_path.read_text(encoding="utf-8"))
    selector_sha = validate_content_hash(selector)
    manifest_path = Path(
        binding["downstream_task_manifest_path"]
    ).resolve()
    if (
        campaign_sha != binding["campaign_spec_sha256"]
        or graph_sha != binding["production_graph_sha256"]
        or task_manifest["node_id"] != binding["downstream_node_id"]
        or intent_sha != binding["continuation_intent_sha256"]
            or selector_sha != binding["selector_output_sha256"]
            or binding_file
            != Path(binding["continuation_binding_path"]).resolve()
        or manifest_path.is_file() is not True
        or canonical_sha256(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        != canonical_sha256(task_manifest)
    ):
        raise ValueError("published dynamic continuation lineage differs")
    for row in task_manifest["rows"]:
        hashes = row["input_artifact_hashes"]
        if (
            hashes.get("selector_output") != selector_sha
            or hashes.get("continuation_intent") != intent_sha
        ):
            raise ValueError(
                "dynamic task row lacks its selector/intent binding"
            )
    return str(binding["content_hash"])


def load_continuation_rows(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("dynamic continuation rows JSON must be an array")
    return [dict(row) for row in raw]


def publish_dynamic_continuation(
    *,
    bundle: Mapping[str, Mapping[str, Any]],
    downstream_manifest_path: str | Path,
    binding_path: str | Path,
) -> dict[str, Any]:
    """Publish the manifest before its final executable binding receipt."""

    validate_content_hash(
        bundle["continuation_intent"],
        expected_contract=DYNAMIC_CONTINUATION_INTENT_CONTRACT,
    )
    validate_content_hash(bundle["task_manifest"])
    validate_content_hash(
        bundle["continuation_binding"],
        expected_contract=DYNAMIC_CONTINUATION_BINDING_CONTRACT,
    )
    manifest_path = Path(downstream_manifest_path).resolve()
    resolved_binding_path = Path(binding_path).resolve()
    if (
        bundle["continuation_binding"][
            "downstream_task_manifest_sha256"
        ]
        != bundle["task_manifest"]["content_hash"]
        or bundle["continuation_binding"]["continuation_intent_sha256"]
        != bundle["continuation_intent"]["content_hash"]
        or manifest_path
        != Path(
            bundle["continuation_binding"][
                "downstream_task_manifest_path"
            ]
        ).resolve()
        or resolved_binding_path
        != Path(
            bundle["continuation_binding"]["continuation_binding_path"]
        ).resolve()
    ):
        raise ValueError("dynamic continuation binding hashes differ")
    return {
        "task_manifest": write_immutable_json(
            manifest_path, bundle["task_manifest"]
        ),
        "continuation_binding": write_immutable_json(
            resolved_binding_path, bundle["continuation_binding"]
        ),
    }


def resolve_selector_continuation(
    *,
    args: argparse.Namespace,
    campaign: Mapping[str, Any],
    campaign_root: str | Path,
    selector_output: Mapping[str, Any],
    selector_output_path: str | Path,
    load_hashed_json: Any,
    dry_run: bool,
) -> dict[str, Any] | None:
    """Resolve the common CLI flags and publish the sealed continuation."""

    if not continuation_requested(args):
        return None
    if (
        args.downstream_node_id is None
        or args.downstream_rows_json is None
    ):
        raise ValueError(
            "continuation requires --downstream-node-id and "
            "--downstream-rows-json"
        )
    root = Path(campaign_root).resolve()
    graph_path = args.production_graph or (
        root / "job_ledgers" / "production_graph.json"
    )
    graph = load_hashed_json(
        graph_path, expected_contract=PRODUCTION_GRAPH_CONTRACT
    )
    rows = load_continuation_rows(args.downstream_rows_json)
    node = _graph_node(graph, args.downstream_node_id)
    manifest_path = args.downstream_manifest or _manifest_path(
        campaign_root=root, node=node
    )
    binding_path = args.continuation_binding or (
        root
        / "selection"
        / "continuations"
        / (
            f"{Path(selector_output_path).stem}_to_"
            f"{args.downstream_node_id}.json"
        )
    )
    bundle = build_dynamic_continuation(
        campaign=campaign,
        production_graph=graph,
        selector_output=selector_output,
        selector_output_path=selector_output_path,
        downstream_node_id=args.downstream_node_id,
        rows=rows,
        campaign_root=root,
        downstream_manifest_path=manifest_path,
        binding_path=binding_path,
    )
    validate_dynamic_continuation(
        bundle,
        campaign=campaign,
        production_graph=graph,
        selector_output=selector_output,
        selector_output_path=selector_output_path,
        rows=rows,
        campaign_root=root,
    )
    result: dict[str, Any] = {
        "downstream_node_id": args.downstream_node_id,
        "task_count": bundle["task_manifest"]["task_count"],
        "task_manifest_sha256": bundle["task_manifest"]["content_hash"],
        "continuation_binding_sha256": bundle["continuation_binding"][
            "content_hash"
        ],
        "downstream_manifest": str(Path(manifest_path).resolve()),
        "continuation_binding": str(Path(binding_path).resolve()),
    }
    if not dry_run:
        result["publication"] = publish_dynamic_continuation(
            bundle=bundle,
            downstream_manifest_path=manifest_path,
            binding_path=binding_path,
        )
    return result


__all__ = [
    "DYNAMIC_CONTINUATION_BINDING_CONTRACT",
    "DYNAMIC_CONTINUATION_INTENT_CONTRACT",
    "add_dynamic_continuation_arguments",
    "build_dynamic_continuation",
    "continuation_requested",
    "load_continuation_rows",
    "publish_dynamic_continuation",
    "resolve_selector_continuation",
    "validate_dynamic_continuation",
    "validate_published_dynamic_continuation",
]
