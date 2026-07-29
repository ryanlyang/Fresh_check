"""Authenticated task-row reuse and complete-manifest attestations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .production import validate_task_manifest


TASK_ROW_COMPLETION_CONTRACT = "retb_task_row_completion_v1"
TASK_MANIFEST_COMPLETION_CONTRACT = "retb_task_manifest_completion_v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_row_completion_path(
    campaign_root: str | Path,
    *,
    node_id: str,
    task_index: int,
) -> Path:
    return (
        Path(campaign_root)
        / "job_ledgers"
        / "completions"
        / str(node_id)
        / f"row_{int(task_index):06d}.json"
    )


def task_manifest_completion_path(
    campaign_root: str | Path,
    *,
    node_id: str,
) -> Path:
    return (
        Path(campaign_root)
        / "job_ledgers"
        / "completions"
        / str(node_id)
        / "manifest_completion.json"
    )


def _output_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(
            f"task expected output is absent or a symlink: {resolved}"
        )
    record: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "file_sha256": _file_sha256(resolved),
        "json_contract": None,
        "json_content_hash": None,
    }
    if resolved.suffix.lower() == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        record["json_content_hash"] = validate_content_hash(payload)
        record["json_contract"] = str(payload["contract"])
    return record


def build_task_row_completion(
    *,
    campaign: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
    task_index: int,
) -> dict[str, Any]:
    campaign_sha = validate_content_hash(campaign)
    manifest_sha = validate_task_manifest(task_manifest)
    index = int(task_index)
    if index < 0 or index >= int(task_manifest["task_count"]):
        raise IndexError("task completion index is outside the manifest")
    row = task_manifest["rows"][index]
    if (
        row["task_index"] != index
        or row["task_id"] != f"{task_manifest['node_id']}:{index}"
        or task_manifest["campaign_spec_sha256"] != campaign_sha
    ):
        raise ValueError("task completion row identity differs")
    outputs = [_output_record(path) for path in row["expected_outputs"]]
    return with_content_hash(
        {
            "contract": TASK_ROW_COMPLETION_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign_sha,
            "task_manifest_sha256": manifest_sha,
            "node_id": str(task_manifest["node_id"]),
            "task_index": index,
            "task_id": str(row["task_id"]),
            "argv": list(row["argv"]),
            "environment": dict(row["environment"]),
            "input_artifact_hashes": dict(
                row["input_artifact_hashes"]
            ),
            "outputs": outputs,
            "output_count": len(outputs),
            "all_expected_outputs_authenticated": True,
            "scientific_performance_evaluated_for_completion": False,
            "source": campaign["source"],
        }
    )


def validate_task_row_completion(
    payload: Mapping[str, Any],
    *,
    campaign: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
    task_index: int,
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=TASK_ROW_COMPLETION_CONTRACT
    )
    expected = build_task_row_completion(
        campaign=campaign,
        task_manifest=task_manifest,
        task_index=task_index,
    )
    if dict(payload) != expected:
        raise ValueError("task row completion semantics differ")
    return digest


def publish_task_row_completion(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
    task_index: int,
) -> dict[str, Any]:
    artifact = build_task_row_completion(
        campaign=campaign,
        task_manifest=task_manifest,
        task_index=task_index,
    )
    path = task_row_completion_path(
        campaign_root,
        node_id=str(task_manifest["node_id"]),
        task_index=task_index,
    )
    publication = write_immutable_json(path, artifact)
    return {
        "artifact": artifact,
        "path": str(path.resolve()),
        "publication": publication,
    }


def reusable_task_row_completion(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
    task_index: int,
) -> dict[str, Any] | None:
    path = task_row_completion_path(
        campaign_root,
        node_id=str(task_manifest["node_id"]),
        task_index=task_index,
    )
    if not path.exists():
        return None
    if not path.is_file() or path.is_symlink():
        raise ValueError("task row completion path is unsafe")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_task_row_completion(
        payload,
        campaign=campaign,
        task_manifest=task_manifest,
        task_index=task_index,
    )
    return payload


def build_task_manifest_completion(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    campaign_sha = validate_content_hash(campaign)
    manifest_sha = validate_task_manifest(task_manifest)
    rows = []
    for index in range(int(task_manifest["task_count"])):
        path = task_row_completion_path(
            campaign_root,
            node_id=str(task_manifest["node_id"]),
            task_index=index,
        )
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"task row completion is absent: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        digest = validate_task_row_completion(
            payload,
            campaign=campaign,
            task_manifest=task_manifest,
            task_index=index,
        )
        rows.append(
            {
                "task_index": index,
                "task_id": payload["task_id"],
                "row_completion_sha256": digest,
                "output_hashes": {
                    output["path"]: output["file_sha256"]
                    for output in payload["outputs"]
                },
            }
        )
    return with_content_hash(
        {
            "contract": TASK_MANIFEST_COMPLETION_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign_sha,
            "task_manifest_sha256": manifest_sha,
            "node_id": str(task_manifest["node_id"]),
            "task_count": int(task_manifest["task_count"]),
            "completed_task_count": len(rows),
            "rows": rows,
            "task_indices_complete_and_contiguous": True,
            "all_outputs_revalidated_after_last_row": True,
            "scientific_underperformance_blocks_completion": False,
            "source": campaign["source"],
        }
    )


def validate_task_manifest_completion(
    payload: Mapping[str, Any],
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=TASK_MANIFEST_COMPLETION_CONTRACT
    )
    expected = build_task_manifest_completion(
        campaign_root=campaign_root,
        campaign=campaign,
        task_manifest=task_manifest,
    )
    if dict(payload) != expected:
        raise ValueError("task manifest completion semantics differ")
    return digest


def publish_task_manifest_completion(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = build_task_manifest_completion(
        campaign_root=campaign_root,
        campaign=campaign,
        task_manifest=task_manifest,
    )
    path = task_manifest_completion_path(
        campaign_root, node_id=str(task_manifest["node_id"])
    )
    publication = write_immutable_json(path, artifact)
    return {
        "artifact": artifact,
        "path": str(path.resolve()),
        "publication": publication,
    }


__all__ = [
    "TASK_MANIFEST_COMPLETION_CONTRACT",
    "TASK_ROW_COMPLETION_CONTRACT",
    "build_task_manifest_completion",
    "build_task_row_completion",
    "publish_task_manifest_completion",
    "publish_task_row_completion",
    "reusable_task_row_completion",
    "task_manifest_completion_path",
    "task_row_completion_path",
    "validate_task_manifest_completion",
    "validate_task_row_completion",
]
