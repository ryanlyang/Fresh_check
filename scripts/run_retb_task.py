#!/usr/bin/env python3
"""Execute exactly one authenticated RETB task-manifest row without a shell."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge import (  # noqa: E402
    TASK_MANIFEST_CONTRACT,
    PRODUCTION_GRAPH_CONTRACT,
    load_hashed_json,
    validate_task_manifest,
    validate_task_manifest_for_graph,
    validate_published_dynamic_continuation,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)
from teacher_logit_reco.relation_expert_token_bridge.task_completion import (  # noqa: E402
    publish_task_manifest_completion,
    publish_task_row_completion,
    reusable_task_row_completion,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--task-manifest", required=True, type=Path)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    manifest = load_hashed_json(
        args.task_manifest, expected_contract=TASK_MANIFEST_CONTRACT
    )
    graph = load_hashed_json(
        args.campaign_root / "job_ledgers" / "production_graph.json",
        expected_contract=PRODUCTION_GRAPH_CONTRACT,
    )
    validate_task_manifest_for_graph(
        manifest,
        production_graph=graph,
        campaign_root=args.campaign_root,
        repo_root=REPO_ROOT,
    )
    graph_node = next(
        row for row in graph["nodes"]
        if row["node_id"] == manifest["node_id"]
    )
    if graph_node["dynamic_continuation"]:
        validate_published_dynamic_continuation(
            campaign=campaign,
            production_graph=graph,
            task_manifest=manifest,
            campaign_root=args.campaign_root,
        )
    if manifest["campaign_spec_sha256"] != campaign["content_hash"]:
        raise ValueError("task manifest belongs to another campaign")
    index = args.task_index
    if index is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None or not raw.isdigit():
            raise ValueError("task index is absent")
        index = int(raw)
    if index < 0 or index >= int(manifest["task_count"]):
        raise IndexError("task index is outside the manifest")
    row = manifest["rows"][index]
    if (
        int(row["task_index"]) != index
        or row["task_id"] != f"{manifest['node_id']}:{index}"
    ):
        raise ValueError("task row identity differs")
    resolved = {
        "task_manifest_sha256": manifest["content_hash"],
        "node_id": manifest["node_id"],
        "task_index": index,
        "task_id": row["task_id"],
        "argv": row["argv"],
        "environment": row["environment"],
        "expected_outputs": row["expected_outputs"],
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(resolved, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    reusable = reusable_task_row_completion(
        campaign_root=args.campaign_root,
        campaign=campaign,
        task_manifest=manifest,
        task_index=index,
    )
    if reusable is not None:
        aggregate = None
        if int(manifest["task_count"]) == 1:
            aggregate = publish_task_manifest_completion(
                campaign_root=args.campaign_root,
                campaign=campaign,
                task_manifest=manifest,
            )
        payload = {
            "status": "reused_authenticated_completion",
            "row_completion_sha256": reusable["content_hash"],
        }
        if aggregate is not None:
            payload["manifest_completion_sha256"] = aggregate[
                "artifact"
            ]["content_hash"]
            payload["manifest_completion_path"] = aggregate["path"]
        print(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    environment = dict(os.environ)
    environment.update(row["environment"])
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        row["argv"],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        return int(completed.returncode)
    missing = [
        value
        for value in row["expected_outputs"]
        if not Path(value).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"task completed without expected outputs: {missing}"
        )
    completion = publish_task_row_completion(
        campaign_root=args.campaign_root,
        campaign=campaign,
        task_manifest=manifest,
        task_index=index,
    )
    result: dict[str, object] = {
        "status": "completed_and_authenticated",
        "row_completion_sha256": completion["artifact"]["content_hash"],
        "row_completion_path": completion["path"],
    }
    if int(manifest["task_count"]) == 1:
        aggregate = publish_task_manifest_completion(
            campaign_root=args.campaign_root,
            campaign=campaign,
            task_manifest=manifest,
        )
        result["manifest_completion_sha256"] = aggregate["artifact"][
            "content_hash"
        ]
        result["manifest_completion_path"] = aggregate["path"]
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
