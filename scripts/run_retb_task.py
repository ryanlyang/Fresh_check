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
from teacher_logit_reco.relation_expert_token_bridge.streamed_execution import (  # noqa: E402
    build_task_lifecycle_receipt,
    is_streamed_profile,
    select_task_local_parent,
    task_local_workspace,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    write_immutable_json,
)


def _prune_streamed_resume_states(row: dict) -> list[str]:
    removed = []
    for output in row["expected_outputs"]:
        candidate = Path(output).parent / "resume_state.pt"
        if candidate.is_file():
            candidate.unlink()
            removed.append(str(candidate))
    return sorted(set(removed))


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
    streamed = is_streamed_profile(graph.get("execution_profile"))
    completed = None
    execution_error: BaseException | None = None
    workspace_parent = select_task_local_parent(environment)
    if streamed:
        try:
            with task_local_workspace(
                campaign_id=str(campaign["campaign_id"]),
                node_id=str(manifest["node_id"]),
                task_index=index,
                environment=environment,
            ) as workspace:
                environment["RETB_TASK_LOCAL_ROOT"] = str(workspace)
                environment["TMPDIR"] = str(workspace)
                environment["TMP"] = str(workspace)
                environment["TEMP"] = str(workspace)
                environment["TORCH_EXTENSIONS_DIR"] = str(
                    workspace / "torch_extensions"
                )
                completed = subprocess.run(
                    row["argv"], cwd=REPO_ROOT, env=environment, check=False
                )
                if completed.returncode != 0:
                    execution_error = RuntimeError(
                        f"task command exited {completed.returncode}"
                    )
                else:
                    missing = [
                        value for value in row["expected_outputs"]
                        if not Path(value).is_file()
                    ]
                    if missing:
                        execution_error = FileNotFoundError(
                            f"task completed without expected outputs: {missing}"
                        )
        except BaseException as error:  # cleanup is guaranteed by the context
            execution_error = error
        receipt = build_task_lifecycle_receipt(
            campaign_spec_sha256=campaign["content_hash"],
            task_manifest_sha256=manifest["content_hash"],
            node_id=manifest["node_id"], task_index=index,
            status="failed" if execution_error is not None else "completed",
            workspace_parent=workspace_parent,
            workspace_removed=True,
            output_paths=row["expected_outputs"], source=campaign["source"],
        )
        receipt_name = f"task_{index:06d}.json"
        if execution_error is not None:
            attempt = environment.get("SLURM_JOB_ID", str(os.getpid()))
            receipt_name = f"task_{index:06d}_failed_{attempt}.json"
        receipt_path = (
            args.campaign_root / "job_ledgers" / "streamed_tasks"
            / str(manifest["node_id"]) / receipt_name
        )
        write_immutable_json(receipt_path, receipt)
        if execution_error is not None:
            if completed is not None and completed.returncode != 0:
                return int(completed.returncode)
            raise execution_error
    else:
        completed = subprocess.run(
            row["argv"], cwd=REPO_ROOT, env=environment, check=False
        )
        if completed.returncode != 0:
            return int(completed.returncode)
        missing = [
            value for value in row["expected_outputs"]
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
    if streamed:
        result["pruned_resume_states"] = _prune_streamed_resume_states(row)
        result["streamed_lifecycle_receipt"] = str(receipt_path)
        result["streamed_lifecycle_receipt_sha256"] = receipt["content_hash"]
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
