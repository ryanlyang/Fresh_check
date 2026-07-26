#!/usr/bin/env python3
"""Execute one authenticated logical graph node inside a Slurm allocation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_node_completion,
    capture_clean_source_checkout,
    load_hashed_json,
    load_quality_warning_jsonl,
    sha256_file,
    validate_particle_view_production_graph,
    write_immutable_json,
    write_quality_warning_jsonl,
    write_quality_warning_summary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    graph = load_hashed_json(args.graph)
    audit = validate_particle_view_production_graph(graph)
    if args.node_id not in audit["nodes"]:
        raise ValueError("logical node is absent from the production graph")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("production nodes require PYTHONNOUSERSITE=1")
    if graph.get("source_checkout_clean") is not True:
        raise RuntimeError("production graph is not bound to a clean checkout")
    checkout = capture_clean_source_checkout(
        PROJECT_ROOT,
        expected_commit=graph["source_commit"],
    )
    for key in (
        "source_tree_git_oid",
        "source_status_sha256",
        "source_checkout_clean",
    ):
        if checkout[key] != graph[key]:
            raise RuntimeError(f"executed source checkout changed: {key}")
    account = os.environ.get("SLURM_JOB_ACCOUNT")
    if account is not None and account != "reu-aisocial":
        raise RuntimeError("Tigris job is running under the wrong account")
    node = audit["nodes"][args.node_id]
    wave_index_text = os.environ.get("PARTICLE_VIEW_TASK_WAVE_INDEX")
    array_task_text = os.environ.get("SLURM_ARRAY_TASK_ID")
    barrier = os.environ.get("PARTICLE_VIEW_NODE_BARRIER") == "1"
    if wave_index_text is not None:
        if barrier or array_task_text is None:
            raise RuntimeError("task-wave execution requires one Slurm array index")
        wave_index = int(wave_index_text)
        array_index = int(array_task_text)
        try:
            task_id = node["task_waves"][wave_index][array_index]
        except (IndexError, TypeError) as exc:
            raise ValueError("Slurm task-wave index is outside the graph") from exc
        command = [*node["command"], "--task-id", task_id]
    else:
        if not barrier and os.environ.get("SLURM_JOB_ID") is not None:
            raise RuntimeError("production allocation is neither a wave nor barrier")
        command = list(node["command"])
    print("NODE_COMMAND:")
    print("  " + " ".join(command))
    if args.dry_run:
        return 0
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return int(completed.returncode)
    if wave_index_text is not None:
        return 0
    execution_manifest = None
    if "--execution-manifest" in node["command"]:
        manifest_index = node["command"].index("--execution-manifest") + 1
        execution_manifest = load_hashed_json(node["command"][manifest_index])
    output_artifacts = []
    task_warning_hashes = []
    task_warnings = []
    if execution_manifest is not None:
        for task in execution_manifest["tasks"]:
            if task["node_id"] != args.node_id:
                continue
            for key in ("result_path", "completion_path"):
                path = Path(task[key]).resolve()
                if not path.is_file() or path.is_symlink():
                    raise FileNotFoundError(
                        f"node barrier is missing {key} for {task['task_id']}"
                    )
                output_artifacts.append(
                    {"path": str(path), "sha256": sha256_file(path)}
                )
            result = load_hashed_json(task["result_path"])
            task_warning_hashes.extend(result.get("warning_sha256", []))
            for artifact in result.get("artifacts", []):
                if Path(artifact["path"]).name == "quality_warnings.jsonl":
                    task_warnings.extend(
                        load_quality_warning_jsonl(artifact["path"])
                    )
        report_root = (
            Path(execution_manifest["artifact_root"])
            / "runtime_node_reports"
            / args.node_id
        )
        reports = sorted(
            report_root.glob("*.json"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not reports:
            raise FileNotFoundError("node barrier emitted no runtime node report")
        report_path = reports[-1].resolve()
        output_artifacts.append(
            {"path": str(report_path), "sha256": sha256_file(report_path)}
        )
    warning_path = (
        Path(graph["artifact_root"])
        / "quality_warnings"
        / args.node_id
        / "quality_warnings.jsonl"
    )
    if not warning_path.exists():
        unique_warnings = {
            row["content_hash"]: row for row in task_warnings
        }
        write_quality_warning_jsonl(
            warning_path,
            [unique_warnings[key] for key in sorted(unique_warnings)],
        )
    warnings = load_quality_warning_jsonl(warning_path)
    warning_hashes = sorted(
        set(
            [warning["content_hash"] for warning in warnings]
            + task_warning_hashes
        )
    )
    if args.node_id == "pv10_hlt_only_final_test":
        campaign_warnings = []
        warning_root = Path(graph["artifact_root"]) / "quality_warnings"
        for path in sorted(warning_root.glob("*/quality_warnings.jsonl")):
            campaign_warnings.extend(load_quality_warning_jsonl(path))
        write_quality_warning_summary(
            output_dir=graph["artifact_root"],
            warnings=campaign_warnings,
        )
    completion = build_node_completion(
        graph=graph,
        node_id=args.node_id,
        output_artifacts=output_artifacts,
        warning_sha256=warning_hashes,
    )
    output = (
        Path(graph["artifact_root"])
        / "node_completions"
        / f"{args.node_id}.json"
    )
    write_immutable_json(output, completion)
    print(
        json.dumps(
            {
                "node_id": args.node_id,
                "warning_count": len(warning_hashes),
                "completion_sha256": completion["content_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
