#!/usr/bin/env python3
"""Evaluate one immutable locked-finalist task on sealed final_test."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import load_hashed_json  # noqa: E402
from scripts.evaluate_relational_part_final_test import (  # noqa: E402
    main as evaluate_main,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--task-registry", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.campaign_root
    registry = load_hashed_json(args.task_registry)
    if registry.get("contract") != "relational_part_final_test_task_registry_v1":
        raise ValueError("final-test task registry contract differs")
    if args.task_index < 0 or args.task_index >= len(registry["tasks"]):
        raise IndexError("final-test task index is outside its registry")
    task = registry["tasks"][args.task_index]
    if int(task["task_index"]) != args.task_index:
        raise ValueError("final-test task indices are not contiguous")
    lock = load_hashed_json(root / "selection" / "locked_finalists.json")
    command = [
        "--locked-finalists",
        str(root / "selection" / "locked_finalists.json"),
        "--campaign-spec",
        str(root / "campaign_spec.json"),
        "--split-manifest-sha256",
        str(lock["split_manifest_sha256"]),
        "--screening-registry",
        str(root / "registry" / "screening_registry.json"),
        "--normalization",
        str(root / "inputs" / "relation_normalization.json"),
        "--region-normalization",
        str(root / "inputs" / "region_normalization.json"),
        "--unary-registry",
        str(
            root
            / "selection"
            / "semantic_controls"
            / "unary_control_registry.json"
        ),
        "--cache-dir",
        str(root / "inputs" / "hlt_cache"),
        "--tree-root",
        str(root / "inputs" / "relation_tree_cache"),
        "--run-id",
        str(task["run_id"]),
        "--seed",
        str(task["seed"]),
        "--model-contract",
        str(task["model_contract_path"]),
        "--checkpoint",
        str(task["checkpoint_path"]),
        "--checkpoint-registration",
        str(task["checkpoint_registration_path"]),
        "--device",
        args.device,
        "--output-dir",
        str(task["output_dir"]),
        "--expected-event-count",
        str(registry["expected_event_count"]),
    ]
    for name, digest in sorted(lock["hlt_cache_hashes"].items()):
        command.extend(("--hlt-cache-hash", f"{name}={digest}"))
    if args.dry_run:
        command.append("--dry-run")
    return evaluate_main(command)


if __name__ == "__main__":
    raise SystemExit(main())
