#!/usr/bin/env python3
"""Resolve one immutable array task and invoke the relational trainer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import (  # noqa: E402
    SCREENING_REGISTRY_CONTRACT,
    SOURCE_RECOVERY_AUTHORIZATION_ENV,
    load_hashed_json,
    validate_campaign_source,
    validate_source_recovery_authorization,
)
from scripts.train_relational_part import main as train_main  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tree-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("screening", "confirmation", "unary"),
        required=True,
    )
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--task-registry", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _screening_task(root: Path, index: int) -> dict[str, Any]:
    registry = load_hashed_json(
        root / "registry" / "screening_registry.json",
        expected_contract=SCREENING_REGISTRY_CONTRACT,
    )
    if index < 0 or index >= len(registry["rows"]):
        raise IndexError("screening task index is outside 0..20")
    row = registry["rows"][index]
    return {
        "run_id": row["run_id"],
        "seed": 101,
        "new_relation_families": list(row["new_relation_families"]),
        "model_contract_path": str(
            (
                root
                / "registry"
                / "model_contracts"
                / f"{row['run_id']}.json"
            ).resolve()
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "screening":
        task = _screening_task(args.campaign_root, args.task_index)
    elif args.mode == "confirmation":
        if args.task_registry is None:
            raise ValueError("confirmation tasks require --task-registry")
        registry = load_hashed_json(args.task_registry)
        contract = registry.get("contract")
        if contract == "relational_part_confirmation_architecture_recovery_tasks_v1":
            campaign = load_hashed_json(args.campaign_root / "campaign_spec.json")
            current = validate_campaign_source(campaign, repo_root=REPO_ROOT)
            authorization_path = os.environ.get(
                SOURCE_RECOVERY_AUTHORIZATION_ENV
            )
            if not authorization_path:
                raise ValueError("recovery tasks require source authorization")
            authorization = validate_source_recovery_authorization(
                authorization_path,
                campaign=campaign,
                current_source=current,
            )
            if authorization.get("recovery_task_registry_sha256") != registry[
                "content_hash"
            ]:
                raise ValueError("recovery task registry is not authorized")
        elif contract != "relational_part_confirmation_task_registry_v1":
            raise ValueError("confirmation task registry contract differs")
        if args.task_index < 0 or args.task_index >= len(registry["tasks"]):
            raise IndexError("confirmation task index is outside its registry")
        task = registry["tasks"][args.task_index]
        if int(task["task_index"]) != int(args.task_index):
            raise ValueError("confirmation task indices are not contiguous")
    else:
        if args.task_index not in (0, 1, 2):
            raise IndexError("unary task index is outside 0..2")
        unary = load_hashed_json(
            args.campaign_root
            / "selection"
            / "semantic_controls"
            / "unary_control_registry.json"
        )
        task = {
            "run_id": "RPT_SELECTED_UNARY",
            "seed": (101, 202, 303)[args.task_index],
            "new_relation_families": list(
                unary["unary_source_relation_set"]
            ),
            "model_contract_path": str(
                (
                    args.campaign_root
                    / "selection"
                    / "semantic_controls"
                    / "unary_model_contract.json"
                ).resolve()
            ),
        }
    resolved = {
        "mode": args.mode,
        "task_index": int(args.task_index),
        **task,
    }
    print(json.dumps(resolved, indent=2, sort_keys=True))
    command = [
        "--campaign-root",
        str(args.campaign_root),
        "--cache-dir",
        str(args.cache_dir),
        "--tree-root",
        str(args.tree_root),
        "--run-id",
        str(task["run_id"]),
        "--seed",
        str(task["seed"]),
        "--model-contract",
        str(task["model_contract_path"]),
        "--selected-families",
        *[str(value) for value in task["new_relation_families"]],
        "--device",
        args.device,
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.mode == "unary":
        command.extend(
            (
                "--unary-registry",
                str(
                    args.campaign_root
                    / "selection"
                    / "semantic_controls"
                    / "unary_control_registry.json"
                ),
            )
        )
    elif args.mode == "confirmation":
        command.extend(
            (
                "--run-registry",
                str(
                    args.campaign_root
                    / "selection"
                    / "confirmation_registry.json"
                ),
            )
        )
    return train_main(command)


if __name__ == "__main__":
    raise SystemExit(main())
