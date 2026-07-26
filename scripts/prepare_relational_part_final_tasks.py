#!/usr/bin/env python3
"""Resolve the sealed final-test task registry from locked finalists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import load_split_manifest  # noqa: E402
from teacher_logit_reco.relational_part import (  # noqa: E402
    LOCKED_FINALISTS_CONTRACT,
    bind_source_provenance,
    load_hashed_json,
    source_snapshot,
    with_content_hash,
    write_immutable_json,
    validate_campaign_source,
)


FINAL_TASK_REGISTRY = "relational_part_final_test_task_registry_v1"


def _model_contract(root: Path, run_id: str) -> Path:
    candidates = [
        root / "registry" / "model_contracts" / f"{run_id}.json",
        root / "registry" / "confirmation_model_contracts" / f"{run_id}.json",
        root
        / "selection"
        / "semantic_controls"
        / "unary_model_contract.json",
    ]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"{run_id} must resolve to exactly one model contract; "
            f"found={existing}"
        )
    contract = load_hashed_json(existing[0])
    if contract.get("run_id") != run_id:
        raise ValueError(f"{existing[0]} belongs to another run")
    return existing[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.campaign_root
    campaign = load_hashed_json(root / "campaign_spec.json")
    validate_campaign_source(campaign, repo_root=REPO_ROOT)
    output = args.output or root / "final_test" / "task_registry.json"
    lock = load_hashed_json(
        root / "selection" / "locked_finalists.json",
        expected_contract=LOCKED_FINALISTS_CONTRACT,
    )
    manifest = load_split_manifest(root / "inputs" / "split_manifest.json.gz")
    tasks = []
    for row in lock["evaluation_rows"]:
        run_id = row["run_id"]
        contract = _model_contract(root, run_id)
        for seed in (101, 202, 303):
            run_dir = root / "runs" / run_id / f"seed_{seed}"
            tasks.append(
                {
                    "task_index": len(tasks),
                    "run_id": run_id,
                    "seed": seed,
                    "new_relation_families": list(
                        row["new_relation_families"]
                    ),
                    "model_contract_path": str(contract.resolve()),
                    "checkpoint_path": str(
                        (run_dir / "best_model_val.pt").resolve()
                    ),
                    "checkpoint_registration_path": str(
                        (run_dir / "checkpoint_registration.json").resolve()
                    ),
                    "output_dir": str(
                        (
                            root
                            / "final_test"
                            / run_id
                            / f"seed_{seed}"
                        ).resolve()
                    ),
                }
            )
    artifact = bind_source_provenance(
        with_content_hash(
            {
                "contract": FINAL_TASK_REGISTRY,
                "schema_version": 1,
                "locked_finalists_sha256": lock["content_hash"],
                "expected_event_count": len(manifest.splits["final_test"]),
                "tasks": tasks,
                "task_count": len(tasks),
                "final_test_used_for_selection": False,
                "hlt_only_inference": True,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    publication = None
    if not args.dry_run:
        publication = write_immutable_json(output, artifact)
    print(
        json.dumps(
            {
                "dry_run": bool(args.dry_run),
                "artifact": artifact,
                "publication": publication,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
