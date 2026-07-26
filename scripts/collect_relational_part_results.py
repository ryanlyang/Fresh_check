#!/usr/bin/env python3
"""Collect authenticated run results into one deterministic JSON sequence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import (  # noqa: E402
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relational_part.workflow import (  # noqa: E402
    build_run_result_envelope,
    expected_training_lineage,
    load_run_result,
    validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("screening", "confirmation", "unary"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    registry = load_hashed_json(args.registry)
    campaign = load_hashed_json(args.campaign_root / "campaign_spec.json")
    source = validate_campaign_source(campaign, repo_root=REPO_ROOT)
    relation_registry = load_hashed_json(
        args.campaign_root / "registry" / "relation_family_registry.json"
    )
    if args.mode == "screening":
        requests = [
            (row["run_id"], 101) for row in registry["rows"]
        ]
    elif args.mode == "confirmation":
        requests = [
            (row["run_id"], int(seed))
            for row in registry["rows"]
            for seed in row["seeds"]
        ]
    else:
        requests = [("RPT_SELECTED_UNARY", int(seed)) for seed in (101, 202, 303)]
    registry_rows = {
        str(row.get("run_id", "RPT_SELECTED_UNARY")): row
        for row in registry.get("rows", ())
    }
    if args.mode == "unary":
        registry_rows["RPT_SELECTED_UNARY"] = registry
    results = []
    for run_id, seed in requests:
        row = registry_rows[run_id]
        families = tuple(
            row.get(
                "new_relation_families",
                row.get("unary_source_relation_set", ()),
            )
        )
        if run_id == "RPT_SELECTED_UNARY":
            contract_path = (
                args.campaign_root
                / "selection"
                / "semantic_controls"
                / "unary_model_contract.json"
            )
        else:
            confirmation_path = (
                args.campaign_root
                / "registry"
                / "confirmation_model_contracts"
                / f"{run_id}.json"
            )
            contract_path = (
                confirmation_path
                if confirmation_path.is_file()
                else args.campaign_root
                / "registry"
                / "model_contracts"
                / f"{run_id}.json"
            )
        model_contract = load_hashed_json(contract_path)
        expected_run_registry = registry["content_hash"]
        if (
            args.mode == "confirmation"
            and int(seed) == 101
            and row.get("seed_101", {}).get("mode") == "reuse_hash_exact"
        ):
            expected_run_registry = load_hashed_json(
                args.campaign_root / "registry" / "screening_registry.json"
            )["content_hash"]
        results.append(
            load_run_result(
                args.campaign_root / "runs" / run_id / f"seed_{seed}",
                expected_lineage=expected_training_lineage(
                    args.campaign_root, families=families
                ),
                expected_run_registry_sha256=expected_run_registry,
                expected_relation_registry_sha256=relation_registry[
                    "content_hash"
                ],
                expected_model_contract_sha256=model_contract["content_hash"],
            )
        )
    result = build_run_result_envelope(
        mode=args.mode,
        registry_sha256=registry["content_hash"],
        campaign_spec_sha256=campaign["content_hash"],
        source=source,
        results=results,
    )
    import json
    print(json.dumps(result, indent=2, sort_keys=True))
    if not args.dry_run:
        write_immutable_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
