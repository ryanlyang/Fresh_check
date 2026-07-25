#!/usr/bin/env python3
"""Bind the confirmed primary, all50, and declared alternate B5 teachers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    load_hashed_json,
)
from teacher_logit_reco.local_particle_residual_field.bridge_teacher_execution import (  # noqa: E402
    bind_teacher_set_from_execution_spec,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-spec", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--selected-consumer", required=True)
    parser.add_argument("--physical45-recipe", required=True)
    parser.add_argument("--all50-recipe", required=True)
    parser.add_argument("--all50-scaler", required=True)
    parser.add_argument("--consumer-evaluation-root", required=True)
    parser.add_argument("--consumer-publication-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-alternate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registry = load_hashed_json(args.registry)
    alternate_rows = [
        row
        for row in registry.get("runs", [])
        if row.get("canonical_run_id") in {"D10_TALT_A0", "D10_TALT_A3"}
    ]
    if {row.get("canonical_run_id") for row in alternate_rows} != {
        "D10_TALT_A0",
        "D10_TALT_A3",
    }:
        raise ValueError("campaign registry omits a clean-consumer comparison")
    alternate_statuses = {
        str(row.get("execution_status")) for row in alternate_rows
    }
    if len(alternate_statuses) != 1:
        raise ValueError("clean-consumer comparison rows have inconsistent status")
    alternate_status = next(iter(alternate_statuses))
    alternate_runnable = bool(
        not args.no_alternate
        and alternate_status == "RUNNABLE"
    )
    if args.dry_run:
        selected = load_hashed_json(args.selected_consumer)
        if selected.get("status") != "CONFIRMED_LOCKED":
            raise PermissionError("Stage B5 requires the confirmed selected consumer")
        result = {
            "ok": True,
            "dry_run": True,
            "selected_consumer_sha256": selected["content_hash"],
            "selected_consumer_status": selected.get("status"),
            "binding_kinds": [
                "primary",
                "all50",
                *(["declared_alternate"] if alternate_runnable else []),
            ],
            "alternate_registry_status": alternate_status,
            "alternate_run_ids": ["D10_TALT_A0", "D10_TALT_A3"],
            "bindings_created_before_cache": True,
            "guessed_consumer_allowed": False,
        }
    else:
        result = bind_teacher_set_from_execution_spec(
            args.execution_spec,
            selected_consumer_path=args.selected_consumer,
            physical45_recipe_path=args.physical45_recipe,
            all50_recipe_path=args.all50_recipe,
            all50_scaler_path=args.all50_scaler,
            consumer_evaluation_root=args.consumer_evaluation_root,
            consumer_publication_root=args.consumer_publication_root,
            output_dir=args.output_dir,
            include_eligible_alternate=alternate_runnable,
        )
        result["dry_run"] = False
        result["alternate_registry_status"] = alternate_status
        result["alternate_run_ids"] = ["D10_TALT_A0", "D10_TALT_A3"]
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
