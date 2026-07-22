#!/usr/bin/env python3
"""Bind the confirmed primary, all50, and eligible alternate B5 teachers."""

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
    if args.dry_run:
        selected = load_hashed_json(args.selected_consumer)
        if selected.get("status") != "CONFIRMED_LOCKED":
            raise PermissionError("Stage B5 requires the confirmed selected consumer")
        result = {
            "ok": True,
            "dry_run": True,
            "selected_consumer_sha256": selected["content_hash"],
            "selected_consumer_status": selected.get("status"),
            "binding_kinds": ["primary", "all50", "eligible_alternate"],
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
            include_eligible_alternate=not bool(args.no_alternate),
        )
        result["dry_run"] = False
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
