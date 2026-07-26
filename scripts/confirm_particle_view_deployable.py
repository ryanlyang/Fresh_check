#!/usr/bin/env python3
"""Publish the selected-path fairness ledger and sealed-split authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_sealed_split_authorization,
    build_selected_path_fairness_ledger,
    write_immutable_json,
)


def _load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument(
        "--fairness-inputs",
        required=True,
        help="JSON with training_ledgers and resource_profiles keyed by config/seed",
    )
    parser.add_argument("--train-identity-sha256", required=True)
    parser.add_argument("--flop-fixture-sha256", required=True)
    parser.add_argument("--flop-counter-sha256", required=True)
    parser.add_argument("--stack-split-sha256", required=True)
    parser.add_argument("--final-test-split-sha256", required=True)
    parser.add_argument("--fairness-output", required=True)
    parser.add_argument("--authorization-output", required=True)
    return parser


def _integer_keys(source):
    return {
        config: {int(seed): payload for seed, payload in rows.items()}
        for config, rows in source.items()
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selection = _load(args.selection)
    inputs = _load(args.fairness_inputs)
    fairness = build_selected_path_fairness_ledger(
        selection=selection,
        replica_training_ledgers=_integer_keys(inputs["training_ledgers"]),
        resource_profiles=_integer_keys(inputs["resource_profiles"]),
        train_identity_sha256=args.train_identity_sha256,
        flop_fixture_sha256=args.flop_fixture_sha256,
        flop_counter_sha256=args.flop_counter_sha256,
    )
    write_immutable_json(args.fairness_output, fairness)
    authorization = build_sealed_split_authorization(
        selection=selection,
        fairness_ledger=fairness,
        stack_split_sha256=args.stack_split_sha256,
        final_test_split_sha256=args.final_test_split_sha256,
        ce_only_comparator_bundles=inputs.get("ce_only_comparators", []),
        stage_g_control_bundles=inputs.get("stage_g_controls", []),
        fusion_recipe_sha256=inputs.get("fusion_recipe_sha256", []),
    )
    write_immutable_json(args.authorization_output, authorization)
    print(
        f"fairness_entries={fairness['distinct_entry_count']} "
        f"authorization_hash={authorization['content_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
