#!/usr/bin/env python3
"""Select, confirm, or bind an exact prediction-anchored bridge teacher."""

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
    sha256_file,
    write_immutable_json,
)
from teacher_logit_reco.local_particle_residual_field.bridge_evaluation import (  # noqa: E402
    ALL50_TEACHER_NAMESPACE,
    ALTERNATE_TEACHER_NAMESPACE,
    PRIMARY_TEACHER_NAMESPACE,
    build_teacher_binding,
    finalize_consumer_confirmation,
    select_bridge_consumer_preconfirmation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select", help="lock the model_val_select aggregate winner")
    select.add_argument("--clean-aggregate", required=True)
    select.add_argument("--robust-aggregate", required=True)
    select.add_argument("--clean-checkpoint", required=True)
    select.add_argument("--robust-checkpoint", required=True)
    select.add_argument("--f0-checkpoint-sha256", default="")
    select.add_argument("--f0-checkpoint", default="")
    select.add_argument("--bridge-recipe-sha256", default="")
    select.add_argument("--bridge-recipe", default="")
    select.add_argument("--output", required=True)
    select.add_argument("--dry-run", action="store_true")

    confirm = subparsers.add_parser("confirm", help="apply the sealed one-shot confirmation")
    confirm.add_argument("--preconfirmation", required=True)
    confirm.add_argument("--confirmation-metrics", required=True)
    confirm.add_argument("--access-receipt", required=True)
    confirm.add_argument("--output-dir", required=True)
    confirm.add_argument("--dry-run", action="store_true")

    bind = subparsers.add_parser("bind", help="bind a primary/all50/alternate median teacher")
    bind.add_argument("--kind", choices=("primary", "all50", "alternate"), required=True)
    bind.add_argument("--run-id", required=True)
    bind.add_argument("--aggregate", required=True)
    bind.add_argument("--checkpoint", required=True)
    bind.add_argument("--checkpoint-sha256", required=True)
    bind.add_argument("--bridge-recipe-sha256", required=True)
    bind.add_argument("--model-val-select-sha256", required=True)
    bind.add_argument("--stack-val-consumer-sha256", required=True)
    bind.add_argument("--selected-consumer", default="")
    bind.add_argument("--all50-scaler", default="")
    bind.add_argument("--output", required=True)
    bind.add_argument("--dry-run", action="store_true")
    return parser


def _plain_json(path: str) -> dict:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"missing/unsafe JSON input: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {source}")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "select":
        if bool(args.f0_checkpoint_sha256) == bool(args.f0_checkpoint):
            raise ValueError("select requires exactly one of --f0-checkpoint-sha256/--f0-checkpoint")
        if bool(args.bridge_recipe_sha256) == bool(args.bridge_recipe):
            raise ValueError("select requires exactly one of --bridge-recipe-sha256/--bridge-recipe")
        clean = load_hashed_json(args.clean_aggregate)
        robust = load_hashed_json(args.robust_aggregate)
        f0_sha256 = (
            str(args.f0_checkpoint_sha256)
            if args.f0_checkpoint_sha256
            else sha256_file(args.f0_checkpoint)
        )
        bridge_sha256 = (
            str(args.bridge_recipe_sha256)
            if args.bridge_recipe_sha256
            else load_hashed_json(args.bridge_recipe)["content_hash"]
        )
        artifact = select_bridge_consumer_preconfirmation(
            [clean, robust],
            f0_checkpoint_sha256=f0_sha256,
            bridge_recipe_sha256=bridge_sha256,
            selected_checkpoint_paths={
                "T10_clean": args.clean_checkpoint,
                "T10_robust": args.robust_checkpoint,
            },
        )
        publication = None if args.dry_run else write_immutable_json(args.output, artifact)
        print(json.dumps({"dry_run": bool(args.dry_run), "selection": artifact, "publication": publication}, indent=2, sort_keys=True))
        return 0

    if args.command == "confirm":
        preconfirmation = load_hashed_json(args.preconfirmation)
        access_receipt = load_hashed_json(args.access_receipt)
        confirmation = _plain_json(args.confirmation_metrics)
        artifact = finalize_consumer_confirmation(
            preconfirmation,
            confirmation,
            access_receipt=access_receipt,
            output_dir=None if args.dry_run else args.output_dir,
        )
        print(json.dumps({"dry_run": bool(args.dry_run), "confirmation": artifact}, indent=2, sort_keys=True))
        return 0 if artifact.get("status") == "CONFIRMED_LOCKED" else 2

    aggregate = load_hashed_json(args.aggregate)
    selected = load_hashed_json(args.selected_consumer) if args.selected_consumer else None
    if args.kind == "primary" and selected is None:
        raise ValueError("primary binding requires --selected-consumer")
    if args.kind != "primary" and selected is not None:
        raise ValueError("non-primary bindings must not receive --selected-consumer")
    if args.kind != "all50" and args.all50_scaler:
        raise ValueError("only an all50 binding may receive --all50-scaler")
    all50_scaler = load_hashed_json(args.all50_scaler) if args.all50_scaler else None
    channel, namespace = {
        "primary": ("physical45", PRIMARY_TEACHER_NAMESPACE),
        "all50": ("all50", ALL50_TEACHER_NAMESPACE),
        "alternate": ("physical45", ALTERNATE_TEACHER_NAMESPACE),
    }[args.kind]
    artifact = build_teacher_binding(
        binding_kind=args.kind,
        run_id=args.run_id,
        aggregate=aggregate,
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        channel_policy=channel,
        validation_manifest_hashes={
            "model_val_select": args.model_val_select_sha256,
            "stack_val_consumer": args.stack_val_consumer_sha256,
        },
        target_cache_namespace=namespace,
        bridge_recipe_sha256=args.bridge_recipe_sha256,
        primary_selection=selected,
        all50_scaler_artifact=all50_scaler,
    )
    publication = None if args.dry_run else write_immutable_json(args.output, artifact)
    print(json.dumps({"dry_run": bool(args.dry_run), "binding": artifact, "publication": publication}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
