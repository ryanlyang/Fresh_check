#!/usr/bin/env python3
"""Build the immutable Step 1 contracts for the prediction-anchored pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import SPLIT_ORDER, load_split_manifest  # noqa: E402
from teacher_logit_reco.local_particle_residual_field.bridge_campaign import (  # noqa: E402
    NORMAL_PILOT_BUDGET_BYTES,
    PAIRED3_HARD_CEILING_BYTES,
    build_campaign_registry,
    build_provisional_storage_projection,
    build_step1_report,
    write_step1_artifacts,
)
from teacher_logit_reco.local_particle_residual_field.bridge_splits import (  # noqa: E402
    ChildSplitSpec,
    ParentPartitionSpec,
    PredictionAnchoredSplitConfig,
    build_child_split_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--budget-gib", type=int, choices=(5, 6), default=5)
    parser.add_argument("--alternate-teacher-valid", action="store_true")
    parser.add_argument(
        "--debug-miniature-splits",
        action="store_true",
        help="derive tiny half-splits from the parent; allowed only with --dry-run",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _debug_split_config(parent) -> PredictionAnchoredSplitConfig:
    counts = tuple((name, len(parent.splits[name])) for name in SPLIT_ORDER)
    child_counts: dict[str, int] = {}
    for parent_name in ("stack_train", "model_val", "stack_val"):
        count = len(parent.splits[parent_name])
        if count % 2 or (count // 2) % len(parent.class_names):
            raise ValueError(
                f"debug parent {parent_name} must divide into two class-balanced halves"
            )
        child_counts[parent_name] = count // 2
    return PredictionAnchoredSplitConfig(
        contract="prediction_anchored_split_config_v1_debug_only",
        parent_split_counts=counts,
        partitions=(
            ParentPartitionSpec(
                "stack_train",
                810_101,
                (
                    ChildSplitSpec(
                        "stack_train_consumer",
                        child_counts["stack_train"],
                        "consumer_training",
                    ),
                    ChildSplitSpec(
                        "stack_train_distill",
                        child_counts["stack_train"],
                        "reconstructor_training",
                    ),
                ),
            ),
            ParentPartitionSpec(
                "model_val",
                810_202,
                (
                    ChildSplitSpec(
                        "model_val_stop",
                        child_counts["model_val"],
                        "checkpoint_selection",
                    ),
                    ChildSplitSpec(
                        "model_val_select",
                        child_counts["model_val"],
                        "configuration_selection",
                    ),
                ),
            ),
            ParentPartitionSpec(
                "stack_val",
                810_303,
                (
                    ChildSplitSpec(
                        "stack_val_consumer",
                        child_counts["stack_val"],
                        "consumer_confirmation",
                        "consumer_preconfirmation",
                    ),
                    ChildSplitSpec(
                        "stack_val_deploy",
                        child_counts["stack_val"],
                        "deployable_confirmation",
                        "deployable_preconfirmation",
                    ),
                ),
            ),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.debug_miniature_splits and not args.dry_run:
        raise PermissionError("debug miniature splits are forbidden outside --dry-run")
    if not args.dry_run and not args.output_dir:
        raise ValueError("--output-dir is required unless --dry-run is used")

    parent = load_split_manifest(args.parent_manifest)
    split_config = _debug_split_config(parent) if args.debug_miniature_splits else None
    child_manifest = build_child_split_manifest(
        parent,
        **({"config": split_config} if split_config is not None else {}),
    )
    registry = build_campaign_registry(
        alternate_teacher_valid=bool(args.alternate_teacher_valid)
    )
    budget = (
        NORMAL_PILOT_BUDGET_BYTES
        if args.budget_gib == 5
        else PAIRED3_HARD_CEILING_BYTES
    )
    projection = build_provisional_storage_projection(
        registry,
        child_manifest,
        particle_width=int(parent.max_constits),
        selected_budget_bytes=budget,
    )
    report = build_step1_report(
        registry=registry,
        child_manifest=child_manifest,
        storage_projection=projection,
    )

    output: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "debug_miniature_splits": bool(args.debug_miniature_splits),
        "report": report,
    }
    if not args.dry_run:
        output["publication"] = write_step1_artifacts(
            args.output_dir,
            child_manifest=child_manifest,
            registry=registry,
            storage_projection=projection,
            report=report,
        )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
