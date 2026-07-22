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
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.local_particle_residual_field.bridge_production import (  # noqa: E402
    build_prediction_anchored_job_ledger,
    build_prediction_anchored_tigris_graph,
    rehearse_prediction_anchored_campaign_cpu,
    render_tigris_sbatch_commands,
    validate_prediction_anchored_tigris_graph,
)
from teacher_logit_reco.local_particle_residual_field.bridge_splits import (  # noqa: E402
    ChildSplitSpec,
    ParentPartitionSpec,
    PredictionAnchoredSplitConfig,
    build_child_split_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-action",
        choices=("contracts", "production-plan", "rehearse-cpu", "validate-production", "finalize-ledger"),
        default="contracts",
    )
    parser.add_argument("--parent-manifest", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--budget-gib", type=int, choices=(5, 6), default=5)
    parser.add_argument("--alternate-teacher-valid", action="store_true")
    parser.add_argument(
        "--debug-miniature-splits",
        action="store_true",
        help="derive tiny half-splits from the parent; allowed only with --dry-run",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--registry", default="", help="final measured Step 8 registry")
    parser.add_argument("--reservations", default="", help="Step 9 measured reservation artifact")
    parser.add_argument("--execution-spec", default="", help="immutable scientific execution specification")
    parser.add_argument("--artifact-root", default="", help="immutable production campaign root")
    parser.add_argument("--production-output", default="", help="immutable graph/rehearsal/ledger JSON")
    parser.add_argument("--graph", default="", help="existing production graph for ledger finalization")
    parser.add_argument("--ledger-tsv", default="", help="node_id<TAB>job_id rows from the submitter")
    parser.add_argument("--include-final-test", action="store_true")
    parser.add_argument("--pack-size", type=int, choices=(1, 2, 3, 4), default=4)
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
    if args.campaign_action != "contracts":
        if args.campaign_action == "finalize-ledger":
            if not args.graph or not args.ledger_tsv:
                raise ValueError("finalize-ledger requires --graph and --ledger-tsv")
            graph = load_hashed_json(args.graph)
            rows = Path(args.ledger_tsv).read_text(encoding="utf-8").splitlines()
            parsed = {}
            for line_number, line in enumerate(rows, start=1):
                if not line.strip():
                    continue
                fields = line.split("\t")
                if len(fields) != 2 or not fields[0] or not fields[1]:
                    raise ValueError(f"invalid job ledger TSV row {line_number}")
                if fields[0] in parsed:
                    raise ValueError(f"duplicate job ledger node {fields[0]}")
                parsed[fields[0]] = fields[1]
            result = build_prediction_anchored_job_ledger(
                graph, job_ids=parsed, include_final_test=bool(args.include_final_test)
            )
        else:
            if not args.registry or not args.reservations or not args.execution_spec or not args.artifact_root:
                raise ValueError(
                    f"{args.campaign_action} requires --registry, --reservations, --execution-spec, and --artifact-root"
                )
            registry = load_hashed_json(args.registry)
            reservations = load_hashed_json(args.reservations)
            execution_spec = load_hashed_json(args.execution_spec)
            graph = build_prediction_anchored_tigris_graph(
                registry,
                reservations=reservations,
                execution_spec=execution_spec,
                artifact_root=args.artifact_root,
                pack_size=int(args.pack_size),
            )
            if args.campaign_action == "production-plan":
                result = {
                    "graph": graph,
                    "commands": render_tigris_sbatch_commands(
                        graph, include_final_test=bool(args.include_final_test)
                    ),
                }
            elif args.campaign_action == "rehearse-cpu":
                result = rehearse_prediction_anchored_campaign_cpu(graph)
            else:
                result = validate_prediction_anchored_tigris_graph(graph)
        output = {
            "campaign_action": args.campaign_action,
            "dry_run": bool(args.dry_run),
            "submission_executed": False,
            "result": result,
        }
        if not args.dry_run:
            if not args.production_output:
                raise ValueError("production actions require --production-output unless --dry-run")
            artifact = result["graph"] if args.campaign_action == "production-plan" else result
            if not isinstance(artifact, dict) or "content_hash" not in artifact:
                raise ValueError(
                    "only hashed production graph/rehearsal/ledger artifacts may be published; "
                    "use --dry-run for validation-only output"
                )
            output["publication"] = write_immutable_json(args.production_output, artifact)
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0

    if args.debug_miniature_splits and not args.dry_run:
        raise PermissionError("debug miniature splits are forbidden outside --dry-run")
    if not args.parent_manifest:
        raise ValueError("contracts action requires --parent-manifest")
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
