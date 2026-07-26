#!/usr/bin/env python3
"""Fit/evaluate a locked particle-view fusion on sealed stack validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_fusion_recipe,
    build_stack_fusion_partition,
    evaluate_fusion_recipe,
    fit_linear_logit_fusion,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--source-key", action="append", required=True)
    parser.add_argument("--source-bundle-sha256", action="append", required=True)
    parser.add_argument("--class-name", action="append", required=True)
    parser.add_argument("--stack-split-sha256", required=True)
    parser.add_argument("--fusion-id", required=True)
    parser.add_argument("--method", choices=("logit_average", "linear_logit"), required=True)
    parser.add_argument("--recipe-output", required=True)
    parser.add_argument("--report-output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(args.source_key) != len(args.source_bundle_sha256):
        raise ValueError("source keys and checkpoint hashes must align")
    with np.load(args.input_npz, allow_pickle=False) as source:
        logits = [np.asarray(source[key]) for key in args.source_key]
        labels = np.asarray(source["labels"])
        identities = np.asarray(source["event_identities"]).tolist()
    partition = build_stack_fusion_partition(
        event_identities=identities,
        stack_split_sha256=args.stack_split_sha256,
    )
    parameters = (
        fit_linear_logit_fusion(
            source_logits=logits,
            labels=labels,
            fit_indices=partition["fit_indices"],
        )
        if args.method == "linear_logit"
        else None
    )
    recipe = build_fusion_recipe(
        fusion_id=args.fusion_id,
        source_bundle_sha256=args.source_bundle_sha256,
        class_order=args.class_name,
        stack_partition=partition,
        method=args.method,
        linear_parameters=parameters,
    )
    write_immutable_json(args.recipe_output, recipe)
    report = evaluate_fusion_recipe(
        recipe=recipe,
        stack_partition=partition,
        source_logits=logits,
        labels=labels,
        source_bundle_sha256=args.source_bundle_sha256,
    )
    write_immutable_json(args.report_output, report)
    print(json.dumps({"recipe_hash": recipe["content_hash"], "accuracy": report["metrics"]["accuracy"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
