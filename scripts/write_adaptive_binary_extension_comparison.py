#!/usr/bin/env python3
"""Freeze one model-val nominal-versus-extension ABPH screening decision."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline import (  # noqa: E402
    build_extension_comparison_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--nominal-checkpoint-hash", required=True)
    parser.add_argument("--extension-checkpoint-hash", required=True)
    parser.add_argument("--matched-a0-artifact-hash", required=True)
    parser.add_argument("--frozen-tagger-recipe-hash", required=True)
    parser.add_argument("--training-budget-hash", required=True)
    parser.add_argument("--nominal-best-loss", type=float, required=True)
    parser.add_argument("--extension-best-loss", type=float, required=True)
    parser.add_argument("--nominal-tagging-gain", type=float, required=True)
    parser.add_argument("--extension-tagging-gain", type=float, required=True)
    parser.add_argument("--initialization-seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_extension_comparison_report(
        variant_name=args.variant,
        nominal_checkpoint_hash=args.nominal_checkpoint_hash,
        extension_checkpoint_hash=args.extension_checkpoint_hash,
        matched_a0_artifact_hash=args.matched_a0_artifact_hash,
        frozen_tagger_recipe_hash=args.frozen_tagger_recipe_hash,
        nominal_best_loss=args.nominal_best_loss,
        extension_best_loss=args.extension_best_loss,
        nominal_tagging_gain=args.nominal_tagging_gain,
        extension_tagging_gain=args.extension_tagging_gain,
        initialization_seed=args.initialization_seed,
        training_budget_hash=args.training_budget_hash,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
