#!/usr/bin/env python
"""Run a diversity audit over frozen JetClass prediction blocks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.ensemble_analysis import (  # noqa: E402
    parse_group_specs,
    resolve_models_and_groups,
    run_diversity_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-dir",
        required=True,
        help="Directory containing <model>/<split>_predictions.npz blocks.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where diversity_report.json and CSV summaries will be written.",
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=None,
        help="Models to include. Defaults to all complete prediction block directories.",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=None,
        help="Optional group spec name:model_a,model_b. Can be repeated. Defaults to m2_only/hlt_plus_m2/hlt_only when possible.",
    )
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Required because this report evaluates final_test after no training choices are made.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    groups = parse_group_specs(args.group)
    model_names, groups = resolve_models_and_groups(
        args.prediction_dir,
        model_names=args.model_names,
        groups=groups,
    )
    report = run_diversity_audit(
        prediction_dir=args.prediction_dir,
        output_dir=args.output_dir,
        model_names=model_names,
        groups=groups,
        confirm_final_test=bool(args.confirm_final_test),
    )
    print(f"Saved diversity report: {Path(args.output_dir) / 'diversity_report.json'}")
    print("Group oracle final_test summary:")
    for row in report["group_oracle_summary"]:
        if row["split"] != "final_test":
            continue
        print(
            "  "
            f"{row['group']}: "
            f"best_single={row['best_single_model_accuracy']:.6f} "
            f"mean_prob={row['mean_probability_accuracy']:.6f} "
            f"oracle_any={row['oracle_any_model_correct_accuracy']:.6f} "
            f"disagree={row['disagreement_prediction_rate']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
