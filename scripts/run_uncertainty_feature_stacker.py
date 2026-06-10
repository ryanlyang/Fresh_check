#!/usr/bin/env python
"""Train uncertainty-feature stackers from frozen JetClass prediction blocks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.ensemble_analysis import (  # noqa: E402
    UNCERTAINTY_FEATURE_MODES,
    UncertaintyStackerConfig,
    default_c_grid,
    parse_group_specs,
    resolve_models_and_groups,
    run_uncertainty_feature_stackers,
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
        help="Directory where uncertainty_stacker_report.json and stackers will be written.",
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
        "--feature-modes",
        nargs="+",
        default=["uncertainty", "mean_uncertainty", "logits_probs_uncertainty"],
        choices=list(UNCERTAINTY_FEATURE_MODES),
        help="Uncertainty feature sets to evaluate.",
    )
    parser.add_argument(
        "--c-grid",
        nargs="+",
        type=float,
        default=default_c_grid(),
        help="Candidate LogisticRegression C values selected on stack_val.",
    )
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Required because this report evaluates final_test after stack_val model selection.",
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
    config = UncertaintyStackerConfig(
        prediction_dir=str(args.prediction_dir),
        output_dir=str(args.output_dir),
        model_names=list(model_names),
        groups=groups,
        feature_modes=list(args.feature_modes),
        c_grid=[float(value) for value in args.c_grid],
        max_iter=int(args.max_iter),
        confirm_final_test=bool(args.confirm_final_test),
    )
    report = run_uncertainty_feature_stackers(config)
    print(f"Saved uncertainty stacker report: {Path(args.output_dir) / 'uncertainty_stacker_report.json'}")
    print("Final-test uncertainty stacker summary:")
    for group_name, group in report["group_uncertainty_stacker_metrics"].items():
        for mode, mode_report in group["feature_modes"].items():
            final = mode_report["metrics"]["final_test"]
            selected = mode_report["selection"]["selected_C"]
            print(
                "  "
                f"{group_name} / {mode}: "
                f"C={selected} "
                f"acc={final['accuracy']:.6f} "
                f"ce={final['cross_entropy']:.6f} "
                f"n_features={mode_report['n_features']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
