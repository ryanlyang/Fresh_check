#!/usr/bin/env python3
"""Build or fit fresh cross-architecture fusion from prediction blocks.

By default this runs the Step 7 feature builder only.  Pass ``--fit-fusers``
to run the Step 8 fuser suite on frozen prediction blocks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.fusion import DEFAULT_C_GRID, STACK_SPLITS  # noqa: E402
from teacher_logit_reco.crossarch_experiment import CrossArchExperimentLayout  # noqa: E402
from teacher_logit_reco.crossarch_fusion import (  # noqa: E402
    ALL_FEATURE_MODES,
    DEFAULT_CONTROL_FEATURE_MODES,
    DEFAULT_CROSSARCH_FUSERS,
    CrossArchFusionFeatureBuildConfig,
    CrossArchFusionFitConfig,
    default_crossarch_feature_groups,
    run_crossarch_feature_builder,
    run_crossarch_fusers,
    validate_crossarch_feature_groups,
)


def parse_group(text: str) -> tuple[str, List[str]]:
    if ":" not in text:
        raise argparse.ArgumentTypeError("Groups must be formatted name:model1,model2,...")
    name, raw_models = text.split(":", 1)
    models = [item.strip() for item in raw_models.split(",") if item.strip()]
    if not name.strip() or not models:
        raise argparse.ArgumentTypeError("Groups must include a nonempty name and at least one model")
    return name.strip(), models


def parse_args() -> argparse.Namespace:
    layout = CrossArchExperimentLayout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", default=str(layout.predictions_dir))
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to fusion/feature_builder, or fusion/fusers with --fit-fusers.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Named default groups to build. Defaults to all required groups.",
    )
    parser.add_argument(
        "--group",
        action="append",
        type=parse_group,
        default=[],
        help="Custom group formatted name:model1,model2. Repeat for multiple groups.",
    )
    parser.add_argument("--include-optional-groups", action="store_true")
    parser.add_argument("--splits", nargs="+", choices=STACK_SPLITS, default=list(STACK_SPLITS))
    parser.add_argument("--feature-modes", nargs="+", choices=ALL_FEATURE_MODES, default=list(ALL_FEATURE_MODES))
    parser.add_argument("--anchor-model-name", default=None)
    parser.add_argument("--quantile-bins", type=int, default=3)
    parser.add_argument(
        "--fit-fusers",
        action="store_true",
        help="Run Step 8 F0-F3 fusers instead of only building feature metadata.",
    )
    parser.add_argument(
        "--fusers",
        nargs="+",
        choices=DEFAULT_CROSSARCH_FUSERS,
        default=list(DEFAULT_CROSSARCH_FUSERS),
        help="Fusers to fit when --fit-fusers is set.",
    )
    parser.add_argument(
        "--c-grid",
        nargs="+",
        type=float,
        default=list(DEFAULT_C_GRID),
        help="Positive inverse-regularization values for logistic fusers.",
    )
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--min-bin-train-rows", type=int, default=2)
    parser.add_argument(
        "--skip-controls",
        action="store_true",
        help="Skip Step 9 label-permutation and row-shuffled negative controls.",
    )
    parser.add_argument("--control-seed", type=int, default=12345)
    parser.add_argument(
        "--control-feature-modes",
        nargs="+",
        choices=ALL_FEATURE_MODES,
        default=list(DEFAULT_CONTROL_FEATURE_MODES),
    )
    parser.add_argument("--control-warning-min-accuracy", type=float, default=0.20)
    parser.add_argument("--control-warning-chance-margin", type=float, default=0.10)
    parser.add_argument(
        "--write-feature-matrices",
        action="store_true",
        help="Persist full feature matrices as NPZ files. Default writes metadata and shapes only.",
    )
    parser.add_argument("--confirm-final-test", action="store_true", help="Required when final_test is requested.")
    args = parser.parse_args()
    if args.output_dir is None:
        leaf = "fusers" if args.fit_fusers else "feature_builder"
        args.output_dir = str(layout.fusion_dir / leaf)
    return args


def selected_groups(args: argparse.Namespace) -> Dict[str, List[str]]:
    default_groups = default_crossarch_feature_groups(include_optional=bool(args.include_optional_groups))
    groups: Dict[str, List[str]]
    if args.group:
        groups = {name: models for name, models in args.group}
    elif args.groups:
        missing = sorted(set(args.groups) - set(default_groups))
        if missing:
            raise SystemExit(f"Unknown default group(s): {missing}")
        groups = {name: default_groups[name] for name in args.groups}
    else:
        groups = default_groups
    validate_crossarch_feature_groups(groups)
    return groups


def main() -> int:
    args = parse_args()
    groups = selected_groups(args)
    if args.fit_fusers:
        config = CrossArchFusionFitConfig(
            prediction_dir=str(args.prediction_dir),
            output_dir=str(args.output_dir),
            groups=groups,
            splits=list(args.splits),
            feature_modes=list(args.feature_modes),
            include_optional_groups=bool(args.include_optional_groups),
            write_feature_matrices=bool(args.write_feature_matrices),
            confirm_final_test=bool(args.confirm_final_test),
            anchor_model_name=args.anchor_model_name,
            quantile_bins=int(args.quantile_bins),
            fusers=list(args.fusers),
            c_grid=[float(value) for value in args.c_grid],
            max_iter=int(args.max_iter),
            min_bin_train_rows=int(args.min_bin_train_rows),
            run_controls=not bool(args.skip_controls),
            control_seed=int(args.control_seed),
            control_feature_modes=list(args.control_feature_modes),
            control_warning_min_accuracy=float(args.control_warning_min_accuracy),
            control_warning_chance_margin=float(args.control_warning_chance_margin),
        )
        report = run_crossarch_fusers(config)
        report_path = Path(args.output_dir) / "fusion_report.json"
        print(f"Saved crossarch fusion report: {report_path}")
        print(f"overall_ok={report['ok']}")
        print(f"controls_ok={report['controls_summary']['ok']} audit_ok={report['audit_summary']['ok']}")
        for group_name, group_report in report["groups"].items():
            print(f"  {group_name}: {group_report['n_models']} models ok={group_report['ok']}")
            for fuser_name, fuser_report in group_report["fusers"].items():
                status = str(fuser_report.get("status", "ok"))
                final_metrics = fuser_report.get("metrics", {}).get("final_test", {})
                accuracy = final_metrics.get("accuracy")
                if accuracy is None:
                    print(f"    {fuser_name}: status={status}")
                else:
                    print(f"    {fuser_name}: status={status} final_test_acc={float(accuracy):.6f}")
    else:
        config = CrossArchFusionFeatureBuildConfig(
            prediction_dir=str(args.prediction_dir),
            output_dir=str(args.output_dir),
            groups=groups,
            splits=list(args.splits),
            feature_modes=list(args.feature_modes),
            include_optional_groups=bool(args.include_optional_groups),
            write_feature_matrices=bool(args.write_feature_matrices),
            confirm_final_test=bool(args.confirm_final_test),
            anchor_model_name=args.anchor_model_name,
            quantile_bins=int(args.quantile_bins),
        )
        report = run_crossarch_feature_builder(config)
        report_path = Path(args.output_dir) / "feature_build_report.json"
        print(f"Saved crossarch feature report: {report_path}")
        for group_name, group_report in report["groups"].items():
            print(f"  {group_name}: {group_report['n_models']} models")
            for split, split_report in group_report["splits"].items():
                shapes = {
                    mode: payload["shape"]
                    for mode, payload in split_report["features"].items()
                }
                print(f"    {split}: {shapes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
