#!/usr/bin/env python3
"""Run independent fusion from saved model-loading-demo prediction blocks.

This script starts from frozen logits/probabilities already saved by
``demo_load_and_score_models_no_fusion.py``.  It does not load neural-network
checkpoints and it does not regenerate predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.fusion import DEFAULT_C_GRID  # noqa: E402
from jetclass_fresh.independent_fusion import (  # noqa: E402
    FEATURE_MODES,
    IndependentFusionConfig,
    default_groups_for_models,
    discover_prediction_models,
    run_independent_fusion,
    validate_fusion_groups,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-dir",
        default="checkpoints/jetclass_fresh_model_loading_demo_with_final_test/predictions",
        help="Directory containing <model>/<split>_predictions.npz blocks.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/jetclass_fresh_independent_fusion",
        help="Fresh output directory for fusion reports.",
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=None,
        help="Models to include in raw reports and alignment audits. Defaults to complete prediction directories.",
    )
    parser.add_argument(
        "--group",
        action="append",
        type=parse_group,
        default=[],
        help="Fusion group formatted name:model1,model2. Repeat for multiple groups.",
    )
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        choices=FEATURE_MODES,
        default=list(FEATURE_MODES),
        help="Stacker feature modes to evaluate.",
    )
    parser.add_argument("--c-grid", nargs="+", type=float, default=list(DEFAULT_C_GRID))
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--confirm-final-test", action="store_true", help="Required to evaluate final_test.")
    parser.add_argument("--skip-controls", action="store_true", help="Skip negative controls for faster debugging.")
    parser.add_argument("--control-seed", type=int, default=12345)
    parser.add_argument(
        "--singleton-models",
        nargs="+",
        default=None,
        help="Optional subset for singleton stacker audits. Defaults to all deployable models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prediction_dir = Path(args.prediction_dir)
    model_names = list(args.model_names or discover_prediction_models(prediction_dir))
    groups: Dict[str, List[str]]
    if args.group:
        groups = {name: models for name, models in args.group}
    else:
        groups = default_groups_for_models(model_names)
    if not groups:
        raise SystemExit("No deployable fusion groups could be inferred. Pass --group name:model1,model2.")
    validate_fusion_groups(groups, model_names)

    config = IndependentFusionConfig(
        prediction_dir=str(prediction_dir),
        output_dir=str(args.output_dir),
        model_names=model_names,
        groups=groups,
        feature_modes=list(args.feature_modes),
        c_grid=[float(value) for value in args.c_grid],
        max_iter=int(args.max_iter),
        confirm_final_test=bool(args.confirm_final_test),
        run_controls=not bool(args.skip_controls),
        control_seed=int(args.control_seed),
        singleton_models=None if args.singleton_models is None else list(args.singleton_models),
    )
    report = run_independent_fusion(config)

    print(f"Saved fusion report: {Path(args.output_dir) / 'fusion_report.json'}")
    print("Fusion groups:")
    for group_name, group_report in report["group_fusion_metrics"].items():
        print(f"  {group_name}: {' '.join(group_report['model_names'])}")
        for mode, mode_report in group_report["feature_modes"].items():
            final = mode_report["metrics"]["final_test"]
            val = mode_report["metrics"]["stack_val"]
            selected_c = mode_report["selection"]["selected_C"]
            print(
                f"    {mode:<12s} C={selected_c:g} "
                f"stack_val_acc={val['accuracy']:.6f} final_test_acc={final['accuracy']:.6f}"
            )
    flags = report.get("suspicious_flags") or []
    if flags:
        print("Suspicious flags:")
        for flag in flags:
            print(f"  {flag['name']}: {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
