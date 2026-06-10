#!/usr/bin/env python3
"""Collect frozen heterogeneous-HLT predictions and run stacked fusion."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.fusion import DEFAULT_C_GRID, STACK_SPLITS  # noqa: E402
from jetclass_fresh.heterogeneous_hlt import (  # noqa: E402
    HETERO_HLT_ARCHITECTURES,
    HeterogeneousHLTFusionConfig,
    collect_heterogeneous_hlt_predictions,
    default_model_name_for_architecture,
    normalize_architecture_name,
)
from jetclass_fresh.hlt_baseline import save_json  # noqa: E402
from jetclass_fresh.independent_fusion import (  # noqa: E402
    FEATURE_MODES,
    IndependentFusionConfig,
    run_independent_fusion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument("--checkpoint-root", required=True, help="Directory containing <arch>/best_model_val.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--architectures", nargs="+", choices=HETERO_HLT_ARCHITECTURES, default=list(HETERO_HLT_ARCHITECTURES))
    parser.add_argument("--splits", nargs="+", choices=STACK_SPLITS, default=list(STACK_SPLITS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stack-train-size", type=int, default=150000)
    parser.add_argument("--stack-val-size", type=int, default=50000)
    parser.add_argument("--final-test-size", type=int, default=300000)
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--no-skip-existing-predictions", action="store_true")
    parser.add_argument("--feature-modes", nargs="+", choices=FEATURE_MODES, default=list(FEATURE_MODES))
    parser.add_argument("--c-grid", nargs="+", type=float, default=list(DEFAULT_C_GRID))
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--control-seed", type=int, default=12345)
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    architectures = [normalize_architecture_name(arch) for arch in args.architectures]
    model_names = [default_model_name_for_architecture(arch) for arch in architectures]
    config = HeterogeneousHLTFusionConfig(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        checkpoint_root=args.checkpoint_root,
        architectures=architectures,
        splits=list(args.splits),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=args.device,
        stack_train_size=int(args.stack_train_size),
        stack_val_size=int(args.stack_val_size),
        final_test_size=int(args.final_test_size),
        overwrite_predictions=bool(args.overwrite_predictions),
        skip_existing_predictions=not bool(args.no_skip_existing_predictions),
        confirm_final_test=bool(args.confirm_final_test),
        feature_modes=list(args.feature_modes),
        c_grid=[float(value) for value in args.c_grid],
        max_iter=int(args.max_iter),
        run_controls=not bool(args.skip_controls),
        control_seed=int(args.control_seed),
    )
    if "final_test" in config.splits and not config.confirm_final_test:
        raise SystemExit("Refusing to evaluate final_test without --confirm-final-test")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "heterogeneous_hlt_fusion_config.json",
        {
            "config": asdict(config),
            "model_names": model_names,
            "leakage_rules": {
                "model_training": "Each architecture trains on model_train and selects best checkpoint on model_val.",
                "stacker_fit_split": "stack_train",
                "stacker_selection_split": "stack_val",
                "final_test_evaluated_after_selection": True,
                "inputs": "cached fixed-HLT tokens only",
            },
        },
    )
    prediction_report = collect_heterogeneous_hlt_predictions(config)
    save_json(output_dir / "prediction_collection_report.json", prediction_report)

    if list(config.splits) != list(STACK_SPLITS):
        print("Prediction collection complete. Fusion skipped because not all stack splits were requested.")
        return 0

    fusion_config = IndependentFusionConfig(
        prediction_dir=str(output_dir / "predictions"),
        output_dir=str(output_dir / "fusion"),
        model_names=model_names,
        groups={"heterogeneous_hlt4": model_names},
        feature_modes=list(config.feature_modes or FEATURE_MODES),
        c_grid=list(config.c_grid or DEFAULT_C_GRID),
        max_iter=int(config.max_iter),
        confirm_final_test=bool(config.confirm_final_test),
        run_controls=bool(config.run_controls),
        control_seed=int(config.control_seed),
        singleton_models=model_names,
    )
    report = run_independent_fusion(fusion_config)
    print(f"Saved heterogeneous HLT fusion report: {output_dir / 'fusion' / 'fusion_report.json'}")
    for mode, mode_report in report["group_fusion_metrics"]["heterogeneous_hlt4"]["feature_modes"].items():
        val = mode_report["metrics"]["stack_val"]
        final = mode_report["metrics"]["final_test"]
        selected_c = mode_report["selection"]["selected_C"]
        print(
            f"  {mode:<12s} C={selected_c:g} "
            f"stack_val_acc={val['accuracy']:.6f} final_test_acc={final['accuracy']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
