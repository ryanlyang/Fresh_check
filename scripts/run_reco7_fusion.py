#!/usr/bin/env python3
"""Run Step 10 reco7+HLT frozen prediction collection and fusion."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.fusion import DEFAULT_C_GRID, FusionRunConfig, run_reco7_fusion  # noqa: E402
from jetclass_fresh.hlt_baseline import save_json  # noqa: E402
from jetclass_fresh.reconstructor import RECONSTRUCTOR_VARIANT_NAMES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument(
        "--hlt-checkpoint",
        default="checkpoints/jetclass_fresh_hlt_baselines/single_hlt_seed101/best_model_val.pt",
    )
    parser.add_argument("--reco-root", default="checkpoints/jetclass_fresh_reco7")
    parser.add_argument("--output-dir", default="checkpoints/jetclass_fresh_fusion/reco7_plus_hlt")
    parser.add_argument("--variants", nargs="+", choices=RECONSTRUCTOR_VARIANT_NAMES, default=list(RECONSTRUCTOR_VARIANT_NAMES))
    parser.add_argument("--splits", nargs="+", choices=["stack_train", "stack_val", "final_test"], default=["stack_train", "stack_val", "final_test"])
    parser.add_argument("--confirm-final-test", action="store_true", help="Required when --splits includes final_test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-jets-per-split", type=int, default=None)
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--no-skip-existing-predictions", action="store_true")
    parser.add_argument("--C-grid", nargs="+", type=float, default=list(DEFAULT_C_GRID))
    parser.add_argument("--feature-mode", choices=["logits", "probs", "logits_probs"], default="logits_probs")
    parser.add_argument("--max-iter", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = FusionRunConfig(
        output_dir=args.output_dir,
        hlt_cache_dir=args.hlt_cache_dir,
        hlt_checkpoint=args.hlt_checkpoint,
        reco_root=args.reco_root,
        variants=list(args.variants),
        splits=list(args.splits),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_jets_per_split=args.max_jets_per_split,
        overwrite_predictions=args.overwrite_predictions,
        skip_existing_predictions=not args.no_skip_existing_predictions,
        confirm_final_test=args.confirm_final_test,
        C_grid=list(args.C_grid),
        feature_mode=args.feature_mode,
        max_iter=args.max_iter,
    )
    report = run_reco7_fusion(config)
    save_json(Path(args.output_dir) / "run_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
