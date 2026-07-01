#!/usr/bin/env python3
"""Cache logits from trained AV10 architecture-view checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.architecture_view_part import (  # noqa: E402
    ARCHITECTURE_VIEW_10CLASS_DEFAULT_VARIANTS,
    ARCHITECTURE_VIEW_10CLASS_VARIANTS,
    ArchitectureView10ClassPredictionCacheConfig,
    cache_architecture_view_10class_predictions,
    normalize_architecture_view_variant,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--variants", nargs="+", choices=ARCHITECTURE_VIEW_10CLASS_VARIANTS, default=list(ARCHITECTURE_VIEW_10CLASS_DEFAULT_VARIANTS))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("model_val", "stack_train", "stack_val", "final_test"),
        default=("model_val", "stack_train", "stack_val", "final_test"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-model-val-jets", type=int, default=None)
    parser.add_argument("--max-stack-train-jets", type=int, default=None)
    parser.add_argument("--max-stack-val-jets", type=int, default=None)
    parser.add_argument("--max-final-test-jets", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--skip-hlt-hash-check", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--seed", type=int, default=7207)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variants = tuple(normalize_architecture_view_variant(variant) for variant in args.variants)
    config = ArchitectureView10ClassPredictionCacheConfig(
        output_dir=args.output_dir,
        hlt_cache_dir=args.hlt_cache_dir,
        checkpoint_root=args.checkpoint_root,
        variants=variants,
        splits=tuple(args.splits),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=args.device,
        max_model_val_jets=args.max_model_val_jets,
        max_stack_train_jets=args.max_stack_train_jets,
        max_stack_val_jets=args.max_stack_val_jets,
        max_final_test_jets=args.max_final_test_jets,
        overwrite=bool(args.overwrite),
        skip_existing=not bool(args.no_skip_existing),
        verify_hlt_hash=not bool(args.skip_hlt_hash_check),
        confirm_final_test=bool(args.confirm_final_test),
        seed=int(args.seed),
    )
    manifest = cache_architecture_view_10class_predictions(config)
    print("architecture_view_10class_prediction_cache_complete:")
    print(f"  output_dir: {args.output_dir}")
    print(f"  prediction_manifest: {Path(args.output_dir) / 'prediction_manifest.json'}")
    print(f"  variants: {' '.join(manifest['variants'])}")
    print(f"  splits: {' '.join(manifest['splits'])}")
    print(f"  rows: {len(manifest['prediction_rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
