#!/usr/bin/env python3
"""Evaluate/cache predictions for one selected PD10 HLT self-dualview model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_self_dualview import (  # noqa: E402
    HLTSDVEvalConfig,
    HLT_SDV_VARIANT_SAME_VIEW,
    default_hlt2_cache_dir,
    default_hlt_sdv_experiment_layout,
    evaluate_hlt_sdv_model,
    hlt_sdv_branch2_mode_from_variant,
    hlt_sdv_dual_hlt2_variant_name,
    hlt_sdv_strength_from_variant,
    normalize_hlt_sdv_variant,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pd10-root",
        default=os.environ.get("PD10_ROOT"),
        help="Existing PD10 root; defaults to $PD10_ROOT or checkpoints/privileged_distill_10class_5m.",
    )
    parser.add_argument("--variant", default=hlt_sdv_dual_hlt2_variant_name(0.20))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--hlt-cache-dir", default=None)
    parser.add_argument("--hlt2-cache-dir", default=None)
    parser.add_argument("--split", default="final_test", choices=["model_train", "model_val", "final_test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-jets", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _pd10_root(args: argparse.Namespace) -> Path:
    if args.pd10_root:
        return Path(args.pd10_root)
    return Path("checkpoints") / "privileged_distill_10class_5m"


def build_config(args: argparse.Namespace) -> HLTSDVEvalConfig:
    pd10_root = _pd10_root(args)
    variant = normalize_hlt_sdv_variant(args.variant)
    layout = default_hlt_sdv_experiment_layout(
        output_root=pd10_root.parent,
        pd10_experiment_name=pd10_root.name,
    )
    branch2_mode = hlt_sdv_branch2_mode_from_variant(variant)
    strength = hlt_sdv_strength_from_variant(variant)
    if args.hlt2_cache_dir:
        hlt2_cache_dir = args.hlt2_cache_dir
    elif variant == HLT_SDV_VARIANT_SAME_VIEW:
        hlt2_cache_dir = None
    else:
        hlt2_cache_dir = str(default_hlt2_cache_dir(pd10_root, float(strength)))
    output_dir = Path(args.output_dir) if args.output_dir else layout.variant_dir(variant)
    return HLTSDVEvalConfig(
        checkpoint=str(Path(args.checkpoint) if args.checkpoint else output_dir / "best_model_val.pt"),
        output_dir=str(output_dir),
        hlt_cache_dir=str(Path(args.hlt_cache_dir) if args.hlt_cache_dir else pd10_root / "hlt_cache"),
        hlt2_cache_dir=hlt2_cache_dir,
        variant_name=variant,
        branch2_mode=branch2_mode,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_jets=args.max_jets,
        max_batches=args.max_batches,
        confirm_final_test=bool(args.confirm_final_test),
        overwrite=bool(args.overwrite),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate_hlt_sdv_model(build_config(args))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
