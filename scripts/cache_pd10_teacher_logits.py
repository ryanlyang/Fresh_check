#!/usr/bin/env python3
"""Cache PD10 HLT/offline teacher logits for model_train/model_val/final_test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.privileged_distill_10class import (  # noqa: E402
    PD10_PART_TEACHER_TARGETS,
    PD10_SPLIT_SIZES,
    PD10_TEACHER_LOGIT_SPLITS,
    PD10TeacherLogitCacheConfig,
    cache_pd10_teacher_logits,
    default_pd10_experiment_layout,
    normalize_pd10_part_teacher_target,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    layout = default_pd10_experiment_layout(output_root="checkpoints")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", choices=PD10_PART_TEACHER_TARGETS, required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Teacher checkpoint; defaults to the canonical PD10 teacher checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(layout.teacher_logits_dir),
        help="Root directory where teacher_logits/<model_name> blocks are written.",
    )
    parser.add_argument("--manifest", default=str(layout.split_manifest_path))
    parser.add_argument("--hlt-cache-dir", default=str(layout.hlt_cache_dir))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--splits", nargs="+", choices=PD10_TEACHER_LOGIT_SPLITS, default=list(PD10_TEACHER_LOGIT_SPLITS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-model-train-jets", type=int, default=PD10_SPLIT_SIZES["model_train"])
    parser.add_argument("--max-model-val-jets", type=int, default=PD10_SPLIT_SIZES["model_val"])
    parser.add_argument("--max-final-test-jets", type=int, default=PD10_SPLIT_SIZES["final_test"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--verify-label-branches", action="store_true")
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    parser.add_argument("--control-seed", type=int, default=7207)
    parser.add_argument("--no-verify-hlt-hash", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    teacher = normalize_pd10_part_teacher_target(args.teacher)
    layout = default_pd10_experiment_layout(output_root="checkpoints")
    checkpoint = args.checkpoint or str(layout.teacher_checkpoint(teacher))
    config = PD10TeacherLogitCacheConfig(
        teacher_target=teacher,
        checkpoint=checkpoint,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        hlt_cache_dir=args.hlt_cache_dir,
        data_dir=args.data_dir,
        splits=tuple(args.splits),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_model_train_jets=args.max_model_train_jets,
        max_model_val_jets=args.max_model_val_jets,
        max_final_test_jets=args.max_final_test_jets,
        overwrite=bool(args.overwrite),
        skip_existing=not bool(args.no_skip_existing),
        confirm_final_test=bool(args.confirm_final_test),
        verify_label_branches=bool(args.verify_label_branches),
        read_chunk_size=args.read_chunk_size,
        control_seed=args.control_seed,
        verify_hlt_hash=not bool(args.no_verify_hlt_hash),
    )
    report = cache_pd10_teacher_logits(config)
    print("pd10_teacher_logits_cached:")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
