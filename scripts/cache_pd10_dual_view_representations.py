#!/usr/bin/env python3
"""Cache PD10 dual-view logit-fusion teacher hidden representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.privileged_distill_10class import (  # noqa: E402
    PD10_DUAL_VIEW_DEFAULT_EVAL_BATCH_SIZE,
    PD10_TEACHER_LOGIT_SPLITS,
    PD10_SPLIT_SIZES,
    PD10DualViewRepresentationCacheConfig,
    cache_pd10_dual_view_teacher_representations,
    default_pd10_experiment_layout,
    pd10_dual_view_teacher_checkpoint,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    layout = default_pd10_experiment_layout(output_root="checkpoints")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(pd10_dual_view_teacher_checkpoint(output_root="checkpoints")))
    parser.add_argument("--teacher-logit-dir", default=str(layout.teacher_logits_dir))
    parser.add_argument(
        "--output-dir",
        default=str(layout.root / "teacher_representations"),
        help="Root where dual-view representation blocks are written.",
    )
    parser.add_argument("--splits", nargs="+", choices=PD10_TEACHER_LOGIT_SPLITS, default=list(PD10_TEACHER_LOGIT_SPLITS))
    parser.add_argument("--batch-size", type=int, default=PD10_DUAL_VIEW_DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-model-train-jets", type=int, default=PD10_SPLIT_SIZES["model_train"])
    parser.add_argument("--max-model-val-jets", type=int, default=PD10_SPLIT_SIZES["model_val"])
    parser.add_argument("--max-final-test-jets", type=int, default=PD10_SPLIT_SIZES["final_test"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = PD10DualViewRepresentationCacheConfig(
        checkpoint=args.checkpoint,
        teacher_logit_dir=args.teacher_logit_dir,
        output_dir=args.output_dir,
        splits=tuple(args.splits),
        batch_size=args.batch_size,
        device=args.device,
        max_model_train_jets=args.max_model_train_jets,
        max_model_val_jets=args.max_model_val_jets,
        max_final_test_jets=args.max_final_test_jets,
        overwrite=bool(args.overwrite),
        skip_existing=not bool(args.no_skip_existing),
        confirm_final_test=bool(args.confirm_final_test),
    )
    report = cache_pd10_dual_view_teacher_representations(config)
    print("pd10_dual_view_representations_cached:")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
