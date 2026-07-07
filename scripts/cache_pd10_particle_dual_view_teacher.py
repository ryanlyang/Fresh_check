#!/usr/bin/env python3
"""Cache PD10-V2 particle dual-view teacher logits and representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.privileged_distill_10class import (  # noqa: E402
    PD10_PARTICLE_DUAL_VIEW_DEFAULT_EVAL_BATCH_SIZE,
    PD10_PARTICLE_DUAL_VIEW_LOGIT_SPLITS,
    PD10_SPLIT_SIZES,
    PD10ParticleDualViewTeacherCacheConfig,
    cache_pd10_particle_dual_view_teacher,
    default_pd10_experiment_layout,
    pd10_particle_dual_view_teacher_checkpoint,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    layout = default_pd10_experiment_layout(output_root="checkpoints")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(pd10_particle_dual_view_teacher_checkpoint(output_root="checkpoints")),
    )
    parser.add_argument("--manifest", default=str(layout.split_manifest_path))
    parser.add_argument("--hlt-cache-dir", default=str(layout.hlt_cache_dir))
    parser.add_argument("--offline-cache-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--logit-output-dir",
        default=str(layout.teacher_logits_dir),
        help="Root where particle dual-view prediction blocks are written.",
    )
    parser.add_argument(
        "--representation-output-dir",
        default=str(layout.root / "teacher_representations"),
        help="Root where particle dual-view representation blocks are written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=PD10_PARTICLE_DUAL_VIEW_LOGIT_SPLITS,
        default=list(PD10_PARTICLE_DUAL_VIEW_LOGIT_SPLITS),
    )
    parser.add_argument("--batch-size", type=int, default=PD10_PARTICLE_DUAL_VIEW_DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-model-train-jets", type=int, default=PD10_SPLIT_SIZES["model_train"])
    parser.add_argument("--max-model-val-jets", type=int, default=PD10_SPLIT_SIZES["model_val"])
    parser.add_argument("--max-final-test-jets", type=int, default=PD10_SPLIT_SIZES["final_test"])
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--verify-label-branches", action="store_true")
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    parser.add_argument("--control-seed", type=int, default=9901)
    parser.add_argument("--no-verify-hlt-hash", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = PD10ParticleDualViewTeacherCacheConfig(
        checkpoint=args.checkpoint,
        manifest_path=args.manifest,
        hlt_cache_dir=args.hlt_cache_dir,
        offline_cache_dir=args.offline_cache_dir,
        logit_output_dir=args.logit_output_dir,
        representation_output_dir=args.representation_output_dir,
        data_dir=args.data_dir,
        splits=tuple(args.splits),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_model_train_jets=args.max_model_train_jets,
        max_model_val_jets=args.max_model_val_jets,
        max_final_test_jets=args.max_final_test_jets,
        max_batches=args.max_batches,
        overwrite=bool(args.overwrite),
        skip_existing=not bool(args.no_skip_existing),
        confirm_final_test=bool(args.confirm_final_test),
        verify_label_branches=bool(args.verify_label_branches),
        read_chunk_size=args.read_chunk_size,
        control_seed=args.control_seed,
        verify_hlt_hash=not bool(args.no_verify_hlt_hash),
    )
    report = cache_pd10_particle_dual_view_teacher(config)
    print("pd10_particle_dual_view_teacher_cached:")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
