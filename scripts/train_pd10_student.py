#!/usr/bin/env python3
"""Train one PD10 HLT-only student with optional privileged KD targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.privileged_distill_10class import (  # noqa: E402
    PD10_DEFAULT_ALPHA,
    PD10_DEFAULT_TEMPERATURE,
    PD10_SPLIT_SIZES,
    PD10_STUDENT_INIT_MODES,
    PD10_STUDENT_INIT_WARM_START,
    PD10_STUDENT_TARGET_MODES,
    PD10_TARGET_FULL_LOGITS,
    PD10_TEACHER_HLT,
    PD10_TEACHER_NONE,
    PD10_TEACHER_TARGETS,
    PD10_TOP_K,
    PD10StudentTrainConfig,
    default_pd10_experiment_layout,
    normalize_pd10_student_init_mode,
    normalize_pd10_teacher_target,
    pd10_student_dir,
    train_pd10_student,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    layout = default_pd10_experiment_layout(output_root="checkpoints")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-init", choices=PD10_STUDENT_INIT_MODES, required=True)
    parser.add_argument("--teacher-target", choices=PD10_TEACHER_TARGETS, required=True)
    parser.add_argument("--target-mode", choices=PD10_STUDENT_TARGET_MODES, default=PD10_TARGET_FULL_LOGITS)
    parser.add_argument("--temperature", type=float, default=PD10_DEFAULT_TEMPERATURE)
    parser.add_argument("--kd-alpha", type=float, default=PD10_DEFAULT_ALPHA)
    parser.add_argument("--kd-warmup-epochs", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=PD10_TOP_K)
    parser.add_argument("--hlt-cache-dir", default=str(layout.hlt_cache_dir))
    parser.add_argument(
        "--teacher-logit-cache",
        default=str(layout.teacher_logits_dir),
        help="Root containing cached teacher logits; ignored for teacher_target=none.",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        default=None,
        help="Warm-start checkpoint; defaults to the canonical HLT ParT teacher checkpoint for warm_start.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Student output directory; defaults to the canonical PD10 student variant directory.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-final-test-batches", type=int, default=None)
    parser.add_argument("--max-train-jets", type=int, default=PD10_SPLIT_SIZES["model_train"])
    parser.add_argument("--max-val-jets", type=int, default=PD10_SPLIT_SIZES["model_val"])
    parser.add_argument("--max-final-test-jets", type=int, default=PD10_SPLIT_SIZES["final_test"])
    parser.add_argument("--model-size", choices=["base", "tiny", "large"], default="base")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Required before evaluating the selected checkpoint on final_test.",
    )
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help="Debug escape hatch; canonical PD10 runs should not use this.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    layout = default_pd10_experiment_layout(output_root="checkpoints")
    student_init = normalize_pd10_student_init_mode(args.student_init)
    teacher_target = normalize_pd10_teacher_target(args.teacher_target)
    baseline_checkpoint = args.baseline_checkpoint
    if student_init == PD10_STUDENT_INIT_WARM_START and baseline_checkpoint is None:
        baseline_checkpoint = str(layout.teacher_checkpoint(PD10_TEACHER_HLT))
    teacher_logit_cache = None if teacher_target == PD10_TEACHER_NONE else args.teacher_logit_cache
    output_dir = args.output_dir or str(
        pd10_student_dir(
            student_init,
            teacher_target,
            args.target_mode,
            temperature=args.temperature,
            kd_alpha=args.kd_alpha,
            top_k=args.top_k,
            output_root="checkpoints",
        )
    )
    config_kwargs = {
        "student_init": student_init,
        "teacher_target": teacher_target,
        "output_dir": output_dir,
        "hlt_cache_dir": args.hlt_cache_dir,
        "teacher_logit_cache": teacher_logit_cache,
        "baseline_checkpoint": baseline_checkpoint,
        "target_mode": args.target_mode,
        "temperature": args.temperature,
        "kd_alpha": args.kd_alpha,
        "kd_warmup_epochs": args.kd_warmup_epochs,
        "top_k": args.top_k,
        "num_workers": args.num_workers,
        "device": args.device,
        "amp": not args.no_amp,
        "grad_clip_norm": args.grad_clip_norm,
        "early_stop_patience": args.early_stop_patience,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "max_final_test_batches": args.max_final_test_batches,
        "max_train_jets": args.max_train_jets,
        "max_val_jets": args.max_val_jets,
        "max_final_test_jets": args.max_final_test_jets,
        "model_size": args.model_size,
        "compile_model": args.compile_model,
        "confirm_final_test": args.confirm_final_test,
        "evaluate_final_test": not args.skip_final_test,
    }
    if args.seed is not None:
        config_kwargs["seed"] = args.seed
    if args.batch_size is not None:
        config_kwargs["batch_size"] = args.batch_size
    if args.epochs is not None:
        config_kwargs["epochs"] = args.epochs
    if args.lr is not None:
        config_kwargs["lr"] = args.lr
    if args.weight_decay is not None:
        config_kwargs["weight_decay"] = args.weight_decay

    config = PD10StudentTrainConfig(**config_kwargs)
    report = train_pd10_student(config)
    print("pd10_student_complete:")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
