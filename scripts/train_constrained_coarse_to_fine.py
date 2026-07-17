#!/usr/bin/env python3
"""Train one constrained coarse-to-fine B- or C-tier reconstructor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.constrained_coarse_to_fine import (  # noqa: E402
    CoarseToFineTrainConfig,
    train_coarse_to_fine_reconstructor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", dest="manifest_path", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--offline-cache-dir", required=True)
    parser.add_argument("--target-cache-dir", required=True)
    parser.add_argument("--variant", default="C5")
    parser.add_argument("--train-split", default="model_train")
    parser.add_argument("--val-split", default="model_val")
    parser.add_argument("--stack-val-split", default=None)
    parser.add_argument("--seed", type=int, default=22031)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--hlt-encoder-lr-scale", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--max-nonfinite-batches", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--runtime-profile",
        default="fp32_reference",
        choices=(
            "fp32_reference",
            "fp16_diagnostic",
            "bf16_calibration",
            "accelerated_candidate_v1",
            "accelerated_approved_v1",
        ),
        help="Named, hash-bound execution profile. Non-FP32 profiles become executable in acceleration Step 2.",
    )
    parser.add_argument(
        "--precision-mode",
        default=None,
        choices=("fp32", "bf16_forward_fp32_loss", "fp16_forward_fp32_loss"),
        help="Explicit forward/loss precision contract; defaults from --runtime-profile.",
    )
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--lr-schedule", choices=("constant", "warmup_cosine"), default="constant")
    parser.add_argument("--warmup-fraction", type=float, default=0.10)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument(
        "--fixed-horizon",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run all requested epochs and disable early stopping for certification jobs.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume only from an epoch-boundary last.pt checkpoint with an exact matching runtime/input contract.",
    )
    parser.add_argument("--hungarian-workers", type=int, default=1)
    parser.add_argument("--hungarian-executor", choices=("serial", "thread"), default="serial")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy FP16 AMP switch. Prefer --precision-mode; BF16 execution is added in acceleration Step 2.",
    )
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--max-train-jets", type=int, default=None)
    parser.add_argument("--max-val-jets", type=int, default=None)
    parser.add_argument("--max-stack-val-jets", type=int, default=None)
    parser.add_argument("--no-save-last-checkpoint", action="store_true")
    parser.add_argument(
        "--progress-interval-batches",
        type=int,
        default=100,
        help="Write training_progress.json and a log heartbeat every N finite batches.",
    )
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=8)
    parser.add_argument("--pool-layers", type=int, default=2)
    parser.add_argument("--decoder-layers-per-level", type=int, default=3)
    parser.add_argument("--ffn-multiplier", type=float, default=4.0)
    parser.add_argument("--pair-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--attention-dropout", type=float, default=0.05)
    parser.add_argument("--unconstrained-slot-accounting", action="store_true")
    parser.add_argument("--direct-particle-decoding", action="store_true")
    parser.add_argument("--hierarchy-loss-weight", type=float, default=1.0)
    parser.add_argument("--slot-loss-weight", type=float, default=1.0)
    parser.add_argument("--hierarchy-global-weight", type=float, default=1.0)
    parser.add_argument("--hierarchy-grid-weight", type=float, default=1.0)
    parser.add_argument("--hierarchy-relative-weight", type=float, default=0.25)
    parser.add_argument("--hierarchy-auxiliary-weight", type=float, default=0.25)
    parser.add_argument("--hierarchy-allocation-kl-weight", type=float, default=0.10)
    parser.add_argument("--hierarchy-uncertainty-weight", type=float, default=0.01)
    return parser


def main() -> int:
    args = vars(build_parser().parse_args())
    profile_precision = {
        "fp32_reference": "fp32",
        "fp16_diagnostic": "fp16_forward_fp32_loss",
        "bf16_calibration": "bf16_forward_fp32_loss",
        "accelerated_candidate_v1": "bf16_forward_fp32_loss",
        "accelerated_approved_v1": "bf16_forward_fp32_loss",
    }
    if args["precision_mode"] is None:
        args["precision_mode"] = profile_precision[args["runtime_profile"]]
    if bool(args["amp"]) and args["precision_mode"] == "fp32":
        args["runtime_profile"] = "fp16_diagnostic"
        args["precision_mode"] = "fp16_forward_fp32_loss"
    elif args["runtime_profile"] == "fp32_reference" and args["precision_mode"] != "fp32":
        args["runtime_profile"] = (
            "bf16_calibration"
            if args["precision_mode"] == "bf16_forward_fp32_loss"
            else "fp16_diagnostic"
        )
    args["verify_hash"] = not args.pop("no_verify_hash")
    args["pin_memory"] = not args.pop("no_pin_memory")
    args["save_last_checkpoint"] = not args.pop("no_save_last_checkpoint")
    args["constrain_slot_accounting"] = not args.pop("unconstrained_slot_accounting")
    report = train_coarse_to_fine_reconstructor(CoarseToFineTrainConfig(**args))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
