#!/usr/bin/env python3
"""Bind site-local caches/checkpoints to an immutable bridge execution spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.bridge_execution import (  # noqa: E402
    build_prediction_anchored_execution_spec,
    write_prediction_anchored_execution_spec,
)
from teacher_logit_reco.local_particle_residual_field.bridge_consumer import (  # noqa: E402
    ConsumerCampaignConfig,
)
from teacher_logit_reco.local_particle_residual_field.bridge_r0 import (  # noqa: E402
    StreamedR0TrainConfig,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--child-manifest", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--offline-cache-dir", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preprocessing-sha256", default="")
    parser.add_argument("--target-schema-sha256", default="")
    parser.add_argument("--r0-epochs", type=int, default=60)
    parser.add_argument("--r0-seed", type=int, default=10421)
    parser.add_argument("--r0-lr", type=float, default=3.0e-4)
    parser.add_argument("--r0-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--r0-early-stop-patience", type=int, default=8)
    parser.add_argument("--r0-device", default="auto")
    parser.add_argument("--consumer-baseline-steps", type=int, default=10_000)
    parser.add_argument("--consumer-bridge-steps", type=int, default=2_000)
    parser.add_argument("--consumer-batch-size", type=int, default=128)
    parser.add_argument("--consumer-evaluation-interval", type=int, default=200)
    parser.add_argument("--consumer-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--consumer-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--consumer-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--consumer-model-size", choices=("tiny", "base", "large"), default="base")
    parser.add_argument(
        "--consumer-data-profile",
        choices=("pilot_250k", "high_data_3m"),
        default="pilot_250k",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    r0_config = StreamedR0TrainConfig(
        output_dir="__RUNTIME_OUTPUT_DIR__",
        epochs=int(args.r0_epochs),
        seed=int(args.r0_seed),
        lr=float(args.r0_lr),
        weight_decay=float(args.r0_weight_decay),
        early_stop_patience=int(args.r0_early_stop_patience),
        device=str(args.r0_device),
    )
    consumer_config = ConsumerCampaignConfig(
        baseline_steps=int(args.consumer_baseline_steps),
        bridge_finetune_steps=int(args.consumer_bridge_steps),
        batch_size=int(args.consumer_batch_size),
        evaluation_interval_steps=int(args.consumer_evaluation_interval),
        learning_rate=float(args.consumer_learning_rate),
        weight_decay=float(args.consumer_weight_decay),
        grad_clip_norm=float(args.consumer_grad_clip_norm),
        model_size=str(args.consumer_model_size),
        data_profile=str(args.consumer_data_profile),
    )
    spec = build_prediction_anchored_execution_spec(
        parent_manifest_path=args.parent_manifest,
        child_manifest_path=args.child_manifest,
        hlt_cache_dir=args.hlt_cache_dir,
        offline_cache_dir=args.offline_cache_dir,
        baseline_checkpoint_path=args.baseline_checkpoint,
        r0_config=r0_config,
        consumer_config=consumer_config,
        preprocessing_sha256=args.preprocessing_sha256 or None,
        target_schema_sha256=args.target_schema_sha256 or None,
    )
    publication = None
    if not args.dry_run:
        publication = write_prediction_anchored_execution_spec(args.output, spec)
    print(
        json.dumps(
            {
                "ok": True,
                "dry_run": bool(args.dry_run),
                "output": str(args.output),
                "execution_spec_sha256": spec["content_hash"],
                "source_splits": list(spec["sources"]),
                "publication": publication,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
