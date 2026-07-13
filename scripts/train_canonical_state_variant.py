#!/usr/bin/env python3
"""Canonical-state variant runner entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.canonical_state import (  # noqa: E402
    CANONICAL_STATE_VARIANT_REGISTRY_CONTRACT,
    CanonicalStateVariantRunConfig,
    canonical_state_required_dependencies,
    canonical_state_variant_spec,
    run_canonical_state_variant,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--phi-hlt-cache-dir", required=True)
    parser.add_argument("--phi-offline-cache-dir", default="")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--variant-root", default="")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--seed", type=int, default=10101)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--adapter-warmup-epochs", type=int, default=2)
    parser.add_argument("--part-lr", type=float, default=3.0e-5)
    parser.add_argument("--adapter-lr", type=float, default=3.0e-4)
    parser.add_argument("--predictor-lr", type=float, default=3.0e-4)
    parser.add_argument("--head-lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--max-train-jets", type=int, default=None)
    parser.add_argument("--max-val-jets", type=int, default=None)
    parser.add_argument("--max-stack-train-jets", type=int, default=None)
    parser.add_argument("--max-stack-val-jets", type=int, default=None)
    parser.add_argument("--max-final-test-jets", type=int, default=None)
    parser.add_argument("--model-size", default="base", choices=("tiny", "base", "large"))
    parser.add_argument(
        "--checkpoint-policy",
        default="all",
        choices=("all", "dependency", "none"),
        help="Which best_model_val.pt checkpoints to keep. 'dependency' keeps only A0/C0.",
    )
    parser.add_argument(
        "--no-save-last-checkpoint",
        action="store_true",
        help="Do not write last.pt after training; reports and prediction caches are still written.",
    )
    parser.add_argument(
        "--emit-planning-stub",
        action="store_true",
        help="Write a non-scientific planning run_report instead of training.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = canonical_state_variant_spec(args.run_id)
    output_dir = Path(args.output_dir)
    if bool(args.emit_planning_stub):
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "ok": True,
            "planning_stub": True,
            "not_scientific_output": True,
            "run_id": spec.run_id,
            "variant_contract": CANONICAL_STATE_VARIANT_REGISTRY_CONTRACT,
            "spec": spec.to_dict(),
            "required_dependencies": list(canonical_state_required_dependencies(spec.run_id)),
            "manifest": str(args.manifest),
            "hlt_cache_dir": str(args.hlt_cache_dir),
            "phi_hlt_cache_dir": str(args.phi_hlt_cache_dir),
            "phi_offline_cache_dir": str(args.phi_offline_cache_dir),
            "baseline_checkpoint": str(args.baseline_checkpoint),
            "confirm_final_test": bool(args.confirm_final_test),
        }
        (output_dir / "run_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    config = CanonicalStateVariantRunConfig(
        run_id=spec.run_id,
        output_dir=args.output_dir,
        manifest=args.manifest,
        hlt_cache_dir=args.hlt_cache_dir,
        phi_hlt_cache_dir=args.phi_hlt_cache_dir,
        phi_offline_cache_dir=args.phi_offline_cache_dir or None,
        baseline_checkpoint=args.baseline_checkpoint or None,
        variant_root=args.variant_root or None,
        confirm_final_test=bool(args.confirm_final_test),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        epochs=int(args.epochs),
        warmup_epochs=int(args.warmup_epochs),
        adapter_warmup_epochs=int(args.adapter_warmup_epochs),
        part_lr=float(args.part_lr),
        adapter_lr=float(args.adapter_lr),
        predictor_lr=float(args.predictor_lr),
        head_lr=float(args.head_lr),
        weight_decay=float(args.weight_decay),
        num_workers=int(args.num_workers),
        device=str(args.device),
        amp=not bool(args.disable_amp),
        grad_clip_norm=float(args.grad_clip_norm),
        early_stop_patience=int(args.early_stop_patience),
        max_train_jets=args.max_train_jets,
        max_val_jets=args.max_val_jets,
        max_stack_train_jets=args.max_stack_train_jets,
        max_stack_val_jets=args.max_stack_val_jets,
        max_final_test_jets=args.max_final_test_jets,
        model_size=str(args.model_size),
        checkpoint_policy=str(args.checkpoint_policy),
        save_last_checkpoint=not bool(args.no_save_last_checkpoint),
    )
    report = run_canonical_state_variant(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
