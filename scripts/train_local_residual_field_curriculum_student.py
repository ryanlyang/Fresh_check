#!/usr/bin/env python3
"""Train a deployable local residual-field curriculum-distillation student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import LABEL_NAMES  # noqa: E402
from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    ALPHA_SCHEDULES,
    CURRICULUM_PILOT_RUN_IDS,
    CURRICULUM_STUDENT_KD_SOURCES,
    FIELD_GATE_MODES,
    FREEZE_SCHEDULES,
    RESIDUAL_PROJECTION_RESET_MODES,
    LocalResidualFieldCurriculumTrainConfig,
    train_local_residual_field_curriculum,
)


def _key_values(values: list[str], *, value_type, label: str) -> dict[str, object]:
    output: dict[str, object] = {}
    for raw in values:
        if "=" not in raw:
            raise argparse.ArgumentTypeError(f"{label} {raw!r} must have form name=value")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError(f"{label} name must not be empty")
        try:
            output[key] = value_type(value)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(f"invalid {label} value in {raw!r}") from exc
    return output


def _piecewise(values: list[str]) -> tuple[dict[str, object], ...]:
    output = _key_values(values, value_type=str, label="piecewise alpha")
    points: list[dict[str, object]] = []
    for epoch, raw_alpha in output.items():
        try:
            epoch_value = int(epoch)
            alpha: object = "selected_endpoint" if raw_alpha == "selected_endpoint" else float(raw_alpha)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid piecewise alpha {epoch}={raw_alpha}") from exc
        points.append({"epoch": epoch_value, "alpha": alpha})
    return tuple(sorted(points, key=lambda item: int(item["epoch"])))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, choices=CURRICULUM_PILOT_RUN_IDS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--target-cache-dir", required=True)
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--train-split", default="model_train")
    parser.add_argument("--val-split", default="model_val")
    parser.add_argument("--stack-val-split", default="stack_val")
    parser.add_argument("--final-test-split", default="final_test")
    parser.add_argument("--selected-consumer-json", default="")
    parser.add_argument("--consumer-id", choices=("Ofull", "Orobust_light"), default=None)
    parser.add_argument("--selected-alpha-endpoint", type=float, default=None)
    parser.add_argument("--confirm-paired-consumers", action="store_true")
    parser.add_argument("--oracle-teacher-checkpoint", default="")
    parser.add_argument("--oracle-teacher-config-path", default="")
    parser.add_argument("--oracle-run-report-path", default="")
    parser.add_argument("--oracle-forward-microbatch-size", type=int, default=None)
    parser.add_argument("--oracle-logit-only-fallback", action="store_true")
    parser.add_argument("--oracle-teacher-logits-dir", default="")
    parser.add_argument("--oracle-teacher-logits-path", action="append", default=[], metavar="SPLIT=PATH")
    parser.add_argument("--offline-teacher-logits-dir", default="")
    parser.add_argument("--offline-teacher-logits-path", action="append", default=[], metavar="SPLIT=PATH")
    parser.add_argument("--student-kd-source", choices=CURRICULUM_STUDENT_KD_SOURCES, default="oracle_true")
    parser.add_argument("--student-warm-start-checkpoint", default="")
    parser.add_argument("--predictor-warm-start-checkpoint", default="")
    parser.add_argument("--field-gate-mode", choices=FIELD_GATE_MODES, default=None)
    parser.add_argument("--initial-gate-bias-prob", type=float, default=0.1)
    parser.add_argument("--gate-reliability-error-scale", type=float, default=1.0)
    parser.add_argument("--residual-projection-reset", choices=RESIDUAL_PROJECTION_RESET_MODES, default=None)
    parser.add_argument("--residual-projection-scale", type=float, default=0.1)
    parser.add_argument("--freeze-schedule", choices=FREEZE_SCHEDULES, default=None)
    parser.add_argument("--freeze-phase1-epochs", type=int, default=2)
    parser.add_argument("--freeze-phase2-epochs", type=int, default=3)
    parser.add_argument("--optimizer-group-lr", action="append", default=[], metavar="GROUP=LR")
    parser.add_argument("--alpha-schedule", choices=ALPHA_SCHEDULES, default=None)
    parser.add_argument("--fixed-alpha", type=float, default=None)
    parser.add_argument("--piecewise-alpha", action="append", default=[], metavar="EPOCH=ALPHA")
    parser.add_argument("--sigmoid-alpha-start", type=float, default=0.25)
    parser.add_argument("--sigmoid-alpha-end", type=float, default=None)
    parser.add_argument("--sigmoid-alpha-midpoint", type=float, default=0.5)
    parser.add_argument("--sigmoid-alpha-sharpness", type=float, default=12.0)
    parser.add_argument("--loss-weight", action="append", default=[], metavar="NAME=WEIGHT")
    parser.add_argument("--loss-weight-schedule-json", default="")
    parser.add_argument("--seed", type=int, default=30421)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--max-train-jets", type=int, default=None)
    parser.add_argument("--max-val-jets", type=int, default=None)
    parser.add_argument("--max-stack-val-jets", type=int, default=None)
    parser.add_argument("--max-final-test-jets", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=len(LABEL_NAMES))
    parser.add_argument("--label-names", nargs="*", default=list(LABEL_NAMES))
    parser.add_argument("--model-size", choices=("tiny", "base", "large"), default="base")
    parser.add_argument("--reconstructor-d-model", type=int, default=160)
    parser.add_argument("--reconstructor-num-heads", type=int, default=5)
    parser.add_argument("--reconstructor-num-layers", type=int, default=4)
    parser.add_argument("--reconstructor-context-layers", type=int, default=1)
    parser.add_argument("--reconstructor-dropout", type=float, default=0.05)
    parser.add_argument("--reconstructor-attention-dropout", type=float, default=0.05)
    parser.add_argument("--residual-field-clip-value", type=float, default=8.0)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--field-huber-beta", type=float, default=0.1)
    parser.add_argument("--min-validation-valid-fraction", type=float, default=0.99)
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--allow-missing-manifest-match", action="store_true")
    parser.add_argument("--no-save-last-checkpoint", action="store_true")
    parser.add_argument("--evaluate-final-test", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    schedule = {}
    if args.loss_weight_schedule_json:
        schedule = json.loads(Path(args.loss_weight_schedule_json).read_text(encoding="utf-8"))
        if not isinstance(schedule, dict):
            raise ValueError("loss-weight schedule JSON must contain an object")
    config = LocalResidualFieldCurriculumTrainConfig(
        output_dir=args.output_dir,
        hlt_cache_dir=args.hlt_cache_dir,
        target_cache_dir=args.target_cache_dir,
        run_id=args.run_id,
        manifest_path=args.manifest_path or None,
        train_split=args.train_split,
        val_split=args.val_split,
        stack_val_split=args.stack_val_split,
        final_test_split=args.final_test_split,
        selected_consumer_json=args.selected_consumer_json or None,
        consumer_id=args.consumer_id,
        selected_alpha_endpoint=args.selected_alpha_endpoint,
        confirm_paired_consumers=bool(args.confirm_paired_consumers),
        oracle_teacher_checkpoint=args.oracle_teacher_checkpoint or None,
        oracle_teacher_config_path=args.oracle_teacher_config_path or None,
        oracle_run_report_path=args.oracle_run_report_path or None,
        oracle_forward_microbatch_size=args.oracle_forward_microbatch_size,
        oracle_logit_only_fallback=bool(args.oracle_logit_only_fallback),
        oracle_teacher_logits_dir=args.oracle_teacher_logits_dir or None,
        oracle_teacher_logits_paths=_key_values(args.oracle_teacher_logits_path, value_type=str, label="oracle logit path"),
        offline_teacher_logits_dir=args.offline_teacher_logits_dir or None,
        offline_teacher_logits_paths=_key_values(args.offline_teacher_logits_path, value_type=str, label="offline logit path"),
        student_kd_source=args.student_kd_source,
        student_warm_start_checkpoint=args.student_warm_start_checkpoint or None,
        predictor_warm_start_checkpoint=args.predictor_warm_start_checkpoint or None,
        field_gate_mode=args.field_gate_mode,
        initial_gate_bias_prob=args.initial_gate_bias_prob,
        gate_reliability_error_scale=args.gate_reliability_error_scale,
        residual_projection_reset=args.residual_projection_reset,
        residual_projection_scale=args.residual_projection_scale,
        freeze_schedule=args.freeze_schedule,
        freeze_phase1_epochs=args.freeze_phase1_epochs,
        freeze_phase2_epochs=args.freeze_phase2_epochs,
        optimizer_group_learning_rates=_key_values(args.optimizer_group_lr, value_type=float, label="optimizer group LR"),
        alpha_schedule=args.alpha_schedule,
        fixed_alpha=args.fixed_alpha,
        piecewise_alpha=_piecewise(args.piecewise_alpha),
        sigmoid_alpha_start=args.sigmoid_alpha_start,
        sigmoid_alpha_end=args.sigmoid_alpha_end,
        sigmoid_alpha_midpoint=args.sigmoid_alpha_midpoint,
        sigmoid_alpha_sharpness=args.sigmoid_alpha_sharpness,
        loss_weight_overrides=_key_values(args.loss_weight, value_type=float, label="loss weight"),
        loss_weight_schedule=schedule,
        seed=args.seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        num_workers=args.num_workers,
        device=args.device,
        amp=not bool(args.disable_amp),
        early_stop_patience=args.early_stop_patience,
        max_train_jets=args.max_train_jets,
        max_val_jets=args.max_val_jets,
        max_stack_val_jets=args.max_stack_val_jets,
        max_final_test_jets=args.max_final_test_jets,
        num_classes=args.num_classes,
        label_names=tuple(args.label_names),
        model_size=args.model_size,
        reconstructor_d_model=args.reconstructor_d_model,
        reconstructor_num_heads=args.reconstructor_num_heads,
        reconstructor_num_layers=args.reconstructor_num_layers,
        reconstructor_context_layers=args.reconstructor_context_layers,
        reconstructor_dropout=args.reconstructor_dropout,
        reconstructor_attention_dropout=args.reconstructor_attention_dropout,
        residual_field_clip_value=args.residual_field_clip_value,
        kd_temperature=args.kd_temperature,
        field_huber_beta=args.field_huber_beta,
        min_validation_valid_fraction=args.min_validation_valid_fraction,
        verify_hash=not bool(args.no_verify_hash),
        require_manifest_match=not bool(args.allow_missing_manifest_match),
        save_last_checkpoint=not bool(args.no_save_last_checkpoint),
        evaluate_final_test=bool(args.evaluate_final_test),
        confirm_final_test=bool(args.confirm_final_test),
    )
    report = train_local_residual_field_curriculum(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
