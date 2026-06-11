#!/usr/bin/env python3
"""Run the first P-CNN teacher-logit reconstructor experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.pcnn_first_experiment import (  # noqa: E402
    DEFAULT_FIRST_EXPERIMENT_SPLITS,
    PredictionComparisonSpec,
    TeacherLogitParticleCnnFirstExperimentConfig,
    run_teacher_logit_particle_cnn_first_experiment,
)
from teacher_logit_reco.teachers import TEACHER_ARCHITECTURES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--manifest-path", default="checkpoints/jetclass_fresh_splits/split_manifest.json.gz")
    parser.add_argument("--hlt-cache-dir", default="checkpoints/jetclass_fresh_hlt_cache")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--teacher-architecture", choices=TEACHER_ARCHITECTURES, default="part")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_FIRST_EXPERIMENT_SPLITS))
    parser.add_argument("--seed", type=int, default=1205)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--predict-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--predict-num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--predict-device", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-predict-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-train-jets", type=int, default=50_000)
    parser.add_argument("--max-val-jets", type=int, default=10_000)
    parser.add_argument("--max-prediction-jets-per-split", type=int, default=50_000)
    parser.add_argument("--skip-hlt-hash-check", action="store_true")
    parser.add_argument("--verify-label-branches", action="store_true")
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--max-constits", type=int, default=128)
    parser.add_argument("--teacher-weight-threshold", type=float, default=0.0)

    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[5, 5, 3, 3, 3, 3])
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 1, 2, 4])
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--context-dims", "--context-mlp-dims", dest="context_mlp_dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--decoder-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--slot-dim", type=int, default=None)
    parser.add_argument("--num-extra-candidates", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-delta-logpt", type=float, default=1.0)
    parser.add_argument("--max-delta-eta", type=float, default=0.35)
    parser.add_argument("--max-delta-phi", type=float, default=0.35)
    parser.add_argument("--max-delta-loge", type=float, default=1.0)
    parser.add_argument("--parent-weight-bias", type=float, default=2.0)
    parser.add_argument("--extra-weight-bias", type=float, default=-3.0)
    parser.add_argument("--max-total-extra-pt-fraction", type=float, default=0.20)
    parser.add_argument("--max-extra-delta-eta", type=float, default=1.25)
    parser.add_argument("--max-extra-delta-phi", type=float, default=1.25)

    parser.add_argument("--teacher-kl-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.25)
    parser.add_argument("--correction-budget-weight", type=float, default=0.05)
    parser.add_argument("--jet-summary-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--no-skip-existing-predictions", action="store_true")
    parser.add_argument("--fit-final-stacker", action="store_true")
    parser.add_argument(
        "--comparison",
        nargs=3,
        action="append",
        metavar=("NAME", "PREDICTION_DIR", "MODEL_NAME"),
        default=[],
        help="Optional saved prediction row to compare against in the report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TeacherLogitParticleCnnFirstExperimentConfig(
        output_dir=args.output_dir,
        teacher_checkpoint=args.teacher_checkpoint,
        manifest_path=args.manifest_path,
        hlt_cache_dir=args.hlt_cache_dir,
        data_dir=args.data_dir,
        teacher_architecture=args.teacher_architecture,
        model_name=args.model_name,
        splits=list(args.splits),
        seed=args.seed,
        batch_size=args.batch_size,
        predict_batch_size=args.predict_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        predict_num_workers=args.predict_num_workers,
        device=args.device,
        predict_device=args.predict_device,
        amp=not bool(args.no_amp),
        predict_amp=not bool(args.no_predict_amp),
        grad_clip_norm=args.grad_clip_norm,
        early_stop_patience=args.early_stop_patience,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        max_train_jets=args.max_train_jets,
        max_val_jets=args.max_val_jets,
        max_prediction_jets_per_split=args.max_prediction_jets_per_split,
        verify_hlt_hash=not bool(args.skip_hlt_hash_check),
        verify_label_branches=args.verify_label_branches,
        read_chunk_size=args.read_chunk_size,
        compile_model=args.compile_model,
        max_constits=args.max_constits,
        teacher_weight_threshold=args.teacher_weight_threshold,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        kernel_sizes=tuple(args.kernel_sizes),
        dilations=tuple(args.dilations),
        context_dim=args.context_dim,
        context_mlp_dims=tuple(args.context_mlp_dims),
        decoder_dims=tuple(args.decoder_dims),
        slot_dim=args.slot_dim,
        num_extra_candidates=args.num_extra_candidates,
        dropout=args.dropout,
        max_delta_logpt=args.max_delta_logpt,
        max_delta_eta=args.max_delta_eta,
        max_delta_phi=args.max_delta_phi,
        max_delta_loge=args.max_delta_loge,
        parent_weight_bias=args.parent_weight_bias,
        extra_weight_bias=args.extra_weight_bias,
        max_total_extra_pt_fraction=args.max_total_extra_pt_fraction,
        max_extra_delta_eta=args.max_extra_delta_eta,
        max_extra_delta_phi=args.max_extra_delta_phi,
        teacher_kl_weight=args.teacher_kl_weight,
        ce_weight=args.ce_weight,
        correction_budget_weight=args.correction_budget_weight,
        jet_summary_weight=args.jet_summary_weight,
        temperature=args.temperature,
        confirm_final_test=bool(args.confirm_final_test),
        overwrite_output=bool(args.overwrite_output),
        overwrite_predictions=bool(args.overwrite_predictions),
        skip_existing_predictions=not bool(args.no_skip_existing_predictions),
        fit_final_stacker=bool(args.fit_final_stacker),
        comparison_specs=[
            PredictionComparisonSpec(name=name, prediction_dir=prediction_dir, model_name=model_name)
            for name, prediction_dir, model_name in args.comparison
        ],
    )
    report = run_teacher_logit_particle_cnn_first_experiment(config)
    print("teacher_logit_particle_cnn_first_experiment_complete:")
    print(f"  output_dir: {report['output_dir']}")
    print(f"  model_name: {report['model_name']}")
    print(f"  train_checkpoint: {report['train_report']['checkpoint']}")
    print(f"  prediction_dir: {report['prediction_dir']}")
    print(f"  report: {Path(args.output_dir) / 'first_experiment_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
