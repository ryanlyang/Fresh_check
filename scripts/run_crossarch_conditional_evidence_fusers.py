#!/usr/bin/env python3
"""Run conditional evidence fusers from frozen crossarch prediction blocks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.fusion import DEFAULT_C_GRID  # noqa: E402
from teacher_logit_reco.conditional_evidence_fusers import (  # noqa: E402
    ConditionalEvidenceFuserConfig,
    SUITES,
    run_conditional_evidence_fusers,
)
from teacher_logit_reco.crossarch_experiment import (  # noqa: E402
    DIRECT_HLT_ARCHITECTURES,
    RECONSTRUCTOR_ARCHITECTURES,
    TEACHER_ARCHITECTURES,
    build_reco_domain_tagger_model_names,
    hlt_model_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument(
        "--hlt-models",
        nargs="+",
        default=[hlt_model_name(arch) for arch in DIRECT_HLT_ARCHITECTURES],
    )
    parser.add_argument(
        "--adapted-models",
        nargs="+",
        default=list(build_reco_domain_tagger_model_names(RECONSTRUCTOR_ARCHITECTURES, TEACHER_ARCHITECTURES)),
    )
    parser.add_argument("--c-grid", nargs="+", type=float, default=list(DEFAULT_C_GRID))
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--residual-penalties", nargs="+", type=float, default=[0.0, 0.001, 0.01, 0.05, 0.1])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[0.0, 0.0001])
    parser.add_argument("--confusion-pair-counts", nargs="+", type=int, default=[4, 8, 12, 20])
    parser.add_argument("--neural-epochs", type=int, default=40)
    parser.add_argument("--neural-batch-size", type=int, default=8192)
    parser.add_argument("--neural-lr", type=float, default=0.001)
    parser.add_argument("--neural-hidden-dims", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--neural-dropout", type=float, default=0.05)
    parser.add_argument("--neural-device", default="auto")
    parser.add_argument("--neural-patience", type=int, default=8)
    parser.add_argument("--control-seed", type=int, default=12345)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ConditionalEvidenceFuserConfig(
        prediction_dir=args.prediction_dir,
        output_dir=args.output_dir,
        suite=args.suite,
        hlt_model_names=tuple(args.hlt_models),
        adapted_model_names=tuple(args.adapted_models),
        c_grid=tuple(float(value) for value in args.c_grid),
        max_iter=int(args.max_iter),
        confirm_final_test=bool(args.confirm_final_test),
        run_controls=not bool(args.skip_controls),
        control_seed=int(args.control_seed),
        residual_penalties=tuple(float(value) for value in args.residual_penalties),
        weight_decays=tuple(float(value) for value in args.weight_decays),
        confusion_pair_counts=tuple(int(value) for value in args.confusion_pair_counts),
        neural_epochs=int(args.neural_epochs),
        neural_batch_size=int(args.neural_batch_size),
        neural_lr=float(args.neural_lr),
        neural_hidden_dims=tuple(int(value) for value in args.neural_hidden_dims),
        neural_dropout=float(args.neural_dropout),
        neural_device=str(args.neural_device),
        neural_patience=int(args.neural_patience),
    )
    report = run_conditional_evidence_fusers(config)
    print(f"Saved conditional evidence fuser report: {Path(args.output_dir) / 'conditional_fuser_report.json'}")
    print(f"ok={report['ok']} suite={report['suite']}")
    for row in report["method_summary"]:
        acc = row.get("final_test_accuracy")
        if acc is None:
            print(f"  {row['method']}: status={row.get('status')}")
        else:
            print(
                f"  {row['method']}: "
                f"stack_val_acc={float(row['stack_val_accuracy']):.6f} "
                f"final_test_acc={float(acc):.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
