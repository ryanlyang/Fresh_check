#!/usr/bin/env python3
"""Write one fixed-row C5-B3/C6 accelerated-vs-FP32 tagger sanity report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.constrained_coarse_to_fine.runtime_tagger_sanity import (  # noqa: E402
    PairedBootstrapConfig,
    write_tagger_sanity_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", choices=("C5-B3", "C6"), required=True)
    parser.add_argument("--accelerated-tagger-dir", required=True)
    parser.add_argument("--fp32-tagger-dir", required=True)
    parser.add_argument("--accelerated-prediction-dir", required=True)
    parser.add_argument("--fp32-prediction-dir", required=True)
    parser.add_argument("--accelerated-model-name", required=True)
    parser.add_argument("--fp32-model-name", required=True)
    parser.add_argument("--accelerated-reconstructor-checkpoint", required=True)
    parser.add_argument("--fp32-reconstructor-checkpoint", required=True)
    parser.add_argument("--candidate-profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=48271)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=128)
    parser.add_argument("--max-accuracy-loss", type=float, default=0.005)
    parser.add_argument("--max-ce-increase", type=float, default=0.010)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = write_tagger_sanity_report(
        path=args.path,
        candidate_tagger_dir=args.accelerated_tagger_dir,
        reference_tagger_dir=args.fp32_tagger_dir,
        candidate_prediction_dir=args.accelerated_prediction_dir,
        reference_prediction_dir=args.fp32_prediction_dir,
        candidate_model_name=args.accelerated_model_name,
        reference_model_name=args.fp32_model_name,
        candidate_reconstructor_checkpoint=args.accelerated_reconstructor_checkpoint,
        reference_reconstructor_checkpoint=args.fp32_reconstructor_checkpoint,
        candidate_profile_path=args.candidate_profile,
        output_path=args.output,
        bootstrap=PairedBootstrapConfig(
            seed=args.bootstrap_seed,
            replicates=args.bootstrap_replicates,
            chunk_size=args.bootstrap_chunk_size,
            max_accuracy_loss=args.max_accuracy_loss,
            max_ce_increase=args.max_ce_increase,
        ),
    )
    print(json.dumps({"ok": report["ok"], "output": args.output}, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
