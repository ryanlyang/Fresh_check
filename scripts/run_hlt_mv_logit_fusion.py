#!/usr/bin/env python3
"""Run cached-logit fusion for one HLT-MV model group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_multiview_source_fusion import (  # noqa: E402
    HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
    HLT_MV_FUSION_HLT_RANDOM_4SEED,
    HLT_MV_FUSION_PRETRAINED_DUALVIEW_4MODEL,
    HLT_MV_FUSION_SCRATCH_DUALVIEW_4MODEL,
    HLT_MV_FUSION_SOURCE_5VIEW,
    HLT_MV_SOURCE_PREDICTION_SPLITS,
    default_hlt_mv_logit_fusion_config,
    normalize_hlt_mv_source_prediction_splits,
    parse_hlt_mv_prediction_specs,
    run_hlt_mv_logit_fusion,
)


BUILTIN_FUSIONS = (
    HLT_MV_FUSION_SOURCE_5VIEW,
    HLT_MV_FUSION_HLT_RANDOM_4SEED,
    HLT_MV_FUSION_PRETRAINED_DUALVIEW_4MODEL,
    HLT_MV_FUSION_SCRATCH_DUALVIEW_4MODEL,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="checkpoints")
    parser.add_argument("--pdv3-experiment-name", default=HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME)
    parser.add_argument("--fusion-name", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        help="Optional explicit member spec, formatted as model_name=/path/to/predictions. Repeat for each member.",
    )
    parser.add_argument("--splits", nargs="+", default=list(HLT_MV_SOURCE_PREDICTION_SPLITS), choices=list(HLT_MV_SOURCE_PREDICTION_SPLITS))
    parser.add_argument("--skip-weighted-average", action="store_true")
    parser.add_argument("--max-weight-steps", type=int, default=30)
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_specs = parse_hlt_mv_prediction_specs(args.model_spec) if args.model_spec else None
    splits = normalize_hlt_mv_source_prediction_splits(args.splits)
    config = default_hlt_mv_logit_fusion_config(
        args.fusion_name,
        output_root=args.output_root,
        pdv3_experiment_name=args.pdv3_experiment_name,
        model_specs=model_specs,
        output_dir=None if args.output_dir is None else Path(args.output_dir),
        splits=splits,
        confirm_final_test=bool(args.confirm_final_test),
        overwrite=bool(args.overwrite),
        fit_weighted_average=not bool(args.skip_weighted_average),
        max_weight_steps=args.max_weight_steps,
    )
    report = run_hlt_mv_logit_fusion(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
