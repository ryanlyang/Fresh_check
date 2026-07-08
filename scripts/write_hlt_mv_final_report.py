#!/usr/bin/env python3
"""Write the aggregate HLT multiview source/fusion final report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_multiview_source_fusion import (  # noqa: E402
    HLTMVFinalReportConfig,
    HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME,
    HLT_MV_ROOT_DIRNAME,
    HLT_MV_TRIVIEW_MODEL_NAME,
    default_hlt_mv_experiment_layout,
    write_hlt_mv_final_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="checkpoints")
    parser.add_argument("--pdv3-experiment-name", default=HLT_MV_DEFAULT_PDV3_EXPERIMENT_NAME)
    parser.add_argument("--root-dirname", default=HLT_MV_ROOT_DIRNAME)
    parser.add_argument("--triview-model-name", default=HLT_MV_TRIVIEW_MODEL_NAME)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--require-triview", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    layout = default_hlt_mv_experiment_layout(
        output_root=args.output_root,
        pdv3_experiment_name=args.pdv3_experiment_name,
        root_dirname=args.root_dirname,
    )
    config = HLTMVFinalReportConfig(
        output_dir=str(Path(args.output_dir) if args.output_dir else layout.final_report_dir),
        output_root=str(Path(args.output_root)),
        pdv3_experiment_name=args.pdv3_experiment_name,
        root_dirname=args.root_dirname,
        triview_model_name=args.triview_model_name,
        allow_missing=bool(args.allow_missing),
        require_triview=bool(args.require_triview),
        overwrite=bool(args.overwrite),
    )
    report = write_hlt_mv_final_report(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
