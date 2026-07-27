#!/usr/bin/env python3
"""Compile immutable Step-10 ABPH evidence into one promotion decision."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline.runtime_acceptance import (  # noqa: E402
    ABPH_RUNTIME_REPRESENTATIVE_VARIANTS,
    build_runtime_acceptance_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-root-run", required=True)
    parser.add_argument("--single-deep-run", required=True)
    parser.add_argument("--ddp4-root-run", required=True)
    parser.add_argument("--ddp4-deep-run", required=True)
    parser.add_argument("--ddp8-root-run")
    parser.add_argument("--ddp8-deep-run")
    parser.add_argument("--single-smoke", required=True)
    parser.add_argument("--ddp4-smoke", required=True)
    parser.add_argument("--ddp8-smoke")
    parser.add_argument("--ddp4-root-batch-contract", required=True)
    parser.add_argument("--ddp4-deep-batch-contract", required=True)
    parser.add_argument("--ddp8-root-batch-contract")
    parser.add_argument("--ddp8-deep-batch-contract")
    parser.add_argument("--single-path-acceptance", required=True)
    parser.add_argument("--root-extension-report")
    parser.add_argument("--deep-extension-report")
    parser.add_argument("--optimized-pilot-report")
    parser.add_argument("--expected-validation-jets", type=int, default=4_096)
    parser.add_argument("--output", required=True)
    return parser


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root_variant, deep_variant = ABPH_RUNTIME_REPRESENTATIVE_VARIANTS
    extensions = None
    if args.root_extension_report or args.deep_extension_report:
        if not args.root_extension_report or not args.deep_extension_report:
            raise ValueError("both representative extension reports are required together")
        extensions = {
            root_variant: args.root_extension_report,
            deep_variant: args.deep_extension_report,
        }
    report = build_runtime_acceptance_report(
        single_run_dirs={
            root_variant: args.single_root_run,
            deep_variant: args.single_deep_run,
        },
        ddp4_run_dirs={
            root_variant: args.ddp4_root_run,
            deep_variant: args.ddp4_deep_run,
        },
        ddp8_run_dirs=(
            {
                root_variant: args.ddp8_root_run,
                deep_variant: args.ddp8_deep_run,
            }
            if args.ddp8_root_run and args.ddp8_deep_run
            else None
        ),
        single_smoke_path=args.single_smoke,
        ddp4_smoke_path=args.ddp4_smoke,
        ddp8_smoke_path=args.ddp8_smoke,
        ddp4_batch_contracts={
            root_variant: args.ddp4_root_batch_contract,
            deep_variant: args.ddp4_deep_batch_contract,
        },
        ddp8_batch_contracts=(
            {
                root_variant: args.ddp8_root_batch_contract,
                deep_variant: args.ddp8_deep_batch_contract,
            }
            if args.ddp8_root_batch_contract and args.ddp8_deep_batch_contract
            else None
        ),
        single_path_acceptance=args.single_path_acceptance,
        extension_reports=extensions,
        optimized_pilot_report=args.optimized_pilot_report,
        expected_validation_jets=args.expected_validation_jets,
    )
    _atomic_json(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
