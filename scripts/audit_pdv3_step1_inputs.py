#!/usr/bin/env python3
"""Verify PDV3 HLT0.2 split, fixed-HLT cache, and paired offline cache artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.privileged_distill_v3 import (  # noqa: E402
    default_pdv3_experiment_layout,
    pdv3_model_split_sizes,
    pdv3_stack_placeholder_split_sizes,
    write_pdv3_step1_input_audit_reports,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    layout = default_pdv3_experiment_layout(output_root="checkpoints")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(layout.split_manifest_path),
        help="PDV3 five-way split manifest path (.json or .json.gz)",
    )
    parser.add_argument(
        "--hlt-cache-dir",
        default=str(layout.hlt_cache_dir),
        help="Directory containing per-split fixed-HLT .npz and metadata JSON files",
    )
    parser.add_argument(
        "--offline-cache-dir",
        default=str(layout.offline_cache_dir),
        help="Directory containing per-split offline .npz and metadata JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(layout.step1_audit_dir),
        help="Directory for PDV3 Step 1 audit reports",
    )
    parser.add_argument("--expected-model-train", type=int, default=None)
    parser.add_argument("--expected-model-val", type=int, default=None)
    parser.add_argument("--expected-final-test", type=int, default=None)
    parser.add_argument("--expected-stack-train", type=int, default=None)
    parser.add_argument("--expected-stack-val", type=int, default=None)
    return parser.parse_args(argv)


def expected_size_overrides(args: argparse.Namespace) -> tuple[dict[str, int] | None, dict[str, int] | None]:
    model_defaults = pdv3_model_split_sizes()
    stack_defaults = pdv3_stack_placeholder_split_sizes()
    model_values = {
        "model_train": args.expected_model_train,
        "model_val": args.expected_model_val,
        "final_test": args.expected_final_test,
    }
    stack_values = {
        "stack_train": args.expected_stack_train,
        "stack_val": args.expected_stack_val,
    }
    expected_model = None
    expected_stack = None
    if any(value is not None for value in model_values.values()):
        expected_model = {
            split: int(model_values[split] if model_values[split] is not None else model_defaults[split])
            for split in model_defaults
        }
    if any(value is not None for value in stack_values.values()):
        expected_stack = {
            split: int(stack_values[split] if stack_values[split] is not None else stack_defaults[split])
            for split in stack_defaults
        }
    return expected_model, expected_stack


def main() -> int:
    args = parse_args()
    expected_model, expected_stack = expected_size_overrides(args)
    result = write_pdv3_step1_input_audit_reports(
        Path(args.manifest),
        Path(args.hlt_cache_dir),
        Path(args.offline_cache_dir),
        Path(args.output_dir),
        expected_split_sizes=expected_model,
        expected_placeholder_sizes=expected_stack,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
