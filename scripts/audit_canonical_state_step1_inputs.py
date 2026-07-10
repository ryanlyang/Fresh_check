#!/usr/bin/env python3
"""Verify Canonical Multi-Scale Jet State split and HLT v2 cache artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.canonical_state import (  # noqa: E402
    canonical_state_split_sizes,
    default_canonical_state_experiment_layout,
    write_canonical_state_step1_input_audit_reports,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    layout = default_canonical_state_experiment_layout(output_root="checkpoints")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(layout.split_manifest_path),
        help="Five-way split manifest path (.json or .json.gz)",
    )
    parser.add_argument(
        "--hlt-cache-dir",
        default=str(layout.hlt_cache_dir),
        help="Directory containing per-split fixed-HLT .npz and metadata JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(layout.step1_audit_dir),
        help="Directory for CMS-JS Step 1 audit reports",
    )
    parser.add_argument("--expected-model-train", type=int, default=None)
    parser.add_argument("--expected-model-val", type=int, default=None)
    parser.add_argument("--expected-stack-train", type=int, default=None)
    parser.add_argument("--expected-stack-val", type=int, default=None)
    parser.add_argument("--expected-final-test", type=int, default=None)
    return parser.parse_args(argv)


def expected_size_overrides(args: argparse.Namespace) -> dict[str, int] | None:
    defaults = canonical_state_split_sizes()
    values = {
        "model_train": args.expected_model_train,
        "model_val": args.expected_model_val,
        "stack_train": args.expected_stack_train,
        "stack_val": args.expected_stack_val,
        "final_test": args.expected_final_test,
    }
    if not any(value is not None for value in values.values()):
        return None
    return {
        split: int(values[split] if values[split] is not None else defaults[split])
        for split in defaults
    }


def main() -> int:
    args = parse_args()
    expected_sizes = expected_size_overrides(args)
    result = write_canonical_state_step1_input_audit_reports(
        Path(args.manifest),
        Path(args.hlt_cache_dir),
        Path(args.output_dir),
        expected_split_sizes=expected_sizes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
