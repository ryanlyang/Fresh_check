#!/usr/bin/env python3
"""Audit PD10 HLT self-dualview HLT2 cache inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import load_split_manifest  # noqa: E402
from teacher_logit_reco.hlt_self_dualview import (  # noqa: E402
    HLT2_AUDIT_SPLITS,
    default_hlt2_audit_dir,
    default_hlt2_cache_dir,
    expected_counts_from_maxima,
    write_hlt2_audit_reports,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pd10-root",
        default=os.environ.get("PD10_ROOT"),
        help="Existing PD10 experiment root. Defaults to $PD10_ROOT when set.",
    )
    parser.add_argument("--manifest", default=None, help="Existing PD10 split manifest path")
    parser.add_argument("--source-hlt-cache-dir", default=None, help="Existing PD10 HLT cache directory")
    parser.add_argument("--hlt2-cache-dir", default=None, help="HLT2 cache directory")
    parser.add_argument("--output-dir", default=None, help="Directory for audit JSON/summary")
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(HLT2_AUDIT_SPLITS),
        choices=list(HLT2_AUDIT_SPLITS),
    )
    parser.add_argument("--expected-model-train", type=int, default=None)
    parser.add_argument("--expected-model-val", type=int, default=None)
    parser.add_argument("--expected-final-test", type=int, default=None)
    return parser.parse_args(argv)


def _pd10_root(args: argparse.Namespace) -> Path:
    if args.pd10_root:
        return Path(args.pd10_root)
    return Path("checkpoints") / "privileged_distill_10class_5m"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pd10_root = _pd10_root(args)
    manifest_path = Path(args.manifest) if args.manifest else pd10_root / "split_manifest" / "split_manifest.json.gz"
    source_hlt_cache_dir = Path(args.source_hlt_cache_dir) if args.source_hlt_cache_dir else pd10_root / "hlt_cache"
    hlt2_cache_dir = Path(args.hlt2_cache_dir) if args.hlt2_cache_dir else default_hlt2_cache_dir(
        pd10_root,
        args.strength,
    )
    output_dir = Path(args.output_dir) if args.output_dir else default_hlt2_audit_dir(pd10_root, args.strength)
    expected_counts = expected_counts_from_maxima(
        model_train=args.expected_model_train,
        model_val=args.expected_model_val,
        final_test=args.expected_final_test,
    )
    manifest = load_split_manifest(manifest_path)
    result = write_hlt2_audit_reports(
        manifest,
        source_hlt_cache_dir,
        hlt2_cache_dir,
        output_dir,
        strength=args.strength,
        splits=args.splits,
        expected_counts=expected_counts,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
