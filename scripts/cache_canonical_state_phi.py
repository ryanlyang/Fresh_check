#!/usr/bin/env python3
"""Build Canonical Multi-Scale Jet State Phi caches from HLT or offline caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.canonical_state import (  # noqa: E402
    CANONICAL_STATE_PHI_HLT_SOURCE,
    CANONICAL_STATE_PHI_HLT_SPLITS,
    CANONICAL_STATE_PHI_OFFLINE_PRIMARY_SPLITS,
    CANONICAL_STATE_PHI_OFFLINE_SOURCE,
    audit_canonical_phi_cache,
    build_phi_cache_from_hlt_cache,
    build_phi_cache_from_offline_cache,
    normalize_phi_source_view,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-view",
        required=True,
        choices=[CANONICAL_STATE_PHI_HLT_SOURCE, CANONICAL_STATE_PHI_OFFLINE_SOURCE],
        help="Build Phi_hlt or Phi_offline cache.",
    )
    parser.add_argument(
        "--input-cache-dir",
        required=True,
        help="Existing raw-token HLT cache dir or offline cache dir.",
    )
    parser.add_argument(
        "--output-cache-dir",
        required=True,
        help="Destination directory for canonical Phi cache files.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional split manifest path. When provided, audit the written Phi cache.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to cache. Defaults to all HLT splits or primary offline splits.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-final-test-offline-oracle",
        action="store_true",
        help="Allow writing/loading offline final_test Phi as explicit oracle diagnostics.",
    )
    parser.add_argument("--expected-model-train", type=int, default=None)
    parser.add_argument("--expected-model-val", type=int, default=None)
    parser.add_argument("--expected-stack-train", type=int, default=None)
    parser.add_argument("--expected-stack-val", type=int, default=None)
    parser.add_argument("--expected-final-test", type=int, default=None)
    return parser.parse_args(argv)


def _expected_size_overrides(args: argparse.Namespace) -> dict[str, int] | None:
    values = {
        "model_train": args.expected_model_train,
        "model_val": args.expected_model_val,
        "stack_train": args.expected_stack_train,
        "stack_val": args.expected_stack_val,
        "final_test": args.expected_final_test,
    }
    if not any(value is not None for value in values.values()):
        return None
    from teacher_logit_reco.canonical_state import canonical_state_split_sizes

    defaults = canonical_state_split_sizes()
    return {
        split: int(values[split] if values[split] is not None else defaults[split])
        for split in defaults
    }


def main() -> int:
    args = parse_args()
    source = normalize_phi_source_view(args.source_view)
    default_splits = (
        CANONICAL_STATE_PHI_HLT_SPLITS
        if source == CANONICAL_STATE_PHI_HLT_SOURCE
        else CANONICAL_STATE_PHI_OFFLINE_PRIMARY_SPLITS
    )
    splits = tuple(args.splits or default_splits)
    if source == CANONICAL_STATE_PHI_HLT_SOURCE:
        reports = build_phi_cache_from_hlt_cache(
            args.input_cache_dir,
            args.output_cache_dir,
            splits=splits,
            overwrite=bool(args.overwrite),
        )
    else:
        reports = build_phi_cache_from_offline_cache(
            args.input_cache_dir,
            args.output_cache_dir,
            splits=splits,
            overwrite=bool(args.overwrite),
            allow_final_test_offline_oracle=bool(args.allow_final_test_offline_oracle),
        )

    audit = None
    if args.manifest:
        audit = audit_canonical_phi_cache(
            args.output_cache_dir,
            source_view=source,
            manifest_path=args.manifest,
            splits=splits,
            expected_split_sizes=_expected_size_overrides(args),
            allow_oracle_final_test=bool(args.allow_final_test_offline_oracle),
        )
    result = {
        "ok": bool(audit is None or audit.get("ok")),
        "source_view": source,
        "input_cache_dir": str(args.input_cache_dir),
        "output_cache_dir": str(args.output_cache_dir),
        "splits": list(splits),
        "reports": reports,
        "audit": audit,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
