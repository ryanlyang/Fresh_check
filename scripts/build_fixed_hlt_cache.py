#!/usr/bin/env python3
"""Generate and cache fixed HLT views for Step 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_cache import (  # noqa: E402
    DEFAULT_HLT_SEEDS,
    audit_hlt_cache,
    fixed_hlt_params_from_profile,
    fixed_hlt_params_dict,
    generate_and_cache_hlt_split,
    hlt_profile_version_from_params,
    normalize_hlt_profile,
)
from jetclass_fresh.jetclass_data import DEFAULT_DATA_DIR, SPLIT_ORDER, load_split_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Step 2 split manifest path (.json or .json.gz)",
    )
    parser.add_argument(
        "--data-dir",
        nargs="+",
        default=None,
        help=f"One or more JetClass data directories; defaults to manifest data_dir or {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--cache-dir",
        default="checkpoints/jetclass_fresh_hlt_cache",
        help="Directory for per-split HLT .npz files and metadata JSON",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLIT_ORDER),
        choices=list(SPLIT_ORDER),
        help="Splits to generate",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing HLT cache files")
    parser.add_argument("--show-progress", action="store_true", help="Show per-jet HLT progress if tqdm exists")
    parser.add_argument(
        "--verify-label-branches",
        action="store_true",
        help="Verify ROOT label branches agree with filename labels while loading offline views",
    )
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    parser.add_argument(
        "--hlt-profile",
        default="fixed_hlt_v1",
        choices=["fixed_hlt_v1", "fixed_hlt_v2_realistic"],
        help=(
            "HLT generation profile. fixed_hlt_v1 is the historical stress-test profile; "
            "fixed_hlt_v2_realistic is the new mild realistic profile."
        ),
    )
    parser.add_argument(
        "--hlt-degradation-strength",
        type=float,
        default=1.0,
        help=(
            "Scale the selected HLT profile. For fixed_hlt_v1, 1.0 is the original harsh "
            "fixed HLT. For fixed_hlt_v2_realistic, 1.0 is the realistic mild target."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_split_manifest(args.manifest)
    data_dir = args.data_dir if args.data_dir is not None else (manifest.data_dir or DEFAULT_DATA_DIR)
    data_dir_report = data_dir if isinstance(data_dir, list) else str(data_dir)
    hlt_profile = normalize_hlt_profile(args.hlt_profile)
    hlt_params = fixed_hlt_params_from_profile(hlt_profile, args.hlt_degradation_strength)
    hlt_profile_version = hlt_profile_version_from_params(hlt_params)
    reports = {}

    for split in args.splits:
        metadata = generate_and_cache_hlt_split(
            manifest,
            split,
            args.cache_dir,
            data_dir=data_dir,
            seed=DEFAULT_HLT_SEEDS[split],
            params=hlt_params,
            hlt_degradation_strength=args.hlt_degradation_strength,
            overwrite=args.overwrite,
            show_progress=args.show_progress,
            verify_label_branches=args.verify_label_branches,
            read_chunk_size=args.read_chunk_size,
        )
        reports[split] = {
            "array_path": metadata["array_path"],
            "metadata_path": metadata["metadata_path"],
            "n_jets": metadata["n_jets"],
            "seed": metadata["seed"],
            "hlt_profile": metadata["hlt_profile"],
            "hlt_profile_version": metadata["hlt_profile_version"],
            "hlt_degradation_strength": float(args.hlt_degradation_strength),
            "hlt_params": metadata["hlt_params"],
            "hlt_content_hash": metadata["hlt_content_hash"],
            "offline_constit_count_summary": metadata["offline_constit_count_summary"],
            "hlt_constit_count_summary": metadata["hlt_constit_count_summary"],
            "hlt_diagnostics_summary": metadata["hlt_diagnostics_summary"],
        }

    audit = audit_hlt_cache(
        manifest,
        args.cache_dir,
        splits=args.splits,
        expected_params=hlt_params,
        expected_hlt_profile=hlt_profile,
        expected_hlt_profile_version=hlt_profile_version,
        expected_hlt_degradation_strength=args.hlt_degradation_strength,
    )
    result = {
        "cache_dir": str(Path(args.cache_dir)),
        "data_dir": data_dir_report,
        "splits": list(args.splits),
        "hlt_profile": hlt_profile,
        "hlt_profile_version": hlt_profile_version,
        "hlt_degradation_strength": float(args.hlt_degradation_strength),
        "hlt_params": fixed_hlt_params_dict(hlt_params),
        "reports": reports,
        "audit": audit,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if audit["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
