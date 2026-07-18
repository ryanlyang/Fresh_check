#!/usr/bin/env python3
"""Build and audit deterministic adaptive-binary hierarchy target caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline import (  # noqa: E402
    ABPH_COMPACT_TARGET_CODEC_NAME,
    ABPH_HIERARCHY_GROUPINGS,
    ABPH_LEGACY_TARGET_CODEC_NAME,
    ABPH_TARGET_CACHE_SPLITS,
    ABPH_TARGET_STORAGE_CODECS,
    audit_adaptive_binary_target_cache,
    build_adaptive_binary_target_caches,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--offline-cache-dir", required=True)
    parser.add_argument("--output-cache-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=list(ABPH_TARGET_CACHE_SPLITS))
    parser.add_argument("--groupings", nargs="+", default=list(ABPH_HIERARCHY_GROUPINGS))
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--feature-dtype", choices=("float16", "float32"), default="float32")
    default_codec = (
        ABPH_COMPACT_TARGET_CODEC_NAME
        if os.environ.get("ABPH_STORAGE_PROFILE") == "streaming_30gb_v1"
        else ABPH_LEGACY_TARGET_CODEC_NAME
    )
    parser.add_argument(
        "--storage-codec",
        choices=ABPH_TARGET_STORAGE_CODECS,
        default=os.environ.get("ABPH_TARGET_STORAGE_CODEC", default_codec),
    )
    parser.add_argument("--forensic-jets-per-class", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report")
    return parser


def main() -> int:
    args = _parser().parse_args()
    cache_set = build_adaptive_binary_target_caches(
        manifest_path=args.manifest,
        hlt_cache_dir=args.hlt_cache_dir,
        offline_cache_dir=args.offline_cache_dir,
        output_cache_dir=args.output_cache_dir,
        splits=tuple(args.splits),
        groupings=tuple(args.groupings),
        chunk_size=args.chunk_size,
        feature_dtype=args.feature_dtype,
        storage_codec=args.storage_codec,
        forensic_jets_per_class=args.forensic_jets_per_class,
        overwrite=args.overwrite,
    )
    audit = audit_adaptive_binary_target_cache(
        args.output_cache_dir,
        manifest_path=args.manifest,
        splits=tuple(args.splits),
        groupings=tuple(args.groupings),
        verify_hash=True,
    )
    report = {"ok": bool(audit["ok"]), "cache_set": cache_set, "audit": audit}
    report_path = Path(args.report) if args.report else Path(args.output_cache_dir) / "step2_target_cache_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
