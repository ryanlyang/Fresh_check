#!/usr/bin/env python3
"""Bind particle-view manifests to exact HLT/offline production cache files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    build_runtime_data_config,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--unified-manifest", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--offline-cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = build_runtime_data_config(
        parent_manifest_path=args.parent_manifest,
        unified_manifest_path=args.unified_manifest,
        hlt_cache_dir=args.hlt_cache_dir,
        offline_cache_dir=args.offline_cache_dir,
    )
    if not args.dry_run:
        write_immutable_json(args.output, config)
    print(
        f"parent_splits={len(config['parent_cache_records'])} "
        f"content_hash={config['content_hash']} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
