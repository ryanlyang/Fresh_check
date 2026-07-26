#!/usr/bin/env python3
"""Build the immutable Step-1 relational Particle Transformer campaign bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import (  # noqa: E402
    RELATIONAL_STORAGE_MEASUREMENTS_CONTRACT,
    RelationalSplitConfig,
    StorageMeasurements,
    build_and_publish_step1,
    build_storage_measurements,
    sha256_file,
    validate_content_hash,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument(
        "--storage-measurements",
        required=True,
        type=Path,
        help="JSON object containing representative measured byte/count fields",
    )
    parser.add_argument("--available-bytes", type=int)
    parser.add_argument("--hlt-cache-dir", type=Path)
    parser.add_argument("--require-hlt-cache", action="store_true")
    parser.add_argument("--budget-gib", type=int, default=20)
    parser.add_argument("--minimum-free-reserve-gib", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--miniature",
        action="store_true",
        help=(
            "Explicit non-production test profile (20/10/0/10/20); "
            "artifacts are marked ineligible for scientific results"
        ),
    )
    return parser


def _available_bytes(output_dir: Path) -> int:
    probe = output_dir.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return int(shutil.disk_usage(probe).free)


def _authenticate_evidence(
    raw_evidence: object,
    *,
    base_dir: Path,
) -> dict:
    if not isinstance(raw_evidence, dict):
        raise ValueError("source_evidence must be a JSON object")
    evidence = {}
    for name, raw in raw_evidence.items():
        if not isinstance(raw, dict):
            raise ValueError(f"source_evidence.{name} must be an object")
        source_path = Path(str(raw["path"]))
        if not source_path.is_absolute():
            source_path = base_dir / source_path
        if source_path.is_symlink() or not source_path.is_file():
            raise FileNotFoundError(
                f"storage evidence file is absent or unsafe: {source_path}"
            )
        actual = {
            "path": str(source_path.resolve()),
            "sha256": sha256_file(source_path),
            "bytes": int(source_path.stat().st_size),
            "purpose": str(raw["purpose"]),
        }
        if "sha256" in raw and str(raw["sha256"]) != actual["sha256"]:
            raise ValueError(f"storage evidence hash mismatch for {name}")
        if "bytes" in raw and int(raw["bytes"]) != actual["bytes"]:
            raise ValueError(f"storage evidence byte-count mismatch for {name}")
        evidence[str(name)] = actual
    return evidence


def _load_measurements(path: Path) -> object:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("storage-measurements must contain a JSON object")
    if payload.get("contract") == RELATIONAL_STORAGE_MEASUREMENTS_CONTRACT:
        validate_content_hash(
            payload, expected_contract=RELATIONAL_STORAGE_MEASUREMENTS_CONTRACT
        )
        evidence = _authenticate_evidence(
            payload["source_evidence"], base_dir=path.parent
        )
        rebuilt = build_storage_measurements(
            StorageMeasurements.from_mapping(payload["measurements"]),
            source_evidence=evidence,
        )
        if rebuilt["content_hash"] != payload["content_hash"]:
            raise ValueError(
                "storage measurement artifact does not match authenticated evidence"
            )
        return rebuilt
    if set(payload) >= {"measurements", "source_evidence"}:
        evidence = _authenticate_evidence(
            payload["source_evidence"], base_dir=path.parent
        )
        return build_storage_measurements(
            StorageMeasurements.from_mapping(payload["measurements"]),
            source_evidence=evidence,
        )
    raise ValueError(
        "storage measurements must be either a hashed "
        "relational_part_storage_measurements_v1 artifact or contain "
        "{measurements, source_evidence}; unbound byte estimates are forbidden"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    measurements = _load_measurements(args.storage_measurements)
    available_bytes = (
        _available_bytes(args.output_dir)
        if args.available_bytes is None
        else int(args.available_bytes)
    )
    split_config = (
        RelationalSplitConfig.miniature()
        if args.miniature
        else RelationalSplitConfig.production()
    )
    result = build_and_publish_step1(
        parent_manifest_path=args.parent_manifest,
        campaign_root=args.output_dir,
        campaign_id=args.campaign_id,
        measurements=measurements,
        available_bytes=available_bytes,
        repo_root=REPO_ROOT,
        split_config=split_config,
        hlt_cache_dir=args.hlt_cache_dir,
        require_hlt_cache=bool(args.require_hlt_cache),
        budget_gib=int(args.budget_gib),
        minimum_free_reserve_gib=int(args.minimum_free_reserve_gib),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
