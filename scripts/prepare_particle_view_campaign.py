#!/usr/bin/env python3
"""Publish Step-1 particle-view split contracts from a parent manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import load_split_manifest  # noqa: E402
from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    LOCKED_PARTICLE_VIEW_500K_SPLIT_CONFIG,
    PARTICLE_VIEW_COORDINATE_BINDING_CONTRACT,
    PARTICLE_VIEW_DEPLOYMENT_CONTRACT,
    PARTICLE_VIEW_LABEL_EXPOSURE_LEDGER_CONTRACT,
    PARTICLE_VIEW_REGISTRY_CONTRACT,
    PARTICLE_VIEW_STEP1_REPORT_CONTRACT,
    PARTICLE_VIEW_TRAINABLE_COMPONENTS,
    PARTICLE_VIEW_UNIFIED_SPLIT_CONTRACT,
    audit_unified_split_manifest,
    build_unified_split_manifest,
    miniature_parent_manifest,
    miniature_split_config,
    sha256_file,
    with_content_hash,
    write_immutable_json,
    write_unified_split_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--parent-manifest")
    source.add_argument(
        "--debug-miniature",
        action="store_true",
        help="Use the deterministic in-memory miniature fixture.",
    )
    parser.add_argument("--debug-rows-per-class", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    if args.debug_miniature:
        parent_path = None
        parent = miniature_parent_manifest(
            rows_per_class=args.debug_rows_per_class
        )
        config = miniature_split_config(
            rows_per_class=args.debug_rows_per_class
        )
    else:
        parent_path = Path(args.parent_manifest).resolve()
        parent = load_split_manifest(parent_path)
        config = LOCKED_PARTICLE_VIEW_500K_SPLIT_CONFIG
    unified = build_unified_split_manifest(
        parent, config=config
    )
    audit = audit_unified_split_manifest(
        unified,
        parent=parent,
        config=config,
    )
    manifest_path = output_dir / "unified_split_manifest.json"
    report = with_content_hash(
        {
            "contract": PARTICLE_VIEW_STEP1_REPORT_CONTRACT,
            "status": "READY",
            "parent_manifest_path": (
                None if parent_path is None else str(parent_path)
            ),
            "parent_manifest_file_sha256": (
                None if parent_path is None else sha256_file(parent_path)
            ),
            "production_eligible": not args.debug_miniature,
            "unified_split_manifest_path": str(manifest_path),
            "unified_split_manifest_sha256": unified["content_hash"],
            "split_config_sha256": unified["split_config_sha256"],
            "training_identity_sha256": audit["training_identity_sha256"],
            "single_training_pool": audit["single_training_pool"],
            "cross_fit": audit["cross_fit"],
            "trainable_component_count": audit[
                "trainable_component_count"
            ],
            "trainable_components": list(PARTICLE_VIEW_TRAINABLE_COMPONENTS),
            "unused_parent_splits": audit["unused_parent_splits"],
            "logical_splits": audit["logical_splits"],
        }
    )
    report_path = output_dir / "particle_view_step1_report.json"
    schema_catalog = with_content_hash(
        {
            "contract": "particle_view_step1_schema_catalog_v1",
            "schemas": {
                "unified_split_manifest": PARTICLE_VIEW_UNIFIED_SPLIT_CONTRACT,
                "coordinate_binding": PARTICLE_VIEW_COORDINATE_BINDING_CONTRACT,
                "campaign_registry": PARTICLE_VIEW_REGISTRY_CONTRACT,
                "deployment_manifest": PARTICLE_VIEW_DEPLOYMENT_CONTRACT,
                "label_exposure_ledger": (
                    PARTICLE_VIEW_LABEL_EXPOSURE_LEDGER_CONTRACT
                ),
            },
            "training_topology": "single_pool_no_crossfit_v1",
        }
    )
    catalog_path = output_dir / "particle_view_step1_schema_catalog.json"
    if args.dry_run:
        publications = {
            name: {
                "path": str(path),
                "content_hash": artifact["content_hash"],
                "status": "dry_run",
            }
            for name, path, artifact in (
                ("manifest", manifest_path, unified),
                ("schema_catalog", catalog_path, schema_catalog),
                ("report", report_path, report),
            )
        }
    else:
        publications = {
            "manifest": write_unified_split_manifest(manifest_path, unified),
            "schema_catalog": write_immutable_json(catalog_path, schema_catalog),
            "report": write_immutable_json(report_path, report),
        }
    print(
        json.dumps(
            {
                "status": "READY",
                **publications,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
