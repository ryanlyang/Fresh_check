#!/usr/bin/env python3
"""Build or dry-run the immutable HOSD Step-1 campaign control plane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import load_split_manifest  # noqa: E402
from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_step1_bundle,
    load_hashed_json,
    publish_step1_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge import (  # noqa: E402
    RetbSplitConfig,
    miniature_storage_measurements,
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.storage import (  # noqa: E402
    STORAGE_MEASUREMENTS_CONTRACT,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--storage-measurements", type=Path)
    parser.add_argument(
        "--inherited-parent",
        action="append",
        default=[],
        metavar="PARENT_ID=PATH",
    )
    parser.add_argument("--miniature", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _parent_paths(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        parent_id, separator, raw_path = value.partition("=")
        if not separator or not parent_id or not raw_path:
            raise ValueError("--inherited-parent must be PARENT_ID=PATH")
        if parent_id in output:
            raise ValueError(f"duplicate inherited parent {parent_id!r}")
        output[parent_id] = Path(raw_path)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = (
        RetbSplitConfig.miniature(validation_role_per_class=2)
        if args.miniature
        else RetbSplitConfig.production()
    )
    if args.storage_measurements is None:
        if not args.miniature:
            raise ValueError("production HOSD requires --storage-measurements")
        measurements = miniature_storage_measurements()
    else:
        measurements = load_hashed_json(
            args.storage_measurements,
            expected_contract=STORAGE_MEASUREMENTS_CONTRACT,
        )
    manifest = load_split_manifest(args.parent_manifest)
    bundle = build_step1_bundle(
        campaign_id=args.campaign_id,
        campaign_root=args.output_dir,
        manifest=manifest,
        split_config=config,
        source_snapshot=source_snapshot(REPO_ROOT),
        storage_measurements=measurements,
        inherited_parent_paths=_parent_paths(args.inherited_parent),
    )
    result: dict[str, object] = {
        "campaign_id": args.campaign_id,
        "campaign_profile": config.profile,
        "campaign_spec_sha256": bundle["campaign_spec"]["content_hash"],
        "dry_run": bool(args.dry_run),
        "stage_job_counts": bundle["dry_run_plan"]["stage_job_counts"],
        "job_count": bundle["dry_run_plan"]["job_count"],
        "artifact_count": bundle["dry_run_plan"]["artifact_count"],
        "unresolved_parent_count": bundle["parent_rebuild_plan"][
            "unresolved_parent_count"
        ],
        "scientific_outputs_created": False,
    }
    if not args.dry_run:
        result["publication"] = publish_step1_bundle(
            campaign_root=args.output_dir,
            manifest=manifest,
            bundle=bundle,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
