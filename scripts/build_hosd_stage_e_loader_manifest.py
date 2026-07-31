#!/usr/bin/env python3
"""Build a source-bound Stage-E loader/intervention manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_stage_d_loader_manifest,
    build_stage_e_loader_manifest,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    STAGE_E_PLAN_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--base-loader-manifest", type=Path)
    parser.add_argument(
        "--roles-json",
        type=Path,
        help="Stage-D-compatible model_train/val_stop/design_select definitions",
    )
    parser.add_argument(
        "--intervention",
        action="append",
        default=[],
        metavar="ROLE=NPZ",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    plan = load_hashed_json(
        args.campaign_root / "job_ledgers" / "stage_e_execution_plan.json",
        expected_contract=STAGE_E_PLAN_CONTRACT,
    )
    rows = [row for row in plan["all_rows"] if row["row_id"] == args.row_id]
    if len(rows) != 1:
        raise ValueError("Stage-E row is absent or duplicated")
    sources = {}
    for value in args.intervention:
        role, separator, path = value.partition("=")
        if not separator or role in sources:
            raise ValueError("intervention arguments must be unique ROLE=NPZ")
        sources[role] = {"npz_path": str(Path(path).resolve())}
    if (args.base_loader_manifest is None) == (args.roles_json is None):
        raise ValueError("provide exactly one base loader manifest or roles JSON")
    base_manifest = args.base_loader_manifest
    if args.roles_json is not None:
        definitions = json.loads(args.roles_json.read_text(encoding="utf-8"))
        base = build_stage_d_loader_manifest(
            row=rows[0],
            role_definitions=definitions,
            campaign_spec_sha256=campaign["content_hash"],
            source=campaign["source"],
        )
        base_manifest = args.output.with_name(f"{args.output.stem}.base.json")
        write_immutable_json(base_manifest, base)
    artifact = build_stage_e_loader_manifest(
        row=rows[0],
        base_loader_manifest=base_manifest,
        intervention_sources=sources,
        campaign_spec_sha256=campaign["content_hash"],
        source=campaign["source"],
    )
    publication = write_immutable_json(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "content_hash": artifact["content_hash"],
                "publication": publication["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
