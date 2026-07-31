#!/usr/bin/env python3
"""Bind a Stage-F graph to its selected member loader manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import build_combination_loader_manifest, load_and_validate_campaign  # noqa: E402
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import STAGE_F_PLAN_CONTRACT, load_hashed_json, write_immutable_json  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--graph-json", type=Path)
    parser.add_argument("--member-loader", action="append", default=[])
    parser.add_argument(
        "--native-relation-target",
        action="append",
        default=[],
        help="ROLE=NPZ; required for C_NATIVE_OFFLINE",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    if args.graph_json:
        graph = json.loads(args.graph_json.read_text(encoding="utf-8"))
    else:
        plan = load_hashed_json(args.campaign_root / "job_ledgers" / "stage_f_execution_plan.json", expected_contract=STAGE_F_PLAN_CONTRACT)
        matches = [row for row in plan["mandatory_combinations"] if row["graph_id"] == args.graph_id]
        if len(matches) != 1:
            raise ValueError("combination graph is absent or duplicated")
        graph = matches[0]
    members = dict(value.split("=", 1) for value in args.member_loader)
    native = dict(value.split("=", 1) for value in args.native_relation_target)
    artifact = build_combination_loader_manifest(
        graph=graph,
        member_loader_manifests=members,
        native_relation_target_files=native or None,
        campaign_spec_sha256=campaign["content_hash"],
        source=campaign["source"],
    )
    publication = write_immutable_json(args.output, artifact)
    print(json.dumps({"content_hash": artifact["content_hash"], "publication": publication["status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
