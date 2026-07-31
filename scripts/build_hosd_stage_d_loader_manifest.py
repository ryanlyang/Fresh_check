#!/usr/bin/env python3
"""Bind one Stage-D row to exact HLT, label, target, tree, and normalizer inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_stage_d_plan,
    load_and_validate_campaign,
    resolve_stage_d_phase_two,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    SINGLE_FAMILY_PHASE_LOCK_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.hlt_offline_structure_distillation.stage_d_data_factory import (  # noqa: E402
    build_stage_d_loader_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--row-id", required=True)
    parser.add_argument(
        "--roles-json",
        required=True,
        type=Path,
        help="JSON object with model_train, val_stop, and design_select definitions",
    )
    parser.add_argument("--phase-lock", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    target_registry = load_hashed_json(
        args.campaign_root / "registry" / "structure_target_registry.json",
        expected_contract="hosd_structure_target_registry_v1",
    )
    plan = build_stage_d_plan(
        campaign_spec_sha256=campaign["content_hash"],
        target_registry=target_registry,
        source=campaign["source"],
    )
    rows = list(plan["all_rows"])
    if args.phase_lock is not None:
        lock = load_hashed_json(
            args.phase_lock, expected_contract=SINGLE_FAMILY_PHASE_LOCK_CONTRACT
        )
        rows.extend(resolve_stage_d_phase_two(stage_d_plan=plan, phase_lock=lock))
    matches = [row for row in rows if row["row_id"] == args.row_id and row["resolved"]]
    if len(matches) != 1:
        raise ValueError("loader-manifest row is absent, duplicated, or unresolved")
    definitions = json.loads(args.roles_json.read_text(encoding="utf-8"))
    artifact = build_stage_d_loader_manifest(
        row=matches[0],
        role_definitions=definitions,
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
