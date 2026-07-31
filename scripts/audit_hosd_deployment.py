#!/usr/bin/env python3
"""Fail-closed HOSD deployable-export coverage audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_deployable_export_audit,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    SCALE_EXECUTION_PLAN_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--export", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    plan = load_hashed_json(
        args.campaign_root / "scale_up" / "execution_plan.json",
        expected_contract=SCALE_EXECUTION_PLAN_CONTRACT,
    )
    artifact = build_deployable_export_audit(
        scale_plan=plan,
        export_rows=[load_hashed_json(path) for path in args.export],
        source=campaign["source"],
    )
    output = args.output or (
        args.campaign_root / "scale_up" / "export_audit.json"
    )
    publication = write_immutable_json(output, artifact)
    print(json.dumps({"content_hash": artifact["content_hash"], "publication": publication["status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
