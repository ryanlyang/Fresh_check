#!/usr/bin/env python3
"""Run the source-bound HOSD research-compute resource preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    load_and_validate_campaign,
    run_resource_preflight,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    STORAGE_MEASUREMENT_CONTRACT,
    RESOURCE_MEASUREMENTS_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--storage-measurements", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resource-measurements", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    artifact = run_resource_preflight(
        campaign_root=args.campaign_root,
        storage_measurements=load_hashed_json(
            args.storage_measurements,
            expected_contract=STORAGE_MEASUREMENT_CONTRACT,
        ),
        profile=args.profile,
        source=campaign["source"],
        require_cuda=args.require_cuda,
        resource_measurements=(
            None
            if args.resource_measurements is None
            else load_hashed_json(
                args.resource_measurements,
                expected_contract=RESOURCE_MEASUREMENTS_CONTRACT,
            )
        ),
    )
    output = args.output or args.campaign_root / "job_ledgers" / "resource_preflight.json"
    publication = write_immutable_json(output, artifact)
    print(
        json.dumps(
            {
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
