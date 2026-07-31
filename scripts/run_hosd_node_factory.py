#!/usr/bin/env python3
"""Resolve and execute one automatic HOSD node-factory coordinate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    RUNTIME_MANIFEST_CONTRACT,
    load_hashed_json,
)
from teacher_logit_reco.hlt_offline_structure_distillation.node_runtime import (  # noqa: E402
    resolve_node_argv,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--coordinate", required=True, type=int)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(
        args.campaign_root, repo_root=REPO_ROOT
    )
    registry = load_hashed_json(
        args.campaign_root / "registry" / "stage_job_registry.json",
        expected_contract="hosd_registry_v1",
    )
    runtime = load_hashed_json(
        args.campaign_root / "registry" / "runtime_manifest.json",
        expected_contract=RUNTIME_MANIFEST_CONTRACT,
    )
    if (
        runtime.get("campaign_spec_sha256") != campaign["content_hash"]
        or runtime.get("source") != campaign["source"]
        or runtime.get("execution_ready") is not True
        or runtime.get("missing_required_options_by_node")
    ):
        raise ValueError("node runtime manifest lineage differs")
    matches = [
        row for row in registry["nodes"] if row["node_id"] == args.node_id
    ]
    if len(matches) != 1:
        raise ValueError("node factory registry coordinate differs")
    command, row = resolve_node_argv(
        node=matches[0],
        runtime_manifest=runtime,
        campaign_root=args.campaign_root,
        coordinate=args.coordinate,
    )
    if command is None:
        print(
            json.dumps(
                {
                    "node_id": args.node_id,
                    "coordinate": args.coordinate,
                    "inactive_upper_bound_slot": True,
                    "scientific_row_omitted": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"registered node worker failed with {completed.returncode}"
        )
    print(
        json.dumps(
            {
                "node_id": args.node_id,
                "coordinate": args.coordinate,
                "resolved_row": row,
                "worker_entrypoint": matches[0]["entrypoint"],
                "scientific_performance_inspected": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
