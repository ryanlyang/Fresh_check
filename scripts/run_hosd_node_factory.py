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
    parser.add_argument("--coordinate", type=int)
    parser.add_argument("--coordinate-start", type=int)
    parser.add_argument("--coordinate-stop", type=int)
    args = parser.parse_args(argv)
    if args.coordinate is not None:
        if args.coordinate_start is not None or args.coordinate_stop is not None:
            raise ValueError("single and batched coordinates are mutually exclusive")
        coordinates = range(args.coordinate, args.coordinate + 1)
    else:
        if (
            args.coordinate_start is None
            or args.coordinate_stop is None
            or args.coordinate_start < 0
            or args.coordinate_stop <= args.coordinate_start
        ):
            raise ValueError("batched coordinates require a positive half-open range")
        coordinates = range(args.coordinate_start, args.coordinate_stop)
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
        or runtime.get("runtime_support_sha256") is None
    ):
        raise ValueError("node runtime manifest lineage differs")
    matches = [
        row for row in registry["nodes"] if row["node_id"] == args.node_id
    ]
    if len(matches) != 1:
        raise ValueError("node factory registry coordinate differs")
    resolved = []
    inactive = []
    for coordinate in coordinates:
        command, row = resolve_node_argv(
            node=matches[0],
            runtime_manifest=runtime,
            campaign_root=args.campaign_root,
            coordinate=coordinate,
        )
        if command is None:
            inactive.append(coordinate)
            continue
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode:
            raise RuntimeError(
                f"registered node worker coordinate {coordinate} failed with "
                f"{completed.returncode}"
            )
        resolved.append({"coordinate": coordinate, "row": row})
    print(
        json.dumps(
            {
                "node_id": args.node_id,
                "coordinate_start": coordinates.start,
                "coordinate_stop": coordinates.stop,
                "resolved_rows": resolved,
                "inactive_upper_bound_slots": inactive,
                "scientific_rows_omitted": False,
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
