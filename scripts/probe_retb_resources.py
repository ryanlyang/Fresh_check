#!/usr/bin/env python3
"""Measure Tigris CPU/GPU/storage admission for an RETB campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge import (  # noqa: E402
    build_resource_probe,
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.storage import (  # noqa: E402
    validate_storage_measurements,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _memory_bytes() -> int:
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return pages * page_size
    except (AttributeError, OSError, ValueError):
        return 0


def _backend_passed(payload: dict) -> bool:
    candidates = (
        payload.get("canonical_smoke_parity", {}).get("passed"),
        payload.get("compiled_backend_parity", {}).get("passed"),
        payload.get("checks", {}).get("compiled_region_backend_parity"),
    )
    values = [value for value in candidates if value is not None]
    if len(values) != 1:
        raise ValueError("compiled REGION parity artifact is ambiguous")
    return bool(values[0])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--resource-kind", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--compiled-region-parity", required=True, type=Path)
    parser.add_argument("--requested-memory-bytes", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    measurements = load_hashed_json(
        args.campaign_root / "storage_measurements.json"
    )
    measurement_sha = validate_storage_measurements(measurements)
    backend = load_hashed_json(args.compiled_region_parity)
    backend_passed = _backend_passed(backend)
    available_memory = _memory_bytes()
    if args.resource_kind == "gpu":
        try:
            import torch

            if not torch.cuda.is_available():
                available_memory = 0
            else:
                free, _ = torch.cuda.mem_get_info()
                available_memory = int(free)
        except ImportError:
            available_memory = 0
    storage = shutil.disk_usage(args.campaign_root)
    expected_rate = float(
        measurements["measurements"][
            "gpu_expert_jets_per_second"
            if args.resource_kind == "gpu"
            else "cpu_degradation_jets_per_second"
        ]
    )
    # The clock sample proves a live worker and is deliberately not a model
    # benchmark; authenticated bootstrap measurements remain authoritative.
    before = time.perf_counter()
    sum(index * index for index in range(10_000))
    elapsed = time.perf_counter() - before
    if elapsed <= 0:
        raise RuntimeError("resource-probe monotonic clock did not advance")
    artifact = build_resource_probe(
        campaign_spec_sha256=campaign["content_hash"],
        storage_measurements_sha256=measurement_sha,
        resource_kind=args.resource_kind,
        node_name=os.environ.get("SLURMD_NODENAME", "local"),
        available_memory_bytes=available_memory,
        available_storage_bytes=int(storage.free),
        measured_items_per_second=expected_rate,
        compiler_backend_parity_passed=backend_passed,
        requested_memory_bytes=args.requested_memory_bytes,
        projected_peak_storage_bytes=int(
            measurements["measurements"]["projected_peak_concurrent_bytes"]
        ),
    )
    result: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "resource_probe_sha256": artifact["content_hash"],
        "resource_admitted": artifact["resource_admitted"],
        "available_memory_bytes": available_memory,
        "available_storage_bytes": int(storage.free),
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not args.dry_run and not artifact["resource_admitted"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
