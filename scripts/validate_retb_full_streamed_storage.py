#!/usr/bin/env python3
"""Validate the complete authenticated full-streamed storage evidence pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import load_hashed_json  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.provenance import source_snapshot  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.storage import (  # noqa: E402
    STORAGE_MEASUREMENTS_CONTRACT,
    validate_storage_measurements,
)
from teacher_logit_reco.relation_expert_token_bridge.streamed_execution import (  # noqa: E402
    STREAMED_STORAGE_PROJECTION_CONTRACT,
    validate_streamed_storage_projection,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", required=True, type=Path)
    parser.add_argument("--projection", type=Path)
    args = parser.parse_args()
    measurement = load_hashed_json(
        args.measurements, expected_contract=STORAGE_MEASUREMENTS_CONTRACT
    )
    projection_path = args.projection or args.measurements.with_name(
        "full_streamed_storage_projection.json"
    )
    projection = load_hashed_json(
        projection_path, expected_contract=STREAMED_STORAGE_PROJECTION_CONTRACT
    )
    validate_storage_measurements(measurement)
    validate_streamed_storage_projection(projection)
    metrics = measurement["measurements"]
    if (
        measurement["evidence_hashes"].get("full_streamed_storage_projection")
        != projection["content_hash"]
        or int(metrics["projected_peak_concurrent_bytes"])
        != int(projection["persistent_peak_bytes"])
        or int(metrics["available_storage_bytes"])
        != int(projection["available_persistent_storage_bytes"])
        or projection["persistent_storage_admitted"] is not True
        or projection.get("source") != source_snapshot(REPO_ROOT)
    ):
        raise ValueError("full-streamed storage evidence pair differs")
    print(
        json.dumps(
            {
                "status": "full_streamed_storage_admitted",
                "measurements_sha256": measurement["content_hash"],
                "projection_sha256": projection["content_hash"],
                "persistent_peak_bytes": projection["persistent_peak_bytes"],
                "available_storage_bytes": projection[
                    "available_persistent_storage_bytes"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
