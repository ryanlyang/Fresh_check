#!/usr/bin/env python3
"""Fit one label-free RETB uncertainty calibrator on val_design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_cache import (  # noqa: E402
    calibrate_predictor_inference_cache,
    load_predictor_inference_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.target_cache import (  # noqa: E402
    load_offline_target_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--predictor-inference", required=True, type=Path)
    parser.add_argument("--predictor-registration", required=True, type=Path)
    parser.add_argument("--target-cache-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    registration = load_hashed_json(args.predictor_registration)
    inference, inference_arrays = load_predictor_inference_cache(
        args.predictor_inference,
        expected_pipeline_seed=int(registration["pipeline_seed"]),
        expected_registration_sha256=registration["content_hash"],
    )
    target_header = load_hashed_json(args.target_cache_manifest)
    target_manifest, target_arrays = load_offline_target_cache(
        args.target_cache_manifest,
        expected_pipeline_seed=int(registration["pipeline_seed"]),
        expected_specification_sha256=target_header[
            "specification_sha256"
        ],
    )
    expert = str(inference["expert_id"])
    if (
        registration.get("source") != campaign.get("source")
        or inference.get("source") != campaign.get("source")
        or target_manifest.get("source") != campaign.get("source")
        or registration.get("expert_id") != expert
        or target_manifest["content_hash"]
        != inference["parents"]["target_cache_manifest"]
        or target_manifest["identity_order_sha256"]
        != inference["identity_order_sha256"]
        or target_arrays["identities"].tolist()
        != inference_arrays["identities"].tolist()
    ):
        raise ValueError("uncertainty-calibration lineage differs")
    result: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "pipeline_seed": int(registration["pipeline_seed"]),
        "expert_id": expert,
        "predictor_inference_sha256": inference["content_hash"],
        "target_cache_manifest_sha256": target_manifest["content_hash"],
        "output": str(args.output.resolve()),
    }
    if not args.dry_run:
        authorize_dataset_access(
            worker_role="design_worker",
            requested_resource="val_design",
        )
        calibration = calibrate_predictor_inference_cache(
            manifest_path=args.predictor_inference,
            expected_pipeline_seed=int(registration["pipeline_seed"]),
            expected_registration_sha256=registration["content_hash"],
            target_tokens=target_arrays["tokens"][expert],
            identity_order_sha256=inference["identity_order_sha256"],
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        result["calibration_sha256"] = calibration["content_hash"]
        result["publication"] = write_immutable_json(
            args.output, calibration
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
