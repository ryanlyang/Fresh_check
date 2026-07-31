#!/usr/bin/env python3
"""Select and lock the RETB J4 N=2/4 bridge-tuning depth."""

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
    bind_source,
    canonical_sha256,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.joint_bridge_training import (  # noqa: E402
    STAGE_J_METRICS_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.step11 import (  # noqa: E402
    select_j4_block_count,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    step11 = load_hashed_json(
        args.campaign_root
        / "registry"
        / "retb_step11_joint_bridge_bundle.json"
    )
    configuration = json.loads(args.configuration.read_text("utf-8"))
    required = {
        "metric_artifact_paths",
        "predictor_bundle_lock_sha256",
        "label_manifest_hashes_by_seed",
    }
    if set(configuration) != required:
        raise ValueError("J4 selector configuration fields differ")
    if (
        step11.get("source") != campaign.get("source")
        or step11["parents"]["predictor_bundle_lock"]
        != configuration["predictor_bundle_lock_sha256"]
    ):
        raise ValueError("J4 selector bundle lineage differs")
    rows = []
    for path in configuration["metric_artifact_paths"]:
        artifact = load_hashed_json(
            Path(path), expected_contract=STAGE_J_METRICS_CONTRACT
        )
        if (
            artifact.get("source") != campaign.get("source")
            or artifact.get("split") != "val_design"
            or artifact.get("label_manifest_sha256")
            != configuration["label_manifest_hashes_by_seed"].get(
                str(artifact.get("pipeline_seed")),
                configuration["label_manifest_hashes_by_seed"].get(
                    artifact.get("pipeline_seed")
                ),
            )
        ):
            raise ValueError("J4 selector metric lineage differs")
        rows.append(
            {
                name: artifact[name]
                for name in (
                    "variant",
                    "final_particle_blocks",
                    "pipeline_seed",
                    "split",
                    "accuracy",
                    "cross_entropy",
                    "normalized_token_error",
                    "inference_flops",
                    "parameter_count",
                    "registration_sha256",
                )
            }
        )
    selection = bind_source(
        select_j4_block_count(
            rows,
            predictor_bundle_lock_sha256=configuration[
                "predictor_bundle_lock_sha256"
            ],
            label_manifest_sha256=canonical_sha256(
                configuration["label_manifest_hashes_by_seed"]
            ),
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    result = {
        "dry_run": args.dry_run,
        "selected_final_particle_blocks": selection[
            "selected_final_particle_blocks"
        ],
        "selection_sha256": selection["content_hash"],
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, selection)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
