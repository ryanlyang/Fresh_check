#!/usr/bin/env python3
"""Authenticate and register a prepared RETB Stage-J dataset cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
    require_sha256,
)
from teacher_logit_reco.relation_expert_token_bridge.joint_bridge_training import (  # noqa: E402
    JointBridgeDataset,
    publish_joint_dataset_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step11 import (  # noqa: E402
    STAGE_J_RUN_CONTRACT,
    validate_materialized_stage_j_run,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--prepared-dataset", required=True, type=Path)
    parser.add_argument("--prepared-dataset-sha256", required=True)
    parser.add_argument("--parent-hashes", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    run = load_hashed_json(
        args.run, expected_contract=STAGE_J_RUN_CONTRACT
    )
    validate_materialized_stage_j_run(run)
    if run.get("source") != campaign.get("source"):
        raise ValueError("joint dataset run source lineage differs")
    expected_payload_hash = require_sha256(
        args.prepared_dataset_sha256, name="prepared_dataset_sha256"
    )
    if (
        not args.prepared_dataset.is_file()
        or args.prepared_dataset.is_symlink()
        or _sha256(args.prepared_dataset) != expected_payload_hash
    ):
        raise ValueError("prepared joint dataset payload differs")
    payload = torch.load(
        args.prepared_dataset, map_location="cpu", weights_only=False
    )
    dataset = payload.get("dataset") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or set(payload) != {
        "dataset"
    } or not isinstance(
        dataset, JointBridgeDataset
    ):
        raise ValueError("prepared joint dataset semantics differ")
    parents = json.loads(args.parent_hashes.read_text("utf-8"))
    split_keys = {
        "model_train": (
            "model_train_identity_manifest",
            "model_train_R_MULTI_view_cache",
        ),
        "val_stop": (
            "val_stop_identity_manifest",
            "val_stop_R_MULTI_view_cache",
        ),
        "val_design": (
            "val_design_identity_manifest",
            "val_design_fixed_view_cache",
        ),
    }
    identity_key, view_key = split_keys[dataset.split]
    expected_parents = {
        "identity_manifest": run["parent_hashes"][identity_key],
        "HLT_view_cache": run["parent_hashes"][view_key],
        "offline_target_cache": run["parent_hashes"][
            "offline_target_cache"
        ],
        "target_normalizer_set": run["parent_hashes"][
            "target_normalizer_set"
        ],
    }
    if not isinstance(parents, dict) or parents != expected_parents:
        raise ValueError("joint dataset parent configuration differs")
    manifest = publish_joint_dataset_cache(
        output_dir=args.output_dir,
        dataset=dataset,
        parent_hashes=parents,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    print(
        json.dumps(
            {
                "joint_dataset_cache_sha256": manifest["content_hash"],
                "prepared_dataset_sha256": expected_payload_hash,
                "split": dataset.split,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
