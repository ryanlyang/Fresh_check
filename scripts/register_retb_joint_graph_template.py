#!/usr/bin/env python3
"""Authenticate and register a prepared RETB Stage-J graph template."""

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
from teacher_logit_reco.relation_expert_token_bridge.joint_bridge import (  # noqa: E402
    JointBridgeGraph,
)
from teacher_logit_reco.relation_expert_token_bridge.joint_bridge_training import (  # noqa: E402
    publish_joint_graph_template,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_bundle import (  # noqa: E402
    PREDICTOR_BUNDLE_LOCK_CONTRACT,
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
    parser.add_argument("--predictor-bundle-lock", required=True, type=Path)
    parser.add_argument("--prepared-graph", required=True, type=Path)
    parser.add_argument("--prepared-graph-sha256", required=True)
    parser.add_argument("--component-parents", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    run = load_hashed_json(
        args.run, expected_contract=STAGE_J_RUN_CONTRACT
    )
    validate_materialized_stage_j_run(run)
    lock = load_hashed_json(
        args.predictor_bundle_lock,
        expected_contract=PREDICTOR_BUNDLE_LOCK_CONTRACT,
    )
    expected_payload_hash = require_sha256(
        args.prepared_graph_sha256, name="prepared_graph_sha256"
    )
    if (
        not args.prepared_graph.is_file()
        or args.prepared_graph.is_symlink()
        or _sha256(args.prepared_graph) != expected_payload_hash
        or run.get("source") != campaign.get("source")
        or lock.get("source") != campaign.get("source")
        or run["predictor_bundle_lock_sha256"] != lock["content_hash"]
    ):
        raise ValueError("prepared joint graph lineage differs")
    payload = torch.load(
        args.prepared_graph, map_location="cpu", weights_only=False
    )
    if not isinstance(payload, dict) or set(payload) != {
        "graph",
        "objective_by_expert",
        "gradnorm_weights_by_expert",
    } or not isinstance(payload["graph"], JointBridgeGraph):
        raise ValueError("prepared joint graph payload differs")
    parents = json.loads(args.component_parents.read_text("utf-8"))
    if not isinstance(parents, dict) or any(
        run["parent_hashes"].get(name) != value
        for name, value in parents.items()
    ):
        raise ValueError("prepared joint graph component parents differ")
    manifest = publish_joint_graph_template(
        output_dir=args.output_dir,
        graph=payload["graph"],
        run_record_sha256=run["content_hash"],
        predictor_bundle_lock_sha256=lock["content_hash"],
        objective_by_expert=payload["objective_by_expert"],
        gradnorm_weights_by_expert=payload[
            "gradnorm_weights_by_expert"
        ],
        component_parent_hashes=parents,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    print(
        json.dumps(
            {
                "joint_graph_template_sha256": manifest["content_hash"],
                "prepared_graph_sha256": expected_payload_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
