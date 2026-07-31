#!/usr/bin/env python3
"""Export and reload-audit one exact HLT-only RETB graph."""

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
from teacher_logit_reco.relation_expert_token_bridge.deployment import (  # noqa: E402
    DeployableRetbGraph,
    export_deployable_retb_graph,
)
from teacher_logit_reco.relation_expert_token_bridge.final_consumer_training import (  # noqa: E402
    FINAL_CONSUMER_CHECKPOINT_CONTRACT,
    FINAL_CONSUMER_REGISTRATION_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step12 import (  # noqa: E402
    FINAL_CONSUMER_RUN_CONTRACT,
    validate_materialized_final_consumer_run,
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
    parser.add_argument("--consumer-registration", required=True, type=Path)
    parser.add_argument("--consumer-checkpoint", type=Path)
    parser.add_argument("--prepared-export", required=True, type=Path)
    parser.add_argument("--prepared-export-sha256", required=True)
    parser.add_argument("--parent-hashes", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    run = load_hashed_json(
        args.run, expected_contract=FINAL_CONSUMER_RUN_CONTRACT
    )
    validate_materialized_final_consumer_run(run)
    registration = load_hashed_json(
        args.consumer_registration,
        expected_contract=(
            FINAL_CONSUMER_REGISTRATION_CONTRACT
            if run["trainable"]
            else "retb_final_consumer_reference_registration_v1"
        ),
    )
    expected_payload_hash = require_sha256(
        args.prepared_export_sha256, name="prepared_export_sha256"
    )
    if (
        run.get("source") != campaign.get("source")
        or registration.get("source") != campaign.get("source")
        or (
            run["trainable"]
            and registration.get("run_record_sha256")
            != run["content_hash"]
        )
        or (
            not run["trainable"]
            and registration.get("run_sha256") != run["content_hash"]
        )
        or not args.prepared_export.is_file()
        or args.prepared_export.is_symlink()
        or _sha256(args.prepared_export) != expected_payload_hash
    ):
        raise ValueError("prepared deployable export lineage differs")
    payload = torch.load(
        args.prepared_export, map_location="cpu", weights_only=False
    )
    if (
        not isinstance(payload, dict)
        or set(payload) != {"graph", "smoke_inputs"}
        or not isinstance(payload["graph"], DeployableRetbGraph)
    ):
        raise ValueError("prepared deployable export semantics differ")
    consumer_kind = run["consumer_kind"]
    if payload["graph"].consumer_kind != consumer_kind:
        raise ValueError("deployable graph consumer kind differs")
    if run["trainable"]:
        if (
            args.consumer_checkpoint is None
            or not args.consumer_checkpoint.is_file()
            or args.consumer_checkpoint.is_symlink()
            or _sha256(args.consumer_checkpoint)
            != registration.get("checkpoint_sha256")
        ):
            raise ValueError(
                "deployable graph selected checkpoint lineage differs"
            )
        checkpoint = torch.load(
            args.consumer_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        graph_state = (
            payload["graph"].token_refiner.state_dict()
            if run["consumer_kind"] == "TR_REFINE"
            else payload["graph"].final_consumer.state_dict()
        )
        selected_state = checkpoint.get("model_state_dict")
        if (
            checkpoint.get("contract")
            != FINAL_CONSUMER_CHECKPOINT_CONTRACT
            or checkpoint.get("kind") != "selected_inference"
            or checkpoint.get("training_contract_sha256")
            != registration.get("training_contract_sha256")
            or not isinstance(selected_state, dict)
            or set(graph_state) != set(selected_state)
            or any(
                not torch.equal(
                    graph_state[name].detach().cpu(),
                    selected_state[name].detach().cpu(),
                )
                for name in graph_state
            )
        ):
            raise ValueError(
                "deployable graph does not contain selected consumer weights"
            )
    elif args.consumer_checkpoint is not None:
        raise ValueError(
            "non-trainable deployable reference forbids a checkpoint"
        )
    parents = json.loads(args.parent_hashes.read_text("utf-8"))
    expected = {
        "campaign_spec": campaign["content_hash"],
        "step12_bundle": run["step12_bundle_sha256"],
        "HLT_frontend_checkpoint": run["parent_hashes"][
            "native_HLT_checkpoint_bundle"
        ],
        "joint_predictor_or_J_checkpoint": run["parent_hashes"][
            "joint_prediction_checkpoint"
        ],
        "final_consumer_checkpoint": registration["content_hash"],
        "frozen_offline_fusion_checkpoint": run["parent_hashes"][
            "frozen_offline_fusion"
        ],
        "frozen_offline_expert_heads": run["parent_hashes"][
            "frozen_offline_expert_heads"
        ],
        "HLT_input_normalizer": run["parent_hashes"][
            "HLT_input_normalizer"
        ],
        "HLT_relation_normalizer": run["parent_hashes"][
            "HLT_relation_normalizer"
        ],
        "HLT_region_normalizer": run["parent_hashes"][
            "HLT_region_normalizer"
        ],
        "degradation_profile": run["parent_hashes"][
            "degradation_profile"
        ],
        "uncertainty_calibration": run["parent_hashes"][
            "uncertainty_calibration"
        ],
    }
    if payload["graph"].token_refiner is not None:
        expected["token_refiner_checkpoint"] = run["parent_hashes"][
            "selected_token_refiner"
        ]
    if parents != expected:
        raise ValueError("deployable export parents differ")
    manifest = export_deployable_retb_graph(
        output_dir=args.output_dir,
        graph=payload["graph"],
        hlt_smoke_inputs=payload["smoke_inputs"],
        parent_hashes=parents,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    print(
        json.dumps(
            {
                "deployment_export_sha256": manifest["content_hash"],
                "graph_sha256": manifest["graph_sha256"],
                "prepared_export_sha256": expected_payload_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
