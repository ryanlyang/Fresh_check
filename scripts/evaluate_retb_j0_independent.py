#!/usr/bin/env python3
"""Evaluate the locked J0 independent faithful reference on val_design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.joint_bridge_training import (  # noqa: E402
    evaluate_joint_bridge,
    load_joint_dataset_cache,
    load_joint_graph_template,
    make_joint_bridge_loader,
    publish_joint_inference,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_bundle import (  # noqa: E402
    PREDICTOR_BUNDLE_LOCK_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.step11 import (  # noqa: E402
    STAGE_J_RUN_CONTRACT,
    validate_materialized_stage_j_run,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--predictor-bundle-lock", required=True, type=Path)
    parser.add_argument("--graph-template", required=True, type=Path)
    parser.add_argument("--val-design-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
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
    if (
        run.get("source") != campaign.get("source")
        or lock.get("source") != campaign.get("source")
        or run.get("variant") != "J0_INDEPENDENT"
        or run.get("predictor_bundle_lock_sha256") != lock["content_hash"]
    ):
        raise ValueError("J0 evaluation lineage differs")
    graph_manifest, graph, objectives, gradnorm = load_joint_graph_template(
        args.graph_template,
        expected_variant="J0_INDEPENDENT",
        expected_run_record_sha256=run["content_hash"],
        expected_predictor_bundle_lock_sha256=lock["content_hash"],
        expected_source=campaign["source"],
    )
    cache_manifest, dataset = load_joint_dataset_cache(
        args.val_design_cache,
        expected_split="val_design",
        expected_source=campaign["source"],
    )
    expected_cache_parents = {
        "identity_manifest": run["parent_hashes"][
            "val_design_identity_manifest"
        ],
        "HLT_view_cache": run["parent_hashes"][
            "val_design_fixed_view_cache"
        ],
        "offline_target_cache": run["parent_hashes"][
            "offline_target_cache"
        ],
        "target_normalizer_set": run["parent_hashes"][
            "target_normalizer_set"
        ],
    }
    expected_graph_parents = {
        name: run["parent_hashes"][name]
        for name in (
            "frozen_offline_fusion",
            "frozen_offline_expert_heads",
            "offline_target_cache",
            "selected_predictor_seed_artifacts",
            "target_normalizer_set",
        )
    }
    if (
        cache_manifest.get("parent_hashes") != expected_cache_parents
        or graph_manifest.get("component_parent_hashes")
        != expected_graph_parents
    ):
        raise ValueError("J0 evaluation component lineage differs")
    authorize_dataset_access(
        worker_role="training_worker", requested_resource="val_design"
    )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    graph.to(device).eval()
    evaluation = evaluate_joint_bridge(
        graph=graph,
        loader=make_joint_bridge_loader(
            dataset, batch_size=args.batch_size, seed=101, training=False
        ),
        objective_by_expert=objectives,
        gradnorm_weights_by_expert=gradnorm,
        device=device,
    )
    snapshot = source_snapshot(REPO_ROOT)
    registration = bind_source(
        with_content_hash(
            {
                "contract": "retb_j0_faithful_registration_v1",
                "schema_version": 1,
                "run_sha256": run["content_hash"],
                "graph_template_sha256": graph_manifest["content_hash"],
                "predictor_bundle_lock_sha256": lock["content_hash"],
                "model_retrained": False,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=snapshot,
    )
    write_immutable_json(
        args.output_dir / "j0_registration.json", registration
    )
    predictions = publish_joint_inference(
        output_dir=args.output_dir,
        evaluation=evaluation,
        split="val_design",
        pipeline_seed=int(run["pipeline_seed"]),
        variant="J0_INDEPENDENT",
        registration_sha256=registration["content_hash"],
        identity_manifest_sha256=run["parent_hashes"][
            "val_design_identity_manifest"
        ],
        hlt_input_cache_sha256=cache_manifest["content_hash"],
        source_snapshot=snapshot,
    )
    write_immutable_json(
        args.output_dir / "joint_predictions_manifest.json", predictions
    )
    reference = bind_source(
        with_content_hash(
            {
                "contract": "retb_j0_faithful_reference_v1",
                "schema_version": 1,
                "run_sha256": run["content_hash"],
                "predictor_bundle_lock_sha256": lock["content_hash"],
                "split": "val_design",
                "pipeline_seed": run["pipeline_seed"],
                "semantic_label": run["semantic_label"],
                "metrics": evaluation["metrics"],
                "normalized_token_rmse_by_expert": evaluation[
                    "normalized_token_rmse_by_expert"
                ],
                "label_manifest_sha256": run["parent_hashes"][
                    "val_design_label_manifest"
                ],
                "prediction_manifest_sha256": predictions["content_hash"],
                "registration_sha256": registration["content_hash"],
                "input_policy": "R_MULTI",
                "model_retrained": False,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=snapshot,
    )
    publication = write_immutable_json(
        args.output_dir / "j0_reference.json", reference
    )
    print(
        json.dumps(
            {
                "J0_reference_sha256": reference["content_hash"],
                "publication": publication,
                "cache_sha256": cache_manifest["content_hash"],
                "prediction_manifest_sha256": predictions["content_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
