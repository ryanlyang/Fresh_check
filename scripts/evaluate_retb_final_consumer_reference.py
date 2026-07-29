#!/usr/bin/env python3
"""Evaluate a non-trainable PF_FROZEN or TR0 Step-12 reference."""

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
from teacher_logit_reco.relation_expert_token_bridge.final_consumer_training import (  # noqa: E402
    evaluate_final_consumer,
    load_final_consumer_dataset,
    load_final_consumer_template,
    make_final_consumer_loader,
    publish_final_consumer_inference,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step12 import (  # noqa: E402
    FINAL_CONSUMER_RUN_CONTRACT,
    validate_materialized_final_consumer_run,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--val-design-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    run = load_hashed_json(
        args.run, expected_contract=FINAL_CONSUMER_RUN_CONTRACT
    )
    validate_materialized_final_consumer_run(run)
    if (
        run.get("source") != campaign.get("source")
        or run["trainable"]
        or not (
            run["consumer_kind"] == "PF_FROZEN"
            or (
                run["consumer_kind"] == "TR_REFINE"
                and run["model_variant"] == "TR0_NONE"
            )
        )
    ):
        raise ValueError("final-consumer reference lineage differs")
    template, model, heads, fusion, refiner = (
        load_final_consumer_template(
            args.template,
            expected_run_record_sha256=run["content_hash"],
            expected_source=campaign["source"],
        )
    )
    cache, dataset = load_final_consumer_dataset(
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
        "joint_prediction_cache": run["parent_hashes"][
            "joint_prediction_checkpoint"
        ],
        "native_HLT_cache": run["parent_hashes"][
            "native_HLT_checkpoint_bundle"
        ],
        "offline_target_cache": run["parent_hashes"][
            "offline_target_cache"
        ],
        "target_normalizer_set": run["parent_hashes"][
            "target_normalizer_set"
        ],
        "uncertainty_calibration": run["parent_hashes"][
            "uncertainty_calibration"
        ],
    }
    if cache["parent_hashes"] != expected_cache_parents:
        raise ValueError("final-consumer reference cache lineage differs")
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
    model.to(device).eval()
    consumer_kind = run["consumer_kind"]
    evaluation = evaluate_final_consumer(
        model=model,
        consumer_kind=consumer_kind,
        loader=make_final_consumer_loader(
            dataset,
            batch_size=args.batch_size,
            seed=int(run["pipeline_seed"]),
            training=False,
        ),
        frozen_expert_heads=heads,
        frozen_offline_fusion=fusion,
        device=device,
        refiner=refiner,
    )
    snapshot = source_snapshot(REPO_ROOT)
    registration = bind_source(
        with_content_hash(
            {
                "contract": "retb_final_consumer_reference_registration_v1",
                "schema_version": 1,
                "run_sha256": run["content_hash"],
                "template_sha256": template["content_hash"],
                "consumer_kind": consumer_kind,
                "model_variant": run["model_variant"],
                "model_retrained": False,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=snapshot,
    )
    write_immutable_json(
        args.output_dir / "reference_registration.json", registration
    )
    inference = publish_final_consumer_inference(
        output_dir=args.output_dir,
        evaluation=evaluation,
        split="val_design",
        run_id=run["run_id"],
        registration_sha256=registration["content_hash"],
        identity_manifest_sha256=run["parent_hashes"][
            "val_design_identity_manifest"
        ],
        HLT_input_cache_sha256=cache["content_hash"],
        source_snapshot=snapshot,
    )
    write_immutable_json(
        args.output_dir / "final_consumer_predictions_manifest.json",
        inference,
    )
    report = bind_source(
        with_content_hash(
            {
                "contract": "retb_final_consumer_reference_v1",
                "schema_version": 1,
                "run_sha256": run["content_hash"],
                "metrics": evaluation["metrics"],
                "token_RMSE_before": evaluation["token_RMSE_before"],
                "token_RMSE_after": evaluation["token_RMSE_after"],
                "prediction_manifest_sha256": inference["content_hash"],
                "label_manifest_sha256": run["parent_hashes"][
                    "val_design_label_manifest"
                ],
                "model_retrained": False,
            }
        ),
        source_snapshot=snapshot,
    )
    publication = write_immutable_json(
        args.output_dir / "reference_metrics.json", report
    )
    print(
        json.dumps(
            {
                "reference_sha256": report["content_hash"],
                "prediction_manifest_sha256": inference["content_hash"],
                "publication": publication,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
