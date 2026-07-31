#!/usr/bin/env python3
"""Train one fixed-budget RETB Step-12 final consumer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

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
    FinalConsumerTrainingConfig,
    evaluate_final_consumer,
    load_final_consumer_dataset,
    load_final_consumer_template,
    make_final_consumer_loader,
    publish_final_consumer_inference,
    train_final_consumer,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.step12 import (  # noqa: E402
    FINAL_CONSUMER_RUN_CONTRACT,
    validate_materialized_final_consumer_run,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_execution import (  # noqa: E402
    SCALE_FINAL_CONSUMER_RUN_CONTRACT,
    validate_scale_final_consumer_run,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def _validate_dataset(
    manifest: Mapping[str, Any],
    *,
    run: Mapping[str, Any],
    split: str,
) -> None:
    keys = {
        "model_train": (
            "model_train_identity_manifest",
            "model_train_R_MULTI_view_cache",
        ),
        "scale_train": (
            "scale_train_identity_manifest",
            "scale_train_R_MULTI_view_cache",
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
    identity, view = keys[split]
    expected = {
        "identity_manifest": run["parent_hashes"][identity],
        "HLT_view_cache": run["parent_hashes"][view],
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
    if manifest.get("parent_hashes") != expected:
        raise ValueError(f"{split} final-consumer cache lineage differs")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--model-train-cache", required=True, type=Path)
    parser.add_argument("--val-stop-cache", required=True, type=Path)
    parser.add_argument("--val-design-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--microbatch-size", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--training-role",
        choices=("model_train", "scale_train"),
        default="model_train",
    )
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    run = load_hashed_json(args.run)
    if run.get("contract") == FINAL_CONSUMER_RUN_CONTRACT:
        validate_materialized_final_consumer_run(run)
    elif run.get("contract") == SCALE_FINAL_CONSUMER_RUN_CONTRACT:
        validate_scale_final_consumer_run(run)
    else:
        raise ValueError("final-consumer run contract differs")
    scale_training = args.training_role == "scale_train"
    if scale_training != (
        run.get("contract") == SCALE_FINAL_CONSUMER_RUN_CONTRACT
    ):
        raise ValueError("final-consumer training population differs")
    step12 = load_hashed_json(
        args.campaign_root
        / "registry"
        / "retb_step12_final_consumers_bundle.json"
    )
    if (
        run.get("source") != campaign.get("source")
        or step12.get("source") != campaign.get("source")
        or run["step12_bundle_sha256"] != step12["content_hash"]
        or not run["trainable"]
    ):
        raise ValueError("final-consumer training lineage differs")
    template, model, heads, fusion, refiner = (
        load_final_consumer_template(
            args.template,
            expected_run_record_sha256=run["content_hash"],
            expected_source=campaign["source"],
        )
    )
    component_names = {
        "joint_prediction_checkpoint",
        "native_HLT_checkpoint_bundle",
        "uncertainty_calibration",
        "frozen_offline_fusion",
        "frozen_offline_expert_heads",
    }
    if refiner is not None:
        component_names.add("selected_token_refiner")
    if template["component_parent_hashes"] != {
        name: run["parent_hashes"][name] for name in component_names
    }:
        raise ValueError("final-consumer template component lineage differs")
    training_role = args.training_role
    loaded = {
        split: load_final_consumer_dataset(
            path,
            expected_split=split,
            expected_source=campaign["source"],
        )
        for split, path in (
            (training_role, args.model_train_cache),
            ("val_stop", args.val_stop_cache),
            ("val_design", args.val_design_cache),
        )
    }
    for split, (manifest, _) in loaded.items():
        _validate_dataset(manifest, run=run, split=split)
    identity_sets = [
        set(loaded[split][1].identities)
        for split in (training_role, "val_stop", "val_design")
    ]
    if any(
        identity_sets[left] & identity_sets[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("final-consumer split identities overlap")
    default_batch = 512 if run["consumer_kind"] == "OF_ROBUST" else 128
    microbatch = args.microbatch_size or default_batch
    miniature = campaign["campaign_profile"] == "miniature_test"
    config = FinalConsumerTrainingConfig(
        seed=int(run["pipeline_seed"]),
        consumer_kind=run["consumer_kind"],
        model_variant=run["model_variant"],
        native_dropout_mode=run["native_dropout_mode"],
        maximum_epochs=2 if miniature else 40,
        microbatch_size=microbatch,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        effective_batch_size=(
            microbatch * args.gradient_accumulation_steps
        ),
        campaign_profile="miniature_test" if miniature else "production",
    )
    summary = {
        "dry_run": args.dry_run,
        "run_id": run["run_id"],
        "consumer_kind": run["consumer_kind"],
        "model_variant": run["model_variant"],
        "native_dropout_mode": run["native_dropout_mode"],
        "template_sha256": template["content_hash"],
        "dataset_hashes": {
            split: loaded[split][0]["content_hash"] for split in loaded
        },
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    for split in (training_role, "val_stop", "val_design"):
        authorize_dataset_access(
            worker_role=(
                "scale_training_worker"
                if split == "scale_train"
                else "training_worker"
                if split == "val_stop"
                else "design_worker"
            ),
            requested_resource=split,
        )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    loaders = {
        split: make_final_consumer_loader(
            loaded[split][1],
            batch_size=microbatch,
            seed=config.seed,
            training=split == training_role,
        )
        for split in loaded
    }
    lineage = {
        "template": template["content_hash"],
        **{
            f"{split}_cache": loaded[split][0]["content_hash"]
            for split in loaded
        },
    }
    snapshot = source_snapshot(REPO_ROOT)
    registration = train_final_consumer(
        model=model,
        train_loader=loaders[training_role],
        val_stop_loader=loaders["val_stop"],
        frozen_expert_heads=heads,
        frozen_offline_fusion=fusion,
        output_dir=args.output_dir,
        run_record=run,
        step12_bundle_sha256=step12["content_hash"],
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        lineage_hashes=lineage,
        source_snapshot=snapshot,
        config=config,
        device=device,
        refiner=refiner,
    )
    design = evaluate_final_consumer(
        model=model,
        consumer_kind=config.consumer_kind,
        loader=loaders["val_design"],
        frozen_expert_heads=heads,
        frozen_offline_fusion=fusion,
        device=device,
        refiner=refiner,
    )
    inference = publish_final_consumer_inference(
        output_dir=args.output_dir / "val_design",
        evaluation=design,
        split="val_design",
        run_id=run["run_id"],
        registration_sha256=registration["content_hash"],
        identity_manifest_sha256=run["parent_hashes"][
            "val_design_identity_manifest"
        ],
        HLT_input_cache_sha256=loaded["val_design"][0]["content_hash"],
        source_snapshot=snapshot,
    )
    write_immutable_json(
        args.output_dir
        / "val_design"
        / "final_consumer_predictions_manifest.json",
        inference,
    )
    metrics = bind_source(
        with_content_hash(
            {
                "contract": "retb_final_consumer_val_design_metrics_v1",
                "schema_version": 1,
                "run_id": run["run_id"],
                "consumer_kind": run["consumer_kind"],
                "model_variant": run["model_variant"],
                "native_dropout_mode": run["native_dropout_mode"],
                "token_input": run["token_input"],
                "pipeline_seed": run["pipeline_seed"],
                "split": "val_design",
                "metrics": design["metrics"],
                "token_RMSE_before": design["token_RMSE_before"],
                "token_RMSE_after": design["token_RMSE_after"],
                "registration_sha256": registration["content_hash"],
                "prediction_manifest_sha256": inference["content_hash"],
                "label_manifest_sha256": run["parent_hashes"][
                    "val_design_label_manifest"
                ],
                "performance_based_termination": False,
                "stack_val_consumed": False,
                "final_test_consumed": False,
            }
        ),
        source_snapshot=snapshot,
    )
    publication = write_immutable_json(
        args.output_dir / "val_design" / "metrics.json", metrics
    )
    summary.update(
        {
            "registration_sha256": registration["content_hash"],
            "prediction_manifest_sha256": inference["content_hash"],
            "metrics_sha256": metrics["content_hash"],
            "metrics_publication": publication,
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
