#!/usr/bin/env python3
"""Train one fixed-budget RETB J1-J5 graph and evaluate val_design."""

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
from teacher_logit_reco.relation_expert_token_bridge.joint_bridge_training import (  # noqa: E402
    JointBridgeTrainingConfig,
    STAGE_J_METRICS_CONTRACT,
    evaluate_joint_bridge,
    load_joint_dataset_cache,
    load_joint_graph_template,
    make_joint_bridge_loader,
    publish_joint_inference,
    train_joint_bridge,
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


def _forward(graph: Any, raw: Mapping[str, Any], device: torch.device) -> Any:
    def move(value):
        if isinstance(value, Mapping):
            return {name: move(item) for name, item in value.items()}
        if isinstance(value, list):
            return value
        return value.to(device) if hasattr(value, "to") else value

    batch = move(raw)
    if graph.variant in {"J4_BRIDGE_FINETUNE", "J5_END_TO_END"}:
        return graph(shared_view=batch["shared_raw_view"])
    return graph(
        evidence={
            "hlt_token_banks": batch["hlt_token_banks"],
            "unbiased_particle_states": batch[
                "unbiased_particle_states"
            ],
            "particle_mask": batch["particle_mask"],
            "relation_particle_states": batch[
                "relation_particle_states"
            ],
            "relation_particle_masks": batch["relation_particle_masks"],
        }
    )


def _profile_flops(graph: Any, loader: Any, device: torch.device) -> int:
    raw = next(iter(loader))
    batch_size = len(raw["identities"])
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    graph.eval()
    with torch.no_grad(), torch.profiler.profile(
        activities=activities,
        with_flops=True,
        record_shapes=True,
    ) as profile:
        _forward(graph, raw, device)
    flops = sum(int(event.flops or 0) for event in profile.key_averages())
    if flops <= 0:
        raise RuntimeError("joint graph profiler emitted no FLOPs")
    return (flops + batch_size - 1) // batch_size


def _validate_cache_lineage(
    *,
    manifest: Mapping[str, Any],
    run: Mapping[str, Any],
    split: str,
) -> None:
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
        "scale_train": (
            "scale_train_identity_manifest",
            "scale_train_R_MULTI_view_cache",
        ),
    }
    identity_key, view_key = split_keys[split]
    expected = {
        "identity_manifest": run["parent_hashes"][identity_key],
        "HLT_view_cache": run["parent_hashes"][view_key],
        "offline_target_cache": run["parent_hashes"][
            "offline_target_cache"
        ],
        "target_normalizer_set": run["parent_hashes"][
            "target_normalizer_set"
        ],
    }
    if manifest.get("parent_hashes") != expected:
        raise ValueError(f"{split} joint cache parent lineage differs")


def _validate_graph_lineage(
    *,
    template: Mapping[str, Any],
    run: Mapping[str, Any],
) -> None:
    names = {
        "frozen_offline_fusion",
        "frozen_offline_expert_heads",
        "offline_target_cache",
        "selected_predictor_seed_artifacts",
        "target_normalizer_set",
    }
    if run["variant"] in {"J4_BRIDGE_FINETUNE", "J5_END_TO_END"}:
        names.add("selected_HLT_expert_seed_artifacts")
    if run["variant"] == "J5_END_TO_END":
        names.update(
            {
                "j4_block_selection",
                "selected_J4_bridge_initialization",
            }
        )
    expected = {name: run["parent_hashes"][name] for name in names}
    if template.get("component_parent_hashes") != expected:
        raise ValueError("joint graph component lineage differs")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument(
        "--training-role",
        choices=("model_train", "scale_train"),
        default="model_train",
    )
    parser.add_argument("--predictor-bundle-lock", required=True, type=Path)
    parser.add_argument("--graph-template", required=True, type=Path)
    parser.add_argument("--model-train-cache", required=True, type=Path)
    parser.add_argument("--val-stop-cache", required=True, type=Path)
    parser.add_argument("--val-design-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--microbatch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    run = load_hashed_json(args.run)
    if run.get("contract") == STAGE_J_RUN_CONTRACT:
        validate_materialized_stage_j_run(run)
    elif (
        run.get("contract") != "retb_scale_stage_j_run_v1"
        or run.get("training_population") != "scale_train"
        or args.training_role != "scale_train"
    ):
        raise ValueError("scale Stage-J run contract differs")
    lock = load_hashed_json(
        args.predictor_bundle_lock,
        expected_contract=PREDICTOR_BUNDLE_LOCK_CONTRACT,
    )
    step11 = load_hashed_json(
        args.campaign_root
        / "registry"
        / "retb_step11_joint_bridge_bundle.json"
    )
    if (
        run.get("source") != campaign.get("source")
        or lock.get("source") != campaign.get("source")
        or step11.get("source") != campaign.get("source")
        or run["predictor_bundle_lock_sha256"] != lock["content_hash"]
        or run["step11_bundle_sha256"] != step11["content_hash"]
        or run["variant"] == "J0_INDEPENDENT"
    ):
        raise ValueError("joint bridge training lineage differs")
    template, graph, objectives, gradnorm = load_joint_graph_template(
        args.graph_template,
        expected_variant=run["variant"],
        expected_run_record_sha256=run["content_hash"],
        expected_predictor_bundle_lock_sha256=lock["content_hash"],
        expected_source=campaign["source"],
    )
    train_manifest, train = load_joint_dataset_cache(
        args.model_train_cache,
        expected_split=args.training_role,
        expected_source=campaign["source"],
    )
    stop_manifest, stop = load_joint_dataset_cache(
        args.val_stop_cache,
        expected_split="val_stop",
        expected_source=campaign["source"],
    )
    design_manifest, design = load_joint_dataset_cache(
        args.val_design_cache,
        expected_split="val_design",
        expected_source=campaign["source"],
    )
    _validate_graph_lineage(template=template, run=run)
    for split, manifest in (
        (args.training_role, train_manifest),
        ("val_stop", stop_manifest),
        ("val_design", design_manifest),
    ):
        _validate_cache_lineage(manifest=manifest, run=run, split=split)
    identity_sets = [
        set(train.identities),
        set(stop.identities),
        set(design.identities),
    ]
    if (
        len(train.identities) != len(identity_sets[0])
        or len(stop.identities) != len(identity_sets[1])
        or len(design.identities) != len(identity_sets[2])
        or any(
            identity_sets[left] & identity_sets[right]
            for left, right in ((0, 1), (0, 2), (1, 2))
        )
    ):
        raise ValueError("joint bridge cache identity populations differ")
    effective_batch = (
        args.microbatch_size * args.gradient_accumulation_steps
    )
    miniature = campaign["campaign_profile"] == "miniature_test"
    config = JointBridgeTrainingConfig(
        seed=int(run["pipeline_seed"]),
        variant=run["variant"],
        final_particle_blocks=run["final_particle_blocks"],
        maximum_epochs=2 if miniature else 40,
        microbatch_size=args.microbatch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        effective_batch_size=effective_batch,
        campaign_profile="miniature_test" if miniature else "production",
    )
    summary = {
        "dry_run": args.dry_run,
        "run_id": run["run_id"],
        "variant": run["variant"],
        "pipeline_seed": run["pipeline_seed"],
        "graph_template_sha256": template["content_hash"],
        "dataset_cache_hashes": {
            "model_train": train_manifest["content_hash"],
            "val_stop": stop_manifest["content_hash"],
            "val_design": design_manifest["content_hash"],
        },
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    authorize_dataset_access(
        worker_role=(
            "scale_training_worker"
            if args.training_role == "scale_train"
            else "training_worker"
        ),
        requested_resource=args.training_role,
    )
    authorize_dataset_access(
        worker_role="training_worker", requested_resource="val_stop"
    )
    authorize_dataset_access(
        worker_role="training_worker", requested_resource="val_design"
    )
    resolved = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    train_loader = make_joint_bridge_loader(
        train,
        batch_size=args.microbatch_size,
        seed=config.seed,
        training=True,
    )
    stop_loader = make_joint_bridge_loader(
        stop,
        batch_size=args.microbatch_size,
        seed=config.seed,
        training=False,
    )
    design_loader = make_joint_bridge_loader(
        design,
        batch_size=args.microbatch_size,
        seed=config.seed,
        training=False,
    )
    lineage = {
        "graph_template": template["content_hash"],
        f"{args.training_role}_cache": train_manifest["content_hash"],
        "val_stop_cache": stop_manifest["content_hash"],
        "val_design_cache": design_manifest["content_hash"],
    }
    registration = train_joint_bridge(
        graph=graph,
        train_loader=train_loader,
        val_stop_loader=stop_loader,
        objective_by_expert=objectives,
        gradnorm_weights_by_expert=gradnorm,
        output_dir=args.output_dir,
        run_record=run,
        step11_bundle_sha256=step11["content_hash"],
        predictor_bundle_lock_sha256=lock["content_hash"],
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        lineage_hashes=lineage,
        source_snapshot=source_snapshot(REPO_ROOT),
        config=config,
        device=resolved,
    )
    design_evaluation = evaluate_joint_bridge(
        graph=graph,
        loader=design_loader,
        objective_by_expert=objectives,
        gradnorm_weights_by_expert=gradnorm,
        device=resolved,
    )
    inference = publish_joint_inference(
        output_dir=args.output_dir / "val_design",
        evaluation=design_evaluation,
        split="val_design",
        pipeline_seed=config.seed,
        variant=config.variant,
        registration_sha256=registration["content_hash"],
        identity_manifest_sha256=run["parent_hashes"][
            "val_design_identity_manifest"
        ],
        hlt_input_cache_sha256=design_manifest["content_hash"],
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(
        args.output_dir / "val_design" / "joint_predictions_manifest.json",
        inference,
    )
    uninitialized = [
        name
        for name, parameter in graph.named_parameters()
        if isinstance(
            parameter, torch.nn.parameter.UninitializedParameter
        )
    ]
    if uninitialized:
        raise RuntimeError(
            "joint graph capacity cannot be attested with uninitialized "
            f"parameters: {uninitialized}"
        )
    parameter_count = sum(parameter.numel() for parameter in graph.parameters())
    inference_flops = _profile_flops(graph, design_loader, resolved)
    metrics = bind_source(
        with_content_hash(
            {
                "contract": STAGE_J_METRICS_CONTRACT,
                "schema_version": 1,
                "run_id": run["run_id"],
                "variant": run["variant"],
                "final_particle_blocks": run["final_particle_blocks"],
                "pipeline_seed": config.seed,
                "split": "val_design",
                "accuracy": design_evaluation["metrics"]["accuracy"],
                "cross_entropy": design_evaluation["metrics"][
                    "cross_entropy"
                ],
                "normalized_token_error": sum(
                    design_evaluation[
                        "normalized_token_rmse_by_expert"
                    ].values()
                )
                / 7.0,
                "inference_flops": inference_flops,
                "parameter_count": parameter_count,
                "registration_sha256": registration["content_hash"],
                "prediction_manifest_sha256": inference["content_hash"],
                "label_manifest_sha256": run["parent_hashes"][
                    "val_design_label_manifest"
                ],
                "input_policy": "R_MULTI",
                "FLOP_accounting": (
                    "torch_profiler_measured_per_event_on_val_design"
                ),
                "performance_based_termination": False,
                "stack_val_consumed": False,
                "final_test_consumed": False,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    metrics_publication = write_immutable_json(
        args.output_dir / "val_design" / "metrics.json", metrics
    )
    summary.update(
        {
            "registration_sha256": registration["content_hash"],
            "prediction_manifest_sha256": inference["content_hash"],
            "metrics_sha256": metrics["content_hash"],
            "metrics_publication": metrics_publication,
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
