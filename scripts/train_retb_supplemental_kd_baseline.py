#!/usr/bin/env python3
"""Train one conventional O_BASE/O_FULLREL student with dominant logit KD."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_retb_offline_expert import _dataset, _load_npz
from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    bind_source,
    load_hashed_json,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import evaluate_classification
from teacher_logit_reco.relation_expert_token_bridge.expert_training import (
    OfflineExpertTrainingConfig,
    _publish_predictions,
    make_offline_expert_loader,
    train_offline_expert,
)
from teacher_logit_reco.relation_expert_token_bridge.offline_capacity_models import (
    FAMILIES,
    OfflineClassifierAdapter,
    analytical_particle_transformer_flops,
)
from teacher_logit_reco.relation_expert_token_bridge.offline_capacity_training import (
    _collect_predictions,
    build_capacity_profile,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import source_snapshot
from teacher_logit_reco.relation_expert_token_bridge.stage_a import (
    load_authenticated_tree_selection,
    validate_stage_a_tree_index,
)
from teacher_logit_reco.relation_expert_token_bridge.supplemental_kd_baselines import (
    KD_BASELINE_ARCHITECTURES,
    KD_BASELINE_RESULT_CONTRACT,
    KD_BASELINE_SEEDS,
    conventional_kd_configuration,
    file_sha256,
    validate_kd_baseline_plan,
)
from teacher_logit_reco.relation_expert_token_bridge.streamed_abc import (
    STREAMED_ABC_TREE_INDEX_CONTRACT,
)
from teacher_logit_reco.relational_part.capacity import pair_encoder_flops
from teacher_logit_reco.relational_part.model import (
    RelationalFamilyParticleTransformer,
    RelationalParticleTransformer,
)


def _model(
    architecture: str,
    *,
    configuration: Mapping[str, Any],
    relation: Mapping[str, Any],
    region: Mapping[str, Any],
    weaver: Any,
) -> Any:
    classifier = (
        RelationalParticleTransformer(weaver_module=weaver)
        if architecture == "O_BASE"
        else RelationalFamilyParticleTransformer(
            FAMILIES,
            normalization_artifact=relation,
            region_normalization_artifact=region,
            weaver_module=weaver,
        )
    )
    return OfflineClassifierAdapter(
        classifier, expert_configuration=configuration
    )


def _load_selected_checkpoint(model: Any, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, Mapping):
        raise ValueError("supplemental KD checkpoint lacks model state")
    model.load_state_dict(state, strict=True)


def _region_trees(
    *,
    parent: Path,
    plan: Mapping[str, Any],
    split: str,
    arrays: Mapping[str, np.ndarray],
) -> tuple[list[Mapping[str, Any]], str]:
    tree_root = (
        parent / "inputs/region_tree/offline" / f"{split}_exclusive_ca_v1"
    )
    trees, manifest = load_authenticated_tree_selection(
        tree_root,
        [str(value) for value in arrays["identities"].tolist()],
    )
    key = f"region_tree_{split}_manifest"
    expected = plan["parent_artifacts"].get(key)
    if expected is not None:
        if (
            manifest["content_hash"] != expected["content_hash"]
            or file_sha256(tree_root / "manifest.json")
            != expected["file_sha256"]
        ):
            raise ValueError(
                f"supplemental KD {split} REGION tree lineage differs"
            )
    else:
        # Recovery compatibility for the already-published v1 plan.  The v1
        # plan predates direct tree-manifest binding, so authenticate it
        # through the parent campaign's immutable Stage-A tree index.
        standard_index = parent / "inputs/region_tree/tree_cache_index.json"
        streamed_index = (
            parent / "inputs/region_tree/tree_cache_index_streamed_abc.json"
        )
        index_path = standard_index if standard_index.is_file() else streamed_index
        tree_index = load_hashed_json(index_path)
        if tree_index.get("contract") == STREAMED_ABC_TREE_INDEX_CONTRACT:
            validate_content_hash(
                tree_index,
                expected_contract=STREAMED_ABC_TREE_INDEX_CONTRACT,
            )
            if (
                int(tree_index.get("schema_version", -1)) != 1
                or int(tree_index.get("view_count", -1))
                != len(tree_index.get("views", ()))
                or tree_index.get("view_ids")
                != [str(row["view_id"]) for row in tree_index["views"]]
            ):
                raise ValueError("supplemental KD streamed tree index differs")
        else:
            validate_stage_a_tree_index(tree_index)
        campaign = load_hashed_json(parent / "campaign_spec.json")
        rows = {
            str(row["view_id"]): row for row in tree_index["views"]
        }
        row = rows.get(f"offline:{split}")
        if (
            row is None
            or row.get("tree_manifest_sha256")
            != manifest["content_hash"]
            or tree_index.get("campaign_spec_sha256")
            != campaign["content_hash"]
            or tree_index.get("source") != campaign.get("source")
            or plan.get("parent_campaign_spec_sha256")
            != campaign["content_hash"]
        ):
            raise ValueError(
                f"supplemental KD legacy {split} REGION tree lineage differs"
            )
    return trees, str(manifest["content_hash"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--architecture", required=True, choices=KD_BASELINE_ARCHITECTURES)
    parser.add_argument("--seed", required=True, type=int, choices=KD_BASELINE_SEEDS)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    plan = load_hashed_json(args.plan)
    validate_kd_baseline_plan(plan)
    coordinate = {"architecture": args.architecture, "seed": args.seed}
    if coordinate not in plan["coordinates"]:
        raise ValueError("supplemental KD coordinate is not sealed")
    output = args.output_root / args.architecture / f"seed_{args.seed}"
    result_path = output / "result.json"
    if result_path.is_file():
        result = load_hashed_json(
            result_path, expected_contract=KD_BASELINE_RESULT_CONTRACT
        )
        if (
            result.get("plan_sha256") != plan["content_hash"]
            or result.get("architecture") != args.architecture
            or int(result.get("seed", -1)) != args.seed
        ):
            raise ValueError("reusable supplemental KD result differs")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    parent = Path(plan["parent_campaign_root"])
    campaign = load_hashed_json(parent / "campaign_spec.json")
    source = source_snapshot(REPO_ROOT)
    relation = load_hashed_json(
        parent / "inputs/normalization/offline_500k/relation.json"
    )
    region = load_hashed_json(
        parent / "inputs/normalization/offline_500k/region.json"
    )
    loss_registry = load_hashed_json(parent / "registry/retb_expert_losses.json")
    step3 = load_hashed_json(parent / "registry/retb_step3_architecture_bundle.json")
    manifest = load_hashed_json(parent / "inputs/teacher_logits/ELOSS_KD_DOMINANT.json")
    train_path = parent / "inputs/teacher_logits/ELOSS_KD_DOMINANT/model_train.npz"
    stop_path = parent / "inputs/teacher_logits/ELOSS_KD_DOMINANT/val_stop.npz"
    design_path = Path(plan["parent_artifacts"]["val_design_inputs"]["path"])
    train_arrays = _load_npz(train_path)
    stop_arrays = _load_npz(stop_path)
    design_arrays = _load_npz(design_path)
    needs_region_trees = args.architecture == "O_FULLREL"
    tree_manifest_hashes: dict[str, str] = {}
    if needs_region_trees:
        train_trees, tree_manifest_hashes["model_train"] = _region_trees(
            parent=parent,
            plan=plan,
            split="model_train",
            arrays=train_arrays,
        )
        stop_trees, tree_manifest_hashes["val_stop"] = _region_trees(
            parent=parent,
            plan=plan,
            split="val_stop",
            arrays=stop_arrays,
        )
        design_trees, tree_manifest_hashes["val_design"] = _region_trees(
            parent=parent,
            plan=plan,
            split="val_design",
            arrays=design_arrays,
        )
    else:
        train_trees = stop_trees = design_trees = None
    train_dataset = _dataset(
        train_arrays,
        region_trees=train_trees,
    )
    stop_dataset = _dataset(
        stop_arrays,
        region_trees=stop_trees,
    )
    design_dataset = _dataset(
        design_arrays,
        region_trees=design_trees,
    )
    train_loader = make_offline_expert_loader(
        train_dataset, seed=args.seed, training=True, batch_size=64
    )
    stop_loader = make_offline_expert_loader(
        stop_dataset, seed=args.seed, training=False, batch_size=64
    )
    design_loader = make_offline_expert_loader(
        design_dataset, seed=args.seed, training=False, batch_size=64
    )
    configuration = conventional_kd_configuration(args.architecture)
    weaver = importlib.import_module("weaver.nn.model.ParticleTransformer")
    model = _model(
        args.architecture,
        configuration=configuration,
        relation=relation,
        region=region,
        weaver=weaver,
    )
    base_flops = analytical_particle_transformer_flops(
        configuration=(128, 4, 8, 8, 2)
    )
    flops = base_flops + pair_encoder_flops(
        4 if args.architecture == "O_BASE" else 62,
        (64, 64, 64),
    )
    profile = build_capacity_profile(
        control_id=args.architecture,
        model=model,
        analytical_flops_batch1=flops,
        source_snapshot=source,
    )
    write_immutable_json(output / "complete_graph_profile.json", profile)
    config = OfflineExpertTrainingConfig(
        seed=args.seed,
        initialization="INIT_SCRATCH",
        loss_id="ELOSS_KD_DOMINANT",
        learning_rate=1.0e-3,
        particle_dropout=0.0,
        maximum_epochs=40,
        microbatch_size=64,
        gradient_accumulation_steps=2,
        effective_batch_size=128,
        campaign_profile="production",
    )
    run_id = f"retb_supp_kd_{args.architecture.lower()}_s{args.seed}"
    run_record = {
        "run_id": run_id,
        "seed": args.seed,
        "configuration": configuration,
    }
    teacher_sha = plan["teacher_checkpoint_sha256"]
    lineage = {
        "supplemental_plan": plan["content_hash"],
        "campaign_spec": campaign["content_hash"],
        "step3_bundle": step3["content_hash"],
        "run_registry": plan["content_hash"],
        "model_train_inputs": file_sha256(train_path),
        "val_stop_inputs": file_sha256(stop_path),
        "val_design_inputs": file_sha256(design_path),
        "teacher_logits_manifest": manifest["content_hash"],
        "teacher_SELECTED_STRONGEST": teacher_sha,
        "relation_normalization": relation["content_hash"],
        "region_normalization": region["content_hash"],
    }
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    registration = train_offline_expert(
        model=model,
        train_loader=train_loader,
        val_stop_loader=stop_loader,
        output_dir=output,
        run_record=run_record,
        run_registry_sha256=plan["content_hash"],
        step3_bundle_sha256=step3["content_hash"],
        global_determinism_sha256=campaign["parent_artifact_hashes"]["global_determinism"],
        expert_loss_registry=loss_registry,
        lineage_hashes=lineage,
        config=config,
        device=device,
        resource_profile=profile,
        teacher_checkpoint_hashes={"SELECTED_STRONGEST": teacher_sha},
        teacher_logits_manifest=manifest,
        resume=True,
    )
    _load_selected_checkpoint(model, output / "best_model_val.pt")
    model.to(device)
    prediction = _collect_predictions(model, design_loader, device=device)
    prediction_publication = _publish_predictions(
        output / "val_design_predictions.npz",
        logits=np.asarray(prediction["logits"], dtype=np.float32),
        labels=np.asarray(prediction["labels"], dtype=np.int64),
        identities=np.asarray(prediction["identities"]),
    )
    design_metrics = bind_source(
        prediction["metrics"], source_snapshot=source
    )
    write_immutable_json(output / "val_design_metrics.json", design_metrics)
    result = bind_source(
        with_content_hash(
            {
                "contract": KD_BASELINE_RESULT_CONTRACT,
                "schema_version": 2,
                "plan_sha256": plan["content_hash"],
                "architecture": args.architecture,
                "seed": args.seed,
                "run_id": run_id,
                "loss_id": "ELOSS_KD_DOMINANT",
                "teacher_id": "O_FULLREL",
                "teacher_checkpoint_sha256": teacher_sha,
                "region_tree_manifest_sha256s": tree_manifest_hashes,
                "checkpoint_registration_sha256": registration["content_hash"],
                "checkpoint_sha256": registration["checkpoint_sha256"],
                "selected_epoch": registration["selected_epoch"],
                "selected_val_stop": registration["selected_val_stop"],
                "val_design_metrics": design_metrics,
                "val_design_predictions_sha256": prediction_publication["file_sha256"],
                "fixed_budget_completed": True,
                "performance_based_termination": False,
                "final_test_accessed": False,
            }
        ),
        source_snapshot=source,
    )
    write_immutable_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
