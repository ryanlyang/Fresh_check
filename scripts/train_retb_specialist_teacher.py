#!/usr/bin/env python3
"""Train one ordinary uncompressed relation-specialist teacher."""

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

from scripts.train_retb_offline_expert import _dataset, _load_npz  # noqa: E402
from scripts.train_retb_supplemental_kd_baseline import _region_trees  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    source_record,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_training import (  # noqa: E402
    OfflineExpertTrainingConfig,
    _publish_predictions,
    make_offline_expert_loader,
    train_offline_expert,
)
from teacher_logit_reco.relation_expert_token_bridge.offline_capacity_models import (  # noqa: E402
    OfflineClassifierAdapter,
    analytical_particle_transformer_flops,
)
from teacher_logit_reco.relation_expert_token_bridge.offline_capacity_training import (  # noqa: E402
    _collect_predictions,
    build_capacity_profile,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.supplemental_specialist_kd import (  # noqa: E402
    NEW_SPECIALIST_TEACHERS,
    SPECIALIST_TEACHER_CONTRACT,
    file_sha256,
    specialist_teacher_configuration,
    validate_specialist_kd_plan,
)
from teacher_logit_reco.relational_part.capacity import pair_encoder_flops  # noqa: E402
from teacher_logit_reco.relational_part.model import (  # noqa: E402
    RelationalFamilyParticleTransformer,
)
from teacher_logit_reco.relational_part.pair_builder import (  # noqa: E402
    SUPPORTED_FAMILY_DIMENSIONS,
)


def _load_checkpoint(model: Any, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expert", required=True, choices=NEW_SPECIALIST_TEACHERS)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    plan = load_hashed_json(args.plan)
    validate_specialist_kd_plan(plan, check_parent_bytes=False)
    source = source_snapshot(REPO_ROOT)
    if plan.get("source") != source_record(source):
        raise ValueError("specialist teacher source differs from sealed plan")
    output = args.output_root / args.expert
    result_path = output / "teacher_manifest.json"
    if result_path.is_file():
        result = load_hashed_json(result_path, expected_contract=SPECIALIST_TEACHER_CONTRACT)
        if (
            result.get("plan_sha256") != plan["content_hash"]
            or result.get("expert") != args.expert
            or result.get("fixed_budget_completed") is not True
            or result.get("performance_based_termination") is not False
        ):
            raise ValueError("reusable specialist teacher differs")
        if result.get("source") != plan.get("source"):
            raise ValueError("reusable specialist teacher source differs")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    parent = Path(plan["parent_campaign_root"])
    campaign = load_hashed_json(parent / "campaign_spec.json")
    step3 = load_hashed_json(parent / "registry/retb_step3_architecture_bundle.json")
    losses = load_hashed_json(parent / "registry/retb_expert_losses.json")
    relation = load_hashed_json(parent / "inputs/normalization/offline_500k/relation.json")
    region = load_hashed_json(parent / "inputs/normalization/offline_500k/region.json")
    paths = {
        "model_train": Path(plan["parent_artifacts"]["common_teacher_train"]["path"]),
        "val_stop": Path(plan["parent_artifacts"]["common_teacher_val_stop"]["path"]),
        "val_design": Path(plan["parent_artifacts"]["val_design_inputs"]["path"]),
    }
    path_keys = {
        "model_train": "common_teacher_train",
        "val_stop": "common_teacher_val_stop",
        "val_design": "val_design_inputs",
    }
    for split, path in paths.items():
        if file_sha256(path) != plan["parent_artifacts"][path_keys[split]]["file_sha256"]:
            raise ValueError(f"specialist teacher {split} input bytes drifted")
    arrays = {name: _load_npz(path) for name, path in paths.items()}
    trees: dict[str, Any] = {name: None for name in paths}
    tree_hashes: dict[str, str] = {}
    if args.expert == "REGION":
        for split in paths:
            trees[split], tree_hashes[split] = _region_trees(
                parent=parent, plan=plan, split=split, arrays=arrays[split]
            )
    datasets = {
        name: _dataset(
            {key: value for key, value in split_arrays.items() if not key.startswith("teacher_logits_")},
            region_trees=trees[name],
        )
        for name, split_arrays in arrays.items()
    }
    loaders = {
        name: make_offline_expert_loader(
            dataset, seed=101, training=name == "model_train", batch_size=64
        )
        for name, dataset in datasets.items()
    }
    inference_loaders = {
        name: make_offline_expert_loader(
            dataset, seed=101, training=False, batch_size=64
        )
        for name, dataset in datasets.items()
    }
    configuration = specialist_teacher_configuration(args.expert)
    weaver = importlib.import_module("weaver.nn.model.ParticleTransformer")
    torch.manual_seed(101)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(101)
    model = OfflineClassifierAdapter(
        RelationalFamilyParticleTransformer(
            (configuration["relation_family"],),
            normalization_artifact=relation,
            region_normalization_artifact=region,
            weaver_module=weaver,
        ),
        expert_configuration=configuration,
    )
    flops = analytical_particle_transformer_flops(configuration=(128, 4, 8, 8, 2))
    flops += pair_encoder_flops(
        4 + SUPPORTED_FAMILY_DIMENSIONS[configuration["relation_family"]],
        (64, 64, 64),
    )
    profile = build_capacity_profile(
        control_id=f"SPECIALIST_{args.expert}", model=model,
        analytical_flops_batch1=flops, source_snapshot=source,
    )
    write_immutable_json(output / "complete_graph_profile.json", profile)
    config = OfflineExpertTrainingConfig(seed=101, campaign_profile="production")
    run_id = f"retb_specialist_teacher_{args.expert.lower()}_s101"
    registration = train_offline_expert(
        model=model,
        train_loader=loaders["model_train"],
        val_stop_loader=loaders["val_stop"],
        output_dir=output,
        run_record={"run_id": run_id, "seed": 101, "configuration": configuration},
        run_registry_sha256=plan["content_hash"],
        step3_bundle_sha256=step3["content_hash"],
        global_determinism_sha256=campaign["parent_artifact_hashes"]["global_determinism"],
        expert_loss_registry=losses,
        lineage_hashes={
            "supplemental_plan": plan["content_hash"],
            "campaign_spec": campaign["content_hash"],
            "model_train_inputs": file_sha256(paths["model_train"]),
            "val_stop_inputs": file_sha256(paths["val_stop"]),
            "val_design_inputs": file_sha256(paths["val_design"]),
            "relation_normalization": relation["content_hash"],
            "region_normalization": region["content_hash"],
        },
        config=config,
        device=torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device),
        resource_profile=profile,
        resume=True,
    )
    _load_checkpoint(model, output / "best_model_val.pt")
    device = next(model.parameters()).device
    cache_records = {}
    split_metrics = {}
    for split, loader in inference_loaders.items():
        prediction = _collect_predictions(model, loader, device=device)
        cache = output / f"teacher_logits_{split}.npz"
        publication = _publish_predictions(
            cache,
            logits=np.asarray(prediction["logits"], dtype=np.float32),
            labels=np.asarray(prediction["labels"], dtype=np.int64),
            identities=np.asarray(prediction["identities"]),
        )
        cache_records[split] = {"path": str(cache.resolve()), **publication}
        split_metrics[split] = prediction["metrics"]
    result = bind_source(
        with_content_hash({
            "contract": SPECIALIST_TEACHER_CONTRACT,
            "schema_version": 1,
            "plan_sha256": plan["content_hash"],
            "expert": args.expert,
            "seed": 101,
            "configuration": configuration,
            "checkpoint_registration_sha256": registration["content_hash"],
            "checkpoint_path": str((output / "best_model_val.pt").resolve()),
            "checkpoint_sha256": registration["checkpoint_sha256"],
            "selected_epoch": registration["selected_epoch"],
            "selected_val_stop": registration["selected_val_stop"],
            "prediction_caches": cache_records,
            "split_metrics": split_metrics,
            "region_tree_manifest_sha256s": tree_hashes,
            "fixed_budget_completed": True,
            "performance_based_termination": False,
            "final_test_accessed": False,
        }),
        source_snapshot=source,
    )
    write_immutable_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
