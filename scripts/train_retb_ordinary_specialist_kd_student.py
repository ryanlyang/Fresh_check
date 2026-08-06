#!/usr/bin/env python3
"""Train one ordinary-head matched- or hybrid-specialist KD control."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_retb_offline_expert import _dataset, _load_npz  # noqa: E402
from scripts.train_retb_specialist_kd_student import (  # noqa: E402
    _assert_join,
    _specialist_arrays,
)
from scripts.train_retb_supplemental_kd_baseline import _region_trees  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    source_record,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (  # noqa: E402
    evaluate_classification,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_training import (  # noqa: E402
    _publish_predictions,
    make_offline_expert_loader,
)
from teacher_logit_reco.relation_expert_token_bridge.offline_capacity_models import (  # noqa: E402
    OfflineClassifierAdapter,
)
from teacher_logit_reco.relation_expert_token_bridge.offline_capacity_training import (  # noqa: E402,E501
    _collect_predictions,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.specialist_kd_training import (  # noqa: E402
    train_specialist_kd_student,
)
from teacher_logit_reco.relation_expert_token_bridge.supplemental_ordinary_specialist_kd import (  # noqa: E402,E501
    ORDINARY_SPECIALIST_CHECKPOINT_CONTRACT,
    ORDINARY_SPECIALIST_RESUME_CONTRACT,
    ORDINARY_SPECIALIST_STUDENT_CONTRACT,
    ordinary_student_configuration,
    validate_ordinary_specialist_kd_plan,
)
from teacher_logit_reco.relation_expert_token_bridge.supplemental_specialist_kd import (  # noqa: E402,E501
    SPECIALIST_CONDITIONS,
    SPECIALIST_EXPERTS,
    file_sha256,
    validate_specialist_kd_plan,
)
from teacher_logit_reco.relational_part.model import (  # noqa: E402
    RelationalFamilyParticleTransformer,
    RelationalParticleTransformer,
)


def _ordinary_model(
    *,
    expert: str,
    configuration: dict[str, Any],
    relation: dict[str, Any],
    region: dict[str, Any],
    weaver: Any,
) -> Any:
    family = configuration["relation_family"]
    if family is None:
        classifier = RelationalParticleTransformer(weaver_module=weaver)
    else:
        classifier = RelationalFamilyParticleTransformer(
            (family,),
            normalization_artifact=relation,
            region_normalization_artifact=region,
            weaver_module=weaver,
        )
    return OfflineClassifierAdapter(
        classifier, expert_configuration=configuration
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--condition", required=True, choices=SPECIALIST_CONDITIONS)
    parser.add_argument("--expert", required=True, choices=SPECIALIST_EXPERTS)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    plan = load_hashed_json(args.plan)
    validate_ordinary_specialist_kd_plan(plan, check_parent_bytes=False)
    source = source_snapshot(REPO_ROOT)
    if plan.get("source") != source_record(source):
        raise ValueError("ordinary specialist KD source differs from sealed plan")
    coordinate = {"condition": args.condition, "expert": args.expert, "seed": 101}
    if coordinate not in plan["student_coordinates"]:
        raise ValueError("ordinary specialist KD coordinate is not sealed")

    output = args.output_root / args.condition / args.expert
    result_path = output / "result.json"
    if result_path.is_file():
        result = load_hashed_json(
            result_path, expected_contract=ORDINARY_SPECIALIST_STUDENT_CONTRACT
        )
        if (
            result.get("plan_sha256") != plan["content_hash"]
            or result.get("coordinate") != coordinate
            or result.get("fixed_budget_completed") is not True
            or result.get("performance_based_termination") is not False
            or result.get("source") != plan.get("source")
        ):
            raise ValueError("reusable ordinary specialist KD student differs")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    compact_plan = load_hashed_json(
        Path(plan["parent_artifacts"]["compact_plan"]["path"])
    )
    validate_specialist_kd_plan(compact_plan, check_parent_bytes=False)
    parent = Path(compact_plan["parent_campaign_root"])
    teacher_root = Path(plan["compact_specialist_root"]) / "runs/teachers"
    common_paths = {
        "model_train": Path(
            compact_plan["parent_artifacts"]["common_teacher_train"]["path"]
        ),
        "val_stop": Path(
            compact_plan["parent_artifacts"]["common_teacher_val_stop"]["path"]
        ),
        "val_design": Path(
            compact_plan["parent_artifacts"]["val_design_inputs"]["path"]
        ),
    }
    arrays = {split: _load_npz(path) for split, path in common_paths.items()}
    common_keys = {
        "model_train": "common_teacher_train",
        "val_stop": "common_teacher_val_stop",
        "val_design": "val_design_inputs",
    }
    for split, path in common_paths.items():
        expected = compact_plan["parent_artifacts"][common_keys[split]][
            "file_sha256"
        ]
        if file_sha256(path) != expected:
            raise ValueError(f"ordinary specialist KD {split} input drifted")

    specialist_hashes: dict[str, str] = {}
    specialist_checkpoint = None
    for split in ("model_train", "val_stop"):
        specialist, checkpoint, cache_hash = _specialist_arrays(
            plan=compact_plan,
            teacher_root=teacher_root,
            expert=args.expert,
            split=split,
        )
        _assert_join(arrays[split], specialist, split=split)
        arrays[split] = {
            **{
                name: value
                for name, value in arrays[split].items()
                if not name.startswith("teacher_logits_")
            },
            "teacher_logits_COMMON": np.asarray(
                arrays[split]["teacher_logits_SELECTED_STRONGEST"],
                dtype=np.float32,
            ),
            "teacher_logits_SPECIALIST": np.asarray(
                specialist["logits"], dtype=np.float32
            ),
        }
        specialist_checkpoint = checkpoint
        specialist_hashes[split] = cache_hash

    trees: dict[str, Any] = {split: None for split in arrays}
    tree_hashes: dict[str, str] = {}
    if args.expert == "REGION":
        for split in arrays:
            trees[split], tree_hashes[split] = _region_trees(
                parent=parent,
                plan=compact_plan,
                split=split,
                arrays=arrays[split],
            )
    datasets = {
        split: _dataset(values, region_trees=trees[split])
        for split, values in arrays.items()
    }
    loaders = {
        split: make_offline_expert_loader(
            dataset,
            seed=101,
            training=split == "model_train",
            batch_size=64,
        )
        for split, dataset in datasets.items()
    }

    configuration = ordinary_student_configuration(args.expert, args.condition)
    relation = load_hashed_json(
        parent / "inputs/normalization/offline_500k/relation.json"
    )
    region = load_hashed_json(
        parent / "inputs/normalization/offline_500k/region.json"
    )
    if (
        relation["content_hash"]
        != compact_plan["parent_artifacts"]["relation_normalization"]["content_hash"]
        or region["content_hash"]
        != compact_plan["parent_artifacts"]["region_normalization"]["content_hash"]
    ):
        raise ValueError("ordinary specialist KD normalization drifted")
    weaver = importlib.import_module("weaver.nn.model.ParticleTransformer")
    torch.manual_seed(101)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(101)
    model = _ordinary_model(
        expert=args.expert,
        configuration=configuration,
        relation=relation,
        region=region,
        weaver=weaver,
    )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    run_id = (
        f"retb_ordinary_{args.condition.lower()}_"
        f"{args.expert.lower()}_s101"
    )
    training = train_specialist_kd_student(
        model=model,
        train_loader=loaders["model_train"],
        val_stop_loader=loaders["val_stop"],
        output_dir=output,
        condition=args.condition,
        run_id=run_id,
        plan_sha256=plan["content_hash"],
        device=device,
        checkpoint_contract=ORDINARY_SPECIALIST_CHECKPOINT_CONTRACT,
        resume_contract=ORDINARY_SPECIALIST_RESUME_CONTRACT,
    )
    predictions: dict[str, dict[str, Any]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for split in ("val_stop", "val_design"):
        prediction = _collect_predictions(model, loaders[split], device=device)
        path = output / f"{split}_predictions.npz"
        publication = _publish_predictions(
            path,
            logits=np.asarray(prediction["logits"], dtype=np.float32),
            labels=np.asarray(prediction["labels"], dtype=np.int64),
            identities=np.asarray(prediction["identities"]),
        )
        predictions[split] = {"path": str(path.resolve()), **publication}
        metrics[split] = evaluate_classification(
            prediction["logits"], prediction["labels"], split=split
        )

    compact_key = f"compact_{args.condition}_{args.expert}"
    result = bind_source(
        with_content_hash(
            {
                "contract": ORDINARY_SPECIALIST_STUDENT_CONTRACT,
                "schema_version": 1,
                "plan_sha256": plan["content_hash"],
                "coordinate": coordinate,
                "run_id": run_id,
                "configuration": configuration,
                "common_teacher_checkpoint_sha256": compact_plan[
                    "parent_artifacts"
                ]["common_teacher_checkpoint"]["file_sha256"],
                "specialist_teacher_checkpoint_sha256": specialist_checkpoint,
                "specialist_teacher_prediction_sha256s": specialist_hashes,
                "compact_counterpart_sha256": plan["parent_artifacts"][
                    compact_key
                ]["content_hash"],
                "objective": plan["objective"],
                "selected_epoch": training["selected_epoch"],
                "selected_val_stop": training["selected_val_stop"],
                "split_metrics": metrics,
                "predictions": predictions,
                "region_tree_manifest_sha256s": tree_hashes,
                "training_record": training,
                "checkpoint_path": str((output / "best_model_val.pt").resolve()),
                "checkpoint_sha256": file_sha256(output / "best_model_val.pt"),
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "epochs_completed": 40,
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
