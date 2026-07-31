#!/usr/bin/env python3
"""Train a mandatory HOSD offline teacher or authenticate a compatible one.

``--execute-training`` runs the real offline-input RPT_BASE/RPT_FULL_ALL
trainer.  Registration mode is restricted to an already produced compatible
checkpoint.  Teacher locking remains a separate non-producing command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_teacher_training_manifest,
    complete_teacher_training,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.teacher_training import (  # noqa: E402
    train_offline_teacher,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    load_hashed_json,
    write_immutable_json,
)


def _load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require_source_if_declared(
    artifact: dict, *, source: dict, name: str
) -> None:
    if artifact.get("source") is not None and artifact["source"] != source:
        raise ValueError(f"{name} source differs from the active HOSD campaign")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--model-contract-o-base", required=True, type=Path)
    parser.add_argument("--model-contract-o-fullrel", required=True, type=Path)
    parser.add_argument("--normalizer-hashes", required=True, type=Path)
    parser.add_argument(
        "--population", choices=("target_500k", "target_scale"), default="target_500k"
    )
    parser.add_argument("--training-manifest-output", type=Path)
    parser.add_argument("--teacher-id", choices=("O_BASE", "O_FULLREL"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--selector-trace", type=Path)
    parser.add_argument("--architecture", type=Path)
    parser.add_argument("--completion-output", type=Path)
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--offline-manifest", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--validation-partition", type=Path)
    parser.add_argument("--screening-registry", type=Path)
    parser.add_argument("--relation-registry", type=Path)
    parser.add_argument("--relation-normalizer", type=Path)
    parser.add_argument("--region-normalizer", type=Path)
    parser.add_argument("--global-determinism", type=Path)
    parser.add_argument("--tree-root", type=Path)
    parser.add_argument("--validation-tree-root", type=Path)
    parser.add_argument("--training-input-npz", type=Path)
    parser.add_argument("--training-labels-npz", type=Path)
    parser.add_argument("--training-output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    base = load_hashed_json(args.model_contract_o_base)
    full = load_hashed_json(args.model_contract_o_fullrel)
    _require_source_if_declared(
        base, source=campaign["source"], name="O_BASE model contract"
    )
    _require_source_if_declared(
        full, source=campaign["source"], name="O_FULLREL model contract"
    )
    normalizers = _load_object(args.normalizer_hashes)
    split_hash = campaign.get("split_manifest_hash") or campaign.get(
        "parent_artifact_hashes", {}
    ).get("split_manifest") or campaign.get("shared_parent_hashes", {}).get(
        (
            "scale_train_manifest"
            if args.population == "target_scale"
            else "validation_partition_manifest"
        )
    )
    if split_hash is None:
        raise ValueError("campaign does not bind a split manifest")
    training = build_teacher_training_manifest(
        campaign_spec_sha256=campaign["content_hash"],
        split_manifest_sha256=split_hash,
        model_contract_hashes={
            "O_BASE": base["content_hash"],
            "O_FULLREL": full["content_hash"],
        },
        normalizer_hashes=normalizers,
        source=campaign["source"],
        population=args.population,
    )
    training_output = args.training_manifest_output or (
        args.campaign_root / "teachers" / "training_manifest.json"
    )
    write_immutable_json(training_output, training)
    result: dict[str, object] = {
        "training_manifest_sha256": training["content_hash"],
        "training_manifest_output": str(training_output.resolve()),
    }
    if args.execute_training:
        required_execution = {
            "--teacher-id": args.teacher_id,
            "--offline-manifest": args.offline_manifest,
            "--validation-partition": args.validation_partition,
            "--screening-registry": args.screening_registry,
            "--relation-registry": args.relation_registry,
            "--relation-normalizer": args.relation_normalizer,
            "--global-determinism": args.global_determinism,
        }
        missing = [name for name, value in required_execution.items() if value is None]
        if missing:
            raise ValueError(
                f"--execute-training lacks required arguments: {missing}"
            )
        if args.population == "target_scale" and (
            args.training_input_npz is None
            or args.training_labels_npz is None
        ):
            raise ValueError(
                "scale teacher execution lacks training input/labels"
            )
        teacher_output = args.training_output_dir or (
            args.campaign_root
            / "teachers"
            / args.population
            / str(args.teacher_id)
        )
        selected_contract = base if args.teacher_id == "O_BASE" else full
        validation = load_hashed_json(
            args.validation_partition,
            expected_contract="retb_validation_partition_manifest_v1",
        )
        screening = load_hashed_json(args.screening_registry)
        relation_registry = load_hashed_json(args.relation_registry)
        relation_normalizer = load_hashed_json(args.relation_normalizer)
        region_normalizer = (
            load_hashed_json(args.region_normalizer)
            if args.region_normalizer is not None
            else None
        )
        determinism = load_hashed_json(args.global_determinism)
        for name, artifact in (
            ("screening registry", screening),
            ("relation registry", relation_registry),
            ("relation normalizer", relation_normalizer),
            ("global determinism", determinism),
            ("REGION normalizer", region_normalizer),
        ):
            if artifact is not None:
                _require_source_if_declared(
                    artifact, source=campaign["source"], name=name
                )
        expected_normalizers = set(
            next(
                row for row in training["teachers"]
                if row["teacher_id"] == args.teacher_id
            )["normalizer_hashes"].values()
        )
        actual_normalizers = {
            relation_normalizer["content_hash"],
            *(
                ()
                if region_normalizer is None
                else (region_normalizer["content_hash"],)
            ),
        }
        if not actual_normalizers.issubset(expected_normalizers):
            raise ValueError(
                "executed teacher normalizers differ from its frozen training manifest"
            )
        registration = train_offline_teacher(
            teacher_id=args.teacher_id,
            campaign=campaign,
            offline_manifest_path=args.offline_manifest,
            data_dir=args.data_dir,
            validation_partition=validation,
            screening_registry=screening,
            relation_registry=relation_registry,
            relation_normalizer=relation_normalizer,
            region_normalizer=region_normalizer,
            global_determinism=determinism,
            model_contract=selected_contract,
            output_dir=teacher_output,
            tree_root=args.tree_root,
            validation_tree_root=args.validation_tree_root,
            training_input_npz=args.training_input_npz,
            training_labels_npz=args.training_labels_npz,
            device=args.device,
            miniature=campaign.get("campaign_profile")
            == "miniature_test",
            training_split=(
                "scale_train"
                if args.population == "target_scale"
                else "model_train"
            ),
        )
        if registration.get("inference_input_role") != "offline_teacher":
            raise RuntimeError("teacher trainer emitted the wrong inference-input role")
        checkpoint = teacher_output / "best_model_val.pt"
        completion = complete_teacher_training(
            training,
            teacher_id=args.teacher_id,
            checkpoint_path=checkpoint,
            selector_trace={
                "selected_epoch": registration["selected_epoch"],
                "selected_val_stop": registration["selected_val_stop"],
                "checkpoint_registration_sha256": registration["content_hash"],
            },
            architecture=selected_contract,
            source=campaign["source"],
        )
        completion_output = args.completion_output or (
            teacher_output / "training_completion.json"
        )
        write_immutable_json(completion_output, completion)
        result.update(
            {
                "teacher_id": args.teacher_id,
                "checkpoint_registration_sha256": registration["content_hash"],
                "training_completion_sha256": completion["content_hash"],
                "checkpoint_sha256": completion["checkpoint_sha256"],
            }
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    completion_inputs = (
        args.teacher_id,
        args.checkpoint,
        args.selector_trace,
        args.architecture,
        args.completion_output,
    )
    if any(value is not None for value in completion_inputs):
        if any(value is None for value in completion_inputs):
            raise ValueError(
                "checkpoint completion requires --teacher-id, --checkpoint, "
                "--selector-trace, --architecture, and --completion-output"
            )
        completion = complete_teacher_training(
            training,
            teacher_id=args.teacher_id,
            checkpoint_path=args.checkpoint,
            selector_trace=_load_object(args.selector_trace),
            architecture=_load_object(args.architecture),
            source=campaign["source"],
        )
        write_immutable_json(args.completion_output, completion)
        result.update(
            {
                "teacher_id": args.teacher_id,
                "training_completion_sha256": completion["content_hash"],
                "checkpoint_sha256": completion["checkpoint_sha256"],
            }
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
