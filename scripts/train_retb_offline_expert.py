#!/usr/bin/env python3
"""Train one immutable RETB Stage-B pure-offline expert row."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_model import (  # noqa: E402
    RetbExpertModel,
    RetbParticleEncoder,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_training import (  # noqa: E402
    EXPERT_LOSS_CANDIDATES,
    OfflineExpertDataset,
    OfflineExpertTrainingConfig,
    make_offline_expert_loader,
    train_offline_expert,
    validate_teacher_logits_manifest,
)
from teacher_logit_reco.relation_expert_token_bridge.step4 import (  # noqa: E402
    resolve_stage_b_run,
    validate_stage_b_run_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)
from teacher_logit_reco.relational_part.ca_tree import (  # noqa: E402
    unpack_tree_shard,
)

import torch  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--train-npz", type=Path)
    parser.add_argument("--val-stop-npz", type=Path)
    parser.add_argument("--relation-normalization", type=Path)
    parser.add_argument("--region-normalization", type=Path)
    parser.add_argument("--region-tree-root", type=Path)
    parser.add_argument("--initialization-checkpoint", type=Path)
    parser.add_argument("--attachment-pretraining-record", type=Path)
    parser.add_argument("--resource-profile", type=Path)
    parser.add_argument("--teacher-logits-manifest", type=Path)
    parser.add_argument(
        "--teacher-checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--microbatch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _teacher_paths(rows: Sequence[str]) -> dict[str, Path]:
    output = {}
    for row in rows:
        if "=" not in row:
            raise ValueError("--teacher-checkpoint requires NAME=PATH")
        name, value = row.split("=", 1)
        if not name or name in output:
            raise ValueError("teacher checkpoint name is empty or duplicated")
        output[name] = Path(value)
    return output


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"tokens", "mask", "labels", "identities"}
        if not required.issubset(payload.files):
            raise ValueError(f"offline NPZ lacks {sorted(required - set(payload.files))}")
        return {name: np.asarray(payload[name]) for name in payload.files}


def _load_trees(
    root: Path,
    *,
    split: str,
    identities: Sequence[str],
) -> list[Mapping[str, Any]]:
    split_root = root / f"{split}_exclusive_ca_v1"
    shards = sorted((split_root / "shards").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"REGION tree shards are absent under {split_root}")
    by_identity: dict[str, Mapping[str, Any]] = {}
    for shard in shards:
        shard_identities, trees = unpack_tree_shard(shard)
        for identity, tree in zip(shard_identities, trees):
            if identity in by_identity:
                raise ValueError("REGION tree identity is duplicated")
            by_identity[str(identity)] = tree
    missing = [identity for identity in identities if identity not in by_identity]
    if missing:
        raise ValueError(f"REGION trees lack {len(missing)} requested identities")
    return [by_identity[str(identity)] for identity in identities]


def _dataset(
    arrays: Mapping[str, np.ndarray],
    *,
    region_trees: Sequence[Mapping[str, Any]] | None,
) -> OfflineExpertDataset:
    teacher = {
        name.removeprefix("teacher_logits_"): value
        for name, value in arrays.items()
        if name.startswith("teacher_logits_")
    }
    return OfflineExpertDataset(
        tokens=arrays["tokens"],
        mask=arrays["mask"],
        labels=arrays["labels"],
        identities=[str(value) for value in arrays["identities"].tolist()],
        teacher_logits=teacher,
        region_trees=region_trees,
    )


def _checkpoint_state(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, Mapping):
        raise ValueError("initialization checkpoint lacks a state dictionary")
    return state


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    registry = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_b_runs.json"
    )
    registry_sha = validate_stage_b_run_registry(registry)
    run = resolve_stage_b_run(registry, run_id=args.run_id)
    configuration = run["configuration"]
    if configuration["tokenizer_mode"] == "TOK_WEAVER_CLASS":
        raise ValueError(
            "TOK_WEAVER_CLASS is routed through the ordinary baseline worker"
        )
    loss_registry = load_hashed_json(
        args.campaign_root / "registry" / "retb_expert_losses.json"
    )
    step3_manifest = load_hashed_json(
        args.campaign_root
        / "registry"
        / "retb_step3_architecture_bundle.json"
    )
    profile = campaign["campaign_profile"]
    miniature = profile == "miniature_test"
    config = OfflineExpertTrainingConfig(
        seed=int(run["seed"]),
        initialization=configuration["initialization"],
        loss_id=configuration["loss_id"],
        learning_rate=float(configuration["learning_rate"]),
        particle_dropout=float(configuration["particle_dropout"]),
        maximum_epochs=2 if miniature else 40,
        microbatch_size=args.microbatch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        effective_batch_size=(
            args.microbatch_size * args.gradient_accumulation_steps
        ),
        campaign_profile="miniature_test" if miniature else "production",
    )
    config.validate()
    teacher_paths = _teacher_paths(args.teacher_checkpoint)
    output = args.output_dir or (
        args.campaign_root
        / "runs"
        / "stage_b"
        / args.run_id
        / f"seed_{run['seed']}"
    )
    resolved = {
        "dry_run": bool(args.dry_run),
        "campaign_root": str(args.campaign_root.resolve()),
        "run_id": args.run_id,
        "run_registry_sha256": registry_sha,
        "step3_bundle_sha256": step3_manifest["content_hash"],
        "configuration": configuration,
        "training_config": config.artifact(
            global_determinism_sha256=campaign["parent_artifact_hashes"][
                "global_determinism"
            ],
            expert_loss_registry_sha256=loss_registry["content_hash"],
        ),
        "train_npz": None if args.train_npz is None else str(args.train_npz.resolve()),
        "val_stop_npz": (
            None if args.val_stop_npz is None else str(args.val_stop_npz.resolve())
        ),
        "output_dir": str(output.resolve()),
        "teacher_checkpoints": {
            name: str(path.resolve()) for name, path in sorted(teacher_paths.items())
        },
    }
    print(json.dumps(resolved, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    if args.train_npz is None or args.val_stop_npz is None:
        raise ValueError("training requires --train-npz and --val-stop-npz")
    authorize_dataset_access(
        worker_role="training_worker",
        requested_resource="model_train",
    )
    authorize_dataset_access(
        worker_role="training_worker",
        requested_resource="val_stop",
    )
    train_npz_sha = _sha256_file(args.train_npz)
    val_stop_npz_sha = _sha256_file(args.val_stop_npz)
    teacher_hashes = {
        name: _sha256_file(path) for name, path in teacher_paths.items()
    }
    teacher_policy = EXPERT_LOSS_CANDIDATES[config.loss_id]["teacher"]
    required_teachers = (
        set()
        if teacher_policy is None
        else {"O_BASE", "O_FULLREL"}
        if teacher_policy == "MEAN_PROBABILITY_O_BASE_O_FULLREL"
        else {str(teacher_policy)}
    )
    if set(teacher_hashes) != required_teachers:
        raise ValueError(
            "teacher checkpoint arguments differ from the expert-loss policy"
        )
    teacher_logits_manifest = (
        load_hashed_json(args.teacher_logits_manifest)
        if args.teacher_logits_manifest is not None
        else None
    )
    if not required_teachers and teacher_logits_manifest is not None:
        raise ValueError("ELOSS_CE cannot consume a teacher-logit manifest")
    if required_teachers and not miniature and teacher_logits_manifest is None:
        raise ValueError(
            "production KD requires --teacher-logits-manifest"
        )
    if teacher_logits_manifest is not None:
        validate_teacher_logits_manifest(teacher_logits_manifest)
        expected_teacher_lineage = {
            "model_train_npz_sha256": train_npz_sha,
            "val_stop_npz_sha256": val_stop_npz_sha,
            "teacher_checkpoint_hashes": teacher_hashes,
            "teacher_fields": sorted(required_teachers),
        }
        drift = {
            name: (
                teacher_logits_manifest.get(name),
                expected,
            )
            for name, expected in expected_teacher_lineage.items()
            if teacher_logits_manifest.get(name) != expected
        }
        if drift:
            raise ValueError(f"teacher-logit manifest lineage drifted: {drift}")
        if teacher_logits_manifest.get("source") != campaign.get("source"):
            raise ValueError(
                "teacher-logit manifest belongs to another source snapshot"
            )
    relation_normalization = (
        None
        if configuration["relation_family"] is None
        else load_hashed_json(args.relation_normalization)
        if args.relation_normalization is not None
        else None
    )
    if configuration["relation_family"] is not None and relation_normalization is None:
        raise ValueError("relation expert requires --relation-normalization")
    region_normalization = (
        load_hashed_json(args.region_normalization)
        if args.region_normalization is not None
        else None
    )
    if configuration["expert_id"] == "REGION" and (
        region_normalization is None or args.region_tree_root is None
    ):
        raise ValueError(
            "REGION expert requires normalization and tree-root arguments"
        )
    train_arrays = _load_npz(args.train_npz)
    val_arrays = _load_npz(args.val_stop_npz)
    train_ids = [str(value) for value in train_arrays["identities"].tolist()]
    val_ids = [str(value) for value in val_arrays["identities"].tolist()]
    train_trees = (
        _load_trees(
            args.region_tree_root,
            split="model_train",
            identities=train_ids,
        )
        if configuration["expert_id"] == "REGION"
        else None
    )
    val_trees = (
        _load_trees(
            args.region_tree_root,
            split="val_stop",
            identities=val_ids,
        )
        if configuration["expert_id"] == "REGION"
        else None
    )
    train_dataset = _dataset(train_arrays, region_trees=train_trees)
    val_dataset = _dataset(val_arrays, region_trees=val_trees)
    train_loader = make_offline_expert_loader(
        train_dataset,
        seed=config.seed,
        training=True,
        batch_size=config.microbatch_size,
    )
    val_loader = make_offline_expert_loader(
        val_dataset,
        seed=config.seed,
        training=False,
        batch_size=config.microbatch_size,
    )
    weaver = importlib.import_module("weaver.nn.model.ParticleTransformer")
    encoder = RetbParticleEncoder(
        expert_id=configuration["expert_id"],
        topology=configuration["topology"],
        weaver_module=weaver,
        normalization_artifact=relation_normalization,
        region_normalization_artifact=region_normalization,
        measurement_embedding=configuration["measurement_embedding"],
        dual_base4_capacity_control=bool(
            configuration.get("dual_base4_capacity_control", False)
        ),
        activation_checkpointing=True,
        particle_dropout=configuration["particle_dropout"],
    )
    model = RetbExpertModel(
        particle_encoder=encoder,
        shape_id=configuration["shape_id"],
        tokenizer_mode=configuration["tokenizer_mode"],
    )
    initialization_state = (
        _checkpoint_state(args.initialization_checkpoint)
        if args.initialization_checkpoint is not None
        else None
    )
    initialization_sha = (
        _sha256_file(args.initialization_checkpoint)
        if args.initialization_checkpoint is not None
        else None
    )
    pretraining = (
        load_hashed_json(args.attachment_pretraining_record)
        if args.attachment_pretraining_record is not None
        else None
    )
    resource_profile = (
        load_hashed_json(args.resource_profile)
        if args.resource_profile is not None
        else None
    )
    lineage = {
        "campaign_spec": campaign["content_hash"],
        "step3_bundle": step3_manifest["content_hash"],
        "run_registry": registry_sha,
        "model_train_inputs": train_npz_sha,
        "val_stop_inputs": val_stop_npz_sha,
    }
    if relation_normalization is not None:
        lineage["relation_normalization"] = relation_normalization["content_hash"]
    if region_normalization is not None:
        lineage["region_normalization"] = region_normalization["content_hash"]
    if initialization_sha is not None:
        lineage["initialization_checkpoint"] = initialization_sha
    if pretraining is not None:
        lineage["attachment_pretraining"] = pretraining["content_hash"]
    if teacher_logits_manifest is not None:
        lineage["teacher_logits_manifest"] = teacher_logits_manifest[
            "content_hash"
        ]
    lineage.update(
        {f"teacher_{name}": digest for name, digest in teacher_hashes.items()}
    )
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    registration = train_offline_expert(
        model=model,
        train_loader=train_loader,
        val_stop_loader=val_loader,
        output_dir=output,
        run_record=run,
        run_registry_sha256=registry_sha,
        step3_bundle_sha256=step3_manifest["content_hash"],
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        expert_loss_registry=loss_registry,
        lineage_hashes=lineage,
        config=config,
        device=device,
        initialization_state=initialization_state,
        initialization_checkpoint_sha256=initialization_sha,
        attachment_pretraining_record=pretraining,
        resource_profile=resource_profile,
        teacher_checkpoint_hashes=teacher_hashes,
        teacher_logits_manifest=teacher_logits_manifest,
        resume=True,
    )
    print(json.dumps(registration, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
