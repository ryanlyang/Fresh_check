#!/usr/bin/env python3
"""Materialize and train one fixed, seed-matched RETB PILOT_T0 row."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.bridge_targets import (  # noqa: E402
    PilotSlotDecoderDirect,
)
from teacher_logit_reco.relation_expert_token_bridge.bridge_training import (  # noqa: E402
    BridgePilotDataset,
    PilotTrainingConfig,
    make_bridge_pilot_loader,
    train_pilot_t0,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion import (  # noqa: E402
    build_fusion_model,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.step7 import (  # noqa: E402
    materialize_stage_e_run,
    validate_stage_e_template_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.summary_tokens import (  # noqa: E402
    TokenOnlyExpertHead,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_state(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, Mapping):
        raise ValueError("bridge-pilot checkpoint lacks a state dictionary")
    return state


def _registration(
    path: Path,
    *,
    checkpoint: Path,
    seed: int,
    expert: str | None,
) -> Mapping[str, Any]:
    artifact = load_hashed_json(path)
    actual_seed = artifact.get("pipeline_seed", artifact.get("seed"))
    actual_expert = artifact.get("expert_id")
    if (
        int(actual_seed) != int(seed)
        or (expert is not None and actual_expert != expert)
        or artifact.get("checkpoint_sha256") != _sha256(checkpoint)
    ):
        raise ValueError("bridge-pilot parent registration differs")
    return artifact


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    required = {
        "identities",
        "labels",
        "unbiased_particle_states",
        "particle_mask",
        "target_tokens",
        "target_expert_logits",
        "target_hybrid_logits",
        *{f"hlt_tokens_{expert}" for expert in EXPERT_ORDER},
        *{f"t0_tokens_{expert}" for expert in EXPERT_ORDER},
    }
    if set(arrays) != required:
        raise ValueError(
            "bridge-pilot dataset fields differ: "
            f"missing={sorted(required - set(arrays))}, "
            f"extra={sorted(set(arrays) - required)}"
        )
    return arrays


def _dataset(
    arrays: Mapping[str, np.ndarray],
    *,
    split: str,
    expert: str,
    normalizer: Mapping[str, Any],
    lineage: Mapping[str, str],
) -> BridgePilotDataset:
    return BridgePilotDataset(
        identities=[str(value) for value in arrays["identities"].tolist()],
        labels=arrays["labels"],
        hlt_token_banks={
            name: arrays[f"hlt_tokens_{name}"] for name in EXPERT_ORDER
        },
        unbiased_particle_states=arrays["unbiased_particle_states"],
        particle_mask=arrays["particle_mask"],
        target_tokens=arrays["target_tokens"],
        token_mean=np.asarray(normalizer["mean"], dtype=np.float32),
        token_standard_deviation=np.asarray(
            normalizer["standard_deviation"], dtype=np.float32
        ),
        target_expert_logits=arrays["target_expert_logits"],
        target_hybrid_logits=arrays["target_hybrid_logits"],
        other_t0_banks={
            name: arrays[f"t0_tokens_{name}"]
            for name in EXPERT_ORDER
            if name != expert
        },
        target_expert_id=expert,
        split=split,
        lineage_hashes=lineage,
    )


def _load_head(
    checkpoint: Path, *, token_dimension: int
) -> TokenOnlyExpertHead:
    state = _checkpoint_state(checkpoint)
    prefix = "head."
    head_state = {
        name.removeprefix(prefix): value
        for name, value in state.items()
        if name.startswith(prefix)
    }
    if not head_state:
        raise ValueError("T0 checkpoint lacks its token-only head")
    head = TokenOnlyExpertHead(token_dimension=token_dimension)
    head.load_state_dict(head_state, strict=True)
    return head


def _slot_queries(checkpoint: Path, *, token_count: int, dimension: int) -> Any:
    state = _checkpoint_state(checkpoint)
    matches = [
        value
        for name, value in state.items()
        if name.endswith("tokenizer.slot_queries")
    ]
    if len(matches) != 1 or tuple(matches[0].shape) != (
        token_count,
        dimension,
    ):
        raise ValueError("T0 checkpoint slot queries differ")
    return matches[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--expert-id", required=True, choices=EXPERT_ORDER)
    parser.add_argument("--shape-id", required=True)
    parser.add_argument("--t0-registration", required=True, type=Path)
    parser.add_argument("--t0-checkpoint", required=True, type=Path)
    parser.add_argument("--hlt-encoder-registration", required=True, type=Path)
    parser.add_argument("--hlt-encoder-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--unbiased-particle-encoder-registration",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--unbiased-particle-encoder-checkpoint",
        required=True,
        type=Path,
    )
    parser.add_argument("--t0-fusion-registration", required=True, type=Path)
    parser.add_argument("--t0-fusion-checkpoint", required=True, type=Path)
    parser.add_argument("--target-normalizer", required=True, type=Path)
    parser.add_argument("--train-dataset", required=True, type=Path)
    parser.add_argument("--val-stop-dataset", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    templates = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_e_templates.json"
    )
    validate_stage_e_template_registry(templates)
    parents = {
        "t0": _registration(
            args.t0_registration,
            checkpoint=args.t0_checkpoint,
            seed=args.pipeline_seed,
            expert=args.expert_id,
        ),
        "hlt": _registration(
            args.hlt_encoder_registration,
            checkpoint=args.hlt_encoder_checkpoint,
            seed=args.pipeline_seed,
            expert=args.expert_id,
        ),
        "unbiased": _registration(
            args.unbiased_particle_encoder_registration,
            checkpoint=args.unbiased_particle_encoder_checkpoint,
            seed=args.pipeline_seed,
            expert="BASE4",
        ),
        "fusion": _registration(
            args.t0_fusion_registration,
            checkpoint=args.t0_fusion_checkpoint,
            seed=args.pipeline_seed,
            expert=None,
        ),
    }
    normalizer = load_hashed_json(args.target_normalizer)
    for artifact in (*parents.values(), normalizer, templates):
        if (
            artifact.get("source") is not None
            and artifact.get("source") != campaign.get("source")
        ):
            raise ValueError("bridge-pilot source lineage differs")

    train_arrays = _arrays(args.train_dataset)
    val_arrays = _arrays(args.val_stop_dataset)
    target_shape = tuple(train_arrays["target_tokens"].shape[1:])
    if (
        len(target_shape) != 2
        or tuple(val_arrays["target_tokens"].shape[1:]) != target_shape
    ):
        raise ValueError("bridge-pilot train/validation target shapes differ")
    token_count, token_dimension = map(int, target_shape)
    materialized = bind_source(
        materialize_stage_e_run(
            template_registry=templates,
            pipeline_seed=args.pipeline_seed,
            expert_id=args.expert_id,
            shape_id=args.shape_id,
            target_mode="T0_PURE",
            lambda_pred=0.0,
            bridge_dimension=None,
            unfreeze_final_two_blocks=False,
            t0_checkpoint_sha256=parents["t0"]["checkpoint_sha256"],
            hlt_encoder_checkpoint_sha256=parents["hlt"][
                "checkpoint_sha256"
            ],
            unbiased_particle_encoder_checkpoint_sha256=parents["unbiased"][
                "checkpoint_sha256"
            ],
            pilot_checkpoint_sha256=None,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    result = {
        "dry_run": bool(args.dry_run),
        "run_id": materialized["run_id"],
        "materialized_run_sha256": materialized["content_hash"],
        "token_shape": [token_count, token_dimension],
        "output_dir": str(args.output_dir.resolve()),
        "train_dataset_sha256": _sha256(args.train_dataset),
        "val_stop_dataset_sha256": _sha256(args.val_stop_dataset),
    }
    if args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    authorize_dataset_access(
        worker_role="training_worker", requested_resource="model_train"
    )
    authorize_dataset_access(
        worker_role="training_worker", requested_resource="val_stop"
    )
    lineage = {
        "T0_checkpoint": parents["t0"]["checkpoint_sha256"],
        "HLT_encoder_checkpoint": parents["hlt"]["checkpoint_sha256"],
        "unbiased_HLT_particle_encoder_checkpoint": parents["unbiased"][
            "checkpoint_sha256"
        ],
        "target_normalizer": normalizer["content_hash"],
        "T0_fusion": parents["fusion"]["checkpoint_sha256"],
    }
    train_dataset = _dataset(
        train_arrays,
        split="model_train",
        expert=args.expert_id,
        normalizer=normalizer,
        lineage=lineage,
    )
    val_dataset = _dataset(
        val_arrays,
        split="val_stop",
        expert=args.expert_id,
        normalizer=normalizer,
        lineage=lineage,
    )
    model = PilotSlotDecoderDirect(
        token_count=token_count,
        token_dimension=token_dimension,
        target_expert_id=args.expert_id,
        offline_slot_queries=_slot_queries(
            args.t0_checkpoint,
            token_count=token_count,
            dimension=token_dimension,
        ),
        dropout=0.0,
    )
    expert_head = _load_head(
        args.t0_checkpoint, token_dimension=token_dimension
    )
    bank_dimensions = {
        expert: int(train_arrays[f"t0_tokens_{expert}"].shape[-1])
        for expert in EXPERT_ORDER
    }
    fusion = build_fusion_model(
        "F_TOKEN_TRANSFORMER", bank_dimensions=bank_dimensions
    )
    fusion.load_state_dict(
        _checkpoint_state(args.t0_fusion_checkpoint), strict=True
    )
    profile = campaign["campaign_profile"]
    miniature = profile == "miniature_test"
    config = PilotTrainingConfig(
        seed=args.pipeline_seed,
        maximum_epochs=2 if miniature else 40,
        batch_size=args.batch_size,
        dropout=0.0,
        campaign_profile="miniature_test" if miniature else "production",
    )
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    registration = train_pilot_t0(
        model=model,
        train_loader=make_bridge_pilot_loader(
            train_dataset,
            batch_size=args.batch_size,
            seed=args.pipeline_seed,
            training=True,
        ),
        val_stop_loader=make_bridge_pilot_loader(
            val_dataset,
            batch_size=args.batch_size,
            seed=0,
            training=False,
        ),
        expert_head=expert_head,
        hybrid_fusion=fusion,
        target_expert_id=args.expert_id,
        output_dir=args.output_dir,
        materialized_run=materialized,
        pilot_architecture_sha256=load_hashed_json(
            args.campaign_root
            / "registry"
            / "retb_pilot_t0_architecture.json"
        )["content_hash"],
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        config=config,
        device=device,
    )
    print(json.dumps(registration, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
