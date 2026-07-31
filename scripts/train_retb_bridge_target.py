#!/usr/bin/env python3
"""Train one source-bound Stage-E bridge-target candidate and design outputs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.bridge_targets import (  # noqa: E402
    BridgeCandidatePredictor,
    BridgeOfflineTarget,
    BridgeProjection,
    PilotSlotDecoderDirect,
    bridge_target_objective,
    deterministic_within_class_negatives,
    directional_token_loss,
    fit_bridge_token_normalizer,
    heteroscedastic_huber_loss,
    normalized_huber_anchor,
    relative_slot_covariance_loss,
    temperature_two_kl,
    within_class_retrieval_loss,
)
from teacher_logit_reco.relation_expert_token_bridge.bridge_training import (  # noqa: E402
    BridgeCandidateTrainingConfig,
    BridgePilotDataset,
    _collate as collate_bridge,
    train_bridge_candidate,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (  # noqa: E402
    evaluate_classification,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_model import (  # noqa: E402
    RetbExpertModel,
    RetbParticleEncoder,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_training import (  # noqa: E402
    DeterministicExpertSampler,
    OfflineExpertDataset,
    collate_offline_expert_batch,
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
from teacher_logit_reco.relational_part.ca_tree import unpack_tree_shard  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint has no model state: {path}")
    return state


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _pilot_arrays(path: Path) -> dict[str, np.ndarray]:
    arrays = _npz(path)
    required = {
        "identities",
        "labels",
        "unbiased_particle_states",
        "particle_mask",
        "target_tokens",
        "target_expert_logits",
        "target_hybrid_logits",
        *{f"hlt_tokens_{name}" for name in EXPERT_ORDER},
        *{f"t0_tokens_{name}" for name in EXPERT_ORDER},
    }
    if set(arrays) != required:
        raise ValueError("bridge-target pilot dataset fields differ")
    return arrays


def _bridge_dataset(
    arrays: Mapping[str, np.ndarray],
    *,
    expert: str,
    split: str,
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


def _trees(
    root: Path, *, split: str, identities: Sequence[str]
) -> list[Mapping[str, Any]]:
    by_identity: dict[str, Mapping[str, Any]] = {}
    for shard in sorted(
        (root / f"{split}_exclusive_ca_v1" / "shards").glob("shard_*.npz")
    ):
        shard_ids, values = unpack_tree_shard(shard)
        for identity, value in zip(shard_ids, values, strict=True):
            if str(identity) in by_identity:
                raise ValueError("bridge-target REGION identity is duplicated")
            by_identity[str(identity)] = value
    if set(identities) - set(by_identity):
        raise ValueError("bridge-target REGION trees are incomplete")
    return [by_identity[str(identity)] for identity in identities]


def _offline_dataset(
    root: Path,
    *,
    split: str,
    identities: Sequence[str],
    expert: str,
) -> OfflineExpertDataset:
    arrays = _npz(
        root / "inputs" / "offline" / split / "offline_inputs.npz"
    )
    actual = [str(value) for value in arrays["identities"].tolist()]
    if actual != list(identities):
        raise ValueError("bridge-target raw/cached offline identities differ")
    region = (
        _trees(
            root / "inputs" / "region_tree" / "offline",
            split=split,
            identities=actual,
        )
        if expert == "REGION"
        else None
    )
    return OfflineExpertDataset(
        tokens=arrays["tokens"],
        mask=arrays["mask"],
        labels=arrays["labels"],
        identities=actual,
        region_trees=region,
    )


class _PairedDataset(torch.utils.data.Dataset):
    def __init__(
        self, bridge: BridgePilotDataset, offline: OfflineExpertDataset
    ) -> None:
        if bridge.identities != offline.identities:
            raise ValueError("bridge-target paired identities differ")
        self.bridge, self.offline = bridge, offline

    def __len__(self) -> int:
        return len(self.bridge)

    def set_epoch(self, epoch: int) -> None:
        self.bridge.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"bridge": self.bridge[index], "offline": self.offline[index]}


def _collate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = collate_bridge([row["bridge"] for row in rows])
    output["offline_batch"] = collate_offline_expert_batch(
        [row["offline"] for row in rows]
    )
    return output


def _loader(
    dataset: _PairedDataset, *, seed: int, batch_size: int, training: bool
) -> Any:
    sampler = (
        DeterministicExpertSampler(dataset, seed=seed)
        if training
        else torch.utils.data.SequentialSampler(dataset)
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=False,
        collate_fn=_collate,
    )


def _model(
    *,
    checkpoint: Path,
    relation: Mapping[str, Any],
    region: Mapping[str, Any],
) -> RetbExpertModel:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    configuration = payload["run_record"]["configuration"]
    encoder = RetbParticleEncoder(
        expert_id=configuration["expert_id"],
        topology=configuration["topology"],
        weaver_module=importlib.import_module(
            "weaver.nn.model.ParticleTransformer"
        ),
        normalization_artifact=(
            None
            if configuration.get("relation_family") is None
            else relation
        ),
        region_normalization_artifact=(
            region if configuration["expert_id"] == "REGION" else None
        ),
        measurement_embedding=configuration["measurement_embedding"],
        dual_base4_capacity_control=False,
        activation_checkpointing=True,
        particle_dropout=0.0,
    )
    model = RetbExpertModel(
        particle_encoder=encoder,
        shape_id=configuration["shape_id"],
        tokenizer_mode=configuration["tokenizer_mode"],
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def _pilot(
    *, checkpoint: Path, expert: str, count: int, dimension: int
) -> PilotSlotDecoderDirect:
    state = _state(checkpoint)
    queries = [
        value
        for name, value in state.items()
        if name == "target_queries"
    ]
    if len(queries) != 1 or tuple(queries[0].shape) != (count, dimension):
        raise ValueError("bridge pilot slot-query state differs")
    model = PilotSlotDecoderDirect(
        token_count=count,
        token_dimension=dimension,
        target_expert_id=expert,
        offline_slot_queries=queries[0],
        dropout=0.0,
    )
    # Materialize lazy evidence projections before strict loading.
    return model


def _initialize_pilot(
    model: PilotSlotDecoderDirect,
    checkpoint: Path,
    sample: Mapping[str, Any],
) -> None:
    with torch.no_grad():
        model(
            hlt_token_banks={
                name: torch.from_numpy(sample[f"hlt_tokens_{name}"][:1]).float()
                for name in EXPERT_ORDER
            },
            unbiased_particle_states=torch.from_numpy(
                sample["unbiased_particle_states"][:1]
            ).float(),
            particle_mask=torch.from_numpy(
                sample["particle_mask"][:1]
            ).bool(),
        )
    model.load_state_dict(_state(checkpoint), strict=True)


def _frozen_head(model: RetbExpertModel) -> TokenOnlyExpertHead:
    head = copy.deepcopy(model.head)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


def _normal(
    values: torch.Tensor, batch: Mapping[str, Any]
) -> torch.Tensor:
    return (
        values - batch["token_mean"]
    ) / batch["token_standard_deviation"].clamp_min(1.0e-4)


def _offline_forward(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: batch[name]
        for name in ("features", "vectors", "mask", "raw_tokens", "region_trees")
        if name in batch
    }


def _metrics(logits: np.ndarray, labels: np.ndarray, split: str) -> dict:
    return evaluate_classification(logits, labels, split=split)


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {name: _move(item, device) for name, item in value.items()}
    return value


def _publish_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        if path.exists():
            if path.is_symlink() or path.read_bytes() != temporary.read_bytes():
                raise FileExistsError(f"bridge-target array reuse differs: {path}")
        else:
            os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--expert-id", required=True, choices=EXPERT_ORDER)
    parser.add_argument("--shape-id", required=True)
    parser.add_argument("--target-mode", required=True)
    parser.add_argument("--lambda-pred", required=True, type=float)
    parser.add_argument("--bridge-dimension", type=int)
    parser.add_argument("--unfreeze-final-two-blocks", action="store_true")
    parser.add_argument("--pilot-registration", required=True, type=Path)
    parser.add_argument("--pilot-checkpoint", required=True, type=Path)
    parser.add_argument("--parent-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    templates = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_e_templates.json"
    )
    validate_stage_e_template_registry(templates)
    pilot_registration = load_hashed_json(args.pilot_registration)
    parent_bundle = load_hashed_json(args.parent_root / "parent_bundle.json")
    t0_registration = load_hashed_json(
        args.parent_root / "t0_registration.json"
    )
    hlt_registration = load_hashed_json(
        args.parent_root / "hlt_encoder_registration.json"
    )
    unbiased_registration = load_hashed_json(
        args.parent_root / "unbiased_particle_encoder_registration.json"
    )
    fusion_registration = load_hashed_json(
        args.parent_root / "t0_fusion_registration.json"
    )
    normalizer = load_hashed_json(args.parent_root / "target_normalizer.json")
    if (
        pilot_registration["checkpoint_sha256"]
        != _sha256(args.pilot_checkpoint)
        or t0_registration["checkpoint_sha256"]
        != _sha256(args.parent_root / "t0_best_model_val.pt")
        or parent_bundle["shape_id"] != args.shape_id
        or parent_bundle["expert_id"] != args.expert_id
        or int(parent_bundle["pipeline_seed"]) != args.pipeline_seed
    ):
        raise ValueError("bridge-target parent lineage differs")
    materialized = bind_source(
        materialize_stage_e_run(
            template_registry=templates,
            pipeline_seed=args.pipeline_seed,
            expert_id=args.expert_id,
            shape_id=args.shape_id,
            target_mode=args.target_mode,
            lambda_pred=args.lambda_pred,
            bridge_dimension=args.bridge_dimension,
            unfreeze_final_two_blocks=args.unfreeze_final_two_blocks,
            t0_checkpoint_sha256=t0_registration["checkpoint_sha256"],
            hlt_encoder_checkpoint_sha256=hlt_registration[
                "checkpoint_sha256"
            ],
            unbiased_particle_encoder_checkpoint_sha256=(
                unbiased_registration["checkpoint_sha256"]
            ),
            pilot_checkpoint_sha256=pilot_registration["checkpoint_sha256"],
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(args.output_dir / "materialized_run.json", materialized)
    for split, role in (
        ("model_train", "training_worker"),
        ("val_stop", "training_worker"),
        ("val_design", "design_worker"),
    ):
        authorize_dataset_access(worker_role=role, requested_resource=split)
    raw = {
        split: _pilot_arrays(
            args.parent_root / f"{split}_pilot_dataset.npz"
        )
        for split in ("model_train", "val_stop", "val_design")
    }
    lineage = {
        "T0_checkpoint": t0_registration["checkpoint_sha256"],
        "HLT_encoder_checkpoint": hlt_registration["checkpoint_sha256"],
        "unbiased_HLT_particle_encoder_checkpoint": unbiased_registration[
            "checkpoint_sha256"
        ],
        "target_normalizer": normalizer["content_hash"],
        "T0_fusion": fusion_registration["checkpoint_sha256"],
    }
    paired = {}
    for split in raw:
        bridge = _bridge_dataset(
            raw[split],
            expert=args.expert_id,
            split=split,
            normalizer=normalizer,
            lineage=lineage,
        )
        paired[split] = _PairedDataset(
            bridge,
            _offline_dataset(
                args.campaign_root,
                split=split,
                identities=bridge.identities,
                expert=args.expert_id,
            ),
        )
    count, source_dimension = map(
        int, raw["model_train"]["target_tokens"].shape[1:]
    )
    pilot = _pilot(
        checkpoint=args.pilot_checkpoint,
        expert=args.expert_id,
        count=count,
        dimension=source_dimension,
    )
    _initialize_pilot(pilot, args.pilot_checkpoint, raw["val_stop"])
    predictor = BridgeCandidatePredictor(
        pilot=pilot,
        target_mode=args.target_mode,
        bridge_dimension=args.bridge_dimension,
    )
    relation = load_hashed_json(
        args.campaign_root
        / "inputs"
        / "normalization"
        / "offline_500k"
        / "relation.json"
    )
    region = load_hashed_json(
        args.campaign_root
        / "inputs"
        / "normalization"
        / "offline_500k"
        / "region.json"
    )
    t0_model = _model(
        checkpoint=args.parent_root / "t0_best_model_val.pt",
        relation=relation,
        region=region,
    )
    frozen_head = _frozen_head(t0_model)
    fusion_dimensions = {
        name: int(raw["model_train"][f"t0_tokens_{name}"].shape[-1])
        for name in EXPERT_ORDER
    }
    frozen_fusion = build_fusion_model(
        "F_TOKEN_TRANSFORMER", bank_dimensions=fusion_dimensions
    )
    frozen_fusion.load_state_dict(
        _state(args.parent_root / "t0_fusion_best_model_val.pt"), strict=True
    )
    for frozen in (frozen_head, frozen_fusion):
        frozen.eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
    offline_target = None
    if args.target_mode != "T3_LOGIT":
        if args.target_mode == "T2_PROJECT":
            projected_dimensions = dict(fusion_dimensions)
            projected_dimensions[args.expert_id] = int(
                args.bridge_dimension
            )
            candidate_fusion = build_fusion_model(
                "F_TOKEN_TRANSFORMER",
                bank_dimensions=projected_dimensions,
            )
            candidate_state = candidate_fusion.state_dict()
            inherited = {
                name: value
                for name, value in frozen_fusion.state_dict().items()
                if name in candidate_state
                and tuple(value.shape) == tuple(candidate_state[name].shape)
            }
            candidate_fusion.load_state_dict(inherited, strict=False)
        else:
            candidate_fusion = copy.deepcopy(frozen_fusion)
        projection = (
            BridgeProjection(source_dimension, int(args.bridge_dimension))
            if args.target_mode == "T2_PROJECT"
            else None
        )
        projected_head = (
            TokenOnlyExpertHead(token_dimension=int(args.bridge_dimension))
            if args.target_mode == "T2_PROJECT"
            else None
        )
        offline_target = BridgeOfflineTarget(
            target_mode=args.target_mode,
            target_expert_id=args.expert_id,
            expert_model=t0_model,
            candidate_fusion=candidate_fusion,
            projection=projection,
            projected_expert_head=projected_head,
        )
    train_loader = _loader(
        paired["model_train"],
        seed=args.pipeline_seed,
        batch_size=args.batch_size,
        training=True,
    )
    validation_loaders = {
        split: _loader(
            paired[split], seed=0, batch_size=args.batch_size, training=False
        )
        for split in ("model_train", "val_stop", "val_design")
    }
    train_lookup = {
        identity: index
        for index, identity in enumerate(
            paired["model_train"].bridge.identities
        )
    }
    class_rings = {
        label: [
            paired["model_train"].bridge.identities[index]
            for index in np.flatnonzero(
                paired["model_train"].bridge.labels == label
            ).tolist()
        ]
        for label in range(10)
    }

    def predictor_output(batch: Mapping[str, Any], model: Any) -> dict:
        return model(
            hlt_token_banks=batch["hlt_token_banks"],
            unbiased_particle_states=batch["unbiased_particle_states"],
            particle_mask=batch["particle_mask"],
        )

    def retrieval_candidates(
        moving: torch.Tensor, batch: Mapping[str, Any]
    ) -> torch.Tensor:
        values = []
        source = paired["model_train"].bridge.target_tokens
        for row, (identity, label) in enumerate(
            zip(batch["identities"], batch["labels"].tolist(), strict=True)
        ):
            negative_ids = deterministic_within_class_negatives(
                identity=identity,
                class_label=int(label),
                class_rings=class_rings,
                pipeline_seed=args.pipeline_seed,
                certification=False,
            )
            selected = [train_lookup[value] for value in negative_ids]
            negatives = torch.from_numpy(source[selected]).to(
                moving.device, moving.dtype
            )
            values.append(torch.cat((moving[row : row + 1], negatives), dim=0))
        return torch.stack(values)

    def phase_loss_builder(
        batch: Mapping[str, Any],
        predictor_model: Any,
        target_model: Any,
        phase: str,
    ) -> torch.Tensor:
        if args.target_mode == "T3_LOGIT":
            output = predictor_output(batch, predictor_model)
            return (
                temperature_two_kl(
                    output["logits"], batch["target_hybrid_logits"]
                )
                + 0.10
                * torch.nn.functional.cross_entropy(
                    output["logits"], batch["labels"]
                )
            )
        if phase == "predictor":
            with torch.no_grad():
                moving = target_model(
                    offline_batch=_offline_forward(batch["offline_batch"]),
                    other_t0_banks=batch["other_t0_banks"],
                )
            predicted = predictor_output(batch, predictor_model)
        else:
            with torch.no_grad():
                predicted = predictor_output(batch, predictor_model)
            moving = target_model(
                offline_batch=_offline_forward(batch["offline_batch"]),
                other_t0_banks=batch["other_t0_banks"],
            )
        moving_tokens = moving["moving_tokens"]
        if args.target_mode == "T2_PROJECT":
            # T2 owns a new bridge coordinate system.  A batch-local
            # standardization would make that coordinate change with batch
            # composition and leave no deterministic inverse at inference.
            # Train the predictor in the raw learned bridge coordinates; fit
            # the immutable train-population bridge normalizer only after the
            # alternating target/predictor optimization has completed.
            moving_normal = moving_tokens
        else:
            moving_normal = _normal(moving_tokens, batch)
        predicted_tokens = predicted["predicted_tokens"]
        if phase == "predictor":
            token_loss = (
                heteroscedastic_huber_loss(
                    predicted_tokens,
                    moving_normal.detach(),
                    predicted["log_variance"],
                )
                + 0.25
                * directional_token_loss(
                    predicted_tokens, moving_normal.detach()
                )
            )
        else:
            token_loss = (
                torch.nn.functional.huber_loss(
                    moving_normal.float(),
                    predicted_tokens.detach().float(),
                    delta=0.5,
                )
                + 0.25
                * directional_token_loss(
                    moving_normal, predicted_tokens.detach()
                )
            )
        decoded = (
            moving["decoded_tokens"]
            if moving["decoded_tokens"] is not None
            else moving_tokens
        )
        frozen_expert_logits = frozen_head(decoded)
        frozen_banks = dict(batch["other_t0_banks"])
        frozen_banks[args.expert_id] = decoded
        frozen_hybrid_logits = frozen_fusion(token_banks=frozen_banks)
        kwargs: dict[str, Any] = {}
        if args.target_mode == "T1_ANCHORED_BRIDGE":
            pure = batch["target_tokens"]
            kwargs.update(
                anchor_loss=normalized_huber_anchor(moving_normal, pure),
                retrieval_loss=within_class_retrieval_loss(
                    predicted_tokens,
                    retrieval_candidates(moving_normal, batch),
                ),
                covariance_loss=relative_slot_covariance_loss(
                    moving_normal, pure
                ),
            )
        if args.target_mode == "T2_PROJECT":
            decoded_normal = _normal(decoded, batch)
            kwargs.update(
                t0_project_loss=normalized_huber_anchor(
                    decoded_normal, batch["target_tokens"]
                ),
                decoded_t0_logit_loss=(
                    temperature_two_kl(
                        frozen_expert_logits,
                        batch["target_expert_logits"],
                    )
                    + temperature_two_kl(
                        frozen_hybrid_logits,
                        batch["target_hybrid_logits"],
                    )
                ),
            )
        total, _ = bridge_target_objective(
            target_mode=args.target_mode,
            offline_expert_loss=torch.nn.functional.cross_entropy(
                moving["expert_logits"], batch["labels"]
            ),
            token_prediction_loss=token_loss,
            offline_fusion_loss=torch.nn.functional.cross_entropy(
                moving["fusion_logits"], batch["labels"]
            ),
            t0_logit_loss=(
                temperature_two_kl(
                    frozen_expert_logits, batch["target_expert_logits"]
                )
                + temperature_two_kl(
                    frozen_hybrid_logits, batch["target_hybrid_logits"]
                )
            ),
            lambda_pred=args.lambda_pred,
            **kwargs,
        )
        return total

    @torch.no_grad()
    def infer(
        model: Any, target_model: Any, device: torch.device, split: str
    ) -> dict[str, np.ndarray]:
        model.eval()
        if target_model is not None:
            target_model.eval()
        for frozen in (frozen_head, frozen_fusion):
            frozen.to(device).eval()
        output: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "labels",
                "moving_tokens",
                "t0_tokens",
                "predicted_hlt_tokens",
                "moving_expert_logits",
                "moving_fusion_logits",
                "predicted_expert_logits",
                "predicted_fusion_logits",
                "t0_expert_logits",
                "t0_fusion_logits",
                "decoded_tokens",
            )
        }
        identities: list[str] = []
        for raw_batch in validation_loaders[split]:
            batch = _move(raw_batch, device)
            predicted = predictor_output(batch, model)
            if target_model is None:
                moving_tokens = batch["target_tokens_original"]
                decoded = moving_tokens
                moving_expert = batch["target_expert_logits"]
                moving_fusion = predicted["logits"]
                predicted_tokens = moving_tokens
                predicted_expert = predicted["logits"]
                predicted_fusion = predicted["logits"]
            else:
                moving = target_model(
                    offline_batch=_offline_forward(batch["offline_batch"]),
                    other_t0_banks=batch["other_t0_banks"],
                )
                moving_tokens = moving["moving_tokens"]
                decoded = (
                    moving["decoded_tokens"]
                    if moving["decoded_tokens"] is not None
                    else moving_tokens
                )
                moving_expert = moving["expert_logits"]
                moving_fusion = moving["fusion_logits"]
                if args.target_mode == "T2_PROJECT":
                    predicted_tokens = predicted["predicted_tokens"]
                    predicted_expert = target_model.projected_expert_head(
                        predicted_tokens
                    )
                else:
                    predicted_tokens = (
                        predicted["predicted_tokens"]
                        * batch["token_standard_deviation"].clamp_min(1.0e-4)
                        + batch["token_mean"]
                    )
                    predicted_expert = target_model.expert_model.head(
                        predicted_tokens
                    )
                predicted_banks = dict(batch["other_t0_banks"])
                predicted_banks[args.expert_id] = predicted_tokens
                predicted_fusion = target_model.candidate_fusion(
                    token_banks=predicted_banks
                )
            t0_banks = dict(batch["other_t0_banks"])
            t0_banks[args.expert_id] = batch["target_tokens_original"]
            values = {
                "labels": batch["labels"],
                "moving_tokens": moving_tokens,
                "t0_tokens": batch["target_tokens_original"],
                "predicted_hlt_tokens": predicted_tokens,
                "moving_expert_logits": moving_expert,
                "moving_fusion_logits": moving_fusion,
                "predicted_expert_logits": predicted_expert,
                "predicted_fusion_logits": predicted_fusion,
                "t0_expert_logits": batch["target_expert_logits"],
                "t0_fusion_logits": frozen_fusion(token_banks=t0_banks),
                "decoded_tokens": decoded,
            }
            for name, value in values.items():
                output[name].append(value.float().cpu().numpy())
            identities.extend(batch["identities"])
        return {
            "identities": np.asarray(identities, dtype="U"),
            **{
                name: np.concatenate(values)
                for name, values in output.items()
            },
        }

    def val_stop_evaluator(
        model: Any, target_model: Any, device: torch.device
    ) -> dict[str, float]:
        arrays = infer(model, target_model, device, "val_stop")
        return _metrics(
            arrays["moving_fusion_logits"], arrays["labels"], "val_stop"
        )

    miniature = campaign["campaign_profile"] == "miniature_test"
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    for frozen in (frozen_head, frozen_fusion):
        frozen.to(device).eval()
    registration = train_bridge_candidate(
        predictor=predictor,
        offline_target=offline_target,
        train_loader=train_loader,
        val_stop_evaluator=val_stop_evaluator,
        phase_loss_builder=phase_loss_builder,
        output_dir=args.output_dir,
        materialized_run=materialized,
        pilot_checkpoint=pilot_registration,
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        config=BridgeCandidateTrainingConfig(
            seed=args.pipeline_seed,
            target_mode=args.target_mode,
            maximum_epochs=2 if miniature else 40,
            effective_batch_size=args.batch_size,
            campaign_profile="miniature_test" if miniature else "production",
        ),
        device=device,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    coordinate_arrays: dict[str, dict[str, np.ndarray]] = {}
    coordinate_array_hashes: dict[str, str] = {}
    for split in ("model_train", "val_stop", "val_design"):
        arrays = infer(predictor, offline_target, device, split)
        coordinate_arrays[split] = arrays
        coordinate_array_hashes[split] = _publish_npz(
            args.output_dir / f"{split}_coordinate_arrays.npz", arrays
        )
        if split == "model_train":
            continue
        metrics = bind_source(
            with_content_hash(
                {
                    "contract": "retb_bridge_candidate_metrics_v1",
                    "schema_version": 1,
                    "split": split,
                    "target_mode": args.target_mode,
                    "expert_id": args.expert_id,
                    "shape_id": args.shape_id,
                    "pipeline_seed": args.pipeline_seed,
                    "checkpoint_sha256": registration["checkpoint_sha256"],
                    "metrics": _metrics(
                        arrays["moving_fusion_logits"],
                        arrays["labels"],
                        split,
                    ),
                    "performance_based_termination": False,
                }
            ),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        write_immutable_json(args.output_dir / f"{split}_metrics.json", metrics)
        if split == "val_design":
            cert_arrays = arrays
    if args.target_mode == "T2_PROJECT":
        train_arrays = coordinate_arrays["model_train"]
        bridge_normalizer = bind_source(
            fit_bridge_token_normalizer(
                train_arrays["moving_tokens"],
                expert_id=args.expert_id,
                shape_id=args.shape_id,
                target_checkpoint_sha256=registration["checkpoint_sha256"],
                token_cache_sha256=coordinate_array_hashes["model_train"],
                identity_manifest_sha256=parent_bundle[
                    "dataset_evidence"
                ]["model_train"]["identity_manifest_sha256"],
            ),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        write_immutable_json(
            args.output_dir / "bridge_normalizer.json", bridge_normalizer
        )
    cert_payload = {
        name: value
        for name, value in cert_arrays.items()
        if name != "decoded_tokens" or args.target_mode == "T2_PROJECT"
    }
    _publish_npz(args.output_dir / "certification_arrays.npz", cert_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
