#!/usr/bin/env python3
"""Assemble, authenticate, and train one RETB Stage-J graph row."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.part_inputs import (  # noqa: E402
    build_particle_transformer_inputs_from_tokens,
)
from scripts.execute_retb_predictor_campaign import (  # noqa: E402
    PredictorCampaignPlanner,
)
from scripts.train_retb_joint_bridge import main as train_main  # noqa: E402
from scripts.train_retb_native_hlt_expert import _trees  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    canonical_sha256,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_model import (  # noqa: E402
    RetbExpertModel,
    RetbParticleEncoder,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion import (  # noqa: E402
    build_fusion_model,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (  # noqa: E402
    load_hlt_v3_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.joint_bridge import (  # noqa: E402
    CoupledExpertDecoder,
    JointBridgeGraph,
)
from teacher_logit_reco.relation_expert_token_bridge.joint_bridge_training import (  # noqa: E402
    JointBridgeDataset,
    publish_joint_dataset_cache,
    publish_joint_graph_template,
)
from teacher_logit_reco.relation_expert_token_bridge.predictors import (  # noqa: E402
    RetbTokenPredictor,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.step11 import (  # noqa: E402
    SEMANTIC_LABELS,
    materialize_stage_j_run,
    validate_materialized_stage_j_run,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_execution import (  # noqa: E402
    build_scale_joint_completion,
    build_scale_stage_j_run,
)
from teacher_logit_reco.relation_expert_token_bridge.target_cache import (  # noqa: E402
    load_frozen_token_head,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _selected_paths(
    root: Path,
    lock: Mapping[str, Any],
    seed: int,
    scale_index: Mapping[str, Any] | None = None,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    if scale_index is not None:
        return {
            expert: (
                Path(scale_index["predictors"][expert]["run_path"]),
                load_hashed_json(
                    Path(scale_index["predictors"][expert]["run_path"])
                ),
            )
            for expert in EXPERT_ORDER
        }
    configuration = json.loads(
        (
            root
            / "selection"
            / "predictor_bundle"
            / "inputs"
            / "selector_configuration.json"
        ).read_text("utf-8")
    )
    output = {}
    for expert in EXPERT_ORDER:
        candidate = lock["selected_candidate_descriptors"][expert][
            "candidate_id"
        ]
        value = configuration["materialized_run_paths"][candidate]
        path = Path(value.get(str(seed), value.get(seed)))
        output[expert] = (path, load_hashed_json(path))
    return output


def _normalizers(
    root: Path, *, coordinate_index: int, shape: str, seed: int
    , scale_index: Mapping[str, Any] | None = None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    location = (
        Path(scale_index["target_normalizer_root"])
        if scale_index is not None
        else (
            root
            / "inputs"
            / "target_normalizers"
            / f"coordinate_{coordinate_index:03d}"
            / shape
            / f"seed_{seed}"
        )
    )
    normalizer_set = load_hashed_json(
        location / "target_normalizer_set.json"
    )
    means, deviations = {}, {}
    for expert in EXPERT_ORDER:
        row = load_hashed_json(
            location / f"target_normalizer_{expert}.json"
        )
        if (
            normalizer_set["normalizer_hashes"][expert]
            != row["content_hash"]
        ):
            raise ValueError("joint target-normalizer set differs")
        means[expert] = np.asarray(row["mean"], dtype=np.float32)
        deviations[expert] = np.maximum(
            np.asarray(row["standard_deviation"], dtype=np.float32),
            1.0e-4,
        )
    return means, deviations, normalizer_set


def _predictors(
    root: Path,
    *,
    lock: Mapping[str, Any],
    selected: Mapping[str, tuple[Path, Mapping[str, Any]]],
    seed: int,
    scale_index: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    models, objectives, gradnorm = {}, {}, {}
    for expert in EXPERT_ORDER:
        _, run = selected[expert]
        output_root = (
            Path(scale_index["predictors"][expert]["output_root"])
            if scale_index is not None
            else root / "runs" / "predictors" / run["run_id"]
        )
        training_role = (
            "scale_train" if scale_index is not None else "model_train"
        )
        prepared = _npz(
            output_root / "prepared" / f"{training_role}.npz"
        )
        model = RetbTokenPredictor(
            architecture=run["architecture"],
            context=run["context"],
            target_expert_id=expert,
            token_count=int(run["token_count"]),
            token_dimension=int(run["token_dimension"]),
            offline_slot_queries=torch.from_numpy(
                np.asarray(
                    prepared["offline_slot_queries"], dtype=np.float32
                )
            ),
            uncertainty_head=run["uncertainty_head"],
            dropout=float(run["dropout"]),
            residual_hidden_width=run.get("residual_hidden_width"),
            zero_evidence_control=False,
        )
        checkpoint_path = (
            output_root / "training" / "best_model_val.pt"
        )
        expected = (
            scale_index["predictors"][expert]["checkpoint_sha256"]
            if scale_index is not None
            else lock["seed_specific_artifacts"][str(seed)][expert][
                "predictor_checkpoint"
            ]
        )
        if _sha256(checkpoint_path) != expected:
            raise ValueError("joint selected predictor checkpoint differs")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        models[expert] = model
        objectives[expert] = str(run["objective_id"])
        state = checkpoint.get("gradnorm_state")
        gradnorm[expert] = (
            None if state is None else dict(state["current"])
        )
    return models, objectives, gradnorm


def _offline_components(
    planner: PredictorCampaignPlanner,
    *,
    selected: Mapping[str, tuple[Path, Mapping[str, Any]]],
    shape: str,
    seed: int,
    lock: Mapping[str, Any],
    scale_index: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any], Path]:
    if scale_index is not None:
        fusion_path = Path(scale_index["offline_fusion_checkpoint"])
        payload = torch.load(
            fusion_path, map_location="cpu", weights_only=False
        )
        fusion = build_fusion_model(
            "F_TOKEN_TRANSFORMER",
            bank_dimensions={
                expert: int(lock["allocation"][expert][1])
                for expert in EXPERT_ORDER
            },
        )
        fusion.load_state_dict(payload["model_state_dict"], strict=True)
        artifacts = {
            expert: {
                "checkpoint": Path(
                    scale_index["target_checkpoints"][expert]["path"]
                ),
                "descriptor": {
                    "checkpoint_sha256": scale_index[
                        "target_checkpoints"
                    ][expert]["sha256"],
                },
            }
            for expert in EXPERT_ORDER
        }
        heads = {
            expert: load_frozen_token_head(
                checkpoint_path=artifacts[expert]["checkpoint"],
                expected_checkpoint_sha256=artifacts[expert][
                    "descriptor"
                ]["checkpoint_sha256"],
                target_mode=lock["target_modes"][expert],
                token_dimension=int(lock["allocation"][expert][1]),
            )
            for expert in EXPERT_ORDER
        }
        for module in (fusion, *heads.values()):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        return fusion, heads, artifacts, fusion_path
    artifacts = {
        expert: planner._artifacts(
            shape=shape,
            expert=expert,
            seed=seed,
            target_mode=lock["target_modes"][expert],
            evidence_mode=selected[expert][1]["hlt_evidence_mode"],
        )
        for expert in EXPERT_ORDER
    }
    fusion_path = artifacts["BASE4"]["fusion"]
    payload = torch.load(
        fusion_path, map_location="cpu", weights_only=False
    )
    fusion = build_fusion_model(
        "F_TOKEN_TRANSFORMER",
        bank_dimensions={
            expert: int(lock["allocation"][expert][1])
            for expert in EXPERT_ORDER
        },
    )
    fusion.load_state_dict(payload["model_state_dict"], strict=True)
    heads = {
        expert: load_frozen_token_head(
            checkpoint_path=artifacts[expert]["checkpoint"],
            expected_checkpoint_sha256=_sha256(
                artifacts[expert]["checkpoint"]
            ),
            target_mode=lock["target_modes"][expert],
            token_dimension=int(lock["allocation"][expert][1]),
        )
        for expert in EXPERT_ORDER
    }
    for module in (fusion, *heads.values()):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return fusion, heads, artifacts, fusion_path


def _selected_hlt_experts(
    root: Path,
    *,
    shape: str,
    seed: int,
    confirmation: Mapping[str, Any],
    allocation: Mapping[str, Sequence[int]],
    scale_index: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    weaver = importlib.import_module("weaver.nn.model.ParticleTransformer")
    population = "hlt_shared_scale" if scale_index is not None else "hlt_shared_500k"
    relation = load_hashed_json(
        root / "inputs" / "normalization" / population / "relation.json"
    )
    region = load_hashed_json(
        root / "inputs" / "normalization" / population / "region.json"
    )
    models, registrations = {}, {}
    for expert in EXPERT_ORDER:
        rows = [
            row
            for row in confirmation["rows"]
            if row["component"] == "HLT_EXPERT"
            and int(row["seed"]) == seed
            and row["configuration"]["shape_id"] == shape
            and row["configuration"]["expert_id"] == expert
            and row["configuration"]["mode"] == "HE_SCRATCH_CE"
            and row["configuration"]["realization_policy"] == "R_MULTI"
            and not row["configuration"]["measurement_embedding"]
        ]
        if len(rows) != 1:
            raise ValueError("joint selected HLT expert differs")
        row = rows[0]
        output = (
            Path(scale_index["native_hlt_experts"][expert]["output_root"])
            if scale_index is not None
            else (
                root
                / "runs"
                / "stage_d"
                / "hlt_experts"
                / row["run_id"]
                / f"seed_{seed}"
            )
        )
        registration = load_hashed_json(
            output / "checkpoint_registration.json"
        )
        offline = load_hashed_json(
            Path(scale_index["offline_experts"][expert]["registration"])
            if scale_index is not None
            else (
                root
                / "selection"
                / "offline_experts"
                / shape
                / expert
                / f"seed_{seed}"
                / "checkpoint_registration.json"
            )
        )
        encoder = RetbParticleEncoder(
            expert_id=expert,
            topology=offline["topology"],
            weaver_module=weaver,
            normalization_artifact=None if expert == "BASE4" else relation,
            region_normalization_artifact=(
                region if expert == "REGION" else None
            ),
            measurement_embedding=False,
            activation_checkpointing=True,
            particle_dropout=0.0,
        )
        model = RetbExpertModel(
            particle_encoder=encoder,
            shape_id=offline["shape_id"],
            tokenizer_mode=offline["tokenizer_mode"],
        )
        checkpoint_path = output / "best_model_val.pt"
        if _sha256(checkpoint_path) != registration["checkpoint_sha256"]:
            raise ValueError("joint selected HLT expert bytes differ")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if [model.token_count, model.token_dimension] != list(
            allocation[expert]
        ):
            raise ValueError("joint HLT expert allocation differs")
        models[expert] = model
        registrations[expert] = registration["content_hash"]
    return models, canonical_sha256(registrations)


def _split_arrays(
    root: Path,
    *,
    selected: Mapping[str, tuple[Path, Mapping[str, Any]]],
    split: str,
    scale_index: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        expert: _npz(
            (
                Path(scale_index["predictors"][expert]["output_root"])
                if scale_index is not None
                else root / "runs" / "predictors" / run["run_id"]
            )
            / "prepared"
            / f"{split}.npz"
        )
        for expert, (_, run) in selected.items()
    }


def _raw_view(
    root: Path,
    *,
    split: str,
    identities: Sequence[str],
) -> tuple[dict[str, Any], str]:
    replicas = (
        (0, 1, 2, 3)
        if split in {"model_train", "scale_train"}
        else (0,)
    )
    arrays, hashes = {}, {}
    for replica in replicas:
        cache_dir = (
            root
            / "inputs"
            / "hlt_v3"
            / split
            / f"replica_{replica}"
            / (
                "R_MULTI"
                if split in {"model_train", "scale_train"}
                else "R_FIXED"
            )
            / "D_NOMINAL"
        )
        values, metadata = load_hlt_v3_cache(cache_dir)
        if tuple(str(value) for value in values["identities"]) != tuple(
            str(value) for value in identities
        ):
            raise ValueError("joint raw HLT identity order differs")
        inputs = build_particle_transformer_inputs_from_tokens(
            values["tokens"],
            values["mask"],
            labels=np.zeros(len(identities), dtype=np.int64),
            source_view="hlt",
        )
        arrays[replica] = {
            "features": inputs.pf_features,
            "vectors": inputs.pf_vectors,
            "mask": inputs.pf_mask,
            "raw_tokens": values["tokens"],
        }
        hashes[replica] = metadata["content_hash"]
    trees = _trees(
        root / "inputs" / "region_tree" / "hlt",
        logical_role=split,
        replicas=replicas,
        identities=tuple(str(value) for value in identities),
        realization_policy=(
            "R_MULTI"
            if split in {"model_train", "scale_train"}
            else "R_FIXED"
        ),
    )
    return (
        {
            **{
                name: {
                    replica: arrays[replica][name]
                    for replica in replicas
                }
                for name in ("features", "vectors", "mask", "raw_tokens")
            },
            "region_trees_by_expert": {
                expert: trees for expert in EXPERT_ORDER
            },
        },
        canonical_sha256(hashes),
    )


def _dataset(
    root: Path,
    *,
    selected: Mapping[str, tuple[Path, Mapping[str, Any]]],
    split: str,
    means: Mapping[str, np.ndarray],
    deviations: Mapping[str, np.ndarray],
    identity_sha: str,
    target_cache_sha: str,
    normalizer_set_sha: str,
    live: bool,
    scale_index: Mapping[str, Any] | None = None,
) -> tuple[JointBridgeDataset, str]:
    by_expert = _split_arrays(
        root,
        selected=selected,
        split=split,
        scale_index=scale_index,
    )
    base = by_expert["BASE4"]
    identities = [str(value) for value in base["identities"].tolist()]
    labels = np.asarray(base["labels"], dtype=np.int64)
    if any(
        not np.array_equal(values["identities"], base["identities"])
        or not np.array_equal(values["labels"], labels)
        for values in by_expert.values()
    ):
        raise ValueError("joint selected predictor populations differ")
    replicas = (
        (0, 1, 2, 3)
        if split in {"model_train", "scale_train"}
        else (0,)
    )
    evidence_hash = canonical_sha256(
        {
            expert: _sha256(
                (
                    Path(
                        scale_index["predictors"][expert]["output_root"]
                    )
                    if scale_index is not None
                    else root
                    / "runs"
                    / "predictors"
                    / selected[expert][1]["run_id"]
                )
                / "prepared"
                / f"{split}.npz"
            )
            for expert in EXPERT_ORDER
        }
    )
    view_hashes = {
        replica: [
            canonical_sha256(
                {
                    "identity": identity,
                    "replica": replica,
                    "evidence": evidence_hash,
                }
            )
            for identity in identities
        ]
        for replica in replicas
    }
    replica_ids = np.asarray(
        [
            __import__(
                "teacher_logit_reco.relation_expert_token_bridge.replicas",
                fromlist=["replica_for"],
            ).replica_for(
                policy="R_MULTI",
                logical_role=split,
                epoch=0,
                canonical_identity=identity,
            )
            for identity in identities
        ],
        dtype=np.int64,
    )
    raw_view, raw_hash = (
        _raw_view(root, split=split, identities=identities)
        if live
        else (None, evidence_hash)
    )
    parent_hashes = {
        "identity_manifest": identity_sha,
        "HLT_view_cache": raw_hash if live else evidence_hash,
        "offline_target_cache": target_cache_sha,
        "target_normalizer_set": normalizer_set_sha,
    }
    dataset = JointBridgeDataset(
        identities=identities,
        labels=labels,
        replica_ids=replica_ids,
        degraded_view_hashes=view_hashes,
        split=split,
        hlt_token_banks={
            expert: by_expert[expert][f"hlt_tokens_{expert}"]
            for expert in EXPERT_ORDER
        },
        unbiased_particle_states=base["unbiased_particle_states"],
        particle_mask=base["particle_mask"],
        relation_particle_states={
            expert: by_expert[expert][
                f"relation_particle_states_{expert}"
            ]
            for expert in ("PT", "TRACK", "REGION")
        },
        relation_particle_masks={
            expert: by_expert[expert][
                f"relation_particle_mask_{expert}"
            ]
            for expert in ("PT", "TRACK", "REGION")
        },
        target_normalized_banks={
            expert: (
                by_expert[expert]["target_tokens"] - means[expert]
            )
            / deviations[expert]
            for expert in EXPERT_ORDER
        },
        oracle_banks={
            expert: by_expert[expert]["target_tokens"]
            for expert in EXPERT_ORDER
        },
        target_expert_logits={
            expert: by_expert[expert]["target_expert_logits"]
            for expert in EXPERT_ORDER
        },
        oracle_fusion_logits=base["target_hybrid_logits"],
        shared_raw_view=raw_view,
        lineage_hashes=parent_hashes,
    )
    return dataset, raw_hash if live else evidence_hash


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--predictor-bundle-lock", type=Path)
    parser.add_argument("--final-particle-blocks", type=int)
    parser.add_argument("--j4-selection", type=Path)
    parser.add_argument("--selected-j4-output", type=Path)
    parser.add_argument(
        "--scale-component-index",
        type=Path,
        help=(
            "Source-bound Stage-M component index. When present, the locked "
            "J4 topology is populated only with scale-trained components."
        ),
    )
    parser.add_argument("--scale-joint-completion-output", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    scale_index = (
        None
        if args.scale_component_index is None
        else load_hashed_json(args.scale_component_index)
    )
    if (scale_index is None) != (
        args.scale_joint_completion_output is None
    ):
        raise ValueError(
            "scale joint completion output must accompany scale components"
        )
    if scale_index is not None and (
        scale_index.get("contract")
        != "retb_scale_component_index_v1"
        or scale_index.get("source") != campaign.get("source")
        or int(scale_index.get("pipeline_seed", -1))
        != args.pipeline_seed
    ):
        raise ValueError("scale component index lineage differs")
    graph_contract = load_hashed_json(
        root / "job_ledgers" / "production_graph.json"
    )
    lock_path = args.predictor_bundle_lock or (
        root / "selection" / "predictor_bundle" / "predictor_bundle_lock.json"
    )
    lock = load_hashed_json(lock_path)
    step11 = load_hashed_json(
        root / "registry" / "retb_step11_joint_bridge_bundle.json"
    )
    selected = _selected_paths(
        root, lock, args.pipeline_seed, scale_index=scale_index
    )
    coordinate_name, shape = str(lock["coordinate_id"]).split(":", 1)
    coordinate_index = int(coordinate_name.rsplit("_", 1)[1])
    planner = PredictorCampaignPlanner(
        root=root, campaign=campaign, graph=graph_contract
    )
    means, deviations, normalizer_set = _normalizers(
        root,
        coordinate_index=coordinate_index,
        shape=shape,
        seed=args.pipeline_seed,
        scale_index=scale_index,
    )
    predictors, objectives, gradnorm = _predictors(
        root,
        lock=lock,
        selected=selected,
        seed=args.pipeline_seed,
        scale_index=scale_index,
    )
    fusion, heads, target_artifacts, fusion_path = _offline_components(
        planner,
        selected=selected,
        shape=shape,
        seed=args.pipeline_seed,
        lock=lock,
        scale_index=scale_index,
    )
    confirmation = load_hashed_json(
        root
        / "selection"
        / "predictor_phases"
        / "stage_d_evidence_confirmations.json"
    )
    live = args.variant in {"J4_BRIDGE_FINETUNE", "J5_END_TO_END"}
    hlt_experts = hlt_sha = None
    if live:
        hlt_experts, hlt_sha = _selected_hlt_experts(
            root,
            shape=shape,
            seed=args.pipeline_seed,
            confirmation=confirmation,
            allocation=lock["allocation"],
            scale_index=scale_index,
        )
    coupled = None
    if args.variant == "J2_COUPLED_DECODER":
        coupled = CoupledExpertDecoder(
            allocation=lock["allocation"],
            offline_slot_queries={
                expert: predictors[expert].target_queries.detach().cpu()
                for expert in EXPERT_ORDER
            },
            uncertainty_widths={
                expert: int(
                    predictors[expert].log_variance_head[-1].out_features
                )
                for expert in EXPERT_ORDER
            },
            dropout=0.1,
        )
    deployable = None
    j4_selection_sha = j4_initialization_sha = None
    if args.variant == "J5_END_TO_END":
        if args.j4_selection is None or args.selected_j4_output is None:
            raise ValueError("J5 requires the locked J4 initialization")
        j4_selection = load_hashed_json(args.j4_selection)
        j4_registration = load_hashed_json(
            args.selected_j4_output / "registration.json"
        )
        j4_template_payload = torch.load(
            args.selected_j4_output
            / "assets"
            / "graph"
            / "joint_graph_template.pt",
            map_location="cpu",
            weights_only=False,
        )
        graph = j4_template_payload["graph"]
        j4_checkpoint = torch.load(
            args.selected_j4_output / "best_model_val.pt",
            map_location="cpu",
            weights_only=False,
        )
        graph.load_state_dict(j4_checkpoint["model_state_dict"], strict=True)
        graph = copy.deepcopy(graph)
        graph.variant = "J5_END_TO_END"
        if scale_index is not None:
            graph.predictors = torch.nn.ModuleDict(predictors)
            graph.frozen_offline_fusion = fusion
            graph.frozen_expert_heads = torch.nn.ModuleDict(heads)
            graph.hlt_experts = torch.nn.ModuleDict(hlt_experts)
            for expert_index, expert in enumerate(EXPERT_ORDER):
                setattr(
                    graph,
                    f"token_mean_{expert_index}",
                    torch.from_numpy(means[expert]).float(),
                )
                setattr(
                    graph,
                    f"token_std_{expert_index}",
                    torch.from_numpy(deviations[expert]).float(),
                )
        else:
            predictors = dict(graph.predictors.items())
            fusion = graph.frozen_offline_fusion
            heads = dict(graph.frozen_expert_heads.items())
            hlt_experts = dict(graph.hlt_experts.items())
        graph.deployable_fusion = copy.deepcopy(fusion)
        deployable = graph.deployable_fusion
        coupled = graph.coupled_decoder
        j4_selection_sha = j4_selection["content_hash"]
        j4_initialization_sha = j4_registration["content_hash"]
    else:
        graph = JointBridgeGraph(
            variant=args.variant,
            predictors=predictors,
            frozen_offline_fusion=fusion,
            frozen_expert_heads=heads,
            token_means=means,
            token_standard_deviations=deviations,
            hlt_experts=hlt_experts,
            deployable_fusion=deployable,
            coupled_decoder=coupled,
        )

    target_root = (
        Path(scale_index["target_cache_root"])
        if scale_index is not None
        else (
            root
            / "inputs"
            / "target_caches"
            / f"coordinate_{coordinate_index:03d}"
            / shape
            / f"seed_{args.pipeline_seed}"
        )
    )
    training_role = (
        "scale_train" if scale_index is not None else "model_train"
    )
    split_manifests = {
        split: load_hashed_json(
            target_root / split / "target_cache_manifest.json"
        )
        for split in (training_role, "val_stop", "val_design")
    }
    identity_manifests = {
        split: load_hashed_json(
            target_root / split / "identity_manifest.json"
        )
        for split in (training_role, "val_stop", "val_design")
    }
    predictor_sha = canonical_sha256(
        lock["seed_specific_artifacts"][str(args.pipeline_seed)]
    )
    heads_sha = canonical_sha256(
        {
            expert: target_artifacts[expert]["descriptor"][
                "checkpoint_sha256"
            ]
            for expert in EXPERT_ORDER
        }
    )
    parent_hashes = {
        f"{training_role}_identity_manifest": identity_manifests[
            training_role
        ]["content_hash"],
        "val_stop_identity_manifest": identity_manifests["val_stop"][
            "content_hash"
        ],
        "val_design_identity_manifest": identity_manifests["val_design"][
            "content_hash"
        ],
        "val_design_label_manifest": lock["selection_data_hashes"][
            "label_manifests"
        ][str(args.pipeline_seed)],
        f"{training_role}_R_MULTI_view_cache": canonical_sha256(
            {"split": training_role, "seed": args.pipeline_seed, "lock": lock["content_hash"]}
        ),
        "val_stop_R_MULTI_view_cache": canonical_sha256(
            {"split": "val_stop", "seed": args.pipeline_seed, "lock": lock["content_hash"]}
        ),
        "val_design_fixed_view_cache": canonical_sha256(
            {"split": "val_design", "seed": args.pipeline_seed, "lock": lock["content_hash"]}
        ),
        "offline_target_cache": canonical_sha256(
            {
                split: manifest["content_hash"]
                for split, manifest in split_manifests.items()
            }
        ),
        "target_normalizer_set": normalizer_set["content_hash"],
        "frozen_offline_fusion": _sha256(fusion_path),
        "frozen_offline_expert_heads": heads_sha,
        "selected_predictor_seed_artifacts": predictor_sha,
    }
    if live:
        parent_hashes["selected_HLT_expert_seed_artifacts"] = hlt_sha
    if args.variant == "J5_END_TO_END":
        parent_hashes["j4_block_selection"] = j4_selection_sha
        parent_hashes["selected_J4_bridge_initialization"] = (
            j4_initialization_sha
        )
    run_id = f"RETB_{args.variant}_S{args.pipeline_seed}"
    if args.variant == "J4_BRIDGE_FINETUNE":
        run_id += f"_N{args.final_particle_blocks}"
    assets = args.output_dir / "assets"
    run_path = assets / "run.json"
    dataset_paths, datasets = {}, {}
    for split in (training_role, "val_stop", "val_design"):
        dataset, view_sha = _dataset(
            root,
            selected=selected,
            split=split,
            means=means,
            deviations=deviations,
            identity_sha=identity_manifests[split]["content_hash"],
            target_cache_sha=parent_hashes["offline_target_cache"],
            normalizer_set_sha=normalizer_set["content_hash"],
            live=live,
            scale_index=scale_index,
        )
        view_key = {
            training_role: f"{training_role}_R_MULTI_view_cache",
            "val_stop": "val_stop_R_MULTI_view_cache",
            "val_design": "val_design_fixed_view_cache",
        }[split]
        parent_hashes[view_key] = view_sha
        location = assets / "datasets" / split
        datasets[split] = (dataset, view_sha, location)
        dataset_paths[split] = location / "joint_dataset.json"
    if scale_index is None:
        run = bind_source(
            materialize_stage_j_run(
                run_id=run_id,
                variant=args.variant,
                pipeline_seed=args.pipeline_seed,
                final_particle_blocks=args.final_particle_blocks,
                predictor_bundle_lock_sha256=lock["content_hash"],
                step11_bundle_sha256=step11["content_hash"],
                parent_hashes=parent_hashes,
                semantic_label=SEMANTIC_LABELS[args.variant],
            ),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        validate_materialized_stage_j_run(run)
    else:
        if args.variant != "J5_END_TO_END":
            raise ValueError("scale continuation is locked to selected J5")
        base_run = load_hashed_json(scale_index["base_j5_run_path"])
        run = bind_source(
            build_scale_stage_j_run(
                base_run=base_run,
                scale_parent_hashes=parent_hashes,
            ),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
    write_immutable_json(run_path, run)
    for split, (dataset, view_sha, location) in datasets.items():
        publish_joint_dataset_cache(
            output_dir=location,
            dataset=dataset,
            parent_hashes={
                "identity_manifest": identity_manifests[split][
                    "content_hash"
                ],
                "HLT_view_cache": view_sha,
                "offline_target_cache": parent_hashes[
                    "offline_target_cache"
                ],
                "target_normalizer_set": normalizer_set["content_hash"],
            },
            source_snapshot=source_snapshot(REPO_ROOT),
        )
    graph_parents = {
        name: parent_hashes[name]
        for name in (
            "frozen_offline_fusion",
            "frozen_offline_expert_heads",
            "offline_target_cache",
            "selected_predictor_seed_artifacts",
            "target_normalizer_set",
        )
    }
    if live:
        graph_parents["selected_HLT_expert_seed_artifacts"] = hlt_sha
    if args.variant == "J5_END_TO_END":
        graph_parents["j4_block_selection"] = j4_selection_sha
        graph_parents["selected_J4_bridge_initialization"] = (
            j4_initialization_sha
        )
    graph_dir = assets / "graph"
    publish_joint_graph_template(
        output_dir=graph_dir,
        graph=graph,
        run_record_sha256=run["content_hash"],
        predictor_bundle_lock_sha256=lock["content_hash"],
        objective_by_expert=objectives,
        gradnorm_weights_by_expert=gradnorm,
        component_parent_hashes=graph_parents,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    train_main(
        [
            "--campaign-root",
            str(root),
            "--run",
            str(run_path),
            "--predictor-bundle-lock",
            str(lock_path),
            "--graph-template",
            str(graph_dir / "joint_graph_template.json"),
            "--model-train-cache",
            str(dataset_paths[training_role]),
            "--val-stop-cache",
            str(dataset_paths["val_stop"]),
            "--val-design-cache",
            str(dataset_paths["val_design"]),
            "--output-dir",
            str(args.output_dir),
            "--device",
            args.device,
            *(
                ["--training-role", "scale_train"]
                if scale_index is not None
                else []
            ),
        ]
    )
    if scale_index is not None:
        registration = load_hashed_json(
            args.output_dir / "registration.json"
        )
        curves = load_hashed_json(
            args.output_dir / "training_curves.json"
        )
        completion = bind_source(
            build_scale_joint_completion(
                graph_id=scale_index["graph_id"],
                pipeline_seed=args.pipeline_seed,
                scale_component_index_sha256=scale_index["content_hash"],
                scale_stage_j_run_sha256=run["content_hash"],
                joint_checkpoint_sha256=_sha256(
                    args.output_dir / "best_model_val.pt"
                ),
                joint_registration_sha256=registration["content_hash"],
                joint_training_curves_sha256=curves["content_hash"],
            ),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        write_immutable_json(
            args.scale_joint_completion_output, completion
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
