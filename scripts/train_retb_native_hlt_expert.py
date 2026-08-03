#!/usr/bin/env python3
"""Train one immutable RETB Stage-D native-HLT evidence expert."""

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
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_model import (  # noqa: E402
    RetbExpertModel,
    RetbParticleEncoder,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (  # noqa: E402
    load_hlt_v3_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_experts import (  # noqa: E402
    HLT_EVALUATION_REALIZATION_POLICY,
    NativeHLTExpertDataset,
    NativeHLTExpertTrainingConfig,
    infer_native_hlt_expert_replica,
    make_native_hlt_expert_loader,
    train_native_hlt_expert,
)
from teacher_logit_reco.relation_expert_token_bridge.step6 import (  # noqa: E402
    resolve_stage_d_confirmation_run,
    resolve_stage_d_run,
    validate_stage_d_confirmation_registry,
    validate_stage_d_run_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.token_shape_registry import (  # noqa: E402
    HET_PHYSICS,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)
from scripts.materialize_retb_stage_d_offline_targets import (  # noqa: E402
    TARGET_INDEX_CONTRACT,
)
from teacher_logit_reco.relational_part.ca_tree import unpack_tree_shard  # noqa: E402

import torch  # noqa: E402


def _mapping(rows: Sequence[str], *, argument: str) -> dict[int, Path]:
    output = {}
    for row in rows:
        if "=" not in row:
            raise ValueError(f"{argument} requires REPLICA=PATH")
        key, value = row.split("=", 1)
        replica = int(key)
        if replica in output or replica not in range(4):
            raise ValueError(f"{argument} has a duplicate/invalid replica")
        output[replica] = Path(value)
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_output(path: Path, payload: Mapping[str, Any]) -> str:
    arrays = {
        name: np.asarray(payload[name])
        for name in (
            "identities",
            "labels",
            "tokens",
            "logits",
            "particle_states",
            "particle_mask",
        )
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays) or any(
                not np.array_equal(existing[name], value)
                for name, value in arrays.items()
            ):
                raise FileExistsError("native HLT output differs on reuse")
        return _sha256(path)
    np.savez_compressed(path, **arrays)
    return _sha256(path)


def _labels(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    with np.load(path, allow_pickle=False) as payload:
        if not {"identities", "labels"}.issubset(payload.files):
            raise ValueError("native HLT labels NPZ lacks identity/label fields")
        identities = tuple(str(value) for value in payload["identities"].tolist())
        labels = np.asarray(payload["labels"], dtype=np.int64)
    if (
        not identities
        or len(identities) != len(set(identities))
        or labels.shape != (len(identities),)
        or bool(((labels < 0) | (labels >= 10)).any())
    ):
        raise ValueError("native HLT labels population differs")
    return labels, identities


def _targets(
    path: Path | None, identities: tuple[str, ...]
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if path is None:
        return None, None
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"identities", "tokens", "logits"}:
            raise ValueError("native HLT offline-target fields differ")
        target_ids = tuple(str(value) for value in payload["identities"].tolist())
        tokens = np.asarray(payload["tokens"], dtype=np.float32)
        logits = np.asarray(payload["logits"], dtype=np.float32)
    if target_ids != identities:
        raise ValueError("native HLT offline-target identities differ")
    return tokens, logits


def _checkpoint(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, Mapping):
        raise ValueError("offline expert checkpoint lacks model state")
    return state


def _trees(
    root: Path | None,
    *,
    logical_role: str,
    replicas: Sequence[int],
    identities: tuple[str, ...],
) -> dict[int, list[Mapping[str, Any]]] | None:
    if root is None:
        return None
    output = {}
    for replica in replicas:
        shard_root = (
            root
            / f"{logical_role}_r{replica}_exclusive_ca_v1"
            / "shards"
        )
        shards = sorted(shard_root.glob("shard_*.npz"))
        if not shards:
            raise FileNotFoundError(f"REGION shards are absent under {shard_root}")
        by_identity = {}
        for shard in shards:
            shard_ids, shard_trees = unpack_tree_shard(shard)
            for identity, tree in zip(shard_ids, shard_trees):
                if str(identity) in by_identity:
                    raise ValueError("REGION tree identity is duplicated")
                by_identity[str(identity)] = tree
        if set(by_identity) != set(identities):
            raise ValueError("REGION tree identity coverage differs")
        output[replica] = [by_identity[identity] for identity in identities]
    return output


def _resolved_shape(
    *,
    alias: str,
    expert: str,
    uniform_selection: Mapping[str, Any] | None,
    heterogeneous_selection: Mapping[str, Any] | None,
) -> tuple[str | None, int, int]:
    if alias == "S1_128":
        return "S1_128", 1, 128
    if alias in {"SHAPE_COMPACT", "SHAPE_HIGH"}:
        if uniform_selection is None:
            raise ValueError("selected uniform shape requires --uniform-shapes")
        row = uniform_selection[alias]
        return str(row["shape_id"]), int(row["K"]), int(row["D"])
    if alias == "HET_PHYSICS":
        return None, int(HET_PHYSICS[expert]), 128
    if alias in {"HET_SELECTED", "HET_BEAM"}:
        if heterogeneous_selection is None:
            raise ValueError(
                "selected heterogeneous shape requires --heterogeneous-shapes"
            )
        return None, int(
            heterogeneous_selection[alias]["allocation"][expert]
        ), 128
    raise ValueError("native HLT shape alias is unknown")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--training-role",
        choices=("model_train", "scale_train"),
        default="model_train",
    )
    parser.add_argument("--confirmation-registry", type=Path)
    parser.add_argument("--train-cache", action="append", default=[])
    parser.add_argument("--val-stop-cache", action="append", default=[])
    parser.add_argument("--val-design-cache", action="append", default=[])
    parser.add_argument("--train-labels", required=True, type=Path)
    parser.add_argument("--val-stop-labels", required=True, type=Path)
    parser.add_argument("--val-design-labels", type=Path)
    parser.add_argument("--uniform-shapes", type=Path)
    parser.add_argument("--heterogeneous-shapes", type=Path)
    parser.add_argument("--offline-registration", required=True, type=Path)
    parser.add_argument("--offline-checkpoint", required=True, type=Path)
    parser.add_argument("--offline-train-targets", type=Path)
    parser.add_argument("--relation-normalization", type=Path)
    parser.add_argument("--region-normalization", type=Path)
    parser.add_argument("--region-tree-root", type=Path)
    parser.add_argument("--microbatch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    registry = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_d_runs.json"
    )
    base_registry_sha = validate_stage_d_run_registry(registry)
    if args.confirmation_registry is None:
        registry_sha = base_registry_sha
        run = resolve_stage_d_run(registry, run_id=args.run_id)
    else:
        confirmation = load_hashed_json(args.confirmation_registry)
        registry_sha = validate_stage_d_confirmation_registry(confirmation)
        if (
            confirmation["stage_d_run_registry_sha256"] != base_registry_sha
            or confirmation.get("source") != campaign.get("source")
        ):
            raise ValueError("Stage-D confirmation lineage differs")
        run = resolve_stage_d_confirmation_run(
            confirmation, run_id=args.run_id
        )
    configuration = run["configuration"]
    if configuration.get("kind") != "NATIVE_HLT_EXPERT":
        raise ValueError("Stage-D row is not a native HLT expert")
    offline_registration = load_hashed_json(args.offline_registration)
    if (
        offline_registration["expert_id"] != configuration["expert_id"]
        or _sha256(args.offline_checkpoint)
        != offline_registration["checkpoint_sha256"]
        or offline_registration.get("lineage_hashes", {}).get("campaign_spec")
        != campaign["content_hash"]
    ):
        raise ValueError("corresponding offline expert lineage differs")
    if (
        offline_registration.get("source") is not None
        and offline_registration["source"] != campaign.get("source")
    ):
        raise ValueError("offline expert belongs to another source snapshot")
    uniform = (
        load_hashed_json(args.uniform_shapes)
        if args.uniform_shapes is not None
        else None
    )
    heterogeneous = (
        load_hashed_json(args.heterogeneous_shapes)
        if args.heterogeneous_shapes is not None
        else None
    )
    for artifact in (uniform, heterogeneous):
        if artifact is not None and artifact.get("source") != campaign.get("source"):
            raise ValueError("native HLT shape selection belongs to another source")
    shape_id, token_count, token_dimension = _resolved_shape(
        alias=configuration["shape_id"],
        expert=configuration["expert_id"],
        uniform_selection=uniform,
        heterogeneous_selection=heterogeneous,
    )
    if (
        int(offline_registration["token_count"]) != token_count
        or int(offline_registration["token_dimension"]) != token_dimension
    ):
        raise ValueError("offline/native expert token shapes differ")
    train_paths = _mapping(args.train_cache, argument="--train-cache")
    val_paths = _mapping(args.val_stop_cache, argument="--val-stop-cache")
    design_paths = _mapping(
        args.val_design_cache, argument="--val-design-cache"
    )
    train_arrays, train_metadata = {}, {}
    val_arrays, val_metadata = {}, {}
    design_arrays, design_metadata = {}, {}
    for replica, path in train_paths.items():
        train_arrays[replica], train_metadata[replica] = load_hlt_v3_cache(path)
    for replica, path in val_paths.items():
        val_arrays[replica], val_metadata[replica] = load_hlt_v3_cache(path)
    for replica, path in design_paths.items():
        design_arrays[replica], design_metadata[replica] = load_hlt_v3_cache(
            path
        )
    train_labels, train_ids = _labels(args.train_labels)
    val_labels, val_ids = _labels(args.val_stop_labels)
    if (args.val_design_labels is None) != (not design_paths):
        raise ValueError(
            "val_design labels and cache must be provided together"
        )
    design_labels, design_ids = (
        (None, None)
        if args.val_design_labels is None
        else _labels(args.val_design_labels)
    )
    target_tokens, target_logits = _targets(
        args.offline_train_targets, train_ids
    )
    mode = configuration["mode"]
    if (mode == "HE_DUAL_OBJECTIVE") != (target_tokens is not None):
        raise ValueError("native HLT target availability differs from evidence mode")
    if args.offline_train_targets is not None:
        target_index = load_hashed_json(
            args.campaign_root
            / "inputs"
            / "stage_d_offline_targets"
            / "index.json",
            expected_contract=TARGET_INDEX_CONTRACT,
        )
        relative = args.offline_train_targets.resolve().relative_to(
            args.campaign_root.resolve()
        ).as_posix()
        matching = [
            row
            for row in target_index["records"]
            if row["relative_path"] == relative
        ]
        if (
            target_index.get("source") != campaign.get("source")
            or len(matching) != 1
            or matching[0]["file_sha256"]
            != _sha256(args.offline_train_targets)
            or matching[0]["shape_alias"] != configuration["shape_id"]
            or matching[0]["expert_id"] != configuration["expert_id"]
            or int(matching[0]["pipeline_seed"]) != int(run["seed"])
        ):
            raise ValueError("native HLT offline-target index lineage differs")
    train_dataset = NativeHLTExpertDataset(
        replica_arrays=train_arrays,
        replica_metadata=train_metadata,
        labels=train_labels,
        identities=train_ids,
        logical_role=args.training_role,
        realization_policy=configuration["realization_policy"],
        offline_target_tokens=target_tokens,
        offline_target_logits=target_logits,
        region_trees_by_replica=_trees(
            args.region_tree_root,
            logical_role=args.training_role,
            replicas=sorted(train_paths),
            identities=train_ids,
        ),
    )
    val_dataset = NativeHLTExpertDataset(
        replica_arrays=val_arrays,
        replica_metadata=val_metadata,
        labels=val_labels,
        identities=val_ids,
        logical_role="val_stop",
        realization_policy=HLT_EVALUATION_REALIZATION_POLICY,
        region_trees_by_replica=_trees(
            args.region_tree_root,
            logical_role="val_stop",
            replicas=sorted(val_paths),
            identities=val_ids,
        ),
    )
    design_dataset = (
        None
        if design_labels is None
        else NativeHLTExpertDataset(
            replica_arrays=design_arrays,
            replica_metadata=design_metadata,
            labels=design_labels,
            identities=design_ids,
            logical_role="val_design",
            realization_policy="R_FIXED",
            region_trees_by_replica=_trees(
                args.region_tree_root,
                logical_role="val_design",
                replicas=sorted(design_paths),
                identities=design_ids,
            ),
        )
    )
    relation_normalization = (
        load_hashed_json(args.relation_normalization)
        if args.relation_normalization is not None
        else None
    )
    region_normalization = (
        load_hashed_json(args.region_normalization)
        if args.region_normalization is not None
        else None
    )
    for artifact in (relation_normalization, region_normalization):
        if artifact is not None and artifact.get("source") != campaign.get("source"):
            raise ValueError("native HLT normalizer belongs to another source")
    expert = configuration["expert_id"]
    if expert != "BASE4" and relation_normalization is None:
        raise ValueError("relation expert requires shared HLT normalization")
    if expert == "REGION" and (
        region_normalization is None or args.region_tree_root is None
    ):
        raise ValueError("REGION expert requires HLT REGION normalization/trees")
    profile = campaign["campaign_profile"]
    miniature = profile == "miniature_test"
    config = NativeHLTExpertTrainingConfig(
        seed=int(run["seed"]),
        mode=mode,
        realization_policy=configuration["realization_policy"],
        measurement_embedding=configuration["measurement_embedding"],
        lambda_token=float(configuration["lambda_token"]),
        lambda_logit=float(configuration["lambda_logit"]),
        maximum_epochs=2 if miniature else 40,
        microbatch_size=args.microbatch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        effective_batch_size=(
            args.microbatch_size * args.gradient_accumulation_steps
        ),
        campaign_profile="miniature_test" if miniature else "production",
    )
    output = args.output_dir or (
        args.campaign_root
        / "runs"
        / "stage_d"
        / "hlt_experts"
        / args.run_id
        / f"seed_{run['seed']}"
    )
    resolved = {
        "dry_run": bool(args.dry_run),
        "run": run,
        "resolved_shape": [token_count, token_dimension],
        "offline_registration_sha256": offline_registration["content_hash"],
        "output_dir": str(output.resolve()),
    }
    print(json.dumps(resolved, indent=2, sort_keys=True))
    if args.dry_run:
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
    if design_dataset is not None:
        authorize_dataset_access(
            worker_role="design_worker", requested_resource="val_design"
        )
    torch.manual_seed(int(run["seed"]))
    weaver = importlib.import_module("weaver.nn.model.ParticleTransformer")
    encoder = RetbParticleEncoder(
        expert_id=expert,
        topology=offline_registration["topology"],
        weaver_module=weaver,
        normalization_artifact=relation_normalization,
        region_normalization_artifact=region_normalization,
        measurement_embedding=configuration["measurement_embedding"],
        activation_checkpointing=True,
        particle_dropout=0.0,
    )
    model = RetbExpertModel(
        particle_encoder=encoder,
        shape_id=shape_id,
        token_count=None if shape_id is not None else token_count,
        token_dimension=None if shape_id is not None else token_dimension,
        tokenizer_mode=offline_registration["tokenizer_mode"],
    )
    train_loader = make_native_hlt_expert_loader(
        train_dataset,
        seed=config.seed,
        training=True,
        batch_size=config.microbatch_size,
    )
    val_loader = make_native_hlt_expert_loader(
        val_dataset,
        seed=0,
        training=False,
        batch_size=config.microbatch_size,
    )
    step6 = load_hashed_json(
        args.campaign_root / "registry" / "retb_step6_native_hlt_bundle.json"
    )
    lineage = {
        "campaign_spec": campaign["content_hash"],
        "run_registry": registry_sha,
        "offline_expert_checkpoint": offline_registration["checkpoint_sha256"],
        "offline_expert_registration": offline_registration["content_hash"],
        f"{args.training_role}_labels": _sha256(args.train_labels),
        "val_stop_labels": _sha256(args.val_stop_labels),
        **{
            f"{args.training_role}_hlt_replica_{replica}": metadata[
                "content_hash"
            ]
            for replica, metadata in train_metadata.items()
        },
        **{
            f"val_stop_hlt_replica_{replica}": metadata["content_hash"]
            for replica, metadata in val_metadata.items()
        },
        **{
            f"val_design_hlt_replica_{replica}": metadata["content_hash"]
            for replica, metadata in design_metadata.items()
        },
    }
    if relation_normalization is not None:
        lineage["shared_hlt_relation_normalization"] = relation_normalization[
            "content_hash"
        ]
    if region_normalization is not None:
        lineage["shared_hlt_region_normalization"] = region_normalization[
            "content_hash"
        ]
    if args.offline_train_targets is not None:
        lineage["offline_train_targets"] = _sha256(
            args.offline_train_targets
        )
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    registration = train_native_hlt_expert(
        model=model,
        train_loader=train_loader,
        val_stop_loader=val_loader,
        output_dir=output,
        run_id=args.run_id,
        run_registry_sha256=registry_sha,
        lineage_hashes=lineage,
        global_determinism_sha256=campaign["parent_artifact_hashes"][
            "global_determinism"
        ],
        evidence_mode_contract_sha256=step6["artifact_hashes"][
            "hlt_evidence_modes"
        ],
        config=config,
        device=device,
        offline_initialization_state=(
            None if mode == "HE_SCRATCH_CE" else _checkpoint(args.offline_checkpoint)
        ),
        offline_checkpoint_sha256=(
            None
            if mode == "HE_SCRATCH_CE"
            else offline_registration["checkpoint_sha256"]
        ),
    )
    output_files = {}
    for split, dataset in (
        (args.training_role, train_dataset),
        ("val_stop", val_dataset),
        *(()
          if design_dataset is None
          else (("val_design", design_dataset),)),
    ):
        for replica in sorted(dataset.replicas):
            prediction = infer_native_hlt_expert_replica(
                model=model,
                dataset=dataset,
                replica_id=replica,
                batch_size=config.microbatch_size,
                device=device,
            )
            relative = Path("native_outputs") / split / f"replica_{replica}.npz"
            output_files[f"{split}_replica_{replica}"] = {
                "relative_path": relative.as_posix(),
                "file_sha256": _publish_output(output / relative, prediction),
                "event_count": int(len(prediction["labels"])),
                "token_shape": list(prediction["tokens"].shape),
                "logit_shape": list(prediction["logits"].shape),
            }
    output_manifest = bind_source(
        with_content_hash(
            {
            "contract": (
                "retb_native_hlt_expert_outputs_v6"
                if args.training_role == "scale_train"
                else "retb_native_hlt_expert_outputs_v5"
            ),
            "schema_version": 6 if args.training_role == "scale_train" else 5,
            "run_id": args.run_id,
            "expert_id": expert,
            "expert_registration_sha256": registration["content_hash"],
            "pipeline_seed": int(run["seed"]),
            "shape_alias": configuration["shape_id"],
            "resolved_token_shape": [token_count, token_dimension],
            "realization_policy": configuration["realization_policy"],
            "training_realization_policy": configuration["realization_policy"],
            "evaluation_realization_policy": (
                HLT_EVALUATION_REALIZATION_POLICY
            ),
            **(
                {"training_population": "scale_train"}
                if args.training_role == "scale_train"
                else {}
            ),
            "files": output_files,
            "offline_targets_persisted": False,
            "val_design_inference_after_checkpoint_lock": (
                design_dataset is not None
            ),
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(output / "native_output_manifest.json", output_manifest)
    print(json.dumps(registration, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
