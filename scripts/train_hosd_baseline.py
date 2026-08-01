#!/usr/bin/env python3
"""Compile Stage C and optionally train one exact HOSD baseline row."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    authorize_access,
    build_baseline_model,
    build_baseline_registry,
    build_stage_c_plan,
    component_seed,
    exact_trainable_parameter_count,
    load_materialized_hlt_input_view,
    load_and_validate_campaign,
    monolithic_flop_ledger,
)
from teacher_logit_reco.hlt_offline_structure_distillation.baselines import (  # noqa: E402
    HOSDTrainingProtocol,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    COMBINATION_RESULT_CONTRACT,
    with_content_hash,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.hlt_offline_structure_distillation.stage_c_training import (  # noqa: E402
    evaluate_classifier,
    train_stage_c_baseline,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (  # noqa: E402
    identity_order_hash,
    load_hlt_v3_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_experts import (  # noqa: E402
    NativeHLTExpertDataset,
    make_native_hlt_expert_loader,
)
from teacher_logit_reco.hlt_offline_structure_distillation.wave_completion import (  # noqa: E402
    try_finalize_row_wave,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(
    rows: list[str], *, name: str, required_replicas: set[int] | None = None
) -> dict[int, Path]:
    output = {}
    for row in rows:
        if "=" not in row:
            raise ValueError(f"{name} requires REPLICA=PATH")
        key, value = row.split("=", 1)
        replica = int(key)
        if replica not in range(4) or replica in output:
            raise ValueError(f"{name} has invalid/duplicate replica")
        output[replica] = Path(value)
    expected = set(range(4)) if required_replicas is None else set(required_replicas)
    if set(output) != expected:
        raise ValueError(f"{name} requires exact replicas {sorted(expected)}")
    return output


def _labels(
    path: Path,
    *,
    canonical_identities: Sequence[str] | None = None,
) -> tuple[np.ndarray, Sequence[str]]:
    from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (
        identity_order_hash,
    )

    with np.load(path, allow_pickle=False) as payload:
        if not {"identities", "labels"}.issubset(payload.files):
            raise ValueError("labels NPZ lacks identities/labels")
        raw_identities = payload["identities"]
        observed_identity_hash = identity_order_hash(raw_identities)
        labels = np.asarray(payload["labels"], dtype=np.int64)
    identities = (
        tuple(str(value) for value in raw_identities.tolist())
        if canonical_identities is None
        else canonical_identities
    )
    if (
        len(identities) == 0
        or labels.shape != (len(identities),)
        or bool(((labels < 0) | (labels >= 10)).any())
        or observed_identity_hash != identity_order_hash(identities)
        or (
            canonical_identities is None
            and len(identities) != len(set(identities))
        )
    ):
        raise ValueError("label population differs")
    return labels, identities


def _privileged(path: Path | None, identities: Sequence[str], field: str):
    if path is None:
        return None
    if path.is_dir():
        from teacher_logit_reco.hlt_offline_structure_distillation import (
            load_target_cache,
        )

        spec = load_hashed_json(
            path / "cache_spec.json",
            expected_contract="hosd_target_cache_spec_v1",
        )
        cache = load_target_cache(path, cache_spec=spec)
        target_ids = [
            target_id
            for target_id in cache.values
            if target_id.startswith("T_OFFLINE_LOGITS_")
        ]
        if field != "logits" or len(target_ids) != 1:
            raise ValueError("privileged target-cache coordinate differs")
        if cache.identities != identities:
            raise ValueError("privileged target-cache identities differ")
        return np.asarray(cache.values[target_ids[0]], dtype=np.float32)
    mmap_manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if mmap_manifest_path.is_file():
        from teacher_logit_reco.hlt_offline_structure_distillation import (
            identity_order_sha256,
        )

        artifact = load_hashed_json(
            mmap_manifest_path,
            expected_contract="hosd_scale_teacher_logits_mmap_v1",
        )
        member = artifact.get("member", {})
        member_path = Path(str(member.get("path", "")))
        if (
            field != "logits"
            or artifact.get("parents", {}).get("identity_order_sha256")
            != identity_order_sha256(identities)
            or _sha256_file(path) != artifact.get("npz_sha256")
            or member_path.is_symlink()
            or not member_path.is_file()
            or _sha256_file(member_path) != member.get("sha256")
        ):
            raise ValueError("privileged mmap target lineage differs")
        value = np.load(member_path, mmap_mode="r", allow_pickle=False)
        if (
            list(value.shape) != member.get("shape")
            or str(value.dtype) != member.get("dtype")
            or value.shape[0] != len(identities)
        ):
            raise ValueError("privileged mmap target shape differs")
        return value
    with np.load(path, allow_pickle=False) as payload:
        if not {"identities", field}.issubset(payload.files):
            raise ValueError(f"privileged NPZ lacks identities/{field}")
        observed = tuple(str(value) for value in payload["identities"].tolist())
        value = np.asarray(payload[field], dtype=np.float32)
    if observed != identities or value.shape[0] != len(identities):
        raise ValueError("privileged target identity order differs")
    return value


def _native_targets(path: Path | None, identities: Sequence[str]):
    if path is None:
        return None
    from teacher_logit_reco.hlt_offline_structure_distillation.native_relations import (
        NATIVE_RELATION_TARGET_CONTRACT,
    )

    artifact = load_hashed_json(
        path.with_suffix(".manifest.json"),
        expected_contract=NATIVE_RELATION_TARGET_CONTRACT,
    )
    digest = _sha256_file(path)
    store_definition = artifact.get("mmap_store")
    if (
        digest != artifact.get("npz_sha256")
        or artifact.get("storage_layout")
        != "compressed_npz_plus_authenticated_npy_mmap_v1"
        or not isinstance(store_definition, dict)
    ):
        raise ValueError("native relation artifact lineage differs")
    store = path.parent / str(store_definition.get("directory", ""))
    mapped = {}
    for name in ("identities", "targets", "target_mask", "availability"):
        member = store_definition.get("members", {}).get(name)
        member_path = store / str(member.get("filename", ""))
        if (
            not isinstance(member, dict)
            or member_path.parent != store
            or member_path.is_symlink()
            or not member_path.is_file()
            or _sha256_file(member_path) != member.get("sha256")
        ):
            raise ValueError("native relation memory-map member differs")
        mapped[name] = np.load(member_path, mmap_mode="r", allow_pickle=False)
    target = mapped["targets"]
    mask = mapped["target_mask"]
    availability = mapped["availability"]
    target_identities = tuple(str(value) for value in mapped["identities"])
    requested_identities = tuple(str(value) for value in identities)
    target_positions = {
        identity: index for index, identity in enumerate(target_identities)
    }
    if (
        len(target_positions) != len(target_identities)
        or len(requested_identities) != len(set(requested_identities))
        or set(target_positions) != set(requested_identities)
        or target.shape != (len(identities), 545)
        or mask.shape != target.shape
        or availability.shape != (len(identities), 7)
    ):
        raise ValueError("native relation target identity/shape differs")
    return {
        "targets": target,
        "target_mask": mask,
        "availability": availability,
        "source_indices": np.asarray(
            [target_positions[identity] for identity in requested_identities],
            dtype=np.int64,
        ),
    }


class _ReplicaNativeTargetDataset:
    """Attach the native summary extracted from the exact sampled HLT replica."""

    def __init__(self, base, targets_by_replica):
        self.base = base
        self.targets_by_replica = {
            int(key): dict(value)
            for key, value in targets_by_replica.items()
        }
        if set(self.targets_by_replica) != set(base.replicas):
            raise ValueError("native target/HLT replica coverage differs")
        if any(
            value["targets"].shape != (len(base), 545)
            or value["target_mask"].shape != (len(base), 545)
            or value["availability"].shape != (len(base), 7)
            or value["source_indices"].shape != (len(base),)
            for value in self.targets_by_replica.values()
        ):
            raise ValueError("native relation packed target shape differs")
        self.identities = base.identities
        self.logical_role = base.logical_role
        self.realization_policy = base.realization_policy
        self.replicas = base.replicas
        self.metadata = base.metadata
        # Stage-C uses this only as an availability marker; values are selected
        # in item_for_replica below.
        self.offline_target_tokens = True

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def __getitem__(self, index):
        row = dict(self.base[index])
        replica = int(row["replica_id"])
        target = self.targets_by_replica[replica]
        target_index = int(target["source_indices"][index])
        row["offline_target_tokens"] = np.concatenate(
            (
                target["targets"][target_index],
                target["target_mask"][target_index].astype(np.float32),
                target["availability"][target_index],
            )
        )[None, :]
        row["offline_target_logits"] = np.zeros(10, dtype=np.float32)
        return row


def _dataset(
    caches: dict[int, Path],
    labels_path: Path,
    *,
    role: str,
    teacher_logits: Path | None,
    native_relation_targets: dict[int, Path] | None,
):
    arrays, metadata = {}, {}
    for replica, path in caches.items():
        arrays[replica], metadata[replica] = (
            load_materialized_hlt_input_view(path)
            if path.is_file()
            else load_hlt_v3_cache(path)
        )
    canonical_identities = (
        arrays[min(arrays)]["identities"] if role == "scale_train" else None
    )
    labels, identities = _labels(
        labels_path, canonical_identities=canonical_identities
    )
    source_indices = {}
    source_roles = {str(value["logical_role"]) for value in metadata.values()}
    if len(source_roles) != 1:
        raise ValueError("baseline HLT cache logical roles differ")
    for replica, replica_arrays in arrays.items():
        raw_source_ids = replica_arrays["identities"]
        if (
            len(raw_source_ids) == len(identities)
            and identity_order_hash(raw_source_ids) == identity_order_hash(identities)
        ):
            source_indices[replica] = range(len(identities))
        else:
            source_ids = tuple(str(value) for value in raw_source_ids)
            positions = {value: index for index, value in enumerate(source_ids)}
            if len(positions) != len(source_ids) or not set(identities).issubset(positions):
                raise ValueError("baseline HLT cache lacks label identities")
            source_indices[replica] = np.asarray(
                [positions[value] for value in identities], dtype=np.int64
            )
    logits = _privileged(teacher_logits, identities, "logits")
    native = (
        None
        if native_relation_targets is None
        else {
            replica: _native_targets(path, identities)
            for replica, path in native_relation_targets.items()
        }
    )
    if logits is not None and native is not None:
        raise ValueError("one Stage-C row cannot consume KD and native-aux targets")
    tokens = None
    paired_logits = None
    if logits is not None:
        tokens = np.broadcast_to(
            np.zeros((1, 1, 1), dtype=np.float32),
            (len(identities), 1, 1),
        )
        paired_logits = logits
    dataset = NativeHLTExpertDataset(
        replica_arrays=arrays,
        replica_metadata=metadata,
        labels=labels,
        identities=identities,
        logical_role=role,
        realization_policy=(
            "R_MULTI" if role in {"model_train", "scale_train"} else "R_FIXED"
        ),
        source_indices_by_replica=source_indices,
        source_logical_role=next(iter(source_roles)),
        offline_target_tokens=tokens,
        offline_target_logits=paired_logits,
    )
    return (
        dataset
        if native is None
        else _ReplicaNativeTargetDataset(dataset, native)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--baseline-id")
    parser.add_argument("--train-cache", action="append", default=[])
    parser.add_argument("--val-stop-cache", action="append", default=[])
    parser.add_argument("--design-select-cache", action="append", default=[])
    parser.add_argument("--train-labels", type=Path)
    parser.add_argument("--val-stop-labels", type=Path)
    parser.add_argument("--design-select-labels", type=Path)
    parser.add_argument("--train-teacher-logits", type=Path)
    parser.add_argument("--train-native-relation-targets", type=Path)
    parser.add_argument(
        "--train-native-relation-target",
        action="append",
        default=[],
        metavar="REPLICA=NPZ",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args(argv)

    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    target_registry = load_hashed_json(
        args.campaign_root / "registry" / "structure_target_registry.json",
        expected_contract="hosd_structure_target_registry_v1",
    )
    if target_registry.get("source") != campaign["source"]:
        raise ValueError("target registry source differs")
    baseline_registry = build_baseline_registry(source=campaign["source"])
    baseline_path = args.campaign_root / "registry" / "stage_c_baselines.json"
    write_immutable_json(baseline_path, baseline_registry)
    plan = build_stage_c_plan(
        campaign_spec_sha256=campaign["content_hash"],
        target_registry=target_registry,
        baseline_registry=baseline_registry,
        source=campaign["source"],
    )
    plan_path = args.campaign_root / "job_ledgers" / "stage_c_execution_plan.json"
    write_immutable_json(plan_path, plan)
    result = {
        "baseline_registry_sha256": baseline_registry["content_hash"],
        "stage_c_plan_sha256": plan["content_hash"],
        "baseline_row_count": len(plan["baseline_rows"]),
        "probe_row_count": plan["probe_row_count"],
    }
    if args.baseline_id is None or args.dry_run:
        result["executed"] = False
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    for resource in (
        "model_train_hlt",
        "model_train_labels",
        "model_train_targets",
        "val_stop_hlt",
        "val_stop_labels",
    ):
        authorize_access(worker_role="train_worker", requested_resource=resource)
    row = next(
        (row for row in plan["baseline_rows"] if row["baseline_id"] == args.baseline_id),
        None,
    )
    if row is None:
        raise ValueError("baseline ID is not in the frozen Stage-C plan")
    row = {
        **row,
        "pipeline_seed": int(args.seed),
        "component_seed": component_seed(
            int(args.seed), "baseline", args.baseline_id
        ),
    }
    required = {
        "--train-labels": args.train_labels,
        "--val-stop-labels": args.val_stop_labels,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing or not args.train_cache or not args.val_stop_cache:
        raise ValueError(f"baseline execution lacks caches/arguments: {missing}")
    native_paths = None
    if args.baseline_id == "H_NATIVE_REL_AUX":
        if args.train_native_relation_targets is not None:
            raise ValueError(
                "singular --train-native-relation-targets is unsafe under "
                "R_MULTI; use exact replica bindings"
            )
        native_paths = (
            _mapping(
                args.train_native_relation_target,
                name="--train-native-relation-target",
            )
            if args.train_native_relation_target
            else {
                replica: args.campaign_root
                / "targets"
                / "native_relations"
                / "model_train"
                / f"replica_{replica}.npz"
                for replica in range(4)
            }
        )
    train = _dataset(
        _mapping(args.train_cache, name="--train-cache"),
        args.train_labels,
        role="model_train",
        teacher_logits=args.train_teacher_logits,
        native_relation_targets=native_paths,
    )
    val = _dataset(
        _mapping(
            args.val_stop_cache,
            name="--val-stop-cache",
            required_replicas={0},
        ),
        args.val_stop_labels,
        role="val_stop",
        teacher_logits=None,
        native_relation_targets=None,
    )
    miniature = campaign["campaign_profile"] == "miniature_test"
    epochs = 2 if miniature else int(row["epochs"])
    protocol = HOSDTrainingProtocol(
        maximum_epochs=epochs,
        campaign_profile="miniature_test" if miniature else "production",
    )
    module = importlib.import_module("weaver.nn.model.ParticleTransformer")
    import torch
    torch.manual_seed(int(row["component_seed"]))
    model = build_baseline_model(args.baseline_id, weaver_module=module)
    output = args.output_dir or (
        args.campaign_root / "baselines" / args.baseline_id / f"seed_{args.seed}"
    )
    completion = train_stage_c_baseline(
        model=model,
        train_loader=make_native_hlt_expert_loader(
            train, seed=int(row["component_seed"]), training=True, batch_size=64
        ),
        val_stop_loader=make_native_hlt_expert_loader(
            val, seed=int(row["component_seed"]), training=False, batch_size=64
        ),
        output_dir=output,
        baseline_id=args.baseline_id,
        seed=int(row["pipeline_seed"]),
        component_seed=int(row["component_seed"]),
        baseline_registry_sha256=baseline_registry["content_hash"],
        campaign_spec_sha256=campaign["content_hash"],
        lineage_hashes={
            "target_registry": target_registry["content_hash"],
            "stage_c_plan": plan["content_hash"],
        },
        protocol=protocol,
        teacher_id=row["teacher_id"],
        teacher_logit_key=(
            "offline_target_logits" if row["teacher_id"] is not None else None
        ),
        device=(
            "cuda" if args.device == "auto" and torch.cuda.is_available()
            else "cpu" if args.device == "auto" else args.device
        ),
        source=campaign["source"],
    )
    if args.baseline_id == "H_BASE_BEAM_BUDGET":
        if not args.design_select_cache or args.design_select_labels is None:
            raise ValueError(
                "beam-budget H_BASE requires design-select cache and labels"
            )
        design = _dataset(
            _mapping(
                args.design_select_cache,
                name="--design-select-cache",
                required_replicas={0},
            ),
            args.design_select_labels,
            role="design_select",
            teacher_logits=None,
            native_relation_targets=None,
        )
        checkpoint_path = output / completion["checkpoint_file"]
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if (
            checkpoint.get("source") != campaign["source"]
            or checkpoint.get("baseline_id") != args.baseline_id
            or checkpoint.get("lineage_hashes") != completion["lineage_hashes"]
        ):
            raise ValueError("beam-root checkpoint lineage differs")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        resolved_device = (
            "cuda"
            if args.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if args.device == "auto"
            else args.device
        )
        design_metrics = evaluate_classifier(
            model,
            make_native_hlt_expert_loader(
                design,
                seed=int(row["component_seed"]),
                training=False,
                batch_size=64,
            ),
            device=torch.device(resolved_device),
            split="design_select",
        )
        result_artifact = with_content_hash(
            {
                "contract": COMBINATION_RESULT_CONTRACT,
                "schema_version": 1,
                "source": dict(campaign["source"]),
                "graph_id": "H_BASE_BEAM_BUDGET",
                "combination_id": "H_BASE_BEAM_BUDGET",
                "members": [],
                "normalized_weights": {},
                "budget": "BEAM_5_EPOCH",
                "weighting": "W_FIXED",
                "selection_eligible": False,
                "fixed_epoch_budget": 5,
                "source_result_reused_for_omit": True,
                "campaign_spec_sha256": campaign["content_hash"],
                "baseline_completion_sha256": completion["content_hash"],
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint_path.read_bytes()
                ).hexdigest(),
                "design_select": {
                    "classification_metrics": design_metrics,
                },
                "deployed_analytical_flops": float(
                    monolithic_flop_ledger(
                        {
                            "embed_dim": 128,
                            "attention_heads": 8,
                            "particle_blocks": 8,
                            "class_blocks": 2,
                        }
                    )["total_flops"]
                ),
                "deployed_parameter_count": exact_trainable_parameter_count(
                    model
                ),
                "training_gpu_hours": 0.0,
            }
        )
        write_immutable_json(
            output / "design_select_result.json", result_artifact
        )
        result["beam_root_result_sha256"] = result_artifact["content_hash"]
    result.update({"executed": True, "completion_sha256": completion["content_hash"]})
    wave = try_finalize_row_wave(
        wave_id="stage_c_baselines",
        expected_paths={
            item["baseline_id"]: (
                args.campaign_root
                / "baselines"
                / item["baseline_id"]
                / f"seed_{args.seed}"
                / "baseline_completion.json"
            )
            for item in plan["baseline_rows"]
        },
        expected_rows={
            item["baseline_id"]: {
                "baseline_id": item["baseline_id"],
                "epochs": int(item["epochs"]),
            }
            for item in plan["baseline_rows"]
        },
        expected_contract="hosd_baseline_completion_v1",
        parent_hashes={
            "stage_c_plan": plan["content_hash"],
            "baseline_registry": baseline_registry["content_hash"],
        },
        source=campaign["source"],
        output=args.campaign_root
        / "baselines"
        / "baseline_completion.json",
    )
    result["wave_completion_sha256"] = (
        None if wave is None else wave["content_hash"]
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
