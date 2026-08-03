"""Executable shared-encoder multi-target Stage-F graph runtime."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from teacher_logit_reco.relation_expert_token_bridge.determinism import (
    optimizer_update_counts,
    scheduled_learning_rate,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (
    evaluate_classification,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_experts import (
    DeterministicExpertSampler,
)

from .auxiliary import global_auxiliary_loss
from .auxiliary_data import DeterministicScaleShardSampler
from .baselines import HOSDTrainingProtocol, component_seed
from .combinations import pcgrad_project
from .contracts import (
    COMBINATION_RESULT_CONTRACT,
    canonical_sha256,
    load_hashed_json,
    require_sha256,
    with_content_hash,
    write_immutable_json,
)
from .heads import GlobalTargetHead
from .stage_c_training import _balanced_metrics, _move, _restore_rng_state, _rng_state
from .stage_d_data_factory import (
    LOADER_MANIFEST_CONTRACT_V7,
    load_stage_d_loaders_from_manifest,
)
from .target_schemas import (
    target_component_availability_groups,
    target_declarations,
)
from .taps import HBaseParticleTransformer

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


COMBINATION_LOADER_MANIFEST_CONTRACT = "hosd_combination_loader_manifest_v1"
COMBINATION_LOADER_MANIFEST_CONTRACT_V2 = (
    "hosd_combination_loader_manifest_v2"
)
COMBINATION_LOADER_MANIFEST_CONTRACT_V3 = (
    "hosd_combination_loader_manifest_v3"
)
COMBINATION_LOADER_MANIFEST_CONTRACT_V4 = (
    "hosd_combination_loader_manifest_v4"
)
COMBINATION_LOADER_MANIFEST_CONTRACT_V5 = (
    "hosd_combination_loader_manifest_v5"
)
COMBINATION_CHECKPOINT_CONTRACT = "hosd_combination_checkpoint_v1"
COMBINATION_COMPLETION_CONTRACT = "hosd_combination_completion_v1"


class CombinationDataset(torch.utils.data.Dataset if torch is not None else object):
    def __init__(
        self,
        datasets: Mapping[str, Any],
        *,
        native_relation_targets: Mapping[
            int, Mapping[str, np.ndarray]
        ] | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for combination datasets")
        if not datasets:
            raise ValueError("combination dataset has no targets")
        self.datasets = dict(sorted(datasets.items()))
        first = next(iter(self.datasets.values()))
        self.base_dataset = first.base_dataset
        self.identities = first.identities
        identity_hash = getattr(
            self.base_dataset, "identity_order_sha256", None
        )
        if identity_hash is None:
            raise ValueError("combination base lacks authenticated identity order")
        if any(
            len(dataset) != len(first)
            or getattr(dataset.base_dataset, "identity_order_sha256", None)
            != identity_hash
            for dataset in self.datasets.values()
        ):
            raise ValueError("combination target datasets are not identity aligned")
        for dataset in self.datasets.values():
            dataset.base_dataset = self.base_dataset
            dataset.identities = self.identities
        self.identity_order_sha256 = identity_hash
        target_stores = {}
        for dataset in self.datasets.values():
            for store in getattr(dataset, "target_stores", ()):
                target_stores[id(store)] = store
        self.target_store_count = len(target_stores)
        self.resident_target_shard_budget = sum(
            bool(getattr(dataset, "target_stores", ()))
            for dataset in self.datasets.values()
        )
        self.control_kind = None
        self.native_relation_targets = (
            None
            if native_relation_targets is None
            else {
                int(replica): {
                    key: np.asarray(value)
                    for key, value in fields.items()
                }
                for replica, fields in native_relation_targets.items()
            }
        )
        if self.native_relation_targets is not None:
            expected_shapes = {
                "targets": (len(self.identities), 545),
                "target_mask": (len(self.identities), 545),
                "availability": (len(self.identities), 7),
            }
            replicas = set(first.base_dataset.replicas)
            if set(self.native_relation_targets) != replicas or any(
                any(
                    key not in fields or fields[key].shape != shape
                    for key, shape in expected_shapes.items()
                )
                for fields in self.native_relation_targets.values()
            ):
                raise ValueError("native relation combination targets differ")

    def __len__(self) -> int:
        return len(next(iter(self.datasets.values())))

    @property
    def logical_role(self) -> str:
        return str(self.base_dataset.logical_role)

    def replica_for_index(self, index: int) -> int:
        return self.base_dataset.replica_for_index(index)

    def locality_boundaries(self) -> tuple[int, ...]:
        boundaries = set(self.base_dataset.locality_boundaries())
        for dataset in self.datasets.values():
            boundaries.update(dataset.locality_boundaries())
        return tuple(sorted(boundaries))

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base_dataset, "set_epoch"):
            self.base_dataset.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_sample = self.base_dataset[index]
        rows = {
            target: dataset.attach_target(index, base_sample)
            for target, dataset in self.datasets.items()
        }
        first = dict(next(iter(rows.values())))
        identity = str(first.get("event_identity"))
        if any(str(row.get("event_identity")) != identity for row in rows.values()):
            raise ValueError("combination sample identity join differs")
        first["combination_targets"] = {
            target: {
                "target": row["target"],
                "target_mask": row["target_mask"],
            }
            for target, row in rows.items()
        }
        if self.native_relation_targets is not None:
            replica = int(first["replica_id"])
            native = self.native_relation_targets[replica]
            first["native_relation_target"] = {
                "target": native["targets"][index],
                "target_mask": native["target_mask"][index],
                "availability": native["availability"][index],
            }
        return first


def collate_combination_batch(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from .auxiliary_data import collate_auxiliary_batch

    proxy = [
        {
            **sample,
            "target": next(iter(sample["combination_targets"].values()))["target"],
            "target_mask": next(iter(sample["combination_targets"].values()))[
                "target_mask"
            ],
        }
        for sample in samples
    ]
    output = collate_auxiliary_batch(proxy)
    targets = sorted(samples[0]["combination_targets"])
    if any(set(sample["combination_targets"]) != set(targets) for sample in samples):
        raise ValueError("combination batch target coverage differs")
    output["combination_targets"] = {
        target: {
            "target": torch.from_numpy(
                np.stack(
                    [
                        sample["combination_targets"][target]["target"]
                        for sample in samples
                    ]
                )
            ).float(),
            "target_mask": torch.from_numpy(
                np.stack(
                    [
                        sample["combination_targets"][target]["target_mask"]
                        for sample in samples
                    ]
                )
            ).bool(),
        }
        for target in targets
    }
    has_native = ["native_relation_target" in sample for sample in samples]
    if any(has_native) and not all(has_native):
        raise ValueError("native relation target batch coverage differs")
    if all(has_native):
        output["native_relation_target"] = {
            "target": torch.from_numpy(
                np.stack(
                    [sample["native_relation_target"]["target"] for sample in samples]
                )
            ).float(),
            "target_mask": torch.from_numpy(
                np.stack(
                    [
                        sample["native_relation_target"]["target_mask"]
                        for sample in samples
                    ]
                )
            ).bool(),
            "availability": torch.from_numpy(
                np.stack(
                    [
                        sample["native_relation_target"]["availability"]
                        for sample in samples
                    ]
                )
            ).float(),
        }
    return output


def make_combination_loader(
    dataset: CombinationDataset,
    *,
    seed: int,
    training: bool,
    batch_size: int,
) -> Any:
    sampler = (
        (
            DeterministicScaleShardSampler(dataset, seed=int(seed))
            if dataset.logical_role == "scale_train"
            else DeterministicExpertSampler(dataset, seed=int(seed))
        )
        if training
        else torch.utils.data.SequentialSampler(dataset)
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_combination_batch,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_combination_loader_manifest(
    *,
    graph: Mapping[str, Any],
    member_loader_manifests: Mapping[str, str | Path],
    campaign_spec_sha256: str,
    source: Mapping[str, Any],
    native_relation_target_files: Mapping[str, str | Path] | None = None,
    evaluation_role: str = "design_select",
    training_role: str = "model_train",
) -> dict[str, Any]:
    if evaluation_role not in {"design_select", "design_confirm"}:
        raise ValueError("combination evaluation role differs")
    if training_role not in {"model_train", "scale_train"}:
        raise ValueError("combination training role differs")
    expected = {row["target_id"] for row in graph["members"]}
    if set(member_loader_manifests) != expected:
        raise ValueError("combination loader member coverage differs")
    rows = {}
    order_attestation = None
    expected_roles = {training_role, "val_stop", evaluation_role}
    for target, raw_path in sorted(member_loader_manifests.items()):
        path = Path(raw_path).resolve()
        digest = _sha256_file(path)
        member_manifest = load_hashed_json(
            path, expected_contract=LOADER_MANIFEST_CONTRACT_V7
        )
        if (
            member_manifest.get("source") != dict(source)
            or member_manifest.get("campaign_spec_sha256")
            != campaign_spec_sha256
            or member_manifest.get("target_id") != target
            or member_manifest.get("training_role") != training_role
            or member_manifest.get("evaluation_role") != evaluation_role
            or set(member_manifest.get("roles", {})) != expected_roles
        ):
            raise ValueError("combination member loader lineage differs")
        candidate = {
            "pipeline_seed": int(member_manifest["pipeline_seed"]),
            "data_order_contract": str(
                member_manifest["data_order_contract"]
            ),
            "sampler_seed_by_role": {
                str(role): int(seed)
                for role, seed in member_manifest[
                    "sampler_seed_by_role"
                ].items()
            },
            "sampler_contract_by_role": dict(
                member_manifest["sampler_contract_by_role"]
            ),
            "seed_derivation_contract_by_role": dict(
                member_manifest["seed_derivation_contract_by_role"]
            ),
        }
        if order_attestation is None:
            order_attestation = candidate
        elif candidate != order_attestation:
            raise ValueError("combination member data-order contracts differ")
        rows[target] = {
            "path": str(path),
            "sha256": digest,
            "artifact_sha256": member_manifest["content_hash"],
        }
    if order_attestation is None:
        raise ValueError("combination member data-order attestation is absent")
    native_rows = {}
    if graph.get("native_relation_auxiliary") is not None:
        expected_keys = {
            *{f"{training_role}:{replica}" for replica in range(4)},
            "val_stop:0",
            f"{evaluation_role}:0",
        }
        if (
            native_relation_target_files is None
            or set(native_relation_target_files) != expected_keys
        ):
            raise ValueError(
                "C_NATIVE_OFFLINE requires exact role/replica native targets"
            )
        for role_replica, raw_path in sorted(
            native_relation_target_files.items()
        ):
            role, raw_replica = role_replica.rsplit(":", 1)
            replica = int(raw_replica)
            path = Path(raw_path).resolve()
            from .native_relations import NATIVE_RELATION_TARGET_CONTRACT

            artifact = load_hashed_json(
                path.with_suffix(".manifest.json"),
                expected_contract=NATIVE_RELATION_TARGET_CONTRACT,
            )
            if (
                artifact.get("source") != dict(source)
                or artifact.get("npz_sha256") != _sha256_file(path)
            ):
                raise ValueError("native relation artifact lineage differs")
            native_rows.setdefault(role, {})[str(replica)] = {
                "path": str(path),
                "sha256": artifact["npz_sha256"],
                "artifact_sha256": artifact["content_hash"],
            }
    elif native_relation_target_files:
        raise ValueError("ordinary combinations cannot bind native relation targets")
    return with_content_hash(
        {
            "contract": COMBINATION_LOADER_MANIFEST_CONTRACT_V5,
            "schema_version": 5,
            "source": dict(source),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "graph_id": graph["graph_id"],
            "member_manifests": rows,
            "native_relation_target_files": native_rows,
            "evaluation_role": evaluation_role,
            "training_role": training_role,
            **order_attestation,
            "identity_join": "exact_before_batching",
        }
    )


def load_combination_loaders(
    *,
    manifest: Mapping[str, Any],
    graph: Mapping[str, Any],
    campaign_root: Path,
    campaign: Mapping[str, Any],
    target_registry: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        manifest.get("contract") != COMBINATION_LOADER_MANIFEST_CONTRACT_V5
        or manifest.get("source") != campaign["source"]
        or manifest.get("campaign_spec_sha256") != campaign["content_hash"]
        or manifest.get("graph_id") != graph["graph_id"]
    ):
        raise ValueError("combination loader lineage differs")
    members = {row["target_id"]: row for row in graph["members"]}
    evaluation_role = str(manifest.get("evaluation_role", "design_select"))
    if evaluation_role not in {"design_select", "design_confirm"}:
        raise ValueError("combination loader evaluation role differs")
    training_role = str(manifest.get("training_role", "model_train"))
    if training_role not in {"model_train", "scale_train"}:
        raise ValueError("combination loader training role differs")
    roles = (training_role, "val_stop", evaluation_role)
    order_attestation = {
        "pipeline_seed": int(manifest.get("pipeline_seed", -1)),
        "data_order_contract": manifest.get("data_order_contract"),
        "sampler_seed_by_role": manifest.get("sampler_seed_by_role"),
        "sampler_contract_by_role": manifest.get("sampler_contract_by_role"),
        "seed_derivation_contract_by_role": manifest.get(
            "seed_derivation_contract_by_role"
        ),
    }
    if (
        order_attestation["pipeline_seed"] < 0
        or set(order_attestation["sampler_seed_by_role"] or {}) != set(roles)
        or set(order_attestation["sampler_contract_by_role"] or {}) != set(roles)
        or set(order_attestation["seed_derivation_contract_by_role"] or {})
        != set(roles)
    ):
        raise ValueError("combination loader data-order attestation differs")
    loaded_by_target = {}
    lineage = {"combination_loader_manifest": manifest["content_hash"]}
    member_definitions = []
    for target, definition in manifest["member_manifests"].items():
        path = Path(definition["path"])
        if _sha256_file(path) != definition["sha256"]:
            raise ValueError("combination member loader bytes differ")
        member_manifest = load_hashed_json(path)
        member_order_attestation = {
            "pipeline_seed": int(member_manifest.get("pipeline_seed", -1)),
            "data_order_contract": member_manifest.get("data_order_contract"),
            "sampler_seed_by_role": member_manifest.get(
                "sampler_seed_by_role"
            ),
            "sampler_contract_by_role": member_manifest.get(
                "sampler_contract_by_role"
            ),
            "seed_derivation_contract_by_role": member_manifest.get(
                "seed_derivation_contract_by_role"
            ),
        }
        if (
            member_manifest.get("contract") != LOADER_MANIFEST_CONTRACT_V7
            or member_manifest.get("content_hash")
            != definition.get("artifact_sha256")
            or member_order_attestation != order_attestation
        ):
            raise ValueError("combination member data-order lineage differs")
        tree_role_count = sum(
            bool(role_definition.get("tree_caches"))
            for role_definition in member_manifest.get("roles", {}).values()
        )
        member_definitions.append(
            (0 if tree_role_count else 1, -tree_role_count, target, definition, path)
        )
    shared_bases_by_role = {}
    for _, _, target, definition, path in sorted(member_definitions):
        member = members[target]
        row = {
            **member,
            "row_id": member["selected_row_id"],
            "row_kind": "SCIENTIFIC",
            "resolved": True,
            "pipeline_seed": order_attestation["pipeline_seed"],
        }
        loaded = load_stage_d_loaders_from_manifest(
            manifest_path=path,
            campaign_root=campaign_root,
            row=row,
            campaign=campaign,
            target_registry=target_registry,
            shared_base_datasets_by_role=(
                shared_bases_by_role if shared_bases_by_role else None
            ),
        )
        loaded_by_target[target] = loaded
        for role, key in (
            (training_role, "train_loader"),
            ("val_stop", "val_stop_loader"),
            (evaluation_role, "evaluation_loader"),
        ):
            shared_bases_by_role.setdefault(
                role, loaded[key].dataset.base_dataset
            )
        lineage.update(
            {
                f"{target}__{key}": value
                for key, value in loaded["lineage_hashes"].items()
            }
        )
    result = {"lineage_hashes": lineage, "component_group_ids": {}}
    for role, key, training in (
        (training_role, "train_loader", True),
        ("val_stop", "val_stop_loader", False),
        (evaluation_role, "evaluation_loader", False),
    ):
        datasets = {
            target: loaded[key].dataset
            for target, loaded in loaded_by_target.items()
        }
        native_targets = None
        native_definition = manifest.get(
            "native_relation_target_files", {}
        ).get(role)
        if graph.get("native_relation_auxiliary") is not None:
            if native_definition is None:
                raise ValueError("native relation target role is absent")
            expected_replicas = (
                {0, 1, 2, 3} if role == training_role else {0}
            )
            if {int(key) for key in native_definition} != expected_replicas:
                raise ValueError("native relation role replica coverage differs")
            native_targets = {}
            for raw_replica, definition in native_definition.items():
                replica = int(raw_replica)
                native_path = Path(definition["path"])
                if (
                    _sha256_file(native_path)
                    != definition["sha256"]
                ):
                    raise ValueError("native relation target bytes differ")
                from .native_relations import NATIVE_RELATION_TARGET_CONTRACT

                artifact = load_hashed_json(
                    native_path.with_suffix(".manifest.json"),
                    expected_contract=NATIVE_RELATION_TARGET_CONTRACT,
                )
                if artifact.get("content_hash") != definition.get(
                    "artifact_sha256"
                ):
                    raise ValueError("native relation target artifact differs")
                store_definition = artifact.get("mmap_store")
                if (
                    artifact.get("storage_layout")
                    != "compressed_npz_plus_authenticated_npy_mmap_v1"
                    or not isinstance(store_definition, Mapping)
                ):
                    raise ValueError("native relation memory-map contract differs")
                store = native_path.parent / str(
                    store_definition.get("directory", "")
                )
                if store.is_symlink() or not store.is_dir():
                    raise ValueError("native relation memory-map store is unsafe")
                mapped = {}
                for name in (
                    "identities",
                    "targets",
                    "target_mask",
                    "availability",
                ):
                    member = store_definition.get("members", {}).get(name)
                    path = store / str(member.get("filename", ""))
                    if (
                        not isinstance(member, Mapping)
                        or path.parent != store
                        or path.is_symlink()
                        or not path.is_file()
                        or _sha256_file(path) != member.get("sha256")
                    ):
                        raise ValueError("native relation memory-map member differs")
                    value = np.load(path, mmap_mode="r", allow_pickle=False)
                    if (
                        list(value.shape) != member.get("shape")
                        or str(value.dtype) != member.get("dtype")
                    ):
                        raise ValueError("native relation memory-map metadata differs")
                    mapped[name] = value
                expected_dataset = next(iter(datasets.values()))
                expected_ids = expected_dataset.identities
                if (
                    mapped["identities"].shape != (len(expected_ids),)
                    or artifact.get("identity_order_sha256")
                    != getattr(
                        expected_dataset.base_dataset,
                        "identity_order_sha256",
                        None,
                    )
                ):
                    raise ValueError("native relation target identities differ")
                native_targets[replica] = {
                    name: mapped[name]
                    for name in ("targets", "target_mask", "availability")
                }
                lineage[
                    f"native_relation_targets_{role}_{replica}"
                ] = definition["sha256"]
        combination_dataset = CombinationDataset(
            datasets, native_relation_targets=native_targets
        )
        combination_dataset.pipeline_seed = order_attestation["pipeline_seed"]
        combination_dataset.data_order_contract = order_attestation[
            "data_order_contract"
        ]
        combination_dataset.sampler_contract_by_role = dict(
            order_attestation["sampler_contract_by_role"]
        )
        combination_dataset.sampler_seed_by_role = dict(
            order_attestation["sampler_seed_by_role"]
        )
        result[key] = make_combination_loader(
            combination_dataset,
            seed=int(order_attestation["sampler_seed_by_role"][role]),
            training=training,
            batch_size=next(iter(loaded_by_target.values()))[key].batch_size,
        )
        if role == "design_select":
            result["design_select_loader"] = result[key]
        elif role == "design_confirm":
            result["design_confirm_loader"] = result[key]
    result["evaluation_role"] = evaluation_role
    result["training_role"] = training_role
    result.update(order_attestation)
    for target, member in members.items():
        components = {
            row.target_id: row.components for row in target_declarations()
        }[target]
        result["component_group_ids"][target] = (
            target_component_availability_groups(target, components)
        )
    return result


class CombinationHBaseClassifier(
    torch.nn.Module if torch is not None else object
):
    def __init__(
        self,
        classifier: HBaseParticleTransformer,
        members: Sequence[Mapping[str, Any]],
        *,
        particle_dimension: int = 128,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for combination graphs")
        super().__init__()
        self.classifier = classifier
        declarations = {row.target_id: row for row in target_declarations()}
        self.members = {row["target_id"]: dict(row) for row in members}
        self.keys = {
            target: f"target_{index}"
            for index, target in enumerate(sorted(self.members))
        }
        self.heads = torch.nn.ModuleDict()
        for target, member in self.members.items():
            declaration = declarations[target]
            member["component_group_ids"] = target_component_availability_groups(
                target, declaration.components
            )
            availability_group_count = len(
                dict.fromkeys(member["component_group_ids"])
            )
            self.heads[self.keys[target]] = GlobalTargetHead(
                len(declaration.components),
                input_dimension=particle_dimension,
                availability_groups=availability_group_count,
                heteroscedastic=member["parameterization"] == "HET",
            )
        self.native_relation_head = (
            GlobalTargetHead(545, availability_groups=7)
            if any(
                bool(row.get("native_relation_auxiliary"))
                for row in members
            )
            else None
        )

    def shared_parameters(self) -> tuple[Any, ...]:
        return tuple(self.classifier.parameters())

    def head_parameters_by_target(self) -> dict[str, tuple[Any, ...]]:
        return {
            target: tuple(self.heads[key].parameters())
            for target, key in self.keys.items()
        }

    def forward_with_auxiliaries(
        self, points: Any, features: Any, vectors: Any, mask: Any
    ) -> tuple[Any, dict[str, Mapping[str, Any]]]:
        result = self.classifier.forward_with_taps(
            points, features, vectors, mask, capture=("TAP_LATE",)
        )
        state, active = result.states["TAP_LATE"], result.masks["TAP_LATE"]
        predictions = {
            target: self.heads[key](state, active)
            for target, key in self.keys.items()
        }
        if self.native_relation_head is not None:
            predictions["H_NATIVE_REL_AUX"] = self.native_relation_head(state, active)
        return result.logits, predictions

    def forward(self, points: Any, features: Any, vectors: Any, mask: Any) -> Any:
        return self.forward_with_auxiliaries(points, features, vectors, mask)[0]


def build_combination_model(
    graph: Mapping[str, Any],
    *,
    seed: int,
    weaver_module: Any | None = None,
    particle_dimension: int = 128,
) -> CombinationHBaseClassifier:
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(component_seed(seed, "encoder", "H_BASE"))
        classifier = HBaseParticleTransformer(weaver_module=weaver_module)
        torch.manual_seed(component_seed(seed, "combination_heads", graph["graph_id"]))
        members = [
            {
                **member,
                "native_relation_auxiliary": bool(
                    graph.get("native_relation_auxiliary")
                ),
            }
            for member in graph["members"]
        ]
        return CombinationHBaseClassifier(
            classifier, members, particle_dimension=particle_dimension
        )


def combination_losses(
    *,
    model: CombinationHBaseClassifier,
    batch: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    vectors = batch.get("lorentz_vectors", batch.get("vectors"))
    points = batch.get("points", batch["features"][:, 15:17])
    logits, predictions = model.forward_with_auxiliaries(
        points, batch["features"], vectors, batch["mask"]
    )
    classification = torch.nn.functional.cross_entropy(
        logits, batch["labels"].long()
    )
    auxiliaries = {}
    for member in graph["members"]:
        target_id = member["target_id"]
        target = batch["combination_targets"][target_id]
        loss, _ = global_auxiliary_loss(
            predictions[target_id],
            target["target"],
            target["target_mask"],
            parameterization=member["parameterization"],
            component_group_ids=model.members[target_id].get(
                "component_group_ids",
                tuple(
                    f"component_{index}"
                    for index in range(target["target"].shape[-1])
                ),
            ),
            target_id=target_id,
        )
        auxiliaries[target_id] = loss
    if graph.get("native_relation_auxiliary") is not None:
        native = batch.get("native_relation_target")
        if native is None or "H_NATIVE_REL_AUX" not in predictions:
            raise ValueError("C_NATIVE_OFFLINE batch lacks native relation targets")
        prediction = predictions["H_NATIVE_REL_AUX"]
        mask = native["target_mask"].bool()
        per_component = torch.nn.functional.huber_loss(
            prediction["value"],
            native["target"],
            delta=1.0,
            reduction="none",
        )
        count = mask.sum(dim=-1)
        applicable = count > 0
        value_loss = (
            (
                per_component.masked_fill(~mask, 0).sum(dim=-1)
                / count.clamp_min(1)
            )[applicable].mean()
            if bool(applicable.any())
            else prediction["value"].sum() * 0
        )
        availability_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction["availability_logits"], native["availability"]
        )
        auxiliaries["H_NATIVE_REL_AUX"] = value_loss + availability_loss
    total = classification + sum(
        float(graph["normalized_weights"][target]) * loss
        for target, loss in auxiliaries.items()
    )
    return total, logits, {
        "classification": classification,
        "auxiliaries": auxiliaries,
    }


def _atomic_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _evaluate(
    model: Any, loader: Any, graph: Mapping[str, Any], *, split: str, device: Any
) -> dict[str, Any]:
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            _, values, _ = combination_losses(model=model, batch=batch, graph=graph)
            logits.append(values.float().cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
    values, truth = np.concatenate(logits), np.concatenate(labels)
    return (
        evaluate_classification(values, truth, split=split)
        if split in {"design_select", "design_confirm"}
        else _balanced_metrics(values, truth)
    )


def train_combination(
    *,
    model: CombinationHBaseClassifier,
    train_loader: Any,
    val_stop_loader: Any,
    design_select_loader: Any,
    graph: Mapping[str, Any],
    output_dir: str | Path,
    stage_f_plan_sha256: str,
    campaign_spec_sha256: str,
    lineage_hashes: Mapping[str, str],
    protocol: HOSDTrainingProtocol,
    source: Mapping[str, Any],
    deployed_analytical_flops: float,
    deployed_parameter_count: int,
    device: str | Any = "cpu",
    resume: bool = True,
    evaluation_split: str = "design_select",
    checkpoint_contract: str = COMBINATION_CHECKPOINT_CONTRACT,
    completion_contract: str = COMBINATION_COMPLETION_CONTRACT,
    result_contract: str = COMBINATION_RESULT_CONTRACT,
    plan_hash_field: str = "stage_f_plan_sha256",
    completion_filename: str = "combination_completion.json",
) -> dict[str, Any]:
    protocol.validate()
    if evaluation_split not in {"design_select", "design_confirm"}:
        raise ValueError("combination evaluation split differs")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    completion_path = root / completion_filename
    best_path, last_path = root / "best_model_val.pt", root / "last.pt"
    checked_lineage = {
        key: require_sha256(value, name=f"lineage.{key}")
        for key, value in sorted(lineage_hashes.items())
    }
    training_role = str(train_loader.dataset.logical_role)
    data_order_attestation = {
        "pipeline_seed": int(train_loader.dataset.pipeline_seed),
        "data_order_contract": str(train_loader.dataset.data_order_contract),
        "training_sampler_seed": int(
            train_loader.dataset.sampler_seed_by_role[training_role]
        ),
        "training_sampler_contract": str(
            train_loader.dataset.sampler_contract_by_role[training_role]
        ),
    }
    if (
        int(getattr(train_loader.sampler, "seed", -1))
        != data_order_attestation["training_sampler_seed"]
        or str(getattr(train_loader.sampler, "contract", ""))
        != data_order_attestation["training_sampler_contract"]
    ):
        raise ValueError("combination training sampler differs from manifest")
    resolved = torch.device(device)
    model.to(resolved)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=protocol.base_learning_rate,
        betas=(protocol.beta1, protocol.beta2),
        weight_decay=protocol.weight_decay,
    )
    counts = optimizer_update_counts(
        training_event_count=len(train_loader.dataset),
        maximum_epochs=protocol.maximum_epochs,
        microbatch_size=protocol.microbatch_size,
        gradient_accumulation_steps=protocol.gradient_accumulation_steps,
    )
    rows, states, start_epoch, update = [], {}, 1, 0
    started, elapsed_before = time.perf_counter(), 0.0
    if resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if any(
            state.get(key) != value
            for key, value in {
                "contract": checkpoint_contract,
                "graph_id": graph["graph_id"],
                plan_hash_field: stage_f_plan_sha256,
                "campaign_spec_sha256": campaign_spec_sha256,
                "source": dict(source),
                "lineage_hashes": checked_lineage,
                **data_order_attestation,
            }.items()
        ):
            raise ValueError("combination resume lineage differs")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        rows, states = list(state["rows"]), dict(state["candidate_states"])
        start_epoch, update = int(state["epoch_completed"]) + 1, int(
            state["optimizer_update_ordinal"]
        )
        elapsed_before = float(state["elapsed_training_seconds"])
        _restore_rng_state(state["rng_state"])
    for epoch in range(start_epoch, protocol.maximum_epochs + 1):
        train_loader.dataset.set_epoch(epoch)
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        event_sum, objective_sum, accumulated = 0, 0.0, 0
        for batch_index, raw in enumerate(train_loader, 1):
            batch = _move(raw, resolved)
            total, _, pieces = combination_losses(
                model=model, batch=batch, graph=graph
            )
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("combination objective is nonfinite")
            events = int(batch["labels"].numel())
            if graph["weighting"] == "W_PCGRAD":
                tasks = [
                    pieces["classification"],
                    *[
                        float(graph["normalized_weights"][target]) * loss
                        for target, loss in pieces["auxiliaries"].items()
                    ],
                ]
                shared = model.shared_parameters()
                accumulated_shared = [
                    None if parameter.grad is None else parameter.grad.clone()
                    for parameter in shared
                ]
                task_gradients = []
                for task in tasks:
                    gradients = torch.autograd.grad(
                        task, shared, retain_graph=True, allow_unused=True
                    )
                    task_gradients.append(
                        [
                            torch.zeros_like(parameter)
                            if gradient is None
                            else gradient
                            for parameter, gradient in zip(shared, gradients)
                        ]
                    )
                projected = pcgrad_project(
                    task_gradients, update_ordinal=update + 1
                )
                (total * events).backward()
                for parameter, previous, gradient in zip(
                    shared, accumulated_shared, projected
                ):
                    parameter.grad = gradient * events + (
                        0 if previous is None else previous
                    )
            else:
                (total * events).backward()
            event_sum += events
            objective_sum += float(total.detach().cpu()) * events
            accumulated += events
            if (
                batch_index % protocol.gradient_accumulation_steps
                and batch_index != len(train_loader)
            ):
                continue
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), protocol.gradient_clip_norm
            )
            update += 1
            learning_rate = scheduled_learning_rate(
                update_ordinal=update,
                total_optimizer_updates=counts["total_optimizer_updates"],
                warmup_updates=counts["warmup_updates"],
                base_learning_rate=protocol.base_learning_rate,
                minimum_learning_rate=protocol.minimum_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
        validation = _evaluate(
            model, val_stop_loader, graph, split="val_stop", device=resolved
        )
        rows.append(
            {
                "epoch": epoch,
                "optimizer_update_ordinal": update,
                "train_objective": objective_sum / event_sum,
                "val_stop": validation,
            }
        )
        state_dict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        states[epoch] = state_dict
        selected = min(
            rows,
            key=lambda row: (
                -row["val_stop"]["balanced_accuracy"],
                row["val_stop"]["cross_entropy"],
                row["epoch"],
            ),
        )
        selected_epoch = int(selected["epoch"])
        states = {
            key: value
            for key, value in states.items()
            if int(key) in {selected_epoch, epoch}
        }
        common = {
            "contract": checkpoint_contract,
            "schema_version": 1,
            "graph_id": graph["graph_id"],
            "source": dict(source),
            "lineage_hashes": checked_lineage,
            plan_hash_field: stage_f_plan_sha256,
            "campaign_spec_sha256": campaign_spec_sha256,
            **data_order_attestation,
        }
        _atomic_save(
            {
                **common,
                "kind": "selected_inference",
                "epoch": selected_epoch,
                "model_state_dict": states[selected_epoch],
            },
            best_path,
        )
        _atomic_save(
            {
                **common,
                "kind": "resumable_last",
                "epoch_completed": epoch,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "optimizer_update_ordinal": update,
                "rows": rows,
                "candidate_states": states,
                "rng_state": _rng_state(),
                "elapsed_training_seconds": elapsed_before
                + time.perf_counter()
                - started,
            },
            last_path,
        )
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    evaluation = _evaluate(
        model,
        design_select_loader,
        graph,
        split=evaluation_split,
        device=resolved,
    )
    checkpoint_sha = _sha256_file(best_path)
    result = with_content_hash(
        {
            "contract": result_contract,
            "schema_version": 1,
            "source": dict(source),
            **dict(graph),
            plan_hash_field: stage_f_plan_sha256,
            "campaign_spec_sha256": campaign_spec_sha256,
            "checkpoint_sha256": checkpoint_sha,
            evaluation_split: {"classification_metrics": evaluation},
            "deployed_analytical_flops": float(deployed_analytical_flops),
            "deployed_parameter_count": int(deployed_parameter_count),
            **data_order_attestation,
            "training_gpu_hours": (
                elapsed_before + time.perf_counter() - started
            )
            / 3600,
        }
    )
    write_immutable_json(root / f"{evaluation_split}_result.json", result)
    completion = with_content_hash(
        {
            "contract": completion_contract,
            "schema_version": 1,
            "source": dict(source),
            "graph_id": graph["graph_id"],
            plan_hash_field: stage_f_plan_sha256,
            "campaign_spec_sha256": campaign_spec_sha256,
            "lineage_hashes": checked_lineage,
            **data_order_attestation,
            "checkpoint_sha256": checkpoint_sha,
            "result_sha256": result["content_hash"],
            "epochs_completed": len(rows),
            "optimizer_updates_completed": update,
            "performance_based_termination": False,
        }
    )
    write_immutable_json(completion_path, completion)
    if last_path.exists():
        last_path.unlink()
    return completion


__all__ = [
    "COMBINATION_CHECKPOINT_CONTRACT",
    "COMBINATION_COMPLETION_CONTRACT",
    "COMBINATION_LOADER_MANIFEST_CONTRACT",
    "COMBINATION_LOADER_MANIFEST_CONTRACT_V2",
    "COMBINATION_LOADER_MANIFEST_CONTRACT_V3",
    "COMBINATION_LOADER_MANIFEST_CONTRACT_V4",
    "COMBINATION_LOADER_MANIFEST_CONTRACT_V5",
    "CombinationDataset",
    "CombinationHBaseClassifier",
    "build_combination_loader_manifest",
    "build_combination_model",
    "collate_combination_batch",
    "load_combination_loaders",
    "make_combination_loader",
    "train_combination",
]
