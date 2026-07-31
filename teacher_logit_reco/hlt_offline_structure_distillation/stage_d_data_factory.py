"""Manifest-driven, source-bound Stage-D loader construction."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (
    load_hlt_v3_cache,
)
from .input_views import load_materialized_hlt_input_view
from teacher_logit_reco.relational_part.ca_tree import unpack_tree_shard

from .auxiliary_data import (
    AuxiliaryTargetDataset,
    HLTArrayDataset,
    StreamedHLTTarget,
    make_auxiliary_loader,
)
from .contracts import (
    load_hashed_json,
    require_sha256,
    with_content_hash,
)
from .extractors import ExtractorResources
from .normalization import normalize_target, whiten_latents
from .target_cache import load_target_cache
from .target_cache import identity_order_sha256
from .target_schemas import target_component_availability_groups, target_declarations


LOADER_MANIFEST_CONTRACT = "hosd_stage_d_loader_manifest_v1"
LOADER_MANIFEST_CONTRACT_V2 = "hosd_stage_d_loader_manifest_v2"
LOADER_MANIFEST_CONTRACT_V3 = "hosd_stage_d_loader_manifest_v3"
ROLES = ("model_train", "val_stop", "design_select")


def build_default_stage_d_role_definitions(
    *,
    row: Mapping[str, Any],
    campaign_root: str | Path,
    base_role_definitions: Mapping[str, Mapping[str, Any]],
    evaluation_role: str = "design_select",
    training_role: str = "model_train",
) -> dict[str, dict[str, Any]]:
    """Resolve row-specific target paths from fixed HLT/label infrastructure."""

    root = Path(campaign_root)
    roles = (training_role, "val_stop", evaluation_role)
    if set(base_role_definitions) != set(roles):
        raise ValueError("default Stage-D base-role coverage differs")
    split_by_role = {
        "model_train": "model_train",
        "scale_train": "scale_train",
        "val_stop": "val_stop",
        "design_select": "val_design",
        "design_confirm": "val_design",
    }
    pair = row["target_id"] in {
        "T_HLT_TRACK_PAIR_13",
        "T_HLT_REGION_PAIR_8",
    }
    control_kind = {
        "TARGET_MEAN": "target_mean",
        "GLOBAL_SHUFFLE": "global_shuffle",
        "WITHIN_CLASS_SHUFFLE": "within_class_shuffle",
    }.get(row["row_kind"])
    output = {}
    for role in roles:
        base = dict(base_role_definitions[role])
        if not {"labels", "hlt_caches"}.issubset(base):
            raise ValueError("default Stage-D base role is incomplete")
        split = split_by_role[role]
        if pair:
            target = {
                "mode": "stream_same_view",
                "normalizer": str(
                    root
                    / "normalization"
                    / (
                        "target_scale"
                        if training_role == "scale_train"
                        else "target_500k"
                    )
                    / "normalizer_manifest.json"
                ),
                "relation_normalizer": str(
                    root
                    / "normalization"
                    / (
                        "relation_scale"
                        if training_role == "scale_train"
                        else "relation_500k"
                    )
                    / "normalizer.json"
                ),
                "control_kind": control_kind,
            }
            if row["row_kind"] in {
                "GLOBAL_SHUFFLE",
                "WITHIN_CLASS_SHUFFLE",
            }:
                target["shuffle_plan"] = str(
                    root
                    / "targets"
                    / "controls"
                    / "plans"
                    / split
                    / {
                        "GLOBAL_SHUFFLE": "global",
                        "WITHIN_CLASS_SHUFFLE": "within_class",
                    }[row["row_kind"]]
                    / f"{row['target_id']}.json"
                )
        elif row["row_kind"] in {
            "TARGET_MEAN",
            "GLOBAL_SHUFFLE",
            "WITHIN_CLASS_SHUFFLE",
        }:
            if row["target_id"] == "T_OFFLINE_POOLED_LATENT":
                control_source = "O_BASE"
            elif row["target_id"].startswith("T_OFFLINE_LOGITS_"):
                control_source = row["target_id"].removeprefix(
                    "T_OFFLINE_LOGITS_"
                )
            else:
                control_source = "physical"
            target = {
                "mode": "static_cache",
                "caches": {
                    "shared": str(
                        root
                        / "targets"
                        / "controls"
                        / control_kind
                        / split
                        / control_source
                    )
                },
                "normalizer": str(
                    root
                    / "normalization"
                    / "target_500k"
                    / "normalizer_manifest.json"
                ),
                "control_kind": control_kind,
            }
        elif row["parameterization"] == "RES":
            target = {
                "mode": "static_cache",
                "caches": {
                    str(replica): str(
                        root
                        / "targets"
                        / "residuals"
                        / split
                        / f"replica_{replica}"
                    )
                    for replica in sorted(int(key) for key in base["hlt_caches"])
                },
                "normalizer": str(
                    root
                    / "normalization"
                    / (
                        "residual_scale"
                        if training_role == "scale_train"
                        else "residual_500k"
                    )
                    / "normalizer_manifest.json"
                ),
                "control_kind": None,
            }
        elif row["parameterization"] == "KD":
            teacher_id = row["target_id"].removeprefix(
                "T_OFFLINE_LOGITS_"
            )
            target = {
                "mode": "static_cache",
                "caches": {
                    "shared": str(
                        root
                        / "teachers"
                        / "outputs"
                        / split
                        / teacher_id
                    )
                },
                "control_kind": None,
            }
        elif row["parameterization"] == "WHITENED_ABS":
            target = {
                "mode": "static_cache",
                "caches": {
                    "shared": str(
                        root
                        / "teachers"
                        / "outputs"
                        / split
                        / "O_BASE"
                    )
                },
                "whitening": str(
                    root
                    / "normalization"
                    / (
                        "target_scale"
                        if training_role == "scale_train"
                        else "target_500k"
                    )
                    / "latent_whitening.json"
                ),
                "control_kind": None,
            }
        elif row["row_kind"] == "HLT_SELF":
            coordinate = row.get("source_target_id") or row[
                "target_id"
            ].replace("T_HLT_SELF_", "T_OFFLINE_", 1)
            target = {
                "mode": "static_cache",
                "caches": {
                    str(replica): str(
                        root
                        / "targets"
                        / "hlt_analogues"
                        / split
                        / f"replica_{replica}"
                    )
                    for replica in sorted(int(key) for key in base["hlt_caches"])
                },
                "coordinate_id": coordinate,
                "normalizer": str(
                    root
                    / "normalization"
                    / "target_500k"
                    / "normalizer_manifest.json"
                ),
                "control_kind": None,
            }
        else:
            target = {
                "mode": "static_cache",
                "caches": {
                    "shared": str(
                        root / "targets" / "canonical" / split
                    )
                },
                "normalizer": str(
                    root
                    / "normalization"
                    / (
                        "target_scale"
                        if training_role == "scale_train"
                        else "target_500k"
                    )
                    / "normalizer_manifest.json"
                ),
                "control_kind": None,
            }
        output[role] = {**base, "target": target}
    return output


def build_stage_d_loader_manifest(
    *,
    row: Mapping[str, Any],
    role_definitions: Mapping[str, Mapping[str, Any]],
    campaign_spec_sha256: str,
    source: Mapping[str, Any],
    evaluation_role: str = "design_select",
    training_role: str = "model_train",
) -> dict[str, Any]:
    if evaluation_role not in {"design_select", "design_confirm"}:
        raise ValueError("Stage-D evaluation role differs")
    if training_role not in {"model_train", "scale_train"}:
        raise ValueError("Stage-D training role differs")
    roles = (training_role, "val_stop", evaluation_role)
    if not bool(row.get("resolved")) or set(role_definitions) != set(roles):
        raise ValueError("Stage-D loader manifest row/role coverage differs")
    checked = {}
    for role in roles:
        definition = dict(role_definitions[role])
        if not {"labels", "hlt_caches", "target"}.issubset(definition):
            raise ValueError(f"Stage-D {role} loader definition is incomplete")
        target = dict(definition["target"])
        if target.get("mode") not in {"static_cache", "stream_same_view"}:
            raise ValueError("Stage-D target source mode differs")
        if target["mode"] == "static_cache" and not target.get("caches"):
            raise ValueError("static Stage-D target source lacks caches")
        if target["mode"] == "stream_same_view" and not {
            "normalizer",
            "relation_normalizer",
        }.issubset(target):
            raise ValueError("streamed Stage-D target source lacks normalizers")
        expected_control = {
            "TARGET_MEAN": "target_mean",
            "GLOBAL_SHUFFLE": "global_shuffle",
            "WITHIN_CLASS_SHUFFLE": "within_class_shuffle",
        }.get(row["row_kind"])
        if target.get("control_kind") != expected_control:
            raise ValueError("Stage-D target source control semantics differ")
        if (
            row["row_kind"] in {"GLOBAL_SHUFFLE", "WITHIN_CLASS_SHUFFLE"}
            and target["mode"] == "stream_same_view"
            and "shuffle_plan" not in target
        ):
            raise ValueError("streamed Stage-D shuffle lacks its plan")
        definition["target"] = target
        checked[role] = definition
    return with_content_hash(
        {
            "contract": (
                LOADER_MANIFEST_CONTRACT
                if (
                    evaluation_role == "design_select"
                    and training_role == "model_train"
                )
                else LOADER_MANIFEST_CONTRACT_V2
                if training_role == "model_train"
                else LOADER_MANIFEST_CONTRACT_V3
            ),
            "schema_version": (
                1
                if (
                    evaluation_role == "design_select"
                    and training_role == "model_train"
                )
                else 2
                if training_role == "model_train"
                else 3
            ),
            "source": dict(source),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "row_id": row["row_id"],
            "target_id": row["target_id"],
            "parameterization": row["parameterization"],
            "roles": checked,
            "evaluation_role": evaluation_role,
            "training_role": training_role,
            "identity_join": "authenticated_cache_identity_to_role_label_identity",
            "dense_pair_targets_persisted": False,
            "performance_dependent_inputs": False,
        }
    )


def _file_sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _labels(path: Path) -> tuple[tuple[str, ...], np.ndarray, str]:
    with np.load(path, allow_pickle=False) as payload:
        if not {"identities", "labels"}.issubset(payload.files):
            raise ValueError("Stage-D labels lack identities/labels")
        identities = tuple(str(value) for value in payload["identities"].tolist())
        labels = np.asarray(payload["labels"], dtype=np.int64)
    if (
        not identities
        or len(identities) != len(set(identities))
        or labels.shape != (len(identities),)
        or bool(((labels < 0) | (labels >= 10)).any())
    ):
        raise ValueError("Stage-D label population differs")
    return identities, labels, _file_sha(path)


def _subset_indices(
    source_identities: tuple[str, ...], requested: tuple[str, ...]
) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(source_identities)}
    if len(lookup) != len(source_identities) or not set(requested).issubset(lookup):
        raise ValueError("Stage-D identity join lacks exact source coverage")
    return np.asarray([lookup[value] for value in requested], dtype=np.int64)


def _load_hlt(
    caches: Mapping[str, Any],
    identities: tuple[str, ...],
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, str], str, str]:
    arrays_by_replica = {}
    lineage = {}
    logical_roles, policies = set(), set()
    for raw_replica, raw_path in sorted(caches.items(), key=lambda item: int(item[0])):
        replica, path = int(raw_replica), Path(raw_path)
        arrays, metadata = (
            load_materialized_hlt_input_view(path)
            if path.is_file()
            else load_hlt_v3_cache(path)
        )
        source_ids = tuple(str(value) for value in arrays["identities"].tolist())
        order = _subset_indices(source_ids, identities)
        arrays_by_replica[replica] = {
            name: np.asarray(arrays[name])[order]
            for name in ("tokens", "mask", "measurement_states")
        }
        lineage[f"hlt_replica_{replica}"] = metadata["content_hash"]
        logical_roles.add(str(metadata["logical_role"]))
        policies.add(str(metadata["realization_policy"]))
    if len(logical_roles) != 1 or len(policies) != 1:
        raise ValueError("Stage-D HLT cache roles/policies differ")
    return (
        arrays_by_replica,
        lineage,
        next(iter(logical_roles)),
        next(iter(policies)),
    )


def _trees(
    paths: Mapping[str, Any],
    identities: tuple[str, ...],
) -> tuple[dict[int, tuple[Mapping[str, Any], ...]], dict[str, str]]:
    output, lineage = {}, {}
    for raw_replica, raw_path in sorted(paths.items(), key=lambda item: int(item[0])):
        replica, root = int(raw_replica), Path(raw_path)
        manifest = load_hashed_json(root / "manifest.json")
        by_identity = {}
        for shard in sorted((root / "shards").glob("shard_*.npz")):
            shard_ids, rows = unpack_tree_shard(shard)
            for identity, tree in zip(shard_ids, rows):
                if identity in by_identity:
                    raise ValueError("Stage-D tree cache duplicates an identity")
                by_identity[identity] = tree
        if not set(identities).issubset(by_identity):
            raise ValueError("Stage-D tree cache lacks role identities")
        output[replica] = tuple(by_identity[value] for value in identities)
        lineage[f"tree_replica_{replica}"] = manifest["content_hash"]
    return output, lineage


def _load_target_cache(path: Path):
    spec = load_hashed_json(
        path / "cache_spec.json", expected_contract="hosd_target_cache_spec_v1"
    )
    return load_target_cache(path, cache_spec=spec)


def _static_targets(
    *,
    definition: Mapping[str, Any],
    identities: tuple[str, ...],
    row: Mapping[str, Any],
    source: Mapping[str, Any],
) -> tuple[np.ndarray | dict[int, np.ndarray], np.ndarray | dict[int, np.ndarray], dict[str, str]]:
    cache_paths = definition.get("caches")
    if not isinstance(cache_paths, Mapping) or not cache_paths:
        raise ValueError("static Stage-D target source has no caches")
    normalizer = None
    if row["parameterization"] not in {"KD", "WHITENED_ABS"}:
        normalizer = load_hashed_json(
            Path(definition["normalizer"]),
            expected_contract="hosd_target_normalizer_v1",
        )
        if normalizer.get("source") != dict(source):
            raise ValueError("Stage-D target normalizer source differs")
    whitening = None
    if row["parameterization"] == "WHITENED_ABS":
        whitening = load_hashed_json(
            Path(definition["whitening"]),
            expected_contract="hosd_latent_whitening_v1",
        )
        if whitening.get("source") != dict(source):
            raise ValueError("Stage-D whitening source differs")
    values, masks, lineage = {}, {}, {}
    for key, raw_path in sorted(cache_paths.items()):
        cache_root = Path(raw_path)
        cache = _load_target_cache(cache_root)
        if cache.manifest.get("source") != dict(source):
            raise ValueError("Stage-D target cache source differs")
        expected_control = {
            "TARGET_MEAN": "target_mean",
            "GLOBAL_SHUFFLE": "global_shuffle",
            "WITHIN_CLASS_SHUFFLE": "within_class_shuffle",
        }.get(row["row_kind"])
        if expected_control is not None:
            if cache.manifest.get("artifact_kind") != "control":
                raise ValueError("Stage-D null intervention did not use a control cache")
            control = load_hashed_json(
                cache_root / "control_manifest.json",
                expected_contract="hosd_target_control_manifest_v1",
            )
            if (
                control.get("source") != dict(source)
                or control.get("control_kind") != expected_control
                or control.get("control_cache_manifest_sha256")
                != cache.manifest["content_hash"]
            ):
                raise ValueError("Stage-D control-cache lineage differs")
            lineage[f"control_manifest_{key}"] = control["content_hash"]
        order = _subset_indices(cache.identities, identities)
        coordinate_id = definition.get("coordinate_id")
        if coordinate_id is None:
            if row["parameterization"] == "RES":
                candidates = [
                    target_id
                    for target_id in cache.manifest["persisted_target_ids"]
                    if target_id.startswith(f"{row['target_id']}__RES__")
                ]
                if len(candidates) != 1:
                    raise ValueError("Stage-D residual coordinate is not unique")
                coordinate_id = candidates[0]
            else:
                coordinate_id = row["target_id"]
        coordinate_id = str(coordinate_id)
        if coordinate_id not in cache.values:
            raise ValueError("Stage-D target coordinate is absent from cache")
        raw = cache.values[coordinate_id][order]
        mask = cache.masks[coordinate_id][order]
        if row["parameterization"] == "KD":
            transformed = raw.astype(np.float32, copy=True)
        elif row["parameterization"] == "WHITENED_ABS":
            transformed = whiten_latents(raw, whitening=whitening)
            transformed[~mask] = 0
        else:
            transformed = normalize_target(
                raw,
                mask,
                target_id=coordinate_id,
                normalizer=normalizer,
            )
        target_key = -1 if str(key) == "shared" else int(key)
        values[target_key] = transformed
        masks[target_key] = mask
        lineage[f"target_cache_{key}"] = cache.manifest["content_hash"]
    if normalizer is not None:
        lineage["target_normalizer"] = normalizer["content_hash"]
    if whitening is not None:
        lineage["latent_whitening"] = whitening["content_hash"]
    if set(values) == {-1}:
        return values[-1], masks[-1], lineage
    return values, masks, lineage


def load_stage_d_loaders_from_manifest(
    *,
    manifest_path: str | Path,
    campaign_root: Path,
    row: Mapping[str, Any],
    campaign: Mapping[str, Any],
    target_registry: Mapping[str, Any],
) -> dict[str, Any]:
    del campaign_root, target_registry
    manifest = load_hashed_json(Path(manifest_path))
    if manifest.get("contract") not in {
        LOADER_MANIFEST_CONTRACT,
        LOADER_MANIFEST_CONTRACT_V2,
        LOADER_MANIFEST_CONTRACT_V3,
    }:
        raise ValueError("Stage-D loader manifest contract differs")
    evaluation_role = str(manifest.get("evaluation_role", "design_select"))
    if evaluation_role not in {"design_select", "design_confirm"}:
        raise ValueError("Stage-D loader evaluation role differs")
    training_role = str(manifest.get("training_role", "model_train"))
    if training_role not in {"model_train", "scale_train"}:
        raise ValueError("Stage-D loader training role differs")
    roles = (training_role, "val_stop", evaluation_role)
    if (
        manifest.get("source") != campaign["source"]
        or manifest.get("campaign_spec_sha256") != campaign["content_hash"]
        or manifest.get("row_id") != row["row_id"]
    ):
        raise ValueError("Stage-D loader manifest lineage differs")
    if set(manifest.get("roles", {})) != set(roles):
        raise ValueError("Stage-D loader manifest role coverage differs")
    declarations = {item.target_id: item for item in target_declarations()}
    declaration = declarations[row["target_id"]]
    loaders, lineage = {}, {
        "loader_manifest": manifest["content_hash"],
        "campaign_spec": campaign["content_hash"],
    }
    for role in roles:
        definition = manifest["roles"][role]
        identities, labels, label_sha = _labels(Path(definition["labels"]))
        hlt_arrays, hlt_lineage, logical_role, realization_policy = _load_hlt(
            definition["hlt_caches"], identities
        )
        expected_hlt_roles = {
            "model_train": {"model_train"},
            "scale_train": {"scale_train"},
            "val_stop": {"val_stop"},
            "design_select": {"design_select", "val_design"},
            "design_confirm": {"design_confirm", "val_design"},
        }[role]
        if logical_role not in expected_hlt_roles:
            raise ValueError("Stage-D HLT cache logical role differs")
        lineage.update({f"{role}_{key}": value for key, value in hlt_lineage.items()})
        lineage[f"{role}_labels"] = label_sha
        trees = None
        if definition.get("tree_caches"):
            trees, tree_lineage = _trees(definition["tree_caches"], identities)
            lineage.update(
                {f"{role}_{key}": value for key, value in tree_lineage.items()}
            )
        base = HLTArrayDataset(
            replica_arrays=hlt_arrays,
            labels=labels,
            identities=identities,
            logical_role=logical_role,
            realization_policy=realization_policy,
            region_trees_by_replica=trees,
        )
        target_definition = definition["target"]
        mode = str(target_definition["mode"])
        control_kind = target_definition.get("control_kind")
        if mode == "static_cache":
            values, masks, target_lineage = _static_targets(
                definition=target_definition,
                identities=identities,
                row=row,
                source=campaign["source"],
            )
            dataset = AuxiliaryTargetDataset(
                base,
                target_id=row["target_id"],
                target_values=values,
                target_masks=masks,
                control_kind=control_kind,
                target_parent_hashes=target_lineage,
            )
        elif mode == "stream_same_view":
            normalizer = load_hashed_json(Path(target_definition["normalizer"]))
            from .normalization import validate_target_normalizer
            validate_target_normalizer(normalizer)
            relation = load_hashed_json(Path(target_definition["relation_normalizer"]))
            if (
                normalizer.get("source") != campaign["source"]
                or relation.get("source") != campaign["source"]
            ):
                raise ValueError("Stage-D streamed normalizer source differs")
            floors = relation.get("track_uncertainty_floors", {})
            stream = StreamedHLTTarget(
                target_id=row["target_id"],
                normalizer=normalizer,
                resources=ExtractorResources(
                    d0_uncertainty_floor=float(floors.get("d0", {}).get("floor", 0)),
                    dz_uncertainty_floor=float(floors.get("dz", {}).get("floor", 0)),
                    sentinel_policy=relation.get("track_sentinel_policy"),
                ),
                control_kind=(
                    "target_mean"
                    if row["row_kind"] == "TARGET_MEAN"
                    else None
                ),
            )
            stream_lineage = {
                "target_normalizer": normalizer["content_hash"],
                "relation_normalizer": relation["content_hash"],
            }
            donor_indices = None
            if row["row_kind"] in {"GLOBAL_SHUFFLE", "WITHIN_CLASS_SHUFFLE"}:
                shuffle = load_hashed_json(
                    Path(target_definition["shuffle_plan"]),
                    expected_contract="hosd_target_shuffle_plan_v1",
                )
                expected_kind = {
                    "GLOBAL_SHUFFLE": "global",
                    "WITHIN_CLASS_SHUFFLE": "within_class",
                }[row["row_kind"]]
                if (
                    shuffle.get("source") != campaign["source"]
                    or shuffle.get("target_id") != row["target_id"]
                    or shuffle.get("shuffle_kind") != expected_kind
                    or shuffle.get("canonical_identity_order_sha256")
                    != identity_order_sha256(identities)
                ):
                    raise ValueError("Stage-D streamed shuffle plan lineage differs")
                donor_indices = shuffle["mapping_recipient_to_donor"]
                stream_lineage["target_shuffle_plan"] = shuffle["content_hash"]
            dataset = AuxiliaryTargetDataset(
                base,
                target_id=row["target_id"],
                streamed_target=stream,
                control_kind=control_kind,
                target_parent_hashes=stream_lineage,
                donor_indices=donor_indices,
            )
            target_lineage = stream_lineage
        else:
            raise ValueError("unknown Stage-D target source mode")
        lineage.update(
            {f"{role}_{key}": value for key, value in target_lineage.items()}
        )
        loaders[role] = make_auxiliary_loader(
            dataset,
            seed=int(row["head_component_seed"]),
            training=role == training_role,
            batch_size=64,
        )
    for name, value in lineage.items():
        require_sha256(value, name=f"stage_d_lineage.{name}")
    return {
        "train_loader": loaders[training_role],
        "training_role": training_role,
        "val_stop_loader": loaders["val_stop"],
        "evaluation_loader": loaders[evaluation_role],
        "evaluation_role": evaluation_role,
        **(
            {"design_select_loader": loaders["design_select"]}
            if evaluation_role == "design_select"
            else {"design_confirm_loader": loaders["design_confirm"]}
        ),
        "component_group_ids": target_component_availability_groups(
            row["target_id"], declaration.components
        ),
        "lineage_hashes": lineage,
    }


__all__ = [
    "LOADER_MANIFEST_CONTRACT",
    "build_stage_d_loader_manifest",
    "build_default_stage_d_role_definitions",
    "load_stage_d_loaders_from_manifest",
    "LOADER_MANIFEST_CONTRACT_V2",
    "LOADER_MANIFEST_CONTRACT_V3",
]
