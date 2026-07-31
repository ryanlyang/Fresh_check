"""Manifest-driven, source-bound Stage-D loader construction."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (
    load_hlt_v3_cache,
)
from .input_views import load_materialized_hlt_input_view
from .authenticated_tree import AuthenticatedTreeSplit

from .auxiliary_data import (
    AuxiliaryTargetDataset,
    HLTArrayDataset,
    SCALE_SHARD_SAMPLER_CONTRACT,
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
from .target_cache import load_target_cache, load_target_cache_sharded
from .target_cache import identity_order_sha256
from .target_schemas import target_component_availability_groups, target_declarations


LOADER_MANIFEST_CONTRACT = "hosd_stage_d_loader_manifest_v1"
LOADER_MANIFEST_CONTRACT_V2 = "hosd_stage_d_loader_manifest_v2"
LOADER_MANIFEST_CONTRACT_V3 = "hosd_stage_d_loader_manifest_v3"
LOADER_MANIFEST_CONTRACT_V4 = "hosd_stage_d_loader_manifest_v4"
LOADER_MANIFEST_CONTRACT_V5 = "hosd_stage_d_loader_manifest_v5"
DATA_ORDER_CONTRACT = "hosd_data_order_v2"
ROLES = ("model_train", "val_stop", "design_select")


def data_order_seed(pipeline_seed: int, role: str) -> int:
    """Graph-independent sampler seed for one logical population role."""

    if str(role) not in {
        "model_train",
        "scale_train",
        "val_stop",
        "design_select",
        "design_confirm",
    }:
        raise ValueError("data-order role differs")
    payload = f"{DATA_ORDER_CONTRACT}\0{int(pipeline_seed)}\0{str(role)}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def sampler_contract(role: str) -> str:
    if role == "scale_train":
        return SCALE_SHARD_SAMPLER_CONTRACT
    if role == "model_train":
        return "retb_deterministic_full_permutation_sampler_v1"
    if role in {"val_stop", "design_select", "design_confirm"}:
        return "torch_sequential_sampler_v1"
    raise ValueError("data-order sampler role differs")


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
        if base.get("tree_caches"):
            tree_resource = load_hashed_json(
                root / "inputs" / "inherited_angular_tree_resource.json"
            )
            tree_backend = load_hashed_json(
                root / "inputs" / "region_tree" / "backend_manifest.json"
            )
            expected_parents = {}
            for raw_replica, raw_path in base["hlt_caches"].items():
                path = Path(raw_path)
                _, metadata = (
                    load_materialized_hlt_input_view(path)
                    if path.is_file()
                    else load_hlt_v3_cache(path)
                )
                expected_parents[str(int(raw_replica))] = {
                    "hlt_content_sha256": metadata["array_content_sha256"],
                    "tree_resource_sha256": tree_resource["content_hash"],
                    "backend_manifest_sha256": tree_backend["content_hash"],
                }
            base["tree_expected_parents"] = expected_parents
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
            "contract": LOADER_MANIFEST_CONTRACT_V5,
            "schema_version": 5,
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
            "data_order_contract": DATA_ORDER_CONTRACT,
            "sampler_contract_by_role": {
                role: sampler_contract(role) for role in roles
            },
            "sampler_seed_by_role": {
                role: data_order_seed(int(row["pipeline_seed"]), role)
                for role in roles
            },
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


class _AuthenticatedIdentitySequence(Sequence[str]):
    def __init__(self, values: Sequence[str], identity_hash: str) -> None:
        self.values = values
        self.identity_order_sha256 = require_sha256(
            identity_hash, name="identity_order_sha256"
        )

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


def _scale_labels(
    path: Path,
    authenticated_identities: Sequence[str],
    authenticated_identity_hash: str,
) -> tuple[Sequence[str], np.ndarray, str]:
    """Validate scale labels without a population-sized Python tuple/set."""

    with np.load(path, allow_pickle=False) as payload:
        if not {"identities", "labels"}.issubset(payload.files):
            raise ValueError("Stage-D scale labels lack identities/labels")
        label_identities = payload["identities"]
        labels = np.asarray(payload["labels"], dtype=np.int64)
        label_hash = identity_order_sha256(label_identities)
    if (
        len(authenticated_identities) == 0
        or len(label_identities) != len(authenticated_identities)
        or label_hash != authenticated_identity_hash
        or labels.shape != (len(authenticated_identities),)
        or bool(((labels < 0) | (labels >= 10)).any())
    ):
        raise ValueError("Stage-D scale label population differs")
    return (
        _AuthenticatedIdentitySequence(
            authenticated_identities, authenticated_identity_hash
        ),
        labels,
        _file_sha(path),
    )


def _subset_indices(
    source_identities: Sequence[str],
    requested: Sequence[str],
    *,
    require_positional: bool = False,
) -> Sequence[int]:
    source_hash = getattr(source_identities, "identity_order_sha256", None)
    if source_hash is None:
        source_hash = identity_order_sha256(source_identities)
    requested_hash = getattr(requested, "identity_order_sha256", None)
    if requested_hash is None:
        requested_hash = identity_order_sha256(requested)
    if len(source_identities) == len(requested) and source_hash == requested_hash:
        return range(len(requested))
    if require_positional:
        raise ValueError("scale identity populations are not positionally identical")
    lookup = {value: index for index, value in enumerate(source_identities)}
    if len(lookup) != len(source_identities) or not set(requested).issubset(lookup):
        raise ValueError("Stage-D identity join lacks exact source coverage")
    return np.asarray([lookup[value] for value in requested], dtype=np.int64)


def _load_hlt(
    caches: Mapping[str, Any],
    identities: Sequence[str] | None,
) -> tuple[
    dict[int, dict[str, np.ndarray]],
    dict[int, Sequence[int]],
    dict[str, str],
    str,
    str,
    dict[int, str],
]:
    arrays_by_replica = {}
    source_indices_by_replica = {}
    lineage = {}
    content_hashes = {}
    logical_roles, policies = set(), set()
    canonical_source_hash = None
    for raw_replica, raw_path in sorted(caches.items(), key=lambda item: int(item[0])):
        replica, path = int(raw_replica), Path(raw_path)
        arrays, metadata = (
            load_materialized_hlt_input_view(path)
            if path.is_file()
            else load_hlt_v3_cache(path)
        )
        source_ids = arrays["identities"]
        source_hash = metadata.get("identity_order_sha256")
        if source_hash is None:
            source_hash = identity_order_sha256(source_ids)
        if canonical_source_hash is None:
            canonical_source_hash = source_hash
        elif source_hash != canonical_source_hash:
            raise ValueError("Stage-D HLT replica identity orders differ")
        order = (
            range(len(source_ids))
            if identities is None
            else _subset_indices(
                source_ids,
                identities,
                require_positional=str(metadata["logical_role"]) == "scale_train",
            )
        )
        arrays_by_replica[replica] = {
            name: arrays[name]
            for name in ("tokens", "mask", "measurement_states", "identities")
        }
        arrays_by_replica[replica]["identity_order_sha256"] = source_hash
        source_indices_by_replica[replica] = order
        lineage[f"hlt_replica_{replica}"] = metadata["content_hash"]
        content_hashes[replica] = metadata["array_content_sha256"]
        logical_roles.add(str(metadata["logical_role"]))
        policies.add(str(metadata["realization_policy"]))
    if len(logical_roles) != 1 or len(policies) != 1:
        raise ValueError("Stage-D HLT cache roles/policies differ")
    return (
        arrays_by_replica,
        source_indices_by_replica,
        lineage,
        next(iter(logical_roles)),
        next(iter(policies)),
        content_hashes,
    )


class _IdentityAlignedTreeShards(Sequence[Mapping[str, Any]]):
    """One-shard resident tree sequence with exact canonical-order binding."""

    def __init__(
        self,
        root: Path,
        identities: Sequence[str],
        expected_parents: Mapping[str, str],
        cache_coordinator: Any | None = None,
    ) -> None:
        self.split = AuthenticatedTreeSplit(
            root,
            expected_identities=identities,
            expected_parents=expected_parents,
        )
        self._cached_shard = -1
        self._cached_rows: tuple[Mapping[str, Any], ...] = ()
        self._cache_coordinator = cache_coordinator
        self.decoded_shard_load_count = 0

    def clear_cached_shard(self) -> None:
        self._cached_shard = -1
        self._cached_rows = ()

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard = self.split.shard_for_event(index)
        start = self.split.records[shard].start
        if shard != self._cached_shard:
            if self._cache_coordinator is not None:
                self._cache_coordinator.activate(self)
            _, rows = self.split.load_shard(shard)
            self.decoded_shard_load_count += 1
            self._cached_rows = tuple(rows)
            self._cached_shard = shard
        return copy.deepcopy(self._cached_rows[index - start])


class _TreeShardCoordinator:
    def __init__(self) -> None:
        self.active_store = None

    def activate(self, store: Any) -> None:
        if self.active_store is store:
            return
        if self.active_store is not None:
            self.active_store.clear_cached_shard()
        self.active_store = store


def _trees(
    paths: Mapping[str, Any],
    identities: Sequence[str],
    expected_parents: Mapping[str, Any],
) -> tuple[dict[int, tuple[Mapping[str, Any], ...]], dict[str, str]]:
    output, lineage = {}, {}
    coordinator = _TreeShardCoordinator()
    for raw_replica, raw_path in sorted(paths.items(), key=lambda item: int(item[0])):
        replica, root = int(raw_replica), Path(raw_path)
        parents = expected_parents.get(str(replica))
        if not isinstance(parents, Mapping):
            raise ValueError("Stage-D tree definition lacks exact active parents")
        output[replica] = _IdentityAlignedTreeShards(
            root, identities, parents, cache_coordinator=coordinator
        )
        lineage[f"tree_replica_{replica}"] = output[replica].split.manifest[
            "content_hash"
        ]
    return output, lineage


def _load_target_cache(path: Path):
    spec = load_hashed_json(
        path / "cache_spec.json", expected_contract="hosd_target_cache_spec_v1"
    )
    loader = (
        load_target_cache_sharded
        if spec.get("split") == "scale_train"
        else load_target_cache
    )
    return loader(path, cache_spec=spec)


class _IndexedTransformedTarget:
    def __init__(
        self,
        *,
        raw: Any,
        mask: Any,
        order: np.ndarray,
        coordinate_id: str,
        parameterization: str,
        normalizer: Mapping[str, Any] | None,
        whitening: Mapping[str, Any] | None,
        return_mask: bool,
    ) -> None:
        self.raw = raw
        self.mask = mask
        self.order = order
        self.coordinate_id = coordinate_id
        self.parameterization = parameterization
        self.normalizer = normalizer
        self.whitening = whitening
        self.return_mask = return_mask
        self.shape = (len(order), *tuple(raw.shape[1:]))
        self.dtype = np.dtype(bool if return_mask else np.float32)

    def __getitem__(self, index: int) -> np.ndarray:
        source = int(self.order[int(index)])
        mask = np.asarray(self.mask[source], dtype=bool)
        if self.return_mask:
            return mask
        raw = np.asarray(self.raw[source])
        if self.parameterization == "KD":
            return raw.astype(np.float32, copy=True)
        if self.parameterization == "WHITENED_ABS":
            transformed = whiten_latents(
                raw[None], whitening=self.whitening
            )[0]
            transformed[~mask] = 0
            return transformed.astype(np.float32, copy=False)
        return normalize_target(
            raw[None],
            mask[None],
            target_id=self.coordinate_id,
            normalizer=self.normalizer,
        )[0]


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
            expected_contract="hosd_latent_whitening_v2",
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
        order = _subset_indices(
            cache.identities,
            identities,
            require_positional=cache.manifest.get("split") == "scale_train",
        )
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
        raw = cache.values[coordinate_id]
        mask = cache.masks[coordinate_id]
        transformed = _IndexedTransformedTarget(
            raw=raw,
            mask=mask,
            order=order,
            coordinate_id=coordinate_id,
            parameterization=row["parameterization"],
            normalizer=normalizer,
            whitening=whitening,
            return_mask=False,
        )
        aligned_mask = _IndexedTransformedTarget(
            raw=raw,
            mask=mask,
            order=order,
            coordinate_id=coordinate_id,
            parameterization=row["parameterization"],
            normalizer=normalizer,
            whitening=whitening,
            return_mask=True,
        )
        target_key = -1 if str(key) == "shared" else int(key)
        values[target_key] = transformed
        masks[target_key] = aligned_mask
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
    shared_base_datasets_by_role: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del campaign_root, target_registry
    manifest = load_hashed_json(Path(manifest_path))
    if manifest.get("contract") != LOADER_MANIFEST_CONTRACT_V5:
        raise ValueError("Stage-D loader manifest contract differs")
    evaluation_role = str(manifest.get("evaluation_role", "design_select"))
    if evaluation_role not in {"design_select", "design_confirm"}:
        raise ValueError("Stage-D loader evaluation role differs")
    training_role = str(manifest.get("training_role", "model_train"))
    if training_role not in {"model_train", "scale_train"}:
        raise ValueError("Stage-D loader training role differs")
    roles = (training_role, "val_stop", evaluation_role)
    expected_sampler_seeds = {
        role: data_order_seed(int(row["pipeline_seed"]), role)
        for role in roles
    }
    if (
        manifest.get("source") != campaign["source"]
        or manifest.get("campaign_spec_sha256") != campaign["content_hash"]
        or manifest.get("row_id") != row["row_id"]
        or manifest.get("data_order_contract") != DATA_ORDER_CONTRACT
        or manifest.get("sampler_contract_by_role")
        != {role: sampler_contract(role) for role in roles}
        or manifest.get("sampler_seed_by_role") != expected_sampler_seeds
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
    exact_hlt_runtime = None
    for role in roles:
        definition = manifest["roles"][role]
        shared_base = (
            None
            if shared_base_datasets_by_role is None
            else shared_base_datasets_by_role.get(role)
        )
        if shared_base is not None:
            label_sha = _file_sha(Path(definition["labels"]))
            expected_hlt_paths = {
                str(key): str(Path(value).resolve())
                for key, value in sorted(definition["hlt_caches"].items())
            }
            expected_tree_paths = {
                str(key): str(Path(value).resolve())
                for key, value in sorted(definition.get("tree_caches", {}).items())
            }
            if (
                label_sha != getattr(shared_base, "hosd_label_sha256", None)
                or expected_hlt_paths
                != getattr(shared_base, "hosd_hlt_cache_paths", None)
                or (
                    expected_tree_paths
                    and expected_tree_paths
                    != getattr(shared_base, "hosd_tree_cache_paths", None)
                )
            ):
                raise ValueError("shared Stage-D base lineage differs")
            identities = shared_base.identities
            identity_hash = shared_base.identity_order_sha256
            labels = shared_base.labels
            hlt_lineage = dict(shared_base.hosd_hlt_lineage)
            hlt_content_hashes = dict(shared_base.hosd_hlt_content_hashes)
            tree_lineage = dict(shared_base.hosd_tree_lineage)
            logical_role = shared_base.logical_role
            realization_policy = shared_base.realization_policy
            base = shared_base
        else:
            if role == "scale_train":
                loaded_hlt = _load_hlt(definition["hlt_caches"], None)
                compatibility_loader = len(loaded_hlt) == 5
                if compatibility_loader:
                    identities, labels, label_sha = _labels(
                        Path(definition["labels"])
                    )
                    identity_hash = identity_order_sha256(identities)
                else:
                    mmap_identities = loaded_hlt[0][min(loaded_hlt[0])][
                        "identities"
                    ]
                    identity_hash = loaded_hlt[0][min(loaded_hlt[0])][
                        "identity_order_sha256"
                    ]
                    identities, labels, label_sha = _scale_labels(
                        Path(definition["labels"]),
                        mmap_identities,
                        identity_hash,
                    )
            else:
                identities, labels, label_sha = _labels(
                    Path(definition["labels"])
                )
                loaded_hlt = _load_hlt(definition["hlt_caches"], identities)
                compatibility_loader = len(loaded_hlt) == 5
                identity_hash = identity_order_sha256(identities)
            if compatibility_loader:
                # Compatibility for injected non-tree test loaders; real loaders
                # always return authenticated content hashes.
                (
                    hlt_arrays,
                    source_indices,
                    hlt_lineage,
                    logical_role,
                    realization_policy,
                ) = loaded_hlt
                hlt_content_hashes = {}
            else:
                (
                    hlt_arrays,
                    source_indices,
                    hlt_lineage,
                    logical_role,
                    realization_policy,
                    hlt_content_hashes,
                ) = loaded_hlt
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
        if shared_base is None and definition.get("tree_caches"):
            expected_tree_parents = definition.get("tree_expected_parents")
            if not isinstance(expected_tree_parents, Mapping):
                raise ValueError("Stage-D tree consumers require exact parents")
            for replica, content_hash in hlt_content_hashes.items():
                parent = expected_tree_parents.get(str(replica), {})
                if parent.get("hlt_content_sha256") != content_hash:
                    raise ValueError("Stage-D tree/HLT content parent differs")
            trees, tree_lineage = _trees(
                definition["tree_caches"], identities, expected_tree_parents
            )
            lineage.update(
                {f"{role}_{key}": value for key, value in tree_lineage.items()}
            )
        elif shared_base is not None and definition.get("tree_caches"):
            lineage.update(
                {f"{role}_{key}": value for key, value in tree_lineage.items()}
            )
        if shared_base is None:
            base = HLTArrayDataset(
                replica_arrays=hlt_arrays,
                labels=labels,
                identities=identities,
                logical_role=logical_role,
                realization_policy=realization_policy,
                source_indices_by_replica=source_indices,
                region_trees_by_replica=trees,
            )
            base.identity_order_sha256 = identity_hash
            base.hosd_label_sha256 = label_sha
            base.hosd_hlt_cache_paths = {
                str(key): str(Path(value).resolve())
                for key, value in sorted(definition["hlt_caches"].items())
            }
            base.hosd_tree_cache_paths = {
                str(key): str(Path(value).resolve())
                for key, value in sorted(definition.get("tree_caches", {}).items())
            }
            base.hosd_hlt_lineage = dict(hlt_lineage)
            base.hosd_hlt_content_hashes = dict(hlt_content_hashes)
            base.hosd_tree_lineage = dict(tree_lineage if trees is not None else {})
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
            if row.get("control") == "EXACT_HLT":
                candidate = {
                    "target_normalizer": normalizer,
                    "relation_normalizer": relation,
                }
                if exact_hlt_runtime is None:
                    exact_hlt_runtime = candidate
                elif (
                    exact_hlt_runtime["target_normalizer"]["content_hash"]
                    != normalizer["content_hash"]
                    or exact_hlt_runtime["relation_normalizer"]["content_hash"]
                    != relation["content_hash"]
                ):
                    raise ValueError(
                        "exact HLT runtime normalization differs across roles"
                    )
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
            seed=expected_sampler_seeds[role],
            training=role == training_role,
            batch_size=64,
        )
    for name, value in lineage.items():
        require_sha256(value, name=f"stage_d_lineage.{name}")
    if row.get("control") == "EXACT_HLT" and exact_hlt_runtime is None:
        raise ValueError("exact HLT reference requires a streamed same-view target")
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
        "data_order_contract": DATA_ORDER_CONTRACT,
        "sampler_contract_by_role": {
            role: sampler_contract(role) for role in roles
        },
        "sampler_seed_by_role": expected_sampler_seeds,
        **(
            {"exact_hlt_runtime": exact_hlt_runtime}
            if exact_hlt_runtime is not None
            else {}
        ),
    }


__all__ = [
    "LOADER_MANIFEST_CONTRACT",
    "build_stage_d_loader_manifest",
    "build_default_stage_d_role_definitions",
    "load_stage_d_loaders_from_manifest",
    "LOADER_MANIFEST_CONTRACT_V2",
    "LOADER_MANIFEST_CONTRACT_V3",
    "LOADER_MANIFEST_CONTRACT_V4",
    "LOADER_MANIFEST_CONTRACT_V5",
    "DATA_ORDER_CONTRACT",
    "data_order_seed",
    "sampler_contract",
]
