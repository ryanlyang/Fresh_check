"""Stage-J target-refit helpers with bounded streamed-pair memory."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from teacher_logit_reco.relational_part.ca_tree import unpack_tree_shard
from teacher_logit_reco.relational_part.normalization import (
    NORMALIZATION_PAIR_LIMIT_PER_JET,
    NORMALIZATION_PAIR_SALT,
    select_normalization_jet_indices,
)

from .contracts import (
    SCALE_EXECUTION_PLAN_CONTRACT,
    SCALE_TARGET_COMPLETION_CONTRACT,
    SCALE_TARGET_WAVE_COMPLETION_CONTRACT,
    SCALE_GRAPH_WAVE_COMPLETION_CONTRACT,
    SCALE_ROW_RESULT_CONTRACT,
    TARGET_NORMALIZER_CONTRACT,
    load_hashed_json,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .extractors import ExtractorResources, extract_registered_target
from .normalization import fit_streamed_target_normalizer, validate_target_normalizer
from .target_schemas import target_declarations


PAIR_TARGETS = frozenset({"T_HLT_TRACK_PAIR_13", "T_HLT_REGION_PAIR_8"})


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def offline_to_hlt_target(target_id: str) -> str | None:
    """Return the exact same-view HLT coordinate for an offline physical target."""

    target_id = str(target_id)
    if target_id.startswith(("T_OFFLINE_LOGITS_", "T_OFFLINE_POOLED_")):
        return None
    if target_id.startswith("T_OFFLINE_RELATION_"):
        return target_id.replace(
            "T_OFFLINE_RELATION_", "T_HLT_SELF_RELATION_", 1
        )
    if target_id.startswith("T_OFFLINE_"):
        return target_id.replace("T_OFFLINE_", "T_HLT_SELF_", 1)
    return None


def target_component_kinds(target_id: str) -> tuple[str, ...]:
    rows = {row.target_id: row for row in target_declarations()}
    if target_id not in rows:
        raise ValueError(f"unknown target declaration {target_id!r}")
    if target_id == "T_HLT_REGION_PAIR_8":
        return ("binary", "binary", "binary", *(["continuous"] * 5))
    if target_id == "T_HLT_TRACK_PAIR_13":
        return ("continuous",) * 13
    raise ValueError("component kinds require the compiled target registry")


def merge_target_normalizers(
    normalizers: Sequence[Mapping[str, Any]],
    *,
    fitting_population: str,
    normalization_role: str,
    parent_hashes: Mapping[str, str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine disjoint coordinate rows without changing per-replica statistics."""

    if not normalizers:
        raise ValueError("target-normalizer merge is empty")
    targets: dict[str, Mapping[str, Any]] = {}
    fit_split = None
    for normalizer in normalizers:
        validate_target_normalizer(normalizer)
        if (
            normalizer.get("source") != dict(source)
            or normalizer.get("fitting_population") != fitting_population
            or normalizer.get("normalization_role") != normalization_role
        ):
            raise ValueError("target-normalizer merge lineage differs")
        if fit_split is None:
            fit_split = str(normalizer["fit_split"])
        elif fit_split != str(normalizer["fit_split"]):
            raise ValueError("target-normalizer merge splits differ")
        for row in normalizer["targets"]:
            target_id = str(row["target_id"])
            if target_id in targets:
                raise ValueError("target-normalizer merge duplicates a coordinate")
            targets[target_id] = dict(row)
    return with_content_hash(
        {
            "contract": TARGET_NORMALIZER_CONTRACT,
            "schema_version": 1,
            "fitting_population": fitting_population,
            "normalization_role": normalization_role,
            "fit_split": fit_split,
            "population_role": (
                "model_train_only"
                if fit_split == "model_train"
                else "scale_train_only"
            ),
            "parent_normalizer_hashes": [
                normalizer["content_hash"] for normalizer in normalizers
            ],
            "parent_hashes": {
                str(key): require_sha256(value, name=f"parent_hashes.{key}")
                for key, value in sorted(parent_hashes.items())
            },
            "identity_values_stored": False,
            "quantile_method": "numpy_linear",
            "continuous_scale": "max((q75-q25)/1.349,1e-6)",
            "normalized_clipping": [-12.0, 12.0],
            "targets": [targets[key] for key in sorted(targets)],
            "source": dict(source),
        }
    )


def _load_input(path: Path) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        forbidden = names & {"label", "labels", "class", "classes", "y"}
        if forbidden:
            raise ValueError(f"scale target input exposes labels: {sorted(forbidden)}")
        identity_key = "identity" if "identity" in names else "identities"
        required = {identity_key, "raw_tokens", "mask"}
        if not required.issubset(names):
            raise ValueError("scale target input lacks identity/raw_tokens/mask")
        identities = tuple(str(value) for value in archive[identity_key].tolist())
        raw = np.asarray(archive["raw_tokens"], dtype=np.float32)
        mask = np.asarray(archive["mask"], dtype=bool)
        vectors = (
            np.asarray(archive["vectors"], dtype=np.float32)
            if "vectors" in names
            else None
        )
    if (
        len(identities) != raw.shape[0]
        or mask.shape != raw.shape[:2]
        or len(identities) != len(set(identities))
    ):
        raise ValueError("scale target input population differs")
    return identities, raw, mask, vectors


def _tree_lookup(root: Path, identities: Sequence[str]) -> tuple[Mapping[str, Any], ...]:
    manifest = load_hashed_json(root / "manifest.json")
    by_identity = {}
    for shard in sorted((root / "shards").glob("shard_*.npz")):
        shard_ids, trees = unpack_tree_shard(shard)
        for identity, tree in zip(shard_ids, trees):
            if identity in by_identity:
                raise ValueError("scale tree cache duplicates an identity")
            by_identity[identity] = tree
    if not set(identities).issubset(by_identity):
        raise ValueError("scale tree cache lacks selected identities")
    return tuple(by_identity[value] for value in identities), manifest


def _sample_coordinates(
    identity: str, mask: np.ndarray
) -> tuple[tuple[int, int], ...]:
    applicable = np.argwhere(np.asarray(mask, dtype=bool).any(axis=0))
    coordinates = [(int(left), int(right)) for left, right in applicable]
    coordinates.sort(
        key=lambda pair: (
            hashlib.sha256(
                "\0".join(
                    (
                        NORMALIZATION_PAIR_SALT,
                        str(identity),
                        str(pair[0]),
                        str(pair[1]),
                    )
                ).encode("utf-8")
            ).digest(),
            pair,
        )
    )
    return tuple(coordinates[:NORMALIZATION_PAIR_LIMIT_PER_JET])


def fit_pair_normalizer_from_views(
    *,
    target_id: str,
    view_paths_by_replica: Mapping[int, str | Path],
    relation_normalizer: Mapping[str, Any],
    tree_paths_by_replica: Mapping[int, str | Path] | None,
    fitting_population: str,
    split: str,
    source: Mapping[str, Any],
    batch_size: int = 128,
) -> dict[str, Any]:
    """Extract only deterministic normalization coordinates from R_MULTI views."""

    if target_id not in PAIR_TARGETS or not view_paths_by_replica:
        raise ValueError("streamed pair normalizer target/views differ")
    if relation_normalizer.get("source") != dict(source):
        raise ValueError("streamed pair relation normalizer source differs")
    requires_tree = target_id == "T_HLT_REGION_PAIR_8"
    if requires_tree != bool(tree_paths_by_replica):
        raise ValueError("streamed REGION tree resources differ")
    declarations = {row.target_id: row for row in target_declarations()}
    declaration = declarations[target_id]
    component_samples: list[list[np.ndarray]] = [
        [] for _ in declaration.components
    ]
    pair_digest = hashlib.sha256(b"hosd_streamed_pair_normalizer_pairs_v1\0")
    jet_digest = hashlib.sha256(b"hosd_streamed_pair_normalizer_jets_v1\0")
    parent_hashes = {"relation_normalizer": relation_normalizer["content_hash"]}
    selected_total = 0
    floors = relation_normalizer.get("track_uncertainty_floors", {})
    resources = ExtractorResources(
        d0_uncertainty_floor=float(floors.get("d0", {}).get("floor", 0.0)),
        dz_uncertainty_floor=float(floors.get("dz", {}).get("floor", 0.0)),
        sentinel_policy=relation_normalizer.get("track_sentinel_policy"),
    )
    for replica, raw_path in sorted(view_paths_by_replica.items()):
        path = Path(raw_path)
        identities, raw, valid, vectors = _load_input(path)
        selected = select_normalization_jet_indices(identities)
        selected_total += int(selected.size)
        parent_hashes[f"hlt_view_{replica}"] = file_sha256(path)
        tree_rows = None
        if requires_tree:
            tree_root = Path(tree_paths_by_replica[int(replica)])
            selected_ids = [identities[int(index)] for index in selected]
            tree_rows, tree_manifest = _tree_lookup(tree_root, selected_ids)
            parent_hashes[f"tree_view_{replica}"] = tree_manifest["content_hash"]
        for start in range(0, len(selected), int(batch_size)):
            coordinate = selected[start : start + int(batch_size)]
            batch_trees = (
                None
                if tree_rows is None
                else tree_rows[start : start + len(coordinate)]
            )
            batch = extract_registered_target(
                target_id,
                raw[coordinate],
                valid[coordinate],
                resources=resources,
                vectors=None if vectors is None else vectors[coordinate],
                trees=batch_trees,
            )
            values = batch.values.detach().cpu().numpy()
            masks = batch.loss_mask.detach().cpu().numpy().astype(bool)
            for local, raw_index in enumerate(coordinate):
                identity = f"replica={int(replica)}:{identities[int(raw_index)]}"
                jet_digest.update(identity.encode("utf-8") + b"\0")
                pairs = _sample_coordinates(identity, masks[local])
                for left, right in pairs:
                    pair_digest.update(
                        f"{identity}#{left}>{right}".encode("utf-8") + b"\0"
                    )
                if not pairs:
                    continue
                left = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
                right = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
                for component in range(values.shape[1]):
                    applicable = masks[local, component, left, right]
                    if bool(applicable.any()):
                        component_samples[component].append(
                            np.asarray(
                                values[local, component, left[applicable], right[applicable]],
                                dtype=np.float64,
                            )
                        )
    flattened = tuple(
        np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
        for chunks in component_samples
    )
    return fit_streamed_target_normalizer(
        target_id=target_id,
        component_names=tuple(declaration.components),
        component_kinds=target_component_kinds(target_id),
        component_samples=flattened,
        fitting_population=fitting_population,
        split=split,
        selected_jet_count=selected_total,
        selected_jet_identity_sha256=jet_digest.hexdigest(),
        sampled_pair_identity_sha256=pair_digest.hexdigest(),
        parent_hashes=parent_hashes,
        source=source,
    )


def build_scale_target_completion(
    *,
    scale_plan: Mapping[str, Any],
    target_id: str,
    artifact_hashes: Mapping[str, str],
    training_target_definitions: Mapping[str, Mapping[str, Any]],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(scale_plan, expected_contract=SCALE_EXECUTION_PLAN_CONTRACT)
    matches = [
        row for row in scale_plan["target_refit_rows"]
        if str(row["target_id"]) == str(target_id)
    ]
    if len(matches) != 1 or not artifact_hashes:
        raise ValueError("scale target completion coordinate differs")
    expected_parameterizations = set(matches[0]["required_parameterizations"])
    if set(training_target_definitions) != expected_parameterizations:
        raise ValueError("scale target parameterization coverage differs")
    return with_content_hash(
        {
            "contract": SCALE_TARGET_COMPLETION_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "scale_execution_plan_sha256": scale_plan["content_hash"],
            "target_id": str(target_id),
            "plan_row": dict(matches[0]),
            "artifact_hashes": {
                str(key): require_sha256(value, name=f"artifact_hashes.{key}")
                for key, value in sorted(artifact_hashes.items())
            },
            "training_target_definitions": {
                str(key): dict(value)
                for key, value in sorted(training_target_definitions.items())
            },
            "fit_split": "scale_train",
            "fitting_population": "target_scale",
            "all_required_statistics_refit": True,
            "dense_pair_target_persisted": False,
        }
    )


def build_scale_target_wave_completion(
    *,
    scale_plan: Mapping[str, Any],
    completions: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(scale_plan, expected_contract=SCALE_EXECUTION_PLAN_CONTRACT)
    expected = {str(row["target_id"]) for row in scale_plan["target_refit_rows"]}
    observed = {}
    for completion in completions:
        validate_content_hash(
            completion, expected_contract=SCALE_TARGET_COMPLETION_CONTRACT
        )
        target_id = str(completion["target_id"])
        if (
            target_id in observed
            or completion.get("source") != dict(source)
            or completion.get("scale_execution_plan_sha256")
            != scale_plan["content_hash"]
        ):
            raise ValueError("scale target wave lineage differs")
        observed[target_id] = completion
    if set(observed) != expected:
        raise ValueError("scale target wave coverage differs")
    return with_content_hash(
        {
            "contract": SCALE_TARGET_WAVE_COMPLETION_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "scale_execution_plan_sha256": scale_plan["content_hash"],
            "target_completion_hashes": {
                key: observed[key]["content_hash"] for key in sorted(observed)
            },
            "coverage_exact": True,
            "performance_based_termination": False,
        }
    )


def build_scale_graph_wave_completion(
    *,
    scale_plan: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(scale_plan, expected_contract=SCALE_EXECUTION_PLAN_CONTRACT)
    expected = {
        (str(row["graph_id"]), int(row["seed"]))
        for row in scale_plan["graph_rows"]
    }
    observed = {}
    for result in results:
        validate_content_hash(result, expected_contract=SCALE_ROW_RESULT_CONTRACT)
        key = (str(result["graph_id"]), int(result["seed"]))
        if (
            key in observed
            or result.get("source") != dict(source)
            or result.get("scale_execution_plan_sha256")
            != scale_plan["content_hash"]
        ):
            raise ValueError("scale graph wave lineage differs")
        observed[key] = result
    if set(observed) != expected:
        raise ValueError("scale graph wave coverage differs")
    return with_content_hash(
        {
            "contract": SCALE_GRAPH_WAVE_COMPLETION_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "scale_execution_plan_sha256": scale_plan["content_hash"],
            "result_hashes": {
                f"{graph_id}__seed_{seed}": observed[(graph_id, seed)][
                    "content_hash"
                ]
                for graph_id, seed in sorted(observed)
            },
            "coverage_exact": True,
            "all_registered_rows_executed": True,
            "performance_based_termination": False,
        }
    )


__all__ = [
    "PAIR_TARGETS",
    "build_scale_target_completion",
    "build_scale_target_wave_completion",
    "build_scale_graph_wave_completion",
    "file_sha256",
    "fit_pair_normalizer_from_views",
    "merge_target_normalizers",
    "offline_to_hlt_target",
    "target_component_kinds",
]
