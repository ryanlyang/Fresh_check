"""Contracts, backend authentication, probes, and compact REGION sidecars."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    canonical_json_bytes,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)
from .region_tree import (
    EXCLUSIVE_RESOLUTIONS,
    TREE_SCHEMA_CONTRACT,
    tree_content_sha256,
    validate_tree,
)


ANGULAR_TREE_RESOURCE_CONTRACT = "relational_part_angular_tree_resource_v1"
ANGULAR_TREE_BACKEND_CONTRACT = "relational_ca_tree_v1"
ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT = "relational_ca_tree_backend_manifest_v1"
ANGULAR_TREE_SHARD_CONTRACT = "relational_ca_tree_shard_v1"
ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT = "relational_ca_tree_split_manifest_v1"
ANGULAR_TREE_PROBE_CONTRACT = "relational_ca_tree_throughput_probe_v1"
TREE_SHARD_MAX_JETS = 10_000
TREE_PROBE_STRATA = (
    (0, 8), (9, 16), (17, 24), (25, 32), (33, 40),
    (41, 48), (49, 64), (65, 80), (81, 96), (97, 128),
)


def build_angular_tree_resource_contract(
    *,
    split_binding_sha256: str,
) -> dict[str, Any]:
    split_binding_sha256 = require_sha256(
        split_binding_sha256, name="split_binding_sha256"
    )
    return with_content_hash(
        {
            "contract": ANGULAR_TREE_RESOURCE_CONTRACT,
            "schema_version": 1,
            "split_binding_sha256": split_binding_sha256,
            "backend_contract": ANGULAR_TREE_BACKEND_CONTRACT,
            "algorithm": {
                "kind": "beam_free_exclusive_angular_tree",
                "distance": "delta_r_squared",
                "beam_distance": None,
                "recombination": "E_scheme",
                "merge_until_roots": 1,
                "reference_radius": 0.8,
                "reference_radius_affects_topology": False,
                "exclusive_resolutions": [2, 4, 8],
                "n_less_than_k_policy": "n_valid_singleton_clusters",
                "all_invalid_policy": "zero_relation_tensor",
            },
            "canonicalization": {
                "leaf_order": "lexicographic_canonical_raw_hlt_physics_bytes",
                "signed_zero_canonicalized": True,
                "nan_policy_from_raw_input_schema": True,
                "merge_tie_break": "ordered_cluster_leaf_key_multisets",
                "input_row_index_allowed": False,
            },
            "backend": {
                "mechanism": "torch_cpp_extension_cpu",
                "cpp_standard": "c++17",
                "openmp_required": True,
                "compile_once_per_campaign": True,
                "worker_jit_allowed": False,
                "required_flags": [
                    "-O3", "-fno-fast-math", "-fno-associative-math",
                    "-ffp-contract=off",
                ],
                "prohibited_flags": ["-Ofast", "-ffast-math"],
                "calculation_precision": "ieee_float64",
                "storage_precision": "ieee_float32",
                "time_complexity": "O(N^2 log N)",
                "temporary_memory_complexity": "O(N^2)",
                "maximum_constituents": 128,
            },
            "required_backend_identity": [
                "contract_id", "schema_version", "source_sha256",
                "binary_sha256", "compiler_identity",
                "compiler_major_version", "compiler_flags",
                "platform_architecture", "python_major_minor",
                "pytorch_version", "pytorch_cxx11_abi",
                "openmp_available", "self_test_sha256",
            ],
            "sharding": {
                "maximum_jets_per_shard": 10_000,
                "ordering": "split_manifest_identity_order",
                "publication": "job_unique_temporary_then_atomic_rename",
                "resume": "reuse_only_fully_hash_valid_shards",
                "split_manifest_publish_after_all_shards": True,
            },
            "throughput_probe": {
                "sample_split": "model_train",
                "sample_count": 20_000,
                "strata": [list(value) for value in TREE_PROBE_STRATA],
                "initial_quota_per_stratum": 2_000,
                "selection_salt": "rpt_tree_probe_v1",
                "undersized_policy": (
                    "population_proportional_largest_remainder"
                ),
                "parity_sample_count": 1_000,
                "parity_salt": "rpt_tree_probe_parity_v1",
                "maximum_projected_shard_hours": 2.0,
                "maximum_projected_cpu_node_hours": 48.0,
            },
            "persistent_pair_matrices_allowed": False,
            "runtime_recomputation_from_hlt_only": True,
        }
    )


def validate_backend_manifest(
    manifest: Mapping[str, Any],
    *,
    binary_path: Path | None = None,
    source_path: Path | None = None,
) -> str:
    digest = validate_content_hash(
        manifest, expected_contract=ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT
    )
    required = {
        "contract_id", "schema_version", "source_sha256", "binary_sha256",
        "compiler_identity", "compiler_major_version", "compiler_flags",
        "platform_architecture", "python_major_minor", "pytorch_version",
        "pytorch_cxx11_abi", "openmp_available", "self_test_sha256",
    }
    if not required.issubset(manifest):
        raise ValueError("tree backend manifest lacks required ABI identity")
    if manifest["contract_id"] != ANGULAR_TREE_BACKEND_CONTRACT:
        raise ValueError("tree backend contract ID differs")
    flags = list(manifest["compiler_flags"])
    required_flags = {
        "-O3", "-fno-fast-math", "-fno-associative-math", "-ffp-contract=off"
    }
    if not required_flags.issubset(flags) or any(
        flag in flags for flag in ("-Ofast", "-ffast-math")
    ):
        raise ValueError("tree backend compiler flags differ")
    if manifest.get("openmp_available") is not True:
        raise ValueError("tree backend lacks required OpenMP")
    for field in ("source_sha256", "binary_sha256", "self_test_sha256"):
        require_sha256(manifest[field], name=field)
    if binary_path is not None and sha256_file(binary_path) != manifest["binary_sha256"]:
        raise ValueError("tree backend binary hash differs")
    if source_path is not None and sha256_file(source_path) != manifest["source_sha256"]:
        raise ValueError("tree backend source hash differs")
    return digest


def load_tree_backend(
    binary_path: Path,
    manifest_path: Path,
    *,
    source_path: Path,
) -> Any:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_backend_manifest(
        manifest, binary_path=binary_path, source_path=source_path
    )
    spec = importlib.util.spec_from_file_location(
        "relational_ca_tree_v1_ext", binary_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct tree backend import specification")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime_manifest = module.backend_manifest()
    for field in (
        "contract_id", "schema_version", "compiler_flags",
        "pytorch_cxx11_abi", "openmp_available",
    ):
        runtime_value = runtime_manifest.get(field)
        manifest_value = manifest.get(field)
        if field == "compiler_flags":
            runtime_value = list(runtime_value)
            manifest_value = list(manifest_value)
        if runtime_value != manifest_value:
            raise RuntimeError(f"loaded tree backend differs at {field}")
    self_test = module.self_test()
    self_test_sha = hashlib.sha256(
        canonical_json_bytes(self_test)
    ).hexdigest()
    if self_test_sha != manifest["self_test_sha256"]:
        raise RuntimeError("loaded tree backend self-test differs")
    return module


def build_compiled_tree(
    module: Any,
    vectors: np.ndarray,
    raw_tokens: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("compiled tree execution requires PyTorch") from exc
    raw = module.build_tree(
        torch.as_tensor(vectors, dtype=torch.float64),
        torch.as_tensor(raw_tokens, dtype=torch.float64),
        torch.as_tensor(mask, dtype=torch.bool),
    )
    tree: dict[str, Any] = {
        "contract": TREE_SCHEMA_CONTRACT,
        "n_particles": int(raw["n_particles"]),
        "n_valid": int(raw["n_valid"]),
        "n_nodes": int(raw["n_nodes"]),
        "root": int(raw["root"]),
        "assignments": {},
        "actual_cluster_counts": {},
    }
    for name in (
        "leaf_to_node", "parent", "left", "right", "depth", "vectors", "pt",
        "mass", "multiplicity", "merge_delta_r", "merge_kt", "merge_z",
        "merge_mass",
    ):
        tree[name] = raw[name].detach().cpu().numpy()
    for resolution in EXCLUSIVE_RESOLUTIONS:
        tree["assignments"][str(resolution)] = (
            raw[f"assignment_K{resolution}"].detach().cpu().numpy()
        )
        tree["actual_cluster_counts"][str(resolution)] = int(
            raw[f"actual_count_K{resolution}"]
        )
    validate_tree(tree)
    return tree


def _identity_key(identity: Any) -> str:
    method = getattr(identity, "key", None)
    if callable(method):
        return str(method())
    if isinstance(identity, Mapping):
        return f"{identity['file']}#{int(identity['entry'])}"
    return str(identity)


def _identity_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def pack_tree_shard(
    trees: Sequence[Mapping[str, Any]],
    identities: Sequence[Any],
) -> dict[str, np.ndarray]:
    if len(trees) != len(identities) or not trees:
        raise ValueError("tree shard requires matching nonempty inputs")
    for tree in trees:
        validate_tree(tree)
    particle_counts = {int(tree["n_particles"]) for tree in trees}
    if len(particle_counts) != 1:
        raise ValueError("tree shard particle widths differ")
    width = particle_counts.pop()
    node_offsets = [0]
    for tree in trees:
        node_offsets.append(node_offsets[-1] + int(tree["n_nodes"]))
    packed: dict[str, np.ndarray] = {
        "identity": np.asarray([_identity_key(value) for value in identities]),
        "n_valid": np.asarray([tree["n_valid"] for tree in trees], dtype=np.int16),
        "root": np.asarray([tree["root"] for tree in trees], dtype=np.int32),
        "node_offsets": np.asarray(node_offsets, dtype=np.int64),
        "leaf_to_node": np.stack([tree["leaf_to_node"] for tree in trees]),
    }
    for name in (
        "parent", "left", "right", "depth", "vectors", "pt", "mass",
        "multiplicity", "merge_delta_r", "merge_kt", "merge_z", "merge_mass",
    ):
        packed[name] = np.concatenate([np.asarray(tree[name]) for tree in trees])
    for resolution in EXCLUSIVE_RESOLUTIONS:
        packed[f"assignment_K{resolution}"] = np.stack(
            [tree["assignments"][str(resolution)] for tree in trees]
        ).reshape(len(trees), width)
        packed[f"actual_count_K{resolution}"] = np.asarray(
            [tree["actual_cluster_counts"][str(resolution)] for tree in trees],
            dtype=np.int16,
        )
    return packed


def unpack_tree_shard(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with np.load(path, allow_pickle=False) as packed:
        identities = [str(value) for value in packed["identity"]]
        offsets = packed["node_offsets"]
        trees: list[dict[str, Any]] = []
        for row in range(len(identities)):
            start, stop = map(int, offsets[row:row + 2])
            width = int(packed["leaf_to_node"].shape[1])
            n_valid = int(packed["n_valid"][row])
            tree = {
                "contract": TREE_SCHEMA_CONTRACT,
                "n_particles": width,
                "n_valid": n_valid,
                "n_nodes": stop - start,
                "root": int(packed["root"][row]),
                "leaf_to_node": packed["leaf_to_node"][row].copy(),
                "assignments": {
                    str(k): packed[f"assignment_K{k}"][row].copy()
                    for k in EXCLUSIVE_RESOLUTIONS
                },
                "actual_cluster_counts": {
                    str(k): int(packed[f"actual_count_K{k}"][row])
                    for k in EXCLUSIVE_RESOLUTIONS
                },
            }
            for name in (
                "parent", "left", "right", "depth", "vectors", "pt", "mass",
                "multiplicity", "merge_delta_r", "merge_kt", "merge_z",
                "merge_mass",
            ):
                tree[name] = packed[name][start:stop].copy()
            validate_tree(tree)
            trees.append(tree)
    return identities, trees


def write_tree_shard(
    output_path: Path,
    trees: Sequence[Mapping[str, Any]],
    identities: Sequence[Any],
    *,
    hlt_content_sha256: str,
    tree_resource_sha256: str,
    backend_manifest_sha256: str,
) -> dict[str, Any]:
    """Atomically publish or hash-validate and reuse one compact shard."""

    output_path = output_path.resolve()
    metadata_path = output_path.with_suffix(".metadata.json")
    identity_keys = [_identity_key(value) for value in identities]
    expected_parent = {
        "hlt_content_sha256": require_sha256(
            hlt_content_sha256, name="hlt_content_sha256"
        ),
        "tree_resource_sha256": require_sha256(
            tree_resource_sha256, name="tree_resource_sha256"
        ),
        "backend_manifest_sha256": require_sha256(
            backend_manifest_sha256, name="backend_manifest_sha256"
        ),
    }
    if output_path.exists() and metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        validate_content_hash(existing, expected_contract=ANGULAR_TREE_SHARD_CONTRACT)
        if (
            existing.get("parents") == expected_parent
            and existing.get("identity_sha256") == _identity_hash(identity_keys)
            and existing.get("npz_sha256") == sha256_file(output_path)
            and int(existing.get("jet_count", -1)) == len(trees)
        ):
            return {"reused": True, "metadata": existing}
        raise FileExistsError("existing tree shard is stale or incompatible")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    packed = pack_tree_shard(trees, identities)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, suffix=".npz", delete=False
    ) as handle:
        temporary_npz = Path(handle.name)
    temporary_metadata = metadata_path.with_name(
        f".{metadata_path.name}.{os.getpid()}.tmp"
    )
    try:
        np.savez_compressed(temporary_npz, **packed)
        metadata = with_content_hash(
            {
                "contract": ANGULAR_TREE_SHARD_CONTRACT,
                "schema_version": 1,
                "tree_schema": TREE_SCHEMA_CONTRACT,
                "parents": expected_parent,
                "jet_count": len(trees),
                "identity_sha256": _identity_hash(identity_keys),
                "tree_content_sha256": _identity_hash(
                    [tree_content_sha256(tree) for tree in trees]
                ),
                "npz_sha256": sha256_file(temporary_npz),
                "storage_precision": "ieee_float32",
                "topology_dtype": "int32",
            }
        )
        temporary_metadata.write_bytes(canonical_json_bytes(metadata) + b"\n")
        os.replace(temporary_npz, output_path)
        os.replace(temporary_metadata, metadata_path)
    finally:
        temporary_npz.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
    return {"reused": False, "metadata": metadata}


def finalize_tree_split(
    output_path: Path,
    shard_metadata_paths: Sequence[Path],
    *,
    split: str,
    expected_jet_count: int,
    hlt_content_sha256: str,
    tree_resource_sha256: str,
    backend_manifest_sha256: str,
) -> dict[str, Any]:
    rows = []
    total = 0
    for index, path in enumerate(shard_metadata_paths):
        row = json.loads(path.read_text(encoding="utf-8"))
        validate_content_hash(row, expected_contract=ANGULAR_TREE_SHARD_CONTRACT)
        if row["parents"] != {
            "hlt_content_sha256": hlt_content_sha256,
            "tree_resource_sha256": tree_resource_sha256,
            "backend_manifest_sha256": backend_manifest_sha256,
        }:
            raise ValueError("tree shard parent differs")
        total += int(row["jet_count"])
        rows.append(
            {
                "shard_index": index,
                "metadata_sha256": row["content_hash"],
                "jet_count": row["jet_count"],
                "identity_sha256": row["identity_sha256"],
                "npz_sha256": row["npz_sha256"],
            }
        )
    if total != int(expected_jet_count):
        raise ValueError("tree split shard jet count differs")
    manifest = with_content_hash(
        {
            "contract": ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT,
            "schema_version": 1,
            "split": str(split),
            "jet_count": total,
            "maximum_jets_per_shard": TREE_SHARD_MAX_JETS,
            "parents": {
                "hlt_content_sha256": require_sha256(
                    hlt_content_sha256, name="hlt_content_sha256"
                ),
                "tree_resource_sha256": require_sha256(
                    tree_resource_sha256, name="tree_resource_sha256"
                ),
                "backend_manifest_sha256": require_sha256(
                    backend_manifest_sha256, name="backend_manifest_sha256"
                ),
            },
            "shards": rows,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise FileExistsError("existing split tree manifest differs")
        return existing
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_json_bytes(manifest) + b"\n")
    os.replace(temporary, output_path)
    return manifest


def select_tree_probe(
    identities: Sequence[Any],
    valid_counts: Sequence[int],
    *,
    sample_count: int = 20_000,
) -> dict[str, Any]:
    if len(identities) != len(valid_counts) or len(identities) < sample_count:
        raise ValueError("model_train is too small for the locked tree probe")
    keys = [_identity_key(value) for value in identities]
    strata: list[list[int]] = [[] for _ in TREE_PROBE_STRATA]
    for index, count_value in enumerate(valid_counts):
        count = int(count_value)
        matches = [
            position for position, (lower, upper) in enumerate(TREE_PROBE_STRATA)
            if lower <= count <= upper
        ]
        if len(matches) != 1:
            raise ValueError("valid count lies outside tree probe strata")
        strata[matches[0]].append(index)
    for values in strata:
        values.sort(key=lambda index: (
            hashlib.sha256(
                b"rpt_tree_probe_v1" + keys[index].encode("utf-8")
            ).digest(),
            keys[index],
        ))
    quotas = [min(2_000, len(values)) for values in strata]
    deficit = sample_count - sum(quotas)
    remaining = [len(values) - quotas[index] for index, values in enumerate(strata)]
    while deficit > 0:
        total_remaining = sum(remaining)
        if total_remaining < deficit or total_remaining == 0:
            raise ValueError("tree probe quota redistribution is impossible")
        exact = [deficit * value / total_remaining for value in remaining]
        additions = [min(remaining[i], int(np.floor(exact[i]))) for i in range(10)]
        assigned = sum(additions)
        order = sorted(
            range(10), key=lambda i: (-(exact[i] - np.floor(exact[i])), i)
        )
        for index in order:
            if assigned >= deficit:
                break
            if additions[index] < remaining[index]:
                additions[index] += 1
                assigned += 1
        if assigned == 0:
            raise AssertionError("tree probe redistribution stalled")
        for index in range(10):
            quotas[index] += additions[index]
            remaining[index] -= additions[index]
        deficit -= assigned
    selected = [index for s, values in enumerate(strata) for index in values[:quotas[s]]]
    selected.sort(key=lambda index: keys[index])
    parity = sorted(
        selected,
        key=lambda index: (
            hashlib.sha256(
                b"rpt_tree_probe_parity_v1" + keys[index].encode("utf-8")
            ).digest(),
            keys[index],
        ),
    )[: min(1_000, sample_count)]
    return {
        "selected_indices": np.asarray(selected, dtype=np.int64),
        "parity_indices": np.asarray(parity, dtype=np.int64),
        "stratum_populations": [len(values) for values in strata],
        "final_quotas": quotas,
        "selected_identity_sha256": _identity_hash([keys[i] for i in selected]),
        "selection_salt": "rpt_tree_probe_v1",
        "parity_salt": "rpt_tree_probe_parity_v1",
    }


def build_tree_probe_artifact(
    selection: Mapping[str, Any],
    valid_counts: Sequence[int],
    elapsed_milliseconds: Sequence[float],
    persisted_bytes: Sequence[int],
    *,
    peak_resident_bytes: int,
    parity_topology_exact: bool,
    parity_max_continuous_absolute_error: float,
    total_campaign_jets: int = 1_750_000,
    operational_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected = np.asarray(selection["selected_indices"], dtype=np.int64)
    elapsed = np.asarray(elapsed_milliseconds, dtype=np.float64)
    storage = np.asarray(persisted_bytes, dtype=np.float64)
    if elapsed.shape != selected.shape or storage.shape != selected.shape:
        raise ValueError("tree probe measurement counts differ")
    if (
        np.any(~np.isfinite(elapsed))
        or np.any(elapsed <= 0)
        or np.any(~np.isfinite(storage))
        or np.any(storage < 0)
    ):
        raise ValueError("tree probe measurements are invalid")
    populations = np.asarray(selection["stratum_populations"], dtype=np.float64)
    weights = populations / populations.sum()
    stratum_rows = []
    weighted_ms = 0.0
    weighted_bytes = 0.0
    counts = np.asarray(valid_counts)
    for index, (lower, upper) in enumerate(TREE_PROBE_STRATA):
        positions = np.flatnonzero(
            (counts[selected] >= lower) & (counts[selected] <= upper)
        )
        if positions.size == 0:
            mean_ms = 0.0
            mean_bytes = 0.0
            jets_per_second = None
        else:
            mean_ms = float(elapsed[positions].mean())
            mean_bytes = float(storage[positions].mean())
            jets_per_second = 1000.0 / mean_ms
        weighted_ms += float(weights[index]) * mean_ms
        weighted_bytes += float(weights[index]) * mean_bytes
        stratum_rows.append(
            {
                "bounds": [lower, upper],
                "population": int(populations[index]),
                "probe_count": int(positions.size),
                "mean_milliseconds": mean_ms,
                "jets_per_second": jets_per_second,
                "mean_persisted_bytes": mean_bytes,
            }
        )
    projected_shard_hours = TREE_SHARD_MAX_JETS * weighted_ms / 3.6e6
    projected_cpu_node_hours = int(total_campaign_jets) * weighted_ms / 3.6e6
    limits_ok = (
        projected_shard_hours <= 2.0
        and projected_cpu_node_hours <= 48.0
        and parity_topology_exact
    )
    if not limits_ok and operational_override is None:
        raise RuntimeError("tree throughput/parity probe blocks bulk-cache submission")
    return with_content_hash(
        {
            "contract": ANGULAR_TREE_PROBE_CONTRACT,
            "schema_version": 1,
            "selection_salt": selection["selection_salt"],
            "parity_salt": selection["parity_salt"],
            "selected_identity_sha256": selection[
                "selected_identity_sha256"
            ],
            "sample_count": int(selected.size),
            "parity_sample_count": int(
                len(selection["parity_indices"])
            ),
            "strata": stratum_rows,
            "weighted_jets_per_second": 1000.0 / weighted_ms,
            "milliseconds": {
                "p50": float(np.quantile(elapsed, .5, method="linear")),
                "p95": float(np.quantile(elapsed, .95, method="linear")),
                "maximum": float(elapsed.max()),
            },
            "peak_resident_bytes": int(peak_resident_bytes),
            "weighted_persisted_bytes_per_jet": weighted_bytes,
            "projected_campaign_storage_bytes": (
                weighted_bytes * int(total_campaign_jets)
            ),
            "projected_shard_hours": projected_shard_hours,
            "projected_cpu_node_hours": projected_cpu_node_hours,
            "parity": {
                "topology_exact": bool(parity_topology_exact),
                "max_continuous_absolute_error": float(
                    parity_max_continuous_absolute_error
                ),
                "storage_cast": "float64_to_float32_once",
            },
            "limits": {
                "maximum_projected_shard_hours": 2.0,
                "maximum_projected_cpu_node_hours": 48.0,
                "passed": limits_ok,
            },
            "operational_override": (
                None if operational_override is None else dict(operational_override)
            ),
        }
    )


__all__ = [
    "ANGULAR_TREE_BACKEND_CONTRACT",
    "ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT",
    "ANGULAR_TREE_PROBE_CONTRACT",
    "ANGULAR_TREE_RESOURCE_CONTRACT",
    "ANGULAR_TREE_SHARD_CONTRACT",
    "ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT",
    "TREE_PROBE_STRATA",
    "TREE_SHARD_MAX_JETS",
    "build_angular_tree_resource_contract",
    "build_compiled_tree",
    "build_tree_probe_artifact",
    "finalize_tree_split",
    "load_tree_backend",
    "pack_tree_shard",
    "select_tree_probe",
    "unpack_tree_shard",
    "validate_backend_manifest",
    "write_tree_shard",
]
