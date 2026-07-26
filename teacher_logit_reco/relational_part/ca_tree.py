"""Step-1 resource contract for the future beam-free angular-tree backend."""

from __future__ import annotations

from typing import Any

from .contracts import require_sha256, with_content_hash


ANGULAR_TREE_RESOURCE_CONTRACT = "relational_part_angular_tree_resource_v1"
ANGULAR_TREE_BACKEND_CONTRACT = "relational_ca_tree_v1"


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
                    "-O3",
                    "-fno-fast-math",
                    "-fno-associative-math",
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
                "contract_id",
                "schema_version",
                "source_sha256",
                "binary_sha256",
                "compiler_identity",
                "compiler_major_version",
                "compiler_flags",
                "platform_architecture",
                "python_major_minor",
                "pytorch_version",
                "pytorch_cxx11_abi",
                "openmp_available",
                "self_test_sha256",
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
                "strata": [
                    [0, 8],
                    [9, 16],
                    [17, 24],
                    [25, 32],
                    [33, 40],
                    [41, 48],
                    [49, 64],
                    [65, 80],
                    [81, 96],
                    [97, 128],
                ],
                "initial_quota_per_stratum": 2_000,
                "selection_salt": "rpt_tree_probe_v1",
                "undersized_policy": "population_proportional_largest_remainder",
                "parity_sample_count": 1_000,
                "parity_salt": "rpt_tree_probe_parity_v1",
                "maximum_projected_shard_hours": 2.0,
                "maximum_projected_cpu_node_hours": 48.0,
            },
            "persistent_pair_matrices_allowed": False,
            "runtime_recomputation_from_hlt_only": True,
        }
    )


__all__ = [
    "ANGULAR_TREE_BACKEND_CONTRACT",
    "ANGULAR_TREE_RESOURCE_CONTRACT",
    "build_angular_tree_resource_contract",
]
