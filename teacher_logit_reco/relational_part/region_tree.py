"""Deterministic float64 oracle for the beam-free exclusive angular tree."""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any, Mapping, Sequence

import numpy as np


TREE_SCHEMA_CONTRACT = "relational_ca_tree_packed_v1"
EXCLUSIVE_RESOLUTIONS = (2, 4, 8)
REFERENCE_RADIUS = 0.8
MAX_CONSTITUENTS = 128


def _canonical_float_bytes(value: float) -> bytes:
    numeric = float(value)
    if math.isnan(numeric):
        return bytes.fromhex("7ff8000000000000")
    if numeric == 0.0:
        numeric = 0.0
    return struct.pack(">d", numeric)


def canonical_leaf_key(vector: np.ndarray, raw_token: np.ndarray) -> bytes:
    values = np.concatenate(
        (
            np.asarray(vector, dtype=np.float64),
            np.asarray(raw_token, dtype=np.float64),
        )
    )
    return b"".join(_canonical_float_bytes(value) for value in values)


def _kinematics(vector: np.ndarray) -> tuple[float, float, float]:
    px, py, pz, energy = map(float, vector)
    pt = math.hypot(px, py)
    phi = math.atan2(py, px)
    eta = math.asinh(pz / max(pt, 1.0e-300))
    return pt, eta, phi


def _mass(vector: np.ndarray) -> float:
    px, py, pz, energy = map(float, vector)
    return math.sqrt(max(energy * energy - px * px - py * py - pz * pz, 0.0))


def _wrapped_delta_phi(left: float, right: float) -> float:
    return math.atan2(math.sin(left - right), math.cos(left - right))


def _merge_geometry(
    left_vector: np.ndarray,
    right_vector: np.ndarray,
) -> tuple[float, float, float]:
    left_pt, left_eta, left_phi = _kinematics(left_vector)
    right_pt, right_eta, right_phi = _kinematics(right_vector)
    delta_r = math.hypot(
        left_eta - right_eta,
        _wrapped_delta_phi(left_phi, right_phi),
    )
    kt = min(left_pt, right_pt) * delta_r
    z = min(left_pt, right_pt) / (left_pt + right_pt + 1.0e-6)
    return delta_r, kt, z


def build_reference_tree(
    vectors: np.ndarray,
    raw_tokens: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Build one canonical tree; all topology arithmetic is float64."""

    vector_array = np.asarray(vectors)
    token_array = np.asarray(raw_tokens)
    valid_mask = np.asarray(mask, dtype=bool)
    if vector_array.ndim != 2 or vector_array.shape[1] != 4:
        raise ValueError("vectors must have shape [particles,4]")
    if token_array.shape != (vector_array.shape[0], 14):
        raise ValueError("raw_tokens must have shape [particles,14]")
    if valid_mask.shape != (vector_array.shape[0],):
        raise ValueError("mask must have shape [particles]")
    n_valid = int(valid_mask.sum())
    if n_valid > MAX_CONSTITUENTS:
        raise ValueError("tree exceeds the locked 128-constituent limit")
    if not np.isfinite(vector_array[valid_mask]).all():
        raise FloatingPointError("valid tree vectors contain NaN or infinity")
    if not np.isfinite(token_array[valid_mask]).all():
        raise FloatingPointError("valid tree tokens contain NaN or infinity")
    length = int(vector_array.shape[0])
    if n_valid == 0:
        return {
            "contract": TREE_SCHEMA_CONTRACT,
            "n_particles": length,
            "n_valid": 0,
            "n_nodes": 0,
            "root": -1,
            "leaf_to_node": np.full(length, -1, dtype=np.int32),
            "parent": np.empty(0, dtype=np.int32),
            "left": np.empty(0, dtype=np.int32),
            "right": np.empty(0, dtype=np.int32),
            "depth": np.empty(0, dtype=np.int32),
            "vectors": np.empty((0, 4), dtype=np.float32),
            "pt": np.empty(0, dtype=np.float32),
            "mass": np.empty(0, dtype=np.float32),
            "multiplicity": np.empty(0, dtype=np.int32),
            "merge_delta_r": np.empty(0, dtype=np.float32),
            "merge_kt": np.empty(0, dtype=np.float32),
            "merge_z": np.empty(0, dtype=np.float32),
            "merge_mass": np.empty(0, dtype=np.float32),
            "assignments": {
                str(k): np.full(length, -1, dtype=np.int32)
                for k in EXCLUSIVE_RESOLUTIONS
            },
            "actual_cluster_counts": {str(k): 0 for k in EXCLUSIVE_RESOLUTIONS},
        }

    valid_indices = np.flatnonzero(valid_mask)
    key_rows = [
        (
            canonical_leaf_key(vector_array[index], token_array[index]),
            int(index),
        )
        for index in valid_indices
    ]
    key_rows.sort(key=lambda row: row[0])
    ordered_original = [row[1] for row in key_rows]
    leaf_keys = [row[0] for row in key_rows]
    node_vectors: list[np.ndarray] = [
        np.asarray(vector_array[index], dtype=np.float64).copy()
        for index in ordered_original
    ]
    parent = [-1] * n_valid
    left = [-1] * n_valid
    right = [-1] * n_valid
    multiplicity = [1] * n_valid
    merge_delta_r = [0.0] * n_valid
    merge_kt = [0.0] * n_valid
    merge_z = [0.0] * n_valid
    node_leaf_keys: list[tuple[bytes, ...]] = [(key,) for key in leaf_keys]
    active = list(range(n_valid))
    while len(active) > 1:
        best: tuple[Any, ...] | None = None
        chosen: tuple[int, int, float, float, float] | None = None
        for position, first in enumerate(active[:-1]):
            first_pt, first_eta, first_phi = _kinematics(node_vectors[first])
            for second in active[position + 1:]:
                second_pt, second_eta, second_phi = _kinematics(
                    node_vectors[second]
                )
                delta_r = math.hypot(
                    first_eta - second_eta,
                    _wrapped_delta_phi(first_phi, second_phi),
                )
                first_key = node_leaf_keys[first]
                second_key = node_leaf_keys[second]
                ordered_keys = (
                    (first_key, second_key)
                    if first_key <= second_key
                    else (second_key, first_key)
                )
                candidate = (delta_r * delta_r, *ordered_keys)
                if best is None or candidate < best:
                    best = candidate
                    kt = min(first_pt, second_pt) * delta_r
                    z = min(first_pt, second_pt) / (
                        first_pt + second_pt + 1.0e-6
                    )
                    chosen = (first, second, delta_r, kt, z)
        if chosen is None:  # pragma: no cover - active length proves otherwise
            raise AssertionError("tree merge selection failed")
        first, second, delta_r, kt, z = chosen
        new_index = len(node_vectors)
        node_vectors.append(node_vectors[first] + node_vectors[second])
        parent.extend([-1])
        left.append(first)
        right.append(second)
        multiplicity.append(multiplicity[first] + multiplicity[second])
        merge_delta_r.append(delta_r)
        merge_kt.append(kt)
        merge_z.append(z)
        node_leaf_keys.append(
            tuple(sorted((*node_leaf_keys[first], *node_leaf_keys[second])))
        )
        parent[first] = new_index
        parent[second] = new_index
        active = [node for node in active if node not in (first, second)]
        active.append(new_index)
    root = active[0]
    n_nodes = len(node_vectors)
    depth = np.full(n_nodes, -1, dtype=np.int32)
    depth[root] = 0
    stack = [root]
    while stack:
        node = stack.pop()
        for child in (left[node], right[node]):
            if child >= 0:
                depth[child] = depth[node] + 1
                stack.append(child)
    leaf_to_node = np.full(length, -1, dtype=np.int32)
    for leaf_node, original in enumerate(ordered_original):
        leaf_to_node[original] = leaf_node
    vectors64 = np.stack(node_vectors)
    pt64 = np.asarray([_kinematics(vector)[0] for vector in vectors64])
    mass64 = np.asarray([_mass(vector) for vector in vectors64])
    assignments: dict[str, np.ndarray] = {}
    actual_counts: dict[str, int] = {}
    for requested in EXCLUSIVE_RESOLUTIONS:
        count = min(requested, n_valid)
        clusters = {root}
        while len(clusters) < count:
            split = max(node for node in clusters if left[node] >= 0)
            clusters.remove(split)
            clusters.update((left[split], right[split]))
        assignment = np.full(length, -1, dtype=np.int32)
        for original in valid_indices:
            node = int(leaf_to_node[original])
            while node not in clusters:
                node = parent[node]
            assignment[original] = node
        assignments[str(requested)] = assignment
        actual_counts[str(requested)] = count
    return {
        "contract": TREE_SCHEMA_CONTRACT,
        "n_particles": length,
        "n_valid": n_valid,
        "n_nodes": n_nodes,
        "root": root,
        "leaf_to_node": leaf_to_node,
        "parent": np.asarray(parent, dtype=np.int32),
        "left": np.asarray(left, dtype=np.int32),
        "right": np.asarray(right, dtype=np.int32),
        "depth": depth,
        "vectors": vectors64.astype(np.float32),
        "pt": pt64.astype(np.float32),
        "mass": mass64.astype(np.float32),
        "multiplicity": np.asarray(multiplicity, dtype=np.int32),
        "merge_delta_r": np.asarray(merge_delta_r, dtype=np.float32),
        "merge_kt": np.asarray(merge_kt, dtype=np.float32),
        "merge_z": np.asarray(merge_z, dtype=np.float32),
        "merge_mass": mass64.astype(np.float32),
        "assignments": assignments,
        "actual_cluster_counts": actual_counts,
    }


def validate_tree(tree: Mapping[str, Any]) -> None:
    if tree.get("contract") != TREE_SCHEMA_CONTRACT:
        raise ValueError("tree schema contract differs")
    n_valid = int(tree.get("n_valid", -1))
    n_nodes = int(tree.get("n_nodes", -1))
    if n_valid < 0 or n_nodes != (0 if n_valid == 0 else 2 * n_valid - 1):
        raise ValueError("tree node count is inconsistent")
    length = int(tree.get("n_particles", -1))
    if np.asarray(tree.get("leaf_to_node")).shape != (length,):
        raise ValueError("tree leaf map shape differs")
    for name in ("parent", "left", "right", "depth", "pt", "mass", "multiplicity",
                 "merge_delta_r", "merge_kt", "merge_z", "merge_mass"):
        values = np.asarray(tree.get(name))
        if values.shape != (n_nodes,):
            raise ValueError(f"tree {name} shape differs")
        expected_dtype = (
            np.dtype(np.int32)
            if name in ("parent", "left", "right", "depth", "multiplicity")
            else np.dtype(np.float32)
        )
        if values.dtype != expected_dtype:
            raise TypeError(f"tree {name} dtype differs")
        if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
            raise FloatingPointError(f"tree {name} contains NaN or infinity")
    vectors = np.asarray(tree.get("vectors"))
    if vectors.shape != (n_nodes, 4):
        raise ValueError("tree vector shape differs")
    if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise TypeError("tree vector storage must be finite float32")
    if np.asarray(tree["leaf_to_node"]).dtype != np.int32:
        raise TypeError("tree leaf map dtype differs")
    assignments = tree.get("assignments")
    if not isinstance(assignments, Mapping) or set(assignments) != {
        str(k) for k in EXCLUSIVE_RESOLUTIONS
    }:
        raise ValueError("tree assignments differ")
    for values in assignments.values():
        if np.asarray(values).shape != (length,):
            raise ValueError("tree assignment shape differs")
        if np.asarray(values).dtype != np.int32:
            raise TypeError("tree assignment dtype differs")


def tree_content_sha256(tree: Mapping[str, Any]) -> str:
    validate_tree(tree)
    digest = hashlib.sha256()
    for field in (
        "n_particles", "n_valid", "n_nodes", "root",
    ):
        digest.update(struct.pack(">q", int(tree[field])))
    for name in (
        "leaf_to_node", "parent", "left", "right", "depth", "vectors", "pt",
        "mass", "multiplicity", "merge_delta_r", "merge_kt", "merge_z",
        "merge_mass",
    ):
        values = np.ascontiguousarray(tree[name])
        digest.update(name.encode())
        digest.update(str(values.dtype).encode())
        digest.update(np.asarray(values.shape, dtype=">i8").tobytes())
        digest.update(values.tobytes())
    for resolution in EXCLUSIVE_RESOLUTIONS:
        digest.update(np.ascontiguousarray(
            tree["assignments"][str(resolution)]
        ).tobytes())
    return digest.hexdigest()


def build_reference_trees(
    vectors: np.ndarray,
    raw_tokens: np.ndarray,
    mask: np.ndarray,
) -> list[dict[str, Any]]:
    vector_array = np.asarray(vectors)
    if vector_array.ndim != 3 or vector_array.shape[2] != 4:
        raise ValueError("vectors must have shape [jets,particles,4]")
    return [
        build_reference_tree(vector_array[row], raw_tokens[row], mask[row])
        for row in range(int(vector_array.shape[0]))
    ]


__all__ = [
    "EXCLUSIVE_RESOLUTIONS",
    "MAX_CONSTITUENTS",
    "REFERENCE_RADIUS",
    "TREE_SCHEMA_CONTRACT",
    "build_reference_tree",
    "build_reference_trees",
    "canonical_leaf_key",
    "tree_content_sha256",
    "validate_tree",
]
