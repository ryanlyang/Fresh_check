"""Minimal pseudo-view tensor contract consumed by hierarchy-aware taggers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .config import canonical_hash


ABPH_CONSUMER_PSEUDO_CONTRACT = "adaptive_binary_pseudooffline_consumer_pseudo_v1"
ABPH_CONSUMER_PSEUDO_SCHEMA_VERSION = "1.0"

ABPH_CONSUMER_GLOBAL_FIELDS: tuple[str, ...] = (
    "shared_root_ledger",
    "hypothesis_latent",
    "hypothesis_prior_log_prob",
)
ABPH_CONSUMER_PARTICLE_FIELDS: tuple[str, ...] = (
    "canonical_features",
    "side_channels",
    "mask",
    "group_indices",
    "uncertainty",
)
ABPH_CONSUMER_FRONTIER_FIELDS: tuple[str, ...] = (
    "ledger",
    "hidden",
    "support",
    "uncertainty",
    "mask",
    "topology",
    "parent_indices",
)

ABPH_RENDERER_ONLY_PARTICLE_FIELDS: tuple[str, ...] = (
    "four_vector",
    "mass",
    "local_slot_indices",
    "slot_hidden",
)
ABPH_RENDERER_ONLY_FRONTIER_FIELDS: tuple[str, ...] = ("source_child_indices",)


def pseudo_array_key(
    hierarchy: str,
    category: str,
    field: str,
    depth: int | None = None,
) -> str:
    hierarchy_name = str(hierarchy)
    if not hierarchy_name or "__" in hierarchy_name:
        raise ValueError("hierarchy names must be nonempty and cannot contain '__'")
    if category == "particle":
        return f"particle__{hierarchy_name}__{field}"
    if category == "frontier":
        if depth is None or int(depth) < 0:
            raise ValueError("frontier keys require a nonnegative depth")
        return f"frontier__{hierarchy_name}__depth_{int(depth):02d}__{field}"
    raise ValueError(f"unknown pseudo array category {category!r}")


def consumer_pseudo_array_names(
    hierarchy_names: Sequence[str],
    frontier_depths: Mapping[str, int],
) -> tuple[str, ...]:
    names = list(ABPH_CONSUMER_GLOBAL_FIELDS)
    for hierarchy in hierarchy_names:
        depth_count = int(frontier_depths.get(str(hierarchy), 0))
        if depth_count <= 0:
            raise ValueError(f"hierarchy {hierarchy!r} has no frontier depths")
        names.extend(
            pseudo_array_key(hierarchy, "particle", field)
            for field in ABPH_CONSUMER_PARTICLE_FIELDS
        )
        for depth in range(depth_count):
            names.extend(
                pseudo_array_key(hierarchy, "frontier", field, depth)
                for field in ABPH_CONSUMER_FRONTIER_FIELDS
            )
    return tuple(names)


def renderer_only_array_names(
    hierarchy_names: Sequence[str],
    frontier_depths: Mapping[str, int],
) -> tuple[str, ...]:
    names: list[str] = []
    for hierarchy in hierarchy_names:
        names.extend(
            pseudo_array_key(hierarchy, "particle", field)
            for field in ABPH_RENDERER_ONLY_PARTICLE_FIELDS
        )
        for depth in range(int(frontier_depths.get(str(hierarchy), 0))):
            names.extend(
                pseudo_array_key(hierarchy, "frontier", field, depth)
                for field in ABPH_RENDERER_ONLY_FRONTIER_FIELDS
            )
    return tuple(names)


def project_consumer_pseudo_arrays(
    arrays: Mapping[str, Any],
    *,
    hierarchy_names: Sequence[str],
    frontier_depths: Mapping[str, int],
) -> dict[str, Any]:
    """Select tagger-read fields without copying or detaching their values."""

    required = consumer_pseudo_array_names(hierarchy_names, frontier_depths)
    missing = [name for name in required if name not in arrays]
    if missing:
        raise KeyError(f"pseudo object is missing consumer arrays: {missing}")
    return {name: arrays[name] for name in required}


def _array_schema(value: Any) -> dict[str, Any]:
    shape = tuple(int(item) for item in value.shape)
    if not shape:
        raise ValueError("pseudo arrays must have a batch axis")
    dtype = str(getattr(value, "dtype", "unknown"))
    if dtype.startswith("torch."):
        dtype = dtype[len("torch.") :]
    return {"dtype": dtype, "shape_after_batch": list(shape[1:])}


def consumer_pseudo_schema(
    arrays: Mapping[str, Any],
    *,
    hierarchy_names: Sequence[str],
    frontier_depths: Mapping[str, int],
) -> dict[str, Any]:
    projected = project_consumer_pseudo_arrays(
        arrays,
        hierarchy_names=hierarchy_names,
        frontier_depths=frontier_depths,
    )
    return {name: _array_schema(value) for name, value in sorted(projected.items())}


def consumer_pseudo_schema_hash(
    arrays: Mapping[str, Any],
    *,
    hierarchy_names: Sequence[str],
    frontier_depths: Mapping[str, int],
) -> str:
    return canonical_hash(
        {
            "contract": ABPH_CONSUMER_PSEUDO_CONTRACT,
            "schema_version": ABPH_CONSUMER_PSEUDO_SCHEMA_VERSION,
            "arrays": consumer_pseudo_schema(
                arrays,
                hierarchy_names=hierarchy_names,
                frontier_depths=frontier_depths,
            ),
        }
    )


def validate_consumer_pseudo_arrays(
    arrays: Mapping[str, Any],
    *,
    hierarchy_names: Sequence[str],
    frontier_depths: Mapping[str, int],
    exact: bool,
) -> dict[str, Any]:
    expected = set(consumer_pseudo_array_names(hierarchy_names, frontier_depths))
    observed = set(arrays)
    missing = sorted(expected - observed)
    if missing:
        raise KeyError(f"pseudo object is missing consumer arrays: {missing}")
    if exact and observed != expected:
        raise ValueError(
            "consumer-only pseudo object has unexpected arrays: "
            f"{sorted(observed - expected)}"
        )
    forbidden = sorted(observed & set(renderer_only_array_names(hierarchy_names, frontier_depths)))
    if exact and forbidden:
        raise ValueError(f"consumer-only pseudo object retains renderer-only arrays: {forbidden}")
    schema = consumer_pseudo_schema(
        arrays,
        hierarchy_names=hierarchy_names,
        frontier_depths=frontier_depths,
    )
    return {
        "ok": True,
        "contract": ABPH_CONSUMER_PSEUDO_CONTRACT,
        "schema": schema,
        "schema_hash": canonical_hash(
            {
                "contract": ABPH_CONSUMER_PSEUDO_CONTRACT,
                "schema_version": ABPH_CONSUMER_PSEUDO_SCHEMA_VERSION,
                "arrays": schema,
            }
        ),
        "retained_array_count": len(expected),
        "removed_renderer_only_arrays": list(
            renderer_only_array_names(hierarchy_names, frontier_depths)
        ),
    }

