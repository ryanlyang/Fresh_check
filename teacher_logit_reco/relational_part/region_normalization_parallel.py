"""Authenticated map/reduce contracts for REGION normalization."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import require_sha256, validate_content_hash, with_content_hash
from .normalization import (
    NORMALIZATION_JET_LIMIT,
    NORMALIZATION_JET_SALT,
    _identity_key,
    _identity_sequence_hash,
)
from .region_normalization import _REGION_DOMAIN_FEATURE_NAMES


REGION_NORMALIZATION_PLAN_CONTRACT = (
    "relational_part_region_normalization_plan_v1"
)
REGION_NORMALIZATION_PARTIAL_CONTRACT = (
    "relational_part_region_normalization_partial_v1"
)
REGION_SAMPLE_DOMAINS = tuple(_REGION_DOMAIN_FEATURE_NAMES)
_PLAN_ROW_KEYS = {
    "shard_index",
    "shard_jet_count",
    "global_start",
    "global_stop",
    "selected_count",
    "selected_local_indices",
    "selection_ranks",
    "selected_identity_sha256",
    "selected_input_filename",
    "selected_input_npz_sha256",
    "tree_shard_metadata_sha256",
}


def build_region_normalization_plan(
    *,
    tree_manifest_sha256: str,
    tree_resource_sha256: str,
    relation_normalization_sha256: str,
    hlt_content_sha256: str,
    selected_identities: Sequence[Any],
    shard_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe immutable selected-HLT inputs for every physical tree shard."""

    rows = [dict(row) for row in shard_rows]
    if not rows or [int(row.get("shard_index", -1)) for row in rows] != list(
        range(len(rows))
    ):
        raise ValueError("REGION plan shard indices are not contiguous")
    selected_count = sum(int(row.get("selected_count", -1)) for row in rows)
    if selected_count != len(selected_identities):
        raise ValueError("REGION plan selected coverage differs")
    selection_ranks = sorted(
        int(rank)
        for row in rows
        for rank in row.get("selection_ranks", ())
    )
    if selection_ranks != list(range(selected_count)):
        raise ValueError("REGION plan selection ranks are incomplete")
    expected_global_start = 0
    for shard_index, row in enumerate(rows):
        count = int(row.get("selected_count", -1))
        local = [int(value) for value in row.get("selected_local_indices", ())]
        ranks = [int(value) for value in row.get("selection_ranks", ())]
        shard_jet_count = int(row.get("shard_jet_count", -1))
        global_start = int(row.get("global_start", -1))
        global_stop = int(row.get("global_stop", -1))
        if (
            count < 0
            or len(local) != count
            or len(ranks) != count
            or local != sorted(local)
            or len(local) != len(set(local))
            or shard_jet_count < 0
            or global_start != expected_global_start
            or global_stop - global_start != shard_jet_count
            or any(value < 0 or value >= shard_jet_count for value in local)
            or row.get("selected_input_filename")
            != f"shard_{shard_index:05d}.npz"
            or set(row) != _PLAN_ROW_KEYS
        ):
            raise ValueError("REGION plan shard selection differs")
        expected_global_start = global_stop
        require_sha256(
            row.get("selected_input_npz_sha256"),
            name="selected_input_npz_sha256",
        )
        require_sha256(
            row.get("selected_identity_sha256"),
            name="selected_identity_sha256",
        )
        require_sha256(
            row.get("tree_shard_metadata_sha256"),
            name="tree_shard_metadata_sha256",
        )
    return with_content_hash(
        {
            "contract": REGION_NORMALIZATION_PLAN_CONTRACT,
            "schema_version": 1,
            "fit_split": "model_train",
            "selection_policy": {
                "jet_limit": NORMALIZATION_JET_LIMIT,
                "jet_salt": NORMALIZATION_JET_SALT,
                "canonical_reduction_order": "salted_selection_rank",
                "physical_map_partition": "tree_shard_index",
            },
            "parents": {
                "tree_manifest_sha256": require_sha256(
                    tree_manifest_sha256, name="tree_manifest_sha256"
                ),
                "tree_resource_sha256": require_sha256(
                    tree_resource_sha256, name="tree_resource_sha256"
                ),
                "relation_normalization_sha256": require_sha256(
                    relation_normalization_sha256,
                    name="relation_normalization_sha256",
                ),
                "hlt_content_sha256": require_sha256(
                    hlt_content_sha256, name="hlt_content_sha256"
                ),
            },
            "selected_jet_count": selected_count,
            "selected_jet_identity_sha256": _identity_sequence_hash(
                [_identity_key(identity) for identity in selected_identities]
            ),
            "shard_count": len(rows),
            "shards": rows,
        }
    )


def validate_region_normalization_plan(
    artifact: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(
        artifact, expected_contract=REGION_NORMALIZATION_PLAN_CONTRACT
    )
    if int(artifact.get("schema_version", -1)) != 1:
        raise ValueError("REGION plan schema version differs")
    expected_keys = {
        "contract",
        "schema_version",
        "fit_split",
        "selection_policy",
        "parents",
        "selected_jet_count",
        "selected_jet_identity_sha256",
        "shard_count",
        "shards",
        "content_hash",
    }
    if "source" in artifact:
        expected_keys.add("source")
    if set(artifact) != expected_keys or set(artifact.get("parents", ())) != {
        "tree_manifest_sha256",
        "tree_resource_sha256",
        "relation_normalization_sha256",
        "hlt_content_sha256",
    }:
        raise ValueError("REGION plan fields differ")
    if artifact.get("fit_split") != "model_train":
        raise ValueError("REGION plan fit split differs")
    policy = artifact.get("selection_policy")
    if policy != {
        "jet_limit": NORMALIZATION_JET_LIMIT,
        "jet_salt": NORMALIZATION_JET_SALT,
        "canonical_reduction_order": "salted_selection_rank",
        "physical_map_partition": "tree_shard_index",
    }:
        raise ValueError("REGION plan selection policy differs")
    build_region_normalization_plan(
        tree_manifest_sha256=artifact["parents"]["tree_manifest_sha256"],
        tree_resource_sha256=artifact["parents"]["tree_resource_sha256"],
        relation_normalization_sha256=artifact["parents"][
            "relation_normalization_sha256"
        ],
        hlt_content_sha256=artifact["parents"]["hlt_content_sha256"],
        selected_identities=[
            f"rank-{index}"
            for index in range(int(artifact.get("selected_jet_count", -1)))
        ],
        shard_rows=artifact.get("shards", ()),
    )
    require_sha256(
        artifact.get("selected_jet_identity_sha256"),
        name="selected_jet_identity_sha256",
    )
    if int(artifact.get("shard_count", -1)) != len(artifact.get("shards", ())):
        raise ValueError("REGION plan shard count differs")
    return digest


def build_region_normalization_partial(
    *,
    plan: Mapping[str, Any],
    shard_index: int,
    tree_shard_metadata_sha256: str,
    sample_npz_sha256: str,
    selected_count: int,
    sample_counts: Mapping[str, int],
    sample_identity_sha256: Mapping[str, str],
) -> dict[str, Any]:
    plan_sha = validate_region_normalization_plan(plan)
    shard_index = int(shard_index)
    if shard_index < 0 or shard_index >= len(plan["shards"]):
        raise ValueError("REGION partial shard index is out of range")
    plan_row = plan["shards"][shard_index]
    if int(selected_count) != int(plan_row["selected_count"]):
        raise ValueError("REGION partial selected count differs from plan")
    counts = {
        domain: int(sample_counts.get(domain, -1))
        for domain in REGION_SAMPLE_DOMAINS
    }
    if set(sample_counts) != set(REGION_SAMPLE_DOMAINS) or any(
        value < 0 for value in counts.values()
    ):
        raise ValueError("REGION partial sample counts differ")
    hashes = {
        domain: require_sha256(
            sample_identity_sha256.get(domain),
            name=f"{domain}.sample_identity_sha256",
        )
        for domain in REGION_SAMPLE_DOMAINS
    }
    if set(sample_identity_sha256) != set(REGION_SAMPLE_DOMAINS):
        raise ValueError("REGION partial sample hash domains differ")
    tree_metadata_sha = require_sha256(
        tree_shard_metadata_sha256,
        name="tree_shard_metadata_sha256",
    )
    if tree_metadata_sha != plan_row["tree_shard_metadata_sha256"]:
        raise ValueError("REGION partial tree shard parent differs from plan")
    return with_content_hash(
        {
            "contract": REGION_NORMALIZATION_PARTIAL_CONTRACT,
            "schema_version": 1,
            "fit_split": "model_train",
            "shard_index": shard_index,
            "selected_count": int(selected_count),
            "sample_counts": counts,
            "sample_identity_sha256": hashes,
            "parents": {
                "plan_sha256": plan_sha,
                "selected_input_npz_sha256": plan_row[
                    "selected_input_npz_sha256"
                ],
                "tree_shard_metadata_sha256": require_sha256(
                    tree_metadata_sha, name="tree_shard_metadata_sha256",
                ),
                "sample_npz_sha256": require_sha256(
                    sample_npz_sha256, name="sample_npz_sha256"
                ),
            },
        }
    )


def validate_region_normalization_partial(
    artifact: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    shard_index: int,
) -> str:
    digest = validate_content_hash(
        artifact, expected_contract=REGION_NORMALIZATION_PARTIAL_CONTRACT
    )
    if int(artifact.get("schema_version", -1)) != 1:
        raise ValueError("REGION partial schema version differs")
    expected_keys = {
        "contract",
        "schema_version",
        "fit_split",
        "shard_index",
        "selected_count",
        "sample_counts",
        "sample_identity_sha256",
        "parents",
        "content_hash",
    }
    if "source" in artifact:
        expected_keys.add("source")
    if set(artifact) != expected_keys or set(artifact.get("parents", ())) != {
        "plan_sha256",
        "selected_input_npz_sha256",
        "tree_shard_metadata_sha256",
        "sample_npz_sha256",
    }:
        raise ValueError("REGION partial fields differ")
    rebuilt = build_region_normalization_partial(
        plan=plan,
        shard_index=shard_index,
        tree_shard_metadata_sha256=artifact["parents"][
            "tree_shard_metadata_sha256"
        ],
        sample_npz_sha256=artifact["parents"]["sample_npz_sha256"],
        selected_count=artifact["selected_count"],
        sample_counts=artifact["sample_counts"],
        sample_identity_sha256=artifact["sample_identity_sha256"],
    )
    for key in (
        "fit_split",
        "shard_index",
        "selected_count",
        "sample_counts",
        "sample_identity_sha256",
        "parents",
    ):
        if artifact.get(key) != rebuilt.get(key):
            raise ValueError(f"REGION partial {key} differs")
    return digest


def _update_digest(digest: Any, value: str) -> None:
    digest.update(value.encode("utf-8"))
    digest.update(b"\n")


def validate_region_normalization_partial_arrays(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, str]:
    selected_count = int(metadata["selected_count"])
    identities = np.asarray(arrays["identity"])
    ranks = np.asarray(arrays["selection_rank"])
    if (
        identities.ndim != 1
        or ranks.dtype != np.int64
        or ranks.shape != (selected_count,)
        or identities.shape != (selected_count,)
        or len(set(map(int, ranks.tolist()))) != selected_count
    ):
        raise ValueError("REGION partial identity/rank arrays differ")
    digests = {domain: hashlib.sha256() for domain in REGION_SAMPLE_DOMAINS}
    coordinate_names = {
        "REGION_pair": ("REGION_pair_query", "REGION_pair_context"),
        "REGION_merge": ("REGION_merge_query", "REGION_merge_context"),
        "REGION_node": ("REGION_node_index",),
    }
    for domain, names in _REGION_DOMAIN_FEATURE_NAMES.items():
        samples = np.asarray(arrays[f"{domain}_samples"])
        offsets = np.asarray(arrays[f"{domain}_offsets"])
        if (
            samples.dtype != np.float64
            or samples.ndim != 2
            or samples.shape[1] != len(names)
            or offsets.dtype != np.int64
            or offsets.shape != (selected_count + 1,)
            or int(offsets[0]) != 0
            or int(offsets[-1]) != samples.shape[0]
            or np.any(offsets[1:] < offsets[:-1])
            or not np.isfinite(samples).all()
            or samples.shape[0] != int(metadata["sample_counts"][domain])
        ):
            raise ValueError(f"{domain} partial sample layout differs")
        coordinates = [
            np.asarray(arrays[name]) for name in coordinate_names[domain]
        ]
        if any(
            values.dtype != np.int16 or values.shape != (samples.shape[0],)
            for values in coordinates
        ):
            raise ValueError(f"{domain} partial coordinates differ")
        for jet_row in range(selected_count):
            identity = str(identities[jet_row])
            start = int(offsets[jet_row])
            stop = int(offsets[jet_row + 1])
            for sample_row in range(start, stop):
                if domain == "REGION_node":
                    index = int(coordinates[0][sample_row])
                    suffix = f"{index}>{index}"
                else:
                    suffix = (
                        f"{int(coordinates[0][sample_row])}>"
                        f"{int(coordinates[1][sample_row])}"
                    )
                _update_digest(
                    digests[domain],
                    f"{identity}#{domain}:{suffix}",
                )
    actual = {
        domain: digest.hexdigest() for domain, digest in digests.items()
    }
    if actual != metadata["sample_identity_sha256"]:
        raise ValueError("REGION partial sample identity hashes differ")
    return actual


def assemble_region_normalization_partials(
    plan: Mapping[str, Any],
    partials: Sequence[tuple[Mapping[str, Any], Mapping[str, np.ndarray]]],
) -> tuple[dict[str, np.ndarray], dict[str, str], list[str]]:
    """Reassemble partial samples in the original salted jet order."""

    validate_region_normalization_plan(plan)
    if len(partials) != int(plan["shard_count"]):
        raise ValueError("REGION partial coverage differs from plan")
    by_rank: list[tuple[Mapping[str, np.ndarray], int] | None] = [
        None
    ] * int(plan["selected_jet_count"])
    for shard_index, (metadata, arrays) in enumerate(partials):
        validate_region_normalization_partial(
            metadata, plan=plan, shard_index=shard_index
        )
        validate_region_normalization_partial_arrays(arrays, metadata)
        if metadata.get("source") != plan.get("source"):
            raise ValueError("REGION partial source differs from plan")
        ranks = np.asarray(arrays["selection_rank"])
        identities_array = np.asarray(arrays["identity"])
        plan_row = plan["shards"][shard_index]
        if (
            ranks.tolist() != plan_row["selection_ranks"]
            or _identity_sequence_hash(
                [str(value) for value in identities_array]
            )
            != plan_row["selected_identity_sha256"]
        ):
            raise ValueError("REGION partial selection differs from plan")
        for local_row, rank_value in enumerate(ranks):
            rank = int(rank_value)
            if rank < 0 or rank >= len(by_rank) or by_rank[rank] is not None:
                raise ValueError("REGION partial selection ranks overlap")
            by_rank[rank] = (arrays, local_row)
    if any(entry is None for entry in by_rank):
        raise ValueError("REGION partial selection ranks are incomplete")

    chunks = {domain: [] for domain in REGION_SAMPLE_DOMAINS}
    digests = {domain: hashlib.sha256() for domain in REGION_SAMPLE_DOMAINS}
    identities: list[str] = []
    for entry in by_rank:
        if entry is None:  # pragma: no cover - guarded above
            raise AssertionError("unreachable incomplete REGION rank")
        arrays, local_row = entry
        identity = str(np.asarray(arrays["identity"])[local_row])
        identities.append(identity)
        for domain in REGION_SAMPLE_DOMAINS:
            offsets = np.asarray(arrays[f"{domain}_offsets"])
            start = int(offsets[local_row])
            stop = int(offsets[local_row + 1])
            chunks[domain].append(
                np.asarray(arrays[f"{domain}_samples"])[start:stop]
            )
            if domain == "REGION_node":
                indices = np.asarray(arrays["REGION_node_index"])
                for sample_row in range(start, stop):
                    index = int(indices[sample_row])
                    suffix = f"{index}>{index}"
                    _update_digest(
                        digests[domain],
                        f"{identity}#{domain}:{suffix}",
                    )
            else:
                queries = np.asarray(arrays[f"{domain}_query"])
                contexts = np.asarray(arrays[f"{domain}_context"])
                for sample_row in range(start, stop):
                    _update_digest(
                        digests[domain],
                        (
                            f"{identity}#{domain}:"
                            f"{int(queries[sample_row])}>"
                            f"{int(contexts[sample_row])}"
                        ),
                    )
    samples = {
        domain: np.concatenate(chunks[domain], axis=0)
        for domain in REGION_SAMPLE_DOMAINS
    }
    hashes = {
        domain: digest.hexdigest() for domain, digest in digests.items()
    }
    if _identity_sequence_hash(identities) != plan[
        "selected_jet_identity_sha256"
    ]:
        raise ValueError("REGION reconstructed selected identity hash differs")
    return samples, hashes, identities


__all__ = [
    "REGION_NORMALIZATION_PARTIAL_CONTRACT",
    "REGION_NORMALIZATION_PLAN_CONTRACT",
    "REGION_SAMPLE_DOMAINS",
    "assemble_region_normalization_partials",
    "build_region_normalization_partial",
    "build_region_normalization_plan",
    "validate_region_normalization_partial",
    "validate_region_normalization_partial_arrays",
    "validate_region_normalization_plan",
]
