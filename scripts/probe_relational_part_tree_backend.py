#!/usr/bin/env python3
"""Run the locked angular-tree throughput, storage, and parity probe."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import resource
except ImportError:  # pragma: no cover - Windows development host
    resource = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_cache import (  # noqa: E402
    jet_identity_hash,
    load_cached_hlt_view,
)
from jetclass_fresh.part_inputs import (  # noqa: E402
    build_particle_transformer_inputs_from_tokens,
)
from teacher_logit_reco.relational_part import (  # noqa: E402
    ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT,
    ANGULAR_TREE_RESOURCE_CONTRACT,
    RELATIONAL_HLT_BINDING_CONTRACT,
    RELATIONAL_STORAGE_PROJECTION_CONTRACT,
    bind_source_provenance,
    build_compiled_tree,
    build_reference_tree,
    build_tree_probe_artifact,
    load_hashed_json,
    load_tree_backend,
    pack_tree_shard,
    select_tree_probe,
    source_snapshot,
    write_immutable_json,
)


_TOPOLOGY_ARRAYS = (
    "leaf_to_node",
    "parent",
    "left",
    "right",
    "depth",
    "multiplicity",
)
_CONTINUOUS_ARRAYS = (
    "vectors",
    "pt",
    "mass",
    "merge_delta_r",
    "merge_kt",
    "merge_z",
    "merge_mass",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--hlt-binding", type=Path, required=True)
    parser.add_argument("--tree-resource", type=Path, required=True)
    parser.add_argument("--backend-manifest", type=Path, required=True)
    parser.add_argument("--backend-binary", type=Path, required=True)
    parser.add_argument("--storage-projection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=20_000)
    parser.add_argument(
        "--operational-override-json",
        type=Path,
        help=(
            "Explicit {reason, authorized_by, recorded_at_utc} override for "
            "throughput/storage limits; numerical parity remains mandatory."
        ),
    )
    parser.add_argument("--miniature", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _compressed_bytes(
    trees: Sequence[Mapping[str, Any]],
    identities: Sequence[Any],
) -> int:
    packed = pack_tree_shard(trees, identities)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **packed)
    return len(buffer.getbuffer())


def _parity(
    compiled: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> tuple[bool, float]:
    exact = all(
        int(compiled[name]) == int(reference[name])
        for name in ("n_particles", "n_valid", "n_nodes", "root")
    )
    exact = exact and all(
        np.array_equal(compiled[name], reference[name])
        for name in _TOPOLOGY_ARRAYS
    )
    exact = exact and compiled["actual_cluster_counts"] == reference[
        "actual_cluster_counts"
    ]
    exact = exact and all(
        np.array_equal(
            compiled["assignments"][str(resolution)],
            reference["assignments"][str(resolution)],
        )
        for resolution in (2, 4, 8)
    )
    errors = []
    for name in _CONTINUOUS_ARRAYS:
        left = np.asarray(compiled[name], dtype=np.float64)
        right = np.asarray(reference[name], dtype=np.float64)
        if left.shape != right.shape:
            return False, float("inf")
        if left.size:
            errors.append(float(np.max(np.abs(left - right))))
    return exact, max(errors, default=0.0)


def _peak_resident_bytes() -> int:
    if resource is None:  # pragma: no cover - Windows development host
        return 0
    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes. Production Tigris is Linux.
    return maximum * 1024 if sys.platform.startswith("linux") else maximum


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    binding = load_hashed_json(
        args.hlt_binding, expected_contract=RELATIONAL_HLT_BINDING_CONTRACT
    )
    resource_contract = load_hashed_json(
        args.tree_resource, expected_contract=ANGULAR_TREE_RESOURCE_CONTRACT
    )
    backend_manifest = load_hashed_json(
        args.backend_manifest,
        expected_contract=ANGULAR_TREE_BACKEND_MANIFEST_CONTRACT,
    )
    storage = load_hashed_json(
        args.storage_projection,
        expected_contract=RELATIONAL_STORAGE_PROJECTION_CONTRACT,
    )
    report = binding["split_reports"]["model_train"]
    view = load_cached_hlt_view(args.cache_dir, "model_train", verify_hash=True)
    if (
        view.metadata.get("hlt_content_hash") != report["hlt_content_hash"]
        or jet_identity_hash(view.jet_ids) != report["jet_identity_hash"]
    ):
        raise ValueError("tree probe HLT cache differs from its binding")
    requested = int(args.sample_count)
    if args.miniature:
        requested = min(requested, len(view.jet_ids))
    if requested <= 0:
        raise ValueError("tree probe sample count must be positive")
    selection = select_tree_probe(
        view.jet_ids,
        view.mask.sum(axis=1),
        sample_count=requested,
    )
    source = (
        REPO_ROOT
        / "teacher_logit_reco"
        / "relational_part"
        / "csrc"
        / "relational_ca_tree_v1.cpp"
    )
    backend = load_tree_backend(
        args.backend_binary, args.backend_manifest, source_path=source
    )
    particle_inputs = build_particle_transformer_inputs_from_tokens(
        view.tokens, view.mask, source_view="fixed_hlt"
    )
    vectors = particle_inputs.pf_vectors.transpose(0, 2, 1)
    parity_indices = {
        int(value) for value in selection["parity_indices"].tolist()
    }
    elapsed_ms = []
    persisted_bytes = np.zeros(
        len(selection["selected_indices"]), dtype=np.float64
    )
    storage_buffers: dict[
        int, list[tuple[int, Mapping[str, Any], Any]]
    ] = {index: [] for index in range(10)}

    def flush_storage(stratum: int) -> None:
        buffered = storage_buffers[stratum]
        if not buffered:
            return
        total_bytes = _compressed_bytes(
            [row[1] for row in buffered],
            [row[2] for row in buffered],
        )
        per_jet = float(total_bytes) / len(buffered)
        for position, _, _ in buffered:
            persisted_bytes[position] = per_jet
        buffered.clear()

    topology_exact = True
    maximum_error = 0.0
    valid_counts = view.mask.sum(axis=1)
    for position, raw_index in enumerate(
        selection["selected_indices"].tolist()
    ):
        index = int(raw_index)
        started = time.perf_counter_ns()
        tree = build_compiled_tree(
            backend, vectors[index], view.tokens[index], view.mask[index]
        )
        elapsed_ms.append((time.perf_counter_ns() - started) / 1.0e6)
        count = int(valid_counts[index])
        stratum = next(
            row
            for row, (lower, upper) in enumerate(
                (
                    (0, 8),
                    (9, 16),
                    (17, 24),
                    (25, 32),
                    (33, 40),
                    (41, 48),
                    (49, 64),
                    (65, 80),
                    (81, 96),
                    (97, 128),
                )
            )
            if lower <= count <= upper
        )
        storage_buffers[stratum].append(
            (position, tree, view.jet_ids[index])
        )
        if len(storage_buffers[stratum]) >= 256:
            flush_storage(stratum)
        if index in parity_indices:
            reference = build_reference_tree(
                vectors[index], view.tokens[index], view.mask[index]
            )
            exact, error = _parity(tree, reference)
            topology_exact = topology_exact and exact
            maximum_error = max(maximum_error, error)
    for stratum in range(10):
        flush_storage(stratum)
    if bool((persisted_bytes <= 0).any()):
        raise RuntimeError("tree probe storage measurement is incomplete")
    total_campaign_jets = 60 if args.miniature else 1_750_000
    operational_override = None
    if args.operational_override_json is not None:
        operational_override = json.loads(
            args.operational_override_json.read_text(encoding="utf-8")
        )
        if not isinstance(operational_override, dict):
            raise ValueError("operational override JSON must contain an object")
    artifact = build_tree_probe_artifact(
        selection,
        valid_counts,
        elapsed_ms,
        persisted_bytes,
        peak_resident_bytes=_peak_resident_bytes(),
        parity_topology_exact=topology_exact,
        parity_max_continuous_absolute_error=maximum_error,
        total_campaign_jets=total_campaign_jets,
        hlt_content_sha256=report["hlt_content_hash"],
        tree_resource_sha256=resource_contract["content_hash"],
        backend_manifest_sha256=backend_manifest["content_hash"],
        storage_projection=storage,
        storage_measurement_policy=(
            "npz_compressed_same_stratum_chunks_up_to_256_jets_"
            "including_metadata_arrays"
        ),
        operational_override=operational_override,
    )
    artifact = bind_source_provenance(
        artifact, source_snapshot=source_snapshot(REPO_ROOT)
    )
    publication = None
    if not args.dry_run:
        publication = write_immutable_json(args.output, artifact)
    print(
        json.dumps(
            {
                "dry_run": bool(args.dry_run),
                "resolved_configuration": {
                    "sample_count": requested,
                    "production_sample_count": 20_000,
                    "miniature": bool(args.miniature),
                    "output": str(args.output.resolve()),
                },
                "artifact": artifact,
                "publication": publication,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
