#!/usr/bin/env python3
"""Build source-evidence-bound storage measurements for campaign bootstrap."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.part_inputs import (  # noqa: E402
    build_particle_transformer_inputs_from_tokens,
)
from teacher_logit_reco.relational_part import (  # noqa: E402
    StorageMeasurements,
    build_reference_tree,
    build_storage_measurements,
    canonical_sha256,
    pack_tree_shard,
    sha256_file,
    write_immutable_json,
)
from teacher_logit_reco.relational_part.evaluation import (  # noqa: E402
    write_final_predictions,
)


def _build_tree(payload: tuple[np.ndarray, np.ndarray, np.ndarray]):
    vectors, tokens, mask = payload
    return build_reference_tree(vectors, tokens, mask)


def _load_hlt_sample(
    path: Path,
    *,
    sample_count: int,
) -> tuple[int, np.ndarray, np.ndarray, list[str]]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"HLT-cache evidence is absent or unsafe: {path}")
    with np.load(path, allow_pickle=False) as packed:
        required = {
            "tokens",
            "mask",
            "labels",
            "jet_file_indices",
            "jet_entries",
        }
        if not required.issubset(packed.files):
            raise ValueError(
                f"HLT-cache evidence lacks arrays "
                f"{sorted(required - set(packed.files))}"
            )
        tokens = packed["tokens"]
        mask = packed["mask"]
        labels = packed["labels"]
        file_indices = packed["jet_file_indices"]
        entries = packed["jet_entries"]
        event_count = int(tokens.shape[0])
        if (
            tokens.ndim != 3
            or tokens.shape[1:] != (128, 14)
            or mask.shape != tokens.shape[:2]
            or labels.shape != (event_count,)
            or file_indices.shape != (event_count,)
            or entries.shape != (event_count,)
            or event_count <= 0
        ):
            raise ValueError(
                "HLT-cache evidence has incompatible production arrays"
            )
        if mask.dtype != np.bool_ or not np.isfinite(tokens).all():
            raise ValueError(
                "HLT-cache evidence has invalid mask or nonfinite tokens"
            )
        selected_count = min(int(sample_count), event_count)
        if selected_count <= 0:
            raise ValueError("--tree-sample-jets must be positive")
        indices = np.linspace(
            0,
            event_count - 1,
            num=selected_count,
            dtype=np.int64,
        )
        sample_tokens = np.asarray(tokens[indices], dtype=np.float32)
        sample_mask = np.asarray(mask[indices], dtype=np.bool_)
        identities = [
            f"storage-measurement-file-{int(file_indices[index])}"
            f"#{int(entries[index])}"
            for index in indices
        ]
    if np.any(sample_mask.sum(axis=1) <= 0):
        raise ValueError("HLT-cache tree sample contains an empty event")
    return event_count, sample_tokens, sample_mask, identities


def _write_tree_evidence(
    path: Path,
    *,
    tokens: np.ndarray,
    mask: np.ndarray,
    identities: Sequence[str],
    workers: int,
) -> None:
    if path.exists():
        raise FileExistsError(f"tree evidence already exists: {path}")
    inputs = build_particle_transformer_inputs_from_tokens(
        tokens,
        mask,
        source_view="fixed_hlt",
    )
    vectors = inputs.pf_vectors.transpose(0, 2, 1)
    payloads = [
        (vectors[index], tokens[index], mask[index])
        for index in range(len(tokens))
    ]
    worker_count = min(max(int(workers), 1), len(payloads))
    if worker_count == 1:
        trees = [_build_tree(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            trees = list(pool.map(_build_tree, payloads, chunksize=1))
    packed = pack_tree_shard(trees, identities)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **packed)


def _write_prediction_evidence(path: Path, *, event_count: int) -> None:
    generator = np.random.Generator(np.random.PCG64(41_771))
    logits = generator.standard_normal((event_count, 10)).astype(np.float32)
    labels = np.arange(event_count, dtype=np.int16) % 10
    identities = [
        f"storage-prediction-event-{index}" for index in range(event_count)
    ]
    write_final_predictions(
        path,
        {
            "logits": logits,
            "labels": labels,
            "predictions": logits.argmax(axis=1).astype(np.int16),
            "event_identities": np.asarray(identities, dtype=np.str_),
            "event_identity_sha256": canonical_sha256(identities),
        },
        run_id="RPT_STORAGE_MEASUREMENT",
        seed=101,
        checkpoint_sha256="0" * 64,
        locked_finalists_sha256="0" * 64,
    )


def build_measurement_artifact(
    *,
    hlt_cache: Path,
    checkpoint: Path,
    output: Path,
    tree_sample_jets: int,
    prediction_sample_events: int,
    fixed_overhead_bytes: int,
    workers: int,
) -> dict[str, Any]:
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint evidence is absent or unsafe: {checkpoint}"
        )
    if int(prediction_sample_events) <= 0:
        raise ValueError("--prediction-sample-events must be positive")
    if int(fixed_overhead_bytes) < 0:
        raise ValueError("--fixed-overhead-bytes must be nonnegative")
    evidence_root = output.parent / "evidence"
    tree_path = evidence_root / "representative_region_tree_sidecar.npz"
    prediction_path = evidence_root / "representative_final_predictions.npz"
    hlt_event_count, tokens, mask, identities = _load_hlt_sample(
        hlt_cache,
        sample_count=tree_sample_jets,
    )
    _write_tree_evidence(
        tree_path,
        tokens=tokens,
        mask=mask,
        identities=identities,
        workers=workers,
    )
    _write_prediction_evidence(
        prediction_path,
        event_count=int(prediction_sample_events),
    )
    measurements = StorageMeasurements(
        hlt_sample_jets=hlt_event_count,
        hlt_sample_bytes=int(hlt_cache.stat().st_size),
        tree_sample_jets=len(tokens),
        tree_sample_bytes=int(tree_path.stat().st_size),
        checkpoint_sample_count=1,
        checkpoint_sample_bytes=int(checkpoint.stat().st_size),
        prediction_sample_events=int(prediction_sample_events),
        prediction_sample_bytes=int(prediction_path.stat().st_size),
        fixed_overhead_bytes=int(fixed_overhead_bytes),
    )
    purpose = {
        "hlt_cache": "exact fixed-HLT production-format bytes-per-jet sample",
        "tree_sidecar": (
            "exact REGION reference-tree NPZ bytes-per-jet bootstrap sample; "
            "the locked 20k compiled probe revalidates storage before bulk shards"
        ),
        "checkpoint": "representative retained Particle Transformer checkpoint",
        "predictions": "exact sealed-prediction-format bytes-per-event sample",
    }
    paths = {
        "hlt_cache": hlt_cache.resolve(),
        "tree_sidecar": tree_path.resolve(),
        "checkpoint": checkpoint.resolve(),
        "predictions": prediction_path.resolve(),
    }
    artifact = build_storage_measurements(
        measurements,
        source_evidence={
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
                "purpose": purpose[name],
            }
            for name, path in paths.items()
        },
    )
    write_immutable_json(output, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hlt-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tree-sample-jets", type=int, default=256)
    parser.add_argument("--prediction-sample-events", type=int, default=10_000)
    parser.add_argument(
        "--fixed-overhead-bytes",
        type=int,
        default=1024**3,
        help="Conservative fixed allowance for logs, metrics, and registries.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    args = parser.parse_args(argv)
    artifact = build_measurement_artifact(
        hlt_cache=args.hlt_cache,
        checkpoint=args.checkpoint,
        output=args.output,
        tree_sample_jets=args.tree_sample_jets,
        prediction_sample_events=args.prediction_sample_events,
        fixed_overhead_bytes=args.fixed_overhead_bytes,
        workers=args.workers,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
