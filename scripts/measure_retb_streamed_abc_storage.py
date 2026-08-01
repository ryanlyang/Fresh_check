#!/usr/bin/env python3
"""Project persistent storage for streamed A--C from a completed RETB run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.storage import (  # noqa: E402
    build_storage_measurements,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _first(root: Path, pattern: str) -> Path:
    path = next((row for row in root.glob(pattern) if row.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"streamed storage evidence is absent under {root}: {pattern}"
        )
    return path


def _npz_rate(paths: Sequence[Path]) -> tuple[int, int, float]:
    total_bytes = 0
    total_events = 0
    for path in paths:
        metadata = json.loads(
            path.with_name("offline_input_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        total_bytes += path.stat().st_size
        total_events += int(metadata["event_count"])
    if total_events <= 0:
        raise ValueError("offline storage evidence has no events")
    return total_bytes, total_events, total_bytes / total_events


def _tree_rate(root: Path) -> tuple[Path, float]:
    candidates = []
    for npz in root.glob("inputs/region_tree/**/shards/shard_*.npz"):
        metadata_path = npz.with_suffix(".metadata.json")
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        count = int(metadata["jet_count"])
        if count > 0:
            candidates.append((npz, npz.stat().st_size / count))
    if not candidates:
        raise FileNotFoundError("REGION shard storage evidence is absent")
    # The largest observed rate is the conservative source-bound estimator.
    return max(candidates, key=lambda row: row[1])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-campaign-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--available-storage-path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.evidence_campaign_root.resolve()
    offline_paths = [
        root / "inputs" / "offline" / role / "offline_inputs.npz"
        for role in ("model_train", "val_stop", "val_design")
    ]
    if not all(path.is_file() for path in offline_paths):
        raise FileNotFoundError("offline A-C source caches are incomplete")
    offline_bytes, offline_events, offline_rate = _npz_rate(offline_paths)
    hlt_npz = _first(
        root,
        "inputs/hlt_v3/model_train/replica_*/R_MULTI/D_NOMINAL/hlt_v3_arrays.npz",
    )
    hlt_meta = json.loads(
        hlt_npz.with_name("hlt_v3_metadata.json").read_text(encoding="utf-8")
    )
    hlt_events = int(hlt_meta["shape"][0])
    if hlt_events <= 0:
        raise ValueError("HLT source cache has no events")
    hlt_rate = hlt_npz.stat().st_size / hlt_events
    tree_npz, tree_rate = _tree_rate(root)
    checkpoint = max(
        (path for path in root.glob("**/best_model_val.pt") if path.is_file()),
        key=lambda path: path.stat().st_size,
        default=None,
    )
    if checkpoint is None:
        raise FileNotFoundError("checkpoint size evidence is absent")
    checkpoint_bytes = checkpoint.stat().st_size

    # Only A-C populations are persistent.  HLT normalizer views are 4*12,500.
    input_projection = math.ceil(
        600_000 * offline_rate
        + 50_000 * hlt_rate
        + 650_000 * tree_rate
    )
    model_projection = 400 * checkpoint_bytes
    serialized_reserve = 16 * 2**30
    projected_peak = input_projection + model_projection + serialized_reserve
    storage_path = (
        args.available_storage_path or args.output.parent
    ).resolve()
    storage_path.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(storage_path).free
    measurements = {
        "hlt_v3_compressed_bytes_per_jet": hlt_rate,
        "region_sidecar_bytes_per_jet": tree_rate,
        "expert_target_fp16_bytes_per_scalar": 2,
        "expert_target_fp32_bytes_per_scalar": 4,
        "logits_bytes_per_jet": 40,
        "identity_bytes_per_jet": 64,
        "checkpoint_bytes": checkpoint_bytes,
        "projected_peak_concurrent_bytes": projected_peak,
        "available_storage_bytes": available,
        # Rates do not choose or stop scientific rows; live probes remain authoritative.
        "cpu_degradation_jets_per_second": 1,
        "gpu_expert_jets_per_second": 1,
        "gpu_predictor_jets_per_second": 1,
    }
    evidence = {
        "offline_model_train": {
            "path": str(offline_paths[0]),
            "purpose": "compressed_offline_input_rate",
        },
        "offline_val_stop": {
            "path": str(offline_paths[1]),
            "purpose": "compressed_offline_input_rate",
        },
        "offline_val_design": {
            "path": str(offline_paths[2]),
            "purpose": "compressed_offline_input_rate",
        },
        "hlt_model_train": {
            "path": str(hlt_npz),
            "purpose": "compressed_hlt_input_rate",
        },
        "region_shard": {
            "path": str(tree_npz),
            "purpose": "conservative_region_sidecar_rate",
        },
        "checkpoint": {
            "path": str(checkpoint),
            "purpose": "largest_observed_model_checkpoint",
        },
    }
    artifact = build_storage_measurements(
        measurements=measurements,
        source_evidence=evidence,
        measurement_profile="production_source_evidence",
    )
    projection = with_content_hash(
        {
            "contract": "retb_streamed_abc_storage_projection_v1",
            "schema_version": 1,
            "storage_measurements_sha256": artifact["content_hash"],
            "evidence_campaign_root": str(root),
            "offline_source_bytes": offline_bytes,
            "offline_source_events": offline_events,
            "offline_compressed_bytes_per_jet": offline_rate,
            "hlt_compressed_bytes_per_jet": hlt_rate,
            "region_bytes_per_jet": tree_rate,
            "persistent_input_projection_bytes": input_projection,
            "persistent_model_projection_bytes": model_projection,
            "serialized_reserve_bytes": serialized_reserve,
            "projected_peak_concurrent_bytes": projected_peak,
            "available_storage_bytes": available,
            "storage_admitted": projected_peak <= available,
            "ephemeral_frozen_token_banks_included_in_persistent_peak": False,
            "ephemeral_location": "node_local_memory_or_slurm_tmp",
            "evidence_sha256s": {
                name: _sha256(Path(str(row["path"])))
                for name, row in evidence.items()
            },
        }
    )
    result = {
        "dry_run": bool(args.dry_run),
        "storage_measurements_sha256": artifact["content_hash"],
        "projection_sha256": projection["content_hash"],
        "projected_peak_concurrent_bytes": projected_peak,
        "available_storage_bytes": available,
        "storage_admitted": projection["storage_admitted"],
    }
    if not args.dry_run:
        result["measurement_publication"] = write_immutable_json(
            args.output, artifact
        )
        result["projection_publication"] = write_immutable_json(
            args.output.with_name("streamed_abc_projection.json"), projection
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if projection["storage_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
