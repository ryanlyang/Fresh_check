#!/usr/bin/env python3
"""Build an authenticated persistent/transient storage projection for full RETB."""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import load_hashed_json, write_immutable_json
from teacher_logit_reco.relation_expert_token_bridge.provenance import source_snapshot
from teacher_logit_reco.relation_expert_token_bridge.storage import STORAGE_MEASUREMENTS_CONTRACT, build_storage_measurements
from teacher_logit_reco.relation_expert_token_bridge.streamed_execution import build_streamed_storage_projection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-measurements", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--available-storage-path", type=Path)
    parser.add_argument("--maximum-concurrent-allocations", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source_measurements = load_hashed_json(
        args.source_measurements, expected_contract=STORAGE_MEASUREMENTS_CONTRACT
    )
    measured = dict(source_measurements["measurements"])
    hlt_rate = float(measured["hlt_v3_compressed_bytes_per_jet"])
    tree_rate = float(measured["region_sidecar_bytes_per_jet"])
    checkpoint = int(measured["checkpoint_bytes"])
    # Conservative compressed-offline rate: use the largest authenticated
    # offline evidence file and its adjacent event-count manifest when present.
    offline_rates = []
    for evidence in source_measurements.get("source_evidence", {}).values():
        path = Path(str(evidence["path"]))
        if path.name != "offline_inputs.npz":
            continue
        metadata = path.with_name("offline_input_manifest.json")
        if metadata.is_file():
            count = int(json.loads(metadata.read_text(encoding="utf-8"))["event_count"])
            if count > 0:
                offline_rates.append(path.stat().st_size / count)
    offline_rate = max(offline_rates, default=max(hlt_rate, 1.0))
    persistent = {
        "offline_inputs_pre_scale": math.ceil(950_000 * offline_rate),
        "hlt_inputs_pre_scale": math.ceil(2_450_000 * hlt_rate),
        "region_sidecars_pre_scale": math.ceil(3_400_000 * tree_rate),
        "selected_and_candidate_checkpoints": 1_600 * checkpoint,
        "compact_logits_metrics_locks_receipts": 24 * 2**30,
    }
    # K=16, D=128, seven experts, fp16 is the largest token-cache term.
    scale_tokens = 3_000_000 * 16 * 128 * 7 * 2
    transient = {
        "stage_d_native_fusion_banks": math.ceil(500_000 * 16 * 128 * 7 * 2),
        "stage_f_g_target_and_predictor_arrays": math.ceil(500_000 * 16 * 128 * 14 * 2),
        "stage_i_j_joint_and_consumer_arrays": math.ceil(500_000 * 16 * 128 * 21 * 2),
        "stage_k_l_evaluation_working_set": math.ceil(500_000 * (40 + 64 + 16 * 128 * 2)),
        "stage_m_scale_refit_working_set": math.ceil(scale_tokens * 2.5 + 3_000_000 * (hlt_rate + tree_rate)),
        "stage_n_sealed_final_working_set": math.ceil(300_000 * (40 + 64 + hlt_rate + tree_rate)),
    }
    reserve = 16 * 2**30
    storage_path = (args.available_storage_path or args.output.parent).resolve()
    storage_path.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(storage_path).free
    projected_peak = sum(persistent.values()) + reserve
    updated = dict(measured)
    updated["projected_peak_concurrent_bytes"] = projected_peak
    updated["available_storage_bytes"] = available
    evidence = {
        name: {"path": row["path"], "sha256": row["sha256"], "bytes": row["bytes"], "purpose": row["purpose"]}
        for name, row in source_measurements.get("source_evidence", {}).items()
    }
    measurements = build_storage_measurements(
        measurements=updated, source_evidence=evidence,
        measurement_profile="production_source_evidence",
    )
    projection = build_streamed_storage_projection(
        storage_measurements_sha256=measurements["content_hash"],
        persistent_classes=persistent, transient_classes=transient,
        maximum_concurrent_allocations=args.maximum_concurrent_allocations,
        serialized_reserve_bytes=reserve,
        available_storage_bytes=available, source=source_snapshot(REPO_ROOT),
    )
    result = {
        "storage_measurements_sha256": measurements["content_hash"],
        "projection_sha256": projection["content_hash"],
        "persistent_peak_bytes": projection["persistent_peak_bytes"],
        "per_allocation_transient_peak_bytes": projection["per_allocation_transient_peak_bytes"],
        "cluster_transient_peak_bytes": projection["cluster_transient_peak_bytes"],
        "available_storage_bytes": available,
        "storage_admitted": projection["persistent_storage_admitted"],
    }
    if not args.dry_run:
        result["measurement_publication"] = write_immutable_json(args.output, measurements)
        result["projection_publication"] = write_immutable_json(
            args.output.with_name("full_streamed_storage_projection.json"), projection
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["storage_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
