#!/usr/bin/env python3
"""Measure the locked HOSD latency/resource contract on one CUDA node."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_label_free_hlt_loader,
    load_and_validate_campaign,
    load_deployable_graph,
    measure_deployable_efficiency,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (  # noqa: E402
    load_hlt_v3_cache,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--production-batch-size", required=True, type=int)
    parser.add_argument("--analytical-training-flops", required=True, type=int)
    parser.add_argument(
        "--analytical-inference-flops",
        action="append",
        required=True,
        help="BATCH=FLOPS; must cover 1,128,production",
    )
    parser.add_argument("--complete-parameter-count", required=True, type=int)
    parser.add_argument("--target-cache-bytes-per-jet", required=True, type=float)
    parser.add_argument("--training-gpu-hours", required=True, type=float)
    parser.add_argument("--clock-power-mode", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    arrays, metadata = load_hlt_v3_cache(args.cache)
    identities = tuple(str(value) for value in arrays["identities"].tolist())
    required_count = max(128, args.production_batch_size)
    if len(identities) < required_count:
        raise ValueError("efficiency cache is smaller than a required batch")
    module = importlib.import_module("weaver.nn.model.ParticleTransformer")
    model, payload = load_deployable_graph(
        args.export, weaver_module=module, source=campaign["source"]
    )
    if payload["descriptor"].get("graph_id") != args.graph_id:
        raise ValueError("efficiency export graph differs")
    batches = {}
    for batch_size in {1, 128, args.production_batch_size}:
        loader = build_label_free_hlt_loader(
            cache_paths={0: args.cache},
            identities=identities[:batch_size],
            logical_role=str(metadata["logical_role"]),
            realization_policy="R_FIXED",
            batch_size=batch_size,
        )
        batches[batch_size] = next(iter(loader))
    flops = {}
    for value in args.analytical_inference_flops:
        key, separator, raw = value.partition("=")
        if not separator or int(key) in flops:
            raise ValueError("inference FLOPs must be unique BATCH=FLOPS")
        flops[int(key)] = int(raw)
    artifact = measure_deployable_efficiency(
        model=model,
        representative_batches=batches,
        production_batch_size=args.production_batch_size,
        graph_id=args.graph_id,
        seed=args.seed,
        export_path=args.export,
        export_sha256=hashlib.sha256(args.export.read_bytes()).hexdigest(),
        analytical_training_flops=args.analytical_training_flops,
        analytical_inference_flops_by_batch=flops,
        complete_parameter_count=args.complete_parameter_count,
        target_cache_bytes_per_jet=args.target_cache_bytes_per_jet,
        training_gpu_hours=args.training_gpu_hours,
        clock_power_mode=args.clock_power_mode,
        training_evidence={
            "kind": "explicit_cli_measurement",
            "training_gpu_hours": float(args.training_gpu_hours),
        },
        analytical_training_flop_convention=(
            "explicit_cli_analytical_training_flops"
        ),
        source=campaign["source"],
        device="cuda",
    )
    publication = write_immutable_json(args.output, artifact)
    print(
        json.dumps(
            {
                "content_hash": artifact["content_hash"],
                "publication": publication["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
