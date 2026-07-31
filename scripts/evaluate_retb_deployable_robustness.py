#!/usr/bin/env python3
"""Evaluate one exported RETB graph on one immutable Stage-K view."""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_bytes,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.deployment import (  # noqa: E402
    load_deployable_retb_graph,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (  # noqa: E402
    evaluate_classification,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


CONTRACT = "retb_stage_k_robustness_evaluation_v1"
VIEW_CONTRACT = "retb_stage_k_robustness_view_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--export-manifest", required=True, type=Path)
    parser.add_argument("--view-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    view_manifest = load_hashed_json(
        args.view_manifest, expected_contract=VIEW_CONTRACT
    )
    view_path = args.view_manifest.parent / view_manifest["view_filename"]
    if (
        view_manifest.get("source") != campaign.get("source")
        or _sha256(view_path) != view_manifest["view_sha256"]
    ):
        raise ValueError("robustness view lineage differs")
    packed = torch.load(view_path, map_location="cpu", weights_only=False)
    if packed.get("contract") != VIEW_CONTRACT:
        raise ValueError("robustness view payload differs")
    view: dict[str, Any] = packed["view"]
    graph = load_deployable_retb_graph(
        args.export_manifest, expected_source=campaign["source"]
    )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    graph.to(device).eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(view["identities"]), args.batch_size):
            stop = min(start + args.batch_size, len(view["identities"]))
            batch = {
                "identities": list(view["identities"][start:stop]),
                "replica_ids": view["replica_ids"][start:stop].to(device),
                "degraded_view_hashes": list(
                    view["degraded_view_hashes"][start:stop]
                ),
                "features": view["features"][start:stop].to(device),
                "vectors": view["vectors"][start:stop].to(device),
                "mask": view["mask"][start:stop].to(device),
                "raw_tokens": view["raw_tokens"][start:stop].to(device),
                "region_trees_by_expert": {
                    expert: view["region_trees_by_expert"][expert][start:stop]
                    for expert in EXPERT_ORDER
                },
            }
            logits.append(graph(hlt_inputs=batch)["logits"].float().cpu())
    values = torch.cat(logits).numpy()
    labels = np.asarray(view["labels"], dtype=np.int64)
    metrics = bind_source(
        evaluate_classification(values, labels, split="val_design"),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        identities=np.asarray(view["identities"]),
        logits=np.asarray(values, dtype=np.float32),
    )
    predictions = write_immutable_bytes(
        args.output_dir / "predictions.npz", stream.getvalue()
    )
    write_immutable_json(args.output_dir / "metrics.json", metrics)
    export = load_hashed_json(args.export_manifest)
    record = bind_source(
        with_content_hash(
            {
                "contract": CONTRACT,
                "schema_version": 1,
                "graph_id": args.graph_id,
                "profile": view_manifest["profile"],
                "replica": view_manifest["replica"],
                "pipeline_seed": int(
                    args.graph_id.rsplit("_S", 1)[-1]
                ),
                "split": "val_design",
                "event_count": len(labels),
                "deployment_export_sha256": export["content_hash"],
                "robustness_view_sha256": view_manifest["content_hash"],
                "prediction_sha256": predictions["file_sha256"],
                "metrics_sha256": metrics["content_hash"],
                "scientific_underperformance_blocks_continuation": False,
                "stack_val_consumed": False,
                "final_test_consumed": False,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(args.output_dir / "evaluation.json", record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
