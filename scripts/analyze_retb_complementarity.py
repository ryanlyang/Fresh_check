#!/usr/bin/env python3
"""Build deterministic complementarity diagnostics from frozen expert outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.complementarity import (
    build_complementarity_report,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion_cache import load_frozen_token_cache
from teacher_logit_reco.relation_expert_token_bridge.provenance import source_snapshot
from teacher_logit_reco.relation_expert_token_bridge.registry import EXPERT_ORDER
from teacher_logit_reco.relation_expert_token_bridge.workflow import (
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--cache-manifest", required=True, type=Path)
    parser.add_argument("--subset-metrics", required=True, type=Path)
    parser.add_argument("--attention-npz", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(args.campaign_root, repo_root=REPO_ROOT)
    cache_meta, arrays = load_frozen_token_cache(args.cache_manifest)
    if cache_meta.get("source") != campaign.get("source"):
        raise ValueError("frozen token cache belongs to another source")
    requested = cache_meta["split"]
    role = "design_worker" if requested == "val_design" else "training_worker"
    authorize_dataset_access(worker_role=role, requested_resource=requested)
    subset = load_hashed_json(args.subset_metrics)
    if subset.get("contract") != "retb_offline_subset_metrics_v1":
        raise ValueError("subset metric artifact contract differs")
    if subset.get("source") != campaign.get("source"):
        raise ValueError("subset metrics belong to another source")
    rows = {
        int(row["bitmask"]): {
            "accuracy": float(row["accuracy"]),
            "cross_entropy": float(row["cross_entropy"]),
        }
        for row in subset["rows"]
    }
    attention = None
    if args.attention_npz is not None:
        with np.load(args.attention_npz, allow_pickle=False) as payload:
            if set(payload.files) != {f"attention_{name}" for name in EXPERT_ORDER}:
                raise ValueError("attention diagnostic fields differ")
            attention = {
                name: np.asarray(payload[f"attention_{name}"])
                for name in EXPERT_ORDER
            }
    result = {
        "dry_run": bool(args.dry_run),
        "shape_id": cache_meta["shape_id"],
        "pipeline_seed": cache_meta["pipeline_seed"],
        "subset_coverage": len(rows),
        "output": str(args.output.resolve()),
    }
    if not args.dry_run:
        report = build_complementarity_report(
            shape_id=cache_meta["shape_id"],
            pipeline_seed=cache_meta["pipeline_seed"],
            cache_manifest_sha256=cache_meta["content_hash"],
            logits_by_expert=arrays["expert_logits"],
            labels=arrays["labels"],
            tokens_by_expert=arrays["token_banks"],
            subset_metrics=rows,
            attention_by_expert=attention,
        )
        report = bind_source(report, source_snapshot=source_snapshot(REPO_ROOT))
        result["publication"] = write_immutable_json(args.output, report)
        result["report_sha256"] = report["content_hash"]
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
