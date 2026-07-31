#!/usr/bin/env python3
"""Publish one label-free, identity-bound stack-val prediction shard."""

from __future__ import annotations

import argparse
import importlib
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    authorize_access,
    build_stack_prediction_manifest,
    build_label_free_hlt_loader,
    infer_deployable_graph,
    load_deployable_graph,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    SCALE_EXECUTION_PLAN_CONTRACT,
    STACK_PREDICTION_MANIFEST_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)


def _try_finalize(root: Path, campaign):
    plan = load_hashed_json(
        root / "scale_up" / "execution_plan.json",
        expected_contract=SCALE_EXECUTION_PLAN_CONTRACT,
    )
    hashes = {}
    for row in plan["graph_rows"]:
        row_id = f"{row['graph_id']}__seed_{row['seed']}"
        path = (
            root / "selection_predictions" / "stack_val" / "rows"
            / f"{row_id}.json"
        )
        if not path.is_file():
            return None
        artifact = load_hashed_json(
            path, expected_contract=STACK_PREDICTION_MANIFEST_CONTRACT
        )
        if (
            artifact.get("source") != campaign["source"]
            or artifact.get("graph_id") != row["graph_id"]
            or int(artifact.get("seed", -1)) != int(row["seed"])
        ):
            raise ValueError("stack prediction wave coordinate differs")
        hashes[row_id] = artifact["content_hash"]
    completion = with_content_hash(
        {
            "contract": "hosd_stack_prediction_wave_v1",
            "schema_version": 1,
            "source": dict(campaign["source"]),
            "campaign_spec_sha256": campaign["content_hash"],
            "scale_execution_plan_sha256": plan["content_hash"],
            "prediction_hashes": dict(sorted(hashes.items())),
            "prediction_count": len(hashes),
            "coverage_exact": True,
            "labels_consumed": False,
            "performance_based_termination": False,
        }
    )
    write_immutable_json(
        root / "selection_predictions" / "stack_val" / "completion.json",
        completion,
    )
    return completion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--identities-npz", required=True, type=Path)
    parser.add_argument("--cache", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--export-sha256", required=True)
    parser.add_argument("--lineage", action="append", default=[], metavar="NAME=SHA256")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    authorize_access(worker_role="stack_inference", requested_resource="stack_val_hlt")
    with np.load(args.identities_npz, allow_pickle=False) as payload:
        if set(payload.files) != {"identities"}:
            raise ValueError("stack identity NPZ must contain identities only")
        identities = [str(value) for value in payload["identities"].tolist()]
    caches = {}
    for value in args.cache:
        replica, separator, path = value.partition("=")
        if not separator or int(replica) in caches:
            raise ValueError("cache arguments must be unique REPLICA=PATH")
        caches[int(replica)] = Path(path)
    if set(caches) != {0}:
        raise ValueError("stack-val deployable inference requires fixed replica 0")
    module = importlib.import_module("weaver.nn.model.ParticleTransformer")
    model, _ = load_deployable_graph(
        args.export, weaver_module=module, source=campaign["source"]
    )
    import torch

    loader = build_label_free_hlt_loader(
        cache_paths=caches,
        identities=identities,
        logical_role="stack_val",
        realization_policy="R_FIXED",
        batch_size=args.batch_size,
    )
    inferred_ids, logits = infer_deployable_graph(
        model,
        loader,
        device=(
            "cuda"
            if args.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if args.device == "auto"
            else args.device
        ),
    )
    if tuple(identities) != inferred_ids:
        raise ValueError("stack inference identity order changed")
    lineage = {}
    for value in args.lineage:
        key, separator, digest = value.partition("=")
        if not separator or key in lineage:
            raise ValueError("lineage must be unique NAME=SHA256")
        lineage[key] = digest
    lineage["deployable_export_manifest"] = hashlib.sha256(
        args.export.read_bytes()
    ).hexdigest()
    artifact = build_stack_prediction_manifest(
        graph_id=args.graph_id,
        seed=args.seed,
        identities=identities,
        logits=logits,
        checkpoint_sha256=args.checkpoint_sha256,
        export_sha256=args.export_sha256,
        lineage_hashes=lineage,
        source=campaign["source"],
    )
    publication = write_immutable_json(args.output, artifact)
    completion = _try_finalize(args.campaign_root.resolve(), campaign)
    print(json.dumps({"content_hash": artifact["content_hash"], "publication": publication["status"], "wave_complete": completion is not None}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
