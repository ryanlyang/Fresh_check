#!/usr/bin/env python3
"""Train one immutable Stage-C frozen offline fusion row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import load_hashed_json
from teacher_logit_reco.relation_expert_token_bridge.fusion import build_fusion_model
from teacher_logit_reco.relation_expert_token_bridge.fusion_cache import load_frozen_token_cache
from teacher_logit_reco.relation_expert_token_bridge.fusion_training import (
    OfflineFusionTrainingConfig,
    train_frozen_fusion,
)
from teacher_logit_reco.relation_expert_token_bridge.step5 import (
    resolve_stage_c_run,
    validate_stage_c_run_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-train-cache", type=Path)
    parser.add_argument("--val-stop-cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    registry = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_c_runs.json"
    )
    registry_sha = validate_stage_c_run_registry(registry)
    run = resolve_stage_c_run(registry, run_id=args.run_id)
    configuration = run["configuration"]
    if configuration.get("kind") not in {
        "CANONICAL_UNIFORM_FUSION_CONFIRMATION",
        "UNIFORM_FUSION_CONTROL",
    }:
        raise ValueError("run is not a cached offline fusion row")
    variant = configuration["fusion_variant"]
    if variant == "F_UNIFORM_LOGIT_MEAN":
        raise ValueError("logit mean is evaluated without a training worker")
    miniature = campaign["campaign_profile"] == "miniature_test"
    config = OfflineFusionTrainingConfig(
        seed=run["seed"],
        variant=variant,
        maximum_epochs=2 if miniature else 40,
        batch_size=args.batch_size,
        campaign_profile="miniature_test" if miniature else "production",
    )
    config.validate()
    output = args.output_dir or (
        args.campaign_root / "runs" / "stage_c" / args.run_id
    )
    result = {
        "dry_run": bool(args.dry_run),
        "run_id": args.run_id,
        "configuration": configuration,
        "training_config": config.artifact(
            global_determinism_sha256=campaign["parent_artifact_hashes"]["global_determinism"],
            fusion_architecture_sha256=load_hashed_json(
                args.campaign_root / "registry" / "retb_offline_fusion.json"
            )["content_hash"],
        ),
        "output_dir": str(output.resolve()),
    }
    if args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.model_train_cache is None or args.val_stop_cache is None:
        raise ValueError("fusion training requires both cache manifests")
    authorize_dataset_access(worker_role="training_worker", requested_resource="model_train")
    authorize_dataset_access(worker_role="training_worker", requested_resource="val_stop")
    train_meta, _ = load_frozen_token_cache(args.model_train_cache)
    val_meta, _ = load_frozen_token_cache(args.val_stop_cache)
    if (
        train_meta.get("source") != campaign.get("source")
        or val_meta.get("source") != campaign.get("source")
    ):
        raise ValueError("frozen fusion caches belong to another source")
    dimensions = {
        name: int(shape[1]) for name, shape in train_meta["allocation"].items()
    }
    model = build_fusion_model(variant, bank_dimensions=dimensions)
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    registration = train_frozen_fusion(
        model=model,
        model_train_manifest=args.model_train_cache,
        val_stop_manifest=args.val_stop_cache,
        output_dir=output,
        run_id=args.run_id,
        run_registry_sha256=registry_sha,
        global_determinism_sha256=campaign["parent_artifact_hashes"]["global_determinism"],
        fusion_architecture_sha256=load_hashed_json(
            args.campaign_root / "registry" / "retb_offline_fusion.json"
        )["content_hash"],
        config=config,
        device=device,
    )
    print(json.dumps(registration, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
