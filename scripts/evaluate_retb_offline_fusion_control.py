#!/usr/bin/env python3
"""Evaluate one immutable parameter-free Stage-C offline fusion control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion_cache import (
    load_frozen_token_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion_training import (
    evaluate_parameter_free_fusion,
    select_best_single_expert,
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
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--val-stop-cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    registry = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_c_runs.json"
    )
    validate_stage_c_run_registry(registry)
    run = resolve_stage_c_run(registry, run_id=args.run_id)
    configuration = run["configuration"]
    variant = configuration.get("fusion_variant")
    if variant not in {"F_BEST_SINGLE", "F_UNIFORM_LOGIT_MEAN"}:
        raise ValueError("run is not a parameter-free fusion control")

    cache_meta, _ = load_frozen_token_cache(args.cache)
    role = (
        "training_worker"
        if cache_meta["split"] == "val_stop"
        else "design_worker"
    )
    authorize_dataset_access(
        worker_role=role, requested_resource=cache_meta["split"]
    )
    if (
        cache_meta.get("source") != campaign.get("source")
        or cache_meta["shape_id"] != configuration["shape_id"]
        or cache_meta["pipeline_seed"] != run["seed"]
    ):
        raise ValueError("fusion-control cache lineage differs")

    selection = None
    if variant == "F_BEST_SINGLE":
        if args.val_stop_cache is None:
            raise ValueError("F_BEST_SINGLE requires --val-stop-cache")
        authorize_dataset_access(
            worker_role="training_worker", requested_resource="val_stop"
        )
        selection = select_best_single_expert(
            val_stop_manifest=args.val_stop_cache
        )
        if (
            selection.get("source") != campaign.get("source")
            or selection["shape_id"] != configuration["shape_id"]
            or selection["pipeline_seed"] != run["seed"]
        ):
            raise ValueError("best-single selection lineage differs")

    output_dir = args.output_dir or (
        args.campaign_root / "runs" / "stage_c" / args.run_id
    )
    output_path = (
        output_dir / f"{cache_meta['split']}_parameter_free_evaluation.json"
    )
    result = {
        "dry_run": bool(args.dry_run),
        "run_id": args.run_id,
        "variant": variant,
        "split": cache_meta["split"],
        "output": str(output_path.resolve()),
        "best_single_selection": selection,
    }
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        if selection is not None:
            write_immutable_json(
                output_dir / "best_single_selection.json", selection
            )
        result["evaluation"] = evaluate_parameter_free_fusion(
            cache_manifest=args.cache,
            output_path=output_path,
            run_id=args.run_id,
            variant=variant,
            best_single_selection=selection,
            device=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
