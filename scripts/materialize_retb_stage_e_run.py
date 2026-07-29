#!/usr/bin/env python3
"""Materialize one immutable seed-matched RETB Stage-E run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.dynamic_continuation import (  # noqa: E402
    add_dynamic_continuation_arguments,
    resolve_selector_continuation,
)
from teacher_logit_reco.relation_expert_token_bridge.step7 import (  # noqa: E402
    materialize_stage_e_run,
)
from teacher_logit_reco.relation_expert_token_bridge.token_shape_registry import (  # noqa: E402
    HET_PHYSICS,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _checkpoint_sha(
    path: Path,
    *,
    expected_expert: str,
    expected_shapes: set[str],
    seed: int,
    resolved_run: Mapping[str, object] | None = None,
) -> str:
    artifact: Mapping[str, object] = load_hashed_json(path)
    configuration = (
        {}
        if resolved_run is None
        else dict(resolved_run.get("configuration", {}))
    )
    actual_seed = artifact.get(
        "pipeline_seed", artifact.get("seed", resolved_run.get("seed") if resolved_run else -1)
    )
    actual_expert = artifact.get(
        "expert_id", configuration.get("expert_id")
    )
    actual_shape = artifact.get("shape_id", configuration.get("shape_id"))
    if (
        actual_expert != expected_expert
        or actual_shape not in expected_shapes
        or int(actual_seed) != seed
        or not isinstance(artifact.get("checkpoint_sha256"), str)
    ):
        raise ValueError("Stage-E checkpoint registration identity differs")
    return str(artifact["checkpoint_sha256"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--expert-id", required=True)
    parser.add_argument("--shape-id", required=True)
    parser.add_argument("--target-mode", required=True)
    parser.add_argument("--lambda-pred", type=float, default=0.0)
    parser.add_argument("--bridge-dimension", type=int)
    parser.add_argument("--unfreeze-final-two-blocks", action="store_true")
    parser.add_argument("--t0-registration", required=True, type=Path)
    parser.add_argument("--hlt-encoder-registration", required=True, type=Path)
    parser.add_argument(
        "--unbiased-particle-encoder-registration",
        required=True,
        type=Path,
    )
    parser.add_argument("--pilot-registration", type=Path)
    parser.add_argument("--uniform-shapes", type=Path)
    parser.add_argument("--heterogeneous-shapes", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    add_dynamic_continuation_arguments(parser)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    registry = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_e_templates.json"
    )
    stage_d_registry = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_d_runs.json"
    )
    confirmation_path = (
        args.campaign_root / "selection" / "retb_stage_d_confirmations.json"
    )
    confirmation = (
        load_hashed_json(confirmation_path)
        if confirmation_path.is_file()
        else None
    )
    uniform_path = args.uniform_shapes or (
        args.campaign_root / "selection" / "retb_offline_shapes.json"
    )
    heterogeneous_path = args.heterogeneous_shapes or (
        args.campaign_root / "selection" / "retb_heterogeneous_shapes.json"
    )
    if args.shape_id in {"SHAPE_COMPACT", "SHAPE_HIGH"}:
        uniform = load_hashed_json(uniform_path)
        concrete_shape = str(uniform[args.shape_id]["shape_id"])
        shape_selection = uniform
    elif args.shape_id == "HET_PHYSICS":
        concrete_shape = f"HET_K{int(HET_PHYSICS[args.expert_id])}_D128"
        shape_selection = None
    else:
        heterogeneous = load_hashed_json(heterogeneous_path)
        concrete_shape = (
            f"HET_K{int(heterogeneous[args.shape_id]['allocation'][args.expert_id])}"
            "_D128"
        )
        shape_selection = heterogeneous
    parent_paths = [
        args.t0_registration,
        args.hlt_encoder_registration,
        args.unbiased_particle_encoder_registration,
        *([args.pilot_registration] if args.pilot_registration else []),
    ]
    parents = [load_hashed_json(path) for path in parent_paths]
    if (
        registry.get("source") != campaign.get("source")
        or stage_d_registry.get("source") != campaign.get("source")
        or (
            confirmation is not None
            and confirmation.get("source") != campaign.get("source")
        )
        or (
            shape_selection is not None
            and shape_selection.get("source") != campaign.get("source")
        )
        or any(
        parent.get("source") is not None
        and parent.get("source") != campaign.get("source")
        for parent in parents
        )
    ):
        raise ValueError("Stage-E materialization source lineage differs")
    t0_sha = _checkpoint_sha(
        args.t0_registration,
        expected_expert=args.expert_id,
        expected_shapes={concrete_shape},
        seed=args.pipeline_seed,
    )
    hlt_artifact = parents[1]
    def stage_d_run(artifact):
        rows = [
            row
            for section in (
                "scratch_expert_rows",
                "encoder_screen_rows",
                "native_fusion_rows",
                "baseline_rows",
            )
            for row in stage_d_registry[section]
            if row["run_id"] == artifact.get("run_id")
        ]
        if confirmation is not None:
            rows.extend(
                row
                for row in confirmation["rows"]
                if row["run_id"] == artifact.get("run_id")
            )
        if len(rows) != 1:
            raise ValueError("HLT encoder registration run is not in Stage D")
        return rows[0]
    hlt_sha = _checkpoint_sha(
        args.hlt_encoder_registration,
        expected_expert=args.expert_id,
        expected_shapes={args.shape_id, concrete_shape},
        seed=args.pipeline_seed,
        resolved_run=stage_d_run(hlt_artifact),
    )
    unbiased_sha = _checkpoint_sha(
        args.unbiased_particle_encoder_registration,
        expected_expert="BASE4",
        expected_shapes={args.shape_id, concrete_shape},
        seed=args.pipeline_seed,
        resolved_run=stage_d_run(parents[2]),
    )
    pilot_sha = (
        None
        if args.pilot_registration is None
        else _checkpoint_sha(
            args.pilot_registration,
            expected_expert=args.expert_id,
            expected_shapes={args.shape_id},
            seed=args.pipeline_seed,
        )
    )
    run = bind_source(
        materialize_stage_e_run(
            template_registry=registry,
            pipeline_seed=args.pipeline_seed,
            expert_id=args.expert_id,
            shape_id=args.shape_id,
            target_mode=args.target_mode,
            lambda_pred=args.lambda_pred,
            bridge_dimension=args.bridge_dimension,
            unfreeze_final_two_blocks=args.unfreeze_final_two_blocks,
            t0_checkpoint_sha256=t0_sha,
            hlt_encoder_checkpoint_sha256=hlt_sha,
            unbiased_particle_encoder_checkpoint_sha256=unbiased_sha,
            pilot_checkpoint_sha256=pilot_sha,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    output = args.output or (
        args.campaign_root
        / "runs"
        / "stage_e"
        / "materialized"
        / f"{run['run_id']}.json"
    )
    result = {
        "dry_run": bool(args.dry_run),
        "run": run,
        "output": str(output.resolve()),
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(output, run)
    continuation = resolve_selector_continuation(
        args=args,
        campaign=campaign,
        campaign_root=args.campaign_root,
        selector_output=run,
        selector_output_path=output,
        load_hashed_json=load_hashed_json,
        dry_run=bool(args.dry_run),
    )
    if continuation is not None:
        result["continuation"] = continuation
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
