#!/usr/bin/env python3
"""Compile Stage D and execute one immutable auxiliary row."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    HOSDTrainingProtocol,
    authorize_access,
    build_auxiliary_model,
    build_default_stage_d_role_definitions,
    build_stage_d_loader_manifest,
    build_stage_d_plan,
    exact_trainable_parameter_count,
    load_and_validate_campaign,
    monolithic_flop_ledger,
    resolve_stage_d_phase_two,
    component_seed,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    AUXILIARY_COMPLETION_CONTRACT,
    SINGLE_FAMILY_PHASE_LOCK_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.hlt_offline_structure_distillation.stage_d_training import (  # noqa: E402
    train_stage_d_auxiliary,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--row-id")
    parser.add_argument("--phase-lock", type=Path)
    parser.add_argument(
        "--loader-manifest",
        type=Path,
        help="source-bound hosd_stage_d_loader_manifest_v1 for the built-in loader",
    )
    parser.add_argument("--base-roles-json", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--deployed-analytical-flops", type=float)
    parser.add_argument("--deployed-parameter-count", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args(argv)

    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    target_registry = load_hashed_json(
        args.campaign_root / "registry" / "structure_target_registry.json",
        expected_contract="hosd_structure_target_registry_v1",
    )
    if target_registry.get("source") != campaign["source"]:
        raise ValueError("Stage-D target registry source differs")
    plan = build_stage_d_plan(
        campaign_spec_sha256=campaign["content_hash"],
        target_registry=target_registry,
        source=campaign["source"],
    )
    plan_path = args.campaign_root / "job_ledgers" / "stage_d_execution_plan.json"
    write_immutable_json(plan_path, plan)
    rows = list(plan["all_rows"])
    if args.phase_lock is not None:
        lock = load_hashed_json(
            args.phase_lock, expected_contract=SINGLE_FAMILY_PHASE_LOCK_CONTRACT
        )
        if lock.get("source") != campaign["source"]:
            raise ValueError("Stage-D phase lock source differs")
        resolved = resolve_stage_d_phase_two(stage_d_plan=plan, phase_lock=lock)
        rows = [
            *[
                row
                for row in rows
                if row["phase"] not in {
                    "LOCKED_RELATION_HET",
                    "MATCHED_HLT_SELF",
                }
            ],
            *resolved,
        ]
    result = {
        "stage_d_plan_sha256": plan["content_hash"],
        "row_count": plan["row_count"],
        "hard_maximum": plan["hard_maximum"],
        "executed": False,
    }
    if args.row_id is None or args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    matches = [row for row in rows if row["row_id"] == args.row_id]
    if len(matches) != 1:
        raise ValueError("Stage-D row is absent, duplicated, or requires --phase-lock")
    row = matches[0]
    if not row["resolved"]:
        raise ValueError("Stage-D row requires the single-family phase lock")
    canonical_graph = (
        f"{row.get('source_target_id') or row['target_id']}||"
        f"{row['parameterization']}||"
        f"{0.30 if row['auxiliary_weight'] is None else row['auxiliary_weight']}"
    )
    row = {
        **row,
        "pipeline_seed": int(args.seed),
        "encoder_component_seed": component_seed(
            int(args.seed), "encoder", "H_BASE"
        ),
        "head_component_seed": component_seed(
            int(args.seed), "target_head", canonical_graph
        ),
    }
    if args.loader_manifest is None:
        if args.base_roles_json is None:
            raise ValueError(
                "Stage-D execution requires loader or base-role definitions"
            )
        roles = build_default_stage_d_role_definitions(
            row=row,
            campaign_root=args.campaign_root,
            base_role_definitions=json.loads(
                args.base_roles_json.read_text(encoding="utf-8")
            ),
        )
        loader = build_stage_d_loader_manifest(
            row=row,
            role_definitions=roles,
            campaign_spec_sha256=campaign["content_hash"],
            source=campaign["source"],
        )
        args.loader_manifest = (
            args.campaign_root
            / "loaders"
            / "stage_d"
            / f"{row['row_id']}.json"
        )
        write_immutable_json(args.loader_manifest, loader)
    for resource in (
        "model_train_hlt",
        "model_train_labels",
        "model_train_targets",
        "val_stop_hlt",
        "val_stop_labels",
        "val_stop_targets",
        "design_select_hlt",
        "design_select_labels",
        "design_select_targets",
    ):
        authorize_access(worker_role="train_worker", requested_resource=resource)
    from teacher_logit_reco.hlt_offline_structure_distillation.stage_d_data_factory import (
        load_stage_d_loaders_from_manifest,
    )

    loaded = load_stage_d_loaders_from_manifest(
        manifest_path=args.loader_manifest,
        campaign_root=args.campaign_root,
        row=dict(row),
        campaign=dict(campaign),
        target_registry=dict(target_registry),
    )
    required = {
        "train_loader",
        "val_stop_loader",
        "design_select_loader",
        "component_group_ids",
        "lineage_hashes",
    }
    if not isinstance(loaded, dict) or not required.issubset(loaded):
        raise ValueError("Stage-D loader factory returned an incomplete interface")
    if not loaded["lineage_hashes"]:
        raise ValueError("Stage-D loader factory omitted source artifact hashes")
    module = importlib.import_module("weaver.nn.model.ParticleTransformer")
    model, declared_groups = build_auxiliary_model(row, weaver_module=module)
    deployed_flops = float(
        monolithic_flop_ledger(
            {
                "embed_dim": 128,
                "attention_heads": 8,
                "particle_blocks": 8,
                "class_blocks": 2,
            }
        )["total_flops"]
    )
    deployed_parameters = exact_trainable_parameter_count(model.classifier)
    if (
        args.deployed_analytical_flops is not None
        and float(args.deployed_analytical_flops) != deployed_flops
    ):
        raise ValueError("Stage-D supplied deployed FLOPs differ analytically")
    if (
        args.deployed_parameter_count is not None
        and int(args.deployed_parameter_count) != deployed_parameters
    ):
        raise ValueError("Stage-D supplied deployed parameters differ")
    if tuple(loaded["component_group_ids"]) != tuple(declared_groups):
        raise ValueError("Stage-D loader availability-group semantics differ")
    miniature = campaign["campaign_profile"] == "miniature_test"
    protocol = HOSDTrainingProtocol(
        maximum_epochs=2 if miniature else 40,
        campaign_profile="miniature_test" if miniature else "production",
    )
    import torch

    output = args.output_dir or (
        args.campaign_root
        / "auxiliary"
        / row["row_id"]
        / f"seed_{args.seed}"
    )
    completion = train_stage_d_auxiliary(
        model=model,
        train_loader=loaded["train_loader"],
        val_stop_loader=loaded["val_stop_loader"],
        design_select_loader=loaded["design_select_loader"],
        output_dir=output,
        row=row,
        component_group_ids=declared_groups,
        stage_d_plan_sha256=plan["content_hash"],
        campaign_spec_sha256=campaign["content_hash"],
        lineage_hashes=loaded["lineage_hashes"],
        protocol=protocol,
        source=campaign["source"],
        deployed_analytical_flops=deployed_flops,
        deployed_parameter_count=deployed_parameters,
        device=(
            "cuda"
            if args.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if args.device == "auto"
            else args.device
        ),
    )
    result.update(
        {"executed": True, "completion_sha256": completion["content_hash"]}
    )
    from teacher_logit_reco.hlt_offline_structure_distillation.wave_completion import (
        try_finalize_row_wave,
    )

    if row["row_kind"] == "SCIENTIFIC" and row["phase"] == "PRIMARY":
        wave_id = "stage_d_primary_scientific"
        expected = [
            item
            for item in rows
            if item["row_kind"] == "SCIENTIFIC"
            and item["phase"] == "PRIMARY"
        ]
        filename = "primary_scientific_completion.json"
    elif row["phase"] == "LOCKED_RELATION_HET":
        wave_id = "stage_d_relation_het"
        expected = [
            item for item in rows if item["phase"] == "LOCKED_RELATION_HET"
        ]
        filename = "relation_het_completion.json"
    elif row["row_kind"] == "HLT_SELF":
        wave_id = "stage_d_hlt_self"
        expected = [
            item for item in rows if item["row_kind"] == "HLT_SELF"
        ]
        filename = "hlt_self_control_completion.json"
    else:
        wave_id = "stage_d_null_controls"
        expected = [
            item
            for item in rows
            if item["row_kind"] not in {"SCIENTIFIC", "HLT_SELF"}
        ]
        filename = "null_control_completion.json"
    wave = try_finalize_row_wave(
        wave_id=wave_id,
        expected_paths={
            item["row_id"]: args.campaign_root
            / "auxiliary"
            / item["row_id"]
            / f"seed_{args.seed}"
            / "auxiliary_completion.json"
            for item in expected
        },
        expected_rows={
            item["row_id"]: {
                "row_id": item["row_id"],
                "target_id": item["target_id"],
                "parameterization": item["parameterization"],
                "row_kind": item["row_kind"],
            }
            for item in expected
        },
        expected_contract=AUXILIARY_COMPLETION_CONTRACT,
        parent_hashes={"stage_d_plan": plan["content_hash"]},
        source=campaign["source"],
        output=args.campaign_root / "auxiliary" / filename,
    )
    result["wave_completion_sha256"] = (
        None if wave is None else wave["content_hash"]
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
