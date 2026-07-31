#!/usr/bin/env python3
"""Compile Stage E and execute one immutable predicted-feedback row."""

from __future__ import annotations

import argparse
import hashlib
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
    build_feedback_model,
    build_default_stage_d_role_definitions,
    build_stage_d_loader_manifest,
    build_stage_e_loader_manifest,
    build_stage_e_plan,
    exact_trainable_parameter_count,
    evaluate_posthoc_feedback_control,
    export_deployable_graph,
    feedback_model_flop_ledger,
    load_and_validate_campaign,
    load_stage_d_loaders_from_manifest,
    load_stage_e_loaders_from_manifest,
    materialize_feedback_intervention,
    train_stage_e_feedback,
    component_seed,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    FEEDBACK_COMPLETION_CONTRACT,
    SINGLE_FAMILY_SELECTION_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--row-id")
    parser.add_argument("--loader-manifest", type=Path)
    parser.add_argument(
        "--base-roles-json",
        type=Path,
        help="fixed HLT/label/tree role definitions used to build the loader",
    )
    parser.add_argument(
        "--intervention",
        action="append",
        default=[],
        metavar="ROLE=NPZ",
    )
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
    lock = load_hashed_json(
        args.campaign_root / "auxiliary" / "locked_single_family_choices.json",
        expected_contract=SINGLE_FAMILY_SELECTION_CONTRACT,
    )
    if any(value.get("source") != campaign["source"] for value in (target_registry, lock)):
        raise ValueError("Stage-E parent source differs")
    plan = build_stage_e_plan(
        single_family_selection=lock,
        campaign_spec_sha256=campaign["content_hash"],
        source=campaign["source"],
    )
    write_immutable_json(
        args.campaign_root / "job_ledgers" / "stage_e_execution_plan.json", plan
    )
    summary = {
        "stage_e_plan_sha256": plan["content_hash"],
        "row_count": plan["row_count"],
        "hard_maximum": plan["hard_maximum"],
        "executed": False,
    }
    if args.row_id is None or args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    rows = [row for row in plan["all_rows"] if row["row_id"] == args.row_id]
    if len(rows) != 1:
        raise ValueError("Stage-E row is absent or duplicated")
    row = rows[0]
    row = {
        **row,
        "pipeline_seed": int(args.seed),
        "encoder_component_seed": component_seed(
            int(args.seed), "encoder", "H_BASE"
        ),
        "feedback_component_seed": component_seed(
            int(args.seed), "feedback", row["row_id"]
        ),
    }
    if args.loader_manifest is None:
        if args.base_roles_json is None:
            raise ValueError(
                "Stage-E execution requires loader or base-role definitions"
            )
        base_roles = json.loads(
            args.base_roles_json.read_text(encoding="utf-8")
        )
        definitions = build_default_stage_d_role_definitions(
            row=row,
            campaign_root=args.campaign_root,
            base_role_definitions=base_roles,
        )
        base_manifest = build_stage_d_loader_manifest(
            row=row,
            role_definitions=definitions,
            campaign_spec_sha256=campaign["content_hash"],
            source=campaign["source"],
        )
        loader_root = args.campaign_root / "loaders" / "stage_e"
        base_path = loader_root / f"{row['row_id']}.base.json"
        write_immutable_json(base_path, base_manifest)
        intervention_sources = {}
        for raw in args.intervention:
            role, separator, path = raw.partition("=")
            if (
                not separator
                or role in intervention_sources
                or role
                not in {"model_train", "val_stop", "design_select"}
            ):
                raise ValueError(
                    "--intervention requires unique ROLE=NPZ values"
                )
            intervention_sources[role] = {
                "npz_path": str(Path(path).resolve())
            }
        control = row.get("control")
        needs_intervention = control in {
            "SHUFFLED_PREDICTION",
            "SHUFFLED",
            "ORACLE_SUB",
            "ORACLE_TRAINED",
            "EXACT_HLT",
        }
        if needs_intervention and not intervention_sources:
            import torch

            base_loaded = load_stage_d_loaders_from_manifest(
                manifest_path=base_path,
                campaign_root=args.campaign_root,
                row=row,
                campaign=campaign,
                target_registry=target_registry,
            )
            prediction_model = None
            if control in {"SHUFFLED_PREDICTION", "SHUFFLED"}:
                source_rows = [
                    value
                    for value in plan["scientific_rows"]
                    if value["target_id"] == row["target_id"]
                    and value["interface"] == row["interface"]
                    and value["gradient_path"] == "END_TO_END"
                ]
                if len(source_rows) != 1:
                    raise ValueError(
                        "feedback shuffle source row is absent or duplicated"
                    )
                source_row = {
                    **source_rows[0],
                    "pipeline_seed": int(args.seed),
                    "encoder_component_seed": component_seed(
                        int(args.seed), "encoder", "H_BASE"
                    ),
                    "feedback_component_seed": component_seed(
                        int(args.seed),
                        "feedback",
                        source_rows[0]["row_id"],
                    ),
                }
                module = importlib.import_module(
                    "weaver.nn.model.ParticleTransformer"
                )
                prediction_model = build_feedback_model(
                    source_row, weaver_module=module
                )
                checkpoint_path = (
                    args.campaign_root
                    / "feedback"
                    / source_row["row_id"]
                    / f"seed_{args.seed}"
                    / "best_model_val.pt"
                )
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                if (
                    checkpoint.get("contract")
                    != "hosd_feedback_checkpoint_v2"
                    or checkpoint.get("row_id") != source_row["row_id"]
                    or checkpoint.get("stage_e_plan_sha256")
                    != plan["content_hash"]
                    or checkpoint.get("campaign_spec_sha256")
                    != campaign["content_hash"]
                    or checkpoint.get("source") != campaign["source"]
                ):
                    raise ValueError(
                        "feedback shuffle source checkpoint lineage differs"
                    )
                prediction_model.load_state_dict(
                    checkpoint["model_state_dict"], strict=True
                )
            required_roles = (
                ("design_select",)
                if control == "ORACLE_SUB"
                else ("model_train", "val_stop", "design_select")
            )
            loader_by_role = {
                "model_train": base_loaded["train_loader"],
                "val_stop": base_loaded["val_stop_loader"],
                "design_select": base_loaded["design_select_loader"],
            }
            split_by_role = {
                "model_train": "model_train",
                "val_stop": "val_stop",
                "design_select": "val_design",
            }
            intervention_root = (
                args.campaign_root
                / "feedback"
                / "interventions"
                / row["row_id"]
            )
            for role in required_roles:
                shuffle_plan = (
                    load_hashed_json(
                        args.campaign_root
                        / "targets"
                        / "controls"
                        / "plans"
                        / split_by_role[role]
                        / "global"
                        / f"{row['target_id']}.json",
                        expected_contract="hosd_target_shuffle_plan_v1",
                    )
                    if prediction_model is not None
                    else None
                )
                output_path = intervention_root / f"{role}.npz"
                materialize_feedback_intervention(
                    row=row,
                    loader=loader_by_role[role],
                    output_path=output_path,
                    role=role,
                    prediction_model=prediction_model,
                    shuffle_plan=shuffle_plan,
                    device=(
                        "cuda"
                        if args.device == "auto"
                        and torch.cuda.is_available()
                        else "cpu"
                        if args.device == "auto"
                        else args.device
                    ),
                )
                intervention_sources[role] = {
                    "npz_path": str(output_path.resolve())
                }
        stage_e_manifest = build_stage_e_loader_manifest(
            row=row,
            base_loader_manifest=base_path,
            intervention_sources=intervention_sources,
            campaign_spec_sha256=campaign["content_hash"],
            source=campaign["source"],
        )
        args.loader_manifest = loader_root / f"{row['row_id']}.json"
        write_immutable_json(args.loader_manifest, stage_e_manifest)
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
    loaded = load_stage_e_loaders_from_manifest(
        manifest_path=args.loader_manifest,
        campaign_root=args.campaign_root,
        row=row,
        campaign=campaign,
        target_registry=target_registry,
    )
    module = importlib.import_module("weaver.nn.model.ParticleTransformer")
    miniature = campaign["campaign_profile"] == "miniature_test"
    protocol = HOSDTrainingProtocol(
        maximum_epochs=2 if miniature else 40,
        campaign_profile="miniature_test" if miniature else "production",
    )
    import torch

    output_dir = (
        args.output_dir
        or args.campaign_root
        / "feedback"
        / row["row_id"]
        / f"seed_{args.seed}"
    )
    resolved_device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if row.get("control") in {
        "ORACLE_SUB",
        "SHUFFLED_PREDICTION",
        "SHUFFLED",
    }:
        source_rows = [
            value
            for value in plan["scientific_rows"]
            if value["target_id"] == row["target_id"]
            and value["interface"] == row["interface"]
            and value["gradient_path"] == "END_TO_END"
        ]
        if len(source_rows) != 1:
            raise ValueError(
                "post-hoc feedback source row is absent or duplicated"
            )
        source_row = {
            **source_rows[0],
            "pipeline_seed": int(args.seed),
            "encoder_component_seed": component_seed(
                int(args.seed), "encoder", "H_BASE"
            ),
            "feedback_component_seed": component_seed(
                int(args.seed), "feedback", source_rows[0]["row_id"]
            ),
        }
        source_model = build_feedback_model(
            source_row, weaver_module=module
        )
        source_root = (
            args.campaign_root
            / "feedback"
            / source_row["row_id"]
            / f"seed_{args.seed}"
        )
        source_checkpoint = source_root / "best_model_val.pt"
        checkpoint = torch.load(
            source_checkpoint, map_location="cpu", weights_only=False
        )
        if (
            checkpoint.get("contract") != "hosd_feedback_checkpoint_v2"
            or checkpoint.get("row_id") != source_row["row_id"]
            or checkpoint.get("stage_e_plan_sha256") != plan["content_hash"]
            or checkpoint.get("campaign_spec_sha256")
            != campaign["content_hash"]
            or checkpoint.get("source") != campaign["source"]
        ):
            raise ValueError("post-hoc feedback checkpoint lineage differs")
        source_model.load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )
        source_completion = load_hashed_json(
            source_root / "feedback_completion.json",
            expected_contract=FEEDBACK_COMPLETION_CONTRACT,
        )
        completion = evaluate_posthoc_feedback_control(
            source_model=source_model,
            design_select_loader=loaded["design_select_loader"],
            output_dir=output_dir,
            control_row=row,
            source_row=source_row,
            component_group_ids=loaded["component_group_ids"],
            source_checkpoint_path=source_checkpoint,
            source_completion=source_completion,
            stage_e_plan_sha256=plan["content_hash"],
            campaign_spec_sha256=campaign["content_hash"],
            lineage_hashes=loaded["lineage_hashes"],
            source=campaign["source"],
            device=resolved_device,
        )
        export_model = source_model
        export_row = source_row
        export_checkpoint = source_checkpoint
    else:
        model = build_feedback_model(row, weaver_module=module)
        deployed_ledger = feedback_model_flop_ledger(model)
        deployed_flops = float(deployed_ledger["deployed_total_flops"])
        deployed_parameters = exact_trainable_parameter_count(model)
        if (
            args.deployed_analytical_flops is not None
            and float(args.deployed_analytical_flops) != deployed_flops
        ):
            raise ValueError("Stage-E supplied FLOPs differ analytically")
        if (
            args.deployed_parameter_count is not None
            and int(args.deployed_parameter_count) != deployed_parameters
        ):
            raise ValueError("Stage-E supplied parameter count differs")
        completion = train_stage_e_feedback(
            model=model,
            train_loader=loaded["train_loader"],
            val_stop_loader=loaded["val_stop_loader"],
            design_select_loader=loaded["design_select_loader"],
            output_dir=output_dir,
            row=row,
            component_group_ids=loaded["component_group_ids"],
            stage_e_plan_sha256=plan["content_hash"],
            campaign_spec_sha256=campaign["content_hash"],
            lineage_hashes=loaded["lineage_hashes"],
            protocol=protocol,
            source=campaign["source"],
            deployed_analytical_flops=deployed_flops,
            deployed_parameter_count=deployed_parameters,
            device=resolved_device,
        )
        export_model = model
        export_row = row
        export_checkpoint = output_dir / "best_model_val.pt"
    export_manifest = None
    if bool(row.get("deployable")):
        export_descriptor = {
            "graph_id": row["row_id"],
            "graph_kind": "FEEDBACK",
            "row": dict(export_row),
            "seed": int(args.seed),
            "semantic_control_row": dict(row),
        }
        if row.get("control") in {
            "SHUFFLED_PREDICTION",
            "SHUFFLED",
        }:
            export_descriptor.update(
                {
                    "inference_control": "IDENTITY_BOUND_WRONG_EVENT_PREDICTION",
                    "shuffle_construction_rule": (
                        "sort_sha256(hosd_final_feedback_shuffle_v1||"
                        "target_id||split||identity)_then_rotate_left_one"
                    ),
                }
            )
        export_manifest = export_deployable_graph(
            descriptor=export_descriptor,
            research_model=export_model,
            representative_batch=next(iter(loaded["val_stop_loader"])),
            output_path=output_dir / "deployable_control.pt",
            checkpoint_sha256=hashlib.sha256(
                export_checkpoint.read_bytes()
            ).hexdigest(),
            lineage_hashes={
                "stage_e_plan": plan["content_hash"],
                "training_or_diagnostic_completion": completion[
                    "content_hash"
                ],
                **(
                    {
                        "stage_e_loader_manifest": loaded[
                            "lineage_hashes"
                        ]["stage_e_loader_manifest"]
                    }
                    if "stage_e_loader_manifest"
                    in loaded["lineage_hashes"]
                    else {}
                ),
            },
            source=campaign["source"],
            weaver_module=module,
            analytical_inference_flops_batch1_n128=int(
                feedback_model_flop_ledger(export_model)[
                    "deployed_total_flops"
                ]
            ),
        )
    summary.update(
        {
            "executed": True,
            "completion_sha256": completion["content_hash"],
            "deployable_export_sha256": (
                None
                if export_manifest is None
                else export_manifest["content_hash"]
            ),
        }
    )
    from teacher_logit_reco.hlt_offline_structure_distillation.wave_completion import (
        try_finalize_row_wave,
    )

    kind = "SCIENTIFIC" if row["row_kind"] == "SCIENTIFIC" else "CONTROL"
    expected = [
        item for item in plan["all_rows"] if item["row_kind"] == kind
    ]
    wave = try_finalize_row_wave(
        wave_id=f"stage_e_{kind.lower()}",
        expected_paths={
            item["row_id"]: args.campaign_root
            / "feedback"
            / item["row_id"]
            / f"seed_{args.seed}"
            / "feedback_completion.json"
            for item in expected
        },
        expected_rows={
            item["row_id"]: {
                "row_id": item["row_id"],
                "target_id": item["target_id"],
                "interface": item["interface"],
                "control": item["control"],
                "row_kind": item["row_kind"],
            }
            for item in expected
        },
        expected_contract=FEEDBACK_COMPLETION_CONTRACT,
        parent_hashes={"stage_e_plan": plan["content_hash"]},
        source=campaign["source"],
        output=args.campaign_root
        / "feedback"
        / (
            "scientific_row_completion.json"
            if kind == "SCIENTIFIC"
            else "mechanism_control_completion.json"
        ),
    )
    summary["wave_completion_sha256"] = (
        None if wave is None else wave["content_hash"]
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
