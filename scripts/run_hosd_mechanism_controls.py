#!/usr/bin/env python3
"""Compile or finalize the locked Stage-G mechanism-control bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    authorize_access,
    build_mechanism_control_plan,
    build_mechanism_summary,
    execute_mechanism_plan,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    COMBINATION_SELECTION_CONTRACT,
    MECHANISM_CONTROL_PLAN_CONTRACT,
    MECHANISM_RESULT_CONTRACT,
    STAGE_F_PLAN_CONTRACT,
    SINGLE_FAMILY_PHASE_LOCK_CONTRACT,
    STAGE_D_PLAN_CONTRACT,
    STAGE_E_PLAN_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--mode", choices=("compile", "execute", "finalize"), required=True
    )
    parser.add_argument("--result", action="append", default=[], type=Path)
    parser.add_argument("--base-roles-json", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    if args.mode in {"compile", "execute"}:
        lock = load_hashed_json(
            args.campaign_root / "combinations" / "locked_combination_choices.json",
            expected_contract=COMBINATION_SELECTION_CONTRACT,
        )
        selected = lock["selected_graph_definition"]
        artifact = build_mechanism_control_plan(
            combination_selection=lock,
            selected_combination=selected,
            source=campaign["source"],
        )
        plan_path = (
            args.campaign_root / "mechanism_controls" / "control_plan.json"
        )
        write_immutable_json(plan_path, artifact)
        if args.mode == "compile":
            output = args.output or plan_path
        else:
            if args.base_roles_json is None:
                raise ValueError(
                    "mechanism execution requires design-confirm base roles"
                )
            stage_f = load_hashed_json(
                args.campaign_root
                / "job_ledgers"
                / "stage_f_execution_plan.json",
                expected_contract=STAGE_F_PLAN_CONTRACT,
            )
            stage_d = load_hashed_json(
                args.campaign_root
                / "job_ledgers"
                / "stage_d_execution_plan.json",
                expected_contract=STAGE_D_PLAN_CONTRACT,
            )
            phase_lock = load_hashed_json(
                args.campaign_root
                / "auxiliary"
                / "single_family_phase_lock.json",
                expected_contract=SINGLE_FAMILY_PHASE_LOCK_CONTRACT,
            )
            stage_e = load_hashed_json(
                args.campaign_root
                / "job_ledgers"
                / "stage_e_execution_plan.json",
                expected_contract=STAGE_E_PLAN_CONTRACT,
            )
            target_registry = load_hashed_json(
                args.campaign_root
                / "registry"
                / "structure_target_registry.json",
                expected_contract="hosd_structure_target_registry_v1",
            )
            import importlib
            import torch

            device = (
                "cuda"
                if args.device == "auto" and torch.cuda.is_available()
                else "cpu"
                if args.device == "auto"
                else args.device
            )
            results = execute_mechanism_plan(
                plan=artifact,
                selected_graph=selected,
                stage_f_plan_sha256=stage_f["content_hash"],
                stage_d_plan=stage_d,
                phase_lock=phase_lock,
                stage_e_plan=stage_e,
                campaign=campaign,
                target_registry=target_registry,
                base_roles=json.loads(
                    args.base_roles_json.read_text(encoding="utf-8")
                ),
                campaign_root=args.campaign_root,
                weaver_module=importlib.import_module(
                    "weaver.nn.model.ParticleTransformer"
                ),
                device=device,
            )
            from teacher_logit_reco.hlt_offline_structure_distillation.wave_completion import (
                build_row_wave_completion,
            )

            completion = build_row_wave_completion(
                wave_id="stage_g_mechanism_controls",
                expected_rows={
                    row["intervention_id"]: {
                        "intervention_id": row["intervention_id"],
                        "kind": row["kind"],
                    }
                    for row in artifact["interventions"]
                },
                artifacts={
                    row["intervention_id"]: row for row in results
                },
                expected_contract=MECHANISM_RESULT_CONTRACT,
                parent_hashes={
                    "mechanism_plan": artifact["content_hash"],
                    "stage_f_plan": stage_f["content_hash"],
                },
                source=campaign["source"],
            )
            output = args.output or (
                args.campaign_root
                / "mechanism_controls"
                / "control_completion.json"
            )
            artifact = completion
    else:
        authorize_access(
            worker_role="design_confirmer",
            requested_resource="design_confirm_predictions",
        )
        plan = load_hashed_json(
            args.campaign_root / "mechanism_controls" / "control_plan.json",
            expected_contract=MECHANISM_CONTROL_PLAN_CONTRACT,
        )
        result_paths = args.result or [
            args.campaign_root
            / "mechanism_controls"
            / "results"
            / f"{row['intervention_id']}.json"
            for row in plan["interventions"]
        ]
        artifact = build_mechanism_summary(
            plan=plan,
            results=[
                load_hashed_json(path, expected_contract=MECHANISM_RESULT_CONTRACT)
                for path in result_paths
            ],
            source=campaign["source"],
        )
        output = args.output or (
            args.campaign_root
            / "mechanism_controls"
            / "design_confirm_summary.json"
        )
    publication = write_immutable_json(output, artifact)
    print(json.dumps({"content_hash": artifact["content_hash"], "publication": publication["status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
