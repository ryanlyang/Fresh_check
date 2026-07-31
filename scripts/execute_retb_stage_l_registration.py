#!/usr/bin/env python3
"""Register concrete Stage-L graphs and emit all confirmation rows."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.confirmation import (  # noqa: E402
    build_stage_l_graph_registry,
    validate_stage_l_graph_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.confirmation_execution import (  # noqa: E402
    CONFIRMATION_EXECUTION_PLAN_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    canonical_sha256,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.late_plan_factories import (  # noqa: E402
    build_late_factory_input,
    publish_late_factory_input,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


SEEDS = (101, 202, 303)


def _run_id(
    *,
    role: str,
    consumer: str,
    variant: str,
    token_input: str = "TOKEN_PREDICTED",
    seed: int,
) -> str:
    return (
        f"RETB_{role}_{consumer}_{variant}_ND0_NONE_"
        f"{token_input}_S{seed}"
    )


def _definition(
    *,
    graph_id: str,
    role: str,
    category: str,
    carried_role: str,
    consumer: str,
    variant: str,
    baseline: str,
    token_input: str = "TOKEN_PREDICTED",
    registry_carried_role: str | None = None,
) -> dict[str, Any]:
    recipe = {
        "carried_shape_role": (
            carried_role
            if registry_carried_role is None
            else registry_carried_role
        ),
        "source_carried_shape_role": carried_role,
        "consumer_kind": consumer,
        "model_variant": variant,
        "native_dropout_mode": "ND0_NONE",
        "token_input": token_input,
    }
    run_ids = {
        str(seed): _run_id(
            role=carried_role,
            consumer=consumer,
            variant=variant,
            token_input=token_input,
            seed=seed,
        )
        for seed in SEEDS
    }
    return {
        "graph_id": graph_id,
        "role": role,
        "semantic_category": category,
        "shortlist_eligible": role == "scientific_candidate",
        "named_baseline_graph_id": baseline,
        "shape_id": carried_role,
        "complete_graph_definition_sha256": canonical_sha256(
            {"contract": "retb_complete_graph_definition_v1", **recipe}
        ),
        "training_recipe_sha256": canonical_sha256(
            {
                "contract": "retb_500k_training_recipe_v1",
                **recipe,
                "epochs": 40,
                "matched_pipeline_seeds": list(SEEDS),
            }
        ),
        "inference_recipe_sha256": canonical_sha256(
            {
                "contract": "retb_hlt_only_inference_recipe_v1",
                **recipe,
            }
        ),
        "deployable_without_offline_or_oracle": True,
        "predicts_tokens": True,
        "configuration": {
            **recipe,
            "run_ids_by_seed": run_ids,
        },
    }


def _definitions(candidate_roles: list[str]) -> list[dict[str, Any]]:
    primary = "G_PRIMARY_PF"
    uniform = [
        role
        for role in ("SHAPE_COMPACT", "SHAPE_HIGH")
        if role in candidate_roles
    ]
    if len(uniform) == 1:
        uniform_label = "SHAPE_COMPACT_AND_HIGH"
        uniform_rows = [(uniform_label, uniform[0])]
    else:
        uniform_rows = [(role, role) for role in uniform]
    rows = [
        _definition(
            graph_id=primary,
            role="reference_baseline",
            category="PRIMARY_BASELINE",
            carried_role="HET_SELECTED",
            consumer="PF_FROZEN",
            variant="PF_FROZEN",
            baseline=primary,
        ),
        _definition(
            graph_id="G_FROZEN_RECONSTRUCTION",
            role="architecture_control",
            category="FROZEN_RECONSTRUCTION",
            carried_role="HET_PHYSICS",
            consumer="PF_FROZEN",
            variant="PF_FROZEN",
            baseline=primary,
        ),
        _definition(
            graph_id="G_TOKEN_REFINER",
            role="scientific_candidate",
            category="TOKEN_REFINER",
            carried_role="HET_SELECTED",
            consumer="TR_REFINE",
            variant="TR2_ALL_NATIVE",
            baseline=primary,
        ),
        _definition(
            graph_id="G_CONSTRAINED_ADAPTER",
            role="scientific_candidate",
            category="CONSTRAINED_ADAPTER",
            carried_role="HET_SELECTED",
            consumer="HF_ADAPTER",
            variant="R2_PREDICTED_PLUS_ALL_NATIVE_EXPERTS",
            baseline=primary,
        ),
        _definition(
            graph_id="G_UNRESTRICTED",
            role="scientific_candidate",
            category="UNRESTRICTED_FUSION",
            carried_role="HET_SELECTED",
            consumer="HF_UNRESTRICTED",
            variant="F_TOKEN_PLUS_EXPERT_LOGITS",
            token_input="TOKEN_PREDICTED",
            baseline=primary,
        ),
    ]
    rows.extend(
        _definition(
            graph_id=f"G_UNIFORM_{label}",
            role="scientific_candidate",
            category="UNIFORM_FINALIST",
            carried_role=source_role,
            consumer="HF_UNRESTRICTED",
            variant="F_TOKEN_PLUS_EXPERT_LOGITS",
            token_input="TOKEN_REFINED_SELECTED",
            baseline=primary,
            registry_carried_role=label,
        )
        for label, source_role in uniform_rows
    )
    rows.extend(
        _definition(
            graph_id=f"G_NATIVE_{label}",
            role="architecture_control",
            category="NATIVE_HLT_FUSION",
            carried_role=source_role,
            registry_carried_role=label,
            consumer="HF_ADAPTER",
            variant="R3_NATIVE_ONLY_MATCHED_TO_R2",
            baseline=primary,
        )
        for label, source_role in uniform_rows
    )
    rows.extend(
        _definition(
            graph_id=f"G_FROZEN_{label}",
            role="architecture_control",
            category="FROZEN_RECONSTRUCTION",
            carried_role=source_role,
            registry_carried_role=label,
            consumer="PF_FROZEN",
            variant="PF_FROZEN",
            baseline=primary,
        )
        for label, source_role in uniform_rows
    )
    rows.extend(
        _definition(
            graph_id=f"G_HETEROGENEOUS_{role}",
            role="scientific_candidate",
            category="HETEROGENEOUS_FINALIST",
            carried_role=role,
            consumer="HF_UNRESTRICTED",
            variant="F_TOKEN_PLUS_EXPERT_LOGITS",
            token_input="TOKEN_REFINED_SELECTED",
            baseline=primary,
        )
        for role in ("HET_PHYSICS", "HET_SELECTED", "HET_BEAM")
    )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    graph = load_hashed_json(root / "job_ledgers" / "production_graph.json")
    step12 = load_hashed_json(
        root / "registry" / "retb_step12_final_consumers_bundle.json"
    )
    export_index = load_hashed_json(
        root / "exports" / "deployable_export_index.json"
    )
    robustness = load_hashed_json(
        root / "controls" / "robustness" / "robustness_bundle.json"
    )
    semantic = load_hashed_json(
        root / "controls" / "semantic" / "semantic_controls_bundle.json"
    )
    carried = {
        role: load_hashed_json(
            root
            / "selection"
            / "predictor_bundle"
            / "carried"
            / f"{role}.json"
        )
        for role in (
            "SHAPE_COMPACT",
            "SHAPE_HIGH",
            "HET_PHYSICS",
            "HET_SELECTED",
            "HET_BEAM",
        )
    }
    candidate_roles = list(
        dict.fromkeys(
            (
                "SHAPE_COMPACT"
                if carried["SHAPE_COMPACT"]["coordinate_id"]
                != carried["SHAPE_HIGH"]["coordinate_id"]
                else "SHAPE_COMPACT_AND_HIGH",
                *(
                    ["SHAPE_HIGH"]
                    if carried["SHAPE_COMPACT"]["coordinate_id"]
                    != carried["SHAPE_HIGH"]["coordinate_id"]
                    else []
                ),
            )
        )
    )
    # The registry's candidate shape IDs are the actual immutable coordinate
    # IDs, while carried roles remain explicit in each graph configuration.
    candidate_shape_ids = list(
        dict.fromkeys(
            [
                carried["SHAPE_COMPACT"]["coordinate_id"],
                carried["SHAPE_HIGH"]["coordinate_id"],
            ]
        )
    )
    definitions = _definitions(
        ["SHAPE_COMPACT", "SHAPE_HIGH"]
        if len(candidate_shape_ids) == 2
        else ["SHAPE_COMPACT"]
    )
    required_runs = {
        run_id
        for definition in definitions
        for run_id in definition["configuration"]["run_ids_by_seed"].values()
    }
    if not required_runs <= set(export_index["exports"]):
        raise ValueError("Stage-L definitions lack completed deployable exports")
    snapshot = source_snapshot(REPO_ROOT)
    registry = bind_source(
        build_stage_l_graph_registry(
            definitions=definitions,
            step12_bundle_sha256=step12["content_hash"],
            candidate_shape_ids=candidate_shape_ids,
            robustness_controls_completion_sha256=robustness["content_hash"],
            semantic_controls_completion_sha256=semantic["content_hash"],
        ),
        source_snapshot=snapshot,
    )
    validate_stage_l_graph_registry(registry)
    registry_path = root / "selection" / "stage_l" / "graph_registry.json"
    write_immutable_json(registry_path, registry)
    rows = []
    evidence_root = root / "confirmation" / "stage_l"
    for index, (definition, seed) in enumerate(
        (definition, seed)
        for definition in registry["definitions"]
        for seed in SEEDS
    ):
        graph_id = definition["graph_id"]
        evidence = evidence_root / graph_id / f"seed_{seed}" / "evidence"
        confirmation = (
            evidence_root / graph_id / f"seed_{seed}" / "confirmation.json"
        )
        component_paths = {
            name: str(
                (
                    root
                    / "exports"
                    / definition["configuration"]["run_ids_by_seed"][str(seed)]
                    / (
                        "deployable_retb_graph.json"
                        if name == "deployable_export"
                        else "complete_graph_capacity.json"
                    )
                )
                if name in {"deployable_export", "complete_graph_capacity"}
                else (
                    root
                    / "runs"
                    / "final_consumers"
                    / definition["configuration"]["run_ids_by_seed"][str(seed)]
                    / (
                        (
                            "val_design/"
                            "final_consumer_predictions_manifest.json"
                        )
                        if name == "prediction_manifest"
                        else ""
                    )
                )
                if name == "prediction_manifest"
                else evidence
                / (
                    "metrics.json"
                    if name == "metrics_artifact"
                    else f"{name}.json"
                )
            )
            for name in (
                "offline_experts",
                "offline_fusion",
                "offline_target_cache",
                "native_hlt_experts",
                "native_hlt_fusion",
                "predictor_bundle",
                "refiner_or_identity",
                "final_consumer",
                "deployable_export",
                "complete_graph_capacity",
                "prediction_manifest",
                "metrics_artifact",
                "paired_statistics",
            )
        }
        run_id = definition["configuration"]["run_ids_by_seed"][str(seed)]
        run_root = root / "runs" / "final_consumers" / run_id
        reference_prediction = (
            run_root / "final_consumer_predictions_manifest.json"
        )
        if reference_prediction.is_file():
            component_paths["prediction_manifest"] = str(reference_prediction)
        label_lock = carried[
            definition["configuration"]["source_carried_shape_role"]
        ]
        plan = bind_source(
            with_content_hash(
                {
                    "contract": CONFIRMATION_EXECUTION_PLAN_CONTRACT,
                    "schema_version": 1,
                    "graph_id": graph_id,
                    "pipeline_seed": seed,
                    "stage_l_graph_registry_sha256": registry["content_hash"],
                    "steps": [
                        {
                            "step_id": "step_000",
                            "argv": [
                                "python",
                                "scripts/prepare_retb_500k_confirmation_evidence.py",
                                "--campaign-root",
                                str(root),
                                "--graph-registry",
                                str(registry_path),
                                "--graph-id",
                                graph_id,
                                "--pipeline-seed",
                                str(seed),
                                "--output-dir",
                                str(evidence),
                            ],
                            "expected_outputs": [
                                str(evidence / "evidence_index.json"),
                                str(evidence / "training_summary.json"),
                                str(evidence / "metrics.json"),
                                str(evidence / "paired_statistics.json"),
                            ],
                        }
                    ],
                    "component_artifacts": component_paths,
                    "training_summary": str(evidence / "training_summary.json"),
                    "val_design_label_manifest_sha256": label_lock[
                        "selection_data_hashes"
                    ]["label_manifests"][str(seed)],
                }
            ),
            source_snapshot=snapshot,
        )
        plan_path = evidence.parent / "execution_plan.json"
        write_immutable_json(plan_path, plan)
        export = export_index["exports"][run_id]
        rows.append(
            {
                "task_id": f"confirmation_500k:{index}",
                "argv": [
                    "python",
                    "scripts/execute_retb_500k_seed_confirmation.py",
                    "--campaign-root",
                    str(root),
                    "--graph-registry",
                    str(registry_path),
                    "--execution-plan",
                    str(plan_path),
                    "--output",
                    str(confirmation),
                ],
                "expected_outputs": [str(confirmation)],
                "input_artifact_hashes": {
                    "campaign_spec": campaign["content_hash"],
                    "production_graph": graph["content_hash"],
                    "graph_registry": registry["content_hash"],
                    "execution_plan": plan["content_hash"],
                    "deployment_export": export["export_sha256"],
                    "complete_graph_capacity": export[
                        "complete_graph_capacity_sha256"
                    ],
                },
                "environment": {
                    "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
                    "RETB_GRAPH_ID": graph_id,
                    "RETB_PIPELINE_SEED": str(seed),
                },
            }
        )
    factory = build_late_factory_input(
        target_node_id="confirmation_500k",
        producer_node_id="stage_l_graph_registration",
        campaign_spec_sha256=campaign["content_hash"],
        production_graph_sha256=graph["content_hash"],
        producer_task_manifest_sha256=canonical_sha256(
            {
                "kind": "stage_l_registration_controller_v1",
                "graph_registry_sha256": registry["content_hash"],
            }
        ),
        rows=rows,
        coverage={
            "all_predeclared_rows_present": True,
            "scientific_metric_used_for_membership": False,
            "incomplete_wave_permitted": False,
            "required_graph_ids": [
                definition["graph_id"]
                for definition in registry["definitions"]
            ],
        },
        source=campaign["source"],
    )
    # The factory validates the producer identity against its completion
    # later. Directly use the current task-manifest hash recorded by the
    # materialized stage_l_graph_registration task.
    task_manifest = load_hashed_json(
        root / "job_ledgers" / "tasks" / "stage_l_graph_registration.json"
    )
    factory = build_late_factory_input(
        target_node_id="confirmation_500k",
        producer_node_id="stage_l_graph_registration",
        campaign_spec_sha256=campaign["content_hash"],
        production_graph_sha256=graph["content_hash"],
        producer_task_manifest_sha256=task_manifest["content_hash"],
        rows=rows,
        coverage=factory["coverage"],
        source=campaign["source"],
    )
    publish_late_factory_input(
        campaign_root=root,
        payload=factory,
        target_node_id="confirmation_500k",
        producer_node_id="stage_l_graph_registration",
        campaign=campaign,
        production_graph=graph,
        producer_task_manifest_sha256=task_manifest["content_hash"],
    )
    controller = bind_source(
        with_content_hash(
            {
                "contract": "retb_stage_l_registration_controller_v1",
                "schema_version": 1,
                "stage_l_graph_registry_sha256": registry["content_hash"],
                "confirmation_factory_input_sha256": factory["content_hash"],
                "definition_count": len(registry["definitions"]),
                "confirmation_row_count": len(rows),
                "all_scientific_rows_emitted": True,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=snapshot,
    )
    write_immutable_json(args.output, controller)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
