"""Execution-complete Stage-B--E downstream manifest-plan factories.

The factories in this module are deliberately boring: they derive every row
from immutable registries and authenticated producer completions.  Scientific
metrics never affect row creation, so a negative result cannot shorten a wave.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import (
    canonical_sha256,
    load_hashed_json,
    validate_content_hash,
)
from .manifest_orchestration import (
    build_manifest_materialization_plan,
    publish_manifest_materialization_plan,
)
from .production import validate_production_campaign_binding
from .registry import EXPERT_ORDER
from .step4 import validate_stage_b_run_registry
from .step5 import validate_stage_c_run_registry
from .step7 import validate_stage_e_template_registry
from .task_completion import (
    TASK_MANIFEST_COMPLETION_CONTRACT,
    task_manifest_completion_path,
)


EARLY_CONTINUATION_FACTORY_CONTRACT = (
    "retb_stage_b_e_manifest_plan_factories_v2"
)
EARLY_CONTINUATION_TARGETS = (
    "offline_optimization_selector",
    "offline_shape_selector",
    "offline_complementarity",
    "offline_capacity_controls",
    "bridge_target_training",
    "bridge_content_certification",
    "target_coordinate_selector",
)


def _producer_completion(
    root: Path,
    *,
    producer_node_id: str,
) -> tuple[Path, dict[str, Any]]:
    path = task_manifest_completion_path(
        root, node_id=producer_node_id
    )
    if path.is_file():
        payload = load_hashed_json(
            path, expected_contract=TASK_MANIFEST_COMPLETION_CONTRACT
        )
    else:
        from .direct_completion import (
            DIRECT_NODE_COMPLETION_CONTRACT,
            direct_node_completion_path,
        )

        path = direct_node_completion_path(
            root, node_id=producer_node_id
        )
        direct = load_hashed_json(
            path, expected_contract=DIRECT_NODE_COMPLETION_CONTRACT
        )
        payload = {
            **direct,
            "task_count": 1,
            "completed_task_count": 1,
            "all_outputs_revalidated_after_last_row": True,
            "task_manifest_sha256": direct["content_hash"],
            "rows": [
                {
                    "task_index": 0,
                    "task_id": f"{producer_node_id}:direct",
                    "output_hashes": direct["output_hashes"],
                }
            ],
        }
    if (
        payload.get("node_id") != producer_node_id
        or payload.get("task_count") != payload.get("completed_task_count")
        or payload.get("all_outputs_revalidated_after_last_row") is not True
    ):
        raise ValueError(
            f"{producer_node_id} producer completion is incomplete"
        )
    return path, payload


def _row(
    *,
    target: str,
    index: int,
    argv: Sequence[str],
    outputs: Sequence[str | Path],
    campaign_sha256: str,
    graph_sha256: str,
    producer_completion_sha256: str,
    extra_input_hashes: Mapping[str, str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    inputs = {
        "campaign_spec": campaign_sha256,
        "production_graph": graph_sha256,
        "producer_completion": producer_completion_sha256,
        **dict(extra_input_hashes or {}),
    }
    return {
        "task_id": f"{target}:{int(index)}",
        "argv": [str(value) for value in argv],
        "environment": {
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
            **dict(environment or {}),
        },
        "expected_outputs": [str(Path(value)) for value in outputs],
        "input_artifact_hashes": inputs,
    }


def _publish(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str,
    target_node_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root = Path(campaign_root).resolve()
    validate_production_campaign_binding(production_graph, campaign)
    completion_path, completion = _producer_completion(
        root, producer_node_id=producer_node_id
    )
    node = next(
        value
        for value in production_graph["nodes"]
        if value["node_id"] == target_node_id
    )
    dynamic = bool(node["dynamic_continuation"])
    plan = build_manifest_materialization_plan(
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id=target_node_id,
        rows=rows,
        trigger_artifact_path=completion_path if dynamic else None,
        trigger_artifact_sha256=(
            completion["content_hash"] if dynamic else None
        ),
    )
    publication = publish_manifest_materialization_plan(
        campaign_root=root,
        plan=plan,
        campaign=campaign,
        production_graph=production_graph,
    )
    return {
        "target_node_id": target_node_id,
        "plan_sha256": plan["content_hash"],
        "row_count": len(rows),
        "publication": publication,
    }


def _context(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str,
) -> tuple[Path, str, str, str]:
    root = Path(campaign_root).resolve()
    campaign_sha = validate_content_hash(campaign)
    graph_sha = validate_content_hash(production_graph)
    _, completion = _producer_completion(
        root, producer_node_id=producer_node_id
    )
    return root, campaign_sha, graph_sha, completion["content_hash"]


def build_offline_optimization_selector_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "offline_expert_training",
) -> dict[str, Any]:
    target = "offline_optimization_selector"
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    registry = load_hashed_json(
        root / "registry" / "retb_stage_b_runs.json"
    )
    registry_sha = validate_stage_b_run_registry(registry)
    output_root = root / "selection" / "stage_b"
    outputs = (
        output_root / "optimization_candidate_metrics.json",
        output_root / "locked_optimization_selection.json",
        output_root / "optimization_followup_rows.json",
    )
    row = _row(
        target=target,
        index=0,
        argv=(
            "python",
            "scripts/execute_retb_offline_optimization_wave.py",
            "--campaign-root",
            str(root),
            "--output-dir",
            str(output_root),
        ),
        outputs=outputs,
        campaign_sha256=campaign_sha,
        graph_sha256=graph_sha,
        producer_completion_sha256=completion_sha,
        extra_input_hashes={"stage_b_registry": registry_sha},
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id=target,
        rows=[row],
    )


def _stage_c_factory_row(
    *,
    target: str,
    worker: str,
    outputs: Sequence[str | Path],
    root: Path,
    campaign_sha: str,
    graph_sha: str,
    completion_sha: str,
    registry_sha: str,
) -> dict[str, Any]:
    return _row(
        target=target,
        index=0,
        argv=(
            "python",
            worker,
            "--campaign-root",
            str(root),
        ),
        outputs=outputs,
        campaign_sha256=campaign_sha,
        graph_sha256=graph_sha,
        producer_completion_sha256=completion_sha,
        extra_input_hashes={"stage_c_registry": registry_sha},
    )


def build_offline_shape_selector_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "offline_fusion_training",
) -> dict[str, Any]:
    target = "offline_shape_selector"
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    registry = load_hashed_json(
        root / "registry" / "retb_stage_c_runs.json"
    )
    registry_sha = validate_stage_c_run_registry(registry)
    selection_root = root / "selection" / "stage_c"
    aliases = (
        "SHAPE_COMPACT",
        "SHAPE_HIGH",
        "HET_PHYSICS",
        "HET_SELECTED",
        "HET_BEAM",
    )
    shape_outputs: list[Path] = [
        selection_root / "uniform_shape_metrics.json",
        selection_root / "locked_offline_shapes.json",
        root / "selection" / "retb_offline_shapes.json",
        root / "selection" / "retb_heterogeneous_shapes.json",
    ]
    for alias in aliases:
        for seed in (101, 202, 303):
            fusion = (
                root
                / "selection"
                / "offline_fusions"
                / alias
                / f"seed_{seed}"
            )
            shape_outputs.extend(
                [
                    fusion / "fusion_registration.json",
                    fusion / "best_model_val.pt",
                    fusion / "val_design_inference.json",
                ]
            )
            for expert in EXPERT_ORDER:
                expert_root = (
                    root
                    / "selection"
                    / "offline_experts"
                    / alias
                    / expert
                    / f"seed_{seed}"
                )
                shape_outputs.extend(
                    [
                        expert_root / "checkpoint_registration.json",
                        expert_root / "best_model_val.pt",
                    ]
                )
    row = _stage_c_factory_row(
        target=target,
        worker="scripts/execute_retb_offline_shape_wave.py",
        outputs=shape_outputs,
        root=root,
        campaign_sha=campaign_sha,
        graph_sha=graph_sha,
        completion_sha=completion_sha,
        registry_sha=registry_sha,
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id=target,
        rows=[row],
    )


def build_offline_complementarity_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "offline_fusion_training",
) -> dict[str, Any]:
    target = "offline_complementarity"
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    registry = load_hashed_json(
        root / "registry" / "retb_stage_c_runs.json"
    )
    registry_sha = validate_stage_c_run_registry(registry)
    report_root = root / "reports" / "stage_c"
    row = _stage_c_factory_row(
        target=target,
        worker="scripts/execute_retb_offline_complementarity_wave.py",
        outputs=(
            report_root / "subset_readout_metrics.json",
            report_root / "offline_complementarity.json",
        ),
        root=root,
        campaign_sha=campaign_sha,
        graph_sha=graph_sha,
        completion_sha=completion_sha,
        registry_sha=registry_sha,
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id=target,
        rows=[row],
    )


def build_offline_capacity_controls_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "offline_shape_selector",
) -> dict[str, Any]:
    target = "offline_capacity_controls"
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    registry = load_hashed_json(
        root / "registry" / "retb_stage_c_runs.json"
    )
    registry_sha = validate_stage_c_run_registry(registry)
    report = root / "reports" / "stage_c" / "offline_capacity_controls.json"
    control_root = root / "runs" / "stage_c" / "capacity_controls"
    controls = (
        "O_BASE",
        "O_WIDE",
        "O_MONO_PARAM",
        "O_MONO_FLOP",
        "O_BASE_LONG",
        "O_FULLREL",
        "O_GROUPED_HEAD_REL",
        "O_7X_UNBIASED_ENSEMBLE",
        "O_7X_UNBIASED_TOKEN_FUSION",
        "O_RELATION_EXPERT_TOKEN_FUSION",
    )
    profile_paths = {
        control: (
            control_root
            / control
            / (
                "complete_graph_profile.json"
                if control
                in {
                    "O_7X_UNBIASED_ENSEMBLE",
                    "O_RELATION_EXPERT_TOKEN_FUSION",
                }
                else "fusion_training/complete_graph_profile.json"
                if control == "O_7X_UNBIASED_TOKEN_FUSION"
                else "training/complete_graph_profile.json"
            )
        )
        for control in controls
    }
    directly_trained = (
        "O_BASE",
        "O_WIDE",
        "O_MONO_PARAM",
        "O_MONO_FLOP",
        "O_BASE_LONG",
        "O_FULLREL",
        "O_GROUPED_HEAD_REL",
    )
    seven_seeds = (101, 202, 303, 404, 505, 606, 707)
    row = _stage_c_factory_row(
        target=target,
        worker="scripts/execute_retb_offline_capacity_wave.py",
        outputs=(
            report,
            *(
                control_root / control / "control_registration.json"
                for control in controls
            ),
            *(
                control_root / control / "label_exposure.json"
                for control in controls
            ),
            *(profile_paths[control] for control in controls),
            *(
                control_root
                / control
                / "training"
                / "training_registration.json"
                for control in directly_trained
            ),
            *(
                control_root / control / "training" / "best_model_val.pt"
                for control in directly_trained
            ),
            *(
                control_root
                / "O_7X_UNBIASED_ENSEMBLE"
                / f"member_seed_{seed}"
                / name
                for seed in seven_seeds
                for name in (
                    "training_registration.json",
                    "best_model_val.pt",
                )
            ),
            *(
                control_root
                / "O_7X_UNBIASED_TOKEN_FUSION"
                / f"member_seed_{seed}"
                / name
                for seed in seven_seeds
                for name in (
                    "training_registration.json",
                    "best_model_val.pt",
                )
            ),
            control_root
            / "O_7X_UNBIASED_TOKEN_FUSION"
            / "fusion_training"
            / "training_registration.json",
            control_root
            / "O_7X_UNBIASED_TOKEN_FUSION"
            / "fusion_training"
            / "best_model_val.pt",
            control_root
            / "O_7X_UNBIASED_TOKEN_FUSION"
            / "composite_training_lineage.json",
        ),
        root=root,
        campaign_sha=campaign_sha,
        graph_sha=graph_sha,
        completion_sha=completion_sha,
        registry_sha=registry_sha,
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id=target,
        rows=[row],
    )


def _pilot_output(root: Path, *, shape: str, expert: str, seed: int) -> Path:
    identity = f"pilot_t0:{shape}:{expert}:seed_{seed}"
    return (
        root
        / "runs"
        / "stage_e"
        / "pilots"
        / canonical_sha256(identity)[:20]
    )


def _template_slug(template: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(template))[:16]


def build_bridge_target_training_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "bridge_pilot_training",
) -> dict[str, Any]:
    target = "bridge_target_training"
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    registry = load_hashed_json(
        root / "registry" / "retb_stage_e_templates.json"
    )
    registry_sha = validate_stage_e_template_registry(registry)
    rows: list[dict[str, Any]] = []
    for shape in registry["shapes"]:
        for expert in registry["expert_order"]:
            for seed in registry["pipeline_seeds"]:
                pilot = _pilot_output(
                    root, shape=shape, expert=expert, seed=int(seed)
                )
                pilot_registration = load_hashed_json(
                    pilot / "checkpoint_registration.json"
                )
                if (
                    pilot_registration.get("expert_id") != expert
                    or pilot_registration.get("shape_id") != shape
                    or int(pilot_registration.get("pipeline_seed", -1))
                    != int(seed)
                ):
                    raise ValueError("bridge pilot identity differs")
                parent_root = (
                    root
                    / "selection"
                    / "stage_e_parents"
                    / shape
                    / expert
                    / f"seed_{seed}"
                )
                for template in registry["candidate_templates"]:
                    slug = _template_slug(template)
                    output = (
                        root
                        / "runs"
                        / "stage_e"
                        / "targets"
                        / shape
                        / expert
                        / f"seed_{seed}"
                        / slug
                    )
                    argv = [
                        "python",
                        "scripts/train_retb_bridge_target.py",
                        "--campaign-root",
                        str(root),
                        "--pipeline-seed",
                        str(seed),
                        "--expert-id",
                        expert,
                        "--shape-id",
                        shape,
                        "--target-mode",
                        str(template["target_mode"]),
                        "--lambda-pred",
                        str(template["lambda_pred"]),
                        "--pilot-registration",
                        str(pilot / "checkpoint_registration.json"),
                        "--pilot-checkpoint",
                        str(pilot / "best_model_val.pt"),
                        "--parent-root",
                        str(parent_root),
                        "--output-dir",
                        str(output),
                    ]
                    if template["bridge_dimension"] is not None:
                        argv.extend(
                            [
                                "--bridge-dimension",
                                str(template["bridge_dimension"]),
                            ]
                        )
                    if template["unfreeze_final_two_blocks"]:
                        argv.append("--unfreeze-final-two-blocks")
                    rows.append(
                        _row(
                            target=target,
                            index=len(rows),
                            argv=argv,
                            outputs=(
                                output / "materialized_run.json",
                                output / "checkpoint_registration.json",
                                output / "best_model_val.pt",
                                output / "val_stop_metrics.json",
                                output / "val_design_metrics.json",
                                output / "model_train_coordinate_arrays.npz",
                                output / "val_stop_coordinate_arrays.npz",
                                output / "val_design_coordinate_arrays.npz",
                                output / "certification_arrays.npz",
                            ),
                            campaign_sha256=campaign_sha,
                            graph_sha256=graph_sha,
                            producer_completion_sha256=completion_sha,
                            extra_input_hashes={
                                "stage_e_templates": registry_sha,
                                "pilot_registration": pilot_registration[
                                    "content_hash"
                                ],
                                "candidate_template": canonical_sha256(
                                    dict(template)
                                ),
                            },
                            environment={
                                "RETB_TARGET_TEMPLATE_SHA256": (
                                    canonical_sha256(dict(template))
                                )
                            },
                        )
                    )
    expected = int(registry["candidate_membership_count"])
    if len(rows) != expected:
        raise RuntimeError(
            f"bridge target row count differs: {len(rows)} != {expected}"
        )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id=target,
        rows=rows,
    )


def build_bridge_content_certification_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "bridge_target_training",
) -> dict[str, Any]:
    target = "bridge_content_certification"
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    registry = load_hashed_json(
        root / "registry" / "retb_stage_e_templates.json"
    )
    registry_sha = validate_stage_e_template_registry(registry)
    output_root = root / "selection" / "stage_e"
    row = _row(
        target=target,
        index=0,
        argv=(
            "python",
            "scripts/execute_retb_bridge_certification_wave.py",
            "--campaign-root",
            str(root),
            "--output-dir",
            str(output_root),
        ),
        outputs=(
            output_root / "bridge_certification_index.json",
            output_root / "bridge_coordinate_score_table.json",
            output_root / "bridge_eligibility_index.json",
        ),
        campaign_sha256=campaign_sha,
        graph_sha256=graph_sha,
        producer_completion_sha256=completion_sha,
        extra_input_hashes={"stage_e_templates": registry_sha},
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id=target,
        rows=[row],
    )


def build_target_coordinate_selector_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "bridge_content_certification",
) -> dict[str, Any]:
    target = "target_coordinate_selector"
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    index = load_hashed_json(
        root / "selection" / "stage_e" / "bridge_certification_index.json"
    )
    output = root / "selection" / "locked_bridge_coordinates.json"
    row = _row(
        target=target,
        index=0,
        argv=(
            "python",
            "scripts/execute_retb_target_coordinate_selection.py",
            "--campaign-root",
            str(root),
            "--output",
            str(output),
        ),
        outputs=(output,),
        campaign_sha256=campaign_sha,
        graph_sha256=graph_sha,
        producer_completion_sha256=completion_sha,
        extra_input_hashes={
            "bridge_certification_index": index["content_hash"]
        },
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id=target,
        rows=[row],
    )


EARLY_PLAN_FACTORIES: dict[str, Callable[..., dict[str, Any]]] = {
    "offline_optimization_selector": (
        build_offline_optimization_selector_manifest_plan
    ),
    "offline_shape_selector": build_offline_shape_selector_manifest_plan,
    "offline_complementarity": (
        build_offline_complementarity_manifest_plan
    ),
    "offline_capacity_controls": (
        build_offline_capacity_controls_manifest_plan
    ),
    "bridge_target_training": build_bridge_target_training_manifest_plan,
    "bridge_content_certification": (
        build_bridge_content_certification_manifest_plan
    ),
    "target_coordinate_selector": (
        build_target_coordinate_selector_manifest_plan
    ),
}


__all__ = [
    "EARLY_CONTINUATION_FACTORY_CONTRACT",
    "EARLY_CONTINUATION_TARGETS",
    "EARLY_PLAN_FACTORIES",
    *[
        f"build_{target}_manifest_plan"
        for target in EARLY_CONTINUATION_TARGETS
    ],
]
