"""Automatic Stage-F--J manifest-plan factories.

Factories derive rows only from immutable campaign registries, selector locks,
and revalidated producer completions.  Row membership never depends on
whether an upstream scientific metric was positive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from .contracts import load_hashed_json
from .early_continuation import _context, _publish, _row
from .registry import EXPERT_ORDER


MIDDLE_PLAN_FACTORY_CONTRACT = "retb_stage_f_j_plan_factories_v1"
MIDDLE_PLAN_FACTORY_TARGETS = (
    "target_cache_build",
    "target_normalizers",
    "predictor_training",
    "uncertainty_calibration",
    "predictor_bundle_selector",
    "oracle_substitutions",
    "joint_predictor_training",
    "joint_predictor_selector",
    "final_consumer_training",
    "deployable_export",
)


def _coordinate_root(
    root: Path, coordinate_index: int, shape_id: str
) -> Path:
    return (
        root
        / "inputs"
        / "target_caches"
        / f"coordinate_{coordinate_index:03d}"
        / shape_id
    )


def build_target_cache_build_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "step8_target_cache_contracts",
) -> dict[str, Any]:
    target = "target_cache_build"
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    selection = load_hashed_json(
        root / "selection" / "locked_bridge_coordinates.json"
    )
    systems = list(selection["locked_coordinate_systems"])
    stage_e_registry = load_hashed_json(
        root / "registry" / "retb_stage_e_templates.json"
    )
    shapes = tuple(str(value) for value in stage_e_registry["shapes"])
    if not systems:
        raise ValueError("locked target-coordinate systems are empty")
    rows = []
    for coordinate_index, system in enumerate(systems):
        coordinate_sha = system["coordinate_contract_sha256"]
        for shape in shapes:
            for seed in (101, 202, 303):
                output = (
                    _coordinate_root(root, coordinate_index, shape)
                    / f"seed_{seed}"
                )
                outputs = []
                for split in ("model_train", "val_stop", "val_design"):
                    split_output = output / split
                    outputs.extend(
                        [
                            split_output / "identity_manifest.json",
                            split_output / "fusion_registration.json",
                            split_output / "target_lineage.json",
                            split_output
                            / "target_cache_specification.json",
                            split_output / "target_cache_manifest.json",
                        ]
                    )
                rows.append(
                    _row(
                        target=target,
                        index=len(rows),
                        argv=(
                            "python",
                            "scripts/execute_retb_target_cache_row.py",
                            "--campaign-root",
                            str(root),
                            "--coordinate-index",
                            str(coordinate_index),
                            "--shape-id",
                            shape,
                            "--pipeline-seed",
                            str(seed),
                            "--split",
                            "all",
                            "--output-dir",
                            str(output),
                        ),
                        outputs=outputs,
                        campaign_sha256=campaign_sha,
                        graph_sha256=graph_sha,
                        producer_completion_sha256=completion_sha,
                        extra_input_hashes={
                            "locked_bridge_coordinates": selection[
                                "content_hash"
                            ],
                            "coordinate_contract": coordinate_sha,
                        },
                        environment={
                            "RETB_COORDINATE_INDEX": str(
                                coordinate_index
                            ),
                            "RETB_SHAPE_ID": shape,
                            "RETB_PIPELINE_SEED": str(seed),
                            "RETB_DATA_SPLIT": (
                                "all_prelock_splits"
                            ),
                        },
                    )
                )
    expected = len(systems) * len(shapes) * 3
    if len(rows) != expected:
        raise RuntimeError("target-cache row coverage differs")
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id=target,
        rows=rows,
    )


def build_target_normalizers_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "target_cache_build",
) -> dict[str, Any]:
    target = "target_normalizers"
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    selection = load_hashed_json(
        root / "selection" / "locked_bridge_coordinates.json"
    )
    stage_e_registry = load_hashed_json(
        root / "registry" / "retb_stage_e_templates.json"
    )
    shapes = tuple(str(value) for value in stage_e_registry["shapes"])
    rows = []
    for coordinate_index, system in enumerate(
        selection["locked_coordinate_systems"]
    ):
        for shape in shapes:
            for seed in (101, 202, 303):
                cache = (
                    _coordinate_root(root, coordinate_index, shape)
                    / f"seed_{seed}"
                    / "model_train"
                )
                manifest = load_hashed_json(
                    cache / "target_cache_manifest.json"
                )
                specification = load_hashed_json(
                    cache / "target_cache_specification.json"
                )
                if (
                    manifest["target_cache_specification_sha256"]
                    != specification["content_hash"]
                    or manifest["split"] != "model_train"
                    or manifest["pipeline_seed"] != seed
                    or specification[
                        "locked_coordinate_contract_sha256"
                    ]
                    != system["coordinate_contract_sha256"]
                    or not manifest["complete_coverage"]
                ):
                    raise ValueError(
                        "normalizer source target-cache coverage differs"
                    )
                output = (
                    root
                    / "inputs"
                    / "target_normalizers"
                    / f"coordinate_{coordinate_index:03d}"
                    / shape
                    / f"seed_{seed}"
                )
                outputs = [
                    output / f"target_normalizer_{expert}.json"
                    for expert in EXPERT_ORDER
                ]
                outputs.append(output / "target_normalizer_set.json")
                rows.append(
                    _row(
                        target=target,
                        index=len(rows),
                        argv=(
                            "python",
                            "scripts/fit_retb_target_normalizers.py",
                            "--campaign-root",
                            str(root),
                            "--target-cache-manifest",
                            str(cache / "target_cache_manifest.json"),
                            "--pipeline-seed",
                            str(seed),
                            "--specification-sha256",
                            specification["content_hash"],
                            "--output-dir",
                            str(output),
                        ),
                        outputs=outputs,
                        campaign_sha256=campaign_sha,
                        graph_sha256=graph_sha,
                        producer_completion_sha256=completion_sha,
                        extra_input_hashes={
                            "target_cache_manifest": manifest[
                                "content_hash"
                            ],
                            "target_cache_specification": specification[
                                "content_hash"
                            ],
                            "coordinate_contract": system[
                                "coordinate_contract_sha256"
                            ],
                        },
                        environment={
                            "RETB_COORDINATE_INDEX": str(
                                coordinate_index
                            ),
                            "RETB_SHAPE_ID": shape,
                            "RETB_PIPELINE_SEED": str(seed),
                            "RETB_NORMALIZER_FIT_SPLIT": "model_train",
                        },
                    )
                )
    expected = (
        len(selection["locked_coordinate_systems"]) * len(shapes) * 3
    )
    if len(rows) != expected:
        raise RuntimeError("target-normalizer row coverage differs")
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id=target,
        rows=rows,
    )


def build_predictor_training_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "step9_predictor_contracts",
) -> dict[str, Any]:
    """Launch the complete ordered F--H campaign as one public task."""

    target = "predictor_training"
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    step9 = load_hashed_json(
        root / "registry" / "retb_step9_predictor_bundle.json"
    )
    output = (
        root
        / "job_ledgers"
        / "internal_phases"
        / "predictor_campaign"
        / "controller_completion.json"
    )
    factory_input = (
        root
        / "job_ledgers"
        / "factory_inputs"
        / "uncertainty_calibration.json"
    )
    row = _row(
        target=target,
        index=0,
        argv=(
            "python",
            "scripts/execute_retb_predictor_campaign.py",
            "--campaign-root",
            str(root),
            "--output",
            str(output),
        ),
        outputs=(output, factory_input),
        campaign_sha256=campaign_sha,
        graph_sha256=graph_sha,
        producer_completion_sha256=completion_sha,
        extra_input_hashes={"step9_bundle": step9["content_hash"]},
        environment={
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
            "RETB_PUBLIC_CONTROLLER_ID": "predictor_campaign",
        },
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id=target,
        rows=(row,),
    )


def build_predictor_bundle_selector_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "step10_predictor_bundle_contracts",
) -> dict[str, Any]:
    target = "predictor_bundle_selector"
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    step10 = load_hashed_json(
        root / "registry" / "retb_step10_joint_predictor_bundle.json"
    )
    output = root / "selection" / "predictor_bundle"
    outputs = (
        output / "bundle_input_index.json",
        output / "predictor_cache_index.json",
        output / "bundle_search_policy.json",
        output / "bundle_search.json",
        output / "predictor_bundle_lock.json",
        output / "carried_predictor_bundle_index.json",
        *(
            output / "carried" / f"{role}.json"
            for role in (
                "SHAPE_COMPACT",
                "SHAPE_HIGH",
                "HET_PHYSICS",
                "HET_SELECTED",
                "HET_BEAM",
            )
        ),
    )
    row = _row(
        target=target,
        index=0,
        argv=(
            "python",
            "scripts/execute_retb_predictor_bundle_selection.py",
            "--campaign-root",
            str(root),
            "--output-dir",
            str(output),
        ),
        outputs=outputs,
        campaign_sha256=campaign_sha,
        graph_sha256=graph_sha,
        producer_completion_sha256=completion_sha,
        extra_input_hashes={"step10_bundle": step10["content_hash"]},
        environment={
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
        },
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id=target,
        rows=(row,),
    )


def build_oracle_substitutions_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "predictor_bundle_selector",
) -> dict[str, Any]:
    target = "oracle_substitutions"
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    lock = load_hashed_json(
        root / "selection" / "predictor_bundle" / "predictor_bundle_lock.json"
    )
    output = root / "selection" / "stage_i"
    outputs = [output / "stage_i_index.json"]
    for seed in (101, 202, 303):
        seed_root = output / f"seed_{seed}"
        outputs.extend(
            (
                seed_root / "identity_projected_hlt.json",
                seed_root / "no_reconstruction_predictions.npz",
                seed_root / "no_reconstruction.json",
                seed_root / "stage_i_inputs.npz",
                seed_root / "configuration.json",
                seed_root / "evaluation.json",
            )
        )
    row = _row(
        target=target,
        index=0,
        argv=(
            "python",
            "scripts/execute_retb_stage_i_oracle_wave.py",
            "--campaign-root",
            str(root),
            "--bundle-lock",
            str(
                root
                / "selection"
                / "predictor_bundle"
                / "predictor_bundle_lock.json"
            ),
            "--output-dir",
            str(output),
        ),
        outputs=outputs,
        campaign_sha256=campaign_sha,
        graph_sha256=graph_sha,
        producer_completion_sha256=completion_sha,
        extra_input_hashes={"predictor_bundle_lock": lock["content_hash"]},
        environment={
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
        },
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id=target,
        rows=(row,),
    )


def build_joint_predictor_training_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "step11_joint_bridge_contracts",
) -> dict[str, Any]:
    target = "joint_predictor_training"
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    step11 = load_hashed_json(
        root / "registry" / "retb_step11_joint_bridge_bundle.json"
    )
    output = (
        root
        / "job_ledgers"
        / "internal_phases"
        / "joint_predictor_campaign"
        / "controller_completion.json"
    )
    j4_selections = tuple(
        root / "selection" / "joint" / role / "j4_blocks.json"
        for role in (
            "SHAPE_COMPACT",
            "SHAPE_HIGH",
            "HET_PHYSICS",
            "HET_SELECTED",
            "HET_BEAM",
        )
    )
    row = _row(
        target=target,
        index=0,
        argv=(
            "python",
            "scripts/execute_retb_joint_campaign.py",
            "--campaign-root",
            str(root),
            "--output",
            str(output),
        ),
        outputs=(output, *j4_selections),
        campaign_sha256=campaign_sha,
        graph_sha256=graph_sha,
        producer_completion_sha256=completion_sha,
        extra_input_hashes={"step11_bundle": step11["content_hash"]},
        environment={
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
            "RETB_PUBLIC_CONTROLLER_ID": "joint_predictor_campaign",
        },
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id=target,
        rows=(row,),
    )


def build_joint_predictor_selector_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "joint_predictor_training",
) -> dict[str, Any]:
    target = "joint_predictor_selector"
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    output = root / "selection" / "joint" / "joint_campaign_lock.json"
    row = _row(
        target=target,
        index=0,
        argv=(
            "python",
            "scripts/finalize_retb_joint_campaign.py",
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
            f"j4_block_selection_{index}": load_hashed_json(path)[
                "content_hash"
            ]
            for index, path in enumerate(
                (
                    root / "selection" / "joint" / role / "j4_blocks.json"
                    for role in (
                        "SHAPE_COMPACT",
                        "SHAPE_HIGH",
                        "HET_PHYSICS",
                        "HET_SELECTED",
                        "HET_BEAM",
                    )
                )
            )
        },
        environment={
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0"
        },
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id=target,
        rows=(row,),
    )


def build_final_consumer_training_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "step12_final_consumer_contracts",
) -> dict[str, Any]:
    target = "final_consumer_training"
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    step12 = load_hashed_json(
        root / "registry" / "retb_step12_final_consumers_bundle.json"
    )
    joint_lock = load_hashed_json(
        root / "selection" / "joint" / "joint_campaign_lock.json"
    )
    output = (
        root
        / "job_ledgers"
        / "internal_phases"
        / "final_consumer_campaign"
        / "controller_completion.json"
    )
    refiner_locks = tuple(
        root
        / "selection"
        / "final_consumers"
        / role
        / "token_refiner_lock.json"
        for role in (
            "SHAPE_COMPACT",
            "SHAPE_HIGH",
            "HET_PHYSICS",
            "HET_SELECTED",
            "HET_BEAM",
        )
    )
    row = _row(
        target=target,
        index=0,
        argv=(
            "python",
            "scripts/execute_retb_final_consumer_campaign.py",
            "--campaign-root",
            str(root),
            "--output",
            str(output),
        ),
        outputs=(output, *refiner_locks),
        campaign_sha256=campaign_sha,
        graph_sha256=graph_sha,
        producer_completion_sha256=completion_sha,
        extra_input_hashes={
            "step12_bundle": step12["content_hash"],
            "joint_campaign_lock": joint_lock["content_hash"],
        },
        environment={
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
            "RETB_PUBLIC_CONTROLLER_ID": "final_consumer_campaign",
        },
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id=target,
        rows=(row,),
    )


def build_deployable_export_manifest_plan(
    *,
    campaign_root: str | Path,
    campaign: Mapping[str, Any],
    production_graph: Mapping[str, Any],
    producer_node_id: str = "final_consumer_training",
) -> dict[str, Any]:
    target = "deployable_export"
    root, campaign_sha, graph_sha, completion_sha = _context(
        campaign_root=campaign_root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
    )
    final_completion = load_hashed_json(
        root
        / "job_ledgers"
        / "internal_phases"
        / "final_consumer_campaign"
        / "controller_completion.json"
    )
    output = (
        root
        / "job_ledgers"
        / "internal_phases"
        / "deployable_export_campaign"
        / "controller_completion.json"
    )
    export_index = root / "selection" / "deployable_export_index.json"
    row = _row(
        target=target,
        index=0,
        argv=(
            "python",
            "scripts/execute_retb_deployable_export_campaign.py",
            "--campaign-root",
            str(root),
            "--output",
            str(output),
        ),
        outputs=(
            output,
            export_index,
            root
            / "job_ledgers"
            / "factory_inputs"
            / "robustness_controls.json",
            root
            / "job_ledgers"
            / "factory_inputs"
            / "semantic_controls.json",
        ),
        campaign_sha256=campaign_sha,
        graph_sha256=graph_sha,
        producer_completion_sha256=completion_sha,
        extra_input_hashes={
            "final_consumer_controller": final_completion["content_hash"]
        },
        environment={
            "RETB_SCIENTIFIC_UNDERPERFORMANCE_BLOCKS_CONTINUATION": "0",
            "RETB_PUBLIC_CONTROLLER_ID": "deployable_export_campaign",
        },
    )
    return _publish(
        campaign_root=root,
        campaign=campaign,
        production_graph=production_graph,
        producer_node_id=producer_node_id,
        target_node_id=target,
        rows=(row,),
    )


MIDDLE_PLAN_FACTORIES: dict[str, Callable[..., dict[str, Any]]] = {
    "target_cache_build": build_target_cache_build_manifest_plan,
    "target_normalizers": build_target_normalizers_manifest_plan,
    "predictor_training": build_predictor_training_manifest_plan,
    "predictor_bundle_selector": (
        build_predictor_bundle_selector_manifest_plan
    ),
    "oracle_substitutions": build_oracle_substitutions_manifest_plan,
    "joint_predictor_training": build_joint_predictor_training_manifest_plan,
    "joint_predictor_selector": build_joint_predictor_selector_manifest_plan,
    "final_consumer_training": build_final_consumer_training_manifest_plan,
    "deployable_export": build_deployable_export_manifest_plan,
}


__all__ = [
    "MIDDLE_PLAN_FACTORIES",
    "MIDDLE_PLAN_FACTORY_CONTRACT",
    "MIDDLE_PLAN_FACTORY_TARGETS",
    "build_target_cache_build_manifest_plan",
    "build_target_normalizers_manifest_plan",
    "build_predictor_training_manifest_plan",
    "build_predictor_bundle_selector_manifest_plan",
    "build_oracle_substitutions_manifest_plan",
    "build_joint_predictor_training_manifest_plan",
    "build_joint_predictor_selector_manifest_plan",
    "build_final_consumer_training_manifest_plan",
    "build_deployable_export_manifest_plan",
]
