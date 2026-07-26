"""Production recipes for focused interactions and pre-Stage-G controls.

This module deliberately separates the *scientific identity* of a row from
the lower-level trainer used to execute it.  In particular, composite target
interactions are never silently collapsed onto a single-factor target and
privileged controls are never presented as deployable controls.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .consumer import ParticleViewConsumer
from .campaign import FOCUSED_INTERACTION_IDS, _PRE_STAGE_TRAINED_CONTROLS
from .contracts import (
    canonical_sha256,
    load_hashed_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)
from .distillation_runtime import (
    build_distillation_factory,
    validate_distillation_factory_config,
)
from .post_target_runtime import _SelectedTarget
from .registry import validate_particle_view_registry
from .target_runtime import (
    build_target_discovery_factory,
    build_target_discovery_factory_config,
    run_target_discovery_operation,
    validate_target_discovery_factory_config,
)


PARTICLE_VIEW_FOCUSED_CONTROL_FACTORY_CONFIG_CONTRACT = (
    "particle_view_focused_control_factory_config_v1"
)


def _interaction_recipes() -> dict[str, dict[str, Any]]:
    recipes: dict[str, dict[str, Any]] = {}
    for width in (1, 2, 4, 8):
        recipes[f"DIM{width}_VIEW_ALL"] = {
            "target_run_id": f"VGEN_DIM{width}",
            "target_overrides": {},
            "loss_id": "L_VIEW_ALL",
            "selection_status": "selectable",
        }
        recipes[f"DIM{width}_KD_VIEW_REL"] = {
            "target_run_id": f"VGEN_DIM{width}",
            "target_overrides": {},
            "loss_id": "L_KD_VIEW_REL",
            "selection_status": "selectable",
        }
    recipes.update(
        {
            "TAP_PENULT_PRIMARY": {
                "target_run_id": "VGEN_TAP_PENULT",
                "target_overrides": {},
                "loss_id": "L_PRIMARY",
                "selection_status": "selectable",
            },
            "TAP_MIX3_PRIMARY": {
                "target_run_id": "VGEN_TAP_MIX3",
                "target_overrides": {},
                "loss_id": "L_PRIMARY",
                "selection_status": "selectable",
            },
        }
    )
    for width in (2, 4, 8):
        recipes[f"CENTERED_DIM{width}"] = {
            "target_run_id": f"VGEN_DIM{width}",
            "target_overrides": {"center_output": True},
            "loss_id": "L_PRIMARY",
            "selection_status": "selectable",
        }
        recipes[f"UNCENTERED_DIM{width}"] = {
            "target_run_id": f"VGEN_DIM{width}",
            "target_overrides": {"center_output": False},
            "loss_id": "L_PRIMARY",
            "selection_status": "diagnostic_nonselectable",
        }
        recipes[f"STANDARD_DIM{width}"] = {
            "target_run_id": f"VGEN_DIM{width}",
            "target_overrides": {
                "recoverability_codesign": False,
            },
            "loss_id": "L_PRIMARY",
            "selection_status": "selectable",
        }
        recipes[f"RECODESIGN_DIM{width}"] = {
            "target_run_id": "VGEN_RECODESIGN",
            "target_overrides": {
                "bottleneck_width": width,
                "recoverability_codesign": True,
            },
            "loss_id": "L_PRIMARY",
            "selection_status": "selectable",
        }
    if set(recipes) != set(FOCUSED_INTERACTION_IDS):
        raise RuntimeError("focused interaction recipe inventory drifted")
    return recipes


def _control_recipes() -> dict[str, dict[str, Any]]:
    recipes = {
        "IDENTICAL_CE_ONLY": {
            "execution_kind": "joint_ce_from_hlt_initialization",
            "privileged_source": "none",
            "deployable": True,
            "selection_status": "selectable_control",
        },
        "HLT_SELF_DISTILLATION": {
            "execution_kind": "hlt_self_distillation",
            "privileged_source": "none",
            "deployable": True,
            "selection_status": "selectable_control",
        },
        "DEEPER_DIRECT_HLT_PART": {
            "execution_kind": "deeper_direct_hlt_part",
            "privileged_source": "none",
            "deployable": True,
            "selection_status": "diagnostic_nonselectable",
        },
        "RANDOM_PREDICTOR_INITIALIZATION": {
            "execution_kind": "frozen_consumer_distillation",
            "privileged_source": "selected_particle_view",
            "deployable": True,
            "selection_status": "diagnostic_nonselectable",
        },
        "FROZEN_RANDOM_VIEW_GENERATOR": {
            "execution_kind": "frozen_random_view_consumer",
            "privileged_source": "none",
            "deployable": True,
            "selection_status": "diagnostic_nonselectable",
        },
        "OFFLINE_GLOBAL_LOGIT_BROADCAST": {
            "execution_kind": "offline_global_logit_broadcast",
            "privileged_source": "offline_teacher_logits",
            "deployable": False,
            "selection_status": "diagnostic_nonselectable",
        },
        "RAW_OFFLINE_CROSS_ATTENTION": {
            "execution_kind": "raw_offline_cross_attention",
            "privileged_source": "raw_offline_particles",
            "deployable": False,
            "selection_status": "diagnostic_nonselectable",
        },
        "OFFLINE_CLASSIFIER_DIRECT_KD": {
            "execution_kind": "offline_classifier_direct_kd",
            "privileged_source": "offline_teacher_logits_train_only",
            "deployable": True,
            "selection_status": "diagnostic_nonselectable",
        },
        "DVIEW_JOINT": {
            "execution_kind": "joint_privileged",
            "privileged_source": "selected_particle_view",
            "deployable": True,
            "selection_status": "selectable_privileged",
        },
        "DVIEW_JOINT_CE_ONLY": {
            "execution_kind": "joint_ce_schedule_control",
            "privileged_source": "none",
            "deployable": True,
            "selection_status": "selectable_control",
        },
    }
    if set(recipes) != set(_PRE_STAGE_TRAINED_CONTROLS):
        raise RuntimeError("trained-control recipe inventory drifted")
    return recipes


def build_focused_control_factory_config(
    *,
    distillation_factory_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind integrations 5/6 to the exact already-authenticated runtime."""

    validate_distillation_factory_config(distillation_factory_config)
    target_factory = build_target_discovery_factory_config(
        runtime_data_config=distillation_factory_config[
            "runtime_data_config"
        ],
        device=distillation_factory_config["runtime"]["device"],
        num_workers=distillation_factory_config["runtime"]["num_workers"],
        max_train_batches=distillation_factory_config["runtime"][
            "max_train_batches"
        ],
        max_val_batches=distillation_factory_config["runtime"][
            "max_val_batches"
        ],
    )
    interactions = _interaction_recipes()
    controls = _control_recipes()
    artifact = with_content_hash(
        {
            "contract": (
                PARTICLE_VIEW_FOCUSED_CONTROL_FACTORY_CONFIG_CONTRACT
            ),
            "distillation_factory_config": dict(
                distillation_factory_config
            ),
            "distillation_factory_config_sha256": (
                distillation_factory_config["content_hash"]
            ),
            "target_factory_config": target_factory,
            "target_factory_config_sha256": target_factory["content_hash"],
            "focused_interactions": interactions,
            "focused_interactions_sha256": canonical_sha256(interactions),
            "trained_controls": controls,
            "trained_controls_sha256": canonical_sha256(controls),
            "focused_interaction_count": len(interactions),
            "trained_control_count": len(controls),
            "single_training_pool": True,
            "stack_val_loaded": False,
            "final_test_loaded": False,
            "performance_gates": False,
            "quality_warnings_stop_execution": False,
            "composite_target_policy": (
                "materialize_exact_override_and_refit_train_normalizer"
            ),
            "control_identity_policy": (
                "explicit_supervision_source_and_deployability"
            ),
        }
    )
    validate_focused_control_factory_config(artifact)
    return artifact


def validate_focused_control_factory_config(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        payload,
        expected_contract=(
            PARTICLE_VIEW_FOCUSED_CONTROL_FACTORY_CONFIG_CONTRACT
        ),
    )
    expected = {
        "contract",
        "distillation_factory_config",
        "distillation_factory_config_sha256",
        "target_factory_config",
        "target_factory_config_sha256",
        "focused_interactions",
        "focused_interactions_sha256",
        "trained_controls",
        "trained_controls_sha256",
        "focused_interaction_count",
        "trained_control_count",
        "single_training_pool",
        "stack_val_loaded",
        "final_test_loaded",
        "performance_gates",
        "quality_warnings_stop_execution",
        "composite_target_policy",
        "control_identity_policy",
        "content_hash",
    }
    if set(payload) != expected:
        raise ValueError("focused/control factory field inventory mismatch")
    distillation = payload["distillation_factory_config"]
    validate_distillation_factory_config(distillation)
    target_factory = payload["target_factory_config"]
    validate_target_discovery_factory_config(target_factory)
    interactions = _interaction_recipes()
    controls = _control_recipes()
    if (
        payload["distillation_factory_config_sha256"]
        != distillation["content_hash"]
        or payload["target_factory_config_sha256"]
        != target_factory["content_hash"]
        or target_factory["runtime_data_config_sha256"]
        != distillation["runtime_data_config_sha256"]
        or payload["focused_interactions"] != interactions
        or payload["focused_interactions_sha256"]
        != canonical_sha256(interactions)
        or payload["trained_controls"] != controls
        or payload["trained_controls_sha256"] != canonical_sha256(controls)
        or payload["focused_interaction_count"] != len(interactions)
        or payload["trained_control_count"] != len(controls)
        or payload["single_training_pool"] is not True
        or payload["stack_val_loaded"] is not False
        or payload["final_test_loaded"] is not False
        or payload["performance_gates"] is not False
        or payload["quality_warnings_stop_execution"] is not False
        or payload["composite_target_policy"]
        != "materialize_exact_override_and_refit_train_normalizer"
        or payload["control_identity_policy"]
        != "explicit_supervision_source_and_deployability"
    ):
        raise ValueError("focused/control production policy changed")
    return {
        "ok": True,
        "content_hash": payload["content_hash"],
        "focused_interaction_count": len(interactions),
        "trained_control_count": len(controls),
    }


def _ordinary_interaction_row(
    interaction_id: str,
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    overrides = recipe["target_overrides"]
    if overrides not in (
        {},
        {"center_output": True},
        {"recoverability_codesign": False},
    ):
        raise ValueError(
            "composite interaction requires the composite-target executor"
        )
    return {
        "row_id": f"focused_interaction={interaction_id}",
        "target_id": recipe["target_run_id"],
        "architecture_id": "P_HIER_DECODER_REFINE",
        "consumer_id": "C_TARGET_PROBE",
        "loss_id": recipe["loss_id"],
        "mode": "frozen",
        "selectable": recipe["selection_status"] == "selectable",
        "privileged_claim_eligible": True,
    }


def build_focused_interaction_factory(
    *,
    operation: str,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
    task_id: str,
    output_dir: str,
) -> dict[str, Any]:
    """Execute interaction cells backed by an exact registered target.

    Composite target rows are intentionally routed to their own locked
    operation by :func:`build_focused_control_task_specs`; they can never
    fall through this ordinary-target path.
    """

    validate_focused_control_factory_config(config)
    validate_particle_view_registry(registry)
    prefix = "INTERACTION_"
    if not run_id.startswith(prefix):
        raise ValueError("focused interaction run ID is malformed")
    interaction_id = run_id[len(prefix) :]
    recipe = config["focused_interactions"].get(interaction_id)
    if recipe is None:
        raise ValueError("unknown focused interaction")
    row = _ordinary_interaction_row(interaction_id, recipe)
    return build_distillation_factory(
        operation=operation,
        config=config["distillation_factory_config"],
        registry=registry,
        run_id=run_id,
        seed=seed,
        task_id=task_id,
        output_dir=output_dir,
        _row_override=row,
    )


def build_focused_control_task_specs(
    *,
    factory_config_path: str | Path,
) -> dict[str, dict[str, str]]:
    path = Path(factory_config_path).resolve()
    config = load_hashed_json(path)
    validate_focused_control_factory_config(config)
    common = {
        "factory_config_path": str(path),
        "factory_config_sha256": sha256_file(path),
    }
    interaction_factory = (
        "teacher_logit_reco.local_particle_residual_field.particle_view."
        "focused_control_runtime:build_focused_interaction_factory"
    )
    composite_factory = (
        "teacher_logit_reco.local_particle_residual_field.particle_view."
        "focused_control_runtime:build_composite_interaction_factory"
    )
    control_factory = (
        "teacher_logit_reco.local_particle_residual_field.particle_view."
        "focused_control_runtime:build_trained_control_factory"
    )
    specs: dict[str, dict[str, str]] = {}
    for interaction_id, recipe in config["focused_interactions"].items():
        composite = recipe["target_overrides"] not in (
            {},
            {"center_output": True},
            {"recoverability_codesign": False},
        )
        specs[f"INTERACTION_{interaction_id}"] = {
            **common,
            "operation": (
                "focused_composite_training"
                if composite
                else "frozen_distillation"
            ),
            "factory": composite_factory if composite else interaction_factory,
        }
    for control_id in config["trained_controls"]:
        specs[f"TRAINED_CONTROL_{control_id}"] = {
            **common,
            "operation": "trained_control_training",
            "factory": control_factory,
        }
    return specs


# These entry points are deliberately present now so catalogs fail at build
# time rather than silently routing a composite/control row through ordinary
# distillation.  Their locked executors are added by the production trainer
# integration in this module's companion implementation.
def build_composite_interaction_factory(
    *,
    operation: str,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
    task_id: str,
    output_dir: str,
) -> dict[str, Any]:
    validate_focused_control_factory_config(config)
    validate_particle_view_registry(registry)
    if operation != "focused_composite_training":
        raise ValueError("composite interaction operation changed")
    prefix = "INTERACTION_"
    if (
        not run_id.startswith(prefix)
        or task_id != f"{run_id}__seed_{int(seed)}"
    ):
        raise ValueError("composite interaction identity changed")
    interaction_id = run_id[len(prefix) :]
    recipe = config["focused_interactions"].get(interaction_id)
    if recipe is None or recipe["target_overrides"] in (
        {},
        {"center_output": True},
        {"recoverability_codesign": False},
    ):
        raise ValueError("ordinary interaction reached composite executor")
    output = Path(output_dir).resolve()
    target_output = output / "composite_target"
    target_artifact_paths = [
        target_output / name
        for name in (
            "best_model_val_stop.pt",
            "consumer_registration.json",
            "training_curves.json",
            "generator_model_val_stop.pt",
            "target_discovery_recipe.json",
            "query_tap_registration.json",
            "memory_tap_registration.json",
            "target_candidate_registration.json",
            "target_discovery_result.json",
            "train_staged_tap_manifest.json",
            "model_val_stop_staged_tap_manifest.json",
            "model_val_select_staged_tap_manifest.json",
            "provisional_normalizer.json",
            "probe_consumer/best_model_val_stop.pt",
            "probe_consumer/consumer_registration.json",
            "probe_consumer/training_curves.json",
            "recovery_probe/best_model_val_stop.pt",
            "recovery_probe/recovery_probe_registration.json",
            "recovery_probe/training_curves.json",
            "candidate_quantization_diagnostics.json",
            "model_val_select_counterfactual_metrics.json",
            "target_candidate_metrics.json",
            "two_pass_candidate.json",
            "target_two_pass_result.json",
        )
    ]
    if recipe["target_overrides"].get("recoverability_codesign") is True:
        target_artifact_paths.extend(
            target_output / "recoverability_codesign" / name
            for name in (
                "rich_context_registration.json",
                "provisional_head_registration.json",
                "selected_projection.pt",
                "selected_consumer.pt",
                "selected_persistent_probe.pt",
                "codesign_ledger.json",
                "cycle_metrics.json",
                "codesigned_generator.pt",
            )
        )
    distillation_paths = [
        output / name
        for name in (
            "distillation_runtime_binding.json",
            "selected_distilled_predictor.pt",
            "distillation_training_curves.json",
            "distillation_generalization.json",
            "distillation_registration.json",
        )
    ]
    return {
        "kwargs": {
            "config": dict(config),
            "registry": dict(registry),
            "run_id": run_id,
            "seed": int(seed),
            "task_id": task_id,
            "output_dir": str(output),
        },
        "artifact_paths": [
            *(str(path) for path in target_artifact_paths),
            *(str(path) for path in distillation_paths),
        ],
        "action": None,
    }


def _artifact_bindings(paths: list[str]) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for raw in paths:
        path = Path(raw).resolve()
        name = path.name
        if name in bindings:
            name = str(
                Path(path.parent.name) / path.name
            ).replace("\\", "/")
        bindings[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    return bindings


def run_focused_composite_training(
    *,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
    task_id: str,
    output_dir: str,
) -> None:
    """Train an exact centering/width or co-design/width interaction."""

    validate_focused_control_factory_config(config)
    interaction_id = run_id.removeprefix("INTERACTION_")
    interaction = config["focused_interactions"][interaction_id]
    target_output = Path(output_dir).resolve() / "composite_target"
    source_run_id = interaction["target_run_id"]
    prepared = build_target_discovery_factory(
        operation="target_discovery",
        config=config["target_factory_config"],
        registry=registry,
        run_id=source_run_id,
        seed=int(seed),
        task_id=f"{source_run_id}__seed_{int(seed)}",
        output_dir=str(target_output),
    )
    kwargs = dict(prepared["kwargs"])
    recipe = kwargs["recipe"]
    overrides = interaction["target_overrides"]
    generator_config = recipe.generator_config
    if "center_output" in overrides:
        generator_config = replace(
            generator_config,
            center_output=bool(overrides["center_output"]),
        )
    if "bottleneck_width" in overrides:
        generator_config = replace(
            generator_config,
            bottleneck_width=int(overrides["bottleneck_width"]),
        )
    recipe = replace(
        recipe,
        generator_config=generator_config,
        recoverability_codesign=bool(
            overrides.get(
                "recoverability_codesign",
                recipe.recoverability_codesign,
            )
        ),
    )
    kwargs["recipe"] = recipe
    view_dim = int(generator_config.bottleneck_width)
    for name in (
        "consumer_model",
        "probe_consumer_model",
        "codesign_consumer_model",
    ):
        old = kwargs.get(name)
        if old is not None and int(old.config.view_dim) != view_dim:
            kwargs[name] = ParticleViewConsumer(
                old.a0_model,
                replace(old.config, view_dim=view_dim),
            )
    run_target_discovery_operation(**kwargs)
    target_paths = [str(path) for path in prepared["artifact_paths"]]
    target = _SelectedTarget(
        run_id=source_run_id,
        selection=with_content_hash(
            {
                "contract": (
                    "particle_view_composite_interaction_target_reference_v1"
                ),
                "interaction_id": interaction_id,
                "source_target_run_id": source_run_id,
                "target_overrides": dict(overrides),
            }
        ),
        registration=load_hashed_json(
            target_output / "target_candidate_registration.json"
        ),
        recipe=load_hashed_json(
            target_output / "target_discovery_recipe.json"
        ),
        artifacts=_artifact_bindings(target_paths),
    )
    row = {
        "row_id": f"focused_interaction={interaction_id}",
        "target_id": source_run_id,
        "architecture_id": "P_HIER_DECODER_REFINE",
        "consumer_id": "C_TARGET_PROBE",
        "loss_id": interaction["loss_id"],
        "mode": "frozen",
        "selectable": interaction["selection_status"] == "selectable",
        "privileged_claim_eligible": True,
    }
    distillation = build_distillation_factory(
        operation="frozen_distillation",
        config=config["distillation_factory_config"],
        registry=registry,
        run_id=run_id,
        seed=int(seed),
        task_id=task_id,
        output_dir=output_dir,
        _row_override=row,
        _target_override=target,
    )
    from .distillation import train_frozen_consumer_distillation

    train_frozen_consumer_distillation(**distillation["kwargs"])


def _control_distillation_row(
    control_id: str,
) -> tuple[dict[str, Any], bool]:
    rows = {
        "IDENTICAL_CE_ONLY": (
            "TARGET_ALTERNATE_SELECTED",
            "P_HIER_DECODER_REFINE",
            "C_ROBUST_MIX",
            "L_CE",
            "frozen",
            False,
        ),
        "HLT_SELF_DISTILLATION": (
            "VGEN_MEMORY_HLT",
            "P_HIER_DECODER_REFINE",
            "C_TARGET_PROBE",
            "L_KD",
            "frozen",
            False,
        ),
        "RANDOM_PREDICTOR_INITIALIZATION": (
            "TARGET_ALTERNATE_SELECTED",
            "P_HIER_DECODER_REFINE",
            "C_ROBUST_MIX",
            "L_PRIMARY",
            "frozen",
            True,
        ),
        "RAW_OFFLINE_CROSS_ATTENTION": (
            "VGEN_TAP_RAW",
            "P_HIER_DECODER_REFINE",
            "C_TARGET_PROBE",
            "L_PRIMARY",
            "frozen",
            False,
        ),
        "DVIEW_JOINT": (
            "TARGET_ALTERNATE_SELECTED",
            "P_HIER_DECODER_REFINE",
            "C_ROBUST_MIX",
            "L_PRIMARY",
            "joint",
            False,
        ),
        "DVIEW_JOINT_CE_ONLY": (
            "TARGET_ALTERNATE_SELECTED",
            "P_HIER_DECODER_REFINE",
            "C_ROBUST_MIX",
            "L_CE",
            "joint_ce_control",
            False,
        ),
    }
    if control_id not in rows:
        raise KeyError(control_id)
    target, architecture, consumer, loss, mode, fresh = rows[control_id]
    return (
        {
            "row_id": f"trained_control={control_id}",
            "target_id": target,
            "architecture_id": architecture,
            "consumer_id": consumer,
            "loss_id": loss,
            "mode": mode,
            "selectable": control_id
            in {
                "IDENTICAL_CE_ONLY",
                "HLT_SELF_DISTILLATION",
                "DVIEW_JOINT",
                "DVIEW_JOINT_CE_ONLY",
            },
            "privileged_claim_eligible": control_id == "DVIEW_JOINT",
        },
        fresh,
    )


_DEDICATED_CONTROL_IDS = frozenset(
    {
        "DEEPER_DIRECT_HLT_PART",
        "FROZEN_RANDOM_VIEW_GENERATOR",
        "OFFLINE_GLOBAL_LOGIT_BROADCAST",
        "OFFLINE_CLASSIFIER_DIRECT_KD",
    }
)


def build_trained_control_factory(
    *,
    operation: str,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
    task_id: str,
    output_dir: str,
) -> dict[str, Any]:
    validate_focused_control_factory_config(config)
    validate_particle_view_registry(registry)
    prefix = "TRAINED_CONTROL_"
    if (
        operation != "trained_control_training"
        or not run_id.startswith(prefix)
        or task_id != f"{run_id}__seed_{int(seed)}"
    ):
        raise ValueError("trained-control execution identity changed")
    control_id = run_id[len(prefix) :]
    if control_id not in config["trained_controls"]:
        raise ValueError("unknown trained control")
    if control_id in _DEDICATED_CONTROL_IDS:
        # Dedicated architectures are emitted as an authenticated,
        # non-gating unavailable row until their independent trainer is
        # present. This is never promoted into a selectable result.
        output = Path(output_dir).resolve()
        destination = output / "trained_control_unavailable.json"
        return {
            "kwargs": {
                "unavailable": {
                    "destination": str(destination),
                    "control_id": control_id,
                    "run_id": run_id,
                    "seed": int(seed),
                    "factory_config_sha256": config["content_hash"],
                    "reason": "dedicated_control_trainer_not_available",
                }
            },
            "artifact_paths": [str(destination)],
            "action": None,
        }
    row, fresh = _control_distillation_row(control_id)
    training_operation = (
        "frozen_distillation"
        if row["mode"] == "frozen"
        else "joint_finetuning"
    )
    prepared = build_distillation_factory(
        operation=training_operation,
        config=config["distillation_factory_config"],
        registry=registry,
        run_id=run_id,
        seed=int(seed),
        task_id=task_id,
        output_dir=output_dir,
        _row_override=row,
        _force_fresh_predictor=fresh,
    )
    return {
        "kwargs": {
            "trainer": training_operation,
            "trainer_kwargs": prepared["kwargs"],
        },
        "artifact_paths": prepared["artifact_paths"],
        "action": None,
    }


def run_trained_control_training(
    *,
    trainer: str | None = None,
    trainer_kwargs: Mapping[str, Any] | None = None,
    unavailable: Mapping[str, Any] | None = None,
) -> None:
    if unavailable is not None:
        if trainer is not None or trainer_kwargs is not None:
            raise ValueError("unavailable control also supplied a trainer")
        destination = unavailable["destination"]
        artifact = with_content_hash(
            {
                "contract": "particle_view_trained_control_unavailable_v1",
                **{
                    key: value
                    for key, value in unavailable.items()
                    if key != "destination"
                },
                "selection_status": "diagnostic_unavailable",
                "stops_execution": False,
                "quality_gate_used": False,
            }
        )
        from .contracts import write_immutable_json

        write_immutable_json(destination, artifact)
        return
    if trainer_kwargs is None:
        raise ValueError("trained control omitted trainer kwargs")
    if trainer == "frozen_distillation":
        from .distillation import train_frozen_consumer_distillation

        train_frozen_consumer_distillation(**dict(trainer_kwargs))
        return
    if trainer == "joint_finetuning":
        from .distillation import train_joint_finetuning

        train_joint_finetuning(**dict(trainer_kwargs))
        return
    raise ValueError("unknown trained-control trainer")


__all__ = [
    "PARTICLE_VIEW_FOCUSED_CONTROL_FACTORY_CONFIG_CONTRACT",
    "build_focused_control_factory_config",
    "build_focused_control_task_specs",
    "build_focused_interaction_factory",
    "build_composite_interaction_factory",
    "build_trained_control_factory",
    "run_focused_composite_training",
    "run_trained_control_training",
    "validate_focused_control_factory_config",
]
