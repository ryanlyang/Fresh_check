"""Production integration after the Stage-B particle-view target screen.

This module turns the selected target into immutable coordinates and caches,
trains the final clean consumer, then runs the Pview_0 -> residual sampler ->
robust consumer chain.  Large contextual taps and residual banks are rebuilt
inside each allocation and remain RAM-resident; only compact registrations,
checkpoints, and the selected true-view caches are persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .campaign import PARTICLE_VIEW_LOW_DATA_CAMPAIGN_ID
from .consumer import ParticleViewConsumer, ParticleViewConsumerConfig
from .consumer_train import ParticleViewConsumerTrainConfig
from .contracts import (
    COORDINATE_PARENT_HASH_FIELDS,
    build_view_coordinate_binding,
    canonical_sha256,
    load_hashed_json,
    sha256_file,
    validate_content_hash,
    validate_view_coordinate_binding,
    with_content_hash,
    write_immutable_json,
)
from .offline_teacher import (
    FrozenContextualParticleTeacher,
    FrozenTokenLayerMixture,
    ParticleTokenTapSpec,
    reload_registered_teacher,
    validate_teacher_registration,
)
from .oracle_discovery import (
    RecoverabilityCoDesignConfig,
    RecoverabilityCoDesignProjection,
)
from .predictor import (
    HierarchicalParticleViewPredictor,
    build_predictor_architecture_config,
)
from .registry import validate_particle_view_registry
from .robust_consumer import (
    CorrelatedResidualSampler,
    SnapshotDropoutPredictionBank,
    fit_correlated_residual_sampler,
    publish_correlated_residual_sampler,
)
from .runtime import validate_runtime_task_result
from .runtime_data import (
    PARTICLE_VIEW_RUNTIME_DATA_CONFIG_CONTRACT,
    AlignedLogicalJetView,
    load_aligned_logical_jet_view,
    make_logical_data_loader,
    validate_runtime_data_config,
)
from .splits import logical_split_binding
from .target_generator import (
    MatchingFreeParticleViewGenerator,
    particle_view_generator_config_from_payload,
)
from .target_provenance import validate_target_candidate_registration
from .target_runtime import (
    CandidateViewSet,
    IndexedCandidateViewProvider,
    PARTICLE_VIEW_GENERATOR_CHECKPOINT_CONTRACT,
    StagedDiscoveryViewProvider,
    TwoTeacherTokenMixture,
    materialize_candidate_views_in_ram,
    stage_contextual_memory,
)
from .train_predictor import (
    PVIEW0_LINEAGE_FIELDS,
    ParticleViewWarmupConfig,
    collect_pview_predictions,
    validate_pview0_registration,
)
from .view_cache import (
    PARTICLE_VIEW_FINAL_COORDINATE_CONTRACT,
    ParticleViewNormalizer,
    audit_live_selected_view_equivalence,
    fit_particle_view_normalizer,
    load_particle_view_normalizer,
    load_selected_view_cache,
    normalize_particle_view,
    publish_selected_view_cache,
    write_particle_view_normalizer,
)


PARTICLE_VIEW_POST_TARGET_FACTORY_CONFIG_CONTRACT = (
    "particle_view_post_target_factory_config_v1"
)
PARTICLE_VIEW_SELECTED_CACHE_INDEX_CONTRACT = (
    "particle_view_selected_cache_index_v1"
)
PARTICLE_VIEW_SELECTED_COORDINATE_RESULT_CONTRACT = (
    "particle_view_selected_coordinate_result_v1"
)

POST_TARGET_RUN_IDS = (
    "SELECTED_COORDINATE_BINDING",
    "SELECTED_VIEW_CACHE",
    "FINAL_CLEAN_CONSUMER",
    "PVIEW0",
    "RESIDUAL_SAMPLER",
    "ROBUST_CONSUMER",
)
SELECTED_CACHE_SPLITS = ("train", "model_val_stop", "model_val_select")


@dataclass(frozen=True)
class _SelectedTarget:
    run_id: str
    selection: Mapping[str, Any]
    registration: Mapping[str, Any]
    recipe: Mapping[str, Any]
    artifacts: Mapping[str, Mapping[str, str]]


class SelectedViewLoader:
    """Attach an authenticated selected-view array to an aligned HLT loader."""

    def __init__(
        self,
        loader,
        *,
        aligned: AlignedLogicalJetView,
        views: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        values = torch.from_numpy(
            np.ascontiguousarray(views, dtype=np.float32)
        )
        valid = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.bool_))
        parents = torch.from_numpy(
            np.asarray(aligned.parent_row_indices, dtype=np.int64)
        )
        self.view_set = CandidateViewSet(
            views=values,
            mask=valid,
            parent_indices=parents,
            logical_split_sha256=aligned.logical_split_sha256,
            ordered_identity_sha256=aligned.ordered_identity_sha256,
        )
        self.loader = loader
        self.batch_size = getattr(loader, "batch_size", None)

    def __iter__(self):
        for raw in self.loader:
            batch = dict(raw)
            positions = self.view_set.positions(batch["parent_indices"])
            expected = self.view_set.mask[positions]
            if not torch.equal(expected, batch["mask"][:, 0]):
                raise ValueError("selected-view mask differs from HLT batch")
            batch["true_view"] = self.view_set.views[positions]
            # Parent-row identity is immutable within each logical split and
            # is therefore the lookup key shared by the selected-view and
            # frozen-target-logit caches.  Expose it explicitly so Stage-E/F
            # loaders cannot fall back to batch position or shuffled order.
            batch["event_ids"] = batch["parent_indices"].to(torch.long)
            yield batch

    def __len__(self) -> int:
        return len(self.loader)


def build_post_target_factory_config(
    *,
    runtime_data_config: Mapping[str, Any],
    device: str = "auto",
    num_workers: int = 0,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, Any]:
    validate_runtime_data_config(runtime_data_config, verify_cache_files=True)
    if not isinstance(device, str) or not device:
        raise ValueError("device must be nonempty")
    if int(num_workers) < 0:
        raise ValueError("num_workers must be nonnegative")
    for name, value in (
        ("max_train_batches", max_train_batches),
        ("max_val_batches", max_val_batches),
    ):
        if value is not None and int(value) <= 0:
            raise ValueError(f"{name} must be positive when set")
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_POST_TARGET_FACTORY_CONFIG_CONTRACT,
            "runtime_data_config": dict(runtime_data_config),
            "runtime_data_config_sha256": runtime_data_config["content_hash"],
            "runtime": {
                "device": device,
                "num_workers": int(num_workers),
                "max_train_batches": (
                    None
                    if max_train_batches is None
                    else int(max_train_batches)
                ),
                "max_val_batches": (
                    None if max_val_batches is None else int(max_val_batches)
                ),
            },
            "selected_cache_splits": list(SELECTED_CACHE_SPLITS),
            "final_test_view_cache_forbidden": True,
            "residual_bank_persisted": False,
            "quality_warnings_stop_execution": False,
        }
    )
    validate_post_target_factory_config(artifact)
    return artifact


def validate_post_target_factory_config(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        payload,
        expected_contract=PARTICLE_VIEW_POST_TARGET_FACTORY_CONFIG_CONTRACT,
    )
    if set(payload) != {
        "contract",
        "runtime_data_config",
        "runtime_data_config_sha256",
        "runtime",
        "selected_cache_splits",
        "final_test_view_cache_forbidden",
        "residual_bank_persisted",
        "quality_warnings_stop_execution",
        "content_hash",
    }:
        raise ValueError("post-target factory field inventory mismatch")
    data = payload["runtime_data_config"]
    validate_content_hash(
        data, expected_contract=PARTICLE_VIEW_RUNTIME_DATA_CONFIG_CONTRACT
    )
    if payload["runtime_data_config_sha256"] != data["content_hash"]:
        raise ValueError("post-target runtime-data hash mismatch")
    validate_runtime_data_config(data, verify_cache_files=False)
    runtime = payload["runtime"]
    if set(runtime) != {
        "device",
        "num_workers",
        "max_train_batches",
        "max_val_batches",
    }:
        raise ValueError("post-target runtime field inventory mismatch")
    if (
        not isinstance(runtime["device"], str)
        or not runtime["device"]
        or int(runtime["num_workers"]) < 0
        or any(
            value is not None and int(value) <= 0
            for value in (
                runtime["max_train_batches"],
                runtime["max_val_batches"],
            )
        )
        or payload["selected_cache_splits"] != list(SELECTED_CACHE_SPLITS)
        or payload["final_test_view_cache_forbidden"] is not True
        or payload["residual_bank_persisted"] is not False
        or payload["quality_warnings_stop_execution"] is not False
    ):
        raise ValueError("post-target production policy changed")
    return {"ok": True, "content_hash": payload["content_hash"]}


def _task_artifacts(
    root: Path,
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
) -> dict[str, dict[str, str]]:
    rows = {row["run_id"]: row for row in registry["runs"]}
    if run_id not in rows:
        raise KeyError(f"unknown run {run_id!r}")
    seeds = [int(value) for value in rows[run_id]["seed_ids"]]
    resolved_seed = int(seed) if int(seed) in seeds else 101
    if resolved_seed not in seeds:
        raise ValueError(f"no compatible seed for {run_id}")
    task_id = f"{run_id}__seed_{resolved_seed}"
    result_path = root / "runtime_tasks" / task_id / "task_result.json"
    result = load_hashed_json(result_path)
    validate_runtime_task_result(result, expected_task_id=task_id)
    artifacts: dict[str, dict[str, str]] = {}
    for binding in result["artifacts"]:
        path = Path(binding["path"]).resolve()
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ValueError(f"runtime artifact changed for {task_id}")
        relative = str(path.relative_to(path.parent))
        name = path.name
        if name in artifacts:
            # Nested checkpoints can share a basename.  Their relative
            # task-root path remains unambiguous.
            task_root = result_path.parent
            relative = str(path.relative_to(task_root)).replace("\\", "/")
            name = relative
        artifacts[name] = dict(binding)
    return artifacts


def _artifact(
    artifacts: Mapping[str, Mapping[str, str]], name: str
) -> Path:
    try:
        binding = artifacts[name]
    except KeyError as exc:
        raise ValueError(f"required runtime artifact {name!r} is absent") from exc
    path = Path(binding["path"]).resolve()
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise ValueError(f"runtime artifact {name!r} changed")
    return path


def _selected_target(
    *,
    root: Path,
    registry: Mapping[str, Any],
    seed: int,
) -> _SelectedTarget:
    selection_artifacts = _task_artifacts(
        root, registry, "SELECT_TARGET_DEFINITIONS", seed
    )
    selection = load_hashed_json(
        _artifact(selection_artifacts, "selected_targets.json")
    )
    ranked = [
        row
        for row in selection["ranked"]
        if row["selection_status"] in {"selectable", "canonical_selectable"}
    ]
    if not ranked:
        raise ValueError("target selection has no selectable target")
    selected_row = ranked[0]
    run_id = str(selected_row["run_id"])
    artifacts = _task_artifacts(root, registry, run_id, seed)
    registration = load_hashed_json(
        _artifact(artifacts, "target_candidate_registration.json")
    )
    validate_target_candidate_registration(registration)
    if (
        registration["target_id"] != selected_row["target_id"]
        or registration["content_hash"]
        != selected_row["target_registration_sha256"]
    ):
        raise ValueError("selected target registration/metric lineage mismatch")
    recipe = load_hashed_json(
        _artifact(artifacts, "target_discovery_recipe.json")
    )
    if (
        recipe["run_id"] != run_id
        or recipe["generator_config"] != registration["generator_config"]
    ):
        raise ValueError("selected target recipe/registration mismatch")
    return _SelectedTarget(
        run_id=run_id,
        selection=selection,
        registration=registration,
        recipe=recipe,
        artifacts=artifacts,
    )


def registered_target_for_run(
    *,
    root: Path,
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
) -> _SelectedTarget:
    """Resolve one completed target row without consulting target ranking."""

    artifacts = _task_artifacts(root, registry, run_id, seed)
    registration = load_hashed_json(
        _artifact(artifacts, "target_candidate_registration.json")
    )
    validate_target_candidate_registration(registration)
    recipe = load_hashed_json(
        _artifact(artifacts, "target_discovery_recipe.json")
    )
    if (
        recipe["run_id"] != run_id
        or recipe["generator_config"] != registration["generator_config"]
    ):
        raise ValueError("target recipe/registration mismatch")
    synthetic_selection = with_content_hash(
        {
            "contract": "particle_view_unranked_target_reference_v1",
            "run_id": run_id,
            "target_registration_sha256": registration["content_hash"],
            "used_for_consumer_interface_screen_only": True,
        }
    )
    return _SelectedTarget(
        run_id=run_id,
        selection=synthetic_selection,
        registration=registration,
        recipe=recipe,
        artifacts=artifacts,
    )


def _teacher_from_task(
    root: Path,
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
) -> tuple[dict[str, Any], Path, torch.nn.Module]:
    artifacts = _task_artifacts(root, registry, run_id, seed)
    registration = load_hashed_json(
        _artifact(artifacts, "teacher_registration.json")
    )
    validate_teacher_registration(registration)
    checkpoint = _artifact(artifacts, "best_model_val_stop.pt")
    return (
        registration,
        checkpoint,
        reload_registered_teacher(
            registration=registration, checkpoint_path=checkpoint
        ),
    )


def _reload_selected_generator(
    selected: _SelectedTarget,
) -> tuple[
    MatchingFreeParticleViewGenerator,
    FrozenTokenLayerMixture | None,
    FrozenTokenLayerMixture | None,
    TwoTeacherTokenMixture | None,
    RecoverabilityCoDesignProjection | None,
]:
    expected_generator_sha256 = selected.registration[
        "generator_checkpoint_sha256"
    ]
    matches = [
        Path(binding["path"]).resolve()
        for binding in selected.artifacts.values()
        if (
            Path(binding["path"]).suffix == ".pt"
            and binding["sha256"] == expected_generator_sha256
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            "selected generator checkpoint is absent or ambiguous"
        )
    checkpoint_path = matches[0]
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = particle_view_generator_config_from_payload(
        checkpoint["generator_config"]
    )
    generator = MatchingFreeParticleViewGenerator(config)
    codesign = checkpoint.get("contract") == "particle_view_codesigned_generator_v1"
    state_name = "generator_state_dict" if codesign else "model_state_dict"
    generator.load_state_dict(checkpoint[state_name], strict=True)
    query_mixture = (
        FrozenTokenLayerMixture()
        if selected.recipe["query_tap_choice"] == "mix_last3"
        else None
    )
    memory_mixture = (
        FrozenTokenLayerMixture()
        if (
            selected.recipe["memory_tap_choice"] == "mix_last3"
            and selected.recipe["teacher_source"] != "base_large_mix"
        )
        else None
    )
    teacher_mixture = (
        TwoTeacherTokenMixture()
        if selected.recipe["teacher_source"] == "base_large_mix"
        else None
    )
    adapters = checkpoint.get("source_adapter_state_dicts", {})
    expected = {
        name
        for name, value in (
            ("query_layer_mixture", query_mixture),
            ("memory_layer_mixture", memory_mixture),
            ("teacher_source_mixture", teacher_mixture),
        )
        if value is not None
    }
    if not codesign:
        if set(adapters) != expected:
            raise ValueError("selected generator source-adapter inventory changed")
        for name, module in (
            ("query_layer_mixture", query_mixture),
            ("memory_layer_mixture", memory_mixture),
            ("teacher_source_mixture", teacher_mixture),
        ):
            if module is not None:
                module.load_state_dict(adapters[name], strict=True)
                module.eval()
    elif expected:
        raise ValueError("recoverability co-design unexpectedly used adapters")
    projection = None
    if codesign:
        projection_config = checkpoint["projection_config"]
        config_object = RecoverabilityCoDesignConfig(
            view_dim=int(projection_config["view_dim"]),
            seed=int(projection_config["seed"]),
        )
        if config_object.to_payload() != projection_config:
            raise ValueError("co-design projection config changed")
        projection = RecoverabilityCoDesignProjection(config_object)
        projection.load_state_dict(
            checkpoint["projection_state_dict"], strict=True
        )
        projection.eval()
    generator.eval()
    for module in (
        generator,
        query_mixture,
        memory_mixture,
        teacher_mixture,
        projection,
    ):
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    return (
        generator,
        query_mixture,
        memory_mixture,
        teacher_mixture,
        projection,
    )


def _materialize_selected_split(
    *,
    selected: _SelectedTarget,
    aligned: AlignedLogicalJetView,
    root: Path,
    registry: Mapping[str, Any],
    seed: int,
    device: str,
    num_workers: int,
) -> CandidateViewSet:
    a0_registration, _, a0_model = _teacher_from_task(
        root, registry, "A0_VIEW", seed
    )
    teacher_source = selected.recipe["teacher_source"]
    memory_run = (
        "A0_VIEW"
        if teacher_source == "hlt"
        else "TOFF_VIEW_LARGE"
        if teacher_source == "large"
        else "TOFF_VIEW_BASE"
    )
    memory_registration, _, memory_model = _teacher_from_task(
        root, registry, memory_run, seed
    )
    secondary_registration = secondary_model = None
    if teacher_source == "base_large_mix":
        secondary_registration, _, secondary_model = _teacher_from_task(
            root, registry, "TOFF_VIEW_LARGE", seed
        )
    query_choice = selected.recipe["query_tap_choice"]
    query_spec = ParticleTokenTapSpec(
        particle_source="fixed_hlt",
        architecture=a0_registration["recipe"]["architecture_name"],
        tap_choice=(
            "raw_embed" if query_choice == "raw_features" else query_choice
        ),
    )
    memory_spec = ParticleTokenTapSpec(
        particle_source=(
            "fixed_hlt"
            if selected.registration["memory_source"] == "hlt"
            else "offline"
        ),
        architecture=memory_registration["recipe"]["architecture_name"],
        tap_choice=selected.recipe["memory_tap_choice"],
    )
    query_teacher = FrozenContextualParticleTeacher(a0_model, query_spec)
    memory_teacher = FrozenContextualParticleTeacher(
        memory_model, memory_spec
    )
    staged = stage_contextual_memory(
        aligned=aligned,
        teacher=memory_teacher,
        teacher_checkpoint_sha256=memory_registration["checkpoint_sha256"],
        tap_spec_sha256=memory_spec.content_hash,
        source_manifest_sha256=selected.registration[
            "source_manifest_sha256"
        ],
        source_role=(
            "hlt_memory_control"
            if selected.registration["memory_source"] == "hlt"
            else "offline_teacher"
        ),
        source_view=(
            "fixed_hlt"
            if selected.registration["memory_source"] == "hlt"
            else "offline"
        ),
        device=device,
        num_workers=num_workers,
    )
    secondary_staged = None
    if secondary_registration is not None:
        secondary_spec = ParticleTokenTapSpec(
            particle_source="offline",
            architecture=secondary_registration["recipe"][
                "architecture_name"
            ],
            tap_choice=selected.recipe["memory_tap_choice"],
        )
        secondary_staged = stage_contextual_memory(
            aligned=aligned,
            teacher=FrozenContextualParticleTeacher(
                secondary_model, secondary_spec
            ),
            teacher_checkpoint_sha256=secondary_registration[
                "checkpoint_sha256"
            ],
            tap_spec_sha256=secondary_spec.content_hash,
            source_manifest_sha256=selected.registration[
                "source_manifest_sha256"
            ],
            source_role="offline_teacher_secondary",
            source_view="offline",
            device=device,
            num_workers=num_workers,
        )
    (
        generator,
        query_mixture,
        memory_mixture,
        teacher_mixture,
        projection,
    ) = _reload_selected_generator(selected)
    provider = StagedDiscoveryViewProvider(
        generator=generator,
        query_teacher=query_teacher,
        staged_memory=staged,
        query_tap_choice=query_choice,
        memory_source=selected.registration["memory_source"],
        query_mixture=query_mixture,
        memory_mixture=memory_mixture,
        memory_layer_width=(
            int(memory_registration["recipe"]["architecture"]["embed_dims"][-1])
            if memory_mixture is not None
            else None
        ),
        secondary_staged_memory=secondary_staged,
        teacher_mixture=teacher_mixture,
        codesign_projection=projection,
    )
    return materialize_candidate_views_in_ram(
        aligned=aligned,
        provider=provider,
        device=device,
        num_workers=num_workers,
    )


def _coordinate_ancestry(
    *,
    selected: _SelectedTarget,
    root: Path,
    registry: Mapping[str, Any],
    seed: int,
    normalizer_sha256: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    a0_registration, _, _ = _teacher_from_task(
        root, registry, "A0_VIEW", seed
    )
    memory_run = (
        "TOFF_VIEW_LARGE"
        if selected.recipe["teacher_source"] == "large"
        else "TOFF_VIEW_BASE"
    )
    memory_registration, _, _ = _teacher_from_task(
        root, registry, memory_run, seed
    )
    query_tap = load_hashed_json(
        _artifact(selected.artifacts, "query_tap_registration.json")
    )
    memory_tap = load_hashed_json(
        _artifact(selected.artifacts, "memory_tap_registration.json")
    )
    memory_tap_spec = (
        memory_tap["primary_tap"]["tap_spec"]
        if memory_tap.get("contract")
        == "particle_view_two_teacher_tap_binding_v1"
        else memory_tap["tap_spec"]
    )
    generator_config = selected.registration["generator_config"]
    parents = {
        "source_manifest_sha256": selected.registration[
            "source_manifest_sha256"
        ],
        "unified_split_manifest_sha256": selected.registration[
            "unified_split_manifest_sha256"
        ],
        "train_identity_sha256": selected.registration[
            "train_identity_sha256"
        ],
        "hlt_source_sha256": a0_registration["source_sha256"],
        "offline_source_sha256": selected.registration[
            "offline_source_sha256"
        ],
        "a0_checkpoint_sha256": a0_registration["checkpoint_sha256"],
        "a0_config_sha256": canonical_sha256(
            a0_registration["recipe"]["architecture"]
        ),
        "a0_query_tap_sha256": query_tap["content_hash"],
        "a0_input_normalization_sha256": a0_registration[
            "preprocessing_sha256"
        ],
        "offline_teacher_checkpoint_sha256": memory_registration[
            "checkpoint_sha256"
        ],
        "offline_teacher_config_sha256": canonical_sha256(
            memory_registration["recipe"]["architecture"]
        ),
        "offline_tap_spec_sha256": canonical_sha256(memory_tap_spec),
        "generator_checkpoint_sha256": selected.registration[
            "generator_checkpoint_sha256"
        ],
        "normalizer_sha256": normalizer_sha256,
    }
    if set(parents) != set(COORDINATE_PARENT_HASH_FIELDS):
        raise RuntimeError("coordinate ancestry inventory drifted")
    definition = {
        "offline_tap_layer": str(memory_tap_spec["tap_choice"]),
        "offline_tap_tensor_location": str(
            memory_tap_spec["tensor_location"]
        ),
        "cross_attention_config_sha256": canonical_sha256(
            {
                "width": generator_config["width"],
                "num_heads": generator_config["num_heads"],
                "blocks": generator_config["num_cross_attention_blocks"],
                "feed_forward_expansion": generator_config[
                    "feed_forward_expansion"
                ],
            }
        ),
        "pair_feature_schema_sha256": canonical_sha256(
            {
                "schema": generator_config["pair_feature_schema"],
                "order": generator_config["pair_feature_order"],
                "dim": generator_config["pair_feature_dim"],
                "enabled": generator_config["use_pair_bias"],
            }
        ),
        "centering_policy": (
            "masked_particle_mean_center"
            if generator_config["center_output"]
            else "uncentered_diagnostic"
        ),
        "bounded_coordinate_policy": "tanh_then_train_standardize_clip_6",
        "rate_budget_policy": (
            "oracle_rate_covariance_budget"
            if selected.recipe["oracle_objective"]["rate_budget_enabled"]
            else "rate_budget_disabled_diagnostic"
        ),
        "null_token_policy": (
            "learned_null_token"
            if generator_config["use_null_token"]
            else "no_null_token"
        ),
        "bottleneck_width": int(generator_config["bottleneck_width"]),
    }
    return parents, definition


def publish_selected_coordinate_binding(
    *,
    output_dir: str,
    selected: _SelectedTarget,
    train_views: CandidateViewSet,
    root: Path,
    registry: Mapping[str, Any],
    seed: int,
) -> None:
    output = Path(output_dir)
    normalizer = fit_particle_view_normalizer(
        train_views.views,
        train_views.mask,
        train_split_sha256=train_views.logical_split_sha256,
        generator_checkpoint_sha256=selected.registration[
            "generator_checkpoint_sha256"
        ],
    )
    normalizer_artifact = write_particle_view_normalizer(
        output / "selected_view_normalizer.json", normalizer
    )
    parents, definition = _coordinate_ancestry(
        selected=selected,
        root=root,
        registry=registry,
        seed=seed,
        normalizer_sha256=normalizer_artifact["content_hash"],
    )
    coordinate = build_view_coordinate_binding(
        parent_hashes=parents, coordinate_definition=definition
    )
    write_immutable_json(output / "selected_view_coordinate.json", coordinate)
    result = with_content_hash(
        {
            "contract": PARTICLE_VIEW_SELECTED_COORDINATE_RESULT_CONTRACT,
            "campaign_id": PARTICLE_VIEW_LOW_DATA_CAMPAIGN_ID,
            "selected_target_id": selected.registration["target_id"],
            "selected_target_run_id": selected.run_id,
            "target_selection_sha256": selected.selection["content_hash"],
            "target_registration_sha256": selected.registration[
                "content_hash"
            ],
            "generator_checkpoint_sha256": selected.registration[
                "generator_checkpoint_sha256"
            ],
            "normalizer_sha256": normalizer_artifact["content_hash"],
            "coordinate_binding_sha256": coordinate["content_hash"],
            "fit_split": "train",
            "fit_split_sha256": train_views.logical_split_sha256,
            "model_val_select_loaded": False,
            "stack_val_loaded": False,
            "final_test_loaded": False,
            "quality_gate_used": False,
        }
    )
    write_immutable_json(output / "selected_coordinate_result.json", result)


def publish_selected_view_caches(
    *,
    output_dir: str,
    selected: _SelectedTarget,
    views_by_split: Mapping[str, CandidateViewSet],
    coordinate_path: str,
    normalizer_path: str,
) -> None:
    coordinate = load_hashed_json(coordinate_path)
    validate_view_coordinate_binding(coordinate)
    normalizer = load_particle_view_normalizer(normalizer_path)
    if (
        coordinate["parents"]["normalizer_sha256"]
        != normalizer.content_hash
    ):
        raise ValueError("selected coordinate/normalizer lineage mismatch")
    if set(views_by_split) != set(SELECTED_CACHE_SPLITS):
        raise ValueError("selected-view cache split inventory mismatch")
    output = Path(output_dir)
    manifests: dict[str, str] = {}
    paths: dict[str, str] = {}
    audits = {}
    for split in SELECTED_CACHE_SPLITS:
        values = views_by_split[split]
        manifest = publish_selected_view_cache(
            output / "caches",
            split=split,
            split_sha256=values.logical_split_sha256,
            ordered_identity_sha256=values.ordered_identity_sha256,
            raw_view=values.views,
            mask=values.mask,
            normalizer=normalizer,
            coordinate_binding_sha256=coordinate["content_hash"],
            target_id=selected.registration["target_id"],
        )
        manifest_path = (
            output / "caches" / f"{split}_selected_views.json"
        )
        cached, mask, _ = load_selected_view_cache(
            manifest_path,
            expected_coordinate_binding_sha256=coordinate["content_hash"],
            expected_split_sha256=values.logical_split_sha256,
            expected_normalizer_sha256=normalizer.content_hash,
        )
        audits[split] = audit_live_selected_view_equivalence(
            raw_view=values.views,
            mask=values.mask,
            cached_view=cached,
            cached_mask=mask,
            normalizer=normalizer,
        )
        manifests[split] = manifest["content_hash"]
        paths[split] = str(manifest_path.resolve())
    publication = with_content_hash(
        {
            "contract": PARTICLE_VIEW_FINAL_COORDINATE_CONTRACT,
            "target_id": selected.registration["target_id"],
            "target_selection_sha256": selected.selection["content_hash"],
            "generator_checkpoint_sha256": selected.registration[
                "generator_checkpoint_sha256"
            ],
            "normalizer_sha256": normalizer.content_hash,
            "coordinate_binding_sha256": coordinate["content_hash"],
            "cache_manifest_sha256_by_split": manifests,
            "live_publication_audits": audits,
            "normalizer_recomputed_after_selection": True,
            "consumer_reinitialization_required_after_publication": True,
            "final_test_materialized": False,
        }
    )
    write_immutable_json(output / "selected_view_publication.json", publication)
    index = with_content_hash(
        {
            "contract": PARTICLE_VIEW_SELECTED_CACHE_INDEX_CONTRACT,
            "selected_view_publication_sha256": publication["content_hash"],
            "coordinate_binding_sha256": coordinate["content_hash"],
            "normalizer_sha256": normalizer.content_hash,
            "target_selection_sha256": selected.selection["content_hash"],
            "target_registration_sha256": selected.registration[
                "content_hash"
            ],
            "cache_manifest_path_by_split": paths,
            "cache_manifest_sha256_by_split": manifests,
            "allowed_splits": list(SELECTED_CACHE_SPLITS),
            "final_test_cache_forbidden": True,
        }
    )
    write_immutable_json(output / "selected_view_cache_index.json", index)


def _limited(loader, maximum: int | None):
    if maximum is None:
        return loader

    class _Limited:
        batch_size = getattr(loader, "batch_size", None)
        full_loader = loader

        def __iter__(self):
            from itertools import islice

            return islice(iter(loader), maximum)

        def __len__(self):
            return min(len(loader), maximum)

    return _Limited()


def _selected_loader(
    *,
    data: Mapping[str, Any],
    index: Mapping[str, Any],
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    maximum_batches: int | None,
) -> tuple[SelectedViewLoader, AlignedLogicalJetView, dict[str, Any]]:
    aligned = load_aligned_logical_jet_view(data, split)
    view, mask, manifest = load_selected_view_cache(
        index["cache_manifest_path_by_split"][split],
        expected_coordinate_binding_sha256=index[
            "coordinate_binding_sha256"
        ],
        expected_split_sha256=aligned.logical_split_sha256,
        expected_normalizer_sha256=index["normalizer_sha256"],
    )
    if (
        manifest["content_hash"]
        != index["cache_manifest_sha256_by_split"][split]
    ):
        raise ValueError("selected cache index/manifest hash mismatch")
    base = make_logical_data_loader(
        aligned,
        mode="aligned",
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        seed=seed,
    )
    return (
        _limited(
            SelectedViewLoader(
                base, aligned=aligned, views=view, mask=mask
            ),
            maximum_batches,
        ),
        aligned,
        manifest,
    )


def _load_selected_cache_index(
    root: Path, registry: Mapping[str, Any], seed: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cache_artifacts = _task_artifacts(
        root, registry, "SELECTED_VIEW_CACHE", seed
    )
    index = load_hashed_json(
        _artifact(cache_artifacts, "selected_view_cache_index.json")
    )
    validate_content_hash(
        index, expected_contract=PARTICLE_VIEW_SELECTED_CACHE_INDEX_CONTRACT
    )
    publication = load_hashed_json(
        _artifact(cache_artifacts, "selected_view_publication.json")
    )
    coordinate_artifacts = _task_artifacts(
        root, registry, "SELECTED_COORDINATE_BINDING", seed
    )
    coordinate = load_hashed_json(
        _artifact(coordinate_artifacts, "selected_view_coordinate.json")
    )
    if (
        publication["content_hash"]
        != index["selected_view_publication_sha256"]
        or coordinate["content_hash"] != index["coordinate_binding_sha256"]
    ):
        raise ValueError("selected cache index ancestry mismatch")
    return index, publication, coordinate


def _clean_consumer_model(
    *,
    root: Path,
    registry: Mapping[str, Any],
    seed: int,
    view_dim: int,
) -> tuple[ParticleViewConsumer, dict[str, Any], Path]:
    a0_registration, a0_checkpoint, model = _teacher_from_task(
        root, registry, "A0_VIEW", seed
    )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    architecture = a0_registration["recipe"]["architecture"]
    interface_selection = _selected_consumer_interface(
        root, registry, seed
    )
    recipe = interface_selection["selected_recipe"]
    consumer = ParticleViewConsumer(
        model,
        ParticleViewConsumerConfig(
            view_dim=view_dim,
            hidden_dim=int(architecture["embed_dims"][-1]),
            num_heads=int(architecture["num_heads"]),
            injection_block=int(recipe["injection_block"]),
            view_path=str(recipe["view_path"]),
            learned_trust=bool(recipe["learned_trust"]),
        ),
    )
    return consumer, a0_registration, a0_checkpoint


def _selected_consumer_interface(
    root: Path, registry: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    selection_artifacts = _task_artifacts(
        root, registry, "SELECT_TARGET_DEFINITIONS", seed
    )
    artifact = load_hashed_json(
        _artifact(
            selection_artifacts, "selected_consumer_interface.json"
        )
    )
    validate_content_hash(
        artifact,
        expected_contract="particle_view_consumer_interface_selection_v1",
    )
    return artifact


def _load_clean_consumer(
    *,
    root: Path,
    registry: Mapping[str, Any],
    seed: int,
    view_dim: int,
) -> tuple[ParticleViewConsumer, dict[str, Any], Path]:
    artifacts = _task_artifacts(
        root, registry, "FINAL_CLEAN_CONSUMER", seed
    )
    registration = load_hashed_json(
        _artifact(artifacts, "consumer_registration.json")
    )
    checkpoint = _artifact(artifacts, "best_model_val_stop.pt")
    model, _, _ = _clean_consumer_model(
        root=root, registry=registry, seed=seed, view_dim=view_dim
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload["role"] != "Cview_clean"
        or payload["consumer_config"] != model.config.to_payload()
        or sha256_file(checkpoint) != registration["checkpoint_sha256"]
    ):
        raise ValueError("clean consumer checkpoint/registration mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, registration, checkpoint


def _pview_models(
    registration: Mapping[str, Any],
    root: Path,
) -> list[HierarchicalParticleViewPredictor]:
    config = build_predictor_architecture_config(
        registration["architecture_config"]["architecture_id"],
        view_dim=int(registration["architecture_config"]["view_dim"]),
        input_dim=int(registration["architecture_config"]["input_dim"]),
    )
    if config.to_payload() != registration["architecture_config"]:
        raise ValueError("Pview_0 architecture registration changed")
    models = []
    for relative, digest in zip(
        registration["snapshot_files"], registration["snapshot_sha256"]
    ):
        path = root / relative
        if sha256_file(path) != digest:
            raise ValueError("Pview_0 snapshot changed")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = HierarchicalParticleViewPredictor(config)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models.append(model)
    return models


def run_residual_sampler_fit(
    *,
    train_true_view,
    train_prediction,
    train_mask,
    model_val_stop_true_view,
    model_val_stop_prediction,
    model_val_stop_mask,
    train_identity_sha256: str,
    model_val_stop_split_sha256: str,
    coordinate_binding_sha256: str,
    pview0_checkpoint_sha256: str,
    snapshot_sha256: Sequence[str],
    output_dir: str,
) -> None:
    sampler, warning = fit_correlated_residual_sampler(
        train_true_view=train_true_view,
        train_prediction=train_prediction,
        train_mask=train_mask,
        model_val_stop_true_view=model_val_stop_true_view,
        model_val_stop_prediction=model_val_stop_prediction,
        model_val_stop_mask=model_val_stop_mask,
        train_identity_sha256=train_identity_sha256,
        model_val_stop_split_sha256=model_val_stop_split_sha256,
        coordinate_binding_sha256=coordinate_binding_sha256,
        pview0_checkpoint_sha256=pview0_checkpoint_sha256,
        snapshot_sha256=snapshot_sha256,
    )
    publish_correlated_residual_sampler(
        output_dir, sampler=sampler, warning=warning
    )


def _refit_sampler(
    *,
    train_loader,
    stop_loader,
    models: Sequence[HierarchicalParticleViewPredictor],
    registration: Mapping[str, Any],
    coordinate_binding_sha256: str,
    train_identity_sha256: str,
    model_val_stop_split_sha256: str,
    device: str,
) -> tuple[CorrelatedResidualSampler, dict[str, Any] | None]:
    final_model = models[-1]
    train_rows = collect_pview_predictions(
        final_model, train_loader, device=device
    )
    stop_rows = collect_pview_predictions(
        final_model, stop_loader, device=device
    )
    return fit_correlated_residual_sampler(
        train_true_view=train_rows["true_view"],
        train_prediction=train_rows["prediction"],
        train_mask=train_rows["mask"],
        model_val_stop_true_view=stop_rows["true_view"],
        model_val_stop_prediction=stop_rows["prediction"],
        model_val_stop_mask=stop_rows["mask"],
        train_identity_sha256=train_identity_sha256,
        model_val_stop_split_sha256=model_val_stop_split_sha256,
        coordinate_binding_sha256=coordinate_binding_sha256,
        pview0_checkpoint_sha256=registration["checkpoint_sha256"],
        snapshot_sha256=registration["snapshot_sha256"],
    )


def build_post_target_factory(
    *,
    operation: str,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
    task_id: str,
    output_dir: str,
) -> dict[str, Any]:
    validate_post_target_factory_config(config)
    validate_particle_view_registry(registry)
    if (
        run_id not in POST_TARGET_RUN_IDS
        or task_id != f"{run_id}__seed_{int(seed)}"
    ):
        raise ValueError("post-target task identity is invalid")
    rows = {row["run_id"]: row for row in registry["runs"]}
    if run_id not in rows or int(seed) not in rows[run_id]["seed_ids"]:
        raise ValueError("post-target seed is not registered")
    output = Path(output_dir).resolve()
    root = output.parent.parent
    data = config["runtime_data_config"]
    runtime = config["runtime"]
    selected = _selected_target(root=root, registry=registry, seed=seed)

    if run_id == "SELECTED_COORDINATE_BINDING":
        if operation != "selected_view_publication":
            raise ValueError("coordinate binding operation changed")
        train = load_aligned_logical_jet_view(data, "train")
        views = _materialize_selected_split(
            selected=selected,
            aligned=train,
            root=root,
            registry=registry,
            seed=seed,
            device=runtime["device"],
            num_workers=runtime["num_workers"],
        )
        return {
            "kwargs": {
                "output_dir": str(output),
                "selected": selected,
                "train_views": views,
                "root": root,
                "registry": registry,
                "seed": int(seed),
            },
            "artifact_paths": [
                str(output / "selected_view_normalizer.json"),
                str(output / "selected_view_coordinate.json"),
                str(output / "selected_coordinate_result.json"),
            ],
            "action": publish_selected_coordinate_binding,
        }

    if run_id == "SELECTED_VIEW_CACHE":
        if operation != "selected_view_publication":
            raise ValueError("selected-view cache operation changed")
        parent = _task_artifacts(
            root, registry, "SELECTED_COORDINATE_BINDING", seed
        )
        coordinate_path = _artifact(
            parent, "selected_view_coordinate.json"
        )
        normalizer_path = _artifact(
            parent, "selected_view_normalizer.json"
        )
        views_by_split = {
            split: _materialize_selected_split(
                selected=selected,
                aligned=load_aligned_logical_jet_view(data, split),
                root=root,
                registry=registry,
                seed=seed,
                device=runtime["device"],
                num_workers=runtime["num_workers"],
            )
            for split in SELECTED_CACHE_SPLITS
        }
        cache_paths = [
            str(output / "caches" / f"{split}_selected_views.{suffix}")
            for split in SELECTED_CACHE_SPLITS
            for suffix in ("npz", "json")
        ]
        return {
            "kwargs": {
                "output_dir": str(output),
                "selected": selected,
                "views_by_split": views_by_split,
                "coordinate_path": str(coordinate_path),
                "normalizer_path": str(normalizer_path),
            },
            "artifact_paths": [
                *cache_paths,
                str(output / "selected_view_publication.json"),
                str(output / "selected_view_cache_index.json"),
            ],
            "action": publish_selected_view_caches,
        }

    index, publication, coordinate = _load_selected_cache_index(
        root, registry, seed
    )
    view_dim = int(
        selected.registration["generator_config"]["bottleneck_width"]
    )
    train_loader, train, train_manifest = _selected_loader(
        data=data,
        index=index,
        split="train",
        batch_size=128,
        shuffle=True,
        num_workers=runtime["num_workers"],
        seed=seed,
        maximum_batches=runtime["max_train_batches"],
    )
    stop_loader, stop, stop_manifest = _selected_loader(
        data=data,
        index=index,
        split="model_val_stop",
        batch_size=128,
        shuffle=False,
        num_workers=runtime["num_workers"],
        seed=seed + 1,
        maximum_batches=runtime["max_val_batches"],
    )
    unified = load_hashed_json(data["unified_manifest"]["path"])
    _, _, train_identity = logical_split_binding(unified, "train")

    if run_id == "FINAL_CLEAN_CONSUMER":
        if operation != "consumer_training":
            raise ValueError("clean-consumer operation changed")
        model, a0_registration, _ = _clean_consumer_model(
            root=root, registry=registry, seed=seed, view_dim=view_dim
        )
        return {
            "kwargs": {
                "model": model,
                "train_loader": train_loader,
                "model_val_stop_loader": stop_loader,
                "config": ParticleViewConsumerTrainConfig.for_role(
                    "Cview_clean", seed=seed
                ),
                "output_dir": str(output),
                "lineage": {
                    "a0_registration_sha256": a0_registration[
                        "content_hash"
                    ],
                    "target_registration_sha256": selected.registration[
                        "content_hash"
                    ],
                    "train_identity_sha256": train_identity,
                    "model_val_stop_split_sha256": (
                        stop.logical_split_sha256
                    ),
                    "normalizer_sha256": index["normalizer_sha256"],
                    "selected_view_publication_sha256": publication[
                        "content_hash"
                    ],
                },
                "augment_clean_view_override": bool(
                    _selected_consumer_interface(
                        root, registry, seed
                    )["selected_recipe"]["augment_clean_view"]
                ),
                "device": runtime["device"],
            },
            "artifact_paths": [
                str(output / "best_model_val_stop.pt"),
                str(output / "consumer_registration.json"),
                str(output / "training_curves.json"),
            ],
            "action": None,
        }

    clean_model, clean_registration, clean_checkpoint = _load_clean_consumer(
        root=root, registry=registry, seed=seed, view_dim=view_dim
    )
    pview_lineage = {
        "source_manifest_sha256": selected.registration[
            "source_manifest_sha256"
        ],
        "train_identity_sha256": train_identity,
        "model_val_stop_split_sha256": stop.logical_split_sha256,
        "hlt_preprocessing_sha256": _teacher_from_task(
            root, registry, "A0_VIEW", seed
        )[0]["preprocessing_sha256"],
        "target_selection_sha256": selected.selection["content_hash"],
        "coordinate_binding_sha256": coordinate["content_hash"],
        "selected_view_publication_sha256": publication["content_hash"],
        "train_view_cache_manifest_sha256": train_manifest["content_hash"],
        "model_val_stop_view_cache_manifest_sha256": stop_manifest[
            "content_hash"
        ],
        "clean_consumer_registration_sha256": clean_registration[
            "content_hash"
        ],
        "clean_consumer_checkpoint_sha256": sha256_file(clean_checkpoint),
    }
    if set(pview_lineage) != set(PVIEW0_LINEAGE_FIELDS):
        raise RuntimeError("Pview_0 production lineage inventory drifted")

    if run_id == "PVIEW0":
        if operation != "pview0_training":
            raise ValueError("Pview_0 operation changed")
        model = HierarchicalParticleViewPredictor(
            build_predictor_architecture_config(
                "P_HIER_DECODER_REFINE", view_dim=view_dim
            )
        )
        return {
            "kwargs": {
                "model": model,
                "train_loader": train_loader,
                "model_val_stop_loader": stop_loader,
                "output_dir": str(output),
                "lineage": pview_lineage,
                "config": ParticleViewWarmupConfig(seed=seed),
                "device": runtime["device"],
            },
            "artifact_paths": [
                str(output / "snapshots" / "pview0_epoch_002.pt"),
                str(output / "snapshots" / "pview0_epoch_003.pt"),
                str(output / "snapshots" / "pview0_epoch_004.pt"),
                str(output / "pview0_registration.json"),
                str(output / "pview0_training_curves.json"),
            ],
            "action": None,
        }

    pview_artifacts = _task_artifacts(root, registry, "PVIEW0", seed)
    pview_registration_path = _artifact(
        pview_artifacts, "pview0_registration.json"
    )
    pview_registration = load_hashed_json(pview_registration_path)
    pview_root = pview_registration_path.parent
    validate_pview0_registration(
        pview_registration,
        root=pview_root,
        expected_lineage=pview_lineage,
    )
    models = _pview_models(pview_registration, pview_root)

    if run_id == "RESIDUAL_SAMPLER":
        if operation != "residual_sampler_fit":
            raise ValueError("residual-sampler operation changed")
        train_rows = collect_pview_predictions(
            models[-1], train_loader, device=runtime["device"]
        )
        stop_rows = collect_pview_predictions(
            models[-1], stop_loader, device=runtime["device"]
        )
        warning_path = output / "predictor_tail_warning.json"
        # The warning is optional, so the task result authenticates the
        # always-present publication and sampler registrations.
        return {
            "kwargs": {
                "train_true_view": train_rows["true_view"],
                "train_prediction": train_rows["prediction"],
                "train_mask": train_rows["mask"],
                "model_val_stop_true_view": stop_rows["true_view"],
                "model_val_stop_prediction": stop_rows["prediction"],
                "model_val_stop_mask": stop_rows["mask"],
                "train_identity_sha256": train_identity,
                "model_val_stop_split_sha256": stop.logical_split_sha256,
                "coordinate_binding_sha256": coordinate["content_hash"],
                "pview0_checkpoint_sha256": pview_registration[
                    "checkpoint_sha256"
                ],
                "snapshot_sha256": pview_registration["snapshot_sha256"],
                "output_dir": str(output),
            },
            "artifact_paths": [
                str(output / "correlated_residual_sampler.json"),
                str(output / "residual_sampler_publication.json"),
            ],
            "action": None,
        }

    if operation != "robust_consumer_training":
        raise ValueError("robust-consumer operation changed")
    sampler_artifacts = _task_artifacts(
        root, registry, "RESIDUAL_SAMPLER", seed
    )
    published_sampler = load_hashed_json(
        _artifact(
            sampler_artifacts, "correlated_residual_sampler.json"
        )
    )
    sampler, warning = _refit_sampler(
        train_loader=train_loader,
        stop_loader=stop_loader,
        models=models,
        registration=pview_registration,
        coordinate_binding_sha256=coordinate["content_hash"],
        train_identity_sha256=train_identity,
        model_val_stop_split_sha256=stop.logical_split_sha256,
        device=runtime["device"],
    )
    if sampler.registration != published_sampler:
        raise ValueError(
            "RAM-refit residual sampler differs from registered parent"
        )
    del warning
    bank = SnapshotDropoutPredictionBank(
        models,
        snapshot_sha256=pview_registration["snapshot_sha256"],
        pview0_registration_sha256=pview_registration["content_hash"],
        pview0_lineage=pview_registration["lineage"],
    )
    robust_lineage = {
        "source_manifest_sha256": selected.registration[
            "source_manifest_sha256"
        ],
        "train_identity_sha256": train_identity,
        "model_val_stop_split_sha256": stop.logical_split_sha256,
        "target_selection_sha256": selected.selection["content_hash"],
        "coordinate_binding_sha256": coordinate["content_hash"],
        "selected_view_publication_sha256": publication["content_hash"],
        "train_view_cache_manifest_sha256": train_manifest["content_hash"],
        "model_val_stop_view_cache_manifest_sha256": stop_manifest[
            "content_hash"
        ],
        "clean_consumer_registration_sha256": clean_registration[
            "content_hash"
        ],
        "clean_consumer_checkpoint_sha256": sha256_file(clean_checkpoint),
        "pview0_registration_sha256": pview_registration["content_hash"],
        "pview0_checkpoint_sha256": pview_registration[
            "checkpoint_sha256"
        ],
        "residual_sampler_registration_sha256": sampler.registration[
            "content_hash"
        ],
    }
    return {
        "kwargs": {
            "clean_consumer": clean_model,
            "train_loader": train_loader,
            "model_val_stop_loader": stop_loader,
            "prediction_bank": bank,
            "sampler": sampler,
            "output_dir": str(output),
            "lineage": robust_lineage,
            "clean_consumer_checkpoint_path": str(clean_checkpoint),
            "device": runtime["device"],
        },
        "artifact_paths": [
            str(output / "best_model_val_stop.pt"),
            str(output / "robust_consumer_registration.json"),
            str(output / "robust_consumer_training_curves.json"),
        ],
        "action": None,
    }


def build_post_target_task_specs(
    *,
    factory_config_path: str | Path,
) -> dict[str, dict[str, str]]:
    path = Path(factory_config_path).resolve()
    validate_post_target_factory_config(load_hashed_json(path))
    operations = {
        "SELECTED_COORDINATE_BINDING": "selected_view_publication",
        "SELECTED_VIEW_CACHE": "selected_view_publication",
        "FINAL_CLEAN_CONSUMER": "consumer_training",
        "PVIEW0": "pview0_training",
        "RESIDUAL_SAMPLER": "residual_sampler_fit",
        "ROBUST_CONSUMER": "robust_consumer_training",
    }
    return {
        run_id: {
            "operation": operation,
            "factory": (
                "teacher_logit_reco.local_particle_residual_field."
                "particle_view.post_target_runtime:"
                "build_post_target_factory"
            ),
            "factory_config_path": str(path),
            "factory_config_sha256": sha256_file(path),
        }
        for run_id, operation in operations.items()
    }


__all__ = [
    "PARTICLE_VIEW_POST_TARGET_FACTORY_CONFIG_CONTRACT",
    "PARTICLE_VIEW_SELECTED_CACHE_INDEX_CONTRACT",
    "PARTICLE_VIEW_SELECTED_COORDINATE_RESULT_CONTRACT",
    "POST_TARGET_RUN_IDS",
    "SELECTED_CACHE_SPLITS",
    "SelectedViewLoader",
    "build_post_target_factory",
    "build_post_target_factory_config",
    "build_post_target_task_specs",
    "publish_selected_coordinate_binding",
    "publish_selected_view_caches",
    "registered_target_for_run",
    "run_residual_sampler_fit",
    "validate_post_target_factory_config",
]
